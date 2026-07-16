use glam::Mat2;

use crate::materials::physical_props::{Elastic, FromSI, scale_lame};
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particles;

/// Compressible Neo-Hookean hyperelastic solid (jelly, soft tissue).
///
/// STALE DOC FIXED 2026-07-07: this comment described an older, simpler form
/// (`τ = µ(FFᵀ − I) + λ·ln(J)·I`, still what `ViscoelasticMaterial` actually
/// implements for its own elastic term) that this material's CODE no longer
/// matches -- the actual `kirchhoff_stress` below uses a Simo-Pister
/// volumetric-deviatoric split (`k=λ+µ`, the 2D plane-strain bulk modulus,
/// fixed 2026-07-06 for dimensional correctness) instead. See the real,
/// current formula documented directly in `kirchhoff_stress`'s own body below.
/// Free energy: Ψ = µ/2·(tr(FᵀF)−d) − µ·ln(J) + λ/2·ln(J)²
/// Reference: standard hyperelasticity; used in Stomakhin et al. 2013 (snow paper) §2.
#[derive(Debug, Clone, Copy)]
pub struct NeoHookeanMaterial {
    pub lambda: f32,
    pub mu: f32,
    pub min_density: f32,
    /// Thermal modulus scale: λ_eff = λ·(1 + thermal_expansion·T), same for µ.
    /// Negative = thermal softening (typical). 0.0 = isothermal (default).
    pub thermal_expansion: f32,
    /// Active stress coefficient for muscle/motile-cell behaviour.
    /// τ_total = τ_elastic + activation × coeff × I  (contractile: pulls inward like a muscle).
    /// Independent of elastic state — generates force even at rest.
    /// 0.0 = passive (default). Tune to be on the order of µ for visible locomotion.
    pub active_stress_coeff: f32,
    /// Continuum damage softening rate — real mechanical consequence of accumulated
    /// structural damage (`Particle::friction_hardening`, e.g. from
    /// `rankine_damage_estimate`), not just a passive health readout. Effective
    /// stiffness: µ_eff = µ·exp(−rate·damage), λ_eff = λ·exp(−rate·damage) — the
    /// same exponential softening `RankineMaterial` uses for its own tensile
    /// strength (continuum damage mechanics), applied here to elastic stiffness
    /// instead. Damaged tissue gets progressively softer/weaker as a smooth,
    /// continuous function of real accumulated strain — not a hard on/off failure
    /// threshold. 0.0 = no damage coupling (default, unchanged behavior).
    pub damage_softening_rate: f32,
    /// EXPERIMENTAL, not yet FD-verified for the differentiable trainer (see
    /// `kirchhoff_stress_vjp` doc): Kelvin-Voigt internal viscosity, same
    /// `η·dev(D)` term `ViscoelasticMaterial` already implements (D = the
    /// symmetric part of the APIC velocity gradient). 0.0 = passive elastic
    /// only (default, unchanged behavior for every existing user).
    pub viscosity: f32,
}

impl NeoHookeanMaterial {
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            min_density: 1.0e-6,
            thermal_expansion: 0.0,
            active_stress_coeff: 0.0,
            damage_softening_rate: 0.0,
            viscosity: 0.0,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    ///
    /// Canonical values: E = 5e6, ν = 0.2 (wgsparkl elasticity2 — stiff soft solid).
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }

    /// Analytic adjoint (vector-Jacobian product) of `kirchhoff_stress` w.r.t.
    /// the deformation gradient F -- the first real building block toward
    /// differentiable stepping (offline gradient-based controller training,
    /// same real technique ChainQueen/DiffTaichi/SoftZoo use, applied here as
    /// a from-scratch hand derivation rather than a compiler-generated one --
    /// see `project_domain_taxonomy`/locomotion research notes for why no
    /// Rust autodiff crate fits this problem shape).
    ///
    /// Given `d_loss_d_tau` = ∂L/∂τ (the gradient flowing backward from
    /// wherever this particle's stress feeds into a scalar loss), returns
    /// ∂L/∂F.
    ///
    /// Derivation: τ(F) = (µ/J)·dev(B) + k·ln(J)·I, where B = F·Fᵀ,
    /// J = det(F), dev(B) = B − (tr(B)/2)·I (matching `kirchhoff_stress`
    /// exactly -- updated 2026-07-11 alongside the forward formula's
    /// volumetric-term fix, see that function's doc for why). Reverse-mode
    /// chain rule through B → A=dev(B) → τ, and separately through J (using
    /// the standard cofactor identity ∂J/∂F = J·F⁻ᵀ, so ∂ln(J)/∂F = F⁻ᵀ),
    /// gives:
    ///
    ///   B̄ = (µ/J)·dev(Ḡ)
    ///   ∂L/∂F = (B̄ + B̄ᵀ)·F + [k·tr(Ḡ) − (µ/J)·(Ḡ:A)] · F⁻ᵀ
    ///
    /// where Ḡ = ∂L/∂τ, A = dev(B), and Ḡ:A is the Frobenius inner product
    /// (sum of elementwise products). The `B̄ + B̄ᵀ` (NOT `2·B̄`) matters: B̄
    /// is only symmetric when Ḡ itself is, which isn't guaranteed just
    /// because B and A are -- a real derivation bug first-draft code hit
    /// here, caught by the finite-difference tests below, not by inspection.
    /// Verified against central-difference numerical gradients in this
    /// module's own tests -- hand-derived tensor calculus is exactly where
    /// sign/transpose/symmetry-assumption errors hide, so this is not
    /// trusted on derivation alone.
    ///
    /// Covers only the core elastic term (thermal/damage scaling folded into
    /// µ/λ as constants here, matching how `kirchhoff_stress` already treats
    /// them per-call; the active-stress term is additive and its own
    /// gradient is trivial, not yet wired in). Does NOT cover `viscosity`'s
    /// contribution: that term depends on `velocity_gradient` (the APIC C
    /// matrix), not F, so it's simply absent from ∂L/∂F -- correct as long as
    /// `viscosity` stays 0.0 (its default) for any differentiable use. This
    /// is checked with a REAL (non-debug) assert: training runs are exactly
    /// where `--release` gets used, so a `debug_assert` here would silently
    /// compile away and hand back a wrong gradient with no protection at all.
    pub fn kirchhoff_stress_vjp(
        &self,
        particles: &Particles,
        i: usize,
        d_loss_d_tau: Mat2,
    ) -> Mat2 {
        assert_eq!(
            self.viscosity, 0.0,
            "kirchhoff_stress_vjp does not differentiate through the viscosity term; \
             only viscosity=0.0 materials are safe to use with the differentiable trainer"
        );
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        let damage_scale = (-self.damage_softening_rate * particles.friction_hardening[i]).exp();
        let mu = self.mu * t_scale * damage_scale;
        let lambda = self.lambda * t_scale * damage_scale;
        let k = lambda + mu;

        let b = f * f.transpose();
        let tr_b = b.x_axis.x + b.y_axis.y;
        let dev_b = b - Mat2::from_diagonal(glam::Vec2::splat(tr_b * 0.5));

        let g = d_loss_d_tau;
        let tr_g = g.x_axis.x + g.y_axis.y;
        let dev_g = g - Mat2::from_diagonal(glam::Vec2::splat(tr_g * 0.5));
        // Frobenius inner product G:A = sum of elementwise products.
        let g_dot_a = g.x_axis.x * dev_b.x_axis.x
            + g.x_axis.y * dev_b.x_axis.y
            + g.y_axis.x * dev_b.y_axis.x
            + g.y_axis.y * dev_b.y_axis.y;

        let f_inv_t = f.inverse().transpose();

        // B = F·Fᵀ's VJP is (B̄ + B̄ᵀ)·F for a general (not necessarily
        // symmetric) incoming adjoint B̄ = (µ/J)·dev(Ḡ). B itself is always
        // symmetric, but the GRADIENT flowing into it isn't -- Ḡ = ∂L/∂τ has
        // no reason to be symmetric in general (e.g. it won't be once this
        // feeds into an asymmetric downstream operation like a P2G weight).
        // The simplification (B̄+B̄ᵀ)·F = 2·B̄·F only holds when B̄ itself is
        // symmetric, which is NOT guaranteed just because B and A=dev(B) are.
        let b_bar = (mu / j) * dev_g;
        let term1 = (b_bar + b_bar.transpose()) * f;
        let scalar2 = k * tr_g - (mu / j) * g_dot_a;
        let term2 = scalar2 * f_inv_t;

        term1 + term2
    }
}

impl FromSI<Elastic> for NeoHookeanMaterial {
    fn from_physical(props: &Elastic, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(props.e_pa, props.nu, props.rho_kg_m3, config);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for NeoHookeanMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::NeoHookean
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        // Thermal modulus scaling: λ_eff = λ·(1 + α·T), same for µ.
        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        // Damage softening: µ_eff = µ·exp(−rate·damage), same exponential form
        // RankineMaterial uses for tensile strength -- continuum damage mechanics,
        // not a hand-picked curve. rate=0.0 (default) leaves this at 1.0, no-op.
        let damage_scale = (-self.damage_softening_rate * particles.friction_hardening[i]).exp();
        let mu = self.mu * t_scale * damage_scale;
        let lambda = self.lambda * t_scale * damage_scale;

        // Simo-Pister volumetric-deviatoric split, adapted to plane strain (this
        // solver is 2D-only for now; a 3D bulk term would be revisited if/when
        // that changes -- see project notes).
        // B = F·Fᵀ (left Cauchy-Green), d = 2 in 2D.
        // Deviatoric Kirchhoff: µ · J^{-2/d} · dev(B)  with d=2 → µ/J · dev(B)
        //   dev(B) = B − (tr(B)/2)·I  (2D traceless part)
        // Volumetric Kirchhoff: k · ln(J) · I  (from U(J) = k/2·(ln J)², the
        //   actual Simo & Pister 1984 log-barrier volumetric potential -- NOT
        //   k/2·(J²−1), a bounded polynomial this code used until 2026-07-11.
        //   k = λ + µ  (2D PLANE-STRAIN bulk modulus -- NOT the 3D relation
        //   k=λ+2µ/3, which an earlier version of this code used to match
        //   `sparkl`, a 3D reference engine. Real derivation: linearizing
        //   k·(J−1) against small-strain plane-strain pressure gives k=λ+µ;
        //   the 3D relation is off by µ/3, a real (1−2ν)/3 fractional error
        //   in bulk stiffness -- negligible near ν=0.5 (soft-tissue presets)
        //   but ~20% at ν≈0.2 (compressible/granular-like presets). Fixed
        //   2026-07-06 in favor of dimensional correctness over reference-
        //   engine parity.)
        //
        // REAL BUG FIXED 2026-07-11: `k/2·(J²−1)` is bounded as J→0 (its
        // Kirchhoff contribution approaches a finite `-k/2`, never more), so
        // it supplies only a FINITE ceiling on how hard the material resists
        // further compression, no matter how large k is scaled. A sustained
        // driven load (a creature's own muscle activation, cyclically
        // compressing tissue every gait cycle with nothing to fully release
        // it) can always eventually overpower a finite ceiling given enough
        // cycles -- exactly what a real long-horizon `basic_creature`
        // diagnostic found: net crawl drift collapsed to ~0 while min(J) kept
        // falling and NEVER recovered, and neither raising material stiffness
        // nor adding numerical (APIC) damping fixed it -- both only delayed
        // the same eventual collapse, because neither changes the bounded
        // ceiling itself. The log form `k·ln(J)` has NO such ceiling: as J→0,
        // ln(J)→−∞, so the restoring Kirchhoff stress diverges too -- a
        // genuine physical barrier against total compression, the actual
        // reason Simo & Pister's own 1984 formulation uses `(ln J)²` rather
        // than a bounded polynomial in J. This was a citation/implementation
        // mismatch as much as a stability bug: the doc already cited Simo &
        // Pister for this term while implementing a different, weaker one.
        // Reference: Simo & Pister 1984; Bonet & Wood §6.4 (2D plane-strain form).
        let b = f * f.transpose();
        let tr_b = b.x_axis.x + b.y_axis.y;
        let dev_b = b - Mat2::from_diagonal(glam::Vec2::splat(tr_b * 0.5));
        let k = lambda + mu;

        let dev_stress = (mu / j) * dev_b;
        let vol_stress = (k * j.ln()) * Mat2::IDENTITY;

        let viscous_stress = if self.viscosity > 0.0 {
            let c = particles.velocity_gradient[i];
            let sym = c + c.transpose();
            let d = sym * 0.5;
            let d_trace = d.x_axis.x + d.y_axis.y;
            let d_dev = d - Mat2::from_diagonal(glam::Vec2::splat(d_trace * 0.5));
            self.viscosity * d_dev
        } else {
            Mat2::ZERO
        };

        dev_stress + vol_stress + viscous_stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        // Kirchhoff stress is returned directly → scatter with V₀, not current volume.
        particles.initial_volume[i]
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * particles.velocity_gradient[i];
        particles.deformation_gradient[i] = fp_new * particles.deformation_gradient[i];
        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
    }

    fn activation_scale(&self) -> f32 {
        self.active_stress_coeff
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::NeoHookean as u32,
            lambda: self.lambda,
            mu: self.mu,
            thermal_expansion: self.thermal_expansion,
            active_stress_coeff: self.active_stress_coeff,
            // cohesion_coeff is documented as reusable padding (Snow-only otherwise,
            // zero for all other materials) -- repurposed here for damage_softening_rate.
            cohesion_coeff: self.damage_softening_rate,
            dynamic_viscosity: self.viscosity,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let elastic_dt = elastic_wave_dt(
            self.lambda,
            self.mu,
            1.0,
            density,
            self.min_density,
            cell_width,
            material_cfl,
        );
        // Real bug caught 2026-07-11: `viscosity` was added (stress term) without this
        // bound, so a high-viscosity NeoHookean body took substeps sized only for elastic
        // stability -- far too large for the added viscous (parabolic/diffusive) term,
        // which has its own, much stricter stability requirement. Explicit integration of
        // a diffusive term needs dt ~ h²/ν, not h/c (elastic wave speed) -- a real, standard
        // numerical-stability fact, not tuned to this case. Caught by its actual symptom:
        // deformation gradient inverting (J < 0) within ~500 steps at viscosity=150+,
        // identical formula and bound `ViscoelasticMaterial::timestep_bound` already uses.
        let viscous_dt = if self.viscosity > 0.0 {
            let density = density.max(1.0e-6);
            let kinematic = self.viscosity / density;
            if kinematic > f32::EPSILON {
                viscous_cfl * cell_width * cell_width / kinematic
            } else {
                f32::INFINITY
            }
        } else {
            f32::INFINITY
        };
        elastic_dt.min(viscous_dt)
    }
}

#[cfg(test)]
mod small_strain_linear_elasticity_tests {
    use super::*;
    use crate::Particle;
    use glam::Vec2;

    /// **Small-strain limit must recover exact linear elasticity (Hooke's law).**
    ///
    /// `NeoHookeanMaterial` had zero test comparing its stress-strain response to
    /// any real/analytical elasticity result (confirmed via a full test-file
    /// audit, 2026-07-07) -- only stability (J>0, symmetry) and damage-direction
    /// checks existed. Every well-formed hyperelastic model must reduce to
    /// isotropic linear elasticity as strain -> 0: sigma = lambda*tr(eps)*I +
    /// 2*mu*eps for infinitesimal strain eps. Derivation for THIS model's exact
    /// formula (tau = (mu/J)*dev(B) + (k/2)*(J^2-1)*I, k=lambda+mu, B=F*F^T):
    /// for F = I + delta*E (E symmetric, delta small), linearizing to O(delta)
    /// gives tau ~= 2*mu*delta*dev(E) + (lambda+mu)*delta*tr(E)*I, which is
    /// EXACTLY sigma = lambda*tr(eps)*I + 2*mu*eps with eps=delta*E (the plane-
    /// strain form, matching this material's own k=lambda+mu bulk modulus
    /// fix). Verified numerically here, not just derived by hand.
    fn particle_with_f(f: Mat2) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: Vec2::ZERO,
            v: Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient: f,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening: 0.0,
            log_volume_strain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            _pad: [0; 2],
        });
        particles
    }

    /// Real analytical Hooke's law prediction: sigma = lambda*tr(eps)*I + 2*mu*eps.
    fn linear_elastic_prediction(lambda: f32, mu: f32, eps: Mat2) -> Mat2 {
        let tr_eps = eps.x_axis.x + eps.y_axis.y;
        Mat2::from_diagonal(Vec2::splat(lambda * tr_eps)) + 2.0 * mu * eps
    }

    #[test]
    fn small_uniaxial_strain_matches_hookes_law() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = NeoHookeanMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        // Uniaxial strain: stretch in x, zero in y (E = diag(1, 0)).
        let e = Mat2::from_diagonal(Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain NeoHookean stress should match linear elasticity (Hooke's law) \
             to O(delta^2): predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    #[test]
    fn small_shear_strain_matches_hookes_law() {
        let lambda = 500.0;
        let mu = 1200.0;
        let mat = NeoHookeanMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        // Pure shear strain (symmetric, zero trace): E = [[0, 1], [1, 0]].
        let e = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain NeoHookean shear stress should match linear elasticity to O(delta^2): \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    /// Confirms the match is a real convergence to exact linear elasticity as
    /// strain shrinks, not a coincidence at one specific delta.
    ///
    /// The ABSOLUTE residual (actual minus predicted stress) is genuine O(delta^2)
    /// -- hand-derived and hand-verified against the exact numbers at delta=0.01
    /// (predicted residual from the O(delta^2) term in this model's own
    /// linearization matched the measured residual to within rounding). But
    /// RELATIVE error here is that O(delta^2) absolute residual divided by the
    /// O(delta) leading-order predicted stress, so it correctly scales as
    /// O(delta^2)/O(delta) = O(delta) -- linear, roughly 2x per halving, NOT 4x.
    /// (First version of this test wrongly expected 4x by conflating absolute
    /// and relative error order -- fixed after the measured ~2x ratio held
    /// consistently across delta=1e-2 down to 5e-4, five halvings, before f32
    /// precision noise took over below that.)
    #[test]
    fn hookes_law_match_improves_as_strain_shrinks() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = NeoHookeanMaterial::new(lambda, mu);
        let e = Mat2::from_diagonal(Vec2::new(1.0, -0.3));

        let rel_err_at = |delta: f32| -> f32 {
            let f = Mat2::IDENTITY + delta * e;
            let particles = particle_with_f(f);
            let tau = mat.kirchhoff_stress(&particles, 0);
            let predicted = linear_elastic_prediction(lambda, mu, delta * e);
            let diff = tau - predicted;
            let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
            let scale =
                (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
            err / scale
        };

        let err_large = rel_err_at(1.0e-2);
        let err_small = rel_err_at(5.0e-3);
        assert!(
            err_small < err_large * 0.7 && err_small > err_large * 0.3,
            "halving strain should roughly halve relative error (O(delta) relative \
             error from an O(delta^2) absolute residual over an O(delta) leading term): \
             err(1e-2)={err_large:.2e} err(5e-3)={err_small:.2e} ratio={:.2}",
            err_small / err_large
        );
    }
}

#[cfg(test)]
mod damage_softening_tests {
    use super::*;
    use crate::Particle;

    fn particle_with(deformation_gradient: Mat2, friction_hardening: f32) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: glam::Vec2::ZERO,
            v: glam::Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening,
            log_volume_strain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: glam::Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            _pad: [0; 2],
        });
        particles
    }

    #[test]
    fn zero_softening_rate_matches_undamaged_stress() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 0.0;

        let undamaged = particle_with(f, 0.0);
        let damaged = particle_with(f, 5.0);
        let tau_undamaged = mat.kirchhoff_stress(&undamaged, 0);
        let tau_damaged = mat.kirchhoff_stress(&damaged, 0);

        assert_eq!(
            tau_undamaged, tau_damaged,
            "rate=0.0 must leave stress unaffected by damage (backward compatible default)"
        );
    }

    #[test]
    fn damage_softens_stress_magnitude() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 0.5;

        let healthy = particle_with(f, 0.0);
        let damaged = particle_with(f, 3.0);
        let tau_healthy = mat.kirchhoff_stress(&healthy, 0);
        let tau_damaged = mat.kirchhoff_stress(&damaged, 0);

        assert!(
            tau_damaged.x_axis.length() < tau_healthy.x_axis.length(),
            "damaged tissue must produce weaker stress for the same deformation: \
             healthy={:?} damaged={:?}",
            tau_healthy,
            tau_damaged
        );
    }

    #[test]
    fn severe_damage_approaches_near_zero_stiffness() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 1.0;

        let severely_damaged = particle_with(f, 20.0); // exp(-20) ~ 2e-9, near-total loss
        let tau = mat.kirchhoff_stress(&severely_damaged, 0);
        assert!(
            tau.x_axis.length() < 1.0e-3,
            "severe damage must drive stiffness (and thus stress) toward zero, got {:?}",
            tau
        );
    }
}

#[cfg(test)]
mod kirchhoff_stress_vjp_tests {
    use super::*;
    use crate::Particle;

    fn particle_with_f(f: Mat2) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: glam::Vec2::ZERO,
            v: glam::Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient: f,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening: 0.0,
            log_volume_strain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: glam::Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            _pad: [0; 2],
        });
        particles
    }

    /// Scalar loss L(F) = tau(F) : g (Frobenius inner product with a fixed g) --
    /// the standard way to reduce a matrix-to-matrix function to something a
    /// central-difference check can validate one scalar output at a time.
    fn loss(mat: &NeoHookeanMaterial, f: Mat2, g: Mat2) -> f32 {
        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        tau.x_axis.x * g.x_axis.x
            + tau.x_axis.y * g.x_axis.y
            + tau.y_axis.x * g.y_axis.x
            + tau.y_axis.y * g.y_axis.y
    }

    /// Central-difference numerical gradient of `loss` w.r.t. each of F's 4
    /// components, compared against the analytic `kirchhoff_stress_vjp`.
    ///
    /// This is the real verification the hand derivation needed -- matching
    /// this project's standing "verify numerically" discipline for anything
    /// hand-derived, doubly so for tensor calculus where sign/transpose
    /// errors are exactly the class of mistake that doesn't show up as a
    /// compile error or a crash, only as silently wrong gradients.
    /// Perturbs one scalar component of F by ±h, returns the central-difference
    /// numerical derivative of `loss` w.r.t. that component.
    fn numeric_grad_component(
        mat: &NeoHookeanMaterial,
        mut f: Mat2,
        g: Mat2,
        h: f32,
        set: impl Fn(&mut Mat2, f32),
        get: impl Fn(Mat2) -> f32,
    ) -> f32 {
        let base = get(f);
        set(&mut f, base + h);
        let loss_plus = loss(mat, f, g);
        set(&mut f, base - h);
        let loss_minus = loss(mat, f, g);
        (loss_plus - loss_minus) / (2.0 * h)
    }

    fn check_vjp_matches_finite_difference(mat: &NeoHookeanMaterial, f: Mat2, g: Mat2) {
        let analytic = {
            let particles = particle_with_f(f);
            mat.kirchhoff_stress_vjp(&particles, 0, g)
        };

        let h = 1.0e-3_f32;
        // glam's Mat2 stores columns (x_axis, y_axis); x_axis.y is row 1 of
        // column 0, i.e. F[1][0] in row-major reading.
        let checks: [(&str, f32); 4] = [
            (
                "F[0][0]",
                numeric_grad_component(mat, f, g, h, |m, v| m.x_axis.x = v, |m| m.x_axis.x),
            ),
            (
                "F[1][0]",
                numeric_grad_component(mat, f, g, h, |m, v| m.x_axis.y = v, |m| m.x_axis.y),
            ),
            (
                "F[0][1]",
                numeric_grad_component(mat, f, g, h, |m, v| m.y_axis.x = v, |m| m.y_axis.x),
            ),
            (
                "F[1][1]",
                numeric_grad_component(mat, f, g, h, |m, v| m.y_axis.y = v, |m| m.y_axis.y),
            ),
        ];
        let analytic_vals = [
            analytic.x_axis.x,
            analytic.x_axis.y,
            analytic.y_axis.x,
            analytic.y_axis.y,
        ];

        for ((label, numeric), analytic_val) in checks.iter().zip(analytic_vals) {
            let diff = (numeric - analytic_val).abs();
            let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "kirchhoff_stress_vjp mismatch at {label}: analytic={analytic_val:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e} \
                 (F={f:?}, g={g:?})",
                diff / scale
            );
        }
    }

    #[test]
    fn vjp_matches_finite_difference_at_identity() {
        let mat = NeoHookeanMaterial::new(1000.0, 800.0);
        check_vjp_matches_finite_difference(&mat, Mat2::IDENTITY, Mat2::IDENTITY);
    }

    #[test]
    fn vjp_matches_finite_difference_under_stretch() {
        let mat = NeoHookeanMaterial::new(1000.0, 800.0);
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.05), glam::Vec2::new(-0.02, 0.9));
        let g = Mat2::from_cols(glam::Vec2::new(0.7, -0.3), glam::Vec2::new(0.4, 1.1));
        check_vjp_matches_finite_difference(&mat, f, g);
    }

    #[test]
    fn vjp_matches_finite_difference_under_shear() {
        let mat = NeoHookeanMaterial::new(500.0, 1200.0);
        let f = Mat2::from_cols(glam::Vec2::new(1.0, 0.4), glam::Vec2::new(0.15, 1.05));
        let g = Mat2::from_cols(glam::Vec2::new(-0.5, 0.9), glam::Vec2::new(0.2, -0.6));
        check_vjp_matches_finite_difference(&mat, f, g);
    }

    #[test]
    fn vjp_matches_finite_difference_with_nonsymmetric_g() {
        // g need not be symmetric in general (only the dev(B)-derived internal
        // adjoint happens to be) -- confirms the derivation handles the fully
        // general case, not just the symmetric one it happens to be called
        // with in a real P2G force-scatter backward pass.
        let mat = NeoHookeanMaterial::new(800.0, 800.0);
        let f = Mat2::from_cols(glam::Vec2::new(1.1, -0.1), glam::Vec2::new(0.2, 0.95));
        let g = Mat2::from_cols(glam::Vec2::new(0.3, 1.2), glam::Vec2::new(-0.8, 0.1));
        check_vjp_matches_finite_difference(&mat, f, g);
    }

    #[test]
    fn vjp_respects_thermal_and_damage_scaling() {
        let mut mat = NeoHookeanMaterial::new(900.0, 700.0);
        mat.thermal_expansion = -0.01;
        mat.damage_softening_rate = 0.3;

        let f = Mat2::from_cols(glam::Vec2::new(1.15, 0.08), glam::Vec2::new(-0.05, 0.92));
        let temperature = 12.0;
        let friction_hardening = 2.0;

        let particle_with = |f: Mat2| -> Particles {
            let mut particles = particle_with_f(f);
            particles.temperature[0] = temperature;
            particles.friction_hardening[0] = friction_hardening;
            particles
        };

        let g = Mat2::from_cols(glam::Vec2::new(0.6, -0.4), glam::Vec2::new(0.5, 0.7));
        let analytic = mat.kirchhoff_stress_vjp(&particle_with(f), 0, g);

        let h = 1.0e-3_f32;
        let mut f_plus = f;
        f_plus.x_axis.x += h;
        let mut f_minus = f;
        f_minus.x_axis.x -= h;

        let tau_plus = mat.kirchhoff_stress(&particle_with(f_plus), 0);
        let tau_minus = mat.kirchhoff_stress(&particle_with(f_minus), 0);
        let dot = |t: Mat2| {
            t.x_axis.x * g.x_axis.x
                + t.x_axis.y * g.x_axis.y
                + t.y_axis.x * g.y_axis.x
                + t.y_axis.y * g.y_axis.y
        };
        let numeric = (dot(tau_plus) - dot(tau_minus)) / (2.0 * h);

        let diff = (numeric - analytic.x_axis.x).abs();
        let scale = numeric.abs().max(analytic.x_axis.x.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "vjp must still match finite-difference with thermal/damage scaling active: \
             analytic={:.6} numeric={numeric:.6} relative_diff={:.2e}",
            analytic.x_axis.x,
            diff / scale
        );
    }
}
