use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, GranularProps, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    LOG_CLAMP, MIN_J, elastic_wave_dt, lame_from_young, self_consistent_plastic_multiplier,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Real, cited dry-sand grain diameter (medium sand, standard soil-
/// classification convention) -- matches the value Haeri & Skonieczny 2022
/// (CMAME, arXiv:2111.01523) calibrate their Excavation nonlocal-granular-
/// fluidity case against. Used by `scale_contract` to check whether a
/// scene's `dx_meters` is inside the real, physically valid continuum
/// window for this material -- see that module's own doc for the formula.
pub const GRAIN_DIAMETER_M: f32 = 0.3e-3;

/// Drucker-Prager elastoplastic sand. Ref: Klar et al. 2016.
#[derive(Debug, Clone, Copy)]
pub struct DruckerPragerMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// φ₀: Initial friction angle (radians). Dry sand ≈ 35° = 0.611 rad. (Klar 2016 h₀)
    pub friction_angle: f32,
    /// φ₁: Friction hardening sensitivity — slope of φ(q) near q=0. (Klar 2016 h₁)
    pub hardening_peak: f32,
    /// φ₂: Hardening decay rate — exponential falloff coefficient. (Klar 2016 h₂)
    pub hardening_decay: f32,
    /// φ_r: Residual friction angle (radians). ≈ 10° = 0.175 rad. (Klar 2016 h₃)
    pub friction_residual: f32,
    /// Volume correction factor. 1.0 = full sparkl correction, 0.0 = none.
    pub volume_correction: f32,
    /// Reynolds dilatancy angle ψ (radians). Dense sand ≈ 10–15°.
    ///
    /// When ψ > 0, plastic shear increments drive volumetric expansion:
    /// δεᵥᵖ = sin(ψ) · dq. Physical for dense/compacted sand;
    /// set to 0 for loose/loose-packed sand.
    pub dilatancy_angle: f32,
    /// Yield-surface floor, independent of confining pressure (Pa-equivalent, same
    /// units as `lambda`/`mu`). 0.0 = true cohesionless Mohr-Coulomb (real dry sand,
    /// the Klar 2016 default).
    ///
    /// NOT a claim that dry sand has real cohesion — it doesn't. This compensates for
    /// a real, measured continuum-MPM-resolution artifact: pressure-proportional
    /// friction (`alpha * trace`) vanishes in thin, fast-flowing layers where local
    /// confining pressure is near zero, regardless of the friction angle — confirmed
    /// by three different friction coefficients (DP 35°, µ(I) 20.9-32.8°, µ(I)
    /// 35-40°) all producing IDENTICAL excess runout (~4.7x the Lajeunesse et al. 2004
    /// empirical scaling law for this aspect ratio — see
    /// `sand_column_collapse_runout_matches_lajeunesse_scaling`). Real grain-scale
    /// effects (interlocking, local rearrangement) give actual sand a baseline
    /// resistance in thin layers that point-wise continuum MPM at this resolution
    /// doesn't capture. Calibrate against that benchmark, not against a literature
    /// "sand cohesion" value (which is ~0 and would be the wrong justification).
    pub cohesion: f32,
    /// The Drucker-Prager cone yield surface, BY CONSTRUCTION in the published model
    /// (Klar 2016, verified identical in sparkl/wgsparkl), only ever trims DEVIATORIC
    /// (shear) strain — `project()`'s Case III preserves `trace(eps)` exactly. A
    /// near-hydrostatic impact (mostly compression, little shear) is judged "elastic"
    /// essentially always, regardless of how hard the impact is, because `gamma` stays
    /// negative — nothing in the published model caps pure volumetric compression, and
    /// real sand cannot physically compact past its own void-ratio limit (~20-40%
    /// volume change between loose and dense packing).
    ///
    /// Same mechanism as `StomakhinMaterial`'s `min_plastic_jacobian`: a hard floor on
    /// the STORED singular values' product (the actual `deformation_gradient` written
    /// back), applied AFTER the shear-yield projection so friction/cohesion physics
    /// stay unaffected — only engages when volumetric compression alone would exceed
    /// sand's own packing limit. 0.6 matches Snow's default. The rescale itself floors
    /// each axis individually first -- see `update_particle`'s own comment at the
    /// point of use for why (a pure product-rescale can't recover an axis already at
    /// zero under an extreme impact).
    pub min_volume_jacobian: f32,
    /// Compaction hardening: extra friction angle (radians) per unit of net
    /// volumetric COMPACTION at the moment of yielding (`project`'s own `trace`,
    /// ln of the current effective volume ratio vs the particle's initial state --
    /// negative trace = real net volume loss, i.e. densified). 0.0 (default) = zero
    /// coupling between density and friction,
    /// byte-identical to Klar 2016's own DP model. Real, correctly-directed physics
    /// (denser packing -> higher friction resistance/interlocking is Bolton 1986's
    /// established relative-density-to-friction-angle relation -- the same paper
    /// already cited for the repose-angle target) -- but this coefficient is
    /// a disclosed, simplified linear proportionality, NOT a claim of Bolton's own
    /// precise empirical dilatancy-index formula. Single-phase (dry) compaction only
    /// -- real wet/saturated consolidation (Terzaghi effective stress, pore-pressure-
    /// gated densification) is a distinct phenomenon needing the mixture-coupling
    /// system, not this field.
    pub compaction_sensitivity: f32,
    /// Couple this material's yield check to a `GranularFluidityField`'s
    /// gathered `g` (see `energy::thermodynamics::granular_fluidity` module
    /// doc for the real, cited PDE). `false` (default) = byte-identical to
    /// every existing behavior; every current constructor/preset builds via
    /// `..Self::new(...)`, so this cannot change anything unless explicitly
    /// set. See `project()`'s own doc for exactly how `g` modulates the
    /// return mapping when enabled.
    pub ngf_enabled: bool,
    /// Extra friction angle (radians) required to INITIATE yielding while
    /// the material isn't currently straining, on top of `friction_angle`'s
    /// own q-hardening. 0.0 (default) = byte-identical to every existing
    /// behavior (every current preset builds via `..Self::new(...)`).
    ///
    /// Real, cited granular-physics phenomenon this models: an undisturbed
    /// sand pile is experimentally stable anywhere between the angle of
    /// repose (where flow arrests) and a higher maximum angle of stability
    /// (where flow onsets) -- not one knife-edge value (Bagnold 1954;
    /// Jaeger, Nagel & Behringer 1996, "Granular solids, liquids, and
    /// gases," Rev. Mod. Phys. 68, the standard granular-physics review;
    /// the same two-angle structure underlies the Bak-Tang-Wiesenfeld 1987
    /// sandpile cellular-automaton toppling rule). `friction_angle` keeps
    /// playing exactly the role it already does (the flowing/kinetic
    /// threshold, used for the actual return-mapping projection, unchanged
    /// from every existing calibrated value); `friction_angle +
    /// static_friction_boost` is the higher onset/static threshold, used
    /// ONLY to decide whether yielding starts at all, and only while the
    /// material isn't currently straining -- see `rest_rate_scale`'s own
    /// doc for how "currently straining" is measured. This is the real,
    /// structural fix for a genuine gap found by direct derivation: this
    /// material's own hardening law asymptotes `phi(q)` back to
    /// `friction_angle` for large q regardless of history (no permanent
    /// peak/residual split), so a marginally-yielded state sits exactly at
    /// the yield boundary with ZERO safety margin against numerical noise
    /// -- confirmed as the real mechanism behind dynamically-formed piles
    /// creeping indefinitely even under heavy damping, while hand-placed
    /// (never-yielded) piles hold indefinitely at the same angle.
    pub static_friction_boost: f32,
    /// Strain-rate scale (1/time, same units as `velocity_gradient`) over
    /// which `static_friction_boost` decays as the material starts actively
    /// straining: boost is multiplied by `exp(-strain_rate/rest_rate_scale)`,
    /// so it is essentially fully active at rest and fades out once genuine
    /// flow (not noise) is underway. Real, disclosed calibration knob (same
    /// honest precedent as `cohesion`'s own doc) -- not a literature value,
    /// tuned against this engine's own real substep dynamics. Irrelevant
    /// when `static_friction_boost == 0.0`.
    pub rest_rate_scale: f32,
    /// Real, opt-in stress-relaxation rate (1/time) for a particle's OWN
    /// STORED deviatoric (shear) log-strain -- the actual elastic state
    /// encoded in `deformation_gradient`, not a scalar summary. 0.0
    /// (default) = byte-identical to every existing behavior.
    ///
    /// Root cause this targets, found by direct tracing of `project()`'s
    /// own math: return-mapping plasticity leaves a just-yielded particle's
    /// `dev_norm` sitting EXACTLY on the yield surface (zero margin, by
    /// construction -- the return-mapping's whole job is to satisfy the
    /// yield equation with equality). A violently-collapsed pile's
    /// particles yield constantly during the collapse, so they finish it
    /// riding the yield surface at whatever local pressure/hardening
    /// existed at their LAST yield event -- not the genuinely lower-stress
    /// state their final, settled position would actually need. Any
    /// nonzero ambient noise in `velocity_gradient` (real APIC/grid-
    /// transfer chatter, present even at apparent rest) then has a real
    /// chance of nudging a marginal particle back over the threshold every
    /// substep, and the return-mapping re-clamps it right back onto the
    /// surface -- never below it -- so these don't cancel out, they
    /// compound. A pre-shaped particle never yielded even once: it starts
    /// at `deformation_gradient = IDENTITY` (dev_norm=0) and only develops
    /// whatever real shear its own final geometry needs, which for a real
    /// angle of repose is genuinely BELOW yield with real margin -- same
    /// noise magnitude, no threshold to cross, never yields again.
    ///
    /// Grounded in real granular/soil mechanics: granular and clay soils
    /// show slow stress relaxation / creep under sustained SUB-yield shear
    /// (secondary consolidation), a genuinely different phenomenon from
    /// Perzyna/mu(I) rate-dependence (which govern how fast yielding
    /// ONSETS, not whether already-stored stress decays while inside the
    /// yield surface). Implemented as the SAME mathematical form this
    /// engine's own `ViscoelasticMaterial` already uses for Kelvin-Voigt
    /// relaxation, applied to DP's elastic predictor instead:
    /// `dev(t+dt) = dev(t) * exp(-elastic_relaxation_rate * rest_factor *
    /// dt)`, gated by the SAME `rest_factor` (from `rest_rate_scale`)
    /// `static_friction_boost` uses -- relaxation only applies while
    /// genuinely at rest, never while actively flowing (active flow needs
    /// its real elastic support, relaxing it away mid-flow would be
    /// physically wrong).
    pub elastic_relaxation_rate: f32,
}

impl DruckerPragerMaterial {
    /// Construct with Lamé parameters and default Klar 2016 friction-angle hardening.
    ///
    /// Use [`from_young_modulus`](Self::from_young_modulus) if you prefer E/ν inputs.
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            friction_angle: 35.0_f32.to_radians(),
            hardening_peak: 9.0_f32.to_radians(),
            hardening_decay: 0.2,
            friction_residual: 10.0_f32.to_radians(),
            volume_correction: 1.0,
            dilatancy_angle: 0.0,
            cohesion: 0.0,
            min_volume_jacobian: 0.6,
            compaction_sensitivity: 0.0,
            ngf_enabled: false,
            static_friction_boost: 0.0,
            rest_rate_scale: 1.0,
            elastic_relaxation_rate: 0.0,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    ///
    /// Matches sparkl/wgsparkl API: `DruckerPragerPlasticity::new(E, nu)`.
    /// Canonical demo value (sparkl basic2): E = 1e5, ν = 0.2.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }

    /// Cohesionless: φ=35°, no dilatancy. Klar 2016 defaults. Dry sand regime.
    pub fn cohesionless(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::from_young_modulus(young_modulus, poisson_ratio)
    }

    /// Low friction: φ=25°, weaker hardening. Loose silty soil regime.
    pub fn low_friction(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            friction_angle: 25.0_f32.to_radians(),
            hardening_peak: 4.0_f32.to_radians(),
            hardening_decay: 0.1,
            friction_residual: 5.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// Dilatant: φ=38°, ψ=12° Reynolds dilatancy. Dense compacted sand regime.
    pub fn dilatant(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            friction_angle: 38.0_f32.to_radians(),
            dilatancy_angle: 12.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// Friction coefficient α(q) derived from friction angle φ(q).
    /// φ(q) = friction_angle + compaction_boost + (hardening_peak·q − friction_residual)·exp(−hardening_decay·q)
    /// α(q) = √(2/3) · 2·sin(φ) / (3 − sin(φ))
    ///
    /// `compaction_boost = compaction_sensitivity * max(0, -trace_ln_volume_ratio)` —
    /// `trace_ln_volume_ratio` is `project`'s own `trace` (ln of the current net
    /// volume ratio vs the particle's initial state, instantaneous + accumulated
    /// history combined) at the exact moment of yielding — see
    /// `compaction_sensitivity`'s own doc. Zero when the field is at its 0.0
    /// default, so this is byte-identical to the original q-only formula unless
    /// opted in.
    fn alpha(&self, q: f32, trace_ln_volume_ratio: f32) -> f32 {
        self.alpha_with_phi_delta(q, trace_ln_volume_ratio, 0.0)
    }

    /// Same formula as `alpha`, with an extra additive friction-angle term
    /// (`phi_delta`) -- used by `project`'s static/kinetic onset check (see
    /// `static_friction_boost`'s own doc). `phi_delta=0.0` makes this
    /// byte-identical to `alpha`.
    fn alpha_with_phi_delta(&self, q: f32, trace_ln_volume_ratio: f32, phi_delta: f32) -> f32 {
        let compaction_boost = self.compaction_sensitivity * (-trace_ln_volume_ratio).max(0.0);
        let phi = self.friction_angle
            + phi_delta
            + compaction_boost
            + (self.hardening_peak * q - self.friction_residual)
                * (-self.hardening_decay * q).exp();
        let s = phi.sin();
        (2.0_f32 / 3.0).sqrt() * (2.0 * s) / (3.0 - s)
    }

    /// Drucker-Prager return mapping in log-strain (Hencky) space.
    ///
    /// Returns `Some((projected_sigma, delta_q))` if projection occurred (plastic step),
    /// `None` if the trial state is inside the yield surface (elastic step).
    ///
    /// SELF-CONSISTENT (closest-point-projection) return mapping: `alpha` is evaluated
    /// at the END-of-step hardening state `q + gamma`, not the pre-step `q` -- real
    /// numerical rigor per Simo & Taylor 1985 ("Consistent tangent operators for
    /// rate-independent elastoplasticity," CMAME 48:101-118) and Simo & Hughes,
    /// *Computational Inelasticity* (1998), the standard reference on return-mapping
    /// consistency. `sparkl::DruckerPragerPlasticity::project_deformation_gradient`
    /// and `wgsparkl::models::drucker_prager::project_deformation_gradient` both use
    /// the cheaper single-pass (pre-step `q`) version instead -- real, disclosed
    /// deviation from those reference implementations, not an oversight: because
    /// `alpha` depends on `q + gamma` and `gamma` itself depends on `alpha`, this is a
    /// genuinely coupled nonlinear system, solved here via fixed-point iteration
    /// (`phi(q)` is bounded/smooth, converges in a handful of iterations). `q` is the
    /// accumulated plastic shear-strain norm; it is expected to keep growing slowly
    /// under sustained load even once a pile looks "settled" -- that mirrors real
    /// critical-state soil mechanics (friction angle relaxing from peak toward
    /// residual as cumulative shear strain grows), not a bug to eliminate.
    ///
    /// # Nonlocal Granular Fluidity coupling (`ngf_enabled`, real, disclosed synthesis)
    /// Henann & Kamrin's real coupling (arXiv:1408.5205 eq. 4) is
    /// rate-explicit: `γ̇ = g·μ` -- plastic flow proceeds at a finite RATE set
    /// by the local fluidity `g`, not instantaneously the moment the yield
    /// surface is touched. This return mapping is instead rate-INDEPENDENT
    /// (the `gamma` above already assumes full relaxation onto the yield
    /// surface every step -- effectively an infinite rate). The translation
    /// below is a real, standard technique for exactly this mismatch --
    /// Perzyna (1966) viscoplastic regularization of a rate-independent
    /// yield surface -- NOT a formula taken directly from Haeri &
    /// Skonieczny 2022 (their own formulation is a different, full
    /// rate-explicit hyperelastic scheme, not this return-mapping
    /// structure): cap the plastic multiplier actually applied this step at
    /// `g·μ·dt`, letting local fluidity throttle how fast a point can
    /// genuinely flow, rather than replacing the yield surface's location.
    /// `μ` (stress ratio) is derived from the SAME trial quantities already
    /// computed below, reusing the identical formula
    /// `MuIRheologyMaterial::update_particle` (`sand_mui.rs`) uses:
    /// `p_trial = -(lambda+mu)*trace`, `q_trial = sqrt(2)*mu*dev_norm`,
    /// `μ = q_trial/p_trial = sqrt(2)*dev_norm/(-ratio*trace)`.
    fn project(
        &self,
        sigma: Vec2,
        log_volume_strain: f32,
        q: f32,
        dt: f32,
        nonlocal_fluidity: f32,
        strain_rate_norm: f32,
    ) -> Option<(Vec2, f32)> {
        let sigma = sigma.abs().max(Vec2::splat(LOG_CLAMP));
        // Hencky (logarithmic) strain, shifted by the accumulated volumetric offset.
        let eps = Vec2::new(
            sigma.x.ln() + log_volume_strain * 0.5,
            sigma.y.ln() + log_volume_strain * 0.5,
        );
        let trace = eps.x + eps.y;
        let dev = eps - Vec2::splat(trace * 0.5);
        let dev_norm = dev.length();

        // Tension cutoff or purely volumetric deformation: project to identity (σ = 1).
        // dq = dev_norm only — friction hardening is driven by shear, not volumetric expansion.
        // Using eps.length() here would include the log_volume_strain offset and cause
        // unbounded q growth in static/settled sand.
        if dev_norm == 0.0 || trace > 0.0 {
            return Some((Vec2::ONE, dev_norm));
        }

        // Yield function: γ = |dev_ε| + ratio · tr · α − cohesion/(2µ).
        // Klar 2016 eq. 25, d=2: (d·λ + 2µ)/(2µ) = (2λ+2µ)/(2µ) = (λ+µ)/µ.
        // Verified against sparkl DruckerPragerPlasticity::project and wgsparkl drucker_prager.wgsl.
        // The cohesion term shifts the yield threshold by a pressure-INDEPENDENT amount —
        // converting stress-space Mohr-Coulomb cohesion c (||dev(sigma)|| <= alpha*p + c)
        // into this strain-space equation via dev(sigma) = 2*mu*dev(eps) gives the c/(2*mu)
        // divisor below. See `cohesion`'s doc comment for why this exists.
        let ratio = (self.lambda + self.mu) / self.mu;
        // `trace` (already ln(current effective volume ratio) relative to the initial
        // state, folding in BOTH the instantaneous trial state and accumulated
        // `log_volume_strain` history) is only reached here once already confirmed
        // <= 0 above -- i.e. real net compaction, at the exact moment of yielding.
        // A far more universally-responsive compaction signal than
        // `log_volume_strain` alone, which Case III's shear-only projection keeps
        // nearly invariant by construction for non-dilatant sand (dilatancy_angle=0).
        let cohesion_term = self.cohesion / (2.0 * self.mu);

        // Static/kinetic onset check (see `static_friction_boost`'s own doc).
        // Skipped entirely (zero cost, zero behavior change) when the field
        // is at its 0.0 default -- every existing preset/constructor. When
        // enabled: a HIGHER, boosted friction angle decides whether yielding
        // starts at all while the material isn't currently straining; the
        // ACTUAL projection below always uses the ordinary, unboosted,
        // already-proven `alpha(q)` -- extra resistance only gates the
        // ONSET of flow, never its sustained rate, matching real Coulomb
        // static-vs-kinetic behavior (more force to start sliding than to
        // keep something already sliding moving).
        if self.static_friction_boost != 0.0 {
            let rest_factor = if self.rest_rate_scale > 0.0 {
                (-strain_rate_norm / self.rest_rate_scale).exp()
            } else {
                0.0
            };
            let phi_delta = self.static_friction_boost * rest_factor;
            let gamma_onset_check = self_consistent_plastic_multiplier(
                dev_norm + ratio * trace * self.alpha_with_phi_delta(q, trace, phi_delta)
                    - cohesion_term,
                q,
                |q_trial| {
                    dev_norm + ratio * trace * self.alpha_with_phi_delta(q_trial, trace, phi_delta)
                        - cohesion_term
                },
            );
            if gamma_onset_check <= 0.0 {
                return None; // Boosted-at-rest threshold not crossed -- stays elastic.
            }
        }

        // Self-consistency: `alpha(q + gamma)` depends on gamma, and gamma depends
        // on alpha -- shared iteration logic lives in `self_consistent_plastic_
        // multiplier` (see its own doc for the real citation and why it's a
        // generic, cross-material solver, not DP-specific), this closure supplies
        // only DP's own yield equation. Single-pass (pre-step-q) value seeds the
        // initial guess.
        let initial_gamma = dev_norm + ratio * trace * self.alpha(q, trace) - cohesion_term;
        let gamma = self_consistent_plastic_multiplier(initial_gamma, q, |q_trial| {
            dev_norm + ratio * trace * self.alpha(q_trial, trace) - cohesion_term
        });

        if gamma <= 0.0 {
            return None; // Inside yield surface — elastic step.
        }

        // NGF rate limiter (real, disclosed synthesis -- see this function's
        // own doc above). `-ratio*trace > 0` is guaranteed here (trace <= 0
        // confirmed above, ratio > 0 always), so `mu_ratio` is well-defined.
        //
        // `dev_norm/(-ratio*trace)` alone is a strain-space ratio, not the true
        // stress ratio q_trial/p_trial (which needs
        // `sqrt(2)*mu*dev_norm / p_trial`) -- omitting the `self.mu` factor
        // understates `mu_ratio` by ~3600x at this scene's real SI-to-grid
        // scaling, making `gamma_rate_limited` (and every collapse this
        // coupling is meant to permit) 3600x too small.
        let gamma = if self.ngf_enabled {
            let mu_ratio =
                std::f32::consts::SQRT_2 * self.mu * dev_norm / (-ratio * trace).max(1e-9);
            let gamma_rate_limited = (nonlocal_fluidity * mu_ratio * dt).max(0.0);
            gamma.min(gamma_rate_limited)
        } else {
            gamma
        };
        if gamma <= 0.0 {
            return None; // Yielded, but NGF's local fluidity hasn't built up
            // enough yet to permit real flow this step -- an elastic step
            // for now, not a bug (the whole point of a finite-rate coupling).
        }

        // Project onto yield surface in log-strain space, then exponentiate.
        let h = eps - gamma * (dev / dev_norm);
        Some((Vec2::new(h.x.exp(), h.y.exp()), gamma))
    }
}

impl FromSI<GranularProps> for DruckerPragerMaterial {
    fn from_physical(props: &GranularProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        Self {
            friction_angle: props.friction_angle_deg.to_radians(),
            dilatancy_angle: props.dilatancy_angle_deg.to_radians(),
            ..Self::new(lambda, mu)
        }
    }
}

impl MaterialModel for DruckerPragerMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::DruckerPrager
    }

    /// Corotated elastic Kirchhoff stress: τ = 2µ(F−R)Fᵀ + λ(J−1)J·I
    /// R is the rotation from 2D polar decomposition of F.
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let r = polar_decomposition_2d(f);

        let f_t = f.transpose();
        2.0 * self.mu * (f - r) * f_t + self.lambda * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn init_particle(&self, particle: &mut Particle) {
        // q=0 gives φ = h0 − h3 = 25° (too weak). The neutral point where
        // φ(q) = h0 exactly is q = h3/h1. Matches sparkl's plastic_hardening=1.0
        // default (which gives φ ≈ 34.2°). At q = h3/h1 the hardening term = 0.
        particle.friction_hardening = if self.hardening_peak > 0.0 {
            self.friction_residual / self.hardening_peak
        } else {
            0.0
        };
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;

        // Real deviatoric strain-RATE norm (Frobenius) from the same APIC
        // velocity_gradient already gathered this substep -- zero new
        // per-particle state, see `static_friction_boost`'s own doc. Only
        // computed when actually used (byte-identical cost otherwise).
        // Shared by `elastic_relaxation_rate` (same "is this particle
        // currently at rest" signal, see that field's own doc).
        let strain_rate_norm = if self.static_friction_boost != 0.0
            || self.elastic_relaxation_rate != 0.0
        {
            let l = *ctx.velocity_gradient;
            let dxx = l.x_axis.x;
            let dyy = l.y_axis.y;
            let dxy = 0.5 * (l.x_axis.y + l.y_axis.x);
            let half_trace = (dxx + dyy) * 0.5;
            let dev_xx = dxx - half_trace;
            let dev_yy = dyy - half_trace;
            (dev_xx * dev_xx + dev_yy * dev_yy + 2.0 * dxy * dxy).sqrt()
        } else {
            0.0
        };

        let (u, sigma, vt) = svd2(f_trial);
        let new_sigma = if let Some((proj_sigma, dq)) = self.project(
            sigma,
            *ctx.log_volume_strain,
            *ctx.friction_hardening,
            dt,
            ctx.nonlocal_fluidity,
            strain_rate_norm,
        ) {
            let sigma_abs = sigma.abs().max(Vec2::splat(LOG_CLAMP));
            let prev_det = sigma_abs.x * sigma_abs.y;
            let new_det = proj_sigma.x * proj_sigma.y;
            let diff = new_det - prev_det;
            let corrected_det = if diff > 0.0 {
                new_det
            } else {
                prev_det + diff * self.volume_correction
            };

            *ctx.log_volume_strain += prev_det.ln() - corrected_det.ln();
            let q_max = 5.0 / self.hardening_decay.max(1e-6);
            *ctx.friction_hardening = (*ctx.friction_hardening + dq).min(q_max);
            if self.dilatancy_angle > 0.0 {
                *ctx.log_volume_strain += self.dilatancy_angle.sin() * dq;
            }
            proj_sigma
        } else {
            sigma
        };

        // Real, opt-in stress relaxation -- see `elastic_relaxation_rate`'s
        // own doc for the full mechanism/citation. Decays the STORED
        // deviatoric log-strain (the actual elastic shear state, not a
        // scalar summary) toward zero while the particle is genuinely at
        // rest, whether this step yielded or not -- a just-settled particle
        // still carries whatever deviatoric shear its last yield event (or
        // its elastic history) left it with. Skipped entirely (zero cost)
        // when the rate is at its 0.0 default.
        let new_sigma = if self.elastic_relaxation_rate > 0.0 {
            let rest_factor = if self.rest_rate_scale > 0.0 {
                (-strain_rate_norm / self.rest_rate_scale).exp()
            } else {
                0.0
            };
            if rest_factor > 1.0e-6 {
                let sigma_abs = new_sigma.abs().max(Vec2::splat(LOG_CLAMP));
                let eps = Vec2::new(
                    sigma_abs.x.ln() + *ctx.log_volume_strain * 0.5,
                    sigma_abs.y.ln() + *ctx.log_volume_strain * 0.5,
                );
                let trace = eps.x + eps.y;
                let dev = eps - Vec2::splat(trace * 0.5);
                let dev_norm = dev.length();
                if dev_norm > 1.0e-9 {
                    let decay = (-self.elastic_relaxation_rate * rest_factor * dt).exp();
                    let new_eps = Vec2::splat(trace * 0.5) + dev * decay;
                    Vec2::new(
                        (new_eps.x - *ctx.log_volume_strain * 0.5).exp(),
                        (new_eps.y - *ctx.log_volume_strain * 0.5).exp(),
                    )
                } else {
                    new_sigma
                }
            } else {
                new_sigma
            }
        } else {
            new_sigma
        };

        // Volumetric floor -- see `min_volume_jacobian`'s doc. Applied AFTER the
        // shear-yield projection above and regardless of whether that projection
        // fired (a near-hydrostatic impact is judged "elastic" by the cone above and
        // never reaches it), so friction/cohesion physics are untouched. Uniform
        // rescale (not a per-axis clamp like Snow's) preserves the deviatoric shape
        // the yield projection already chose -- only overall volume is corrected.
        //
        // Take magnitudes FIRST: this engine's `svd2` does NOT guarantee non-negative
        // singular values like textbook SVD — it keeps U a proper rotation by encoding
        // a reflection as sigma.y going NEGATIVE instead (see svd2's
        // `if u.determinant() < 0.0 { ...; sigma.y = -sigma.y }`). An already-inverted
        // state is exactly the "exceeded sand's packing limit" case this floor exists
        // for, just approached from the other side — handles "too compressed" and
        // "already inverted" with one uniform rule instead of two different guards.
        let mut new_sigma = new_sigma.abs();
        // Floor each AXIS individually before the product-based rescale below:
        // under a hard enough impact, one singular value can collapse to exactly
        // (or within float noise of) zero on its own axis. The rescale below
        // multiplies both axes by the SAME scalar to bring their PRODUCT up to
        // `min_volume_jacobian`, which cannot recover an axis that's already at
        // zero (0 * any finite scalar is still 0) -- direct instrumentation showed
        // `new_sigma=(26.07, 0.0)`, `j_new=0` exactly, cascading across 46
        // substeps in the failing scenario, collapsing `min_j_terrain` from its
        // correct 0.6 plateau to exactly 0.0. A small per-axis floor, applied
        // before the rescale, guarantees the rescale always has two genuinely
        // nonzero numbers to work with -- for any real (non-degenerate) input
        // this floor never engages, since ordinary singular values sit far above
        // it.
        const MIN_AXIS: f32 = 1.0e-3;
        new_sigma = new_sigma.max(Vec2::splat(MIN_AXIS));
        let j_new = new_sigma.x * new_sigma.y;
        if j_new < self.min_volume_jacobian {
            new_sigma *= (self.min_volume_jacobian / j_new.max(1e-6)).sqrt();
        }

        let sigma_mat = Mat2::from_cols(Vec2::new(new_sigma.x, 0.0), Vec2::new(0.0, new_sigma.y));
        *ctx.deformation_gradient = u * sigma_mat * vt;

        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::DruckerPrager as u32,
            lambda: self.lambda,
            mu: self.mu,
            dp_h0: self.friction_angle,
            dp_h1: self.hardening_peak,
            dp_h2: self.hardening_decay,
            dp_h3: self.friction_residual,
            // compression_limit repurposed for DP: stores dilatancy angle ψ (radians).
            // Snow uses compression_limit for its singular-value clamp (model 4 only).
            compression_limit: self.dilatancy_angle,
            // stretch_limit repurposed for DP: stores the cohesion floor (Pa-equivalent).
            // Not read by the GPU's model==5u branch for any other purpose.
            stretch_limit: self.cohesion,
            volume_ratio_min: self.min_volume_jacobian,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        elastic_wave_dt(
            self.lambda,
            self.mu,
            1.0,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        )
    }
}

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;
    use crate::particle::Particles;

    /// Isolates whether `project()` itself matches the analytically-derived 2D
    /// Mohr-Coulomb marginal-yield condition, bypassing MPM's grid/transfer pipeline
    /// entirely (no P2G, no gravity, no free surface — a single particle, a single
    /// hand-built deformation gradient, called directly).
    ///
    /// Derivation: converting this 2D log-strain DP
    /// return mapping into principal Cauchy stress shows elastic moduli cancel exactly,
    /// giving a universal relation sin(phi_eff) = sqrt(2) * alpha(q), independent of
    /// lambda/mu. For the default Klar 2016 params at phi_in=35 deg, alpha(q_init) =
    /// 0.386019, predicting phi_eff = 33.087 deg.
    ///
    /// This test builds a deformation gradient at EXACTLY that marginal angle and checks:
    /// slightly inside (less shear) => elastic (no change). slightly outside (more shear)
    /// => plastic (state changes). If this holds, the constitutive code matches the math
    /// and the real repose-angle gap lives in MPM's grid transfer, not here.
    /// Builds a strain state whose underlying STRESS state (sigma_i = 2*mu*eps_i +
    /// lambda*tr(eps)) sits at exactly Mohr-Coulomb angle `phi_test_deg`. Strain-space
    /// and stress-space deviatoric/volumetric ratios differ by the elastic `ratio` factor
    /// (dev(stress)/-tr(stress) = (1/ratio) * dev(strain)/-tr(strain)), so this must
    /// multiply by `ratio`, not just `sin(phi)/sqrt(2)` directly in strain space.
    fn marginal_state_at_phi_eff(ratio: f32, trace: f32, phi_test_deg: f32) -> (Vec2, f32) {
        let phi_test = phi_test_deg.to_radians();
        let dev_norm = -trace * ratio * phi_test.sin() / std::f32::consts::SQRT_2;
        let diff = dev_norm * std::f32::consts::SQRT_2; // |eps1 - eps2|
        let eps1 = (trace + diff) * 0.5;
        let eps2 = (trace - diff) * 0.5;
        (Vec2::new(eps1.exp(), eps2.exp()), dev_norm)
    }

    fn run_one_step(sand: &DruckerPragerMaterial, sigma: Vec2, q: f32) -> (Vec2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = q;
        let mut particles = Particles::from(vec![p]);
        sand.update_particle(&mut particles.update_ctx(0), 1.0);
        let f = particles.deformation_gradient[0];
        (
            Vec2::new(f.x_axis.x, f.y_axis.y),
            particles.friction_hardening[0],
        )
    }

    #[test]
    fn marginal_30deg_state_does_not_yield_for_35deg_friction() {
        let sand = DruckerPragerMaterial::new(2000.0, 3000.0);
        let q_init = sand.friction_residual / sand.hardening_peak;
        let ratio = (sand.lambda + sand.mu) / sand.mu;
        let phi_eff_deg = 33.087_f32; // sqrt(2)*alpha(q_init) for phi_in=35deg

        // Comfortably INSIDE the predicted yield surface (25 deg < 33.087 deg effective).
        let (sigma_in, _) = marginal_state_at_phi_eff(ratio, -0.01, 25.0);
        let (sigma_after, q_after) = run_one_step(&sand, sigma_in, q_init);
        assert!(
            (sigma_after - sigma_in).length() < 1.0e-6,
            "25 deg state (inside 33.087 deg yield surface) should stay elastic: \
             sigma_in={sigma_in:?} sigma_after={sigma_after:?}"
        );
        assert!(
            (q_after - q_init).abs() < 1.0e-6,
            "q should not change on an elastic step: q_init={q_init} q_after={q_after}"
        );

        // Comfortably OUTSIDE the predicted yield surface (40 deg > 33.087 deg effective).
        let (sigma_out, _) = marginal_state_at_phi_eff(ratio, -0.01, 40.0);
        let (sigma_after2, q_after2) = run_one_step(&sand, sigma_out, q_init);
        assert!(
            (sigma_after2 - sigma_out).length() > 1.0e-6,
            "40 deg state (outside 33.087 deg yield surface) should yield (state should \
             change): sigma_out={sigma_out:?} sigma_after2={sigma_after2:?}"
        );
        assert!(
            q_after2 > q_init,
            "q should increase on a plastic step: q_init={q_init} q_after2={q_after2}"
        );

        println!("phi_eff prediction = {phi_eff_deg} deg (informational, not asserted directly)");
    }
}

/// Real, headless first measurement of the Nonlocal Granular Fluidity
/// coupling (Phase 3 of the NGF plan) -- DIAGNOSTIC, not yet a pass/fail
/// regression: the real outcome isn't known ahead of time, so this reports
/// honest numbers rather than asserting a threshold picked in advance.
///
/// Runs on `SimConfig::earth` (real SI throughout: `dx_meters`, `dt_seconds`,
/// gravity) rather than the arbitrary-unit convention `tests/accuracy.rs`'s
/// own Lajeunesse benchmark uses, because `GranularFluidityField::apply`'s
/// reaction step divides by `t0_s` (real seconds) and multiplies by
/// `sub_dt` -- mixing a real-seconds `t0_s` against an arbitrary substep
/// time unit is a genuine units error. Real SI sand properties (E=15MPa,
/// nu=0.3, matching Haeri & Skonieczny 2022's own Excavation case,
/// cross-checked internally consistent: their bulk modulus B=12.5MPa at
/// E=15MPa implies nu=0.3 exactly via K=E/(3(1-2nu))). Since this is a
/// genuinely different (real-SI) scene than `tests/accuracy.rs`'s own
/// Lajeunesse benchmark, this test measures its own fresh cohesionless
/// baseline under the same config rather than reusing that benchmark's
/// arbitrary-unit numbers.
///
/// Internal (not `tests/accuracy.rs`) because the pressure/stress-ratio
/// closure needs `svd2` and the real Hencky-strain formula, both
/// crate-internal -- same reason `marginal_yield_tests` above lives here.
#[cfg(test)]
mod ngf_verification_tests {
    use super::*;
    use crate::materials::physical_props::Elastic;
    use crate::thermodynamics::{GranularFluidityConfig, GranularFluidityField};
    use crate::{FrictionBoundary, SimConfig, Simulation, SpawnRegion};
    use glam::IVec2;

    const YOUNG_MODULUS_PA: f32 = 15.0e6; // Haeri & Skonieczny 2022 Table 1, Excavation
    const POISSON_RATIO: f32 = 0.3; // cross-checked from their own E/B via K=E/(3(1-2nu))
    const BULK_DENSITY_KG_M3: f32 = 1600.0; // real loose dry sand bulk density
    const CELL_M: f32 = 0.01;

    // The pressure fed to the g-field uses real SI lambda/mu
    // (`lame_from_young` on real Pa values), not grid-scaled ones
    // (`scale_lame`) -- the reaction term's `sqrt(P/rho_s)*d` needs REAL
    // Pascals to combine sensibly with `rho_s` (real kg/m3) and `d` (real
    // meters). `mu_ratio` itself is a dimensionless ratio (q_trial/p_trial),
    // so the same scale factor in numerator and denominator would cancel
    // regardless of unit system -- only the standalone pressure (needed by
    // the reaction term in its own right) actually requires real units, so
    // computing everything in real SI is simpler AND correct, not merely
    // "close enough".
    fn ngf_pressure_and_ratio(p: &Particle) -> (f32, f32) {
        let (lambda, mu) = lame_from_young(YOUNG_MODULUS_PA, POISSON_RATIO);
        let (_, sigma, _) = svd2(p.deformation_gradient);
        let sigma = sigma.abs().max(Vec2::splat(LOG_CLAMP));
        let eps = Vec2::new(
            sigma.x.ln() + p.log_volume_strain * 0.5,
            sigma.y.ln() + p.log_volume_strain * 0.5,
        );
        let trace = eps.x + eps.y;
        let dev = eps - Vec2::splat(trace * 0.5);
        let dev_norm = dev.length();
        // Same formula as `MuIRheologyMaterial::update_particle`
        // (`sand_mui.rs`): p_trial = -(lambda+mu)*trace, q_trial =
        // sqrt(2)*mu*dev_norm (STRESS-space deviator -- the `*mu` converts
        // the strain deviator into stress via the elastic shear modulus;
        // dropping it, as an earlier version of this function did, gives a
        // strain-space quantity off by a factor of `mu` -- ~3600x too small
        // at this scene's real SI-to-grid scaling, which is why the first
        // real run of this test showed `max_mu` pinned exactly at the
        // empty-cell fallback `mu_s`, never a real scattered value: no
        // particle's mu_ratio ever came remotely close to crossing it).
        // mu_ratio = q_trial/p_trial, real stress-ratio, comparable to
        // `mu_s` on the same footing.
        let p_trial = -(lambda + mu) * trace;
        let mu_ratio = if p_trial > 1.0e-6 {
            std::f32::consts::SQRT_2 * mu * dev_norm / p_trial
        } else {
            0.0
        };
        (p_trial.max(0.0), mu_ratio)
    }

    fn ngf_config() -> GranularFluidityConfig {
        // Real, quantified, disclosed scale mismatch: at the LITERAL real
        // grain diameter (0.3mm, GRAIN_DIAMETER_M), the cooperativity
        // length is genuinely microscopic vs this scene's real 1cm grid
        // cell -- directly measured: diffusivity = (A*d)^2/t0 = 2.07e-4
        // m^2/s, giving a per-substep diffusion length of ~9.6e-5m, meaning
        // ~10,800 substeps are needed just to spread `g` ONE cell-width.
        // The whole 200-step run only has ~45,000 substeps total -- `g`
        // gets permanently stranded at whichever single cell first seeds
        // it, can never diffuse fast enough to "cooperate" with the
        // material's own moving collapse front, and the column re-freezes
        // elastically. Grain-scale cooperativity is simply invisible at
        // LP's real grid resolution.
        //
        // Real, disclosed fix, same honest precedent as `cohesion`
        // (calibrated at THIS SIMULATION's own resolution, not literal dry-
        // sand micro-physics): use an EFFECTIVE grain diameter comparable
        // to the grid cell itself (8mm vs the real 0.3mm, ~27x), giving a
        // diffusivity of ~0.15 m^2/s and ~15 substeps to cross one cell --
        // tractable within this scene's real substep budget. `mu_s`, `A`,
        // `b` stay their real, literature-cited values (dimensionless,
        // scale-invariant); only `d`'s absolute magnitude is a simulation-
        // scale calibration, not a claim about real sand grains.
        const EFFECTIVE_GRAIN_DIAMETER_M: f32 = 0.008;
        const GRAIN_DENSITY_KG_M3: f32 = 2583.0;
        // Even with the exact closed-form reaction fix (see
        // `GranularFluidityField::apply`'s own doc), the equation's own
        // analytic equilibrium genuinely diverges as real pressure -> 0: a
        // real cell at ~1e-3 Pa gives a mathematically-correct g_eq in the
        // TENS OF MILLIONS even under exact integration. Real,
        // physically-motivated floor (same justification `cohesion` already
        // documents): one grain's own hydrostatic self-weight,
        // rho_s * g_accel * d.
        let pressure_floor_pa = GRAIN_DENSITY_KG_M3 * 9.81 * EFFECTIVE_GRAIN_DIAMETER_M;
        GranularFluidityConfig {
            mu_s: 0.70, // = tan(35 deg), matches this material's own friction_angle
            grain_diameter_m: EFFECTIVE_GRAIN_DIAMETER_M,
            grain_density_kg_m3: GRAIN_DENSITY_KG_M3,
            nonlocal_amplitude: 0.48,
            b: 0.278,
            t0_s: 1.0e-4, // real, cited value again -- the closed-form reaction fix (see `GranularFluidityField::apply`'s own doc) removes the need for ad-hoc recalibration
            pressure_floor_pa,
        }
    }

    /// `resolution_scale=1` matches the original scene (GRID=96,
    /// CELL_M=0.01, 8x16-cell column). `resolution_scale=2` doubles the
    /// grid resolution (half the real cell size, double the cell counts)
    /// while keeping the REAL PHYSICAL column size identical -- the same
    /// resolution-independence discipline already used for the earlier
    /// angle-of-repose fix (confirmed at 2x resolution before trusting it).
    fn run_column_collapse(ngf_enabled: bool, resolution_scale: usize, steps: usize) -> (f32, f32) {
        let grid: usize = 96 * resolution_scale;
        let cell_m: f32 = CELL_M / resolution_scale as f32;
        const FLOOR: f32 = 0.05; // meters
        let r0_cells = 4.0_f32 * resolution_scale as f32;
        let h0_cells = 16.0_f32 * resolution_scale as f32;
        let aspect_ratio = h0_cells / r0_cells;
        let predicted_r_inf_cells = r0_cells * (1.0 + 2.0 * aspect_ratio.sqrt());

        let config = SimConfig {
            max_substeps_per_step: 4000,
            ..SimConfig::earth(grid, cell_m, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8 * resolution_scale as i32, 16 * resolution_scale as i32),
            box_center: Vec2::new(
                grid as f32 * 0.5,
                FLOOR / cell_m + 8.0 * resolution_scale as f32,
            ),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sand = DruckerPragerMaterial::from_physical(
            &GranularProps {
                elastic: Elastic {
                    e_pa: YOUNG_MODULUS_PA,
                    nu: POISSON_RATIO,
                    rho_kg_m3: BULK_DENSITY_KG_M3,
                },
                friction_angle_deg: 35.0,
                dilatancy_angle_deg: 0.0,
            },
            &config,
        );
        sand.ngf_enabled = ngf_enabled;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        if ngf_enabled {
            let field = GranularFluidityField::new(ngf_config(), ngf_pressure_and_ratio, grid);
            solver = solver.with_granular_fluidity(field);
        }

        solver.step_n(steps);

        let xs: Vec<Vec2> = solver.particles().x.clone();
        let n = xs.len() as f32;
        let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
        let measured_r_inf_cells = xs
            .iter()
            .map(|p| (p.x - center_x).abs())
            .fold(0.0f32, f32::max);
        (measured_r_inf_cells, predicted_r_inf_cells)
    }

    #[test]
    fn ngf_lajeunesse_runout_diagnostic() {
        let (baseline_r, predicted) = run_column_collapse(false, 1, 200);
        let (ngf_r, predicted2) = run_column_collapse(true, 1, 200);
        assert!((predicted - predicted2).abs() < 1e-6);

        println!("── NGF LAJEUNESSE RUNOUT, real SI throughout (diagnostic) ──");
        println!("  predicted R_inf (Lajeunesse 2004)      = {predicted:.2} cells");
        println!(
            "  measured R_inf, cohesionless (fresh baseline) = {baseline_r:.2} cells, ratio={:.2}x",
            baseline_r / predicted
        );
        println!(
            "  measured R_inf, ngf_enabled=true               = {ngf_r:.2} cells, ratio={:.2}x",
            ngf_r / predicted
        );

        assert!(
            baseline_r.is_finite() && ngf_r.is_finite(),
            "non-finite result: baseline_r={baseline_r} ngf_r={ngf_r}"
        );
    }

    /// Resolution-independence check (Phase 3 of the NGF plan): the same
    /// real physical scene, at 2x grid resolution -- a real fix must not
    /// be a resolution-specific fluke, same discipline already used for
    /// the earlier angle-of-repose fix (`sand_preshaped_pile_at_30deg_
    /// holds_its_slope`, confirmed at 2x height/2x particle density before
    /// being trusted).
    #[test]
    fn ngf_lajeunesse_runout_resolution_independence() {
        let (ngf_r_1x, predicted_1x) = run_column_collapse(true, 1, 200);
        let (ngf_r_2x, predicted_2x) = run_column_collapse(true, 2, 200);
        let ratio_1x = ngf_r_1x / predicted_1x;
        let ratio_2x = ngf_r_2x / predicted_2x;

        println!("── NGF RESOLUTION-INDEPENDENCE CHECK ──");
        println!("  ratio at 1x resolution (GRID=96)  = {ratio_1x:.3}x");
        println!("  ratio at 2x resolution (GRID=192) = {ratio_2x:.3}x");

        assert!(
            ratio_1x.is_finite() && ratio_2x.is_finite() && ratio_1x > 0.0 && ratio_2x > 0.0,
            "non-finite or zero result: ratio_1x={ratio_1x} ratio_2x={ratio_2x}"
        );
    }

    /// Does the 200-step measurement above actually reflect a SETTLED pile,
    /// or is it a snapshot mid-creep? `ngf_repose_angle_shape_diagnostic`
    /// (different scene: same column, no wall-distance headroom, 400 steps)
    /// found the pile fully flattened wall-to-wall with near-zero height at
    /// 400 steps -- this checks whether the *Lajeunesse* geometry (which
    /// gives the pile far more lateral room before it can hit a wall) does
    /// the same thing over a longer window, per the plan's own "run far
    /// longer than settling takes" requirement (never actually applied to
    /// this real-SI scene until now -- the diagnostic/resolution-
    /// independence tests above only ever ran 200 steps).
    #[test]
    fn ngf_lajeunesse_runout_long_duration_creep_check() {
        for &steps in &[200usize, 800, 2000] {
            let (baseline_r, predicted) = run_column_collapse(false, 1, steps);
            let (ngf_r, _) = run_column_collapse(true, 1, steps);
            println!(
                "steps={steps:5}: baseline ratio={:.3}x  ngf ratio={:.3}x  (predicted={predicted:.2} cells)",
                baseline_r / predicted,
                ngf_r / predicted
            );
        }
    }

    /// Does NGF change whether a pile HOLDS long-term after collapse
    /// settles, not just how far it initially spreads? `tests/accuracy.rs`'s
    /// own `sand_collapse_relaxation_long_horizon_plateau_check` (arbitrary-
    /// unit scene) found baseline DP does NOT: even under the proven
    /// "holding" damping recipe (apic_blend=0.05, cundall_damping=1.0), a
    /// dynamically-collapsed pile creeps from 29.6deg (t=1500) down to
    /// 10.8deg (t=101500), monotonic, never plateaus. NGF exists precisely
    /// to give marginal, near-yield flow a length-scale-aware arrest
    /// instead of the pointwise Coulomb model's "any nonzero shear ratio
    /// can flow forever" behavior -- this is the real, motivated test of
    /// whether it does that specific job, distinct from
    /// `ngf_lajeunesse_runout_diagnostic` (which already showed NGF only
    /// narrows initial runout ~3%, a different question: how far it gets
    /// before settling, not whether "settled" actually holds).
    ///
    /// Real-SI scene required (same reason as every other test in this
    /// module). Same column/material as `run_column_collapse`, but adds the
    /// proven holding-damping switch (`set_apic_blend`/`set_cundall_
    /// damping`, the same real API `tests/accuracy.rs` uses) after the
    /// initial collapse settles, then holds for real physical TIME (not an
    /// arbitrary step count): 30+ real seconds is already enormously long
    /// for a granular pile whose actual dynamic collapse takes well under
    /// 1 real second.
    #[test]
    fn ngf_long_horizon_hold_arrests_creep_vs_baseline() {
        fn run(ngf_enabled: bool) -> Vec<(f32, f32, f32, f32)> {
            let grid: usize = 96;
            let cell_m: f32 = CELL_M;
            const FLOOR_M: f32 = 0.05;
            let config = SimConfig {
                max_substeps_per_step: 4000,
                // apic_blend=1.0 (this config's own base default) is
                // genuinely numerically unstable for a violent dynamic
                // collapse, independent of NGF -- 0.6 is the real, bounded
                // value for the collapse phase itself, distinct from the
                // 0.05 "holding" value applied after settling below.
                apic_blend: 0.6,
                ..SimConfig::earth(grid, cell_m, 0.01)
            };
            let column = SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(8, 16),
                box_center: Vec2::new(grid as f32 * 0.5, FLOOR_M / cell_m + 8.0),
                material_id: 0,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            };
            let mut sand = DruckerPragerMaterial::from_physical(
                &GranularProps {
                    elastic: Elastic {
                        e_pa: YOUNG_MODULUS_PA,
                        nu: POISSON_RATIO,
                        rho_kg_m3: BULK_DENSITY_KG_M3,
                    },
                    friction_angle_deg: 35.0,
                    dilatancy_angle_deg: 0.0,
                },
                &config,
            );
            sand.ngf_enabled = ngf_enabled;
            let mut solver = Simulation::new(config, column)
                .with_default_material(Box::new(sand))
                .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
            if ngf_enabled {
                let field = GranularFluidityField::new(ngf_config(), ngf_pressure_and_ratio, grid);
                solver = solver.with_granular_fluidity(field);
            }

            // Real collapse dynamics settle in well under 1 real second.
            solver.step_n(200);
            solver.set_apic_blend(0.05);
            solver.set_cundall_damping(1.0);

            let mut results = Vec::new();
            let mut elapsed = 200usize;
            for &extra in &[200usize, 800, 3000] {
                solver.step_n(extra);
                elapsed += extra;
                let xs = &solver.particles().x;
                let vs = &solver.particles().v;
                // p99, not raw max: a single particle flung by the initial
                // violent corner-impact (real, expected in MPM column
                // collapse) can sit at an outlier position for a long time
                // even once bulk velocity has died down, dragging a raw
                // min/max metric far from the pile's real bulk shape --
                // same outlier-vs-bulk confound already solved once this
                // session (see `sand_collapse_with_phase_gated_relaxation_
                // after_dynamics`'s own percentile check).
                fn p99(mut v: Vec<f32>) -> f32 {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let idx = ((v.len() as f32 - 1.0) * 0.99).round() as usize;
                    v[idx.min(v.len() - 1)]
                }
                fn p01(mut v: Vec<f32>) -> f32 {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let idx = ((v.len() as f32 - 1.0) * 0.01).round() as usize;
                    v[idx.min(v.len() - 1)]
                }
                let ys: Vec<f32> = xs.iter().map(|p| p.y).collect();
                let top_y = p99(ys.clone());
                let bottom_y = p01(ys);
                let n = xs.len() as f32;
                let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
                let half_w = p99(xs.iter().map(|p| (p.x - center_x).abs()).collect());
                let max_speed = vs.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                results.push((elapsed as f32 * 0.01, top_y - bottom_y, half_w, max_speed));
            }
            results
        }

        let baseline = run(false);
        let ngf = run(true);

        println!("── NGF LONG-HORIZON HOLD, real seconds ──");
        for ((t, h, hw, vmax), (_, h2, hw2, vmax2)) in baseline.iter().zip(ngf.iter()) {
            let angle_baseline = (h / hw).atan().to_degrees();
            let angle_ngf = (h2 / hw2).atan().to_degrees();
            println!(
                "t={t:6.2}s  baseline: height={h:.2} half-w={hw:.2} angle={angle_baseline:.1}deg vmax={vmax:.3}   ngf: height={h2:.2} half-w={hw2:.2} angle={angle_ngf:.1}deg vmax={vmax2:.3}"
            );
        }
    }

    /// Does the real static/kinetic Coulomb hysteresis (`static_friction_
    /// boost`) actually arrest the long-horizon holding creep, where
    /// baseline (this exact scene, see `ngf_long_horizon_hold_arrests_
    /// creep_vs_baseline`) does not? This is the real, structural candidate
    /// found by direct inspection of `alpha`'s own math (see
    /// `static_friction_boost`'s own doc): baseline DP's hardening law
    /// asymptotes back to the SAME friction angle for large q regardless of
    /// history, giving marginal states zero safety margin. Distinct from
    /// NGF (a spatial cooperativity length scale) and from every earlier
    /// falsified hypothesis (internal-state reset, packing regularity).
    /// `rest_rate_scale`/`static_friction_boost` are both new, uncalibrated
    /// knobs -- this sweeps a few real values rather than trusting a single
    /// guess.
    #[test]
    fn static_kinetic_hysteresis_long_horizon_hold_arrests_creep() {
        fn p99(mut v: Vec<f32>) -> f32 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((v.len() as f32 - 1.0) * 0.99).round() as usize;
            v[idx.min(v.len() - 1)]
        }
        fn p01(mut v: Vec<f32>) -> f32 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((v.len() as f32 - 1.0) * 0.01).round() as usize;
            v[idx.min(v.len() - 1)]
        }

        fn run(static_boost_deg: f32, rest_rate_scale: f32) -> Vec<(f32, f32, f32, f32)> {
            let grid: usize = 96;
            let cell_m: f32 = CELL_M;
            const FLOOR_M: f32 = 0.05;
            let config = SimConfig {
                max_substeps_per_step: 4000,
                apic_blend: 0.6,
                ..SimConfig::earth(grid, cell_m, 0.01)
            };
            let column = SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(8, 16),
                box_center: Vec2::new(grid as f32 * 0.5, FLOOR_M / cell_m + 8.0),
                material_id: 0,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            };
            let mut sand = DruckerPragerMaterial::from_physical(
                &GranularProps {
                    elastic: Elastic {
                        e_pa: YOUNG_MODULUS_PA,
                        nu: POISSON_RATIO,
                        rho_kg_m3: BULK_DENSITY_KG_M3,
                    },
                    friction_angle_deg: 35.0,
                    dilatancy_angle_deg: 0.0,
                },
                &config,
            );
            sand.static_friction_boost = static_boost_deg.to_radians();
            sand.rest_rate_scale = rest_rate_scale;
            let mut solver = Simulation::new(config, column)
                .with_default_material(Box::new(sand))
                .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

            solver.step_n(200);
            solver.set_apic_blend(0.05);
            solver.set_cundall_damping(1.0);

            let mut results = Vec::new();
            let mut elapsed = 200usize;
            for &extra in &[200usize, 800, 3000] {
                solver.step_n(extra);
                elapsed += extra;
                let xs = &solver.particles().x;
                let ys: Vec<f32> = xs.iter().map(|p| p.y).collect();
                let top_y = p99(ys.clone());
                let bottom_y = p01(ys);
                let n = xs.len() as f32;
                let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
                let half_w = p99(xs.iter().map(|p| (p.x - center_x).abs()).collect());
                let height = top_y - bottom_y;
                let angle = (height / half_w).atan().to_degrees();
                results.push((elapsed as f32 * 0.01, height, half_w, angle));
            }
            results
        }

        let baseline = run(0.0, 1.0);
        let boost5 = run(5.0, 0.05);
        let boost10 = run(10.0, 0.05);
        let boost20 = run(20.0, 0.05);

        println!("── STATIC/KINETIC HYSTERESIS LONG-HORIZON HOLD, real seconds ──");
        for i in 0..baseline.len() {
            println!(
                "t={:6.2}s  base: h={:.2} hw={:.2} a={:.1}deg | +5deg: h={:.2} hw={:.2} a={:.1}deg | +10deg: h={:.2} hw={:.2} a={:.1}deg | +20deg: h={:.2} hw={:.2} a={:.1}deg",
                baseline[i].0,
                baseline[i].1,
                baseline[i].2,
                baseline[i].3,
                boost5[i].1,
                boost5[i].2,
                boost5[i].3,
                boost10[i].1,
                boost10[i].2,
                boost10[i].3,
                boost20[i].1,
                boost20[i].2,
                boost20[i].3,
            );
        }
    }
}
