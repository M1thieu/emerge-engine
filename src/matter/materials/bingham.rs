use glam::{Mat2, Vec2};

use crate::materials::fluid_state::{
    init_particle as init_fluid_particle, tait_pressure, update_particle as update_fluid_particle,
    volume_j,
};
use crate::materials::physical_props::{BinghamProps, FromSI};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Bingham viscoplastic fluid.
///
/// Below yield stress τ₀: rigid plug (no deviatoric flow).
/// Above yield stress τ₀: Newtonian with apparent viscosity η_app = τ₀/γ̇ + η.
///
/// Stress decomposition: σ = −p·I + τ_deviatoric
/// Pressure: Tait EOS — p = k·((ρ/ρ₀)^γ − 1), same as NewtonianFluid.
/// Deviatoric:
///   γ̇ = √(2·D_dev:D_dev)   (scalar shear rate, D_dev = deviatoric part of D)
///   τ = 2(τ₀/γ̇ + η)·D_dev if γ̇ > critical_shear_rate, else 0
///
/// Reference: Bingham 1916. MPM formulation: GeoTaichi BinghamModel (Taichi lang).
///
/// # Natural phenomena
/// - Mud / wet clay: τ₀ = 50–500 Pa, η = 0.1–5 Pa·s
/// - Lava (basaltic): τ₀ = 100–2000 Pa, η = 10–10000 Pa·s
/// - Biological cytoplasm: τ₀ ≈ 0.5–5 Pa, η ≈ 0.005–0.05 Pa·s
/// - Dense biological fluids (mucus, blood clot): τ₀ = 1–50 Pa, η = 0.01–1 Pa·s
#[derive(Debug, Clone, Copy)]
pub struct BinghamFluidMaterial {
    pub rest_density: f32,
    /// Dynamic viscosity η (Pa·s) — slope of stress-rate curve above yield.
    pub dynamic_viscosity: f32,
    /// Tait EOS stiffness k (pressure scale factor).
    pub eos_stiffness: f32,
    /// Tait EOS exponent γ (7 for water-like, 1 for linear).
    pub eos_power: f32,
    /// Yield stress τ₀ — shear stress required to initiate flow.
    /// Below this, deviatoric stress is zero (plug flow).
    pub yield_stress: f32,
    /// Papanastasiou regularization scale for the yield transition: a
    /// numerical parameter, not a physical shear rate, controlling how
    /// sharply apparent viscosity ramps from near-rigid to true Bingham
    /// (`tau0/shear_rate + eta`) as shear rate grows. Smaller = closer to
    /// true (discontinuous) Bingham but a stiffer, more expensive worst-case
    /// CFL bound (`timestep_bound` must budget for apparent viscosity up to
    /// `tau0/critical_shear_rate`); larger = cheaper/more stable but a
    /// softer, less sharp yield onset. Default: 1e-2, not the traditional
    /// "avoid literal division by zero" 1e-4 -- that value makes the CFL's
    /// honest worst-case viscosity 100x more pessimistic than this one for
    /// no corresponding gain in how well the plug/flow transition is
    /// resolved, which is what pinned a real demo's substep to ~1e-5s
    /// (investigated 2026-08-06, see `deviatoric_stress`'s doc).
    pub critical_shear_rate: f32,
    /// Physical second viscosity ζ in `τ = 2μD_dev + ζ div(v)I - pI`.
    pub bulk_viscosity: f32,
}

impl BinghamFluidMaterial {
    pub const fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        eos_stiffness: f32,
        eos_power: f32,
        yield_stress: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            eos_stiffness,
            eos_power,
            yield_stress,
            critical_shear_rate: 1.0e-2,
            bulk_viscosity: 0.0,
        }
    }

    /// High yield stress, low viscosity: τ₀=100 Pa, η=0.5 Pa·s. Wet mud regime.
    pub fn high_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.5, eos_stiffness, 7.0, 100.0)
    }

    /// High yield stress, high viscosity: τ₀=1000 Pa, η=500 Pa·s. Basaltic lava regime.
    pub fn viscous_high_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 500.0, eos_stiffness, 7.0, 1000.0)
    }

    /// Low yield stress, low viscosity: τ₀=1 Pa, η=0.01 Pa·s. Biological cytoplasm regime.
    pub fn low_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.01, eos_stiffness, 7.0, 1.0)
    }

    /// Medium yield stress, medium viscosity: τ₀=10 Pa, η=0.1 Pa·s. Dense biological fluid regime.
    pub fn medium_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.1, eos_stiffness, 7.0, 10.0)
    }

    /// Compute deviatoric Bingham stress from the APIC velocity gradient C.
    ///
    /// D = (C + Cᵀ)/2 (symmetric strain rate)
    /// γ̇ = √(2·D_dev:D_dev) (scalar shear rate — deviatoric only: a yield criterion
    /// must not respond to pure volumetric expansion/compression, which isn't shear)
    /// Papanastasiou-regularized: smoothly bounded near rest, converges to the
    /// true Bingham law (τ₀/γ̇ + η)·D_dev once γ̇ well exceeds `critical_shear_rate`.
    fn deviatoric_stress(&self, c: Mat2) -> Mat2 {
        assert!(
            self.dynamic_viscosity.is_finite()
                && self.dynamic_viscosity >= 0.0
                && self.yield_stress.is_finite()
                && self.yield_stress >= 0.0
                && self.critical_shear_rate.is_finite()
                && self.critical_shear_rate >= 0.0,
            "BinghamFluidMaterial: viscosity, yield stress, and critical shear rate must be finite and nonnegative"
        );
        assert!(
            self.yield_stress == 0.0 || self.critical_shear_rate > 0.0,
            "BinghamFluidMaterial: positive yield stress requires a positive critical_shear_rate for an explicit resolved-rate model"
        );
        // Symmetric strain rate D = (C + Cᵀ) / 2
        let sym = c + c.transpose();
        let d = sym * 0.5;

        // Deviatoric: remove isotropic part
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));

        // Scalar shear rate γ̇ = √(2·D_dev:D_dev) — Frobenius norm of deviatoric D, scaled.
        let d_xx = d_dev.x_axis.x;
        let d_yy = d_dev.y_axis.y;
        let d_xy = d_dev.x_axis.y; // = d_dev.y_axis.x for symmetric D
        let d_sq = d_xx * d_xx + d_yy * d_yy + 2.0 * d_xy * d_xy;
        let shear_rate = (2.0 * d_sq).sqrt();

        if self.yield_stress == 0.0 {
            return d_dev * (2.0 * self.dynamic_viscosity);
        }

        // Papanastasiou (1987) exponential regularization, replacing a hard
        // rigid/flowing switch at critical_shear_rate with a smooth ramp.
        // A piecewise cutoff makes a particle sitting near the yield surface
        // flip between "rigid" (zero stress) and "flowing" (apparent
        // viscosity up to tau0/critical_shear_rate) every substep -- with
        // nothing clamping the resulting velocity spike, that flip becomes a
        // non-decaying oscillation instead of a transient (see the mud/water
        // lock this fixes, investigated 2026-08-06).
        //
        // eta_app(g) = tau0 * (1 - exp(-m*g))/g + eta, m = 1/critical_shear_rate.
        // As g -> 0, (1-exp(-m*g))/g -> m (L'Hopital), so eta_app stays finite
        // at tau0*m + eta instead of the true formula's 1/g singularity. As
        // g grows past critical_shear_rate, it converges to the real Bingham
        // law tau0/g + eta. `exp_m1` avoids the 1-exp(-x) cancellation error
        // for small x.
        let m = 1.0 / self.critical_shear_rate;
        let ramp = if shear_rate > f32::EPSILON {
            -(-m * shear_rate).exp_m1() / shear_rate
        } else {
            m
        };
        let eta_app = self.yield_stress * ramp + self.dynamic_viscosity;
        // Cauchy stress is 2*eta_app*D_dev because D=(grad(v)+grad(v)^T)/2.
        d_dev * (2.0 * eta_app)
    }
}

impl FromSI<BinghamProps> for BinghamFluidMaterial {
    fn from_physical(props: &BinghamProps, config: &crate::SimConfig) -> Self {
        // Tait EOS polytropic exponent -- Cole 1948, "Underwater Explosions"; standard
        // in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994). Applies to the
        // volumetric/EOS part of a Bingham fluid same as any other weakly-compressible
        // liquid; the yield-stress physics (tau0 below) is separate and unaffected.
        const GAMMA: f32 = 7.0;
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "BinghamFluidMaterial::from_physical requires a positive dx_meters"
        );
        // See `NewtonianFluidMaterial::from_physical`'s doc -- rest_density
        // must match `particles.density[i]`'s real units, not an extra `/dt_seconds^2`.
        let rho_grid = props.rho_kg_m3 * config.dx_meters * config.dx_meters;
        Self::new(
            rho_grid,
            props.eta_pa_s,
            props.bulk_modulus_pa / GAMMA,
            GAMMA,
            props.yield_stress_pa,
        )
    }
}

impl MaterialModel for BinghamFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Pressure from the same unmodified Tait EOS as NewtonianFluid.
        let j = volume_j(
            particles.initial_volume[i],
            particles.volume[i],
            "BinghamFluidMaterial",
        );
        let pressure = tait_pressure(
            self.eos_stiffness,
            self.eos_power,
            j,
            "BinghamFluidMaterial",
        );

        let hydrostatic = Mat2::from_diagonal(Vec2::splat(-pressure));
        let deviatoric = self.deviatoric_stress(particles.velocity_gradient[i]);

        // Bulk viscosity ζ: τ += ζ·(∇·v)·I — damps the acoustic/volumetric
        // oscillation mode, same real mechanism as
        // `NewtonianFluidMaterial::kirchhoff_stress` (see that file's own doc).
        let bulk = if self.bulk_viscosity > 0.0 {
            let c = particles.velocity_gradient[i];
            let sym_strain = c + c.transpose();
            let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;
            Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5))
        } else {
            Mat2::ZERO
        };

        assert!(
            self.bulk_viscosity.is_finite() && self.bulk_viscosity >= 0.0,
            "BinghamFluidMaterial: bulk_viscosity must be finite and nonnegative"
        );

        // Real, sourced (von Neumann-Richtmyer + Landshoff, see
        // fluid_state::artificial_bulk_viscosity's own doc) numerical
        // stabilizer for extreme local compression -- same real mechanism
        // as NewtonianFluidMaterial::kirchhoff_stress, root-caused
        // 2026-08-08 against basic_fluids_gpu.rs's real crash. `h=1.0`:
        // same disclosed scope limit as that file (no config/grid_cell_size
        // threaded through this trait method; every scene tonight uses
        // grid_cell_size=1.0, so this is exact today, not approximate).
        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = 0.5 * (sym_strain.x_axis.x + sym_strain.y_axis.y);
        let q = crate::materials::fluid_state::artificial_bulk_viscosity(
            self.eos_stiffness,
            self.eos_power,
            self.rest_density,
            j,
            div_v,
            1.0,
        );
        let artificial = Mat2::from_diagonal(Vec2::splat(-q));

        hydrostatic + deviatoric + bulk + artificial
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        let _ = volume_j(
            particles.initial_volume[i],
            particles.volume[i],
            "BinghamFluidMaterial",
        );
        particles.volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        update_fluid_particle(ctx, dt, self.rest_density, "BinghamFluidMaterial");
    }

    fn init_particle(&self, particle: &mut Particle) {
        init_fluid_particle(particle, self.rest_density, "BinghamFluidMaterial");
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            dynamic_viscosity: self.dynamic_viscosity,
            compression_limit: self.yield_stress,
            critical_shear_rate: self.critical_shear_rate,
            bulk_viscosity: self.bulk_viscosity,
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
        assert!(
            density.is_finite()
                && density > 0.0
                && self.rest_density.is_finite()
                && self.rest_density > 0.0,
            "BinghamFluidMaterial: density state must be finite and positive"
        );
        let ratio = density / self.rest_density;

        let mut dt_bound = f32::INFINITY;

        // Acoustic bound from EOS
        let c2 = self.eos_stiffness
            * self.eos_power
            * crate::materials::utils::fast_pow(ratio, self.eos_power - 1.0)
            / self.rest_density;
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Explicit diffusion must resolve the largest admissible Bingham
        // apparent viscosity, not only the post-yield slope.  At the cutoff,
        // eta_app = eta + tau_y / critical_shear_rate; omitting this term
        // makes a high-yield plug take an unstable timestep exactly as it
        // starts to flow.
        assert!(
            self.dynamic_viscosity.is_finite()
                && self.dynamic_viscosity >= 0.0
                && self.bulk_viscosity.is_finite()
                && self.bulk_viscosity >= 0.0
                && self.yield_stress.is_finite()
                && self.yield_stress >= 0.0
                && self.critical_shear_rate.is_finite()
                && self.critical_shear_rate >= 0.0
                && (self.yield_stress == 0.0 || self.critical_shear_rate > 0.0),
            "BinghamFluidMaterial: explicit CFL requires finite nonnegative viscosities and a positive cutoff for positive yield stress"
        );
        let yield_viscosity = if self.yield_stress > 0.0 {
            self.yield_stress / self.critical_shear_rate
        } else {
            0.0
        };
        let total_viscosity = self.dynamic_viscosity + self.bulk_viscosity + yield_viscosity;
        if total_viscosity > 0.0 {
            let kinematic_viscosity = total_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod analytical_validation_tests {
    use super::*;

    /// Papanastasiou regularization must stay *finite and bounded* as shear
    /// rate -> 0 (the true Bingham law's `tau0/shear_rate` term is singular
    /// there) -- not exactly zero, since a hard zero is exactly the
    /// discontinuity this regularization replaces (see `deviatoric_stress`'s
    /// doc for why that discontinuity locked a real demo, 2026-08-06).
    #[test]
    fn near_rest_deviatoric_stress_is_small_and_finite_not_singular() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        let c = Mat2::from_cols(Vec2::new(0.0, 1.0e-6), Vec2::new(1.0e-6, 0.0));
        let tau = mat.deviatoric_stress(c);
        assert!(
            tau.x_axis.x.is_finite() && tau.y_axis.y.is_finite() && tau.x_axis.y.is_finite(),
            "near-rest deviatoric stress must be finite, not a 1/shear_rate singularity: {tau:?}"
        );
        // Bounded by the regularized ceiling 2*(tau0/critical_shear_rate + eta)*shear_rate,
        // not by the unregularized (and here enormous) tau0/shear_rate ceiling.
        let ceiling =
            2.0 * (mat.yield_stress / mat.critical_shear_rate + mat.dynamic_viscosity) * 1.0e-6;
        assert!(
            tau.x_axis.y.abs() <= ceiling * 1.01,
            "near-rest stress {} must not exceed the regularized ceiling {ceiling}",
            tau.x_axis.y
        );
    }

    /// The whole point of Papanastasiou regularization: no jump at
    /// `critical_shear_rate`. A hard cutoff there is what let a particle
    /// sitting near the yield surface flip between near-zero and
    /// `tau0/critical_shear_rate` every substep -- a real, non-decaying lock
    /// this fix removes (investigated 2026-08-06).
    #[test]
    fn deviatoric_stress_is_continuous_across_critical_shear_rate() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        let g0 = mat.critical_shear_rate * 0.999 * 0.5; // shear_rate = 2*g, just below
        let g1 = mat.critical_shear_rate * 1.001 * 0.5; // just above
        let tau0 = mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, g0), Vec2::new(g0, 0.0)));
        let tau1 = mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, g1), Vec2::new(g1, 0.0)));
        let jump = (tau1.x_axis.y - tau0.x_axis.y).abs();
        assert!(
            jump < 0.01 * mat.yield_stress,
            "stress must not jump discontinuously across critical_shear_rate: tau0={tau0:?} tau1={tau1:?}"
        );
    }

    #[test]
    fn above_yield_matches_bingham_formula_exactly() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        // Pure shear strain rate: C = [[0, g], [g, 0]] gives D=C (already symmetric),
        // D_dev=D (already traceless), d_xx=d_yy=0, d_xy=g, d_sq=2*g^2,
        // shear_rate=sqrt(2*2*g^2)=2*g (real, hand-derivable from the formula).
        let g = 5.0_f32;
        let c = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0));
        let tau = mat.deviatoric_stress(c);

        let shear_rate = 2.0 * g;
        let eta_app = mat.yield_stress / shear_rate + mat.dynamic_viscosity;
        let d_dev = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0)); // D_dev = D here
        let predicted = d_dev * (2.0 * eta_app);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        assert!(
            err < 1.0e-3,
            "above-yield deviatoric stress should match tau0/gamma_dot+eta exactly: \
             predicted={predicted:?} actual={tau:?}"
        );
    }

    /// Real, checkable monotonic claim: apparent viscosity (and thus deviatoric
    /// stress magnitude at a FIXED shear rate) must DECREASE as shear rate
    /// increases -- shear-thinning behavior intrinsic to the Bingham model
    /// (tau0/gamma_dot term shrinks as gamma_dot grows), not an assumption.
    #[test]
    fn apparent_viscosity_decreases_as_shear_rate_increases() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        let tau_slow =
            mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)));
        let tau_fast =
            mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, 10.0), Vec2::new(10.0, 0.0)));

        // Stress DOES grow with shear rate overall (more strain rate -> more
        // stress), but the EFFECTIVE viscosity (stress/shear_rate) must shrink --
        // check the ratio, not the raw magnitude.
        let eff_visc_slow = tau_slow.x_axis.y / 2.0; // shear_rate=2*g=2 here
        let eff_visc_fast = tau_fast.x_axis.y / 20.0; // shear_rate=2*g=20 here
        assert!(
            eff_visc_fast < eff_visc_slow,
            "apparent viscosity must decrease as shear rate increases (shear-thinning): \
             slow={eff_visc_slow:.4} fast={eff_visc_fast:.4}"
        );
    }

    #[test]
    fn explicit_cfl_includes_maximum_resolved_bingham_viscosity() {
        let mut yielded = BinghamFluidMaterial::new(1000.0, 0.5, 0.0, 7.0, 100.0);
        yielded.critical_shear_rate = 0.5;
        let mut no_yield = yielded;
        no_yield.yield_stress = 0.0;

        let yielded_dt = yielded.timestep_bound(1000.0, 0.0, 1.0, 0.5, 0.5);
        let no_yield_dt = no_yield.timestep_bound(1000.0, 0.0, 1.0, 0.5, 0.5);
        assert!(
            yielded_dt < no_yield_dt,
            "the CFL limit must include tau_y/critical_shear_rate: yielded={yielded_dt}, no_yield={no_yield_dt}"
        );
    }
}
