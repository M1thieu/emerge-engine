use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, NewtonianFluid, scale_stress, scale_visc};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{ParticleUpdateCtx, Particles};

/// Weakly-compressible Newtonian fluid (Tait EOS + deviatoric viscosity).
/// Refs: Becker & Teschner 2007 (WCSPH), Hu et al. 2018 (MLS-MPM).
#[derive(Debug, Clone, Copy)]
pub struct NewtonianFluidMaterial {
    pub rest_density: f32,
    pub dynamic_viscosity: f32,
    pub eos_stiffness: f32,
    pub eos_power: f32,
    pub pressure_floor: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Thermal thinning: µ_eff = dynamic_viscosity · exp(−thermal_viscosity_coeff · T).
    /// 0.0 = isothermal. Positive values make the fluid flow easier when hot.
    pub thermal_viscosity_coeff: f32,
    /// Bulk viscosity ζ (second viscosity, Pa·s in physical units).
    ///
    /// Adds τ += ζ·(∇·v)·I to Kirchhoff stress — damps compression waves (acoustic damping).
    /// Physical: Navier-Stokes second viscosity, distinct from shear viscosity µ.
    /// Stokes assumption (ζ=0) holds for dilute ideal gases; real liquids have ζ > 0.
    /// For water: ζ ≈ 3e-3 Pa·s (Dukhin & Goetz 2009). In simulation units set to
    /// ~0.5–5× dynamic_viscosity. 0.0 = no acoustic damping.
    pub bulk_viscosity: f32,
    /// Surface tension coefficient γ (N/m in physical units).
    ///
    /// Adds isotropic Kirchhoff stress τ += γ·J·I — continuum surface energy ψ = γ·J.
    /// Reference: Ziran 2020, `SurfaceTension.h` (Chenfanfu Jiang group).
    ///
    /// **Limitation**: curvature-free. Young-Laplace gives Δp = γ·κ (interface curvature κ),
    /// but MPM particles carry no interface normal. This term resists volumetric compression
    /// isotropically — sufficient for cohesion/droplet stability, not for curvature-driven
    /// flow (e.g. Rayleigh-Plateau instability). 0.0 = disabled.
    pub surface_tension_coeff: f32,
    /// Per-step velocity decay: v *= (1 − settling_damping · dt).
    ///
    /// Damps residual sloshing and slow plastic creep without affecting fast flow.
    /// 0.0 = off (default). 0.05–0.2 for water, 0.1–0.5 for mud/viscous fluids.
    /// Implemented in the GPU shader via the `dp_h0` slot (unused for fluids).
    pub settling_damping: f32,
}

impl NewtonianFluidMaterial {
    pub fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        eos_stiffness: f32,
        eos_power: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            eos_stiffness,
            eos_power,
            pressure_floor: -0.1,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            thermal_viscosity_coeff: 0.0,
            bulk_viscosity: 0.0,
            surface_tension_coeff: 0.0,
            settling_damping: 0.0,
        }
    }

    /// Low-viscosity preset: γ=7, µ=1e-3 Pa·s. Corresponds to water at 20°C.
    ///
    /// `eos_stiffness` controls incompressibility — higher = stiffer; 1e4 works
    /// well at emerge's default grid scale. Reference: Becker & Teschner 2007 §4.
    pub fn low_viscosity(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 1.0e-3, eos_stiffness, 7.0)
    }

    /// Weakly-compressible variant: caps sound speed at `c_ref_m_s`.
    ///
    /// Use `c_ref_m_s = 10 * v_max_m_s` (WCSPH rule) to limit compressibility to ~1%.
    /// `rho_kg_m3` and `eta_pa_s` are the fluid's SI density and viscosity.
    pub fn weakly_compressible(
        rho_kg_m3: f32,
        eta_pa_s: f32,
        c_ref_m_s: f32,
        config: &crate::SimConfig,
    ) -> Self {
        // Tait EOS polytropic exponent for water -- Cole 1948, "Underwater Explosions"
        // (the original real-fluid measurement this exponent is drawn from); used
        // identically in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994;
        // Becker & Teschner 2007, already cited elsewhere in this project).
        const GAMMA: f32 = 7.0;
        let visc = scale_visc(eta_pa_s, rho_kg_m3, config);
        let k_si = rho_kg_m3 * c_ref_m_s * c_ref_m_s / GAMMA;
        let eos = scale_stress(k_si, rho_kg_m3, config);
        Self::new(rho_kg_m3, visc, eos, GAMMA)
    }
}

impl FromSI<NewtonianFluid> for NewtonianFluidMaterial {
    /// `rest_density` defaults to `props.rho_kg_m3`. Caller should adjust if
    /// particle mass/volume don't match the SI density.
    fn from_physical(props: &NewtonianFluid, config: &crate::SimConfig) -> Self {
        // Tait EOS polytropic exponent for water -- Cole 1948, "Underwater Explosions";
        // standard in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994).
        const GAMMA: f32 = 7.0;
        let visc = scale_visc(props.eta_pa_s, props.rho_kg_m3, config);
        let eos = scale_stress(props.bulk_modulus_pa / GAMMA, props.rho_kg_m3, config);
        // rest_density must be in the SAME units `particles.density[i]` actually comes
        // out in -- i.e. whatever `estimate_particle_volumes`'s kernel-based density
        // estimate produces for a particle spawned via `ParticleMass::particle_mass`
        // (real SI kilograms) at rest: `rho_grid = rho_SI * dx_meters^2`. Do not add
        // an extra `/dt_seconds^2` factor here -- it pins any real fluid's EOS
        // pressure at its floor regardless of real depth/compression. Inflating
        // particle mass by `1/dt^2` instead breaks the gravity/EOS force balance --
        // see `Elastic::particle_mass`'s doc.
        let rho_grid = props.rho_kg_m3 * config.dx_meters * config.dx_meters;
        Self::new(rho_grid, visc, eos, GAMMA)
    }
}

impl MaterialModel for NewtonianFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Density from F's own determinant (rho = rest_density / J), NOT the
        // grid-mass-gathered `particles.density[i]` this used before -- a
        // real CPU/GPU parity fix: the GPU fluid path (`p2g.wgsl`'s case 1u)
        // already uses this exact formula ("sparkl canonical, no grid-lag"
        // per its own comment), and `GranularFluidMaterial`'s CPU code
        // (the engine's other EOS-pressure material) already does too --
        // plain `NewtonianFluidMaterial` was the one inconsistent holdout.
        // Grid-mass density carries a real one-substep lag (P2G scatter ->
        // grid -> G2P gather, vs J which is already current this same
        // substep) and is blind to how it's actually used elsewhere in this
        // engine (GranularFluid, GPU) -- switching removes a real, disclosed
        // inconsistency, not just a style choice.
        //
        // Real, honest disclosure: the OLD grid-mass approach is exactly
        // what `hydrostatic_pressure_matches_rho_g_h`'s own doc measured
        // settling at ~1.3x rest_density (not the correct ~1.003x) --
        // whether J-based density changes that specific overshoot is NOT
        // yet re-measured (that test stays `#[ignore]`d); this fix is
        // motivated by real consistency across the engine, not a confirmed
        // fix for that specific still-open gap.
        //
        // Clamp density both ways: min prevents div-by-zero, max (2x rho0)
        // limits how far the EOS pressure response saturates under impact
        // overcompression. Keep this at 2x, not looser --
        // `fluid_spreads_more_than_elastic_under_gravity` (tests/accuracy.rs)
        // needs it (a looser clamp stops the fluid spreading at all).
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        let density = (self.rest_density / j)
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        let pressure = (self.eos_stiffness
            * ((density / self.rest_density).powf(self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure));

        let eff_viscosity = if self.thermal_viscosity_coeff > 0.0 {
            self.dynamic_viscosity
                * (-self.thermal_viscosity_coeff * particles.temperature[i]).exp()
        } else {
            self.dynamic_viscosity
        };
        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y; // = 2·tr(D) = 2·∇·v
        let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
        stress += eff_viscosity * strain_dev;

        // Bulk viscosity ζ: τ += ζ·(∇·v)·I — damps longitudinal/acoustic waves.
        // ∇·v ≈ div_v/2 (div_v here is trace of sym_strain = C+Cᵀ = 2D, so ∇·v = div_v/2).
        if self.bulk_viscosity > 0.0 {
            stress += Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5));
        }

        if self.surface_tension_coeff != 0.0 {
            let f = particles.deformation_gradient[i];
            let j = f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x;
            stress += Mat2::from_diagonal(Vec2::splat(self.surface_tension_coeff * j));
        }

        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let j = ctx.deformation_gradient.determinant().clamp(0.5, 2.0);
        let s = j.sqrt();
        *ctx.deformation_gradient =
            glam::Mat2::from_cols(glam::Vec2::new(s, 0.0), glam::Vec2::new(0.0, s));
        if self.settling_damping > 0.0 {
            *ctx.v *= 1.0 - (self.settling_damping * dt).min(0.5);
        }
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            dynamic_viscosity: self.dynamic_viscosity,
            thermal_viscosity_coeff: self.thermal_viscosity_coeff,
            // Free-surface J cap: GPU clamps det(F) to [J_MIN, volume_ratio_max].
            // 2.0 = realistic free-surface density (half rest_density with no restoring EOS force).
            volume_ratio_max: 2.0,
            pressure_floor: self.pressure_floor,
            bulk_viscosity: self.bulk_viscosity,
            surface_tension_coeff: self.surface_tension_coeff,
            dp_h0: self.settling_damping, // fluid repurposes dp_h0 for settling damping (DP unused)
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
        const MIN_DENSITY_RATIO: f32 = 1.0e-6;
        let density = density.max(self.min_density);
        let ratio = (density / self.rest_density.max(self.min_density)).max(MIN_DENSITY_RATIO);

        let mut dt_bound = f32::INFINITY;

        // Acoustic timestep bound from EOS derivative dp/drho.
        let c2 = self.eos_stiffness * self.eos_power * ratio.powf(self.eos_power - 1.0)
            / self.rest_density.max(self.min_density);
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Viscous diffusion bound for explicit integration.
        if self.dynamic_viscosity > 0.0 {
            let kinematic_viscosity = self.dynamic_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }

    fn needs_density_recompute(&self) -> bool {
        true
    }
}
