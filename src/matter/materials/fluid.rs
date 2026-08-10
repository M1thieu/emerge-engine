use glam::{Mat2, Vec2};

use crate::materials::fluid_state::{
    init_particle as init_fluid_particle, tait_pressure, update_particle as update_fluid_particle,
    volume_j,
};
use crate::materials::physical_props::{FromSI, NewtonianFluid};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Weakly-compressible Newtonian fluid discretising the barotropic
/// Navier--Stokes equations with a Tait equation of state.
///
/// The material owns `rho = rho0 / J` and `V = V0 J`.  It deliberately does
/// not remeasure either quantity from a kernel density gather: that gather is
/// useful for rendering/occupancy, but is biased at a free surface and is not
/// a conservative thermodynamic state update.
#[derive(Debug, Clone, Copy)]
pub struct NewtonianFluidMaterial {
    /// Reference areal density in solver units.  For SI construction this is
    /// `rho_kg_m3 * dx_meters^2`.
    pub rest_density: f32,
    /// Dynamic shear viscosity `mu` in the stress units used by the solver.
    pub dynamic_viscosity: f32,
    /// Tait coefficient `B`: `p = B ((rho/rho0)^gamma - 1)`.
    pub eos_stiffness: f32,
    /// Tait exponent `gamma` (usually 7 for a water-like WC model).
    pub eos_power: f32,
    /// Thermal thinning coefficient for `mu_eff = mu exp(-k T)`.
    pub thermal_viscosity_coeff: f32,
    /// Physical second (bulk) viscosity `zeta` in
    /// `tau = 2 mu D_dev + zeta div(v) I - p I`.
    pub bulk_viscosity: f32,
}

impl NewtonianFluidMaterial {
    pub const fn new(
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
            thermal_viscosity_coeff: 0.0,
            bulk_viscosity: 0.0,
        }
    }

    /// Low-viscosity water-like preset in caller-selected solver units.
    pub fn low_viscosity(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 1.0e-3, eos_stiffness, 7.0)
    }

    /// Construct a WC-MPM liquid from SI values and a stated artificial sound
    /// speed.  Solver time is already seconds and positions are grid cells, so
    /// pressure and dynamic viscosity remain in SI stress units; only density
    /// is converted to mass per grid-cell area.
    ///
    /// Choose `c_ref_m_s` from an explicit Mach target, commonly at least ten
    /// times the largest expected physical flow speed.  This is still an
    /// explicit acoustic model and therefore carries its acoustic CFL cost.
    pub fn weakly_compressible(
        rho_kg_m3: f32,
        eta_pa_s: f32,
        c_ref_m_s: f32,
        config: &crate::SimConfig,
    ) -> Self {
        const GAMMA: f32 = 7.0;
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "weakly_compressible requires a positive dx_meters"
        );
        let rho_grid = rho_kg_m3 * config.dx_meters * config.dx_meters;
        let tait_b_pa = rho_kg_m3 * c_ref_m_s * c_ref_m_s / GAMMA;
        Self::new(rho_grid, eta_pa_s, tait_b_pa, GAMMA)
    }
}

impl FromSI<NewtonianFluid> for NewtonianFluidMaterial {
    fn from_physical(props: &NewtonianFluid, config: &crate::SimConfig) -> Self {
        const GAMMA: f32 = 7.0;
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "NewtonianFluidMaterial::from_physical requires a positive dx_meters"
        );
        let rho_grid = props.rho_kg_m3 * config.dx_meters * config.dx_meters;
        // The MPM transfer uses grid-cell coordinates but real seconds.  With
        // V_grid = V_SI/dx^2 and rho_grid = rho_SI dx^2, a stress coefficient
        // in Pa produces exactly sigma/(rho dx^2) grid acceleration.  Applying
        // the legacy dt^2/(rho dx^2) conversion here would double-scale it.
        Self::new(
            rho_grid,
            props.eta_pa_s,
            props.bulk_modulus_pa / GAMMA,
            GAMMA,
        )
    }
}

impl MaterialModel for NewtonianFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = volume_j(
            particles.initial_volume[i],
            particles.volume[i],
            "NewtonianFluidMaterial",
        );
        let pressure = tait_pressure(
            self.eos_stiffness,
            self.eos_power,
            j,
            "NewtonianFluidMaterial",
        );
        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure));

        let viscosity = if self.thermal_viscosity_coeff > 0.0 {
            self.dynamic_viscosity
                * (-self.thermal_viscosity_coeff * particles.temperature[i]).exp()
        } else {
            self.dynamic_viscosity
        };
        assert!(
            viscosity.is_finite()
                && viscosity >= 0.0
                && self.bulk_viscosity.is_finite()
                && self.bulk_viscosity >= 0.0,
            "NewtonianFluidMaterial: viscosities must be finite and nonnegative"
        );

        let gradient = particles.velocity_gradient[i];
        let twice_d = gradient + gradient.transpose();
        let twice_div = twice_d.x_axis.x + twice_d.y_axis.y;
        let twice_d_dev = twice_d - Mat2::from_diagonal(Vec2::splat(0.5 * twice_div));
        stress += viscosity * twice_d_dev;
        stress += Mat2::from_diagonal(Vec2::splat(0.5 * self.bulk_viscosity * twice_div));

        // Real, sourced (von Neumann-Richtmyer + Landshoff, see
        // fluid_state::artificial_bulk_viscosity's own doc) numerical
        // stabilizer for extreme local compression -- root-caused
        // 2026-08-08 against basic_fluids_gpu.rs's real crash (a genuine MPM
        // cell-crossing-style instability, CPU/GPU chaotically diverging
        // under a violent wall impact). `h=grid_cell_size` hardcoded to 1.0:
        // `kirchhoff_stress`'s trait signature has no config/grid_cell_size
        // parameter to thread it through (a real, disclosed scope limit --
        // every scene in this engine currently uses grid_cell_size=1.0, so
        // this is exactly correct today, not an approximation of a
        // different value).
        let q = crate::materials::fluid_state::artificial_bulk_viscosity(
            self.eos_stiffness,
            self.eos_power,
            self.rest_density,
            j,
            0.5 * twice_div,
            1.0,
        );
        stress += Mat2::from_diagonal(Vec2::splat(-q));
        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        let _ = volume_j(
            particles.initial_volume[i],
            particles.volume[i],
            "NewtonianFluidMaterial",
        );
        particles.volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        update_fluid_particle(ctx, dt, self.rest_density, "NewtonianFluidMaterial");
    }

    fn init_particle(&self, particle: &mut Particle) {
        init_fluid_particle(particle, self.rest_density, "NewtonianFluidMaterial");
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            dynamic_viscosity: self.dynamic_viscosity,
            thermal_viscosity_coeff: self.thermal_viscosity_coeff,
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
            "NewtonianFluidMaterial: density state must be finite and positive"
        );
        let density_ratio = density / self.rest_density;
        let mut dt_bound = f32::INFINITY;

        let c2 = self.eos_stiffness
            * self.eos_power
            * crate::materials::utils::fast_pow(density_ratio, self.eos_power - 1.0)
            / self.rest_density;
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        let total_viscosity = self.dynamic_viscosity + self.bulk_viscosity;
        if total_viscosity > 0.0 {
            let kinematic_viscosity = total_viscosity / density;
            if kinematic_viscosity.is_finite() && kinematic_viscosity > f32::EPSILON {
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
mod tests {
    use super::*;

    #[test]
    fn si_constructor_preserves_pressure_and_viscosity_units() {
        let cfg = crate::SimConfig::earth(32, 0.01, 0.1);
        let material = NewtonianFluidMaterial::weakly_compressible(1000.0, 1.0e-3, 20.0, &cfg);
        assert!((material.rest_density - 0.1).abs() < 1.0e-7);
        assert!((material.dynamic_viscosity - 1.0e-3).abs() < 1.0e-9);
        assert!((material.eos_stiffness - (1000.0 * 20.0 * 20.0 / 7.0)).abs() < 1.0e-3);
    }

    #[test]
    fn initial_volume_and_density_are_mass_conservative() {
        let material = NewtonianFluidMaterial::new(4.0, 0.0, 10.0, 7.0);
        let mut particle = Particle {
            mass: 2.0,
            ..Particle::zeroed()
        };
        material.init_particle(&mut particle);
        assert!((particle.initial_volume - 0.5).abs() < 1.0e-6);
        assert!((particle.volume - 0.5).abs() < 1.0e-6);
        assert!((particle.density - 4.0).abs() < 1.0e-6);
    }

    #[test]
    fn exponential_volume_update_is_not_j_clamped() {
        let material = NewtonianFluidMaterial::new(4.0, 0.0, 10.0, 7.0);
        let mut particles = Particles::from(vec![Particle {
            mass: 2.0,
            ..Particle::zeroed()
        }]);
        // Initialise, then prescribe div(v) = -ln(4).
        let mut p = particles.get(0);
        material.init_particle(&mut p);
        particles.set(0, p);
        particles.velocity_gradient[0] = Mat2::from_diagonal(Vec2::splat(-0.5 * 4.0_f32.ln()));
        material.update_particle(&mut particles.update_ctx(0), 1.0);
        let j = particles.deformation_gradient[0].determinant();
        assert!(
            (j - 0.25).abs() < 1.0e-6,
            "J must follow exp(integral div v), got {j}"
        );
        assert!((particles.volume[0] - 0.125).abs() < 1.0e-6);
        assert!((particles.density[0] - 16.0).abs() < 1.0e-5);
    }

    #[test]
    fn tait_eos_is_not_pressure_clamped_in_tension() {
        let pressure = tait_pressure(10.0, 2.0, 2.0, "test");
        assert!(
            pressure < 0.0,
            "an expanded barotropic state must retain its Tait tension instead of being pressure-clamped"
        );
        assert!((pressure + 7.5).abs() < 1.0e-6);
    }

    #[test]
    #[should_panic(expected = "Tait pressure is unrepresentable")]
    fn tait_eos_rejects_unrepresentable_compression_instead_of_capping_it() {
        let _ = tait_pressure(1.0, 7.0, 1.0e-10, "test");
    }
}
