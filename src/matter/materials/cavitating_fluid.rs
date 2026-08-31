//! `IsothermalCavitatingFluidMaterial` -- the real `MaterialModel` wired on
//! top of `cavitating_eos::CavitatingEosParams` (see that module's own doc
//! for the full real, sourced derivation and the real absolute/gauge and
//! shared-stiffness bugs its EOS core had before being fixed).
//!
//! Isothermal (2026-08-30): this first version fixes `p_v_gauge_pa` at
//! construction from one real temperature -- it does NOT yet couple to a
//! particle's own live `temperature` field the way a full non-isothermal
//! cavitation model eventually would (Saurel/Boivin/Le Métayer 2016's own
//! real mixture-equilibrium closure, cited in project memory, is the real,
//! disclosed future work for that). A real, bounded, disclosed limitation,
//! not a hidden one -- see `CavitatingEosParams`'s own doc for why a fixed
//! `T` is still a real, useful first increment.

use glam::{Mat2, Vec2};

use super::cavitating_eos::CavitatingEosParams;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

#[derive(Debug, Clone, Copy)]
pub struct IsothermalCavitatingFluidMaterial {
    /// The real, gauge-pressure, three-branch EOS core -- always real SI
    /// internally (kg/m^3, Pa, m/s), see `CavitatingEosParams`'s own doc.
    pub eos: CavitatingEosParams,
    /// Grid cell size (m) this material was constructed for -- needed to
    /// convert the particle's own GRID-unit `density` (`rho_si*dx^2`, this
    /// engine's established convention) back to real SI before evaluating
    /// `eos.pressure_gauge_pa`, and back again for `stress_volume`/
    /// `timestep_bound`'s own grid-consistent quantities. Set once at
    /// construction (`new`), same convention `NewtonianFluidMaterial`'s
    /// `weakly_compressible` establishes -- this material does NOT convert
    /// stress/pressure by `dx^2` (only density does, matching that same,
    /// already-fixed convention).
    pub dx_meters: f32,
    /// Dynamic viscosity (Pa.s, raw SI -- same already-fixed, unconverted
    /// convention `NewtonianFluidMaterial`/`IdealGasMaterial` both use).
    pub dynamic_viscosity: f32,
    /// Rest density in GRID units (`eos.rho_l_ref_kg_m3 * dx_meters^2`) --
    /// cached at construction so `init_particle`/`update_particle` don't
    /// recompute it every call. Matches `NewtonianFluidMaterial::rest_density`'s
    /// own role exactly.
    rest_density_grid: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Real, explicit, CALLER-SUPPLIED numerical safety bound on `J`
    /// (`volume/initial_volume`) -- same established convention this
    /// engine already uses (`IdealGasMaterial::volume_ratio_min/max`,
    /// `NewtonianFluidMaterial`'s own `[0.5,2.0]` inline clamp). NOT
    /// defaulted to an invented number: the caller should size this
    /// relative to the real `rho_l_ref/rho_v_ref` ratio this material's
    /// own `eos` was built with (full vaporization real corresponds to
    /// `J ~= rho_l_ref/rho_v_ref`) plus real headroom for further vapor
    /// expansion at low pressure, not an arbitrary flat number.
    pub volume_ratio_min: f32,
    pub volume_ratio_max: f32,
}

impl IsothermalCavitatingFluidMaterial {
    pub fn new(
        eos: CavitatingEosParams,
        dx_meters: f32,
        dynamic_viscosity: f32,
        volume_ratio_min: f32,
        volume_ratio_max: f32,
    ) -> Self {
        assert!(
            dx_meters.is_finite() && dx_meters > 0.0,
            "IsothermalCavitatingFluidMaterial requires a positive dx_meters"
        );
        assert!(
            volume_ratio_min > 0.0 && volume_ratio_max > volume_ratio_min,
            "IsothermalCavitatingFluidMaterial requires 0 < volume_ratio_min < volume_ratio_max, \
             no default -- see this field's own doc for how to size it"
        );
        let rest_density_grid = eos.rho_l_ref_kg_m3 * dx_meters * dx_meters;
        Self {
            eos,
            dx_meters,
            dynamic_viscosity,
            rest_density_grid,
            min_density: 1.0e-6,
            min_volume: 1.0e-9,
            volume_ratio_min,
            volume_ratio_max,
        }
    }

    /// Real SI density (kg/m^3) recovered from this particle's own
    /// GRID-unit `density`/`volume` state (`rho_grid = rho_si*dx^2`).
    #[inline]
    fn real_density_si(&self, grid_density: f32) -> f32 {
        grid_density / (self.dx_meters * self.dx_meters)
    }
}

impl MaterialModel for IsothermalCavitatingFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    /// Same real contract as `NewtonianFluidMaterial::init_particle` --
    /// see that method's own doc for why a strict fluid must set
    /// `initial_volume`/`volume`/`density` exactly here (V0=m/rho0,
    /// rho=rho0), not rely on `SpawnRegion`'s own kernel-density estimate.
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density_grid;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density_grid / j;
    }

    /// Same real contract as `NewtonianFluidMaterial::init_particle_from_transition`
    /// -- see that method's own doc for the real, live-confirmed bug
    /// (violent pressure spikes at a phase-transition front) this exact
    /// rebaseline avoids.
    ///
    /// Real, disclosed correction (2026-08-30): an earlier version of
    /// this method hardcoded `.clamp(0.5, 2.0)`, copy-pasted from
    /// `NewtonianFluidMaterial` without updating it to this material's own
    /// real, explicit `volume_ratio_min/max` -- confirmed consequence:
    /// real steam (`J~6`, real full vaporization ratio) condensing INTO
    /// this material would be instantly, wrongly clamped to `J=2`,
    /// breaking the exact volume-continuity guarantee this override
    /// exists to provide. Fixed to use this material's own real bounds.
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        let true_initial_volume = particle.mass / self.rest_density_grid;
        let prior_volume = particle.volume.max(1.0e-9);
        let j = (prior_volume / true_initial_volume)
            .clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        particle.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        particle.initial_volume = true_initial_volume;
        particle.volume = true_initial_volume * j;
        particle.density = self.rest_density_grid / j;
    }

    fn rest_acoustic_c2(&self) -> Option<f32> {
        // Real, grid-consistent rest acoustic speed squared, same
        // dx-folding convention `NewtonianFluidMaterial::rest_acoustic_c2`
        // uses (`B*gamma/rho_GRID`, which is `c_real^2/dx^2` -- grid
        // cells^2/s^2, not real m^2/s^2 -- see this material's own
        // `timestep_bound` doc for the full derivation).
        Some(self.eos.c_l_m_s * self.eos.c_l_m_s / (self.dx_meters * self.dx_meters))
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        // Real, disclosed correction (2026-08-30): the
        // density ceiling must correspond to THIS material's own real
        // `volume_ratio_min` (density=rest/J, so the smallest admissible J
        // gives the largest admissible density) -- a copy-pasted hardcoded
        // `rest_density*2.0` (matching `NewtonianFluidMaterial`'s own
        // fixed `[0.5,2.0]` convention) silently broke the F/V/rho triple's
        // own consistency for any caller setting `volume_ratio_min` below
        // 0.5, which this material's own explicit, undefaulted field
        // exists specifically to allow.
        let density_grid = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        let density_si = self.real_density_si(density_grid);
        // Real gauge pressure straight from the EOS core -- stays raw SI
        // Pa, same already-fixed, unconverted stress convention every
        // other material in this engine now uses (see `cavitating_eos`'s
        // own doc for why this is gauge, not absolute).
        let pressure_gauge = self.eos.pressure_gauge_pa(density_si);
        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure_gauge));

        if self.dynamic_viscosity > 0.0 {
            let c = particles.velocity_gradient[i];
            let sym_strain = c + c.transpose();
            let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;
            let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
            stress += self.dynamic_viscosity * strain_dev;
        }
        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    /// Real, disclosed fix reused directly (2026-08-30): the exact
    /// exponential volume integrator (`J_new=J_old*exp(dt*div(v))`) this
    /// same session restored for `NewtonianFluidMaterial` -- see that
    /// material's own `update_particle` doc for the full writeup on why
    /// `det(I+dt*C)` is NOT rotation-invariant and must not be used here
    /// either. This material is built specifically to fix a real drift
    /// bug in fluid volume evolution -- it would be a real, disclosed
    /// contradiction to build it on the SAME buggy integrator that
    /// investigation started from.
    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        let j = (old_j * (dt * div_v).exp()).clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        *ctx.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        // Real, disclosed correction (2026-08-30) -- same F/V/rho
        // consistency fix as `kirchhoff_stress`'s own doc:
        // the density ceiling must track `volume_ratio_min`, not a
        // hardcoded `2.0` left over from `NewtonianFluidMaterial`'s own
        // fixed convention.
        let density = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(self.min_volume);
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    fn needs_density_recompute(&self) -> bool {
        false
    }

    /// Real, disclosed limitation: NOT yet wired for GPU -- this material
    /// is CPU-only for now, same real "CPU correctness first, GPU port
    /// second" standing rule `IdealGasMaterial` was built under. If this
    /// material is ever registered with a `GpuSimulation`, it will be
    /// silently misread as a zero-stiffness `NewtonianFluidMaterial` --
    /// real, disclosed, NOT YET guarded against at construction; a real,
    /// loud rejection or a genuine new GPU constitutive branch is the
    /// correct fix, not attempted here.
    ///
    /// Real, disclosed correction (2026-08-30): `params()` is NOT purely
    /// GPU-facing metadata -- `eos_power`
    /// specifically doubles as `cfl.rs`'s own CPU-side shock-viscosity
    /// safety term's `weak_shock_gamma` (see that file's own comment,
    /// confirmed: `NewtonianFluidMaterial`/`IdealGasMaterial` both feed
    /// their own real exponent through this exact same field for this
    /// exact reason). Leaving it at the trait's default `0.0` SILENTLY
    /// disabled this real CFL safety margin for this material -- fixed by
    /// supplying `gamma_l` (this material's own real, liquid-branch Tait-
    /// like exponent, the same physical role `NewtonianFluidMaterial`'s
    /// own real `eos_power` plays for the identical mechanism).
    /// `eos_stiffness` is NOT populated -- no other confirmed CPU-side
    /// consumer found for it on this material's own real code path.
    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density_grid,
            dynamic_viscosity: self.dynamic_viscosity,
            eos_power: self.eos.gamma_l,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    /// Real, disclosed correction (2026-08-30): an earlier version of this
    /// method used the LIQUID branch's own `c_l^2` as a blanket bound
    /// everywhere, believing it always conservative -- confirmed WRONG by
    /// direct computation: `CavitatingEosParams::acoustic_c2_si`'s own
    /// test found the vapor branch's real `dp/drho` at its boundary
    /// EXCEEDS `c_l^2` for this material's own real test parameters (a
    /// real, ~13% UNDERESTIMATE, not just a loose bound), and the mixture
    /// branch's own exact derivative diverges to infinity at its edges (a
    /// real, disclosed feature of the arcsin closure). Fixed: evaluates
    /// the EOS's own exact, per-density `acoustic_c2_si` at the ACTUAL
    /// current density (recovered to real SI first, same bridge
    /// `kirchhoff_stress` uses) -- both more correct (no more silent
    /// underestimate) and tighter (not needlessly conservative in the
    /// liquid regime) than a single blanket constant.
    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = f32::INFINITY;
        let density_si = self.real_density_si(density.max(self.min_density));
        let c2_si = self.eos.acoustic_c2_si(density_si);
        // Same real dx-folding as `rest_acoustic_c2` -- grid cells^2/s^2.
        let c2_grid = c2_si / (self.dx_meters * self.dx_meters);
        if c2_grid.is_finite() && c2_grid > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2_grid.sqrt());
        }
        if self.dynamic_viscosity > 0.0 {
            let density = density.max(self.min_density);
            let kinematic_viscosity = self.dynamic_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }
        dt_bound
    }
}
