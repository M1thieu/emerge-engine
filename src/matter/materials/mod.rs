pub mod bingham;
pub mod corotated;
pub mod elastic;
pub mod fluid;
pub mod granular_fluid;
pub mod nacc;
pub mod no_compression;
pub mod params;
pub mod physical_props;
mod property_dispatch;
pub mod rankine;
pub mod registry;
pub mod sand;
pub mod sand_mui;
pub mod snow;
pub(crate) mod svd;
pub mod utils;
pub mod viscoelastic;
pub mod von_mises;

pub use physical_props::{
    BrittleProps, Elastic, Elastoplastic, Fluid, FluidGranular, FromSI, NoCompression,
    ParticleMass, PlasticityModel, Pressurized, Viscoelastic,
};

pub use bingham::BinghamFluidMaterial;
pub use corotated::CorotatedMaterial;
pub use elastic::NeoHookeanMaterial;
pub use fluid::NewtonianFluidMaterial;
pub use granular_fluid::GranularFluidMaterial;
pub use nacc::NaccMaterial;
pub use no_compression::NoCompressionMaterial;
pub use params::MaterialParams;
pub use rankine::RankineMaterial;
pub use registry::{MAX_MATERIAL_SLOTS, MaterialRegistry};
pub use sand::DruckerPragerMaterial;
pub use sand_mui::MuIRheologyMaterial;
pub use snow::StomakhinMaterial;
pub use utils::{
    elastic_wave_dt, gravity_to_grid, lame_from_si, lame_from_young, polar_decomposition_2d,
    rankine_damage_estimate,
};
pub use viscoelastic::ViscoelasticMaterial;
pub use von_mises::VonMisesMaterial;

use glam::Mat2;

use crate::particle::{Particle, Particles};

/// Identifies which constitutive model a material implements.
/// `repr(u32)` so this discriminant can be stored directly in GPU uniform buffers.
/// Explicit values are stable across recompiles — do not change them.
#[non_exhaustive]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstitutiveModel {
    Fallback = 0,
    Fluid = 1,            // Weakly-compressible Newtonian fluid, Tait EOS
    NeoHookean = 2,       // Neo-Hookean hyperelastic (jelly, soft solids)
    Corotated = 3,        // Corotated linear elastic (stiffer baseline)
    Snow = 4,             // Corotated + SVD plasticity (Stomakhin 2013)
    DruckerPrager = 5,    // Corotated elastic + DP yield surface (sand, soil, rock)
    VonMises = 6,         // J2 perfect plasticity — ductile flow, no hardening (lava, metal, clay)
    Rankine = 7,          // Tensile cutoff + exponential softening — brittle rock, bone, ice
    DruckerPragerMuI = 8, // Rate-dependent DP — µ(I) rheology, granular flow
    Viscoelastic = 9,     // Kelvin-Voigt: NeoHookean elastic + viscous dashpot in parallel
    Nacc = 10,            // Non-Associated Cam-Clay — wet soil, clay, bio tissue under compression
    GranularFluid = 11, // Granular-fluid mixture — Tait EOS + corotated deviatoric + SVD plasticity
    NoCompression = 12, // Tension-only (no-compression) reversible elastic — silk, tendons, membranes
}

// WGSL shaders (p2g.wgsl, particles_update.wgsl) index material branches by the
// ConstitutiveModel discriminant cast to u32. These assertions catch any enum reordering
// that would silently run the wrong GPU stress branch on a material.
const _: () = {
    use ConstitutiveModel as C;
    assert!(C::Fallback as u32 == 0);
    assert!(C::Fluid as u32 == 1);
    assert!(C::NeoHookean as u32 == 2);
    assert!(C::Corotated as u32 == 3);
    assert!(C::Snow as u32 == 4);
    assert!(C::DruckerPrager as u32 == 5);
    assert!(C::VonMises as u32 == 6);
    assert!(C::Rankine as u32 == 7);
    assert!(C::DruckerPragerMuI as u32 == 8);
    assert!(C::Viscoelastic as u32 == 9);
    assert!(C::Nacc as u32 == 10);
    assert!(C::GranularFluid as u32 == 11);
    assert!(C::NoCompression as u32 == 12);
};

/// Which role a material plays in two-phase mixture coupling (Tampubolon et al.
/// 2017, "Multi-species simulation of porous sand and water mixtures" --
/// interpenetrating granular-fluid Darcy drag, e.g. water soaking into sand).
/// This is a MATERIAL-level classification (via `MaterialModel::mixture_phase`),
/// not a per-particle field -- every particle of a given material shares the
/// same phase, matching how `constitutive_model` already works. `None` (the
/// default for every existing material) opts a scene entirely out of mixture
/// coupling at zero cost -- see `Grid::has_mixture_activity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixturePhase {
    /// The porous solid (e.g. sand/soil) -- keeps its own full elastic/plastic
    /// deformation, unaffected by mixture coupling beyond the drag force itself.
    Solid,
    /// The interpenetrating fluid (e.g. water) -- exchanges momentum with the
    /// solid phase via Darcy-style drag at every node both phases touch.
    Fluid,
}

pub trait MaterialModel: Send + Sync + core::fmt::Debug {
    /// Which constitutive law this material implements.
    /// Used by the GPU shader to select the correct stress branch per particle.
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fallback
    }
    // Returns the Kirchhoff-like stress used by the transfer kernel.
    // The kernel applies geometry/time factors (dt, kernel_d_inverse, cell_dist, weight).
    fn kirchhoff_stress(&self, _particles: &Particles, _i: usize) -> Mat2 {
        Mat2::ZERO
    }

    // Returns the particle volume used in the stress contribution.
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    /// CFL timestep bound for one particle. Takes `density`/`hardening_scale` as plain
    /// scalars rather than `&Particles, i: usize` — every implementation only ever reads
    /// these two fields, both of which exist directly on `Particle` (AoS) too, so the CPU
    /// (SoA) and GPU (AoS) CFL scans can both call this without either one needing the
    /// other's storage representation.
    fn timestep_bound(
        &self,
        _density: f32,
        _hardening_scale: f32,
        _cell_width: f32,
        _material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        f32::INFINITY
    }

    fn update_particle(&self, _particles: &mut Particles, _i: usize, _dt: f32) {}

    /// Seed per-particle plastic state at spawn time.
    ///
    /// Called once per particle immediately after position/volume assignment.
    /// Default: no-op (elastic materials need no initial plastic state).
    /// Override for materials that have a non-zero neutral accumulator (e.g. sand).
    fn init_particle(&self, _particle: &mut Particle) {}

    /// Whether `update_particle` does real work on the CPU.
    ///
    /// Return `false` if plasticity is fully handled on GPU (default).
    /// Return `true` for CPU-only plasticity paths — the GPU solver uses this to
    /// decide whether to download particles and run the CPU pass each frame.
    fn needs_cpu_update(&self) -> bool {
        false
    }

    /// Whether particles of this material require a per-substep density recompute.
    ///
    /// Fluid EOS materials (Newtonian, Bingham) need up-to-date density each substep
    /// because their pressure is a function of current ρ. Elastic/plastic materials
    /// do not — density is derived from J at the end of update_particle.
    /// Default: false. Override in fluid models.
    fn needs_density_recompute(&self) -> bool {
        false
    }

    /// Which two-phase mixture role this material plays, if any -- see
    /// `MixturePhase`'s own doc. `None` (default) means this material never
    /// participates in mixture coupling, the zero-cost-when-unused case that
    /// covers every material/scene that doesn't need this feature.
    fn mixture_phase(&self) -> Option<MixturePhase> {
        None
    }

    /// Scaling coefficient for activation-driven deviatoric stress.
    ///
    /// When non-zero, the per-particle `activation` field (0.0–1.0) modulates the
    /// deviatoric component of the Kirchhoff stress. This is the engine-level hook for
    /// active matter: muscles, motile cells, contractile tissue.
    ///
    /// Physics: τ_total = τ_elastic + activation × coeff × I  (contractile active pressure)
    /// Default: 0.0 — activation has no effect on passive materials.
    fn activation_scale(&self) -> f32 {
        0.0
    }

    /// Scaling coefficient for internal pre-stress pressure.
    ///
    /// When non-zero, the per-particle `internal_pressure` field (already SI-
    /// converted to grid stress units) contributes an isotropic `-P·I` term to
    /// the Kirchhoff stress — the standard "prestressed structure" treatment
    /// (a balloon: envelope tension balanced against internal gas pressure).
    /// Generic engine-level hook, not plant-specific: real motivating case is
    /// turgor pressure (plants aren't held up by cell-wall elasticity alone —
    /// see Niklas 1992's "hydro-skeleton" theory), but applies to any
    /// internally-pressurized body a material wants to model this way.
    ///
    /// Physics: τ_total = τ_elastic + τ_active − internal_pressure × coeff × I
    /// Default: 0.0 — pre-stress has no effect on materials that don't opt in
    /// (fluids already carry their own EOS pressure and should not double up).
    fn pressure_scale(&self) -> f32 {
        0.0
    }

    /// Returns this material's parameters as a flat, GPU-uploadable struct.
    /// Default returns zeroed params (Fallback model).
    fn params(&self) -> MaterialParams {
        MaterialParams::default()
    }

    /// Energy cost (J/kg, in whatever temperature unit `Particle::temperature` uses)
    /// of transitioning INTO this material via `Simulation::phase_transition` /
    /// `add_phase_rule`. Positive = endothermic (e.g. melting into a liquid — absorbs
    /// energy, cooling the particle). Negative = exothermic (e.g. freezing into a
    /// solid — releases energy, warming the particle). Default 0.0 = no energy cost
    /// (existing behavior for every material, unchanged).
    ///
    /// Applied in `Simulation::phase_transition`/`add_phase_rule` (CPU) against
    /// `ThermalDiffusion::heat_capacity` when a thermal model is configured, and in
    /// `GpuSimulation::phase_transition` against the `heat_capacity` passed to
    /// `attach_thermal_gpu` -- same debit, same formula, on both. GPU has no automatic
    /// `add_phase_rule` counterpart yet (only the manual, one-shot `phase_transition`);
    /// that gap is real and separate from this energy accounting.
    fn latent_heat(&self) -> f32 {
        0.0
    }
}

/// The `MaterialModel` methods every delegating wrapper below (`WithLatentHeat`,
/// `WithMixturePhase`, `WithPreStress`) forwards to `self.inner` byte-for-byte.
/// Factored into one macro so these three impls can't drift out of sync — a new
/// `MaterialModel` method that should default-forward gets added here ONCE, not
/// copy-pasted three times.
///
/// The 3 methods each wrapper actually overrides (`init_particle`, `mixture_phase`,
/// `latent_heat`) are NOT in this list -- each wrapper still writes those by hand,
/// forwarding the two it doesn't override itself.
macro_rules! forward_material_model_common {
    () => {
        fn constitutive_model(&self) -> ConstitutiveModel {
            self.inner.constitutive_model()
        }
        fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
            self.inner.kirchhoff_stress(particles, i)
        }
        fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
            self.inner.stress_volume(particles, i)
        }
        fn timestep_bound(
            &self,
            density: f32,
            hardening_scale: f32,
            cell_width: f32,
            material_cfl: f32,
            viscous_cfl: f32,
        ) -> f32 {
            self.inner.timestep_bound(
                density,
                hardening_scale,
                cell_width,
                material_cfl,
                viscous_cfl,
            )
        }
        fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
            self.inner.update_particle(particles, i, dt)
        }
        fn needs_cpu_update(&self) -> bool {
            self.inner.needs_cpu_update()
        }
        fn needs_density_recompute(&self) -> bool {
            self.inner.needs_density_recompute()
        }
        fn activation_scale(&self) -> f32 {
            self.inner.activation_scale()
        }
        fn pressure_scale(&self) -> f32 {
            self.inner.pressure_scale()
        }
        fn params(&self) -> MaterialParams {
            self.inner.params()
        }
    };
}

/// Wraps any `MaterialModel` to give it a non-zero `latent_heat()` without writing a full
/// delegating impl by hand — none of the 12 built-in materials expose a settable
/// `latent_heat` field directly, since most users never need one.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{NewtonianFluidMaterial, WithLatentHeat};
/// // Water absorbs 334 (sim-unit) energy per unit mass when transitioning into this material.
/// let water = WithLatentHeat::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1.0e5), 334.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithLatentHeat<M> {
    pub inner: M,
    pub latent_heat: f32,
}

impl<M> WithLatentHeat<M> {
    pub fn new(inner: M, latent_heat: f32) -> Self {
        Self { inner, latent_heat }
    }
}

impl<M: MaterialModel> MaterialModel for WithLatentHeat<M> {
    forward_material_model_common!();
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle)
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        self.inner.mixture_phase()
    }
    fn latent_heat(&self) -> f32 {
        self.latent_heat
    }
}

/// Wraps any `MaterialModel` to opt it into two-phase mixture coupling as either
/// the `Solid` or `Fluid` phase -- see `MixturePhase`'s own doc. Same pattern as
/// `WithLatentHeat`: existing materials (`DruckerPragerMaterial`,
/// `NewtonianFluidMaterial`, etc.) never opt in by themselves, so combining sand
/// and water in a scene that DOESN'T wrap either one behaves exactly as before
/// (two ordinary single-phase materials sharing a grid, no drag coupling) --
/// mixture physics is explicit, per-scene, not a silent side effect of which
/// materials happen to coexist.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{DruckerPragerMaterial, MixturePhase, WithMixturePhase};
/// let sand = WithMixturePhase::new(DruckerPragerMaterial::cohesionless(1.0e5, 0.2), MixturePhase::Solid);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithMixturePhase<M> {
    pub inner: M,
    pub phase: MixturePhase,
}

impl<M> WithMixturePhase<M> {
    pub fn new(inner: M, phase: MixturePhase) -> Self {
        Self { inner, phase }
    }
}

impl<M: MaterialModel> MaterialModel for WithMixturePhase<M> {
    forward_material_model_common!();
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle)
    }
    fn latent_heat(&self) -> f32 {
        self.inner.latent_heat()
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        Some(self.phase)
    }
}

/// Wraps any `MaterialModel` to give particles a nonzero `internal_pressure` at spawn
/// time, without writing a full delegating impl by hand — same pattern as
/// `WithLatentHeat`/`WithMixturePhase`. The wrapped material's own `pressure_scale()`
/// still gates whether the pressure actually contributes stress (see
/// `combined_kirchhoff_stress`); this wrapper only supplies the per-particle value.
///
/// Real motivating case: turgor pressure in plants (see `Particle::internal_pressure`
/// doc) — but generic, not plant-specific: any internally-pressurized body.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{NeoHookeanMaterial, WithPreStress};
/// // A turgid plant-tissue stalk: 0.5 MPa turgor pressure already SI-converted to
/// // grid stress units (see `Pressurized::material` for the real conversion).
/// let stalk = WithPreStress::new(NeoHookeanMaterial::new(4000.0, 6000.0), 12.5);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithPreStress<M> {
    pub inner: M,
    pub pressure: f32,
}

impl<M> WithPreStress<M> {
    pub fn new(inner: M, pressure: f32) -> Self {
        Self { inner, pressure }
    }
}

impl<M: MaterialModel> MaterialModel for WithPreStress<M> {
    forward_material_model_common!();
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle);
        particle.internal_pressure = self.pressure;
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        self.inner.mixture_phase()
    }
    fn latent_heat(&self) -> f32 {
        self.inner.latent_heat()
    }
}

/// Internal fallback used when no material is registered for a particle ID.
/// Zero stress, no timestep constraint, no state updates.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FallbackMaterial;

impl MaterialModel for FallbackMaterial {}
