//! Energy domain: how it flows and transforms.
//!
//! `thermodynamics` — `ThermalDiffusion` (Fourier heat), `ScalarDiffusionField`
//! (generic reaction-diffusion: pheromone, nutrients, morphogen). `acoustics`
//! [feature = "experimental"] — `WaveEquation2D`, pressure-wave propagation.
//! `electromagnetics` [feature = "experimental"] — `ElectromagneticWave`,
//! optical `MaterialProperties` (refractive index, permittivity/permeability);
//! the point-charge force-application half lives in `forces::electromagnetics`
//! instead.
//!
//! Part of the emerge/LP domain taxonomy (matter/forces/energy/information/
//! spacetime/organism/systems) -- see `project_domain_taxonomy` design notes.
//! Re-exported at crate root (`pub use energy::thermodynamics;` etc in
//! `lib.rs`) so every existing `crate::thermodynamics::`/`crate::acoustics::`
//! path and every LP `emerge::thermodynamics::` path keeps resolving
//! unchanged -- this move only changes where the files physically live, not
//! any public API.

#[cfg(feature = "experimental")]
pub mod acoustics;
#[cfg(feature = "experimental")]
pub mod electromagnetics;
pub mod thermodynamics;
