//! Energy domain: how it flows and transforms.
//!
//! `thermodynamics` — `ThermalDiffusion` (Fourier heat), `ScalarDiffusionField`
//! (generic reaction-diffusion: pheromone, nutrients, morphogen). `acoustics`
//! [feature = "experimental"] — `WaveEquation2D`, pressure-wave propagation.
//!
//! Electromagnetism's radiative/energy-transfer half (`ElectromagneticWave`,
//! optical `MaterialProperties` in `electromagnetics::interactions`) belongs
//! here too once that module's Forces/Energy split lands -- its point-charge
//! force-application half (`electromagnetics::fields`) stays in Forces.
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
pub mod thermodynamics;
