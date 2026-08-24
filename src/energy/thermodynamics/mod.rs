//! Thermodynamics -- heat and generic scalar transport, MPM-coupled.
//!
//! - `diffusion.rs`    -- Fourier heat diffusion ∂T/∂t = α∇²T + Newton cooling
//! - `scalar_field.rs` -- generic ∂φ/∂t = D·∇²φ − λ·φ + S (pheromone, nutrients, morphogen)
//! - `stencil.rs`      -- shared Laplacian FD step used by both of the above
//! - `transfer.rs`     -- scalar IRL primitives: conduction, Stefan-Boltzmann radiation, entropy/2nd law

pub mod cosserat_field;
pub mod diffusion;
pub mod granular_fluidity;
pub mod ideal_gas;
pub mod scalar_field;
mod stencil;
pub mod transfer;

pub use cosserat_field::{CosseratConfig, CosseratField};
pub use diffusion::{ThermalConfig, ThermalDiffusion};
pub use granular_fluidity::{GranularFluidityConfig, GranularFluidityField};
pub use scalar_field::{ScalarDiffusionConfig, ScalarDiffusionField};
pub use transfer::{
    STEFAN_BOLTZMANN, entropy_change_heat_transfer, entropy_change_irreversible, heat_conduction,
    heat_radiation, saturating_uptake, second_law_holds, thermal_diffusivity,
};
