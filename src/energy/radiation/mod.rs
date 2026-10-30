//! Radiation: the energy matter carries as light.
//!
//! `thermodynamics::transfer` already owns the *integrated* side of thermal
//! radiation -- Stefan-Boltzmann exchange, `sigma*eps*A*F*(T_h^4 - T_c^4)`, in
//! Watts. This module owns the *spectral* side: how that power is distributed
//! over wavelength, and what that distribution amounts to as a colour.
//!
//! - `blackbody` -- Planck's law, Wien's displacement law, Lambertian
//!   blackbody radiance. Pure SI physics, no observer, no display.
//! - `attenuation` -- Beer-Lambert: what a medium removes from light passing
//!   through it. The coefficients belong to the substance, so they live in
//!   `matter::materials::optical`, not here.
//! - `fresnel` -- reflectance at an interface: what bounces off instead
//!   of entering. Maxwell's boundary conditions, unpolarized.
//! - `spectrum` -- a spectrum reduced to a colour through the CIE 1931
//!   standard observer, then to linear sRGB. Radiometry becoming photometry.
//!
//! Rendering lives in `systems::render` and only *consumes* this: a renderer
//! that computed its own emission colour would be inventing physics in the
//! wrong domain. `systems/render/shaders/blackbody.inc.wgsl` is a deliberate,
//! tested mirror of [`spectrum::blackbody_linear_srgb_locus_fit`], because a
//! fragment shader cannot integrate a spectrum per pixel -- the mirror is
//! held honest by `spectrum`'s own tests, not by inspection.

pub mod attenuation;
pub mod blackbody;
pub mod fresnel;
pub mod spectrum;

pub use attenuation::{
    OpticalCoefficientsError, OpticalCoefficientsSi, beer_lambert_transmittance,
    mix_by_volume_fraction, penetration_attenuation_m_inv, scattering_from_reduced, slab_radiance,
    volume_fractions_from_masses,
};
pub use blackbody::{
    BOLTZMANN, PLANCK, SPEED_OF_LIGHT, WIEN_DISPLACEMENT, blackbody_radiance_w_m2_sr,
    planck_spectral_radiance, wien_peak_wavelength_m,
};
pub use fresnel::{fresnel_r0_dielectric, schlick_reflectance};
pub use spectrum::{
    blackbody_band_luminance_w_m2_sr, blackbody_band_radiance_w_m2_sr, blackbody_chromaticity_xy,
    blackbody_display_exposure, blackbody_linear_srgb, blackbody_linear_srgb_locus_fit,
    chromaticity_to_linear_srgb, cie_1931_observer, linear_srgb_luminance_weights,
    spectral_band_average, spectral_band_effective_absorption, spectral_band_transmittance,
    srgb_channel_sensitivity,
};
