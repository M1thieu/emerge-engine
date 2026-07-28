//! GPU-side wire structs for the renderer -- split out of `mod.rs` (was its own
//! "GPU-side structs (must match WGSL)" section, ~90 of the file's ~895 lines).
//! Every `repr(C)` struct here must stay byte-identical to its WGSL counterpart;
//! the `size_of` asserts are the real guard against silent drift.

use std::mem;

use bytemuck::{Pod, Zeroable};

/// Mirrors `grid_volume.wgsl`'s `GridVolumeParams` -- see that shader's own doc for
/// the real technique (samples the solver's own P2G mass field directly instead of
/// per-particle splats).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GridVolumeParams {
    pub(super) sx: f32,
    pub(super) tx: f32,
    pub(super) sy: f32,
    pub(super) ty: f32,
    pub(super) grid_res: u32,
    pub(super) mass_floor: f32,
    pub(super) material_mass_enabled: u32,
    pub(super) _pad1: f32,
}
const _: () = assert!(mem::size_of::<GridVolumeParams>() == 32);

/// Bundles `render_grid_volume`'s buffer args -- same real precedent as
/// `spacetime::transfer::P2GParticleState` (a struct instead of a suppressed
/// argument-count lint).
pub struct GridVolumeSource<'a> {
    /// `GpuSimulation::grid_buffer()`.
    pub grid: &'a wgpu::Buffer,
    /// `GpuSimulation::material_mass_buffer()` -- pass it regardless of whether
    /// `attach_grid_material_render_gpu` was called; `material_mass_enabled` gates
    /// whether the shader actually reads it.
    pub material_mass: &'a wgpu::Buffer,
    pub material_mass_enabled: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct InstanceData {
    pub(super) deform_col0: [f32; 2],
    pub(super) deform_col1: [f32; 2],
    pub(super) position: [f32; 2],
    pub(super) _pad: [f32; 2],
    pub(super) color: [f32; 4],
}
const _: () = assert!(mem::size_of::<InstanceData>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraParams {
    pub(super) view_proj: [f32; 16],
    pub(super) particle_scale: f32,
    pub(super) round_particles: u32,
    pub(super) _pad: [f32; 2],
}
const _: () = assert!(mem::size_of::<CameraParams>() == 80);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RenderConfig {
    pub(super) mode: u32,
    pub(super) particle_count: u32,
    pub(super) vel_scale: f32,
    pub(super) _pad: u32,
}
const _: () = assert!(mem::size_of::<RenderConfig>() == 16);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct OpticalTable {
    /// rgb = sigma_a, absorption coefficient (Beer-Lambert). .w = sigma_s, reduced
    /// scattering coefficient (single scalar, not per-channel -- real tissue
    /// scattering is much less wavelength-dependent than absorption in the visible
    /// range, Jacques 2013, a legitimate simplification for that reason).
    pub(super) slots: [[f32; 4]; 16],
    /// .x = specular Fresnel base reflectance R0 (Schlick 1994 approximation),
    /// rest padding. Real, cited, but bounded: this renderer has no surface-normal
    /// estimation (it tints particle instances, doesn't raytrace a reconstructed
    /// surface), so this is a constant near-normal-incidence reflectance, NOT a
    /// full view-angle-dependent Fresnel term -- honestly a simplification, not a
    /// claim of full BRDF accuracy.
    pub(super) specular: [[f32; 4]; 16],
}
const _: () = assert!(mem::size_of::<OpticalTable>() == 512);
