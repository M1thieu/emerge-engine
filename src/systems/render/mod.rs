/// emerge particle renderer -- physics-driven, no assets.
///
/// # Two rendering paths
///
/// **CPU path** (`render_slice`):
///   Builds `InstanceData` per particle on CPU, uploads via `write_buffer`.
///
/// **GPU path** (`render_gpu`):
///   Runs `prep_instances.wgsl` compute to fill instance buffer directly from
///   the particle storage buffer -- zero CPU readback, zero stall.
///   Pass `sim.particle_buffer()` + `sim.particle_count()`. No `sync_particles_blocking()`.
use std::mem;

use crate::particle::{Particle, Particles};
use crate::systems::gpu::MAX_RENDER_MATERIAL_SLOTS;

const RENDER_SHADER: &str = include_str!("shaders/render_particles.wgsl");
const PREP_SHADER: &str = include_str!("shaders/prep_instances.wgsl");
const GRID_VOLUME_SHADER: &str = include_str!("shaders/grid_volume.wgsl");
const CURVATURE_FLOW_SHADER: &str = include_str!("shaders/curvature_flow.wgsl");
const PREP_WG: u32 = 64;
const SURFACE_CLEAR_WG: u32 = 64;
const SURFACE_SPLAT_WG: u32 = 64;
/// Finer-than-physics-grid resolution multiplier (see `curvature_flow.wgsl`'s
/// own top doc). Cost scales with the SQUARE of this constant --
/// `(N*grid_res)^2` f32 cells through every pass (splat, convert,
/// curvature-iterate, thermal diffusion, wave, visibility, band-hysteresis).
const SURFACE_RES_MULTIPLIER: u32 = 6;
/// Real, fixed EVEN iteration count -- keeping this even means the settled
/// result always lands in the SAME buffer (`surface_a`) regardless of N,
/// avoiding a runtime-conditional final bind group (see
/// `render_surface_reconstruction`'s own doc). 12 is within van der Laan et
/// al. 2009's own "several iterations per frame" range.
const CURVATURE_ITERATIONS: u32 = 12;
const _: () = assert!(
    CURVATURE_ITERATIONS.is_multiple_of(2),
    "CURVATURE_ITERATIONS must stay even so the result always settles in surface_a"
);

// ── Color mode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    ByMaterial = 0,
    ByVelocity = 1,
    ByVolume = 2,
    ByPhysics = 3,
    ByThermal = 4,
    ByActivation = 5,
    /// Generic second scalar carrier (resource/grass level, pheromone, nutrients).
    /// See `Particle::scalar_field`'s own doc. Distinct wire value (6, not the next
    /// unused slot after ByActivation's implicit WGSL else-branch) so the GPU shader's
    /// existing fallback `else` can keep meaning ByActivation without renumbering it.
    ByScalarField = 6,
}

// GPU-side wire structs (InstanceData/CameraParams/RenderConfig/OpticalTable,
// GridVolumeParams/GridVolumeSource) live in gpu_types.rs -- see that file's doc.
mod gpu_types;
use gpu_types::{CameraParams, InstanceData, OpticalTable, RenderConfig};
pub use gpu_types::{DualPhaseSurfaceSource, GridVolumeSource, SurfaceReconstructionSource};

// GPU buffer allocation (the RenderBuffers struct + its own constructor)
// lives in buffers.rs -- see that file's doc.
mod buffers;
use buffers::RenderBuffers;

// Grid-native volumetric render path (`render_grid_volume`) lives in
// grid_volume.rs -- see that file's doc.
mod grid_volume;

// wgpu pipeline construction (the three build_*_pipeline functions + their
// bind-group-layout helpers) lives in pipelines.rs -- see that file's doc.
mod pipelines;

// Curvature-flow surface reconstruction (render_surface_reconstruction[_dual_phase]
// + its own capacity helpers) lives in surface_reconstruction.rs -- see that file's doc.
mod surface_reconstruction;
use gpu_types::{
    BandHysteresisParams, LightDiffuseParams, SurfaceParams, SurfaceRenderParams, VisibilityParams,
    WaveStepParams,
};
use pipelines::{
    build_band_hysteresis_step_pipeline, build_grid_visibility_step_pipeline,
    build_grid_volume_pipeline, build_light_diffuse_pipeline, build_particle_pipeline,
    build_post_total_reduce_pipeline, build_prep_pipeline, build_surface_clear_pipeline,
    build_surface_convert_pipeline, build_surface_dual_render_pipeline,
    build_surface_iterate_pipeline, build_surface_render_pipeline, build_surface_splat_pipeline,
    build_temp_avg_pipeline, build_temp_diffuse_pipeline, build_visibility_step_pipeline,
    build_volume_correct_pipeline, build_wave_step_pipeline,
};

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer, // VERTEX | COPY_DST — drawn as per-instance attributes
    storage_instances: wgpu::Buffer, // STORAGE | COPY_SRC — compute write target (GPU path)
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    max_particles: usize,

    prep_pipeline: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    render_config_buf: wgpu::Buffer,
    optical_table_buf: wgpu::Buffer,

    grid_volume_pipeline: wgpu::RenderPipeline,
    grid_volume_bgl: wgpu::BindGroupLayout,
    grid_volume_params_buf: wgpu::Buffer,
    /// Real, persistent (across frames) hysteresis visible/invisible state
    /// for the grid-native render path -- the SAME technique as
    /// `visibility_buf` above (see `curvature_flow.wgsl`'s "Pass 2c" doc),
    /// ported to `grid_volume.wgsl`'s own `mass_floor` discard. Separate
    /// buffer/resolution tracking since this operates at `grid_res`, not
    /// `surface_res`.
    grid_visibility_step_pipeline: wgpu::ComputePipeline,
    grid_visibility_step_bgl: wgpu::BindGroupLayout,
    grid_visibility_buf: wgpu::Buffer,
    grid_visibility_params_buf: wgpu::Buffer,
    /// Resolution `grid_visibility_buf` is currently allocated at --
    /// `ensure_grid_visibility_capacity` regrows it when a caller's
    /// `grid_res` exceeds this, same lazy-growth convention as `surface_res`.
    grid_visibility_res: u32,
    /// Cached ortho projection + grid_res (set by `set_camera`) -- lets
    /// `render_grid_volume` take just (device, queue, grid_buf, material_mass_buf,
    /// view, clear) instead of repeating width/height/grid_res, keeping it under
    /// clippy's argument-count lint.
    cached_ortho: (f32, f32, f32, f32),
    cached_grid_res: u32,
    /// Real light direction for `render_grid_volume`/`render_surface_
    /// reconstruction`(`_dual_phase`)'s Lambertian + specular shading -- set
    /// via `set_light_dir`, defaults to the same value those shaders used
    /// to hardcode directly (so behavior is unchanged until a caller
    /// explicitly wires in a real value, e.g. `SimConfig::light_dir`).
    light_dir: (f32, f32),

    // ── Curvature-flow surface reconstruction (see curvature_flow.wgsl) ────
    surface_clear_pipeline: wgpu::ComputePipeline,
    surface_clear_bgl: wgpu::BindGroupLayout,
    surface_splat_pipeline: wgpu::ComputePipeline,
    surface_splat_bgl: wgpu::BindGroupLayout,
    surface_convert_pipeline: wgpu::ComputePipeline,
    surface_convert_bgl: wgpu::BindGroupLayout,
    surface_iterate_pipeline: wgpu::ComputePipeline,
    surface_iterate_bgl: wgpu::BindGroupLayout,
    surface_render_pipeline: wgpu::RenderPipeline,
    surface_render_bgl: wgpu::BindGroupLayout,
    /// Fixed-point atomic splat target (see `curvature_flow.wgsl`'s own
    /// `clear_surface_main`/`splat_density_main` doc).
    surface_atomic_buf: wgpu::Buffer,
    /// Real mass-weighted temperature: fixed-point atomic scatter target +
    /// converted plain-f32 buffer `fs_main` samples for blackbody emission
    /// -- see `surface_temp_atomic`'s own doc in the shader. Single-phase
    /// only (grown/reused by phase A of the dual-phase path too, same as
    /// `surface_atomic_buf` itself -- see that field's own sharing note
    /// below); phase B gets its own separate pair.
    surface_temp_atomic_buf: wgpu::Buffer,
    surface_temp_float_buf: wgpu::Buffer,
    /// Real volume-preserving correction (`curvature_flow.wgsl`'s "Pass
    /// 1d"): `pre_total_atomic_buf` accumulates the TRUE ground-truth
    /// particle mass during splat; `post_total_atomic_buf` sums the settled
    /// (post-curvature-flow) density; `volume_correct_pipeline` rescales
    /// the settled result by their ratio. Both are single-element (4-byte)
    /// buffers, NEVER grown with `surface_res` (a global scalar per phase,
    /// not a per-cell field).
    post_total_reduce_pipeline: wgpu::ComputePipeline,
    post_total_reduce_bgl: wgpu::BindGroupLayout,
    volume_correct_pipeline: wgpu::ComputePipeline,
    volume_correct_bgl: wgpu::BindGroupLayout,
    pre_total_atomic_buf: wgpu::Buffer,
    post_total_atomic_buf: wgpu::Buffer,
    /// Real 2D thermal-diffusion PDE on the recovered temperature field
    /// (`curvature_flow.wgsl`'s "Pass 1c") -- `temp_avg_pipeline` divides by
    /// final settled density once, `temp_diffuse_pipeline` then runs one
    /// real heat-equation step. `surface_temp_b_buf` is the ping-pong
    /// partner: avg writes into it, diffuse reads it and writes the result
    /// back into `surface_temp_float_buf` (the buffer `fs_main` already
    /// reads), so no render-bind-group change was needed for this addition.
    temp_avg_pipeline: wgpu::ComputePipeline,
    temp_avg_bgl: wgpu::BindGroupLayout,
    temp_diffuse_pipeline: wgpu::ComputePipeline,
    temp_diffuse_bgl: wgpu::BindGroupLayout,
    surface_temp_b_buf: wgpu::Buffer,
    /// Real diffusion approximation to light transport (`curvature_flow.
    /// wgsl`'s "Pass 1e") -- see that entry point's own doc for the full
    /// real derivation. `light_phi_bufs` are the two ping-pong buffers (see
    /// their own doc in `buffers.rs`); `light_frame_index` alternates which
    /// one is "current" (read) vs "next" (write) each frame, the same real
    /// role `wave_frame_index` plays for the (second-order) wave field,
    /// just a simpler 2-way rotation for this first-order equation.
    light_diffuse_pipeline: wgpu::ComputePipeline,
    light_diffuse_bgl: wgpu::BindGroupLayout,
    light_phi_bufs: [wgpu::Buffer; 2],
    light_diffuse_params_buf: wgpu::Buffer,
    light_frame_index: u32,
    /// Ping-pong plain-f32 buffers. `CURVATURE_ITERATIONS` (even) means the
    /// settled result always lands in `surface_a` -- `render_surface_
    /// reconstruction` only ever reads that one, never `surface_b` directly.
    surface_a_buf: wgpu::Buffer,
    surface_b_buf: wgpu::Buffer,
    surface_params_buf: wgpu::Buffer,
    surface_render_params_buf: wgpu::Buffer,
    /// Resolution the surface buffers were allocated at (`grid_res *
    /// SURFACE_RES_MULTIPLIER`) -- `ensure_surface_capacity` reallocates all
    /// three buffers when a caller's `grid_res` needs a bigger one.
    surface_res: u32,

    /// N-material extension (see `curvature_flow.wgsl`'s own doc),
    /// single-phase path only: flat `surface_res² × MAX_RENDER_MATERIAL_
    /// SLOTS` per-cell mass array, same real technique as `grid_volume.
    /// wgsl`'s own `material_mass`. Grown LAZILY by `ensure_surface_
    /// material_mass_capacity`, called only when a caller opts in
    /// (`SurfaceReconstructionSource::material_mass_enabled`) -- unlike
    /// every other surface buffer above, NOT grown unconditionally by
    /// `ensure_surface_capacity`, since it's 16x the size of a single
    /// per-cell field and most callers never use it (real cost accounting
    /// in the plan this shipped from).
    surface_material_mass_buf: wgpu::Buffer,
    /// Resolution `surface_material_mass_buf` is currently sized for.
    /// Independent of `surface_res` itself -- 0 means still the
    /// constructor's 4-byte placeholder, never opted into.
    surface_material_mass_res: u32,

    /// Real, persistent (across frames, unlike everything else in this
    /// section) 2D wave-equation height field -- see `curvature_flow.wgsl`'s
    /// own "Pass 2b" doc. THREE physical buffers, not two: wgpu's usage-
    /// scope validator rejects binding the SAME buffer as both read-only
    /// and read_write within one dispatch, even when the actual access
    /// pattern is index-disjoint and logically hazard-free (confirmed via a
    /// real validation error when a 2-buffer aliasing scheme was tried
    /// first) -- a genuine leapfrog integrator only ever needs u(t) and
    /// u(t-dt) to read, but writing u(t+dt) needs a THIRD distinct slot to
    /// satisfy wgpu's conservative rule. `wave_frame_index` rotates which
    /// of the 3 buffers plays which of the 3 roles (current/previous/next)
    /// each call -- see that method's own doc for the exact rotation.
    wave_step_pipeline: wgpu::ComputePipeline,
    wave_step_bgl: wgpu::BindGroupLayout,
    wave_bufs: [wgpu::Buffer; 3],
    wave_params_buf: wgpu::Buffer,
    wave_frame_index: u32,
    /// Last frame's settled density (a copy of `surface_a_buf` taken right
    /// after each frame's wave step reads it), so `wave_step_main` can
    /// excite from a genuine TEMPORAL disturbance rather than a permanent
    /// spatial-edge artifact.
    wave_density_prev_buf: wgpu::Buffer,

    /// Real, persistent (across frames) hysteresis visible/invisible state
    /// -- see `curvature_flow.wgsl`'s own "Pass 2c" doc. A SINGLE buffer
    /// (no rotation needed, unlike the wave field: this pass only ever
    /// reads/writes its OWN index, no neighbor stencil, so there's no
    /// wgpu usage-scope conflict to avoid).
    visibility_step_pipeline: wgpu::ComputePipeline,
    visibility_step_bgl: wgpu::BindGroupLayout,
    visibility_buf: wgpu::Buffer,
    visibility_params_buf: wgpu::Buffer,

    /// Real, persistent hysteresis color-band state -- see `curvature_
    /// flow.wgsl`'s own "Pass 2d" doc. Same single-buffer, no-rotation
    /// shape as `visibility_buf` (self-index only, downstream/one-way of
    /// the density field, never fed back into it).
    band_hysteresis_step_pipeline: wgpu::ComputePipeline,
    band_hysteresis_step_bgl: wgpu::BindGroupLayout,
    band_state_buf: wgpu::Buffer,
    band_hysteresis_params_buf: wgpu::Buffer,

    /// Real, persistent RAW splat density history for neighborhood-
    /// clamped temporal smoothing -- see `convert_atomic_to_float_main`'s
    /// own doc (Lottes 2011 / Karis 2014 TAA technique, adapted to a
    /// scalar field).
    raw_splat_history_buf: wgpu::Buffer,

    // ── Two-phase extension (see curvature_flow.wgsl's own doc) ────────────
    /// A second, fully independent set of splat/ping-pong buffers for
    /// "phase B" -- `render_surface_reconstruction_dual_phase` runs the
    /// SAME clear/splat/convert/iterate pipelines twice, once into phase A's
    /// existing `surface_*_buf` fields above and once into these, so each
    /// phase gets its own real, independently-smoothed surface (see the
    /// shader's own VOF/phase-fraction citation for why that's the correct
    /// choice, not a shared blended field).
    phase_b_atomic_buf: wgpu::Buffer,
    /// Phase B's own temperature atomic/float pair -- see `surface_temp_
    /// atomic_buf`'s own doc. Written every dual-phase frame for symmetry
    /// with phase A's scatter, but NOT read by `fs_main_dual_phase` (no
    /// binding for it there -- see that entry point's own doc for why).
    phase_b_temp_atomic_buf: wgpu::Buffer,
    phase_b_temp_float_buf: wgpu::Buffer,
    /// Phase B's own volume-preserving-correction totals -- see
    /// `pre_total_atomic_buf`'s own doc.
    phase_b_pre_total_atomic_buf: wgpu::Buffer,
    phase_b_post_total_atomic_buf: wgpu::Buffer,
    phase_b_a_buf: wgpu::Buffer,
    phase_b_b_buf: wgpu::Buffer,
    /// Phase B's own raw-splat history -- `surface_convert_bgl` now needs
    /// this 4th binding on EVERY caller.
    phase_b_raw_splat_history_buf: wgpu::Buffer,
    phase_b_params_buf: wgpu::Buffer,
    render_params_b_buf: wgpu::Buffer,
    /// Phase B's own wave/visibility/band state -- SAME real techniques as
    /// `wave_bufs`/`visibility_buf`/`band_state_buf` above (see
    /// `curvature_flow.wgsl`'s Pass 3b doc), just a second independent set
    /// since phase B is a fully separate density field. Phase A reuses the
    /// single-phase fields directly (`render_surface_reconstruction` and
    /// `render_surface_reconstruction_dual_phase` are never called the same
    /// frame, so sharing is safe) -- only phase B needs its own buffers.
    /// `wave_params_buf`/`visibility_params_buf`/`band_hysteresis_params_buf`
    /// are ALSO shared across both phases: they only carry `surface_res`/
    /// `mass_floor`, identical for both phases every frame.
    phase_b_wave_bufs: [wgpu::Buffer; 3],
    /// Phase B's own previous-density copy -- see `wave_density_prev_buf`'s
    /// own doc.
    phase_b_wave_density_prev_buf: wgpu::Buffer,
    phase_b_visibility_buf: wgpu::Buffer,
    phase_b_band_state_buf: wgpu::Buffer,
    surface_dual_render_pipeline: wgpu::RenderPipeline,
    surface_dual_render_bgl: wgpu::BindGroupLayout,

    scratch: Vec<InstanceData>,
    color_mode: ColorMode,
    vel_scale: f32,
    sigma_a: [[f32; 3]; 16],
    /// Reduced scattering coefficient per material slot (single scalar -- see
    /// `OpticalTable`'s own doc for why this isn't per-channel).
    sigma_s: [f32; 16],
    /// Specular Fresnel base reflectance R0 per material slot (see `OpticalTable`'s
    /// own doc for the real-but-bounded caveat).
    specular_r0: [f32; 16],
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        max_particles: usize,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let cap = max_particles.max(1);

        let RenderBuffers {
            instance_buffer,
            storage_instances,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            render_config_buf,
            optical_table_buf,
            grid_volume_params_buf,
            grid_visibility_buf,
            grid_visibility_params_buf,
            surface_atomic_buf,
            surface_temp_atomic_buf,
            surface_temp_float_buf,
            pre_total_atomic_buf,
            post_total_atomic_buf,
            surface_temp_b_buf,
            surface_a_buf,
            surface_b_buf,
            surface_params_buf,
            surface_render_params_buf,
            surface_material_mass_buf,
            phase_b_atomic_buf,
            phase_b_temp_atomic_buf,
            phase_b_temp_float_buf,
            phase_b_pre_total_atomic_buf,
            phase_b_post_total_atomic_buf,
            phase_b_a_buf,
            phase_b_b_buf,
            phase_b_raw_splat_history_buf,
            phase_b_params_buf,
            render_params_b_buf,
            phase_b_wave_bufs,
            phase_b_wave_density_prev_buf,
            phase_b_visibility_buf,
            phase_b_band_state_buf,
            wave_bufs,
            wave_params_buf,
            wave_density_prev_buf,
            visibility_buf,
            visibility_params_buf,
            band_state_buf,
            band_hysteresis_params_buf,
            raw_splat_history_buf,
            light_phi_bufs,
            light_diffuse_params_buf,
        } = RenderBuffers::new(device, cap);

        let (render_pipeline, render_bgl) = build_particle_pipeline(device, output_format);
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let (prep_pipeline, prep_bgl) = build_prep_pipeline(device);
        let (grid_volume_pipeline, grid_volume_bgl) =
            build_grid_volume_pipeline(device, output_format);
        let (grid_visibility_step_pipeline, grid_visibility_step_bgl) =
            build_grid_visibility_step_pipeline(device);
        let (surface_clear_pipeline, surface_clear_bgl) = build_surface_clear_pipeline(device);
        let (surface_splat_pipeline, surface_splat_bgl) = build_surface_splat_pipeline(device);
        let (surface_convert_pipeline, surface_convert_bgl) =
            build_surface_convert_pipeline(device);
        let (surface_iterate_pipeline, surface_iterate_bgl) =
            build_surface_iterate_pipeline(device);
        let (surface_render_pipeline, surface_render_bgl) =
            build_surface_render_pipeline(device, output_format);
        let (surface_dual_render_pipeline, surface_dual_render_bgl) =
            build_surface_dual_render_pipeline(device, output_format);
        let (wave_step_pipeline, wave_step_bgl) = build_wave_step_pipeline(device);
        let (visibility_step_pipeline, visibility_step_bgl) =
            build_visibility_step_pipeline(device);
        let (band_hysteresis_step_pipeline, band_hysteresis_step_bgl) =
            build_band_hysteresis_step_pipeline(device);
        let (temp_avg_pipeline, temp_avg_bgl) = build_temp_avg_pipeline(device);
        let (temp_diffuse_pipeline, temp_diffuse_bgl) = build_temp_diffuse_pipeline(device);
        let (light_diffuse_pipeline, light_diffuse_bgl) = build_light_diffuse_pipeline(device);
        let (post_total_reduce_pipeline, post_total_reduce_bgl) =
            build_post_total_reduce_pipeline(device);
        let (volume_correct_pipeline, volume_correct_bgl) = build_volume_correct_pipeline(device);

        Self {
            render_pipeline,
            render_bind_group,
            instance_buffer,
            storage_instances,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            max_particles: cap,
            prep_pipeline,
            prep_bgl,
            render_config_buf,
            grid_volume_pipeline,
            grid_volume_bgl,
            grid_volume_params_buf,
            grid_visibility_step_pipeline,
            grid_visibility_step_bgl,
            grid_visibility_buf,
            grid_visibility_params_buf,
            grid_visibility_res: 1,
            cached_ortho: (1.0, 0.0, 1.0, 0.0),
            cached_grid_res: 1,
            light_dir: (-0.5, 0.7),
            surface_clear_pipeline,
            surface_clear_bgl,
            surface_splat_pipeline,
            surface_splat_bgl,
            surface_convert_pipeline,
            surface_convert_bgl,
            surface_iterate_pipeline,
            surface_iterate_bgl,
            surface_render_pipeline,
            surface_render_bgl,
            surface_atomic_buf,
            surface_temp_atomic_buf,
            surface_temp_float_buf,
            post_total_reduce_pipeline,
            post_total_reduce_bgl,
            volume_correct_pipeline,
            volume_correct_bgl,
            pre_total_atomic_buf,
            post_total_atomic_buf,
            temp_avg_pipeline,
            temp_avg_bgl,
            temp_diffuse_pipeline,
            temp_diffuse_bgl,
            light_diffuse_pipeline,
            light_diffuse_bgl,
            light_phi_bufs,
            light_diffuse_params_buf,
            light_frame_index: 0,
            surface_temp_b_buf,
            surface_a_buf,
            surface_b_buf,
            surface_params_buf,
            surface_render_params_buf,
            surface_res: 1,
            surface_material_mass_buf,
            surface_material_mass_res: 0,
            wave_step_pipeline,
            wave_step_bgl,
            wave_bufs,
            wave_params_buf,
            wave_frame_index: 0,
            wave_density_prev_buf,
            visibility_step_pipeline,
            visibility_step_bgl,
            visibility_buf,
            visibility_params_buf,
            band_hysteresis_step_pipeline,
            band_hysteresis_step_bgl,
            band_state_buf,
            band_hysteresis_params_buf,
            raw_splat_history_buf,
            phase_b_atomic_buf,
            phase_b_temp_atomic_buf,
            phase_b_temp_float_buf,
            phase_b_pre_total_atomic_buf,
            phase_b_post_total_atomic_buf,
            phase_b_a_buf,
            phase_b_b_buf,
            phase_b_raw_splat_history_buf,
            phase_b_params_buf,
            render_params_b_buf,
            phase_b_wave_bufs,
            phase_b_wave_density_prev_buf,
            phase_b_visibility_buf,
            phase_b_band_state_buf,
            surface_dual_render_pipeline,
            surface_dual_render_bgl,
            optical_table_buf,
            scratch: Vec::with_capacity(cap),
            color_mode: ColorMode::ByMaterial,
            vel_scale: 0.05,
            sigma_a: [[0.3f32; 3]; 16],
            sigma_s: [0.0f32; 16],
            specular_r0: [0.0f32; 16],
        }
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Call at init and on every resize.
    pub fn set_camera(
        &mut self,
        queue: &wgpu::Queue,
        grid_res: u32,
        width: u32,
        height: u32,
        particle_scale: f32,
        round_particles: bool,
    ) {
        let gr = grid_res as f32;
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let (sx, tx, sy, ty) = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        self.cached_ortho = (sx, tx, sy, ty);
        self.cached_grid_res = grid_res;
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraParams {
                view_proj: [
                    sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
                ],
                particle_scale,
                round_particles: round_particles as u32,
                _pad: [0.0; 2],
            }),
        );
    }

    /// Real light direction for `render_grid_volume`/surface-reconstruction
    /// shading, sourced from wherever the caller's own real light lives --
    /// LP callers should pass `SimConfig::light_dir` (the SAME real value
    /// already driving `rod::Phototropism`), not invent a separate one.
    /// Replaces a value each fragment shader used to hardcode independently
    /// (and inconsistently with the sim's own real light direction).
    pub fn set_light_dir(&mut self, x: f32, y: f32) {
        self.light_dir = (x, y);
    }

    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }
    pub fn set_vel_scale(&mut self, s: f32) {
        self.vel_scale = s;
    }

    /// Sets the Beer-Lambert absorption coefficient for `slot` AND uploads it to
    /// the GPU immediately, not just CPU-side state -- `render()`'s per-particle
    /// `particle_color()` path reads CPU state directly, but `render_grid_volume`'s
    /// GPU shader reads a GPU-resident buffer that would otherwise silently keep
    /// whatever was uploaded last, ignoring every material's real color in
    /// grid-volume mode. The redundant-write cost when setting several slots in a
    /// row is negligible (scene-setup-time only, never a per-frame path).
    pub fn set_optical_params(&mut self, queue: &wgpu::Queue, slot: usize, sigma_a: [f32; 3]) {
        self.sigma_a[slot % 16] = sigma_a;
        self.upload_optical_params(queue);
    }

    /// Reduced scattering coefficient for `slot` -- see `OpticalTable`'s doc for
    /// what this represents physically (real subsurface scattering, single-
    /// scattering approximation) and its real citation (Jacques 2013). Auto-
    /// uploads immediately -- see `set_optical_params`'s own doc for why.
    pub fn set_optical_scattering(&mut self, queue: &wgpu::Queue, slot: usize, sigma_s: f32) {
        self.sigma_s[slot % 16] = sigma_s;
        self.upload_optical_params(queue);
    }

    /// Specular Fresnel base reflectance R0 for `slot` -- see `OpticalTable`'s doc
    /// for the real-but-bounded caveat (constant near-normal reflectance, no
    /// surface-normal-dependent angle term). Auto-uploads immediately -- see
    /// `set_optical_params`'s own doc for why.
    pub fn set_specular_r0(&mut self, queue: &wgpu::Queue, slot: usize, r0: f32) {
        self.specular_r0[slot % 16] = r0;
        self.upload_optical_params(queue);
    }

    fn upload_optical_params(&self, queue: &wgpu::Queue) {
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );
    }

    // ── GPU compute render path ────────────────────────────────────────────────

    /// Zero-readback GPU render. No `sync_particles_blocking()` needed.
    pub fn render_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        if particle_count == 0 {
            return;
        }
        self.ensure_capacity(device, particle_count);

        queue.write_buffer(
            &self.render_config_buf,
            0,
            bytemuck::bytes_of(&RenderConfig {
                mode: self.color_mode as u32,
                particle_count: particle_count as u32,
                vel_scale: self.vel_scale,
                _pad: 0,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prep_bg"),
            layout: &self.prep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.storage_instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.render_config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_gpu"),
        });
        // Compute: fill the storage instance buffer from the particle buffer.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prep_instances"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.prep_pipeline);
            cp.set_bind_group(0, &prep_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(PREP_WG), 1, 1);
        }
        // GPU->GPU copy into the vertex buffer (decouples storage and vertex roles).
        let bytes = (particle_count * mem::size_of::<InstanceData>()) as u64;
        enc.copy_buffer_to_buffer(&self.storage_instances, 0, &self.instance_buffer, 0, bytes);
        // Render: draw instanced quads from the vertex buffer.
        self.draw_pass(&mut enc, output_view, clear, particle_count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Grid-volume render path ────────────────────────────────────────────────

    // ── CPU render path ────────────────────────────────────────────────────────

    /// CPU-fill render for the SoA `Particles` store (CPU `Simulation`).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &Particles,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles.iter() {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(&p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_soa"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    pub fn render_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[Particle],
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_cpu"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn ensure_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count > self.max_particles {
            let size = (count * mem::size_of::<InstanceData>()) as u64;
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances_storage"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.max_particles = count;
        }
    }

    fn draw_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: bool,
        count: usize,
    ) {
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            })
        } else {
            wgpu::LoadOp::Load
        };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, &self.render_bind_group, &[]);
        rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rp.set_vertex_buffer(1, self.instance_buffer.slice(..));
        rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..count as u32);
    }
}

// particle_color (the CPU-path per-particle color computation) is split into
// color.rs alongside the rest of the "Color helpers" section below -- see
// that file's own doc comment.
mod color;
use color::write_optical_table;

// Test suite split into its own file -- was ~150 of this file's ~930 lines,
// same pattern as `gpu/solver/device_lost_tests.rs`.
#[cfg(test)]
mod tests;
