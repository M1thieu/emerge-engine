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

use wgpu::util::DeviceExt;

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
use gpu_types::{CameraParams, GridVolumeParams, InstanceData, OpticalTable, RenderConfig};
pub use gpu_types::{DualPhaseSurfaceSource, GridVolumeSource, SurfaceReconstructionSource};

// wgpu pipeline construction (the three build_*_pipeline functions + their
// bind-group-layout helpers) lives in pipelines.rs -- see that file's doc.
mod pipelines;
use gpu_types::{
    BandHysteresisParams, GridVisibilityParams, SurfaceParams, SurfaceRenderParams,
    VisibilityParams, WaveStepParams,
};
use pipelines::{
    build_band_hysteresis_step_pipeline, build_grid_visibility_step_pipeline,
    build_grid_volume_pipeline, build_particle_pipeline, build_post_total_reduce_pipeline,
    build_prep_pipeline, build_surface_clear_pipeline, build_surface_convert_pipeline,
    build_surface_dual_render_pipeline, build_surface_iterate_pipeline,
    build_surface_render_pipeline, build_surface_splat_pipeline, build_temp_avg_pipeline,
    build_temp_diffuse_pipeline, build_visibility_step_pipeline, build_volume_correct_pipeline,
    build_wave_step_pipeline,
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

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            // VERTEX for draw; COPY_DST for both the CPU fill path and the GPU compute copy.
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GPU compute write target. Kept distinct from the vertex buffer: wgpu treats a
        // read_write storage buffer as an exclusive usage, so sharing one buffer for both
        // compute-write and vertex-read trips its usage tracker. Copied into instance_buffer.
        let storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances_storage"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_verts"),
            contents: bytemuck::cast_slice::<[f32; 2], u8>(&[
                [-0.5f32, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_idx"),
            contents: bytemuck::cast_slice::<u16, u8>(&[0u16, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_camera"),
            size: mem::size_of::<CameraParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_config"),
            size: mem::size_of::<RenderConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let optical_table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_optics"),
            size: mem::size_of::<OpticalTable>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_volume_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_volume_params"),
            size: mem::size_of::<GridVolumeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Real hysteresis visibility state for the grid-native path --
        // same real, disclosed all-zero starting bias as `visibility_buf`
        // above (every cell starts "not visible" until it genuinely earns
        // visibility on its own first real frame).
        let grid_visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_visibility_state"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let grid_visibility_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_visibility_params"),
            size: mem::size_of::<GridVisibilityParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Curvature-flow surface buffers -- allocated at a minimal 1-cell
        // placeholder size; `ensure_surface_capacity` (called from
        // `render_surface_reconstruction`, the only place `grid_res` is
        // actually known) grows all three together the first time it's
        // needed, same lazy-growth pattern `ensure_capacity` already uses
        // for the particle instance buffers.
        let surface_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_atomic"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Real mass-weighted temperature pair -- same minimal-placeholder-
        // then-grow convention as `surface_atomic_buf` above.
        let surface_temp_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_atomic"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_temp_float_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_float"),
            size: 4,
            // COPY_SRC: `fs_main`'s own final temperature source, same real
            // "readback/diagnostic tools need to copy FROM it" reason
            // `surface_a_buf` is COPY_SRC (see that field's own doc).
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Real volume-preserving-correction totals -- see `pre_total_atomic_
        // buf`'s own doc. Always exactly 4 bytes (one atomic i32), never
        // grown by `ensure_surface_capacity` -- a global scalar, not a
        // per-cell field.
        let pre_total_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pre_total_atomic"),
            size: 4,
            // COPY_SRC: diagnostic/test readback, same real reason
            // `surface_a_buf` has it (see that field's own doc).
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let post_total_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_total_atomic"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Ping-pong partner for the real thermal-diffusion PDE -- see
        // `temp_avg_pipeline`'s own doc.
        let surface_temp_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_b"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_a"),
            size: 4,
            // COPY_SRC: `surface_a` is where `CURVATURE_ITERATIONS` (even)
            // always settles the final result -- readback/diagnostic tools
            // need to copy FROM it, not just the render pass reading it.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let surface_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_b"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_params"),
            size: mem::size_of::<SurfaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface_render_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_render_params"),
            size: mem::size_of::<SurfaceRenderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // N-material extension -- minimal placeholder, same lazy-growth
        // convention as the surface buffers above, but grown independently
        // (only on opt-in) by `ensure_surface_material_mass_capacity`, not
        // by `ensure_surface_capacity`. See the struct field's own doc.
        let surface_material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_material_mass"),
            size: 4,
            // COPY_SRC: diagnostic/test readback, same real reason
            // `surface_a_buf` has it (see that field's own doc).
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Two-phase extension's own phase-B buffers -- same minimal-
        // placeholder-then-grow convention as phase A's own buffers above.
        let phase_b_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_atomic"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Phase B's own temperature pair -- see `surface_temp_atomic_buf`'s
        // own doc.
        let phase_b_temp_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_temp_atomic"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let phase_b_temp_float_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_temp_float"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Phase B's own volume-preserving-correction totals -- see
        // `pre_total_atomic_buf`'s own doc.
        let phase_b_pre_total_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_pre_total_atomic"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let phase_b_post_total_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_post_total_atomic"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let phase_b_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_a"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let phase_b_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_b"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let phase_b_raw_splat_history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_raw_splat_history"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let phase_b_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_params"),
            size: mem::size_of::<SurfaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let render_params_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_render_params_b"),
            size: mem::size_of::<SurfaceRenderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Phase B's own wave/visibility/band state -- same real, disclosed
        // placeholder-then-grow convention as every other buffer above.
        let phase_b_wave_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "phase_b_wave_0",
                    1 => "phase_b_wave_1",
                    _ => "phase_b_wave_2",
                }),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let phase_b_wave_density_prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_wave_density_prev"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let phase_b_visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_visibility_state"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let phase_b_band_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_band_state"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Real, persistent wave field -- placeholder-then-grow, same
        // convention as the surface buffers above. WebGPU/wgpu guarantees
        // newly created buffers start zero-filled (undisturbed water genuinely
        // starts at zero height -- no explicit clear pass needed). THREE
        // buffers, not two -- see this struct's own `wave_bufs` field doc.
        let wave_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "wave_0",
                    1 => "wave_1",
                    _ => "wave_2",
                }),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let wave_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave_params"),
            size: mem::size_of::<WaveStepParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Real temporal-disturbance history -- see `wave_density_prev_buf`'s
        // own doc. Starts all-zero (WebGPU guarantee), same real one-time
        // "body just appeared" excitation bias as the wave field itself.
        let wave_density_prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave_density_prev"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Real hysteresis visibility state -- single persistent buffer,
        // starts all-zero (WebGPU guarantee), meaning every cell starts
        // "not visible" until it genuinely earns visibility on its own
        // first real frame (a harmless, expected one-time conservative
        // bias from hysteresis itself, not a bug).
        let visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_state"),
            size: 4,
            // COPY_SRC: see `surface_a_buf`'s own doc -- readback/diagnostic
            // tools (incl. this crate's own tests) need to copy FROM it.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let visibility_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_params"),
            size: mem::size_of::<VisibilityParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Real hysteresis color-band state -- single persistent buffer,
        // same real, disclosed all-zero starting bias as `visibility_buf`
        // (every cell starts in band 0 until it genuinely earns a
        // different one on its own first real frame).
        let band_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("band_state"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let band_hysteresis_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("band_hysteresis_params"),
            size: mem::size_of::<BandHysteresisParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Real, persistent raw-splat history -- starts all-zero (WebGPU
        // guarantee), a real, harmless one-time bias (first frame's blend
        // ramps up to the true value over a few frames).
        let raw_splat_history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw_splat_history"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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

    /// Renders the solver's own grid mass field directly (see `grid_volume.wgsl`'s
    /// own doc for the real technique). Requires `set_camera` to have been called
    /// first (same as `render_gpu` needs for its own bind group) -- reuses the
    /// identical cached orthographic projection/grid_res so both modes line up on
    /// screen without re-deriving them.
    pub fn render_grid_volume(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: GridVolumeSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let (sx, tx, sy, ty) = self.cached_ortho;
        let grid_res = self.cached_grid_res;
        self.ensure_grid_visibility_capacity(device, grid_res);
        // Real per-particle cell-mass scale here is order 0.5-4 per occupied
        // cell; 0.15 requires non-trivial local density before showing anything,
        // instead of any measurable trace (which combined with bilinear smoothing
        // would overshoot true particle extent). SAME floor the visibility step
        // below gates on, so the hysteresis band and the raw discard agree.
        let mass_floor = 0.15;
        queue.write_buffer(
            &self.grid_volume_params_buf,
            0,
            bytemuck::bytes_of(&GridVolumeParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                grid_res,
                mass_floor,
                material_mass_enabled: source.material_mass_enabled as u32,
                _pad1: 0.0,
                _pad2: [0.0, 0.0],
            }),
        );
        queue.write_buffer(
            &self.grid_visibility_params_buf,
            0,
            bytemuck::bytes_of(&GridVisibilityParams {
                grid_res,
                mass_floor,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

        // Real hysteresis visibility step -- see `grid_volume.wgsl`'s own
        // `grid_visibility_step_main` doc. Reads the SAME raw grid buffer
        // the render pass below samples, must run before it in this
        // encoder so `fs_main`'s discard sees this frame's decision, not
        // last frame's.
        let grid_visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_visibility_step_bg"),
            layout: &self.grid_visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.grid_visibility_params_buf.as_entire_binding(),
                },
            ],
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_volume_bg"),
            layout: &self.grid_volume_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_volume_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: source.material_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.grid_visibility_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_grid_volume"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_visibility_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.grid_visibility_step_pipeline);
            cp.set_bind_group(0, &grid_visibility_step_bg, &[]);
            cp.dispatch_workgroups(grid_res.div_ceil(8), grid_res.div_ceil(8), 1);
        }
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
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_grid_volume"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
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
            rp.set_pipeline(&self.grid_volume_pipeline);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Curvature-flow surface reconstruction ──────────────────────────────────

    /// Real, finer-than-physics-grid surface reconstruction (see
    /// `curvature_flow.wgsl`'s own top doc for the full real technique --
    /// van der Laan et al. 2009 mean curvature flow on a particle-splatted
    /// auxiliary buffer, resolution-independent from the solver's own
    /// `grid_res`). Requires `set_camera` to have been called first (reuses
    /// its cached orthographic projection, rescaled to this pass's own
    /// finer `surface_res` -- see the real derivation in this function's
    /// body). `material_slot` picks ONE `OpticalTable` slot for the whole
    /// surface (real v1 scope: single dominant material, not full per-
    /// material phase-fraction separation -- see the shader's own doc).
    pub fn render_surface_reconstruction(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: SurfaceReconstructionSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let SurfaceReconstructionSource {
            particle_buf,
            particle_count,
            grid_res,
            material_slot,
            material_mass_enabled,
        } = source;
        if particle_count == 0 {
            return;
        }
        self.ensure_surface_capacity(device, grid_res);
        let surface_res = self.surface_res;
        if material_mass_enabled {
            self.ensure_surface_material_mass_capacity(device, surface_res);
        }

        queue.write_buffer(
            &self.surface_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceParams {
                grid_res,
                surface_res,
                particle_count: particle_count as u32,
                phase_filter_material_id: -1, // v1 behavior: every particle contributes
                material_mass_enabled: material_mass_enabled as u32,
            }),
        );

        // Real derivation, not a fresh camera computation: `set_camera`'s own
        // orthographic formula has `sx/sy` scale as 1/grid_res and `tx/ty`
        // independent of resolution entirely (translation, not scale) --
        // rescaling the ALREADY-cached grid-space transform by
        // `grid_res/surface_res` on sx/sy alone (tx/ty unchanged) gives the
        // exact same real projection expressed directly in this pass's own,
        // finer coordinate units, with no separate width/height bookkeeping.
        let (sx_grid, tx, sy_grid, ty) = self.cached_ortho;
        let res_ratio = grid_res as f32 / surface_res as f32;
        let sx = sx_grid * res_ratio;
        let sy = sy_grid * res_ratio;

        queue.write_buffer(
            &self.surface_render_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceRenderParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                surface_res,
                // Same real reasoning as `render_grid_volume`'s own mass_floor
                // (see that method's doc): a floor low enough to need real
                // local density, high enough that bilinear smoothing doesn't
                // overshoot true particle extent. The B-spline kernel here
                // deposits real mass (not a normalized [0,1] density), same
                // units `render_grid_volume` already uses.
                mass_floor: 0.15,
                material_slot,
                material_mass_enabled: material_mass_enabled as u32,
                _pad2: [0.0, 0.0],
            }),
        );

        queue.write_buffer(
            &self.wave_params_buf,
            0,
            bytemuck::bytes_of(&WaveStepParams {
                surface_res,
                _pad: [0; 3],
            }),
        );

        queue.write_buffer(
            &self.visibility_params_buf,
            0,
            bytemuck::bytes_of(&VisibilityParams {
                surface_res,
                mass_floor: 0.15,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        queue.write_buffer(
            &self.band_hysteresis_params_buf,
            0,
            bytemuck::bytes_of(&BandHysteresisParams {
                surface_res,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_splat_bg"),
            layout: &self.surface_splat_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.surface_temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pre_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.post_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.surface_material_mass_buf.as_entire_binding(),
                },
            ],
        });
        // clear_surface_main only ever reads binding 1/2/3/4/5/6 (its own
        // atomic buffers + params); binding 0 is unused but the layout is
        // shared with the splat pass, so bind SOMETHING real there too.
        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_clear_bg"),
            layout: &self.surface_clear_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.surface_temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pre_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.post_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.surface_material_mass_buf.as_entire_binding(),
                },
            ],
        });
        let convert_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_convert_bg"),
            layout: &self.surface_convert_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.raw_splat_history_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.surface_temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_iterate_a_to_b"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_iterate_b_to_a"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        // Real thermal-diffusion pair (see `curvature_flow.wgsl`'s own
        // "Pass 1c" doc): `temp_avg_bg` recovers real temperature from the
        // mass-weighted scatter using THIS frame's now-settled density
        // (`surface_a_buf`); `temp_diffuse_bg` then runs one real heat-
        // equation step, reading `surface_temp_b_buf` (avg's output) and
        // writing back into `surface_temp_float_buf` (the buffer `fs_main`
        // already binds -- no render-bind-group change needed).
        let temp_avg_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temp_avg_bg"),
            layout: &self.temp_avg_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_temp_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        let temp_diffuse_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temp_diffuse_bg"),
            layout: &self.temp_diffuse_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_temp_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        // Real, persistent (across frames) wave field -- see `curvature_
        // flow.wgsl`'s own "Pass 2b" doc. THREE distinct physical buffers
        // rotate through the "current" (read with neighbor offsets),
        // "previous" (self-index read only), and "next" (self-index write
        // only) roles each call -- wgpu's usage-scope validator rejects
        // binding the SAME buffer as both read-only and read_write within
        // one dispatch (confirmed via a real validation error when a
        // cheaper 2-buffer aliasing scheme was tried first), even though
        // that scheme's actual access pattern was index-disjoint and
        // logically hazard-free -- 3 buffers is the correct, always-valid
        // way to satisfy that rule for a leapfrog integrator. After this
        // dispatch, the buffer that served as "next" holds the freshly
        // computed state, which is what `render_bg` below must read.
        let cur_idx = (self.wave_frame_index % 3) as usize;
        let prev_idx = ((self.wave_frame_index + 2) % 3) as usize;
        let next_idx = ((self.wave_frame_index + 1) % 3) as usize;
        let wave_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave_step_bg"),
            layout: &self.wave_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.wave_bufs[cur_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.wave_bufs[prev_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.wave_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.wave_density_prev_buf.as_entire_binding(),
                },
            ],
        });

        // Real hysteresis visibility step -- a SINGLE persistent buffer
        // (no rotation needed: self-index read+write only, no neighbor
        // stencil, so no wgpu usage-scope conflict). Reads this frame's
        // settled density (`surface_a_buf`), same as the wave step above.
        let visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility_step_bg"),
            layout: &self.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visibility_params_buf.as_entire_binding(),
                },
            ],
        });

        // Real hysteresis color-band step -- same single-buffer, one-way-
        // downstream shape as the visibility step above (see "Pass 2d"
        // doc in the shader).
        let band_hysteresis_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("band_hysteresis_step_bg"),
            layout: &self.band_hysteresis_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.band_hysteresis_params_buf.as_entire_binding(),
                },
            ],
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_render_bg"),
            layout: &self.surface_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_render_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.surface_material_mass_buf.as_entire_binding(),
                },
            ],
        });

        let cell_count = surface_res * surface_res;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_surface_reconstruction"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_clear"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_clear_pipeline);
            cp.set_bind_group(0, &clear_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_splat"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_splat_pipeline);
            cp.set_bind_group(0, &splat_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(SURFACE_SPLAT_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_convert"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_convert_pipeline);
            cp.set_bind_group(0, &convert_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        // `CURVATURE_ITERATIONS` real dispatches, ping-ponged -- kept EVEN so
        // the settled result always lands back in `surface_a` (see this
        // struct's own field doc, and the module-level const assertion
        // below), letting `render_bg` above bind `surface_a`
        // unconditionally rather than choosing at runtime.
        let iterate_wg_x = surface_res.div_ceil(8);
        let iterate_wg_y = surface_res.div_ceil(8);
        for i in 0..CURVATURE_ITERATIONS {
            let bg = if i % 2 == 0 {
                &iterate_a_to_b
            } else {
                &iterate_b_to_a
            };
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_iterate"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_iterate_pipeline);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Real thermal-diffusion pair (see `curvature_flow.wgsl`'s own
        // "Pass 1c" doc) -- must run AFTER the density iterate loop above
        // (needs the final settled `surface_a_buf`) and BEFORE the render
        // pass below reads `surface_temp_float_buf` for blackbody emission.
        // Real ordering requirement, caught by a real test failure: this
        // MUST run BEFORE the volume-preserving correction below --
        // `temp_avg_main` divides `surface_temp_float_buf` (raw weighted
        // sum) by `surface_a_buf` (mass) to recover a real per-cell AVERAGE
        // temperature; that ratio is only meaningful against the mass value
        // the weighted sum was ORIGINALLY splatted against. Rescaling mass
        // first (for a completely different, unrelated reason -- volume
        // preservation) before this division silently corrupted the
        // recovered temperature by the same rescale factor (confirmed via
        // the real test: hot/cold both shifted by the identical ratio).
        // Average temperature is an INTENSIVE quantity -- it doesn't need
        // "volume preservation" at all, so it must be computed from the
        // real, as-settled mass, before that mass is corrected for anything
        // else.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temp_avg"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.temp_avg_pipeline);
            cp.set_bind_group(0, &temp_avg_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temp_diffuse"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.temp_diffuse_pipeline);
            cp.set_bind_group(0, &temp_diffuse_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Real volume-preserving correction (see `curvature_flow.wgsl`'s own
        // "Pass 1d" doc) -- must run AFTER temperature recovery above (see
        // that step's own doc for why) and BEFORE the wave/visibility/band
        // steps and the render pass, all of which need the CORRECTED mass
        // for alpha/banding/excitation purposes.
        {
            let post_total_reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post_total_reduce_bg"),
                layout: &self.post_total_reduce_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.surface_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.post_total_atomic_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.surface_params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("post_total_reduce"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.post_total_reduce_pipeline);
            cp.set_bind_group(0, &post_total_reduce_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let volume_correct_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("volume_correct_bg"),
                layout: &self.volume_correct_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.pre_total_atomic_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.post_total_atomic_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.surface_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.surface_params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("volume_correct"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.volume_correct_pipeline);
            cp.set_bind_group(0, &volume_correct_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        // Real wave-equation step, reading this frame's now-settled density
        // (`surface_a_buf`) as its excitation source -- must run AFTER the
        // iterate loop above finishes writing it, and BEFORE the render
        // pass below reads the wave field for shading.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wave_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.wave_step_pipeline);
            cp.set_bind_group(0, &wave_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Snapshot this frame's settled density as "previous" for next
        // frame's temporal-disturbance wave excitation. Must happen AFTER
        // the wave step above (which needed this frame's density as "now").
        // Plain buffer copy, no shader needed.
        enc.copy_buffer_to_buffer(
            &self.surface_a_buf,
            0,
            &self.wave_density_prev_buf,
            0,
            (cell_count as u64) * mem::size_of::<f32>() as u64,
        );
        // Real hysteresis visibility step -- also reads this frame's now-
        // settled density, independent of the wave step above (order
        // between the two doesn't matter, neither reads the other's
        // output).
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visibility_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.visibility_step_pipeline);
            cp.set_bind_group(0, &visibility_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Real hysteresis color-band step -- also reads this frame's now-
        // settled density, independent of the wave/visibility steps above.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("band_hysteresis_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.band_hysteresis_step_pipeline);
            cp.set_bind_group(0, &band_hysteresis_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }

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
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface_render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
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
            rp.set_pipeline(&self.surface_render_pipeline);
            rp.set_bind_group(0, &render_bg, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
        // Rotate which of the 3 wave buffers plays which role next call --
        // see the bind-group construction above's own doc for the full
        // rotation reasoning.
        self.wave_frame_index = self.wave_frame_index.wrapping_add(1);
    }

    /// Encodes clear+splat+convert+`CURVATURE_ITERATIONS`-iterate for ONE
    /// phase into the shared encoder -- the real, common body both
    /// `render_surface_reconstruction` (inlined, unchanged, zero risk to
    /// already-shipped code) and `render_surface_reconstruction_dual_phase`
    /// (below, calls this twice) both need. `params_buf` must already carry
    /// the real `phase_filter_material_id` for this specific phase.
    #[allow(clippy::too_many_arguments)]
    fn encode_phase_pipeline(
        &self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
        surface_res: u32,
        params_buf: &wgpu::Buffer,
        atomic_buf: &wgpu::Buffer,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        raw_splat_history_buf: &wgpu::Buffer,
        temp_atomic_buf: &wgpu::Buffer,
        temp_float_buf: &wgpu::Buffer,
        pre_total_buf: &wgpu::Buffer,
        post_total_buf: &wgpu::Buffer,
        // N-material extension's own buffer -- always bound (layout is
        // shared with the single-phase path), but a harmless dead-code
        // path here: both dual-phase calls set `material_mass_enabled: 0`
        // in `params_buf`, so the shader branch that reads this never
        // executes.
        material_mass_buf: &wgpu::Buffer,
    ) {
        let cell_count = surface_res * surface_res;
        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_clear_bg"),
            layout: &self.surface_clear_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pre_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: post_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: material_mass_buf.as_entire_binding(),
                },
            ],
        });
        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_splat_bg"),
            layout: &self.surface_splat_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pre_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: post_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: material_mass_buf.as_entire_binding(),
                },
            ],
        });
        let convert_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_convert_bg"),
            layout: &self.surface_convert_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raw_splat_history_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: temp_float_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_iterate_a_to_b"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_iterate_b_to_a"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_clear"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_clear_pipeline);
            cp.set_bind_group(0, &clear_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_splat"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_splat_pipeline);
            cp.set_bind_group(0, &splat_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(SURFACE_SPLAT_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_convert"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_convert_pipeline);
            cp.set_bind_group(0, &convert_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        let iterate_wg_x = surface_res.div_ceil(8);
        let iterate_wg_y = surface_res.div_ceil(8);
        for i in 0..CURVATURE_ITERATIONS {
            let bg = if i % 2 == 0 {
                &iterate_a_to_b
            } else {
                &iterate_b_to_a
            };
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_iterate"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_iterate_pipeline);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Real volume-preserving correction for THIS phase -- see
        // `render_surface_reconstruction`'s own identical step for the full
        // doc (`curvature_flow.wgsl`'s "Pass 1d").
        {
            let post_total_reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("phase_post_total_reduce_bg"),
                layout: &self.post_total_reduce_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: post_total_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_post_total_reduce"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.post_total_reduce_pipeline);
            cp.set_bind_group(0, &post_total_reduce_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let volume_correct_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("phase_volume_correct_bg"),
                layout: &self.volume_correct_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pre_total_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: post_total_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_volume_correct"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.volume_correct_pipeline);
            cp.set_bind_group(0, &volume_correct_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
    }

    /// Two-phase extension of `render_surface_reconstruction` (see
    /// `curvature_flow.wgsl`'s own "two-phase extension" doc and
    /// `DualPhaseSurfaceSource`'s doc): runs the real clear/splat/convert/
    /// iterate pipeline TWICE, once per material, into two fully
    /// independent buffer sets, so each phase gets its own real,
    /// independently-smoothed surface instead of merging at a shared
    /// interface into one blob. The final composite picks whichever
    /// phase has more real local density at each pixel (discrete
    /// winner-take-all, matching `grid_volume.wgsl`'s own "dominant
    /// material wins" convention).
    pub fn render_surface_reconstruction_dual_phase(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: DualPhaseSurfaceSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let DualPhaseSurfaceSource {
            particle_buf,
            particle_count,
            grid_res,
            material_id_a,
            material_id_b,
        } = source;
        if particle_count == 0 {
            return;
        }
        self.ensure_surface_capacity(device, grid_res);
        let surface_res = self.surface_res;

        queue.write_buffer(
            &self.surface_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceParams {
                grid_res,
                surface_res,
                particle_count: particle_count as u32,
                phase_filter_material_id: material_id_a as i32,
                // N-material extension is a single-phase-only mechanism,
                // unrelated to this 2-phase filter -- always off here.
                material_mass_enabled: 0,
            }),
        );
        queue.write_buffer(
            &self.phase_b_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceParams {
                grid_res,
                surface_res,
                particle_count: particle_count as u32,
                phase_filter_material_id: material_id_b as i32,
                material_mass_enabled: 0,
            }),
        );

        // Same real derivation as `render_surface_reconstruction`'s own doc
        // -- identical for both phases, they share one camera/surface_res.
        let (sx_grid, tx, sy_grid, ty) = self.cached_ortho;
        let res_ratio = grid_res as f32 / surface_res as f32;
        let sx = sx_grid * res_ratio;
        let sy = sy_grid * res_ratio;

        queue.write_buffer(
            &self.surface_render_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceRenderParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                surface_res,
                mass_floor: 0.15,
                material_slot: material_id_a,
                material_mass_enabled: 0,
                _pad2: [0.0, 0.0],
            }),
        );
        queue.write_buffer(
            &self.render_params_b_buf,
            0,
            bytemuck::bytes_of(&SurfaceRenderParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                surface_res,
                mass_floor: 0.15,
                material_slot: material_id_b,
                material_mass_enabled: 0,
                _pad2: [0.0, 0.0],
            }),
        );

        // Real per-phase wave/visibility/band params -- SHARED across both
        // phases (only `surface_res`/`mass_floor` matter here, identical for
        // both), same single-write-covers-both-dispatches convention the
        // single-phase path already established.
        queue.write_buffer(
            &self.wave_params_buf,
            0,
            bytemuck::bytes_of(&WaveStepParams {
                surface_res,
                _pad: [0; 3],
            }),
        );
        queue.write_buffer(
            &self.visibility_params_buf,
            0,
            bytemuck::bytes_of(&VisibilityParams {
                surface_res,
                mass_floor: 0.15,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &self.band_hysteresis_params_buf,
            0,
            bytemuck::bytes_of(&BandHysteresisParams {
                surface_res,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_surface_reconstruction_dual_phase"),
        });

        self.encode_phase_pipeline(
            device,
            &mut enc,
            particle_buf,
            particle_count,
            surface_res,
            &self.surface_params_buf,
            &self.surface_atomic_buf,
            &self.surface_a_buf,
            &self.surface_b_buf,
            &self.raw_splat_history_buf,
            &self.surface_temp_atomic_buf,
            &self.surface_temp_float_buf,
            &self.pre_total_atomic_buf,
            &self.post_total_atomic_buf,
            &self.surface_material_mass_buf,
        );
        self.encode_phase_pipeline(
            device,
            &mut enc,
            particle_buf,
            particle_count,
            surface_res,
            &self.phase_b_params_buf,
            &self.phase_b_atomic_buf,
            &self.phase_b_a_buf,
            &self.phase_b_b_buf,
            &self.phase_b_raw_splat_history_buf,
            &self.phase_b_temp_atomic_buf,
            &self.phase_b_temp_float_buf,
            &self.phase_b_pre_total_atomic_buf,
            &self.phase_b_post_total_atomic_buf,
            &self.surface_material_mass_buf,
        );

        // Real per-phase wave/visibility/band steps -- reuses the SAME
        // wave_frame_index rotation (see `render_surface_reconstruction`'s
        // own doc for the 3-buffer reasoning) applied to phase A's existing
        // buffers AND phase B's own separate set, so both phases get the
        // identical proven flicker fixes. Must run AFTER both
        // `encode_phase_pipeline` calls above (they need the now-settled
        // `surface_a_buf`/`phase_b_a_buf`) and BEFORE the render pass below
        // (which reads their output).
        let iterate_wg_x = surface_res.div_ceil(8);
        let iterate_wg_y = surface_res.div_ceil(8);
        let cur_idx = (self.wave_frame_index % 3) as usize;
        let prev_idx = ((self.wave_frame_index + 2) % 3) as usize;
        let next_idx = ((self.wave_frame_index + 1) % 3) as usize;
        let phase_a_wave_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_a_wave_step_bg"),
            layout: &self.wave_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.wave_bufs[cur_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.wave_bufs[prev_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.wave_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.wave_density_prev_buf.as_entire_binding(),
                },
            ],
        });
        let phase_b_wave_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_b_wave_step_bg"),
            layout: &self.wave_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_wave_bufs[cur_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.phase_b_wave_bufs[prev_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.phase_b_wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.wave_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.phase_b_wave_density_prev_buf.as_entire_binding(),
                },
            ],
        });
        let phase_a_visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_a_visibility_step_bg"),
            layout: &self.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visibility_params_buf.as_entire_binding(),
                },
            ],
        });
        let phase_b_visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_b_visibility_step_bg"),
            layout: &self.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visibility_params_buf.as_entire_binding(),
                },
            ],
        });
        let phase_a_band_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_a_band_step_bg"),
            layout: &self.band_hysteresis_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.band_hysteresis_params_buf.as_entire_binding(),
                },
            ],
        });
        let phase_b_band_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_b_band_step_bg"),
            layout: &self.band_hysteresis_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.band_hysteresis_params_buf.as_entire_binding(),
                },
            ],
        });
        for (label, bg) in [
            ("phase_a_wave_step", &phase_a_wave_step_bg),
            ("phase_b_wave_step", &phase_b_wave_step_bg),
            ("phase_a_visibility_step", &phase_a_visibility_step_bg),
            ("phase_b_visibility_step", &phase_b_visibility_step_bg),
            ("phase_a_band_step", &phase_a_band_step_bg),
            ("phase_b_band_step", &phase_b_band_step_bg),
        ] {
            let pipeline = if label.ends_with("wave_step") {
                &self.wave_step_pipeline
            } else if label.ends_with("visibility_step") {
                &self.visibility_step_pipeline
            } else {
                &self.band_hysteresis_step_pipeline
            };
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            cp.set_pipeline(pipeline);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Per-phase snapshot -- see the single-phase path's identical copy
        // above for the full doc.
        let wave_history_bytes = (surface_res * surface_res) as u64 * mem::size_of::<f32>() as u64;
        enc.copy_buffer_to_buffer(
            &self.surface_a_buf,
            0,
            &self.wave_density_prev_buf,
            0,
            wave_history_bytes,
        );
        enc.copy_buffer_to_buffer(
            &self.phase_b_a_buf,
            0,
            &self.phase_b_wave_density_prev_buf,
            0,
            wave_history_bytes,
        );

        let dual_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_dual_render_bg"),
            layout: &self.surface_dual_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_render_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.render_params_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.phase_b_wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.phase_b_visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.phase_b_band_state_buf.as_entire_binding(),
                },
            ],
        });

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
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface_dual_render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
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
            rp.set_pipeline(&self.surface_dual_render_pipeline);
            rp.set_bind_group(0, &dual_bg, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
        // Rotate which of the 3 wave-buffer slots plays which role next call
        // -- same real rotation `render_surface_reconstruction` performs,
        // shared here since both phases index off the SAME frame counter
        // (see the wave-step bind-group construction above's own doc).
        self.wave_frame_index = self.wave_frame_index.wrapping_add(1);
    }

    /// Grows the three curvature-flow surface buffers together when a
    /// caller's `grid_res * SURFACE_RES_MULTIPLIER` exceeds the currently
    /// allocated `surface_res` -- same lazy-growth pattern `ensure_capacity`
    /// already uses for the particle instance buffers.
    fn ensure_surface_capacity(&mut self, device: &wgpu::Device, grid_res: u32) {
        let needed = grid_res * SURFACE_RES_MULTIPLIER;
        if needed <= self.surface_res {
            return;
        }
        let cell_count = (needed * needed) as u64;
        let float_size = cell_count * mem::size_of::<f32>() as u64;
        self.surface_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Real mass-weighted temperature pair, grown together -- see
        // `surface_temp_atomic_buf`'s own doc.
        self.surface_temp_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.surface_temp_float_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_float"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Ping-pong partner, grown together -- see `temp_avg_pipeline`'s own
        // doc.
        self.surface_temp_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_b"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.surface_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_a"),
            size: float_size,
            // COPY_SRC: see the constructor's own placeholder allocation doc.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.surface_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_b"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Two-phase extension's own phase-B buffers, same sizes -- grown
        // together so `render_surface_reconstruction_dual_phase` never has
        // to special-case a mismatched capacity between the two phases.
        self.phase_b_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Phase B's own temperature pair, grown together -- see
        // `surface_temp_atomic_buf`'s own doc.
        self.phase_b_temp_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_temp_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.phase_b_temp_float_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_temp_float"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.phase_b_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_a"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.phase_b_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_b"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.phase_b_raw_splat_history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_raw_splat_history"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Real wave field, grown together -- a resize resets it to flat
        // (zero), the same real, disclosed behavior `surface_a_buf`/
        // `surface_b_buf` already have on resize (a rare, one-time event,
        // and "undisturbed" is a physically sensible reset state).
        self.wave_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "wave_0",
                    1 => "wave_1",
                    _ => "wave_2",
                }),
                size: float_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        // Real temporal-disturbance history, grown together -- see
        // `wave_density_prev_buf`'s own doc.
        self.wave_density_prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave_density_prev"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Real hysteresis visibility state, grown together -- a resize
        // resets it to all-"not visible" (the same real, disclosed,
        // harmless one-time bias the constructor's own placeholder
        // allocation already has).
        self.visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Real hysteresis color-band state, grown together -- a resize
        // resets it to band 0, same real, disclosed harmless bias the
        // constructor's own placeholder already has.
        self.band_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("band_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.raw_splat_history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw_splat_history"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Phase B's own wave/visibility/band state, grown together -- same
        // real, disclosed reset-to-flat/not-visible/band-0 bias as phase
        // A's own fields above.
        self.phase_b_wave_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "phase_b_wave_0",
                    1 => "phase_b_wave_1",
                    _ => "phase_b_wave_2",
                }),
                size: float_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        self.phase_b_wave_density_prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_wave_density_prev"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.phase_b_visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_visibility_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.phase_b_band_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_band_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.surface_res = needed;
    }

    /// N-material extension (see `surface_material_mass_buf`'s own doc) --
    /// grows it to `surface_res² × MAX_RENDER_MATERIAL_SLOTS × 4` bytes.
    /// Deliberately NOT called from `ensure_surface_capacity` above: only
    /// invoked when a caller actually opts in
    /// (`SurfaceReconstructionSource::material_mass_enabled`), so a
    /// `Renderer` that never opts in keeps paying only the 4-byte
    /// placeholder -- same real, disclosed lazy-growth reasoning as
    /// `GpuBuffers::grow_material_mass` on the solver side (`buffers.rs`),
    /// not the always-grow-together convention every other surface buffer
    /// above uses.
    fn ensure_surface_material_mass_capacity(&mut self, device: &wgpu::Device, surface_res: u32) {
        if self.surface_material_mass_res >= surface_res {
            return;
        }
        let cell_count = (surface_res * surface_res) as u64;
        self.surface_material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_material_mass"),
            size: cell_count * MAX_RENDER_MATERIAL_SLOTS as u64 * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.surface_material_mass_res = surface_res;
    }

    /// Grows `grid_visibility_buf` when a caller's `grid_res` exceeds the
    /// currently allocated `grid_visibility_res` -- same lazy-growth
    /// pattern `ensure_surface_capacity` uses above, just keyed on the
    /// solver's own `grid_res` instead of the finer `surface_res`.
    fn ensure_grid_visibility_capacity(&mut self, device: &wgpu::Device, grid_res: u32) {
        if grid_res <= self.grid_visibility_res {
            return;
        }
        let cell_count = (grid_res * grid_res) as u64;
        self.grid_visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_visibility_state"),
            size: cell_count * mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.grid_visibility_res = grid_res;
    }

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
