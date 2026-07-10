//! GPU uniform-buffer parameter structs: per-substep step params, force-field
//! entries, impulse entries, sleep/wake params, and spatial-block constants.
//!
//! Split out of `gpu/mod.rs` -- pure `#[repr(C)]`/`bytemuck::Pod` data plus
//! constructors. No wgpu device/buffer handling lives here; that's
//! `gpu::solver`.

use crate::solver::config::SimConfig;

/// Re-export so GPU code reads the same limit as the registry.
/// Injected into WGSL shaders at pipeline creation — change only in `materials/registry.rs`.
pub use crate::materials::registry::MAX_MATERIAL_SLOTS as MAX_MATERIALS;

/// Per-substep solver constants uploaded to the GPU uniform buffer before each substep.
///
/// 48 bytes, 16-byte aligned — satisfies WGSL uniform binding requirements.
/// Fields mirror `struct StepParams` in every WGSL shader exactly (same offsets, same types).
///
/// All values come from `SimConfig` or are computed from it — no hardcoded physics here.
/// Uniform data uploaded once per GPU substep.
///
/// Layout (48 bytes, 16-byte aligned — WGSL uniform binding requirement):
///   offset  0: grid_res       u32
///   offset  4: particle_count u32
///   offset  8: dt             f32
///   offset 12: kernel_d_inverse      f32  (always 4.0 — quadratic B-spline)
///   offset 16: gravity        `vec2<f32>`  (8 bytes; 8-byte aligned in WGSL ✓)
///   offset 24: boundary_thickness u32
///   offset 28: vel_limit      f32
///   offset 32: sleep_threshold f32  (0.0 = sleep/wake disabled, SimConfig default)
///   offset 36: _pad           [u32; 3]
///                             = 48 bytes, 16-byte aligned ✓
///
/// `gravity: Vec2` replaces the old `gravity: f32` + `_pad1: u32` pair —
/// same byte count, no layout change for other fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuStepParams {
    pub grid_res: u32,
    pub particle_count: u32,
    pub dt: f32,
    pub kernel_d_inverse: f32,
    pub gravity: glam::Vec2, // SimConfig::gravity — supports angled/planetary gravity
    pub boundary_thickness: u32,
    pub vel_limit: f32,       // grid_cell_size / sub_dt
    pub sleep_threshold: f32, // SimConfig::sleep_threshold — 0.0 disables sleep/wake entirely
    pub _pad: [u32; 3],
}

impl GpuStepParams {
    pub fn new(config: &SimConfig, sub_dt: f32, particle_count: usize) -> Self {
        Self {
            grid_res: config.grid_res as u32,
            particle_count: particle_count as u32,
            dt: sub_dt,
            kernel_d_inverse: crate::solver::config::KERNEL_D_INVERSE,
            gravity: config.gravity,
            boundary_thickness: config.boundary_thickness as u32,
            vel_limit: config.grid_cell_size / sub_dt,
            sleep_threshold: config.sleep_threshold,
            _pad: [0; 3],
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuStepParams>() == 48);

/// Maximum number of active GPU force-field entries per frame.
/// Must match `MAX_FORCE_FIELDS` in `force_fields.wgsl`.
pub const MAX_FORCE_FIELDS: usize = 16;

/// Field-type discriminants — match `FIELD_*` constants in `force_fields.wgsl`.
pub mod field_type {
    pub const DISABLED: u32 = 0;
    pub const GRAVITY_WELL: u32 = 1;
    pub const COULOMB: u32 = 2;
    pub const AABB_CONFINEMENT: u32 = 3;
    pub const RADIAL_CONFINEMENT: u32 = 4;
    pub const UNIFORM_ELECTRIC: u32 = 5;
    pub const BUOYANCY: u32 = 6;
}

/// One GPU force-field entry — 48 bytes, 16-byte aligned.
/// Matches `struct FieldEntry` in `force_fields.wgsl` exactly (size-asserted).
/// Use the named constructors instead of filling `params` manually.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFieldEntry {
    pub field_type: u32,
    pub material_mask: u32,
    pub _pad: [u32; 2],
    pub params: [f32; 8],
}

const _: () = assert!(core::mem::size_of::<GpuFieldEntry>() == 48);

impl GpuFieldEntry {
    /// material_mask value for a field that affects all materials.
    pub const ALL_MATERIALS: u32 = 0xFFFF_FFFF;

    /// Plummer-softened point-mass gravity: a = −G·M·r / (r²+ε²)^(3/2).
    ///
    /// - `gm`: gravitational_constant × source_mass (positive = attractive)
    /// - `softening_sq`: Plummer ε² (prevents singularity at r=0)
    /// - `cutoff`: hard cutoff distance (0.0 = no cutoff)
    /// - `switch_on`: force-switch onset (< cutoff; force tapers from `switch_on` to `cutoff`)
    pub fn gravity_well(
        pos: glam::Vec2,
        gm: f32,
        softening_sq: f32,
        cutoff: f32,
        switch_on: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = pos.x;
        p[1] = pos.y;
        p[2] = gm;
        p[3] = softening_sq;
        p[6] = cutoff;
        p[7] = switch_on;
        Self {
            field_type: field_type::GRAVITY_WELL,
            material_mask: Self::ALL_MATERIALS,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Plummer-softened Coulomb interaction for one (source, material) pair.
    ///
    /// - `charge_factor`: k × q_source × q_particle (signed; positive = repulsion)
    /// - `softening_sq`: Plummer ε²
    /// - `material_id`: which material's particles are affected (bitmask = 1 << id)
    /// - `cutoff` / `switch_on`: same as `gravity_well`
    pub fn coulomb(
        pos: glam::Vec2,
        charge_factor: f32,
        softening_sq: f32,
        material_id: u32,
        cutoff: f32,
        switch_on: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = pos.x;
        p[1] = pos.y;
        p[2] = charge_factor;
        p[3] = softening_sq;
        p[6] = cutoff;
        p[7] = switch_on;
        Self {
            field_type: field_type::COULOMB,
            material_mask: 1 << material_id,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Soft repulsive walls of an axis-aligned bounding box.
    ///
    /// Particles that penetrate within `thickness` cells of any wall get a
    /// restoring acceleration proportional to penetration depth × `stiffness`.
    pub fn aabb_confinement(
        min: glam::Vec2,
        max: glam::Vec2,
        stiffness: f32,
        thickness: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = min.x;
        p[1] = min.y;
        p[2] = max.x;
        p[3] = max.y;
        p[4] = stiffness;
        p[5] = thickness;
        Self {
            field_type: field_type::AABB_CONFINEMENT,
            material_mask: Self::ALL_MATERIALS,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Soft inward repulsion outside a radial shell.
    ///
    /// Particles beyond `radius − thickness` receive an inward acceleration
    /// proportional to excess penetration × `stiffness`.
    pub fn radial_confinement(
        center: glam::Vec2,
        radius: f32,
        stiffness: f32,
        thickness: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = center.x;
        p[1] = center.y;
        p[2] = radius;
        p[3] = stiffness;
        p[4] = thickness;
        Self {
            field_type: field_type::RADIAL_CONFINEMENT,
            material_mask: Self::ALL_MATERIALS,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Spatially-constant electric field: a = q · E / m.
    ///
    /// - `field`: E-field vector (simulation units — force per unit charge)
    /// - `charge`: per-particle charge for `material_id` (same units as the Coulomb constant)
    /// - `material_id`: only particles of this material are affected
    pub fn uniform_electric(field: glam::Vec2, charge: f32, material_id: u32) -> Self {
        let mut p = [0f32; 8];
        p[0] = field.x;
        p[1] = field.y;
        p[2] = charge;
        Self {
            field_type: field_type::UNIFORM_ELECTRIC,
            material_mask: 1 << material_id,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Archimedes buoyancy for particles of `material_id` floating in a denser fluid.
    ///
    /// - `gravity`: must match `SimConfig::gravity` (solver gravity, including sign)
    /// - `fluid_density_grid`: surrounding fluid's rest_density in grid units
    ///   (`ρ_SI · dx_m²` — same value as `NewtonianFluidMaterial::rest_density`, fixed
    ///   2026-07-07 to drop an incorrect extra `/dt_s²` factor)
    /// - `material_id`: only particles of this material receive the buoyancy force
    ///
    /// Uses particle rest density (`mass / initial_volume`) not instantaneous density,
    /// preventing the expansion-buoyancy runaway where expanded fluid appears falsely light.
    /// Applies `Δv = −gravity · (fluid_density / ρ₀_particle − 1) · dt` each substep.
    pub fn buoyancy(gravity: glam::Vec2, fluid_density_grid: f32, material_id: u32) -> Self {
        let mut p = [0f32; 8];
        p[0] = gravity.x;
        p[1] = gravity.y;
        p[2] = fluid_density_grid;
        p[3] = 1.0e-4; // min_density floor — mirrors BuoyancyField::new default
        Self {
            field_type: field_type::BUOYANCY,
            material_mask: 1 << material_id,
            _pad: [0; 2],
            params: p,
        }
    }
}

/// Uniform buffer containing all active GPU force-field entries — 784 bytes.
/// Matches `struct FieldsParams` in `force_fields.wgsl` exactly (size-asserted).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFieldsParams {
    pub count: u32,
    pub _pad: [u32; 3],
    pub entries: [GpuFieldEntry; MAX_FORCE_FIELDS],
}

const _: () = assert!(core::mem::size_of::<GpuFieldsParams>() == 784);

/// Max impulses per frame submitted via `apply_impulse` / `apply_radial_impulse`.
/// Must match `array<ImpulseEntry, 16>` in `apply_impulses.wgsl`.
pub const MAX_GPU_IMPULSES: usize = 16;

/// One impulse descriptor — 32 bytes, matches `struct ImpulseEntry` in WGSL.
///
/// mode 0 = radial: `v += normalize(p - center) * strength * falloff`
/// mode 1 = directional: `v += force * falloff`
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuImpulseEntry {
    pub center: [f32; 2], // grid-space origin
    pub radius: f32,
    pub strength: f32,   // radial only (signed)
    pub force: [f32; 2], // directional only
    pub mode: u32,       // 0 = radial, 1 = directional
    pub _pad: u32,
}

const _: () = assert!(core::mem::size_of::<GpuImpulseEntry>() == 32);

/// Uniform data for the apply_impulses compute pass — 528 bytes.
/// Matches `struct ImpulseParams` in `apply_impulses.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuImpulseParams {
    pub count: u32,
    pub vel_limit: f32,
    pub particle_count: u32,
    pub _pad: u32,
    pub entries: [GpuImpulseEntry; MAX_GPU_IMPULSES],
}

const _: () = assert!(core::mem::size_of::<GpuImpulseParams>() == 528);

/// Max tags per frame for force-sleep/force-wake-by-tag.
/// Must match `array<u32, 8>` in `force_fields.wgsl`.
///
/// Minimal hook for LP's future chunk system (see `mpm_technique_survey` memory
/// note): a chunk leaving camera range force-sleeps its particles by `user_tag`
/// regardless of velocity; a chunk re-entering range force-wakes them. The chunk
/// system itself — tagging particles by chunk, tracking camera distance — is
/// LP's job, not emerge's. This is just the primitive it needs.
pub const MAX_SLEEP_WAKE_TAGS: usize = 8;

/// Uniform data for force-sleep/force-wake-by-tag, checked once per substep in
/// `force_fields.wgsl` — 80 bytes. Matches `struct SleepWakeParams` in WGSL.
///
/// Tags are packed 4-per-`vec4<u32>` (`[[u32; 4]; 2]` = 8 tags), not a flat
/// `[u32; 8]` — WGSL requires uniform-address-space arrays to have a 16-byte
/// element stride, so a flat u32 array would be rejected by naga at shader-module
/// creation (same class of gotcha as `vec3<u32>` padding elsewhere in this file).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSleepWakeParams {
    pub sleep_count: u32,
    pub wake_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub sleep_tags: [[u32; 4]; MAX_SLEEP_WAKE_TAGS / 4],
    pub wake_tags: [[u32; 4]; MAX_SLEEP_WAKE_TAGS / 4],
}

const _: () = assert!(core::mem::size_of::<GpuSleepWakeParams>() == 80);

/// Spatial-block bucket geometry for the particle_sort histogram AND the
/// active-block detection it now also feeds (GPU sparse grid, Phase 1 — see
/// `mpm_technique_survey` memory note). Single Rust-side source of truth: must
/// match `NUM_BLOCKS_PER_DIM`/`NUM_BLOCKS` in `particle_sort.wgsl` and
/// `grid_clear.wgsl` exactly. Re-deriving from `grid_res` at runtime is not an
/// option — this sizes `block_counts`/`active_block_ids`, both allocated once
/// at `GpuBuffers::new()`, so it must be a fixed compile-time constant, same
/// class as `MAX_FORCE_FIELDS`.
pub const NUM_BLOCKS_PER_DIM: usize = 16;
pub const NUM_BLOCKS: usize = NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM; // 256
