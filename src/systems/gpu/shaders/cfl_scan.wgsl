// cfl_scan — per-substep GPU-native CFL bound reduction for strict WC-MPM fluids.
//
// CPU's `Simulation::step()` re-evaluates its CFL bound every single substep (near-
// zero cost: an in-process, synchronous array scan -- see `choose_substep_dt`).
// GPU's batch-oriented `step_frame` previously only rescanned once per up-to-64-
// substep batch, via a CPU-side mirror of particle state. Under a violent enough
// compression event (basic_fluids_gpu.rs's real, reproduced crash, 2026-08-08: J
// up to 34653 under real gravity into a wall/mud contact), a whole batch could run
// on a dt that was safe when the batch started but became unsafe partway through,
// before the next rescan/sync ever caught it -- confirmed by testing a much finer
// SUBSTEP_BATCH_SIZE=8, which measurably reduced (not eliminated) the blowup,
// proving finer-grained rescanning helps but per-batch reactivity alone is not
// sufficient at this violence level.
//
// This pass restores CPU-parity, per-substep reactivity, GPU-natively (no CPU
// particle-buffer round trip): it reduces four raw, state-dependent quantities --
//   [0] max |v|                                          (velocity CFL term)
//   [1] max deformation-gradient rate (Frobenius norm of velocity_gradient)
//                                                          (deformation_gradient_cfl_bound term)
//   [2] max Tait EOS c² numerator, strict fluids only     (material acoustic term)
//   [3] 1.0 iff any strict-fluid particle sits within `boundary_thickness`
//       cells of a wall, else 0.0 -- the SAME real, CPU-proven mechanism as
//       `SimConfig::fluid_near_wall_cfl_scale` (see that field's own doc,
//       and MEMORY.md's fluid-recovery notes Round 7-9 for the full,
//       independently-verified derivation), ported to GPU 2026-08-09 (the
//       CPU version predates tonight; GPU never had it -- "GPU-side
//       equivalent not yet ported" was an explicitly disclosed gap). No
//       compression gate here (unlike CPU's ORIGINAL acoustic-only use) --
//       deliberately: a compression gate is reactive by construction,
//       exactly what fails at the critical FIRST substep before anything
//       has moved (the same real reasoning CPU's gravity-bound use already
//       established). `fluid_near_wall_cfl_scale` itself is applied in Rust
//       (see `reactive_gpu_substep_dt`), not read here -- this just detects
//       whether it should apply at all, keeping the shader free of another
//       scene-wide constant.
// -- via atomicMax on bitcast<u32>(x). Exact, not approximate, for x>0: IEEE754 bit
// patterns are monotonically increasing with magnitude for positive floats, a
// standard order-preserving trick (no atomic<f32> in WebGPU/WGSL). The scene-wide,
// CPU-known coefficients (cfl_coefficient, material_cfl_coefficient, grid_cell_size,
// gravity, fluid_near_wall_cfl_scale) are applied to these raw maxima AFTERWARD in
// Rust (see step.rs's `reactive_gpu_substep_dt`) -- this avoids growing
// `StepParams`, which would ripple through all 11 shaders that mirror its layout
// for no benefit (these coefficients never vary mid-scene).

struct Particle {
    x:                    vec2<f32>,
    v:                    vec2<f32>,
    velocity_gradient:    mat2x2<f32>,
    deformation_gradient: mat2x2<f32>,
    mass:                 f32,
    initial_volume:       f32,
    volume:               f32,
    density:              f32,
    material_id:          u32,
    plastic_volume_ratio: f32,
    hardening_scale:      f32,
    friction_hardening:   f32,
    log_volume_strain:    f32,
    temperature:          f32,
    user_tag:             u32,
    activation:           f32,
    activation_dir:       vec2<f32>,
    muscle_group_id:      u32,
    contact_group:        u32,
    sleeping:             u32,
    pinned:               u32,
    scalar_field:         f32,
    internal_pressure:    f32,
}

struct MaterialParams {
    model:                   u32,
    lambda:                  f32,
    mu:                      f32,
    hardening_exponent:      f32,
    compression_limit:       f32,
    stretch_limit:           f32,
    rest_density:            f32,
    eos_stiffness:           f32,
    eos_power:               f32,
    dynamic_viscosity:       f32,
    volume_ratio_min:        f32,
    volume_ratio_max:        f32,
    dp_h0:                   f32,
    dp_h1:                   f32,
    dp_h2:                   f32,
    dp_h3:                   f32,
    active_stress_coeff:     f32,
    hardening_modulus:       f32,
    thermal_viscosity_coeff: f32,
    thermal_expansion:       f32,
    pressure_floor:          f32,
    bulk_viscosity:          f32,
    critical_shear_rate:     f32,
    cohesion_coeff:          f32,
}

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    reserved_velocity_slot: f32,
    sleep_threshold:    f32,
    contact_friction:   f32,
    grid_cell_size:     f32,
    contact_active:     u32,
}

const MAX_MATERIALS: u32 = {{MAX_MATERIALS}}u;
// ConstitutiveModel::Fluid = 1 (src/matter/materials/mod.rs) -- strict WC-MPM
// liquid, the only model this reduction's acoustic term applies to.
const MODEL_FLUID: u32 = 1u;

// Regional-substepping Step 1 (2026-08-12, purring-swinging-cookie.md Part A
// section 1): duplicated from particle_sort.wgsl's own `block_index`/
// `NUM_BLOCKS_PER_DIM`/`NUM_BLOCKS` -- same real, already-established
// convention every shader module in this pipeline uses (each WGSL
// compilation unit independently redeclares its own mirror of shared
// structs/constants; there is no cross-file `#include` in WGSL). Not a new
// partition -- the SAME 256-block spatial bucketing particle_sort/grid_clear/
// grid_update already use for the sparse-grid active-block dispatch.
override NUM_BLOCKS_PER_DIM: u32;
const NUM_BLOCKS: u32 = 256u; // NUM_BLOCKS_PER_DIM² -- array sizes can't be override-derived

fn block_index(pos: vec2<f32>, grid_res: u32) -> u32 {
    let max_cell = grid_res - 1u;
    let cell_x = u32(clamp(pos.x, 0.0, f32(max_cell)));
    let cell_y = u32(clamp(pos.y, 0.0, f32(max_cell)));
    let block_size = (grid_res + NUM_BLOCKS_PER_DIM - 1u) / NUM_BLOCKS_PER_DIM; // ceil div
    let block_x = min(cell_x / block_size, NUM_BLOCKS_PER_DIM - 1u);
    let block_y = min(cell_y / block_size, NUM_BLOCKS_PER_DIM - 1u);
    return block_y * NUM_BLOCKS_PER_DIM + block_x;
}

@group(0) @binding(0) var<storage, read_write> particles:    array<Particle>;
@group(0) @binding(2) var<uniform>              materials:   array<MaterialParams, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform>              step_params: StepParams;

// Cleared to 0u before each dispatch (Rust side) -- 0u bitcasts to the float
// 0.0, a valid, harmless starting point for an atomicMax reduction over
// positive values.
@group(2) @binding(33) var<storage, read_write> cfl_reduction: array<atomic<u32>>;

// Regional-substepping Step 1: the SAME 4 quantities as `cfl_reduction`
// above, reduced PER-BLOCK instead of globally -- lets a future batch-start
// classification pass (not built yet, this is Step 1 only: populate real
// data, no consumer yet) ask "what CFL bound does THIS block alone need"
// instead of only the whole-domain minimum. NUM_BLOCKS*4 = 1024 atomics,
// structurally derived from the existing 256-block partition x the existing
// 4-quantity layout above, not a chosen size. Binding 35, not 34 -- 34 is
// already `cohesion_params` (added earlier tonight, see grid_update.wgsl).
@group(2) @binding(35) var<storage, read_write> block_cfl_reduction: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn cfl_scan_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= step_params.particle_count { return; }
    let p = particles[i];
    if p.sleeping != 0u { return; }
    // Regional-substepping Step 1: this particle's own block, computed once,
    // reused by all 4 per-block writes below (same 4 quantities, same slot
    // order, as the global `cfl_reduction` above).
    let b = block_index(p.x, step_params.grid_res);

    let speed = length(p.v);
    if speed > 0.0 && speed < 1.0e30 {
        atomicMax(&cfl_reduction[0], bitcast<u32>(speed));
        atomicMax(&block_cfl_reduction[b * 4u + 0u], bitcast<u32>(speed));
    }

    let cg   = p.velocity_gradient;
    let rate = sqrt(dot(cg[0], cg[0]) + dot(cg[1], cg[1]));
    if rate > 0.0 && rate < 1.0e30 {
        atomicMax(&cfl_reduction[1], bitcast<u32>(rate));
        atomicMax(&block_cfl_reduction[b * 4u + 1u], bitcast<u32>(rate));
    }

    let mat = materials[p.material_id];
    if mat.model == MODEL_FLUID && p.initial_volume > 0.0 {
        let j     = p.volume / p.initial_volume;
        if j > 0.0 && mat.rest_density > 0.0 {
            let ratio = (mat.rest_density / j) / mat.rest_density;
            let c2    = mat.eos_stiffness * mat.eos_power
                * pow(max(ratio, 1.0e-8), mat.eos_power - 1.0) / mat.rest_density;
            if c2 > 0.0 && c2 < 1.0e30 {
                atomicMax(&cfl_reduction[2], bitcast<u32>(c2));
                atomicMax(&block_cfl_reduction[b * 4u + 2u], bitcast<u32>(c2));
            }
        }
        // Same wall-proximity zone `apply_slip_wall_velocity`/CPU's `is_near_wall`
        // already treat specially, not a new margin.
        //
        // ROOT-CAUSE FIX (2026-08-11, real, tested): this flag was raised on wall
        // PROXIMITY ALONE, missing CPU's own compression gate
        // (`SimConfig::fluid_near_wall_compression_threshold`, `cfl.rs`,
        // added 2026-08-08). A floor is a wall, so any fluid simply RESTING
        // paid the 20x `fluid_near_wall_cfl_scale` tightening permanently ->
        // ~655 substeps/frame -> ~5000 GPU dispatches/frame -> the measured
        // 1 fps. CPU gates on a particle genuinely being squeezed RIGHT NOW;
        // GPU never did. `j` is already computed just above, so this reuses
        // real existing state rather than adding any new measurement.
        let t  = f32(step_params.boundary_thickness);
        let hi = f32(step_params.grid_res) - t;
        let near_wall_zone = p.x.x < t || p.x.x > hi || p.x.y < t || p.x.y > hi;
        // GPU PORT (2026-08-11) of CPU's Mach-relative compression threshold
        // (`SimConfig::fluid_near_wall_compression_mach_margin`, `cfl.rs`) --
        // replaces a fixed absolute percentage, which self-defeats for a
        // deliberately Monaghan-softened EOS: weakly-compressible WC-MPM's
        // own design tolerates ~1% compression AS THE NORM (that IS the
        // Mach<0.1 validity target), not an anomaly, so a fixed 1% threshold
        // reads ordinary hydrostatic compression as a permanent wall-contact
        // event for a soft EOS. Real relation: Ma^2 ~= density fluctuation
        // (Zhang et al., "A variable speed of sound formulation for weakly
        // compressible SPH", arXiv:2310.04139) -- compares |J-1| to what
        // THIS material's own acoustic stiffness predicts as normal at the
        // scene's actual current flow speed, not a scene-independent
        // constant. `step_params.reserved_velocity_slot` carries the
        // previous batch's real measured max particle speed (Rust-side
        // `GpuSimulation::last_max_particle_speed`, one-batch-lagged --
        // same real data CPU's own `last_max_speed` param uses, not an
        // estimate; repurposes the former "always zero" legacy ABI slot,
        // same convention as `contact_friction`/`grid_cell_size`/
        // `contact_active` already repurposing the other 3 pad slots).
        const NEAR_WALL_COMPRESSION_THRESHOLD: f32 = 0.01;
        // Matches `SimConfig::fluid_near_wall_compression_mach_margin`'s
        // real default (2.0) -- disclosed shader constant, same convention
        // as STRICT_FLUID_J_MAX; no scene currently overrides it, and
        // threading it through `StepParams` would change that struct's
        // fixed 48-byte layout every shader mirrors exactly.
        const NEAR_WALL_COMPRESSION_MACH_MARGIN: f32 = 2.0;
        var near_wall_threshold = NEAR_WALL_COMPRESSION_THRESHOLD;
        if mat.eos_stiffness > 0.0 && mat.rest_density > 0.0 {
            let c2_rest = mat.eos_stiffness * mat.eos_power / mat.rest_density;
            if c2_rest > 1.0e-8 {
                let mach = step_params.reserved_velocity_slot / sqrt(c2_rest);
                // REAL FIX (2026-08-12): was a bare `(mach*mach)*MARGIN`, with
                // no floor. As the scene settles, `mach -> 0` (last_max_speed
                // -> 0), so this threshold ALSO shrinks toward zero -- making
                // the gate MORE sensitive exactly as the fluid calms down,
                // backwards from the intended design. A fully settled fluid's
                // own inevitable floating-point/discretization residual in J
                // then trips this ever-shrinking threshold every substep,
                // forever, even at max_speed<1 -- confirmed live: a water-only
                // dam-break scene stayed at sub=1000-1700+/frame (fps=0) long
                // after fully settling (max_speed 0.68-0.96). Floored at the
                // SAME `NEAR_WALL_COMPRESSION_THRESHOLD` (0.01) already used
                // as the fallback for materials with no acoustic term -- its
                // own doc already establishes this as the real, disclosed
                // "real water compressibility is negligible past ~1%"
                // baseline, reused here as a floor rather than inventing a
                // new constant: the Mach-relative term can only TIGHTEN this
                // baseline for genuinely violent events, never shrink below
                // it for an already-calm one.
                near_wall_threshold = max(
                    (mach * mach) * NEAR_WALL_COMPRESSION_MACH_MARGIN,
                    NEAR_WALL_COMPRESSION_THRESHOLD,
                );
            }
        }
        if near_wall_zone && abs(j - 1.0) > near_wall_threshold {
            atomicMax(&cfl_reduction[3], bitcast<u32>(1.0));
            atomicMax(&block_cfl_reduction[b * 4u + 3u], bitcast<u32>(1.0));
        }
    }
}
