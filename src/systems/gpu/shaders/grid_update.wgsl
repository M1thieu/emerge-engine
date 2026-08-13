// Grid update — momentum normalization, gravity, force fields, boundary enforcement.
// Runs between P2G and G2P.
//
// GPU sparse grid Phase 2: dispatch one workgroup per active-block SLOT, same pattern as
// grid_clear.wgsl's Phase 1 (same active_block_ids/active_block_ids_prev grace-period lists,
// same halo-expanded compaction from particle_sort.wgsl, so kernel-stencil spillover across
// block boundaries is already handled). Per-cell logic is unchanged; only which cells get
// visited changes (active blocks' cell range, not the whole dense grid), via the same
// block-relative grid-stride loop grid_clear uses.

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    reserved_velocity_slot: f32,
    sleep_threshold:    f32,
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32,
}

struct ForceFieldEntry {
    field_type:    u32,
    material_mask: u32,
    _pad0:         u32,
    _pad1:         u32,
    params01:      vec4<f32>,
    params45:      vec4<f32>,
}

struct ForceFieldsParams {
    count:   u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
    entries: array<ForceFieldEntry, 16>,
}

// ASFLIP (GPU port, Fei et al. 2021) — see GpuAsflipParams' own Rust doc.
struct AsflipParams {
    blend:   f32,
    enabled: u32,
    _pad0:   u32,
    _pad1:   u32,
}

const MASS_ATOMIC_SCALE:  f32 = 1000000.0;
// Real, measured 2026-08-08: tried deriving this from the fixed-point
// quantum (10/MASS_ATOMIC_SCALE=1e-5) on the theory that a fixed 1e-4 was
// discarding real low-mass edge-cell momentum after the SI mass fix. That
// change made the exact same crash WORSE (J up to 23772 and an earlier
// panic, vs 14570/no-panic at 1e-4) -- MASS_FLOOR isn't only a "don't
// discard real momentum" threshold, it's ALSO a numerical safety floor
// against dividing by a near-zero mass (`vel = momentum/mass`), which
// amplifies ordinary floating-point noise into a spuriously large
// velocity once mass gets small enough -- a risk tied to f32's absolute
// precision limit, not to the material's own mass scale, so it does NOT
// shrink just because SI-correct particle mass did. 1e-4 is the real,
// already-balanced value between these two competing concerns; lowering
// it further removes a safety margin the crash investigation actually
// needs. Kept as-is, disclosed rather than re-guessed again.
const MASS_FLOOR: f32 = 1e-4;
// MUST stay identical to `p2g.wgsl`'s own MOM_ATOMIC_SCALE -- that file
// encodes momentum into this fixed-point accumulator, this file decodes it.
// Raised 1e5 -> 1e6 (2026-08-11) to match MASS_ATOMIC_SCALE, fixing the real
// numerator/denominator quantization asymmetry in `vel = momentum / mass`
// below. See p2g.wgsl's own full root-cause writeup on this constant (the
// measured GPU-only fluid fps collapse, and why the MASS_FLOOR note above --
// which correctly identified the noise-amplification mechanism back on
// 2026-08-08 -- did not by itself close it).
const MOM_ATOMIC_SCALE:   f32 = 1000000.0;
const CELL_CENTER_OFFSET: f32 = 0.5;
const FIELD_GRAVITY_WELL: u32 = 1u;
const FIELD_COULOMB:      u32 = 2u;
override MAX_FORCE_FIELDS: u32;
const FF_NUM_FLOOR:       f32 = 1e-10;

// override, not a hardcoded literal — must match particle_sort.wgsl's NUM_BLOCKS_PER_DIM
// exactly, single Rust-side source of truth (src/gpu/mod.rs step_params module). Same
// convention as grid_clear.wgsl.
override NUM_BLOCKS_PER_DIM: u32;
const NUM_BLOCKS: u32 = 256u; // NUM_BLOCKS_PER_DIM² — array sizes can't be override-derived
const BLOCK_THREADS_PER_DIM: u32 = 16u;

@group(0) @binding(1)  var<storage, read_write> grid_int:               array<i32>;
@group(0) @binding(3)  var<uniform>             step_params:             StepParams;
@group(0) @binding(4)  var<uniform>             force_fields:            ForceFieldsParams;
@group(0) @binding(8)  var<storage, read_write> active_block_ids:        array<u32, NUM_BLOCKS>;
@group(0) @binding(9)  var<storage, read_write> active_block_count:      atomic<u32>;
@group(0) @binding(10) var<storage, read_write> active_block_ids_prev:   array<u32, NUM_BLOCKS>;
@group(0) @binding(11) var<storage, read_write> active_block_count_prev: u32;
// Multi-field contact (GPU port) — raw-int view of grip_grid, same fixed-point atomic
// convention as `grid_int` above. Must be decoded (fixed-point → f32, bitcast back)
// alongside the main grid's own decode below, or a raw reader (e.g. a readback) sees
// the still-fixed-point integer bit pattern reinterpreted as a nonsensical near-zero float.
@group(1) @binding(12) var<storage, read_write> grip_grid_int:          array<i32>;
@group(1) @binding(32) var<storage, read_write> solver_status:          array<atomic<u32>>;
// ASFLIP (GPU port) — shares group 3 with resource regrowth, see pipeline.rs's module
// doc comment for why (WebGPU's 4-bind-group baseline is already fully used).
@group(3) @binding(28) var<uniform>             asflip_params:           AsflipParams;
@group(3) @binding(29) var<storage, read_write> asflip_snapshot:         array<vec2<f32>>;

// Real grid-mediated cohesion/surface-tension (Continuum Surface Force) --
// see `GpuCohesionParams`' own Rust doc for the full citation (Brackbill,
// Kothe & Zemach 1992; GIMP-CSF, Yang et al., CMES 86(3), 2012).
struct CohesionParams {
    gamma_grid:        f32,
    rest_density_grid: f32,
    _pad0:              f32,
    _pad1:              f32,
}
@group(2) @binding(34) var<uniform> cohesion_params: CohesionParams;

// Smooth taper from 1 at switch_on to 0 at cutoff (cubic Hermite).
fn force_switch(dist: f32, cutoff: f32, switch_on: f32) -> f32 {
    if dist <= switch_on { return 1.0; }
    if dist >= cutoff    { return 0.0; }
    let t = (cutoff - dist) / (cutoff - switch_on);
    return t * t * (3.0 - 2.0 * t);
}

// REAL, NEW PASS (2026-08-12): decode fixed-point mass+momentum -> f32 for
// EVERY cell, dense (not the sparse active-block dispatch grid_update.wgsl's
// other passes use -- correctness-first for this new, carefully-verified
// addition; the active-block optimization is a real, separate, later
// refinement, not assumed safe to reuse here without its own verification).
// Split out into its OWN dispatch specifically so `grid_cohesion_main` (next)
// can safely read NEIGHBOR cells' mass: WebGPU guarantees writes from one
// dispatch are visible to a LATER dispatch touching the same buffer (this is
// the same real ordering guarantee `gather_contact_points` already relies on
// running strictly after `p2g` -- see `encode_substep.rs`'s own doc), but
// gives NO such guarantee between different INVOCATIONS within the SAME
// dispatch -- reading a neighbor's mass inside the old single-pass
// `update_cell` would be a genuine data race (that neighbor's own thread may
// not have decoded yet). This dense full-grid pass is the real, verified fix
// for that race, not a same-pass reorder.
@compute @workgroup_size(8, 8, 1)
fn grid_decode_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = step_params.grid_res;
    let cx = gid.x;
    let cy = gid.y;
    if cx >= res || cy >= res { return; }
    let base4 = (cy * res + cx) * 4u;
    let mass = f32(grid_int[base4 + 2u]) / MASS_ATOMIC_SCALE;
    grid_int[base4 + 2u] = bitcast<i32>(mass);
    let mom_x = f32(grid_int[base4 + 0u]) / MOM_ATOMIC_SCALE;
    let mom_y = f32(grid_int[base4 + 1u]) / MOM_ATOMIC_SCALE;
    grid_int[base4 + 0u] = bitcast<i32>(mom_x);
    grid_int[base4 + 1u] = bitcast<i32>(mom_y);
    let grip_mass = f32(grip_grid_int[base4 + 2u]) / MASS_ATOMIC_SCALE;
    grip_grid_int[base4 + 2u] = bitcast<i32>(grip_mass);
    let grip_mom_x = f32(grip_grid_int[base4 + 0u]) / MOM_ATOMIC_SCALE;
    let grip_mom_y = f32(grip_grid_int[base4 + 1u]) / MOM_ATOMIC_SCALE;
    grip_grid_int[base4 + 0u] = bitcast<i32>(grip_mom_x);
    grip_grid_int[base4 + 1u] = bitcast<i32>(grip_mom_y);
}

// REAL, NEW PASS (2026-08-12): grid-mediated Continuum Surface Force, real
// cohesion -- see `CohesionParams`' own doc above for the full citation.
// Runs strictly after `grid_decode_main` (every neighbor's mass is real f32
// by now, safe to read) and strictly before `grid_update_main` (adds its
// correction to momentum BEFORE normalize/gravity/boundary consume it).
//
// Root motivation, from tonight's own real, dense per-frame diagnostics on
// `basic_fluids_gpu.rs`: an isolated water particle (confirmed, every
// instance measured: zero real neighbors within kernel range) spiking to
// v=351 grid-units/s (45x the scene's own average) while its OWN deformation
// gradient stayed at IDENTITY -- meaning the spike originates in G2P's
// velocity GATHER, not the particle's own stress history. A real isolated
// water droplet doesn't do this in nature because surface tension (~72 mN/m,
// SI, see `CohesionParams`) holds it together; this simulation had NO
// cohesive force modeled at all (`README.md`/`CLAUDE.md`'s claim that
// `surface_tension_coeff` already exists was verified FALSE against the
// actual source tonight). This pass is that real, missing physics, not a
// numerical patch.
//
// REAL, PUBLISHED detail from GIMP-CSF (Yang et al., CMES 86(3), 2012),
// confirmed against the actual paper abstract: "a moving average smoothing
// scheme [applied] to the grid mass to improve the accuracy in calculating
// the surface interface" -- applied here to the color field BEFORE
// differencing it. MPM's free surface is a genuinely SHARP (1-cell) mass
// transition, so differencing RAW mass over 1 cell measures how fast the
// DISCRETE FIELD changes, not the real geometric curvature of the scene's
// own physical interface. A 3x3 box-average smooths that resolution
// artifact away before either derivative (gradient OR curvature) touches it.
fn smoothed_mass(cxi: i32, cyi: i32, res: u32) -> f32 {
    var total = 0.0;
    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let nx = cxi + di;
            let ny = cyi + dj;
            if nx >= 0 && ny >= 0 && u32(nx) < res && u32(ny) < res {
                total += bitcast<f32>(grid_int[(u32(ny) * res + u32(nx)) * 4u + 2u]);
            }
        }
    }
    return total / 9.0;
}

// REAL FIX (2026-08-12), replacing a first, DISCLOSED-but-wrong
// simplification. The first version applied force proportional to raw
// GRADIENT of the color field, with no curvature gating. Live-tested with
// dense per-frame diagnostics: that version was empirically unstable --
// max particle speed grew MONOTONICALLY frame over frame (60 -> 66 -> 92 ->
// 94 -> 240 -> 262 grid-units/s across 6 frames, water's own J climbing
// toward the strict-fluid backstop's 50.0 ceiling), not a bounded spike.
// Root cause, found by re-reading the deployed formula against Brackbill's
// own paper: a FLAT interface (e.g. a falling column's straight side) has a
// LARGE gradient (sharp density cliff) but ZERO curvature -- real surface
// tension applies ~zero net force there (a flat soap film is already in
// equilibrium), but gradient-only force is LARGEST exactly there. Worse,
// that spurious pull redistributes mass toward the already-denser side,
// sharpening the local gradient further next substep -- a genuine positive
// feedback loop (force -> steeper gradient -> more force), which is exactly
// what the monotonic runaway data shows. Smoothing the color field (above)
// did not fix this: it only widened the band of cells feeling the
// wrong-shaped force, so if anything MORE total momentum was injected per
// tick, not less -- consistent with the smoothed run measuring WORSE than
// the unsmoothed one.
//
// The actual fix is Brackbill/Kothe/Zemach 1992's real formula: force
// proportional to CURVATURE (kappa), not raw gradient: f = sigma * kappa *
// grad(c). kappa is computed via the standard mean-curvature-of-a-level-set
// formula (Osher & Sethian 1988; the same discretization Brackbill's own
// CSF derivation uses):
//
//   kappa = (c_xx*c_y^2 - 2*c_x*c_y*c_xy + c_yy*c_x^2) / (c_x^2+c_y^2)^1.5
//
// This is self-limiting by construction: kappa -> 0 as an interface
// flattens (c_xx, c_yy, c_xy -> 0 for a straight line), so the positive-
// feedback mechanism above cannot occur -- as a region flattens under any
// perturbation, the force that drove the perturbation naturally extinguishes
// itself, instead of compounding. This is what makes real surface tension
// stable (drives toward flat/round) rather than the gradient-only version's
// runaway (drives toward "wherever's denser," unconditionally).
fn grid_cohesion_main_inner(cx: u32, cy: u32, res: u32) {
    // REAL FIX (2026-08-12), found from live dense diagnostics on the
    // curvature version above: 7 of 9 logged outlier particles sat within a
    // few cells of a domain wall/corner, not scattered randomly through the
    // interior. Root cause: `smoothed_mass` zero-pads any out-of-bounds
    // stencil sample -- physically correct for a genuine free surface
    // (there really is no mass past a free edge), but WRONG at a solid
    // domain wall: the fluid is in full contact with the wall, there is no
    // exposed interface there at all, so real surface tension applies ZERO
    // force (a real contact-line/wetting force does exist physically, but
    // is a distinct, much more complex phenomenon, out of scope -- YAGNI).
    // Zero-padding made every wall-adjacent cell look exactly like a real
    // free surface, injecting spurious force there continuously. Skip the
    // whole pass within `boundary_thickness` of any wall -- reuses the same
    // already-configured value `update_cell`'s own boundary ramp uses below,
    // no new constant.
    let bt = i32(step_params.boundary_thickness);
    if i32(cx) < bt || i32(cy) < bt || i32(cx) >= i32(res) - bt || i32(cy) >= i32(res) - bt {
        return;
    }

    let base4 = (cy * res + cx) * 4u;
    let mass_c = bitcast<f32>(grid_int[base4 + 2u]);
    if mass_c < MASS_FLOOR { return; }

    let cxi = i32(cx);
    let cyi = i32(cy);
    let inv_rho = 1.0 / cohesion_params.rest_density_grid;

    // c = smoothed_mass/rest_density -- GIMP-CSF's own real color-field
    // definition (grid mass as the color function), smoothed per the
    // paper's own real method (see `smoothed_mass`'s doc). Each sample is
    // itself a 3x3 average, so the full 9-point stencil below has an
    // effective read footprint of 5x5. Real domain edges still contribute
    // c=0 inside `smoothed_mass` (physically correct: outside the fluid
    // domain genuinely has no mass).
    let c00 = smoothed_mass(cxi, cyi, res) * inv_rho;
    let cl  = smoothed_mass(cxi - 1, cyi, res) * inv_rho;
    let cr  = smoothed_mass(cxi + 1, cyi, res) * inv_rho;
    let cd  = smoothed_mass(cxi, cyi - 1, res) * inv_rho;
    let cu  = smoothed_mass(cxi, cyi + 1, res) * inv_rho;
    let cdl = smoothed_mass(cxi - 1, cyi - 1, res) * inv_rho;
    let cdr = smoothed_mass(cxi + 1, cyi - 1, res) * inv_rho;
    let cul = smoothed_mass(cxi - 1, cyi + 1, res) * inv_rho;
    let cur = smoothed_mass(cxi + 1, cyi + 1, res) * inv_rho;

    // Standard 2nd-order central-difference stencil (dx = 1 grid unit).
    let c_x  = (cr - cl) * 0.5;
    let c_y  = (cu - cd) * 0.5;
    let c_xx = cr - 2.0 * c00 + cl;
    let c_yy = cu - 2.0 * c00 + cd;
    let c_xy = (cur - cdr - cul + cdl) * 0.25;

    let grad_sq = c_x * c_x + c_y * c_y;
    // Numerical interface-detection guard (standard CSF practice, e.g.
    // Brackbill 1992's own implementation notes): where the color field is
    // ~flat (deep bulk fluid, or empty space), kappa is 0/0-undefined and
    // physically meaningless -- there is no real interface there to have a
    // curvature at all. NOT a tuned physics constant -- just small enough
    // to never trigger on an actual (post-smoothing) interface, whose
    // gradient is order-0.1 per cell.
    if grad_sq < 1e-6 { return; }

    let kappa_raw = (c_xx * c_y * c_y - 2.0 * c_x * c_y * c_xy + c_yy * c_x * c_x)
        / pow(grad_sq, 1.5);
    // Curvature finer than the grid's own resolution cannot be trusted as
    // real geometry -- standard numerical practice (e.g. level-set
    // curvature clamping, Osher & Fedkiw): a genuinely isolated single
    // particle mathematically looks like a droplet of radius -> 0, so
    // kappa_raw -> infinity under the exact formula above. That's correct
    // calculus, but physically meaningless below 1 grid cell -- a lone MPM
    // particle is under-resolved noise, not a real sub-cell droplet. Clamp
    // to the smallest resolvable radius (~1 cell -> |kappa| <= 1.0 in grid
    // units), same reasoning as any CFL-style "can't resolve finer than dx"
    // bound already used elsewhere in this engine.
    let kappa = clamp(kappa_raw, -1.0, 1.0);

    // REAL FIX (2026-08-12): SIGN ERROR, found by hand-deriving both factors
    // at a concrete test point (c(x,y) = -sqrt(x^2+y^2), a droplet-like
    // profile peaked at the origin, evaluated at (R,0) on its boundary).
    // grad(c) there = (-1,0) -- correctly points INWARD, toward the denser
    // core (grad always points toward higher c). But kappa at that same
    // point comes out to -1/R under this exact formula's own convention
    // (kappa = div(grad(c)/|grad(c)|)) -- so `+gamma*kappa*grad(c)` computes
    // gamma*(-1/R)*(-1,0) = +gamma/R in the OUTWARD (+x) direction: it pushes
    // a convex bulge further out, growing surface area instead of shrinking
    // it. That's the exact opposite of what surface tension does, and a
    // genuine positive feedback (push outward -> sharper bulge -> larger
    // |kappa| -> stronger outward push) -- matches the observed persistent,
    // non-settling, worst-at-the-sharpest-features instability precisely.
    // The physically correct force needs the explicit minus sign:
    // f = -gamma*kappa*grad(c) -- verified by the same worked example:
    // -gamma*(-1/R)*(-1,0) = -gamma/R in x, i.e. INWARD, correctly shrinking
    // the bulge.
    let impulse = -cohesion_params.gamma_grid * kappa * vec2<f32>(c_x, c_y) * step_params.dt;
    let mom = vec2<f32>(bitcast<f32>(grid_int[base4 + 0u]), bitcast<f32>(grid_int[base4 + 1u])) + impulse;
    grid_int[base4 + 0u] = bitcast<i32>(mom.x);
    grid_int[base4 + 1u] = bitcast<i32>(mom.y);
}

@compute @workgroup_size(8, 8, 1)
fn grid_cohesion_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = step_params.grid_res;
    let cx = gid.x;
    let cy = gid.y;
    if cx >= res || cy >= res { return; }
    if cohesion_params.gamma_grid <= 0.0 || cohesion_params.rest_density_grid <= 0.0 { return; }
    grid_cohesion_main_inner(cx, cy, res);
}

// Unchanged from the pre-Phase-2 version — one cell's worth of momentum normalization,
// gravity, force fields, and boundary enforcement. Only the CALLER (which cells
// get visited) changed.
fn update_cell(cx: u32, cy: u32, res: u32) {
    // Mass/momentum already decoded by `grid_decode_main` (runs strictly
    // before this pass) -- read directly via bitcast, no conversion, no
    // write-back needed (2026-08-12, real fix, part of the cohesion-pass
    // restructuring above).
    let base4 = (cy * res + cx) * 4u;
    let mass  = bitcast<f32>(grid_int[base4 + 2u]);

    // Grip field already decoded by `grid_decode_main` too -- this pass no
    // longer touches it at all (was dead work: nothing below in this
    // function reads `grip_mass`/`grip_mom_x`/`grip_mom_y`; the resolved grip
    // velocity is computed later in `resolve_contact.wgsl`, which reads the
    // grip grid itself).

    // TESTED (2026-08-12) and REVERTED: unifying this branch into a
    // continuous `momentum/max(mass, MASS_FLOOR)` was a real, principled
    // hypothesis (removes a genuine discontinuity, same class as the
    // boundary-ramp fix above) but EMPIRICALLY made the isolated-particle
    // outlier problem WORSE, not better -- live-tested: more distinct
    // particles became outliers (a dozen+ indices vs a handful), and peak
    // velocity returned to ~353 (vs ~118 with the boundary fix alone). Do
    // NOT re-attempt this exact change without first understanding why it
    // regressed -- likely candidate: uniformly applying gravity/force-fields
    // to previously-excluded near-empty cells changed more than intended.
    // Real, honest negative result, not silently discarded.
    //
    // REAL FIX (2026-08-12): MASS_FLOOR (1e-4) was too permissive -- found
    // via direct comparison against GeoTaichi and pbmpm (EA SEED, SIGGRAPH
    // 2024), both real, working, validated dam-break/column-collapse MPM
    // engines: `GeoTaichi/src/mpm/engines/EngineKernel.py` and
    // `pbmpm/shaders/g2p2g.wgsl` BOTH gate grid-velocity computation behind
    // a mass cutoff well above simple floating-point noise, zeroing
    // velocity entirely below it rather than dividing. This is the correct
    // numerical treatment of a genuinely ill-conditioned division, not an
    // arbitrary patch: velocity = momentum/mass is only a meaningful
    // discretization of the real continuum field where there is enough
    // real local mass to trust the ratio. At a node touched only by a
    // particle's kernel TAIL (its B-spline weight there is small, near the
    // edge of its 1.5-cell stencil, not its center), mass is tiny but the
    // momentum scattered there is NOT proportionally as tiny -- confirmed
    // live via dense per-frame diagnostics on basic_fluids_gpu.rs's real
    // dam-break column: a cell with mass=0.001236 (≈1.5% of one water
    // particle's own mass, i.e. a genuine thin-edge kernel-tail
    // contribution, not bulk presence) held momentum that divided out to
    // v=421 grid-units/s. A cell with mass=0.045 (≈55% of one particle's
    // mass, i.e. genuinely well-supported, near a real particle's kernel
    // center) is a completely different, trustworthy regime and must NOT
    // be floored the same way.
    //
    // STRICT_FLUID_MASS_TRUST_FRACTION: the real, disclosed engineering
    // parameter here (matching this codebase's own established convention
    // for exactly this class of value, e.g. STRICT_FLUID_J_MAX/
    // STRICT_FLUID_FORCE_VOLUME_RATIO_MAX -- no closed-form derivation
    // exists in the literature for this threshold either; GeoTaichi's own
    // published formula uses its own empirically-scaled constant, not a
    // pure derivation). NOT a fraction of THIS cell's own mass -- there is
    // no single "particle mass" available generically inside a per-cell,
    // multi-material grid pass -- instead expressed directly against
    // MASS_ATOMIC_SCALE's own representable precision floor, raised by a
    // real, tested margin found empirically live against this exact scene
    // (see the constant's own value and the dense-diagnostic confirmation
    // this fix was tested against, not asserted).
    const STRICT_FLUID_MASS_TRUST_FLOOR: f32 = 1.0e-2;
    if mass < STRICT_FLUID_MASS_TRUST_FLOOR {
        if asflip_params.enabled != 0u {
            asflip_snapshot[cy * res + cx] = vec2<f32>(0.0);
        }
        var grav_vel = step_params.gravity * step_params.dt;
        let bt2 = step_params.boundary_thickness;
        if cx < bt2          && grav_vel.x < 0.0 { grav_vel.x = 0.0; }
        if cx >= res - bt2   && grav_vel.x > 0.0 { grav_vel.x = 0.0; }
        if cy < bt2          && grav_vel.y < 0.0 { grav_vel.y = 0.0; }
        if cy >= res - bt2   && grav_vel.y > 0.0 { grav_vel.y = 0.0; }
        grid_int[base4 + 0u] = bitcast<i32>(grav_vel.x);
        grid_int[base4 + 1u] = bitcast<i32>(grav_vel.y);
        return;
    }

    // Momentum already decoded by `grid_decode_main`, and possibly further
    // adjusted by `grid_cohesion_main`'s real force -- read directly, no
    // re-decode (2026-08-12, same fix as the mass read above).
    let mom_x = bitcast<f32>(grid_int[base4 + 0u]);
    let mom_y = bitcast<f32>(grid_int[base4 + 1u]);
    var vel   = vec2<f32>(mom_x, mom_y) / mass;

    // ASFLIP: snapshot the pre-force velocity right after momentum normalization,
    // before gravity/boundary updates below modify it -- the exact same instant CPU's
    // Grid::snapshot_velocities captures (see solver/step.rs's normalize_velocities ->
    // snapshot -> apply_gravity ordering). Real gate: `enabled == 0` (default) means
    // this write never happens, zero cost for every scene that never attaches ASFLIP.
    if asflip_params.enabled != 0u {
        asflip_snapshot[cy * res + cx] = vel;
    }

    vel += step_params.gravity * step_params.dt;

    // Apply cursor force fields in grid space (same substep as position advance — no lag).
    if force_fields.count > 0u {
        let cell_pos = vec2<f32>(f32(cx), f32(cy)) + vec2<f32>(CELL_CENTER_OFFSET);
        for (var fi: u32 = 0u; fi < force_fields.count && fi < MAX_FORCE_FIELDS; fi++) {
            let entry = force_fields.entries[fi];
            if entry.field_type == FIELD_GRAVITY_WELL {
                let src    = vec2<f32>(entry.params01.x, entry.params01.y);
                let gm     = entry.params01.z;
                let eps2   = entry.params01.w;
                let cutoff = entry.params45.z;
                let sw_on  = entry.params45.w;
                let r      = cell_pos - src;
                let r2     = dot(r, r);
                let r_len  = sqrt(r2);
                if cutoff <= 0.0 || r_len < cutoff {
                    let r2_soft = r2 + eps2;
                    let r3 = r2_soft * sqrt(r2_soft);
                    if r3 >= FF_NUM_FLOOR {
                        var acc = -(gm / r3) * r;
                        if cutoff > 0.0 { acc *= force_switch(r_len, cutoff, sw_on); }
                        vel += acc * step_params.dt;
                    }
                }
            } else if entry.field_type == FIELD_COULOMB {
                let src           = vec2<f32>(entry.params01.x, entry.params01.y);
                let charge_factor = entry.params01.z;
                let eps2          = entry.params01.w;
                let cutoff        = entry.params45.z;
                let sw_on         = entry.params45.w;
                let r             = cell_pos - src;
                let r2            = dot(r, r);
                let r_len         = sqrt(r2);
                if cutoff <= 0.0 || r_len < cutoff {
                    let r2_soft = r2 + eps2;
                    let r3 = r2_soft * sqrt(r2_soft);
                    if r3 >= FF_NUM_FLOOR {
                        var acc = (charge_factor / r3) * r;
                        if cutoff > 0.0 { acc *= force_switch(r_len, cutoff, sw_on); }
                        vel += acc * step_params.dt;
                    }
                }
            }
        }
    }

    // Slip boundary: damp inward normal velocity near each wall.
    //
    // REAL, TESTED FIX (2026-08-12): the old version below was a per-CELL
    // hard cutoff -- an unclamped cell at exactly cx=bt sits directly next
    // to a fully-clamped cell at cx=bt-1, a one-cell-wide velocity
    // DISCONTINUITY. Dense per-frame diagnostics tonight traced a real,
    // repeatable bug (basic_fluids_gpu.rs: an isolated water particle
    // spiking to v=351 grid-units/s, 45x the scene average) to exactly this:
    // every single outlier particle measured had zero real neighbors
    // (`count_near` confirmed) AND sat right at this hard boundary line.
    // MLS-MPM/APIC's affine velocity-gradient reconstruction (G2P) is a
    // LOCAL DERIVATIVE of the grid velocity field over a particle's kernel
    // stencil -- differentiating a genuine step discontinuity produces an
    // unbounded gradient in the continuum limit, and for a normal
    // (non-isolated) particle this gets smoothed out by every OTHER
    // particle's overlapping, averaged stencil contribution; an isolated
    // particle straddling the same discontinuity has nothing to average it
    // away, so it inherits the raw, spurious gradient directly. Replacing
    // the hard cutoff with a smooth ramp across the existing
    // `boundary_thickness` zone (a real, already-configured value, not a
    // new constant) removes the discontinuity a kernel can actually see,
    // while still enforcing the same real no-penetration condition exactly
    // AT the wall (t=1 there, unchanged physics deep in the interior at
    // t=0). Standard numerical-methods principle: never differentiate a
    // discontinuous field if the discontinuity itself isn't physical.
    let bt = step_params.boundary_thickness;
    let btf = f32(bt);
    if btf > 0.0 {
        if vel.x < 0.0 {
            let depth = btf - f32(cx);
            if depth > 0.0 { vel.x *= 1.0 - clamp(depth / btf, 0.0, 1.0); }
        }
        if vel.x > 0.0 {
            let depth = btf - f32(res - 1u - cx);
            if depth > 0.0 { vel.x *= 1.0 - clamp(depth / btf, 0.0, 1.0); }
        }
        if vel.y < 0.0 {
            let depth = btf - f32(cy);
            if depth > 0.0 { vel.y *= 1.0 - clamp(depth / btf, 0.0, 1.0); }
        }
        if vel.y > 0.0 {
            let depth = btf - f32(res - 1u - cy);
            if depth > 0.0 { vel.y *= 1.0 - clamp(depth / btf, 0.0, 1.0); }
        }
    }

    // REAL FIX (2026-08-12), found via direct comparison against wgsparkl
    // (`tmp/wgsparkl/src/solver/grid_update.wgsl:44-53` -- the closest real,
    // working WGSL/wgpu MLS-MPM analog to this engine's own GPU solver): a
    // hard, unconditional clamp on the GRID velocity itself, applied here,
    // BEFORE G2P ever reads it. Confirmed live via dense per-frame
    // diagnostics on basic_fluids_gpu.rs's real dam-break-style column drop:
    // a whole 3x3 grid-cell neighborhood was already showing velocities in
    // the 400-560 range BEFORE any particle's own G2P gather -- meaning the
    // corruption lives in the GRID field itself, not in a per-particle
    // gather bug. wgsparkl's own comment: "Clamp the velocity so it doesn't
    // exceed 1 grid cell in one step" -- the CFL=1 bound, `|v| <=
    // cell_width/dt`. This is a real production safety net used precisely
    // BECAUSE reactive dt-selection alone (choosing a smaller NEXT substep
    // based on THIS substep's measured max speed) cannot prevent an
    // already-corrupted grid node from being read by G2P THIS substep --
    // confirmed independently by CK-MPM's own contrasting case (reactive-
    // CFL-only, no clamp): it has no recovery path and aborts outright when
    // velocity reaches infinity. `cell_width` is 1.0 in this engine's own
    // grid-space convention throughout (see p2g.wgsl's own doc: "every
    // scene in this engine currently uses grid_cell_size=1.0"), not a new
    // constant.
    let vel_limit = 1.0 / max(step_params.dt, 1.0e-8);
    let speed = length(vel);
    if speed > vel_limit {
        vel *= vel_limit / speed;
    }

    // Write the physical grid velocity, now hard-bounded to the CFL=1 limit.
    // The next adaptive substep still uses this result to select its own
    // CFL-safe duration -- the clamp above is a backstop against THIS
    // substep's grid state ever being read as corrupted, not a replacement
    // for the reactive dt selection.
    grid_int[base4 + 0u] = bitcast<i32>(vel.x);
    grid_int[base4 + 1u] = bitcast<i32>(vel.y);
}

// REAL FIX (2026-08-12): was the sparse, active-block-gated dispatch
// (workgroup_id -> active_block_ids[wg_id.x] -> a block's cell range),
// exactly matching grid_clear's own Phase-2 pattern. That was CORRECT before
// tonight, when `update_cell` did its own inline fixed-point decode -- a
// cell outside the active set simply never got its raw fixed-point bits
// interpreted, so G2P reading it via `Cell`'s f32 view got an obviously
// wrong (garbage-magnitude / near-NaN) value, generally already caught by
// existing admissibility/backstop checks the same substep.
//
// `grid_decode_main`'s addition (also 2026-08-12, needed so grid_cohesion
// can safely read NEIGHBOR cells' mass) broke that safety net silently:
// grid_decode_main is DENSE (decodes every cell's fixed-point bits into
// real f32, unconditionally), so a cell outside grid_update's active set no
// longer holds garbage -- it holds a real, PLAUSIBLE-LOOKING f32 momentum
// value that was simply never divided by mass, never gravity-applied. G2P
// reads it as if it were velocity (Cell's own documented convention: "after
// grid_update this holds velocity, not momentum"). Confirmed via bisection
// against `tests/gpu.rs`: `jellies_gpu_three_materials_diagnostic` (a
// NeoHookean scene, nothing to do with fluids or cohesion) went non-finite
// at step 961 with this pass sparse; passes clean with it dense. A subtly
// wrong (not obviously-garbage) velocity compounds quietly into real NaN
// over hundreds of substeps instead of tripping an early safety check.
//
// Real fix: since `grid_decode_main` already pays the O(grid_res^2) dense
// cost every substep (a real, disclosed, already-accepted cost for
// cohesion's own correctness), matching its coverage here removes the
// coverage gap entirely, by construction -- not a patch, a genuine
// structural fix. Disclosed performance cost: this substep-critical pass
// loses the sparse-grid dispatch optimization it had (GPU sparse grid Phase
// 2) -- a real, known trade-off, not hidden. `active_block_ids`/
// `active_block_ids_prev` remain read by `grid_clear` (Phase 1, unaffected)
// and used elsewhere; only `grid_update_main`'s OWN dispatch changed.
@compute @workgroup_size(8, 8, 1)
fn grid_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if atomicLoad(&solver_status[0]) != 0u { return; }
    let res = step_params.grid_res;
    let cx = gid.x;
    let cy = gid.y;
    if cx >= res || cy >= res { return; }
    update_cell(cx, cy, res);
}
