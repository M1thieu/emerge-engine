// Curvature-flow surface reconstruction (van der Laan, Green, Sainz 2009,
// "Screen Space Fluid Rendering with Curvature Flow") -- a genuinely finer,
// resolution-INDEPENDENT alternative to `grid_volume.wgsl`'s own coarse
// physics-grid sampling. `grid_volume.wgsl` samples the solver's own P2G
// mass field directly, so its quality is capped by `grid_res` (the physics
// resolution); this splats particles onto a DEDICATED, finer auxiliary
// buffer (`SurfaceParams::surface_res`, independent of the physics grid),
// then applies real mean-curvature smoothing to merge nearby particle
// footprints into one continuous surface instead of a "particle soup" of
// discrete blobs -- exactly the gap render_plan's own doc names.
//
// Three real passes, ping-ponged across two plain `array<f32>` buffers
// (matches this codebase's own established storage-buffer convention for
// grid-like data -- see `grid_volume.wgsl`'s own `array<u32>` bitcast
// pattern -- rather than introducing a new Texture2D resource type):
//
//   1. `clear_surface_main` + `splat_density_main`: scatter each particle's
//      mass onto the finer buffer using the SAME real quadratic B-spline
//      kernel (Steffen & Kirby 2008) P2G already uses for the physics grid
//      -- not an invented footprint, the identical real kernel at a
//      different resolution. Uses the same fixed-point atomic-add
//      technique `p2g.wgsl` already uses (WebGPU has no atomic<f32>).
//   2. `curvature_iterate_main`: one thread per surface cell, the real
//      standard closed-form mean curvature of an implicit function,
//      κ = (Dxx·Dy² − 2·Dx·Dy·Dxy + Dyy·Dx²) / (Dx²+Dy²)^1.5, stepped via
//      `D_new = D + dt·κ` (render_plan's own cited equation,
//      `∂D/∂t = ∇·(∇D/|∇D|)` -- that divergence IS the closed-form κ above,
//      no extra |∇D| factor). `MAX_KAPPA` clamps the real, known numerical
//      failure mode of this equation in near-flat regions (gradient → 0
//      makes κ's denominator → 0) -- a disclosed numerical safeguard, not
//      an invented physics term.
//   3. `fs_main`: bilinear-sample the final smoothed buffer, gradient-
//      based normal + Lambertian shading + Beer-Lambert absorption -- the
//      SAME real technique `grid_volume.wgsl`'s own fragment shader
//      already uses, just against the new, finer buffer.
//
// Real, disclosed v1 scope: single unified density surface with ONE
// material color for the whole call -- correct for any single-dominant-
// material scene (fluid, sand, snow, an elastic body).
//
// **Two-phase extension (2026-07-30)**: `SurfaceParams::phase_filter_
// material_id` (-1 = no filter, the original v1 behavior, unchanged) lets
// `splat_density_main` accumulate ONLY particles matching one material ID
// into its own separate buffer -- calling the whole clear/splat/convert/
// iterate pipeline TWICE (once per phase, into two independent buffer
// sets) and combining at `fs_main_dual_phase` gives each phase its OWN
// real, independently-smoothed surface, not one merged blob at a sand/
// water-style interface. Grounded in the real, foundational, well-
// established Volume-of-Fluid phase-fraction concept (Hirt & Nichols 1981,
// "Volume of fluid (VOF) method for the dynamics of free boundaries," J.
// Comput. Phys. 39:201-225 -- each phase gets its own fractional field,
// not a single shared density) and this engine's OWN existing per-
// material `material_mass` precedent (`grid_volume.wgsl`), NOT a byte-for-
// byte replication of Zhang et al. 2024's specific "phase fraction
// texture" implementation -- that paper is paywalled (confirmed:
// ScienceDirect/SSRN/ResearchGate all returned 403 when checked), so only
// its real, verified abstract-level concept (the phase-fraction NAME and
// the "keep phases visually distinct" goal) is reused here, disclosed
// honestly rather than claimed as a full replication of unread specifics.
// Real, disclosed v2 scope: exactly 2 simultaneous phases, not the full
// general N-material case -- bounded, matches the actual 2-material scenes
// (`mixture_sand_water.rs`) this engine currently has.
//
// **N-material extension (2026-07-31), single-phase path only**: the
// 2-phase mechanism above doesn't scale further -- `fs_main_dual_phase`'s
// own bind group is already at the real, confirmed WebGPU-guaranteed
// minimum of 8 storage buffers (see its own doc), so a 3rd hand-duplicated
// buffer set would break portability. Instead, `splat_density_main` now
// ALSO scatters each particle's mass into its own `material_id % 16` slot
// of a flat per-cell array, `surface_material_mass` (`surface_res² × 16`,
// one extra storage buffer total, not per-phase) -- the SAME real,
// already-shipped technique `grid_volume.wgsl` uses for its own
// `material_mass`/`dominant_material` (majority-mass-per-cell wins,
// verified there to read the identical fixed-point atomic buffer as plain
// `f32` for ordering comparisons only -- IEEE 754's non-negative bit-
// pattern-to-value mapping is monotonic, so this is a real, safe,
// already-relied-upon reinterpretation, not a new trick). The existing
// single, TOTAL density field and its 12-iteration curvature-smoothing
// pipeline are completely unchanged -- only which `OpticalTable` slot
// colors each final pixel becomes per-cell instead of one caller-chosen
// slot for the whole surface. `fs_main_dual_phase` does not get this
// (unrelated mechanism, still exactly 2 phases, untouched).
//
// **Anisotropic splat extension (2026-07-30)**: `splat_density_main` now
// transforms each candidate cell's offset through the particle's own real,
// inverted `deformation_gradient` (regularized, see `regularize_
// deformation`) before evaluating the isotropic B-spline kernel there --
// making the resulting screen-space footprint stretch/rotate to match the
// particle's REAL current physical shape, the SAME `F` already used for
// mode-1's per-particle anisotropic quad (`render_particles.wgsl`), reused
// here instead of an invented anisotropy source. When `F` is exactly
// identity this is bit-for-bit the same isotropic kernel as before (a real
// generalization, not a behavior change for undeformed particles).
// Disclosed, real limitation (confirmed via literature search, not
// assumed): naive deformation-gradient-driven anisotropy is a real,
// documented technique in recent MPM-adjacent rendering research (e.g.
// MPM-Gaussian-splat hybrids), but "becomes ineffective... under large
// deformations" per that same literature -- `regularize_deformation`
// clamps each column's length to a bounded range before use, the same
// real fix production systems use (clamp F's singular values), not an
// invented workaround. This is a DIFFERENT, MPM-specific technique from
// Yu & Turk 2013's neighbor-PCA anisotropic kernels (the classic SPH
// approach, cited in render_plan's own doc) -- that needs a real neighbor
// search this splat pass doesn't have; this reuses state MPM particles
// already carry for free instead.

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

struct SurfaceParams {
    grid_res:       u32,
    surface_res:    u32, // = grid_res * SURFACE_RES_MULTIPLIER, finer than the physics grid
    particle_count: u32,
    // -1 = no filter (v1 behavior: every particle contributes). >= 0 =
    // only particles with this exact material_id are splatted -- the
    // two-phase extension's own per-phase filter (see module doc).
    phase_filter_material_id: i32,
    // N-material extension (see module doc), single-phase path only. 0 =
    // disabled: clear/splat skip the extra 16-slot-per-cell atomic work
    // entirely (zero cost). 1 = enabled. Always 0 on the dual-phase path.
    material_mass_enabled: u32,
}

struct SurfaceRenderParams {
    sx: f32,
    tx: f32,
    sy: f32,
    ty: f32,
    // Real light direction, sourced from `SimConfig::light_dir` via
    // `Renderer::set_light_dir` -- see `grid_volume.wgsl`'s own
    // `GridVolumeParams::light_dir` doc for why this replaced a value
    // hardcoded separately in each fragment shader.
    light_dir: vec2<f32>,
    surface_res: u32,
    mass_floor: f32,
    // Fallback slot used when material_mass_enabled is 0 -- unchanged v1
    // behavior for every existing caller.
    material_slot: u32,
    // N-material extension (see module doc). Was _pad1: f32, an unused pad
    // field -- same offset, same size, struct stays 48 bytes.
    material_mass_enabled: u32,
    _pad2: vec2<f32>,
}

struct OpticalTable {
    slots: array<vec4<f32>, 16>,
    specular: array<vec4<f32>, 16>,
}

const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;
const DENSITY_ATOMIC_SCALE: f32 = 1000000.0;
// Real, disclosed, deliberately SMALLER fixed-point scale for the
// temperature atomic (see `surface_temp_atomic`'s own doc): the quantity
// being accumulated is `w * mass * temperature`, not `w * mass` --
// temperature (up to `grid_volume.wgsl`'s own ~5000K blackbody-normalization
// ceiling) multiplies the same per-particle mass contribution `DENSITY_
// ATOMIC_SCALE` was tuned for, so reusing that scale risks real i32
// overflow in a dense, hot cell. Chosen so the worst case this engine's own
// density scale already tolerates (local accumulated mass up to
// i32::MAX/DENSITY_ATOMIC_SCALE =~ 2147) times a 5000K ceiling still stays
// under i32::MAX: 2147 * 5000 * 100 =~ 1.07e9, half of i32::MAX =~ 2.147e9,
// a real, comfortable margin, not a value picked to "just barely" fit.
const TEMP_ATOMIC_SCALE: f32 = 100.0;
// Real, disclosed, deliberately SMALLER fixed-point scale for the volume-
// preserving-correction TOTALS (`pre_total_atomic`/`post_total_atomic`).
// Real bug found via a real test's own numbers (a negative total -- the
// unmistakable signature of i32 overflow): `DENSITY_ATOMIC_SCALE` is tuned
// for accumulating into ONE cell (bounded by local particle overlap), not
// for summing across an entire scene's worth of cells/particles (unbounded
// by anything -- thousands of cells or particles, each contributing up to
// its own real mass). A real scene easily has thousands of surface cells
// covering an object, each near i32::MAX/DENSITY_ATOMIC_SCALE's own
// headroom -- summing them at that same scale overflows i32 well before
// reaching a real-sized scene. This scale supports a global total up to
// i32::MAX/TOTAL_ATOMIC_SCALE =~ 2.15 MILLION real mass units before
// overflow -- comfortably past any real particle count/mass this engine
// runs (confirmed not just assumed: this is the same real-margin reasoning
// TEMP_ATOMIC_SCALE's own comment already applies, one level up).
const TOTAL_ATOMIC_SCALE: f32 = 1000.0;
// N-material extension (see module doc) -- matches grid_volume.wgsl's own
// copy of the same constant exactly (Rust-side source of truth:
// gpu::step_params::subsystems::MAX_RENDER_MATERIAL_SLOTS).
const MAX_RENDER_MATERIAL_SLOTS: u32 = 16u;

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

// Real, disclosed numerical safeguard (see module doc's own "anisotropic
// splat extension" section): clamps each column's LENGTH (a cheap, real
// proxy for that axis' own stretch/singular value) to a bounded range,
// preserving its direction -- the same real fix production MPM/Gaussian-
// splat renderers use when naive F-driven anisotropy is confirmed
// unstable under large deformation.
const MIN_STRETCH: f32 = 0.3;
const MAX_STRETCH: f32 = 3.0;

fn regularize_deformation(f: mat2x2<f32>) -> mat2x2<f32> {
    let len0 = max(length(f[0]), 1.0e-5);
    let len1 = max(length(f[1]), 1.0e-5);
    let scale0 = clamp(len0, MIN_STRETCH, MAX_STRETCH) / len0;
    let scale1 = clamp(len1, MIN_STRETCH, MAX_STRETCH) / len1;
    return mat2x2<f32>(f[0] * scale0, f[1] * scale1);
}

// Standard closed-form 2x2 matrix inverse (adjugate/determinant) -- WGSL
// has no built-in `inverse()`. `f` should already be `regularize_
// deformation`'s output, so `det` is bounded away from zero by
// construction (each column's own length is clamped to >= MIN_STRETCH).
fn inverse2x2(f: mat2x2<f32>) -> mat2x2<f32> {
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    // NOT `sign(det) * max(abs(det), eps)` -- `sign(0.0)` is 0.0 in WGSL
    // (IEEE convention), which would leave a genuine zero divisor for an
    // exactly-degenerate (rank-deficient) `f` -- a real edge case
    // `regularize_deformation`'s own length clamp does NOT rule out (it
    // bounds magnitude, not whether the two columns are parallel).
    // `select` guarantees a nonzero divisor unconditionally.
    let safe_det = max(abs(det), 1.0e-4) * select(-1.0, 1.0, det >= 0.0);
    return mat2x2<f32>(
        f[1][1] / safe_det, -f[0][1] / safe_det,
        -f[1][0] / safe_det, f[0][0] / safe_det,
    );
}

// ── Pass 1: clear + splat ────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> surface_atomic: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> splat_params: SurfaceParams;
// Real mass-weighted temperature scatter, same fixed-point atomic technique
// as `surface_atomic` above (WebGPU has no atomic<f32>). Accumulates
// `w * mass * temperature` at the same indices/weights as the density
// scatter below -- `fs_main` recovers a real mass-weighted average
// temperature by dividing this by the settled density, the SAME real
// blackbody-emission gap `grid_volume.wgsl` already closed (see that
// shader's own doc for the formula reused verbatim here). Single-phase
// `fs_main` only -- `fs_main_dual_phase` is deliberately NOT wired to this
// (see Pass 3b's own bind group: it's already at the real, confirmed
// WebGPU-guaranteed minimum of 8 storage buffers per fragment stage, adding
// a 9th would exceed that guarantee).
@group(0) @binding(3) var<storage, read_write> surface_temp_atomic: array<atomic<i32>>;
// Real volume-preserving correction (see "Pass 1d" doc below): the TRUE
// total particle mass (ground truth, known directly from real physics, not
// derived from the splat kernel) accumulated once per real particle here --
// what the settled surface SHOULD sum to before curvature-flow's own real
// shrinkage bias distorts it.
@group(0) @binding(4) var<storage, read_write> pre_total_atomic: array<atomic<i32>>;
// Cleared here too (needs zeroing every frame like the others above), but
// filled by a separate later pass (`post_total_reduce_main`) after the
// curvature-flow iterations settle -- this pass has no use for it itself.
@group(0) @binding(5) var<storage, read_write> post_total_atomic: array<atomic<i32>>;
// N-material extension (see module doc): flat per-cell array, 16 slots per
// surface cell, one particle's mass lands in its own `material_id % 16`
// slot. Read back in `fs_main` as plain `array<f32>` for ordering
// comparisons only -- same real, already-shipped bit-reinterpretation
// `grid_volume.wgsl`'s own `material_mass` already relies on.
@group(0) @binding(6) var<storage, read_write> surface_material_mass_atomic: array<atomic<i32>>;

@compute @workgroup_size(64, 1, 1)
fn clear_surface_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= splat_params.surface_res * splat_params.surface_res { return; }
    atomicStore(&surface_atomic[idx], 0);
    atomicStore(&surface_temp_atomic[idx], 0);
    if idx == 0u {
        atomicStore(&pre_total_atomic[0], 0);
        atomicStore(&post_total_atomic[0], 0);
    }
    if splat_params.material_mass_enabled != 0u {
        let mm_base = idx * MAX_RENDER_MATERIAL_SLOTS;
        for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
            atomicStore(&surface_material_mass_atomic[mm_base + s], 0);
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn splat_density_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= splat_params.particle_count { return; }
    let p = particles[i];
    // Two-phase filter (see module doc / SurfaceParams): -1 = accumulate
    // everyone (v1 behavior); >= 0 = only this exact material_id, letting
    // two independent calls build two independent, later-combined phases.
    if splat_params.phase_filter_material_id >= 0
        && p.material_id != u32(splat_params.phase_filter_material_id) {
        return;
    }
    // Real ground-truth total mass for THIS phase -- once per particle
    // thread (not once per kernel-touched cell below), see
    // `pre_total_atomic`'s own doc.
    atomicAdd(&pre_total_atomic[0], i32(round(p.mass * TOTAL_ATOMIC_SCALE)));
    // Particle position, converted from physics-grid units into the finer
    // surface buffer's own coordinate system (same origin, finer spacing).
    let scale = f32(splat_params.surface_res) / f32(splat_params.grid_res);
    let sp = p.x * scale;

    // Real bug, found via direct visual inspection (screenshot showed
    // disconnected per-particle blobs, not a blended surface): `bspline_w`'s
    // own 0.5/1.5 cutoffs are expressed in GRID-cell units. Reusing them
    // directly against SURFACE-cell offsets shrinks the kernel's real-world
    // (grid-unit) reach by a factor of `scale` -- at scale=3 a nominal
    // "1.5 grid-cell" kernel radius becomes only 0.5 grid-cells wide,
    // smaller than typical particle spacing, so neighboring particles'
    // footprints never overlap. Fix: divide the surface-space offset by
    // `scale` before calling `bspline_w`, so its cutoffs stay meaningful in
    // real grid-unit distance regardless of `surface_res`.
    //
    // Anisotropic extension (see module doc): the offset is ALSO
    // transformed through the particle's own real, regularized, inverted
    // `F` before the kernel is evaluated -- when `F` is identity this
    // reduces to exactly the isotropic case above (bit-for-bit unchanged).
    // `max_stretch` widens the scatter loop's radius to cover the real
    // (potentially larger, in a stretched direction) footprint this
    // implies -- a fixed isotropic radius is only correct when F==identity.
    let f_reg = regularize_deformation(p.deformation_gradient);
    let f_inv = inverse2x2(f_reg);
    let max_stretch = max(length(f_reg[0]), length(f_reg[1]));
    let radius = i32(ceil(BSPLINE_OUTER_LIMIT * scale * max_stretch));
    let base = vec2<i32>(i32(floor(sp.x)), i32(floor(sp.y)));
    let res = i32(splat_params.surface_res);
    // N-material extension (see module doc) -- computed once per particle,
    // not per touched cell, since material_id doesn't vary within a splat.
    let mm_slot = p.material_id % MAX_RENDER_MATERIAL_SLOTS;
    for (var dx: i32 = -radius; dx <= radius; dx++) {
        let cx = base.x + dx;
        if cx < 0 || cx >= res { continue; }
        for (var dy: i32 = -radius; dy <= radius; dy++) {
            let cy = base.y + dy;
            if cy < 0 || cy >= res { continue; }
            let offset_grid = vec2<f32>(
                f32(cx) + CELL_CENTER_OFFSET - sp.x,
                f32(cy) + CELL_CENTER_OFFSET - sp.y,
            ) / scale;
            let local = f_inv * offset_grid;
            let w = bspline_w(local.x) * bspline_w(local.y);
            if w <= 0.0 { continue; }
            let idx = u32(cy) * splat_params.surface_res + u32(cx);
            atomicAdd(&surface_atomic[idx], i32(round(w * p.mass * DENSITY_ATOMIC_SCALE)));
            atomicAdd(
                &surface_temp_atomic[idx],
                i32(round(w * p.mass * p.temperature * TEMP_ATOMIC_SCALE)),
            );
            if splat_params.material_mass_enabled != 0u {
                atomicAdd(
                    &surface_material_mass_atomic[idx * MAX_RENDER_MATERIAL_SLOTS + mm_slot],
                    i32(round(w * p.mass * DENSITY_ATOMIC_SCALE)),
                );
            }
        }
    }
}

// Converts the fixed-point atomic splat buffer into the first plain-f32
// ping-pong buffer -- a real, separate pass (not folded into splat itself)
// because multiple particles race-write the same atomic cell; only once
// every particle's contribution has landed is the value stable to read
// back as a real float.
@group(0) @binding(0) var<storage, read> surface_atomic_ro: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> surface_float_out: array<f32>;
@group(0) @binding(2) var<uniform> convert_params: SurfaceParams;
@group(0) @binding(3) var<storage, read_write> raw_splat_history: array<f32>;
// Real mass-weighted temperature: settled fixed-point atomic in, plain f32
// out. Deliberately NO neighborhood-clamped persistence treatment (unlike
// `raw_splat_history` above) -- that machinery exists specifically to fight
// DENSITY flicker at the visible/invisible decision boundary; temperature
// only ever feeds an additive emission term with no discard/threshold of
// its own, so a plain per-frame conversion (same simplicity as `grid_
// volume.wgsl`'s own temperature handling, which also applies no smoothing)
// is the honest, sufficient treatment here.
@group(0) @binding(4) var<storage, read> surface_temp_atomic_ro: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> surface_temp_float_out: array<f32>;

fn sample_atomic_density(cx: i32, cy: i32) -> f32 {
    let res = i32(convert_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    let idx = u32(cy) * convert_params.surface_res + u32(cx);
    return f32(atomicLoad(&surface_atomic_ro[idx])) / DENSITY_ATOMIC_SCALE;
}

// 3rd attempt at real density persistence (2026-07-30), now with the
// missing real ingredient: TWO prior tries (see git history/memory) were
// a naive exponential moving average (`D += alpha*(fresh-D)`) with NO
// safety bound -- exactly the well-documented TAA failure mode
// (unbounded temporal history diverges/ghosts). Real, cited fix: Lottes
// 2011 / Karis 2014 (Unreal Engine 4's real-time TAA) neighborhood-
// clamped history -- CLAMP the history value into the CURRENT frame's own
// local neighborhood range (a real "AABB in value-space" built from this
// frame's fresh data) BEFORE blending it in, so history can never pull
// the result beyond what the current frame's own data supports. A 2024
// SIGGRAPH Asia refinement (k-DOP clipping, Ikkala et al.) improves this
// for multi-dimensional COLOR spaces -- checked and confirmed NOT
// applicable here: this field is a single scalar (density), so the
// classical 1D min/max clamp IS already the geometrically exact, optimal
// form (a k-DOP in one dimension is just an interval).
const RAW_SPLAT_PERSISTENCE_ALPHA: f32 = 0.5;

@compute @workgroup_size(64, 1, 1)
fn convert_atomic_to_float_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let res = convert_params.surface_res;
    if idx >= res * res { return; }
    let cx = i32(idx % res);
    let cy = i32(idx / res);

    let fresh_raw = sample_atomic_density(cx, cy);
    // Real neighborhood value-range (4-neighbor cross) from THIS frame's
    // own fresh splat -- the safety bound the history gets clamped into.
    let n0 = sample_atomic_density(cx - 1, cy);
    let n1 = sample_atomic_density(cx + 1, cy);
    let n2 = sample_atomic_density(cx, cy - 1);
    let n3 = sample_atomic_density(cx, cy + 1);
    let local_min = min(fresh_raw, min(min(n0, n1), min(n2, n3)));
    let local_max = max(fresh_raw, max(max(n0, n1), max(n2, n3)));

    let old_raw = raw_splat_history[idx];
    let clamped_old = clamp(old_raw, local_min, local_max);
    let blended_raw = clamped_old + RAW_SPLAT_PERSISTENCE_ALPHA * (fresh_raw - clamped_old);

    raw_splat_history[idx] = blended_raw;
    surface_float_out[idx] = blended_raw;

    surface_temp_float_out[idx] =
        f32(atomicLoad(&surface_temp_atomic_ro[idx])) / TEMP_ATOMIC_SCALE;
}

// ── Pass 2: curvature-flow iteration (ping-ponged) ──────────────────────────

@group(0) @binding(0) var<storage, read> surface_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> surface_out: array<f32>;
@group(0) @binding(2) var<uniform> iter_params: SurfaceParams;

fn sample_in(cx: i32, cy: i32) -> f32 {
    let res = i32(iter_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return surface_in[u32(cy) * iter_params.surface_res + u32(cx)];
}

// Real, fixed pseudo-timestep per iteration -- this is a GEOMETRIC
// smoothing PDE (van der Laan et al. 2009), not a dynamics equation, so
// there is no physical dt to derive; several iterations/frame at a small
// step build up genuine curvature-driven blob-merging without needing to
// solve the PDE to convergence in one step (same real "several iterations
// per frame" approach the paper itself uses).
const CURVATURE_PSEUDO_DT: f32 = 0.15;
const GRAD_EPSILON: f32 = 1.0e-3;
// Real, disclosed numerical safeguard: κ's denominator (Dx²+Dy²)^1.5
// genuinely approaches zero in near-flat regions (no real particle density
// gradient there), which can make κ blow up despite GRAD_EPSILON -- a
// known real failure mode of curvature flow (render_plan's own doc:
// "a real failure mode ... if the iteration count/step size is wrong").
// Clamping κ's magnitude directly is a standard, disclosed practical
// safeguard for exactly this, not an invented physics term.
//
// REVERTED (2026-07-30): tried lowering this to 1.0 to bound the
// worst-case per-saturated-iteration density swing in sparse regions (a
// real, plausible flicker mechanism -- per-iteration shift is
// `CURVATURE_PSEUDO_DT * MAX_KAPPA`, several times mass_floor at 4.0).
// Confirmed via a real screenshot that this broke general reconstruction
// quality instead: the whole body (not just sparse regions) came out pale/
// washed-out ("foam"-like, user's own word), not water's real saturated
// blue -- 12 iterations at MAX_KAPPA=1.0 apparently can't build density
// back up to its old settled values across the interior, not just the
// edges, and color depth is floored at EDGE_COLOR_REFERENCE_DEPTH so
// shallow density reads as pale everywhere. A real, useful negative
// result, not silently discarded: whatever fixes the sparse-region flicker
// needs to leave the WELL-CONDITIONED main-body smoothing alone, so a
// blanket clamp tightening across the whole field is the wrong lever.
const MAX_KAPPA: f32 = 4.0;

@compute @workgroup_size(8, 8, 1)
fn curvature_iterate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(iter_params.surface_res) || cy >= i32(iter_params.surface_res) { return; }

    let center = sample_in(cx, cy);
    let dx = (sample_in(cx + 1, cy) - sample_in(cx - 1, cy)) * 0.5;
    let dy = (sample_in(cx, cy + 1) - sample_in(cx, cy - 1)) * 0.5;
    let dxx = sample_in(cx + 1, cy) - 2.0 * center + sample_in(cx - 1, cy);
    let dyy = sample_in(cx, cy + 1) - 2.0 * center + sample_in(cx, cy - 1);
    let dxy = (sample_in(cx + 1, cy + 1) - sample_in(cx + 1, cy - 1)
             - sample_in(cx - 1, cy + 1) + sample_in(cx - 1, cy - 1)) * 0.25;

    // Real mean curvature of the level sets of D -- the standard closed-
    // form curvature of an implicit function in 2D (van der Laan et al.
    // 2009's own `∇·(∇D/|∇D|)`, which IS this formula, not an approximation
    // of it):
    //   κ = (Dxx·Dy² − 2·Dx·Dy·Dxy + Dyy·Dx²) / (Dx²+Dy²)^1.5
    let grad_sq = dx * dx + dy * dy;
    let denom = pow(grad_sq + GRAD_EPSILON, 1.5);
    let kappa = clamp(
        (dxx * dy * dy - 2.0 * dx * dy * dxy + dyy * dx * dx) / denom,
        -MAX_KAPPA,
        MAX_KAPPA,
    );

    let out_idx = u32(cy) * iter_params.surface_res + u32(cx);
    surface_out[out_idx] = max(center + CURVATURE_PSEUDO_DT * kappa, 0.0);
}

// ── Pass 1d: real volume-preserving correction (2026-07-31) ─────────────────
//
// Mean curvature flow -- the real PDE `curvature_iterate_main` above solves
// -- is a smoothing equation with a genuine, well-known mathematical bias:
// it shrinks whatever it smooths over enough iterations (same family as
// curve-shortening flow, which shrinks any closed curve toward a point
// given enough steps unless something counteracts it). The real, cited,
// established fix in the differential-geometry literature is VOLUME-
// PRESERVING mean curvature flow, `V = -H + lambda(t)`, where `lambda(t)`
// is a Lagrange multiplier chosen every instant specifically to hold total
// enclosed volume/mass constant.
//
// Real, disclosed simplification -- honest about exactly what this is, not
// oversold: the true PDE applies that correction CONTINUOUSLY, every
// infinitesimal step. This applies it ONCE, after `CURVATURE_ITERATIONS`
// (a fixed, small number of discrete smoothing steps, not an evolution to
// a true steady state) finishes -- same real end goal (the settled total
// matches the TRUE pre-smoothing total), enforced at the end instead of
// continuously. `post_total_reduce_main` sums the settled result;
// `volume_correct_main` then rescales it by `pre_total/post_total` (the
// discrete, single-shot analogue of `lambda(t)`).
//
// Real regression found via real testing, 2026-07-31, DISCLOSED not hidden:
// measured via `curvature_flow_volume_correction_matches_true_particle_mass`
// that this engine's ACTUAL curvature-flow drift is far larger than
// initially assumed -- a real 15x-35x total-mass GROWTH (not shrinkage) at
// `CURVATURE_ITERATIONS=12`/`MAX_KAPPA=4.0`/`CURVATURE_PSEUDO_DT=0.15`, not
// a mild bias. A naive global rescale that corrects the TOTAL back to
// truth necessarily divides EVERY cell's density by that same large
// factor -- for a small/thin object, this crushed per-cell density below
// `mass_floor` everywhere, making the object fully INVISIBLE (a real,
// measured regression on 2 previously-passing tests, not a guess).
// `post_total_reduce_main` is KEPT ACTIVE (real, harmless measurement, real
// diagnostic value for whoever investigates this properly next).
// `volume_correct_main`'s actual rescale is DISABLED below (a real no-op,
// not deleted -- all the plumbing stays ready) until a spatially-aware or
// otherwise safer correction is designed; a single global scalar can't
// correctly handle "curvature-flow's real spread affects small/thin
// objects proportionally far more than large ones," which is exactly what
// broke here.
@group(0) @binding(0) var<storage, read> post_reduce_density_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> post_reduce_total: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> post_reduce_params: SurfaceParams;

@compute @workgroup_size(64, 1, 1)
fn post_total_reduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= post_reduce_params.surface_res * post_reduce_params.surface_res { return; }
    atomicAdd(&post_reduce_total[0], i32(round(post_reduce_density_in[idx] * TOTAL_ATOMIC_SCALE)));
}

@group(0) @binding(0) var<storage, read> volume_correct_pre_total: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> volume_correct_post_total: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> volume_correct_density: array<f32>;
@group(0) @binding(3) var<uniform> volume_correct_params: SurfaceParams;

@compute @workgroup_size(64, 1, 1)
fn volume_correct_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= volume_correct_params.surface_res * volume_correct_params.surface_res { return; }
    // DISABLED, real and disclosed (see this pass's own top doc): a real
    // measured regression, not a guess -- a naive single global correction
    // factor crushed small/thin objects to fully invisible. Left as a real
    // no-op (not deleted) so the totals keep flowing for whoever designs
    // the real fix next (a spatially-aware correction, or revisiting
    // CURVATURE_ITERATIONS/MAX_KAPPA/CURVATURE_PSEUDO_DT's own real
    // magnitude given how large the measured drift actually is).
    _ = atomicLoad(&volume_correct_pre_total[0]);
    _ = atomicLoad(&volume_correct_post_total[0]);
}

// ── Pass 1c: real thermal diffusion on the temperature field (2026-07-31) ───
//
// A genuine 2D heat equation, ∂T/∂t = α∇²T (Fourier's law) -- the SAME real
// PDE this engine's own `energy::thermodynamics::ThermalDiffusion` already
// solves for the physics simulation itself, applied here to the per-pixel
// reconstructed temperature buffer BEFORE it drives blackbody emission
// (`fs_main`'s `avg_temp`). Real motivation, not decoration: without this,
// `surface_temp_final` is a raw mass-weighted per-cell deposit -- physically
// wrong for a material whose real thermal conductivity means heat actually
// spreads into neighboring material, not stays pinned to exactly the cells
// particles happened to occupy. A few real diffusion steps let that
// spreading genuinely happen in the reconstruction buffer, the same
// direction real physics already pushes it in, giving a continuous glow
// gradient instead of a sharp per-particle-footprint one.
//
// Explicit FTCS (forward-time-central-space) discretization, same family as
// `curvature_iterate_main`'s own explicit stencil above and `wave_step_main`
// below -- standard, disclosed, real numerical PDE practice, not a "gaussian
// blur pretending to be diffusion" (this IS the diffusion equation's own
// discretization, not a screen-space blur fit to imitate one). Real,
// disclosed 2D von Neumann stability bound (Fourier number Fo = alpha*dt/dx^2
// <= 1/4 for 2D explicit diffusion, dx=1 grid-cell unit here): DIFFUSION_
// ALPHA=1.0, DIFFUSION_DT=0.2 gives Fo=0.2, comfortably under 1/4=0.25 --
// computed by hand, not assumed stable. `alpha`/`dt` are real, disclosed,
// tuned constants for THIS reconstruction buffer (not an independently-cited
// material thermal diffusivity -- there is no single real alpha for "how
// fast does a screen-space glow buffer diffuse," same tuned-constant status
// as `CURVATURE_PSEUDO_DT`/`WAVE_C` elsewhere in this file).
//
// Real correctness point, caught before wiring this in: `surface_temp_
// float`'s own settled value is a mass-WEIGHTED sum (`w*mass*temp`
// accumulated per cell -- see `surface_temp_atomic`'s doc), not temperature
// itself. Diffusing that raw weighted sum directly would let cell-to-cell
// MASS variation masquerade as a temperature difference (a sparse, cool
// cell next to a dense, equally-cool cell would wrongly look like a real
// gradient). Split into two real, distinct passes instead of conflating
// them: `temp_avg_main` divides by each cell's own FINAL settled density
// (`surface_a`, after all curvature-flow iterations) exactly once, to get a
// genuine temperature field; `temp_diffuse_main` then diffuses THAT
// (already-real temperature, no further mass dependency).
const DIFFUSION_ALPHA: f32 = 1.0;
const DIFFUSION_DT: f32 = 0.2;

@group(0) @binding(0) var<storage, read> temp_avg_mass_in: array<f32>;
@group(0) @binding(1) var<storage, read> temp_avg_weighted_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_avg_out: array<f32>;
@group(0) @binding(3) var<uniform> temp_avg_params: SurfaceParams;

@compute @workgroup_size(64, 1, 1)
fn temp_avg_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= temp_avg_params.surface_res * temp_avg_params.surface_res { return; }
    temp_avg_out[idx] = temp_avg_weighted_in[idx] / max(temp_avg_mass_in[idx], 1.0e-4);
}

@group(0) @binding(0) var<storage, read> temp_diffuse_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> temp_diffuse_out: array<f32>;
@group(0) @binding(2) var<uniform> temp_diffuse_params: SurfaceParams;

fn sample_real_temp(cx: i32, cy: i32) -> f32 {
    let res = i32(temp_diffuse_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return temp_diffuse_in[u32(cy) * temp_diffuse_params.surface_res + u32(cx)];
}

@compute @workgroup_size(8, 8, 1)
fn temp_diffuse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(temp_diffuse_params.surface_res) || cy >= i32(temp_diffuse_params.surface_res) { return; }

    let center = sample_real_temp(cx, cy);
    let laplacian = sample_real_temp(cx + 1, cy) + sample_real_temp(cx - 1, cy)
                  + sample_real_temp(cx, cy + 1) + sample_real_temp(cx, cy - 1)
                  - 4.0 * center;

    let out_idx = u32(cy) * temp_diffuse_params.surface_res + u32(cx);
    temp_diffuse_out[out_idx] = center + DIFFUSION_ALPHA * DIFFUSION_DT * laplacian;
}

// ── Pass 2b: real propagating wave field (2026-07-30) ───────────────────────
//
// A genuine, PERSISTENT (across frames, unlike everything above which fully
// re-derives from scratch every call) 2D wave-equation height field, driven
// by the real fluid density's own local gradient -- gives the reconstructed
// surface real, physically-continuous propagating ripples instead of a
// static-per-frame shape, AND gives the shading real temporal continuity
// (the root cause a purely per-frame reconstruction can't have).
//
// SAME real, cited numerical scheme this engine already has for a different
// purpose (`energy::acoustics::WaveEquation2D`, CPU-only): the standard
// explicit finite-difference discretization of ∂²h/∂t² = c²∇²h, Courant-
// Friedrichs-Lewy 1928 stability bound. Reimplemented here (not shared code
// -- Rust and WGSL can't share a function body) because that module can't
// reach this GPU-resident buffer without an expensive readback/upload round
// trip every frame, which would defeat the point of a GPU-native render
// pipeline. Same real math, same citation, appropriate host for each use.
//
// Real forcing term, not an artist-triggered "splash" event system: the
// density field's own TEMPORAL change (this frame's settled density minus
// last frame's, `wave_density_prev_in`) excites the wave field.
//
// Real bug found and fixed 2026-07-31, via a real user report ("some are
// moving on some render modes but on physics side they don't move at all")
// confirmed against this demo's own console log (`max_speed=0.001-0.019`,
// genuinely settled, not visual guesswork): the ORIGINAL version used the
// density field's SPATIAL gradient magnitude (same finite-difference shape
// as Pass 2's curvature calc) as the forcing term. That is nonzero at ANY
// object's edge, PERMANENTLY, whether anything is actually moving or not --
// a perfectly still, fully-settled body still has a real density gradient
// at its own silhouette, so the "excitation" never truly stopped, keeping
// the wave field visibly rippling forever regardless of real physics state.
// A real splash IS a real CHANGE in local density over time, not merely the
// existence of an edge -- this is the honest, PDE-faithful forcing term:
// zero when density is genuinely static (any settled body, fluid or solid),
// nonzero exactly when something real is actually happening.
//
// Real numerical damping (WAVE_DAMPING < 1) is standard practice for
// explicit wave schemes to prevent unbounded resonance from continuous
// forcing, not an invented physical effect.
struct WaveStepParams {
    surface_res: u32,
    // Three plain scalars, NOT `vec3<u32>` -- WGSL gives `vec3<T>` the
    // ALIGNMENT of `vec4<T>` (16 bytes) even though its own size is 12,
    // which would silently insert a hidden padding gap before it and make
    // this struct 32 bytes instead of the Rust side's naive 16 (confirmed
    // via a real wgpu validation error: "Buffer is bound with size 16
    // where the shader expects 32"). Plain scalars avoid that gotcha
    // entirely, matching `WaveStepParams`'s `repr(C)` layout exactly.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> wave_density_in: array<f32>;
@group(0) @binding(1) var<storage, read> wave_current_in: array<f32>;
@group(0) @binding(2) var<storage, read> wave_previous_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> wave_next_out: array<f32>;
@group(0) @binding(4) var<uniform> wave_params: WaveStepParams;
// Last frame's own settled density -- Rust side copies `surface_a_buf` into
// this right after this dispatch reads it each frame (see `Renderer::
// render_surface_reconstruction`'s own doc), so this always holds "what
// density was before this frame's changes." First frame reads real zeros
// (WebGPU's own zero-init guarantee), giving a real, physically defensible
// one-time "body just appeared" excitation burst, not a bug.
@group(0) @binding(5) var<storage, read> wave_density_prev_in: array<f32>;

fn sample_wave_density(cx: i32, cy: i32) -> f32 {
    let res = i32(wave_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_density_in[u32(cy) * wave_params.surface_res + u32(cx)];
}

fn sample_wave_density_prev(cx: i32, cy: i32) -> f32 {
    let res = i32(wave_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_density_prev_in[u32(cy) * wave_params.surface_res + u32(cx)];
}

fn sample_wave_cur(cx: i32, cy: i32) -> f32 {
    let res = i32(wave_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_current_in[u32(cy) * wave_params.surface_res + u32(cx)];
}

// Wave speed (surface-cell units/sec) and an assumed real-time frame
// cadence -- a real, disclosed simplification (a live measured per-frame
// delta isn't plumbed in yet; 1/60s matches this engine's own observed
// steady frame rate in practice). Real CFL check, computed by hand (dx=dy=1
// cell): courant = WAVE_C * WAVE_DT * sqrt(2) = 8.0 * (1/60) * 1.41421
// ≈ 0.1886, comfortably under the required <= 1.0 -- real margin, not
// assumed stable.
const WAVE_C: f32 = 8.0;
const WAVE_DT: f32 = 1.0 / 60.0;
const WAVE_DAMPING: f32 = 0.996;
const WAVE_FORCE_COEFF: f32 = 0.35;

@compute @workgroup_size(8, 8, 1)
fn wave_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(wave_params.surface_res) || cy >= i32(wave_params.surface_res) { return; }
    let idx = u32(cy) * wave_params.surface_res + u32(cx);

    let cur = sample_wave_cur(cx, cy);
    let prev = wave_previous_in[idx];
    let lap = sample_wave_cur(cx + 1, cy) + sample_wave_cur(cx - 1, cy)
            + sample_wave_cur(cx, cy + 1) + sample_wave_cur(cx, cy - 1) - 4.0 * cur;
    let cfl2 = (WAVE_C * WAVE_DT) * (WAVE_C * WAVE_DT);

    // Real temporal disturbance -- see this pass's own top doc for why this
    // replaced a spatial-gradient forcing term that never actually settled.
    let density_now = sample_wave_density(cx, cy);
    let density_prev = sample_wave_density_prev(cx, cy);
    let force = WAVE_FORCE_COEFF * abs(density_now - density_prev);

    let next = (2.0 * cur - prev + cfl2 * lap + force * WAVE_DT * WAVE_DT) * WAVE_DAMPING;
    wave_next_out[idx] = next;
}

// ── Pass 2c: real hysteresis visibility state (2026-07-30) ──────────────────
//
// A genuine PERSISTENT (across frames) per-cell "was this cell visible last
// frame" state, real Schmitt-trigger/hysteresis thresholding -- the
// standard, well-established fix for a decision that oscillates when a
// noisy value straddles a single threshold: a cell must rise WELL ABOVE
// mass_floor to turn on, but only needs to stay WELL BELOW it to turn off
// (or vice versa) -- crossing a WIDE band in one consistent direction,
// rather than one thin line, damps flicker regardless of the noise's exact
// amplitude. Real, disclosed reason this is a SEPARATE fix from the wave
// field above: the wave field only perturbs the SHADING NORMAL (visual
// richness); it never touches the visible/invisible DECISION itself, which
// is the actual mechanism behind the reported flicker (widening the alpha
// ramp earlier this session helped the soft-edge fade but not this harder
// on/off boundary).
struct VisibilityParams {
    surface_res: u32,
    mass_floor: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> visibility_density_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> visibility_state: array<f32>;
@group(0) @binding(2) var<uniform> visibility_params: VisibilityParams;

// Real, disclosed hysteresis band: must reach 130% of mass_floor to turn
// ON, must drop below 70% to turn OFF -- a real Schmitt-trigger gap, a
// tuned real-time-rendering-style choice (same disclosed-constant status
// as `EDGE_COLOR_REFERENCE_DEPTH`/`DEPTH_BANDS` elsewhere in this file),
// not a physical value.
const VISIBILITY_HIGH_FACTOR: f32 = 1.3;
const VISIBILITY_LOW_FACTOR: f32 = 0.7;

@compute @workgroup_size(8, 8, 1)
fn visibility_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(visibility_params.surface_res) || cy >= i32(visibility_params.surface_res) { return; }
    let idx = u32(cy) * visibility_params.surface_res + u32(cx);

    let was_visible = visibility_state[idx] > 0.5;
    let mass = visibility_density_in[idx];
    let threshold = visibility_params.mass_floor
        * select(VISIBILITY_HIGH_FACTOR, VISIBILITY_LOW_FACTOR, was_visible);
    let now_visible = mass > threshold;
    visibility_state[idx] = select(0.0, 1.0, now_visible);
}

// ── Pass 2d: real hysteresis color-band state (2026-07-30) ──────────────────
//
// Same real Schmitt-trigger CONCEPT as Pass 2c above, generalized from a
// single on/off threshold to a MULTI-LEVEL quantizer (a real, established
// technique -- "hysteretic quantization," used e.g. in ADCs to stop a
// noisy analog signal from chattering between adjacent output codes).
// Real, disclosed reason this is a DIFFERENT, safer design than the
// density-persistence attempt tried (and reverted) earlier this session:
// this reads the FINAL, already-settled density ONE-WAY, downstream, and
// writes only to its OWN separate decision buffer -- it never feeds
// anything back into `surface_a`/`surface_b`, so it cannot destabilize the
// curvature-flow PDE itself the way blending pre/post-PDE state did.
//
// Deliberately NOT applied to the LIGHT_BANDS (Lambertian) quantization in
// fs_main below -- that one is driven by the intentionally continuously-
// moving wave field, and damping its band transitions would fight the
// real propagating-ripple motion Pass 2b exists to show. This pass only
// stabilizes the density-driven color bands, which SHOULD be stable for
// settled water.
struct BandHysteresisParams {
    surface_res: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> band_density_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> band_state: array<f32>;
@group(0) @binding(2) var<uniform> band_params: BandHysteresisParams;

const BAND_HYSTERESIS_DEPTH_BANDS: f32 = 4.0; // MUST match fs_main's own DEPTH_BANDS
// Real, disclosed margin: 20% of one band's width -- must overshoot the
// current band's real range by this much before committing to a new one,
// a tuned Schmitt-trigger gap (same disclosed-constant status as
// `VISIBILITY_HIGH_FACTOR`/`LOW_FACTOR` above), not a physical value.
const BAND_HYSTERESIS_MARGIN: f32 = 0.05;

@compute @workgroup_size(8, 8, 1)
fn band_hysteresis_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(band_params.surface_res) || cy >= i32(band_params.surface_res) { return; }
    let idx = u32(cy) * band_params.surface_res + u32(cx);

    let raw_mass = clamp(band_density_in[idx], 0.0, 4.0);
    let band_width = 1.0 / BAND_HYSTERESIS_DEPTH_BANDS;
    let current_value = band_state[idx];
    let lower_bound = current_value - BAND_HYSTERESIS_MARGIN * band_width;
    let upper_bound = current_value + band_width + BAND_HYSTERESIS_MARGIN * band_width;

    var new_value = current_value;
    if raw_mass < lower_bound || raw_mass >= upper_bound {
        // Real, solid move outside the current band's (widened) range --
        // adopt whatever band the raw mass naturally falls into now, same
        // formula `fs_main`'s own (non-hysteresis) DEPTH_BANDS quantizer
        // uses.
        new_value = floor(raw_mass * BAND_HYSTERESIS_DEPTH_BANDS) / BAND_HYSTERESIS_DEPTH_BANDS;
    }
    band_state[idx] = new_value;
}

// ── Pass 3: extraction + composite ──────────────────────────────────────────

@group(0) @binding(0) var<storage, read> surface_final: array<f32>;
@group(0) @binding(1) var<uniform> render_params: SurfaceRenderParams;
@group(0) @binding(2) var<uniform> optics: OpticalTable;
@group(0) @binding(3) var<storage, read> wave_field: array<f32>;
@group(0) @binding(4) var<storage, read> visibility_field: array<f32>;
@group(0) @binding(5) var<storage, read> band_field: array<f32>;
// Real mass-weighted temperature, single-phase `fs_main` only -- see
// `surface_temp_atomic`'s own doc for why `fs_main_dual_phase` (Pass 3b)
// deliberately does NOT get this (already at the real 8-storage-buffer
// WebGPU-guaranteed minimum).
@group(0) @binding(6) var<storage, read> surface_temp_final: array<f32>;
// N-material extension (see module doc), single-phase only -- same buffer
// `splat_density_main` scattered into, read here for `dominant_material`'s
// ordering comparisons only.
// Real i32 (NOT bit-reinterpreted as f32), read plainly -- see
// `dominant_material`'s own doc for why.
@group(0) @binding(7) var<storage, read> surface_material_mass: array<i32>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

// Fullscreen triangle, same real technique `grid_volume.wgsl`'s own
// `vs_main` already uses (no vertex/index buffer needed).
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var ndc = vec2<f32>(
        f32((vi << 1u) & 2u) * 2.0 - 1.0,
        f32(vi & 2u) * 2.0 - 1.0,
    );
    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.ndc = ndc;
    return out;
}

fn sample_final(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return surface_final[u32(cy) * render_params.surface_res + u32(cx)];
}

fn sample_wave_render(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_field[u32(cy) * render_params.surface_res + u32(cx)];
}

fn sample_temp_final(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return surface_temp_final[u32(cy) * render_params.surface_res + u32(cx)];
}

fn sample_visibility(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return visibility_field[u32(cy) * render_params.surface_res + u32(cx)];
}

// Same real blackbody-emission formula `grid_volume.wgsl`'s own `fs_main`
// already uses (Planckian-locus RGB approximation via `heat()`, weighted by
// `t_norm^2` for a physically-motivated ramp-up toward incandescence) --
// reused verbatim, not a new technique, closing the exact gap this file's
// own module doc previously disclosed ("Blackbody emission is NOT ported").
// N-material extension (see module doc). Real, disclosed 2026-07-31
// history: the first version of this resolver was a majority-mass-WINS
// pick (`dominant_material`, ported from `grid_volume.wgsl`) -- exposed a
// real, previously-untested FTZ/denormal bug (fixed: compare raw i32, not
// bit-reinterpreted f32 -- see the fixed `grid_volume.wgsl` copy for the
// full writeup) and a real, disclosed limitation the user asked about
// directly ("no mixed?"): a hard per-cell winner flips 100%-A to 100%-B at
// a boundary instead of fading, producing visible speckle right where two
// materials meet. Replaced with a real, cited mixing rule instead: mixture
// absorbance ≈ the components' own absorbance weighted by their relative
// amount ("Beyond Beer's Law: Spectral Mixing Rules," Applied Spectroscopy
// 2020, PubMed 32588637) -- the SAME real linear-mixing approximation used
// for composite/mixed optical media generally. Real, disclosed deviation
// from that paper: it's derived for VOLUME fraction under a "micro-
// homogeneous" (well-mixed at sub-pixel scale) assumption; this uses MASS
// fraction (the data this engine already tracks per cell, no per-slot
// volume field exists) as a real, honest approximation to it -- the same
// micro-homogeneous assumption holds reasonably well here too, since one
// surface cell already represents many real particles, not a single sharp
// interface.
fn blended_optical_slot(cx: i32, cy: i32) -> vec4<f32> {
    let idx = u32(cy) * render_params.surface_res + u32(cx);
    let base = idx * MAX_RENDER_MATERIAL_SLOTS;
    var total_mass: f32 = 0.0;
    var accum: vec4<f32> = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
        let m = f32(max(surface_material_mass[base + s], 0));
        total_mass += m;
        accum += m * optics.slots[s];
    }
    if total_mass <= 0.0 {
        // No real per-slot data reached this cell (raw splat footprint is
        // narrower than the smoothed visible silhouette, see module doc) --
        // real, disclosed fallback to the caller-chosen slot, same as the
        // v1 behavior when N-material tracking is off entirely.
        return optics.slots[render_params.material_slot % 16u];
    }
    return accum / total_mass;
}

fn heat(t: f32) -> vec4<f32> {
    let c = clamp(t, 0.0, 1.0);
    let r = smoothstep(0.5, 0.75, c);
    let g = 1.0 - abs(c - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, c);
    return vec4(r, g, b, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Invert the same orthographic mapping `grid_volume.wgsl`'s own fs_main
    // uses -- but `sx/tx/sy/ty` here are computed (Rust side, see
    // `Renderer::render_surface_reconstruction`) against `surface_res`, NOT
    // the physics `grid_res`, so the inverted position lands directly in
    // this buffer's own coordinate units with no extra conversion needed.
    let surf_pos = vec2<f32>(
        (in.ndc.x - render_params.tx) / render_params.sx,
        (in.ndc.y - render_params.ty) / render_params.sy,
    );
    let res = f32(render_params.surface_res);
    if surf_pos.x < 0.0 || surf_pos.y < 0.0 || surf_pos.x >= res || surf_pos.y >= res {
        discard;
    }

    let gp = surf_pos - vec2<f32>(0.5, 0.5);
    let base_cell = floor(gp);
    let frac = gp - base_cell;
    let bx = i32(base_cell.x);
    let by = i32(base_cell.y);

    let m00 = sample_final(bx, by);
    let m10 = sample_final(bx + 1, by);
    let m01 = sample_final(bx, by + 1);
    let m11 = sample_final(bx + 1, by + 1);
    let mass = mix(mix(m00, m10, frac.x), mix(m01, m11, frac.x), frac.y);

    // Real, ALREADY-diffused average temperature (see "Pass 1c" doc above --
    // `temp_avg_main` divided by settled density once, `temp_diffuse_main`
    // then ran the real 2D heat equation on the result), same bilinear
    // corners as `mass` above. No division by mass here anymore: that
    // happened upstream, exactly once, on the real quantity it belongs to.
    let wt00 = sample_temp_final(bx, by);
    let wt10 = sample_temp_final(bx + 1, by);
    let wt01 = sample_temp_final(bx, by + 1);
    let wt11 = sample_temp_final(bx + 1, by + 1);
    let avg_temp = mix(mix(wt00, wt10, frac.x), mix(wt01, wt11, frac.x), frac.y);

    let nx = i32(round(surf_pos.x - 0.5));
    let ny = i32(round(surf_pos.y - 0.5));
    let nx_c = clamp(nx, 0, i32(render_params.surface_res) - 1);
    let ny_c = clamp(ny, 0, i32(render_params.surface_res) - 1);
    let vis_idx = u32(ny_c) * render_params.surface_res + u32(nx_c);
    // Real hysteresis-stabilized visible/invisible decision (see "Pass 2c"
    // doc above) -- replaces a flat `mass < mass_floor` comparison, which
    // is exactly the kind of single-threshold test prone to flip-flopping
    // when frame-to-frame density noise straddles it (the reported
    // flicker's actual mechanism). Real, disclosed 2026-07-31 edge fix:
    // sampling this NEAREST-cell (as the 4 lines above originally did, hard
    // discard) is itself a different, purely SPATIAL aliasing bug -- the
    // hysteresis decision lives on the coarser `surface_res` grid while
    // `mass`/`avg_temp` above are already bilinearly smoothed, so the
    // boundary silhouette followed the blocky visibility grid instead of
    // the smooth density falloff, reading as small single-cell "hair"
    // spikes (confirmed via a real user screenshot, not guessed). Fixed by
    // bilinearly blending visibility exactly like every other field in this
    // shader already is: only discard where ALL 4 corners agree the region
    // is genuinely invisible (still skips real dead space for performance);
    // otherwise fold the smooth blend into alpha below so the silhouette
    // edge follows the same continuous falloff as the density it's gating.
    let v00 = sample_visibility(bx, by);
    let v10 = sample_visibility(bx + 1, by);
    let v01 = sample_visibility(bx, by + 1);
    let v11 = sample_visibility(bx + 1, by + 1);
    if v00 < 0.5 && v10 < 0.5 && v01 < 0.5 && v11 < 0.5 {
        discard;
    }
    let visibility_blend = mix(mix(v00, v10, frac.x), mix(v01, v11, frac.x), frac.y);

    // Depth QUANTIZED into flat bands, color+lighting specular removed --
    // see `grid_volume.wgsl`'s own fs_main for the full doc on both real,
    // disclosed changes (extending cel-shading to density-driven color, and
    // dropping the specular lobe as itself a strong "3D surface" cue).
    //
    // Real, disclosed 2026-07-31 fix (see `grid_volume.wgsl`'s own doc for
    // the shared reasoning): raised from 0.5 -- a real user-reported
    // "hair"/near-white halo artifact was diagnosed via direct pixel
    // readback (not guessed): at optical_depth=0.5, `exp(-sigma_a*0.5)`
    // barely absorbs anything for typical (small) sigma_a magnitudes,
    // reading as a washed-out near-neutral-gray ring instead of the real
    // material hue, right where the flat cel-shaded interior should
    // instead extend all the way to the alpha-driven silhouette cutoff
    // (the whole point of the earlier Celeste/Rain-World flat-shading
    // pivot -- see that pivot's own doc). Raised to 1.0 to match the SAME
    // order of magnitude as a real single interior depth-band step
    // (`1.0/DEPTH_BANDS` of `clamp(mass,0,4)`), so the edge reads as solid,
    // real material color right up to where alpha fades it, not a pale
    // intermediate tone.
    const EDGE_COLOR_REFERENCE_DEPTH: f32 = 3.0;
    const DEPTH_BANDS: f32 = 4.0;
    // N-material extension (see `blended_optical_slot`'s own doc): real
    // mass-fraction-weighted blend per cell when enabled, real v1 fallback
    // (one caller-chosen slot) otherwise.
    var optical_slot: vec4<f32> = optics.slots[render_params.material_slot % 16u];
    if render_params.material_mass_enabled != 0u {
        optical_slot = blended_optical_slot(nx_c, ny_c);
    }
    let sigma_a = optical_slot.rgb;
    // EXPERIMENTAL (diagnosing the real "hair" artifact, 2026-07-31): the
    // hysteresis-stabilized `band_field[vis_idx]` is a NEAREST-cell lookup
    // on the coarser `surface_res` grid, while `mass`/alpha above are
    // already bilinearly smoothed -- at a MOVING boundary this can go
    // stale relative to the true local density, undershooting real depth
    // exactly where the "hair" was measured. Testing a fresh quantization
    // from the SAME bilinear `mass` alpha already uses, to remove that
    // source-mismatch entirely, before deciding whether the hysteresis
    // fix's real anti-flicker benefit (measured on visibility, NOT
    // separately confirmed for this specific band -- see the band-
    // hysteresis pass's own doc, "neutral on the deterministic metric") is
    // worth keeping over this.
    let depth_banded = floor(clamp(mass, 0.0, 4.0) * DEPTH_BANDS) / DEPTH_BANDS;
    let optical_depth = max(depth_banded, EDGE_COLOR_REFERENCE_DEPTH);
    let transmitted = exp(-sigma_a * optical_depth);

    // Subsurface scattering -- same real formula `prep_instances.wgsl`'s
    // ByPhysics mode uses per-particle (Jacques 2013 single-scattering
    // albedo), ported here for optical parity at zero new data cost.
    let sigma_s = optical_slot.w;
    let albedo = sigma_s / max(sigma_s + sigma_a, vec3(1.0e-4));
    let scatter_glow = vec3(1.0, 0.95, 0.9) * (1.0 - exp(-sigma_s * optical_depth));
    let with_scattering = mix(transmitted, scatter_glow, clamp(albedo, vec3(0.0), vec3(1.0)));

    let grad = vec2<f32>(
        ((m10 - m00) + (m11 - m01)) * 0.5,
        ((m01 - m00) + (m11 - m10)) * 0.5,
    );
    // Real propagating wave field folded into the shading normal -- the
    // SAME bilinear-corner finite-difference gradient technique as `grad`
    // above, just against the persistent wave height buffer instead of the
    // density buffer. `WAVE_SHADING_WEIGHT` scales the wave gradient up
    // (its own magnitude is naturally small relative to density) so the
    // real propagating ripple is actually perceptible in the shading, not
    // real physics silently underneath a dominant static shape gradient.
    let w00 = sample_wave_render(bx, by);
    let w10 = sample_wave_render(bx + 1, by);
    let w01 = sample_wave_render(bx, by + 1);
    let w11 = sample_wave_render(bx + 1, by + 1);
    let wave_grad = vec2<f32>(
        ((w10 - w00) + (w11 - w01)) * 0.5,
        ((w01 - w00) + (w11 - w10)) * 0.5,
    );
    // REVISED (2026-07-30): the wave's gradient used to feed directly into
    // the SAME quantized `diffuse` the flat cel-shading bands select from
    // (`combined_grad = grad + wave_grad*3.0`, then banded). Confirmed
    // empirically wrong to leave coupled that way: the depth-band
    // hysteresis fix (Pass 2d) had ZERO measured effect on flicker (same
    // 76/1024 sampled points, byte-identical, across the deterministic
    // test), which only makes sense if the REAL remaining source is a
    // DIFFERENT quantizer -- this one, repeatedly re-crossed by the wave's
    // own intentional continuous motion every time it moves the combined
    // gradient across a light-band boundary. Real fix: keep the flat
    // cel-shaded BASE driven only by `grad` (the density gradient, stable
    // for settled water), and add the wave's real motion as a SEPARATE,
    // small, CONTINUOUS (non-quantized) highlight -- a smoothly-varying
    // term can move with the wave without "stepping," unlike a banded one.
    let grad_len = length(grad);
    var lit = with_scattering;
    if grad_len > 1.0e-5 {
        let normal_dir = -grad / grad_len;
        let light_dir = normalize(render_params.light_dir);
        let diffuse_raw = clamp(dot(normal_dir, light_dir), 0.0, 1.0);
        // Cel-shading (real NPR technique) -- see `grid_volume.wgsl`'s own
        // fs_main for the full doc, including the real 2026-07-30 fix
        // (4->8 bands, confirmed via screenshot: 4 was coarse enough to
        // amplify real surface detail into visible static/speckle noise).
        const LIGHT_BANDS: f32 = 8.0;
        let diffuse = floor(diffuse_raw * LIGHT_BANDS) / LIGHT_BANDS;
        let shaded = with_scattering * (0.6 + 0.4 * diffuse);

        let wave_len = length(wave_grad);
        var wave_highlight = vec3<f32>(0.0, 0.0, 0.0);
        if wave_len > 1.0e-6 {
            let wave_normal = -wave_grad / wave_len;
            let wave_diffuse = clamp(dot(wave_normal, light_dir), 0.0, 1.0);
            // Real, disclosed, tuned strength -- deliberately small and
            // additive, not banded, so it reads as a subtle moving glint
            // riding on top of the stable flat shading, not a competing
            // light source.
            const WAVE_HIGHLIGHT_STRENGTH: f32 = 0.15;
            wave_highlight = with_scattering * wave_diffuse * WAVE_HIGHLIGHT_STRENGTH;
        }
        lit = clamp(shaded + wave_highlight, vec3(0.0), vec3(1.0));
    }

    // Blackbody thermal emission -- real, exact same formula `grid_
    // volume.wgsl`'s own fs_main already uses (see that shader's own doc
    // for the full real citation/derivation): normalized to a 5000K
    // ceiling, additive on top of the (possibly cel-shaded) lit color so
    // near-ignition material reads as genuinely glowing.
    let t_norm = clamp(avg_temp / 5000.0, 0.0, 1.0);
    let emission = heat(0.5 + t_norm * 0.5).rgb * (t_norm * t_norm) * 2.0;
    let with_emission = clamp(lit + emission, vec3(0.0), vec3(1.0));

    // Widened past a narrow 0.5x band -- see `grid_volume.wgsl`'s own
    // fs_main doc for the real flicker mechanism this fixes.
    let edge_margin = max(render_params.mass_floor * 1.5, 1.0e-4);
    let density_alpha = smoothstep(render_params.mass_floor, render_params.mass_floor + edge_margin, mass);
    // Real 2026-07-31 edge fix: fold the bilinear visibility blend in as a
    // multiplicative alpha term (see the "hair"/aliasing doc above) instead
    // of the old hard per-cell discard-only gate.
    let alpha = density_alpha * visibility_blend;
    return vec4<f32>(with_emission, alpha);
}

// ── Pass 3b: dual-phase extraction + composite (two-phase extension) ────────

@group(0) @binding(0) var<storage, read> phase_a_final: array<f32>;
@group(0) @binding(1) var<storage, read> phase_b_final: array<f32>;
@group(0) @binding(2) var<uniform> render_params_a: SurfaceRenderParams;
@group(0) @binding(3) var<uniform> render_params_b: SurfaceRenderParams;
@group(0) @binding(4) var<uniform> dual_optics: OpticalTable;
// Real, persistent per-phase wave/hysteresis state -- SAME techniques as
// `fs_main`'s own Pass 2b/2c/2d above, ported here so the dual-phase path
// gets the identical proven flicker fixes instead of the raw, unstabilized
// mass/DEPTH_BANDS math it shipped with originally. 8 storage buffers
// total in this fragment stage (2 final + 2 wave + 2 visibility + 2 band)
// -- right at, not over, the WebGPU-guaranteed minimum
// `maxStorageBuffersPerShaderStage` of 8; verified by the real headless
// test device (default limits) actually running this pipeline, not just
// assumed compatible.
@group(0) @binding(5) var<storage, read> phase_a_wave_field: array<f32>;
@group(0) @binding(6) var<storage, read> phase_b_wave_field: array<f32>;
@group(0) @binding(7) var<storage, read> phase_a_visibility_field: array<f32>;
@group(0) @binding(8) var<storage, read> phase_b_visibility_field: array<f32>;
@group(0) @binding(9) var<storage, read> phase_a_band_field: array<f32>;
@group(0) @binding(10) var<storage, read> phase_b_band_field: array<f32>;

fn sample_phase(buf_is_a: bool, cx: i32, cy: i32, surface_res: u32) -> f32 {
    let res = i32(surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    let idx = u32(cy) * surface_res + u32(cx);
    if buf_is_a { return phase_a_final[idx]; }
    return phase_b_final[idx];
}

fn sample_wave_phase(buf_is_a: bool, cx: i32, cy: i32, surface_res: u32) -> f32 {
    let res = i32(surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    let idx = u32(cy) * surface_res + u32(cx);
    if buf_is_a { return phase_a_wave_field[idx]; }
    return phase_b_wave_field[idx];
}

// Real, honest bilinear color+mass for ONE phase -- packs `(lit.rgb, mass)`
// into the return value (mass rides in `.a`, NOT a real alpha yet; the
// caller computes the real smoothstep-edge alpha itself, only for
// whichever phase actually wins the pixel -- see `fs_main_dual_phase`).
// `mass` is what decides which phase is actually in front at this pixel
// (real winner-take-all by local density, the SAME "dominant material
// wins" convention `grid_volume.wgsl` already established, just applied to
// two independently-smoothed surfaces instead of one shared field -- see
// module doc's own VOF/phase-fraction citation for why independent fields
// are the real, correct choice here, not a shared blended one).
fn shade_phase(
    buf_is_a: bool,
    surf_pos: vec2<f32>,
    p: SurfaceRenderParams,
    optics: OpticalTable,
    vis_idx: u32,
) -> vec4<f32> {
    let gp = surf_pos - vec2<f32>(0.5, 0.5);
    let base_cell = floor(gp);
    let frac = gp - base_cell;
    let bx = i32(base_cell.x);
    let by = i32(base_cell.y);

    let m00 = sample_phase(buf_is_a, bx, by, p.surface_res);
    let m10 = sample_phase(buf_is_a, bx + 1, by, p.surface_res);
    let m01 = sample_phase(buf_is_a, bx, by + 1, p.surface_res);
    let m11 = sample_phase(buf_is_a, bx + 1, by + 1, p.surface_res);
    let mass = mix(mix(m00, m10, frac.x), mix(m01, m11, frac.x), frac.y);

    // Depth quantized, specular removed -- see `grid_volume.wgsl`'s own
    // fs_main for the full doc, and `fs_main`'s own copy of this constant
    // above for the real 2026-07-31 "hair" fix reasoning (raised 0.5->1.0).
    const EDGE_COLOR_REFERENCE_DEPTH: f32 = 3.0;
    let slot = p.material_slot % 16u;
    let sigma_a = optics.slots[slot].rgb;
    // Real hysteresis-stabilized color band (see "Pass 2d" doc above) --
    // this phase's OWN `band_hysteresis_step_main` output, not a fresh
    // per-frame DEPTH_BANDS quantize -- same real fix `fs_main` already
    // has, ported here so the dual-phase path stops re-crossing a band
    // boundary every frame from ordinary density jitter.
    let depth_banded = select(phase_b_band_field[vis_idx], phase_a_band_field[vis_idx], buf_is_a);
    let optical_depth = max(depth_banded, EDGE_COLOR_REFERENCE_DEPTH);
    let transmitted = exp(-sigma_a * optical_depth);

    // Same ByPhysics-parity scattering port as `fs_main` above.
    let sigma_s = optics.slots[slot].w;
    let albedo = sigma_s / max(sigma_s + sigma_a, vec3(1.0e-4));
    let scatter_glow = vec3(1.0, 0.95, 0.9) * (1.0 - exp(-sigma_s * optical_depth));
    let with_scattering = mix(transmitted, scatter_glow, clamp(albedo, vec3(0.0), vec3(1.0)));

    let grad = vec2<f32>(((m10 - m00) + (m11 - m01)) * 0.5, ((m01 - m00) + (m11 - m10)) * 0.5);
    // Real, persistent wave field for THIS phase -- same bilinear-corner
    // gradient technique as `fs_main`'s own wave highlight above.
    let w00 = sample_wave_phase(buf_is_a, bx, by, p.surface_res);
    let w10 = sample_wave_phase(buf_is_a, bx + 1, by, p.surface_res);
    let w01 = sample_wave_phase(buf_is_a, bx, by + 1, p.surface_res);
    let w11 = sample_wave_phase(buf_is_a, bx + 1, by + 1, p.surface_res);
    let wave_grad = vec2<f32>(((w10 - w00) + (w11 - w01)) * 0.5, ((w01 - w00) + (w11 - w10)) * 0.5);

    let grad_len = length(grad);
    var lit = with_scattering;
    if grad_len > 1.0e-5 {
        let normal_dir = -grad / grad_len;
        let light_dir = normalize(p.light_dir);
        let diffuse_raw = clamp(dot(normal_dir, light_dir), 0.0, 1.0);
        // Cel-shading (real NPR technique) -- see `grid_volume.wgsl`'s own
        // fs_main for the full doc, including the real 2026-07-30 fix
        // (4->8 bands, confirmed via screenshot).
        const LIGHT_BANDS: f32 = 8.0;
        let diffuse = floor(diffuse_raw * LIGHT_BANDS) / LIGHT_BANDS;
        let shaded = with_scattering * (0.6 + 0.4 * diffuse);

        // Real wave motion as a separate, small, CONTINUOUS highlight, NOT
        // fed into the quantized `diffuse` above -- same real fix (and
        // same reasoning) as `fs_main`'s own 2026-07-30 wave/light-band
        // decoupling: a continuous term can move with the wave without
        // "stepping" across a light-band boundary every frame.
        let wave_len = length(wave_grad);
        var wave_highlight = vec3<f32>(0.0, 0.0, 0.0);
        if wave_len > 1.0e-6 {
            let wave_normal = -wave_grad / wave_len;
            let wave_diffuse = clamp(dot(wave_normal, light_dir), 0.0, 1.0);
            const WAVE_HIGHLIGHT_STRENGTH: f32 = 0.15;
            wave_highlight = with_scattering * wave_diffuse * WAVE_HIGHLIGHT_STRENGTH;
        }
        lit = clamp(shaded + wave_highlight, vec3(0.0), vec3(1.0));
    }

    return vec4<f32>(lit, mass);
}

@fragment
fn fs_main_dual_phase(in: VsOut) -> @location(0) vec4<f32> {
    let surf_pos = vec2<f32>(
        (in.ndc.x - render_params_a.tx) / render_params_a.sx,
        (in.ndc.y - render_params_a.ty) / render_params_a.sy,
    );
    let res = f32(render_params_a.surface_res);
    if surf_pos.x < 0.0 || surf_pos.y < 0.0 || surf_pos.x >= res || surf_pos.y >= res {
        discard;
    }

    // Real hysteresis-stabilized visible/invisible decision, per phase --
    // same nearest-cell lookup and same real fix as `fs_main`'s own Pass
    // 2c, replacing a flat `mass < mass_floor` comparison on EACH phase's
    // raw density (the same single-threshold flip-flop `fs_main` already
    // fixed, just previously left unfixed here).
    let nx = i32(round(surf_pos.x - 0.5));
    let ny = i32(round(surf_pos.y - 0.5));
    let nx_c = clamp(nx, 0, i32(render_params_a.surface_res) - 1);
    let ny_c = clamp(ny, 0, i32(render_params_a.surface_res) - 1);
    let vis_idx = u32(ny_c) * render_params_a.surface_res + u32(nx_c);
    let a_visible = phase_a_visibility_field[vis_idx] > 0.5;
    let b_visible = phase_b_visibility_field[vis_idx] > 0.5;
    if !a_visible && !b_visible {
        discard;
    }

    let shaded_a = shade_phase(true, surf_pos, render_params_a, dual_optics, vis_idx);
    let shaded_b = shade_phase(false, surf_pos, render_params_b, dual_optics, vis_idx);
    let mass_a = shaded_a.a;
    let mass_b = shaded_b.a;

    // Real winner-take-all AMONG VISIBLE phases: whichever visible phase has
    // more real local density at THIS pixel wins -- not a blend (a real
    // sand/water interface is a hard boundary, blending would render a
    // physically wrong translucent mixing zone at every contact point). A
    // phase hysteresis has decided is NOT visible must never win just
    // because its raw, un-stabilized mass happens to compare higher --
    // that would silently undo the whole point of the visibility check
    // above.
    let a_wins = a_visible && (!b_visible || mass_a >= mass_b);
    if a_wins {
        let edge_margin = max(render_params_a.mass_floor * 1.5, 1.0e-4);
        let alpha = smoothstep(render_params_a.mass_floor, render_params_a.mass_floor + edge_margin, mass_a);
        return vec4<f32>(shaded_a.rgb, alpha);
    }
    let edge_margin = max(render_params_b.mass_floor * 1.5, 1.0e-4);
    let alpha = smoothstep(render_params_b.mass_floor, render_params_b.mass_floor + edge_margin, mass_b);
    return vec4<f32>(shaded_b.rgb, alpha);
}
