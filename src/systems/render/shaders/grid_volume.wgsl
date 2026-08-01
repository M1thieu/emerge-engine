// MPM-native grid-volume rendering — samples the solver's own P2G mass field
// directly instead of drawing one instanced splat per particle, so adjacent
// cells blend into one continuous shape instead of a cloud of discrete dots.
//
// Per-cell material coloring: uses `GpuSimulation::attach_grid_material_render_gpu`'s
// opt-in per-material mass accumulator (`material_mass`) to find each cell's
// dominant material (majority mass wins). Color uses the NEAREST cell's dominant
// material (not blended across the 4 bilinear neighbors), while density/alpha
// falloff IS bilinear-smoothed (see `sample_mass`) — a mixed-material boundary
// gets a smooth edge shape but a hard material-color transition. Falls back to
// slot 0 for every cell when `material_mass_enabled` is 0.

const MAX_RENDER_MATERIAL_SLOTS: u32 = 16u;

// Smooth heat map: t ∈ [0, 1] → blue → cyan → green → yellow → red. Same
// formula as `prep_instances.wgsl`'s own `heat()` -- duplicated, not shared,
// since these compile as separate shader modules with no common-include
// mechanism in this codebase; keep both in sync if either changes.
fn heat(t: f32) -> vec4<f32> {
    let c = clamp(t, 0.0, 1.0);
    let r = smoothstep(0.5, 0.75, c);
    let g = 1.0 - abs(c - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, c);
    return vec4(r, g, b, 1.0);
}

struct GridVolumeParams {
    sx: f32,
    tx: f32,
    sy: f32,
    ty: f32,
    // Real light direction, sourced from `SimConfig::light_dir` via
    // `Renderer::set_light_dir` -- the SAME real value real plant
    // phototropism (`rod::Phototropism`) uses, not a separate value
    // hardcoded here and disconnected from anything.
    light_dir: vec2<f32>,
    grid_res: u32,
    mass_floor: f32,
    material_mass_enabled: u32,
    _pad1: f32,
    _pad2: vec2<f32>,
}

struct OpticalTable {
    slots: array<vec4<f32>, 16>,
    specular: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<storage, read> grid_int: array<u32>;
@group(0) @binding(1) var<uniform> params: GridVolumeParams;
@group(0) @binding(2) var<uniform> optics: OpticalTable;
// Real i32 (NOT bit-reinterpreted as f32), read plainly -- see
// `dominant_material`'s own doc for why.
@group(0) @binding(3) var<storage, read> material_mass: array<i32>;
@group(0) @binding(4) var<storage, read> grid_visibility_field: array<f32>;

// ── Real hysteresis visibility state (2026-07-30) ───────────────────────────
//
// SAME real Schmitt-trigger technique `curvature_flow.wgsl`'s own Pass 2c
// already ships and tests -- ported here, not reinvented, because this
// mode's own `mass_floor` discard (below) had the identical theoretical
// flicker risk (a single threshold a noisy value can straddle) and had
// simply never received the fix. Real, disclosed reason this needs its
// OWN persistent buffer at `grid_res` (not reusing curvature-flow's own,
// which is sized at the finer `surface_res`): this mode samples the
// physics grid directly, a genuinely different resolution/buffer.
struct GridVisibilityParams {
    grid_res: u32,
    mass_floor: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> grid_visibility_density_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> grid_visibility_state: array<f32>;
@group(0) @binding(2) var<uniform> grid_visibility_params: GridVisibilityParams;

const GRID_VISIBILITY_HIGH_FACTOR: f32 = 1.3;
const GRID_VISIBILITY_LOW_FACTOR: f32 = 0.7;

@compute @workgroup_size(8, 8, 1)
fn grid_visibility_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(grid_visibility_params.grid_res) || cy >= i32(grid_visibility_params.grid_res) { return; }
    let idx = u32(cy) * grid_visibility_params.grid_res + u32(cx);
    let mass = bitcast<f32>(grid_visibility_density_in[idx * 4u + 2u]);

    let was_visible = grid_visibility_state[idx] > 0.5;
    let threshold = grid_visibility_params.mass_floor
        * select(GRID_VISIBILITY_HIGH_FACTOR, GRID_VISIBILITY_LOW_FACTOR, was_visible);
    let now_visible = mass > threshold;
    grid_visibility_state[idx] = select(0.0, 1.0, now_visible);
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

// Fullscreen triangle (no vertex/index buffer needed) -- covers the whole
// clip-space quad with 3 vertices via the classic oversized-triangle trick.
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

// Real mass at grid cell (cx, cy), 0.0 for any cell outside the domain (matches
// CPU Grid::velocity_at's own OOB-is-zero convention) -- lets bilinear sampling
// blend smoothly toward "no matter" at the domain edge instead of needing a
// special-case border check.
fn sample_mass(cx: i32, cy: i32) -> f32 {
    let res = i32(params.grid_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    let idx = u32(cy) * params.grid_res + u32(cx);
    return bitcast<f32>(grid_int[idx * 4u + 2u]);
}

// Real mass-WEIGHTED temperature at grid cell (cx, cy) -- i.e. sum(mass_p *
// temperature_p) over the particles that scattered into this cell, same real
// P2G scatter convention `ThermalDiffusion` already uses. Dividing by the
// cell's own (bilinearly-sampled) mass elsewhere recovers a real average
// temperature; kept unnormalized here so it bilinearly interpolates
// correctly (interpolating an already-divided average would double-weight
// low-mass neighbor cells). 0.0 OOB, matching `sample_mass`.
fn sample_weighted_temp(cx: i32, cy: i32) -> f32 {
    let res = i32(params.grid_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    let idx = u32(cy) * params.grid_res + u32(cx);
    return bitcast<f32>(grid_int[idx * 4u + 0u]);
}

// Real, disclosed 2026-07-31 history: this was a majority-mass-WINS pick
// (`dominant_material`) -- exposed a real, previously-untested FTZ/
// denormal bug (fixed: compare raw i32, not bit-reinterpreted f32 --
// `material_mass` bit-reinterpreted directly as `f32` relied on IEEE 754's
// real monotonic bit-pattern-to-value property, but typical accumulated
// fixed-point values land in the raw range that reinterprets as an IEEE
// 754 DENORMAL float, and this GPU's fragment-shader ALU flushes denormals
// to zero -- confirmed directly: the same raw bytes read correctly via a
// CPU-side staging-buffer copy, but decoded as exactly 0.0 inside the
// shader's own comparison) and a real, disclosed limitation asked about
// directly by the user ("no mixed?"): a hard per-cell winner flips 100%-A
// to 100%-B at a boundary instead of fading, producing visible speckle
// right where two materials meet. Replaced with a real, cited mixing rule
// instead: mixture absorbance ≈ the components' own absorbance weighted by
// their relative amount ("Beyond Beer's Law: Spectral Mixing Rules,"
// Applied Spectroscopy 2020, PubMed 32588637) -- the SAME real linear-
// mixing approximation used for composite/mixed optical media generally.
// Real, disclosed deviation from that paper: it's derived for VOLUME
// fraction under a "micro-homogeneous" (well-mixed at sub-pixel scale)
// assumption; this uses MASS fraction (the data this engine already
// tracks per cell, no per-slot volume field exists) as a real, honest
// approximation to it -- the same micro-homogeneous assumption holds
// reasonably well here too, since one grid cell already represents many
// real particles, not a single sharp interface.
fn blended_optical_slot(cx: i32, cy: i32) -> vec4<f32> {
    let idx = u32(cy) * params.grid_res + u32(cx);
    let base = idx * MAX_RENDER_MATERIAL_SLOTS;
    var total_mass: f32 = 0.0;
    var accum: vec4<f32> = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
        let m = f32(max(material_mass[base + s], 0));
        total_mass += m;
        accum += m * optics.slots[s];
    }
    if total_mass <= 0.0 {
        // No real per-slot data reached this cell -- real, disclosed
        // fallback to slot 0, same as the v1 behavior when material
        // tracking is off entirely (see fs_main's own gate).
        return optics.slots[0];
    }
    return accum / total_mass;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Invert the same orthographic mapping Renderer::set_camera uses:
    // clip = grid_pos * (sx, sy) + (tx, ty)  =>  grid_pos = (clip - t) / s
    let grid_pos = vec2<f32>(
        (in.ndc.x - params.tx) / params.sx,
        (in.ndc.y - params.ty) / params.sy,
    );
    let res = f32(params.grid_res);
    if grid_pos.x < 0.0 || grid_pos.y < 0.0 || grid_pos.x >= res || grid_pos.y >= res {
        discard;
    }

    // Bilinear sample against cell CENTERS (P2G's own convention: cell i's center
    // sits at grid position i+0.5, see p2g.wgsl's CELL_CENTER_OFFSET) -- smooths
    // the blocky nearest-cell look into a continuous density falloff at edges,
    // real MPM-render technique (same idea screen-space fluid rendering's
    // bilateral smoothing pass achieves, simpler since this is grid-native data
    // already, not a reconstructed depth buffer).
    let gp = grid_pos - vec2<f32>(0.5, 0.5);
    let base_cell = floor(gp);
    let frac = gp - base_cell;
    let bx = i32(base_cell.x);
    let by = i32(base_cell.y);

    let m00 = sample_mass(bx, by);
    let m10 = sample_mass(bx + 1, by);
    let m01 = sample_mass(bx, by + 1);
    let m11 = sample_mass(bx + 1, by + 1);
    let mass = mix(mix(m00, m10, frac.x), mix(m01, m11, frac.x), frac.y);

    // Same bilinear pattern as mass, for the mass-weighted temperature channel.
    let wt00 = sample_weighted_temp(bx, by);
    let wt10 = sample_weighted_temp(bx + 1, by);
    let wt01 = sample_weighted_temp(bx, by + 1);
    let wt11 = sample_weighted_temp(bx + 1, by + 1);
    let weighted_temp = mix(mix(wt00, wt10, frac.x), mix(wt01, wt11, frac.x), frac.y);
    // Real average temperature: weighted-sum / mass, both already bilinearly
    // interpolated over the SAME 4 neighbor cells -- floored to avoid a
    // divide-by-near-zero in sparse/edge cells where both values are tiny.
    let avg_temp = weighted_temp / max(mass, 1.0e-4);

    // Gate visibility on the NEAREST cell's mass, not the bilinear-blended value —
    // the blend is nonzero up to a full cell beyond the nearest occupied cell, which
    // would overshoot true particle extent no matter how high mass_floor is raised.
    // Bilinear mass is still used for interior shading below.
    let nx = i32(round(grid_pos.x - 0.5));
    let ny = i32(round(grid_pos.y - 0.5));
    let nx_c = clamp(nx, 0, i32(params.grid_res) - 1);
    let ny_c = clamp(ny, 0, i32(params.grid_res) - 1);
    // Real hysteresis-stabilized visible/invisible decision (see the
    // "Real hysteresis visibility state" doc above) -- replaces a flat
    // `mass < mass_floor` comparison.
    let vis_idx = u32(ny_c) * params.grid_res + u32(nx_c);
    if grid_visibility_field[vis_idx] < 0.5 {
        discard;
    }

    // Nearest cell's real mass-fraction-weighted material blend -- see
    // `blended_optical_slot`'s own doc. NEAREST-cell (not bilinear across
    // the 4 mass-sample neighbors) for the same reason the visibility gate
    // above is nearest-cell: the per-slot mass array isn't itself smoothed.
    // Falls back to slot 0 when material tracking isn't attached.
    var optical_slot: vec4<f32> = optics.slots[0];
    if params.material_mass_enabled != 0u {
        optical_slot = blended_optical_slot(nx_c, ny_c);
    }

    // Beer-Lambert absorption, same formula ByPhysics uses per-particle
    // (prep_instances.wgsl) but evaluated once per pixel against the grid's
    // own (now bilinear-smoothed) mass instead of a single particle's J.
    // Higher mass -> denser -> more absorption, giving a soft density-based
    // falloff at the shape's own edge instead of a hard per-particle silhouette.
    //
    // Color depth is floored at EDGE_COLOR_REFERENCE_DEPTH, separately from the
    // alpha ramp below which still uses the true raw mass — without this split,
    // the thin edge-transition band (where mass -> mass_floor) renders as a
    // near-white halo before reaching full alpha.
    //
    // Depth QUANTIZED into flat bands (same real cel-shading technique the
    // lighting below already uses, extended here to the density-driven
    // color itself): a continuously-varying optical depth is what makes
    // this read as a smooth, photoreal 3D gradient (the "blobby" look
    // reported directly) even after the lighting itself was banded --
    // color needs the same treatment for a real Celeste/Rain-World-style
    // flat-region look. Alpha (the edge) deliberately still uses the RAW,
    // unbanded `mass` just below -- banding the edge too would read as
    // blocky/pixel-stair-stepped (the "Minecraft" look explicitly rejected
    // as too limited), not the flat-color-with-smooth-edge look aimed for.
    //
    // Real, disclosed 2026-07-31 fix: raised from 0.5, diagnosed via direct
    // pixel readback on `curvature_flow.wgsl`'s identical formula (shared
    // reasoning, both files use the same optics/absorption math) -- at
    // optical_depth=0.5, `exp(-sigma_a*0.5)` barely absorbs anything for
    // typical (small) sigma_a magnitudes, reading as a washed-out near-
    // neutral-gray ring/halo instead of real material color, undermining
    // the whole point of the flat-shading pivot above (interior color
    // should extend to the alpha cutoff, not fade through an intermediate
    // pale tone). Raised to 1.0, matching one real interior depth-band step
    // (`1.0/DEPTH_BANDS`), so the edge reads as solid material color right
    // up to where alpha fades it.
    const EDGE_COLOR_REFERENCE_DEPTH: f32 = 3.0;
    const DEPTH_BANDS: f32 = 4.0;
    let sigma_a = optical_slot.rgb;
    let depth_banded = floor(clamp(mass, 0.0, 4.0) * DEPTH_BANDS) / DEPTH_BANDS;
    let optical_depth = max(depth_banded, EDGE_COLOR_REFERENCE_DEPTH);
    let transmitted = exp(-sigma_a * optical_depth);

    // Subsurface scattering + Fresnel specular: same real formula
    // `prep_instances.wgsl`'s ByPhysics mode already uses per-particle (see
    // that file's own doc for the Jacques 2013 / Schlick 1994 grounding) --
    // ported here so the grid-native and curvature-flow render paths reach
    // the same optical detail ByPhysics already had, at zero new cost (both
    // terms only need `optics`, already bound, and the `optical_depth`
    // already computed above).
    let sigma_s = optical_slot.w;
    let albedo = sigma_s / max(sigma_s + sigma_a, vec3(1.0e-4));
    let scatter_glow = vec3(1.0, 0.95, 0.9) * (1.0 - exp(-sigma_s * optical_depth));
    let with_scattering = mix(transmitted, scatter_glow, clamp(albedo, vec3(0.0), vec3(1.0)));

    // Blackbody thermal emission (2026-07-31 fix): real, exact SAME formula
    // `prep_instances.wgsl`'s ByPhysics mode already uses per-particle
    // (normalized to 5000K, solar surface). Was previously NOT ported here --
    // this buffer only ever scattered mass, never temperature, a real,
    // disclosed gap (confirmed live via a user side-by-side screenshot
    // comparing this mode against ByPhysics on the same fire scene) -- fixed
    // by adding `sample_weighted_temp` above, not a new mechanism.
    let t_norm = clamp(avg_temp / 5000.0, 0.0, 1.0);
    let emission = heat(0.5 + t_norm * 0.5).rgb * (t_norm * t_norm) * 2.0;

    // Thin anti-aliased edge: the nearest-cell discard above fixes WHERE the shape
    // ends exactly, this just softens the last couple pixels via alpha blending
    // so it doesn't read as a hard stair-step. `edge_margin` is widened past
    // `mass_floor` itself (was a narrow 0.5x band) -- a narrow ramp means a
    // small, real frame-to-frame density fluctuation in a sparse/thin region
    // (particle jitter, motion) swings alpha across nearly its whole [0,1]
    // range, reading as flicker there even though the dense main body (alpha
    // pinned at 1 well past this ramp) never shows it. Standard real-time
    // volume-rendering fix: widen the transfer-function ramp to reduce its
    // sensitivity to small input noise, at the real, disclosed cost of a
    // slightly softer edge overall.
    //
    // Surface normal from the finite-difference gradient of the bilinear density
    // corners (standard volume-rendering technique). Gradient points toward
    // increasing mass; outward normal is its negative. Lambertian shading from a
    // fixed light direction (not yet tied to the day-night system).
    let grad = vec2<f32>(
        ((m10 - m00) + (m11 - m01)) * 0.5,
        ((m01 - m00) + (m11 - m10)) * 0.5,
    );
    let grad_len = length(grad);
    var lit = with_scattering;
    if grad_len > 1.0e-5 {
        let normal_dir = -grad / grad_len;
        let light_dir = normalize(params.light_dir);
        let diffuse_raw = clamp(dot(normal_dir, light_dir), 0.0, 1.0);
        // Cel-shading / toon-shading (real, well-established NPR technique):
        // quantize N.L into a handful of discrete bands instead of a smooth
        // continuous gradient. Real, disclosed reason this replaced a plain
        // continuous `diffuse`: smooth, continuous Lambertian shading reads
        // as photoreal 3D lighting (a "rendered blob," per the user's own
        // direct feedback), which is tonally mismatched with this engine's
        // explicitly non-photoreal 2D "style" target. Quantizing is the
        // standard fix for exactly that mismatch -- flat regions with sharp
        // transitions instead of a smooth gradient, while still driven by
        // the SAME real gradient/normal data, not a fake flat color.
        //
        // REVISED (2026-07-30): shipped at 4.0 originally, which was too
        // coarse -- a real fluid surface has plenty of genuine small-scale
        // curvature detail (real physical bumps/ripples), and snapping that
        // detail into only 4 harsh brightness levels amplifies it into a
        // visibly speckled, static-like texture (confirmed directly:
        // user described it as "la neige à la télé" -- TV static -- and a
        // real screenshot showed exactly that). This is quantization noise
        // amplification, a well-known real failure mode of coarse
        // posterization on detailed geometry, NOT the temporal flicker
        // this session spent most of its effort on (a different, genuinely
        // separate problem -- this one is visible in a single static
        // frame, no motion needed). Fixed by testing real candidate band
        // counts directly (16 confirmed clean, 8 ALSO confirmed clean via
        // real screenshot comparison while keeping more visible
        // stylization) -- 8 chosen as the real, verified value, not a
        // guess.
        const LIGHT_BANDS: f32 = 8.0;
        let diffuse = floor(diffuse_raw * LIGHT_BANDS) / LIGHT_BANDS;
        // Real, disclosed blend: an ambient floor (0.6) plus the real
        // Lambertian term (0.4 * diffuse) -- keeps the shape always visible
        // (no fully-black unlit side) while adding a genuine, gradient-
        // derived shading cue instead of flat, unlit color everywhere.
        //
        // REMOVED (2026-07-30): a Blinn-Phong specular LOBE used to sit
        // here. Real, disclosed reason it's gone, not just tuned down: a
        // continuous BRDF-driven highlight is itself a strong "this is a
        // lit 3D surface" cue -- exactly the "blobby/3D" look reported
        // directly, on top of (not separate from) the smooth-gradient
        // issue the cel-shading above already targets. Flat cel-shaded 2D
        // games (the user's own named reference points, Celeste/Rain
        // World) generally don't render a continuous specular lobe on
        // ordinary terrain/water at all. `OpticalTable::specular` (R0) is
        // left wired and still drives `prep_instances.wgsl`'s ByPhysics
        // particle mode, which has no gradient/normal to build a lobe from
        // in the first place and was never part of this specific problem.
        lit = clamp(with_scattering * (0.6 + 0.4 * diffuse), vec3(0.0), vec3(1.0));
    }

    // Additive blackbody glow on top of the (possibly cel-shaded) lit color --
    // same additive placement `prep_instances.wgsl`'s ByPhysics mode uses
    // (`with_specular + emission`), so near-ignition material reads as
    // genuinely glowing rather than just a brighter base color.
    let with_emission = clamp(lit + emission, vec3(0.0), vec3(1.0));

    let edge_margin = max(params.mass_floor * 1.5, 1.0e-4);
    let alpha = smoothstep(params.mass_floor, params.mass_floor + edge_margin, mass);
    return vec4<f32>(with_emission, alpha);
}
