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

struct GridVolumeParams {
    sx: f32,
    tx: f32,
    sy: f32,
    ty: f32,
    grid_res: u32,
    mass_floor: f32,
    material_mass_enabled: u32,
    _pad1: f32,
}

struct OpticalTable {
    slots: array<vec4<f32>, 16>,
    specular: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<storage, read> grid_int: array<u32>;
@group(0) @binding(1) var<uniform> params: GridVolumeParams;
@group(0) @binding(2) var<uniform> optics: OpticalTable;
@group(0) @binding(3) var<storage, read> material_mass: array<f32>;

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

// Majority-mass-wins dominant material for cell (cx, cy). Returns 0 (and is never
// called) when material tracking isn't attached -- see fs_main's gate.
fn dominant_material(cx: i32, cy: i32) -> u32 {
    let idx = u32(cy) * params.grid_res + u32(cx);
    let base = idx * MAX_RENDER_MATERIAL_SLOTS;
    var best_slot: u32 = 0u;
    var best_mass: f32 = -1.0;
    for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
        let m = material_mass[base + s];
        if m > best_mass {
            best_mass = m;
            best_slot = s;
        }
    }
    return best_slot;
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

    // Gate visibility on the NEAREST cell's mass, not the bilinear-blended value —
    // the blend is nonzero up to a full cell beyond the nearest occupied cell, which
    // would overshoot true particle extent no matter how high mass_floor is raised.
    // Bilinear mass is still used for interior shading below.
    let nx = i32(round(grid_pos.x - 0.5));
    let ny = i32(round(grid_pos.y - 0.5));
    let nx_c = clamp(nx, 0, i32(params.grid_res) - 1);
    let ny_c = clamp(ny, 0, i32(params.grid_res) - 1);
    if sample_mass(nx_c, ny_c) < params.mass_floor {
        discard;
    }

    // Nearest cell's dominant material (majority mass wins) -- see this file's own
    // top doc comment for why this is nearest-cell, not blended across the 4
    // bilinear neighbors. Falls back to slot 0 when material tracking isn't attached.
    var slot: u32 = 0u;
    if params.material_mass_enabled != 0u {
        slot = dominant_material(nx_c, ny_c);
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
    const EDGE_COLOR_REFERENCE_DEPTH: f32 = 0.5;
    let sigma_a = optics.slots[slot].rgb;
    let optical_depth = max(clamp(mass, 0.0, 4.0), EDGE_COLOR_REFERENCE_DEPTH);
    let transmitted = exp(-sigma_a * optical_depth);

    // Thin anti-aliased edge: the nearest-cell discard above fixes WHERE the shape
    // ends exactly, this just softens the last couple pixels via alpha blending
    // so it doesn't read as a hard stair-step. `edge_margin` is deliberately small
    // relative to `mass_floor`.
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
    var lit = transmitted;
    if grad_len > 1.0e-5 {
        let normal_dir = -grad / grad_len;
        let light_dir = normalize(vec2<f32>(-0.5, 0.7));
        let diffuse = clamp(dot(normal_dir, light_dir), 0.0, 1.0);
        // Real, disclosed blend: an ambient floor (0.6) plus the real
        // Lambertian term (0.4 * diffuse) -- keeps the shape always visible
        // (no fully-black unlit side) while adding a genuine, gradient-
        // derived shading cue instead of flat, unlit color everywhere.
        lit = transmitted * (0.6 + 0.4 * diffuse);
    }

    let edge_margin = max(params.mass_floor * 0.5, 1.0e-4);
    let alpha = smoothstep(params.mass_floor, params.mass_floor + edge_margin, mass);
    return vec4<f32>(lit, alpha);
}
