//! Standalone rod time integration -- no grid, no `Simulation` (Phase 0/1).
//! Phase 2's grid-coupled path reuses `forces::compute_internal_forces`
//! directly (see `coupling.rs`) rather than this integrator.

use glam::Vec2;

use super::{RodMaterial, RodPoints, RodRestState, compute_internal_forces};

/// One explicit (symplectic Euler) rod substep. Pinned points held at
/// `v=0`/position fixed -- identical semantics to `Particle::pinned`'s own
/// G2P handling (forces v=0 instead of gathering, position left completely
/// untouched, mass/forces still computed normally so the anchor is real).
pub fn step_rod(
    rod: &mut RodPoints,
    material: &RodMaterial,
    gravity: Vec2,
    wind_velocity: Vec2,
    wind_drag_coeff: f32,
    dx_meters: f32,
    dt: f32,
) {
    let n = rod.len();
    if n == 0 {
        return;
    }

    let internal = compute_internal_forces(
        &rod.x,
        &rod.v,
        RodRestState {
            rest_edge_length: &rod.rest_edge_length,
            rest_curvature: &rod.rest_curvature,
            ea: &rod.ea,
            ei: &rod.ei,
        },
        material,
        dx_meters,
    );

    for (i, internal_force) in internal.iter().enumerate() {
        if rod.pinned[i] != 0 {
            rod.v[i] = Vec2::ZERO;
            continue;
        }
        // Internal force (Newtons) -> grid acceleration: a = F/(m*dx_meters),
        // mirrors gravity_to_grid's own g_grid = g_SI/dx_meters (mass handled
        // explicitly here since force, unlike gravity, isn't already
        // per-unit-mass).
        let a_internal = *internal_force / (rod.mass[i] * dx_meters);
        // Wind drag: a = k*(target - v), same law LinearDragField implements.
        let a_wind = wind_drag_coeff * (wind_velocity - rod.v[i]);
        let a = gravity + a_internal + a_wind;
        rod.v[i] += a * dt;
    }
    // Symplectic Euler: advance position with the JUST-updated velocity.
    for i in 0..n {
        if rod.pinned[i] != 0 {
            continue;
        }
        rod.x[i] += rod.v[i] * dt;
    }
}

/// Longest step the rod's own explicit integrator (`step_rod`: symplectic
/// Euler, damping from the step's starting velocity) stays stable at, per
/// point, then the smallest; `fraction` of it is returned. The fraction is
/// `SimConfig::material_cfl_coefficient`'s definition.
///
/// Derived from the scheme. A mode `x'' = -omega^2 x - b x'` stepped this
/// way is stable exactly while `dt <= 4 / (b + sqrt(b^2 + 4 omega^2))`
/// (the Jury conditions on its 2x2 update; `2 / omega` without damping).
/// `omega^2` and `b` are bounded by Gershgorin row sums of the linearised
/// stiffness and damping over the point's mass (`point_stability_sums`).
///
/// It replaces a per-point sum that counted each edge's axial stiffness once
/// and each bending vertex once with weight one, 2 and 16/3 times too little,
/// with an empirical 0.4 in front: a cantilever at 1 cm cells under real
/// gravity blew up at step 32 at that step (`tests/subsystem_time_steps.rs`,
/// `probe_cantilever_coupling_cut`).
pub fn rod_cfl_dt(rod: &RodPoints, material: &RodMaterial, fraction: f32) -> f32 {
    let mut min_dt = f32::INFINITY;
    for i in 0..rod.len() {
        if rod.pinned[i] != 0 {
            continue;
        }
        let m = rod.mass[i].max(1.0e-9);
        let (k, c) = point_stability_sums(rod, material, i);
        let (omega_sq, b) = (k / m, c / m);
        if omega_sq > 0.0 || b > 0.0 {
            min_dt = min_dt.min(4.0 / (b + (b * b + 4.0 * omega_sq).sqrt()));
        }
    }
    fraction * min_dt
}

/// Point `i`'s Gershgorin row sums of the linearised stiffness `K` (N/m) and
/// damping `C` (N s/m), about the straight rest state, shared by
/// `rod_cfl_dt` and `apply_mass_scaling_for_target_dt`.
///
/// Axial: each edge is a spring `EA / l0` and a dashpot `c_a` along it, a
/// 2x2 block `[[1, -1], [-1, 1]]`, so each adjacent edge adds twice its
/// value to the row. Bending (`forces::compute_internal_forces`): vertex
/// `k` stores `EI / (2 L_v) kappa^2`, and for a straight rod
/// `kappa = (w2 - w1) / l_n - (w1 - w0) / l_p` in the lateral displacements,
/// so its gradient is `g = (1/l_p, -(1/l_p + 1/l_n), 1/l_n)`, its stiffness
/// `(EI / L_v) g g^T` and its damping `c_b g g^T`; point `i` at position `j`
/// in the vertex adds `|g_j| * sum|g|` times each. An interior point on
/// equal edges gets `4 EA / l0` and `16 EI / l0^3`.
fn point_stability_sums(rod: &RodPoints, material: &RodMaterial, i: usize) -> (f32, f32) {
    let n = rod.len();
    let ea_at = |k: usize| {
        if rod.ea.is_empty() {
            material.ea
        } else {
            rod.ea[k]
        }
    };
    let ei_at = |k: usize| {
        if rod.ei.is_empty() {
            material.ei
        } else {
            rod.ei[k]
        }
    };
    let (mut k_sum, mut c_sum) = (0.0f32, 0.0f32);
    for edge in [i.checked_sub(1), (i + 1 < n).then_some(i)]
        .into_iter()
        .flatten()
    {
        let l0 = rod.rest_edge_length[edge].max(1.0e-9);
        k_sum += 2.0 * ea_at(edge) / l0;
        c_sum += 2.0 * material.axial_damping;
    }
    for vertex in [i.checked_sub(2), i.checked_sub(1), Some(i)]
        .into_iter()
        .flatten()
    {
        if vertex + 2 >= n {
            continue;
        }
        let l_p = rod.rest_edge_length[vertex].max(1.0e-9);
        let l_n = rod.rest_edge_length[vertex + 1].max(1.0e-9);
        let voronoi_length = 0.5 * (l_p + l_n);
        let g = [1.0 / l_p, 1.0 / l_p + 1.0 / l_n, 1.0 / l_n];
        let row = g[i - vertex] * (g[0] + g[1] + g[2]);
        k_sum += ei_at(vertex) / voronoi_length * row;
        c_sum += material.bending_damping * row;
    }
    (k_sum, c_sum)
}

/// Mass scaling (a standard explicit-dynamics technique, e.g. LS-DYNA's own
/// `*CONTROL_TIMESTEP` option): raise a point's own inertia just enough
/// that `rod_cfl_dt` reaches `target_dt` at `fraction`. Only ever raises
/// mass, and changes real dynamics (gravity, wind, push response), so it is
/// an opt-in call, never applied silently.
///
/// From `rod_cfl_dt`: with `tau = target_dt / fraction`, `omega^2 = K / m`
/// and `b = C / m`, `4 / (b + sqrt(b^2 + 4 omega^2)) >= tau` exactly when
/// `m >= (K tau^2 + 2 C tau) / 4`.
pub fn apply_mass_scaling_for_target_dt(
    rod: &mut RodPoints,
    material: &RodMaterial,
    fraction: f32,
    target_dt: f32,
) {
    let tau = target_dt / fraction;
    for i in 0..rod.len() {
        if rod.pinned[i] != 0 {
            continue;
        }
        let (k, c) = point_stability_sums(rod, material, i);
        let m_required = (k * tau * tau + 2.0 * c * tau) / 4.0;
        rod.mass[i] = rod.mass[i].max(m_required);
    }
}
