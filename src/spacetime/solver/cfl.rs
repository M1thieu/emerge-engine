//! Adaptive-timestep (CFL) selection -- split out of `step.rs` (was ~85 of that
//! file's ~730 lines). A distinct concern from the substep pipeline itself:
//! picking how big a step is safe, not advancing the simulation by one.
//! `choose_substep_dt`/`cfl_bound`/`affine_cfl_speed_contribution` are reused
//! by the GPU solver's own CFL scan (`systems::gpu::solver::step`), which is
//! why the latter two stay `pub(crate)` and re-exported from `solver/mod.rs`.

use glam::Mat2;

use super::{MaterialRegistry, SimConfig};
use crate::particle::Particles;
use crate::rod::{Rod, rod_cfl_dt};

// choose_substep_dt: picks the largest CFL-safe dt ≤ max_dt.
// Called inside step()'s substep loop — max_dt is the remaining frame time.
// pub(crate) so the GPU solver can reuse this without duplicating CFL logic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn choose_substep_dt(
    config: &SimConfig,
    particles: &Particles,
    active_count: usize,
    materials: &MaterialRegistry,
    rods: &[Rod],
    max_dt: f32,
    granular_fluidity_dt_bound: Option<f32>,
    thermal_dt_bound: Option<f32>,
) -> f32 {
    if !config.adaptive_timestep {
        return max_dt.min(config.dt);
    }
    // Single pass for both velocity CFL and material timestep bound.
    let mut max_speed = 0.0f32;
    let mut min_mat_dt = max_dt;
    for i in 0..active_count {
        let mut s = particles.v[i].length();
        if config.cfl_include_affine_speed {
            s += affine_cfl_speed_contribution(
                &particles.velocity_gradient[i],
                config.grid_cell_size,
            );
        }
        max_speed = max_speed.max(s);
        let mdt = materials.get(particles.material_id[i]).timestep_bound(
            particles.density[i],
            particles.hardening_scale[i],
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        if mdt.is_finite() && mdt > 0.0 {
            min_mat_dt = min_mat_dt.min(mdt);
        }
    }
    // Rods aren't scanned by the particle loop above (separate SoA) -- fold
    // in their own CFL bound the same way a stiff material would clamp
    // min_mat_dt, so a rod going unstable can never silently escape the
    // adaptive substep logic (the exact bug class this whole rod effort
    // started from: something CFL never knew about). Skipped for sleeping
    // rods -- this is the real cost fix for many simultaneous rods (a grass
    // field): `rod_cfl_dt` is a per-point Gershgorin sum over every stiffness
    // term touching it, paid EVERY substep for EVERY rod before this; a
    // settled rod contributing nothing to the min anyway (its own dt bound
    // stays constant while frozen) has no reason to keep paying for it.
    for rod in rods {
        // An implicit-integration rod is advanced ONCE per `step()` call, entirely
        // outside this substep loop (see `Simulation::step`'s own implicit-rod
        // pass) -- its stability no longer depends on this shared adaptive dt at
        // all (the whole point of implicit integration: unconditionally stable
        // regardless of the rod's own stiffness), so it correctly contributes
        // nothing here, same spirit as a sleeping rod contributing nothing while
        // frozen.
        if rod.sleeping || rod.use_implicit_integration {
            continue;
        }
        let rod_dt = rod_cfl_dt(&rod.points, &rod.material, config.rod_cfl_coefficient);
        if rod_dt.is_finite() && rod_dt > 0.0 {
            min_mat_dt = min_mat_dt.min(rod_dt);
        }
    }
    // Nonlocal Granular Fluidity's own real, quoted Von Neumann stability
    // bound (`GranularFluidityConfig::stability_dt`, Haeri & Skonieczny
    // 2022). `None` (every scene without a configured `GranularFluidityField`)
    // leaves this exactly as it always was.
    if let Some(bound) = granular_fluidity_dt_bound
        && bound.is_finite()
        && bound > 0.0
    {
        min_mat_dt = min_mat_dt.min(bound);
    }
    // `ThermalDiffusion`'s own explicit-diffusion stability bound
    // (`ThermalConfig::stability_dt`) -- normally many orders of magnitude
    // larger than MPM's own CFL (real thermal diffusivity is tiny), so this
    // is a no-op for any correctly-configured scene. It only bites on a
    // real, already-reproduced misconfiguration (passing `grid_cell_size`
    // instead of `dx_meters`, see that field's own doc) -- folding it in
    // turns that from a silent runaway into an automatically clamped,
    // still-correct substep, same precedent as NGF above.
    if let Some(bound) = thermal_dt_bound
        && bound.is_finite()
        && bound > 0.0
    {
        min_mat_dt = min_mat_dt.min(bound);
    }
    cfl_bound(config, max_speed, min_mat_dt, max_dt)
}

/// Shared CFL formula: clamps dt to advection + material bounds.
/// Called by both SoA and AoS scan paths after computing their respective max values.
pub(crate) fn cfl_bound(config: &SimConfig, max_speed: f32, min_mat_dt: f32, max_dt: f32) -> f32 {
    let mut dt = max_dt;
    if max_speed > f32::EPSILON {
        dt = dt.min(config.cfl_coefficient * config.grid_cell_size / max_speed);
    }
    dt = dt.min(min_mat_dt);
    dt.clamp(config.min_dt.min(max_dt), max_dt)
}

pub(crate) fn affine_cfl_speed_contribution(c: &Mat2, cell_width: f32) -> f32 {
    // The APIC affine matrix C encodes the local velocity gradient.
    // The farthest point in the quadratic B-spline 3×3 stencil is at 1.5 cells per axis,
    // so its corner distance is 1.5*√2 cells — the effective maximum affine speed contribution.
    const STENCIL_CORNER_DISTANCE: f32 = 1.5 * std::f32::consts::SQRT_2;
    let grad_norm = (c.x_axis.length_squared() + c.y_axis.length_squared()).sqrt();
    grad_norm * STENCIL_CORNER_DISTANCE * cell_width
}
