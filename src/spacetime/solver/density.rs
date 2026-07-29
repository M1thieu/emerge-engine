use glam::{IVec2, Vec2};

use crate::materials::MaterialRegistry;
use crate::transfer::scatter_particle_mass;
use crate::{grid::Grid, grid::kernel::quadratic_weights, particle::Particles};

/// Fluid free-surface rarefaction ceiling, mirroring the GPU path's own
/// `FLUID_J_MAX` clamp exactly (`systems/gpu/shaders/particles_update.wgsl`,
/// `mat.volume_ratio_max`, default 2.0 when a material doesn't set it — same
/// `> 1.0` fallback test as the shader's `select(2.0, mat.volume_ratio_max,
/// mat.volume_ratio_max > 1.0)`).
///
/// Real bug this fixes (found 2026-07-26 investigating Martin & Moyce 1952
/// dam-break validation): this kernel-mass-based density estimate has no
/// upper bound on the resulting `volume = mass/density`. As a particle's
/// local mass support genuinely thins (bulk fluid surges away, e.g. near a
/// dam-break's back wall), `density` correctly shrinks toward zero and
/// `volume` explodes without limit (measured: 18x in one substep sequence,
/// 0.20 -> 3.69) — this inflated volume is the P2G quadrature weight, so even
/// bounded stress dumps disproportionate momentum into the grid (measured:
/// particle speed reaching 130+ cells/s vs a real target of ~4). The GPU path
/// already guards this exact failure via `volume_ratio_max` on `det(F)`; this
/// CPU path (the one every fluid substep actually uses, since fluid density
/// is estimated from grid mass, not tracked via F — see
/// `NewtonianFluidMaterial::update_particle`'s F-reset, which does NOT write
/// `particles.volume`/`density`) never had the equivalent bound. Only
/// engages on the per-substep recompute (`write_initial=false`) — spawn-time
/// calls (`write_initial=true`) skip it, since `initial_volume` is what's
/// being established there, not yet a reference to clamp against.
///
/// `materials: None` (the only callers with no registry in scope are all
/// `write_initial=true` spawn-time calls, which never reach this function —
/// see call sites) falls back to the GPU shader's own hardcoded default (2.0).
///
/// HONEST STATUS (2026-07-26): this closes the specific volume/quadrature-
/// weight blowup it documents above (verified: volume plateaus at ~1.8-1.9x
/// initial instead of ballooning to 18x). It does NOT by itself stabilize the
/// dam-break scenario that surfaced it — with this fix alone, particle speed
/// still runs away (measured up to 173 cells/s) via a SEPARATE mechanism:
/// `velocity_gradient` (the APIC C matrix) grows unbounded at the same
/// free-surface corner / wall region (measured C-norm climbing 1 -> 706 over
/// 14 samples), most likely an explicit-viscosity feedback loop (deviatoric
/// stress depends on C, feeds P2G force, updates grid velocity, updates next
/// substep's C) not fully covered by the existing global viscous CFL bound
/// at this local, severely-thinned density. Real, separate, NOT yet root-
/// caused or fixed — kept here as a real, verified, standalone improvement
/// regardless of that second open issue.
fn clamp_rarefied_volume(
    materials: Option<&MaterialRegistry>,
    material_id: u32,
    raw_volume: f32,
    initial_volume: f32,
) -> f32 {
    let declared = materials
        .map(|m| m.get(material_id).params().volume_ratio_max)
        .unwrap_or(0.0);
    let ratio_max = if declared > 1.0 { declared } else { 2.0 };
    raw_volume.min(initial_volume.max(f32::EPSILON) * ratio_max)
}

/// Export the mass-density field as a flat `grid_res × grid_res` buffer.
///
/// Each cell value is Σ(w_ij · mass_j) — the same mass accumulation used internally
/// by P2G. Values are NOT normalized by cell volume; callers can divide by
/// `grid_cell_size²` if physical units are needed.
///
/// # Use case — LP metaball surface rendering
/// Call once per render frame (not per substep) after `solver.step()`.
/// Upload the result to a `wgpu::Texture` and threshold in a fragment shader
/// to get a particle density surface.
///
/// Layout: column-major, index = x * grid_res + y — matches mechanics grid.
pub fn compute_density_grid(particles: &Particles, grid_res: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; grid_res * grid_res];
    let res = grid_res as i32;
    for i in 0..particles.len() {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let w = quadratic_weights(x);
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                    continue;
                }
                let weight = w.wx[gx as usize] * w.wy[gy as usize];
                buf[(cell.x * res + cell.y) as usize] += weight * mass;
            }
        }
    }
    buf
}

/// Compute density and volume for `count` particles (scatter + gather).
/// Pass `write_initial = true` at spawn time to also set `initial_volume`.
pub fn estimate_particle_volumes(
    particles: &mut Particles,
    grid: &mut Grid,
    materials: Option<&MaterialRegistry>,
    count: usize,
    write_initial: bool,
) {
    grid.clear();
    scatter_particle_mass(particles, grid, count);

    for i in 0..count {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        let mut density = 0.0;

        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                density += grid.mass_at(cell_pos) * weight;
            }
        }

        if density > f32::EPSILON {
            let raw_volume = mass / density;
            if write_initial {
                particles.density[i] = density;
                particles.volume[i] = raw_volume;
                particles.initial_volume[i] = raw_volume;
            } else {
                let volume = clamp_rarefied_volume(
                    materials,
                    particles.material_id[i],
                    raw_volume,
                    particles.initial_volume[i],
                );
                particles.volume[i] = volume;
                particles.density[i] = mass / volume;
            }
        }
    }
}

/// Compute density and volume only for particles in `[new_start..active_count]`.
///
/// Scatters only particles whose positions fall within the AABB of the new group
/// expanded by 3 grid cells (the quadratic B-spline influence radius). All other
/// active particles are ignored — their density contribution to the new group is zero.
///
/// O(active_count) scan but O(local × stencil) grid work — fast for sparse spawns.
pub fn estimate_particle_volumes_local(
    particles: &mut Particles,
    grid: &mut Grid,
    materials: Option<&MaterialRegistry>,
    active_count: usize,
    new_start: usize,
    write_initial: bool,
) {
    if new_start >= active_count {
        return;
    }

    // AABB of new particles in grid coords.
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for i in new_start..active_count {
        lo = lo.min(particles.x[i]);
        hi = hi.max(particles.x[i]);
    }
    // Expand by 3 cells: quadratic stencil reaches 1.5 cells per side,
    // and we need to capture particles that contribute mass to those cells.
    const MARGIN: f32 = 3.0;
    lo -= Vec2::splat(MARGIN);
    hi += Vec2::splat(MARGIN);

    grid.clear();
    for i in 0..active_count {
        let x = particles.x[i];
        if x.x < lo.x || x.y < lo.y || x.x > hi.x || x.y > hi.y {
            continue;
        }
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                grid.add_mass_momentum(cell_pos, weight * mass, Vec2::ZERO);
            }
        }
    }

    for i in new_start..active_count {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        let mut density = 0.0;

        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                density += grid.mass_at(cell_pos) * weight;
            }
        }

        if density > f32::EPSILON {
            let raw_volume = mass / density;
            if write_initial {
                particles.density[i] = density;
                particles.volume[i] = raw_volume;
                particles.initial_volume[i] = raw_volume;
            } else {
                let volume = clamp_rarefied_volume(
                    materials,
                    particles.material_id[i],
                    raw_volume,
                    particles.initial_volume[i],
                );
                particles.volume[i] = volume;
                particles.density[i] = mass / volume;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::clamp_rarefied_volume;
    use crate::materials::NewtonianFluidMaterial;
    use crate::materials::{MaterialRegistry, NeoHookeanMaterial};

    #[test]
    fn volume_within_ratio_max_passes_through_unchanged() {
        let registry = MaterialRegistry::with_default(Box::new(
            NewtonianFluidMaterial::new(1.0, 1.0e-3, 10.0, 7.0), // volume_ratio_max = 2.0
        ));
        let initial_volume = 0.2_f32;
        let raw_volume = initial_volume * 1.5; // inside [_, 2.0x] -- must not clamp
        let clamped = clamp_rarefied_volume(Some(&registry), 0, raw_volume, initial_volume);
        assert!(
            (clamped - raw_volume).abs() < 1.0e-6,
            "volume within the real ratio_max must pass through unchanged, got {clamped} \
             from raw {raw_volume}"
        );
    }

    #[test]
    fn volume_beyond_ratio_max_clamps_to_the_boundary() {
        let registry = MaterialRegistry::with_default(Box::new(NewtonianFluidMaterial::new(
            1.0, 1.0e-3, 10.0, 7.0,
        )));
        let initial_volume = 0.2_f32;
        let raw_volume = initial_volume * 18.0; // reproduces the real measured blowup ratio
        let clamped = clamp_rarefied_volume(Some(&registry), 0, raw_volume, initial_volume);
        let expected = initial_volume * 2.0; // this material's real, declared volume_ratio_max
        assert!(
            (clamped - expected).abs() < 1.0e-6,
            "volume beyond ratio_max must clamp exactly to initial_volume*volume_ratio_max \
             ({expected}), got {clamped}"
        );
    }

    #[test]
    fn material_without_declared_ratio_max_falls_back_to_gpu_default_of_2x() {
        // NeoHookeanMaterial never sets volume_ratio_max (MaterialParams::default() = 0.0) --
        // must fall back to the same 2.0 the GPU shader's own `select(2.0, ..., > 1.0)` uses.
        let registry = MaterialRegistry::with_default(Box::new(
            NeoHookeanMaterial::from_young_modulus(1.0e4, 0.3),
        ));
        let initial_volume = 1.0_f32;
        let raw_volume = 5.0_f32;
        let clamped = clamp_rarefied_volume(Some(&registry), 0, raw_volume, initial_volume);
        assert!(
            (clamped - 2.0).abs() < 1.0e-6,
            "material with no declared volume_ratio_max must fall back to 2.0x, got {clamped}"
        );
    }

    #[test]
    fn no_registry_falls_back_to_gpu_default_of_2x() {
        let initial_volume = 1.0_f32;
        let raw_volume = 5.0_f32;
        let clamped = clamp_rarefied_volume(None, 0, raw_volume, initial_volume);
        assert!(
            (clamped - 2.0).abs() < 1.0e-6,
            "no registry in scope must fall back to 2.0x, got {clamped}"
        );
    }
}
