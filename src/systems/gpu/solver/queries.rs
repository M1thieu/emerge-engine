//! Read-only diagnostics and spatial-query API on `GpuSimulation`.
//!
//! Split out of `gpu/solver/mod.rs` -- everything here reads the CPU particle
//! mirror + spatial hash, never encodes a wgpu command. Lower-risk slice than
//! `step_frame`/`encode_substep`, which touch live device/buffer state.
//! Mirrors the CPU `solver::queries` split.

use crate::particle::Particle;

use super::GpuSimulation;

impl GpuSimulation {
    /// Physics snapshot from the CPU particle mirror (one frame behind GPU when strided).
    /// Grid-side fields (mass error, momentum error, active cells) are zero — GPU grid is
    /// not readable on CPU. All particle-side fields are exact.
    pub fn diagnostics_snapshot(&self) -> crate::diagnostics::SimSnapshot {
        crate::diagnostics::collect_snapshot_particles_only(
            self.frame_index,
            &self.particles,
            &self.config,
            self.last_sub_dt,
            self.last_substeps,
        )
    }

    /// Iterate over (index, &Particle) pairs within `radius` grid-cells of `center`.
    /// Reads the internal CPU particle mirror — one frame behind GPU when strided.
    /// O(candidates) via the internal spatial hash, not O(N) -- see `spatial_hash`
    /// field's own doc for why this matters at real scale (many creatures/queries
    /// per frame against a large terrain+water buffer).
    pub fn particles_near(
        &self,
        center: glam::Vec2,
        radius: f32,
    ) -> impl Iterator<Item = (usize, &Particle)> {
        let r2 = radius * radius;
        self.spatial_hash
            .query(center, radius)
            .filter_map(move |i| {
                let p = &self.particles[i];
                ((p.x - center).length_squared() <= r2).then_some((i, p))
            })
    }

    /// Count particles of `material_id` within `radius` grid-cells of `center`.
    /// O(candidates) via the internal spatial hash, not O(N).
    pub fn count_near(&self, center: glam::Vec2, radius: f32, material_id: u32) -> usize {
        let r2 = radius * radius;
        self.spatial_hash
            .query(center, radius)
            .filter(|&i| {
                let p = &self.particles[i];
                p.material_id == material_id && (p.x - center).length_squared() <= r2
            })
            .count()
    }

    /// Indices of the `k` particles nearest to `center`, sorted by distance
    /// ascending -- see `Simulation::particles_knn` (CPU, `src/solver/mod.rs`)
    /// for the full rationale (Ballerini et al. 2008, PNAS: real starling
    /// flocks use a topological ~6-7-nearest-neighbor rule, not a fixed
    /// radius) and exactness argument. Identical algorithm, mirrored here
    /// because the GPU backend keeps its own CPU-side spatial hash mirror.
    pub fn particles_knn(&self, center: glam::Vec2, k: usize) -> Vec<usize> {
        if k == 0 || self.particles.is_empty() {
            return Vec::new();
        }
        let domain_diag =
            self.config.grid_res as f32 * self.config.grid_cell_size * std::f32::consts::SQRT_2;
        let mut radius = self.config.grid_cell_size * (k as f32).sqrt().max(1.0);
        let mut candidates: Vec<(usize, f32)>;
        loop {
            let r2 = radius * radius;
            candidates = self
                .spatial_hash
                .query(center, radius)
                .map(|i| (i, (self.particles[i].x - center).length_squared()))
                .filter(|&(_, d2)| d2 <= r2)
                .collect();
            if candidates.len() >= k || radius >= domain_diag {
                break;
            }
            radius *= 2.0;
        }
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.truncate(k);
        candidates.into_iter().map(|(i, _)| i).collect()
    }

    /// Center of mass for particles in `range`. O(range.len()). GPU has no tag_index
    /// like CPU `Simulation::group_centroid` -- `range` (from `spawn_region`'s return)
    /// is the stable group identity here instead of a `u32` tag.
    pub fn group_centroid(&self, range: std::ops::Range<usize>) -> glam::Vec2 {
        let particles = &self.particles[range.clone()];
        if particles.is_empty() {
            return glam::Vec2::ZERO;
        }
        let sum: glam::Vec2 = particles.iter().map(|p| p.x).sum();
        sum / range.len() as f32
    }

    /// Aggregate state for all particles of the given material.
    pub fn material_state(&self, material_id: u32) -> crate::solver::query::BodyState {
        crate::solver::query::body_state_of_slice(&self.particles, material_id)
    }

    /// Aggregate state for all particles within `radius` grid-cells of `center`.
    pub fn region_state(&self, center: glam::Vec2, radius: f32) -> crate::solver::query::BodyState {
        crate::solver::query::region_body_state_of_slice(&self.particles, center, radius)
    }
}
