//! Particle-mutation API on `GpuSimulation`: material reassignment and
//! impulses.
//!
//! Split out of `gpu/solver/mod.rs` -- self-contained particle mutation,
//! not raw wgpu command encoding (impulses queue into `pending_impulses`,
//! applied by a dedicated compute pass the next `step_frame` call — see
//! `step.rs`). Mirrors the CPU `solver::particles` split.

use crate::particle::Particle;

use super::super::step_params::{GpuImpulseEntry, MAX_GPU_IMPULSES};
use super::GpuSimulation;

impl GpuSimulation {
    /// Reassign material for all particles matching `predicate`. Marks dirty so GPU
    /// sees the change on the next `step_frame` call.
    pub fn phase_transition<F>(&mut self, predicate: F, new_material_id: u32)
    where
        F: Fn(&Particle) -> bool,
    {
        assert!(
            self.registry.is_registered(new_material_id),
            "phase_transition: material_id {new_material_id} is not registered — \
             call solver.set_material({new_material_id}, ...) first"
        );
        for p in self.particles.iter_mut() {
            if predicate(p) {
                p.material_id = new_material_id;
            }
        }
        self.layout_dirty = true; // material_id changed — sort order may differ
    }

    /// Add `force` to every particle within `radius` cells of `center`, scaled by proximity.
    /// Applied on the GPU at the start of the next step_frame — reads LIVE GPU positions,
    /// avoiding any stale-CPU-mirror artifacts. No CPU particle scan.
    pub fn apply_impulse(&mut self, center: glam::Vec2, radius: f32, force: glam::Vec2) {
        if self.pending_impulses.len() < MAX_GPU_IMPULSES {
            self.pending_impulses.push(GpuImpulseEntry {
                center: center.to_array(),
                radius,
                strength: 0.0,
                force: force.to_array(),
                mode: 1,
                _pad: 0,
            });
        } else {
            eprintln!(
                "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) — impulse dropped"
            );
        }
    }

    /// Push every particle within `radius` cells outward from `center`.
    /// Applied on the GPU at the start of the next step_frame — reads LIVE GPU positions.
    /// `strength` may be negative to pull. No CPU particle scan.
    pub fn apply_radial_impulse(&mut self, center: glam::Vec2, radius: f32, strength: f32) {
        if self.pending_impulses.len() < MAX_GPU_IMPULSES {
            self.pending_impulses.push(GpuImpulseEntry {
                center: center.to_array(),
                radius,
                strength,
                force: [0.0; 2],
                mode: 0,
                _pad: 0,
            });
        } else {
            eprintln!(
                "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) — impulse dropped"
            );
        }
    }
}
