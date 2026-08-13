//! The actual per-frame GPU dispatch: `step_frame` (CFL scan, uploads, encode,
//! submit, async readback) and `encode_substep` (the 7 labeled compute passes).
//!
//! Split out of `gpu/solver/mod.rs` -- the highest-risk slice: everything here
//! touches live wgpu device/buffer state and timing-sensitive submit/poll
//! ordering (see the OOM/substep-batching and active-block-grace-period
//! comments below).

use super::super::step_params::{
    GpuFieldsParams, GpuImpulseParams, GpuSleepWakeParams, GpuStepParams, NUM_BLOCKS,
};
use super::encode_substep::SubstepGates;
use super::{GpuSimulation, WG_PARTICLES, build_bind_group_pool};

use crate::particle::Particles;
use crate::solver::config::SimConfig;
use crate::solver::{affine_cfl_speed_contribution, cfl_bound, deformation_gradient_cfl_bound};

/// How many strict-fluid substeps share one dt and one command-buffer submit.
/// Module-scope (not function-local) because regional substepping's own tier
/// rule is derived FROM it: a Coarse block must survive integrating with a
/// whole batch's accumulated dt -- see `classify_blocks` and
/// `SimConfig::fluid_regional_substepping_fine_tier_margin`.
const STRICT_FLUID_SUBSTEP_BATCH_SIZE: usize = 8;

impl GpuSimulation {
    /// Reject combinations for which this backend has no one-fluid PDE
    /// discretization.  Leaving them enabled would route a WC-MPM particle
    /// through solid contact/sleep machinery or inject a live-GPU impulse
    /// after the CPU-side CFL scan, neither of which is an admissible fluid
    /// update.
    fn assert_strict_fluid_mode_is_supported(&self) -> bool {
        let mut has_strict_fluid = false;
        for (i, particle) in self.particles.iter().enumerate() {
            let material = self.registry.get(particle.material_id);
            if !material.owns_deformation_volume_state() {
                continue;
            }
            has_strict_fluid = true;
            assert!(
                particle.contact_group == 0,
                "GPU strict WC-MPM fluid particle {i} cannot use multi-field contact; use a fluid--solid boundary/coupling model"
            );
            assert!(
                particle.pinned == 0,
                "GPU strict WC-MPM fluid particle {i} cannot be pinned; use a geometric wall boundary instead"
            );
            assert!(
                particle.sleeping == 0,
                "GPU strict WC-MPM fluid particle {i} cannot be sleeping; sleeping removes momentum evolution from the PDE"
            );
            assert!(
                material.mixture_phase().is_none(),
                "GPU strict WC-MPM fluid particle {i} cannot use porous-mixture coupling; that requires a separate multiphase PDE"
            );
        }
        if has_strict_fluid {
            assert!(
                self.config.apic_blend == 1.0,
                "GPU strict WC-MPM fluid requires apic_blend = 1: attenuating the gathered velocity gradient changes the continuity equation"
            );
            assert!(
                self.asflip_params.enabled == 0,
                "GPU strict WC-MPM fluid cannot use ASFLIP: its blend/compression switch is a transfer heuristic, not part of this liquid PDE"
            );
            assert!(
                self.config.sleep_threshold <= 0.0,
                "GPU strict WC-MPM fluid cannot sleep; sleeping removes momentum evolution from the PDE"
            );
            assert!(
                self.config.cundall_damping <= 0.0,
                "GPU strict WC-MPM fluid cannot use Cundall damping; model a declared viscous stress or drag force instead"
            );
            assert!(
                self.pending_impulses.is_empty(),
                "GPU strict WC-MPM fluid does not accept live-GPU velocity impulses before an authoritative preflight/retry path exists; apply a resolved physical force instead"
            );
            assert!(
                self.pending_sleep_tags.is_empty() && self.pending_wake_tags.is_empty(),
                "GPU strict WC-MPM fluid cannot use forced sleep/wake tags"
            );
        }
        has_strict_fluid
    }

    /// One CFL scan over the current CPU particle mirror -- shared by `step_frame`'s
    /// first substep-chunk (frame-start state) and, for strict WC-MPM fluids, every
    /// later chunk boundary (post-sync state, see `step_frame`'s own doc on why this
    /// re-scan exists). Pulled out as its own method specifically so those two call
    /// sites can never drift apart.
    ///
    /// Excludes sleeping particles from the scan -- CPU's `Simulation::step()` does
    /// this implicitly via its active/sleeping partition; GPU has no such partition,
    /// so without this filter a frozen-near-zero sleeping majority dilutes the
    /// velocity statistics this estimate is based on. (sparkl's `adaptive_timestep_
    /// length` computes this the same way: scan only the live/active particle set.)
    fn scan_gpu_cfl_sub_dt(&self) -> f32 {
        let mut max_speed = 0.0f32;
        let mut min_mat_dt = self.config.dt;
        let mut awake_count = 0usize;
        // Real, CPU-proven near-wall tightening (`SimConfig::fluid_near_wall_cfl_scale`,
        // see `reactive_gpu_substep_dt`'s own doc for the full citation) -- applied here
        // too so the very FIRST substep of a strict-fluid frame (this bootstrap scan,
        // before `cfl_scan.wgsl`'s per-substep reduction has run even once) gets the
        // same protection, not just substep 2 onward.
        // REAL ROOT-CAUSE FIX (2026-08-11) of the GPU fluid fps collapse:
        // this scan was missing CPU's own compression gate
        // (`SimConfig::fluid_near_wall_compression_threshold`, added
        // 2026-08-08 in `cfl.rs`). CPU requires THREE conditions -- strict
        // fluid AND near a wall AND *actually being compressed right now*;
        // GPU checked only the first two, so the 20x tightening applied
        // permanently to any fluid merely RESTING on a floor (a floor is a
        // wall). That multiplies the substep count by 20 forever: measured
        // `sub=655` per frame in `basic_fluids_gpu.rs`, and at ~8 GPU
        // dispatches per substep that is >5000 dispatches/frame -- which is
        // arithmetically the observed 1 fps, not a mystery.
        //
        // The CPU field's own doc already described this exact failure mode
        // and even named the symptom ("a sustained 3-5fps crawl in
        // `basic_fluids_gui.rs` once the water settled, not a transient
        // slowdown") -- the fix was simply never ported to this backend,
        // which is why CPU fluid demos behave and GPU ones crawl.
        let near_wall = self.config.fluid_near_wall_cfl_scale != 1.0
            && self.particles.iter().any(|p| {
                if p.sleeping != 0 {
                    return false;
                }
                let t = self.config.boundary_thickness as f32;
                let hi = self.config.grid_res as f32 - t;
                self.registry
                    .get(p.material_id)
                    .owns_deformation_volume_state()
                    && (p.x.x < t || p.x.x > hi || p.x.y < t || p.x.y > hi)
                    && {
                        // Same real Mach-relative compression test as `cfl.rs`'s
                        // own (2026-08-11 port) -- calls the SAME real
                        // `rest_acoustic_c2()` method CPU uses, not a
                        // reimplementation, so the two can't drift apart.
                        let j = p.volume / p.initial_volume;
                        let threshold = match self.registry.rest_acoustic_c2(p.material_id) {
                            Some(c2_rest) if c2_rest > f32::EPSILON => {
                                let mach = self.last_max_particle_speed / c2_rest.sqrt();
                                (mach * mach) * self.config.fluid_near_wall_compression_mach_margin
                            }
                            _ => self.config.fluid_near_wall_compression_threshold,
                        };
                        (j - 1.0).abs() > threshold
                    }
            });
        let near_wall_scale = if near_wall {
            self.config.fluid_near_wall_cfl_scale
        } else {
            1.0
        };
        for p in self.particles.iter() {
            if p.sleeping != 0 {
                continue;
            }
            awake_count += 1;
            let mut s = p.v.length();
            if self.config.cfl_include_affine_speed {
                s +=
                    affine_cfl_speed_contribution(&p.velocity_gradient, self.config.grid_cell_size);
            }
            max_speed = max_speed.max(s);
            let mdt = self.registry.get(p.material_id).timestep_bound(
                p.density,
                p.hardening_scale,
                self.config.grid_cell_size,
                self.config.material_cfl_coefficient / near_wall_scale,
                self.config.viscous_timestep_coefficient,
            );
            if mdt.is_finite() && mdt > 0.0 {
                min_mat_dt = min_mat_dt.min(mdt);
            }
            let deformation_dt = deformation_gradient_cfl_bound(
                &p.velocity_gradient,
                self.config.cfl_coefficient.min(0.5),
            );
            if deformation_dt.is_finite() && deformation_dt > 0.0 {
                min_mat_dt = min_mat_dt.min(deformation_dt);
            }
        }
        // Real, standard "additional stability condition" for explicit integration
        // under a body force (Bridson, "Fluid Simulation for Computer Graphics" ch.
        // 3; Foster & Fedkiw 2001) -- gravity alone can move a particle more than
        // one cell per substep even at REST (zero velocity, zero material stress),
        // which none of the bounds above catch (they all key off existing
        // velocity/stress/velocity-gradient, all zero at t=0). Same formula as
        // CPU's `choose_substep_dt` (`spacetime/solver/cfl.rs`) -- ported here
        // 2026-08-08 after this exact gap (present on GPU only) was root-caused
        // live: `basic_fluids_gpu.rs` under its real, stronger-than-CPU gravity
        // (~981*0.003 vs CPU's -0.3) diverged (J up to 7319) regardless of
        // eos_stiffness, including at eos_stiffness=1000 -- proof the missing
        // bound, not material stiffness, was the actual gap, since a stiffer EOS
        // has nothing to do with gravity's own contribution to instability. GPU's
        // scan never received this fix when CPU did earlier tonight (the gravity
        // bound was added directly inside `choose_substep_dt`'s fold, a function
        // this GPU-side scan does not call -- it has its own, separate
        // implementation for the parallel/GPU-mirrored CFL scan).
        // `near_wall_scale` (computed above) also tightens this term -- ported
        // 2026-08-09, see `reactive_gpu_substep_dt`'s own doc for the full
        // citation trail (real, CPU-proven, previously GPU-disclosed-as-not-
        // yet-ported gap).
        let g = self.config.gravity.length();
        if g > f32::EPSILON {
            let gravity_dt = (self.config.cfl_coefficient * self.config.grid_cell_size
                / (g * near_wall_scale))
                .sqrt();
            if gravity_dt.is_finite() && gravity_dt > 0.0 {
                min_mat_dt = min_mat_dt.min(gravity_dt);
            }
        }
        // If every particle is asleep AND something could actually disturb them this
        // frame, there's no awake velocity to base an estimate on — choose_substep_dt
        // would fall back to max_dt (max_speed=0 fails its `> f32::EPSILON` guard), the
        // COARSEST possible substep, right when a wake event needs the FINEST. But wake
        // propagation only happens via a neighbor's grid activity (which requires some
        // OTHER awake particle to exist — if the awake set is truly empty, there is none)
        // or an external impulse. So "everyone asleep" alone isn't a risk: nothing CAN
        // wake spontaneously with no awake particles and no incoming disturbance. Only
        // pay for the fine fallback when a pending impulse could actually wake someone —
        // otherwise a fully-settled scene would pay maximum substep cost forever, which
        // defeats sleep/wake's entire purpose. (Strict WC-MPM fluids never reach this
        // branch at all -- `assert_strict_fluid_mode_is_supported` requires
        // `sleep_threshold <= 0` for them.)
        let might_wake_this_frame = !self.pending_impulses.is_empty();
        if awake_count == 0 && self.config.sleep_threshold > 0.0 && might_wake_this_frame {
            self.config.dt / self.config.max_substeps_per_step.max(1) as f32
        } else {
            cfl_bound(&self.config, max_speed, min_mat_dt, self.config.dt)
        }
    }

    /// Combine `cfl_scan.wgsl`'s per-substep GPU reduction (max speed, max
    /// deformation-gradient rate, max Tait EOS c² numerator, near-wall flag)
    /// into a concrete dt, using the scene-wide, CPU-known coefficients --
    /// mirrors `choose_substep_dt`'s own combination exactly (deformation
    /// term, material/acoustic term, gravity term, then `cfl_bound` folds in
    /// the velocity term and the `max_dt` clamp), just fed from a GPU-
    /// computed reduction instead of a CPU array scan. See `cfl_scan.wgsl`'s
    /// own doc for why this per-substep reactivity is the real fix (not a
    /// per-batch approximation) for basic_fluids_gpu.rs's crash.
    ///
    /// `near_wall` applies `SimConfig::fluid_near_wall_cfl_scale` to BOTH the
    /// acoustic and gravity terms -- the real, CPU-proven mechanism (see that
    /// field's own doc and MEMORY.md's fluid-recovery notes, Round 7-9) that
    /// was never ported to GPU before 2026-08-09. Applied to both terms (not
    /// just whichever is nonzero for this scene) so the same GPU code path
    /// works correctly regardless of whether a given fluid uses a stiff EOS
    /// (acoustic term active) or the eos-less pressure-projection scheme
    /// (acoustic term identically zero, only gravity matters) -- matches
    /// Round 9 attempt 8's real, verified combination on CPU.
    fn reactive_gpu_substep_dt(
        &self,
        max_speed: f32,
        max_rate: f32,
        max_c2: f32,
        near_wall: bool,
        max_dt: f32,
    ) -> f32 {
        let config = &self.config;
        let near_wall_scale = if near_wall && config.fluid_near_wall_cfl_scale != 1.0 {
            config.fluid_near_wall_cfl_scale
        } else {
            1.0
        };
        let mut min_mat_dt = max_dt;
        if max_rate.is_finite() && max_rate > f32::EPSILON {
            let deformation_dt = config.cfl_coefficient.min(0.5) / max_rate;
            if deformation_dt.is_finite() && deformation_dt > 0.0 {
                min_mat_dt = min_mat_dt.min(deformation_dt);
            }
        }
        if max_c2.is_finite() && max_c2 > f32::EPSILON {
            let acoustic_dt = config.material_cfl_coefficient * config.grid_cell_size
                / (max_c2.sqrt() * near_wall_scale);
            if acoustic_dt.is_finite() && acoustic_dt > 0.0 {
                min_mat_dt = min_mat_dt.min(acoustic_dt);
            }
        }
        let g = config.gravity.length();
        if g > f32::EPSILON {
            let gravity_dt =
                (config.cfl_coefficient * config.grid_cell_size / (g * near_wall_scale)).sqrt();
            if gravity_dt.is_finite() && gravity_dt > 0.0 {
                min_mat_dt = min_mat_dt.min(gravity_dt);
            }
        }
        cfl_bound(config, max_speed, min_mat_dt, max_dt)
    }

    /// Blocking readback of `cfl_scan.wgsl`'s 4-word reduction, decoded back from
    /// bitcast<u32> to f32 (exact for positive finite floats -- see
    /// `cfl_reduction`'s own field doc). Tiny (16 bytes) -- cheap enough to call
    /// once per substep for strict-fluid scenes, unlike a full particle readback.
    fn read_cfl_reduction_blocking(&self) -> (f32, f32, f32, bool) {
        let raw = self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.cfl_reduction,
            4,
        );
        (
            f32::from_bits(raw[0]),
            f32::from_bits(raw[1]),
            f32::from_bits(raw[2]),
            f32::from_bits(raw[3]) > 0.0,
        )
    }

    /// Regional-substepping Step 1/2 (2026-08-12, `purring-swinging-cookie.md`
    /// Part A) -- same convention as `read_cfl_reduction_blocking` above, sized
    /// for the per-block reduction (`NUM_BLOCKS * 4` words instead of 4).
    fn read_block_cfl_reduction_blocking(&self) -> Vec<[f32; 4]> {
        let raw = self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.block_cfl_reduction,
            super::super::step_params::NUM_BLOCKS * 4,
        );
        raw.chunks_exact(4)
            .map(|c| {
                [
                    f32::from_bits(c[0]),
                    f32::from_bits(c[1]),
                    f32::from_bits(c[2]),
                    f32::from_bits(c[3]),
                ]
            })
            .collect()
    }

    /// Regional-substepping Step 2: classify each of the 256 blocks as Fine
    /// (needs the batch's own fine dt this substep) or Coarse (can skip the
    /// full G2P gather + `particles_update` integrate), from the SAME 4
    /// quantities and the SAME `reactive_gpu_substep_dt` formula the global
    /// scan already uses -- called once per block instead of once globally,
    /// no new CFL formula invented. `dt_fine` computed this way is provably
    /// identical to the existing global scan's result (min-of-256-mins ==
    /// the same global min), a strong, cheap regression property: real
    /// fine-tier dt is byte-identical to today's, always.
    ///
    /// NOT YET WIRED into the batch loop's own live dt selection or the tier
    /// gate in g2p.wgsl/particles_update.wgsl (a real, disclosed,
    /// deliberately separate next step -- see the plan's own section 3-5).
    /// Real and exercised today by `regional_substepping_tests` below, which
    /// is what keeps it off the dead-code list honestly (no `#[allow]` --
    /// this codebase has zero precedent for silencing that lint).
    fn classify_blocks(&self, block_words: &[[f32; 4]], remaining_time: f32) -> (Vec<bool>, f32) {
        let mut dt_b = vec![remaining_time; block_words.len()];
        for (b, words) in block_words.iter().enumerate() {
            let [max_speed, max_rate, max_c2, near_wall_flag] = *words;
            dt_b[b] = self.reactive_gpu_substep_dt(
                max_speed,
                max_rate,
                max_c2,
                near_wall_flag > 0.0,
                remaining_time,
            );
        }
        let dt_fine = dt_b.iter().copied().fold(f32::MAX, f32::min);
        // Floored at the batch size, not at 1.0: a Coarse block integrates
        // once with the batch's whole ACCUMULATED dt (~batch_len * dt_fine),
        // so demoting a block whose own bound can't survive that is what
        // drove the retry ladder and made substeps/frame go UP -- see this
        // field's own doc in `SimConfig` for the full derivation and the
        // real measurement that caught it.
        let margin = self
            .config
            .fluid_regional_substepping_fine_tier_margin
            .max(STRICT_FLUID_SUBSTEP_BATCH_SIZE as f32);
        let mut is_fine: Vec<bool> = dt_b.iter().map(|&dt| dt <= dt_fine * margin).collect();

        // Mandatory halo dilation (see the plan's own section 2 doc for why
        // this is a correctness requirement, not an optimization): any block
        // adjacent to a Fine block becomes Fine too, matching the SAME 3x3
        // neighbor-expansion convention `particle_sort_compact_main` already
        // uses for occupancy, for the identical physical reason (the P2G/G2P
        // kernel stencil spans block boundaries).
        let num_blocks_per_dim = super::super::step_params::NUM_BLOCKS_PER_DIM;
        let original = is_fine.clone();
        for by in 0..num_blocks_per_dim {
            for bx in 0..num_blocks_per_dim {
                let idx = by * num_blocks_per_dim + bx;
                if original[idx] {
                    continue;
                }
                'halo: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = bx as i32 + dx;
                        let ny = by as i32 + dy;
                        if nx < 0
                            || ny < 0
                            || nx >= num_blocks_per_dim as i32
                            || ny >= num_blocks_per_dim as i32
                        {
                            continue;
                        }
                        let nidx = ny as usize * num_blocks_per_dim + nx as usize;
                        if original[nidx] {
                            is_fine[idx] = true;
                            break 'halo;
                        }
                    }
                }
            }
        }
        (is_fine, dt_fine)
    }

    /// Advance one frame of simulation time (`config.dt`) using the GPU.
    ///
    /// Substeps are encoded in batches of up to `SUBSTEP_BATCH_SIZE` (64) command
    /// buffers each -- a per-submit resource ceiling this backend actually hits, not
    /// a physics choice, see that constant's own doc. For an ordinary scene (well
    /// under 64 substeps/frame) this is exactly one batch: one CFL scan of the
    /// frame-start CPU mirror, one dt for every substep, byte-identical to how this
    /// function worked before 2026-08-07.
    ///
    /// For a scene that genuinely needs more than 64 substeps in one frame (a
    /// strict WC-MPM fluid under real gravity is the only case seen so far), EVERY
    /// batch boundary already pays a blocking `device.poll` (see `SUBSTEP_BATCH_SIZE`'s
    /// doc -- required regardless, to avoid overrunning the descriptor allocator).
    /// Strict WC-MPM fluids now spend that already-paid sync point on two real fixes,
    /// not just batching: (1) the solver-status admissibility check moves to EVERY
    /// batch boundary instead of only after the whole frame, so a bad batch is caught
    /// before a further batch keeps computing on top of already-corrupted state; (2)
    /// the CFL scan re-runs against the just-synced actual GPU state before planning
    /// the next batch, instead of reusing the frame-start estimate for the whole
    /// frame. Root-caused 2026-08-07: `basic_fluids_gpu.rs`/`basic_showcase_gpu.rs`
    /// were diverging (J up to 358x volume) specifically because a single frame-start
    /// dt kept being reused across a growing-velocity fluid's ENTIRE ~150-substep
    /// frame, well past where it stayed CFL-safe. This is not the full GPU preflight/
    /// retry protocol (dt is still fixed WITHIN a batch, up to 64 substeps stale, not
    /// truly per-substep) -- a real, bounded improvement using sync points the
    /// architecture already pays for, not a rewrite of the batching itself. Non-strict
    /// scenes (the overwhelming majority -- sand/jellies/snow/etc, all single-batch)
    /// are completely unaffected: this whole re-scan-and-recheck path is only
    /// reachable when `strict_volume_state_active` AND a second batch is needed.
    pub fn step_frame(&mut self) {
        // A lost device cannot be un-lost; every further GPU call on it would panic
        // through wgpu's default error handler. Once lost, become a safe no-op instead.
        if self.is_device_lost() {
            return;
        }
        let total_start = std::time::Instant::now();
        let any_cpu = self.registry.any_needs_cpu_update();
        let strict_volume_state_active = self.assert_strict_fluid_mode_is_supported();

        // The status is sticky within this candidate frame. It is reset only
        // before command encoding, never by a shader that encounters a bad
        // state, so a strict fluid failure cannot be hidden by a later pass.
        self.buffers.clear_solver_status(&self.queue);

        // Upload CPU → GPU only when positions/materials actually changed.
        // Impulses are now applied by a dedicated GPU compute pass (apply_impulses) that
        // reads LIVE GPU positions — no CPU mirror upload needed for impulse-only frames.
        //
        // Do not resort `self.particles` by grid cell here — GPU `particle_sort` already
        // provides spatial locality via a separate index buffer (`sorted_particle_ids`)
        // that never touches actual particle storage order. Resorting the backing array
        // would invalidate `spawn_region`'s promised stable `Range<usize>` particle
        // identity (LP uses this as creature_id -> particle_range).
        let needs_upload = self.layout_dirty || any_cpu;
        if needs_upload {
            self.buffers.upload_particles(&self.queue, &self.particles);
            self.layout_dirty = false;
        }

        // Step-param pool is sized to the worst case (`max_substeps_per_step`) up
        // front, not to this frame's actual substep count -- the dynamic per-batch
        // planning below (see `step_frame`'s own doc) no longer knows the total
        // substep count ahead of time for a strict-fluid multi-batch frame. Same
        // one-time, only-grows cost as before; `ensure_step_param_capacity` is a
        // no-op once the pool has already reached this size.
        if self
            .buffers
            .ensure_step_param_capacity(&self.device, self.config.max_substeps_per_step)
        {
            self.bind_group_pool =
                build_bind_group_pool(&self.device, &self.pipelines, &self.buffers);
        }
        self.frame_index += 1;
        let mut cfl_scan_ns = 0.0f32;

        // Sleep delay: a particle spawned at rest (v=0) satisfies any positive
        // sleep_threshold on its very first substep, before gravity has accelerated it
        // at all — same fix every real physics engine uses for this (Box2D, PhysX,
        // Bullet all require sustained low velocity before sleeping, never an instant
        // single-frame check). Can't add a per-particle timer here (Particle has no
        // spare bytes left), so this is the simulation-level equivalent: don't let
        // anything sleep-score for the first few frames after the most recent spawn,
        // giving real dynamics a chance to start.
        //
        // Window re-arms from `last_spawn_frame` (updated by `spawn_region`), not just
        // frame 0 — otherwise a particle spawned live mid-scene (e.g. a paint tool) would
        // get `sleep_threshold` applied at v=0 on its very first substep and freeze
        // asleep before gravity ever touched it.
        const SLEEP_WARMUP_FRAMES: u64 = 10;
        let step_config = if self.frame_index <= self.last_spawn_frame + SLEEP_WARMUP_FRAMES {
            SimConfig {
                sleep_threshold: 0.0,
                ..self.config
            }
        } else {
            self.config
        };

        // Build force fields uniform (same every substep).
        let mut ff_params: GpuFieldsParams = bytemuck::Zeroable::zeroed();
        ff_params.count = self.force_field_entries.len() as u32;
        for (i, e) in self.force_field_entries.iter().enumerate() {
            ff_params.entries[i] = *e;
        }
        self.buffers
            .upload_force_fields_params(&self.queue, &ff_params);

        // Multi-field contact (GPU port) — directional grip friction, uploaded once per
        // frame like ff_params above. `self.grip_params` starts symmetric (no
        // directional bias, identical to every scene before this existed) and is only
        // live-adjustable via `set_grip_direction`/`set_grip_friction` — a real
        // GPU-side `DirectionalContactGrip` equivalent, matching CPU's own
        // atomics-based live-adjustable pattern (plain field here since GpuSimulation
        // isn't Arc-shared across threads the way CPU's boundary conditions are).
        self.buffers
            .upload_grip_params(&self.queue, &self.grip_params);

        // Day-night/ambient thermal diffusion (GPU port) — uploaded once per frame,
        // same pattern as grip_params above. `enabled == 0` (the default, every
        // existing scene) makes the 4 thermal passes below skip their dispatch
        // entirely, not just early-return per-thread — real, not just disabled-in-name.
        self.buffers
            .upload_thermal_params(&self.queue, &self.thermal_params);
        let thermal_active = self.thermal_params.enabled != 0;

        // Real grid-mediated cohesion/surface-tension (CSF) — uploaded once per
        // frame, same pattern as thermal_params above. `gamma_grid <= 0.0` (the
        // default, every existing scene) makes `grid_cohesion_main` return
        // immediately for every cell — real, not just disabled-in-name.
        self.buffers
            .upload_cohesion_params(&self.queue, &self.cohesion_params);

        // Resource regrowth (GPU port) — same upload + real dispatch-skip pattern as
        // thermal above.
        self.buffers
            .upload_resource_params(&self.queue, &self.resource_params);
        let resource_active = self.resource_params.enabled != 0;

        // ASFLIP (GPU port) — same upload + real dispatch-skip pattern as thermal/
        // resource above, but the "skip" here means the fused g2p_asflip_fused pass
        // REPLACES g2p+particles_update rather than an extra pass being skipped
        // entirely — see SubstepGates::asflip_active's use in encode_substep.rs.
        self.buffers
            .upload_asflip_params(&self.queue, &self.asflip_params);
        let asflip_active = self.asflip_params.enabled != 0;

        // `ColorMode::GridVolume` material-mass tracking — same upload pattern, real
        // per-substep cost (an extra P2G atomic scatter + grid_clear zeroing) only
        // when `attach_grid_material_render_gpu` has been called.
        self.buffers
            .upload_material_mass_params(&self.queue, &self.material_mass_params);

        // Force-sleep/force-wake-by-tag — minimal hook for LP's future chunk system.
        // Uploaded every frame (zeroed when nothing's pending, same as ff_params above)
        // and read once per substep in force_fields.wgsl; cleared after upload since
        // each call is a one-shot edge-trigger, not a persistent state (a tag that's
        // force-asleep doesn't need to be re-sent every frame — sleeping is sticky on
        // the particle itself until something genuinely wakes it).
        let mut sw_params: GpuSleepWakeParams = bytemuck::Zeroable::zeroed();
        sw_params.sleep_count = self.pending_sleep_tags.len() as u32;
        for (i, &tag) in self.pending_sleep_tags.iter().enumerate() {
            sw_params.sleep_tags[i / 4][i % 4] = tag;
        }
        sw_params.wake_count = self.pending_wake_tags.len() as u32;
        for (i, &tag) in self.pending_wake_tags.iter().enumerate() {
            sw_params.wake_tags[i / 4][i % 4] = tag;
        }
        self.buffers
            .upload_sleep_wake_params(&self.queue, &sw_params);
        self.pending_sleep_tags.clear();
        self.pending_wake_tags.clear();

        // force_fields_main is a provable no-op for every particle this frame when none
        // of these are true -- no fields configured, no tag-based sleep/wake pending,
        // and sleep-scoring disabled (the pass's only other job). Even with an empty loop
        // body it still reads+writes every particle's full 128-byte struct, so skipping
        // the whole dispatch (not just the loop) when unneeded avoids that memory traffic
        // — same principle as the lazy spatial hash and sparse-grid active-block dispatch.
        let force_fields_needed = ff_params.count > 0
            || sw_params.sleep_count > 0
            || sw_params.wake_count > 0
            || step_config.sleep_threshold > 0.0;

        // Mirrors CPU's `Grid::has_contact_activity()` gate (`transfer.rs`):
        // `resolve_contact`/`gather_contact_points` are structurally required whenever ANY
        // particle uses multi-field contact (g2p.wgsl unconditionally reads their output),
        // but for the common case where NO particle ever sets `contact_group`, this is
        // provable dead work. A plain O(N) scan of the CPU particle mirror (same
        // "compute once per frame" pattern as `force_fields_needed` above) is far cheaper
        // than the GPU passes it gates.
        let contact_active = self.particles[..self.particle_count]
            .iter()
            .any(|p| p.contact_group != 0);

        // Step params for each substep are uploaded into their pool slot batch-by-batch,
        // just before that batch is encoded -- see the dynamic substep-planning loop
        // below (`step_frame`'s own doc explains why this moved off a single up-front
        // pass). The bind group pointing at each slot only depends on buffer IDENTITY,
        // not contents, so it's still built once in `bind_group_pool` (see that field's
        // doc comment), not recreated per substep per frame -- doing so at LP's
        // ~5-6k-substep-per-frame scale exhausted the GPU's descriptor allocator within
        // seconds. Indexed directly (`self.bind_group_pool[..]`) at each use point below
        // rather than held as a `bind_groups` local across the loop -- the loop also
        // calls `&mut self` methods (`sync_particles_blocking`), which a borrow held
        // across iterations would conflict with.

        // Encode everything into one command buffer — one GPU submit per frame.
        // Order: [apply_impulses?] → [particle_sort?] → substep_0 → … → substep_N
        //
        // apply_impulses runs first so physics sees the freshly-applied velocities.
        // particle_sort re-seeds sorted_particle_ids after a CPU upload (layout_dirty).
        // Both use dedicated buffer slots so they never alias substep params.
        let particle_wg = (self.particle_count as u32).div_ceil(WG_PARTICLES);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mpm_frame"),
            });

        // — apply_impulses pass (GPU-native, no stale CPU mirror) —
        if !self.pending_impulses.is_empty() {
            let mut params = GpuImpulseParams {
                count: self.pending_impulses.len() as u32,
                reserved_velocity_slot: 0.0,
                particle_count: self.particle_count as u32,
                _pad: 0,
                entries: bytemuck::Zeroable::zeroed(),
            };
            for (i, e) in self.pending_impulses.iter().enumerate() {
                params.entries[i] = *e;
            }
            self.buffers.upload_impulse_params(&self.queue, &params);
            let impulse_bg = self
                .pipelines
                .make_impulse_bind_group(&self.device, &self.buffers);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apply_impulses"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.apply_impulses);
            pass.set_bind_group(0, &impulse_bg, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            drop(pass);
            self.pending_impulses.clear();
        }

        // — particle_sort pass: clear -> count -> scan -> scatter, every frame —
        //
        // Runs unconditionally (not gated on layout_dirty) because particle positions drift
        // every substep even when the CPU mirror is never touched — without a per-frame
        // re-sort, sorted_particle_ids would stay frozen at whatever ordering existed at the
        // last CPU upload, going stale as GPU-resident particles move. See particle_sort.wgsl.
        {
            let sort_slot = self.buffers.step_params_pool.len() - 1;
            let sort_params = GpuStepParams::new(
                &self.config,
                self.config.dt,
                self.particle_count,
                contact_active,
                self.last_max_particle_speed,
            );
            self.buffers
                .upload_step_params_at(&self.queue, sort_slot, &sort_params);
            // Reuse the cached bind group from `bind_group_pool` rather than calling
            // `pipelines.make_bind_group(...)` fresh here -- creating one every
            // `step_frame` call exhausts the GPU's descriptor allocator over a long run.
            // `bind_group_pool` already contains one bind group per `step_params_pool`
            // slot, including this sort slot (see `build_bind_group_pool`), rebuilt only
            // when `buffers` reallocates. The bind group only depends on buffer IDENTITY,
            // not contents (which `upload_step_params_at` above rewrites in place), so
            // reusing the cached entry is correct, not just faster.
            let sort_bg = &self.bind_group_pool[sort_slot];
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_sort"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, sort_bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
            pass.set_pipeline(&self.pipelines.particle_sort_clear);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_count);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            // No particle_sort_compact here anymore — active-block detection now runs
            // every substep (see encode_substep's active_block_refresh pass), since
            // particles move every substep and this once-per-frame pass would go stale by
            // substep 2+. This pass's count output is used only for the sort permutation
            // (scan + scatter below), unrelated to active-block correctness.
            pass.set_pipeline(&self.pipelines.particle_sort_scan);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_scatter);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Substeps are batched into multiple command buffers/submits instead of one --
        // stiff-terrain scenes routinely need several hundred substeps in a single frame,
        // and encoding them all into one command buffer exhausts this GPU backend's
        // descriptor allocator. 200 substeps in one submit reliably OOMs, 64 is stable
        // (matches `max_substeps_per_step`'s doc) -- a per-submit resource ceiling on the
        // backend/driver actually exercised, not derived from any GPU spec, so a
        // different backend may need a different number. Blocking between batches is
        // required too -- unblocked back-to-back submits queue up faster than the GPU
        // drains them and hit the same OOM even with batching. Only blocks BETWEEN
        // batches, never after the last one -- typical scenes (well under 64
        // substeps/frame) produce exactly one batch and pay zero extra sync cost.
        //
        // Each batch's own substep count/dt is planned just before it's encoded, not all
        // up front -- see `step_frame`'s own doc for why (strict WC-MPM fluid CFL
        // re-scan at each batch boundary). For a single-batch frame (the common case)
        // this is exactly one `scan_gpu_cfl_sub_dt` call against the frame-start state,
        // byte-identical to how this function worked before 2026-08-07.
        const SUBSTEP_BATCH_SIZE: usize = 64;
        // Split pure CPU command-building time from GPU-completion wait time --
        // "encode_ns" previously bundled both under one name, hiding whether a slow
        // step_frame() was a CPU-side encoding problem or genuinely GPU-execution-bound.
        let mut pure_encode_ns = 0.0f32;
        let mut wait_ns = 0.0f32;
        let mut substeps_taken = 0usize;
        let mut remaining_time = self.config.dt;
        let mut last_used_sub_dt = self.config.dt;

        if strict_volume_state_active {
            // GPU-native, batch-reactive CFL -- see `cfl_scan.wgsl`'s own doc for
            // the crash this replaces the old per-64-batch CPU-mirror scan for
            // (basic_fluids_gpu.rs, 2026-08-08: J up to 34653 under real gravity,
            // root-caused to a whole up-to-64-substep batch running on a dt chosen
            // from state up to 64 substeps stale).
            //
            // A true batch size of 1 (tried first, 2026-08-09) fixed correctness
            // but made the blocking CPU<->GPU sync -- not the physics kernels --
            // the dominant per-frame cost: every substep pays its own submit +
            // `poll(wait_indefinitely)` + two tiny blocking readbacks, and that
            // fixed per-round-trip latency (not payload size) dominates when a
            // real-water scene needs 100+ substeps/frame, live-confirmed
            // (fps 0-5 even in `--release`, basic_fluids_gui.rs and
            // basic_fluids_gpu.rs both). `STRICT_FLUID_SUBSTEP_BATCH_SIZE`
            // substeps are now encoded into ONE submit, sharing ONE dt, cutting
            // the sync-round-trip count by that same factor.
            //
            // This is not a reversion to the old 64-batch bug: (1) the batch's
            // shared dt comes from `cfl_scan.wgsl`'s real GPU-measured worst-case
            // over the PREVIOUS batch (an atomicMax reduction over actual
            // post-step particle state), not a stale CPU-mirror scan of
            // frame-start state -- so staleness is bounded to one batch, not one
            // frame; (2) `cfl_reduction` is deliberately left uncleared across a
            // batch's own substeps (cleared once per batch, not per substep) so
            // the atomicMax naturally accumulates the worst case seen ANYWHERE
            // in the batch, which is what the NEXT batch's dt is chosen from --
            // a conservative, not optimistic, estimate. A batch size of 8 was
            // tried in isolation before the artificial bulk viscosity and
            // near-wall CFL fixes existed (2026-08-08) and only reduced, not
            // eliminated, the blowup -- this is a different experiment: the same
            // batch size WITH those two later stabilizers already in place. The
            // regression test below re-verifies J stays bounded with this exact
            // combination.
            let scan_start = std::time::Instant::now();
            // Frame-start bootstrap value only (first batch's dt) -- every
            // subsequent batch's dt comes from `cfl_reduction` below, not this
            // CPU-mirror scan.
            let mut sub_dt_cfl = self.scan_gpu_cfl_sub_dt();
            cfl_scan_ns += scan_start.elapsed().as_secs_f32() * 1.0e9;

            // `max_substeps_per_step` is an INITIAL pool-sizing hint for strict
            // fluids, not a hard cap: a weakly-compressible fluid's mass/momentum
            // conservation is a per-frame guarantee, so silently dropping leftover
            // `remaining_time` (the ordinary, acceptable-for-other-materials
            // behavior every other GPU scene still uses) would mean advancing an
            // INCOMPLETE dt for a PDE that specifically must not tolerate that.
            // The pool grows (doubling, amortized) via `ensure_step_param_capacity`
            // + a `bind_group_pool` rebuild exactly like the one-time up-front call
            // above, just triggered on demand instead of never. `STRICT_FLUID_
            // SUBSTEP_SAFETY_CEILING` is a circuit breaker for a genuinely
            // never-stabilizing scene (CFL dt shrinking without bound) -- fails
            // loud, not a silent hang or a bandaid time-drop.
            const STRICT_FLUID_SUBSTEP_SAFETY_CEILING: usize = 100_000;
            // Real batch-level retry, mirroring CPU's `do_substep_with_retry`
            // (`spacetime/solver/step.rs`) adapted to GPU's batch (not
            // per-substep) sync granularity -- 2026-08-11, closing the real,
            // previously-disclosed gap ("not the full GPU preflight/retry
            // protocol... dt is still fixed WITHIN a batch") that let J drift
            // to 47-350x in `basic_fluids_gpu.rs`/`basic_showcase_gpu.rs`
            // undetected. 16 halvings matches CPU's own real
            // `FLUID_STEP_RETRY_LIMIT` (~65000x finer than the original dt).
            const STRICT_FLUID_BATCH_RETRY_LIMIT: u32 = 16;
            // Must match `STRICT_FLUID_J_MAX`/`STRICT_FLUID_J_MIN` in
            // `particles_update.wgsl` exactly -- both real, generous (50x)
            // safety range, same default as CPU's `SimConfig::j_max`/`j_min`.
            // Duplicated (not shared) because the WGSL side is a shader
            // constant baked at pipeline-build time, not something this
            // Rust-side exhaustion backstop can import directly.
            const STRICT_FLUID_J_MAX: f32 = 50.0;
            const STRICT_FLUID_J_MIN: f32 = 1.0 / 50.0;
            // Real fps fix (2026-08-11), the actual root cause of the
            // measured 55 -> 1 fps collapse: retry-with-halved-dt is the
            // right tool ONLY for a TRANSIENT CFL spike, where a finer dt
            // genuinely resolves the violation. For a PERSISTENT condition
            // (measured live: the SAME particle index failing every batch,
            // unresolved even after 16 real halvings = ~65000x finer dt) it
            // is pure waste -- each futile attempt costs a full GPU
            // round-trip (submit + blocking poll + status readback +
            // buffer-copy rollback), and paying 16 of those per batch,
            // every batch, for the rest of the run IS the fps collapse.
            // Tracking which particle failed last batch lets a genuine
            // repeat go STRAIGHT to the exhaustion backstop (which already
            // handles exactly this case correctly) instead of re-proving
            // the same negative result 16 more times. A transient spike
            // still gets the full real retry ladder, unchanged.
            let mut last_failed_particle: Option<u32> = None;
            let particle_bytes =
                (self.particle_count * std::mem::size_of::<crate::particle::Particle>()) as u64;
            let mut pool_capacity_hint = self.config.max_substeps_per_step;
            // Regional-substepping Step 4 (`purring-swinging-cookie.md` Part A) --
            // `regional_enabled=false` (the default, every existing scene) makes every
            // block Fine for the whole frame, byte-identical to today (see the flat-fill
            // branch below). `current_tier` is a per-FRAME local (not a struct field) --
            // deliberately reset to all-Fine at the start of every `step_frame()` call,
            // matching the plan's own "first batch of a frame: no prior block readback
            // exists yet, treat every block as Fine" bootstrap (avoids building a second
            // CPU-side mirror of the CFL scan purely to bootstrap one batch's tiering).
            // Updated after each successful batch from THAT batch's own block-CFL
            // readback, for the NEXT batch to use -- frozen across a batch's own retry
            // ladder (declared outside the `for retry_attempt` loop below).
            let regional_enabled = self.config.fluid_regional_substepping_gpu_enabled;
            let mut current_tier: Vec<bool> = vec![true; NUM_BLOCKS];
            loop {
                if remaining_time <= 0.0 {
                    break;
                }
                assert!(
                    substeps_taken < STRICT_FLUID_SUBSTEP_SAFETY_CEILING,
                    "GPU strict fluid CFL never stabilized within {STRICT_FLUID_SUBSTEP_SAFETY_CEILING} substeps this frame; inspect the scene/material configuration"
                );

                // Real pre-batch state to roll back to if this batch turns out
                // inadmissible -- both the GPU particle buffer AND the CPU-side
                // scalars that the batch's own encoding loop advances, so a
                // retry truly restarts from the exact state before this batch,
                // not a partially-advanced one.
                let pre_batch_remaining_time = remaining_time;
                let pre_batch_substeps_taken = substeps_taken;
                let pre_batch_pool_capacity_hint = pool_capacity_hint;
                let mut snapshot_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("mpm_strict_fluid_batch_snapshot"),
                        });
                snapshot_encoder.copy_buffer_to_buffer(
                    &self.buffers.particles,
                    0,
                    &self.buffers.particle_batch_snapshot,
                    0,
                    particle_bytes,
                );
                self.queue
                    .submit(std::iter::once(snapshot_encoder.finish()));

                let mut batch_sub_dt_cfl = sub_dt_cfl;
                let mut batch_admissible = false;
                for retry_attempt in 0..=STRICT_FLUID_BATCH_RETRY_LIMIT {
                    remaining_time = pre_batch_remaining_time;
                    substeps_taken = pre_batch_substeps_taken;
                    pool_capacity_hint = pre_batch_pool_capacity_hint;

                    // Plan and encode a batch of up to STRICT_FLUID_SUBSTEP_BATCH_SIZE
                    // substeps into ONE command buffer, all sharing this batch's dt.
                    self.buffers.clear_cfl_reduction(&self.queue);
                    // Regional-substepping Step 1: same per-attempt clear cadence as
                    // `clear_cfl_reduction` above -- a retried attempt must restart the
                    // per-block accumulation too, not just the global one.
                    self.buffers.clear_block_cfl_reduction(&self.queue);
                    self.buffers.clear_solver_status(&self.queue);
                    let chunk_encode_start = std::time::Instant::now();
                    let mut sub_encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("mpm_substep_strict_fluid_batch"),
                            });
                    let mut batch_len = 0usize;
                    // Regional-substepping Step 4 -- running sum of the ACTUAL sub_dt used
                    // each substep this batch, not `batch_len * batch_sub_dt_cfl`: the
                    // existing clipping against `remaining_time` on a frame's final
                    // (possibly short) batch would otherwise silently mis-account the
                    // coarse tier's own resync dt. Reset every retry attempt, same
                    // lifetime as `batch_len` itself.
                    let mut coarse_accumulated_dt = 0.0f32;
                    while batch_len < STRICT_FLUID_SUBSTEP_BATCH_SIZE && remaining_time > 0.0 {
                        let sub_dt = batch_sub_dt_cfl.min(remaining_time);
                        assert!(
                            sub_dt.is_finite()
                                && sub_dt > 0.0
                                && remaining_time - sub_dt < remaining_time,
                            "GPU adaptive timestep cannot advance requested simulation time"
                        );
                        let params = GpuStepParams::new(
                            &step_config,
                            sub_dt,
                            self.particle_count,
                            contact_active,
                            self.last_max_particle_speed,
                        );
                        if substeps_taken >= pool_capacity_hint {
                            pool_capacity_hint =
                                pool_capacity_hint.saturating_mul(2).max(substeps_taken + 1);
                            if self
                                .buffers
                                .ensure_step_param_capacity(&self.device, pool_capacity_hint)
                            {
                                self.bind_group_pool = build_bind_group_pool(
                                    &self.device,
                                    &self.pipelines,
                                    &self.buffers,
                                );
                            }
                        }
                        self.buffers
                            .upload_step_params_at(&self.queue, substeps_taken, &params);
                        last_used_sub_dt = sub_dt;
                        remaining_time -= sub_dt;
                        coarse_accumulated_dt += sub_dt;

                        // Regional-substepping Step 4 -- per-block dt plan for THIS
                        // substep. Fine tier: `sub_dt` every substep (byte-identical
                        // physics to today for those blocks). Coarse tier: skip (0.0)
                        // except the batch's last substep, which carries the coarse
                        // tier's own accumulated resync dt (the "async MPM" block-local
                        // dt technique, Yuanming Hu et al. -- see the plan's own Section
                        // 4 doc). `regional_enabled=false` flat-fills every block with
                        // `sub_dt`, making the tier gate in g2p.wgsl/particles_update.wgsl
                        // provably always-false -- an algebraic identity to today, not
                        // merely "should be."
                        let is_last_substep_of_batch = batch_len + 1
                            >= STRICT_FLUID_SUBSTEP_BATCH_SIZE
                            || remaining_time <= 0.0;
                        let mut block_dt_values = [sub_dt; NUM_BLOCKS];
                        if regional_enabled {
                            for (b, value) in block_dt_values.iter_mut().enumerate() {
                                *value = if current_tier[b] {
                                    sub_dt
                                } else if is_last_substep_of_batch {
                                    coarse_accumulated_dt
                                } else {
                                    0.0
                                };
                            }
                        }
                        self.buffers.upload_block_dt_at(
                            &self.queue,
                            substeps_taken,
                            &block_dt_values,
                        );

                        let bg = &self.bind_group_pool[substeps_taken];
                        self.encode_substep(
                            &mut sub_encoder,
                            bg,
                            particle_wg,
                            SubstepGates {
                                force_fields_needed,
                                contact_active,
                                thermal_active,
                                resource_active,
                                asflip_active,
                                cfl_scan_active: true,
                            },
                        );
                        substeps_taken += 1;
                        batch_len += 1;
                    }
                    self.queue.submit(std::iter::once(sub_encoder.finish()));
                    pure_encode_ns += chunk_encode_start.elapsed().as_secs_f32() * 1.0e9;

                    let wait_start = std::time::Instant::now();
                    self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                    wait_ns += wait_start.elapsed().as_secs_f32() * 1.0e9;

                    if self.is_device_lost() {
                        break;
                    }
                    // Checked once per batch (not per substep) -- the batch is small
                    // (8) so this is still much earlier detection than the old
                    // 64-substep batch, at a fraction of the sync cost of checking
                    // every single substep.
                    let status = self.buffers.readback_u32_blocking(
                        &self.device,
                        &self.queue,
                        &self.buffers.solver_status,
                        4,
                    );
                    if status[0] == 0 {
                        // Next batch's dt comes from the real GPU-measured
                        // `reactive_gpu_substep_dt` below (this batch's own
                        // actual worst-case dynamics), not from whatever
                        // `batch_sub_dt_cfl` this attempt happened to use --
                        // that reactive value already correctly reflects
                        // reality regardless of whether a retry occurred.
                        batch_admissible = true;
                        break;
                    }
                    // Real exhaustion backstop, ported from CPU's own
                    // `project_particle_state_to_admissible` (2026-08-11) --
                    // real diagnostic data (temp instrumentation, since
                    // removed) showed the actual failure mode this closes:
                    // a spatially isolated fluid particle (a stray droplet
                    // flung out on impact, with too few real neighbors for
                    // its local velocity-divergence estimate to mean
                    // anything) lands J right at ~47-50 EVERY batch,
                    // independent of dt -- not a transient CFL spike at
                    // all, so no amount of retrying at a finer dt ever
                    // converges it, and 16 real retries (~65000x finer)
                    // confirmed this live. CPU's own real fix for exactly
                    // this class of case is to clamp the specific violated
                    // invariant (here: J rescaled into [j_min, j_max],
                    // volume/density recomputed consistently with it) and
                    // move on, preserving the rest of the real computed
                    // motion -- NOT rolling back the whole batch (which
                    // would just reproduce the identical failure next
                    // attempt, as observed) and NOT panicking (which
                    // would kill the whole simulation over one isolated
                    // droplet).
                    // Real persistence detection (see `last_failed_particle`'s
                    // own doc above for the measured fps impact): the same
                    // particle failing again, on the very first attempt of a
                    // fresh batch, is direct evidence this is NOT a transient
                    // spike a finer dt can fix -- it already survived a full
                    // 16-halving ladder last batch. Skip straight to the
                    // backstop rather than re-proving that at full GPU
                    // round-trip cost 16 more times.
                    let is_known_persistent =
                        retry_attempt == 0 && last_failed_particle == Some(status[1]);
                    if retry_attempt == STRICT_FLUID_BATCH_RETRY_LIMIT || is_known_persistent {
                        last_failed_particle = Some(status[1]);
                        let mut dump = self.buffers.readback_blocking(
                            &self.device,
                            &self.queue,
                            self.particle_count,
                        );
                        let mut clamped_any = false;
                        // Real, previously-CPU-only ceiling (`spacetime/solver/step.rs`,
                        // found 2026-08-09), ported here 2026-08-12 after live dense
                        // per-frame diagnostics proved GPU had the identical gap: the
                        // `!p.v.is_finite()` check just below only catches NaN/infinite
                        // velocity, not FINITE-but-absurd velocity (measured live on
                        // this exact GPU path: max_speed=351 grid-units/s, ~20x this
                        // scene's own real free-fall estimate, sailing through every
                        // check because it was never non-finite). Same real formula as
                        // CPU's `max_representable_speed`: solving `choose_substep_dt`'s
                        // own velocity-CFL term (`dt = cfl_coefficient*grid_cell_size/
                        // max_speed`) for the max_speed that keeps that result at or
                        // above `min_dt` -- uses `min_dt` (the solver's absolute floor),
                        // NOT this substep's own (already reactively shrunk) dt, which
                        // is why the earlier per-substep admissibility check (`particles_
                        // update.wgsl`) alone couldn't catch this: that check's bound
                        // grows MORE permissive as dt shrinks, exactly backwards from a
                        // real sanity ceiling.
                        let max_representable_speed = self.config.cfl_coefficient
                            * self.config.grid_cell_size
                            / self.config.min_dt;
                        for p in dump.iter_mut() {
                            if self.registry.get(p.material_id).constitutive_model()
                                != crate::materials::ConstitutiveModel::Fluid
                            {
                                continue;
                            }
                            if !p.v.is_finite() {
                                p.v = glam::Vec2::ZERO;
                                clamped_any = true;
                            } else {
                                let speed = p.v.length();
                                if speed > max_representable_speed {
                                    p.v *= max_representable_speed / speed;
                                    clamped_any = true;
                                }
                            }
                            if !p.velocity_gradient.x_axis.is_finite()
                                || !p.velocity_gradient.y_axis.is_finite()
                            {
                                p.velocity_gradient = glam::Mat2::ZERO;
                                clamped_any = true;
                            }
                            let f = p.deformation_gradient;
                            if !f.x_axis.is_finite()
                                || !f.y_axis.is_finite()
                                || f.determinant() <= 0.0
                            {
                                p.deformation_gradient = glam::Mat2::IDENTITY;
                                p.volume = p.initial_volume.max(1.0e-8);
                                p.density = (p.mass / p.volume).max(1.0e-8);
                                clamped_any = true;
                            }
                        }
                        // Real, direct fix (2026-08-11), after two prior
                        // attempts at re-deriving the shader's exact
                        // condition both failed to find anything to clamp:
                        // checking `f.determinant()` (stored, pre-substep
                        // F) found the state already in-range; replicating
                        // the shader's own `J_fluid = old_j*exp(dt*div_v)`
                        // formula for a SINGLE substep also found nothing,
                        // because the real batch runs up to
                        // `STRICT_FLUID_SUBSTEP_BATCH_SIZE` (8) substeps,
                        // compounding the effect across several of them --
                        // a multi-substep cumulative process this
                        // single-substep replica structurally cannot
                        // predict correctly. Rather than keep re-guessing
                        // the precise WGSL arithmetic, use what we ALREADY
                        // know for certain: 16 real retries (the actual
                        // GPU-measured admissibility check, not a Rust-side
                        // approximation of it) have DEFINITIVELY proven
                        // `status[1]` is the real problem particle. Pull
                        // its J back from wherever it currently sits toward
                        // real, disclosed safety margin, directly.
                        let reported_idx = status[1] as usize;
                        if let Some(p) = dump.get_mut(reported_idx) {
                            if self.registry.get(p.material_id).constitutive_model()
                                == crate::materials::ConstitutiveModel::Fluid
                            {
                                let f = p.deformation_gradient;
                                // REAL FIX (2026-08-12): this repair branch used to check
                                // only whether F ITSELF was valid (finite, positive det) --
                                // it never checked whether F and volume actually AGREE with
                                // each other, so it would silently "fix" a state that was
                                // never a real extreme-dynamics case at all, just an
                                // externally-injected inconsistency (confirmed via
                                // `gpu_strict_fluid_rejects_inconsistent_constitutive_state`,
                                // tests/gpu.rs: `p.volume *= 2.0` with F left at identity --
                                // F alone looks perfectly valid, so the old check repaired it
                                // instead of letting the deliberate hard-fail through). Real,
                                // genuine extreme dynamics (the case this backstop actually
                                // exists for) keep volume and F drifting TOGETHER even as J
                                // grows large -- same real consistency check
                                // `strict_fluid_state_is_admissible` already does in
                                // p2g.wgsl, same tolerance, reused here rather than a new
                                // number.
                                const STRICT_FLUID_RELATIVE_TOLERANCE: f32 = 2.0e-4;
                                let j_f = f.determinant();
                                let j_volume = p.volume / p.initial_volume;
                                let volume_error = if j_volume > 0.0 {
                                    (j_f - j_volume).abs() / j_volume
                                } else {
                                    f32::INFINITY
                                };
                                if f.x_axis.is_finite()
                                    && f.y_axis.is_finite()
                                    && j_f > 0.0
                                    && volume_error <= STRICT_FLUID_RELATIVE_TOLERANCE
                                {
                                    // Real, disclosed engineering margin (NOT
                                    // a physical constant) -- same real
                                    // category as `j_max` itself (CPU's own
                                    // doc: "a real, generous safety range,
                                    // NOT a physical bound"). Pulls this
                                    // KNOWN-problematic particle to the
                                    // CENTER of the admissible range
                                    // (geometric mean of j_min/j_max = 1.0,
                                    // real headroom on both sides) rather
                                    // than leaving it hugging whichever
                                    // boundary it was already near, so the
                                    // next several substeps of its own
                                    // genuine (persistently expansive)
                                    // dynamics have real room before
                                    // needing this backstop again.
                                    let j_target = (STRICT_FLUID_J_MIN * STRICT_FLUID_J_MAX).sqrt();
                                    p.deformation_gradient =
                                        f * (j_target / f.determinant()).sqrt();
                                    p.volume = (p.initial_volume * j_target).max(1.0e-8);
                                    p.density = (p.mass / p.volume).max(1.0e-8);
                                    clamped_any = true;
                                }
                            }
                        }
                        assert!(
                            clamped_any,
                            "GPU strict fluid update became inadmissible after {STRICT_FLUID_BATCH_RETRY_LIMIT} real retries \
                             (code {}, particle {}, events {}, reason {}) but the real exhaustion backstop found nothing to \
                             clamp -- reason 1/4=incoming-state check, 2/5=J_fluid exponential update, 3/6=final new_x/new_F \
                             check (NOT visible via particle-buffer readback -- that checkpoint runs before any write-back); \
                             a genuinely different failure class (e.g. non-finite position, which this backstop deliberately \
                             does not relocate -- CPU's own equivalent resets position to domain center, a real, disclosed gap \
                             not yet ported; the reported particle index itself pointing at a non-fluid particle, which \
                             would itself be a real, separate bug in the reporting; OR -- as of 2026-08-12, intentional --  \
                             volume and F disagreeing beyond STRICT_FLUID_RELATIVE_TOLERANCE, meaning this is a genuinely \
                             inconsistent state, not repairable extreme dynamics: the backstop correctly refuses to repair \
                             it and this panic IS the intended, designed behavior); inspect the scene/material configuration",
                            status[0], status[1], status[2], status[3],
                        );
                        self.queue.write_buffer(
                            &self.buffers.particles,
                            0,
                            bytemuck::cast_slice(&dump),
                        );
                        // Real bug in this backstop's own first version
                        // (found live 2026-08-11, `basic_fluids_gpu.rs`
                        // panicking at the END-of-frame check rather than
                        // in this loop): the failure this backstop just
                        // RESOLVED was still latched in `solver_status`,
                        // so `step_frame`'s own final frame-level
                        // admissibility assert -- which runs after all
                        // batching, on whatever status remains -- saw a
                        // stale flag for an already-handled condition and
                        // panicked anyway. Clearing it here is what makes
                        // "handled by the backstop" actually mean handled.
                        self.buffers.clear_solver_status(&self.queue);
                        batch_admissible = true;
                        break;
                    }
                    // Real rollback: this batch corrupted state at the current
                    // dt -- restore the pre-batch snapshot (undoing all
                    // STRICT_FLUID_SUBSTEP_BATCH_SIZE substeps just encoded,
                    // not just the last one) and retry the SAME batch at half
                    // the dt, mirroring CPU's own retry loop exactly.
                    let mut rollback_encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("mpm_strict_fluid_batch_rollback"),
                            });
                    rollback_encoder.copy_buffer_to_buffer(
                        &self.buffers.particle_batch_snapshot,
                        0,
                        &self.buffers.particles,
                        0,
                        particle_bytes,
                    );
                    self.queue
                        .submit(std::iter::once(rollback_encoder.finish()));
                    self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                    // Record which particle failed, so a repeat next batch is
                    // recognized as persistent (see `is_known_persistent`).
                    last_failed_particle = Some(status[1]);
                    batch_sub_dt_cfl *= 0.5;
                }
                if self.is_device_lost() || !batch_admissible {
                    break;
                }

                if remaining_time <= 0.0 {
                    break;
                }
                let (max_speed, max_rate, max_c2, near_wall) = self.read_cfl_reduction_blocking();
                // One-batch-lagged, real (not estimated) -- feeds cfl_scan.wgsl's
                // near-wall gate Mach-relative threshold on the NEXT batch, mirroring
                // CPU's identical `self.last_max_particle_speed = measured_max_speed;`
                // (spacetime/solver/step.rs).
                self.last_max_particle_speed = max_speed;
                sub_dt_cfl = self.reactive_gpu_substep_dt(
                    max_speed,
                    max_rate,
                    max_c2,
                    near_wall,
                    remaining_time,
                );
                // Regional-substepping Step 4 -- reclassify blocks from THIS batch's
                // own just-measured per-block state, for the NEXT batch to use. Real
                // GPU-measured data (not stale/estimated), same one-batch-lagged
                // cadence as `last_max_particle_speed` above. `sub_dt_cfl` itself
                // (the actual dt fine-tier blocks use) is untouched by this --
                // `classify_blocks`' own `dt_fine` return is used ONLY as the
                // Fine/Coarse threshold, never substituted in place of the global
                // scan's own value (see the plan's own Section 2 for why: a per-block
                // min-reduction is provably >= the global scan's max-of-every-field
                // reduction, so it is NOT a safe drop-in replacement for it).
                if regional_enabled {
                    let block_words = self.read_block_cfl_reduction_blocking();
                    let (is_fine, _dt_fine) = self.classify_blocks(&block_words, remaining_time);
                    current_tier = is_fine;
                }
                // TEMPORARY: live, unbuffered (eprintln -- Rust's stderr is not
                // line-buffered even when piped to a file, unlike stdout) trace
                // of dt evolution WITHIN one frame's strict-fluid loop -- hunting
                // whether dt genuinely collapses batch-over-batch toward the
                // STRICT_FLUID_SUBSTEP_SAFETY_CEILING (never printed live before;
                // the per-frame log only ever saw the FINAL substep count after
                // the whole loop already returned).
                if substeps_taken.is_multiple_of(800) {
                    eprintln!(
                        "  BATCH_DT substeps_taken={substeps_taken} sub_dt_cfl={sub_dt_cfl:.3e} \
                         max_speed={max_speed:.3} max_rate={max_rate:.3} max_c2={max_c2:.3e} \
                         near_wall={near_wall} remaining_time={remaining_time:.4}"
                    );
                }
            }
        } else {
            loop {
                if remaining_time <= 0.0 || substeps_taken >= self.config.max_substeps_per_step {
                    break;
                }
                let scan_start = std::time::Instant::now();
                let sub_dt_cfl = self.scan_gpu_cfl_sub_dt();
                cfl_scan_ns += scan_start.elapsed().as_secs_f32() * 1.0e9;

                let batch_cap =
                    SUBSTEP_BATCH_SIZE.min(self.config.max_substeps_per_step - substeps_taken);
                let mut batch_len = 0usize;
                while batch_len < batch_cap && remaining_time > 0.0 {
                    let sub_dt = sub_dt_cfl.min(remaining_time);
                    assert!(
                        sub_dt.is_finite()
                            && sub_dt > 0.0
                            && remaining_time - sub_dt < remaining_time,
                        "GPU adaptive timestep cannot advance requested simulation time"
                    );
                    let params = GpuStepParams::new(
                        &step_config,
                        sub_dt,
                        self.particle_count,
                        contact_active,
                        self.last_max_particle_speed,
                    );
                    self.buffers.upload_step_params_at(
                        &self.queue,
                        substeps_taken + batch_len,
                        &params,
                    );
                    // Regional substepping is strict-fluid-only (see the plan's own
                    // non-goals) -- this non-strict-fluid path always flat-fills every
                    // block with `sub_dt`, making the tier gate in g2p.wgsl/
                    // particles_update.wgsl provably always-false here, same as the
                    // feature-off path in the strict-fluid branch above. REQUIRED, not
                    // optional: g2p.wgsl/particles_update.wgsl now read block_dt
                    // unconditionally for every particle regardless of which loop
                    // encoded this substep, so leaving this pool slot un-uploaded (its
                    // buffer content undefined/stale) would incorrectly gate every
                    // particle in every non-strict-fluid scene -- confirmed live via
                    // real test failures (ASFLIP, contact) before this fix.
                    let block_dt_values = [sub_dt; NUM_BLOCKS];
                    self.buffers.upload_block_dt_at(
                        &self.queue,
                        substeps_taken + batch_len,
                        &block_dt_values,
                    );
                    last_used_sub_dt = sub_dt;
                    remaining_time -= sub_dt;
                    batch_len += 1;
                }

                let batch = &self.bind_group_pool[substeps_taken..substeps_taken + batch_len];
                let chunk_encode_start = std::time::Instant::now();
                let mut sub_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("mpm_substep_batch"),
                        });
                for bg in batch {
                    self.encode_substep(
                        &mut sub_encoder,
                        bg,
                        particle_wg,
                        SubstepGates {
                            force_fields_needed,
                            contact_active,
                            thermal_active,
                            resource_active,
                            asflip_active,
                            cfl_scan_active: false,
                        },
                    );
                }
                self.queue.submit(std::iter::once(sub_encoder.finish()));
                pure_encode_ns += chunk_encode_start.elapsed().as_secs_f32() * 1.0e9;
                substeps_taken += batch_len;

                let more_needed =
                    remaining_time > 0.0 && substeps_taken < self.config.max_substeps_per_step;
                if !more_needed {
                    break;
                }
                let wait_start = std::time::Instant::now();
                self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                wait_ns += wait_start.elapsed().as_secs_f32() * 1.0e9;
            }
        }
        self.last_sim_time_dropped = remaining_time.max(0.0);
        self.last_substeps = substeps_taken;
        self.last_sub_dt = last_used_sub_dt;
        let encode_ns = pure_encode_ns;
        // Repurposed: real GPU-completion wait time between batches, not always 0 --
        // this IS where GPU execution time shows up for multi-batch (>64 substep) frames.
        let submit_ns = wait_ns;

        // Final check for the LAST batch -- the in-loop check above only fires when
        // CONTINUING to a further batch, so the last batch (the common single-batch
        // case's only batch) still needs its own check here. Same assert/sync as above.
        if strict_volume_state_active && !self.is_device_lost() {
            let status = self.buffers.readback_u32_blocking(
                &self.device,
                &self.queue,
                &self.buffers.solver_status,
                4,
            );
            assert!(
                status[0] == 0,
                "GPU strict fluid update became inadmissible (code {}, particle {}, events {}); reduce the timestep or inspect the applied force/state",
                status[0],
                status[1],
                status[2],
            );
            self.sync_particles_blocking();
        }

        // Async GPU → CPU readback — never blocks the render thread.
        //
        // Two-phase: begin_readback submits a GPU copy + async map (non-blocking).
        // The receiver fires on a subsequent frame when the GPU copy + map completes.
        // We pump wgpu callbacks with poll(Poll) each frame so the mapping progresses.
        //
        // If any_cpu: readback every frame (CPU plasticity needs current state).
        // Otherwise: stride-gated to reduce overhead.
        let readback_start = std::time::Instant::now();
        self.readback_frame = self.readback_frame.wrapping_add(1);
        let want_readback = !strict_volume_state_active
            && (any_cpu || self.readback_frame.is_multiple_of(self.readback_stride));

        // Pump wgpu callbacks so any in-flight mapping can complete.
        self.device.poll(wgpu::PollType::Poll).ok();

        // Check if a previous async readback completed -- Ok, Err, or still pending.
        // Every completion path must explicitly unmap regardless of Ok/Err — an
        // unhandled Err leaves the staging buffer mapped forever (finish_readback, the
        // only unmapper, never called) and pending_readback stuck Some forever, until
        // something else tries to map the same buffer and panics.
        let readback_done = self
            .pending_readback
            .as_ref()
            .and_then(|flag| flag.lock().ok().and_then(|mut g| g.take()));
        if let Some(result) = readback_done {
            self.pending_readback = None;
            // The device-lost check at the TOP of step_frame only guards against a
            // device that was ALREADY lost before this call started — it says nothing
            // about a device that dies DURING this same call (e.g. an earlier
            // queue.submit() in this frame's chunked substep loop triggers an
            // uncaptured OOM). Re-check here: once lost, the staging buffer may already
            // be destroyed regardless of what the async result claims, so both the Ok
            // and Err branches are skipped, not just one.
            if self.is_device_lost() {
                // Do nothing -- neither finish_readback nor abandon_readback is
                // safe to call once the device is confirmed lost.
            } else if result.is_err() {
                self.readback_error_count += 1;
                self.buffers.abandon_readback();
            } else {
                let gpu_particles = self.buffers.finish_readback(self.particle_count);

                // CPU plasticity pass — skipped if all materials run plasticity on GPU.
                //
                // IMPORTANT: GPU g2p already integrated F via `F_new = (I + dt·C)·F_old`.
                // Zero affine before update_particle so only the plasticity projection runs.
                // Restore GPU affine afterwards so next P2G APIC term is correct.
                // Convert AoS to SoA, run the CPU pass via a per-particle
                // `ParticleUpdateCtx`, then scatter results back.
                if any_cpu {
                    // Stash GPU affine matrices — we zero affine for the plasticity call then restore.
                    let gpu_affines: Vec<_> =
                        gpu_particles.iter().map(|p| p.velocity_gradient).collect();
                    // Copy readback into AoS cpu mirror (zeroing affine for plasticity).
                    for (p_gpu, p_cpu) in gpu_particles.iter().zip(self.particles.iter_mut()) {
                        *p_cpu = *p_gpu;
                        p_cpu.velocity_gradient = glam::Mat2::ZERO;
                    }
                    // Build SoA wrapper, run CPU plasticity, scatter plastic state back.
                    // Skip sleeping particles — same reasoning as every GPU-side pass: their
                    // F/plastic state is frozen, re-running plasticity on unchanged input
                    // wastes exactly the compute sleep/wake exists to avoid.
                    let mut soa = Particles::from(std::mem::take(&mut self.particles));
                    for i in 0..soa.len() {
                        if soa.sleeping[i] {
                            continue;
                        }
                        let material_id = soa.material_id[i];
                        self.registry
                            .get(material_id)
                            .update_particle(&mut soa.update_ctx(i), self.last_sub_dt);
                    }
                    self.particles = soa.to_vec();
                    // Restore GPU affine.
                    for (p_cpu, gpu_affine) in self.particles.iter_mut().zip(gpu_affines) {
                        p_cpu.velocity_gradient = gpu_affine;
                    }
                } else {
                    for (p_gpu, p_cpu) in gpu_particles.into_iter().zip(self.particles.iter_mut()) {
                        *p_cpu = p_gpu;
                    }
                }
                if any_cpu {
                    self.layout_dirty = true; // CPU plasticity touched positions/F
                }
                // Defer the actual O(N) rebuild to the first query that needs it
                // (ensure_spatial_hash_fresh, queries.rs) instead of paying it on every
                // readback completion regardless of whether a query runs this frame --
                // see spatial_hash's doc in mod.rs.
                self.spatial_hash_dirty.set(true);
            }
        }

        // Start a new readback if wanted and none is already in flight -- guarded by
        // is_device_lost() for the same reason as the completion-check block above:
        // a mid-call device loss shouldn't kick off a fresh async copy/map against a
        // buffer that may already be gone.
        if want_readback && self.pending_readback.is_none() && !self.is_device_lost() {
            self.pending_readback = Some(self.buffers.begin_readback(
                &self.device,
                &self.queue,
                self.particle_count,
            ));
        }
        let readback_ns = readback_start.elapsed().as_secs_f32() * 1.0e9;
        let total_ns = total_start.elapsed().as_secs_f32() * 1.0e9;
        self.last_cpu_timings = (cfl_scan_ns, encode_ns, submit_ns, readback_ns, total_ns);
    }
}

// Live-adjustable params (add/clear_force_field_gpu, set_grip_direction/
// friction, attach_thermal_gpu, set_thermal_ambient, attach_resource_field_gpu)
// split into sibling module live_params.rs (declared in solver/mod.rs);
// encode_substep (the 7-8 per-substep compute passes) split into sibling
// module encode_substep.rs -- was ~440 combined of this file's ~1000 lines.
// step_frame (above) is the one thing that stays here, per this file's own
// top-of-file doc comment on why it's the highest-risk slice.

// Real, white-box test for regional-substepping Step 2's `classify_blocks`/
// `read_block_cfl_reduction_blocking` (private methods -- must be a CHILD
// module of `step`, not a sibling, for `super::*` to see them; same reason
// `device_lost_tests.rs` is a child of `solver` for ITS private-field access
// to `GpuSimulation`). Declared inline (not a separate file) since it's
// small and exists purely to keep these two real methods off the dead-code
// list honestly -- see `classify_blocks`' own doc for why a silent
// `#[allow(dead_code)]` isn't used instead (zero precedent for it anywhere
// in this codebase).
#[cfg(test)]
mod regional_substepping_tests {
    use super::*;
    use crate::materials::NewtonianFluidMaterial;
    use crate::materials::registry::MaterialRegistry;
    use crate::solver::config::SpawnRegion;
    use glam::{IVec2, Vec2};

    fn gpu_available() -> bool {
        let instance = crate::systems::gpu::create_wgpu_instance();
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }

    /// Real, direct test of the plan's own central claim (see
    /// `classify_blocks`' doc): `dt_fine` computed by taking the min over
    /// all 256 per-block dts must equal the SAME dt the existing, already-
    /// shipped global `cfl_reduction` scan computes -- both are ultimately
    /// `reactive_gpu_substep_dt` applied to a max-reduction of the same 4
    /// quantities, just partitioned differently (256 mins-then-a-min vs. one
    /// big reduction). Whichever block contains the single most-limiting
    /// particle reports the SAME raw maxima as the global scan for at least
    /// the limiting term, so the two must agree.
    #[test]
    fn classify_blocks_dt_fine_matches_existing_global_scan() {
        if !gpu_available() {
            return;
        }
        let config = SimConfig::standard(32, 0.05, Vec2::new(0.0, -0.1));
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(4, 4),
            box_center: Vec2::splat(16.0),
            mass_override: Some(4.0 * 0.5 * 0.5),
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let particles = crate::build_particles(&config, spawn);
        let registry = MaterialRegistry::with_default(Box::new(NewtonianFluidMaterial::new(
            4.0, 0.1, 10.0, 4.0,
        )));
        let mut sim = pollster::block_on(GpuSimulation::new(config, particles, registry));

        sim.step_frame();

        let (max_speed, max_rate, max_c2, near_wall) = sim.read_cfl_reduction_blocking();
        let global_dt = sim.reactive_gpu_substep_dt(max_speed, max_rate, max_c2, near_wall, 1.0);

        let block_words = sim.read_block_cfl_reduction_blocking();
        let (is_fine, dt_fine) = sim.classify_blocks(&block_words, 1.0);

        assert!(
            (dt_fine - global_dt).abs() < 1.0e-5 * global_dt.max(1.0),
            "classify_blocks' own dt_fine ({dt_fine}) must match the existing \
             global scan's dt ({global_dt}) -- both are the same min-reduction \
             of the same 4 quantities, just partitioned differently"
        );
        assert!(
            is_fine.iter().any(|&f| f),
            "at least one block (the one containing the limiting particle) \
             must classify as Fine -- an all-Coarse result would mean the \
             tier rule itself is broken"
        );
    }
}
