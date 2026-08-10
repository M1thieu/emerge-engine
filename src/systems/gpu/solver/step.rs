//! The actual per-frame GPU dispatch: `step_frame` (CFL scan, uploads, encode,
//! submit, async readback) and `encode_substep` (the 7 labeled compute passes).
//!
//! Split out of `gpu/solver/mod.rs` -- the highest-risk slice: everything here
//! touches live wgpu device/buffer state and timing-sensitive submit/poll
//! ordering (see the OOM/substep-batching and active-block-grace-period
//! comments below).

use super::super::step_params::{
    GpuFieldsParams, GpuImpulseParams, GpuSleepWakeParams, GpuStepParams,
};
use super::encode_substep::SubstepGates;
use super::{GpuSimulation, WG_PARTICLES, build_bind_group_pool};

use crate::particle::Particles;
use crate::solver::config::SimConfig;
use crate::solver::{affine_cfl_speed_contribution, cfl_bound, deformation_gradient_cfl_bound};

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
            const STRICT_FLUID_SUBSTEP_BATCH_SIZE: usize = 8;
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
            let mut pool_capacity_hint = self.config.max_substeps_per_step;
            loop {
                if remaining_time <= 0.0 {
                    break;
                }
                assert!(
                    substeps_taken < STRICT_FLUID_SUBSTEP_SAFETY_CEILING,
                    "GPU strict fluid CFL never stabilized within {STRICT_FLUID_SUBSTEP_SAFETY_CEILING} substeps this frame; inspect the scene/material configuration"
                );

                // Plan and encode a batch of up to STRICT_FLUID_SUBSTEP_BATCH_SIZE
                // substeps into ONE command buffer, all sharing this batch's dt.
                self.buffers.clear_cfl_reduction(&self.queue);
                let chunk_encode_start = std::time::Instant::now();
                let mut sub_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("mpm_substep_strict_fluid_batch"),
                        });
                let mut batch_len = 0usize;
                while batch_len < STRICT_FLUID_SUBSTEP_BATCH_SIZE && remaining_time > 0.0 {
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
                    );
                    if substeps_taken >= pool_capacity_hint {
                        pool_capacity_hint =
                            pool_capacity_hint.saturating_mul(2).max(substeps_taken + 1);
                        if self
                            .buffers
                            .ensure_step_param_capacity(&self.device, pool_capacity_hint)
                        {
                            self.bind_group_pool =
                                build_bind_group_pool(&self.device, &self.pipelines, &self.buffers);
                        }
                    }
                    self.buffers
                        .upload_step_params_at(&self.queue, substeps_taken, &params);
                    last_used_sub_dt = sub_dt;
                    remaining_time -= sub_dt;

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
                assert!(
                    status[0] == 0,
                    "GPU strict fluid update became inadmissible (code {}, particle {}, events {}); reduce the timestep or inspect the applied force/state",
                    status[0],
                    status[1],
                    status[2],
                );

                if remaining_time <= 0.0 {
                    break;
                }
                let (max_speed, max_rate, max_c2, near_wall) = self.read_cfl_reduction_blocking();
                sub_dt_cfl = self.reactive_gpu_substep_dt(
                    max_speed,
                    max_rate,
                    max_c2,
                    near_wall,
                    remaining_time,
                );
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
                    );
                    self.buffers.upload_step_params_at(
                        &self.queue,
                        substeps_taken + batch_len,
                        &params,
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
