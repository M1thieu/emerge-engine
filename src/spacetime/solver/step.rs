//! The adaptive-substep physics step: CFL timestep selection, P2G, grid update,
//! G2P, force fields, thermal/scalar diffusion, phase rules, and sleep scoring.
//!
//! Split out of `solver/mod.rs` (was 1536 lines, doing 5-6 jobs in one file) --
//! this is the one piece that's purely "advance the simulation by one step,"
//! distinct from construction, queries, and particle-lifecycle management that
//! live alongside `Simulation` in the parent module. `do_substep`'s own body
//! stays a single ordering-sensitive sequence on purpose (see its inline
//! comments for why each phase must run where it does) -- only the two
//! genuinely self-contained pieces split further, into sibling files:
//! adaptive-timestep selection (`cfl.rs`) and per-substep NaN/invalid-state
//! guards (`projection.rs`).

use glam::Vec2;

use super::Simulation;
use super::cfl::choose_substep_dt;
use super::projection::{apply_boundary_conditions_to_grid, project_particle_state_to_admissible};
use crate::rod::{
    RodForceParams, RodImplicitStepParams, apply_bending_plasticity, apply_gravitropism,
    apply_growth, apply_phototropism, apply_rod_internal_and_wind_forces, apply_secondary_growth,
    gather_grid_to_rod, scatter_rod_to_grid, step_rod_implicit,
};
use crate::solver::density::estimate_particle_volumes;
use crate::transfer::{
    G2PParams, gather_contact_point_cloud, gather_grid_to_particles, scatter_particles_to_grid,
};

impl Simulation {
    /// One MLS-MPM timestep: particle→grid→particle cycle.
    /// The grid is temporary scratch — only particles hold long-term material memory.
    pub fn step(&mut self) {
        // Adaptive substep loop: step() always advances exactly config.dt of simulation time,
        // but uses smaller sub-steps when CFL requires it (stiff materials, high velocities).
        // Without this loop, the FixedStepController accounts for config.dt per call but the
        // simulation only advances sub_dt — causing it to run orders of magnitude too slowly.
        let step_start = std::time::Instant::now();
        let mut remaining = self.config.dt;
        let mut substeps_taken = 0;
        self.last_vel_clamp_count = 0;
        self.last_j_projection_count = 0;
        self.last_timing = crate::diagnostics::StepTiming::default();

        // Implicit-integration rods (Baraff & Witkin 1998, see
        // `rod::implicit` doc): advanced ONCE per `step()` at the full
        // frame dt, outside the substep loop. Gravity/wind/push stay inside
        // this solve, not on the shared grid -- `Grid::apply_gravity` is a
        // plain explicit `v += g*dt`, only safe elsewhere because every other
        // body has its own CFL ceiling keeping dt small; an implicit rod has
        // none, so that path is unconditionally unstable at the full frame dt.
        // Not grid-coupled yet -- see the struct field's own doc.
        for rod in &mut self.rods {
            if !rod.use_implicit_integration {
                continue;
            }
            if rod.sleeping {
                // Wake on push/wind HERE, before the skip below: the grid-touch
                // wake check runs only after this loop, so waking there alone
                // would miss the same frame a push or wind is first applied.
                // Must check wind_velocity too, not just push_strength -- wind
                // never touches the grid, so it's the only way a sleeping rod
                // can wake from wind alone.
                let has_external_force = (rod.push_strength > 0.0 && rod.push_center.is_some())
                    || rod.wind_velocity.length_squared() > 0.0;
                if has_external_force {
                    rod.sleeping = false;
                    rod.below_threshold_time = 0.0;
                } else {
                    continue;
                }
            }
            // Splitting frame `dt` into smaller implicit substeps reduces backward
            // Euler's numerical damping, letting the rod's tuned damping ratio show
            // as visible sway instead of a smooth glide. Default 1 = prior behavior.
            let substeps = rod.implicit_substeps.max(1);
            let sub_dt = self.config.dt / substeps as f32;
            for _ in 0..substeps {
                step_rod_implicit(
                    &mut rod.points,
                    &rod.material,
                    RodImplicitStepParams {
                        gravity: self.config.gravity,
                        wind_velocity: rod.wind_velocity,
                        wind_drag_coeff: rod.wind_drag_coeff,
                        push_center: rod.push_center,
                        push_strength: rod.push_strength,
                        push_radius: rod.push_radius,
                        dx_meters: self.config.dx_meters,
                        dt: sub_dt,
                    },
                );
            }
            if let Some(gravitropism) = &rod.gravitropism {
                apply_gravitropism(
                    &mut rod.points,
                    gravitropism,
                    self.config.gravity,
                    &self.grid,
                    self.config.dt,
                );
            }
            if let Some(phototropism) = &rod.phototropism {
                apply_phototropism(
                    &mut rod.points,
                    phototropism,
                    self.config.light_dir,
                    &self.grid,
                    self.config.dt,
                );
            }
            if let Some(growth) = &mut rod.growth {
                apply_growth(
                    &mut rod.points,
                    growth,
                    &self.grid,
                    self.config.light_dir,
                    self.config.dx_meters,
                    self.config.dt,
                );
            }
            // Gated on the rod still being over-critical (same Greenhill buckling
            // gate as gravitropism/phototropism above, opposite direction):
            // unconditional secondary growth keeps stiffening the rod past the
            // point it's mechanically needed, making it progressively less
            // responsive to later pushes.
            if let Some(secondary_growth) = &rod.secondary_growth {
                let gravity_si = self.config.gravity.length() * self.config.dx_meters;
                if rod.buckling_warning(gravity_si).is_some() {
                    apply_secondary_growth(
                        &mut rod.points,
                        secondary_growth,
                        self.config.dx_meters,
                        self.config.dt,
                    );
                }
            }
            // Real elastic-perfectly-plastic bending (see `plasticity`
            // module doc) -- applied AFTER any biological reshaping above,
            // so mechanical yield acts on top of whatever active tropism/
            // growth already did this step, not instead of it.
            if let Some(plasticity) = &rod.plasticity {
                apply_bending_plasticity(&mut rod.points, plasticity, self.config.dx_meters);
            }
        }
        while remaining > f32::EPSILON && substeps_taken < self.config.max_substeps_per_step {
            // Cap sub-step at remaining time so we don't overshoot the configured frame dt.
            let t_cfl = std::time::Instant::now();
            let sub_dt = choose_substep_dt(
                &self.config,
                &self.particles,
                self.active_count,
                &self.materials,
                &self.rods,
                remaining,
                self.granular_fluidity
                    .as_ref()
                    .map(|f| f.config.stability_dt(self.config.dx_meters)),
            );
            self.last_timing.cfl_us += t_cfl.elapsed().as_micros() as u64;
            self.do_substep(sub_dt);
            remaining -= sub_dt;
            self.last_step_dt = sub_dt;
            substeps_taken += 1;
        }
        self.last_substeps = substeps_taken;
        self.last_sim_time_dropped = remaining.max(0.0);
        // Rebuild once per step, not per substep — LP queries happen between step() calls,
        // never mid-substep, so one rebuild after the loop is sufficient and correct.
        let t_hash = std::time::Instant::now();
        self.spatial_hash
            .rebuild(&self.particles.x, self.active_count);
        self.last_timing.spatial_hash_us = t_hash.elapsed().as_micros() as u64;
        self.last_timing.total_us = step_start.elapsed().as_micros() as u64;
        self.frame_index = self.frame_index.saturating_add(1);
    }

    fn do_substep(&mut self, sub_dt: f32) {
        // Project invalid particle state before it can corrupt the grid scatter.
        // Running pre-P2G (not post) means a bad particle from a previous substep is
        // fixed before its momentum enters the grid — no NaN cascade possible.
        let t_pre = std::time::Instant::now();
        if self.config.project_invalid_state {
            for i in 0..self.active_count {
                if project_particle_state_to_admissible(&mut self.particles, i, &self.config) {
                    self.last_j_projection_count += 1;
                }
            }
        }
        self.last_timing.project_us += t_pre.elapsed().as_micros() as u64;

        // Density recompute: fluid EOS materials need current ρ each substep (pressure = f(ρ)).
        // Auto-enabled when any registered material declares needs_density_recompute=true.
        // Manual override via config.recompute_density_each_step for edge cases.
        let t_density = std::time::Instant::now();
        if self.config.recompute_density_each_step || self.materials.any_needs_density_recompute() {
            estimate_particle_volumes(
                &mut self.particles,
                &mut self.grid,
                Some(&self.materials),
                self.active_count,
                false,
            );
        }
        self.last_timing.density_us += t_density.elapsed().as_micros() as u64;

        // ── P2G ──────────────────────────────────────────────────────────────
        let t0 = std::time::Instant::now();
        self.grid.clear();
        scatter_particles_to_grid(
            &self.particles,
            &mut self.grid,
            &self.materials,
            sub_dt,
            self.active_count,
        );
        // Second particle pass for the contact-normal point cloud (see
        // `gather_contact_point_cloud` doc) -- must run after the above, since
        // contact-active nodes aren't fully known until every grip particle's mass
        // has been scattered. No-op when `contact_group` is unused anywhere.
        gather_contact_point_cloud(&self.particles, &mut self.grid, self.active_count);
        // Rod -> grid scatter, same P2G pass, same shared `Grid` -- BEFORE the
        // wake pass below so a rod touching settled sand/fluid wakes it with
        // zero new code (the wake scan just sees active cells the rod itself
        // created). No-op for every scene that never calls add_rod/with_rod.
        // Sleeping rods skip this entirely (see `Rod::sleeping` doc) -- they
        // neither scatter mass/momentum nor self-trigger their own wake check
        // below; they're woken only by genuinely external activity.
        //
        for rod in &self.rods {
            if !rod.sleeping && !rod.use_implicit_integration {
                scatter_rod_to_grid(&rod.points, &mut self.grid);
            }
        }
        self.last_timing.p2g_us += t0.elapsed().as_micros() as u64;

        // Wake any sleeping particle whose kernel overlaps a MEANINGFULLY active
        // grid cell. This propagates activity from moving regions into
        // neighbouring sleeping ones without a separate O(N) scan — we only
        // visit the sleeping partition.
        //
        // Must gate on the neighbour's actual velocity, not just `cell_is_active`
        // (has ANY mass, regardless of speed): a body that scatters into the grid
        // every substep but never itself goes to sleep (e.g. a rod gated awake by
        // ongoing `Growth`, see `Rod::is_growing`) would otherwise count as
        // permanent "activity" for every neighbour touching its cells, even once
        // its own residual speed is tiny — causing spurious sleep/wake cycling.
        // Requiring the neighbour's actual velocity (momentum/mass — `cell.momentum`
        // is still RAW scattered momentum at this point in the substep, before
        // `update_velocities` normalizes it) to exceed THIS body's own sleep
        // threshold gives the same hysteresis a "settled" body already assumes:
        // something merely present but equally quiescent shouldn't wake it up.
        let wake_speed_sq = self.config.sleep_threshold * self.config.sleep_threshold;
        if self.active_count < self.particles.len() {
            let total = self.particles.len();
            self.scratch_indices.clear();
            for i in self.active_count..total {
                let x = self.particles.x[i];
                let base = crate::grid::kernel::quadratic_weights(x).base_cell;
                'outer: for gx in 0i32..3 {
                    for gy in 0i32..3 {
                        let cell = base + glam::IVec2::new(gx - 1, gy - 1);
                        let mass = self.grid.mass_at(cell);
                        if mass <= 0.0 {
                            continue;
                        }
                        let speed_sq = (self.grid.velocity_at(cell) / mass).length_squared();
                        if speed_sq > wake_speed_sq {
                            self.scratch_indices.push(i);
                            break 'outer;
                        }
                    }
                }
            }
            // Index directly — wake_particle doesn't touch scratch_indices, capacity preserved.
            for j in 0..self.scratch_indices.len() {
                let i = self.scratch_indices[j];
                self.wake_particle(i);
            }
        }

        // Same wake test, rod granularity: a sleeping rod's own (frozen) points
        // didn't scatter above, so any overlap found here comes from genuinely
        // external activity (another body's P2G, or another awake rod) -- the
        // particle wake pass's no-self-trigger property. Also wakes unconditionally
        // on an active push, since `push_strength > 0` is a direct move request the
        // grid-activity test can't see yet (nothing has touched the grid near it).
        // Requires actual velocity over rod_sleep_threshold, not merely "some mass
        // present" -- otherwise a permanently-active neighbour (e.g. a growing root
        // that never sleeps) keeps waking every sleeping rod touching its cells.
        // Must also check wind_velocity, not just push_strength: wind is
        // rod-internal and never touches the grid, so it's the only way a sleeping
        // rod can wake from wind alone (same as the implicit-rod path above).
        let rod_wake_speed_sq = self.config.rod_sleep_threshold * self.config.rod_sleep_threshold;
        for rod in &mut self.rods {
            if !rod.sleeping {
                continue;
            }
            let has_push = rod.push_strength > 0.0 && rod.push_center.is_some();
            let has_wind = rod.wind_velocity.length_squared() > 0.0;
            let touched = has_push
                || has_wind
                || rod.points.x.iter().any(|&x| {
                    let base = crate::grid::kernel::quadratic_weights(x).base_cell;
                    (0i32..3).any(|gx| {
                        (0i32..3).any(|gy| {
                            let cell = base + glam::IVec2::new(gx - 1, gy - 1);
                            let mass = self.grid.mass_at(cell);
                            if mass <= 0.0 {
                                return false;
                            }
                            let speed_sq = (self.grid.velocity_at(cell) / mass).length_squared();
                            speed_sq > rod_wake_speed_sq
                        })
                    })
                });
            if touched {
                rod.sleeping = false;
                rod.below_threshold_time = 0.0;
            }
        }

        // ── Grid update ───────────────────────────────────────────────────────
        let t1 = std::time::Instant::now();
        // ASFLIP (SimConfig::asflip_blend, Fei et al. 2021) needs the grid's velocity
        // right after P2G's own momentum normalization -- before THIS substep's gravity,
        // boundary conditions, or contact resolution modify it -- to compute G2P's FLIP
        // residual. Cundall damping (SimConfig::cundall_damping) needs the exact same
        // pre-force reference point -- see `Grid::apply_cundall_damping`'s own doc --
        // so both features share one snapshot. Taking it only when either feature is
        // enabled keeps every other scene on the exact original single-call path (zero
        // cost, zero behavior change).
        let pre_force_snapshot =
            if self.config.asflip_blend > 0.0 || self.config.cundall_damping > 0.0 {
                self.grid.normalize_velocities();
                let snapshot = self.grid.snapshot_velocities();
                self.grid.apply_gravity(sub_dt, self.config.gravity);
                Some(snapshot)
            } else {
                self.grid.update_velocities(sub_dt, self.config.gravity);
                None
            };
        let grid_res = self.grid.resolution();
        for boundary in &self.boundaries {
            apply_boundary_conditions_to_grid(&mut self.grid, grid_res, boundary.as_ref());
        }
        // Clamp grid velocity before G2P — bounds both v_p and C_p at the source.
        // Post-G2P clamping misses C_p: large C_p → F = (I + dt·C)·F blows up → J→0.
        let vel_limit = self.config.grid_cell_size / sub_dt;
        {
            for cell in self.grid.active_cells_mut() {
                if cell.mass > 0.0 {
                    let spd = cell.momentum.length();
                    if spd > vel_limit {
                        cell.momentum *= vel_limit / spd;
                    }
                }
            }
        }
        // Multi-field frictional contact (Bardenhagen 2001) — AFTER the clamp above, so
        // the grip/rest split is resolved against an already-safe total, and passed
        // `vel_limit` to apply the SAME clamp to the grip field's own raw velocity and
        // to both resolved outputs (a tiny-mass grip node could otherwise carry a huge
        // raw velocity even when the total is fine). No-op (no dirty contact cells) for
        // every scene that never sets `Particle::contact_group` — see
        // `Grid::resolve_contact` doc.
        self.grid.resolve_contact(
            sub_dt,
            self.config.gravity,
            self.config.contact_friction,
            vel_limit,
            self.config.grid_cell_size,
            self.contact_grip.as_deref(),
        );
        // Two-phase mixture coupling (Tampubolon et al. 2017) — same "after the
        // clamp, no-op when unused" positioning as contact above. No-op (no dirty
        // mixture cells) for every scene that never uses `WithMixturePhase` — see
        // `Grid::resolve_mixture_coupling` doc.
        self.grid.resolve_mixture_coupling(
            sub_dt,
            self.config.gravity,
            self.config.mixture_drag_coefficient,
            self.config.grid_cell_size,
            self.config.mixture_pressure_iterations,
        );
        // Cundall damping (see the snapshot comment above) -- applied LAST, after
        // gravity/boundary/contact/mixture have all had their say, so it damps the
        // real NET result of everything this substep, not just one contributor.
        if self.config.cundall_damping > 0.0
            && let Some(snapshot) = &pre_force_snapshot
        {
            self.grid
                .apply_cundall_damping(snapshot, self.config.cundall_damping);
        }
        self.last_timing.grid_update_us += t1.elapsed().as_micros() as u64;

        // ── G2P ──────────────────────────────────────────────────────────────
        let t2 = std::time::Instant::now();
        let g_len = self.active_count.min(self.granular_fluidity_g.len());
        self.last_vel_clamp_count += gather_grid_to_particles(
            &mut self.particles,
            &self.grid,
            sub_dt,
            &self.boundaries,
            &self.materials,
            G2PParams {
                vel_limit: self.config.grid_cell_size / sub_dt,
                apic_blend: self.config.apic_blend,
                active_count: self.active_count,
                asflip_blend: self.config.asflip_blend,
                // Real, honest, minor shared cost: if only `cundall_damping` is enabled
                // (asflip_blend still 0.0), G2P still takes the `Some` branch and computes
                // the extra pre-force stencil gather -- harmless (asflip_blend=0.0 zeroes
                // its own contribution exactly) but not free. Reusing one snapshot for both
                // features beats duplicating the mechanism; this is the real tradeoff.
                pre_force_snapshot: pre_force_snapshot.as_ref(),
                // Computed at the END of the PREVIOUS substep, by the
                // granular-fluidity pass alongside thermal/scalar diffusion
                // below -- same one-substep-lag convention those already
                // use. Empty when no `GranularFluidityField` is configured
                // for this scene (every existing scene) -- every read falls
                // back to 0.0, `ParticleUpdateCtx::nonlocal_fluidity`'s own
                // real-rest-state default.
                nonlocal_fluidity: &self.granular_fluidity_g[..g_len],
            },
        );
        // Grid -> rod gather (this rod's own G2P): pulls velocity (gravity
        // already baked in via the shared grid-update step above) AND
        // advances `rod.points.x`, mirroring `gather_grid_to_particles`'s own
        // position-advection contract exactly (see `coupling::gather_grid_to_rod`'s
        // doc) so rod force integration below only ever touches velocity,
        // matching how particle force fields never touch `particles.x` either.
        for rod in &mut self.rods {
            if !rod.sleeping && !rod.use_implicit_integration {
                gather_grid_to_rod(&mut rod.points, &self.grid, sub_dt);
            }
        }
        self.last_timing.g2p_us += t2.elapsed().as_micros() as u64;

        // ── Force fields ──────────────────────────────────────────────────────
        // External body force fields: v += dt × acceleration(p) per particle.
        // Applied after G2P so each field sees the fully gathered particle state.
        // A post-field velocity clamp (same limit as G2P) prevents large impulses from
        // leaving particles with >1 cell/substep velocity that P2G then scatters as extreme
        // momentum — the clamp re-asserts the CFL contract after external perturbation.
        // prepare() is called first so stateful fields (e.g. Barnes-Hut tree) can
        // rebuild their internal state from the current particle snapshot.
        if !self.force_fields.is_empty() {
            let t3 = std::time::Instant::now();
            let mut fields = std::mem::take(&mut self.force_fields);
            for (_, field) in &mut fields {
                field.prepare(&self.particles);
            }
            for i in 0..self.active_count {
                // Dirichlet/kinematic anchor (`Particle::pinned`): must stay at v=0,
                // matching G2P's own unconditional pinned branch just before this pass.
                // Force fields ran AFTER G2P with no pinned check, silently un-zeroing
                // pinned particles' velocity every substep -- P2G then scatters that as
                // real momentum next substep (`scatter_particles_to_grid` doesn't special-
                // case pinned particles either, since a pinned particle's mass/stress
                // SHOULD still be felt by neighbors, just not its velocity). A supposedly-
                // fixed anchor was quietly injecting wind-driven momentum into the grid
                // every substep -- a real, confirmed root cause of long-horizon energy
                // injection at every pinned+force-field composition, not just this scene.
                if self.particles.pinned[i] != 0 {
                    continue;
                }
                let mut dv = Vec2::ZERO;
                for (_, field) in &fields {
                    dv += field.acceleration(&self.particles, i);
                }
                self.particles.v[i] += sub_dt * dv;
            }
            self.force_fields = fields;
            // Re-clamp velocity after force fields — large external impulses (explosions,
            // creature bursts, planetary impacts) must not enter P2G with >1 cell/substep.
            let vel_limit = self.config.grid_cell_size / sub_dt;
            for i in 0..self.active_count {
                if self.particles.pinned[i] != 0 {
                    continue;
                }
                let spd = self.particles.v[i].length();
                if spd > vel_limit {
                    self.particles.v[i] *= vel_limit / spd;
                }
            }
            self.last_timing.fields_us += t3.elapsed().as_micros() as u64;
        }

        // ── Rod internal + wind forces ──────────────────────────────────────────
        // Runs where particle force fields just ran, on the SAME real convention:
        // velocity-only (position already advanced in the gather above), so a
        // rod's own stretch/bend/damping + wind drag land exactly like an
        // ordinary force field would. Gravity is NOT reapplied here — the rod
        // already received it via the shared grid-update step, same mechanism
        // ordinary particles use. No-op for every scene with no rods.
        //
        for rod in &mut self.rods {
            if rod.sleeping || rod.use_implicit_integration {
                continue;
            }
            apply_rod_internal_and_wind_forces(
                &mut rod.points,
                &rod.material,
                RodForceParams {
                    wind_velocity: rod.wind_velocity,
                    wind_drag_coeff: rod.wind_drag_coeff,
                    push_center: rod.push_center,
                    push_strength: rod.push_strength,
                    push_radius: rod.push_radius,
                    dx_meters: self.config.dx_meters,
                    dt: sub_dt,
                },
            );
            // Real root gravitropism (Porat, Rivière, Meroz 2024 -- see
            // `rod::gravitropism` module doc): evolves the tip's own
            // rest_curvature toward gravity-alignment. No-op for every rod
            // that doesn't opt in (plain stems/blades don't grow toward
            // gravity).
            if let Some(gravitropism) = &rod.gravitropism {
                apply_gravitropism(
                    &mut rod.points,
                    gravitropism,
                    self.config.gravity,
                    &self.grid,
                    sub_dt,
                );
            }
            // Real phototropism (Cholodny & Went auxin-asymmetry theory --
            // see `rod::gravitropism` module doc's own "Phototropism reuses
            // the SAME core" section). No-op for every rod that doesn't
            // opt in.
            if let Some(phototropism) = &rod.phototropism {
                apply_phototropism(
                    &mut rod.points,
                    phototropism,
                    self.config.light_dir,
                    &self.grid,
                    sub_dt,
                );
            }
            // Real elongation growth (Verhulst 1838 logistic law -- see
            // `rod::growth` module doc). No-op for every rod that doesn't
            // opt in.
            if let Some(growth) = &mut rod.growth {
                apply_growth(
                    &mut rod.points,
                    growth,
                    &self.grid,
                    self.config.light_dir,
                    self.config.dx_meters,
                    sub_dt,
                );
            }
            // Real stress-driven secondary growth (Jaffe 1973, Mattheck &
            // Kübler 1995 -- see `rod::secondary_growth` module doc). No-op
            // for every rod that doesn't opt in. Gated on the rod STILL
            // being over-critical -- see the other call site's own doc for
            // the real, measured bug this fixes (unbounded stiffening long
            // past the point it was actually needed).
            if let Some(secondary_growth) = &rod.secondary_growth {
                let gravity_si = self.config.gravity.length() * self.config.dx_meters;
                if rod.buckling_warning(gravity_si).is_some() {
                    apply_secondary_growth(
                        &mut rod.points,
                        secondary_growth,
                        self.config.dx_meters,
                        sub_dt,
                    );
                }
            }
            // Real elastic-perfectly-plastic bending (see `rod::plasticity`
            // module doc) -- same ordering rationale as the implicit branch's
            // own call site: mechanical yield applies on top of whatever
            // biological reshaping already happened this substep.
            if let Some(plasticity) = &rod.plasticity {
                apply_bending_plasticity(&mut rod.points, plasticity, self.config.dx_meters);
            }
        }

        // ── Thermal / scalar diffusion ────────────────────────────────────────
        let t4 = std::time::Instant::now();
        if let Some(thermal) = &mut self.thermal {
            thermal.apply(&mut self.particles, sub_dt);
        }
        for field in &mut self.scalar_fields {
            field.apply(&mut self.particles, sub_dt);
        }
        // Nonlocal Granular Fluidity (see `energy::thermodynamics::
        // granular_fluidity` module doc) -- same one-substep-lag placement
        // as thermal/scalar diffusion above: computed here from THIS
        // substep's just-updated particle stress state, read by next
        // substep's G2P via `G2PParams::nonlocal_fluidity`.
        if let Some(field) = &mut self.granular_fluidity {
            if self.granular_fluidity_g.len() < self.active_count {
                self.granular_fluidity_g.resize(self.active_count, 0.0);
            }
            field.apply(
                &self.particles,
                sub_dt,
                &mut self.granular_fluidity_g[..self.active_count],
            );
        }
        self.last_timing.thermal_us += t4.elapsed().as_micros() as u64;

        // ── Phase rules + sleep scoring ───────────────────────────────────────
        let t5 = std::time::Instant::now();
        if !self.phase_rules.is_empty() {
            let rules = std::mem::take(&mut self.phase_rules);
            let heat_capacity = self.thermal.as_ref().map(|t| t.config.heat_capacity);
            for i in 0..self.active_count {
                let p = self.particles.get(i);
                for rule in &rules {
                    if let Some(new_id) = rule(&p) {
                        self.particles.material_id[i] = new_id;
                        let latent_heat = self.materials.get(new_id).latent_heat();
                        if let (true, Some(cp)) = (latent_heat != 0.0, heat_capacity) {
                            self.particles.temperature[i] -= latent_heat / cp;
                        }
                        // See `Simulation::phase_transition`'s doc: reset
                        // material-specific plastic state to the new material's
                        // own defaults instead of silently inheriting the old
                        // material's stale values under a different meaning.
                        let mut p = self.particles.get(i);
                        self.materials.get(new_id).init_particle(&mut p);
                        self.particles.set(i, p);
                        break;
                    }
                }
            }
            self.phase_rules = rules;
        }
        let threshold = self.config.sleep_threshold;
        if threshold > 0.0 {
            let threshold_sq = threshold * threshold;
            self.scratch_indices.clear();
            self.scratch_indices
                .extend((0..self.active_count).filter(|&i| {
                    self.particles.activation[i] == 0.0
                        && self.particles.v[i].length_squared() < threshold_sq
                }));
            // Descending order: sleep_particle swaps i↔last_active (high end of active zone).
            // Processing high-to-low ensures each displacement lands in already-processed
            // positions, so no sleeping candidate is accidentally skipped.
            self.scratch_indices.sort_unstable_by(|a, b| b.cmp(a));
            for j in 0..self.scratch_indices.len() {
                self.sleep_particle(self.scratch_indices[j]);
            }
        }
        // Rod sleep scoring: same threshold-crossing test as particles above,
        // but scored over the WHOLE rod (max point speed) since points are
        // elastically coupled -- one point can't sleep while its neighbor
        // keeps swinging. Never sleeps mid-push (`push_strength > 0`), since
        // that's a live interaction the caller is actively driving.
        //
        // Must sleep on a sustained duration below threshold, not the instant
        // `max_speed_sq < threshold_sq`: a freshly-constructed rod trivially
        // satisfies that (`v = Vec2::ZERO` at birth) before gravity/grid coupling
        // gets a chance to act within one tiny substep, so it could fall asleep on
        // its very first substep, then skip its own gravity entirely while
        // "asleep" until external activity woke it -- receiving the entire
        // deferred gravitational transient at once as an unphysical velocity
        // spike. Every major real-time physics engine (Box2D's documented
        // `b2_timeToSleep = 0.5s`, Bullet, PhysX) requires staying below threshold
        // for a minimum duration, not one instant, for exactly this reason.
        //
        // Real root-cause fix (user-reported "never settles straight",
        // headlessly confirmed): a FIXED settle-duration (this used to be a
        // single constant, 0.5s) is wrong for ANY rod whose own natural
        // period is comparable to or longer than that fixed window. A
        // rod's velocity genuinely dips near zero at every swing peak, not
        // just at true rest -- if the fixed window is short enough relative
        // to the period, the sustained-below-threshold requirement can
        // complete DURING a single slow peak of a still-large-amplitude
        // swing, freezing the rod there at a real, wrong, off-rest position.
        // A soft demo blade (period ~0.53s) froze
        // several cells from true vertical rest at a fixed 0.5s window, and
        // even bumping that fixed constant up only shifts the same failure
        // to an even slower rod -- the real fix is SCALING the window to
        // each rod's OWN period, not picking a bigger universal constant.
        // `ROD_SLEEP_SETTLE_PERIODS=3.0`: three full natural periods of
        // sustained quiet is real headroom past any single swing peak's own
        // dwell time, for any rod's own stiffness/mass. Clamped to
        // `[ROD_SLEEP_SETTLE_MIN_SECONDS, ROD_SLEEP_SETTLE_MAX_SECONDS]`:
        // the floor preserves the original anti-instant-sleep protection
        // above for a very stiff/fast rod (three periods of a very fast rod
        // could be under a millisecond); the ceiling keeps an extremely
        // soft/slow rod from waiting an impractically long real time.
        const ROD_SLEEP_SETTLE_PERIODS: f32 = 3.0;
        const ROD_SLEEP_SETTLE_MIN_SECONDS: f32 = 0.3;
        const ROD_SLEEP_SETTLE_MAX_SECONDS: f32 = 8.0;
        let rod_threshold = self.config.rod_sleep_threshold;
        if rod_threshold > 0.0 {
            let threshold_sq = rod_threshold * rod_threshold;
            for rod in &mut self.rods {
                if rod.sleeping
                    || rod.push_strength > 0.0
                    || rod.is_growing()
                    || rod.is_correcting_gravitropically(self.config.gravity, &self.grid)
                    || rod.is_correcting_phototropically(self.config.light_dir, &self.grid)
                {
                    rod.below_threshold_time = 0.0;
                    continue;
                }
                let max_speed_sq = rod
                    .points
                    .v
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.length_squared()));
                if max_speed_sq < threshold_sq {
                    rod.below_threshold_time += sub_dt;
                    let period =
                        crate::rod::RodMaterial::fundamental_period_s(&rod.points, rod.material.ei);
                    let settle_seconds = (period * ROD_SLEEP_SETTLE_PERIODS)
                        .clamp(ROD_SLEEP_SETTLE_MIN_SECONDS, ROD_SLEEP_SETTLE_MAX_SECONDS);
                    if rod.below_threshold_time >= settle_seconds {
                        rod.sleeping = true;
                    }
                } else {
                    rod.below_threshold_time = 0.0;
                }
            }
        }
        self.last_timing.phase_sleep_us += t5.elapsed().as_micros() as u64;
    }

    pub fn effective_dt(&self) -> f32 {
        self.last_step_dt
    }

    pub fn last_substeps(&self) -> usize {
        self.last_substeps
    }

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

// apply_boundary_conditions_to_grid, project_particle_state_to_admissible: projection.rs
// choose_substep_dt, cfl_bound, affine_cfl_speed_contribution: cfl.rs
