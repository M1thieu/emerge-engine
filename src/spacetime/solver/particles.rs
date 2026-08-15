//! Phase rules, impulses, and per-particle lifecycle: spawn (`add_body`),
//! removal, sleep/wake, and tag bookkeeping.
//!
//! Split out of `solver/mod.rs` -- the remaining piece after construction
//! (`solver::lifecycle`), the step loop (`solver::step`), and read-only
//! queries (`solver::queries`) were pulled out. Everything here still
//! mutates the particle buffer between `step()` calls, unlike `queries`.

use std::collections::{HashMap, HashSet};

use glam::{Mat2, Vec2};

use super::{LcgRng, Simulation, SpawnRegion, density, initialize_particles};
use crate::particle::Particle;
use crate::solver::density::estimate_particle_volumes;
use crate::thermodynamics::ScalarDiffusionField;

impl Simulation {
    /// Real, shared phase-transition logic -- the single place both
    /// `phase_transition` (this file) and the per-substep `add_phase_rule`
    /// evaluation (`solver::step`) apply a material change, so the two
    /// can never drift apart on what a transition actually does.
    ///
    /// Real fix (2026-08-14) for a genuine "spring" artifact live-reported
    /// on a fluid->solid transition (e.g. the water->ice freeze rule): a
    /// particle arriving from a material with no real rest-shape memory (a
    /// fluid's `F` only ever encodes volume ratio, `sqrt(J)*I` -- see
    /// `NewtonianFluidMaterial`'s own doc) into a material that interprets
    /// `F` as elastic strain away from a rest configuration (any solid) had
    /// its leftover, almost-never-exactly-1.0 `J` misread as real elastic
    /// strain -- generating a spurious restoring stress that visibly
    /// oscillates instead of the particle freezing smoothly into its
    /// current shape. `MaterialModel::init_particle`'s default is a no-op
    /// and no existing solid material touches `deformation_gradient` there
    /// (confirmed by reading `CorotatedMaterial::init_particle` and the
    /// trait default), so nothing rebaselined it before this fix.
    ///
    /// Real physical justification, not a workaround: a genuine phase
    /// transition (water becoming ice, rock becoming lava) changes the
    /// material's actual physical structure, so whatever elastic strain
    /// existed relative to the OLD phase's reference configuration has no
    /// meaning in the NEW phase -- the new phase's zero-strain reference is
    /// wherever the particle physically IS at the instant of transition.
    /// Standard treatment for phase-transforming materials in computational
    /// solid mechanics, applied generically here rather than as a per-solid-
    /// material patch (this project's own standing preference for a single
    /// reusable mechanism over duplicated per-material boilerplate).
    ///
    /// Deliberately volume/density-CONTINUOUS, not a reset to some default:
    /// `initial_volume` is rebaselined to the particle's CURRENT volume (not
    /// its original spawn volume), so there is no discontinuous size jump
    /// at the instant of transition -- only the ELASTIC STRAIN memory is
    /// cleared, not the particle's real physical size. A material whose own
    /// `init_particle` subsequently redefines volume/density from ITS OWN
    /// rest_density (e.g. `NewtonianFluidMaterial`, for a melting
    /// transition) is free to do so -- that is a real, physically expected
    /// density change on phase transition (most real materials change
    /// density when they melt/freeze too), not a bug this rebaseline
    /// introduces.
    ///
    /// ## Real, disclosed roadmap toward a genuine Stefan-condition treatment
    ///
    /// This function (and every `add_phase_rule` predicate, e.g. "water
    /// freezes below 273K") implements phase change as an instantaneous,
    /// per-particle THRESHOLD -- a particle flips the instant its own local
    /// temperature crosses a fixed point, with no notion of how fast that
    /// transformation should actually happen. The real, governing PDE this
    /// approximates is the Stefan condition for a moving phase boundary:
    /// `L * rho * v_interface = -[k * grad(T)]` -- the interface velocity is
    /// set by the JUMP in heat flux across it, i.e. by how much MORE energy
    /// is leaving one side than entering the other. A threshold switch is
    /// exactly that condition's zeroth-order limit (`v_interface ->
    /// infinity`, the phase change treated as instantaneous once enough
    /// energy has crossed the threshold at all, regardless of RATE) -- a
    /// real, named simplification, not an unexamined one.
    ///
    /// What is ALREADY real and load-bearing in that equation, today: `L`
    /// (latent heat, debited below from `MaterialModel::latent_heat()`) and
    /// `k`/`grad(T)` (real thermal conductivity and gradient, already
    /// computed every substep by `ThermalDiffusion`, see
    /// `energy::thermodynamics::diffusion`). The missing piece is using
    /// `grad(T)` at the moment of transition to RATE-LIMIT the transition
    /// itself, instead of switching materials outright once `L` has been
    /// debited -- the real next increment, not attempted here. This is
    /// disclosed so it can be picked up as a genuine PDE-consistency
    /// upgrade to this exact function later, without re-deriving where the
    /// gap is.
    pub(super) fn apply_phase_transition(&mut self, i: usize, new_material_id: u32) {
        self.particles.material_id[i] = new_material_id;

        let current_volume = self.particles.volume[i];
        if current_volume.is_finite() && current_volume > 0.0 {
            self.particles.deformation_gradient[i] = Mat2::IDENTITY;
            self.particles.initial_volume[i] = current_volume;
            self.particles.density[i] = self.particles.mass[i] / current_volume;
        }

        let latent_heat = self.materials.get(new_material_id).latent_heat();
        if latent_heat != 0.0
            && let Some(thermal) = &self.thermal
        {
            self.particles.temperature[i] -= latent_heat / thermal.config.heat_capacity;
        }

        let mut p = self.particles.get(i);
        self.materials.get(new_material_id).init_particle(&mut p);
        self.particles.set(i, p);
    }

    /// Switch material for every particle where `predicate` returns true.
    ///
    /// Rebaselines each transitioned particle's elastic reference state to
    /// its current physical configuration and calls the new material's own
    /// `init_particle` -- see `apply_phase_transition`'s own doc for the
    /// real reasoning (this is not a cosmetic reset: without it, a solid
    /// material arriving from a fluid's leftover volumetric state visibly
    /// springs/oscillates instead of freezing smoothly). Otherwise a
    /// transitioned particle would also inherit stale material-specific
    /// plastic fields (`hardening_scale`, `friction_hardening`,
    /// `plastic_volume_ratio`, etc.) from its OLD material, reinterpreted
    /// under the new material's semantics (e.g. Rankine's damage
    /// accumulator read as Drucker-Prager's friction accumulator) --
    /// `init_particle` resets those to the new material's own defaults,
    /// exactly like `reinit_all_particle_state`/`add_body` do for every
    /// other material-assignment path.
    ///
    /// If `new_material_id`'s `MaterialModel::latent_heat()` is non-zero and a thermal
    /// model is configured (`with_thermal`/`set_thermal`), debits `temperature` by
    /// `latent_heat / heat_capacity` for every transitioned particle — see
    /// `MaterialModel::latent_heat` for the sign convention.
    pub fn phase_transition<F>(&mut self, predicate: F, new_material_id: u32)
    where
        F: Fn(&Particle) -> bool,
    {
        assert!(
            self.materials.is_registered(new_material_id),
            "phase_transition: material_id {new_material_id} is not registered — \
             call solver.with_material({new_material_id}, ...) first"
        );
        for i in 0..self.particles.len() {
            let p = self.particles.get(i);
            if predicate(&p) {
                self.apply_phase_transition(i, new_material_id);
            }
        }
    }

    /// Register an automatic phase transition rule, evaluated every substep.
    ///
    /// `rule` receives a particle and returns `Some(new_material_id)` to transition it,
    /// or `None` to leave it unchanged. Rules are checked in registration order;
    /// first match wins for each particle.
    ///
    /// All `new_material_id` values returned by the rule must be pre-registered via
    /// `solver.with_material(id, ...)` before any step is taken.
    ///
    /// Applies the same `latent_heat` energy debit as `phase_transition` — see there.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # extern crate emerge_engine as emerge;
    /// // Water freezes below 273 K
    /// solver.add_phase_rule(|p| {
    ///     if p.material_id == WATER_ID && p.temperature < 273.0 { Some(ICE_ID) } else { None }
    /// });
    /// // Rock melts above 1500 K
    /// solver.add_phase_rule(|p| {
    ///     if p.material_id == ROCK_ID && p.temperature > 1500.0 { Some(LAVA_ID) } else { None }
    /// });
    /// ```
    pub fn add_phase_rule<F>(&mut self, rule: F)
    where
        F: Fn(&Particle) -> Option<u32> + Send + Sync + 'static,
    {
        self.phase_rules.push(Box::new(rule));
    }

    /// Builder-style variant of `add_phase_rule`.
    pub fn with_phase_rule<F>(mut self, rule: F) -> Self
    where
        F: Fn(&Particle) -> Option<u32> + Send + Sync + 'static,
    {
        self.add_phase_rule(rule);
        self
    }

    /// Apply a velocity delta to all particles within `radius` of `center`, with linear falloff.
    /// `force` units: grid-cell/s (instantaneous velocity change).
    /// This applies the requested impulse exactly; it is not velocity-clamped.
    /// The next substep is selected from the resulting actual CFL state. Supply
    /// a physically meaningful impulse (or a resolved force history) rather
    /// than using this as a hidden settling/stability control.
    pub fn apply_impulse(&mut self, center: Vec2, radius: f32, force: Vec2) {
        let r2 = radius * radius;
        let mut to_wake = Vec::new();
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                if self.particles.sleeping[i] {
                    to_wake.push(i);
                }
                let falloff = 1.0 - (dist2 / r2).sqrt();
                self.particles.v[i] += force * falloff;
            }
        }
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    /// Apply an outward radial velocity delta to particles within `radius`, with linear falloff.
    /// This applies the requested radial impulse exactly; the next substep
    /// uses the resulting actual CFL state.
    pub fn apply_radial_impulse(&mut self, center: Vec2, radius: f32, strength: f32) {
        let r2 = radius * radius;
        let mut to_wake = Vec::new();
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                if self.particles.sleeping[i] {
                    to_wake.push(i);
                }
                let dist = dist2.sqrt();
                let falloff = 1.0 - dist / radius;
                self.particles.v[i] += (d / dist) * strength * falloff;
            }
        }
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    pub fn recompute_initial_volumes(&mut self) {
        estimate_particle_volumes(
            &mut self.particles,
            &mut self.grid,
            Some(&self.materials),
            self.active_count,
            true,
        );
    }

    /// Remove all particles where `predicate` returns true. Returns count removed.
    ///
    /// Uses stable retain (preserves order). O(N).
    ///
    /// LP pattern: tag particles with a sentinel before the step, then call
    /// `solver.remove_particles(|p| p.user_tag == DEAD)`. The tag-based group API
    /// remains valid after removal — tag_index is rebuilt internally.
    pub fn remove_particles<F: Fn(&Particle) -> bool>(&mut self, predicate: F) -> usize {
        let before = self.particles.len();
        self.particles.retain(|p| !predicate(p));
        let removed = before - self.particles.len();
        if removed > 0 {
            // retain() compacted the array — all physical indices in tag_index are stale.
            // Rebuild from scratch and re-establish the sleep partition.
            self.tag_index.clear();
            // Re-partition: move all sleeping particles to the back.
            let n = self.particles.len();
            let mut write = 0usize;
            for read in 0..n {
                if !self.particles.sleeping[read] {
                    if write != read {
                        self.particles.swap(write, read);
                    }
                    write += 1;
                }
            }
            self.active_count = write;
            // Rebuild tag_index over the freshly-partitioned array.
            for i in 0..n {
                self.tag_index
                    .entry(self.particles.user_tag[i])
                    .or_default()
                    .insert(i);
            }
        }
        removed
    }

    /// Iterate physical indices of all particles with `tag`. O(group_size) via tag_index.
    ///
    /// Returns indices only — read particle data via `solver.particles().x[i]` etc.
    /// This avoids cloning 112B per particle on every call.
    pub fn particles_with_tag(&self, tag: u32) -> impl Iterator<Item = usize> + '_ {
        self.tag_index
            .get(&tag)
            .into_iter()
            .flat_map(|s| s.iter())
            .copied()
    }

    // ── Sleep / wake ─────────────────────────────────────────────────────────

    /// Put particle at physical index `i` to sleep.
    ///
    /// Swaps it with the last active particle, decrementing `active_count`.
    /// Updates `tag_index` for both affected particles.
    /// No-op if already sleeping.
    pub(super) fn sleep_particle(&mut self, i: usize) {
        if self.particles.sleeping[i] || self.active_count == 0 {
            return;
        }
        self.particles.sleeping[i] = true;
        let last_active = self.active_count - 1;
        if i != last_active {
            let tag_i = self.particles.user_tag[i];
            let tag_j = self.particles.user_tag[last_active];
            // Same-tag swap: both indices stay in the same set — no update needed.
            // Different-tag swap: each particle moves to the other's former position.
            if tag_i != tag_j {
                Self::tag_index_replace(&mut self.tag_index, tag_i, i, last_active);
                Self::tag_index_replace(&mut self.tag_index, tag_j, last_active, i);
            }
            self.particles.swap(i, last_active);
        }
        self.active_count -= 1;
    }

    /// Wake particle at physical index `i`.
    ///
    /// Swaps it with the first sleeping particle, incrementing `active_count`.
    /// Updates `tag_index` for both affected particles.
    /// No-op if already awake.
    pub(super) fn wake_particle(&mut self, i: usize) {
        if !self.particles.sleeping[i] {
            return;
        }
        self.particles.sleeping[i] = false;
        let first_sleeping = self.active_count;
        if i != first_sleeping {
            let tag_i = self.particles.user_tag[i];
            let tag_j = self.particles.user_tag[first_sleeping];
            if tag_i != tag_j {
                Self::tag_index_replace(&mut self.tag_index, tag_i, i, first_sleeping);
                Self::tag_index_replace(&mut self.tag_index, tag_j, first_sleeping, i);
            }
            self.particles.swap(i, first_sleeping);
        }
        self.active_count += 1;
    }

    /// Wake all particles belonging to `tag`. O(group_size · log group_size).
    pub fn wake_tag(&mut self, tag: u32) {
        // Snapshot sleeping indices, then wake ascending.
        // wake_particle(i) swaps i↔first_sleeping (low end of sleeping zone).
        // Ascending order: each displacement lands at an already-processed lower position,
        // so no sleeping tag particle is silently displaced to an unvisited index.
        let mut to_wake: Vec<usize> = self
            .tag_index
            .get(&tag)
            .map(|s| {
                s.iter()
                    .filter(|&&i| self.particles.sleeping[i])
                    .copied()
                    .collect()
            })
            .unwrap_or_default();
        to_wake.sort_unstable();
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    /// Sleep all particles belonging to `tag`. O(group_size · log group_size).
    pub fn sleep_tag(&mut self, tag: u32) {
        // Snapshot active indices, then sleep descending.
        // sleep_particle(i) swaps i↔last_active (high end of active zone).
        // Descending order: each displacement lands at an already-processed higher position.
        let mut to_sleep: Vec<usize> = self
            .tag_index
            .get(&tag)
            .map(|s| {
                s.iter()
                    .filter(|&&i| !self.particles.sleeping[i])
                    .copied()
                    .collect()
            })
            .unwrap_or_default();
        to_sleep.sort_unstable_by(|a, b| b.cmp(a));
        for i in to_sleep {
            self.sleep_particle(i);
        }
    }

    /// Number of currently active (non-sleeping) particles.
    pub const fn active_count(&self) -> usize {
        self.active_count
    }

    fn tag_index_replace(
        tag_index: &mut HashMap<u32, HashSet<usize>>,
        tag: u32,
        old_idx: usize,
        new_idx: usize,
    ) {
        if let Some(s) = tag_index.get_mut(&tag) {
            s.remove(&old_idx);
            s.insert(new_idx);
        }
    }

    /// Collect all particles into a `Vec<Particle>` (for diagnostics or GPU upload).
    pub fn collect_particles(&self) -> Vec<Particle> {
        self.particles.to_vec()
    }

    /// Spawn particles, tag them, and return the stable tag.
    ///
    /// The returned `u32` tag is the only stable identity for this group — physical
    /// indices change whenever particles sleep or wake. Pass it to `group_state`,
    /// `set_group_activation`, `group_centroid`, etc.
    ///
    /// All particles in the region are stamped with `user_tag = tag` (overrides
    /// any `user_tag` set in the SpawnRegion).
    ///
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::solver::Simulation;
    /// # use emerge::{SimConfig, SpawnRegion};
    /// # let config = SimConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = Simulation::empty(config);
    /// let creature = solver.add_body(SpawnRegion::for_sim(&config));
    /// solver.set_group_activation(creature, 1.0);
    /// let centroid = solver.group_centroid(creature);
    /// ```
    #[must_use = "store the tag — it is the only stable identity for this group"]
    pub fn add_body(&mut self, spawn: SpawnRegion) -> u32 {
        let tag = self.next_tag;
        self.next_tag += 1;

        let old_active = self.active_count;
        let old_len = self.particles.len();
        // sleeping zone is [old_active..old_len] — new particles must land before it.

        spawn.validate_for_sim(&self.config);
        debug_assert!(
            self.materials.is_registered(spawn.material_id),
            "add_body: material_id {} is not registered",
            spawn.material_id,
        );
        let mut rng = LcgRng::new(spawn.rng_seed);
        let new_particles = initialize_particles(&self.config, spawn, &mut rng);
        for p in new_particles {
            self.particles.push(p);
        }
        let new_len = self.particles.len();
        let new_count = new_len - old_len;
        let sleeping_count = old_len - old_active;

        // Stamp tag and init material plastic state (new particles still at [old_len..new_len]).
        let mat_id = spawn.material_id;
        for i in old_len..new_len {
            self.particles.user_tag[i] = tag;
            if self.materials.is_registered(mat_id) {
                let mut p = self.particles.get(i);
                self.materials.get(mat_id).init_particle(&mut p);
                self.particles.set(i, p);
            }
        }

        // If sleeping particles sit between the active zone and the new particles, rotate new
        // particles before the sleeping zone so the partition invariant is maintained:
        //   before: [0..old_active] active | [old_active..old_len] sleeping | [old_len..new_len] new
        //   after:  [0..old_active] active | [old_active..old_active+new_count] new | [...] sleeping
        if sleeping_count > 0 {
            self.particles.rotate_range(old_active, old_len, new_len);
            // Sleeping particles moved from [old_active+k] → [old_active+new_count+k].
            // Update tag_index for each displaced sleeping particle.
            for k in 0..sleeping_count {
                let old_pos = old_active + k;
                let new_pos = old_active + new_count + k;
                let t = self.particles.user_tag[new_pos];
                Self::tag_index_replace(&mut self.tag_index, t, old_pos, new_pos);
            }
        }

        // New particles are at [old_active..old_active+new_count].
        let group_start = old_active;
        let group_end = old_active + new_count;
        self.tag_index
            .insert(tag, (group_start..group_end).collect::<HashSet<usize>>());
        self.active_count = group_end;

        // Scatter only particles in the spawn region + 3-cell margin.
        // O(active_count) scan but O(local × stencil) grid work — fast for sparse spawns.
        density::estimate_particle_volumes_local(
            &mut self.particles,
            &mut self.grid,
            Some(&self.materials),
            self.active_count,
            group_start,
            true,
        );
        self.spatial_hash
            .borrow_mut()
            .rebuild(&self.particles.x, self.active_count);
        self.spatial_hash_dirty.set(false);
        tag
    }

    /// Attach a scalar diffusion field (pheromone, nutrients, morphogen).
    ///
    /// Attached fields are applied automatically every substep — LP does not need
    /// to call `field.apply()` manually.
    ///
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::solver::Simulation;
    /// # use emerge::{SimConfig, SpawnRegion, ScalarDiffusionConfig, ScalarDiffusionField};
    /// # let config = SimConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = Simulation::new(config, SpawnRegion::default());
    /// let pheromone = ScalarDiffusionField::new(
    ///     ScalarDiffusionConfig { diffusivity: 0.5, decay_rate: 0.1, ambient: 0.0 },
    ///     |p| p.temperature,
    ///     |p, d| p.temperature += d,
    ///     64,
    /// );
    /// solver.attach_scalar_field(pheromone);
    /// // No manual field.apply() needed — runs automatically in solver.step()
    /// ```
    pub fn attach_scalar_field(&mut self, field: ScalarDiffusionField) {
        self.scalar_fields.push(field);
    }

    /// Builder variant of `attach_scalar_field`.
    pub fn with_scalar_field(mut self, field: ScalarDiffusionField) -> Self {
        self.attach_scalar_field(field);
        self
    }
}
