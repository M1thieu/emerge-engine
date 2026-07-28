use std::collections::HashMap;

use glam::{IVec2, Vec2};

use super::{FxU32BuildHasher, Grid, flat_index};

/// Two-phase mixture coupling cell (Tampubolon et al. 2017 — see `MixturePhase`'s
/// own doc). Only allocated at nodes touched by at least one `MixturePhase::Solid`
/// OR `MixturePhase::Fluid` particle (via `WithMixturePhase`) — a scene that never
/// wraps a material this way never allocates a single one of these.
///
/// `solid_mass`/`solid_momentum` and `fluid_mass`/`fluid_momentum` accumulate
/// during P2G exactly like `Cell`'s own fields, but from each phase's particles
/// separately (both are ADDITIVE alongside the ordinary `Cell` scatter, not a
/// replacement — mirrors `ContactCell`'s own convention). `resolved_solid_v`/
/// `resolved_fluid_v` are filled in by `Grid::resolve_mixture_coupling` (after
/// `update_velocities` + gravity, same pipeline position as `resolve_contact`)
/// and are what G2P reads for solid/fluid particles respectively at nodes where
/// this cell exists.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MixtureCell {
    solid_mass: f32,
    solid_momentum: Vec2,
    fluid_mass: f32,
    fluid_momentum: Vec2,
    resolved_solid_v: Vec2,
    resolved_fluid_v: Vec2,
}

pub(super) type MixtureCellMap = HashMap<u32, MixtureCell, FxU32BuildHasher>;

impl Grid {
    /// Accumulate mass and momentum for one mixture phase during P2G, additively
    /// alongside the normal `add_mass_momentum` call for the SAME particle — see
    /// `MixtureCell` doc. OOB silently ignored.
    pub fn add_mixture_mass_momentum(
        &mut self,
        cell_pos: IVec2,
        phase: crate::materials::MixturePhase,
        mass: f32,
        momentum: Vec2,
    ) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        use crate::materials::MixturePhase;
        match self.mixture_cells.entry(idx) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let cell = e.get_mut();
                match phase {
                    MixturePhase::Solid => {
                        cell.solid_mass += mass;
                        cell.solid_momentum += momentum;
                    }
                    MixturePhase::Fluid => {
                        cell.fluid_mass += mass;
                        cell.fluid_momentum += momentum;
                    }
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                self.mixture_dirty.push(idx);
                let mut cell = MixtureCell::default();
                match phase {
                    MixturePhase::Solid => {
                        cell.solid_mass = mass;
                        cell.solid_momentum = momentum;
                    }
                    MixturePhase::Fluid => {
                        cell.fluid_mass = mass;
                        cell.fluid_momentum = momentum;
                    }
                }
                e.insert(cell);
            }
        }
    }

    /// Resolved solid-phase velocity at `cell_pos` — valid after
    /// `resolve_mixture_coupling()`. Falls back to the ordinary total velocity
    /// when no mixture coupling was ever registered at this node, same
    /// convention as `grip_velocity_at`.
    pub fn resolved_solid_velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        self.mixture_cells
            .get(&idx)
            .map_or_else(|| self.velocity_at(cell_pos), |c| c.resolved_solid_v)
    }

    /// Resolved fluid-phase velocity at `cell_pos` — valid after
    /// `resolve_mixture_coupling()`. Same fallback convention as
    /// `resolved_solid_velocity_at`.
    pub fn resolved_fluid_velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        self.mixture_cells
            .get(&idx)
            .map_or_else(|| self.velocity_at(cell_pos), |c| c.resolved_fluid_v)
    }

    /// Resolves two-phase mixture coupling (Tampubolon et al. 2017 Darcy-style
    /// momentum exchange) at every mixture-active node — call after
    /// `update_velocities()` (needs the gravity-applied total field), same
    /// pipeline position as `resolve_contact`.
    ///
    /// Exact closed-form solve, not an iterative approximation: implicit
    /// backward-Euler drag exchange between two masses reduces to a 2x2 linear
    /// system per velocity COMPONENT (x and y decouple since drag is isotropic),
    /// solved directly here rather than needing Newton iteration or a global
    /// sparse solver. Let `v_s`, `v_f` be each phase's own pre-coupling velocity
    /// (its own momentum/mass, gravity already applied), `a = dt*k/m_s`,
    /// `b = dt*k/m_f`:
    ///   (1+a) v_s' - a v_f' = v_s
    ///   -b v_s' + (1+b) v_f' = v_f
    ///   det = 1 + a + b  (always > 0, unconditionally stable, no ill-conditioning
    ///   at any real k/dt/mass combination — this is what makes the LOCAL,
    ///   per-node simplification valid instead of needing the paper's own global
    ///   MINRES solve, which exists there specifically to handle full elastic/
    ///   plastic coupling this simplified drag-only model doesn't attempt).
    ///   v_s' = [(1+b) v_s + a v_f] / det
    ///   v_f' = [b v_s + (1+a) v_f] / det
    /// Momentum is exactly conserved by construction (`m_s*(v_s'-v_s) =
    /// -m_f*(v_f'-v_f)` falls out of the shared `det` and the `a*m_s = b*m_f =
    /// dt*k` identity), verified by a real test, not just claimed.
    ///
    /// Real, disclosed simplification vs. the paper: `drag_coefficient` (k) is a
    /// single scalar (mass/time), not the paper's own permeability/porosity-
    /// derived `c_E` field — mapping to real soil permeability is future work,
    /// not attempted here. Nodes with only one phase present get no correction
    /// at all (both resolved velocities just read the ordinary total field),
    /// matching `resolve_contact`'s own "no real second field" fallback.
    ///
    /// `cell_width`/`pressure_iterations` feed `project_mixture_incompressibility`
    /// (see `mixture::pressure`'s own doc): the drag solve above conserves
    /// momentum but never enforces the mixture's incompressibility constraint,
    /// so under sustained/confined loading (e.g. water settled into sand) the
    /// violation compounds silently over hundreds of steps until velocities
    /// blow past the CFL bound. `pressure_iterations == 0` skips the
    /// projection entirely.
    pub fn resolve_mixture_coupling(
        &mut self,
        dt: f32,
        gravity: Vec2,
        drag_coefficient: f32,
        cell_width: f32,
        pressure_iterations: u32,
    ) {
        const MIN_MASS_FRACTION: f32 = 1.0e-6;
        if drag_coefficient <= 0.0 {
            // Disabled: both fields just read the ordinary total velocity —
            // matches every other opt-in system's "true default is a no-op".
            for &idx in &self.mixture_dirty {
                let Some(&total) = self.cells.get(&idx) else {
                    continue;
                };
                if let Some(cell) = self.mixture_cells.get_mut(&idx) {
                    cell.resolved_solid_v = total.momentum;
                    cell.resolved_fluid_v = total.momentum;
                }
            }
            return;
        }
        for &idx in &self.mixture_dirty {
            let Some(&total) = self.cells.get(&idx) else {
                continue;
            };
            let Some(cell) = self.mixture_cells.get(&idx) else {
                continue;
            };
            let m_s = cell.solid_mass;
            let m_f = cell.fluid_mass;
            if m_s <= MIN_MASS_FRACTION || m_f <= MIN_MASS_FRACTION {
                let cell = self.mixture_cells.get_mut(&idx).unwrap();
                cell.resolved_solid_v = total.momentum;
                cell.resolved_fluid_v = total.momentum;
                continue;
            }
            let v_s = cell.solid_momentum / m_s + gravity * dt;
            let v_f = cell.fluid_momentum / m_f + gravity * dt;
            let a = dt * drag_coefficient / m_s;
            let b = dt * drag_coefficient / m_f;
            let det = 1.0 + a + b;
            let v_s_new = ((1.0 + b) * v_s + a * v_f) / det;
            let v_f_new = (b * v_s + (1.0 + a) * v_f) / det;
            let cell = self.mixture_cells.get_mut(&idx).unwrap();
            cell.resolved_solid_v = v_s_new;
            cell.resolved_fluid_v = v_f_new;
        }
        if pressure_iterations > 0 {
            self.project_mixture_incompressibility(cell_width, pressure_iterations);
        }
    }
}

// The variable-mobility Jacobi pressure projection that enforces the mixture's
// actual incompressibility constraint (`project_mixture_incompressibility`,
// `pub(super)` so `resolve_mixture_coupling` above can call it) -- a distinct
// algorithm (Zhao & Choo 2020 / Bridson Chorin-style projection) from the
// momentum-exchange drag coupling in this file -- lives in pressure.rs, along
// with its own private helpers and its own test. See that file's own doc.
mod pressure;

#[cfg(test)]
mod mixture_coupling_tests {
    use super::*;
    use crate::materials::MixturePhase;

    /// White-box: constructs a single mixture-active node with known solid/fluid
    /// mass+momentum directly (bypassing P2G), so the resolved velocities can be
    /// checked against the exact closed-form solve `resolve_mixture_coupling`'s
    /// own doc derives (backward-Euler drag exchange reduces to a 2x2 linear
    /// system, solved directly) -- not just "runs without crashing."
    fn setup(m_s: f32, v_s0: Vec2, m_f: f32, v_f0: Vec2) -> Grid {
        let mut grid = Grid::new(8);
        let cell_pos = IVec2::new(2, 2);
        // Ordinary total-field scatter too (resolve_mixture_coupling reads
        // `self.cells` for its "no real second field" fallback check via the
        // shared total, though the dual-mass branch below doesn't use it).
        grid.add_mass_momentum(cell_pos, m_s + m_f, m_s * v_s0 + m_f * v_f0);
        grid.add_mixture_mass_momentum(cell_pos, MixturePhase::Solid, m_s, m_s * v_s0);
        grid.add_mixture_mass_momentum(cell_pos, MixturePhase::Fluid, m_f, m_f * v_f0);
        // Normalize the total field's raw momentum into true velocity, matching the
        // real pipeline's ordering (resolve_mixture_coupling always runs after
        // update_velocities) -- resolve_mixture_coupling's own "no drag"/"no real
        // second field" fallbacks read `Cell.momentum` assuming it's already velocity.
        grid.update_velocities(0.0, Vec2::ZERO);
        grid
    }

    #[test]
    fn resolved_velocities_match_closed_form_2x2_solve() {
        let (m_s, m_f) = (4.0_f32, 1.0_f32);
        let (v_s0, v_f0) = (Vec2::new(0.0, 0.0), Vec2::new(0.0, -2.0));
        let dt = 0.1_f32;
        let k = 3.0_f32;
        let gravity = Vec2::ZERO; // isolate the drag exchange, no extra gravity term

        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(dt, gravity, k, 1.0, 0);

        let a = dt * k / m_s;
        let b = dt * k / m_f;
        let det = 1.0 + a + b;
        let expected_v_s = ((1.0 + b) * v_s0 + a * v_f0) / det;
        let expected_v_f = (b * v_s0 + (1.0 + a) * v_f0) / det;

        let cell_pos = IVec2::new(2, 2);
        let got_v_s = grid.resolved_solid_velocity_at(cell_pos);
        let got_v_f = grid.resolved_fluid_velocity_at(cell_pos);
        assert!(
            (got_v_s - expected_v_s).length() < 1.0e-5,
            "solid velocity mismatch: got={got_v_s:?} expected={expected_v_s:?}"
        );
        assert!(
            (got_v_f - expected_v_f).length() < 1.0e-5,
            "fluid velocity mismatch: got={got_v_f:?} expected={expected_v_f:?}"
        );
    }

    #[test]
    fn momentum_is_exactly_conserved_across_the_coupling() {
        let (m_s, m_f) = (7.0_f32, 2.5_f32);
        let (v_s0, v_f0) = (Vec2::new(1.0, 0.5), Vec2::new(-3.0, 2.0));
        let dt = 0.05_f32;
        let k = 10.0_f32;
        let gravity = Vec2::ZERO;

        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(dt, gravity, k, 1.0, 0);

        let cell_pos = IVec2::new(2, 2);
        let v_s = grid.resolved_solid_velocity_at(cell_pos);
        let v_f = grid.resolved_fluid_velocity_at(cell_pos);

        let p_before = m_s * v_s0 + m_f * v_f0;
        let p_after = m_s * v_s + m_f * v_f;
        assert!(
            (p_before - p_after).length() < 1.0e-4,
            "mixture coupling must conserve momentum exactly: before={p_before:?} after={p_after:?}"
        );
    }

    #[test]
    fn drag_pulls_phases_toward_a_shared_velocity_not_apart() {
        // Real, qualitative physical sanity check: whatever the exact numbers,
        // drag must reduce the RELATIVE speed between phases, never increase it
        // (that would mean the coupling is doing something backwards).
        let (m_s, m_f) = (3.0_f32, 3.0_f32);
        let (v_s0, v_f0) = (Vec2::new(0.0, 0.0), Vec2::new(5.0, 0.0));
        let dt = 0.1_f32;
        let k = 1.0_f32;

        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(dt, Vec2::ZERO, k, 1.0, 0);

        let cell_pos = IVec2::new(2, 2);
        let v_s = grid.resolved_solid_velocity_at(cell_pos);
        let v_f = grid.resolved_fluid_velocity_at(cell_pos);
        let relative_before = (v_s0 - v_f0).length();
        let relative_after = (v_s - v_f).length();
        assert!(
            relative_after < relative_before,
            "drag should reduce relative velocity: before={relative_before} after={relative_after}"
        );
    }

    #[test]
    fn disabled_when_drag_coefficient_is_zero() {
        // 0.0 is the documented "disabled" sentinel -- both phases must read the
        // ordinary total field, completely unaffected by their own individual
        // momenta (matching every other opt-in system's true-no-op convention).
        let (m_s, m_f) = (4.0_f32, 1.0_f32);
        let (v_s0, v_f0) = (Vec2::new(2.0, 0.0), Vec2::new(-6.0, 0.0));
        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(0.1, Vec2::ZERO, 0.0, 1.0, 0);

        let cell_pos = IVec2::new(2, 2);
        let total_v = (m_s * v_s0 + m_f * v_f0) / (m_s + m_f);
        let v_s = grid.resolved_solid_velocity_at(cell_pos);
        let v_f = grid.resolved_fluid_velocity_at(cell_pos);
        assert!((v_s - total_v).length() < 1.0e-5);
        assert!((v_f - total_v).length() < 1.0e-5);
    }
}
