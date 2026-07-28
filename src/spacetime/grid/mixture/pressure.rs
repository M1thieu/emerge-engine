//! Two-phase mixture incompressibility pressure projection -- split out of
//! `mixture.rs` (was its "Enforces the mixture's incompressibility constraint"
//! section, ~180 of the original file's ~595 lines). A distinct algorithm from
//! the momentum-exchange drag coupling in the parent module (`mixture::mod`):
//! a Jacobi-iterated variable-mobility Poisson solve, run AFTER
//! `resolve_mixture_coupling`'s own closed-form drag solve to enforce the
//! mixture's actual incompressibility constraint instead of just conserving
//! momentum. See `project_mixture_incompressibility`'s own doc for the full
//! derivation and citations (Zhao & Choo 2020; Bridson's Chorin-style
//! projection).

use std::collections::HashMap;

use glam::{IVec2, Vec2};

use super::{FxU32BuildHasher, Grid, flat_index};

impl Grid {
    /// Flat index -> cell position. Inverse of `flat_index`.
    fn idx_to_pos(&self, idx: u32) -> IVec2 {
        let idx = idx as usize;
        IVec2::new(
            (idx / self.resolution) as i32,
            (idx % self.resolution) as i32,
        )
    }

    fn mixture_solid_v_or_zero(&self, pos: IVec2) -> Vec2 {
        flat_index(pos, self.resolution)
            .and_then(|idx| self.mixture_cells.get(&idx))
            .map_or(Vec2::ZERO, |c| c.resolved_solid_v)
    }

    fn mixture_fluid_v_or_zero(&self, pos: IVec2) -> Vec2 {
        flat_index(pos, self.resolution)
            .and_then(|idx| self.mixture_cells.get(&idx))
            .map_or(Vec2::ZERO, |c| c.resolved_fluid_v)
    }

    /// Enforces the mixture's incompressibility constraint (Zhao & Choo 2020,
    /// "Stabilized material point methods for coupled large deformation and
    /// fluid flow in porous materials", arXiv:1905.00671):
    ///   (1 - n)*div(v_solid) + n*div(v_fluid) = 0
    /// where `n` is the local fluid volume fraction (porosity). Naive momentum-
    /// only coupling lets this drift under sustained/confined loading (water
    /// settled into sand -- close to the "undrained" regime the paper names as
    /// the specific failure case) until the accumulated violation destabilizes
    /// velocities well past the CFL bound.
    ///
    /// Variable-density Chorin-style pressure projection (Bridson, "Fluid
    /// Simulation for Computer Graphics", ch. 5 -- the standard real-time-
    /// graphics form of enforcing incompressibility, generalized here to a
    /// two-phase mixture instead of one fluid), solved with a fixed number of
    /// Jacobi iterations rather than an exact sparse solve -- the real-time-
    /// affordable approximation both Zhao & Choo and Stam's own "Real-Time
    /// Fluid Dynamics for Games" independently point to. Per active mixture
    /// cell, using each cell's OWN local mass fractions as the porosity
    /// estimate `n = fluid_mass / (solid_mass + fluid_mass)`:
    ///
    ///   D = (1-n)*div(v_s) + n*div(v_f)                  (divergence residual)
    ///   alpha_s = 1/max(solid_mass, eps), alpha_f = 1/max(fluid_mass, eps)
    ///   K = (1-n)*alpha_s + n*alpha_f                     (local "mobility")
    ///   div( K * grad(p) ) = D                            (variable-mobility Poisson eq.)
    ///   v_s' = v_s - alpha_s * grad(p)
    ///   v_f' = v_f - alpha_f * grad(p)
    ///
    /// The mobility `K` must be folded into the Laplacian operator itself via
    /// harmonic-mean FACE coefficients (`K_face = 2*K_i*K_j/(K_i+K_j)`), not
    /// divided out of a constant-coefficient Laplacian's right-hand side and
    /// reapplied only in the final correction step — `alpha = 1/mass` is
    /// unbounded at the near-zero-mass nodes that are ordinary at MPM
    /// kernel-support edges, and a mismatched formulation lets that unbounded
    /// alpha produce an unbounded velocity correction. Harmonic-mean faces are
    /// the standard, correct treatment for variable-density/variable-mobility
    /// pressure projection (Bridson; Foster & Fedkiw) and are structurally
    /// self-limiting: a face where either side has near-zero mass (huge K)
    /// contributes almost nothing (harmonic mean of a huge value and a normal
    /// value is close to the smaller one), and a face where BOTH sides are
    /// near-empty contributes ~0 instead of blowing up. Missing/OOB neighbors
    /// are treated as `K_j = 0` (a natural no-flux Neumann boundary at the
    /// material's own edge, not an arbitrary Dirichlet p=0).
    ///
    /// The Poisson equation is solved via `pressure_iterations` Jacobi sweeps.
    /// Disclosed limitation from Stam's own paper: a *settled, confined* liquid
    /// is the documented worst case for a low-iteration Jacobi solve -- pick
    /// `pressure_iterations` by measuring against the actual long-settle
    /// scenario, not by assuming a small fixed count is free.
    ///
    /// `pub(super)`: called by `resolve_mixture_coupling` in the parent
    /// `mixture` module when `pressure_iterations > 0`.
    pub(super) fn project_mixture_incompressibility(
        &mut self,
        cell_width: f32,
        pressure_iterations: u32,
    ) {
        const MIN_MASS: f32 = 1.0e-6;
        let h = cell_width.max(1.0e-6);

        // Per-cell constants: mobility K, inverse-mass weights, and the
        // divergence residual -- computed once from the post-drag-solve velocity
        // field, all in local (cell_pos, value) pairs so we're not fighting the
        // borrow checker against `self.mixture_cells` while reading neighbors.
        let mut alpha_s: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut alpha_f: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut significant: HashMap<u32, (bool, bool), FxU32BuildHasher> = HashMap::default();
        let mut mobility: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut rhs: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut pressure: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();

        for &idx in &self.mixture_dirty {
            let Some(cell) = self.mixture_cells.get(&idx) else {
                continue;
            };
            let pos = self.idx_to_pos(idx);
            let m_s = cell.solid_mass.max(0.0);
            let m_f = cell.fluid_mass.max(0.0);
            let n = if m_s + m_f > MIN_MASS {
                m_f / (m_s + m_f)
            } else {
                0.0
            };
            let a_s = 1.0 / m_s.max(MIN_MASS);
            let a_f = 1.0 / m_f.max(MIN_MASS);
            let k = (1.0 - n) * a_s + n * a_f;
            // A phase with negligible LOCAL mass at this node (ordinary at MPM
            // kernel-support edges) has an unbounded `alpha = 1/mass`. The Poisson
            // solve above is safe (harmonic-mean faces saturate it), but applying
            // that raw, unbounded alpha to a moderate grad(p) in the correction
            // step below would produce huge velocity corrections at nodes with
            // essentially no real fluid (or solid) there. A phase's velocity is
            // only meaningful, and only gets corrected, where it holds a real
            // fraction of this node's total mass -- mirrors
            // `resolve_mixture_coupling`'s own "no real second field" skip
            // convention, just with a threshold large enough to matter.
            const MIN_MASS_FRACTION: f32 = 0.01;
            let total_mass = (m_s + m_f).max(MIN_MASS);
            let solid_significant = m_s / total_mass > MIN_MASS_FRACTION;
            let fluid_significant = m_f / total_mass > MIN_MASS_FRACTION;

            let vs_r = self.mixture_solid_v_or_zero(pos + IVec2::new(1, 0)).x;
            let vs_l = self.mixture_solid_v_or_zero(pos - IVec2::new(1, 0)).x;
            let vs_u = self.mixture_solid_v_or_zero(pos + IVec2::new(0, 1)).y;
            let vs_d = self.mixture_solid_v_or_zero(pos - IVec2::new(0, 1)).y;
            let div_vs = (vs_r - vs_l) / (2.0 * h) + (vs_u - vs_d) / (2.0 * h);

            let vf_r = self.mixture_fluid_v_or_zero(pos + IVec2::new(1, 0)).x;
            let vf_l = self.mixture_fluid_v_or_zero(pos - IVec2::new(1, 0)).x;
            let vf_u = self.mixture_fluid_v_or_zero(pos + IVec2::new(0, 1)).y;
            let vf_d = self.mixture_fluid_v_or_zero(pos - IVec2::new(0, 1)).y;
            let div_vf = (vf_r - vf_l) / (2.0 * h) + (vf_u - vf_d) / (2.0 * h);

            let residual = (1.0 - n) * div_vs + n * div_vf;

            alpha_s.insert(idx, a_s);
            alpha_f.insert(idx, a_f);
            significant.insert(idx, (solid_significant, fluid_significant));
            mobility.insert(idx, k);
            rhs.insert(idx, residual);
            pressure.insert(idx, 0.0);
        }

        let k_or_zero = |pos: IVec2| -> f32 {
            flat_index(pos, self.resolution)
                .and_then(|idx| mobility.get(&idx).copied())
                .unwrap_or(0.0)
        };
        let p_or_zero = |p: &HashMap<u32, f32, FxU32BuildHasher>, pos: IVec2| -> f32 {
            flat_index(pos, self.resolution)
                .and_then(|idx| p.get(&idx).copied())
                .unwrap_or(0.0)
        };
        // Harmonic mean of two mobilities -- 0 if either side is ~0 (no material,
        // no flux through that face), never blows up even if one side is huge.
        let face_k = |k_i: f32, k_j: f32| -> f32 {
            if k_i + k_j > 1.0e-12 {
                2.0 * k_i * k_j / (k_i + k_j)
            } else {
                0.0
            }
        };

        for _ in 0..pressure_iterations {
            let mut next = pressure.clone();
            for &idx in &self.mixture_dirty {
                let (Some(&r), Some(&k_i)) = (rhs.get(&idx), mobility.get(&idx)) else {
                    continue;
                };
                let pos = self.idx_to_pos(idx);
                let k_r = face_k(k_i, k_or_zero(pos + IVec2::new(1, 0)));
                let k_l = face_k(k_i, k_or_zero(pos - IVec2::new(1, 0)));
                let k_u = face_k(k_i, k_or_zero(pos + IVec2::new(0, 1)));
                let k_d = face_k(k_i, k_or_zero(pos - IVec2::new(0, 1)));
                let k_sum = (k_r + k_l + k_u + k_d).max(1.0e-9);

                let p_r = p_or_zero(&pressure, pos + IVec2::new(1, 0));
                let p_l = p_or_zero(&pressure, pos - IVec2::new(1, 0));
                let p_u = p_or_zero(&pressure, pos + IVec2::new(0, 1));
                let p_d = p_or_zero(&pressure, pos - IVec2::new(0, 1));
                let weighted_neighbors = k_r * p_r + k_l * p_l + k_u * p_u + k_d * p_d;
                next.insert(idx, (weighted_neighbors - h * h * r) / k_sum);
            }
            pressure = next;
        }

        for &idx in &self.mixture_dirty {
            let (Some(&a_s), Some(&a_f), Some(&(solid_significant, fluid_significant))) =
                (alpha_s.get(&idx), alpha_f.get(&idx), significant.get(&idx))
            else {
                continue;
            };
            let pos = self.idx_to_pos(idx);
            let p_r = p_or_zero(&pressure, pos + IVec2::new(1, 0));
            let p_l = p_or_zero(&pressure, pos - IVec2::new(1, 0));
            let p_u = p_or_zero(&pressure, pos + IVec2::new(0, 1));
            let p_d = p_or_zero(&pressure, pos - IVec2::new(0, 1));
            let grad_p = Vec2::new((p_r - p_l) / (2.0 * h), (p_u - p_d) / (2.0 * h));
            if let Some(cell) = self.mixture_cells.get_mut(&idx) {
                if solid_significant {
                    cell.resolved_solid_v -= a_s * grad_p;
                }
                if fluid_significant {
                    cell.resolved_fluid_v -= a_f * grad_p;
                }
            }
        }
    }
}

#[cfg(test)]
mod pressure_projection_tests {
    use super::*;
    use crate::materials::MixturePhase;

    /// Real test for the incompressibility projection itself: build a small
    /// neighborhood of mixture-active nodes with a deliberately divergent
    /// solid velocity field (radiating outward from a center node -- a real,
    /// nonzero div(v_s)), run the projection, and confirm the projected
    /// divergence residual actually SHRINKS relative to the unprojected one.
    /// This is the real, checkable claim behind
    /// `project_mixture_incompressibility` -- not just "runs without crashing."
    #[test]
    fn pressure_projection_reduces_divergence_residual() {
        let mut grid = Grid::new(16);
        let center = IVec2::new(8, 8);
        let m_s = 2.0_f32;
        let m_f = 2.0_f32;
        // Solid velocity field radiating outward from `center` -- real nonzero
        // divergence by construction (a source, not a rotation/shear).
        // A uniform dilation (v = 0.5*d) has constant divergence everywhere and
        // a closed (Neumann) system can never fully cancel that -- it's a
        // net source with nowhere to drain. Use a decaying (Gaussian-weighted)
        // radial field instead: real, concentrated divergence near `center`
        // that fades toward the patch edge, so a closed system CAN resolve
        // it (the total divergence over the patch is close to zero).
        let v_s_at = |pos: IVec2| -> Vec2 {
            let d = (pos - center).as_vec2();
            let r2 = d.length_squared();
            d * 0.5 * (-r2 / 8.0).exp()
        };
        for dx in -6..=6 {
            for dy in -6..=6 {
                let pos = center + IVec2::new(dx, dy);
                let v_s = v_s_at(pos);
                grid.add_mass_momentum(pos, m_s + m_f, m_s * v_s + m_f * Vec2::ZERO);
                grid.add_mixture_mass_momentum(pos, MixturePhase::Solid, m_s, m_s * v_s);
                grid.add_mixture_mass_momentum(pos, MixturePhase::Fluid, m_f, m_f * Vec2::ZERO);
            }
        }
        grid.update_velocities(0.0, Vec2::ZERO);

        // No drag (phases already at rest relative to their own construction),
        // no projection yet -- just resolve the mixture bookkeeping.
        grid.resolve_mixture_coupling(0.0, Vec2::ZERO, 1.0e-9, 1.0, 0);
        let div_before = |g: &Grid| -> f32 {
            let r = g.resolved_solid_velocity_at(center + IVec2::new(1, 0)).x
                - g.resolved_solid_velocity_at(center - IVec2::new(1, 0)).x;
            let u = g.resolved_solid_velocity_at(center + IVec2::new(0, 1)).y
                - g.resolved_solid_velocity_at(center - IVec2::new(0, 1)).y;
            (r + u) / 2.0
        };
        let residual_unprojected = div_before(&grid).abs();
        assert!(
            residual_unprojected > 1.0e-3,
            "test setup should have real nonzero divergence, got {residual_unprojected}"
        );

        // Same setup, but with the projection applied.
        let mut grid2 = Grid::new(16);
        for dx in -6..=6 {
            for dy in -6..=6 {
                let pos = center + IVec2::new(dx, dy);
                let v_s = v_s_at(pos);
                grid2.add_mass_momentum(pos, m_s + m_f, m_s * v_s + m_f * Vec2::ZERO);
                grid2.add_mixture_mass_momentum(pos, MixturePhase::Solid, m_s, m_s * v_s);
                grid2.add_mixture_mass_momentum(pos, MixturePhase::Fluid, m_f, m_f * Vec2::ZERO);
            }
        }
        grid2.update_velocities(0.0, Vec2::ZERO);
        grid2.resolve_mixture_coupling(0.0, Vec2::ZERO, 1.0e-9, 1.0, 200);
        let residual_projected = div_before(&grid2).abs();

        assert!(
            residual_projected < residual_unprojected * 0.6,
            "projection should substantially shrink the divergence residual: \
             before={residual_unprojected:.5} after={residual_projected:.5}"
        );
    }
}
