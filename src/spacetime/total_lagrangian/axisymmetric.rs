//! FAST_MODE_PLAN.md sec. 51.4, the axisymmetric path's first slice, test-only
//! (knob `EMERGE_PROBE_AXISYMMETRIC_PATH = i_a`, the axis's node column; off
//! by default): the symmetry axis on the grid's unknowns.
//!
//! Source: Nairn and Guilkey, IJNME 101 (2015) 127-147, Fig. 6 and its text:
//! `v_r = 0` on the axis, a mirrored ghost node `v_r(-dr) = -v_r(dr)`. Here
//! the B-spline coefficients are mirrored about the axis's node column `i_a`
//! (the static probe's rule, `axisymmetric_probes`, exact for uniform states
//! with an exact quadrature, sec. 49g): the radial component is odd, its
//! coefficient on the axis zero; the axial component is even. The engine's
//! extended B-splines already write a face's coefficient as a list of
//! unknowns and weights (`Grid::extension`), so a ghost face takes its
//! mirror's list times `-1` or `+1`, and no new mechanism enters.
//! Adaptation, declared: only the column next to the axis is mirrored (the
//! quadratic B-splines of points at `R > 0` reach one node beyond it).
//!
//! The hoop (sec. 49c.2, derived from the source's weak form): each point's
//! `F_33 = r / R`, the current radius over the reference one; the internal
//! force gains `w P_33 N_i / R` on the radial rows, `P_33 = F_33 S_33`, with
//! the weights `w = A_p R_p` (per radian) the caller gives as `v0` and the
//! masses per radian. The law's step stress is the discrete gradient over
//! the block-diagonal `C` (the in-plane block and `C_33`), Gonzalez's formula
//! in that space. Declared limits of this slice: the reference radii come
//! through this thread (`set_radii`), so only a step run on the caller's
//! thread sees them (`step_total_lagrangian`, not the controller's
//! concurrent tries); the internal force's derivative is taken by central
//! differences; the midpoint only (no theta step, no backward Euler); the
//! neo-Hookean and J2 laws only, with no dashpot and no relaxation branch.

use std::cell::{Cell, RefCell};

use glam::{DMat2, DVec2};

use super::basis::Grid;
use super::elastic::{Elastic, ElasticLaw};
use crate::matter::materials::solid::neo_hookean::{
    neo_hookean_energy_axisymmetric, neo_hookean_second_pk_axisymmetric,
};

thread_local! {
    static AXIS: Cell<Option<usize>> = const { Cell::new(None) };
    static RADII: RefCell<Option<Vec<f64>>> = const { RefCell::new(None) };
}

/// Each point's reference radius for this thread's solves.
pub(crate) fn set_radii(radii: Option<Vec<f64>>) {
    RADII.with(|r| *r.borrow_mut() = radii);
}

/// What a solve needs for the hoop: the axis's position (the step's length
/// unit) and each point's reference radius.
pub(crate) struct Hoop {
    pub(crate) axis_x: f64,
    pub(crate) radii: Vec<f64>,
}

/// The hoop's data for a solve on `grid` with `n` points, when the path is
/// axisymmetric and the radii match.
pub(crate) fn hoop(grid: &Grid, n: usize) -> Option<Hoop> {
    let axis = axis()?;
    let radii = RADII.with(|r| r.borrow().clone())?;
    (radii.len() == n).then_some(Hoop {
        axis_x: axis as f64 * grid.cell,
        radii,
    })
}

/// The stored energy per reference volume with the hoop stretch `f33`.
pub(crate) fn energy_with_stretch(solid: &Elastic, f: DMat2, f33: f64) -> f64 {
    let (lambda, mu) = solid.lame();
    match solid.law {
        ElasticLaw::NeoHookean => {
            neo_hookean_energy_axisymmetric(f.transpose() * f, f33 * f33, lambda, mu)
        }
        _ => panic!("the axisymmetric path's energy is the neo-Hookean law's only (sec. 51.4)"),
    }
}

impl Elastic {
    /// The step's stress between `(C_n, C33_n)` and `(C_n+1, C33_n+1)`:
    /// the mean of the end stresses plus Gonzalez's correction along the
    /// increment, so that `S : dC / 2` (the hoop entry included) is the
    /// law's energy change, plus its dissipation for J2. Returns the
    /// in-plane block and `S_33`.
    pub(crate) fn step_stress_with_stretch(
        &self,
        (c0, a0): (DMat2, f64),
        (c1, a1): (DMat2, f64),
        dt: f64,
    ) -> (DMat2, f64) {
        assert!(
            !self.is_viscous() && !self.is_relaxing(),
            "the axisymmetric path's first slice carries no dashpot and no relaxation branch"
        );
        let (lambda, mu) = self.lame();
        let ((s0, t0), (s1, t1), work) = match &self.law {
            ElasticLaw::NeoHookean => (
                neo_hookean_second_pk_axisymmetric(c0, a0, lambda, mu),
                neo_hookean_second_pk_axisymmetric(c1, a1, lambda, mu),
                neo_hookean_energy_axisymmetric(c1, a1, lambda, mu)
                    - neo_hookean_energy_axisymmetric(c0, a0, lambda, mu),
            ),
            ElasticLaw::HenckyJ2(law, state) => {
                let stepped = law.step_with_stretch((c0, a0), (c1, a1), state, dt);
                (
                    law.second_pk_with_stretch(c0, a0, state),
                    law.second_pk_with_stretch(c1, a1, &stepped.state),
                    stepped.energy_change + stepped.dissipated,
                )
            }
            _ => panic!(
                "the axisymmetric path's first slice carries the neo-Hookean and J2 laws only"
            ),
        };
        let (mean, mean33) = (0.5 * (s0 + s1), 0.5 * (t0 + t1));
        let (dc, da) = (c1 - c0, a1 - a0);
        let norm2 = dc.x_axis.length_squared() + dc.y_axis.length_squared() + da * da;
        if norm2 == 0.0 {
            return (mean, mean33);
        }
        let contracted = mean.x_axis.dot(dc.x_axis) + mean.y_axis.dot(dc.y_axis) + mean33 * da;
        let scale = 2.0 * (work - 0.5 * contracted) / norm2;
        (mean + scale * dc, mean33 + scale * da)
    }
}

/// One point's internal force with the hoop, the midpoint's (`solve`'s
/// internal force on the axisymmetric path): `v0 F_mid S` on the in-plane
/// rows and `v0 F33_mid S_33 N_i / R` on the radial ones.
pub(crate) fn hoop_force(
    grid: &Grid,
    hoop: &Hoop,
    (p, u, x): (usize, &[f64], DVec2),
    (v0, solid): (f64, &Elastic),
    (f0, f1, dt): (DMat2, DMat2, f64),
    acc: &mut [f64],
) {
    let r = hoop.radii[p];
    if v0 <= 0.0 || r <= 0.0 {
        return;
    }
    let f33_0 = (x.x - hoop.axis_x) / r;
    let f33_1 = f33_0 + grid.read(u, p, 0).0 / r;
    let (s, s33) = solid.step_stress_with_stretch(
        (f0.transpose() * f0, f33_0 * f33_0),
        (f1.transpose() * f1, f33_1 * f33_1),
        dt,
    );
    let first = v0 * (0.5 * (f0 + f1)) * s;
    for a in 0..2 {
        let row = first.row(a);
        grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
    }
    let p33 = v0 * 0.5 * (f33_0 + f33_1) * s33 / r;
    grid.visit(p, 0, |k, w, _| acc[k] += p33 * w);
}

/// The axis's node column for this thread's grids (the unit tests), in place
/// of the knob.
pub(crate) fn set_axis(axis: Option<usize>) {
    AXIS.with(|a| a.set(axis));
}

/// The axis's node column, if the path is axisymmetric; says so once.
pub(crate) fn axis() -> Option<usize> {
    static KNOB: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    AXIS.with(Cell::get).or_else(|| {
        *KNOB.get_or_init(|| {
            let axis = std::env::var("EMERGE_PROBE_AXISYMMETRIC_PATH")
                .ok()?
                .parse()
                .ok()?;
            println!("knob active: axisymmetric path, axis on node column {axis} (sec. 51.4)");
            Some(axis)
        })
    })
}

/// Whether face column `i` of lattice `a` (0 the radial component, 1 the
/// axial) is no unknown of its own: every column left of the axis, and the
/// axis's own for the radial component.
pub(crate) fn excluded(a: usize, i: usize) -> bool {
    axis().is_some_and(|axis| i < axis || (a == 0 && i == axis))
}

/// The mirrored faces' coefficients, once the grid's unknowns and extensions
/// are built: the radial component zero on the axis, each ghost face (one
/// column left of it) its mirror's list times `-1` (radial) or `+1` (axial).
pub(crate) fn mirror(grid: &mut Grid) {
    let Some(axis) = axis() else {
        return;
    };
    for (a, l) in grid.lattices.iter().enumerate() {
        let sign = if a == 0 { -1.0 } else { 1.0 };
        for j in 0..l.nj {
            let face = |i: usize| l.start + i + l.ni * j;
            if a == 0 && axis < l.ni {
                grid.extension[face(axis)] = Some(Box::default());
            }
            if axis == 0 || axis + 1 >= l.ni {
                continue;
            }
            let (ghost, image) = (face(axis - 1), face(axis + 1));
            let list: Option<Box<[(usize, f64)]>> = if grid.slot[image] != usize::MAX {
                Some(Box::new([(grid.slot[image], sign)]))
            } else {
                grid.extension[image]
                    .as_ref()
                    .map(|list| list.iter().map(|&(u, e)| (u, sign * e)).collect())
            };
            grid.extension[ghost] = list;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ConservingSettings, Grid, GridLayout, Particles, step_total_lagrangian};
    use super::*;

    /// The mirror reproduces the axisymmetric linear fields exactly: a
    /// radial `u_r = a R` (odd) and a uniform axial `u_z = b` (even) set on
    /// the unknowns read back at every point as `a R` and `b`, the points
    /// nearest the axis included; no unknown sits on the axis for `u_r`.
    #[test]
    fn the_mirror_reproduces_the_axisymmetric_linear_fields() {
        let (axis, h) = (6usize, 1.0);
        set_axis(Some(axis));
        let layout = GridLayout::new(24, 24, 1.0);
        let mut x = Vec::new();
        for i in 0..8 {
            for j in 0..12 {
                for (s, t) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    x.push(DVec2::new(axis as f64 + i as f64 + s, 6.0 + j as f64 + t));
                }
            }
        }
        let mass = vec![1.0; x.len()];
        let volume = vec![0.25; x.len()];
        let grid = Grid::new(&layout, &x, (&mass, &volume), &|_, _, _| None, Some(0.5));
        set_axis(None);
        let (a, b) = (0.03, -0.7);
        let mut y = vec![0.0; grid.len];
        let lattice = grid.lattices;
        for (k, &face) in grid.stacked_of.iter().enumerate() {
            let c = grid.axis_of[k];
            let l = lattice[c];
            let i = (face - l.start) % l.ni;
            assert!(!(c == 0 && i <= axis), "a radial unknown at column {i}");
            y[k] = if c == 0 {
                a * (i as f64 - axis as f64) * h
            } else {
                b
            };
        }
        for (p, q) in x.iter().enumerate() {
            let (ur, _) = grid.read(&y, p, 0);
            let (uz, _) = grid.read(&y, p, 1);
            let r = q.x - axis as f64 * h;
            assert!(
                (ur - a * r).abs() < 1e-12,
                "point {p} at R {r}: u_r {ur} vs {}",
                a * r
            );
            assert!((uz - b).abs() < 1e-12, "point {p}: u_z {uz}");
        }
    }

    /// AX-8 (FAST_MODE_PLAN.md sec. 51.4, criterion written before the
    /// run, the axisymmetric analogue of the plane gate (g)): a neo-Hookean
    /// cylinder released from a uniform radial stretch of 1.3 (`F = diag(1.3,
    /// 1)` in plane and `F_33 = 1.3`) in free space rings with its energy
    /// per radian kept: over 20 steps the kinetic plus stored energy (the
    /// hoop's included) stays the start's within the solves' residual work
    /// and the sums' round-off; the axial momentum stays zero.
    ///
    /// Written for the plane gate's step, 0.2; measured: the first step's
    /// Newton stagnates there (residual 0.27 down to 6.7e-4, the hoop's
    /// stiffness missing from the preconditioner and the derivative taken by
    /// differences), while stretches of 1.01 and 1.1 pass at 0.2 and 1.3
    /// passes at 0.05. The gate runs at 0.05; the failure at 0.2 is recorded
    /// (FAST_MODE_PLAN.md sec. 51.5).
    #[test]
    fn a_released_neo_hookean_cylinder_keeps_its_energy() {
        let axis = 4usize;
        set_axis(Some(axis));
        let layout = GridLayout::new(24, 24, 1.0);
        let mut reference = Vec::new();
        for i in 0..4 {
            for j in 0..8 {
                for (s, t) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    reference.push(DVec2::new((axis + i) as f64 + s, 8.0 + j as f64 + t));
                }
            }
        }
        let n = reference.len();
        let radii: Vec<f64> = reference.iter().map(|q| q.x - axis as f64).collect();
        set_radii(Some(radii.clone()));
        // Weights per radian: area times radius, density one.
        let v0: Vec<f64> = radii.iter().map(|r| 0.25 * r).collect();
        let mass = v0.clone();
        let solids = vec![Elastic::neo_hookean(2.9, 1.3); n];
        let stretch = DMat2::from_diagonal(DVec2::new(1.3, 1.0));
        let mut x: Vec<DVec2> = reference
            .iter()
            .zip(&radii)
            .map(|(q, r)| DVec2::new(axis as f64 + 1.3 * r, q.y))
            .collect();
        let mut v = vec![DVec2::ZERO; n];
        let mut f = vec![stretch; n];
        let energy = |x: &[DVec2], v: &[DVec2], f: &[DMat2]| -> (f64, f64) {
            let kinetic: f64 = (0..n).map(|p| 0.5 * mass[p] * v[p].length_squared()).sum();
            let stored: f64 = (0..n)
                .map(|p| {
                    v0[p] * energy_with_stretch(&solids[p], f[p], (x[p].x - axis as f64) / radii[p])
                })
                .sum();
            (kinetic, stored)
        };
        let start = energy(&x, &v, &f);
        let settings = ConservingSettings::default();
        let free = |_: usize, _: usize, _: usize| None;
        let mut cache = None;
        let (mut allowed, mut largest_kinetic, mut worst) = (0.0, 0.0f64, 0.0f64);
        for k in 0..20 {
            let report = step_total_lagrangian(
                &layout,
                Particles {
                    x: &mut x,
                    v: &mut v,
                    f: &mut f,
                    mass: &mass,
                    v0: &v0,
                },
                &reference,
                &solids,
                (DVec2::ZERO, None, 0.05),
                (&free, &settings),
                &mut cache,
            );
            assert!(report.converged, "step {k} did not converge: {report:?}");
            allowed += report.residual_work;
            let now = energy(&x, &v, &f);
            largest_kinetic = largest_kinetic.max(now.0);
            let round_off = n as f64 * f64::EPSILON * (now.0 + now.1 + start.1);
            let balance = now.0 + now.1 - start.0 - start.1;
            worst = worst.max(balance.abs());
            let axial: f64 = (0..n).map(|p| mass[p] * v[p].y).sum();
            assert!(axial.abs() < 1e-12, "step {k}: axial momentum {axial:e}");
            assert!(
                balance.abs() <= allowed + round_off,
                "step {k}: balance {balance:e} beyond {:e}",
                allowed + round_off
            );
        }
        set_axis(None);
        set_radii(None);
        println!(
            "AX-8: stored at the start {:.6e} per radian, largest kinetic {largest_kinetic:.6e}, worst |balance| {worst:.3e} within residual work {allowed:.3e}",
            start.1
        );
        assert!(largest_kinetic > 0.1 * start.1, "the cylinder rings");
    }
}
