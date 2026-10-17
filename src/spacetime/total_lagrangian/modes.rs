//! A body's linear modes on the step's own grid, for its flight between
//! contacts: the exact solution of the linearised elastic problem,
//! `M q'' + K q = 0`, mode by mode.
//!
//! The modes are those of the grid the step keeps on the reference
//! configuration (`contact_grid`), its consistent mass and the corotated
//! law's tangent at rest: `K phi = omega^2 M phi`, `phi^T M phi = 1`, the
//! mass-orthonormal basis `u = U q`, `U^T M U = I`, of Barbic's thesis
//! (CMU-CS-07-145, 2007, sec. 2.6). The grid stays on the reference, so a
//! displacement or velocity the step makes lies in the basis and its modal
//! coordinates are exact projections.
//!
//! Solved by two-sided Jacobi (Demmel and Veselic, LAPACK Working Note 15,
//! Algorithm 3.1) on `M`, then on `M^-1/2 (K + sigma M) M^-1/2`: the shift
//! `sigma` makes the second matrix definite, the case the algorithm
//! states, and is removed from the eigenvalues after. Dense: `n^3` per
//! sweep, 0.1 s for a 108-unknown block, measured; a large body needs a
//! sparse method (not built).

use super::*;
use crate::matter::materials::solid::corotated::stress::corotated_second_pk_tangent;
pub(crate) use crate::tensor_algebra::dense::product;
use crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite;

/// A free body's rigid modes in the plane: two translations and the
/// linearised rotation.
pub(crate) const RIGID_MODES: usize = 3;

/// A body's modes, ascending, the first `RIGID_MODES` its rigid motions.
#[derive(Clone, Debug)]
pub(crate) struct Modes {
    /// Angular frequencies, rad/s (the rigid ones set to zero).
    pub(crate) omega: Vec<f64>,
    /// Per mode, its value at every point of the step's state (particles,
    /// then contact points).
    pub(crate) at: Vec<Vec<DVec2>>,
    /// Per mode, its material gradient at every particle.
    pub(crate) gradients: Vec<Vec<DMat2>>,
}

/// The material gradient `grad_X d` of the grid field `d` at point `p`, as
/// the step forms `F`'s increment.
fn gradient(grid: &Grid, d: &[f64], p: usize) -> DMat2 {
    let (_, gu) = grid.read(d, p, 0);
    let (_, gv) = grid.read(d, p, 1);
    DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y))
}

/// `K d`: the internal force's derivative at rest, `sum_p V0_p dP : grad
/// N`, with `dP = dS` there (`S = 0` at `F = I`) and `dS` the corotated
/// law's own tangent along `dC = dF + dF^T`.
fn stiffness_times(grid: &Grid, d: &[f64], (v0, solids): (&[f64], &[Elastic])) -> Vec<f64> {
    let mut out = vec![0.0; grid.len];
    for (p, (&volume, solid)) in v0.iter().zip(solids).enumerate() {
        if volume <= 0.0 {
            continue;
        }
        let g = gradient(grid, d, p);
        let ds = corotated_second_pk_tangent(
            DMat2::IDENTITY,
            g + g.transpose(),
            solid.bulk - solid.mu,
            solid.mu,
        );
        for a in 0..2 {
            let row = DVec2::new(ds.x_axis[a], ds.y_axis[a]);
            grid.visit(p, a, |k, _, grad| out[k] += volume * row.dot(grad));
        }
    }
    out
}

/// Mirrors the upper triangle's mean into both halves (round-off only).
pub(crate) fn symmetrise(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let mean = 0.5 * (a[i * n + j] + a[j * n + i]);
            (a[i * n + j], a[j * n + i]) = (mean, mean);
        }
    }
}

/// Demmel and Veselic's `tol`, "usually machine precision"; 64 eps here,
/// declared (the modes' orthonormality held to 5e-14 with 45 eps, measured).
pub(crate) const JACOBI_TOLERANCE: f64 = 64.0 * f64::EPSILON;

/// A body's dense mass and stiffness on the grid's active unknowns.
pub(crate) struct System {
    pub(crate) unknowns: Vec<usize>,
    pub(crate) n: usize,
    pub(crate) m: Vec<f64>,
    pub(crate) k: Vec<f64>,
}

/// The dense consistent mass and stiffness at rest of the body whose
/// `particles` points carry mass on `grid`; `None` when a law is not the
/// undamped corotated solid's.
pub(crate) fn system(
    grid: &Grid,
    (mass, v0, solids): (&[f64], &[f64], &[Elastic]),
    particles: usize,
) -> Option<System> {
    if solids[..particles]
        .iter()
        .any(|s| s.law != ElasticLaw::Corotated || s.is_viscous() || s.is_relaxing())
    {
        return None;
    }
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    let (mut m, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (j, &column) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[column] = 1.0;
        let mass_column = grid.mass_times(&unit, mass, 0.0);
        let stiffness_column = stiffness_times(grid, &unit, (&v0[..particles], solids));
        for (i, &row) in unknowns.iter().enumerate() {
            m[i * n + j] = mass_column[row];
            k[i * n + j] = stiffness_column[row];
        }
    }
    symmetrise(&mut m, n);
    symmetrise(&mut k, n);
    Some(System { unknowns, n, m, k })
}

/// The generalised eigenpairs `k phi = omega^2 m phi`, `phi^T m phi = 1`,
/// of an `n` by `n` pair (`m` definite): `omega^2` ascending, the shift
/// removed, and the modes as the columns of a row-major `n` by `n` matrix;
/// also the shift, the scale of the eigenvalues' round-off. `None` when `m`
/// is not definite.
pub(crate) fn eigen_pairs(m: Vec<f64>, k: &[f64], n: usize) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let (d, q) = jacobi_eigen_symmetric_definite(m, n, JACOBI_TOLERANCE);
    if d.iter().any(|&e| e <= 0.0) {
        return None;
    }
    // `M^-1/2 = Q D^-1/2 Q^T`.
    let root: Vec<f64> = (0..n * n)
        .map(|ij| {
            (0..n)
                .map(|e| q[ij / n * n + e] * q[ij % n * n + e] / d[e].sqrt())
                .sum()
        })
        .collect();
    let mut reduced = product(&product(&root, k, n), &root, n);
    symmetrise(&mut reduced, n);
    let shift = (0..n).map(|i| reduced[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        reduced[i * n + i] += shift;
    }
    let (lambda, y) = jacobi_eigen_symmetric_definite(reduced, n, JACOBI_TOLERANCE);
    let phi = product(&root, &y, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| lambda[a].total_cmp(&lambda[b]));
    let squared = order.iter().map(|&e| lambda[e] - shift).collect();
    let sorted = (0..n * n)
        .map(|ij| phi[ij / n * n + order[ij % n]])
        .collect();
    Some((squared, sorted, shift))
}

/// A grid field's values at every point and its material gradients at the
/// particles, from its entries on `unknowns`.
pub(crate) fn field_at(
    grid: &Grid,
    (unknowns, values): (&[usize], &[f64]),
    (particles, points): (usize, usize),
) -> (Vec<DVec2>, Vec<DMat2>) {
    let mut field = vec![0.0; grid.len];
    for (&unknown, &value) in unknowns.iter().zip(values) {
        field[unknown] = value;
    }
    let at = (0..points)
        .map(|p| DVec2::new(grid.read(&field, p, 0).0, grid.read(&field, p, 1).0))
        .collect();
    let gradients = (0..particles).map(|p| gradient(grid, &field, p)).collect();
    (at, gradients)
}

/// The modes of the body whose `particles` points carry mass on `grid`
/// (its contact points after them carry none). `None` when a law is not
/// the undamped corotated solid's (another law's modes are not built), or
/// when the mass is not definite.
pub(crate) fn body_modes(
    grid: &Grid,
    (mass, v0, solids): (&[f64], &[f64], &[Elastic]),
    (particles, points): (usize, usize),
) -> Option<Modes> {
    let System { unknowns, n, m, k } = system(grid, (mass, v0, solids), particles)?;
    if n <= RIGID_MODES {
        return None;
    }
    let (squared, phi, _) = eigen_pairs(m, &k, n)?;
    let omega = squared
        .iter()
        .enumerate()
        .map(|(rank, &w2)| {
            if rank < RIGID_MODES {
                0.0
            } else {
                w2.max(0.0).sqrt()
            }
        })
        .collect();
    let (at, gradients) = (0..n)
        .map(|e| {
            let column: Vec<f64> = (0..n).map(|i| phi[i * n + e]).collect();
            field_at(grid, (&unknowns, &column), (particles, points))
        })
        .unzip();
    Some(Modes {
        omega,
        at,
        gradients,
    })
}
