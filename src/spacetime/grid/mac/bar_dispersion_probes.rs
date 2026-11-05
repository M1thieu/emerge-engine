//! The bar's numerical dispersion on its own grid: the consistent mass
//! and the stiffness at rest that `bar_impact_run`'s grid carries, their
//! free-free longitudinal modes against the exact `omega_n = n pi c0 / L`,
//! and the midpoint's frequency on each. A probe: measurement, nothing held.

use super::MacLayout;
use super::gates::gauss_legendre;
use crate::spacetime::total_lagrangian::{ConservingSettings, Grid};
use crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite;
use glam::{DMat2, DVec2};

/// Row-major `n x n` product.
pub(super) fn product(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let e = a[i * n + k];
            if e != 0.0 {
                for j in 0..n {
                    out[i * n + j] += e * b[k * n + j];
                }
            }
        }
    }
    out
}

/// `K phi = l M phi` on `M` definite (Jacobi on `M`, then on `M^-1/2 K
/// M^-1/2`, as `vibration::modes`): eigenvalues ascending, their vectors.
pub(super) fn generalized(k: &[f64], m: &[f64], n: usize) -> Vec<(f64, Vec<f64>)> {
    let (d, q) = jacobi_eigen_symmetric_definite(m.to_vec(), n, 1e-14);
    let root: Vec<f64> = (0..n * n)
        .map(|ij| {
            (0..n)
                .map(|e| q[ij / n * n + e] * q[ij % n * n + e] / d[e].sqrt())
                .sum()
        })
        .collect();
    let mut reduced = product(&product(&root, k, n), &root, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let mean = 0.5 * (reduced[i * n + j] + reduced[j * n + i]);
            (reduced[i * n + j], reduced[j * n + i]) = (mean, mean);
        }
    }
    let shift = (0..n).map(|i| reduced[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        reduced[i * n + i] += shift;
    }
    let (lambda, y) = jacobi_eigen_symmetric_definite(reduced, n, 1e-14);
    let phi = product(&root, &y, n);
    let mut out: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|e| (lambda[e] - shift, (0..n).map(|i| phi[i * n + e]).collect()))
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// The bar of `bar_impact_run` at `cells` (its defaults: width 2 cells,
/// Poisson's ratio 0, 3 x 3 Gauss points per half cell, per unit density,
/// lengths in cells), free of walls; prints its longitudinal modes.
fn dispersion(cells: usize, courants: &[f64]) {
    let width = 2usize;
    let (e, rho, length) = (11.25e6f64, 1700.0f64, 1.0f64);
    let cell_m = length / cells as f64;
    let c0 = (e / rho).sqrt() / cell_m;
    let mu = 0.5 * c0 * c0;
    // `lambda = bulk - mu = 0` (`Elastic::isochoric(mu, mu)` in the probe).
    let lambda = 0.0;
    let layout = MacLayout::new(width + 8, cells + 8, 1.0);
    let (nodes, weights) = gauss_legendre(3);
    let (x0, bottom) = (4.0, 4.5);
    let (mut reference, mut v0) = (Vec::new(), Vec::new());
    for j in 0..2 * cells {
        for i in 0..2 * width {
            for b in 0..3 {
                for a in 0..3 {
                    reference.push(DVec2::new(
                        x0 + (i as f64 + nodes[a]) * 0.5,
                        bottom + (j as f64 + nodes[b]) * 0.5,
                    ));
                    v0.push(0.25 * weights[a] * weights[b]);
                }
            }
        }
    }
    let mass = v0.clone();
    let occupation = ConservingSettings::default().occupation;
    let grid = Grid::new(
        &layout,
        &reference,
        (&mass, &v0),
        &|_, _, _| None,
        occupation,
    );
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    // `K d`: the linear elasticity of the step's Jacobian at rest,
    // `sigma = 2 mu eps + lambda tr(eps) I`.
    let stiffness_times = |d: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; grid.len];
        for (p, &volume) in v0.iter().enumerate().take(reference.len()) {
            let (_, gu) = grid.read(d, p, 0);
            let (_, gv) = grid.read(d, p, 1);
            let g = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
            let eps = 0.5 * (g + g.transpose());
            let sigma = 2.0 * mu * eps + lambda * (eps.x_axis.x + eps.y_axis.y) * DMat2::IDENTITY;
            for a in 0..2 {
                let row = DVec2::new(sigma.x_axis[a], sigma.y_axis[a]);
                grid.visit(p, a, |k, _, grad| out[k] += volume * row.dot(grad));
            }
        }
        out
    };
    let (mut m, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (j, &col) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[col] = 1.0;
        let mu_col = grid.mass_times(&unit, &mass, 0.0);
        let ku_col = stiffness_times(&unit);
        for (i, &r) in unknowns.iter().enumerate() {
            m[i * n + j] = mu_col[r];
            k[i * n + j] = ku_col[r];
        }
    }
    let largest = (0..n).map(|i| k[i * n + i]).fold(0.0, f64::max);
    println!(
        "bar dispersion, h = L/{cells}, {n} unknowns, c0 {:.1} m/s ({c0:.3} cells/s)",
        c0 * cell_m
    );
    // The particles' rows of equal reference height, for the uniformity
    // across the width.
    let row_of = |p: usize| ((reference[p].y - bottom) * 1e6).round() as i64;
    let mut found = 0usize;
    for (lambda_n, phi) in generalized(&k, &m, n) {
        if lambda_n.abs() <= 1e-9 * largest {
            continue;
        }
        let mut full = vec![0.0; grid.len];
        for (i, &u) in unknowns.iter().enumerate() {
            full[u] = phi[i];
        }
        let (mut ex, mut ey) = (0.0, 0.0);
        let mut rows: std::collections::BTreeMap<i64, (f64, f64)> = Default::default();
        let values: Vec<(f64, f64)> = (0..reference.len())
            .map(|p| (grid.read(&full, p, 0).0, grid.read(&full, p, 1).0))
            .collect();
        for (p, &(ux, uy)) in values.iter().enumerate() {
            ex += v0[p] * ux * ux;
            ey += v0[p] * uy * uy;
            let e = rows.entry(row_of(p)).or_default();
            e.0 += v0[p] * uy;
            e.1 += v0[p];
        }
        let across: f64 = values
            .iter()
            .enumerate()
            .map(|(p, &(_, uy))| {
                let (s, w) = rows[&row_of(p)];
                v0[p] * (uy - s / w).powi(2)
            })
            .sum();
        // Longitudinal: no horizontal motion and uniform across the width.
        if ex > 1e-10 * ey || across > 1e-10 * ey {
            continue;
        }
        found += 1;
        let omega_h = lambda_n.sqrt();
        let omega = found as f64 * std::f64::consts::PI * c0 / cells as f64;
        let theta = found as f64 * std::f64::consts::PI / cells as f64;
        let midpoint: Vec<String> = courants
            .iter()
            .map(|&nu| {
                let dt = nu / c0;
                let omega_d = 2.0 / dt * (0.5 * omega_h * dt).atan();
                format!("{nu} {:+.3e}", omega_d / omega - 1.0)
            })
            .collect();
        println!(
            "  n {found}: theta {theta:.4}, omega_h / omega - 1 {:+.4e} (infinite grid theta^4/1440 {:+.4e}); midpoint omega_d / omega - 1 at Courant {}",
            omega_h / omega - 1.0,
            theta.powi(4) / 1440.0,
            midpoint.join(", ")
        );
        if theta > 2.0 {
            break;
        }
    }
}

/// `EMERGE_PROBE_BAR_CELLS` (default `16,32`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_bar_dispersion() {
    let courants = [0.375, 0.19, 0.095, 0.0475, 0.024];
    let cells = std::env::var("EMERGE_PROBE_BAR_CELLS").unwrap_or_else(|_| "16,32".into());
    for c in cells.split(',').filter_map(|s| s.trim().parse().ok()) {
        dispersion(c, &courants);
    }
}
