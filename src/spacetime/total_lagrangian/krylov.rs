//! The linear solvers the step uses: conjugate gradients for the mass and
//! restarted GMRES for Newton's corrections.

use super::*;

/// Jacobi-preconditioned conjugate gradient on the active entries, to a
/// relative residual of 1e-13 (the mass conditioning test's inverse
/// iteration).
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn conjugate_gradient(
    apply: &dyn Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    diag: &[f64],
    active: &[bool],
) -> Vec<f64> {
    let inv: Vec<f64> = (0..b.len())
        .map(|k| {
            if active[k] && diag[k] > 0.0 {
                1.0 / diag[k]
            } else {
                0.0
            }
        })
        .collect();
    let jacobi = |r: &[f64]| r.iter().zip(&inv).map(|(a, b)| a * b).collect();
    preconditioned_cg(apply, b, &jacobi, active)
}

/// Conjugate gradient on the active entries with the preconditioner
/// `precondition`, to a relative residual of 1e-13.
pub(crate) fn preconditioned_cg(
    apply: Operator<'_>,
    b: &[f64],
    precondition: Operator<'_>,
    active: &[bool],
) -> Vec<f64> {
    let n = b.len();
    let masked = |y: Vec<f64>| -> Vec<f64> {
        y.into_iter()
            .zip(active)
            .map(|(e, &on)| if on { e } else { 0.0 })
            .collect()
    };
    let mut x = masked(precondition(b));
    let ax = apply(&x);
    let mut r: Vec<f64> = (0..n).map(|k| b[k] - ax[k]).collect();
    let mut z = masked(precondition(&r));
    let mut p = z.clone();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let norm_b = dot(b, b).sqrt().max(f64::MIN_POSITIVE);
    let mut rz = dot(&r, &z);
    for _ in 0..10 * n.max(1) {
        if dot(&r, &r).sqrt() <= 1e-13 * norm_b {
            break;
        }
        let ap = apply(&p);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        z = masked(precondition(&r));
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for k in 0..n {
            p[k] = z[k] + beta * p[k];
        }
    }
    x
}

/// A linear map on the unknowns.
pub(crate) type Operator<'a> = &'a dyn Fn(&[f64]) -> Vec<f64>;

/// Restarted GMRES with right preconditioning on the active entries;
/// returns the step and the iterations spent.
pub(crate) fn gmres(
    apply: Operator<'_>,
    b: &[f64],
    (precondition, active): (Operator<'_>, &[bool]),
    tolerance: f64,
    settings: &ConservingSettings,
) -> (Vec<f64>, u32) {
    let n = b.len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let norm_b = dot(b, b).sqrt();
    let mut x = vec![0.0; n];
    let mut iterations = 0u32;
    if norm_b == 0.0 {
        return (x, 0);
    }
    let m = settings.gmres_restart;
    while iterations < settings.max_gmres {
        let ax = apply(&x);
        let r: Vec<f64> = (0..n)
            .map(|k| if active[k] { b[k] - ax[k] } else { 0.0 })
            .collect();
        let beta = dot(&r, &r).sqrt();
        if beta <= tolerance * norm_b {
            break;
        }
        let mut basis: Vec<Vec<f64>> = vec![r.iter().map(|e| e / beta).collect()];
        let mut hess = vec![vec![0.0; m]; m + 1];
        let (mut cs, mut sn) = (vec![0.0; m], vec![0.0; m]);
        let mut g = vec![0.0; m + 1];
        g[0] = beta;
        let mut used = 0;
        for col in 0..m {
            iterations += 1;
            let z = precondition(&basis[col]);
            let mut w = apply(&z);
            for (k, value) in w.iter_mut().enumerate() {
                if !active[k] {
                    *value = 0.0;
                }
            }
            for (row, q) in basis.iter().enumerate() {
                let hval = dot(&w, q);
                hess[row][col] = hval;
                for k in 0..n {
                    w[k] -= hval * q[k];
                }
            }
            let wn = dot(&w, &w).sqrt();
            hess[col + 1][col] = wn;
            for row in 0..col {
                let temp = cs[row] * hess[row][col] + sn[row] * hess[row + 1][col];
                hess[row + 1][col] = -sn[row] * hess[row][col] + cs[row] * hess[row + 1][col];
                hess[row][col] = temp;
            }
            let denom = (hess[col][col].powi(2) + hess[col + 1][col].powi(2)).sqrt();
            if denom == 0.0 {
                used = col;
                break;
            }
            cs[col] = hess[col][col] / denom;
            sn[col] = hess[col + 1][col] / denom;
            hess[col][col] = denom;
            hess[col + 1][col] = 0.0;
            g[col + 1] = -sn[col] * g[col];
            g[col] *= cs[col];
            used = col + 1;
            if g[col + 1].abs() <= tolerance * norm_b || wn == 0.0 {
                break;
            }
            basis.push(w.iter().map(|e| e / wn).collect());
        }
        // Back substitution for the coefficients, then x += M^-1 V y.
        let mut y = vec![0.0; used];
        for row in (0..used).rev() {
            let mut sum = g[row];
            for c in row + 1..used {
                sum -= hess[row][c] * y[c];
            }
            y[row] = sum / hess[row][row];
        }
        let mut combination = vec![0.0; n];
        for (c, coef) in y.iter().enumerate() {
            for k in 0..n {
                combination[k] += coef * basis[c][k];
            }
        }
        for (xk, zk) in x.iter_mut().zip(precondition(&combination)) {
            *xk += zk;
        }
        if g[used].abs() <= tolerance * norm_b {
            break;
        }
    }
    (x, iterations)
}
