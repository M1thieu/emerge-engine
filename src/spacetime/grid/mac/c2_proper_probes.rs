//! C2 proper's P0 (C2_PROPER_PLAN.md sec. 7) on the bar's own grid: the
//! singular dynamic mass `M_bar = M H (H^T M H)^-1 H^T M` (Pozzolini, Renard,
//! Salaun 2013, `B^T C^-1 B` with `H^h` the span of the non-strip unknowns
//! plus the linear fields with a strip part), read only: the inf-sup rank
//! test, the six moments, `K` on `Ker B`, the strip's share of `M_bar - M`,
//! and the C2 diagnostic's Newton matrix's definiteness. A probe.

use super::MacLayout;

use super::gates::gauss_legendre;
use crate::spacetime::total_lagrangian::{ConservingSettings, Grid};
use crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite;
use glam::{DMat2, DVec2};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn mat_vec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| dot(&a[i * n..(i + 1) * n], x)).collect()
}

/// The lower Cholesky factor of a row-major `n x n` definite matrix.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            l[i * n + j] = if i == j {
                assert!(s > 0.0, "not definite at {i}: {s:e}");
                s.sqrt()
            } else {
                s / l[j * n + j]
            };
        }
    }
    l
}

/// `A^-1 b` from `cholesky(A)`.
fn solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..n {
        y[i] = (y[i] - (0..i).map(|k| l[i * n + k] * y[k]).sum::<f64>()) / l[i * n + i];
    }
    for i in (0..n).rev() {
        y[i] = (y[i] - (i + 1..n).map(|k| l[k * n + i] * y[k]).sum::<f64>()) / l[i * n + i];
    }
    y
}

/// The rank of a set of vectors under the inner product `inner`, by
/// Gram-Schmidt, a vector counting when its remainder exceeds `1e-10` of
/// its own size; the orthonormal basis kept.
fn rank(vectors: &[Vec<f64>], inner: &dyn Fn(&[f64], &[f64]) -> f64) -> (usize, Vec<Vec<f64>>) {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for v in vectors {
        let size = inner(v, v).sqrt();
        let mut r = v.clone();
        for b in &basis {
            let c = inner(&r, b);
            r.iter_mut().zip(b).for_each(|(x, y)| *x -= c * y);
        }
        let norm = inner(&r, &r).sqrt();
        if norm > 1e-10 * size.max(f64::MIN_POSITIVE) {
            r.iter_mut().for_each(|x| *x /= norm);
            basis.push(r);
        }
    }
    (basis.len(), basis)
}

/// The smallest and largest eigenvalues of a symmetric `n x n` matrix.
fn extremes(a: Vec<f64>, n: usize) -> (f64, f64) {
    let (d, _) = jacobi_eigen_symmetric_definite(a, n, 1e-14);
    d.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &e| {
            (lo.min(e), hi.max(e))
        })
}

/// P0 at `cells` (the bar of `bar_impact_run`: width 2 cells, Poisson's ratio
/// 0, 3 x 3 Gauss points per half cell, per unit density, lengths in cells).
fn p0(cells: usize) {
    let width = 2usize;
    let (e, rho, length) = (11.25e6f64, 1700.0f64, 1.0f64);
    let cell_m = length / cells as f64;
    let c0 = (e / rho).sqrt() / cell_m;
    let mu = 0.5 * c0 * c0;
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
    let body = reference.len();
    // The contact points of the bottom face (`bar_impact_run`'s), no mass.
    let mut along = vec![x0];
    along.extend((0..width).map(|i| x0 + i as f64 + 0.5));
    along.push(x0 + width as f64);
    for &xc in &along {
        reference.push(DVec2::new(xc, bottom));
        v0.push(0.0);
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
    let contacts: Vec<(usize, DVec2)> = (body..reference.len()).map(|p| (p, DVec2::Y)).collect();
    p0_core(
        &format!("h = L/{cells}"),
        &grid,
        (&reference, &mass, &v0),
        &|_| (mu, 0.0),
        &contacts,
        &[(0.375, 0.375 / c0), (0.0475, 0.0475 / c0)],
    );
}

/// P0's measurements on any body's grid (C2_PROPER_PLAN.md sec. 7):
/// `contacts` the contact points with their outward-to-inward normals,
/// `lame(p)` a particle's `(mu, lambda)`, `diagnostics` the `(Courant
/// label, dt)` at which the C2 diagnostic's Newton matrix is checked.
pub(crate) fn p0_core(
    label: &str,
    grid: &Grid,
    (reference, mass, v0): (&[DVec2], &[f64], &[f64]),
    lame: &dyn Fn(usize) -> (f64, f64),
    contacts: &[(usize, DVec2)],
    diagnostics: &[(f64, f64)],
) {
    let particles: Vec<usize> = (0..reference.len()).filter(|&p| v0[p] > 0.0).collect();
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    let mut slot = vec![usize::MAX; grid.len];
    for (i, &k) in unknowns.iter().enumerate() {
        slot[k] = i;
    }
    // `M` and `K` (the step's linear elasticity at rest, `lambda = 0`).
    let (mut m, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (j, &column) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[column] = 1.0;
        let mc = grid.mass_times(&unit, mass, 0.0);
        let mut kc = vec![0.0; grid.len];
        for &p in &particles {
            let (_, gu) = grid.read(&unit, p, 0);
            let (_, gv) = grid.read(&unit, p, 1);
            let g = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
            let (mu, lambda) = lame(p);
            let sigma =
                mu * (g + g.transpose()) + lambda * (g.x_axis.x + g.y_axis.y) * DMat2::IDENTITY;
            for a in 0..2 {
                let row = DVec2::new(sigma.x_axis[a], sigma.y_axis[a]);
                grid.visit(p, a, |q, _, grad| kc[q] += v0[p] * row.dot(grad));
            }
        }
        for (i, &r) in unknowns.iter().enumerate() {
            m[i * n + j] = mc[r];
            k[i * n + j] = kc[r];
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (
                0.5 * (m[i * n + j] + m[j * n + i]),
                0.5 * (k[i * n + j] + k[j * n + i]),
            );
            (m[i * n + j], m[j * n + i], k[i * n + j], k[j * n + i]) = (a, a, b, b);
        }
    }
    let lm = cholesky(&m, n);
    // The contact rows `G` (the normal component at each contact point) and
    // the strip `S`: the unknowns with a nonzero weight in some row.
    let rows: Vec<Vec<f64>> = contacts
        .iter()
        .map(|&(p, normal)| {
            let mut g = vec![0.0; n];
            for a in 0..2 {
                grid.visit(p, a, |q, w, _| {
                    if slot[q] != usize::MAX {
                        g[slot[q]] += normal[a] * w;
                    }
                });
            }
            g
        })
        .collect();
    let in_strip: Vec<bool> = (0..n).map(|i| rows.iter().any(|g| g[i] != 0.0)).collect();
    let strip = in_strip.iter().filter(|&&s| s).count();
    // The six linear fields, `M^-1 b` (exact if the basis reproduces them),
    // and their reproduction error at the particles.
    let total: f64 = particles.iter().map(|&p| mass[p]).sum();
    let centre = particles
        .iter()
        .fold(DVec2::ZERO, |c, &p| c + mass[p] * reference[p])
        / total;
    let fields: [&dyn Fn(DVec2) -> DVec2; 6] = [
        &|_| DVec2::X,
        &|_| DVec2::Y,
        &|q| DVec2::new((q - centre).x, 0.0),
        &|q| DVec2::new((q - centre).y, 0.0),
        &|q| DVec2::new(0.0, (q - centre).x),
        &|q| DVec2::new(0.0, (q - centre).y),
    ];
    let mut reproduction = 0.0f64;
    let linear: Vec<Vec<f64>> = fields
        .iter()
        .map(|f| {
            let mut b = vec![0.0; n];
            for &p in &particles {
                let value = f(reference[p]);
                for a in 0..2 {
                    grid.visit(p, a, |q, w, _| {
                        if slot[q] != usize::MAX {
                            b[slot[q]] += mass[p] * value[a] * w;
                        }
                    });
                }
            }
            let l = solve(&lm, &b, n);
            let mut full = vec![0.0; grid.len];
            for (i, &u) in unknowns.iter().enumerate() {
                full[u] = l[i];
            }
            for &p in &particles {
                let value = f(reference[p]);
                let got = DVec2::new(grid.read(&full, p, 0).0, grid.read(&full, p, 1).0);
                reproduction = reproduction.max((got - value).length());
            }
            l
        })
        .collect();
    // `H`: the non-strip unit vectors and the fields with a strip part.
    let added: Vec<usize> = (0..6)
        .filter(|&j| {
            let s: f64 = (0..n)
                .filter(|&i| in_strip[i])
                .map(|i| linear[j][i].powi(2))
                .sum();
            s.sqrt() > 1e-12 * dot(&linear[j], &linear[j]).sqrt()
        })
        .collect();
    let mut h: Vec<Vec<f64>> = (0..n)
        .filter(|&i| !in_strip[i])
        .map(|i| {
            let mut e = vec![0.0; n];
            e[i] = 1.0;
            e
        })
        .collect();
    h.extend(added.iter().map(|&j| linear[j].clone()));
    let nh = h.len();
    let mh: Vec<Vec<f64>> = h.iter().map(|v| mat_vec(&m, v, n)).collect();
    let mut c = vec![0.0; nh * nh];
    for a in 0..nh {
        for b in 0..nh {
            c[a * nh + b] = dot(&h[a], &mh[b]);
        }
    }
    let lc = cholesky(&c, nh);
    // `M_bar = (M H) C^-1 (M H)^T`, column by column.
    let mut m_bar = vec![0.0; n * n];
    for j in 0..n {
        let rhs: Vec<f64> = mh.iter().map(|v| v[j]).collect();
        let y = solve(&lc, &rhs, nh);
        for i in 0..n {
            m_bar[i * n + j] = (0..nh).map(|a| mh[a][i] * y[a]).sum();
        }
    }
    // (b) The six moments.
    let moment = |a: &[f64], u: &[f64], v: &[f64]| dot(u, &mat_vec(a, v, n));
    let scale = (0..6)
        .flat_map(|j| (0..6).map(move |l| (j, l)))
        .map(|(j, l)| moment(&m, &linear[j], &linear[l]).abs())
        .fold(0.0, f64::max);
    let moments = (0..6)
        .flat_map(|j| (0..6).map(move |l| (j, l)))
        .map(|(j, l)| {
            (moment(&m_bar, &linear[j], &linear[l]) - moment(&m, &linear[j], &linear[l])).abs()
        })
        .fold(0.0, f64::max)
        / scale;
    // (d) `M_bar - M`: its size, and what of it lies off the strip block
    // (zero by the derivation: `M_bar e_j = M e_j` for `j` not in `S`).
    let (mut total, mut off) = (0.0f64, 0.0f64);
    let size: f64 = m.iter().map(|e| e * e).sum::<f64>().sqrt();
    for i in 0..n {
        for j in 0..n {
            let d = (m_bar[i * n + j] - m[i * n + j]).powi(2);
            total += d;
            if !(in_strip[i] && in_strip[j]) {
                off += d;
            }
        }
    }
    // `Ker B`: the strip unit vectors made `M`-orthogonal to `H`.
    let inner = |a: &[f64], b: &[f64]| dot(a, &mat_vec(&m, b, n));
    let candidates: Vec<Vec<f64>> = (0..n)
        .filter(|&i| in_strip[i])
        .map(|i| {
            let mut u = vec![0.0; n];
            u[i] = 1.0;
            let rhs: Vec<f64> = mh.iter().map(|v| v[i]).collect();
            let y = solve(&lc, &rhs, nh);
            for (a, v) in h.iter().enumerate() {
                u.iter_mut().zip(v).for_each(|(x, hx)| *x -= y[a] * hx);
            }
            u
        })
        .collect();
    let (kernel, z) = rank(&candidates, &inner);
    // (a) The inf-sup rank test: `rank(G Z)` against `rank(G)`.
    let euclid = |a: &[f64], b: &[f64]| dot(a, b);
    let (rank_g, _) = rank(&rows, &euclid);
    let gz: Vec<Vec<f64>> = rows
        .iter()
        .map(|g| z.iter().map(|zv| dot(g, zv)).collect())
        .collect();
    let (rank_gz, _) = rank(&gz, &euclid);
    // (c) `K` on `Ker B` (`Z` is `M`-orthonormal: generalized eigenvalues).
    let mut zk = vec![0.0; kernel * kernel];
    for a in 0..kernel {
        let kz = mat_vec(&k, &z[a], n);
        for b in 0..kernel {
            zk[a * kernel + b] = dot(&z[b], &kz);
        }
    }
    let (k_low, k_high) = extremes(zk, kernel);
    // The C2 diagnostic's Newton matrix, `(2/dt^2) M_masked + K`, the
    // strip's rows and columns of `M` zeroed, at two Courant numbers.
    let diagnostic: Vec<(f64, f64, f64)> = diagnostics
        .iter()
        .map(|&(nu, dt)| {
            let mut j = k.clone();
            for a in 0..n {
                for b in 0..n {
                    if !(in_strip[a] || in_strip[b]) {
                        j[a * n + b] += 2.0 / (dt * dt) * m[a * n + b];
                    }
                }
            }
            let (lo, hi) = extremes(j, n);
            (nu, lo, hi)
        })
        .collect();

    println!(
        "C2 P0, {label}: {n} unknowns, strip {strip}, contact rows {}, linear fields reproduced to {reproduction:.1e}, added to H {} of 6 ({:?}); dim H {nh}",
        rows.len(),
        added.len(),
        added
    );
    println!(
        "  (a) dim Ker B {kernel}, rank G {rank_g}, rank G on Ker B {rank_gz}: {}",
        if rank_gz == rank_g {
            "inf-sup holds"
        } else {
            "inf-sup FAILS"
        }
    );
    println!(
        "  (b) six moments, largest |L^T (M_bar - M) L| over the largest |L^T M L|: {moments:.2e}: {}",
        if moments < 1e-10 { "kept" } else { "NOT kept" }
    );
    println!(
        "  (c) K on Ker B, generalized eigenvalues {k_low:.4e} to {k_high:.4e}: {}",
        if k_low > 0.0 {
            "definite"
        } else {
            "NOT definite"
        }
    );
    for (nu, lo, hi) in diagnostic {
        println!(
            "      the C2 diagnostic's (2/dt^2) M_masked + K at Courant {nu}: eigenvalues {lo:.4e} to {hi:.4e} (ratio {:.2e})",
            lo / hi
        );
    }
    println!(
        "  (d) |M_bar - M| / |M| {:.3e}, of it off the strip block {:.1e}",
        total.sqrt() / size,
        (off / total.max(f64::MIN_POSITIVE)).sqrt()
    );
}

/// `EMERGE_PROBE_BAR_CELLS` (default `16,32,64`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_c2_proper_p0() {
    let cells = std::env::var("EMERGE_PROBE_BAR_CELLS").unwrap_or_else(|_| "16,32,64".into());
    for c in cells.split(',').filter_map(|s| s.trim().parse().ok()) {
        p0(c);
    }
}
