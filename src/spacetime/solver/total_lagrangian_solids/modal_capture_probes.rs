//! How much of reference A's vibration a reduced basis holds
//! (`tmp/papers/dl_realtime/STAGE0_MODAL_CAPTURE_PLAN.md`): A's elastic
//! state at frames 8 and 9 projected on (a) the lowest free elastic modes
//! and (b) a massless Craig-Bampton basis (Monjaraz Tec, Gross, Krack 2021,
//! eqs. 20-23, linear at the reference). A probe: measurement, nothing held.

use super::super::impact_local::{FINE, FRAME_S, drop};
use super::super::modal_flight::split;
use super::*;

/// A basis's captured energy, built column by column: each new column made
/// orthonormal in the stiffness's and in the mass's inner products against
/// the ones before (modified Gram-Schmidt, twice), the captured strain and
/// kinetic energies the sums of `1/2 (e^T K u)^2` and `1/2 (e^T M w)^2`. A
/// column whose remaining norm is below `1e-10` of its own adds nothing in
/// that product (the rigid motions in `K`, a dependent column).
struct Capture<'a> {
    n: usize,
    k: &'a [f64],
    m: &'a [f64],
    by_k: Vec<Vec<f64>>,
    by_m: Vec<Vec<f64>>,
    strain: f64,
    kinetic: f64,
}

impl<'a> Capture<'a> {
    fn new(n: usize, k: &'a [f64], m: &'a [f64]) -> Self {
        Self {
            n,
            k,
            m,
            by_k: Vec::new(),
            by_m: Vec::new(),
            strain: 0.0,
            kinetic: 0.0,
        }
    }

    fn times(&self, a: &[f64], x: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|i| (0..self.n).map(|j| a[i * self.n + j] * x[j]).sum())
            .collect()
    }

    fn add(&mut self, column: &[f64], (u, w): (&[f64], &[f64])) {
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        for (product, kept, state) in [(self.k, 0, u), (self.m, 1, w)] {
            let mut e = column.to_vec();
            let own = dot(&e, &self.times(product, &e));
            let basis = if kept == 0 { &self.by_k } else { &self.by_m };
            for _ in 0..2 {
                for b in basis {
                    let c = dot(b, &self.times(product, &e));
                    for (x, y) in e.iter_mut().zip(b) {
                        *x -= c * y;
                    }
                }
            }
            let left = dot(&e, &self.times(product, &e));
            if left <= 1e-10 * own.max(f64::MIN_POSITIVE) || left <= 0.0 {
                continue;
            }
            let scale = left.sqrt();
            e.iter_mut().for_each(|x| *x /= scale);
            let c = dot(&e, &self.times(product, state));
            if kept == 0 {
                self.strain += 0.5 * c * c;
                self.by_k.push(e);
            } else {
                self.kinetic += 0.5 * c * c;
                self.by_m.push(e);
            }
        }
    }

    fn energy(&self) -> f64 {
        self.strain + self.kinetic
    }
}

/// The symmetric generalized eigenproblem `K x = l M x` on `M` definite,
/// as `vibration::modes` solves it (Jacobi on `M`, then on `M^-1/2 K
/// M^-1/2` shifted definite): the eigenvalues ascending and their vectors,
/// mass-orthonormal.
fn generalized(k: &[f64], m: &[f64], n: usize) -> Vec<(f64, Vec<f64>)> {
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

/// The minimum-norm solution of `K_ii x = b` with `K_ii` symmetric
/// semidefinite (its null space the x translation when only the bottom
/// face's normals are held), from its eigenvectors: `b` lies in its range
/// (`K` annihilates the translation, so `t_i^T K_ib = 0`).
fn semidefinite_solve(spectrum: &[(f64, Vec<f64>)], b: &[f64]) -> Vec<f64> {
    let top = spectrum.iter().map(|s| s.0.abs()).fold(0.0, f64::max);
    let mut x = vec![0.0; b.len()];
    for (l, v) in spectrum {
        if *l <= 1e-10 * top {
            continue;
        }
        let c: f64 = v.iter().zip(b).map(|(a, b)| a * b).sum::<f64>() / l;
        x.iter_mut().zip(v).for_each(|(x, v)| *x += c * v);
    }
    x
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_modal_capture() {
    super::super::cap::capped("probe_modal_capture", 3600, || {
        let mut a = drop(
            FRAME_S / FINE as f64,
            TotalLagrangianConfig::new(f64::INFINITY),
        );
        let mut basis: Option<Modes> = None;
        for frame in 0..10 {
            for _ in 0..FINE {
                a.step();
            }
            let state = a.total_lagrangian.as_ref().expect("on");
            let body = basis.get_or_insert_with(|| modes(state));
            if frame < 8 {
                continue;
            }
            let n = body.unknowns.len();
            let (flight, _) = split(state, body, 0.0);
            // The modal coordinates `(q, q')` from `z = (omega q, -q')`,
            // the rigid ones dropped; on the unknowns `u = Phi q`, `w = Phi
            // q'`.
            let coordinates: Vec<(f64, f64)> = (0..n)
                .map(|r| {
                    if r < RIGID {
                        (0.0, 0.0)
                    } else {
                        (flight.z[r].x / body.omega[r], -flight.z[r].y)
                    }
                })
                .collect();
            let field = |pick: &dyn Fn(&(f64, f64)) -> f64| -> Vec<f64> {
                (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|r| body.phi[i * n + r] * pick(&coordinates[r]))
                            .sum()
                    })
                    .collect()
            };
            let (u, w) = (field(&|c| c.0), field(&|c| c.1));
            let by_mode: Vec<f64> = (0..n)
                .map(|r| 0.5 * flight.z[r].length_squared() * f64::from(u8::from(r >= RIGID)))
                .collect();
            let whole: f64 = by_mode.iter().sum();
            let mut check = Capture::new(n, &body.stiffness, &body.mass);
            for r in 0..n {
                let column: Vec<f64> = (0..n).map(|i| body.phi[i * n + r]).collect();
                check.add(&column, (&u, &w));
            }
            println!(
                "Stage 0 frame {frame}: vibration (sum of the elastic modes' 1/2 |z|^2) {whole:.6e} J/m; on the unknowns, 1/2 u K u + 1/2 w M w through every mode {:.6e} J/m",
                check.energy()
            );
            // (a) the lowest N free elastic body.
            let mut cumulative = 0.0;
            let mut free = Vec::new();
            for (r, e) in by_mode.iter().enumerate().skip(RIGID) {
                cumulative += e;
                free.push((r + 1 - RIGID, cumulative / whole));
            }
            let first_in_band = free.iter().find(|(_, f)| *f >= 0.97);
            // The frequency at the cut and the highest, Hz: the step-gain bound `f_max / f_N`.
            let f_max = body.omega[n - 1] / std::f64::consts::TAU;
            if let Some(&(cut, _)) = first_in_band {
                let f_cut = body.omega[RIGID + cut - 1] / std::f64::consts::TAU;
                println!(
                    "Stage 0 frame {frame} (a) frequency at the cut N = {cut}: {f_cut:.0} Hz; highest {f_max:.0} Hz; f_max / f_N {:.3}",
                    f_max / f_cut
                );
            }
            println!(
                "Stage 0 frame {frame} (a) free modes: first N within 3 %: {:?}; captured fraction by N: {}",
                first_in_band,
                free.iter()
                    .map(|(k, f)| format!("{k}:{f:.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            // (b) the bottom face's normal unknowns held, their constraint
            // modes, then the fixed-interface modes ascending.
            let grid = &state.cache.as_ref().expect("built").grid;
            let outline = &state.outlines[0];
            let lowest = outline
                .iter()
                .map(|&q| state.reference[q].y)
                .fold(f64::INFINITY, f64::min);
            let mut boundary: Vec<usize> = Vec::new();
            for &q in outline {
                if state.reference[q].y > lowest + 1e-9 {
                    continue;
                }
                grid.visit(q, 1, |k, weight, _| {
                    if weight != 0.0
                        && let Some(i) = body.unknowns.iter().position(|&u| u == k)
                        && !boundary.contains(&i)
                    {
                        boundary.push(i);
                    }
                });
            }
            boundary.sort_unstable();
            let interior: Vec<usize> = (0..n).filter(|i| !boundary.contains(i)).collect();
            let sub = |a: &[f64], rows: &[usize], cols: &[usize]| -> Vec<f64> {
                rows.iter()
                    .flat_map(|&i| cols.iter().map(move |&j| a[i * n + j]))
                    .collect()
            };
            let ni = interior.len();
            let k_ii = sub(&body.stiffness, &interior, &interior);
            let m_ii = sub(&body.mass, &interior, &interior);
            let fixed = generalized(&k_ii, &m_ii, ni);
            let k_spectrum = {
                let (l, v) = {
                    let top = (0..ni).map(|i| k_ii[i * ni + i]).fold(0.0, f64::max);
                    let mut shifted = k_ii.clone();
                    for i in 0..ni {
                        shifted[i * ni + i] += top;
                    }
                    let (l, v) = jacobi_eigen_symmetric_definite(shifted, ni, 1e-14);
                    (l.iter().map(|e| e - top).collect::<Vec<_>>(), v)
                };
                (0..ni)
                    .map(|e| (l[e], (0..ni).map(|i| v[i * ni + e]).collect()))
                    .collect::<Vec<(f64, Vec<f64>)>>()
            };
            let top = fixed.iter().map(|f| f.0.abs()).fold(0.0, f64::max);
            let zero = fixed.iter().filter(|f| f.0 <= 1e-10 * top).count();
            let mut cb = Capture::new(n, &body.stiffness, &body.mass);
            for r in 0..RIGID {
                let column: Vec<f64> = (0..n).map(|i| body.phi[i * n + r]).collect();
                cb.add(&column, (&u, &w));
            }
            for &j in &boundary {
                let k_ib: Vec<f64> = interior
                    .iter()
                    .map(|&i| -body.stiffness[i * n + j])
                    .collect();
                let inner = semidefinite_solve(&k_spectrum, &k_ib);
                let mut column = vec![0.0; n];
                column[j] = 1.0;
                for (&i, &x) in interior.iter().zip(&inner) {
                    column[i] = x;
                }
                cb.add(&column, (&u, &w));
            }
            let mut enriched = vec![(boundary.len(), cb.energy() / whole)];
            for (count, (_, mode)) in fixed.iter().skip(zero).enumerate() {
                let mut column = vec![0.0; n];
                for (&i, &x) in interior.iter().zip(mode) {
                    column[i] = x;
                }
                cb.add(&column, (&u, &w));
                enriched.push((boundary.len() + count + 1, cb.energy() / whole));
            }
            let first_in_band = enriched.iter().find(|(_, f)| *f >= 0.97);
            // The highest fixed-interface mode retained at that order, Hz (the constraint modes are static: the
            // boundary is massless), against the free body's highest.
            if let Some(&(order, _)) = first_in_band {
                let retained = order - boundary.len();
                let (l, _) = &fixed[zero + retained - 1];
                let f_cut = l.max(0.0).sqrt() / std::f64::consts::TAU;
                let f_max = body.omega[n - 1] / std::f64::consts::TAU;
                println!(
                    "Stage 0 frame {frame} (b) highest retained fixed-interface frequency at order {order} (N = {retained}): {f_cut:.0} Hz; the free body's highest {f_max:.0} Hz; f_max / f_N {:.3}",
                    f_max / f_cut
                );
            }
            println!(
                "Stage 0 frame {frame} (b) massless Craig-Bampton: boundary unknowns B = {} (bottom face's y), interior {ni}, zero-frequency fixed-interface modes {zero} (in the rigid modes, skipped); first order B + N within 3 %: {:?}; captured fraction by order B + N: {}",
                boundary.len(),
                first_in_band,
                enriched
                    .iter()
                    .map(|(k, f)| format!("{k}:{f:.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    });
}
