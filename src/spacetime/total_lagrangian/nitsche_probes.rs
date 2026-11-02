//! FAST_MODE_PLAN.md sec. 36.18 (R-b), the counts before any code:
//! Chouly, Hild and Renard's Nitsche contact (M2AN 49 (2015), part 1 eqs.
//! 2.4-2.6, `Theta = 1`) at the outline points, each with its trapezoid
//! weight on its wall (their Gauss points, an adaptation), on the linear
//! midpoint step of the solid's linear elasticity at `F = I`
//! (`Grid::approximate_jacobian` with the inertia `4 M / dt^2` of the step
//! in the midpoint displacement). In this engine's signs (gap `g >= 0`,
//! pressure `p = -sigma_n >= 0`) the source's `P_gamma(u) = u_n - gamma
//! sigma_n(u)` is `-g + gamma p`. Test-only, printed, nothing changed.

use super::multipliers::dense_solve;
use super::stabilized_probes::{Wall, conditions, inverse, products, rectangle, sorted};
use super::*;
use crate::forces::fields::EARTH_GRAVITY_M_S2;
use crate::matter::materials::lame_parameters::lame_from_young;
use crate::matter::materials::liquid::liquid_water::LiquidWater;
use crate::matter::materials::relaxation::ViscoelasticRecord;

/// One wall point: its gap row `G`, its pressure row `P` (`p = -sigma_NN`
/// of the linear law, normal on the wall's axis), its trapezoid weight, its
/// wall and its particle.
struct Point {
    gap: Vec<f64>,
    pressure: Vec<f64>,
    weight: f64,
    wall: usize,
    particle: usize,
}

/// Every outline point on a wall, a rectangle's corner weighted a quarter
/// cell on each of its faces, the others half a cell.
fn wall_points(
    grid: &Grid,
    (x, outline, corners): (&[DVec2], &[usize], &[usize]),
    walls: &[Wall],
    (mu, lame, h): (f64, f64, f64),
) -> Vec<Point> {
    let mut all = Vec::new();
    for (r, &q) in outline.iter().enumerate() {
        for (i, wall) in walls.iter().enumerate() {
            if (x[q][wall.axis] - wall.at).abs() > 1e-9 {
                continue;
            }
            let (mut gap, mut pressure) = (vec![0.0; grid.len], vec![0.0; grid.len]);
            grid.visit(q, wall.axis, |k, n, _| gap[k] += wall.inward * n);
            for b in 0..2 {
                grid.visit(q, b, |k, _, g| {
                    let normal = if b == wall.axis {
                        2.0 * mu * g[wall.axis]
                    } else {
                        0.0
                    };
                    pressure[k] -= normal + lame * g[b];
                });
            }
            all.push(Point {
                gap,
                pressure,
                weight: if corners.contains(&r) {
                    0.25 * h
                } else {
                    0.5 * h
                },
                wall: i,
                particle: q,
            });
        }
    }
    all
}

/// The Nitsche terms of the step as `U C U^T`, balanced by a diagonal
/// congruence so that `C^-1`'s blocks are of order one: an active point
/// adds `(w / gamma) G G^T - w G P^T - Theta w P G^T` (rows test, columns
/// trial: `(w / gamma) (G - Theta gamma P)(G - gamma P)^T` with `A^Theta`'s
/// `-Theta w gamma P P^T`; its columns `G sqrt(w / gamma)`, `P sqrt(w
/// gamma)`, `C^-1` block `-(1 / Theta) [[0, 1], [Theta, 1]]`); an inactive
/// one keeps only `-Theta w gamma P P^T` (column `P sqrt(w gamma)`, `C^-1`
/// entry `-1 / Theta`). `Theta` is 1 or -1. Returns the columns and `C^-1`.
fn nitsche_terms(
    points: &[Point],
    active: &[bool],
    (gamma, theta): (f64, f64),
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut columns: Vec<Vec<f64>> = Vec::new();
    let mut blocks: Vec<Vec<(usize, f64)>> = Vec::new();
    for (p, &on) in points.iter().zip(active) {
        let (a, b) = ((p.weight / gamma).sqrt(), (p.weight * gamma).sqrt());
        let i = columns.len();
        if on {
            columns.push(p.gap.iter().map(|g| a * g).collect());
            columns.push(p.pressure.iter().map(|q| b * q).collect());
            blocks.push(vec![(i + 1, -1.0 / theta)]);
            blocks.push(vec![(i, -1.0), (i + 1, -1.0 / theta)]);
        } else {
            columns.push(p.pressure.iter().map(|q| b * q).collect());
            blocks.push(vec![(i, -1.0 / theta)]);
        }
    }
    let k = columns.len();
    let mut c_inverse = vec![vec![0.0; k]; k];
    for (i, row) in blocks.iter().enumerate() {
        for &(j, v) in row {
            c_inverse[i][j] = v;
        }
    }
    (columns, c_inverse)
}

/// `(A + U C U^T)^-1 b` by Woodbury's identity, `A^-1 b - Z (C^-1 + U^T Z)^-1
/// U^T A^-1 b`, `Z = A^-1 U`; `None` when the inner matrix is singular.
fn woodbury(
    a: &Band,
    (columns, c_inverse): (&[Vec<f64>], &[Vec<f64>]),
    b: &[f64],
) -> Option<Vec<f64>> {
    let base = a.solve(b);
    if columns.is_empty() {
        return Some(base);
    }
    let rows: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
    let z: Vec<Vec<f64>> = rows.iter().map(|u| a.solve(u)).collect();
    let mut inner = products(&rows, &z);
    for (row, extra) in inner.iter_mut().zip(c_inverse) {
        for (e, x) in row.iter_mut().zip(extra) {
            *e += x;
        }
    }
    let projected: Vec<f64> = rows
        .iter()
        .map(|u| u.iter().zip(&base).map(|(p, q)| p * q).sum())
        .collect();
    let weights = dense_solve(inner, projected)?;
    let mut y = base;
    for (zj, &wj) in z.iter().zip(&weights) {
        for (yk, zk) in y.iter_mut().zip(zj) {
            *yk -= wj * zk;
        }
    }
    Some(y)
}

/// `gamma0` by K-6's rule (sec. 36.17), on the points: the largest
/// eigenvalue of `D^1/2 P A^-1 P^T D^1/2`, `D = diag(w_q h)`, `A` the
/// operator K-6 took (inertia `2 M / dt^2`), set to 1/2.
fn gamma0(a: &Band, points: &[Point], h: f64) -> f64 {
    let rows: Vec<&[f64]> = points.iter().map(|p| p.pressure.as_slice()).collect();
    let z: Vec<Vec<f64>> = rows.iter().map(|r| a.solve(r)).collect();
    let pp = products(&rows, &z);
    let d: Vec<f64> = points.iter().map(|p| p.weight * h).collect();
    let scaled: Vec<Vec<f64>> = (0..d.len())
        .map(|i| {
            (0..d.len())
                .map(|j| (d[i] * d[j]).sqrt() * pp[i][j])
                .collect()
        })
        .collect();
    0.5 / *sorted(scaled).last().expect("points")
}

/// N-1, by inertia: `A` is positive, so `K_N = A + U C U^T` is positive
/// definite exactly when `-C^-1 - U^T A^-1 U` has as many negative
/// eigenvalues as `-C^-1` (Haynsworth's inertia additivity; Sylvester's law
/// for the balancing congruence). Every point active. Returns the two
/// counts and the eigenvalue of `-C^-1 - U^T A^-1 U` nearest zero.
fn inertia(a: &Band, points: &[Point], gamma: f64) -> (usize, usize, f64) {
    let (columns, c_inverse) = nitsche_terms(points, &vec![true; points.len()], (gamma, 1.0));
    let rows: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
    let z: Vec<Vec<f64>> = rows.iter().map(|u| a.solve(u)).collect();
    let q = products(&rows, &z);
    let schur: Vec<Vec<f64>> = q
        .iter()
        .zip(&c_inverse)
        .map(|(qr, cr)| qr.iter().zip(cr).map(|(x, c)| -x - c).collect())
        .collect();
    let values = sorted(schur);
    let negative = values.iter().filter(|&&v| v < 0.0).count();
    let nearest = values
        .iter()
        .fold(f64::INFINITY, |m, &v| if v.abs() < m.abs() { v } else { m });
    (negative, points.len(), nearest)
}

/// N-2 and N-3 on G2's tank at cell `h`: one linear midpoint step of `dt`
/// from the exact hydrostatic start (projected on the basis as in K-7), at
/// rest, with Nitsche (`Theta = 1`, active set iterated until it repeats)
/// and with today's two-cell multipliers held at the step's end (Part I);
/// the outline points' end gaps, and Nitsche's contact pressure `[-g +
/// gamma p]_+ / gamma` at the midpoint against the exact law's. With
/// `(theta, penalty)` other than `(1, 1)` (N-5, sec. 36.18b) the penalty
/// `1 / gamma` is `penalty` times K-6's minimum and one N-5 line is printed
/// instead, with the stiffness `1 + lambda_max(C U^T A^-1 U)` (power
/// iteration).
fn tank_step(h: f64, dt: f64, (theta, penalty): (f64, f64)) {
    let reported = theta == 1.0 && penalty == 1.0;
    let water = LiquidWater::at(293.15).expect("in range");
    let (rho, bulk) = (water.density(), water.isentropic_bulk());
    let g = EARTH_GRAVITY_M_S2 as f64;
    let (width, depth) = (0.32, 0.12);
    let corner = DVec2::new(0.04, 0.04);
    let (x, mass, outline, corners) = rectangle(corner, DVec2::new(width, depth), h, rho);
    let layout = GridLayout::new(
        ((width + 0.08) / h).round() as usize,
        ((depth + 0.08) / h).round() as usize,
        h as f32,
    );
    let volume: Vec<f64> = mass
        .iter()
        .map(|&m| if m > 0.0 { 0.25 * h * h } else { 0.0 })
        .collect();
    let open = |_: usize, _: usize, _: usize| None;
    let grid = Grid::new(&layout, &x, (&mass, &volume), &open, Some(0.75));
    let solid = Elastic::barotropic(bulk);
    let solids = vec![solid; x.len()];
    let walls = [
        Wall {
            axis: 1,
            inward: 1.0,
            at: corner.y,
        },
        Wall {
            axis: 0,
            inward: 1.0,
            at: corner.x,
        },
        Wall {
            axis: 0,
            inward: -1.0,
            at: corner.x + width,
        },
    ];
    let points = wall_points(&grid, (&x, &outline, &corners), &walls, (0.0, bulk, h));
    let k6 = grid.approximate_jacobian((&mass, &volume, &solids), 0.0, 2.0 / (dt * dt), &[]);
    let gamma = gamma0(&k6, &points, h) * h / penalty;
    let a = grid.approximate_jacobian((&mass, &volume, &solids), 0.0, 4.0 / (dt * dt), &[]);
    // The start: the exact displacement projected (K-7), at rest.
    let displacement: Vec<DVec2> = x
        .iter()
        .map(|p| {
            let big_y = p.y - corner.y;
            DVec2::new(0.0, -rho * g * (depth * big_y - 0.5 * big_y * big_y) / bulk)
        })
        .collect();
    let factor = grid.mass_band(&mass, 0.0);
    let (u_n, _) = grid_velocity(&grid, (&mass, &displacement), 0.0, &factor);
    // `f + 4 M u_n / dt^2`, gravity on the vertical lattice.
    let inertia = grid.mass_times_unmasked(&u_n, &mass, 0.0);
    let b: Vec<f64> = (0..grid.len)
        .map(|k| {
            let weight = if grid.axis_of[k] == 1 {
                -grid.lumped[k] * g
            } else {
                0.0
            };
            weight + 4.0 / (dt * dt) * inertia[k]
        })
        .collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    let end_gaps = |u_m: &[f64]| -> Vec<f64> {
        points
            .iter()
            .map(|p| 2.0 * dot(&p.gap, u_m) - dot(&p.gap, &u_n))
            .collect()
    };
    // Nitsche (`nitsche_terms`); a point is active while `-g + gamma p > 0`
    // at the midpoint, `-(G - gamma P) . u_m` (the start is stress-free and
    // on the walls at `u = 0`).
    let combined: Vec<Vec<f64>> = points
        .iter()
        .map(|p| {
            p.gap
                .iter()
                .zip(&p.pressure)
                .map(|(gk, pk)| gk - gamma * pk)
                .collect()
        })
        .collect();
    let mut active = vec![true; points.len()];
    let mut passes = 0;
    let u_m = loop {
        passes += 1;
        let (columns, c_inverse) = nitsche_terms(&points, &active, (gamma, theta));
        let Some(u_m) = woodbury(&a, (&columns, &c_inverse), &b) else {
            println!(
                "N-2 water tank, cell {h} m: the Woodbury inner matrix is singular; not measured"
            );
            return;
        };
        let next: Vec<bool> = combined.iter().map(|v| -dot(v, &u_m) > 0.0).collect();
        if next == active || passes >= 30 {
            break u_m;
        }
        active = next;
    };
    let nitsche_gaps = end_gaps(&u_m);
    // N-3: Nitsche's pressure at the midpoint against the exact law's.
    let stretch = |big_y: f64| 1.0 - rho * g * (depth - big_y) / bulk;
    let exact = |p: DVec2, axis: usize| {
        let f = DMat2::from_diagonal(DVec2::new(1.0, stretch(p.y - corner.y)));
        let c = f.transpose() * f;
        -(f * solid.step_stress(c, c, 1.0)).col(axis)[axis]
    };
    let scale = rho * g * depth;
    let mut pressure_error = [0.0f64; 3];
    let mut diagnostic = [0.0f64; 3];
    for (p, v) in points.iter().zip(&combined) {
        let returned = (-dot(v, &u_m)).max(0.0) / gamma;
        let truth = exact(x[p.particle], walls[p.wall].axis);
        let e = &mut pressure_error[p.wall];
        *e = e.max((returned - truth).abs() / scale);
        // A diagnostic beside the criterion: the error without the
        // rectangle's corners, and each corner's, with the part `g / gamma`.
        let corner = p.weight < 0.375 * h;
        let d = &mut diagnostic[p.wall];
        if corner {
            if reported {
                println!(
                    "N-3 diagnostic, cell {h} m, wall {} corner at ({:.3}, {:.3}) m: returned {returned:.4e} Pa, exact {truth:.4e} Pa, end gap over gamma {:.4e} Pa",
                    p.wall,
                    x[p.particle].x,
                    x[p.particle].y,
                    nitsche_gaps[points
                        .iter()
                        .position(|o| std::ptr::eq(o, p))
                        .expect("listed")]
                        / gamma
                );
            }
        } else {
            *d = d.max((returned - truth).abs() / scale);
        }
    }
    if !reported {
        let (columns, c_inverse) = nitsche_terms(&points, &active, (gamma, theta));
        let rows: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();
        let z: Vec<Vec<f64>> = rows.iter().map(|u| a.solve(u)).collect();
        let q = products(&rows, &z);
        let c = inverse(&c_inverse);
        let k = q.len();
        let cq: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| (0..k).map(|l| c[i][l] * q[l][j]).sum())
                    .collect()
            })
            .collect();
        let mut v = vec![1.0; k];
        let mut lambda = 0.0;
        for _ in 0..500 {
            let w: Vec<f64> = cq
                .iter()
                .map(|row| row.iter().zip(&v).map(|(x, y)| x * y).sum())
                .collect();
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            lambda = w.iter().zip(&v).map(|(x, y)| x * y).sum::<f64>()
                / v.iter().map(|x| x * x).sum::<f64>();
            v = w.iter().map(|x| x / norm).collect();
        }
        println!(
            "N-5 water tank, theta {theta}, penalty {penalty} x K-6's minimum, cell {h} m: largest |end gap| {:.4e} m; pressure error floor {:.4e} (without corners {:.4e}), left wall {:.4e}, right wall {:.4e}; stiffness 1 + lambda_max {:.4e}; {} of {} active after {passes} passes",
            nitsche_gaps.iter().fold(0.0f64, |m, g| m.max(g.abs())),
            pressure_error[0],
            diagnostic[0],
            pressure_error[1],
            pressure_error[2],
            1.0 + lambda,
            active.iter().filter(|&&a| a).count(),
            points.len()
        );
        return;
    }
    for (name, d) in ["floor", "left wall", "right wall"].iter().zip(diagnostic) {
        println!("N-3 diagnostic, cell {h} m, {name}: largest error without the corners {d:.4e}");
    }
    // Today's two-cell multipliers held at the step's end: `G_c (2 u_m -
    // u_n) = 0`, `A u_m = b + G_c^T lambda`.
    let trace = outline_trace(&x, &outline, &corners, h);
    let patches = conditions(&grid, &x, &outline, &trace, &walls, (0.0, bulk));
    let rows: Vec<&[f64]> = patches.iter().map(|c| c.gap.as_slice()).collect();
    let z_g: Vec<Vec<f64>> = rows.iter().map(|r| a.solve(r)).collect();
    let schur = products(&rows, &z_g);
    let base = a.solve(&b);
    let target: Vec<f64> = rows
        .iter()
        .map(|r| 0.5 * dot(r, &u_n) - dot(r, &base))
        .collect();
    let lambda: Vec<f64> = inverse(&schur)
        .iter()
        .map(|row| row.iter().zip(&target).map(|(p, q)| p * q).sum())
        .collect();
    let mut u_patch = base;
    for (z, l) in z_g.iter().zip(&lambda) {
        for (uk, zk) in u_patch.iter_mut().zip(z) {
            *uk += l * zk;
        }
    }
    let patch_gaps = end_gaps(&u_patch);
    let largest = |v: &[f64]| v.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    println!(
        "N-2 water tank, cell {h} m, dt {dt:.3e} s: gamma {gamma:.4e} m/Pa, {} of {} wall points active after {passes} passes; largest |end gap| over the outline points: Nitsche {:.4e} m, two-cell multipliers {:.4e} m (their multipliers' sign: {} of {} pushing)",
        active.iter().filter(|&&a| a).count(),
        points.len(),
        largest(&nitsche_gaps),
        largest(&patch_gaps),
        lambda.iter().filter(|&&l| l >= 0.0).count(),
        lambda.len()
    );
    for (name, e) in ["floor", "left wall", "right wall"]
        .iter()
        .zip(pressure_error)
    {
        println!(
            "N-3 water tank, cell {h} m, {name}: largest |Nitsche's pressure - exact| / (rho g H0) {e:.4e}"
        );
    }
}

#[test]
#[ignore = "a measurement (FAST_MODE_PLAN.md sec. 36.18): run with --ignored --nocapture"]
fn nitsche_walls_counts() {
    let dt = 1.0 / 120.0;
    let water = LiquidWater::at(293.15).expect("in range");
    let plasticine = ViscoelasticRecord::plasticine_crandall_1970_fresh().relaxed;
    let (lambda, mu) = lame_from_young(plasticine.e_pa, plasticine.nu);
    let scenes = [
        (
            "Plasticine block 8 x 4 cm on a floor",
            Elastic::corotated(lambda as f64, mu as f64),
            plasticine.rho_kg_m3 as f64,
            DVec2::new(0.08, 0.04),
            false,
        ),
        (
            "water tank 32 x 12 cm, floor and side walls",
            Elastic::barotropic(water.isentropic_bulk()),
            water.density(),
            DVec2::new(0.32, 0.12),
            true,
        ),
    ];
    for (name, solid, rho, size, sides) in scenes {
        for h in [0.04, 0.02, 0.01] {
            let corner = DVec2::new(0.04, 0.04);
            let (x, mass, outline, corners) = rectangle(corner, size, h, rho);
            let layout = GridLayout::new(
                ((size.x + 0.08) / h).round() as usize,
                ((size.y + 0.08) / h).round() as usize,
                h as f32,
            );
            let volume: Vec<f64> = mass
                .iter()
                .map(|&m| if m > 0.0 { 0.25 * h * h } else { 0.0 })
                .collect();
            let open = |_: usize, _: usize, _: usize| None;
            let grid = Grid::new(&layout, &x, (&mass, &volume), &open, Some(0.75));
            let solids = vec![solid; x.len()];
            let mut walls = vec![Wall {
                axis: 1,
                inward: 1.0,
                at: corner.y,
            }];
            if sides {
                walls.push(Wall {
                    axis: 0,
                    inward: 1.0,
                    at: corner.x,
                });
                walls.push(Wall {
                    axis: 0,
                    inward: -1.0,
                    at: corner.x + size.x,
                });
            }
            let points = wall_points(
                &grid,
                (&x, &outline, &corners),
                &walls,
                (solid.mu, solid.bulk - solid.mu, h),
            );
            let k6 =
                grid.approximate_jacobian((&mass, &volume, &solids), 0.0, 2.0 / (dt * dt), &[]);
            let gamma = gamma0(&k6, &points, h) * h;
            let a = grid.approximate_jacobian((&mass, &volume, &solids), 0.0, 4.0 / (dt * dt), &[]);
            let (negative, expected, nearest) = inertia(&a, &points, gamma);
            println!(
                "N-1 {name}, cell {h} m, dt {dt:.3e} s: {} wall points, gamma {gamma:.4e} m/Pa; negative eigenvalues of -C^-1 - U^T A^-1 U {negative} against {expected} for a positive K_N, the one nearest zero {nearest:.4e}: K_N {}",
                points.len(),
                if negative == expected {
                    "positive definite"
                } else {
                    "NOT positive definite"
                }
            );
        }
    }
    for h in [0.04, 0.02, 0.01] {
        tank_step(h, dt, (1.0, 1.0));
    }
}

#[test]
#[ignore = "a measurement (FAST_MODE_PLAN.md sec. 36.18b, N-5): run with --ignored --nocapture"]
fn nitsche_penalty_and_theta_characterization() {
    let dt = 1.0 / 120.0;
    for (theta, factors) in [(1.0, vec![1.0, 10.0, 100.0]), (-1.0, vec![1.0, 10.0])] {
        for factor in factors {
            for h in [0.04, 0.02, 0.01] {
                tank_step(h, dt, (theta, factor));
            }
        }
    }
}
