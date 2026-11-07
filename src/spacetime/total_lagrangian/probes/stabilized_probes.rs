//! The counts before the code: Hild and Renard's stabilized multipliers on one-cell patches
//! (Numer. Math. 115 (2010), eq. 12), measured on the step's own operator
//! (`Grid::approximate_jacobian`: the midpoint's `2 M / dt^2` and the
//! solid's linear elasticity at `F = I`) before any engine change. The
//! contact pressure's rows `P` take the same linear law, `sigma = 2 mu
//! eps` plus `(K - mu) tr(eps) I`, so the counts measure the
//! discretization, not a second law. Test-only, printed, nothing changed. Run with
//! `EMERGE_PROBE_PATCH_CELLS=1` for patches of one cell (`outline_trace`),
//! unset for today's two.

use super::multipliers::dense_solve;
use super::weighted_gap::tests::eigenvalues;
use super::*;
use crate::forces::fields::EARTH_GRAVITY_M_S2;
use crate::matter::materials::lame_parameters::lame_from_young;
use crate::matter::materials::liquid::liquid_water::LiquidWater;
use crate::matter::materials::relaxation::ViscoelasticRecord;

/// A rigid wall the body rests on: its normal's axis, the inward sign (the
/// gap `inward (x[axis] - at)` is `>= 0` on the body's side), its position.
pub(super) struct Wall {
    pub(super) axis: usize,
    pub(super) inward: f64,
    pub(super) at: f64,
}

/// A rectangle of particles every half cell from `corner`, `size` metres,
/// its outline every half cell counter-clockwise from `corner` (massless),
/// the outline's corners, at cell `h`, density `rho`.
pub(super) fn rectangle(
    corner: DVec2,
    size: DVec2,
    h: f64,
    rho: f64,
) -> (Vec<DVec2>, Vec<f64>, Vec<usize>, Vec<usize>) {
    let mut x = Vec::new();
    let (ni, nj) = (
        (size.x / (0.5 * h)).round() as usize,
        (size.y / (0.5 * h)).round() as usize,
    );
    for i in 0..ni {
        for j in 0..nj {
            x.push(corner + DVec2::new((i as f64 + 0.5) * 0.5 * h, (j as f64 + 0.5) * 0.5 * h));
        }
    }
    let mut mass = vec![rho * 0.25 * h * h; x.len()];
    let start = x.len();
    let c = [
        corner,
        corner + DVec2::new(size.x, 0.0),
        corner + size,
        corner + DVec2::new(0.0, size.y),
    ];
    let mut corners = Vec::new();
    for e in 0..4 {
        let (a, b) = (c[e], c[(e + 1) % 4]);
        let count = ((b - a).length() / (0.5 * h)).round() as usize;
        corners.push(x.len() - start);
        for s in 0..count {
            x.push(a + (b - a) * (s as f64 / count as f64));
        }
    }
    mass.resize(x.len(), 0.0);
    let outline = (start..x.len()).collect();
    (x, mass, outline, corners)
}

/// One condition's rows on the grid's unknowns: the weighted gap's gradient
/// `G_c`, the weighted contact pressure's `P_c` (`p = -sigma_NN`, the
/// normal's axis `a`), its weights' sum `L_c`, its wall and its members.
pub(super) struct Condition {
    pub(super) gap: Vec<f64>,
    pressure: Vec<f64>,
    length: f64,
    wall: usize,
    members: Vec<(usize, f64)>,
}

pub(super) fn conditions(
    grid: &Grid,
    x: &[DVec2],
    outline: &[usize],
    trace: &OutlineTrace,
    walls: &[Wall],
    (mu, lame): (f64, f64),
) -> Vec<Condition> {
    let mut all = Vec::new();
    for members in &trace.functions {
        let Some(wall) = walls.iter().position(|w| {
            members
                .iter()
                .all(|&(r, _)| (x[outline[r]][w.axis] - w.at).abs() < 1e-9)
        }) else {
            continue;
        };
        let (axis, inward) = (walls[wall].axis, walls[wall].inward);
        let total: f64 = members.iter().map(|&(_, w)| w).sum();
        let (mut gap, mut pressure) = (vec![0.0; grid.len], vec![0.0; grid.len]);
        for &(r, w) in members {
            let share = w / total;
            grid.visit(outline[r], axis, |k, n, _| gap[k] += share * inward * n);
            for b in 0..2 {
                grid.visit(outline[r], b, |k, _, g| {
                    let normal = if b == axis { 2.0 * mu * g[axis] } else { 0.0 };
                    pressure[k] -= share * (normal + lame * g[b]);
                });
            }
        }
        all.push(Condition {
            gap,
            pressure,
            length: total,
            wall,
            members: members.clone(),
        });
    }
    all
}

/// `rows_i . A^-1 rows_j` for two families of rows, from the solves `z`.
pub(super) fn products(rows: &[&[f64]], solved: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|r| {
            solved
                .iter()
                .map(|z| r.iter().zip(z).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect()
}

/// The symmetric matrix's eigenvalues, sorted.
pub(super) fn sorted(a: Vec<Vec<f64>>) -> Vec<f64> {
    let mut values = eigenvalues(a);
    values.sort_by(f64::total_cmp);
    values
}

/// A dense matrix's inverse, column by column.
pub(super) fn inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let columns: Vec<Vec<f64>> = (0..m)
        .map(|j| {
            let e: Vec<f64> = (0..m).map(|i| if i == j { 1.0 } else { 0.0 }).collect();
            dense_solve(a.to_vec(), e).expect("regular")
        })
        .collect();
    (0..m)
        .map(|i| (0..m).map(|j| columns[j][i]).collect())
        .collect()
}

/// One scene's spectra at one step (`spectra`).
struct Spectra {
    /// The ratio at `gamma0` = 1 / Pa, Pa; `gamma0` at the ratio 1/2, 1 / Pa;
    /// the P-wave modulus, Pa.
    ratio: f64,
    gamma0: f64,
    modulus: f64,
    /// The Schur complement's eigenvalues without and with the
    /// stabilization, sorted, 1 / Pa.
    plain: Vec<f64>,
    with: Vec<f64>,
}

/// The relative rank (1e-12) of sorted eigenvalues.
fn rank(v: &[f64]) -> usize {
    v.iter().filter(|&&s| s > 1e-12 * v[v.len() - 1]).count()
}

/// The stability constant and the Schur complement on one scene at one step, patches of `2 trace_cell`: the
/// stabilization's ratio `max_v sum_c gamma_c L_c p_c(v)^2 / a(v, v)` at
/// `gamma_c = h` (`gamma0` = 1 / Pa), `gamma0` set by the ratio 1/2, and
/// the multipliers' Schur complement without and with it.
fn spectra(
    (x, mass, outline, corners): (&[DVec2], &[f64], &[usize], &[usize]),
    (layout, h, walls): (&GridLayout, f64, &[Wall]),
    solid: Elastic,
    (dt, trace_cell): (f64, f64),
) -> Spectra {
    let volume: Vec<f64> = mass
        .iter()
        .map(|&m| if m > 0.0 { 0.25 * h * h } else { 0.0 })
        .collect();
    let open = |_: usize, _: usize, _: usize| None;
    let grid = Grid::new(layout, x, (mass, &volume), &open, Some(0.75));
    let solids = vec![solid; x.len()];
    let a = grid.approximate_jacobian((mass, &volume, &solids), 0.0, 2.0 / (dt * dt), &[]);
    let trace = outline_trace(x, outline, corners, trace_cell, None);
    let (mu, lame) = (solid.mu, solid.bulk - solid.mu);
    let all = conditions(&grid, x, outline, &trace, walls, (mu, lame));
    let m = all.len();
    let gaps: Vec<&[f64]> = all.iter().map(|c| c.gap.as_slice()).collect();
    let pressures: Vec<&[f64]> = all.iter().map(|c| c.pressure.as_slice()).collect();
    let z_g: Vec<Vec<f64>> = gaps.iter().map(|r| a.solve(r)).collect();
    let z_p: Vec<Vec<f64>> = pressures.iter().map(|r| a.solve(r)).collect();
    let (gg, gp, pp) = (
        products(&gaps, &z_g),
        products(&gaps, &z_p),
        products(&pressures, &z_p),
    );
    // The stability constant: with `D = diag(h L_c)`, the largest eigenvalue of `D^1/2 P
    // A^-1 P^T D^1/2`, Pa.
    let d: Vec<f64> = all.iter().map(|c| h * c.length).collect();
    let scaled: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..m).map(|j| (d[i] * d[j]).sqrt() * pp[i][j]).collect())
        .collect();
    let ratio = *sorted(scaled).last().expect("conditions");
    let gamma0 = 0.5 / ratio;
    let modulus = lame + 2.0 * mu;
    // The Schur complement without: `G A^-1 G^T`.
    let plain = sorted(gg.clone());
    // The Schur complement with, `gamma_c = gamma0 h`: the saddle system of the stabilized
    // Lagrangian linearised, `A_gamma = A - P^T D_gamma P` (`D_gamma =
    // diag(gamma_c L_c)`), `B = G - Gamma P`; its Schur complement `B
    // A_gamma^-1 B^T + diag(gamma_c / L_c)` by Woodbury through `A^-1`.
    let gamma = gamma0 * h;
    let mut bab = vec![vec![0.0; m]; m];
    let mut y = vec![vec![0.0; m]; m];
    let mut e = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            bab[i][j] = gg[i][j] - gamma * gp[i][j] - gamma * gp[j][i] + gamma * gamma * pp[i][j];
            y[i][j] = gp[i][j] - gamma * pp[i][j];
            e[i][j] = -pp[i][j];
        }
        e[i][i] += 1.0 / (gamma * all[i].length);
    }
    let e_inv = inverse(&e);
    let mut stabilized = bab;
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..m {
                for l in 0..m {
                    s += y[i][k] * e_inv[k][l] * y[j][l];
                }
            }
            stabilized[i][j] += s;
        }
        stabilized[i][i] += gamma / all[i].length;
    }
    Spectra {
        ratio,
        gamma0,
        modulus,
        plain,
        with: sorted(stabilized),
    }
}

/// The stability constant and the Schur complement printed (`spectra`), today's patch length (`outline_trace`
/// at the cell, `EMERGE_PROBE_PATCH_CELLS` to change it).
fn count(
    label: &str,
    body: (&[DVec2], &[f64], &[usize], &[usize]),
    (layout, h, walls): (&GridLayout, f64, &[Wall]),
    solid: Elastic,
    dt: f64,
) {
    let Spectra {
        ratio,
        gamma0,
        modulus,
        plain,
        with,
    } = spectra(body, (layout, h, walls), solid, (dt, h));
    let m = plain.len();
    println!(
        "stability constant, {label} dt {dt:.3e} s: {m} conditions, ratio at gamma0 = 1/Pa {ratio:.4e} Pa, over the P-wave modulus {:.4e}; gamma0 {gamma0:.4e} 1/Pa",
        ratio / modulus
    );
    println!(
        "Schur complement, {label} dt {dt:.3e} s: without, Schur eigenvalues {:.4e} to {:.4e} 1/Pa, rank {} of {m}, condition {:.4e}; with, {:.4e} to {:.4e}, rank {} of {m}, condition {:.4e}",
        plain[0],
        plain[m - 1],
        rank(&plain),
        plain[m - 1] / plain[0].abs().max(f64::MIN_POSITIVE),
        with[0],
        with[m - 1],
        rank(&with),
        with[m - 1] / with[0].abs().max(f64::MIN_POSITIVE)
    );
}

/// The water tank's exact hydrostatic start (`F = diag(1,
/// a(Y))`, `a = 1 - rho g (H0 - Y) / K`, `y(Y) = Y - rho g (H0 Y - Y^2 / 2)
/// / K`) projected on the basis by the consistent mass (`grid_velocity`),
/// the contact pressure `p = -N0 . P(F) N0` read at the outline points from
/// the projected field through the law itself, each patch's mean against
/// the exact law's at the same members, relative to `rho g H0`; beside it
/// the mean over the particles of the boundary cell row facing the patch.
fn trace_on_exact_start(h: f64) {
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
    let stretch = |big_y: f64| 1.0 - rho * g * (depth - big_y) / bulk;
    let displacement: Vec<DVec2> = x
        .iter()
        .map(|p| {
            let big_y = p.y - corner.y;
            DVec2::new(0.0, -rho * g * (depth * big_y - 0.5 * big_y * big_y) / bulk)
        })
        .collect();
    let factor = grid.mass_band(&mass, 0.0);
    let (u, _) = grid_velocity(&grid, (&mass, &displacement), 0.0, &factor);
    let pressure = |f: DMat2, axis: usize| {
        let c = f.transpose() * f;
        let p = f * solid.step_stress(c, c, 1.0);
        -p.col(axis)[axis]
    };
    let at = |q: usize| {
        let (_, gu) = grid.read(&u, q, 0);
        let (_, gv) = grid.read(&u, q, 1);
        DMat2::IDENTITY + DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y))
    };
    let exact = |p: DVec2| DMat2::from_diagonal(DVec2::new(1.0, stretch(p.y - corner.y)));
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
    let trace = outline_trace(&x, &outline, &corners, h, None);
    let all = conditions(&grid, &x, &outline, &trace, &walls, (0.0, bulk));
    let scale = rho * g * depth;
    let mut worst = [(0.0f64, 0.0f64); 3];
    for c in &all {
        let axis = walls[c.wall].axis;
        let (mut basis, mut truth) = (0.0, 0.0);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(r, w) in &c.members {
            let q = outline[r];
            basis += w / c.length * pressure(at(q), axis);
            truth += w / c.length * pressure(exact(x[q]), axis);
            lo = lo.min(x[q][1 - axis]);
            hi = hi.max(x[q][1 - axis]);
        }
        // The particles of the boundary cell row facing this patch.
        let near: Vec<usize> = (0..x.len())
            .filter(|&p| mass[p] > 0.0)
            .filter(|&p| (x[p][axis] - walls[c.wall].at).abs() < h)
            .filter(|&p| x[p][1 - axis] >= lo && x[p][1 - axis] <= hi)
            .collect();
        let particles = if near.is_empty() {
            f64::NAN
        } else {
            near.iter().map(|&p| pressure(at(p), axis)).sum::<f64>() / near.len() as f64
        };
        if c.members == all[0].members {
            println!(
                "hydrostatic start, cell {h} m, the first floor patch: from the outline points {basis:.9e} Pa, exact {truth:.9e} Pa, the boundary row's particles {particles:.9e} Pa"
            );
        }
        // A NaN is kept, not dropped by `max`.
        let worse = |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        };
        let slot = &mut worst[c.wall];
        slot.0 = worse(slot.0, (basis - truth).abs() / scale);
        slot.1 = worse(slot.1, (particles - truth).abs() / scale);
    }
    for (name, (basis, particles)) in ["floor", "left wall", "right wall"].iter().zip(worst) {
        println!(
            "hydrostatic start, cell {h} m, {name}: largest |p_c - exact| / (rho g H0) from the outline points {basis:.4e}, from the boundary row's particles {particles:.4e}"
        );
    }
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn stabilized_multipliers_counts() {
    let water = LiquidWater::at(293.15).expect("in range");
    let plasticine = ViscoelasticRecord::plasticine_crandall_1970_fresh().relaxed;
    let (lambda, mu) = lame_from_young(plasticine.e_pa, plasticine.nu);
    let solids = [
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
    for (name, solid, rho, size, sides) in solids {
        for h in [0.04, 0.02, 0.01] {
            let corner = DVec2::new(0.04, 0.04);
            let (x, mass, outline, corners) = rectangle(corner, size, h, rho);
            let layout = GridLayout::new(
                ((size.x + 0.08) / h).round() as usize,
                ((size.y + 0.08) / h).round() as usize,
                h as f32,
            );
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
            // The decision's step, the RT contact piece (half a 1/60 s
            // frame), two finer steps and two long ones (the elastostatic
            // limit of Hild and Renard's condition), printed beside.
            for dt in [1.0 / 120.0, 1e-3, 1e-4, 1.0, 100.0] {
                count(
                    &format!("{name}, cell {h} m"),
                    (&x, &mass, &outline, &corners),
                    (&layout, h, &walls),
                    solid,
                    dt,
                );
            }
        }
    }
    for h in [0.04, 0.02, 0.01] {
        trace_on_exact_start(h);
    }
}

/// Criteria written before this
/// probe: on the walls' inf-sup geometry (a block of 6 by 4 cells of 2 cm
/// on a floor, cells 2, 1 and 0.5 cm, offsets 0 and 0.3 cell), `q0 =
/// lambda_min(S, H = h) / lambda_min(S, H = 2h)` and `q` the same with the
/// stabilization at H = h, at the RT contact piece and at 1e-4 s; the bar
/// is the walls' test's own, the finest above half the coarsest.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn stabilized_one_cell_patches_keep_the_walls_inf_sup() {
    let water = LiquidWater::at(293.15).expect("in range");
    let plasticine = ViscoelasticRecord::plasticine_crandall_1970_fresh().relaxed;
    let (lambda, mu) = lame_from_young(plasticine.e_pa, plasticine.nu);
    let materials = [
        (
            "water",
            Elastic::barotropic(water.isentropic_bulk()),
            water.density(),
        ),
        (
            "Plasticine",
            Elastic::corotated(lambda as f64, mu as f64),
            plasticine.rho_kg_m3 as f64,
        ),
    ];
    let size = DVec2::new(0.12, 0.08);
    for (name, solid, rho) in materials {
        for offset in [0.0, 0.3] {
            for dt in [1.0 / 120.0, 1e-4] {
                let mut rows = Vec::new();
                for h in [0.02, 0.01, 0.005] {
                    let corner = DVec2::new(0.04 + offset * h, 0.04);
                    let (x, mass, outline, corners) = rectangle(corner, size, h, rho);
                    let layout = GridLayout::new(
                        ((size.x + 0.08) / h).round() as usize + 1,
                        ((size.y + 0.08) / h).round() as usize,
                        h as f32,
                    );
                    let walls = [Wall {
                        axis: 1,
                        inward: 1.0,
                        at: corner.y,
                    }];
                    let body = (
                        x.as_slice(),
                        mass.as_slice(),
                        outline.as_slice(),
                        corners.as_slice(),
                    );
                    let one = spectra(body, (&layout, h, &walls), solid, (dt, 0.5 * h));
                    let two = spectra(body, (&layout, h, &walls), solid, (dt, h));
                    let (q0, q) = (one.plain[0] / two.plain[0], one.with[0] / two.plain[0]);
                    println!(
                        "{name}, offset {offset} cell, dt {dt:.3e} s, cell {h} m: {} conditions at H = h, gamma0 {:.4e} 1/Pa, lambda_min at H = 2h {:.4e}, at H = h without {:.4e}, with {:.4e} 1/Pa (rank {} of {}); q0 {q0:.4e}, q {q:.4e}",
                        one.plain.len(),
                        one.gamma0,
                        two.plain[0],
                        one.plain[0],
                        one.with[0],
                        rank(&one.with),
                        one.with.len()
                    );
                    rows.push((q0, q, rank(&one.with) == one.with.len()));
                }
                let bar = |first: f64, last: f64| last > 0.5 * first;
                println!(
                    "{name}, offset {offset} cell, dt {dt:.3e} s: control q0 {} the bar ({:.4e} to {:.4e}); stabilized q {} it ({:.4e} to {:.4e}); full rank at every cell: {}",
                    if bar(rows[0].0, rows[2].0) {
                        "PASSES"
                    } else {
                        "FAILS"
                    },
                    rows[0].0,
                    rows[2].0,
                    if bar(rows[0].1, rows[2].1) {
                        "PASSES"
                    } else {
                        "FAILS"
                    },
                    rows[0].1,
                    rows[2].1,
                    rows.iter().all(|r| r.2)
                );
            }
        }
    }
}
