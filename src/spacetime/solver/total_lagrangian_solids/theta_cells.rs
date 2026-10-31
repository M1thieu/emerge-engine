//! MIXED_FORMULATION_PLAN.md sec. 6 (test builds, `EMERGE_PROBE_THETA=cell`):
//! the TL solids path's mean-dilatation cells. Each state point's reference
//! cell of the background grid (`grid.cell` square cells, indexed by the
//! cells that hold points), handed once to `hooks::set_cells`, which turns
//! on `Elastic::mixed_stress` in the step (Gonzalez 2000 eq. 32, sec. 4.2,
//! `m_dis = 1`). Contact points carry no volume and add nothing to their
//! cell's mean `J`.

use std::collections::HashMap;

use glam::DVec2;

use super::TotalLagrangianState;
use crate::spacetime::total_lagrangian::hooks;

/// Whether `EMERGE_PROBE_THETA=cell` is set; says so once, from inside.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("EMERGE_PROBE_THETA").as_deref() == Ok("cell");
        if on {
            println!(
                "knob active: Theta per cell on the TL solids path (MIXED_FORMULATION_PLAN.md)"
            );
        }
        on
    })
}

/// Each state point's cell and the number of cells, kept for the report.
static CELLS: std::sync::OnceLock<(Vec<usize>, usize)> = std::sync::OnceLock::new();

/// Sets the cells once, at the first step with a grid.
pub(crate) fn ensure(state: &TotalLagrangianState) {
    static SET: std::sync::Once = std::sync::Once::new();
    if !on() {
        return;
    }
    let Some(cell) = state.cache.as_ref().map(|c| c.grid.cell) else {
        return;
    };
    SET.call_once(|| {
        let mut index: HashMap<(i64, i64), usize> = HashMap::new();
        let of: Vec<usize> = state
            .reference
            .iter()
            .map(|q| {
                let key = ((q.x / cell).floor() as i64, (q.y / cell).floor() as i64);
                let next = index.len();
                *index.entry(key).or_insert(next)
            })
            .collect();
        println!(
            "Theta cells: {} cells for {} points (cell {cell} m)",
            index.len(),
            of.len()
        );
        let _ = CELLS.set((of.clone(), index.len()));
        hooks::set_cells(Some((of, index.len())));
    });
}

/// Per frame (MIXED_FORMULATION_PLAN.md sec. 9): the particles' least and
/// largest `J` and the least `W_dev = W - K/2 (J - 1)^2` over them (the
/// subtraction split's validity near `J = 1`), a negative one flagged.
pub(crate) fn report(state: &TotalLagrangianState, frame: u64) {
    if !on() {
        return;
    }
    let (mut low, mut high, mut least) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY);
    for p in 0..state.particles {
        if state.v0[p] <= 0.0 {
            continue;
        }
        let j = state.f[p].determinant();
        low = low.min(j);
        high = high.max(j);
        least = least.min(state.solids[p].deviatoric_energy(state.f[p]));
    }
    // The energy the mixed step conserves: `sum_p V0 W_dev + sum_cells 1/2
    // (sum_p V0 K) (Theta - 1)^2`, `Theta` the cell's `V0`-weighted mean `J`
    // (the cell's `K` as `solve.rs` forms its pressure), with the frame's
    // kinetic and gravity parts.
    let mixed = CELLS.get().map_or(f64::NAN, |(of, count)| {
        let mut sums = vec![(0.0f64, 0.0f64, 0.0f64); *count];
        let mut deviatoric = 0.0;
        for p in 0..state.particles {
            let (v0, f) = (state.v0[p], state.f[p]);
            if v0 <= 0.0 {
                continue;
            }
            deviatoric += v0 * state.solids[p].deviatoric_energy(f);
            let cell = &mut sums[of[p]];
            cell.0 += v0;
            cell.1 += v0 * f.determinant();
            cell.2 += v0 * state.solids[p].bulk;
        }
        let volumetric: f64 = sums
            .iter()
            .filter(|c| c.0 > 0.0)
            .map(|&(volume, j, bulk)| 0.5 * bulk * (j / volume - 1.0).powi(2))
            .sum();
        deviatoric + volumetric
    });
    let [kinetic, _, gravity] = state.frame.parts;
    println!(
        "Theta frame {frame}: J in [{low:.9}, {high:.9}], least W_dev {least:+.4e} J/m^3{}; mixed energy {:.12e} J/m (kinetic {kinetic:.6e}, mixed stored {mixed:.6e}, gravity {gravity:.6e}); rigid-relative vibration {:.6e} J/m",
        if least < 0.0 {
            " NEGATIVE: outside the split's validity"
        } else {
            ""
        },
        kinetic + mixed + gravity,
        relative_kinetic(state) + mixed
    );
}

/// The kinetic energy relative to the rigid motion: the total less the
/// centre's translation and the spin `L / I` about the centre, J/m.
fn relative_kinetic(state: &TotalLagrangianState) -> f64 {
    let n = state.particles;
    let (mut mass, mut momentum, mut centre) = (0.0, DVec2::ZERO, DVec2::ZERO);
    let mut kinetic = 0.0;
    for p in 0..n {
        mass += state.mass[p];
        momentum += state.mass[p] * state.v[p];
        centre += state.mass[p] * state.x[p];
        kinetic += 0.5 * state.mass[p] * state.v[p].length_squared();
    }
    let total = mass.max(f64::MIN_POSITIVE);
    let (velocity, centre) = (momentum / total, centre / total);
    let (mut angular, mut inertia) = (0.0, 0.0);
    for p in 0..n {
        let arm = state.x[p] - centre;
        angular += state.mass[p] * arm.perp_dot(state.v[p] - velocity);
        inertia += state.mass[p] * arm.length_squared();
    }
    let spin = angular / inertia.max(f64::MIN_POSITIVE);
    kinetic - 0.5 * mass * velocity.length_squared() - 0.5 * inertia * spin * spin
}

/// The same rigid-relative vibration for a run WITHOUT the Theta cells, the
/// law's own stored energy in place of the mixed one
/// (`EMERGE_PROBE_RELATIVE_VIBRATION`), so a displacement run and a mixed
/// run are compared on one definition.
pub(crate) fn report_displacement(state: &TotalLagrangianState, frame: u64) {
    if on() || std::env::var_os("EMERGE_PROBE_RELATIVE_VIBRATION").is_none() {
        return;
    }
    let stored: f64 = (0..state.particles)
        .map(|p| state.v0[p] * state.solids[p].energy(state.f[p]))
        .sum();
    println!(
        "Displacement frame {frame}: rigid-relative vibration {:.6e} J/m (law stored {stored:.6e})",
        relative_kinetic(state) + stored
    );
}

/// MIXED_FORMULATION_PLAN.md criterion 3: the inf-sup constant of the pair
/// (the quadratic B-spline displacement, one constant pressure per cell)
/// on the block's own grid, at the block refinement `EMERGE_PROBE_REFINE`
/// (8r x 4r cells). `beta^2 = lambda_min` of `Mp^-1/2 B A^-1 B^T Mp^-1/2`:
/// `B` the cells' divergence rows, `(q_e, div v) = sum_(p in e) V0_p div
/// v(X_p)`; `A` the velocity norm, `int grad u : grad v + 1/L^2 int u . v`
/// with `L` the body's longest side (the scale that makes it dimensionally
/// one H1 norm; the rigid motions, divergence-free, are not in `B`'s
/// range); `Mp` the cells' volumes. `lambda_min` by inverse iteration on
/// the Schur complement's Cholesky factor (a zero pivot reports a spurious
/// pressure mode), `lambda_max` by the power method. A probe.
#[cfg(test)]
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_theta_inf_sup() {
    use super::super::TotalLagrangianConfig;
    use super::super::impact_local::{FRAME_S, drop};
    let mut sim = drop(FRAME_S, TotalLagrangianConfig::new(1e-2));
    sim.step();
    let state = sim.total_lagrangian.as_ref().expect("the TL path");
    let grid = &state.cache.as_ref().expect("a step was taken").grid;
    let cell = grid.cell;
    // The cells holding particles with volume, and each particle's.
    let mut index: HashMap<(i64, i64), usize> = HashMap::new();
    let mut of = vec![usize::MAX; state.particles];
    for (p, owner) in of.iter_mut().enumerate() {
        if state.v0[p] <= 0.0 {
            continue;
        }
        let q = state.reference[p];
        let key = ((q.x / cell).floor() as i64, (q.y / cell).floor() as i64);
        let next = index.len();
        *owner = *index.entry(key).or_insert(next);
    }
    let cells = index.len();
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    let mut slot = vec![usize::MAX; grid.len];
    for (i, &k) in unknowns.iter().enumerate() {
        slot[k] = i;
    }
    let (lo, hi) = state.reference[..state.particles].iter().fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(lo, hi), &q| (lo.min(q), hi.max(q)),
    );
    let length = (hi - lo).max_element().max(cell);
    let (mut a, mut b, mut mp) = (vec![0.0; n * n], vec![0.0; cells * n], vec![0.0; cells]);
    for p in 0..state.particles {
        let v0 = state.v0[p];
        if v0 <= 0.0 {
            continue;
        }
        mp[of[p]] += v0;
        for axis in 0..2 {
            let mut touched: Vec<(usize, f64, DVec2)> = Vec::new();
            grid.visit(p, axis, |k, w, g| {
                if slot[k] != usize::MAX {
                    touched.push((slot[k], w, g));
                }
            });
            for &(i, wi, gi) in &touched {
                b[of[p] * n + i] += v0 * gi[axis];
                for &(j, wj, gj) in &touched {
                    a[i * n + j] += v0 * (gi.dot(gj) + wi * wj / (length * length));
                }
            }
        }
    }
    let cholesky = |m: &[f64], size: usize| -> Result<Vec<f64>, usize> {
        let mut l = vec![0.0; size * size];
        for i in 0..size {
            for j in 0..=i {
                let mut s = m[i * size + j];
                for k in 0..j {
                    s -= l[i * size + k] * l[j * size + k];
                }
                if i == j {
                    if s <= 0.0 {
                        return Err(i);
                    }
                    l[i * size + i] = s.sqrt();
                } else {
                    l[i * size + j] = s / l[j * size + j];
                }
            }
        }
        Ok(l)
    };
    let solve = |l: &[f64], rhs: &[f64], size: usize| -> Vec<f64> {
        let mut y = rhs.to_vec();
        for i in 0..size {
            for k in 0..i {
                y[i] -= l[i * size + k] * y[k];
            }
            y[i] /= l[i * size + i];
        }
        for i in (0..size).rev() {
            for k in (i + 1)..size {
                y[i] -= l[k * size + i] * y[k];
            }
            y[i] /= l[i * size + i];
        }
        y
    };
    let la = cholesky(&a, n).expect("the velocity norm is definite");
    // `T = D B A^-1 B^T D`, `D = Mp^-1/2`.
    let ainv_bt: Vec<Vec<f64>> = (0..cells)
        .map(|e| solve(&la, &b[e * n..(e + 1) * n], n))
        .collect();
    let mut t = vec![0.0; cells * cells];
    for e in 0..cells {
        for f in 0..cells {
            let s: f64 = (0..n).map(|i| b[e * n + i] * ainv_bt[f][i]).sum();
            t[e * cells + f] = s / (mp[e] * mp[f]).sqrt();
        }
    }
    let norm = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let times = |x: &[f64]| -> Vec<f64> {
        (0..cells)
            .map(|e| (0..cells).map(|f| t[e * cells + f] * x[f]).sum())
            .collect()
    };
    // The power method for `lambda_max`, from an alternating start (it
    // carries a checkerboard component and the smooth one alike).
    let mut x: Vec<f64> = (0..cells).map(|e| 1.0 + (e % 3) as f64).collect();
    let mut largest = 0.0;
    for _ in 0..500 {
        let y = times(&x);
        let size = norm(&y);
        largest = size / norm(&x);
        x = y.iter().map(|v| v / size).collect();
    }
    let (smallest, iterations) = match cholesky(&t, cells) {
        Err(pivot) => (0.0, pivot),
        Ok(lt) => {
            let mut x: Vec<f64> = (0..cells).map(|e| 1.0 + (e % 3) as f64).collect();
            let mut smallest = 0.0;
            let mut count = 0;
            for k in 0..500 {
                let y = solve(&lt, &x, cells);
                let size = norm(&y);
                let next = norm(&x) / size;
                x = y.iter().map(|v| v / size).collect();
                count = k + 1;
                if (next - smallest).abs() <= 1e-12 * next {
                    smallest = next;
                    break;
                }
                smallest = next;
            }
            (smallest, count)
        }
    };
    println!(
        "inf-sup: refinement {} ({} cells, {n} unknowns, L {length:.4} m): beta^2 = lambda_min {smallest:.6e}, lambda_max {largest:.6e}, beta {:.6e}, ratio {:.3e}; {}",
        super::super::impact_local::refinement().0,
        cells,
        smallest.max(0.0).sqrt(),
        smallest / largest,
        if smallest > 0.0 {
            format!("Schur complement definite, inverse iteration {iterations} steps")
        } else {
            format!("Schur complement SINGULAR at pivot {iterations}: a spurious pressure mode")
        }
    );
}
