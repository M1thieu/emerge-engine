//! Where a held body's depth in a
//! wall comes from, counted before each controlled step under the trace
//! switch. For every condition of the lone outline against every wall (a
//! patch function `f` and a wall plane, the step's key `(3, f, wall)`): its
//! weighted gap to the WALL itself (not to the planes the step sets at the
//! points' start positions), its weighted normal rate, and its least member
//! point's gap. For each held condition its line; for each condition that
//! joined in the step just taken, its start gap, start rate and rate times
//! that step's length (route C's overshoot bound `v_c dt`), against its end
//! gaps. Test-only, printed, nothing changed.

use super::*;

/// One condition's measures: key, weighted gap, weighted rate, least member
/// gap.
type Measure = (ContactKey, f64, f64, f64);

/// A call's measures, held keys and the step length it was taken with.
type Last = Option<(Vec<Measure>, Vec<ContactKey>, f64)>;

thread_local! {
    /// The last call's.
    static LAST: std::cell::RefCell<Last> = const { std::cell::RefCell::new(None) };
}

/// Every condition's measures at the state's positions and velocities.
fn measures(state: &TotalLagrangianState, planes: &[Plane]) -> Option<Vec<Measure>> {
    let cache = state.cache.as_ref()?;
    let trace = cache.traces.first()?;
    let outline = state.outlines.first()?;
    let (x, v) = (&state.x, &state.v);
    let mut all = Vec::new();
    for (f, members) in trace.functions.iter().enumerate() {
        let total: f64 = members.iter().map(|&(_, w)| w).sum();
        if total <= 0.0 {
            continue;
        }
        for (i, plane) in planes.iter().enumerate() {
            let (mut gap, mut rate, mut least) = (0.0, 0.0, f64::INFINITY);
            for &(r, w) in members {
                let q = outline[r];
                let g = plane.inward * (x[q][plane.axis] - plane.at);
                gap += w / total * g;
                rate += w / total * plane.inward * v[q][plane.axis];
                least = least.min(g);
            }
            all.push(((3, f, i), gap, rate, least));
        }
    }
    Some(all)
}

/// A held wall condition's
/// in-patch bound. Its members' wall gaps against their reference
/// arclength along the wall `s` (the reference coordinate across the
/// wall's axis), fitted by least squares with the members' weights to `g0 +
/// theta (s - s_mid) + kappa (s - s_mid)^2 / 2` (a line with fewer than three
/// members); on the patch's length `H` (the weights' sum) the quadratic
/// departs from its mean by at most `|theta| H / 2 + |kappa| H^2 / 8`, and the
/// fit's largest residual is added as measured. The line says whether the
/// least member's gap below the weighted gap lies within that bound.
fn in_patch_bound(
    state: &TotalLagrangianState,
    planes: &[Plane],
    key: ContactKey,
) -> Option<String> {
    let (kind, function, wall) = key;
    if kind != 3 {
        return None;
    }
    let members = state
        .cache
        .as_ref()?
        .traces
        .first()?
        .functions
        .get(function)?;
    let plane = planes.get(wall)?;
    let outline = state.outlines.first()?;
    let along = 1 - plane.axis;
    let total: f64 = members.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 || members.len() < 2 {
        return None;
    }
    let points: Vec<(f64, f64, f64)> = members
        .iter()
        .map(|&(r, w)| {
            let q = outline[r];
            let gap = plane.inward * (state.x[q][plane.axis] - plane.at);
            (state.reference[q][along], gap, w / total)
        })
        .collect();
    let mid = points.iter().map(|p| p.2 * p.0).sum::<f64>();
    let weighted = points.iter().map(|p| p.2 * p.1).sum::<f64>();
    let least = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    // The weighted normal equations for (g0, theta, kappa / 2).
    let degree = if points.len() >= 3 { 3 } else { 2 };
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for &(s, g, w) in &points {
        let basis = [1.0, s - mid, (s - mid).powi(2)];
        for ((row, right), &bi) in a.iter_mut().zip(b.iter_mut()).zip(&basis).take(degree) {
            *right += w * bi * g;
            for (entry, &bj) in row.iter_mut().zip(&basis).take(degree) {
                *entry += w * bi * bj;
            }
        }
    }
    let mut coefficients = [0.0f64; 3];
    // Gaussian elimination on the small system.
    for i in 0..degree {
        let pivot = a[i][i];
        if pivot.abs() <= f64::MIN_POSITIVE {
            return None;
        }
        let (top, bottom) = a.split_at_mut(i + 1);
        let (b_top, b_bottom) = b.split_at_mut(i + 1);
        for (row, right) in bottom
            .iter_mut()
            .zip(b_bottom.iter_mut())
            .take(degree - i - 1)
        {
            let factor = row[i] / pivot;
            for (entry, &above) in row[i..degree].iter_mut().zip(&top[i][i..degree]) {
                *entry -= factor * above;
            }
            *right -= factor * b_top[i];
        }
    }
    for i in (0..degree).rev() {
        let tail: f64 = a[i][i + 1..degree]
            .iter()
            .zip(&coefficients[i + 1..degree])
            .map(|(x, y)| x * y)
            .sum();
        coefficients[i] = (b[i] - tail) / a[i][i];
    }
    let (theta, kappa) = (coefficients[1], 2.0 * coefficients[2]);
    let residual = points
        .iter()
        .map(|&(s, g, _)| {
            let d = s - mid;
            (g - coefficients[0] - coefficients[1] * d - coefficients[2] * d * d).abs()
        })
        .fold(0.0, f64::max);
    let bound = theta.abs() * total / 2.0 + kappa.abs() * total * total / 8.0 + residual;
    let below = weighted - least;
    Some(format!(
        "wall gap {key:?}: weighted wall gap {weighted:+.4e} m, least member {least:+.4e} m, below the weighted {below:.4e} m; theta {theta:+.4e}, kappa {kappa:+.4e} 1/m, fit residual {residual:.3e} m, H {total:.4e} m, in-patch bound {bound:.4e} m: {}",
        if below <= bound { "within" } else { "BEYOND" }
    ))
}

/// The gap lines before a controlled step of length `state.step_length`.
pub(super) fn before_step(state: &TotalLagrangianState, planes: &[Plane]) {
    let Some(all) = measures(state, planes) else {
        return;
    };
    let cache = state.cache.as_ref().expect("measured above");
    let held: Vec<ContactKey> = cache.contacts.iter().map(|c| c.key).collect();
    let find = |key: ContactKey| all.iter().find(|m| m.0 == key);
    for c in &cache.contacts {
        if let Some(line) = in_patch_bound(state, planes, c.key) {
            trace(|| line);
        }
        if let Some(&(key, gap, rate, least)) = find(c.key) {
            trace(|| {
                format!(
                    "held {key:?}: weighted wall gap {gap:+.4e} m, least member {least:+.4e} m, in-patch spread {:+.4e} m, rate {rate:+.4e} m/s, multiplier {:+.4e}",
                    least - gap,
                    c.multiplier
                )
            });
        }
    }
    // Every condition NOT held with a member
    // below its wall, with its in-patch bound (`in_patch_bound`).
    for &(key, gap, rate, least) in all.iter().filter(|m| m.3 < 0.0 && !held.contains(&m.0)) {
        trace(|| {
            format!(
                "unheld {key:?}: weighted wall gap {gap:+.4e} m, least member {least:+.4e} m, rate {rate:+.4e} m/s"
            )
        });
        if let Some(line) = in_patch_bound(state, planes, key) {
            trace(|| format!("unheld {line}"));
        }
    }
    LAST.with(|last| {
        if let Some((before, held_before, dt)) = last.borrow().as_ref() {
            for &key in held.iter().filter(|k| !held_before.contains(k)) {
                let (Some(start), Some(end)) = (before.iter().find(|m| m.0 == key), find(key))
                else {
                    continue;
                };
                trace(|| {
                    format!(
                        "joined {key:?} in the last step ({dt:.6e} s): start weighted gap {:+.4e} m, start rate {:+.4e} m/s, rate x dt {:+.4e} m; end weighted gap {:+.4e} m, end least member {:+.4e} m",
                        start.1,
                        start.2,
                        start.2 * dt,
                        end.1,
                        end.3
                    )
                });
            }
        }
        *last.borrow_mut() = Some((all.clone(), held, state.step_length));
    });
    let [relative, round_off, resolution, largest] =
        crate::spacetime::total_lagrangian::hooks::take_allowance_terms();
    if relative + round_off + resolution > 0.0 {
        trace(|| {
            format!(
                "last step's held allowances bound by: the relative term {relative}, the round-off {round_off}, the resolution {resolution}; the largest relative term {largest:.4e} m"
            )
        });
    }
}
