//! T6 (`tmp/papers/dl_realtime/PUSO_CANDIDATE_PLAN.md` sec. 5 and 9): Puso,
//! Porter and Slavik's persistency update and energy return (LLNL-JRNL-855301,
//! eqs. 29-30, 40-42, App. B) after route C's step, on the weighted-gap
//! contacts. The second variant of the plan's section 5: route C first, then
//! the persistency update; declared as having no source for the pairing.
//! Adaptations, declared: each contact's row is its weighted normal velocity
//! `G_c v = sum_shares weight n . sum_r w_r v(x_r)` over the shares the push
//! uses (`contact_geometry`), Puso's mortar rows' role; a row dependent on the
//! ones before (a pivot below `1e-12` of the system's largest diagonal) is
//! skipped and counted (the pseudo-inverse of the plan's section 5);
//! frictionless only. Test-only, switched by `EMERGE_PROBE_T6`.

use super::*;
use std::collections::HashMap;

/// T6's state, carried in the step's cache (`Factors::persistency`) so that
/// every try of the step controller starts from the step's own and only the
/// kept try's survives (a store outside the cache let rejected tries store
/// and return energy the kept trajectory never saw).
#[derive(Clone, Default)]
pub(crate) struct Store {
    /// Each contact's energy taken by the persistency update since it was
    /// first held (`kappa`, eq. 40), by its key, with the shares of its last
    /// held step: route C rebuilds the contacts each step, so a released one
    /// returns its energy along the row it was last held by (the grid's
    /// stencils stay on the reference configuration, so that row stays valid).
    kappa: HashMap<ContactKey, (f64, Vec<Share>)>,
    /// Since the last `take_accounts`: stored, returned, dropped by the
    /// guard, rows skipped as dependent, the steps it acted on, and the
    /// release excess (substep 3's kinetic change less the energy it returns:
    /// zero by Lemma 2, PUSO_CANDIDATE_PLAN.md sec. 11); the largest relative
    /// residual of an `M^-1` application, and T6's net kinetic change.
    accounts: [f64; 8],
}

impl Store {
    /// The energy stored now (the booked ledger's term: `T + U + sum kappa`,
    /// Puso eq. 43).
    pub(crate) fn held(&self) -> f64 {
        self.kappa.values().map(|e| e.0).sum()
    }

    /// The accounts since the last call, and the energy still stored.
    pub(crate) fn take_accounts(&mut self) -> ([f64; 8], f64) {
        let held = self.kappa.values().map(|e| e.0).sum();
        (std::mem::take(&mut self.accounts), held)
    }
}

/// Whether `EMERGE_PROBE_T6` is set; says so once, from inside.
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_T6").is_some();
        if on {
            println!("knob active: T6 persistency update (route C, then Puso's substeps 2-3)");
        }
        on
    })
}

/// The step's hook (`step_total_lagrangian_contact`, after the end
/// velocities): on route C with the switch on, `M^-1` by the mass band on the
/// massive unknowns, the grid's kinetic energy `1/2 w M_eps w`, then
/// `apply`; the kinetic change added to the step's `work.1`.
pub(super) fn hook(
    mode: ContactMode,
    grid: &Grid,
    (x, used, u, contacts): (&[DVec2], &[Plane], &[f64], &[Contact]),
    (factors, mass, settings): (&mut Factors, &[f64], &ConservingSettings),
    (end_velocity, updates): (&mut [f64], &mut f64),
) {
    if mode != ContactMode::Velocity || !on() {
        return;
    }
    let band = factors.mass.as_ref().expect("built by the solve");
    let massive = grid.massive();
    // The exact `M_eps^-1` on the massive unknowns: conjugate gradients on
    // the mass with its band as the preconditioner, as the step solves eq. 28
    // and the torque probe does (the band alone is the preconditioner's
    // approximation; Lemmas 1 and 2 need the exact inverse).
    let refine: usize = std::env::var("EMERGE_PROBE_T6_REFINE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let worst = std::cell::Cell::new(0.0f64);
    let inverse = |y: &[f64]| -> Vec<f64> {
        let rhs: Vec<f64> = y
            .iter()
            .zip(&massive)
            .map(|(&e, &on)| if on { e } else { 0.0 })
            .collect();
        let apply = |w: &[f64]| grid.mass_times(w, mass, settings.epsilon);
        let solve = |b: &[f64]| preconditioned_cg(&apply, b, &|r: &[f64]| band.solve(r), &massive);
        let mut z = solve(&rhs);
        // `EMERGE_PROBE_T6_REFINE` (PUSO_CANDIDATE_PLAN.md sec. 13): passes of
        // iterative refinement, `z += M^-1 (y - M z)`.
        for _ in 0..refine {
            let mz = apply(&z);
            let r: Vec<f64> = rhs
                .iter()
                .zip(&mz)
                .zip(&massive)
                .map(|((&b, &m), &on)| if on { b - m } else { 0.0 })
                .collect();
            let dz = solve(&r);
            z.iter_mut().zip(&dz).for_each(|(a, b)| *a += b);
        }
        let mz = apply(&z);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for ((&b, &m), &on) in rhs.iter().zip(&mz).zip(&massive) {
            if on {
                num += (b - m) * (b - m);
                den += b * b;
            }
        }
        worst.set(worst.get().max((num / den.max(f64::MIN_POSITIVE)).sqrt()));
        z
    };
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    if refine > 0 {
        ANNOUNCED.call_once(|| println!("knob active: T6 M^-1 refinement passes {refine}"));
    }
    let kinetic = |w: &[f64]| {
        let mw = grid.mass_times(w, mass, settings.epsilon);
        0.5 * w.iter().zip(&mw).map(|(a, b)| a * b).sum::<f64>()
    };
    let store = &mut factors.persistency;
    let change = apply(
        grid,
        (x, used, u),
        (contacts, store),
        (&inverse, &kinetic),
        end_velocity,
    );
    *updates += change;
    let a = &mut factors.persistency.accounts;
    a[6] = a[6].max(worst.get());
    a[7] += change;
}

/// The grid's `M^-1` on the massive unknowns and its kinetic energy.
type GridOperators<'a> = (&'a dyn Fn(&[f64]) -> Vec<f64>, &'a dyn Fn(&[f64]) -> f64);

/// Applies substeps 2 and 3 to `end_velocity` (the grid's massive unknowns'
/// end velocities); `inverse` is `M^-1` on the massive unknowns, `kinetic`
/// the grid's kinetic energy. Returns the kinetic energy's change.
pub(super) fn apply(
    grid: &Grid,
    (x, used, u): (&[DVec2], &[Plane], &[f64]),
    (contacts, store): (&[Contact], &mut Store),
    (inverse, kinetic): GridOperators<'_>,
    end_velocity: &mut [f64],
) -> f64 {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let row_of = |shares: &[Share]| -> Vec<f64> {
        let mut g = vec![0.0; grid.len];
        for s in shares {
            for &(r, w) in &s.spread {
                for a in 0..2 {
                    let coefficient = s.weight * w * s.normal[a];
                    grid.visit(r, a, |k, wk, _| g[k] += coefficient * wk);
                }
            }
        }
        g
    };
    let before = kinetic(end_velocity);
    let held: Vec<&Contact> = contacts.iter().filter(|c| c.multiplier > 0.0).collect();
    let released: Vec<ContactKey> = store
        .kappa
        .keys()
        .filter(|key| !held.iter().any(|c| c.key == **key))
        .copied()
        .collect();
    // Substep 2 on the persistency set `P`: the held contacts (their shares
    // at this step) and the just-released ones (their stored shares), as
    // section 5 states it: Lemma 2's exact return needs `G_C v~ = 0` for every
    // released `C` (sec. 11: without them, the return added `alpha v~ . f`).
    let mut keys: Vec<ContactKey> = held.iter().map(|c| c.key).collect();
    let mut shares: Vec<Vec<Share>> = held
        .iter()
        .map(|c| contact_geometry(grid, x, used, u, c).1)
        .collect();
    for key in &released {
        keys.push(*key);
        shares.push(store.kappa[key].1.clone());
    }
    let rows: Vec<Vec<f64>> = shares.iter().map(|s| row_of(s)).collect();
    let (mut stored, mut skipped) = (0.0, 0.0);
    if !rows.is_empty() {
        let columns: Vec<Vec<f64>> = rows.iter().map(|g| inverse(g)).collect();
        let size = rows.len();
        let mut system: Vec<Vec<f64>> = rows
            .iter()
            .map(|g| columns.iter().map(|z| dot(g, z)).collect())
            .collect();
        let largest = (0..size).map(|i| system[i][i].abs()).fold(0.0, f64::max);
        let rates: Vec<f64> = rows.iter().map(|g| dot(g, end_velocity)).collect();
        let mut impulse = rates.clone();
        let mut live = vec![true; size];
        for i in 0..size {
            let pivot = system[i][i];
            if pivot.abs() <= 1e-12 * largest {
                live[i] = false;
                skipped += 1.0;
                continue;
            }
            let pivot_row = system[i].clone();
            for r in i + 1..size {
                let factor = system[r][i] / pivot;
                for (e, above) in system[r][i..].iter_mut().zip(&pivot_row[i..]) {
                    *e -= factor * above;
                }
                impulse[r] -= factor * impulse[i];
            }
        }
        for i in (0..size).rev() {
            if !live[i] {
                impulse[i] = 0.0;
                continue;
            }
            let tail: f64 = (i + 1..size).map(|c| system[i][c] * impulse[c]).sum();
            impulse[i] = (impulse[i] - tail) / system[i][i];
        }
        for (z, &l) in columns.iter().zip(&impulse) {
            end_velocity
                .iter_mut()
                .zip(z)
                .for_each(|(e, zk)| *e -= l * zk);
        }
        // Each contact's share of the dissipation, `1/2 l (G v)` (eq. 32).
        for (((key, &l), &rate), s) in keys.iter().zip(&impulse).zip(&rates).zip(&shares) {
            let entry = store.kappa.entry(*key).or_insert((0.0, Vec::new()));
            entry.0 += 0.5 * l * rate;
            entry.1 = s.clone();
            stored += 0.5 * l * rate;
        }
    }
    // Substep 3: the released contacts' energy returned (eqs. 41, 51-58), the
    // guard `max(kappa, 0)` (Puso sec. 4) and its dropped part counted.
    let (mut returned, mut dropped) = (0.0, 0.0);
    if !released.is_empty() {
        let (kappas, rows): (Vec<f64>, Vec<Vec<f64>>) = released
            .iter()
            .map(|key| (store.kappa[key].0, row_of(&store.kappa[key].1)))
            .unzip();
        let energy: f64 = kappas.iter().map(|e| e.max(0.0)).sum();
        dropped = kappas.iter().map(|e| e.min(0.0)).sum::<f64>();
        if energy > 0.0 {
            let mut force = vec![0.0; grid.len];
            for (g, &kappa) in rows.iter().zip(&kappas) {
                force
                    .iter_mut()
                    .zip(g)
                    .for_each(|(e, gk)| *e += kappa.max(0.0) * gk);
            }
            let z = inverse(&force);
            let norm = dot(&force, &z);
            if norm > 0.0 {
                let alpha = (2.0 * energy / norm).sqrt();
                let before_return = kinetic(end_velocity);
                end_velocity
                    .iter_mut()
                    .zip(&z)
                    .for_each(|(e, zk)| *e += alpha * zk);
                returned = energy;
                store.accounts[5] += kinetic(end_velocity) - before_return - energy;
            } else {
                // No row to return it along: counted as dropped, never
                // created.
                dropped -= energy;
            }
        }
        for key in &released {
            store.kappa.remove(key);
        }
    }
    let a = &mut store.accounts;
    a[0] += stored;
    a[1] += returned;
    a[2] += dropped;
    a[3] += skipped;
    a[4] += 1.0;
    kinetic(end_velocity) - before
}
