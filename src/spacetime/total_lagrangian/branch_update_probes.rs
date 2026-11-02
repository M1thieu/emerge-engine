//! FAST_MODE_PLAN.md secs. 42-42b, a test-only probe: each Maxwell branch
//! advanced over the WHOLE step (no splitting, no fraction of the step) by
//! an update that stays damped when the step is much longer than the
//! branch's relaxation time `tau = eta_b / mu_b`, where Liu and Guan's
//! midpoint (`strain_change`) gives the factor `(1 - h / 2) / (1 + h / 2)
//! -> -1`, `h = dt / tau`. With `Q = S_P(C) - mu_b E`, `Ups = |Q|^2 / (4
//! mu_b)` and `eta_b dE/dt = Q`, the branch relaxes toward `E* = S_P(C) /
//! mu_b` and dissipates `eta_b |dE/dt|^2 / 2`. Selected by
//! `EMERGE_PROBE_BRANCH_UPDATE`:
//! - `frozen` (C-2a): the exact relaxation toward `E*_n` held over the step,
//!   `E_n+1 = E*_n + exp(-h)(E_n - E*_n)`; the stress the discrete gradient
//!   of `Ups` in `C` at fixed `E_n+1` (Liu and Guan's eq. 4.18 structure),
//!   the heat `Ups(C_n, E_n) - Ups(C_n, E_n+1)`, never negative;
//! - `linear` (C-2b): the exact solution with `E*` linear in time over the
//!   step; the heat that path's exact dissipation; the stress the midpoint's
//!   projected so that `Z : S` is `Ups`'s change plus the heat (derived, not
//!   sourced);
//! - `backward` (C-2d): backward Euler toward `E*_n+1` (Shutov, Landgraf and
//!   Ihlemann, arXiv 1304.3380, eq. 20, the linear branch's form), the
//!   stress and heat as `frozen`'s, the heat's sign not guaranteed.
//!
//! Adaptations, declared: the frozen target is first order; the linear
//! target assumes `S_P` linear in time over the step.

use super::*;

/// The probe's branch update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Update {
    Frozen,
    Linear,
    Backward,
}

/// The update `EMERGE_PROBE_BRANCH_UPDATE` selects, if any; says so once.
pub(super) fn active() -> Option<Update> {
    static UPDATE: std::sync::OnceLock<Option<Update>> = std::sync::OnceLock::new();
    *UPDATE.get_or_init(|| {
        let update = match std::env::var("EMERGE_PROBE_BRANCH_UPDATE").as_deref() {
            Ok("frozen") => Some(Update::Frozen),
            Ok("linear") => Some(Update::Linear),
            Ok("backward") => Some(Update::Backward),
            _ => None,
        };
        if let Some(update) = update {
            println!("knob active: branch update over the whole step, {update:?}");
        }
        update
    })
}

/// `Ups(C, E2) - Ups(C, E1)` at the stress `s = S_P(C)`, from the change.
fn ups_change(s: DMat2, (e1, e2): (DMat2, DMat2), mu: f64) -> f64 {
    colon(e1 - e2, s * 2.0 - (e1 + e2) * mu) / 4.0
}

/// The branch's internal strain at the step's end and the step's heat.
pub(super) fn advance(
    b: &BranchState,
    (s0, s1): (DMat2, DMat2),
    dt: f64,
    update: Update,
) -> (DMat2, f64) {
    let Branch {
        modulus: mu,
        viscosity: eta,
    } = b.branch;
    let h = dt * mu / eta;
    let (target0, target1) = (s0 * mu.recip(), s1 * mu.recip());
    let e0 = b.strain;
    let decay = (-h).exp();
    match update {
        Update::Frozen => {
            let e1 = target0 + (e0 - target0) * decay;
            (e1, -ups_change(s0, (e0, e1), mu))
        }
        Update::Backward => {
            let e1 = (e0 + target1 * h) * (1.0 + h).recip();
            (e1, -ups_change(s0, (e0, e1), mu))
        }
        Update::Linear => {
            let rise = target1 - target0;
            // `1 - (1 - e^-h) / h`, the lag behind a moving target.
            let lag = 1.0 + (-h).exp_m1() / h;
            let e1 = e0 * decay + target0 * (1.0 - decay) + rise * lag;
            // The deviation from the target in the time `s` (in `tau`):
            // `e(s) = (E_n - E*_n) e^-s + rise expm1(-s) / h`; the heat
            // `mu / 2 int_0^h |e(s)|^2 ds`.
            let e00 = e0 - target0;
            let heat = if h > 0.1 {
                // Closed form with `u = e00 + rise / h`, `v = rise / h`.
                let v = rise * h.recip();
                let u = e00 + v;
                0.5 * mu
                    * (0.5 * colon(u, u) * (1.0 - decay * decay)
                        - 2.0 * colon(u, v) * (1.0 - decay)
                        + colon(v, v) * h)
            } else {
                // Gauss-Legendre, 6 points on `[0, h]` (no cancellation).
                const NODES: [f64; 6] = [
                    -0.932_469_514_203_152,
                    -0.661_209_386_466_265,
                    -0.238_619_186_083_197,
                    0.238_619_186_083_197,
                    0.661_209_386_466_265,
                    0.932_469_514_203_152,
                ];
                const WEIGHTS: [f64; 6] = [
                    0.171_324_492_379_170,
                    0.360_761_573_048_139,
                    0.467_913_934_572_691,
                    0.467_913_934_572_691,
                    0.360_761_573_048_139,
                    0.171_324_492_379_170,
                ];
                let integral: f64 = NODES
                    .iter()
                    .zip(WEIGHTS)
                    .map(|(&x, w)| {
                        let s = 0.5 * h * (x + 1.0);
                        let e = e00 * (-s).exp() + rise * ((-s).exp_m1() / h);
                        w * colon(e, e)
                    })
                    .sum::<f64>()
                    * 0.5
                    * h;
                0.5 * mu * integral
            };
            (e1, heat)
        }
    }
}

/// The branch's step stress under `update` (`advance`'s end strain).
pub(super) fn stress(
    b: &BranchState,
    (s0, s1): (DMat2, DMat2),
    (c0, c1, dt): (DMat2, DMat2, f64),
    (l, m): (f64, f64),
    update: Update,
) -> DMat2 {
    let mu = b.branch.modulus;
    let (e1, heat) = advance(b, (s0, s1), dt, update);
    let z = (c1 - c0) * 0.5;
    let norm2 = colon(z, z);
    let mid = 0.5 * (c0 + c1);
    let held = if update == Update::Linear {
        0.5 * (b.strain + e1)
    } else {
        e1
    };
    let q_mid = corotated_second_pk(mid, l, m) - held * mu;
    let s_neq = corotated_second_pk_tangent(mid, q_mid, l, m) * mu.recip();
    if 4.0 * norm2 <= 1e-28 {
        return s_neq;
    }
    // `Ups(C_n+1, E) - Ups(C_n, E)` at the held `E`, from the increment.
    let ds = corotated_second_pk_increment(c0, c1, l, m);
    let in_c = |e: DMat2| colon(ds, s0 + s1 - e * (2.0 * mu)) / (4.0 * mu);
    // What `Z : S` must carry: for `frozen` and `backward`, `Ups(C_n+1,
    // E_n+1) - Ups(C_n, E_n+1)` (the heat booked is `Ups`'s fall at `C_n`);
    // for `linear`, `Ups(C_n+1, E_n+1) - Ups(C_n, E_n)` plus the heat.
    let work = if update == Update::Linear {
        in_c(e1) + ups_change(s0, (b.strain, e1), mu) + heat
    } else {
        in_c(e1)
    };
    s_neq + z * ((work - colon(s_neq, z)) / norm2)
}

/// X-2 (sec. 42): a kept step's branch heat below zero, counted and printed
/// (the first 20, then every 1000th).
pub(super) fn count_negative(heat: f64) {
    static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if heat < 0.0 {
        let k = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if k <= 20 || k.is_multiple_of(1000) {
            println!("X-2 negative branch heat #{k}: {heat:.6e} J/m");
        }
    }
}
