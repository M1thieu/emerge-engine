//! A test-only probe: each Maxwell branch
//! advanced over the WHOLE step (no splitting, no fraction of the step) by
//! an update that stays damped when the step is much longer than the
//! branch's relaxation time `tau = eta_b / mu_b`, where Liu and Guan's
//! midpoint (`strain_change`) gives the factor `(1 - h / 2) / (1 + h / 2)
//! -> -1`, `h = dt / tau`. With `Q = S_P(C) - mu_b E`, `Ups = |Q|^2 / (4
//! mu_b)` and `eta_b dE/dt = Q`, the branch relaxes toward `E* = S_P(C) /
//! mu_b` and dissipates `eta_b |dE/dt|^2 / 2`. Selected by
//! `EMERGE_PROBE_BRANCH_UPDATE`:
//! - `frozen`: the exact relaxation toward `E*_n` held over the step,
//!   `E_n+1 = E*_n + exp(-h)(E_n - E*_n)`; the stress the discrete gradient
//!   of `Ups` in `C` at fixed `E_n+1` (Liu and Guan's eq. 4.18 structure),
//!   the heat `Ups(C_n, E_n) - Ups(C_n, E_n+1)`, never negative;
//! - `backward`: backward Euler toward `E*_n+1` (Shutov, Landgraf and
//!   Ihlemann, arXiv 1304.3380, eq. 20, the linear branch's form), the
//!   stress and heat as `frozen`'s, the heat's sign not guaranteed.
//!
//! Adaptation, declared: the frozen target is first order. (The update with
//! the target linear in time over the step is the default one,
//! `relaxation::exponential_step`.)

use super::*;

/// The probe's branch update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Update {
    Frozen,
    Backward,
}

/// The update `EMERGE_PROBE_BRANCH_UPDATE` selects, if any; says so once.
pub(super) fn active() -> Option<Update> {
    static UPDATE: std::sync::OnceLock<Option<Update>> = std::sync::OnceLock::new();
    *UPDATE.get_or_init(|| {
        let update = match std::env::var("EMERGE_PROBE_BRANCH_UPDATE").as_deref() {
            Ok("frozen") => Some(Update::Frozen),
            Ok("backward") => Some(Update::Backward),
            _ => None,
        };
        if let Some(update) = update {
            println!("knob active: branch update over the whole step, {update:?}");
        }
        update
    })
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
            (e1, -super::ups_change(s0, (e0, e1), mu))
        }
        Update::Backward => {
            let e1 = (e0 + target1 * h) * (1.0 + h).recip();
            (e1, -super::ups_change(s0, (e0, e1), mu))
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
    let (e1, _) = advance(b, (s0, s1), dt, update);
    let z = (c1 - c0) * 0.5;
    let norm2 = colon(z, z);
    let mid = 0.5 * (c0 + c1);
    let q_mid = corotated_second_pk(mid, l, m) - e1 * mu;
    let s_neq = corotated_second_pk_tangent(mid, q_mid, l, m) * mu.recip();
    if 4.0 * norm2 <= 1e-28 {
        return s_neq;
    }
    // What `Z : S` must carry: `Ups(C_n+1, E_n+1) - Ups(C_n, E_n+1)` at the
    // held `E_n+1`, from the increment (the heat booked is `Ups`'s fall at
    // `C_n`).
    let ds = corotated_second_pk_increment(c0, c1, l, m);
    let work = colon(ds, s0 + s1 - e1 * (2.0 * mu)) / (4.0 * mu);
    s_neq + z * ((work - colon(s_neq, z)) / norm2)
}

/// A kept step's branch heat below zero, counted and printed
/// (the first 20, then every 1000th).
pub(super) fn count_negative(heat: f64) {
    static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if heat < 0.0 {
        let k = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if k <= 20 || k.is_multiple_of(1000) {
            println!("negative branch heat #{k}: {heat:.6e} J/m");
        }
    }
}
