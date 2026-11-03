//! FAST_MODE_PLAN.md sec. 56, a test-only measurement (knob
//! `EMERGE_PROBE_SCALED_REUSE`, default off): a kept preconditioner factor
//! reused, scaled, across tries whose steps are all far shorter than a P
//! wave's cell crossing.
//!
//! Derived on the page from `solve`'s band `P(c) = c M_b + K_0` (`M_b`
//! definite, `K_0` semidefinite): the generalized eigenvalues of `(c'/c)
//! P(c')^-1 P(c)` are `(m + k/c) / (m + k/c')` over the modes of `(K_0,
//! M_b)`, so its condition number is at most `1 + w / min(c, c')`, `w =
//! lambda_max(M_b^-1 K_0)`, whatever the ratio `c / c'`. `w` is not
//! computed: the reuse needs both coefficients at least `c* = 64 (K + mu) /
//! (rho h^2)` (a stated threshold, declared: `dt <= h / (sqrt(32) c_P)`;
//! the measured GMRES counts check it). A preconditioner changes the
//! iterations, never the solution beyond the Newton tolerance.

use super::elastic::Elastic;
use super::{Band, Spring};

/// Whether the knob is set; says so once.
fn knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_SCALED_REUSE").is_some();
        if on {
            println!("knob active: scaled reuse of a kept factor above c* = 64 (K + mu) / (rho h^2) (sec. 56)");
        }
        on
    })
}

/// `c*` for these points, the largest over the massive ones.
fn threshold((mass, v0, solids): (&[f64], &[f64], &[Elastic]), cell: f64) -> f64 {
    (0..mass.len())
        .filter(|&p| mass[p] > 0.0 && v0[p] > 0.0)
        .map(|p| 64.0 * (solids[p].bulk + solids[p].mu) * v0[p] / (mass[p] * cell * cell))
        .fold(0.0f64, f64::max)
}

/// The kept factor to reuse scaled, if the knob is set: the same springs,
/// both coefficients at least `c*`, the nearest in ratio.
pub(super) fn far(
    kept: &[(f64, Vec<Spring>, Band, u64)],
    springs: &[Spring],
    coefficient: f64,
    points: (&[f64], &[f64], &[Elastic]),
    cell: f64,
) -> Option<usize> {
    if !knob() {
        return None;
    }
    let star = threshold(points, cell);
    if star <= 0.0 || coefficient < star {
        return None;
    }
    kept.iter()
        .enumerate()
        .filter(|(_, (c, built, _, _))| built.as_slice() == springs && *c >= star)
        .map(|(i, (c, _, _, _))| (i, (c / coefficient).max(coefficient / c)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}
