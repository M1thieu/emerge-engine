//! The walls held at the step's end, with the energy that takes refunded after
//! the solve (test builds, `EMERGE_PROBE_END_GAP` = `gap`, `refund` or
//! `refund_friction`, default off). A measurement: nothing here is the engine's
//! default.
//!
//! Source: Laursen and Love, IJNME 53 (2002) 245-274. The end gap is their eq.
//! 69, Signorini at the step's end (also Doyen, Ern and Piperno 2011, eq. 109),
//! in place of the persistency on the velocity. The solve then loses energy to
//! the walls where a gap closes. The refund repays it after convergence, with no
//! store: `v = v^s + M^-1 G^T p`, `v^s = 2 u / dt - v_n` (their eqs. 46, 75), each
//! condition's impulse solving `1/2 sum_k S_lk p_k p_l + b_l p_l + w_l = 0` (eq.
//! 63) with `S = G M^-1 G^T`, `b = G v^s`, `w_l` the condition's contact work over
//! the step. Summed over the conditions, that is the energy balance of the update
//! exactly. The root is the one that vanishes with `w_l` (eqs. 67-68).
//!
//! Adaptation, declared: their quadrature points are this engine's weighted-gap
//! conditions.
//!
//! With friction (`refund_friction`): each impulse acts along the condition's
//! whole contact force, normal push and friction together, not the normal alone
//! (Love and Laursen, CMAME 192 (2003) 2223-2248, eq. 56). The energy owed stays
//! the normal work: the engine already books friction's work as heat, from the
//! step's displacement, which the refund leaves unchanged. Declared: their
//! examples use a lumped mass, this engine a consistent one; flat walls only.
//!
//! Not addressed, by construction: at a held condition `w_l = 0`, so `p = 0` and
//! `G v_n+1 = -G v_n` stays.

use std::sync::atomic::{AtomicU64, Ordering};

/// What the knob asks for.
#[derive(Clone, Copy, PartialEq)]
enum Asked {
    Off,
    Gap,
    Refund,
    RefundFriction,
}

fn asked() -> Asked {
    static ASKED: std::sync::OnceLock<Asked> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| match std::env::var("EMERGE_PROBE_END_GAP").as_deref() {
        Ok("gap") => {
            println!("knob active: the walls held at the step's end, no refund");
            Asked::Gap
        }
        Ok("refund") => {
            println!("knob active: the walls held at the step's end, the energy refunded");
            Asked::Refund
        }
        Ok("refund_friction") => {
            println!(
                "knob active: the walls held at the step's end, the energy refunded along the whole contact force"
            );
            Asked::RefundFriction
        }
        Ok(other) => panic!("EMERGE_PROBE_END_GAP {other}: gap, refund or refund_friction"),
        Err(_) => Asked::Off,
    })
}

/// Whether the walls are held at the step's end.
pub(crate) fn gap_on() -> bool {
    asked() != Asked::Off
}

/// Whether the energy is refunded after the solve.
pub(crate) fn refund_on() -> bool {
    matches!(asked(), Asked::Refund | Asked::RefundFriction)
}

/// Whether the refund takes friction: its impulses along the whole contact
/// force.
pub(crate) fn with_friction() -> bool {
    asked() == Asked::RefundFriction
}

/// The impulses of one step, and how their solve went.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Impulses {
    pub(crate) p: Vec<f64>,
    /// Conditions whose quadratic has no real root: left unrefunded.
    pub(crate) no_root: u64,
    pub(crate) converged: bool,
}

/// Solves the coupled quadratics by sweeps, each condition on its own root with
/// the others held. The stop is the impulses' round-off, no tolerance chosen. The
/// sweep count only guards against a loop, and hitting it is reported.
pub(crate) fn impulses(s: &[Vec<f64>], b: &[f64], w: &[f64]) -> Impulses {
    const SWEEPS_GUARD: usize = 1000;
    let n = b.len();
    let mut p = vec![0.0; n];
    let mut no_root = 0;
    let mut converged = n == 0;
    for _ in 0..SWEEPS_GUARD {
        no_root = 0;
        let mut change = 0.0f64;
        let mut size = 0.0f64;
        for l in 0..n {
            let a = 0.5 * s[l][l];
            let coupled: f64 = (0..n).filter(|&k| k != l).map(|k| s[l][k] * p[k]).sum();
            let big_b = 0.5 * coupled + b[l];
            let discriminant = big_b * big_b - 4.0 * a * w[l];
            // Eqs. 67-68, in the form that keeps its digits as `w_l` vanishes.
            let next = if discriminant < 0.0 {
                no_root += 1;
                0.0
            } else {
                let sign = if big_b < 0.0 { -1.0 } else { 1.0 };
                let denominator = discriminant.sqrt() + big_b.abs();
                if denominator > 0.0 {
                    -2.0 * w[l] * sign / denominator
                } else {
                    0.0
                }
            };
            change = change.max((next - p[l]).abs());
            size = size.max(next.abs());
            p[l] = next;
        }
        if change <= 8.0 * f64::EPSILON * size {
            converged = true;
            break;
        }
    }
    Impulses {
        p,
        no_root,
        converged,
    }
}

static STEPS: AtomicU64 = AtomicU64::new(0);
static CONDITIONS: AtomicU64 = AtomicU64::new(0);
static NO_ROOT: AtomicU64 = AtomicU64::new(0);
static UNCONVERGED: AtomicU64 = AtomicU64::new(0);
static PULLING: AtomicU64 = AtomicU64::new(0);
static INWARD: AtomicU64 = AtomicU64::new(0);
static REVERSED: AtomicU64 = AtomicU64::new(0);

/// Counts one refund: the impulses, and the end normal velocities after it.
pub(crate) fn count(solved: &Impulses, end_rates: &[f64]) {
    STEPS.fetch_add(1, Ordering::Relaxed);
    CONDITIONS.fetch_add(solved.p.len() as u64, Ordering::Relaxed);
    NO_ROOT.fetch_add(solved.no_root, Ordering::Relaxed);
    UNCONVERGED.fetch_add(u64::from(!solved.converged), Ordering::Relaxed);
    let pulling = solved.p.iter().filter(|&&p| p < 0.0).count();
    PULLING.fetch_add(pulling as u64, Ordering::Relaxed);
    let inward = end_rates.iter().filter(|&&rate| rate < 0.0).count();
    INWARD.fetch_add(inward as u64, Ordering::Relaxed);
}

/// Counts the conditions whose tangential end velocity the refund turned
/// around: its sign before the update against after it (Love and Laursen do
/// not exclude it).
pub(crate) fn count_reversals(before: &[f64], after: &[f64]) {
    let reversed = before
        .iter()
        .zip(after)
        .filter(|(b, a)| **b * **a < 0.0)
        .count();
    REVERSED.fetch_add(reversed as u64, Ordering::Relaxed);
}

/// One line: the refunds so far on all threads.
pub(crate) fn report() -> String {
    let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    format!(
        "{} refunds over {} conditions; no real root {}, unconverged {}; pulling impulses {}, inward end velocities {}, tangential end velocities turned around {}",
        read(&STEPS),
        read(&CONDITIONS),
        read(&NO_ROOT),
        read(&UNCONVERGED),
        read(&PULLING),
        read(&INWARD),
        read(&REVERSED)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The balance each impulse must close: the sum over the conditions is the
    /// update's energy, `1/2 p S p + b p`, against the contact work.
    fn residual(s: &[Vec<f64>], b: &[f64], w: &[f64], p: &[f64], l: usize) -> f64 {
        let coupled: f64 = (0..b.len()).map(|k| s[l][k] * p[k]).sum();
        0.5 * coupled * p[l] + b[l] * p[l] + w[l]
    }

    /// One condition: the exact root, positive with a positive rate.
    #[test]
    fn one_condition_repays_its_work() {
        let solved = impulses(&[vec![2.0]], &[3.0], &[-1.0]);
        assert!(solved.converged && solved.no_root == 0);
        // `p^2 + 3 p - 1 = 0`, the root that vanishes with the work.
        let exact = (-3.0 + 13f64.sqrt()) / 2.0;
        assert!((solved.p[0] - exact).abs() < 1e-14, "{}", solved.p[0]);
    }

    /// Coupled conditions: every quadratic closes to round-off, and the sum is
    /// the energy balance.
    #[test]
    fn coupled_conditions_close_each_balance() {
        let s = vec![
            vec![2.0, 0.5, 0.1],
            vec![0.5, 3.0, 0.4],
            vec![0.1, 0.4, 1.5],
        ];
        let (b, w) = ([1.0, 2.0, 0.5], [-0.3, -0.2, -0.1]);
        let solved = impulses(&s, &b, &w);
        assert!(solved.converged && solved.no_root == 0);
        for l in 0..3 {
            let r = residual(&s, &b, &w, &solved.p, l);
            assert!(r.abs() < 1e-14, "condition {l}: {r}");
        }
    }

    /// No work, no impulse: a held condition keeps its velocity flip.
    #[test]
    fn a_held_condition_gets_no_impulse() {
        let solved = impulses(&[vec![2.0]], &[-3.0], &[0.0]);
        assert!(solved.converged);
        assert_eq!(solved.p[0], 0.0);
    }

    /// Work the root cannot repay is left unrefunded, and counted.
    #[test]
    fn a_quadratic_without_a_root_is_counted() {
        let solved = impulses(&[vec![2.0]], &[0.1], &[1.0]);
        assert_eq!((solved.no_root, solved.p[0]), (1, 0.0));
    }
}
