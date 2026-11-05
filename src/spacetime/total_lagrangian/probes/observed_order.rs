//! The order observed over two refusals in a row, for the next try after the
//! second (test builds, `EMERGE_PROBE_OBSERVED_ORDER`, default off). A
//! measurement: nothing here is the engine's default, and every kept step still
//! passes the same error test.
//!
//! Source: Gustafsson, ACM TOMS 20 (1994) 496-517, eq. 17 and sec. 5.2: with
//! the error model `r = phi h^k` and `phi` the same at one point, two refused
//! tries give `k^ = log(r_n / r_n-1) / log(h_n / h_n-1)`, and the next step is
//! `h (1/r)^(1/k^)`. His limits, stated as "(say)" and so declared as
//! conventions: `k^` within `[0.1, k]`, the step not falling by more than a
//! factor 10.

use std::sync::atomic::{AtomicU64, Ordering};

/// Whether the probe is on.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_OBSERVED_ORDER").is_some();
        if on {
            println!("knob active: the order observed over two refusals in a row");
        }
        on
    })
}

/// His lower limit on the observed order, and on the step's fall.
const LOWEST_ORDER: f64 = 0.1;
const LARGEST_FALL: f64 = 0.1;

static USED: AtomicU64 = AtomicU64::new(0);
static BELOW: AtomicU64 = AtomicU64::new(0);
static ABOVE: AtomicU64 = AtomicU64::new(0);
static FALL_LIMITED: AtomicU64 = AtomicU64::new(0);

/// The ratio of the next try to `h_n`, from the last two refused tries'
/// `(relative error, step)`, `k` the order the error model assumes (the
/// estimate's order plus one). `None` when the two steps are equal, where
/// eq. 17 has no value.
pub(crate) fn ratio(previous: (f64, f64), last: (f64, f64), k: f64) -> Option<f64> {
    let ((r0, h0), (r1, h1)) = (previous, last);
    if h1 == h0 || r0 <= 0.0 || r1 <= 0.0 {
        return None;
    }
    let observed = (r1 / r0).ln() / (h1 / h0).ln();
    let limited = observed.clamp(LOWEST_ORDER, k);
    USED.fetch_add(1, Ordering::Relaxed);
    if observed < LOWEST_ORDER {
        BELOW.fetch_add(1, Ordering::Relaxed);
    } else if observed > k {
        ABOVE.fetch_add(1, Ordering::Relaxed);
    }
    let ratio = (1.0 / r1).powf(1.0 / limited);
    if ratio < LARGEST_FALL {
        FALL_LIMITED.fetch_add(1, Ordering::Relaxed);
    }
    Some(ratio.max(LARGEST_FALL))
}

/// One line: the uses so far on all threads.
pub(crate) fn report() -> String {
    let read = |c: &AtomicU64| c.load(Ordering::Relaxed);
    format!(
        "{} tries after two refusals; observed order below 0.1 {}, above k {}; fall limited to 10 {}",
        read(&USED),
        read(&BELOW),
        read(&ABOVE),
        read(&FALL_LIMITED)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An error that follows `r = phi h^3` gives back the order 3, and the
    /// step that meets the target.
    #[test]
    fn a_model_error_gives_back_its_order() {
        let ratio = ratio((8.0, 2.0), (4.096, 1.6), 3.0).expect("a ratio");
        // `phi = 1`: the target `r = 1` is met at `h = 1`, `1 / 1.6` of `h_n`.
        assert!((ratio - 1.0 / 1.6).abs() < 1e-12, "{ratio}");
    }

    /// An error that grows while the step falls (a negative observed order)
    /// is limited to 0.1, so the step falls by the largest factor, 10.
    #[test]
    fn a_growing_error_falls_by_the_largest_factor() {
        let ratio = ratio((6.9, 2.57e-3), (10.6, 1.30e-3), 3.0).expect("a ratio");
        assert_eq!(ratio, LARGEST_FALL);
    }

    /// Equal steps leave the order undefined.
    #[test]
    fn equal_steps_give_no_order() {
        assert_eq!(ratio((2.0, 1e-3), (3.0, 1e-3), 3.0), None);
    }
}
