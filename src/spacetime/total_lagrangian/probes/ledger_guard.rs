//! A kept step's energy identity as a check on the try itself (test builds,
//! `EMERGE_PROBE_LEDGER_GUARD`, default off).
//!
//! Source: the discrete-gradient midpoint conserves the energy per step for
//! any step length, or with plasticity dissipates exactly the plastic work
//! (Love and Sulsky, HPC@UNM TR 2004-002 and TR 2005-001, Prop. 3.1). At a
//! solution of the step's equations `dE + D = W_c`, with `D` the heat the step
//! books and `W_c` the walls' normal work; at Newton's kept iterate the
//! difference is `u . r`. So, with no constant chosen:
//!
//! `|dE + D| <= residual_work + contact_work_bound + n eps (sum |E parts| + |D|)`
//!
//! the last term the sums' round-off, the form the frame's ledger closes with.
//! A try beyond that bound is no solution of the scheme's equations to the
//! precision it claims. It is refused and the step halved, as an unconverged
//! try is; at the estimator's floor, with nothing shorter to try, it is kept,
//! printed and counted (`ControlledStep::ledger_breaches`, which the frame sums
//! for a host to read).

use super::StepReport;

/// Whether `EMERGE_PROBE_LEDGER_GUARD` is set; says so once.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_LEDGER_GUARD").is_some();
        if on {
            println!("knob active: a kept step must satisfy its energy identity");
        }
        on
    })
}

/// Runs one check and adds its time to `spent`.
pub(crate) fn timed<T>(check: impl FnOnce() -> T) -> T {
    let clock = std::time::Instant::now();
    let out = check();
    SPENT_NS.fetch_add(
        clock.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    out
}

/// The checks' total time so far, s.
pub(crate) fn spent() -> f64 {
    1e-9 * SPENT_NS.load(std::sync::atomic::Ordering::Relaxed) as f64
}

static SPENT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A state's kinetic, stored and potential energy.
pub(crate) type EnergyParts = (f64, f64, f64);

/// The energy balance of a try, and the bound its identity allows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Balance {
    pub(crate) balance: f64,
    pub(crate) bound: f64,
}

impl Balance {
    /// From the try's start and end energies and its pieces' reports, on `n`
    /// points.
    pub(crate) fn of(
        start: EnergyParts,
        end: EnergyParts,
        reports: &[StepReport],
        n: usize,
    ) -> Self {
        let total = |e: EnergyParts| e.0 + e.1 - e.2;
        let heat: f64 = reports
            .iter()
            .map(|r| r.relaxation + r.viscous + r.plastic + r.friction)
            .sum();
        let allowed: f64 = reports
            .iter()
            .map(|r| r.residual_work + r.contact_work_bound)
            .sum();
        let parts = end.0.abs() + end.1.abs() + end.2.abs() + total(start).abs() + heat.abs();
        Self {
            balance: total(end) - total(start) + heat,
            bound: allowed + n as f64 * f64::EPSILON * parts,
        }
    }

    pub(crate) fn breached(&self) -> bool {
        self.balance.abs() > self.bound
    }
}

/// Prints a breach: every one kept at the floor, the first refusals.
pub(crate) fn report(balance: Balance, dt: f64, kept: bool) {
    const REFUSALS_PRINTED: usize = 200;
    static REFUSALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let refusal =
        !kept && REFUSALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= REFUSALS_PRINTED;
    if !refusal {
        println!(
            "LEDGER BREACH {}: dt {dt:.3e} s, balance {:+.4e} J/m, bound {:.4e} J/m",
            if kept { "KEPT at the floor" } else { "refused" },
            balance.balance,
            balance.bound
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report_with(residual_work: f64, heat: f64) -> StepReport {
        StepReport {
            residual_work,
            plastic: heat,
            ..StepReport::default()
        }
    }

    /// The identity with a residual's allowance: a balance inside it passes,
    /// one beyond it is a breach, on either sign.
    #[test]
    fn a_try_beyond_its_identity_is_a_breach() {
        let start = (1.0, 2.0, 0.0);
        // The energy falls by 0.5, the heat books 0.5 less 1e-3: balance -1e-3.
        let end = (1.0, 1.5, 0.0);
        let inside = Balance::of(start, end, &[report_with(2e-3, 0.499)], 100);
        assert!(!inside.breached(), "{inside:?}");
        let beyond = Balance::of(start, end, &[report_with(5e-4, 0.499)], 100);
        assert!(beyond.breached(), "{beyond:?}");
        let above = Balance::of(start, end, &[report_with(5e-4, 0.501)], 100);
        assert!(above.breached() && above.balance > 0.0, "{above:?}");
    }

    /// A balance at the sums' round-off is no breach, with no residual at all.
    #[test]
    fn a_balance_at_the_round_off_is_no_breach() {
        let start = (1.0, 2.0, 0.0);
        let end = (1.0 + 4.0 * f64::EPSILON, 2.0, 0.0);
        let balance = Balance::of(start, end, &[report_with(0.0, 0.0)], 100);
        assert!(balance.balance != 0.0 && !balance.breached(), "{balance:?}");
    }
}
