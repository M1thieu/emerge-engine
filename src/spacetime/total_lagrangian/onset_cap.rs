//! FAST_MODE_PLAN.md sec. 50.6, a test-only measurement (knob
//! `EMERGE_PROBE_ONSET_CAP`, default off): when the error controller
//! refuses a try inside which a contact first holds, its next try is capped
//! at the impact's duration, so the controller does not halve its way down
//! from a frame-long try through two decades of refusals.
//!
//! The duration is the law's: `t_1 = 2 H / c`, the contact time of a body of
//! height `H` struck along that height, `c` its fastest P wave (Wang and
//! Hansen 2016, read, FAST_MODE_PLAN.md sec. 45: 0.443 ms for the V0 block).
//! `c = sqrt((K + mu) / rho)` with the code's `bulk + mu` and `rho = m / V0`,
//! as S1's fixed contact step takes it; `H` is the body's extent along the
//! walls' normal. Every kept step still passes the same error test with the
//! same tolerance: a choice of the next try's length, not of what is kept.
//! Adaptations, declared: the walls' first plane gives the normal; with
//! several bodies, the extent of all of them (one body in the measurement).

use glam::DVec2;

use super::contact::Plane;
use super::elastic::Elastic;

/// Whether the knob is set; says so once.
pub(super) fn knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_ONSET_CAP").is_some();
        if on {
            println!("knob active: onset cap, a refused try with a contact onset caps the next at 2 H / c (sec. 50.6)");
        }
        on
    })
}

/// `2 H / c` for the massive points, or `None` without a wall or matter.
pub(super) fn impact_time(
    x: &[DVec2],
    (mass, v0): (&[f64], &[f64]),
    solids: &[Elastic],
    planes: &[Plane],
) -> Option<f64> {
    let axis = planes.first()?.axis;
    let (mut low, mut high, mut speed) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for p in 0..x.len() {
        if mass[p] <= 0.0 || v0[p] <= 0.0 {
            continue;
        }
        low = low.min(x[p][axis]);
        high = high.max(x[p][axis]);
        speed = speed.max(((solids[p].bulk + solids[p].mu) * v0[p] / mass[p]).sqrt());
    }
    (speed > 0.0 && high > low).then(|| 2.0 * (high - low) / speed)
}

/// Whether `EMERGE_PROBE_NOELS_REJECT` asks for Noels et al.'s reduction on a
/// refused try (FAST_MODE_PLAN.md sec. 50.7); says so once.
pub(super) fn noels_knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_NOELS_REJECT").is_some();
        if on {
            println!("knob active: a refused try's next length times (target / (2 err))^(2/3) (sec. 50.7)");
        }
        on
    })
}

/// The factor on a refused try's length at the error `relative` (its error
/// over the target): Noels, Stainier, Ponthot and Bonini 2002, sec. 3.3,
/// eq. 32 with `eta = 3/2`, "at a rapid change in the physical problem
/// (impact...)": `RAT = [0.5 PRCU / e]^(2/3)`, this engine's target standing
/// for PRCU (an adaptation, declared: their error is the acceleration jump,
/// here it is the step-doubling estimate).
pub(super) fn noels_ratio(relative: f64) -> f64 {
    (0.5 / relative).powf(2.0 / 3.0)
}
