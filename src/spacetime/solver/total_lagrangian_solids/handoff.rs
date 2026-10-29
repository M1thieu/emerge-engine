//! A frame cut where its contact is (`TotalLagrangianConfig::subframe_handoff`):
//! the lone body flown by its modes up to the last time its contact points
//! are provably clear of the walls, the step from there, and the flight
//! resumed for the rest of the frame once the step's contact has ended.
//! Without it a frame is either flown whole or stepped whole, so an impact
//! frame steps its free part too.
//!
//! The cut is the flight's own clearance bound (`Flight::clear`, a lower
//! bound on each contact point's gap, a minimum over the interval, so
//! monotone in the interval's end), bisected to the time's round-off: no
//! new tolerance. A flight resumes only after a contact held in the frame
//! has ended (the last step held none), not before the frame's first
//! contact, so the step and the flight do not alternate as the body nears
//! the wall. Each flown segment's energy change (the linearisation's
//! residual) enters the frame's `modal_energy_change`, as a flown frame's
//! does. Undamped flights only: a damped flight's propagator advances whole
//! frames (`Flight::rebuild`), so with relaxation branches the frame is
//! stepped as before (declared).

use super::flight::Flight;
use super::*;

#[cfg(all(test, feature = "experimental"))]
#[path = "mode_free_probes.rs"]
pub(super) mod mode_free_probes;

/// MIXED_FORMULATION_PLAN.md: the mean-dilatation cells on this path.
#[cfg(all(test, feature = "experimental"))]
#[path = "theta_cells.rs"]
pub(super) mod theta_cells;

/// MODE_FREE_HANDOFF_PLAN.md M0 at a flight start (the frame's and the
/// cut's): the mode-free bound against the modal one.
#[cfg(all(test, feature = "experimental"))]
/// With `EMERGE_PROBE_MODE_FREE_CUT` the started flight's contact-point
/// bounds are replaced by the mode-free trace form (sec. 7's cost of the
/// looser bound, measured on the path; a non-default probe).
pub(super) fn probe_start(
    state: &TotalLagrangianState,
    started: Result<Flight, flight::Refusal>,
    walls: (&[Plane], DVec2),
    interval: (f64, f64),
) -> Result<Flight, flight::Refusal> {
    let flight = started?;
    mode_free_probes::compare(state, &flight, walls, interval);
    Ok(if mode_free_probes::cut_on() {
        flight.with_arm_bounds(&mode_free_probes::free(state).bounds())
    } else {
        flight
    })
}

/// `EMERGE_PROBE_FLIGHT_TRACE`, a measurement: each flight attempt's
/// outcome printed.
#[cfg(all(test, feature = "experimental"))]
pub(super) fn trace(line: impl FnOnce() -> String) {
    if tracing() {
        println!("    flight: {}", line());
    }
}

/// Whether `EMERGE_PROBE_FLIGHT_TRACE` is set: the trace's own measurements
/// run only then, so an untraced run times the path alone.
#[cfg(all(test, feature = "experimental"))]
pub(super) fn tracing() -> bool {
    std::env::var_os("EMERGE_PROBE_FLIGHT_TRACE").is_some()
}

/// The angular-momentum check's measured excess against its exact-arithmetic
/// prediction (`Flight::predicted_angular`), traced at every check.
#[cfg(all(test, feature = "experimental"))]
pub(super) fn trace_angular(
    flight: &Flight,
    modes: &crate::spacetime::total_lagrangian::Modes,
    state: &TotalLagrangianState,
    (t0, t): (f64, f64),
) {
    if !tracing() {
        return;
    }
    let (rebuilt, own, tolerance) = flight.angular_parts(state);
    let ((predicted, turning, leak, gyro), start) = flight.predicted_angular(modes, state, t);
    let excess = rebuilt - own;
    trace(|| {
        format!(
            "L check from {t0:.6} to {t:.6}: kept {}; measured excess {excess:+.3e}, tolerance {tolerance:.3e}; predicted exact {predicted:+.3e} (turning {turning:+.3e}, leak {leak:+.3e}, gyroscopic {gyro:+.3e}; the modal start's own {start:+.3e}, its round-off bound {:.3e} (spin {:.2e}, split {:.2e}, projection {:.2e}), ratio {:.3e}; det back - 1 {:+.2e}, angle error {:.2e}, scaling term {:.2e}, predicted exact {:+.3e}); remainder {:+.3e} = {:.3} of the tolerance",
            excess.abs() <= tolerance,
            flight.start_bound.iter().sum::<f64>(),
            flight.start_bound[0],
            flight.start_bound[1],
            flight.start_bound[2],
            start.abs()
                / flight
                    .start_bound
                    .iter()
                    .sum::<f64>()
                    .max(f64::MIN_POSITIVE),
            flight.start_extra[0],
            flight.start_extra[1],
            flight.start_extra[2],
            flight.start_extra[3],
            excess - predicted,
            (excess - predicted).abs() / tolerance.max(f64::MIN_POSITIVE)
        )
    });
}

#[cfg(all(test, feature = "experimental"))]
thread_local! {
    /// S1 (`tmp/papers/dl_realtime/S1_FIXED_CONTACT_STEP_PLAN.md`): whether
    /// the frame loop is inside a contact phase, the last flight attempt
    /// having stopped at the clearance bound or on a held contact, not on a
    /// refusal.
    static CONTACT_PHASE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(test, feature = "experimental"))]
fn contact_phase(on: bool) {
    CONTACT_PHASE.set(on);
}

/// S1's control for the next controlled step: with
/// `EMERGE_PROBE_CONTACT_FIXED_STEP=k` set and inside a contact phase, an
/// infinite tolerance and the step `2 k h / c` (its two halves of `k h / c`
/// kept, as reference A and the step sweep take theirs), cut to the
/// frame's remaining time; `c` the body's fastest P wave, `sqrt((lambda +
/// 2 mu) / rho)` with `lambda + 2 mu` the code's `bulk + mu`. Otherwise
/// `control` unchanged.
#[cfg(all(test, feature = "experimental"))]
pub(super) fn contact_phase_control(
    control: &StepControl,
    state: &mut TotalLagrangianState,
    remaining: f64,
) -> StepControl {
    // MIXED_FORMULATION_PLAN.md: the Theta cells, once, at the first step.
    theta_cells::ensure(state);
    // S1PP_STEPPED_PLAN.md: the stepped path's own contact phase first.
    if let Some(fixed) = super::reach_cut::stepped_phase::control(control, state, remaining) {
        return fixed;
    }
    static K: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let k = *K.get_or_init(|| {
        std::env::var("EMERGE_PROBE_CONTACT_FIXED_STEP")
            .ok()
            .and_then(|v| v.parse().ok())
    });
    let (Some(k), true) = (k, CONTACT_PHASE.get()) else {
        return *control;
    };
    // The knob's confirmation, once, from inside the patched path.
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    ANNOUNCED.call_once(|| println!("knob active: S1 fixed contact step, k {k}"));
    let Some(cell) = state.cache.as_ref().map(|c| c.grid.cell) else {
        return *control;
    };
    let c = (0..state.particles)
        .filter(|&p| state.v0[p] > 0.0)
        .map(|p| ((state.solids[p].bulk + state.solids[p].mu) * state.v0[p] / state.mass[p]).sqrt())
        .fold(0.0, f64::max);
    state.step_length = (2.0 * k * cell / c).min(remaining);
    trace(|| {
        format!(
            "S1: fixed step {:.6e} s (k {k}, h {cell} m, c {c:.4} m/s)",
            state.step_length
        )
    });
    StepControl {
        tolerance: f64::INFINITY,
        ..*control
    }
}

impl Simulation {
    /// Flies the lone body from `t0` toward `t1` as far as its contact
    /// points stay provably clear of the walls; returns the time reached
    /// (`t0` when no flight could be taken) and the flown part's energy
    /// change (kinetic, stored and gravity's), J/m. The flight is kept when
    /// it reaches `t1` and ends otherwise (a hand-back).
    pub(super) fn fly_as_far_as_clear(
        &self,
        state: &mut TotalLagrangianState,
        (planes, gravity): (&[Plane], DVec2),
        (t0, t1): (f64, f64),
    ) -> (f64, f64) {
        #[cfg(all(test, feature = "experimental"))]
        if let Some(reached) =
            self.fly_exponential_as_far_as_clear(state, (planes, gravity), (t0, t1))
        {
            return reached;
        }
        // Not a contact phase unless the flight stops at the clearance bound
        // or on a held contact (a refusal is not one: no modes, yielded, a
        // bound refused, damped, a failed rebuild or angular check).
        #[cfg(all(test, feature = "experimental"))]
        contact_phase(false);
        if state.flight.is_none() {
            let free = state.outlines.len() == 1
                && state.cache.as_ref().is_some_and(|c| c.contacts.is_empty());
            if !free {
                #[cfg(all(test, feature = "experimental"))]
                trace(|| format!("t {t0:.6}: not free (a contact held at the last step)"));
                // A contact phase only if a contact is held (before the first
                // step there is no cache: not a contact).
                #[cfg(all(test, feature = "experimental"))]
                contact_phase(state.cache.as_ref().is_some_and(|c| !c.contacts.is_empty()));
                return (t0, 0.0);
            }
            if state.modes.is_none() {
                let grid = &state.cache.as_ref().expect("checked above").grid;
                state.modes = Some(body_modes(
                    grid,
                    (&state.mass, &state.v0, &state.solids),
                    (state.particles, state.x.len()),
                ));
            }
            let Some(Some(modes)) = &state.modes else {
                state.modal_refused[0] += 1;
                return (t0, 0.0);
            };
            let started = Flight::start(state, modes, (t0, t1 - t0));
            #[cfg(all(test, feature = "experimental"))]
            let started = probe_start(state, started, (planes, gravity), (t0, t1));
            match started {
                Err(flight::Refusal::Yielded) => state.modal_refused[1] += 1,
                Err(flight::Refusal::Bound) => state.modal_refused[2] += 1,
                _ => {}
            }
            #[cfg(all(test, feature = "experimental"))]
            if let Err(refusal) = &started {
                trace(|| format!("t {t0:.6}: refused to start: {refusal:?}"));
            }
            state.flight = started.ok();
            #[cfg(all(test, feature = "experimental"))]
            if let (Some(flight), Some(Some(modes))) = (&state.flight, &state.modes) {
                super::g0::at_flight_start(flight, modes, state);
            }
        }
        let Some(mut flight) = state.flight.take() else {
            return (t0, 0.0);
        };
        if flight.is_damped() {
            state.hand_backs += 1;
            return (t0, 0.0);
        }
        // The last time the clearance bound holds, bisected until the
        // interval is the times' round-off (the frame loop's own measure
        // of a time left over).
        let reach = if flight.clear(gravity, planes, (t0, t1)) {
            t1
        } else {
            let (mut low, mut high) = (t0, t1);
            while high - low > 4.0 * f64::EPSILON * high.abs().max(t1 - t0) {
                let middle = 0.5 * (low + high);
                if flight.clear(gravity, planes, (t0, middle)) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            low
        };
        #[cfg(all(test, feature = "experimental"))]
        if tracing() {
            let (least, largest) = flight.clearance_report(gravity, planes, (t0, t1));
            let sampled = match &state.modes {
                Some(Some(modes)) => {
                    flight.sampled_gaps(modes, (gravity, planes), state, (t0, t1), 4096)
                }
                _ => (f64::NAN, f64::NAN, f64::NAN),
            };
            trace(|| {
                format!(
                    "t {t0:.6} to {t1:.6}: reach {reach:.6}; over the whole interval the least gap bound {least:+.4e} m, the largest vibration bound {largest:.4e} m; sampled: least gap {:+.4e} m, clear from {:.6}, largest inward vibration {:.3} of its bound",
                    sampled.0, sampled.1, sampled.2
                )
            });
        }
        if reach <= t0 {
            #[cfg(all(test, feature = "experimental"))]
            contact_phase(true);
            state.hand_backs += 1;
            return (t0, 0.0);
        }
        let Some(Some(modes)) = state.modes.take() else {
            return (t0, 0.0);
        };
        let before = self.total_lagrangian_energy(state);
        let start = (state.x.clone(), state.v.clone(), state.f.clone());
        if !flight.rebuild(&modes, gravity, reach, state) {
            state.modes = Some(Some(modes));
            state.modal_check_failed += 1;
            state.hand_backs += 1;
            return (t0, 0.0);
        }
        #[cfg(all(test, feature = "experimental"))]
        trace_angular(&flight, &modes, state, (t0, reach));
        if !flight.keeps_angular_momentum(state) {
            #[cfg(all(test, feature = "experimental"))]
            if tracing() {
                let (rebuilt, own, tolerance) = flight.angular_parts(state);
                trace(|| {
                    format!(
                        "t {t0:.6}: angular momentum not kept: excess {:.3e} = {:.3} of the tolerance",
                        (rebuilt - own).abs(),
                        (rebuilt - own).abs() / tolerance.max(f64::MIN_POSITIVE)
                    )
                });
            }
            (state.x, state.v, state.f) = start;
            state.modes = Some(Some(modes));
            state.hand_backs += 1;
            return (t0, 0.0);
        }
        flight.carry_branches(&modes, state, &start.2);
        state.modes = Some(Some(modes));
        let change = self.total_lagrangian_energy(state) - before;
        if reach >= t1 {
            state.flight = Some(flight);
        } else {
            #[cfg(all(test, feature = "experimental"))]
            contact_phase(true);
            state.hand_backs += 1;
        }
        (reach, change)
    }
}
