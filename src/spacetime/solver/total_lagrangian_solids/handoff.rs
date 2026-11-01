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
    /// FAST_MODE_PLAN.md sec. 21: the last held contact's contact time,
    /// kept through the phase's free runs between pushes (no contact held
    /// there), forgotten once a flight is kept to a frame's end.
    static CONTACT_TIME: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
}

#[cfg(all(test, feature = "experimental"))]
fn contact_phase(on: bool) {
    CONTACT_PHASE.set(on);
}

// KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: with k above about 20 the fixed contact step is cut to the frame's remaining time, so the whole contact phase is one conserving midpoint step and k no longer sets it | source: FAST_MODE_PLAN.md sec. 14b, 21a | replace when: the contact phase is stepped at k <= 1/4 within every frame's budget, or an error-controlled large step bounds the contact's error | state: researched; the sourced contact-time rule (sec. 21) measured and not adopted | gap: k is chosen, not derived (sec. 21a)
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
    planes: &[Plane],
    remaining: f64,
) -> StepControl {
    // MIXED_FORMULATION_PLAN.md: the Theta cells, once, at the first step.
    theta_cells::ensure(state);
    // S1PP_STEPPED_PLAN.md: the stepped path's own contact phase first.
    if let Some(fixed) = super::reach_cut::stepped_phase::control(control, state, remaining) {
        return fixed;
    }
    // FAST_MODE_PLAN.md sec. 13 (F4): with `EMERGE_PROBE_BACKWARD_EULER=dt`
    // (seconds), inside a contact phase, each controlled step is `2 dt` by
    // backward Euler at an infinite tolerance, its two pieces of `dt` kept
    // (with the single-solve knob, solved alone); the midpoint elsewhere.
    static BACKWARD: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let backward = *BACKWARD.get_or_init(|| {
        std::env::var("EMERGE_PROBE_BACKWARD_EULER")
            .ok()
            .and_then(|v| v.parse().ok())
    });
    let in_phase = CONTACT_PHASE.get();
    crate::spacetime::total_lagrangian::hooks::set_backward_euler(backward.is_some() && in_phase);
    if let (Some(dt), true) = (backward, in_phase) {
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| println!("knob active: backward Euler contact step, dt {dt} s"));
        state.step_length = (2.0 * dt).min(remaining);
        trace(|| {
            format!(
                "F4: backward Euler step {:.6e} s (pieces of {dt:.6e} s)",
                state.step_length
            )
        });
        return StepControl {
            tolerance: f64::INFINITY,
            ..*control
        };
    }
    static K: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let k = *K.get_or_init(|| {
        std::env::var("EMERGE_PROBE_CONTACT_FIXED_STEP")
            .ok()
            .and_then(|v| v.parse().ok())
    });
    // FAST_MODE_PLAN.md sec. 21: the contact step from the contact's own
    // duration, in place of k.
    static N: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let n = *N.get_or_init(|| {
        std::env::var("EMERGE_PROBE_CONTACT_STEP_RULE")
            .ok()
            .and_then(|v| v.parse().ok())
    });
    if let Some(n) = n {
        assert!(
            k.is_none(),
            "EMERGE_PROBE_CONTACT_STEP_RULE replaces EMERGE_PROBE_CONTACT_FIXED_STEP: set one"
        );
        if !CONTACT_PHASE.get() {
            return *control;
        }
        let held = round_trip_time(state, planes);
        if held.is_some() {
            CONTACT_TIME.set(held);
        }
        let Some(contact_time) = held.or(CONTACT_TIME.get()).or(smallest_round_trip(state)) else {
            return *control;
        };
        let piece = contact_time / n;
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            println!(
                "knob active: contact step rule, N {n}, first T_c {contact_time:.6e} s, piece {piece:.6e} s"
            );
        });
        state.step_length = (2.0 * piece).min(remaining);
        trace(|| {
            format!(
                "R: rule step {:.6e} s (T_c {contact_time:.6e} s, N {n})",
                state.step_length
            )
        });
        return StepControl {
            tolerance: f64::INFINITY,
            ..*control
        };
    }
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

// KNOWN-LIMIT[DISCRETIZATION/NO-SOURCE]: the contact step's N = 18 pieces per contact time is one example's statement (Love and Sulsky TR 2004-002 sec. 4.1: "about 18 times the time step, so reasonable accuracy is expected"), not a convergence result; the body's extent along the normal and the fastest bulk speed are adaptations | source: FAST_MODE_PLAN.md sec. 21, 21a | replace when: an error estimate inside the contact phase sets the step at a cost within the frame's budget | state: measured, not adopted | gap: fails the 50 % floor on the resting block (P1 sag kept 0.152, k = 39.33 keeps 0.885), cause not measured
/// FAST_MODE_PLAN.md sec. 21: the shortest contact time among the held
/// contacts, `T_c = 2 L_n / c_b`: the time a wave takes across the body and
/// back (Love and Sulsky TR 2004-002 sec. 4.1; Acary 2013, CMAME 256, the
/// elastic bar's exact `2 L / c0`), `L_n` the body's extent along the
/// contact's normal (its outline points, now), `c_b = sqrt(kappa / rho0)`
/// the linearised bulk wave, `kappa = bulk - mu / 3` (`bulk` is `lambda +
/// mu`), the fastest over the solid particles. `None` without a held
/// contact or a wave.
#[cfg(all(test, feature = "experimental"))]
fn round_trip_time(state: &TotalLagrangianState, planes: &[Plane]) -> Option<f64> {
    let cache = state.cache.as_ref()?;
    let c = bulk_wave(state)?;
    let x = &state.x;
    let extent = |outline: &[usize], normal: DVec2| extent(x, outline, normal);
    cache
        .contacts
        .iter()
        .flat_map(|contact| {
            contact
                .points
                .iter()
                .map(move |(point, _)| (contact.key, point))
        })
        .filter_map(|((kind, _, plane), point)| {
            let outline = state
                .outlines
                .iter()
                .find(|o| o.contains(&point.particle))?;
            // The normal: a segment's (against another body), else the
            // wall's axis (kinds 0 and 3 key the wall's plane; a point's
            // own `plane` indexes the step's copies, not `planes`). A pair
            // across a plane of its own (kind 1) names no wall: the body's
            // smaller extent, the shorter time (a declared adaptation).
            let length = match (point.segment, kind) {
                (Some((a, b)), _) => extent(outline, (x[b] - x[a]).perp().normalize_or_zero()),
                (None, 0 | 3) => extent(outline, DVec2::AXES[planes[plane].axis]),
                _ => extent(outline, DVec2::X).min(extent(outline, DVec2::Y)),
            };
            (length > 0.0).then(|| 2.0 * length / c)
        })
        .reduce(f64::min)
}

/// FAST_MODE_PLAN.md sec. 21: at a contact phase's first step no contact is
/// held yet: the shortest round trip through any body's smaller extent, the
/// shorter time (a declared adaptation). `None` without a body or a wave.
#[cfg(all(test, feature = "experimental"))]
fn smallest_round_trip(state: &TotalLagrangianState) -> Option<f64> {
    let c = bulk_wave(state)?;
    state
        .outlines
        .iter()
        .map(|o| extent(&state.x, o, DVec2::X).min(extent(&state.x, o, DVec2::Y)))
        .filter(|&length| length > 0.0)
        .map(|length| 2.0 * length / c)
        .reduce(f64::min)
}

/// The fastest linearised bulk wave over the solid particles, `sqrt(kappa /
/// rho0)`, `kappa = bulk - mu / 3`; `None` without one.
#[cfg(all(test, feature = "experimental"))]
fn bulk_wave(state: &TotalLagrangianState) -> Option<f64> {
    let c = (0..state.particles)
        .filter(|&p| state.v0[p] > 0.0)
        .map(|p| {
            let solid = &state.solids[p];
            ((solid.bulk - solid.mu / 3.0) * state.v0[p] / state.mass[p]).sqrt()
        })
        .fold(0.0, f64::max);
    (c > 0.0).then_some(c)
}

/// An outline's extent along `normal`, m.
#[cfg(all(test, feature = "experimental"))]
fn extent(x: &[DVec2], outline: &[usize], normal: DVec2) -> f64 {
    let along = outline.iter().map(|&q| x[q].dot(normal));
    let (low, high) = along.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), d| {
        (l.min(d), h.max(d))
    });
    high - low
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
            #[cfg(all(test, feature = "experimental"))]
            CONTACT_TIME.set(None);
        } else {
            #[cfg(all(test, feature = "experimental"))]
            contact_phase(true);
            state.hand_backs += 1;
        }
        (reach, change)
    }
}
