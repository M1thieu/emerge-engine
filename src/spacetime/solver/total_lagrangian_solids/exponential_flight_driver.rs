//! The turning flight's drivers (`exponential_flight`): its start, its
//! advance and check, and the two entry points of the frame loop.

use super::*;

/// Whether every outline point of the rebuilt state lies inside every wall.
fn inside(state: &TotalLagrangianState, planes: &[Plane]) -> bool {
    state.outlines.iter().flatten().all(|&o| {
        planes
            .iter()
            .all(|plane| plane.inward * (state.x[o][plane.axis] - plane.at) > 0.0)
    })
}

/// Starts a flight when none is under way (today's conditions); `None` when
/// the trial does not take this body (today's flight then), `Some(false)`
/// when no flight could start.
fn ensure(
    state: &mut TotalLagrangianState,
    (t0, gravity): (f64, DVec2),
    knob: Knob,
) -> Option<bool> {
    if state.exponential.is_some() {
        return Some(true);
    }
    let free =
        state.outlines.len() == 1 && state.cache.as_ref().is_some_and(|c| c.contacts.is_empty());
    if !free {
        return Some(false);
    }
    let undamped = state.solids[..state.particles].iter().all(|s| {
        s.law == ElasticLaw::Corotated && s.shear_viscosity == 0.0 && s.bulk_viscosity == 0.0
    });
    if !undamped {
        return None;
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
        return Some(false);
    };
    if !modes.memory.is_empty() {
        return None;
    }
    state.exponential = Exponential::start(state, modes, t0, (knob.truncated, knob.law));
    // With the law's energy, the start's projection remainder (sec. 26,
    // part (d)): the step's energy of the state rebuilt from the flight's
    // own coordinates at its start, against the stepped state's.
    if let Some(mut flight) = state.exponential.take() {
        if flight.inv.law.is_some()
            && let Some(Some(modes)) = state.modes.take()
        {
            let saved = (state.x.clone(), state.v.clone(), state.f.clone());
            let stepped = law::step_energy(state, gravity).0;
            flight.clone().rebuild(&modes, gravity, state);
            let rebuilt = law::step_energy(state, gravity).0;
            (state.x, state.v, state.f) = saved;
            flight.start_remainder = Some(rebuilt - stepped);
            state.modes = Some(Some(modes));
        }
        state.exponential = Some(flight);
    }
    Some(state.exponential.is_some())
}

/// Advances the flight to `t1` in `steps` steps, rebuilds, checks; the
/// outcome, the flight kept in `state` only when flown.
fn advance(
    state: &mut TotalLagrangianState,
    mut flight: Exponential,
    (planes, gravity): (&[Plane], DVec2),
    t1: f64,
    steps: usize,
) -> Outcome {
    let t0 = flight.t;
    let mut records = Vec::with_capacity(steps);
    for k in 1..=steps {
        let to = if k == steps {
            t1
        } else {
            t0 + (t1 - t0) * k as f64 / steps as f64
        };
        match flight.step(to) {
            Ok(r) => records.push(r),
            Err(outcome) => {
                state.exponential_log.push(Record {
                    t0,
                    t1,
                    outcome,
                    ..Default::default()
                });
                state.hand_backs += 1;
                return outcome;
            }
        }
    }
    let Some(Some(modes)) = state.modes.take() else {
        return Outcome::NotClear;
    };
    let start = (state.x.clone(), state.v.clone(), state.f.clone());
    let before = flight
        .inv
        .law
        .is_some()
        .then(|| law::step_energy(state, gravity));
    flight.rebuild(&modes, gravity, state);
    state.modes = Some(Some(modes));
    // With the law's energy (sec. 26, E-1): the step's energy change over
    // the attempt and the round-off bound of its two evaluations.
    // FAST_MODE_PLAN.md sec. 29 (L-2): the lemma's inequalities on the
    // rebuilt state, measured only.
    law::lemma::check_state(&flight, state);
    if let Some((e0, b0)) = before {
        let (e1, b1) = law::step_energy(state, gravity);
        let remainder = flight.start_remainder.take();
        if let Some(first) = records.first_mut() {
            first.start_remainder = remainder;
        }
        for r in &mut records {
            r.law = true;
            r.gram_defect = flight.inv.gram_defect;
        }
        if let Some(last) = records.last_mut() {
            last.step_energy_change = e1 - e0;
            last.step_energy_bound = b0 + b1 + 0.5 * f64::EPSILON * (e1 - e0).abs();
        }
    }
    let outcome = if !inside(state, planes) {
        Outcome::EndGap
    } else if !flight.keeps_angular_momentum(state) {
        Outcome::AngularMomentum
    } else {
        Outcome::Flown
    };
    if outcome != Outcome::Flown {
        (state.x, state.v, state.f) = start;
        state.hand_backs += 1;
        if let Some(last) = records.last_mut() {
            last.outcome = outcome;
        }
    }
    state.exponential_log.extend(records);
    if outcome == Outcome::Flown {
        state.exponential = Some(flight);
    }
    outcome
}

/// `fly_total_lagrangian_frame` under the trial: `None` when the trial is
/// off or does not take this body.
pub(in super::super) fn fly_frame(
    state: &mut TotalLagrangianState,
    (planes, gravity): (&[Plane], DVec2),
    length: f64,
) -> Option<bool> {
    let knob = knob();
    if !knob.on {
        return None;
    }
    let (t0, t1) = (state.time, state.time + length);
    if !ensure(state, (t0, gravity), knob)? {
        return Some(false);
    }
    let flight = state.exponential.take().expect("started");
    if !flight.clear(gravity, planes, (t0, t1)) {
        state.exponential_log.push(Record {
            t0,
            t1,
            outcome: Outcome::NotClear,
            ..Default::default()
        });
        state.hand_backs += 1;
        handoff::trace(|| format!("t {t0:.6}: whole frame: exponential flight: NotClear"));
        return Some(false);
    }
    let outcome = advance(state, flight, (planes, gravity), t1, knob.steps);
    if outcome == Outcome::Flown {
        state.modal_frames += 1;
    } else {
        handoff::trace(|| format!("t {t0:.6}: whole frame: exponential flight: {outcome:?}"));
    }
    Some(outcome == Outcome::Flown)
}

impl Simulation {
    /// `fly_as_far_as_clear` under the trial: `None` when the trial is off
    /// or does not take this body.
    pub(in super::super) fn fly_exponential_as_far_as_clear(
        &self,
        state: &mut TotalLagrangianState,
        (planes, gravity): (&[Plane], DVec2),
        (t0, t1): (f64, f64),
    ) -> Option<(f64, f64)> {
        let knob = knob();
        if !knob.on {
            return None;
        }
        // KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: on the former real-time baseline (fixed contact step, now removed) this flight removes the angular-check refusals and the long-run chatter of the hopping clay, but each hand-back keeps the linear basis's residual (the law's energy of the rebuilt state against the modes' quadratic energy, -7.5e-2 to +6.7e-2 J/m per segment), a net gain of 0.17 J/m over 420 frames: higher hops, fewer contacts (long run kept 0.46) | source: FAST_MODE_PLAN.md sec. 25a | replace when: the flight's energy is the law's own, so a hand-back keeps the step's energy | state: measured, not adopted | gap: none
        if !ensure(state, (t0, gravity), knob)? {
            let held = state.cache.as_ref().is_some_and(|c| !c.contacts.is_empty());
            if held {
                law::lemma::contact_held();
            }
            handoff::trace(|| {
                format!("t {t0:.6}: exponential flight: not started (contact held {held})")
            });
            return Some((t0, 0.0));
        }
        let flight = state.exponential.take().expect("started");
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
        if reach <= t0 {
            state.exponential_log.push(Record {
                t0,
                t1,
                outcome: Outcome::NotClear,
                ..Default::default()
            });
            state.hand_backs += 1;
            handoff::trace(|| format!("t {t0:.6}: exponential flight: NotClear"));
            return Some((t0, 0.0));
        }
        let before = self.total_lagrangian_energy(state);
        let outcome = advance(state, flight, (planes, gravity), reach, knob.steps);
        if outcome != Outcome::Flown {
            handoff::trace(|| format!("t {t0:.6}: exponential flight: {outcome:?}"));
            return Some((t0, 0.0));
        }
        let change = self.total_lagrangian_energy(state) - before;
        handoff::trace(|| {
            format!(
                "t {t0:.6}: exponential flight: flown to {reach:.6}, energy change {change:+.4e} J/m"
            )
        });
        if reach < t1 {
            state.exponential = None;
            state.hand_backs += 1;
        }
        Some((reach, change))
    }
}
