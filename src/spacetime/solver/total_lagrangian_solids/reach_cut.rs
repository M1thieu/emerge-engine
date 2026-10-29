//! ROW15_REACH_CUT_PLAN.md (test builds, `EMERGE_PROBE_REACH_CUT`): no
//! step's first try spans a contact the reaching test predicts. A free
//! body (no contact held, no point beyond a plane) whose point reaches a
//! plane in the controller's first pair has that step handed back; the
//! step is retried up to the last time the mode-free bound proves every
//! free body clear (`T_clear`), then one critical step to the first time
//! the upper bound proves a contact (`T_must`), its contacts forecast
//! (Acary RR-7151 eqs. 16e and 18, as `step.rs` cites them), after
//! Acary's Algorithm 1 (integrate up to the event, across it in a
//! critical step). No length of its own: the bounds, the estimator's floor.

use std::cell::Cell;

use glam::DVec2;

use super::handoff::mode_free_probes::{free_of, lower_gaps, solve_term_of, upper_gaps_at};
use super::{Simulation, TotalLagrangianState};
use crate::spacetime::total_lagrangian::{ConservingSettings, ControlledStep, Plane, hooks};

/// S1PP_STEPPED_PLAN.md: the fixed contact step on the stepped path, its
/// phases started by this rule's firing.
#[path = "stepped_phase.rs"]
pub(super) mod stepped_phase;

thread_local! {
    static PENDING: Cell<Option<f64>> = const { Cell::new(None) };
    static COUNTS: Cell<[u32; 4]> = const { Cell::new([0; 4]) };
}

/// Whether `EMERGE_PROBE_REACH_CUT` is set; says so once, from inside.
pub(super) fn on() -> bool {
    static SAID: std::sync::Once = std::sync::Once::new();
    let on = std::env::var_os("EMERGE_PROBE_REACH_CUT").is_some();
    if on {
        SAID.call_once(|| println!("knob active: reach cut (ROW15_REACH_CUT_PLAN.md)"));
    }
    on
}

fn count(which: usize) {
    COUNTS.with(|c| {
        let mut n = c.get();
        n[which] += 1;
        c.set(n);
    });
}

/// Each body's particles and outline points (the outlines in spawn order,
/// a body's particles by its tag).
fn bodies(sim: &Simulation, state: &TotalLagrangianState) -> Vec<(Vec<usize>, Vec<usize>)> {
    sim.total_lagrangian_outlines
        .iter()
        .zip(&state.outlines)
        .map(|(spawned, points)| {
            let particles = (0..state.particles)
                .filter(|&p| sim.particles.user_tag[p] == spawned.tag)
                .collect();
            (particles, points.clone())
        })
        .collect()
}

/// Whether a body is free at the state: none of its points beyond a plane,
/// no held contact on any of its particles or points.
fn is_free(
    state: &TotalLagrangianState,
    planes: &[Plane],
    body: &(Vec<usize>, Vec<usize>),
) -> bool {
    let (particles, points) = body;
    let beyond = points.iter().any(|&q| {
        planes
            .iter()
            .any(|plane| plane.inward * (state.x[q][plane.axis] - plane.at) < 0.0)
    });
    let held = state.cache.as_ref().is_some_and(|c| {
        c.contacts.iter().any(|contact| {
            contact.points.iter().any(|(point, _)| {
                particles.contains(&point.particle) || points.contains(&point.particle)
            })
        })
    });
    !beyond && !held
}

/// The window over `[t, t + dt]` for one body, relative to the step's
/// start: the last time every point's lower gap bound less `D_p` stays
/// positive, and the first time some point's upper bound plus `D_p` is at
/// most zero (`dt` if none). The fit and `D_p` from the step's start state;
/// `D_p` at `dt`, conservative for any shorter step (it grows as `dt^2`).
fn window(
    state: &TotalLagrangianState,
    body: &(Vec<usize>, Vec<usize>),
    (planes, gravity): (&[Plane], DVec2),
    (dt, tolerance): (f64, f64),
) -> (f64, f64) {
    let (particles, points) = body;
    let free = free_of(state, particles, points);
    let (terms, _, _) = solve_term_of(state, gravity, (dt, tolerance), (particles, points));
    let t0 = state.time;
    let resolved = |low: f64, high: f64| high - low > 4.0 * f64::EPSILON * high.abs().max(dt);
    let clear = |t: f64| {
        lower_gaps(state, &free, (planes, gravity), (t0, t))
            .iter()
            .zip(&terms)
            .all(|(g, d)| g - d > 0.0)
    };
    let until = if clear(t0 + dt) {
        t0 + dt
    } else {
        let (mut low, mut high) = (t0, t0 + dt);
        while resolved(low, high) {
            let middle = 0.5 * (low + high);
            if clear(middle) {
                low = middle;
            } else {
                high = middle;
            }
        }
        low
    };
    let must = |t: f64| {
        upper_gaps_at(state, &free, planes, gravity, (t0, t))
            .iter()
            .zip(&terms)
            .any(|(g, d)| g + d <= 0.0)
    };
    let first = if !must(t0 + dt) {
        t0 + dt
    } else {
        let (mut low, mut high) = (until, t0 + dt);
        while resolved(low, high) {
            let middle = 0.5 * (low + high);
            if must(middle) {
                high = middle;
            } else {
                low = middle;
            }
        }
        high
    };
    (until - t0, first - t0)
}

/// One step of the frame loop under the rule: `take` runs the controlled
/// step from `state` with the settings given; with the stepped contact step
/// (`stepped_phase`) the phases are followed after it.
pub(super) fn step(
    sim: &Simulation,
    state: &mut TotalLagrangianState,
    (planes, gravity): (&[Plane], DVec2),
    settings: &ConservingSettings,
    take: &dyn Fn(&mut TotalLagrangianState, &ConservingSettings) -> ControlledStep,
) -> ControlledStep {
    if stepped_phase::k().is_some() {
        stepped_phase::set_bodies(|| bodies(sim, state));
    }
    let step = step_inner(sim, state, (planes, gravity), settings, take);
    let tolerance = sim
        .config
        .total_lagrangian
        .as_ref()
        .map_or(settings.newton_tolerance, |c| c.settings.newton_tolerance);
    stepped_phase::after_step(
        state,
        (planes, gravity),
        (tolerance, step.dt, step.rejected),
        &|state: &TotalLagrangianState, body: &(Vec<usize>, Vec<usize>)| {
            is_free(state, planes, body)
        },
        &window,
    );
    step
}

fn step_inner(
    sim: &Simulation,
    state: &mut TotalLagrangianState,
    (planes, gravity): (&[Plane], DVec2),
    settings: &ConservingSettings,
    take: &dyn Fn(&mut TotalLagrangianState, &ConservingSettings) -> ControlledStep,
) -> ControlledStep {
    if let Some(length) = PENDING.with(Cell::take) {
        let length = length.min(state.step_length);
        let reached = critical_points(state);
        let before: Vec<DVec2> = reached.iter().map(|&q| state.v[q]).collect();
        let standoff = reached
            .iter()
            .map(|&q| least_gap(state, planes, q))
            .fold(f64::INFINITY, f64::min);
        hooks::set_reach_critical(Some(length));
        let forecast = ConservingSettings {
            forecast_activation: true,
            ..*settings
        };
        let step = take(state, &forecast);
        let jump = reached
            .iter()
            .zip(&before)
            .map(|(&q, v)| (state.v[q] - *v).length())
            .fold(0.0, f64::max);
        count(2);
        println!(
            "reach cut: critical step {:.6e} s (asked {length:.6e}), contacts {}, stand-off {standoff:+.4e} m, Lemma 1 bound C h |dv| = {:.4e} m (C = 1/2, |dv| {jump:.4e} m/s)",
            step.dt,
            step.kept.iter().map(|r| r.contacts).max().unwrap_or(0),
            0.5 * step.dt * jump
        );
        return step;
    }
    // Before the first step there is no grid, so no bound: the step as
    // today, counted ("no grid").
    if state.cache.is_none() {
        count(3);
        return take(state, settings);
    }
    let all = bodies(sim, state);
    let free: Vec<bool> = all.iter().map(|b| is_free(state, planes, b)).collect();
    let mut watch = vec![false; state.x.len()];
    // A body in its stepped contact phase is not watched: the fixed step
    // resolves its contacts, as in reference A.
    for (b, (body, &f)) in all.iter().zip(&free).enumerate() {
        if f && !stepped_phase::in_phase(b) {
            for &p in body.0.iter().chain(&body.1) {
                watch[p] = true;
            }
        }
    }
    hooks::set_reach_watch(Some(watch));
    let step = take(state, settings);
    hooks::set_reach_watch(None);
    let Some(abort) = hooks::take_reach_abort() else {
        return step;
    };
    let tolerance = sim
        .config
        .total_lagrangian
        .as_ref()
        .map_or(settings.newton_tolerance, |c| c.settings.newton_tolerance);
    let (mut until, mut first) = (abort.dt, abort.dt);
    for (body, &f) in all.iter().zip(&free) {
        if f {
            let (u, m) = window(state, body, (planes, gravity), (abort.dt, tolerance));
            until = until.min(u);
            first = first.min(m.max(u));
        }
    }
    if until > abort.floor && until < abort.dt {
        count(0);
        state.step_length = until;
        let step = take(state, settings);
        if step.reaching {
            println!(
                "reach cut: ALARM: the cut try of {until:.6e} s reached a plane the bound proved clear"
            );
        }
        let critical = (first - until).max(abort.floor);
        PENDING.with(|p| p.set(Some(critical)));
        REACHED.with(|r| *r.borrow_mut() = abort.reached.clone());
        stepped_phase::start(&abort.reached);
        println!(
            "reach cut: fired in the frame from t {:.6}: pair {:.6e} s reached {} points; cut to T_clear {until:.6e} s (kept {:.6e} s), critical {critical:.6e} s to T_must {first:.6e} s",
            state.time,
            abort.dt,
            abort.reached.len(),
            step.dt
        );
        return step;
    }
    count(1);
    if until >= abort.dt {
        println!(
            "reach cut: ALARM: the pair of {:.6e} s reached a plane the bound proved clear to its end",
            abort.dt
        );
    }
    take(state, settings)
}

thread_local! {
    static REACHED: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The points the cut pair reached (every outline point if none recorded).
fn critical_points(state: &TotalLagrangianState) -> Vec<usize> {
    let reached = REACHED.with(|r| std::mem::take(&mut *r.borrow_mut()));
    if reached.is_empty() {
        return state.outlines.iter().flatten().copied().collect();
    }
    reached
}

fn least_gap(state: &TotalLagrangianState, planes: &[Plane], q: usize) -> f64 {
    planes
        .iter()
        .map(|plane| plane.inward * (state.x[q][plane.axis] - plane.at))
        .fold(f64::INFINITY, f64::min)
}

/// The frame's counts, printed when the knob is on: reach cuts fired,
/// pairs redone as today (not provably clear), critical steps, watched
/// points reaching after a rejection (left as today), and steps taken
/// unwatched before the first grid exists; and, with
/// `EMERGE_PROBE_STATE_DIGEST` (knob on or off), the state's sums at full
/// precision, `impact_local`'s form with the deformation's added, so the
/// identity of criterion 2 is judged below the printed digits.
pub(super) fn report(state: &TotalLagrangianState, frame: u64) {
    super::handoff::theta_cells::report(state, frame);
    super::handoff::theta_cells::report_displacement(state, frame);
    if std::env::var_os("EMERGE_PROBE_STATE_DIGEST").is_some() {
        let (mut sx, mut sv, mut sf) = (0.0, 0.0, 0.0);
        for p in 0..state.x.len() {
            sx += state.x[p].x + state.x[p].y;
            sv += state.v[p].x + state.v[p].y;
            let f = state.f[p];
            sf += f.x_axis.x + f.x_axis.y + f.y_axis.x + f.y_axis.y;
        }
        println!("frame digest {frame}: sum x {sx:.17e} m, sum v {sv:.17e} m/s, sum F {sf:.17e}");
    }
    if !on() {
        return;
    }
    let [fired, redone, critical, gridless] = COUNTS.with(|c| c.replace([0; 4]));
    println!(
        "reach cut frame {frame}: fired {fired}, redone {redone}, critical {critical}, late {}, no grid {gridless}",
        hooks::take_reach_late()
    );
    stepped_phase::frame_end(state, frame);
}
