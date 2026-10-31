//! S1PP_STEPPED_PLAN.md (test builds, `EMERGE_PROBE_STEPPED_CONTACT_STEP=k`
//! with `EMERGE_PROBE_REACH_CUT`): the fixed contact step on the stepped
//! path. A body's contact phase starts at the reach cut's firing for it and
//! ends when the reach cut's window proves the body clear for one fixed
//! half-step `k h / c`, or at a frame's end when its held contact keys were
//! the same at every kept step's end of the frame as at its start (the
//! contact set unchanged over a whole frame, `rest::rest_frame`'s rule per
//! body). While any body is in its phase the shared step is `2 k h / c`, its
//! halves of `k h / c` kept (`handoff::contact_phase_control`'s
//! construction), `c` the fastest P wave of the phase bodies.

use std::cell::{Cell, RefCell};

use glam::DVec2;

use super::TotalLagrangianState;
use crate::spacetime::total_lagrangian::{ContactKey, Plane, StepControl};

/// A body: its particles and its outline points.
type Body = (Vec<usize>, Vec<usize>);

/// A body's clear window from a state, the walls and a time span.
type Window = dyn Fn(&TotalLagrangianState, &Body, (&[Plane], DVec2), (f64, f64)) -> (f64, f64);

#[derive(Default, Clone)]
struct Phase {
    on: bool,
    /// The held keys at the frame's start and whether they changed at any
    /// kept step's end since.
    keys: Vec<ContactKey>,
    changed: bool,
    /// Whether the body was free at the last kept step's end.
    was_free: bool,
}

thread_local! {
    static BODIES: RefCell<Vec<Body>> = const { RefCell::new(Vec::new()) };
    static PHASES: RefCell<Vec<Phase>> = const { RefCell::new(Vec::new()) };
    /// Per frame: fixed steps, phases started, ended clear, ended at rest,
    /// contacts begun outside a phase ("unphased"), fixed steps cut to the
    /// frame's remainder, rejected tries inside a phase.
    static COUNTS: Cell<[u32; 7]> = const { Cell::new([0; 7]) };
    /// Per frame: the shortest step kept while a phase was on, s.
    static SHORTEST: Cell<f64> = const { Cell::new(f64::INFINITY) };
    /// Whether the step being taken is a phase's fixed step.
    static FIXED: Cell<bool> = const { Cell::new(false) };
}

/// `k`, from `EMERGE_PROBE_STEPPED_CONTACT_STEP`; says so once, from inside.
pub(super) fn k() -> Option<f64> {
    static K: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let k = *K.get_or_init(|| {
        std::env::var("EMERGE_PROBE_STEPPED_CONTACT_STEP")
            .ok()
            .and_then(|v| v.parse().ok())
    });
    if let Some(k) = k {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            println!("knob active: stepped contact step, k {k} (S1PP_STEPPED_PLAN.md)")
        });
    }
    k
}

fn count(which: usize) {
    COUNTS.with(|c| {
        let mut n = c.get();
        n[which] += 1;
        c.set(n);
    });
}

/// The bodies, once (the reach cut's own partition).
pub(super) fn set_bodies(bodies: impl FnOnce() -> Vec<Body>) {
    BODIES.with(|b| {
        if b.borrow().is_empty() {
            let all = bodies();
            PHASES.with(|p| *p.borrow_mut() = vec![Phase::default(); all.len()]);
            *b.borrow_mut() = all;
        }
    });
}

/// Whether body `b` is in its contact phase.
pub(super) fn in_phase(b: usize) -> bool {
    PHASES.with(|p| p.borrow().get(b).is_some_and(|ph| ph.on))
}

/// The reach cut fired: the bodies of the reached points start their phase.
pub(super) fn start(reached: &[usize]) {
    BODIES.with(|b| {
        PHASES.with(|p| {
            let mut phases = p.borrow_mut();
            for (i, (particles, points)) in b.borrow().iter().enumerate() {
                let hit = reached
                    .iter()
                    .any(|q| particles.contains(q) || points.contains(q));
                if hit && !phases[i].on {
                    phases[i].on = true;
                    count(1);
                }
            }
        })
    });
}

/// The fixed step while a phase is on: the control and the step length set
/// in `state`; `None` when no body is in its phase.
pub(crate) fn control(
    control: &StepControl,
    state: &mut TotalLagrangianState,
    remaining: f64,
) -> Option<StepControl> {
    let k = k()?;
    let cell = state.cache.as_ref()?.grid.cell;
    let c = BODIES.with(|b| {
        PHASES.with(|p| {
            let phases = p.borrow();
            b.borrow()
                .iter()
                .zip(phases.iter())
                .filter(|(_, ph)| ph.on)
                .flat_map(|(body, _)| body.0.iter().copied())
                .filter(|&q| state.v0[q] > 0.0)
                .map(|q| {
                    ((state.solids[q].bulk + state.solids[q].mu) * state.v0[q] / state.mass[q])
                        .sqrt()
                })
                .fold(0.0, f64::max)
        })
    });
    if c <= 0.0 {
        return None;
    }
    let fixed = 2.0 * k * cell / c;
    if fixed > remaining {
        count(5);
    }
    state.step_length = fixed.min(remaining);
    count(0);
    FIXED.with(|f| f.set(true));
    Some(StepControl {
        tolerance: f64::INFINITY,
        ..*control
    })
}

/// The fixed half-step `k h / c` of the phase bodies (for the clear end).
fn half_step(state: &TotalLagrangianState, body: &Body) -> Option<f64> {
    let k = k()?;
    let cell = state.cache.as_ref()?.grid.cell;
    let c = body
        .0
        .iter()
        .filter(|&&q| state.v0[q] > 0.0)
        .map(|&q| {
            ((state.solids[q].bulk + state.solids[q].mu) * state.v0[q] / state.mass[q]).sqrt()
        })
        .fold(0.0, f64::max);
    (c > 0.0).then(|| k * cell / c)
}

/// The keys of the contacts held on a body.
fn held_keys(state: &TotalLagrangianState, body: &Body) -> Vec<ContactKey> {
    let (particles, points) = body;
    let mut keys: Vec<ContactKey> = state
        .cache
        .as_ref()
        .map(|c| {
            c.contacts
                .iter()
                .filter(|contact| {
                    contact.points.iter().any(|(point, _)| {
                        particles.contains(&point.particle) || points.contains(&point.particle)
                    })
                })
                .map(|contact| contact.key)
                .collect()
        })
        .unwrap_or_default();
    keys.sort_unstable();
    keys
}

/// After each kept step: a free phase body proved clear for one fixed
/// half-step ends its phase; a body that begins a contact outside a phase is
/// counted; each phase body's key set is followed for the rest end.
pub(super) fn after_step(
    state: &TotalLagrangianState,
    walls: (&[Plane], DVec2),
    (tolerance, dt, rejected): (f64, f64, u32),
    is_free: &dyn Fn(&TotalLagrangianState, &Body) -> bool,
    window: &Window,
) {
    if k().is_none() || state.cache.is_none() {
        return;
    }
    if FIXED.with(|f| f.replace(false)) {
        SHORTEST.with(|s| s.set(s.get().min(dt)));
        for _ in 0..rejected {
            count(6);
        }
    }
    BODIES.with(|b| {
        PHASES.with(|p| {
            let mut phases = p.borrow_mut();
            for (body, phase) in b.borrow().iter().zip(phases.iter_mut()) {
                let free = is_free(state, body);
                if phase.on {
                    if held_keys(state, body) != phase.keys {
                        phase.changed = true;
                    }
                    let trace = std::env::var_os("EMERGE_PROBE_STEPPED_TRACE").is_some();
                    if trace && !free {
                        println!(
                            "stepped trace: not free, held keys {}",
                            held_keys(state, body).len()
                        );
                    }
                    if free && let Some(half) = half_step(state, body) {
                        let (until, _) = window(state, body, walls, (half, tolerance));
                        if trace {
                            println!(
                                "stepped trace: free, clear for {until:.4e} s of {half:.4e} s"
                            );
                        }
                        // The window's full interval, `(t0 + half) - t0`, the
                        // same rounded quantity it returns when clear to its end.
                        if until >= (state.time + half) - state.time {
                            phase.on = false;
                            count(2);
                        }
                    }
                } else if phase.was_free && !free {
                    count(4);
                }
                phase.was_free = free;
            }
        })
    });
}

/// At a frame's end: the rest end, the counts and the limits printed.
pub(super) fn frame_end(state: &TotalLagrangianState, frame: u64) {
    if k().is_none() {
        return;
    }
    BODIES.with(|b| {
        PHASES.with(|p| {
            let mut phases = p.borrow_mut();
            for (body, phase) in b.borrow().iter().zip(phases.iter_mut()) {
                let keys = held_keys(state, body);
                if phase.on && !phase.changed && keys == phase.keys && !keys.is_empty() {
                    phase.on = false;
                    count(3);
                }
                phase.keys = keys;
                phase.changed = false;
            }
        })
    });
    let on = PHASES.with(|p| p.borrow().iter().filter(|ph| ph.on).count());
    let [fixed, started, clear, rest, unphased, remainder, rejected] =
        COUNTS.with(|c| c.replace([0; 7]));
    let shortest = SHORTEST.with(|s| s.replace(f64::INFINITY));
    println!(
        "stepped phase frame {frame}: fixed steps {fixed} (cut to the frame's remainder {remainder}, rejected tries {rejected}, shortest kept {shortest:.4e} s), phases started {started}, ended clear {clear}, ended at rest {rest}, unphased contacts {unphased}, in phase at the end {on}; limits: momentum observables only, plastic bodies information only, spun starts reproducibility only, one shared step for all bodies"
    );
}
