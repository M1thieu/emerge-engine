//! The chatter's pushes classified by their signature (`tmp/papers/
//! dl_realtime/CHATTER_TRACE_PLANS.md`, M1), from the contact trace of a
//! frame's kept steps (`hooks::ContactRow`). A probe: measurement, nothing
//! held.

use crate::spacetime::total_lagrangian::hooks::TracedStep;

/// Counts M1's signatures over the kept steps of one frame and prints them.
/// A held contact-step has a positive multiplier. (a) Puso's flip: approaching
/// at the start, `G v^n < 0`, and `|G v^{n+1} + G v^n| <= 2 a / dt`, `a` the
/// gap allowance it was held to (the plan names `gap_resolution`, the
/// allowance's resolution term: counted against both). (b) A flip released
/// separating: (a), its key not held at the next kept step, `G v^{n+1} > 0`.
/// A push is a run of kept steps holding at least one contact; the contact
/// phase runs from the first such step to the last.
pub(super) fn report_m1(label: &str, steps: &[TracedStep]) {
    let held = |k: usize| -> Vec<_> {
        steps[k]
            .1
            .iter()
            .filter(|r| r.multiplier > 0.0)
            .cloned()
            .collect()
    };
    let holding: Vec<bool> = (0..steps.len()).map(|k| !held(k).is_empty()).collect();
    let (Some(first), Some(last)) = (
        holding.iter().position(|&h| h),
        holding.iter().rposition(|&h| h),
    ) else {
        println!("{label} M1: no held contact in {} kept steps", steps.len());
        return;
    };
    let mut flips = [0u32; 2];
    let mut contact_steps = 0u32;
    let mut releases = 0u32;
    let mut held_on = 0u32;
    // Per push: its length in steps, whether every held contact-step in it
    // is a flip (by the allowance), and whether it ends with every contact
    // of its last step released separating.
    let mut pushes: Vec<(usize, bool, bool)> = Vec::new();
    let mut k = first;
    while k <= last {
        if !holding[k] {
            k += 1;
            continue;
        }
        let begin = k;
        let mut all_flips = true;
        while k <= last && holding[k] {
            let dt = steps[k].0;
            let next: Vec<_> = if k + 1 < steps.len() {
                held(k + 1)
            } else {
                Vec::new()
            };
            let mut ends_released = true;
            for r in held(k) {
                contact_steps += 1;
                let approaching = r.start < 0.0;
                let sum = (r.end + r.start).abs();
                let by_allowance = approaching && sum <= 2.0 * r.allowance / dt;
                let by_resolution = approaching && sum <= 2.0 * r.resolution / dt;
                flips[0] += u32::from(by_allowance);
                flips[1] += u32::from(by_resolution);
                all_flips &= by_allowance;
                let kept_on = next.iter().any(|q| q.key == r.key);
                if kept_on {
                    held_on += 1;
                    ends_released = false;
                } else if by_allowance && r.end > 0.0 {
                    releases += 1;
                } else {
                    ends_released = false;
                }
            }
            k += 1;
            if k > last || !holding[k] {
                pushes.push((k - begin, all_flips, ends_released));
            }
        }
    }
    let one_then_release = pushes
        .iter()
        .filter(|&&(n, flip, released)| n == 1 && flip && released)
        .count();
    let run_then_release = pushes
        .iter()
        .filter(|&&(_, flip, released)| flip && released)
        .count();
    let elapsed: f64 = steps[first..=last].iter().map(|s| s.0).sum();
    println!(
        "{label} M1: contact phase {elapsed:.6e} s over kept steps {first}..={last} of {}; pushes {}; held contact-steps {contact_steps}: (a) flips by the allowance {}, by the resolution term {}; (b) flips released separating {releases}; contact-steps held on at the next step {held_on}",
        steps.len(),
        pushes.len(),
        flips[0],
        flips[1],
    );
    println!(
        "{label} M1: pushes that are one (a) then (b): {one_then_release} of {}; a run of (a) ended by (b): {run_then_release}; push lengths in steps: {:?}",
        pushes.len(),
        pushes.iter().map(|p| p.0).collect::<Vec<_>>()
    );
    // The pushes' and the free runs' durations (CHATTER_STUDY.md's
    // counting: a push is a run of steps holding a contact, a free run a
    // run of steps holding none, inside the contact phase).
    let mut runs: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut k = first;
    while k <= last {
        let class = usize::from(holding[k]);
        let mut duration = 0.0;
        while k <= last && usize::from(holding[k]) == class {
            duration += steps[k].0;
            k += 1;
        }
        runs[class].push(duration);
    }
    let median = |v: &[f64]| -> f64 {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    };
    println!(
        "{label} M1: pushes {}, median push run {:.3e} s; free runs {}, median free run {:.3e} s",
        runs[1].len(),
        median(&runs[1]),
        runs[0].len(),
        median(&runs[0])
    );
    // Every held contact-step, for the log: step, its length, key,
    // multiplier, G v at the start and end, both tolerances in velocity.
    for (k, (dt, rows)) in steps.iter().enumerate().take(last + 1).skip(first) {
        for r in rows.iter().filter(|r| r.multiplier > 0.0) {
            println!(
                "{label} M1 row: step {k} dt {dt:.3e} key {:?} lambda {:.3e} Gv start {:+.4e} end {:+.4e} sum {:+.3e} 2a/dt {:.3e} 2res/dt {:.3e}",
                r.key,
                r.multiplier,
                r.start,
                r.end,
                r.end + r.start,
                2.0 * r.allowance / dt,
                2.0 * r.resolution / dt
            );
        }
    }
}

/// M2: the vibration a force history alone gives. Reference A (fixed steps
/// of `1/60/315` s, infinite tolerance) runs to frame 7's end; its modal
/// state there (`modal_flight::split`) is then driven, kept step by kept
/// step through frames 8 and 9, by the contact pushes A applies, each mode
/// an undamped oscillator under a force constant over the step. With `z =
/// omega q - i q'` (`vibration::z`), `q'' + omega^2 q = f` reads `z' = i
/// omega z - i f`, so `z_1 = f / omega + e^(i omega h) (z_0 - f / omega)`
/// (derived here; the plan wrote the same increment for the opposite sign
/// convention of `z`). Adaptations, declared: the forces are expressed in
/// the body frame of frame 7's end (no rotation over the two frames), the
/// modes are linear about the reference (A is corotated), and the outline
/// points take their mode values as the particles do (`Modes::at`).
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_force_history_vibration() {
    use super::super::impact_local::{FINE, FRAME_S, drop};
    use super::super::modal_flight::split;
    use super::super::vibration::{RIGID, modes};
    use super::*;
    super::super::cap::capped("probe_force_history_vibration", 4 * 3600, || {
        // The linearisation's measured size: run D's flown frames' modal
        // energy change, J/m (the plan's yardstick, recorded alongside).
        let mut d = drop(
            FRAME_S,
            TotalLagrangianConfig::new(1e-2).with_subframe_handoff(),
        );
        let mut flown_change = 0.0f64;
        for n in 0..8 {
            d.step();
            let f = d.total_lagrangian_frame().expect("on");
            if f.modal {
                flown_change = flown_change.max(f.modal_energy_change.abs());
            }
            println!(
                "M2 run D frame {n}: modal {} flown {:.3e} s, modal energy change {:+.3e} J/m, contacts {}",
                f.modal, f.flown, f.modal_energy_change, f.contacts
            );
        }
        let mut a = drop(
            FRAME_S / FINE as f64,
            TotalLagrangianConfig::new(f64::INFINITY),
        );
        let measure = |sim: &Simulation, basis: &super::super::vibration::Modes| {
            let state = sim.total_lagrangian.as_ref().expect("on");
            let n = state.particles;
            let mass: f64 = state.mass[..n].iter().sum();
            let vy = (0..n).map(|p| state.mass[p] * state.v[p].y).sum::<f64>() / mass;
            let (flight, _) = split(state, basis, 0.0);
            let vibration: f64 = flight.z[RIGID..]
                .iter()
                .map(|z| 0.5 * z.length_squared())
                .sum();
            (flight, vibration, vy, mass)
        };
        for frame in 0..8 {
            let mut contact = false;
            for _ in 0..FINE {
                a.step();
                contact |= a.total_lagrangian_frame().expect("on").contacts > 0;
            }
            println!("M2 A frame {frame}: contact {contact}");
        }
        let basis = &modes(a.total_lagrangian.as_ref().expect("on"));
        let (start, start_vibration, start_vy, mass) = measure(&a, basis);
        let back = glam::DMat2::from_angle(-start.angle);
        let mut z = start.z.clone();
        let mut impulse = DVec2::ZERO;
        let mut elapsed = 0.0;
        let gravity = a.total_lagrangian_gravity();
        println!("M2 frame 7 end: vibration {start_vibration:.6e} J/m, vy {start_vy:+.6e} m/s");
        for frame in 8..10 {
            hooks::watch_contact_rows(true);
            for _ in 0..FINE {
                a.step();
            }
            let steps = hooks::take_contact_rows();
            hooks::watch_contact_rows(false);
            for (h, rows) in &steps {
                let mut force = vec![DVec2::ZERO; basis.at[0].len()];
                for r in rows {
                    for &(p, push) in &r.push {
                        force[p] += push;
                    }
                }
                impulse += *h * force.iter().copied().sum::<DVec2>();
                elapsed += h;
                for ((zi, &omega), at) in z.iter_mut().zip(&basis.omega).zip(&basis.at).skip(RIGID)
                {
                    let f: f64 = force
                        .iter()
                        .zip(at)
                        .map(|(&g, &at)| (back * g).dot(at))
                        .sum();
                    let fixed = DVec2::new(f / omega, 0.0);
                    let (c, s) = ((omega * h).cos(), (omega * h).sin());
                    let w = *zi - fixed;
                    *zi = fixed + DVec2::new(c * w.x - s * w.y, s * w.x + c * w.y);
                }
            }
            let predicted: f64 = z[RIGID..].iter().map(|z| 0.5 * z.length_squared()).sum();
            let (_, vibration, vy, _) = measure(&a, basis);
            let rebound = start_vy + impulse.y / mass + gravity.y * elapsed;
            let off = (predicted / vibration - 1.0).abs();
            println!(
                "M2 frame {frame} end: kept steps {}, A's vibration {vibration:.6e} J/m, predicted {predicted:.6e} J/m, |pred / A - 1| {off:.4e}; A's vy {vy:+.6e} m/s, from the impulse {rebound:+.6e} m/s; flown frames' largest modal energy change {flown_change:.3e} J/m, {:.4e} of A's vibration",
                steps.len(),
                flown_change / vibration.max(f64::MIN_POSITIVE)
            );
        }
    });
}
