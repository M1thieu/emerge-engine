//! The exponential flight's trial measured (`LI_WU_TRIAL_PLAN.md`, section
//! 2 and criteria 1-8). Probes: measurement, nothing held.

use super::exponential_flight::{Exponential, Knob, Outcome, Record, set_knob};
use super::impact_local::{FINE, FRAME_S, FRAMES, asymmetry, clay, drop, sample};
use super::modal_flight::split;
use super::vibration::{Modes as Basis, modes};
use super::*;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

fn knob(truncated: bool, steps: usize) -> Option<Knob> {
    Some(Knob {
        on: true,
        truncated,
        steps,
        law: false,
    })
}

fn off() -> Option<Knob> {
    Some(Knob {
        on: false,
        truncated: false,
        steps: 1,
        law: false,
    })
}

/// The records against criteria 1, 3 and 5, one line each, and the
/// outcomes.
pub(super) fn summary(label: &str, log: &[Record]) {
    let flown: Vec<&Record> = log
        .iter()
        .filter(|r| r.outcome == Outcome::Flown && r.iterations > 0)
        .collect();
    let mut counts = std::collections::BTreeMap::new();
    for r in log {
        *counts.entry(format!("{:?}", r.outcome)).or_insert(0usize) += 1;
    }
    let within = flown
        .iter()
        .filter(|r| r.delta_h.abs() <= r.bound_h)
        .count();
    let zero = flown.iter().filter(|r| r.delta_h == 0.0).count();
    let mut ratios: Vec<f64> = flown
        .iter()
        .filter(|r| r.delta_h != 0.0)
        .map(|r| r.bound_h / r.delta_h.abs())
        .collect();
    ratios.sort_by(f64::total_cmp);
    let pick = |v: &[f64], q: f64| {
        if v.is_empty() {
            f64::NAN
        } else {
            v[((v.len() - 1) as f64 * q).round() as usize]
        }
    };
    let worst = flown
        .iter()
        .map(|r| r.delta_h.abs() / r.bound_h.max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max);
    println!(
        "{label} criterion 1: steps flown {}, |dH| <= B_H on {within} ({} not), dH exactly zero on {zero}; B_H / |dH| median {:.3e} (10 % {:.3e}, 90 % {:.3e}, largest {:.3e}); largest |dH| / B_H {worst:.3e}",
        flown.len(),
        flown.len() - within,
        pick(&ratios, 0.5),
        pick(&ratios, 0.1),
        pick(&ratios, 0.9),
        ratios.last().copied().unwrap_or(f64::NAN)
    );
    let parts = flown.iter().fold([0.0f64; 4], |mut a, r| {
        for (k, p) in r.parts.iter().enumerate() {
            a[k] = a[k].max(p / r.bound_h.max(f64::MIN_POSITIVE));
        }
        a
    });
    println!(
        "{label} B_H parts, largest share: stop {:.3}, discrete gradient {:.3}, update {:.3}, evaluation {:.3}",
        parts[0], parts[1], parts[2], parts[3]
    );
    let above = flown.iter().filter(|r| r.rho_obs > r.rho_pred).count();
    let max = |f: fn(&Record) -> f64| flown.iter().map(|r| f(r)).fold(0.0, f64::max);
    let mean =
        |f: fn(&Record) -> f64| flown.iter().map(|r| f(r)).sum::<f64>() / flown.len().max(1) as f64;
    println!(
        "{label} criterion 3: rho_obs > rho_pred on {above}; rho_pred largest {:.3e} mean {:.3e}; rho_obs largest {:.3e}; M2 largest {:.3e}; iterations mean {:.2} largest {}, K_max largest {}",
        max(|r| r.rho_pred),
        mean(|r| r.rho_pred),
        max(|r| r.rho_obs),
        max(|r| r.m2),
        mean(|r| f64::from(r.iterations)),
        flown.iter().map(|r| r.iterations).max().unwrap_or(0),
        flown.iter().map(|r| r.k_max).max().unwrap_or(0)
    );
    println!(
        "{label} criterion 5: flops per step mean {:.3e} largest {:.3e}; wall per step mean {:.3e} s largest {:.3e} s; spin at the steps' ends largest {:.3e} rad/s",
        mean(|r| r.flops),
        max(|r| r.flops),
        mean(|r| r.wall),
        max(|r| r.wall),
        max(|r| r.spin.abs())
    );
    println!("{label} outcomes: {counts:?}");
    for r in flown.iter().filter(|r| r.rho_obs > r.rho_pred).take(40) {
        println!(
            "{label} rho_obs {:.3e} > rho_pred {:.3e} at {:.6} s: deltas {:.3e} then {:.3e}, the noise floor {:.3e}",
            r.rho_obs, r.rho_pred, r.t0, r.ratio_deltas.0, r.ratio_deltas.1, r.noise_floor
        );
    }
    for r in log.iter().filter(|r| r.outcome != Outcome::Flown).take(40) {
        println!(
            "{label} not flown: {:?} from {:.6} to {:.6} s",
            r.outcome, r.t0, r.t1
        );
    }
    // `H_int` along each flight's steps, against its first: the measured
    // changes accumulated.
    let (mut drift, mut first) = (0.0f64, None::<(f64, f64)>);
    for r in &flown {
        let h0 = match first {
            Some((h0, start)) if start == r.start => h0,
            _ => r.h_int - r.delta_h,
        };
        drift = drift.max((r.h_int - h0).abs() / h0.abs().max(f64::MIN_POSITIVE));
        first = Some((h0, r.start));
    }
    println!("{label} H_int over each flight: largest accumulated change {drift:.3e} of its start");
    // FAST_MODE_PLAN.md sec. 26 (E-1): the step's energy change over each
    // flown attempt against the round-off bound of its two evaluations.
    if log.iter().any(|r| r.law) {
        let attempts: Vec<&Record> = log
            .iter()
            .filter(|r| r.law && r.outcome == Outcome::Flown && r.step_energy_bound > 0.0)
            .collect();
        let within = attempts
            .iter()
            .filter(|r| r.step_energy_change.abs() <= r.step_energy_bound)
            .count();
        let total: f64 = attempts.iter().map(|r| r.step_energy_change.abs()).sum();
        let mut ratios: Vec<f64> = attempts
            .iter()
            .map(|r| r.step_energy_change.abs() / r.step_energy_bound)
            .collect();
        ratios.sort_by(f64::total_cmp);
        let remainders: Vec<f64> = log.iter().filter_map(|r| r.start_remainder).collect();
        println!(
            "{label} E-1: flown attempts {}, within the round-off bound {within} ({} not); sum |dE| {total:.4e} J/m, largest |dE| {:.4e} J/m; |dE| / bound median {:.3e}, largest {:.3e}; start remainders {} (largest |.| {:.3e} J/m, sum |.| {:.3e} J/m); |P - I|_F largest {:.3e}",
            attempts.len(),
            attempts.len() - within,
            attempts
                .iter()
                .map(|r| r.step_energy_change.abs())
                .fold(0.0, f64::max),
            pick(&ratios, 0.5),
            ratios.last().copied().unwrap_or(f64::NAN),
            remainders.len(),
            remainders.iter().map(|e| e.abs()).fold(0.0, f64::max),
            remainders.iter().map(|e| e.abs()).sum::<f64>(),
            log.iter().map(|r| r.gram_defect).fold(0.0, f64::max)
        );
        println!("{label} A-8: {}", super::exponential_flight::proof_report());
        if std::env::var_os("EMERGE_PROBE_CLEARANCE_LEMMA").is_some() {
            for line in super::exponential_flight::lemma_report().lines() {
                println!("{label} {line}");
            }
        }
        for r in attempts
            .iter()
            .filter(|r| r.step_energy_change.abs() > r.step_energy_bound)
            .take(40)
        {
            // The parts, by count: the scheme's own `Delta H` over the
            // attempt (b), the start's projection remainder when the attempt
            // opens its flight (d), and what is left, the representation and
            // the rebuild (c, e).
            let same: Vec<&Record> = log
                .iter()
                .filter(|s| s.law && s.start == r.start && s.t0 >= r.t0 && s.t1 <= r.t1)
                .collect();
            let delta_h: f64 = same.iter().map(|s| s.delta_h).sum();
            let remainder: f64 = same.iter().filter_map(|s| s.start_remainder).sum();
            println!(
                "{label} E-1 beyond the bound at {:.6} to {:.6} s: dE {:+.4e} J/m, bound {:.4e} J/m; scheme dH {:+.4e}, start remainder {:+.4e}, left {:+.4e} J/m",
                r.t0,
                r.t1,
                r.step_energy_change,
                r.step_energy_bound,
                delta_h,
                remainder,
                r.step_energy_change - delta_h - remainder
            );
        }
    }
}

/// Every point's `(x, v, F)` of the path's state.
fn snapshot(sim: &Simulation) -> (Vec<DVec2>, Vec<DVec2>, Vec<DMat2>) {
    let s = sim.total_lagrangian.as_ref().expect("on");
    (s.x.clone(), s.v.clone(), s.f.clone())
}

/// Criterion 6: run D (the sub-frame handoff, tol 1e-2) with the trial off
/// and with its truncated model on, every frame's state compared bit for
/// bit; then from forced zero-spin starts (every velocity set to zero) the
/// truncated flight against today's `Flight`, 30 frames each.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_exponential_identical_digits() {
    super::cap::capped("probe_exponential_identical_digits", 7200, || {
        let tolerance = std::env::var("EMERGE_PROBE_TOLERANCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-2);
        let run = |k: Option<Knob>| {
            set_knob(k);
            let mut sim = drop(
                FRAME_S,
                TotalLagrangianConfig::new(tolerance).with_subframe_handoff(),
            );
            let mut out = Vec::new();
            for _ in 0..FRAMES {
                sim.step();
                let f = sim.total_lagrangian_frame().expect("on");
                out.push((
                    snapshot(&sim),
                    f.steps,
                    f.modal,
                    f.flown,
                    f.hand_backs,
                    f.modal_frames,
                ));
            }
            let log = sim
                .total_lagrangian
                .as_ref()
                .expect("on")
                .exponential_log
                .clone();
            set_knob(None);
            (out, log, sim)
        };
        let (today, _, sim) = run(off());
        let (trial, log, _) = run(knob(true, 1));
        let mut first = None;
        for (n, (a, b)) in today.iter().zip(&trial).enumerate() {
            let same = a.0 == b.0 && (a.1, a.2, a.3, a.4, a.5) == (b.1, b.2, b.3, b.4, b.5);
            if !same && first.is_none() {
                first = Some(n);
            }
        }
        let flown = today.iter().filter(|r| r.2).count();
        println!(
            "run D (tolerance {tolerance:e}), trial off against the truncated trial on: {} frames, {flown} flown whole, flown time {:.4} s; first frame differing: {first:?}; trial steps recorded {}",
            today.len(),
            today.iter().map(|r| r.3).sum::<f64>(),
            log.len()
        );
        summary("truncated run D", &log);
        // Forced zero-spin starts.
        let state = sim.total_lagrangian.as_ref().expect("on");
        let Some(Some(modes)) = state.modes.clone() else {
            panic!("the body's modes")
        };
        let gravity = sim.total_lagrangian_gravity();
        let mut differing = 0;
        let mut copy = state.clone();
        copy.v.iter_mut().for_each(|v| *v = DVec2::ZERO);
        let t0 = copy.time;
        let mut today_state = copy.clone();
        let mut trial_state = copy.clone();
        let mut old = super::flight::Flight::start(&copy, &modes, (t0, FRAME_S)).expect("starts");
        let mut new = Exponential::start(&copy, &modes, t0, (true, false)).expect("starts");
        for k in 1..=30 {
            let t = t0 + k as f64 * FRAME_S;
            assert!(old.rebuild(&modes, gravity, t, &mut today_state));
            new.step(t).expect("a truncated step");
            new.rebuild(&modes, gravity, &mut trial_state);
            if (&today_state.x, &today_state.v, &today_state.f)
                != (&trial_state.x, &trial_state.v, &trial_state.f)
            {
                differing += 1;
            }
        }
        println!("zero-spin start at t {t0:.4} s: 30 frames, {differing} differing bit for bit");
    });
}

/// The controller's energy norm of a state difference (`step_control`):
/// `1/2 sum m |dv|^2 + 1/2 sum V0 (2 mu |sym dF|^2 + lambda (tr dF)^2)`,
/// its square root, J/m.
fn distance(
    state: &TotalLagrangianState,
    a: &(Vec<DVec2>, Vec<DVec2>, Vec<DMat2>),
    b: &(Vec<DVec2>, Vec<DVec2>, Vec<DMat2>),
) -> f64 {
    let mut sum = 0.0;
    for p in 0..state.particles {
        let dv = a.1[p] - b.1[p];
        let df = a.2[p] - b.2[p];
        let sym = 0.5 * (df + df.transpose());
        let (mu, lambda) = (
            state.solids[p].mu,
            state.solids[p].bulk - state.solids[p].mu,
        );
        let tr = df.x_axis.x + df.y_axis.y;
        let sym2 = sym.x_axis.length_squared() + sym.y_axis.length_squared();
        sum += 0.5 * state.mass[p] * dv.length_squared()
            + 0.5 * state.v0[p] * (2.0 * mu * sym2 + lambda * tr * tr);
    }
    sum.sqrt()
}

/// E2's clay block in free space (no gravity, far from every wall), spun at
/// `spin` rad/s about its centre and stretched along its long axis at 1/s.
fn spun(frame_s: f64, path: TotalLagrangianConfig, spin: f64) -> Simulation {
    let mut config = SimConfig::earth(40, 0.01, frame_s as f32);
    config.min_dt = config.min_dt.min(frame_s as f32);
    config.gravity = Vec2::ZERO;
    config.total_lagrangian = Some(path);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, 20.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&clay(), &config),
    );
    let units = GridUnits::of(&config);
    let n = sim.particles.len();
    let centre = (0..n)
        .map(|p| units.position(sim.particles.x[p]))
        .sum::<DVec2>()
        / n as f64;
    for p in 0..n {
        let arm = units.position(sim.particles.x[p]) - centre;
        let v = spin * arm.perp() + DVec2::new(arm.x, 0.0);
        sim.particles.v[p] = units.velocity_to_grid(v);
    }
    sim
}

/// 2.2 and criterion 4: the spun block at `EMERGE_PROBE_SPIN` rad/s (10 by
/// default), 31 frames (the first stepped): A, B, V, V_fine and V at 2, 4, 8
/// steps a frame, each frame's distance in the controller's norm, and per
/// mode the amplitude and phase of V against V_fine.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_exponential_spun_block() {
    super::cap::capped("probe_exponential_spun_block", 14400, || {
        let spin: f64 = std::env::var("EMERGE_PROBE_SPIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0);
        const COUNT: usize = 31;
        type Snap = (Vec<DVec2>, Vec<DVec2>, Vec<DMat2>);
        // `seed`: after the first (stepped) frame the state is replaced by
        // A's at that frame (a diagnostic: a common start, not criterion 4).
        let run_from = |frame: f64,
                        path: TotalLagrangianConfig,
                        per: usize,
                        k: Option<Knob>,
                        seed: Option<&Snap>| {
            set_knob(k);
            let clock = std::time::Instant::now();
            let mut sim = spun(frame, path, spin);
            let mut out = Vec::new();
            let mut steps = 0u32;
            for n in 0..COUNT {
                for _ in 0..per {
                    sim.step();
                    steps += sim.total_lagrangian_frame().expect("on").steps;
                }
                if n == 0
                    && let Some(seed) = seed
                {
                    let state = sim.total_lagrangian.as_mut().expect("on");
                    (state.x, state.v, state.f) = seed.clone();
                }
                out.push(snapshot(&sim));
            }
            let wall = clock.elapsed().as_secs_f64();
            let log = sim
                .total_lagrangian
                .as_ref()
                .expect("on")
                .exponential_log
                .clone();
            set_knob(None);
            (out, log, sim, steps, wall)
        };
        let run = |frame: f64, path: TotalLagrangianConfig, per: usize, k: Option<Knob>| {
            run_from(frame, path, per, k, None)
        };
        let a = run(
            FRAME_S / FINE as f64,
            TotalLagrangianConfig::new(f64::INFINITY),
            FINE,
            off(),
        );
        let b = run(FRAME_S, TotalLagrangianConfig::new(1e-2), 1, off());
        let modal = TotalLagrangianConfig::new(1e-2).with_modal_flight();
        let v = run(FRAME_S, modal, 1, knob(false, 1));
        let fine = run(FRAME_S, modal, 1, knob(false, FINE));
        let state = a.2.total_lagrangian.as_ref().expect("on");
        println!(
            "spin {spin} rad/s: walls A {:.1} s ({} steps), B {:.1} s ({} steps), V {:.3} s ({} steps), V_fine {:.1} s",
            a.4, a.3, b.4, b.3, v.4, v.3, fine.4
        );
        summary(&format!("spin {spin} V"), &v.1);
        summary(&format!("spin {spin} V_fine"), &fine.1);
        let mut failing = 0;
        for n in 0..COUNT {
            let (va, vb, vv, vf) = (
                distance(state, &v.0[n], &a.0[n]),
                distance(state, &b.0[n], &a.0[n]),
                distance(state, &v.0[n], &fine.0[n]),
                distance(state, &fine.0[n], &a.0[n]),
            );
            failing += usize::from(va > vb);
            println!(
                "spin {spin} frame {n}: |V - A| {va:.4e}, |B - A| {vb:.4e} J^(1/2)/m^(1/2) (criterion 4 {}); |V - V_fine| {vv:.4e}; |V_fine - A| {vf:.4e}",
                if va <= vb { "met" } else { "NOT met" }
            );
        }
        println!("spin {spin}: criterion 4 not met on {failing} of {COUNT} frames");
        for s in [2usize, 4, 8] {
            let r = run(FRAME_S, modal, 1, knob(false, s));
            let last = distance(state, &r.0[COUNT - 1], &fine.0[COUNT - 1]);
            let worst = (0..COUNT)
                .map(|n| distance(state, &r.0[n], &fine.0[n]))
                .fold(0.0, f64::max);
            println!(
                "spin {spin} order: {s} steps a frame, |V - V_fine| at the end {last:.4e}, largest {worst:.4e}"
            );
        }
        let one = (0..COUNT)
            .map(|n| distance(state, &v.0[n], &fine.0[n]))
            .fold(0.0, f64::max);
        println!("spin {spin} order: 1 step a frame, |V - V_fine| largest {one:.4e}");
        // Diagnostics beyond the plan's runs: a finer reference (omega h at
        // most 736 / 8192 = 0.09 a step) for the order, and B and V from A's
        // frame-0 state (a common start).
        let resolved = run(FRAME_S, modal, 1, knob(false, 8192));
        for (s, r) in [(1usize, &v), (FINE, &fine)] {
            let worst = (0..COUNT)
                .map(|n| distance(state, &r.0[n], &resolved.0[n]))
                .fold(0.0, f64::max);
            println!(
                "spin {spin} diagnostic order: {s} steps a frame against 8192, largest {worst:.4e}"
            );
        }
        for s in [2usize, 4, 8, 64] {
            let r = run(FRAME_S, modal, 1, knob(false, s));
            let worst = (0..COUNT)
                .map(|n| distance(state, &r.0[n], &resolved.0[n]))
                .fold(0.0, f64::max);
            println!(
                "spin {spin} diagnostic order: {s} steps a frame against 8192, largest {worst:.4e}"
            );
        }
        let seed = a.0[0].clone();
        let b0 = run_from(
            FRAME_S,
            TotalLagrangianConfig::new(1e-2),
            1,
            off(),
            Some(&seed),
        );
        let v0 = run_from(FRAME_S, modal, 1, knob(false, 1), Some(&seed));
        let r0 = run_from(FRAME_S, modal, 1, knob(false, 8192), Some(&seed));
        let mut meets = 0;
        for n in 1..COUNT {
            let (va, vb, vr) = (
                distance(state, &v0.0[n], &a.0[n]),
                distance(state, &b0.0[n], &a.0[n]),
                distance(state, &r0.0[n], &a.0[n]),
            );
            meets += usize::from(va <= vb);
            println!(
                "spin {spin} diagnostic common start frame {n}: |V - A| {va:.4e}, |B - A| {vb:.4e}, |V_8192 - A| (the modes' linearisation) {vr:.4e}"
            );
        }
        println!(
            "spin {spin} diagnostic common start: |V - A| <= |B - A| on {meets} of {}",
            COUNT - 1
        );
        // Per mode at the last frame: amplitude and phase of V against V_fine.
        let basis: Basis = modes(state);
        let at = |snap: &(Vec<DVec2>, Vec<DVec2>, Vec<DMat2>), sim: &Simulation| {
            let mut copy = sim.total_lagrangian.as_ref().expect("on").clone();
            (copy.x, copy.v, copy.f) = snap.clone();
            split(&copy, &basis, 0.0).0.z
        };
        let (zv, zf) = (at(&v.0[COUNT - 1], &v.2), at(&fine.0[COUNT - 1], &fine.2));
        let h = FRAME_S;
        let mut rows: Vec<(f64, f64, f64, f64)> = (super::vibration::RIGID..basis.omega.len())
            .filter(|&i| basis.omega[i] > 0.0)
            .map(|i| {
                let (a, b) = (zv[i], zf[i]);
                let amplitude = (a.length() - b.length()) / b.length().max(f64::MIN_POSITIVE);
                let phase = (a.perp_dot(b)).atan2(a.dot(b));
                (basis.omega[i] * h, b.length(), amplitude, phase)
            })
            .collect();
        rows.sort_by(|x, y| x.0.total_cmp(&y.0));
        let largest = rows.iter().map(|r| r.1).fold(0.0, f64::max);
        for (wh, size, amplitude, phase) in rows.into_iter().filter(|r| r.1 >= 1e-6 * largest) {
            println!(
                "spin {spin} mode omega h {wh:7.2}: |z| (V_fine) {size:.3e}, amplitude error {amplitude:+.3e}, phase error {phase:+.3e} rad"
            );
        }
    });
}

/// Criteria 1, 2, 3, 5 and 7 on run D (`EMERGE_PROBE_TOLERANCE`, 1e-2 by
/// default; 1e-1 is criterion 8's E2), the per-frame lines in
/// `probe_floor_candidates`'s format for the comparison against reference
/// A's log, then the records' summary.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_exponential_run_d() {
    super::cap::capped("probe_exponential_run_d", 7200, || {
        let tolerance = std::env::var("EMERGE_PROBE_TOLERANCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-2);
        let label = format!("exponential_{tolerance:e}");
        set_knob(knob(false, 1));
        let clock = std::time::Instant::now();
        let mut sim = drop(
            FRAME_S,
            TotalLagrangianConfig::new(tolerance).with_subframe_handoff(),
        );
        let mut basis: Option<Basis> = None;
        let mut start = None;
        for n in 0..FRAMES {
            let frame_clock = std::time::Instant::now();
            sim.step();
            let wall = frame_clock.elapsed().as_secs_f64();
            let f = sim.total_lagrangian_frame().expect("on");
            let state = sim.total_lagrangian.as_ref().expect("on");
            let b = basis.get_or_insert_with(|| modes(state));
            let (spin, odd, share) = asymmetry(state);
            let (height, vy, vibration, energy) = sample(&sim, b);
            let start = *start.get_or_insert(energy);
            println!(
                "{label} frame {n}: steps {} rejected {} floor {} contact {} modal {} flown {:.3e} s wall {wall:.3} s; height {height:.6e} m, vy {vy:+.6e} m/s, vibration {vibration:.4e} J/m, energy {:+.3e} of the start; spin {spin:+.3e} rad/s, odd vibration {odd:.3e} J/m ({share:.3e} of it); hand-backs {} modal frames {}",
                f.steps,
                f.rejected,
                f.at_floor,
                f.contacts > 0,
                f.modal,
                f.flown,
                energy / start - 1.0,
                f.hand_backs,
                f.modal_frames
            );
        }
        println!("{label} run wall {:.1} s", clock.elapsed().as_secs_f64());
        let log = sim
            .total_lagrangian
            .as_ref()
            .expect("on")
            .exponential_log
            .clone();
        set_knob(None);
        summary(&label, &log);
    });
}

/// Criterion 3's diagnostic: the spun block's V_fine alone (`EMERGE_PROBE_SPIN`,
/// 10 by default), each step whose observed ratio exceeds `rho_pred` with
/// the deltas it compares and the noise floor.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_exponential_ratio_floor() {
    super::cap::capped("probe_exponential_ratio_floor", 3600, || {
        let spin: f64 = std::env::var("EMERGE_PROBE_SPIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0);
        set_knob(knob(false, FINE));
        let mut sim = spun(
            FRAME_S,
            TotalLagrangianConfig::new(1e-2).with_modal_flight(),
            spin,
        );
        for _ in 0..31 {
            sim.step();
        }
        let log = sim
            .total_lagrangian
            .as_ref()
            .expect("on")
            .exponential_log
            .clone();
        set_knob(None);
        summary(&format!("spin {spin} V_fine"), &log);
    });
}
