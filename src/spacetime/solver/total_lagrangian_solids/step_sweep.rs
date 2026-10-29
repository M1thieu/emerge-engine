//! What a fixed step buys on the path: the clay drop of
//! `tests/realtime_profile.rs` (E2) stepped at `k h/c`, `k` in 1 to 64, its
//! contact observables against the `k = 1` run, and the share of its
//! energy in the modes the step does not resolve. A probe: measurement,
//! nothing held.
//!
//! The step is fixed without changing the controller: with an infinite
//! tolerance every try is accepted, the controller keeps its two half
//! steps and caps the next at the frame, so frames of `2 k h/c` keep one
//! pair of steps `k h/c` each (the discarded whole try is a cost, not a
//! different step). A Newton failure still halves; it is counted. `h` is
//! the cell, `c` the clay's P wave speed `sqrt((lambda + 2 mu) / rho)`.
//!
//! The unresolved band: the body's free elastic modes at its reference
//! configuration (`vibration::modes`, modal flight's basis), the state
//! projected as `modal_flight::split` does, mode `i`'s energy `|z_i|^2 /
//! 2`; a mode is unresolved when `omega dt > 2`, where the midpoint rule's
//! frequency `(2/dt) atan(omega dt / 2)` is 0.785 of the true one.
//! Adaptation, declared: the contact constraints and the current tangent
//! are not in that basis.

use super::modal_flight::split;
use super::vibration::{Modes, modes};
use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

const CELL_M: f32 = 0.01;
/// The run's length, s: the E2 scene's 45 frames of 1/60 s.
const T_S: f64 = 0.75;

fn clay() -> ElasticProps {
    ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    }
}

/// The clay's P wave speed, m/s.
fn p_wave_speed() -> f64 {
    let p = clay();
    let (e, nu, rho) = (f64::from(p.e_pa), f64::from(p.nu), f64::from(p.rho_kg_m3));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    ((lambda + 2.0 * mu) / rho).sqrt()
}

/// E2's block, frames of `frame_s`, the controller's tolerance infinite.
fn drop(frame_s: f32) -> Simulation {
    let mut config = SimConfig::earth(40, CELL_M, frame_s);
    config.min_dt = config.min_dt.min(frame_s);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(f64::INFINITY));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 12.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&clay(), &config),
    );
    sim
}

/// One run's observables and costs.
#[derive(Debug, Clone, Copy)]
struct Observed {
    impact_s: f64,
    rebound_m_s: f64,
    peak_force_n_m: f64,
    contact_s: f64,
    height_m: f64,
    worst_balance: f64,
    newton_per_step: f64,
    gmres_per_step: f64,
    halvings: u32,
    /// The fast band's share of the vibration energy and of the kinetic
    /// plus stored energy: largest and mean over the frames, at `omega dt >
    /// 2`, then at `omega dt > pi/2`.
    fast_of_vibration: [f64; 2],
    fast_of_total: [f64; 2],
    fast_of_vibration_quarter: [f64; 2],
    wall_s_per_frame: f64,
}

/// The centre of mass's height above the floor (m), vertical velocity
/// (m/s) and the mass (kg/m), from the path's `f64` state.
fn centre(sim: &Simulation, floor_m: f64) -> (f64, f64, f64) {
    let state = sim.total_lagrangian.as_ref().expect("on");
    let n = state.particles;
    let mass: f64 = state.mass[..n].iter().sum();
    let (y, vy) = (0..n).fold((0.0, 0.0), |(y, vy), p| {
        (
            y + state.mass[p] * state.x[p].y,
            vy + state.mass[p] * state.v[p].y,
        )
    });
    (y / mass - floor_m, vy / mass, mass)
}

/// The fast band's energy over the vibration energy and over the kinetic
/// plus stored energy, at `omega dt > threshold`.
fn fast_share(state: &TotalLagrangianState, basis: &Modes, dt: f64, threshold: f64) -> (f64, f64) {
    let (flight, _) = split(state, basis, 0.0);
    let (mut fast, mut vibration) = (0.0, 0.0);
    for (z, &omega) in flight
        .z
        .iter()
        .zip(&basis.omega)
        .skip(super::vibration::RIGID)
    {
        let e = 0.5 * z.length_squared();
        vibration += e;
        if omega * dt > threshold {
            fast += e;
        }
    }
    let n = state.particles;
    let total: f64 = (0..n)
        .map(|p| {
            0.5 * state.mass[p] * state.v[p].length_squared()
                + state.v0[p] * state.solids[p].energy(state.f[p])
        })
        .sum();
    (
        fast / vibration.max(f64::MIN_POSITIVE),
        fast / total.max(f64::MIN_POSITIVE),
    )
}

fn run(k: f64) -> Observed {
    let step = k * f64::from(CELL_M) / p_wave_speed();
    let mut sim = drop((2.0 * step) as f32);
    let frame = f64::from(sim.config().dt);
    let step = 0.5 * frame;
    let floor_m = sim.config().boundary_thickness as f64 * f64::from(CELL_M);
    let frames = (T_S / frame).ceil() as usize;
    let g = 9.81;
    let count = sim.particles().len() as f64;
    let (mut impact, mut ended) = (None::<usize>, None::<usize>);
    let (mut rising, mut apex) = (false, false);
    let (mut rebound, mut peak) = (0.0f64, 0.0f64);
    let (mut allowed, mut worst) = (0.0f64, 0.0f64);
    let (mut newton, mut gmres, mut steps, mut halvings) = (0u64, 0u64, 0u64, 0u32);
    let mut basis = None;
    let mut shares = [[0.0f64, 0.0], [0.0, 0.0], [0.0, 0.0]];
    let mut previous = (0.0, 0.0, 0.0);
    let clock = std::time::Instant::now();
    for n in 1..=frames {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
        let round_off = count
            * f64::EPSILON
            * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
        worst = worst.max(f.balance.abs() / (allowed + round_off));
        newton += u64::from(f.newton);
        gmres += u64::from(f.gmres);
        steps += 2 * u64::from(f.steps);
        halvings += f.rejected;
        let state = sim.total_lagrangian.as_ref().expect("on");
        let basis = basis.get_or_insert_with(|| modes(state));
        let (vibration, total) = fast_share(state, basis, step, 2.0);
        let (quarter, _) = fast_share(state, basis, step, std::f64::consts::FRAC_PI_2);
        for (slot, value) in shares.iter_mut().zip([vibration, total, quarter]) {
            slot[0] = slot[0].max(value);
            slot[1] += value / frames as f64;
        }
        let now = centre(&sim, floor_m);
        if f.contacts > 0 && impact.is_none() {
            impact = Some(n);
        }
        // From the first contact to the first apex (the centre's vertical
        // velocity, having been positive, no longer is): the largest upward
        // speed, the last contact frame and the largest frame-mean floor
        // force. Contact may drop out for a frame inside the impact.
        if n > 1 && impact.is_some() && !apex {
            if now.1 > 0.0 {
                rising = true;
            } else if rising {
                apex = true;
            }
            if !apex {
                rebound = rebound.max(now.1);
                if f.contacts > 0 {
                    ended = Some(n + 1);
                    peak = peak.max(now.2 * ((now.1 - previous.1) / frame + g));
                }
            }
        }
        previous = now;
    }
    let first = impact.expect("the block reaches the floor");
    let last = ended.expect("the block touches the floor");
    assert!(apex, "the block rises to an apex within the run");
    Observed {
        impact_s: first as f64 * frame,
        rebound_m_s: rebound,
        peak_force_n_m: peak,
        contact_s: (last - first) as f64 * frame,
        height_m: previous.0,
        worst_balance: worst,
        newton_per_step: newton as f64 / steps as f64,
        gmres_per_step: gmres as f64 / steps as f64,
        halvings,
        fast_of_vibration: shares[0],
        fast_of_total: shares[1],
        fast_of_vibration_quarter: shares[2],
        wall_s_per_frame: clock.elapsed().as_secs_f64() / frames as f64,
    }
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_clay_drop_fixed_step_sweep() {
    super::cap::capped("probe_clay_drop_fixed_step_sweep", 14400, || {
        // `EMERGE_PROBE_SWEEP_K`: the steps, comma separated, the first the
        // reference.
        let ks: Vec<f64> = std::env::var("EMERGE_PROBE_SWEEP_K")
            .unwrap_or_else(|_| "0.25,0.5,1,2,4,8,16,32,64".into())
            .split(',')
            .filter_map(|k| k.trim().parse().ok())
            .collect();
        let h_over_c = f64::from(CELL_M) / p_wave_speed();
        println!("clay drop: h/c {h_over_c:.4e} s, run {T_S} s");
        let mut runs = Vec::new();
        for &k in &ks {
            let o = run(k);
            println!("k {k}: {o:?}");
            runs.push(o);
        }
        let r = runs[0];
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
        for (k, o) in ks.iter().zip(&runs) {
            let errors = [
                rel(o.impact_s, r.impact_s),
                rel(o.rebound_m_s, r.rebound_m_s),
                rel(o.peak_force_n_m, r.peak_force_n_m),
                rel(o.contact_s, r.contact_s),
                rel(o.height_m, r.height_m),
            ];
            let largest = errors.iter().copied().fold(0.0, f64::max);
            println!(
                "k {k} (dt {:.3e} s, {:.1} steps per 1/60 s): errors against the first k: impact {:.3e}, rebound {:.3e}, peak force {:.3e}, contact {:.3e}, height at T {:.3e}; largest {largest:.3e}; fast band (omega dt > 2) of the vibration energy at most {:.3e} (mean {:.3e}), of the energy at most {:.3e}; at omega dt > pi/2 at most {:.3e}; balance at most {:.3} of its bound; Newton {:.2} GMRES {:.1} per step; {} halvings; {:.3} s per frame",
                k * h_over_c,
                1.0 / 60.0 / (k * h_over_c),
                errors[0],
                errors[1],
                errors[2],
                errors[3],
                errors[4],
                o.fast_of_vibration[0],
                o.fast_of_vibration[1],
                o.fast_of_total[0],
                o.fast_of_vibration_quarter[0],
                o.worst_balance,
                o.newton_per_step,
                o.gmres_per_step,
                o.halvings,
                o.wall_s_per_frame
            );
        }
    });
}
