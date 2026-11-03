//! Fine steps only where the impact is: E2's clay drop run three ways and
//! compared at the 1/60 s frames over the ring-down. A probe: measurement,
//! nothing held.
//! - A: fixed steps of about `h/(4c)` (frames of `1/60/315` s with an
//!   infinite tolerance: the controller keeps its two half steps), every
//!   mode resolved; the reference.
//! - B: the path's controller at tolerance 1e-2, frames of 1/60 s.
//! - C: B with `with_modal_flight`: the flights carried by the body's
//!   modes, the step at contact.

use super::modal_flight::split;
use super::vibration::{Modes, modes};
use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

pub(super) const FRAME_S: f64 = 1.0 / 60.0;
/// A's frames per 1/60 s.
pub(super) const FINE: usize = 315;
/// 45 frames of 1/60 s.
pub(super) const FRAMES: usize = 45;

fn clay_record() -> ElasticProps {
    ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    }
}

pub(super) fn clay() -> ElasticProps {
    super::g0::series(clay_record())
}

pub(super) fn drop(frame_s: f64, path: TotalLagrangianConfig) -> Simulation {
    let (r, spacing) = refinement();
    let symmetry = symmetry_knobs();
    // Check 5's similarity: lengths and times both scaled by `alpha`.
    let frame_s = frame_s * f64::from(symmetry.scale);
    let mut config = SimConfig::earth(40 * r, 0.01 * symmetry.scale / r as f32, frame_s as f32);
    config.min_dt = config.min_dt.min(frame_s as f32);
    if symmetry.no_gravity {
        config.gravity = Vec2::ZERO;
    }
    config.total_lagrangian = Some(path);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let r = r as f32;
    let start = drop_start();
    // `pressed` starts touching the floor (the block 4 cells tall, its centre
    // 2 cells above the plane); every other start 12 cells above it.
    let height = if start.as_deref() == Some("pressed") {
        2.0
    } else {
        12.0
    };
    // Check 1's shift along the floor, check 2's mirror about the domain's
    // vertical centre line (the walls are symmetric about it).
    let sign = if symmetry.mirror { -1.0 } else { 1.0 };
    let centre = Vec2::new(20.0 * r + sign * symmetry.shift * r, floor + height * r);
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8 * r as i32, 4 * r as i32))
            .at(centre)
            .spacing(spacing)
            .material(id)
            .mass_from(&clay(), &config),
    );
    if let Some(start) = start {
        // ROW18_COVERAGE_PLAN.md sec. 1; velocities in cells per second. The
        // tilt turns the spawn's outline with its particles.
        let turn = Vec2::from_angle(sign * 15f32.to_radians());
        if start == "tilted" {
            let body = sim.total_lagrangian_outlines.last_mut().expect("spawned");
            for q in body.outline.0.iter_mut() {
                *q = centre + turn.rotate(*q - centre);
            }
        }
        let per_metre = 1.0 / config.dx_meters;
        let particles = sim.particles_mut();
        for (x, v) in particles.x.iter_mut().zip(particles.v.iter_mut()) {
            match start.as_str() {
                "thrown" => *v = Vec2::new(sign * 0.5, -1.0) * per_metre,
                "spun" => *v = sign * spin() * (*x - centre).perp(),
                "tilted" => *x = centre + turn.rotate(*x - centre),
                "pressed" => *v = Vec2::new(0.0, -0.5) * per_metre,
                other => panic!("EMERGE_PROBE_DROP_START: unknown start {other}"),
            }
        }
    }
    sim
}

/// STATE_SPACE_COVERAGE_PLAN.md sec. 7's knobs: `EMERGE_PROBE_SHIFT_X`
/// (cells along the floor), `EMERGE_PROBE_DROP_MIRROR`, `EMERGE_PROBE_SCALE`
/// (`alpha`, lengths and times), `EMERGE_PROBE_NO_GRAVITY`. Says so once.
#[derive(Clone, Copy)]
struct Symmetry {
    shift: f32,
    mirror: bool,
    scale: f32,
    no_gravity: bool,
}

fn symmetry_knobs() -> Symmetry {
    static ON: std::sync::OnceLock<Symmetry> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let float = |name: &str, default: f32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let s = Symmetry {
            shift: float("EMERGE_PROBE_SHIFT_X", 0.0),
            mirror: std::env::var_os("EMERGE_PROBE_DROP_MIRROR").is_some(),
            scale: float("EMERGE_PROBE_SCALE", 1.0),
            no_gravity: std::env::var_os("EMERGE_PROBE_NO_GRAVITY").is_some(),
        };
        if (s.shift, s.mirror, s.scale, s.no_gravity) != (0.0, false, 1.0, false) {
            println!(
                "knob active: shift {} cells, mirror {}, scale {}, no gravity {}",
                s.shift, s.mirror, s.scale, s.no_gravity
            );
        }
        s
    })
}

/// The spun start's angular velocity, rad/s: 10, or one ulp above with
/// `EMERGE_PROBE_SPIN_ULP` (the ensemble rule's kind (iii)).
fn spin() -> f32 {
    if std::env::var_os("EMERGE_PROBE_SPIN_ULP").is_some() {
        f32::from_bits(10f32.to_bits() + 1)
    } else {
        10.0
    }
}

/// `EMERGE_PROBE_DROP_START=thrown|spun|tilted|pressed` (ROW18_COVERAGE_PLAN.md):
/// the block's start; unset, the drop from rest. Says so once.
fn drop_start() -> Option<String> {
    static ON: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| {
        let start = std::env::var("EMERGE_PROBE_DROP_START").ok();
        if let Some(s) = &start {
            println!("knob active: drop start {s}");
        }
        start
    })
    .clone()
}

/// `EMERGE_PROBE_REFINE=r` and `EMERGE_PROBE_SPACING=s`
/// (BLOCK_REFINEMENT_PLAN.md): the same physical scene on cells `r` times
/// finer (the block `8r x 4r` cells, its spawn `12r` cells above the floor),
/// particles at `s` cells; `(1, 0.5)` unset. Says so once, from inside.
pub(super) fn refinement() -> (usize, f32) {
    static ON: std::sync::OnceLock<(usize, f32)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let r = std::env::var("EMERGE_PROBE_REFINE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let s = std::env::var("EMERGE_PROBE_SPACING")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        if (r, s) != (1, 0.5) {
            println!("knob active: block refinement r {r}, particle spacing {s} cells");
        }
        (r, s)
    })
}

/// The state at the end of one 1/60 s frame.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Sample {
    height: f64,
    vy: f64,
    vibration: f64,
    energy: f64,
    contact: bool,
    modal: bool,
    steps: u32,
    wall: f64,
    flown: f64,
    rejected: u32,
    floors: u32,
    spin: f64,
    odd: f64,
    odd_share: f64,
}

/// The spin `L / I` about the centre of mass, and the vibration's part odd
/// under the mirror about the reference block's vertical mid-line: `w = v -
/// v_c - spin perp(x - c)`, `w_odd(p) = (w(p) - M w(p')) / 2` with `p'` the
/// particle at the mirrored reference position (`M` the reflection of x);
/// returns the spin, the odd kinetic energy `1/2 sum m |w_odd|^2` (J/m) and
/// its share of `1/2 sum m |w|^2`.
pub(super) fn asymmetry(state: &TotalLagrangianState) -> (f64, f64, f64) {
    let n = state.particles;
    let mass: f64 = state.mass[..n].iter().sum();
    let centre_of = |q: &[DVec2]| (0..n).map(|p| state.mass[p] * q[p]).sum::<DVec2>() / mass;
    let (c, vc, origin) = (
        centre_of(&state.x),
        centre_of(&state.v),
        centre_of(&state.reference),
    );
    let (l, i) = (0..n).fold((0.0, 0.0), |(l, i), p| {
        let r = state.x[p] - c;
        (
            l + state.mass[p] * r.perp_dot(state.v[p] - vc),
            i + state.mass[p] * r.length_squared(),
        )
    });
    let spin = l / i;
    let w = |p: usize| state.v[p] - vc - spin * (state.x[p] - c).perp();
    let mirror = |q: DVec2| DVec2::new(2.0 * origin.x - q.x, q.y);
    let (mut odd, mut all) = (0.0, 0.0);
    for p in 0..n {
        let target = mirror(state.reference[p]);
        let twin = (0..n)
            .min_by(|&a, &b| {
                (state.reference[a] - target)
                    .length()
                    .total_cmp(&(state.reference[b] - target).length())
            })
            .expect("particles");
        let reflected = DVec2::new(-w(twin).x, w(twin).y);
        let part = 0.5 * (w(p) - reflected);
        odd += 0.5 * state.mass[p] * part.length_squared();
        all += 0.5 * state.mass[p] * w(p).length_squared();
    }
    (spin, odd, odd / all.max(f64::MIN_POSITIVE))
}

/// The centre's height above the floor and vertical velocity, the
/// vibration energy (the modes' energy, rigid motion removed), the
/// kinetic plus stored plus gravity energy.
pub(super) fn sample(sim: &Simulation, basis: &Modes) -> (f64, f64, f64, f64) {
    let state = sim.total_lagrangian.as_ref().expect("on");
    let n = state.particles;
    let floor = sim.config().boundary_thickness as f64 * f64::from(sim.config().dx_meters);
    let mass: f64 = state.mass[..n].iter().sum();
    let (y, vy) = (0..n).fold((0.0, 0.0), |(y, vy), p| {
        (
            y + state.mass[p] * state.x[p].y,
            vy + state.mass[p] * state.v[p].y,
        )
    });
    let (flight, _) = split(state, basis, 0.0);
    let vibration: f64 = flight
        .z
        .iter()
        .skip(super::vibration::RIGID)
        .map(|z| 0.5 * z.length_squared())
        .sum();
    (
        y / mass - floor,
        vy / mass,
        vibration,
        sim.total_lagrangian_energy(state),
    )
}

/// One run, sampled at every 1/60 s; `per` path frames per sample.
fn run(mut sim: Simulation, per: usize) -> (Vec<Sample>, f64) {
    let mut out = Vec::new();
    let mut basis: Option<Modes> = None;
    let mut allowed = 0.0f64;
    let mut worst = 0.0f64;
    let count = sim.particles().len() as f64;
    let digest = std::env::var_os("EMERGE_PROBE_STATE_DIGEST").is_some();
    // `EMERGE_PROBE_FRAMES=n`: only the first `n` frames (the sizing runs).
    let frames = std::env::var("EMERGE_PROBE_FRAMES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map_or(FRAMES, |n| {
            println!("knob active: {n} frames");
            n.min(FRAMES)
        });
    for frame in 0..frames {
        let mut s = Sample::default();
        let clock = std::time::Instant::now();
        for _ in 0..per {
            sim.step();
            let f = sim.total_lagrangian_frame().expect("on");
            allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
            let round_off = count
                * f64::EPSILON
                * (f.parts.iter().map(|e| e.abs()).sum::<f64>()
                    + f.heat.abs()
                    + f.start_energy.abs());
            worst = worst.max(f.balance.abs() / (allowed + round_off));
            // `EMERGE_PROBE_LEDGER_FRAMES` (PUSO_CANDIDATE_PLAN.md sec. 13).
            if std::env::var_os("EMERGE_PROBE_LEDGER_FRAMES").is_some() {
                println!(
                    "ledger frame: balance {:+.6e}, bound so far {:.6e}, ratio {:.4}",
                    f.balance,
                    allowed + round_off,
                    f.balance.abs() / (allowed + round_off)
                );
            }
            s.contact |= f.contacts > 0;
            s.modal |= f.modal;
            s.flown += f.flown;
            s.steps += 2 * f.steps;
            s.rejected += f.rejected;
            s.floors += f.at_floor;
        }
        s.wall = clock.elapsed().as_secs_f64();
        // `EMERGE_PROBE_STATE_DIGEST` (ROW16 sec. 10): the f64 state's sums at
        // full precision, so a perturbation's reach shows below the printed
        // observables' digits.
        if digest {
            let state = sim.total_lagrangian.as_ref().expect("on");
            let (sx, sv) = (0..state.x.len()).fold((0.0, 0.0), |(a, b), p| {
                (
                    a + state.x[p].x + state.x[p].y,
                    b + state.v[p].x + state.v[p].y,
                )
            });
            println!("digest frame {frame}: sum x {sx:.17e} m, sum v {sv:.17e} m/s");
        }
        let state = sim.total_lagrangian.as_ref().expect("on");
        let basis = basis.get_or_insert_with(|| modes(state));
        (s.spin, s.odd, s.odd_share) = asymmetry(state);
        (s.height, s.vy, s.vibration, s.energy) = sample(&sim, basis);
        out.push(s);
    }
    (out, worst)
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_impact_local_stepping() {
    super::cap::capped("probe_impact_local_stepping", 7200, || {
        let a = run(
            drop(
                FRAME_S / FINE as f64,
                TotalLagrangianConfig::new(f64::INFINITY),
            ),
            FINE,
        );
        let b = run(drop(FRAME_S, TotalLagrangianConfig::new(1e-2)), 1);
        let c = run(
            drop(
                FRAME_S,
                TotalLagrangianConfig::new(1e-2).with_modal_flight(),
            ),
            1,
        );
        let d = run(
            drop(
                FRAME_S,
                TotalLagrangianConfig::new(1e-2).with_subframe_handoff(),
            ),
            1,
        );
        let start = a.0[0].energy;
        let impact = a.0.iter().position(|s| s.contact).expect("an impact");
        // The ring-down: after the first impact's frames, to T.
        let after = impact
            + a.0[impact..]
                .iter()
                .position(|s| !s.contact)
                .expect("it leaves");
        println!(
            "A's first contact frame {impact}, ring-down from frame {after}; balances over their bounds: A {:.3}, B {:.3}, C {:.3}, D {:.3}",
            a.1, b.1, c.1, d.1
        );
        for (name, run) in [("A", &a.0), ("B", &b.0), ("C", &c.0), ("D", &d.0)] {
            for (n, s) in run.iter().enumerate() {
                println!(
                    "{name} frame {n}: steps {} contact {} modal {} flown {:.3e} s wall {:.3} s; height {:.6e} m, vy {:+.6e} m/s, vibration {:.4e} J/m, energy {:+.3e} of the start",
                    s.steps,
                    s.contact,
                    s.modal,
                    s.flown,
                    s.wall,
                    s.height,
                    s.vy,
                    s.vibration,
                    s.energy / start - 1.0
                );
            }
        }
        for (name, run) in [("B", &b.0), ("C", &c.0), ("D", &d.0)] {
            let (mut height, mut vy, mut vibration) = (Vec::new(), Vec::new(), Vec::new());
            for (&r, &s) in a.0[after..FRAMES].iter().zip(&run[after..FRAMES]) {
                height.push((s.height - r.height).abs() / r.height.abs());
                vy.push((s.vy - r.vy).abs() / a.0[after].vy.abs().max(f64::MIN_POSITIVE));
                vibration.push(
                    (s.vibration - r.vibration).abs() / r.vibration.abs().max(f64::MIN_POSITIVE),
                );
            }
            let stats = |v: &[f64]| {
                (
                    v.iter().copied().fold(0.0, f64::max),
                    v.iter().sum::<f64>() / v.len() as f64,
                )
            };
            let contact_steps: u32 = run.iter().filter(|s| s.contact).map(|s| s.steps).sum();
            let contact_frames = run.iter().filter(|s| s.contact).count();
            let flight_steps: u32 = run.iter().filter(|s| !s.contact).map(|s| s.steps).sum();
            let flight_frames = run.len() - contact_frames;
            let modal = run.iter().filter(|s| s.modal).count();
            println!(
                "{name} against A over the ring-down: height largest {:.3e} mean {:.3e}; vy (of the take-off speed) largest {:.3e} mean {:.3e}; vibration energy largest {:.3e} mean {:.3e}; steps per frame in contact {:.1} ({contact_frames} frames), in flight {:.1} ({flight_frames} frames, {modal} modal)",
                stats(&height).0,
                stats(&height).1,
                stats(&vy).0,
                stats(&vy).1,
                stats(&vibration).0,
                stats(&vibration).1,
                f64::from(contact_steps) / contact_frames.max(1) as f64,
                f64::from(flight_steps) / flight_frames.max(1) as f64
            );
        }
    });
}

/// Run C alone, each frame's reason for being stepped instead of flown:
/// the deltas of the hand-backs, the refusals (law, yielded, bound) and the
/// failed propagator checks, with the contacts and the centre's and the
/// lowest contact point's heights at the frame's start. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_flight_refusals() {
    super::cap::capped("probe_flight_refusals", 3600, || {
        let mut sim = drop(
            FRAME_S,
            TotalLagrangianConfig::new(1e-2).with_modal_flight(),
        );
        let floor = sim.config().boundary_thickness as f64 * f64::from(sim.config().dx_meters);
        let mut last = [0u64; 5];
        let mut basis: Option<Modes> = None;
        for n in 0..FRAMES {
            // The flight's clearance bound at the frame's start (`Flight::
            // clear`: each mode at `|z_i| / omega_i`, all in phase), its
            // largest over the contact points, against the vibration's real
            // displacement there (`sum_i phi_i q_i`, the modes' sum).
            if let (Some(state), Some(b)) = (sim.total_lagrangian.as_ref(), basis.as_ref()) {
                let (flight, _) = split(state, b, 0.0);
                let p = state.particles;
                let (mut bound, mut real) = (0.0f64, 0.0f64);
                for o in p..state.x.len() {
                    let mut sum = 0.0;
                    let mut displacement = DVec2::ZERO;
                    for i in super::vibration::RIGID..b.omega.len() {
                        if b.omega[i] <= 0.0 {
                            continue;
                        }
                        let z = flight.z[i];
                        sum += b.at[i][o].length() * z.length() / b.omega[i];
                        displacement += b.at[i][o] * (z.x / b.omega[i]);
                    }
                    bound = bound.max(sum);
                    real = real.max(displacement.length());
                }
                println!(
                    "  start of frame {n}: clearance bound at most {bound:.4e} m over the contact points; the vibration's real displacement at most {real:.4e} m"
                );
            }
            // The body's spin at the frame's start: `L`, `I` about the centre,
            // `L / I`.
            if let Some(state) = sim.total_lagrangian.as_ref() {
                let p = state.particles;
                let total: f64 = state.mass[..p].iter().sum();
                let (cx, cv) = (0..p).fold((DVec2::ZERO, DVec2::ZERO), |(a, b), i| {
                    (
                        a + state.mass[i] * state.x[i],
                        b + state.mass[i] * state.v[i],
                    )
                });
                let (centre, velocity) = (cx / total, cv / total);
                let (l, inertia) = (0..p).fold((0.0, 0.0), |(l, j), i| {
                    let arm = state.x[i] - centre;
                    (
                        l + state.mass[i] * arm.perp_dot(state.v[i] - velocity),
                        j + state.mass[i] * arm.length_squared(),
                    )
                });
                println!(
                    "  start of frame {n}: L {l:+.6e} kg m^2/s per m, I {inertia:.6e} kg m^2 per m, spin L / I {:+.4e} rad/s",
                    l / inertia
                );
            }
            // The frame's own three checks replayed on a copy, in its order
            // (`fly_total_lagrangian_frame`): clear of the walls, the rebuild,
            // the angular momentum kept.
            if let Some(state) = sim.total_lagrangian.as_ref() {
                let mut copy = state.clone();
                let (t0, t1) = (copy.time, copy.time + FRAME_S);
                let planes = sim.total_lagrangian_walls().0;
                let gravity = sim.total_lagrangian_gravity();
                let free = copy.outlines.len() == 1
                    && copy.cache.as_ref().is_some_and(|c| c.contacts.is_empty());
                let modes = match copy.modes.take() {
                    Some(Some(m)) => Some(m),
                    _ => None,
                };
                let flight = match (copy.flight.take(), &modes) {
                    (Some(f), _) => Some(f),
                    (None, Some(m)) if free => {
                        super::flight::Flight::start(&copy, m, (t0, FRAME_S)).ok()
                    }
                    _ => None,
                };
                match (flight, modes) {
                    (Some(mut flight), Some(m)) => {
                        let clear = flight.clear(gravity, &planes, (t0, t1));
                        let rebuilt = clear && flight.rebuild(&m, gravity, t1, &mut copy);
                        let kept = rebuilt && flight.keeps_angular_momentum(&copy);
                        println!(
                            "  replay of frame {n}: free {free}, clear {clear}, rebuilt {rebuilt}, angular momentum kept {kept}"
                        );
                        if rebuilt {
                            let (rebuilt_l, flight_l, tolerance) = flight.angular_parts(&copy);
                            println!(
                                "  replay of frame {n}: rebuilt L {rebuilt_l:+.6e}, flight L {flight_l:+.6e} kg m^2/s per m; excess {:.3e} = {:.3e} of the tolerance {tolerance:.3e}",
                                (rebuilt_l - flight_l).abs(),
                                (rebuilt_l - flight_l).abs() / tolerance.max(f64::MIN_POSITIVE)
                            );
                        }
                    }
                    _ => println!("  replay of frame {n}: free {free}, no flight to try"),
                }
            }
            let heights = sim.total_lagrangian.as_ref().map(|state| {
                let p = state.particles;
                let mass: f64 = state.mass[..p].iter().sum();
                let centre =
                    (0..p).map(|i| state.mass[i] * state.x[i].y).sum::<f64>() / mass - floor;
                let lowest = state.x[p..]
                    .iter()
                    .map(|q| q.y)
                    .fold(f64::INFINITY, f64::min)
                    - floor;
                (centre, lowest)
            });
            sim.step();
            if basis.is_none() {
                basis = Some(modes(sim.total_lagrangian.as_ref().expect("on")));
            }
            let f = sim.total_lagrangian_frame().expect("on");
            let now = [
                f.hand_backs,
                f.modal_refused_law,
                f.modal_refused_yielded,
                f.modal_refused_bound,
                f.modal_check_failed,
            ];
            let delta: Vec<u64> = now.iter().zip(&last).map(|(a, b)| a - b).collect();
            last = now;
            println!(
                "frame {n}: modal {} steps {} contacts {}; hand-backs +{} refused law +{} yielded +{} bound +{} check failed +{}; at the start centre {:?} m, lowest contact point {:?} m",
                f.modal,
                2 * f.steps,
                f.contacts,
                delta[0],
                delta[1],
                delta[2],
                delta[3],
                delta[4],
                heights.map(|h| h.0),
                heights.map(|h| h.1)
            );
        }
    });
}

/// Run D alone (the sub-frame handoff), each frame printed in the format of
/// `probe_impact_local_stepping`, to set against that probe's logged A. A
/// probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_handoff_alone() {
    super::cap::capped("probe_handoff_alone", 3600, || {
        let d = run(
            drop(
                FRAME_S,
                TotalLagrangianConfig::new(1e-2).with_subframe_handoff(),
            ),
            1,
        );
        let start = d.0[0].energy;
        println!("D balance at most {:.3} of its bound", d.1);
        for (n, s) in d.0.iter().enumerate() {
            println!(
                "D frame {n}: steps {} contact {} modal {} flown {:.3e} s wall {:.3} s; height {:.6e} m, vy {:+.6e} m/s, vibration {:.4e} J/m, energy {:+.3e} of the start",
                s.steps,
                s.contact,
                s.modal,
                s.flown,
                s.wall,
                s.height,
                s.vy,
                s.vibration,
                s.energy / start - 1.0
            );
        }
    });
}

/// The one-ulp discriminator (`ROW16_GMRES_SWEEP_PLAN.md` sec. 5), before
/// the first step: `EMERGE_PROBE_ULP=v` sets every particle's start y
/// velocity to one ulp of zero downward (the f32 subnormal), `x` raises one
/// particle's start y by one f32 ulp; it says so from inside.
pub(super) fn ulp_perturb(sim: &mut Simulation) {
    let Ok(kind) = std::env::var("EMERGE_PROBE_ULP") else {
        return;
    };
    let particles = sim.particles_mut();
    match kind.as_str() {
        "v" => {
            for v in particles.v.iter_mut() {
                v.y = -f32::from_bits(1);
            }
        }
        // `vp:<particle>`: that particle's vertical velocity moved by one
        // ulp (a start with a nonzero velocity: ROW18_COVERAGE_PLAN.md, the
        // ensemble rule's kind (ii)).
        s if s.starts_with("vp:") => {
            let p: usize = s[3..].parse().unwrap_or(0);
            let v = &mut particles.v[p].y;
            *v = f32::from_bits(v.to_bits() + 1);
            println!("knob active: one ulp on particle {p}'s vertical velocity");
        }
        s if s.starts_with("s:") => {
            // `s:<particle>`: that particle and its mirror about the body's
            // vertical mid-line, each raised by one ulp: the start's
            // left-right symmetry kept.
            let p: usize = s[2..].parse().unwrap_or(0);
            let (lo, hi) = particles
                .x
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), q| {
                    (l.min(q.x), h.max(q.x))
                });
            let target = Vec2::new(lo + hi - particles.x[p].x, particles.x[p].y);
            let twin = (0..particles.x.len())
                .min_by(|&a, &b| {
                    (particles.x[a] - target)
                        .length()
                        .total_cmp(&(particles.x[b] - target).length())
                })
                .expect("particles");
            for q in [p, twin] {
                let y = particles.x[q].y;
                particles.x[q].y = f32::from_bits(y.to_bits() + 1);
            }
            println!("knob active: ulp perturbation {kind} (particles {p} and its mirror {twin})");
            return;
        }
        x if x.starts_with('x') => {
            // `x` or `x:<particle>`: which particle (0 by default).
            let p: usize = x
                .strip_prefix("x:")
                .and_then(|i| i.parse().ok())
                .unwrap_or(0);
            let y = particles.x[p].y;
            particles.x[p].y = f32::from_bits(y.to_bits() + 1);
            println!(
                "ulp perturbation: particle {p} start y {y:.9e} to {:.9e} (bits {:#x} to {:#x})",
                particles.x[p].y,
                y.to_bits(),
                particles.x[p].y.to_bits()
            );
        }
        _ => return,
    }
    println!("knob active: ulp perturbation {kind}");
}

/// Run D under the floor-collapse candidates' knobs (`FLOOR_COLLAPSE_PLAN.md`;
/// `EMERGE_PROBE_PAIR_PIECES`, `EMERGE_PROBE_SWITCH_BY_CONTACTS`), labelled
/// by `EMERGE_PROBE_CANDIDATE`: per frame the controlled steps, rejected
/// tries, floor visits and wall time with the observables, for the
/// comparison against reference A's log. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_floor_candidates() {
    super::cap::capped("probe_floor_candidates", 7200, || {
        let label = std::env::var("EMERGE_PROBE_CANDIDATE").unwrap_or_else(|_| "baseline".into());
        let tolerance = std::env::var("EMERGE_PROBE_TOLERANCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-2);
        let mut path = TotalLagrangianConfig::new(tolerance).with_subframe_handoff();
        // `EMERGE_PROBE_GMRES_TOL` (ROW16_GMRES_SWEEP_PLAN.md).
        if let Some(tol) = std::env::var("EMERGE_PROBE_GMRES_TOL")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            path.settings.gmres_tolerance = tol;
        }
        println!(
            "knob active: gmres_tolerance {:e}",
            path.settings.gmres_tolerance
        );
        // `EMERGE_PROBE_REFERENCE_A`: reference A's construction instead
        // (fixed steps of `1/60/FINE` frames, infinite tolerance, no handoff).
        // On cells `r` times finer A keeps its Courant number: `r FINE` frames.
        let reference = std::env::var_os("EMERGE_PROBE_REFERENCE_A").is_some();
        let fine = (FINE as f64 * refinement().0 as f64 * super::g0::wave_ratio(&clay_record()))
            .ceil() as usize;
        let mut sim = if reference {
            println!("knob active: reference A's construction, {fine} frames per 1/60 s");
            drop(
                FRAME_S / fine as f64,
                TotalLagrangianConfig::new(f64::INFINITY),
            )
        } else {
            drop(FRAME_S, path)
        };
        ulp_perturb(&mut sim);
        let (run, worst) = run(sim, if reference { fine } else { 1 });
        println!("{label} tolerance {tolerance:e}");
        let start = run[0].energy;
        println!("{label} balance at most {worst:.3} of its bound");
        for (n, s) in run.iter().enumerate() {
            println!(
                "{label} frame {n}: steps {} rejected {} floor {} contact {} modal {} flown {:.3e} s wall {:.3} s; height {:.6e} m, vy {:+.6e} m/s, vibration {:.4e} J/m, energy {:+.3e} of the start; spin {:+.3e} rad/s, odd vibration {:.3e} J/m ({:.3e} of it)",
                s.steps / 2,
                s.rejected,
                s.floors,
                s.contact,
                s.modal,
                s.flown,
                s.wall,
                s.height,
                s.vy,
                s.vibration,
                s.energy / start - 1.0,
                s.spin,
                s.odd,
                s.odd_share
            );
        }
    });
}

/// S1's criterion 4 (`tmp/papers/dl_realtime/S1_FIXED_CONTACT_STEP_PLAN.md`):
/// run D to frame 7's end, then frame 8 with every controlled pair sampled
/// against its 64-piece reference (`hooks::EffectivityRow`); for the pairs
/// of the fixed contact-phase step (`EMERGE_PROBE_CONTACT_FIXED_STEP`), the
/// median true error of the kept halves, in the controller's norm (the
/// end-velocity norm when `EMERGE_PROBE_NORM=velocity`), and the whole
/// step's in both forms, for information. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_s1_contact_convergence() {
    use crate::spacetime::total_lagrangian::hooks;
    super::cap::capped("probe_s1_contact_convergence", 7200, || {
        let mut sim = drop(
            FRAME_S,
            TotalLagrangianConfig::new(1e-2).with_subframe_handoff(),
        );
        for _ in 0..8 {
            sim.step();
        }
        hooks::set_effectivity(1, 0);
        sim.step();
        hooks::set_effectivity(0, 0);
        let rows = hooks::take_effectivity();
        let f = sim.total_lagrangian_frame().expect("on");
        // The fixed step's pairs: the most frequent step length among the
        // sampled pairs (the phase's `2 k h / c`; the cut steps differ).
        let mut lengths: Vec<f64> = rows.iter().map(|r| r.dt).collect();
        lengths.sort_by(f64::total_cmp);
        let mode = lengths
            .chunk_by(|a, b| (a - b).abs() <= 1e-12 * a.abs())
            .max_by_key(|run| run.len())
            .map_or(f64::NAN, |run| run[0]);
        let fixed: Vec<&hooks::EffectivityRow> = rows
            .iter()
            .filter(|r| (r.dt - mode).abs() <= 1e-12 * mode.abs())
            .collect();
        let median = |mut v: Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let halves = median(fixed.iter().map(|r| r.halves_error).collect());
        // The split by the pair's class (S1 plan sec. 12): switching
        // (an activation or release inside the pair) or not.
        for (class, switching) in [("switching", true), ("non-switching", false)] {
            let errors: Vec<f64> = fixed
                .iter()
                .filter(|r| r.switching == switching)
                .map(|r| r.halves_error)
                .collect();
            println!(
                "S1 convergence frame 8, {class} pairs: {}, median kept-halves error {:.6e}",
                errors.len(),
                median(errors.clone())
            );
        }
        let whole = median(fixed.iter().map(|r| r.whole_error).collect());
        let whole_velocity = median(fixed.iter().map(|r| r.whole_error_velocity).collect());
        println!(
            "S1 convergence frame 8: sampled pairs {}, at the fixed step {mode:.6e} s {}, of them holding a contact {}; median true error: kept halves {halves:.6e} (controller's norm), whole step {whole:.6e} (dx/dt form) and {whole_velocity:.6e} (end-velocity form); frame steps {} rejected {} wall {:.3} s",
            rows.len(),
            fixed.len(),
            fixed.iter().filter(|r| r.held).count(),
            f.steps,
            f.rejected,
            f.wall
        );
    });
}

/// S3 (`tmp/papers/dl_realtime/S3_DENSE_SOLVE_PLAN.md`): run D's frame 8,
/// in the regime the environment selects (the baseline, or S1' with
/// `EMERGE_PROBE_CONTACT_FIXED_STEP`), its GMRES time and Jacobian products
/// per Newton iteration over every solve, against a dense LU with partial
/// pivoting (textbook Gaussian elimination, timing only, never used by the
/// solver) at the frame's own active-unknown count and at 612 and 2244. A
/// probe: measurement, nothing held.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_s3_dense_solve() {
    use crate::spacetime::total_lagrangian::hooks;
    super::cap::capped("probe_s3_dense_solve", 3600, || {
        let mut sim = drop(
            FRAME_S,
            TotalLagrangianConfig::new(1e-2).with_subframe_handoff(),
        );
        for _ in 0..8 {
            sim.step();
        }
        let _ = hooks::take_parts();
        sim.step();
        let parts = hooks::take_parts();
        let f = sim.total_lagrangian_frame().expect("on");
        let state = sim.total_lagrangian.as_ref().expect("on");
        let grid = &state.cache.as_ref().expect("built").grid;
        let n = grid.active.iter().filter(|&&a| a).count();
        let newton = parts[9].max(1.0);
        println!(
            "S3 frame 8: steps {} rejected {}; solves {:.0}, Newton iterations {newton:.0}, Jacobian products {:.0}; per Newton iteration: GMRES {:.4} ms (products and preconditioner included), products {:.1}, the products' time {:.4} ms; frame wall {:.3} s; active unknowns {n}",
            f.steps,
            f.rejected,
            parts[7],
            parts[0],
            1e3 * f.timing.gmres / newton,
            parts[0] / newton,
            1e3 * f.timing.jacobian / newton,
            f.wall
        );
        // A dense, diagonally dominant matrix of size `m` (a fixed linear
        // congruential sequence, so every run factors the same numbers),
        // factored `reps` times; the median time, and the two triangular
        // solves'.
        let lu_time = |m: usize, reps: usize| -> (f64, f64) {
            let mut seed = 0x2545_f491_4f6c_dd1du64;
            let mut next = || {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            };
            let base: Vec<f64> = (0..m * m)
                .map(|ij| if ij / m == ij % m { m as f64 } else { next() })
                .collect();
            let (mut factor, mut solve) = (Vec::new(), Vec::new());
            for _ in 0..reps {
                let mut a = base.clone();
                let mut pivot = vec![0usize; m];
                let clock = std::time::Instant::now();
                for k in 0..m {
                    let p = (k..m)
                        .max_by(|&i, &j| a[i * m + k].abs().total_cmp(&a[j * m + k].abs()))
                        .expect("a row");
                    pivot[k] = p;
                    if p != k {
                        for j in 0..m {
                            a.swap(k * m + j, p * m + j);
                        }
                    }
                    let d = a[k * m + k];
                    for i in (k + 1)..m {
                        let l = a[i * m + k] / d;
                        a[i * m + k] = l;
                        for j in (k + 1)..m {
                            a[i * m + j] -= l * a[k * m + j];
                        }
                    }
                }
                factor.push(clock.elapsed().as_secs_f64());
                let mut b: Vec<f64> = (0..m).map(|i| i as f64).collect();
                let clock = std::time::Instant::now();
                for (k, &p) in pivot.iter().enumerate() {
                    b.swap(k, p);
                }
                for i in 0..m {
                    let s: f64 = (0..i).map(|j| a[i * m + j] * b[j]).sum();
                    b[i] -= s;
                }
                for i in (0..m).rev() {
                    let s: f64 = ((i + 1)..m).map(|j| a[i * m + j] * b[j]).sum();
                    b[i] = (b[i] - s) / a[i * m + i];
                }
                solve.push(clock.elapsed().as_secs_f64());
                assert!(b.iter().all(|x| x.is_finite()), "a finite solution");
            }
            factor.sort_by(f64::total_cmp);
            solve.sort_by(f64::total_cmp);
            (factor[reps / 2], solve[reps / 2])
        };
        for (m, reps) in [(n, 21), (612, 7), (2244, 3)] {
            let (factor, solve) = lu_time(m, reps);
            println!(
                "S3 dense LU at n = {m}: factor {:.4} ms, two triangular solves {:.4} ms (median of {reps})",
                1e3 * factor,
                1e3 * solve
            );
        }
    });
}

/// C2 proper's P0 on the block (C2_PROPER_PLAN.md sec. 7, before P1): the
/// clay drop's grid after its first frame, the bottom side's outline points
/// as the contact rows (normal `+y`), each particle's `(mu, bulk - mu)`, the
/// C2 diagnostic's Newton matrix at Courant 0.25 and 0.0475 by the P-wave
/// speed. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_c2_proper_p0_block() {
    let mut sim = drop(FRAME_S, TotalLagrangianConfig::new(1e-2));
    sim.step();
    let state = sim.total_lagrangian.as_ref().expect("on");
    let grid = &state.cache.as_ref().expect("built by the step").grid;
    let points: Vec<usize> = state.outlines.iter().flatten().copied().collect();
    let lowest = points
        .iter()
        .map(|&o| state.reference[o].y)
        .fold(f64::INFINITY, f64::min);
    let contacts: Vec<(usize, DVec2)> = points
        .iter()
        .filter(|&&o| state.reference[o].y <= lowest + 1e-9)
        .map(|&o| (o, DVec2::Y))
        .collect();
    let p = (0..state.particles)
        .find(|&p| state.v0[p] > 0.0)
        .expect("a particle");
    let solid = state.solids[p];
    let c = ((solid.bulk + solid.mu) * state.v0[p] / state.mass[p]).sqrt();
    let dt = |nu: f64| (nu, nu * grid.cell / c);
    crate::spacetime::grid::mac::c2_proper_probes::p0_core(
        &format!("the clay block, h {} m, c {c:.2} m/s", grid.cell),
        grid,
        (&state.reference, &state.mass, &state.v0),
        &|q| {
            (
                state.solids[q].mu,
                state.solids[q].bulk - state.solids[q].mu,
            )
        },
        &contacts,
        &[dt(0.25), dt(0.0475)],
    );
}
