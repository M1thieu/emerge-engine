//! Probes of the path inside `Simulation`.

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// The frictionless slide of `tests/total_lagrangian_path.rs` with
/// every Newton iteration of frame `EMERGE_PROBE_TL_FRAME` (default 3)
/// printed. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_wired_slide_newton() {
    super::cap::capped("probe_wired_slide_newton", 1800, || {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let props = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 2.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        for v in sim.particles_mut().v.iter_mut() {
            *v = Vec2::new(50.0, 0.0);
        }
        let watched: u64 = std::env::var("EMERGE_PROBE_TL_FRAME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        for frame in 1..=watched {
            if frame == watched {
                crate::spacetime::total_lagrangian::hooks::start(0);
            }
            sim.step();
            if frame == watched {
                for (solve, lines) in crate::spacetime::total_lagrangian::hooks::finish()
                    .iter()
                    .enumerate()
                {
                    for line in lines {
                        println!("  solve {solve}: {line}");
                    }
                }
            }
            println!("frame {frame}: {:?}", sim.total_lagrangian_frame());
        }
    });
}

/// How far a free body's elastic modes are from orthogonal to its rigid
/// rotation on the particles, `leak_i = sum_p m_p (X_p - C) x phi_i(X_p)`,
/// against the angular momentum allowance modal flight checks
/// (`Flight::keeps_angular_momentum`): the ringing bar of `tests/zener.rs`
/// and the clay block of `tests/modal_flight.rs`, after one stepped frame.
/// A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_modes_rotation_leak() {
    super::cap::capped("probe_modes_rotation_leak", 1800, || {
        let bar = || {
            let mut config = SimConfig::earth(48, 0.01, 2.0e-3);
            config.gravity = Vec2::ZERO;
            config.min_dt = config.min_dt.min(1e-9);
            config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2).with_modal_flight());
            let props = ElasticProps {
                e_pa: 1.0e5,
                nu: 0.0,
                rho_kg_m3: 1000.0,
            };
            let mut sim = Simulation::empty(config);
            let id = sim
                .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
                .0;
            let _bar = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(32, 2))
                    .at(Vec2::new(24.0, 24.0))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&props, &config),
            );
            let x = sim.particles().x.clone();
            for (v, q) in sim.particles_mut().v.iter_mut().zip(x) {
                *v = Vec2::new(
                    5.0e-3 * (std::f32::consts::PI * (q.x - 8.0) / 32.0).cos(),
                    0.0,
                );
            }
            sim
        };
        let block = || {
            let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
            config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1).with_modal_flight());
            let props = ElasticProps {
                e_pa: 1.125e7,
                nu: 0.3,
                rho_kg_m3: 1700.0,
            };
            let mut sim = Simulation::empty(config);
            let id = sim
                .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
                .0;
            let floor = config.boundary_thickness as f32;
            let _block = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(8, 4))
                    .at(Vec2::new(20.0, floor + 12.0))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&props, &config),
            );
            sim
        };
        for (name, mut sim) in [("bar", bar()), ("block", block())] {
            sim.step();
            let state = sim.total_lagrangian.as_ref().expect("on the path");
            let Some(Some(modes)) = &state.modes else {
                println!("{name}: no modes");
                continue;
            };
            let n = state.particles;
            let total: f64 = state.mass[..n].iter().sum();
            let origin =
                (0..n).fold(DVec2::ZERO, |a, p| a + state.mass[p] * state.reference[p]) / total;
            let inertia: f64 = (0..n)
                .map(|p| state.mass[p] * (state.reference[p] - origin).length_squared())
                .sum();
            let velocity = (0..n).fold(DVec2::ZERO, |a, p| a + state.mass[p] * state.v[p]) / total;
            let (mut worst, mut predicted, mut size) = (0.0f64, 0.0f64, 0.0f64);
            for i in crate::spacetime::total_lagrangian::RIGID_MODES..modes.omega.len() {
                let leak: f64 = (0..n)
                    .map(|p| state.mass[p] * (state.reference[p] - origin).perp_dot(modes.at[i][p]))
                    .sum();
                let rate: f64 = (0..n)
                    .map(|p| state.mass[p] * (state.v[p] - velocity).dot(modes.at[i][p]))
                    .sum();
                worst = worst.max(leak.abs() / inertia.sqrt());
                predicted += leak.abs() * rate.abs();
            }
            for p in 0..n {
                let arm = state.x[p] - origin;
                size += state.mass[p] * arm.length() * (state.v[p].length() + velocity.length());
            }
            // One flown frame by hand: the angular momentum before and after
            // the rebuild, and the leak's signed prediction `sum_i leak_i
            // q'_i` from the rebuilt velocities.
            let momentum = |s: &TotalLagrangianState| -> f64 {
                let total: f64 = s.mass[..n].iter().sum();
                let (c, v) = (0..n).fold((DVec2::ZERO, DVec2::ZERO), |(c, v), p| {
                    (c + s.mass[p] * s.x[p], v + s.mass[p] * s.v[p])
                });
                let (c, v) = (c / total, v / total);
                (0..n)
                    .map(|p| s.mass[p] * (s.x[p] - c).perp_dot(s.v[p] - v))
                    .sum()
            };
            let mut trial = state.clone();
            let length = f64::from(sim.config().dt);
            let before = momentum(&trial);
            let mut flight =
                Flight::start(&trial, modes, (trial.time, length)).expect("a flight starts");
            assert!(flight.rebuild(modes, DVec2::ZERO, trial.time + length, &mut trial));
            let after = momentum(&trial);
            let rebuilt_velocity =
                (0..n).fold(DVec2::ZERO, |a, p| a + trial.mass[p] * trial.v[p]) / total;
            let signed: f64 = (crate::spacetime::total_lagrangian::RIGID_MODES..modes.omega.len())
                .map(|i| {
                    let leak: f64 = (0..n)
                        .map(|p| {
                            state.mass[p] * (state.reference[p] - origin).perp_dot(modes.at[i][p])
                        })
                        .sum();
                    let rate: f64 = (0..n)
                        .map(|p| {
                            trial.mass[p] * (trial.v[p] - rebuilt_velocity).dot(modes.at[i][p])
                        })
                        .sum();
                    leak * rate
                })
                .sum();
            // The eigen-solve itself: each elastic mode's residual and its
            // mass inner product with the solve's own rigid modes.
            let grid = &state.cache.as_ref().expect("stepped").grid;
            let system = crate::spacetime::total_lagrangian::system(
                grid,
                (&state.mass, &state.v0, &state.solids),
                n,
            )
            .expect("an elastic body");
            let k = system.n;
            let (squared, phi, _) =
                crate::spacetime::total_lagrangian::eigen_pairs(system.m.clone(), &system.k, k)
                    .expect("definite");
            let column = |e: usize| -> Vec<f64> { (0..k).map(|r| phi[r * k + e]).collect() };
            let times = |a: &[f64], x: &[f64]| -> Vec<f64> {
                (0..k)
                    .map(|r| (0..k).map(|c| a[r * k + c] * x[c]).sum())
                    .collect()
            };
            let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
            let norm_k = dot(&system.k, &system.k).sqrt();
            let (mut residual, mut rigid_overlap) = (0.0f64, 0.0f64);
            let rigid: Vec<Vec<f64>> = (0..3).map(column).collect();
            for (e, &w2) in squared.iter().enumerate().skip(3) {
                let x = column(e);
                let (kx, mx) = (times(&system.k, &x), times(&system.m, &x));
                let r: f64 = kx
                    .iter()
                    .zip(&mx)
                    .map(|(a, b)| (a - w2 * b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                residual = residual.max(r / (norm_k * dot(&x, &x).sqrt()));
                for g in &rigid {
                    rigid_overlap = rigid_overlap.max(dot(g, &mx).abs());
                }
            }
            println!(
                "{name}: one flown frame: L {before:.3e} -> {after:.3e}, change {:.3e}; sum_i leak_i q'_i {signed:.3e}; eigen residual at most {residual:.3e} of |K|, rigid overlap phi_r^T M phi_i at most {rigid_overlap:.3e}",
                after - before
            );
            println!(
                "{name}: {} modes (largest {:.0} Hz, {} above 3 kHz); largest leak over sqrt(I) {worst:.3e}; sum |leak_i| |q'_i| {predicted:.3e} against the allowance 2 n eps sum m |r| (|v| + |v_c|) {:.3e}",
                modes.omega.len(),
                modes.omega.iter().fold(0.0f64, |a, &w| a.max(w)) / std::f64::consts::TAU,
                modes
                    .omega
                    .iter()
                    .filter(|&&w| w / std::f64::consts::TAU > 3000.0)
                    .count(),
                2.0 * n as f64 * f64::EPSILON * size
            );
        }
    });
}

/// Each flown frame of `tests/modal_flight.rs`'s drop against the step
/// replayed from the same start state: the particles' position difference
/// (largest and RMS, m) and the energy plus heat difference (J/m); and, at
/// each frame a flight hands back, which check refused it. A probe.
///
/// The reference is the step at the drop's tolerance, 1e-1, not an exact
/// solution: its own error is not separated from the flight's, and at that
/// tolerance a body is known to hop at the frame rate after impacts (the
/// Plasticine block, `tests/zener.rs`'s `probe_plasticine_block_drop`).
/// Measured: the first flight agrees to 1e-14 m; after the impacts the
/// flown frames differ by RMS 7e-5 to 1.4e-4 m per frame.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_flown_frames_against_the_step() {
    super::cap::capped("probe_flown_frames_against_the_step", 1800, || {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1).with_modal_flight());
        let props = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        let length = f64::from(config.dt);
        for k in 1..=45 {
            let before = sim.total_lagrangian.clone();
            sim.step();
            let frame = sim.total_lagrangian_frame().expect("on");
            let Some(start) = before else { continue };
            // A hand-back: which check the flight under way failed.
            if !frame.modal && frame.contacts == 0 {
                if let (Some(flight), Some(Some(modes))) = (&start.flight, &start.modes) {
                    let gravity = sim.total_lagrangian_gravity();
                    let planes = sim.total_lagrangian_walls().0;
                    let clear = flight.clear(gravity, &planes, (start.time, start.time + length));
                    let mut trial = start.clone();
                    let mut f = flight.clone();
                    let rebuilt = f.rebuild(modes, gravity, start.time + length, &mut trial);
                    println!(
                        "frame {k}: handed back with no contact: clear {clear}, rebuilt {rebuilt}, keeps L {}",
                        f.keeps_angular_momentum(&trial)
                    );
                } else {
                    println!("frame {k}: stepped with no contact, no flight under way");
                }
            }
            if !frame.modal {
                continue;
            }
            let flown = sim.total_lagrangian.take().expect("on");
            sim.total_lagrangian = Some(start);
            let mut stepped_config = config;
            stepped_config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
            sim.config = stepped_config;
            sim.step();
            let replay = sim.total_lagrangian_frame().expect("on");
            let stepped = sim.total_lagrangian.take().expect("on");
            sim.config = config;
            let n = flown.particles;
            let (mut largest, mut squared) = (0.0f64, 0.0f64);
            for p in 0..n {
                let d = (flown.x[p] - stepped.x[p]).length();
                largest = largest.max(d);
                squared += d * d;
            }
            println!(
                "frame {k}: flown against stepped from the same start: positions largest {largest:.3e} m, RMS {:.3e} m; energy+heat {:+.3e} J/m (flown {:.6e}, stepped {:.6e}); stepped contacts {}",
                (squared / n as f64).sqrt(),
                frame.energy + frame.heat - replay.energy - replay.heat,
                frame.energy + frame.heat,
                replay.energy + replay.heat,
                replay.contacts
            );
            sim.total_lagrangian = Some(flown);
        }
    });
}
