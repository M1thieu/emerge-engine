//! A lone body's flights between contacts carried by its modes
//! (`TotalLagrangianConfig::with_modal_flight`): the clay block of
//! `tests/realtime_profile.rs`'s E2 dropped on the floor, every contact
//! stepped, every flight frame flown with no step.

extern crate emerge_engine as emerge;

#[path = "common/cap.rs"]
mod cap;

use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// Medium clay, as the path's tests use it.
fn clay() -> Elastic {
    Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    }
}

/// The 8 x 4 cm block dropped `lift` cells, its flights modal.
fn drop(lift: f32) -> Simulation {
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1).with_modal_flight());
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + lift + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&clay(), &config),
    );
    sim
}

#[test]
fn a_dropped_block_flies_by_its_modes_between_impacts() {
    let mut sim = drop(10.0);
    let unit = f64::from(sim.config().grid_mass_unit_kg_per_m());
    let mass: f64 = sim
        .particles()
        .mass
        .iter()
        .map(|&m| f64::from(m))
        .sum::<f64>()
        * unit;
    let gravity = f64::from(sim.config().gravity.y) * f64::from(sim.config().dx_meters);
    let dt = f64::from(sim.config().dt);
    let count = sim.particles().len() as f64;
    let (mut modal, mut stepped, mut handed_back) = (0usize, 0usize, 0usize);
    let (mut modal_wall, mut stepped_wall) = (0.0f64, 0.0f64);
    let mut allowed = 0.0f64;
    let mut worst_rigid = 0.0f64;
    let mut previous: Option<
        emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianFrame,
    > = None;
    for _ in 0..30 {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        // A frame that held a contact is never modal.
        assert!(
            !(frame.modal && frame.contacts > 0),
            "a modal frame held a contact"
        );
        allowed += frame.residual_work + frame.contact_work_bound + frame.modal_energy_change.abs();
        let round_off = count
            * f64::EPSILON
            * (frame.parts.iter().map(|e| e.abs()).sum::<f64>()
                + frame.heat.abs()
                + frame.start_energy.abs());
        assert!(
            frame.balance.abs() <= allowed + round_off,
            "the balance {:e} J/m exceeds what the solves and the modes allow, {:e} J/m",
            frame.balance,
            allowed + round_off
        );
        if frame.modal {
            modal += 1;
            modal_wall += frame.wall;
            // No contact acts: the momentum changes by `M g dt`, the elastic
            // modes carrying none. They do so to the eigen-solve's accuracy
            // (their mass-orthogonality to the translations, about 5e-14),
            // so the check is relative, at a declared 1e-9.
            if let Some(last) = previous.filter(|p| p.modal) {
                let change = frame.momentum.y - last.momentum.y;
                let expected = mass * gravity * dt;
                let off = (change - expected).abs() / expected.abs();
                worst_rigid = worst_rigid.max(off);
                assert!(
                    off <= 1e-9,
                    "a modal frame's momentum changed by {change:e}, not M g dt = {expected:e}"
                );
            }
        } else {
            stepped += 1;
            stepped_wall += frame.wall;
            if previous.is_some_and(|p| p.modal) {
                handed_back += 1;
            }
        }
        previous = Some(frame);
    }
    println!(
        "modal flight: {modal} modal frames ({:.3} ms each), {stepped} stepped ({:.0} ms each), {handed_back} hand-backs to the step; the modal frames' momentum off M g dt by at most {worst_rigid:.2e} of it",
        1e3 * modal_wall / modal.max(1) as f64,
        1e3 * stepped_wall / stepped.max(1) as f64
    );
    assert!(
        modal >= 10,
        "the flights were not modal: {modal} modal frames"
    );
    assert!(handed_back >= 1, "no flight ended at a contact");
}

/// The same scene with or without modal flight.
fn scene(lift: f32, modal: bool) -> Simulation {
    let mut sim = drop(lift);
    let mut config = *sim.config();
    if !modal {
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let mut stepped = Simulation::empty(config);
        let id = stepped
            .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = stepped.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + lift + 2.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&clay(), &config),
        );
        sim = stepped;
    }
    sim
}

/// The block's mean height and vertical speed, cells and cells/s, and
/// three particles' velocities (nearest the top centre, the top right
/// corner, the right side's middle).
fn observe(sim: &Simulation, sensors: &[usize; 3]) -> (f64, f64, [Vec2; 3]) {
    let p = sim.particles();
    let total: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
    let (y, vy) = (0..p.len()).fold((0.0, 0.0), |(y, vy), i| {
        let m = f64::from(p.mass[i]);
        (y + m * f64::from(p.x[i].y), vy + m * f64::from(p.v[i].y))
    });
    (y / total, vy / total, sensors.map(|i| p.v[i]))
}

fn sensors(sim: &Simulation) -> [usize; 3] {
    let p = sim.particles();
    let (lo, hi) = p.x.iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(lo, hi), &q| (lo.min(q), hi.max(q)),
    );
    let centre = 0.5 * (lo + hi);
    [Vec2::new(centre.x, hi.y), hi, Vec2::new(hi.x, centre.y)].map(|target| {
        (0..p.len())
            .min_by(|&a, &b| {
                (p.x[a] - target)
                    .length()
                    .total_cmp(&(p.x[b] - target).length())
            })
            .expect("particles")
    })
}

/// The modal run against the stepped run on the same drop, on what a
/// viewer sees: the rebounds' heights and speeds, the second impact's
/// frame, the sensors' velocities, the energy and heat. A probe.
#[test]
#[ignore = "probe: run with --ignored --nocapture"]
fn probe_modal_against_stepped_drop() {
    cap::capped("probe_modal_against_stepped_drop", 1800, || {
        let (mut modal, mut stepped) = (scene(10.0, true), scene(10.0, false));
        let sensors = sensors(&stepped);
        let (mut sensor_error, mut sensor_size) = ([0.0f64; 3], [0.0f64; 3]);
        println!(
            "frame | stepped: y cells, vy cells/s, contacts, energy+heat J/m | modal: same, modal?"
        );
        for k in 1..=45 {
            stepped.step();
            modal.step();
            let (fs, fm) = (
                stepped.total_lagrangian_frame().expect("on"),
                modal.total_lagrangian_frame().expect("on"),
            );
            let (ys, vys, ss) = observe(&stepped, &sensors);
            let (ym, vym, sm) = observe(&modal, &sensors);
            for s in 0..3 {
                sensor_error[s] += f64::from((sm[s] - ss[s]).length_squared());
                sensor_size[s] += f64::from(ss[s].length_squared());
            }
            println!(
                "{k:5} | {ys:8.4} {vys:+9.3} {:2} {:.6e} | {ym:8.4} {vym:+9.3} {:2} {:.6e} {}",
                fs.contacts,
                fs.energy + fs.heat,
                fm.contacts,
                fm.energy + fm.heat,
                if fm.modal { "modal" } else { "stepped" }
            );
        }
        for s in 0..3 {
            println!(
                "sensor {s}: modal against stepped, RMS velocity difference {:.3e} of the stepped RMS",
                (sensor_error[s] / sensor_size[s]).sqrt()
            );
        }
        let f = modal.total_lagrangian_frame().expect("on");
        println!(
            "modal run: {} modal frames, {} hand-backs",
            f.modal_frames, f.hand_backs
        );
    });
}

/// A block spun at `spin` rad/s in free flight, modal against stepped: the
/// oscillators leave out the turning frame's Coriolis and centrifugal
/// terms; this measures what that costs where the entry rule lets a
/// turning body fly. A probe.
#[test]
#[ignore = "probe: run with --ignored --nocapture"]
fn probe_spinning_flight() {
    cap::capped("probe_spinning_flight", 1800, || {
        for spin in [10.0f32, 50.0] {
            let (mut modal, mut stepped) = (scene(25.0, true), scene(25.0, false));
            for sim in [&mut modal, &mut stepped] {
                let p = sim.particles();
                let total: f32 = p.mass.iter().sum();
                let centre = (0..p.len()).fold(Vec2::ZERO, |c, i| c + p.mass[i] * p.x[i]) / total;
                let x = p.x.clone();
                for (v, q) in sim.particles_mut().v.iter_mut().zip(&x) {
                    *v = spin * (*q - centre).perp();
                }
            }
            println!("spin {spin} rad/s:");
            for k in 1..=10 {
                stepped.step();
                modal.step();
                let (fs, fm) = (
                    stepped.total_lagrangian_frame().expect("on"),
                    modal.total_lagrangian_frame().expect("on"),
                );
                println!(
                    "  frame {k:2}: stepped L {:.6e}, stored {:.4e} J/m, energy {:.6e}; modal L {:.6e}, stored {:.4e}, energy {:.6e}, {}",
                    fs.angular_momentum,
                    fs.parts[1],
                    fs.energy + fs.heat,
                    fm.angular_momentum,
                    fm.parts[1],
                    fm.energy + fm.heat,
                    if fm.modal { "modal" } else { "stepped" }
                );
            }
        }
    });
}
