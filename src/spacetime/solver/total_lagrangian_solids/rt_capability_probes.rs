//! The capability scorecard in the real-time configuration
//! (tmp/papers/dl_realtime/FAST_MODE_PLAN.md sec. 19, measurement worktree
//! only): rows P1 and P3-P11 of the precise path's capabilities, each in
//! its integration test's own scene (the same laws, records, bodies in
//! metres and frames), run either in the test's own configuration or in
//! the declared real-time one, by `EMERGE_CAP_CONFIG`:
//! - `precise`: the test's cell, tolerance and flight setting;
//! - `rt`: tolerance 1e-1, the sub-frame handoff, the fixed contact step
//!   and the single solve (`rt_knobs`), and two cells through the body's
//!   thinnest dimension (`cell`), the body's size in metres unchanged.
//!
//! Each probe prints one `CAP` line of raw quantities; the kept fractions
//! are computed from them outside (target/scratch_harness). In-crate
//! because the real-time knobs exist only in the library's test build; a
//! child of the ladder harness (its parent file is at its size limit).

use crate::materials::physical_props::Elastic;
use crate::materials::relaxation::RelaxationSpectrum;
use crate::materials::solid::viscoplastic_records::ViscoplasticRecord;
use crate::spacetime::solver::total_lagrangian_solids::{
    TotalLagrangianConfig, TotalLagrangianFrame,
};
use crate::{
    CorotatedMaterial, DuctileProps, FrictionBoundary, FromSI, SimConfig, Simulation, SpawnRegion,
    VonMisesMaterial,
};
use glam::{IVec2, Vec2};

/// Whether the run is in the real-time configuration (`EMERGE_CAP_CONFIG
/// = rt`, or `rt_rule` with the contact step rule of sec. 21 in place of
/// k) or the test's own (`precise`); anything else stops the run.
fn rt() -> bool {
    match std::env::var("EMERGE_CAP_CONFIG").as_deref() {
        Ok(config @ ("rt" | "rt_rule")) => {
            rt_knobs(config == "rt_rule");
            true
        }
        Ok("precise") => false,
        other => panic!("EMERGE_CAP_CONFIG {other:?}: precise, rt or rt_rule"),
    }
}

/// The real-time configuration's knobs, checked, not set (sec. 17): the
/// fixed step k = 39.33, or with `rule` the contact step rule at one of the
/// N fixed in sec. 21 and no k.
fn rt_knobs(rule: bool) {
    let fixed = std::env::var("EMERGE_PROBE_CONTACT_FIXED_STEP").ok();
    let n = std::env::var("EMERGE_PROBE_CONTACT_STEP_RULE").ok();
    let step = if rule {
        fixed.is_none() && matches!(n.as_deref(), Some("18" | "40"))
    } else {
        fixed.as_deref() == Some("39.33") && n.is_none()
    };
    assert!(
        step && std::env::var_os("EMERGE_PROBE_SINGLE_SOLVE").is_some()
            && std::env::var_os("EMERGE_PROBE_BACKWARD_EULER").is_none(),
        "rt needs EMERGE_PROBE_CONTACT_FIXED_STEP=39.33 (rt_rule: EMERGE_PROBE_CONTACT_STEP_RULE=18 or 40 instead) and EMERGE_PROBE_SINGLE_SOLVE, backward Euler unset"
    );
}

/// The path: the test's own (`tolerance`, modal flight when `modal`), or
/// the real-time one.
fn path(tolerance: f64, modal: bool) -> TotalLagrangianConfig {
    if rt() {
        TotalLagrangianConfig::new(1e-1).with_subframe_handoff()
    } else if modal {
        TotalLagrangianConfig::new(tolerance).with_modal_flight()
    } else {
        TotalLagrangianConfig::new(tolerance)
    }
}

/// The cell, m: the test's `precise`, or half the body's `thinnest`
/// dimension (two cells through it) in the real-time configuration.
fn cell(precise: f32, thinnest: f32) -> f32 {
    if rt() { 0.5 * thinnest } else { precise }
}

fn label() -> &'static str {
    match (
        rt(),
        std::env::var_os("EMERGE_PROBE_CONTACT_STEP_RULE").is_some(),
    ) {
        (true, true) => "rt_rule",
        (true, false) => match std::env::var("EMERGE_PROBE_REFUSAL_PHASE").as_deref() {
            Ok("angular") => "rt_c2a",
            Ok(_) => "rt_c2",
            Err(_) => "rt",
        },
        (false, _) => "precise",
    }
}

fn clay() -> Elastic {
    Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    }
}

/// A `size` (m) box of `props` centred at `x` (m) with its bottom `lift`
/// (m) above the floor.
fn box_at(
    config: &SimConfig,
    id: u32,
    props: &Elastic,
    size: (f32, f32),
    x: f32,
    lift: f32,
) -> SpawnRegion {
    let dx = config.dx_meters;
    let (w, h) = ((size.0 / dx).round() as i32, (size.1 / dx).round() as i32);
    let floor = config.boundary_thickness as f32;
    SpawnRegion::for_sim(config)
        .box_of(IVec2::new(w, h))
        .at(Vec2::new(x / dx, floor + lift / dx + 0.5 * h as f32))
        .spacing(0.5)
        .material(id)
        .mass_from(props, config)
}

/// A clay scene on the floor in a `domain` (m) square: the config at the
/// cell for a body `thinnest` m thick, the clay registered.
fn clay_scene(domain: f32, thinnest: f32, modal: bool) -> (SimConfig, Simulation, u32) {
    let dx = cell(0.01, thinnest);
    let mut config = SimConfig::earth((domain / dx).round() as usize, dx, 1.0 / 60.0);
    config.total_lagrangian = Some(path(1e-1, modal));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    (config, sim, id)
}

/// The particles' mass-weighted mean position, m, of those `keep` selects.
fn centroid(sim: &Simulation, keep: &dyn Fn(usize) -> bool) -> (f64, f64) {
    let p = sim.particles();
    let dx = f64::from(sim.config().dx_meters);
    let (mut m, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
    for i in (0..p.len()).filter(|&i| keep(i)) {
        let mass = f64::from(p.mass[i]);
        m += mass;
        x += mass * f64::from(p.x[i].x);
        y += mass * f64::from(p.x[i].y);
    }
    (x / m * dx, y / m * dx)
}

/// The mass-weighted mean velocity, m/s, of the particles `keep` selects.
fn mean_velocity(sim: &Simulation, keep: &dyn Fn(usize) -> bool) -> (f64, f64) {
    let p = sim.particles();
    let dx = f64::from(sim.config().dx_meters);
    let (mut m, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
    for i in (0..p.len()).filter(|&i| keep(i)) {
        let mass = f64::from(p.mass[i]);
        m += mass;
        x += mass * f64::from(p.v[i].x);
        y += mass * f64::from(p.v[i].y);
    }
    (x / m * dx, y / m * dx)
}

/// `1/2 M v^2` of the whole scene at speed `v` (m/s), J/m.
fn kinetic_at(sim: &Simulation, v: f64) -> f64 {
    let unit = f64::from(sim.config().grid_mass_unit_kg_per_m());
    let mass: f64 = sim
        .particles()
        .mass
        .iter()
        .map(|&m| f64::from(m))
        .sum::<f64>()
        * unit;
    0.5 * mass * v * v
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p1_rest() {
    let (config, mut sim, id) = clay_scene(0.40, 0.04, false);
    let _block = sim.add_body(box_at(&config, id, &clay(), (0.08, 0.04), 0.20, 0.0));
    let floor = f64::from(config.boundary_thickness as f32 * config.dx_meters);
    let start = centroid(&sim, &|_| true).1;
    let (mut worst, mut deepest) = (0.0f64, 0.0f64);
    for _ in 0..30 {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        worst = worst.max(f.balance.abs() / f.start_energy.abs());
        let dx = f64::from(config.dx_meters);
        let low = sim
            .particles()
            .x
            .iter()
            .map(|q| f64::from(q.y) * dx)
            .fold(f64::INFINITY, f64::min);
        deepest = deepest.max(floor - low);
    }
    println!(
        "CAP P1 {} dx {} sag_m {:.6e} deepest_m {:.6e} worst_balance_over_e0 {:.3e}",
        label(),
        config.dx_meters,
        start - centroid(&sim, &|_| true).1,
        deepest.max(0.0),
        worst
    );
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p3_modal_flight() {
    let (config, mut sim, id) = clay_scene(0.40, 0.04, true);
    let _block = sim.add_body(box_at(&config, id, &clay(), (0.08, 0.04), 0.20, 0.10));
    let unit = f64::from(config.grid_mass_unit_kg_per_m());
    let mass: f64 = sim
        .particles()
        .mass
        .iter()
        .map(|&m| f64::from(m))
        .sum::<f64>()
        * unit;
    // In f64 throughout, as tests/modal_flight.rs forms it: the f32 product
    // `gravity.y * dx` alone errs by 2.9e-9 of it (measured, first run).
    let expected =
        mass * f64::from(config.gravity.y) * f64::from(config.dx_meters) * f64::from(config.dt);
    let (mut modal, mut after, mut touched, mut off) = (0u32, 0u32, false, 0.0f64);
    let mut previous: Option<TotalLagrangianFrame> = None;
    for _ in 0..30 {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        if f.modal {
            modal += 1;
            after += u32::from(touched);
            if let Some(last) = previous.as_ref().filter(|p| p.modal) {
                off = off.max(((f.momentum.y - last.momentum.y) - expected).abs() / expected.abs());
            }
        }
        touched |= f.contacts > 0;
        previous = Some(f);
    }
    println!(
        "CAP P3 {} dx {} modal_frames {modal} modal_after_contact {after} rigid_off {off:.3e}",
        label(),
        config.dx_meters
    );
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p4_slide() {
    let (config, mut sim, id) = clay_scene(0.40, 0.04, false);
    let _block = sim.add_body(box_at(&config, id, &clay(), (0.08, 0.04), 0.10, 0.0));
    sim.add_boundary_condition(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.3,
    )));
    let speed = 1.0 / config.dx_meters;
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(speed, 0.0);
    }
    let kinetic0 = kinetic_at(&sim, 1.0);
    let start = centroid(&sim, &|_| true).0;
    for _ in 0..30 {
        sim.step();
    }
    let f = sim.total_lagrangian_frame().expect("the path is on");
    println!(
        "CAP P4 {} dx {} travel_m {:.6e} heat_over_kinetic {:.6} end_speed_m_s {:.3e} worst_balance_over_e0 {:.3e}",
        label(),
        config.dx_meters,
        centroid(&sim, &|_| true).0 - start,
        f.heat / kinetic0,
        mean_velocity(&sim, &|_| true).0,
        f.balance.abs() / f.start_energy.abs()
    );
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p5_collision() {
    let (config, mut sim, id) = clay_scene(0.40, 0.04, false);
    let left = sim.add_body(box_at(&config, id, &clay(), (0.06, 0.04), 0.08, 0.0));
    let _right = sim.add_body(box_at(&config, id, &clay(), (0.06, 0.04), 0.18, 0.0));
    let tags = sim.particles().user_tag.clone();
    let speed = 0.25 / config.dx_meters;
    for (v, &t) in sim.particles_mut().v.iter_mut().zip(&tags) {
        if t == left {
            *v = Vec2::new(speed, 0.0);
        }
    }
    for _ in 0..30 {
        sim.step();
    }
    let f = sim.total_lagrangian_frame().expect("the path is on");
    let (vl, vr) = (
        mean_velocity(&sim, &|i| tags[i] == left).0,
        mean_velocity(&sim, &|i| tags[i] != left).0,
    );
    println!(
        "CAP P5 {} dx {} restitution {:.6} v_left {vl:.5e} v_right {vr:.5e} momentum_drift_rel {:.3e} worst_balance_over_e0 {:.3e}",
        label(),
        config.dx_meters,
        (vr - vl) / 0.25,
        (f.momentum.x - f.start_momentum.x).abs() / f.start_momentum.x.abs(),
        f.balance.abs() / f.start_energy.abs()
    );
}

/// The slab tests' centre of mass above the bottom row's mean height, m.
fn slab_centre(sim: &Simulation, bottom: &[bool]) -> f64 {
    let p = sim.particles();
    let dx = f64::from(sim.config().dx_meters);
    let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
    let mean: f64 =
        p.x.iter()
            .zip(&p.mass)
            .map(|(x, &m)| f64::from(x.y) * f64::from(m))
            .sum::<f64>()
            / mass;
    let (sum, count) =
        p.x.iter()
            .zip(bottom)
            .filter(|(_, b)| **b)
            .fold((0.0f64, 0.0f64), |(s, c), (x, _)| {
                (s + f64::from(x.y), c + 1.0)
            });
    (mean - sum / count) * dx
}

/// The bottom row (within a quarter cell of the lowest) and its height
/// above the floor, m.
fn bottom_row(sim: &Simulation) -> (Vec<bool>, f64) {
    let lowest = sim
        .particles()
        .x
        .iter()
        .map(|x| x.y)
        .fold(f32::INFINITY, f32::min);
    let floor = sim.config().boundary_thickness as f32;
    let bottom = sim
        .particles()
        .x
        .iter()
        .map(|x| x.y < lowest + 0.25)
        .collect();
    (bottom, f64::from((lowest - floor) * sim.config().dx_meters))
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p6_settled_slab() {
    // tests/settled_slab.rs: E 1e5 Pa, nu 0, 1000 kg/m^3, 0.30 x 0.12 m,
    // the dashpot eta 400 Pa s (bulk 2 eta / 3); precise at its 2 cm.
    let props = Elastic {
        e_pa: 1.0e5,
        nu: 0.0,
        rho_kg_m3: 1000.0,
    };
    let dx = cell(0.02, 0.12);
    let cells = |m: f32| (m / dx).round() as i32;
    // The domain widened, if 0.48 m does not hold the slab inside the walls.
    let res = ((0.48 / dx).round() as i32).max(cells(0.30) + 2 * 2 + 2) as usize;
    let mut config = SimConfig::earth(res, dx, 1.0 / 60.0);
    config.total_lagrangian = Some(path(1e-1, false));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.elastic_viscosity = config.visc_from_si(400.0);
    material.elastic_bulk_viscosity = config.visc_from_si(2.0 * 400.0 / 3.0);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _slab = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(cells(0.30), cells(0.12)))
            .at(Vec2::new(
                0.5 * res as f32,
                floor + 0.5 * cells(0.12) as f32,
            ))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let (bottom, y_b) = bottom_row(&sim);
    let start = slab_centre(&sim, &bottom);
    let mut worst = 0.0f64;
    for _ in 0..30 {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        worst = worst.max(f.balance.abs() / f.start_energy.abs());
    }
    let drop = start - slab_centre(&sim, &bottom);
    let h = f64::from(cells(0.12) as f32 * dx);
    let exact = 1000.0 * 9.81 / 1.0e5 * (h * h / 3.0 - (h * y_b - 0.5 * y_b * y_b));
    println!(
        "CAP P6 {} dx {dx} res {res} compression_over_exact {:.6} worst_balance_over_e0 {worst:.3e}",
        label(),
        drop / exact
    );
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p7_plastic_slab() {
    // tests/plastic_slab.rs: E 1e6 Pa, nu 0.3, 1000 kg/m^3, yield 300 Pa,
    // 0.12 m high filling a 0.48 m domain's width; precise at its 2 cm.
    let (e, nu, rho, yield_pa, height, g) =
        (1.0e6f64, 0.3f64, 1000.0f64, 300.0f64, 0.12f64, 9.81f64);
    let shear = e / (2.0 * (1.0 + nu));
    let bulk = e / (3.0 * (1.0 - 2.0 * nu));
    let modulus = bulk + 4.0 * shear / 3.0;
    let onset = modulus * yield_pa / (2.0 * shear);
    let strain = |y: f64| -> f64 {
        let load = rho * g * (height - y);
        if load <= onset {
            load / modulus
        } else {
            (load - 2.0 * yield_pa / 3.0) / bulk
        }
    };
    let knee = (height - onset / (rho * g)).clamp(0.0, height);
    let piece = |a: f64, b: f64| -> f64 {
        if b <= a {
            0.0
        } else {
            (b - a) / 6.0 * (strain(a) + 4.0 * strain(0.5 * (a + b)) + strain(b))
        }
    };
    let between = |a: f64, b: f64| -> f64 {
        let (lo, hi, sign) = if a <= b { (a, b, 1.0) } else { (b, a, -1.0) };
        let split = knee.clamp(lo, hi);
        sign * (piece(lo, split) + piece(split, hi))
    };
    let exact_drop = |y_b: f64| -> f64 {
        let n = 20000;
        let step = height / n as f64;
        (0..n)
            .map(|i| {
                let (a, b) = (i as f64 * step, (i + 1) as f64 * step);
                step / 6.0 * (between(y_b, a) + 4.0 * between(y_b, 0.5 * (a + b)) + between(y_b, b))
            })
            .sum::<f64>()
            / height
    };
    let props = DuctileProps {
        elastic: Elastic {
            e_pa: e as f32,
            nu: nu as f32,
            rho_kg_m3: rho as f32,
        },
        yield_stress_pa: yield_pa as f32,
    };
    let dx = cell(0.02, 0.12);
    let res = (0.48 / f64::from(dx)).round() as usize;
    let mut config = SimConfig::earth(res, dx, 1.0 / 60.0);
    config.total_lagrangian = Some(path(1e-4, false));
    let wall = config.boundary_thickness as i32;
    let mut material = VonMisesMaterial::from_physical(&props, &config);
    let omega = std::f64::consts::PI * (modulus / rho).sqrt() / (2.0 * height);
    material.elastic_viscosity = config.visc_from_si((3.0 * 2.0 * modulus / omega * 0.75) as f32);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let cells = (height / f64::from(dx)).round() as i32;
    let _slab = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(res as i32 - 2 * wall, cells))
            .at(Vec2::new(
                res as f32 * 0.5,
                wall as f32 + 0.5 * cells as f32,
            ))
            .spacing(0.5)
            .material(id)
            .mass_from(&props.elastic, &config),
    );
    let (bottom, y_b) = bottom_row(&sim);
    let start = slab_centre(&sim, &bottom);
    let mut plastic = 0.0f64;
    for _ in 0..30 {
        sim.step();
        plastic += sim
            .total_lagrangian_frame()
            .expect("the path is on")
            .plastic_work;
    }
    let drop = start - slab_centre(&sim, &bottom);
    println!(
        "CAP P7 {} dx {dx} compression_over_exact {:.6} plastic_work {plastic:.4e}",
        label(),
        drop / exact_drop(y_b)
    );
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p8_norton() {
    // tests/norton.rs: the record's block at E 30 MPa, its stand-in density.
    let rho = 119.0 * 0.453_592_37 / (0.3048 * 0.3048 * 0.3048);
    let e_pa = 30.0e6;
    let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
    let dx = cell(0.01, 0.04);
    let mut config = SimConfig::earth((0.40 / dx).round() as usize, dx, 1.0 / 60.0);
    config.total_lagrangian = Some(path(1e-2, false));
    let material = record.material((e_pa, rho), &config);
    let props = Elastic {
        e_pa: e_pa as f32,
        nu: record.nu as f32,
        rho_kg_m3: rho as f32,
    };
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let _block = sim.add_body(box_at(&config, id, &props, (0.08, 0.04), 0.20, 0.10));
    let (mut plastic, mut rebound, mut touched) = (0.0f64, 0.0f64, false);
    for _ in 0..30 {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        plastic += f.plastic_work;
        touched |= f.contacts > 0;
        if touched {
            rebound = rebound.max(mean_velocity(&sim, &|_| true).1);
        }
    }
    let f = sim.total_lagrangian_frame().expect("the path is on");
    println!(
        "CAP P8 {} dx {dx} plastic_work {plastic:.6e} heat {:.6e} rebound_m_s {rebound:.6e} worst_balance_over_e0 {:.3e}",
        label(),
        f.heat,
        f.balance.abs() / f.start_energy.abs()
    );
}

/// The damped period and logarithmic decrement per period of `samples`
/// every `dt` (the Zener and Kelvin-Voigt tests' peak reading).
fn period_and_decrement(samples: &[f64], dt: f64) -> (f64, f64) {
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for k in 1..samples.len() - 1 {
        let (a, b, c) = (samples[k - 1].abs(), samples[k].abs(), samples[k + 1].abs());
        if b >= a && b > c {
            let curvature = a - 2.0 * b + c;
            let shift = if curvature != 0.0 {
                0.5 * (a - c) / curvature
            } else {
                0.0
            };
            peaks.push(((k as f64 + shift) * dt, b - 0.25 * (a - c) * shift));
        }
    }
    assert!(peaks.len() >= 5, "too few peaks: {peaks:?}");
    let ((t0, a0), (t4, a4)) = (peaks[0], peaks[4]);
    ((t4 - t0) / 2.0, (a0 / a4).ln() / 2.0)
}

/// A 0.32 x 0.02 m free bar of `props` (`nu` 0), released in its first
/// axial mode at `speed` m/s with the dashpot `eta` or the branches
/// `relaxation`, `frames` frames of `frame_s` at the test's `tolerance` and
/// step floor `min_dt`: the modal amplitude per frame.
fn ring(
    props: &Elastic,
    (eta, relaxation): (f32, Option<RelaxationSpectrum>),
    (speed, frame_s, tolerance, min_dt): (f32, f32, f64, f32),
    frames: usize,
) -> Vec<f64> {
    let dx = cell(0.01, 0.02);
    let mut config = SimConfig::earth((0.48 / dx).round() as usize, dx, frame_s);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(min_dt);
    config.total_lagrangian = Some(path(tolerance, false));
    let mut material = CorotatedMaterial::from_physical(props, &config);
    material.elastic_viscosity = config.visc_from_si(eta);
    material.elastic_bulk_viscosity = config.visc_from_si(2.0 * eta / 3.0);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let (x0, l) = ((0.08 / dx).round(), (0.32 / dx).round());
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(l as i32, (0.02 / dx).round() as i32))
            .at(Vec2::new(x0 + 0.5 * l, (0.24 / dx).round()))
            .spacing(0.5)
            .material(id)
            .mass_from(props, &config),
    );
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        *v = Vec2::new(
            speed / dx * (std::f32::consts::PI * (x.x - x0) / l).cos(),
            0.0,
        );
    }
    let amplitude = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let shape = f64::from((std::f32::consts::PI * (p.x[i].x - x0) / l).cos());
            num += f64::from(p.mass[i]) * f64::from(p.v[i].x) * shape;
            den += f64::from(p.mass[i]) * shape * shape;
        }
        num / den
    };
    let mut samples = vec![amplitude(&sim)];
    for _ in 0..frames {
        sim.step();
        samples.push(amplitude(&sim));
    }
    samples
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p9_kelvin_voigt_bar() {
    // tests/kelvin_voigt_bar.rs: E 1e5 Pa, nu 0, 1000 kg/m^3, eta 50 Pa s,
    // 1 cm/s, 2 ms frames, tolerance 1e-2, 100 frames (its bar sits at a
    // 0.48 m domain's 0.08-0.40 m, as here).
    let props = Elastic {
        e_pa: 1.0e5,
        nu: 0.0,
        rho_kg_m3: 1000.0,
    };
    let samples = ring(&props, (50.0, None), (0.01, 2.0e-3, 1e-2, 1e-7), 100);
    let (period, decrement) = period_and_decrement(&samples, 2.0e-3);
    let omega = 2.0 * std::f64::consts::PI / period;
    let zeta = decrement / (4.0 * std::f64::consts::PI.powi(2) + decrement * decrement).sqrt();
    let exact = 50.0 * omega / 1.0e5;
    let omega_bar = std::f64::consts::PI * (1.0e5f64 / 1000.0).sqrt() / 0.32;
    println!(
        "CAP P9 {} zeta_over_exact {:.6} omega_over_bar {:.6}",
        label(),
        zeta / exact,
        omega / omega_bar
    );
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p10_zener_bar() {
    // tests/zener.rs's ringing gate: E 1e5 Pa, nu 0, 1000 kg/m^3, Plasticine
    // (Crandall et al. 1970), 5e-5 m/s, 2 ms frames, tolerance 1e-2, 100
    // frames; the exact mode's period and decrement are the test's own print.
    let props = Elastic {
        e_pa: 1.0e5,
        nu: 0.0,
        rho_kg_m3: 1000.0,
    };
    let free = ring(&props, (0.0, None), (5.0e-5, 2.0e-3, 1e-2, 1e-9), 100);
    let damped = ring(
        &props,
        (0.0, Some(RelaxationSpectrum::plasticine_crandall_1970())),
        (5.0e-5, 2.0e-3, 1e-2, 1e-9),
        100,
    );
    let (free_period, _) = period_and_decrement(&free, 2.0e-3);
    let (period, decrement) = period_and_decrement(&damped, 2.0e-3);
    println!(
        "CAP P10 {} free_period {free_period:.6e} damped_period {period:.6e} decrement {decrement:.6e}",
        label()
    );
}

#[test]
#[ignore = "capability probe: EMERGE_CAP_CONFIG=precise|rt, --ignored --nocapture"]
fn cap_p11_wave_transit() {
    // tests/wave_transit_density.rs: a 0.48 x 0.02 m clay-stiffness bar, its
    // left half at 1 cm/s, the front timed at a probe 0.10 m right of the
    // middle where its mean velocity crosses a quarter of that; 2.5e-5 s
    // frames, tolerance 1e-1.
    let mut worst = 0.0f64;
    for nu in [0.3f32, 0.0] {
        for rho in [1000.0f32, 1700.0, 2700.0] {
            let props = Elastic {
                e_pa: 1.125e7,
                nu,
                rho_kg_m3: rho,
            };
            let dx = cell(0.01, 0.02);
            let mut config = SimConfig::earth((0.64 / dx).round() as usize, dx, 2.5e-5);
            config.gravity = Vec2::ZERO;
            config.min_dt = config.min_dt.min(1e-7);
            config.total_lagrangian = Some(path(1e-1, false));
            let mut sim = Simulation::empty(config);
            let id = sim
                .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
                .0;
            let middle = (0.32 / dx).round();
            let _bar = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(
                        (0.48 / dx).round() as i32,
                        (0.02 / dx).round() as i32,
                    ))
                    .at(Vec2::new(middle, middle))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&props, &config),
            );
            let speed = 0.01 / dx;
            let positions = sim.particles().x.clone();
            for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
                if x.x < middle {
                    *v = Vec2::new(speed, 0.0);
                }
            }
            let probe = middle + 0.10 / dx;
            let probe_speed = |sim: &Simulation| -> f64 {
                let p = sim.particles();
                let (mut sum, mut count) = (0.0f64, 0.0f64);
                for i in 0..p.len() {
                    if (p.x[i].x - probe).abs() < 0.5 && p.mass[i] > 0.0 {
                        sum += f64::from(p.v[i].x);
                        count += 1.0;
                    }
                }
                sum / count
            };
            let threshold = 0.25 * f64::from(speed);
            let mut previous = probe_speed(&sim);
            let mut arrival = None;
            for frame in 1..=240 {
                sim.step();
                let now = probe_speed(&sim);
                if now >= threshold {
                    arrival = Some(
                        (f64::from(frame) - 1.0 + (threshold - previous) / (now - previous))
                            * 2.5e-5,
                    );
                    break;
                }
                previous = now;
            }
            let c_bar =
                (f64::from(props.e_pa) / ((1.0 - f64::from(nu * nu)) * f64::from(rho))).sqrt();
            let ratio = arrival.map_or(0.0, |t| 0.10 / t / c_bar);
            println!(
                "CAP P11 {} nu {nu} rho {rho} speed_over_c_bar {ratio:.6}",
                label()
            );
            worst = worst.max((ratio - 1.0).abs());
        }
    }
    println!("CAP P11 {} worst_off {worst:.6}", label());
}
