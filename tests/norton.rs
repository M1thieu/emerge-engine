//! Norton's power-law flow on the total Lagrangian path: Sapozhnikov and
//! Ignatova's technical plasticine (`ViscoplasticRecord`) dropped as E2's
//! block, its energy ledger closed every frame; Roma Plastilina's measured
//! law at a drop test's speed; and the paths that do not carry the law
//! refusing it.

extern crate emerge_engine as emerge;

use emerge::materials::MaterialModel;
use emerge::materials::solid::viscoplastic_records::ViscoplasticRecord;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// The scene's density, kg/m^3: the record gives none (its declared gap),
/// so this scene states its own, a stand-in from ANOTHER plasticine
/// (Crandall et al. 1970's, `ViscoelasticRecord`'s 119 lb/ft^3), declared.
const RHO_STAND_IN: f64 = 119.0 * 0.453_592_37 / (0.3048 * 0.3048 * 0.3048);

/// E2's block (8 x 4 cm, its bottom 10 cells above the floor, 1 cm cells,
/// 1/60 s frames) of the record at Young's modulus `e_pa`, step tolerance
/// 1e-2.
fn drop(e_pa: f64) -> Simulation {
    let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let material = record.material((e_pa, RHO_STAND_IN), &config);
    let props = emerge::materials::physical_props::Elastic {
        e_pa: e_pa as f32,
        nu: record.nu as f32,
        rho_kg_m3: RHO_STAND_IN as f32,
    };
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
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
}

/// What a drop measured: the largest balance over its bound, the heat, the
/// particle-steps beyond the record, the most return iterations, the rebound
/// (the centre's largest upward speed after the first contact, m/s) and the
/// first contact's frame.
struct Dropped {
    worst: f64,
    heat: f64,
    plastic: f64,
    outside: [u32; 3],
    iterations: u32,
    rebound: f64,
    first_contact: Option<usize>,
    /// The least plastic work of a frame (never negative: a law's dissipation).
    least_plastic: f64,
    /// The particles' least horizontal extent over the last 10 frames, cells.
    last_width: f32,
}

/// The particles' horizontal extent, cells.
fn width(sim: &Simulation) -> f32 {
    let (low, high) = sim
        .particles()
        .x
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), q| {
            (low.min(q.x), high.max(q.x))
        });
    high - low
}

fn run(mut sim: Simulation, frames: usize) -> Dropped {
    let dx = f64::from(sim.config().dx_meters);
    let count = sim.particles().len() as f64;
    let mut out = Dropped {
        worst: 0.0,
        heat: 0.0,
        plastic: 0.0,
        outside: [0; 3],
        iterations: 0,
        rebound: 0.0,
        first_contact: None,
        least_plastic: f64::INFINITY,
        last_width: f32::INFINITY,
    };
    let mut allowed = 0.0f64;
    for k in 1..=frames {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
        let round_off = count
            * f64::EPSILON
            * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
        out.worst = out.worst.max(f.balance.abs() / (allowed + round_off));
        out.heat = f.heat;
        out.plastic += f.plastic_work;
        out.least_plastic = out.least_plastic.min(f.plastic_work);
        if k + 10 > frames {
            out.last_width = out.last_width.min(width(&sim));
        }
        out.outside[0] += f.norton_slower;
        out.outside[1] += f.norton_faster;
        out.outside[2] += f.norton_tension;
        out.iterations = out.iterations.max(f.norton_iterations);
        if f.contacts > 0 && out.first_contact.is_none() {
            out.first_contact = Some(k);
        }
        if out.first_contact.is_some() {
            let p = sim.particles();
            let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
            let vy: f64 =
                p.v.iter()
                    .zip(&p.mass)
                    .map(|(v, &m)| f64::from(v.y) * f64::from(m))
                    .sum::<f64>()
                    / mass
                    * dx;
            out.rebound = out.rebound.max(vy);
        }
    }
    out
}

/// Criteria fixed before measuring: the record's block dropped,
/// the energy balance within the solves' bound every frame, the heat the
/// plastic work alone (no friction, no dashpot), and both made. Reported:
/// the rebound, the return's iterations and the particle-steps beyond the
/// record (rate outside 4e-4 to 80 1/s, or tension), and their
/// sensitivity to `E` over the record's 25-35 MPa. The impact's strain rate,
/// derived: a 0.10 m fall lands at `sqrt(2 g h) = 1.40 m/s`, about `v / H =
/// 35 1/s` over the block's 4 cm, inside the record's range.
#[test]
fn a_dropped_norton_block_flows_and_its_plastic_work_becomes_heat() {
    const FRAMES: usize = 30;
    for e_pa in [25.0e6, 30.0e6, 35.0e6] {
        let d = run(drop(e_pa), FRAMES);
        println!(
            "E {:.0} MPa: balance at most {:.3} of its bound; heat {:.4e} J/m, plastic work {:.4e} J/m; rebound {:.4} m/s after the first contact (frame {:?}); return iterations at most {}; particle-steps beyond the record (slower, faster, tensile) {:?}",
            e_pa / 1e6,
            d.worst,
            d.heat,
            d.plastic,
            d.rebound,
            d.first_contact,
            d.iterations,
            d.outside
        );
        assert!(d.worst <= 1.0, "E {e_pa}: balance {} of its bound", d.worst);
        assert!(d.plastic > 0.0, "no plastic work");
        assert!(
            (d.heat - d.plastic).abs() <= 1e-9 * d.plastic,
            "heat {} against the plastic work {}",
            d.heat,
            d.plastic
        );
    }
}

/// Roma Plastilina's record (Hernandez et al. 2023) as an 8 x 4 cm block, 2 cm
/// cells, plane strain, reaching the frictionless floor at `sqrt(2 g h)` of
/// a 2.0 m drop, 6.264 m/s: its density from the same papers (1529 kg/m^3),
/// forecast activation on. Not the sphere drop: a verification of this law on
/// the path, compared with no measured depth. Criteria written before
/// measuring: the balance within the solves' bound every frame; the heat the
/// plastic work alone; no frame's plastic work negative; the block wider
/// for good (over the last 10 frames, against its start); and the particle-steps
/// beyond the record counted, as a drop reaches about `v / H` = 157 1/s
/// against the 0.33 1/s its compression tests reach. Measured: about nine
/// tenths of the energy at touch became heat at this resolution.
#[test]
fn roma_plastilina_at_a_drop_tests_speed_flows_for_good_and_its_work_becomes_heat() {
    const FRAMES: usize = 30;
    let record = ViscoplasticRecord::roma_plastilina_hernandez_2023();
    let rho = 1529.0;
    let mut config = SimConfig::earth(20, 0.02, 1.0 / 60.0);
    let mut path = TotalLagrangianConfig::new(1e-2);
    path.settings.forecast_activation = true;
    config.total_lagrangian = Some(path);
    let props = emerge::materials::physical_props::Elastic {
        e_pa: record.e_pa as f32,
        nu: record.nu as f32,
        rho_kg_m3: rho as f32,
    };
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(record.material((record.e_pa, rho), &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(4, 2))
            .at(Vec2::new(10.0, floor + 6.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    // 5 cells above the floor, 0.10 m: the start velocity that reaches it at
    // the drop's 6.264 m/s.
    let touch = (2.0f64 * 9.81 * 2.0).sqrt();
    let start = (touch * touch - 2.0 * 9.81 * 0.10).sqrt();
    sim.particles_mut()
        .v
        .iter_mut()
        .for_each(|v| v.y = -(start / 0.02) as f32);
    let start_width = width(&sim);
    let d = run(sim, FRAMES);
    println!(
        "Roma Plastilina at {touch:.3} m/s: balance at most {:.3} of its bound; heat {:.3} J/m, of 1/2 m v^2 = {:.3} J/m; width {start_width:.2} -> at least {:.2} cells; rebound {:.3} m/s; return iterations at most {}; particle-steps beyond the record (slower, faster, tensile) {:?}",
        d.worst,
        d.heat,
        0.5 * rho * 0.08 * 0.04 * touch * touch,
        d.last_width,
        d.rebound,
        d.iterations,
        d.outside
    );
    assert!(d.worst <= 1.0, "balance {} of its bound", d.worst);
    assert!(d.plastic > 0.0, "no plastic work");
    assert!(
        (d.heat - d.plastic).abs() <= 1e-9 * d.plastic,
        "heat {} against the plastic work {}",
        d.heat,
        d.plastic
    );
    assert!(d.least_plastic >= 0.0, "plastic work {}", d.least_plastic);
    assert!(
        d.last_width > start_width,
        "not wider for good: {start_width} -> {}",
        d.last_width
    );
    assert!(d.outside[1] > 0, "no step beyond the record's rates");
}

/// The explicit path does not carry Norton's flow: it refuses the material
/// rather than run it with the yield-free von Mises law in its place.
#[test]
#[should_panic(expected = "Norton's power-law flow")]
fn the_explicit_path_refuses_norton_flow() {
    let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
    let config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    let mut sim = Simulation::empty(config);
    let _ = sim.register_material(Box::new(
        record.material((record.e_pa, RHO_STAND_IN), &config),
    ));
    sim.step();
}

/// The GPU path does not carry it either, nor relaxation branches: both
/// materials name the reason the GPU solver refuses them with.
#[test]
fn the_gpu_path_names_what_it_does_not_carry() {
    let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
    let config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    let norton = record.material((record.e_pa, RHO_STAND_IN), &config);
    assert!(norton.gpu_unsupported_reason().is_some());
    let props = emerge::materials::physical_props::Elastic {
        e_pa: 2.0e7,
        nu: 0.4,
        rho_kg_m3: 1900.0,
    };
    let mut damped =
        <emerge::CorotatedMaterial as emerge::FromSI<_>>::from_physical(&props, &config);
    assert!(damped.gpu_unsupported_reason().is_none());
    damped.relaxation =
        Some(emerge::materials::relaxation::RelaxationSpectrum::plasticine_crandall_1970());
    assert!(damped.gpu_unsupported_reason().is_some());
}
