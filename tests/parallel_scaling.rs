//! The impact step on 1, 2, 4 and 8 threads: the clay drop's impact frames and a stepped Norton
//! frame, each frame's wall, its `Timing` split and its counts per thread
//! count. A probe: measurement, nothing held.

extern crate emerge_engine as emerge;

use emerge::materials::solid::viscoplastic_records::ViscoplasticRecord;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const FRAME: f64 = 1.0 / 60.0;

/// The clay drop's scene (`impact_local`'s `drop`): E2's clay block, tolerance
/// 1e-2, the sub-frame handoff on.
fn run_d() -> Simulation {
    let clay = emerge::materials::physical_props::Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let mut config = SimConfig::earth(40, 0.01, FRAME as f32);
    config.min_dt = config.min_dt.min(FRAME as f32);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2).with_subframe_handoff());
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 12.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&clay, &config),
    );
    sim
}

/// `tests/norton.rs`'s drop at 30 MPa (its stand-in density), tolerance 1e-2.
fn norton() -> Simulation {
    const RHO_STAND_IN: f64 = 119.0 * 0.453_592_37 / (0.3048 * 0.3048 * 0.3048);
    let e_pa = 30e6;
    let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
    let mut config = SimConfig::earth(40, 0.01, FRAME as f32);
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

/// One timed frame: wall, the `Timing` split, the counts.
#[derive(Clone, Debug, PartialEq)]
struct Timed {
    wall: f64,
    parts: [f64; 12],
    counts: [u64; 5],
}

const PARTS: [&str; 12] = [
    "residual",
    "jacobian",
    "linearisation",
    "preconditioner build",
    "preconditioner solve",
    "gmres",
    "responses",
    "contact",
    "multipliers",
    "grid",
    "finishing",
    "controller",
];

/// The scene run `untimed` frames, then `timed` frames recorded; and the
/// particles' state after them (their `f32` copies, bits) with the last
/// frame's energy.
fn measure(build: fn() -> Simulation, untimed: usize, timed: usize) -> (Vec<Timed>, Vec<u32>) {
    let mut sim = build();
    for _ in 0..untimed {
        sim.step();
    }
    let mut out = Vec::new();
    for _ in 0..timed {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        let t = &f.timing;
        out.push(Timed {
            wall: f.wall,
            parts: [
                t.residual,
                t.jacobian,
                t.linearisation,
                t.preconditioner_build,
                t.preconditioner_solve,
                t.gmres,
                t.responses,
                t.contact,
                t.multipliers,
                t.grid,
                t.finishing,
                t.controller,
            ],
            counts: [
                u64::from(f.steps),
                u64::from(f.rejected),
                u64::from(f.newton),
                u64::from(f.gmres),
                t.internal_evaluations,
            ],
        });
    }
    let p = sim.particles();
    let mut bits: Vec<u32> =
        p.x.iter()
            .flat_map(|q| [q.x.to_bits(), q.y.to_bits()])
            .collect();
    bits.extend(p.v.iter().flat_map(|q| [q.x.to_bits(), q.y.to_bits()]));
    let e = sim.total_lagrangian_frame().expect("on").energy.to_bits();
    bits.extend([(e >> 32) as u32, e as u32]);
    (out, bits)
}

fn scene(label: &str, build: fn() -> Simulation, untimed: usize, timed: usize) {
    let order = [1usize, 2, 4, 8, 8, 4, 2, 1];
    let mut runs: Vec<(usize, Vec<Timed>, Vec<u32>)> = Vec::new();
    for &threads in &order {
        let (t, bits) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool")
            .install(|| measure(build, untimed, timed));
        for (k, f) in t.iter().enumerate() {
            println!(
                "{label} threads {threads} frame {}: wall {:.3} s; counts steps/rejected/newton/gmres/evaluations {:?}; parts {}",
                untimed + k,
                f.wall,
                f.counts,
                PARTS
                    .iter()
                    .zip(&f.parts)
                    .map(|(n, v)| format!("{n} {v:.3}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        runs.push((threads, t, bits));
    }
    // Criterion 1: counts and state identical at every thread count.
    let (counts0, bits0) = (
        runs[0].1.iter().map(|f| f.counts).collect::<Vec<_>>(),
        &runs[0].2,
    );
    let same = runs
        .iter()
        .all(|(_, t, b)| t.iter().map(|f| f.counts).collect::<Vec<_>>() == counts0 && b == bits0);
    println!("{label} criterion 1: counts and state identical at every thread count: {same}");
    // The minimum over the two runs of each thread count, per frame.
    for k in 0..timed {
        let best = |p: usize| -> &Timed {
            runs.iter()
                .filter(|(t, _, _)| *t == p)
                .map(|(_, t, _)| &t[k])
                .min_by(|a, b| a.wall.total_cmp(&b.wall))
                .expect("two runs")
        };
        let one = best(1);
        for p in [2usize, 4, 8] {
            let f = best(p);
            let s = one.wall / f.wall;
            let serial = (1.0 / s - 1.0 / p as f64) / (1.0 - 1.0 / p as f64);
            let phases = PARTS
                .iter()
                .zip(one.parts.iter().zip(&f.parts))
                .filter(|(_, (a, _))| **a > 0.01 * one.wall)
                .map(|(n, (a, b))| format!("{n} {:.2}", a / b))
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "{label} frame {}: {p} threads: wall {:.3} s against {:.3} s, speedup {s:.3}, Amdahl serial fraction {serial:.3}; phase speedups (phases above 1 % of the 1-thread wall): {phases}",
                untimed + k,
                f.wall,
                one.wall
            );
        }
        let shares = PARTS
            .iter()
            .zip(&one.parts)
            .map(|(n, v)| format!("{n} {:.1} %", 100.0 * v / one.wall))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "{label} frame {}: 1-thread shares of the wall: {shares}",
            untimed + k
        );
    }
}

#[test]
#[ignore = "real-time profile: a quiet machine only"]
fn profile_parallel_scaling() {
    scene("the clay drop", run_d, 8, 2);
    scene("Norton", norton, 9, 1);
}
