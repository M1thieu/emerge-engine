//! The total Lagrangian solid path inside `Simulation` (opt-in,
//! `SimConfig::total_lagrangian`), measured at several states: a block at
//! rest on the floor, dropped from two heights, sliding to a stop on a
//! Coulomb floor, two bodies colliding; and what the path refuses.
//!
//! Its claim, checked in every state: the solids' energy (kinetic, stored,
//! gravity's) plus the heat friction made stays the start's, to round-off;
//! no NaN; the matter stays above the floor within the path's declared
//! penetration (a step's motion, `v dt`).

extern crate emerge_engine as emerge;

#[path = "common/cap.rs"]
mod cap;

use emerge::boundary::FrictionBoundary;
use emerge::fields::LinearDragField;
use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    CorotatedMaterial, DuctileProps, FromSI, SimConfig, Simulation, SpawnRegion, VonMisesMaterial,
};
use glam::{IVec2, Vec2};

const CELL_M: f32 = 0.01;
const RES: usize = 40;
/// Where a sliding block's centre starts, cells: at 1 m/s on mu 0.3 it
/// stops `v0^2 / (2 mu g)` = 17 cells further, its right face at 31,
/// clear of the right wall at 38. Started at the domain's middle, it
/// reached that wall at 0.21 s and bounced back off it (measured: the
/// mean velocity from +36 to -26 cells/s in one frame), the bounce's
/// energy missing from the heat.
const SLIDE_START: f32 = 10.0;

/// Medium clay's moduli at a 1 cm cell, the path on with the step
/// controller's tolerance `tolerance`.
fn config(tolerance: f64) -> SimConfig {
    let mut config = SimConfig::earth(RES, CELL_M, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(tolerance));
    config
}

fn clay() -> Elastic {
    Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    }
}

/// A `width` by `height` cell block of clay whose bottom sits `lift` cells
/// above the floor.
fn block(
    config: &SimConfig,
    material: u32,
    (width, height): (i32, i32),
    x: f32,
    lift: f32,
) -> SpawnRegion {
    let floor = config.boundary_thickness as f32;
    SpawnRegion::for_sim(config)
        .box_of(IVec2::new(width, height))
        .at(Vec2::new(x, floor + lift + height as f32 * 0.5))
        .spacing(0.5)
        .material(material)
        .mass_from(&clay(), config)
}

fn scene(config: SimConfig, lift: f32) -> Simulation {
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let _block = sim.add_body(block(&config, id, (8, 4), RES as f32 * 0.5, lift));
    sim
}

/// The scale the balance is judged against: the total weight over one
/// cell's height.
fn weight_scale(sim: &Simulation) -> f64 {
    let mass: f64 = sim.particles().mass.iter().map(|&m| f64::from(m)).sum();
    mass * f64::from(sim.config().gravity.length()) * energy_unit(sim)
}

/// The grid's energy unit in J per metre of depth, `M dx^2`: the path
/// reports its energies in SI, the particles carry grid units.
fn energy_unit(sim: &Simulation) -> f64 {
    let config = sim.config();
    f64::from(config.grid_mass_unit_kg_per_m()) * f64::from(config.dx_meters).powi(2)
}

/// The scene's precondition: a body whose centre starts at `start` with
/// `half_width` cells either side, moving `travel` cells to the right over
/// the run, stays at least `CLEARANCE` cells off both side walls. A test
/// that measures sliding or a collision must not measure a wall's bounce.
fn assert_clear_of_side_walls(config: &SimConfig, start: f32, half_width: f32, travel: f32) {
    const CLEARANCE: f32 = 3.0;
    let thickness = config.boundary_thickness as f32;
    let (left, right) = (thickness, config.grid_res as f32 - thickness);
    assert!(
        start - half_width - left >= CLEARANCE
            && right - (start + travel + half_width) >= CLEARANCE,
        "scene precondition: the body (centre {start}, half width {half_width}, travel {travel} cells) comes within {CLEARANCE} cells of a side wall at {left} or {right}"
    );
}

/// Runs `frames` frames, checking the balance, NaN and the floor at each;
/// returns the worst balance over the weight scale and the deepest point
/// below the floor, cells.
///
/// The balance's bar follows from the solves' tolerances, not a chosen
/// number: the midpoint step's energy identity `dE = u . R(u)` bounds each
/// step's error by `sum |u_k R_k|` (Newton's tolerance on `R`), and each
/// contact condition held to its gap tolerance `a_v` does at most
/// `lambda_v a_v` of work (`StepReport`'s `residual_work` and
/// `contact_work_bound`). Their sum accumulates over the run; the energy's
/// own sums add their round-off: each floating addition errs by at most
/// `eps` of its partial sum, itself at most `sum |terms|`, so `n` terms
/// err by at most `n eps sum |terms|` (to first order in `eps`). Measured,
/// this bound tracks the tolerances: the
/// collision's contact work 2.74e-10 of the weight scale at the default
/// tolerances, 2.71e-11 with them ten times tighter.
/// With `EMERGE_REPLAY_DIR` set, `run` writes a replay of its scene there:
/// per frame the particles' positions (m) and speeds (m/s), the energy,
/// heat and balance (J/m), the contacts held, one JSON file per test,
/// named after it. For viewing, no check.
fn replay_frame(sim: &Simulation, time: f64) -> String {
    let dx = f64::from(sim.config().dx_meters);
    let p = sim.particles();
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let positions: Vec<String> =
        p.x.iter()
            .map(|q| format!("[{:.6},{:.6}]", f64::from(q.x) * dx, f64::from(q.y) * dx))
            .collect();
    let speeds: Vec<String> =
        p.v.iter()
            .map(|v| format!("{:.5}", f64::from(v.length()) * dx))
            .collect();
    format!(
        "{{\"time\":{time:.6},\"positions\":[{}],\"speed_m_s\":[{}],\"energy\":{:.9e},\"heat\":{:.9e},\"balance\":{:.3e},\"contacts\":{}}}",
        positions.join(","),
        speeds.join(","),
        frame.energy,
        frame.heat,
        frame.balance,
        frame.contacts
    )
}

/// Writes the replay `frames` of the running test into `EMERGE_REPLAY_DIR`.
fn write_replay(sim: &Simulation, frames: &[String]) {
    let Some(dir) = std::env::var_os("EMERGE_REPLAY_DIR") else {
        return;
    };
    let name = std::thread::current()
        .name()
        .unwrap_or("scene")
        .rsplit("::")
        .next()
        .unwrap_or("scene")
        .to_string();
    let config = sim.config();
    let header = format!(
        "{{\"scene\":\"{name}\",\"dx_m\":{},\"frame_s\":{},\"gravity_m_s2\":{},\"floor_y_m\":{},\"path\":\"total Lagrangian, weighted-gap contact\",\"precomputed\":true}}",
        config.dx_meters,
        config.dt,
        config.gravity.length() * config.dx_meters,
        config.boundary_thickness as f32 * config.dx_meters
    );
    let path = std::path::Path::new(&dir).join(format!("{name}.json"));
    let body = format!(
        "{{\"header\":{header},\"frames\":[
{}
]}}
",
        frames.join(
            ",
"
        )
    );
    std::fs::write(&path, body).expect("the replay directory is writable");
    println!("replay written: {}", path.display());
}

fn run(sim: &mut Simulation, frames: usize) -> (f64, f32) {
    let (worst, deepest, _) = run_measured(sim, frames);
    (worst, deepest)
}

/// `run`, and what the solves allow the horizontal momentum to drift by,
/// N s/m: the frames' `residual_impulse.x + impulse_round_off` summed.
fn run_measured(sim: &mut Simulation, frames: usize) -> (f64, f32, f64) {
    let floor = sim.config().boundary_thickness as f32;
    let scale = weight_scale(sim);
    let (mut worst, mut deepest) = (0.0f64, 0.0f32);
    let clock = std::time::Instant::now();
    // The drift the solves allow (their residuals' and contacts' work
    // bounds, accumulated) and the round-off of the energy's sums.
    let mut allowed = 0.0f64;
    let mut impulse = 0.0f64;
    let mut tightest = 0.0f64;
    let mut replay = Vec::new();
    let frame_s = f64::from(sim.config().dt);
    let count = sim.particles().len() as f64;
    // The run's cost in counts (no clock): steps, rejected tries, contact
    // passes and Newton iterations, and the heaviest frame's passes, printed
    // once per run as a `cost counts` line for the cost check
    // (`EMERGE_COST_BASELINE` is not read here: the comparison is outside).
    let (mut steps, mut rejected, mut passes, mut newton, mut heaviest) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    // The Jacobian factorizations, appended after the fields the cost check
    // reads; and a kept probe (`EMERGE_PROBE_PROFILE`), the wall model's
    // profile: the frames' always-on `Timing` and wall summed.
    let mut factorizations = 0u64;
    let mut profile: Option<(f64, _)> = None;
    for _ in 0..frames {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        match profile.as_mut() {
            None => profile = Some((frame.wall, frame.timing)),
            Some((wall, timing)) => {
                *wall += frame.wall;
                timing.add(&frame.timing);
            }
        }
        factorizations += frame.timing.factorizations;
        steps += u64::from(frame.steps);
        rejected += u64::from(frame.rejected);
        passes += u64::from(frame.passes);
        newton += u64::from(frame.newton);
        heaviest = heaviest.max(u64::from(frame.passes));
        assert!(frame.energy.is_finite(), "energy not finite");
        allowed += frame.residual_work + frame.contact_work_bound;
        impulse += frame.residual_impulse.x + frame.impulse_round_off;
        let round_off = count
            * f64::EPSILON
            * (frame.parts.iter().map(|e| e.abs()).sum::<f64>()
                + frame.heat.abs()
                + frame.start_energy.abs());
        tightest = tightest.max(frame.balance.abs() / (allowed + round_off));
        if std::env::var_os("EMERGE_REPLAY_DIR").is_some() {
            replay.push(replay_frame(sim, frame_s * (replay.len() + 1) as f64));
        }
        assert!(
            frame.balance.abs() <= allowed + round_off,
            "the balance {:e} J/m exceeds what the solves' tolerances allow, {:e} J/m (residuals and contacts) plus {:e} J/m (sums)",
            frame.balance,
            allowed,
            round_off
        );
        assert!(
            sim.particles().x.iter().all(|q| q.is_finite()),
            "a particle's position is not finite"
        );
        worst = worst.max(frame.balance.abs() / scale);
        // A clean run: no step kept on a stagnated Newton.
        assert_eq!(
            frame.stagnated, 0,
            "a step stagnated on its residual's noise"
        );
        if std::env::var_os("EMERGE_TL_TRACE").is_some() {
            let p = sim.particles();
            for tag in 1..=2u32 {
                let (mut m, mut kinetic, mut y, mut low, mut vx, mut vy) =
                    (0.0f64, 0.0f64, 0.0f64, f32::INFINITY, 0.0f64, 0.0f64);
                for i in 0..p.len() {
                    if p.user_tag[i] == tag {
                        let mass = f64::from(p.mass[i]);
                        m += mass;
                        kinetic += 0.5 * mass * f64::from(p.v[i].length_squared());
                        y += mass * f64::from(p.x[i].y);
                        vx += mass * f64::from(p.v[i].x);
                        vy += mass * f64::from(p.v[i].y);
                        low = low.min(p.x[i].y);
                    }
                }
                if m > 0.0 {
                    println!(
                        "    body {tag}: kinetic {:.3e}, mean y {:.3}, lowest {low:.3}, mean v ({:+.3}, {:+.3})",
                        kinetic * energy_unit(sim) / scale,
                        y / m,
                        vx / m,
                        vy / m
                    );
                }
            }
            println!(
                "  frame: newton {} unconverged {} steps {} rejected {} (unconverged {}) reaching {} at the floor {}, balance {:+.2e}, friction work {:+.3e} contact work {:+.3e} heat {:.3e} energy {:.6e} (kinetic {:.4e}, stored {:.4e}, gravity {:.6e})",
                frame.newton,
                frame.unconverged,
                frame.steps,
                frame.rejected,
                frame.rejected_unconverged,
                frame.reaching,
                frame.at_floor,
                frame.balance / scale,
                frame.friction_work / scale,
                frame.contact_work / scale,
                frame.heat / scale,
                frame.energy / scale,
                frame.parts[0] / scale,
                frame.parts[1] / scale,
                frame.parts[2] / scale
            );
            let momentum: f64 =
                p.v.iter()
                    .zip(&p.mass)
                    .map(|(v, &m)| f64::from(v.x) * f64::from(m))
                    .sum();
            println!(
                "  bounds: residual work {:.3e}, contact work bound {:.3e}, contact work {:+.3e} (J/m)",
                frame.residual_work, frame.contact_work_bound, frame.contact_work
            );
            let t = frame.timing;
            println!(
                "  timing: frame {:.3} s; residual {:.3}, Jacobian products {:.3}, preconditioner build {:.3} solve {:.3}, GMRES {:.3} (algebra {:.3}), responses {:.3}, contact passes {:.3}, multipliers {:.3}, grid {:.3}, finishing {:.3}, controller {:.3}; internal force evaluated {} times",
                frame.wall,
                t.residual,
                t.jacobian,
                t.preconditioner_build,
                t.preconditioner_solve,
                t.gmres,
                t.gmres - t.jacobian - t.preconditioner_solve,
                t.responses,
                t.contact,
                t.multipliers,
                t.grid,
                t.finishing,
                t.controller,
                t.internal_evaluations
            );
            println!(
                "  counters: horizontal momentum {momentum:.12e}, contact passes {}, most contacts {}, gmres {}, stagnated {}, rejected on the estimate {}, wall {:.1} s",
                frame.passes,
                frame.contacts,
                frame.gmres,
                frame.stagnated,
                frame.rejected - frame.rejected_unconverged,
                clock.elapsed().as_secs_f64()
            );
        }
        let lowest = sim
            .particles()
            .x
            .iter()
            .map(|q| q.y)
            .fold(f32::INFINITY, f32::min);
        deepest = deepest.max(floor - lowest);
    }
    // How close the balance came to the bound: a bound far looser than the
    // drift (1e-6 of it, say) would hide a defect below it.
    println!("balance over the solves' bound: at most {tightest:.3e}");
    if std::env::var_os("EMERGE_PROBE_PROFILE").is_some()
        && let Some((wall, t)) = profile
    {
        let share = |x: f64| 100.0 * x / wall.max(f64::MIN_POSITIVE);
        println!(
            "profile: wall {wall:.3} s; residual {:.1} %, jacobian {:.1} % (push derivative {:.1} %, responses' products {:.1} %), linearisation {:.1} %, preconditioner build {:.1} %, preconditioner solve {:.1} %, GMRES {:.1} %, responses {:.1} %, contact {:.1} %, multipliers {:.1} %, grid {:.1} %, finishing {:.1} %, controller {:.1} %; internal-force evaluations {}, factorizations {}",
            share(t.residual),
            share(t.jacobian),
            share(t.push_derivative),
            share(t.responses_jacobian),
            share(t.linearisation),
            share(t.preconditioner_build),
            share(t.preconditioner_solve),
            share(t.gmres),
            share(t.responses),
            share(t.contact),
            share(t.multipliers),
            share(t.grid),
            share(t.finishing),
            share(t.controller),
            t.internal_evaluations,
            t.factorizations
        );
    }
    write_replay(sim, &replay);
    println!(
        "cost counts: {}: frames {frames}, steps {steps}, rejected {rejected}, passes {passes}, newton {newton}, heaviest frame passes {heaviest}, factorizations {factorizations}",
        std::thread::current().name().unwrap_or("unnamed")
    );
    (worst, deepest, impulse)
}

#[test]
fn a_block_resting_on_the_floor_keeps_its_energy() {
    let mut sim = scene(config(1e-1), 0.0);
    let (worst, deepest) = run(&mut sim, 30);
    println!("rest: worst balance {worst:.2e} of the weight scale, deepest {deepest:.2e} cells");
    assert!(worst < 1e-6, "balance {worst:e}");
    assert!(deepest < 0.5, "sank {deepest} cells");
}

#[test]
fn a_dropped_block_bounces_with_its_energy_kept() {
    for lift in [4.0f32, 10.0] {
        let mut sim = scene(config(1e-1), lift);
        let (worst, deepest) = run(&mut sim, 45);
        // The declared penetration: a step's motion at the impact speed.
        let impact = (2.0 * sim.config().gravity.length() * lift).sqrt();
        let bound = impact / 60.0;
        println!(
            "drop {lift} cells: worst balance {worst:.2e}, deepest {deepest:.3} cells (v dt {bound:.3})"
        );
        assert!(worst < 1e-6, "balance {worst:e}");
        assert!(
            deepest <= bound,
            "penetration {deepest} beyond v dt {bound}"
        );
    }
}

/// The block's ledger closes and it comes to rest. The heat claim ("all the
/// kinetic energy becomes heat") is not asserted here: this corotated clay has
/// no material damping, so the vibration that the sliding and the settling
/// excite stays in the block (measured: 6e-4 of `1/2 M v0^2` as kinetic and
/// stored energy at the end, about 20 times what a block at rest can keep,
/// derived). The claim is the damped block's,
/// `a_damped_block_slides_to_rest_and_its_energy_becomes_heat`.
#[test]
fn a_block_slides_to_rest_with_its_ledger_closed() {
    let config = config(1e-1);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let _block = sim.add_body(block(&config, id, (8, 4), SLIDE_START, 0.0));
    sim.add_boundary_condition(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.3,
    )));
    // 1 m/s along the floor.
    let speed = 1.0 / CELL_M;
    let stopping = speed * speed / (2.0 * 0.3 * config.gravity.length());
    assert_clear_of_side_walls(&config, SLIDE_START, 4.0, stopping);
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(speed, 0.0);
    }
    let (worst, _) = run(&mut sim, 30);
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let mass: f64 = sim.particles().mass.iter().map(|&m| f64::from(m)).sum();
    let kinetic0 = 0.5 * mass * f64::from(speed) * f64::from(speed) * energy_unit(&sim);
    let mean_speed: f32 =
        sim.particles().v.iter().map(|v| v.x).sum::<f32>() / sim.particles().len() as f32;
    println!(
        "slide: heat {:.6} of 1/2 M v0^2, worst balance {worst:.2e}, end speed {:.3e} m/s",
        frame.heat / kinetic0,
        mean_speed * CELL_M
    );
    assert!(worst < 1e-6, "balance {worst:e}");
    assert!(
        mean_speed.abs() * CELL_M < 0.05,
        "still sliding at {} m/s",
        mean_speed * CELL_M
    );
}

/// The slide of `a_block_slides_to_rest_with_its_ledger_closed` in a damped
/// material: Harbutt's Plasticine, freshly formed, its relaxed elasticity and
/// three Zener mechanisms measured by Crandall, Ehrlich and Wenk 1970
/// (`ViscoelasticRecord::plasticine_crandall_1970_fresh`). Damped, the
/// vibration the sliding and the settling excite is dissipated into heat, so
/// at the end the block keeps no more than a block at rest can, derived here
/// (first law plus statics; no source beyond the record):
/// - the shear its base traction holds, at most `(mu M g)^2 H / (2 G W)`
///   (simple shear of a `W x H` block per metre of depth, `k_T = G W / H`);
/// - its self-weight energy, `W rho^2 g^2 H^3 (1 - nu^2) / (6 E)` (plane
///   strain, free sides, `sigma_yy = rho g (H - y)`);
/// - the kinetic energy of the translation left (its mean speed);
///
/// all with the relaxed modulus (the slowest state, the largest energies).
/// Asserted: the ledger per frame (`run`), the stop, and the end kinetic plus
/// stored energy within that bound plus the ledger's worst balance: the rest
/// of `1/2 M v0^2` has become heat (the ledger's identity).
#[test]
fn a_damped_block_slides_to_rest_and_its_energy_becomes_heat() {
    let config = config(1e-1);
    let record =
        emerge::materials::relaxation::ViscoelasticRecord::plasticine_crandall_1970_fresh();
    let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
    material.relaxation = Some(record.spectrum);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(SLIDE_START, floor + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&record.relaxed, &config),
    );
    sim.add_boundary_condition(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.3,
    )));
    let speed = 1.0 / CELL_M;
    let stopping = speed * speed / (2.0 * 0.3 * config.gravity.length());
    assert_clear_of_side_walls(&config, SLIDE_START, 4.0, stopping);
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(speed, 0.0);
    }
    let (worst, _) = run(&mut sim, 30);
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let mass: f64 = sim.particles().mass.iter().map(|&m| f64::from(m)).sum();
    let kinetic0 = 0.5 * mass * f64::from(speed) * f64::from(speed) * energy_unit(&sim);
    let mean_speed: f32 =
        sim.particles().v.iter().map(|v| v.x).sum::<f32>() / sim.particles().len() as f32;
    // The bound, SI per metre of depth (the block: 8 x 4 cells of 1 cm).
    let (e, nu, rho) = (
        f64::from(record.relaxed.e_pa),
        f64::from(record.relaxed.nu),
        f64::from(record.relaxed.rho_kg_m3),
    );
    let (w, h, mu, g) = (
        0.08f64,
        0.04f64,
        0.3f64,
        f64::from(config.gravity.length() * CELL_M),
    );
    let m = 2.0 * kinetic0; // 1/2 m v0^2 with v0 = 1 m/s
    let shear = (mu * m * g).powi(2) * h / (2.0 * e / (2.0 * (1.0 + nu)) * w);
    let static_weight = w * rho * rho * g * g * h.powi(3) * (1.0 - nu * nu) / (6.0 * e);
    let translation = 0.5 * m * f64::from(mean_speed * CELL_M).powi(2);
    let bound = shear + static_weight + translation;
    let left = frame.parts[0] + frame.parts[1];
    println!(
        "damped slide: heat {:.6} of 1/2 M v0^2, kinetic {:.3e}, stored {:.3e}, their sum {left:.3e} against the at-rest bound {bound:.3e} (shear {shear:.3e}, self-weight {static_weight:.3e}, translation {translation:.3e}) J/m, worst balance {worst:.2e}, end speed {:.3e} m/s",
        frame.heat / kinetic0,
        frame.parts[0],
        frame.parts[1],
        mean_speed * CELL_M
    );
    assert!(worst < 1e-6, "balance {worst:e}");
    assert!(
        mean_speed.abs() * CELL_M < 0.05,
        "still sliding at {} m/s",
        mean_speed * CELL_M
    );
    assert!(
        left <= bound + worst,
        "the block keeps {left:e} J/m, beyond a block at rest's {bound:e}"
    );
}

/// The same slide in the rigid limit's stand-in, `E` 1 GPa: its ledger
/// closes. The heat claim is not asserted: undamped, the block keeps its
/// settling and sliding vibration (measured: 8e-6 of `1/2 M v0^2` at the end,
/// about 60 times what a block at rest can keep, derived); the claim is the
/// damped block's, `a_damped_block_slides_to_rest_and_its_energy_becomes_heat`.
#[test]
fn a_stiff_block_slides_to_rest_with_its_ledger_closed() {
    let stiff = Elastic {
        e_pa: 1e9,
        ..clay()
    };
    let config = config(1e-1);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&stiff, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(SLIDE_START, floor + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&stiff, &config),
    );
    sim.add_boundary_condition(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.3,
    )));
    let speed = 1.0 / CELL_M;
    let stopping = speed * speed / (2.0 * 0.3 * config.gravity.length());
    assert_clear_of_side_walls(&config, SLIDE_START, 4.0, stopping);
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(speed, 0.0);
    }
    let clock = std::time::Instant::now();
    let (worst, _) = run(&mut sim, 30);
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let mass: f64 = sim.particles().mass.iter().map(|&m| f64::from(m)).sum();
    let kinetic0 = 0.5 * mass * f64::from(speed) * f64::from(speed) * energy_unit(&sim);
    println!(
        "stiff slide: heat {:.6} of 1/2 M v0^2, worst balance {worst:.2e}, {:.1} s for 0.5 s simulated",
        frame.heat / kinetic0,
        clock.elapsed().as_secs_f64()
    );
    assert!(worst < 1e-6, "balance {worst:e}");
}

#[test]
fn two_blocks_collide_with_their_energy_and_momentum_kept() {
    let config = config(1e-1);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    // 4 cells apart, closing at 0.25 m/s: they meet at 0.16 s. No body
    // outruns the moving one's speed over the 0.5 s run, so the right one
    // travels at most 12.5 cells (at 0.5 m/s from the middle it reached
    // the right wall, whose bounce the momentum check then counted).
    let (left_start, right_start) = (8.0, 18.0);
    let speed = 0.25 / CELL_M;
    assert_clear_of_side_walls(&config, left_start, 3.0, 0.0);
    assert_clear_of_side_walls(&config, right_start, 3.0, speed * 0.5);
    let left_tag = sim.add_body(block(&config, id, (6, 4), left_start, 0.0));
    let _right = sim.add_body(block(&config, id, (6, 4), right_start, 0.0));
    let left: Vec<bool> = sim
        .particles()
        .user_tag
        .iter()
        .map(|&t| t == left_tag)
        .collect();
    for (v, &l) in sim.particles_mut().v.iter_mut().zip(&left) {
        if l {
            *v = Vec2::new(speed, 0.0);
        }
    }
    let momentum = |sim: &Simulation| -> f64 {
        sim.particles()
            .v
            .iter()
            .zip(&sim.particles().mass)
            .map(|(v, &m)| f64::from(v.x) * f64::from(m))
            .sum()
    };
    let start = momentum(&sim);
    let (worst, _, allowed) = run_measured(&mut sim, 30);
    let drift = (momentum(&sim) - start).abs() / start.abs();
    // The path's own f64 momentum against what its solves allow: no
    // external horizontal force acts (gravity and the floor's push are
    // vertical, the side walls are not reached, the bodies' pushes are
    // equal and opposite), so it moves only by the residuals' impulses and
    // their round-off (`run_measured`), plus the f64 sums' own round-off,
    // `n eps sum |m v_x|`, `sum |m v_x|` at most `sqrt(2 M KE)` at the end
    // and `|p_x|` at the start (one body moving, uniformly).
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let unit = f64::from(sim.config().grid_mass_unit_kg_per_m());
    let total_mass: f64 = sim
        .particles()
        .mass
        .iter()
        .map(|&m| f64::from(m))
        .sum::<f64>()
        * unit;
    let count = sim.particles().len() as f64;
    let sums = count
        * f64::EPSILON
        * (frame.start_momentum.x.abs() + (2.0 * total_mass * frame.parts[0]).sqrt());
    let moved = (frame.momentum.x - frame.start_momentum.x).abs();
    println!(
        "collision: worst balance {worst:.2e}, horizontal momentum off by {drift:.2e} (f32 copies); f64 drift {moved:.3e} N s/m against the solves' bound {:.3e} ({:.3e} of it)",
        allowed + sums,
        moved / (allowed + sums)
    );
    // The bodies met, and did not pass through each other: Signorini between
    // them, the gap never negative, so their order along the closing axis
    // is kept. Checked at the run's end, after the meeting time (0.16 s of
    // 0.5 s): the left body's rightmost particle stays left of the right
    // body's leftmost one within the path's declared penetration, a step's
    // travel at the closing speed (`v dt`, as the drop test declares). The
    // ledger and the momentum alone hold as well when the bodies pass
    // through each other (measured: a run whose body-body conditions were
    // all dropped kept both and passed).
    let x = &sim.particles().x;
    let left_most = x
        .iter()
        .zip(&left)
        .filter(|&(_, &l)| l)
        .map(|(q, _)| q.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let right_least = x
        .iter()
        .zip(&left)
        .filter(|&(_, &l)| !l)
        .map(|(q, _)| q.x)
        .fold(f32::INFINITY, f32::min);
    let penetration = speed * sim.config().dt;
    println!(
        "collision: the left body's rightmost particle at {left_most:.4} cells, the right body's leftmost at {right_least:.4} (declared penetration {penetration:.4})"
    );
    assert!(worst < 1e-6, "balance {worst:e}");
    assert!(
        moved <= allowed + sums,
        "the momentum moved {moved:e} N s/m, beyond what the solves allow, {allowed:e} plus {sums:e} (sums)"
    );
    assert!(
        left_most <= right_least + penetration,
        "the bodies passed through each other: the left body reaches {left_most} cells, the right body starts at {right_least}"
    );
}

/// Two equal bars meeting head-on separate after the impact. In the
/// centre-of-mass frame each bar meets the symmetry plane as a rigid
/// frictionless wall, the bar-on-wall problem of Doyen, Ern and Piperno
/// (2011, sec. 3.1): the push lasts `2L/c` and the bar leaves; in the lab
/// frame the striker slows, the struck bar leaves faster, and the gap opens.
/// Held here: after the meeting the contact pushes and never pulls (the
/// struck bar ends faster than the striker, the gap grows frame to frame);
/// the momentum's own check is the two blocks' collision's. The
/// exchange's completeness converges with the tolerance (measured, bars
/// 24 x 2 cells, the striker's speed after the meeting: 0.188, 0.0885,
/// 0.0538, 0.0560 of `V` at 1e-1, 1e-2, 1e-3, 1e-4) and is not asserted at
/// this one. Before the admissibility test of the body-body gaps
/// (`weighted_gap`), the striker's back face was paired with the struck
/// bar's front face through its own body and the bars never separated.
#[test]
fn two_equal_bars_meet_push_and_separate() {
    let mut config = SimConfig::earth(80, CELL_M, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    // Bars of 24 x 2 cells 6 cells apart, the striker at 0.25 m/s: they
    // meet after about 0.24 s (frame 15 of 1/60 s).
    let speed = 0.25 / CELL_M;
    let striker_tag = sim.add_body(block(&config, id, (24, 2), 20.0, 0.0));
    let _struck = sim.add_body(block(&config, id, (24, 2), 50.0, 0.0));
    let striker: Vec<bool> = sim
        .particles()
        .user_tag
        .iter()
        .map(|&t| t == striker_tag)
        .collect();
    for (v, &s) in sim.particles_mut().v.iter_mut().zip(&striker) {
        if s {
            *v = Vec2::new(speed, 0.0);
        }
    }
    // Per body its mean velocity along the bars, and the gap between the
    // striker's front particle and the struck bar's back one, cells.
    let read = |sim: &Simulation| -> (f64, f64, f64) {
        let p = sim.particles();
        let (mut ms, mut mt, mut ps, mut pt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut front, mut back) = (f32::NEG_INFINITY, f32::INFINITY);
        for (((&s, &m), v), x) in striker.iter().zip(&p.mass).zip(&p.v).zip(&p.x) {
            let w = f64::from(m);
            if s {
                ms += w;
                ps += w * f64::from(v.x);
                front = front.max(x.x);
            } else {
                mt += w;
                pt += w * f64::from(v.x);
                back = back.min(x.x);
            }
        }
        (ps / ms, pt / mt, f64::from(back - front))
    };
    let mut gaps = Vec::new();
    for _ in 0..20 {
        sim.step();
        let (vs, vt, gap) = read(&sim);
        println!(
            "bars: striker {:.4} of V, struck {:.4} of V, gap {gap:.4} cells",
            vs / f64::from(speed),
            vt / f64::from(speed)
        );
        gaps.push((vs, vt, gap));
    }
    let &(vs, vt, _) = gaps.last().expect("frames");
    assert!(
        vt > vs,
        "the struck bar ({vt}) is not faster than the striker ({vs}): no separation"
    );
    // After the push has ended (its last frames), the gap grows frame to
    // frame: the bars separate and nothing pulls them back.
    let tail = &gaps[gaps.len() - 3..];
    assert!(
        tail.windows(2).all(|w| w[1].2 > w[0].2),
        "the gap does not open: {:?}",
        tail.iter().map(|g| g.2).collect::<Vec<_>>()
    );
}

#[test]
#[should_panic(expected = "does not support force fields")]
fn the_path_refuses_force_fields_it_does_not_carry() {
    let mut sim = scene(config(1e-1), 0.0);
    sim.add_force_field(Box::new(LinearDragField::new(
        Vec2::ZERO,
        1.0,
        LinearDragField::ALL_MATERIALS,
    )));
    sim.step();
}

#[test]
#[should_panic(expected = "does not support phase rules")]
fn the_path_refuses_phase_rules_it_does_not_carry() {
    let mut sim = scene(config(1e-1), 0.0);
    sim.add_phase_rule(|_| None);
    sim.step();
}

#[test]
#[should_panic(expected = "does not support thermal or scalar diffusion")]
fn the_path_refuses_heat_conduction_it_does_not_carry() {
    let config = config(1e-1);
    let mut sim = scene(config, 0.0);
    sim.set_thermal(ThermalDiffusion::new(
        ThermalConfig::default(),
        config.grid_res,
    ));
    sim.step();
}

#[test]
#[should_panic(expected = "does not support muscle activation")]
fn the_path_refuses_a_muscle_activation_it_would_ignore() {
    let mut sim = scene(config(1e-1), 0.0);
    sim.step();
    sim.particles_mut().activation[0] = 0.5;
    sim.step();
}

#[test]
#[should_panic(expected = "does not support adding bodies after its first frame")]
fn the_path_refuses_a_body_added_after_its_first_frame() {
    let config = config(1e-1);
    let mut sim = scene(config, 0.0);
    sim.step();
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let _late = sim.add_body(block(&config, id, (4, 4), 8.0, 0.0));
    sim.step();
}

/// The long rest: 60 simulated seconds on the floor, the balance's drift
/// (the `f64` state's claim: no slow drift from round-tripping `f32`). A
/// probe: run with --ignored --nocapture.
#[test]
#[ignore = "long: 60 s simulated; run with --ignored --nocapture"]
fn probe_a_block_at_rest_for_a_minute() {
    cap::capped("probe_a_block_at_rest_for_a_minute", 3600, || {
        let mut sim = scene(config(1e-1), 0.0);
        let scale = weight_scale(&sim);
        for second in 1..=60 {
            let (worst, deepest) = run(&mut sim, 60);
            let frame = sim.total_lagrangian_frame().expect("the path is on");
            println!(
                "t {second} s: balance {:+.3e} of the weight scale (worst this second {worst:.2e}), deepest {deepest:.2e} cells, steps {} rejected {} at the floor {}",
                frame.balance / scale,
                frame.steps,
                frame.rejected,
                frame.at_floor
            );
        }
    });
}

#[test]
fn a_block_sliding_on_a_frictionless_floor_keeps_its_energy() {
    let config = config(1e-1);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let _block = sim.add_body(block(&config, id, (8, 4), SLIDE_START, 0.0));
    let speed = 0.5 / CELL_M;
    // 20 frames at that speed.
    assert_clear_of_side_walls(&config, SLIDE_START, 4.0, speed * 20.0 / 60.0);
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(speed, 0.0);
    }
    let (worst, _) = run(&mut sim, 20);
    println!("frictionless slide: worst balance {worst:.2e}");
    assert!(worst < 1e-6, "balance {worst:e}");
}

/// A block in free fall gains `g t` of speed, in m/s: the path's state in
/// SI (`m/s^2` gravity, m/s velocities) against the exact answer, to the
/// `f32` copy the particles carry (1e-5 relative).
#[test]
fn a_falling_block_gains_g_t_in_si() {
    let config = config(1e-1);
    let mut sim = scene(config, 15.0);
    let frames = 10;
    for _ in 0..frames {
        sim.step();
    }
    let p = sim.particles();
    let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
    let mean_vy: f64 =
        p.v.iter()
            .zip(&p.mass)
            .map(|(v, &m)| f64::from(v.y) * f64::from(m))
            .sum::<f64>()
            / mass
            * f64::from(CELL_M);
    let exact = -9.81 * f64::from(frames as f32 * config.dt);
    println!("free fall: mean vy {mean_vy:.6} m/s, exact -g t {exact:.6} m/s");
    assert!(
        (mean_vy / exact - 1.0).abs() < 1e-5,
        "free fall {mean_vy} m/s against {exact}"
    );
}

/// A block sliding on a Coulomb floor stops after `v0^2 / (2 mu g)`, in
/// metres (rigid-body Coulomb, the block's centre of mass): within 1
/// percent, the heat criterion's own tolerance, stated before measuring.
/// The block is the measured plasticine (Crandall, Ehrlich and Wenk 1970,
/// its three relaxation mechanisms): a body that dissipates its vibration.
/// The rigid distance is the limit of such a body, not of a lossless one:
/// an undamped elastic block, released unstressed, keeps vibrating, its base
/// leaves the floor and its slip reverses, and it slides further (measured,
/// E 1 MPa: 1.051 of the rigid distance, converged in the step tolerance,
/// 1.0508 at 1e-2 and 1.0512 at 1e-3, its base out of contact at 73 percent
/// of the step ends; the plasticine 0.99998 at both).
#[test]
fn a_sliding_block_stops_after_v0_squared_over_2_mu_g() {
    let config = config(1e-1);
    let record =
        emerge::materials::relaxation::ViscoelasticRecord::plasticine_crandall_1970_fresh();
    let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
    material.relaxation = Some(record.spectrum);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(SLIDE_START, floor + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&record.relaxed, &config),
    );
    sim.add_boundary_condition(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.3,
    )));
    let (v0, mu, g) = (1.0f64, 0.3f64, 9.81f64);
    let stopping = v0 * v0 / (2.0 * mu * g);
    assert_clear_of_side_walls(
        &config,
        SLIDE_START,
        4.0,
        (stopping / f64::from(CELL_M)) as f32,
    );
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new((v0 / f64::from(CELL_M)) as f32, 0.0);
    }
    let centre = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
        p.x.iter()
            .zip(&p.mass)
            .map(|(x, &m)| f64::from(x.x) * f64::from(m))
            .sum::<f64>()
            / mass
            * f64::from(CELL_M)
    };
    let start = centre(&sim);
    // 0.34 s to stop; 30 frames is 0.5 s.
    let _ = run(&mut sim, 30);
    let travel = centre(&sim) - start;
    println!("slide: travel {travel:.5} m, v0^2 / (2 mu g) {stopping:.5} m");
    assert!(
        (travel / stopping - 1.0).abs() < 0.01,
        "travel {travel} m against {stopping} m"
    );
}

/// The sliding block of `a_sliding_block_stops_after_v0_squared_over_2_mu_g`
/// in a von Mises solid (J2 on Hencky elasticity, the record of
/// `a_ductile_block_splats_and_its_plastic_work_becomes_heat`): the centre
/// of mass, decelerated by `mu g` while the whole base slides, stops after
/// `v0^2 / (2 mu g)`. The base's shear, `mu rho g H`, about 200 Pa, stays far
/// below the 20 kPa yield: no plastic flow. The ledger closes within the
/// solves' bound (`run`). Bound on the travel: 0.1 percent, stated before
/// measuring.
/// Known limit (kind: scope; why: no measured dissipation for this record's
/// elastic part): a lossless body is not at the rigid limit (it keeps
/// vibrating and slides further, `a_sliding_block_stops_after_v0_squared_
/// over_2_mu_g`); at this step tolerance its vibration is under-resolved and
/// the travel measured, 1.0008 of the rigid distance, sits near the limit.
/// Removal: a measured dissipation for this record, or the lossless body's
/// own converged answer as the reference.
#[test]
fn a_ductile_block_slides_v0_squared_over_2_mu_g() {
    let config = config(1e-1);
    let props = DuctileProps {
        elastic: Elastic {
            e_pa: 1.0e6,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        },
        yield_stress_pa: 2.0e4,
    };
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(VonMisesMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(SLIDE_START, floor + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props.elastic, &config),
    );
    sim.add_boundary_condition(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.3,
    )));
    let (v0, mu, g) = (1.0f64, 0.3f64, 9.81f64);
    let stopping = v0 * v0 / (2.0 * mu * g);
    assert_clear_of_side_walls(
        &config,
        SLIDE_START,
        4.0,
        (stopping / f64::from(CELL_M)) as f32,
    );
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new((v0 / f64::from(CELL_M)) as f32, 0.0);
    }
    let centre = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
        p.x.iter()
            .zip(&p.mass)
            .map(|(x, &m)| f64::from(x.x) * f64::from(m))
            .sum::<f64>()
            / mass
            * f64::from(CELL_M)
    };
    let start = centre(&sim);
    let _ = run(&mut sim, 30);
    let travel = centre(&sim) - start;
    println!("ductile slide: travel {travel:.5} m, v0^2 / (2 mu g) {stopping:.5} m");
    assert!(
        (travel / stopping - 1.0).abs() < 0.001,
        "travel {travel} m against {stopping} m"
    );
}

/// The first impact of a flat 8 by 4 cm clay block dropped flat from 10 cm
/// on a frictionless floor, 15 frames (the fall, the impact, the rebound's
/// start), at `refine` times finer cells and the box's centre `centre`
/// cells (in 1 cm cells; the same offset from the grid at every
/// refinement). Returns the worst angular momentum about the centre of mass
/// over `I v / r` (`v` the impact speed, `r` the half width: the spin the
/// impact's whole impulse would give at the body's edge), the worst tilt,
/// degrees, and the steps taken.
fn first_impact_spin(centre: f32, refine: usize) -> (f64, f64, u32) {
    let k = refine as f32;
    let cell = CELL_M / k;
    let mut config = SimConfig::earth(RES * refine, cell, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let lift = 10.0 * k;
    let centre = 20.0 * k + (centre - 20.0);
    let (width, height) = (8 * refine as i32, 4 * refine as i32);
    let _block = sim.add_body(block(&config, id, (width, height), centre, lift));
    let impact = f64::from((2.0 * config.gravity.length() * lift).sqrt() * cell);
    let half_width = 0.5 * f64::from(width as f32 * cell);
    let orientation = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let total: f32 = p.mass.iter().sum();
        let c =
            p.x.iter()
                .zip(&p.mass)
                .fold(Vec2::ZERO, |c, (x, &m)| c + m * *x)
                / total;
        let (mut xx, mut yy, mut xy) = (0.0f64, 0.0f64, 0.0f64);
        for (x, &m) in p.x.iter().zip(&p.mass) {
            let d = *x - c;
            xx += f64::from(m * d.x * d.x);
            yy += f64::from(m * d.y * d.y);
            xy += f64::from(m * d.x * d.y);
        }
        0.5 * (2.0 * xy).atan2(xx - yy)
    };
    let start = orientation(&sim);
    let (mut steps, mut worst, mut tilt) = (0u32, 0.0f64, 0.0f64);
    let frames: usize = std::env::var("EMERGE_FLAT_DROP_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(15);
    for _ in 0..frames {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        steps += frame.steps;
        let scale = frame.moment_of_inertia * impact / half_width;
        worst = worst.max(frame.angular_momentum.abs() / scale);
        tilt = tilt.max((orientation(&sim) - start).abs().to_degrees());
        if std::env::var_os("EMERGE_FLAT_DROP_TRACE").is_some() {
            println!(
                "  frame: L {:+.3e} of I v / r, tilt {:+.3e} degrees, contacts {}",
                frame.angular_momentum / scale,
                (orientation(&sim) - start).to_degrees(),
                frame.contacts
            );
        }
    }
    println!(
        "first impact, cell {cell} m, box centre {centre} cells: worst |L| {worst:.3e} of I v / r, worst tilt {tilt:.3e} degrees, {steps} steps"
    );
    (worst, tilt, steps)
}

/// A flat block dropped flat on a flat frictionless floor: nothing can
/// turn it. On a mirror line of the grid (the box centred on a grid line or
/// halfway between two) the discrete problem is mirror symmetric, and its
/// spin is what the solves' tolerances allow. Derived bound: each step's
/// contacts hold to `contact_tolerance` of its motion and its equations to
/// `newton_tolerance`, so a step leaves at most their sum of `I v / r`
/// unbalanced; over `N` steps, `N (tol_c + tol_N)` of it (measured: about
/// 1e-11 against the bound's 4e-7), and the tilt it would give over the
/// run, `bound v / r t`.
///
/// Off the mirror lines the multiplier patches are the body's (the same at
/// every offset) but the grid under its two ends is not, and the first
/// impact leaves a spin of discretization: measured 9.2e-3 to 1.4e-2 of `I
/// v / r` over offsets 0.1 to 0.3 cell at 1 cm cells, tilts 1.9 to 2.8
/// degrees after it, varying continuously with the offset (no jump at the
/// basis's occupation threshold). It must converge: at the worst offset
/// (0.2 cell) the spin and the tilt at half the cell must be at most half
/// those at the cell (first order or better; measured 1.7 at 0.3 cell, 1.1e-2
/// to 3.4e-3). Declared: the later bounces amplify the spin a first impact
/// leaves (a tilted elastic block lands on a corner and turns; no damping
/// in this clay), up to 0.4 of `I v / r` and tens of degrees over 0.75 s at
/// 1 cm cells: a measured behaviour, not asserted here.
#[test]
fn a_flat_drop_does_not_turn_the_block() {
    if let Ok(one) = std::env::var("EMERGE_FLAT_DROP_CENTRE") {
        let refine = std::env::var("EMERGE_FLAT_DROP_REFINE")
            .ok()
            .and_then(|k| k.parse().ok())
            .unwrap_or(1);
        first_impact_spin(one.parse().expect("a centre in cells"), refine);
        return;
    }
    let tolerances = {
        let settings = TotalLagrangianConfig::new(1e-1).settings;
        settings.contact_tolerance + settings.newton_tolerance
    };
    for centre in [20.0f32, 20.5] {
        let (spin, tilt, steps) = first_impact_spin(centre, 1);
        let bound = f64::from(steps) * tolerances + 16.0 * f64::EPSILON;
        // The tilt the bound's spin would give over the 15 frames.
        let impact = (2.0 * 9.81f64 * 0.1).sqrt();
        let tilt_bound = (bound * impact / 0.04 * 0.25).to_degrees();
        assert!(
            spin <= bound,
            "centre {centre}: spin {spin:e} of I v / r above {bound:e}"
        );
        assert!(
            tilt <= tilt_bound,
            "centre {centre}: tilt {tilt:e} degrees above {tilt_bound:e}"
        );
    }
    for centre in [20.1f32, 20.25, 20.3, 20.7] {
        first_impact_spin(centre, 1);
    }
    let (coarse, coarse_tilt, _) = first_impact_spin(20.2, 1);
    let (fine, fine_tilt, _) = first_impact_spin(20.2, 2);
    println!(
        "worst offset: spin {coarse:.3e} -> {fine:.3e} (order {:.2}), tilt {coarse_tilt:.3e} -> {fine_tilt:.3e} degrees",
        (coarse / fine).log2()
    );
    assert!(
        fine <= 0.5 * coarse,
        "the spin does not converge: {coarse:e} -> {fine:e}"
    );
    assert!(
        fine_tilt <= 0.5 * coarse_tilt,
        "the tilt does not converge: {coarse_tilt:e} -> {fine_tilt:e}"
    );
}

/// The weighted gaps' corner sink, measured at two cells: a block resting
/// on the floor (8 by 4 cm), released unstressed, 20 frames; prints the
/// bottom row's deepest sink below its rest height per cell. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_corner_sink_at_two_cells() {
    cap::capped("probe_corner_sink_at_two_cells", 1800, || {
        for (res, cell) in [(40usize, 0.01f32), (80, 0.005)] {
            let mut config = SimConfig::earth(res, cell, 1.0 / 60.0);
            config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
            let mut sim = Simulation::empty(config);
            let id = sim
                .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
                .0;
            let (w, h) = ((0.08 / cell).round() as i32, (0.04 / cell).round() as i32);
            let floor = config.boundary_thickness as f32;
            let _block = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(w, h))
                    .at(Vec2::new(res as f32 * 0.5, floor + h as f32 * 0.5))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&clay(), &config),
            );
            let mut deepest = 0.0f32;
            for _ in 0..20 {
                sim.step();
                let lowest = sim
                    .particles()
                    .x
                    .iter()
                    .map(|q| q.y)
                    .fold(f32::INFINITY, f32::min);
                deepest = deepest.max(floor + 0.25 - lowest);
            }
            println!(
                "cell {} mm: the bottom row's deepest sink below its rest height {:.4} cells = {:.4} mm",
                cell * 1000.0,
                deepest,
                deepest * cell * 1000.0
            );
        }
    });
}

/// The same scene moved by 1 m steps the same: the 4-cell drop as
/// `a_dropped_block_bounces_with_its_energy_kept` sets it, and the scene
/// moved on both axes, floor, walls and block, by a whole number of cells
/// (the walls sit at whole cells), so the grid, the basis and the discrete
/// problem are identical and only the coordinates' magnitudes differ.
/// Asserted over the release, the impact and the rebound (15 frames):
/// - the step tolerance's energy identical (`tolerance_energy`: the same
///   masses, the same extent, the start at rest and unstrained);
/// - the same steps in every frame;
/// - the particles' positions, the shift taken off, within the two `f32`
///   copies' rounding, half an ulp at each run's coordinates: the `f64`
///   states agree far below it unless a decision flips.
///
/// Measured before `tolerance_energy` (the start's total energy, gravity's
/// potential from the grid's corner): 800 steps against 30 over 30 frames,
/// the moved scene taking one step per frame through the impact, the
/// bottom rows 8 mm apart. Measured after, not asserted: past frame 20 the
/// two runs part, round-off amplified through the later contact switches
/// (1e-4 cell at frame 21, 2e-2 at frame 30, frames 29 and 30 differing by
/// 7 percent of their steps).
#[test]
fn a_scene_moved_by_a_metre_steps_the_same() {
    let shift = (1.0 / CELL_M).round() as usize;
    let mut runs = Vec::new();
    for offset in [0usize, shift] {
        let mut config = config(1e-1);
        config.grid_res = RES + 2 * offset;
        config.boundary_thickness += offset;
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
            .0;
        let middle = config.grid_res as f32 * 0.5;
        let _block = sim.add_body(block(&config, id, (8, 4), middle, 4.0));
        let (mut sequence, mut frames) = (Vec::new(), Vec::new());
        for _ in 0..15 {
            sim.step();
            let frame = sim.total_lagrangian_frame().expect("the path is on");
            sequence.push(frame.steps);
            frames.push(sim.particles().x.clone());
        }
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        println!(
            "offset {offset} cells: tolerance energy {:.17e} J/m (start energy {:.4e}), steps per frame {sequence:?}",
            frame.tolerance_energy, frame.start_energy
        );
        runs.push((frame.tolerance_energy, sequence, frames, offset as f32));
    }
    let (here, there) = (&runs[0], &runs[1]);
    assert_eq!(
        here.0, there.0,
        "the tolerance's energy moved with the scene"
    );
    assert_eq!(here.1, there.1, "the step sequence moved with the scene");
    let half_ulp = |x: f32| 0.5 * (x.abs().next_up() - x.abs());
    for (k, (a, b)) in here.2.iter().zip(&there.2).enumerate() {
        for (p, q) in a.iter().zip(b) {
            let moved = *q - Vec2::splat(there.3);
            let bound = Vec2::new(half_ulp(p.x) + half_ulp(q.x), half_ulp(p.y) + half_ulp(q.y));
            let apart = (*p - moved).abs();
            assert!(
                apart.x <= bound.x && apart.y <= bound.y,
                "frame {}: a particle {apart} cells apart, beyond the roundings' {bound}",
                k + 1
            );
        }
    }
}

/// A ductile block (J2 on Hencky elasticity, Love and Sulsky 2005: E 1 MPa,
/// nu 0.3, uniaxial yield 20 kPa, no hardening, 1700 kg/m^3; impact stress
/// `rho c v` about 57 kPa, past yield) dropped 10 cm onto the frictionless
/// floor: it deforms for good, and the energy it loses is exactly the
/// plastic work, turned into heat (Love and Sulsky eq. 83: `1/2 S : dC =
/// dW + dD`, so the step's energy falls by its dissipation and no more).
/// Asserted, each derived:
/// - the balance per frame within the solves' tolerances' bound (`run`),
///   which with no friction and no dashpot is the identity `KE0 + E_grav0 =
///   KE + E_el + E_grav + W_plastic` (the heat is the plastic work alone);
/// - the area kept: J2 flow is isochoric, so the body's area changes only by
///   its elastic part, `int (J_e - 1) dA`, about `int ln J_e dA`, at most
///   `sqrt(A int (ln J_e)^2 dA) <= sqrt(2 A E_el / K)` (Cauchy-Schwarz, the
///   bulk energy `K/2 (ln J_e)^2` being part of the stored energy `E_el`),
///   plus the `f32` particle volumes' round-off;
/// - wider for good: flow at the base under vertical compression is lateral
///   expansion there (isochoric), so the bounding width grows.
///
/// Not asserted: the permanent shape. Measured, the base spreads (7.50 to
/// 8.40 cells) and the box's height barely moves (3.50 to 3.50); predicting
/// that shape needs the plastic wave's theory of an impact on a rigid wall
/// (Taylor's, for a bar), not read here. A first assertion that the block
/// would come out lower, written before measuring, was this unread
/// expectation, and is withdrawn rather than met.
#[test]
fn a_ductile_block_splats_and_its_plastic_work_becomes_heat() {
    let config = config(1e-1);
    let props = DuctileProps {
        elastic: Elastic {
            e_pa: 1.0e6,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        },
        yield_stress_pa: 2.0e4,
    };
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(VonMisesMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let (width, height) = (8, 4);
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(width, height))
            .at(Vec2::new(
                RES as f32 * 0.5,
                floor + 10.0 + height as f32 * 0.5,
            ))
            .spacing(0.5)
            .material(id)
            .mass_from(&props.elastic, &config),
    );
    let extent = |sim: &Simulation| -> (f32, f32) {
        let p = sim.particles();
        let (lo, hi) = p.x.iter().fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(lo, hi), q| (lo.min(*q), hi.max(*q)),
        );
        (hi.x - lo.x, hi.y - lo.y)
    };
    let (w0, h0) = extent(&sim);
    let cell = f64::from(config.dx_meters);
    let area = |sim: &Simulation| -> f64 {
        sim.particles()
            .volume
            .iter()
            .map(|&v| f64::from(v))
            .sum::<f64>()
            * cell
            * cell
    };
    // The reference area is the path's own `V0 = m / rho` (the spawn's
    // packing estimate in `volume` is not, issue #41).
    let a0 = sim
        .particles()
        .mass
        .iter()
        .map(|&m| f64::from(m))
        .sum::<f64>()
        / f64::from(config.density_from_si(props.elastic.rho_kg_m3))
        * cell
        * cell;
    let (worst, _) = run(&mut sim, 40);
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let (w1, h1) = extent(&sim);
    // The plastic work, summed over the run: `run` checked every frame;
    // the heat is the sum of every frame's plastic work (no friction, no
    // dashpot).
    println!(
        "ductile drop: worst balance {worst:.2e}; heat {:.6e} J/m; block {:.2} x {:.2} cells -> {:.2} x {:.2}",
        frame.heat, w0, h0, w1, h1
    );
    assert!(frame.heat > 0.0, "no plastic work");
    assert_eq!(frame.friction_work, 0.0, "the floor is frictionless");
    assert_eq!(frame.dissipated, 0.0, "the material has no dashpot");
    assert!(w1 > w0, "no lateral expansion: width {w0} -> {w1}");
    // The area: `sqrt(2 A E_el / K)`, `K` the material's bulk modulus, plus
    // `n eps_f32` of the area for the particles' volumes.
    let bulk = f64::from(props.elastic.e_pa) / (3.0 * (1.0 - 2.0 * f64::from(props.elastic.nu)));
    let a1 = area(&sim);
    let bound = (2.0 * a0 * frame.parts[1] / bulk).sqrt()
        + sim.particles().len() as f64 * f64::from(f32::EPSILON) * a0;
    println!(
        "area {a0:.6e} -> {a1:.6e} m^2 per m (change {:.3e}, bound {bound:.3e})",
        a1 - a0
    );
    assert!(
        (a1 - a0).abs() <= bound,
        "area {a0} -> {a1}, beyond {bound}"
    );
}
