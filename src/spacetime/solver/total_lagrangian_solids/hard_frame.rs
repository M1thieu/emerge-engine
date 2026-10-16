//! E2's 4th impact frame (`tests/realtime_profile.rs`), which took 46.7 s
//! with the responses' warm start against 8.8 s without, run from one
//! identical state both ways. Equal costs mean the runs' states
//! had diverged by then (the trajectory's sensitivity); unequal costs mean
//! the warm start itself hurts there. A probe: it measures, and changes no
//! answer of the path.

use std::time::Instant;

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// `tests/realtime_profile.rs`'s E2: the 8 x 4 cm clay block dropped 10
/// cells, frame 1/60 s, step tolerance 1e-1.
fn e2() -> Simulation {
    let props = ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 10.0 + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    sim
}

/// The responses' warm start on or off (`hooks::set_cold_responses`).
fn warm_start(on: bool) {
    crate::spacetime::total_lagrangian::hooks::set_cold_responses(!on);
}

/// One frame from `state`, its wall time and its counts.
fn frame_from(sim: &mut Simulation, state: &TotalLagrangianState, on: bool) -> String {
    sim.total_lagrangian = Some(state.clone());
    warm_start(on);
    let clock = Instant::now();
    sim.step();
    let wall = clock.elapsed().as_secs_f64();
    let f = sim.total_lagrangian_frame().expect("on the path");
    format!(
        "warm start {}: {wall:.2} s; steps {}, rejected {}, Newton {}, GMRES {}, passes {}, responses {:.2} s",
        if on { "on " } else { "off" },
        f.steps,
        f.rejected,
        f.newton,
        f.gmres,
        f.passes,
        f.timing.responses
    )
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_e2_hard_frame_from_one_state() {
    // The frame of E2's 4th impact in the profile's frame list (1-based).
    const FRAME: usize = 44;
    for (label, on) in [
        ("the run without warm start", false),
        ("the run with it", true),
    ] {
        let mut sim = e2();
        warm_start(on);
        for _ in 1..FRAME {
            sim.step();
        }
        let state = sim.total_lagrangian.as_ref().expect("on the path").clone();
        println!("frame {FRAME} from the state {label} reached:");
        for second in [false, true] {
            println!("  {}", frame_from(&mut sim, &state, second));
        }
    }
    warm_start(true);
}

/// What the controller does in that frame, from the state the warm-start
/// run reached: its counters by cause (`TotalLagrangianFrame`), and every
/// try printed (`hooks::set_control_trace`): which branch rejects and whether the step-ratio clamp or the proposal
/// after a rejection streak sets the count. Report only.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_e2_hard_frame_controller() {
    const FRAME: usize = 44;
    let mut sim = e2();
    warm_start(true);
    for _ in 1..FRAME {
        sim.step();
    }
    crate::spacetime::total_lagrangian::hooks::set_control_trace(true);
    sim.step();
    crate::spacetime::total_lagrangian::hooks::set_control_trace(false);
    let f = sim.total_lagrangian_frame().expect("on the path");
    println!(
        "frame {FRAME}: kept {}, rejected {} (Newton unconverged {}), at the estimator's floor {}, reaching {}, passes {}, unconverged kept {}, stagnated {}",
        f.steps,
        f.rejected,
        f.rejected_unconverged,
        f.at_floor,
        f.reaching,
        f.passes,
        f.unconverged,
        f.stagnated
    );
}

/// A block at rest on the floor (`tests/realtime_profile.rs`'s E1): why its
/// frames still build Jacobian factors, each build printed with what was
/// new (the step's inertia coefficient, or the held springs).
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_rest_factorizations() {
    let props = ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
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
    for k in 1..=6 {
        crate::spacetime::total_lagrangian::hooks::set_control_trace(k > 4);
        sim.step();
        let f = sim.total_lagrangian_frame().expect("on the path");
        println!(
            "frame {k}: steps {}, rejected {}, passes {}, factorizations {}",
            f.steps, f.rejected, f.passes, f.timing.factorizations
        );
    }
    crate::spacetime::total_lagrangian::hooks::set_control_trace(false);
}
