//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/basic_showcase.rs`'s scene (no GUI, no arrow-key/mouse
//! interaction) to verify the sand terrain's real SI migration
//! (E=15 MPa/nu=0.3/rho=1600, same citation as basic_sand.rs) is stable
//! alongside the unmigrated elastic body and the already-real fluid, under
//! this scene's own deliberately-weak gravity.
extern crate emerge_engine as emerge;

use emerge::{
    DruckerPragerMaterial, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const ELASTIC_ID: u32 = 0;
const SAND_ID: u32 = 1;
const FLUID_ID: u32 = 2;
const SPACING: f32 = 0.7;

const SAND_YOUNG_MODULUS_PA: f32 = 15.0e6;
const SAND_POISSON_RATIO: f32 = 0.3;
const SAND_DENSITY_KG_M3: f32 = 1600.0;

fn make_sim(max_substeps_per_step: usize) -> Simulation {
    let config = SimConfig {
        min_dt: 0.005,
        max_substeps_per_step,
        recompute_density_each_step: true,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let elastic = NeoHookeanMaterial::new(40.0, 80.0);
    let (sand_lambda, sand_mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    let sand = DruckerPragerMaterial::new(sand_lambda, sand_mu);
    let fluid = NewtonianFluidMaterial::low_viscosity(0.1, 0.25);
    let sand_mass = (SAND_DENSITY_KG_M3 / config.reference_density_kg_m3) * SPACING * SPACING;

    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(elastic))
        .with_material(SAND_ID, Box::new(sand))
        .with_material(FLUID_ID, Box::new(fluid))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(19.0, 9.0),
        material_id: SAND_ID,
        precompute_initial_volumes: true,
        mass_override: Some(sand_mass),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(45.0, 9.0),
        material_id: FLUID_ID,
        precompute_initial_volumes: true,
        mass_override: Some(0.1 * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(12, 12),
        box_center: Vec2::new(32.0, 46.0),
        material_id: ELASTIC_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    });
    solver
}

fn run_probe(label: &str, max_substeps_per_step: usize, steps: u64) {
    let mut sim = make_sim(max_substeps_per_step);
    for step in 1..=steps {
        sim.step();
        if step.is_multiple_of(steps / 20) || step == 1 || step == steps {
            let snap = sim.diagnostics_snapshot();
            println!(
                "[{label}] step={step} t={:.2} sub={} J=[{:.4},{:.4}] cfl={:.4} \
                 non_finite={} mass_err={:.2e} time_dropped={:.4}",
                step as f32 * DT,
                snap.substeps_last_step,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.cfl_number,
                snap.non_finite_particle_values,
                snap.relative_mass_error,
                snap.sim_time_dropped,
            );
            assert_eq!(
                snap.non_finite_particle_values, 0,
                "[{label}] NaN/Inf at step {step}"
            );
        }
    }
    let snap = sim.diagnostics_snapshot();
    assert!(
        snap.sim_time_dropped < 1.0e-6,
        "[{label}] sim_time_dropped={} -- max_substeps_per_step too low",
        snap.sim_time_dropped
    );
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_showcase_real_sand_stiffness_settles() {
    run_probe("substeps=3000", 3000, 300);
}
