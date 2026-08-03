//! Temporary diagnostic: real per-phase CPU step() timing breakdown at a
//! realistic particle count, using the engine's own existing `StepTiming`
//! instrumentation (no new profiler needed). Run with --release; debug-mode
//! timing is noise (see perf_opportunities_survey memory's own false-alarm
//! entries).
extern crate emerge_engine as emerge;

use emerge::prelude::*;
use glam::Vec2;

fn main() {
    let config = SimConfig::standard(96, 0.001, Vec2::NEG_Y * 9.8);
    let spawn = SpawnRegion::for_sim(&config)
        .at(Vec2::new(48.0, 48.0))
        .box_of(glam::IVec2::new(70, 70))
        .spacing(0.5)
        .material(0);
    let mut solver = Simulation::new(config, spawn)
        .with_material(0, Box::new(DruckerPragerMaterial::cohesionless(1.0e5, 0.3)));

    // Warm up (settle past initial transients).
    for _ in 0..30 {
        solver.step();
    }

    let n = 60;
    let mut totals = emerge::diagnostics::StepTiming::default();
    for _ in 0..n {
        solver.step();
        let t = solver.diagnostics_snapshot().timing;
        totals.p2g_us += t.p2g_us;
        totals.grid_update_us += t.grid_update_us;
        totals.g2p_us += t.g2p_us;
        totals.fields_us += t.fields_us;
        totals.thermal_us += t.thermal_us;
        totals.cfl_us += t.cfl_us;
        totals.spatial_hash_us += t.spatial_hash_us;
        totals.phase_sleep_us += t.phase_sleep_us;
        totals.project_us += t.project_us;
        totals.density_us += t.density_us;
        totals.total_us += t.total_us;
    }

    let particle_count = solver.particles().len();
    println!("particle_count={particle_count}");
    println!("avg total_us={:.1}", totals.total_us as f64 / n as f64);
    println!("avg p2g_us={:.1}", totals.p2g_us as f64 / n as f64);
    println!(
        "avg grid_update_us={:.1}",
        totals.grid_update_us as f64 / n as f64
    );
    println!("avg g2p_us={:.1}", totals.g2p_us as f64 / n as f64);
    println!("avg fields_us={:.1}", totals.fields_us as f64 / n as f64);
    println!("avg thermal_us={:.1}", totals.thermal_us as f64 / n as f64);
    println!("avg cfl_us={:.1}", totals.cfl_us as f64 / n as f64);
    println!(
        "avg spatial_hash_us={:.1}",
        totals.spatial_hash_us as f64 / n as f64
    );
    println!(
        "avg phase_sleep_us={:.1}",
        totals.phase_sleep_us as f64 / n as f64
    );
    println!("avg project_us={:.1}", totals.project_us as f64 / n as f64);
    println!("avg density_us={:.1}", totals.density_us as f64 / n as f64);
    let accounted = totals.p2g_us
        + totals.grid_update_us
        + totals.g2p_us
        + totals.fields_us
        + totals.thermal_us
        + totals.cfl_us
        + totals.spatial_hash_us
        + totals.phase_sleep_us
        + totals.project_us
        + totals.density_us;
    println!(
        "unaccounted_us={:.1} ({:.1}% of total)",
        (totals.total_us - accounted) as f64 / n as f64,
        100.0 * (totals.total_us - accounted) as f64 / totals.total_us as f64
    );
}
