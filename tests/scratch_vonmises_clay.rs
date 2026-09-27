//! Headless probe of `basic_vonmises`'s clay scene
//! (`examples/cpu/vonmises_clay_scene.rs`, included, not copied): each
//! blob's shape, plastic strain and landing, and the scene's cost, at real
//! gravity. A probe, no criterion.

extern crate emerge_engine as emerge;

#[path = "../examples/cpu/vonmises_clay_scene.rs"]
mod vonmises_clay_scene;
use vonmises_clay_scene::*;

#[test]
#[ignore = "probe: run with --ignored --nocapture"]
fn the_clay_blobs_at_real_gravity() {
    let (mut sim, materials) = make_sim(1.0);
    println!(
        "{} particles, up to {} substeps a frame of {DT:.4} s",
        sim.particles().len(),
        sim.config().max_substeps_per_step
    );
    let report = |sim: &emerge::Simulation, frame: usize| {
        for (slot, (name, _)) in CLAYS.iter().enumerate() {
            let (mut lo, mut hi, mut top, mut bottom) = (f32::MAX, f32::MIN, f32::MIN, f32::MAX);
            let (mut kappa, mut j, mut vy, mut n, mut at_yield) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let p = sim.particles();
            for i in 0..p.len() {
                if p.material_id[i] != slot as u32 {
                    continue;
                }
                let x = p.x[i];
                lo = lo.min(x.x);
                hi = hi.max(x.x);
                top = top.max(x.y);
                bottom = bottom.min(x.y);
                kappa = kappa.max(p.friction_hardening[i]);
                j += p.deformation_gradient[i].determinant();
                vy += p.v[i].y;
                n += 1.0;
                at_yield += f32::from(u8::from(materials[slot].yield_ratio(p, i) >= 0.99));
            }
            println!(
                "  frame {frame} {name}: height {:.2} cells, width {:.2}, bottom {:.2}, largest kappa {kappa:.3}, mean J {:.4}, mean v_y {:.3} cells/s, at yield {:.2}",
                top - bottom,
                hi - lo,
                bottom,
                j / n,
                vy / n,
                at_yield / n
            );
        }
    };
    report(&sim, 0);
    let started = std::time::Instant::now();
    let (mut substeps, mut dropped) = (0usize, 0.0f32);
    for frame in 1..=60 {
        sim.step();
        substeps += sim.last_substeps();
        dropped += sim.diagnostics_snapshot().sim_time_dropped;
        if frame % 10 == 0 {
            report(&sim, frame);
        }
    }
    let wall = started.elapsed().as_secs_f32();
    println!(
        "1 s asked, {dropped:.2e} s of it dropped at the substep cap, in {wall:.1} s wall, {:.1} substeps a frame on average, {:.2} ms per substep, simulated time at {:.3}x real time",
        substeps as f32 / 60.0,
        wall * 1000.0 / substeps as f32,
        1.0 / wall
    );
}
