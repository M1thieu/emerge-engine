//! Start at equilibrium (`settle_total_lagrangian_at_rest`): its design's
//! checks (a)-(c), criteria fixed before the first run, and its refusals.

use super::*;
use crate::matter::materials::physical_props::Elastic as Measured;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

const CELL_M: f32 = 0.01;

/// The integration tests' illustrative solid (`tests/total_lagrangian_path.rs`).
fn illustrative() -> Measured {
    Measured::ILLUSTRATIVE
}

/// A block of `size` cells on the floor of a `res`-cell domain with
/// frictionless walls, its centre at `x` cells, its bottom `lift` cells
/// above the floor; the path on, no modal rest.
fn scene(res: usize, size: (i32, i32), x: f32, lift: f32) -> Simulation {
    let mut config = SimConfig::earth(res, CELL_M, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(
            &illustrative(),
            &config,
        )))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(size.0, size.1))
            .at(Vec2::new(x, floor + lift + 0.5 * size.1 as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&illustrative(), &config),
    );
    sim
}

/// The check scene: a 12 by 2 cell layer exactly between the side walls of
/// a 16-cell domain (walls 2 cells thick), so no rigid motion is free.
fn confined_layer() -> Simulation {
    scene(16, (12, 2), 8.0, 0.0)
}

/// (a) and (b): the settle converges (every increment, the last residual
/// within the step's tolerance), and on frictionless walls its end shape is
/// the linear held-set offset `b / omega^2` (`rest::Rest::settle` from the
/// spawn, on the settle's compressive held set) to O(strain): the largest
/// position difference over the largest offset at most 1e-4 (the static
/// strain is 3.0e-5).
#[test]
fn a_confined_layer_settles_on_its_linear_offset() {
    let mut sim = confined_layer();
    let at = sim
        .settle_total_lagrangian_at_rest()
        .unwrap_or_else(|refused| panic!("the layer settles: {refused:?}"));
    println!(
        "settle: {} increments kept, {} solved, Newton {}, passes {}, GMRES {}, factorizations {}; last residual {:.3e} of its scale (tolerance {:.0e}); end change {:.3e} against {:.3e}; deepest {:.3e} m; held {}",
        at.increments,
        at.solved,
        at.newton,
        at.passes,
        at.gmres,
        at.factorizations,
        at.residual,
        at.tolerance,
        at.change,
        at.target,
        at.deepest,
        at.held
    );
    assert!(
        at.residual <= at.tolerance,
        "(a) residual {:e}",
        at.residual
    );
    let state = sim.total_lagrangian.as_ref().expect("settled");
    let (planes, friction) = sim.total_lagrangian_walls();
    let gravity = sim.total_lagrangian_gravity();
    let length = f64::from(sim.config().dt);
    let mut held = state.clone();
    let cache = held.cache.as_mut().expect("solved");
    let before = cache.contacts.len();
    cache.contacts.retain(|c| c.multiplier > 0.0);
    println!(
        "held set: {} conditions compressive of {before}",
        cache.contacts.len()
    );
    let grid = cache.grid.clone();
    let basis = rest::Basis::build(&held, (&grid, &planes, friction)).expect("a basis");
    // Started from the settled state `u_s` (its held gaps kept, `G u_s + g =
    // 0`), the linear rest's offset is `u_s + Z (Z^T K Z)^-1 Z^T (f - K u_s)
    // = Z (Z^T K Z)^-1 Z^T f`: the linear held-set solution, whatever `u_s`.
    // From the spawn instead, `Rest::start` refuses: its multipliers are the
    // start instant's, and there the side walls carry nothing yet.
    let mut linear_state = held.clone();
    let linear = rest::Rest::start_for_settling(
        basis,
        &linear_state,
        (&grid, &planes),
        (gravity, 0.0, length),
    )
    .unwrap_or_else(|refused| {
        panic!(
            "the linear rest from the settled state refused (beyond the limit {})",
            refused.beyond_limit
        )
    });
    linear.settle(0.0, &mut linear_state);
    let n = state.particles;
    let (mut apart, mut offset) = (0.0f64, 0.0f64);
    for p in 0..n {
        apart = apart.max((state.x[p] - linear_state.x[p]).length());
        offset = offset.max((state.x[p] - state.reference[p]).length());
    }
    println!(
        "(b) static against linear: largest difference {apart:.3e} m over the largest offset {offset:.3e} m = {:.3e}",
        apart / offset
    );
    assert!(apart / offset <= 1e-4, "(b) {:e}", apart / offset);
}

/// (c): 30 frames from the settled layer at `v = 0` stay at rest: each
/// frame's kinetic energy plus the stored energy's change from the start
/// within the run's ledger allowance (the solves' residual work and contact
/// work bounds, accumulated) plus the sums' round-off, `run_measured`'s bar.
#[test]
fn a_settled_layer_stays_at_rest() {
    let mut sim = confined_layer();
    sim.settle_total_lagrangian_at_rest()
        .unwrap_or_else(|refused| panic!("the layer settles: {refused:?}"));
    let state = sim.total_lagrangian.as_ref().expect("settled");
    let n = state.particles;
    let start: Vec<DVec2> = state.x[..n].to_vec();
    // Which held conditions end the settle at a weighted wall gap of zero or
    // more: the step holds a condition only from `g < 0` (forecast off), so
    // such a condition is free at the first step.
    let (planes, _) = sim.total_lagrangian_walls();
    let cache = state.cache.as_ref().expect("settled");
    let gaps: Vec<(ContactKey, f64, f64)> = cache
        .contacts
        .iter()
        .map(|c| {
            let wall = planes[c.key.2];
            let gap: f64 = c
                .points
                .iter()
                .map(|(point, w)| w * wall.inward * (state.x[point.particle][wall.axis] - wall.at))
                .sum();
            (c.key, gap, c.multiplier)
        })
        .collect();
    println!(
        "(c) reading: the settle's {} held conditions (key, weighted wall gap m, multiplier): {gaps:.3?}; at gap >= 0: {}",
        gaps.len(),
        gaps.iter().filter(|g| g.1 >= 0.0).count()
    );
    let stored0: f64 = (0..n)
        .map(|p| state.v0[p] * state.solids[p].energy(state.f[p]))
        .sum();
    let (mut allowed, mut worst, mut moved_most) = (0.0f64, 0.0f64, 0.0f64);
    let (mut steps, mut rejected, mut passes, mut newton, mut factorizations) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    for k in 0..30 {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("on the path");
        steps += u64::from(frame.steps);
        rejected += u64::from(frame.rejected);
        passes += u64::from(frame.passes);
        newton += u64::from(frame.newton);
        factorizations += frame.timing.factorizations;
        allowed += frame.residual_work + frame.contact_work_bound;
        let round_off = n as f64
            * f64::EPSILON
            * (frame.parts.iter().map(|e| e.abs()).sum::<f64>()
                + frame.heat.abs()
                + frame.start_energy.abs());
        let moved = frame.parts[0] + (frame.parts[1] - stored0).abs();
        if k == 0 {
            let keys = rest::held_keys(sim.total_lagrangian.as_ref().expect("on the path"));
            println!(
                "(c) reading, frame 1: most contacts held {}, held at its end {keys:?}; kinetic {:.3e}, stored change {:+.3e} J/m",
                frame.contacts,
                frame.parts[0],
                frame.parts[1] - stored0
            );
        }
        worst = worst.max(moved / (allowed + round_off));
        let state = sim.total_lagrangian.as_ref().expect("on the path");
        for (now, then) in state.x.iter().zip(&start) {
            moved_most = moved_most.max((*now - *then).length());
        }
        assert!(
            moved <= allowed + round_off,
            "(c) frame {}: kinetic {:e} + stored change {:e} beyond {:e}",
            k + 1,
            frame.parts[0],
            (frame.parts[1] - stored0).abs(),
            allowed + round_off
        );
    }
    println!(
        "(c) 30 frames from rest: kinetic plus stored change at most {worst:.3e} of the allowance; largest position change {moved_most:.3e} m; steps {steps}, rejected {rejected}, passes {passes}, Newton {newton}, factorizations {factorizations}"
    );
}

/// A block on frictionless walls touching only the floor: its horizontal
/// translation is free and the load has no component on it, so it is
/// neutral; the settle fixes it by its gauge and converges, the block's mean
/// horizontal displacement printed. A block touching no wall: refused (no
/// support); after a frame: refused.
#[test]
fn the_settle_refuses_what_has_no_unique_rest() {
    let mut on_floor = scene(40, (8, 4), 20.0, 0.0);
    let at = on_floor
        .settle_total_lagrangian_at_rest()
        .unwrap_or_else(|refused| panic!("a frictionless floor alone settles: {refused:?}"));
    let state = on_floor.total_lagrangian.as_ref().expect("settled");
    let n = state.particles;
    let mass: f64 = state.mass[..n].iter().sum();
    let mean = (0..n).fold(DVec2::ZERO, |sum, p| {
        sum + state.mass[p] * (state.x[p] - state.reference[p])
    }) / mass;
    println!(
        "floor only: gauged axis {:?}, {} increments, {} solved, Newton {}, factorizations {}, residual {:.3e} (tolerance {:.0e}); mean displacement ({:+.3e}, {:+.3e}) m",
        at.gauged,
        at.increments,
        at.solved,
        at.newton,
        at.factorizations,
        at.residual,
        at.tolerance,
        mean.x,
        mean.y
    );
    assert_eq!(at.gauged, Some(0), "the horizontal translation gauged");
    assert!(at.residual <= at.tolerance, "residual {:e}", at.residual);
    let mut lifted = scene(40, (8, 4), 20.0, 2.0);
    assert!(matches!(
        lifted.settle_total_lagrangian_at_rest(),
        Err(SettleRefused::FreeRigidMode(_))
    ));
    let mut running = confined_layer();
    running.step();
    assert_eq!(
        running.settle_total_lagrangian_at_rest(),
        Err(SettleRefused::AlreadyRunning)
    );
}
