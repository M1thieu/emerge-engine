//! Criteria for rebuilding multi-field contact (issue #49), frozen before any
//! change to `Grid::resolve_contact`.
//!
//! The method is Nairn, Hammerquist and Smith 2020 (CMAME 362, 112859; the
//! revised author PDF, whose section 2.2 carries the authors' own sign
//! correction to eq. 10): contact exists only where two bodies approach
//! (eq. 14) AND touch, their separation measured from the particles' deformed
//! edges being negative (eq. 22 to 25); the correction is eq. 7 to 10 with
//! Coulomb friction. The Baumgarte velocity floor goes; the small nodal mass
//! bound of Bardenhagen et al. 2001 (section 2.2) stays. Whether emerge needs
//! the paper's second correction per step (section 3.2) is decided by
//! criterion 1, not by preference.
//!
//! # Criteria
//!
//! 1. Perfect interface (the paper's section 4.1, loaded by gravity instead
//!    of a pushed top, which emerge cannot prescribe): two identical
//!    NeoHookean blocks stacked with their interface on a grid line, the top
//!    one a separate contact body, settle under real gravity on the floor.
//!    Their top surface and interface follow the same blocks run as one body
//!    within 1 percent of the one body's own largest compression, at every
//!    frame of 2 simulated seconds.
//!
//!    Amended after the first measurement, with the reason: gravity ramps
//!    from zero to real over the first 200 frames instead of switching on,
//!    so the load is monotonic and slow like the paper's push. Switched on at
//!    once, the column rings, the interface falls into tension, and two
//!    bodies rightly part where one body cannot: 26 percent on the rebuilt
//!    contact, with Poisson's ratio 0 as well as 0.3, 0.3 percent once
//!    ramped. The material is softer (E 10 kPa) so the bottom reaches about
//!    20 percent strain, the paper's range.
//! 2. Contact time (the paper's Fig. 2): a block moving at constant speed
//!    toward a block at rest, no gravity, their facing edges 1.5 cells apart;
//!    the resting block starts moving within one frame of the time its edge
//!    is reached, with the gap closing on a grid line and mid-cell.
//! 3. Resting contact: a block on a slab under real gravity, no Baumgarte;
//!    after settling its centre is within a tenth of a cell of the elastic
//!    rest position, its vertical centre-of-mass speed under `g dt`, and the
//!    slab carries its weight within 5 percent.
//! 4. Fast impact: the same block dropped from 10 cells onto the slab does
//!    not pass into it (no block particle below the slab's top edge by more
//!    than a quarter cell) and comes to rest on it.
//! 5. Coulomb: the resting block given a horizontal speed decelerates at
//!    `mu g` within 5 percent until it sticks; at `mu = 0` it keeps its speed.
//! 6. Directional grip: easy and resisted directions decelerate at
//!    `mu_easy g` and `mu_resist g` within 5 percent.
//! 7. `tests/physics_correctness.rs`: the DP floor tests (the elastic
//!    control's rest body no longer crushed; today its smallest J is 0.0136),
//!    `multi_field_contact_produces_real_coulomb_slip_and_stick`,
//!    `directional_contact_grip_is_real_and_direction_aware`, and
//!    `grip_friction_locomotion_sweep`, each rechecked with its numbers.
//! 8. Unit tests: an approaching pair stops closing and rubs, a separating
//!    pair is left free (`src/spacetime/grid/contact.rs`).
//! 9. CPU/GPU parity on 1 to 6, and no constant beyond the paper's.
//!
//! Criteria 1 and 2 come first, CPU only, measured on the current code
//! before anything changes.

extern crate emerge_engine as emerge;

use emerge::{Elastic, FromSI, NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DX_M: f32 = 0.01;
const SPACING: f32 = 0.5;

/// A soft solid, chosen so each criterion's strains and speeds stay small
/// and cheap: 1000 kg/m3, E 50 kPa, nu 0.3. A numerical-method test, not a
/// material claim.
fn body() -> Elastic {
    Elastic {
        e_pa: 50.0e3,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    }
}

/// Two blocks `CELLS` wide and `HEIGHT` cells tall each, stacked on the
/// floor with the interface on a grid line; the top one is contact group 1
/// when `two_bodies`, otherwise both are one body. Returns the simulation
/// and the index range of the top block.
fn stacked(two_bodies: bool, frame_dt: f32) -> (Simulation, std::ops::Range<usize>) {
    const CELLS: i32 = 12;
    const HEIGHT: i32 = 10;
    let config = SimConfig::earth(GRID, DX_M, frame_dt);
    let floor = config.boundary_thickness as f32;
    // Softer than `body()`: about 20 percent strain at the bottom under
    // `rho g` over 20 cells.
    let soft = Elastic {
        e_pa: 10.0e3,
        ..body()
    };
    let block = |centre_y: f32| {
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(CELLS, HEIGHT),
            box_center: Vec2::new(GRID as f32 * 0.5, centre_y),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&soft, &config)
    };
    let material = NeoHookeanMaterial::from_physical(&soft, &config);
    let mut sim = Simulation::new(config, block(floor + HEIGHT as f32 * 0.5))
        .with_default_material(Box::new(material));
    let start = sim.particles().len();
    let _ = sim.add_body(block(floor + HEIGHT as f32 * 1.5));
    let end = sim.particles().len();
    if two_bodies {
        for i in start..end {
            sim.particles_mut().contact_group[i] = 1;
        }
    }
    (sim, start..end)
}

fn mean_y(sim: &Simulation, range: std::ops::Range<usize>, top: bool) -> f32 {
    let p = sim.particles();
    let ys: Vec<f32> = range.map(|i| p.x[i].y).collect();
    if top {
        ys.iter().copied().fold(f32::MIN, f32::max)
    } else {
        ys.iter().copied().fold(f32::MAX, f32::min)
    }
}

/// Criterion 1.
#[test]
#[ignore = "contact rebuild criterion 1: run with --ignored --nocapture"]
fn criterion1_a_perfect_interface_behaves_as_one_body() {
    let frame_dt = 1.0 / 120.0;
    let (mut one, top_one) = stacked(false, frame_dt);
    let (mut two, top_two) = stacked(true, frame_dt);
    let (top0, interface0) = (
        mean_y(&one, top_one.clone(), true),
        mean_y(&one, top_one.clone(), false),
    );
    let (mut worst, mut largest) = (0.0f32, 0.0f32);
    // Gravity ramps up over the first 200 frames: see criterion 1's doc.
    let g = one.config().gravity;
    for frame in 1..=240 {
        let ramp = (frame as f32 / 200.0).min(1.0);
        one.set_gravity(g * ramp);
        two.set_gravity(g * ramp);
        one.step();
        two.step();
        let (t1, i1) = (
            mean_y(&one, top_one.clone(), true),
            mean_y(&one, top_one.clone(), false),
        );
        let (t2, i2) = (
            mean_y(&two, top_two.clone(), true),
            mean_y(&two, top_two.clone(), false),
        );
        largest = largest.max(top0 - t1).max(interface0 - i1);
        worst = worst.max((t1 - t2).abs()).max((i1 - i2).abs());
        if frame % 40 == 0 {
            println!(
                "frame {frame}: top {t1:.3} one body, {t2:.3} two; interface {i1:.3}, {i2:.3}"
            );
        }
    }
    println!(
        "largest compression of the one body {largest:.3} cells, largest difference {worst:.3} cells, {:.1} percent",
        100.0 * worst / largest
    );
    assert!(
        worst <= 0.01 * largest,
        "two bodies differ from one by {worst:.3} cells against a compression of {largest:.3}"
    );
}

/// Criterion 2: the resting block's first motion against the time the
/// moving block's edge reaches it. `offset` shifts both blocks so the gap
/// closes on a grid line (0.0) or mid-cell (0.5).
fn contact_time(offset: f32) -> (f32, f32) {
    const CELLS: i32 = 8;
    const HEIGHT: i32 = 6;
    const GAP_CELLS: f32 = 1.5;
    const SPEED_CELLS_S: f32 = 15.0;
    let frame_dt = 1.0e-3;
    let mut config = SimConfig::earth(GRID, DX_M, frame_dt);
    config.gravity = Vec2::ZERO;
    let block = |centre_y: f32| {
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(CELLS, HEIGHT),
            box_center: Vec2::new(GRID as f32 * 0.5, centre_y),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&body(), &config)
    };
    let low_centre = 20.0 + offset;
    let high_centre = low_centre + HEIGHT as f32 + GAP_CELLS;
    let material = NeoHookeanMaterial::from_physical(&body(), &config);
    let mut sim =
        Simulation::new(config, block(low_centre)).with_default_material(Box::new(material));
    let low = 0..sim.particles().len();
    let _ = sim.add_body(block(high_centre));
    let high = low.end..sim.particles().len();
    for i in high.clone() {
        sim.particles_mut().contact_group[i] = 1;
        sim.particles_mut().v[i] = Vec2::new(0.0, -SPEED_CELLS_S);
    }
    // Facing edges: particle centres plus half a spacing.
    let p = sim.particles();
    let low_edge = low.clone().map(|i| p.x[i].y).fold(f32::MIN, f32::max) + SPACING * 0.5;
    let high_edge = high.clone().map(|i| p.x[i].y).fold(f32::MAX, f32::min) - SPACING * 0.5;
    let expected = (high_edge - low_edge) / SPEED_CELLS_S;
    let mut onset = f32::NAN;
    for frame in 1..=400 {
        sim.step();
        let p = sim.particles();
        let n = low.len() as f32;
        let vy = low.clone().map(|i| p.v[i].y).sum::<f32>() / n;
        if vy < -0.01 * SPEED_CELLS_S {
            onset = frame as f32 * frame_dt;
            break;
        }
    }
    (onset, expected)
}

/// Criterion 2.
#[test]
#[ignore = "contact rebuild criterion 2: run with --ignored --nocapture"]
fn criterion2_contact_starts_when_the_edges_meet() {
    let mut failures = Vec::new();
    for (label, offset) in [("grid line", 0.0), ("mid-cell", 0.5)] {
        let (onset, expected) = contact_time(offset);
        println!("{label}: resting block moves at {onset:.4} s, edges meet at {expected:.4} s");
        if !(onset - expected).abs().le(&1.0e-3) {
            failures.push(format!("{label}: {onset:.4} s against {expected:.4} s"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
