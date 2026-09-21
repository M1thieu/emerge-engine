//! Core audit (2026-09-21): which defect of the pressure projection amplifies
//! divergence instead of removing it? Archive-branch diagnostic, run with the
//! `EMERGE_P_*` toggles of `src/spacetime/grid/pressure.rs`, one at a time:
//!
//!   EMERGE_P_COMPACT=1 cargo test --profile quick \
//!       --test diag_pressure_projection_amplification -- --ignored --nocapture --test-threads=1
//!
//! Toggles: EMERGE_P_COMPACT (S1), EMERGE_P_SURFACE_IN_SOLVE (S2),
//! EMERGE_P_RELAX1 (S3), EMERGE_P_NO_FILTER (S4), EMERGE_P_CONST_ALPHA (S5).
extern crate emerge_engine as emerge;

use emerge::{Grid, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
use std::panic::{AssertUnwindSafe, catch_unwind};

const GRID: usize = 64;
const DT: f32 = 0.1;
const SPACING: f32 = 0.6;
const WATER_MASS: f32 = 0.1 * SPACING * SPACING;

fn toggles() -> String {
    let names = [
        "EMERGE_P_COMPACT",
        "EMERGE_P_SURFACE_IN_SOLVE",
        "EMERGE_P_RELAX1",
        "EMERGE_P_NO_FILTER",
        "EMERGE_P_CONST_ALPHA",
    ];
    let on: Vec<&str> = names
        .into_iter()
        .filter(|n| std::env::var(n).is_ok_and(|v| v == "1"))
        .collect();
    if on.is_empty() {
        "variant C only".into()
    } else {
        on.join("+")
    }
}

fn config(gravity_fraction: f32, pressure_iterations: u32) -> SimConfig {
    SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: pressure_iterations,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        gravity: Vec2::new(0.0, -981.0 * gravity_fraction),
        ..SimConfig::earth(GRID, 0.01, DT)
    }
}

fn water_sim(config: SimConfig, box_size: IVec2, center: Vec2) -> Simulation {
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size,
        box_center: center,
        material_id: 0,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn)
        .with_default_material(Box::new(NewtonianFluidMaterial::low_viscosity(0.1, 0.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

/// Deterministic value in [-1, 1] per node.
fn noise(pos: IVec2, salt: u32) -> f32 {
    let mut x = (pos.x as u32).wrapping_mul(0x9E37_79B9)
        ^ (pos.y as u32).wrapping_mul(0x85EB_CA6B)
        ^ salt.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Divergence with variant C's reading (massless neighbour = centre velocity):
/// central differences over 2h, or the backward difference S1 uses.
fn divergence(grid: &Grid, pos: IVec2, compact: bool) -> f32 {
    let own = grid.velocity_at(pos);
    let read = |q: IVec2| {
        if grid.mass_at(q) > 0.0 {
            grid.velocity_at(q)
        } else {
            own
        }
    };
    if compact {
        (own.x - read(pos - IVec2::X).x) + (own.y - read(pos - IVec2::Y).y)
    } else {
        0.5 * (read(pos + IVec2::X).x - read(pos - IVec2::X).x)
            + 0.5 * (read(pos + IVec2::Y).y - read(pos - IVec2::Y).y)
    }
}

/// RMS divergence over fluid nodes, and over interior nodes only.
fn rms_divergence(grid: &Grid, nodes: &[IVec2], compact: bool) -> (f32, f32) {
    let interior = |p: IVec2| {
        [IVec2::X, -IVec2::X, IVec2::Y, -IVec2::Y]
            .iter()
            .all(|&o| grid.mass_at(p + o) > 0.0)
    };
    let (mut all, mut n_all, mut inner, mut n_inner) = (0.0f64, 0usize, 0.0f64, 0usize);
    for &p in nodes {
        let d = divergence(grid, p, compact) as f64;
        all += d * d;
        n_all += 1;
        if interior(p) {
            inner += d * d;
            n_inner += 1;
        }
    }
    (
        (all / n_all.max(1) as f64).sqrt() as f32,
        (inner / n_inner.max(1) as f64).sqrt() as f32,
    )
}

/// The porte: a random velocity field on a real spawned mass distribution
/// (free surface all around), one projection, then a second one. A projection
/// must shrink the divergence it measures, and leave a projected field alone.
#[test]
#[ignore = "core-audit diagnostic, prints measurements, asserts the gate"]
fn gate_injected_divergence_must_shrink() {
    let compact = std::env::var("EMERGE_P_COMPACT").is_ok_and(|v| v == "1");
    println!(
        "toggles: {} (gate measured with the {} divergence)",
        toggles(),
        if compact { "compact" } else { "central 2h" }
    );
    let mut worst = 0.0f32;
    for (name, box_size) in [
        ("droplet 7x7", IVec2::new(7, 7)),
        ("block 20x20", IVec2::new(20, 20)),
    ] {
        let mut sim = water_sim(config(0.0, 0), box_size, Vec2::new(32.0, 32.0));
        sim.step();
        let rest = sim.grid();
        let mut grid = Grid::new(GRID);
        let mut nodes = Vec::new();
        for x in 0..GRID as i32 {
            for y in 0..GRID as i32 {
                let p = IVec2::new(x, y);
                let m = rest.mass_at(p);
                if m > 0.0 {
                    let v = Vec2::new(noise(p, 1), noise(p, 2));
                    grid.add_mass_momentum(p, m, m * v);
                    nodes.push(p);
                }
            }
        }
        grid.update_velocities(0.0, Vec2::ZERO);
        let masses: Vec<f32> = nodes.iter().map(|&p| grid.mass_at(p)).collect();
        let avg = masses.iter().sum::<f32>() / masses.len() as f32;
        let min = masses.iter().copied().fold(f32::MAX, f32::min);
        let light = masses.iter().filter(|&&m| m < 0.3 * avg).count();
        println!(
            "  {name:<12} node mass: average {avg:.4}, min {min:.5} (average/min x{:.1}), {light} nodes under 0.3 x average",
            avg / min
        );
        let d0 = rms_divergence(&grid, &nodes, compact);
        grid.project_fluid_incompressibility(1.0, 1);
        let d1 = rms_divergence(&grid, &nodes, compact);
        grid.project_fluid_incompressibility(1.0, 1);
        let d2 = rms_divergence(&grid, &nodes, compact);
        println!(
            "  {name:<12} {} nodes | RMS div all: {:.4} -> {:.4} -> {:.4} (x{:.3}, x{:.3}) | interior: {:.4} -> {:.4} -> {:.4} (x{:.3}, x{:.3})",
            nodes.len(),
            d0.0,
            d1.0,
            d2.0,
            d1.0 / d0.0,
            d2.0 / d1.0.max(1e-30),
            d0.1,
            d1.1,
            d2.1,
            d1.1 / d0.1,
            d2.1 / d1.1.max(1e-30),
        );
        worst = worst.max(d1.0 / d0.0);
    }
    assert!(
        worst < 1.0,
        "injected divergence grew (worst ratio x{worst:.3}) under {}",
        toggles()
    );
}

struct SceneReport {
    frames: u32,
    panicked: bool,
    j1: (f32, f32),
    speed1: f32,
    first_off_1pct: Option<u32>,
    first_clamp: Option<u32>,
    last_j: (f32, f32),
    first_wall_contact: Option<u32>,
}

/// True once any grid node inside the slip boundary band carries mass.
fn touches_wall(sim: &Simulation) -> bool {
    let band = sim.config().boundary_thickness as i32;
    let grid = sim.grid();
    let last = GRID as i32 - 1;
    (0..GRID as i32).any(|i| {
        (0..band).any(|b| {
            grid.mass_at(IVec2::new(i, b)) > 0.0
                || grid.mass_at(IVec2::new(i, last - b)) > 0.0
                || grid.mass_at(IVec2::new(b, i)) > 0.0
                || grid.mass_at(IVec2::new(last - b, i)) > 0.0
        })
    })
}

fn run_scene(
    gravity_fraction: f32,
    box_size: IVec2,
    center: Vec2,
    frames: u32,
    stop_y: f32,
) -> SceneReport {
    let mut rep = SceneReport {
        frames: 0,
        panicked: false,
        j1: (1.0, 1.0),
        speed1: 0.0,
        first_off_1pct: None,
        first_clamp: None,
        last_j: (1.0, 1.0),
        first_wall_contact: None,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut sim = water_sim(config(gravity_fraction, 1), box_size, center);
        for frame in 1..=frames {
            sim.step();
            let s = sim.diagnostics_snapshot();
            let j = (s.min_deformation_j, s.max_deformation_j);
            if frame == 1 {
                rep.j1 = j;
                rep.speed1 = s.max_particle_speed;
            }
            if rep.first_off_1pct.is_none() && (j.0 < 0.99 || j.1 > 1.01) {
                rep.first_off_1pct = Some(frame);
            }
            if rep.first_clamp.is_none() && (j.0 <= 0.5001 || j.1 >= 1.9999) {
                rep.first_clamp = Some(frame);
            }
            if rep.first_wall_contact.is_none() && touches_wall(&sim) {
                rep.first_wall_contact = Some(frame);
            }
            rep.last_j = j;
            rep.frames = frame;
            let lowest = sim
                .particles()
                .x
                .iter()
                .map(|x| x.y)
                .fold(f32::MAX, f32::min);
            if lowest < stop_y {
                break;
            }
        }
    }));
    rep.panicked = result.is_err();
    rep
}

fn print_scene(name: &str, g_t: f32, r: &SceneReport) {
    let fmt = |f: Option<u32>| f.map_or("never".to_string(), |v| v.to_string());
    println!(
        "  {name:<22} frames {:>3}{} | frame 1: J [{:.4}, {:.4}], speed {:.3} (g t = {:.3}) | |J-1| > 1% at {} | clamp at {} | wall contact at {} | last J [{:.4}, {:.4}]",
        r.frames,
        if r.panicked { " PANIC" } else { "" },
        r.j1.0,
        r.j1.1,
        r.speed1,
        g_t,
        fmt(r.first_off_1pct),
        fmt(r.first_clamp),
        fmt(r.first_wall_contact),
        r.last_j.0,
        r.last_j.1,
    );
}

/// The three scenes of the re-test, under whatever toggles are set.
#[test]
#[ignore = "core-audit diagnostic, prints measurements"]
fn scenes_under_current_toggles() {
    println!("toggles: {}", toggles());
    let floor = 2.0 + 7.0 * SPACING + 2.0;
    let droplet = IVec2::new(7, 7);
    let r = run_scene(1.0, droplet, Vec2::new(32.0, 32.0), 60, floor);
    print_scene("droplet, 1 g", 981.0 * DT, &r);
    let r = run_scene(0.003, droplet, Vec2::new(32.0, 32.0), 60, floor);
    print_scene("droplet, 0.003 g", 981.0 * 0.003 * DT, &r);
    let r = run_scene(
        0.003,
        IVec2::new(14, 52),
        Vec2::new(11.0, 30.0),
        120,
        f32::MIN,
    );
    print_scene("wall column, 0.003 g", 981.0 * 0.003 * DT, &r);
}
