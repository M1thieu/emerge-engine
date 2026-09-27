//! Time steps of the subsystems beside the mechanics: grains, rods, heat and
//! scalar fields. Criteria written before the code and frozen: they do not
//! move once results have been seen.
//!
//! # What reading the code found
//!
//! - Heat and scalar fields are applied once per `step()`, the whole frame's
//!   time in one explicit step. The heat bound is folded into the mechanics
//!   substep, where it slows the mechanics and protects nothing. The scalar
//!   field has no bound, and decays by `1 - lambda dt`, negative past
//!   `lambda dt = 1`, while its doc says `exp(-lambda dt)`.
//! - Grain contacts are integrated at the mechanics substep; their critical
//!   step exists but only tests call it.
//! - A stiff explicit rod slows the whole scene; the implicit rod is wired
//!   but chosen by a flag set by hand.
//!
//! # The junction
//!
//! - Every subsystem declares the largest step its own scheme stays stable
//!   at, from its own parameters, through one interface shared by grains,
//!   rods, heat, scalar fields, granular fluidity and the Cosserat field.
//!   Each limit is derived from its scheme, next to the code.
//! - The step a subsystem takes is `material_cfl_coefficient` times its
//!   limit: the definition the materials already use, the fraction of the
//!   stability limit, and no new free number.
//! - Explicit rods sub-cycle their own forces inside the mechanics substep
//!   to their own step and stay coupled to the grid; a free rod no longer
//!   bounds the substep. A rod touching other matter still does, a declared
//!   approximation: the grid exchanges momentum once per substep, and a
//!   loaded cantilever sub-cycled under a large substep let the particles on
//!   it fall through (measured, `rod_deflects_and_mpm_particles_feel_
//!   reaction`). Rewritten before gate 3 passed, from "the substep takes the
//!   minimum", once sub-cycling was chosen (next point).
//! - Grains: their contacts are sub-cycled inside the mechanics substep to
//!   their own step, rather than clamping the substep; with a real
//!   mineral's stiffness a minimum would collapse the whole scene's step.
//!   The cost is measured and reported.
//! - Heat and scalar fields, advanced once per frame on their own clock:
//!   they sub-cycle to their own step and no longer clamp the mechanics.
//! - Rewritten before gate 3 passed: "a rod goes implicit by itself when its
//!   explicit step would fall below the rest of the scene's" is dropped. The
//!   implicit rod is not coupled to the grid (its own doc), so switching a
//!   stiff rod to it would have silently cut it off from the scene;
//!   sub-cycling keeps the coupling. The implicit flag stays, set by hand.
//! - A scene with none of these subsystems gives bit-identical results.
//!
//! # Gate 1: the diffusion of a peak matches the analytic
//!
//! Through the real path of `ScalarDiffusionField` (particles to grid,
//! finite difference, grid to particles) with its transfer blend at the
//! default, 1.0, and through `ThermalDiffusion`. A body of particles at rest,
//! no gravity, a Gaussian peak of standard deviation 3 cells.
//!
//! - First, with no physical diffusion, the variance change the transfers
//!   alone make over the same time is measured and reported. The criteria
//!   below count only if it is under 1 percent of `2 D T`, so they cannot
//!   pass or fail because of the transfers.
//! - With a frame at half the explicit limit, and at twenty times it: the
//!   variance along each axis grows as `sigma0^2 + 2 D t` within 5
//!   percent; without decay the total is conserved within 0.1 percent; with
//!   decay at `lambda * frame = 3`, the total follows `exp(-lambda t)`
//!   within 0.5 percent (for heat, the excess over ambient under Newton
//!   cooling).
//! - Added before any result, because the two above cannot see an unstable
//!   step: the discrete Laplacian's second moment is exactly `2 D dt` times
//!   the total whatever `dt`, so an update far past its limit still gets the
//!   variance and the total right while its profile turns to noise. Also at
//!   both frames: the largest value follows the analytic peak
//!   `phi0 sigma0^2 / sigma(t)^2` within 5 percent, and no value falls
//!   below minus 1 percent of the initial peak.
//!
//! # Gate 2: two stiff grains on soft sand invent no energy
//!
//! A bed of soft sand at real gravity settles alone for 1 s. Two grains are
//! placed on it, one on the other, at rest, and the scene runs 3 s. Their
//! contact stiffness comes from the existing physical preset with the Young's
//! modulus of quartz, read from two sources, never softened to pass.
//!
//! - The energy counted is the whole system, sand and grains, kinetic,
//!   rotational and gravitational, because energy passes between them. The
//!   same settled bed also runs alone over the same 3 s; the rise of the
//!   energy with the grains, above the bed alone's, is at most 1 percent of
//!   the grains' own `m g R`.
//! - Reported: grain contact sub-steps per mechanics substep, and the wall
//!   time with and without the grains.
//!
//! # Gate 3: a rod bends the same explicit and implicit
//!
//! A cantilever under its own weight, clamped at one end, its tip deflection
//! under 5 percent of its length.
//!
//! - Settled tip deflection, explicit and implicit, within 1 percent of each
//!   other, and within 5 percent of the Euler-Bernoulli value
//!   `q L^4 / (8 E I)`, derived in the test. Settled means the mean over the
//!   last 10 s of 20: the rod rings around its equilibrium, and a single
//!   instant read 3 and 7 percent where the means agree within 0.2
//!   (clarified after the first measurement, on review).
//! - A stiff explicit rod touching nothing, sub-cycled, leaves the scene's
//!   largest substeps per frame unchanged.
//! - The switch between the two regimes (touching, free) makes no jump: a
//!   particle body falling past the tip of a cantilever, touching it and
//!   leaving it, moves the tip along the same path, within 5 percent of the
//!   tip's largest excursion, as a reference run whose every step is within
//!   the rod's own stable step.
//!
//! # Gate 4: nothing else moves
//!
//! The existing suite is unchanged, except tests that assert the old fold of
//! the heat bound into the mechanics substep, rewritten with the reason.
//!
//! # Order
//!
//! Diffusion, then grains, then rods, each measured before its change and
//! after.

extern crate emerge_engine as emerge;

use emerge::particle::RodPoints;
use emerge::particle::{Particle, Particles};
use emerge::rod::forces::discrete_curvature;
use emerge::rod::{
    Rod, RodForceParams, RodMaterial, advance_rod, build_straight_rod, rod_cfl_dt, step_rod,
};
use emerge::thermodynamics::{
    ScalarDiffusionConfig, ScalarDiffusionField, ThermalConfig, ThermalDiffusion,
};
use emerge::{MaterialRegistry, NeoHookeanMaterial, SimConfig, Simulation};
use glam::Vec2;

// ── Gate 1: diffusion ─────────────────────────────────────────────────────

const GRID: usize = 96;
/// Half-width of the body, in cells: the peak never reaches its edge (at the
/// end sigma is 7 cells, the edge 5.7 sigma away).
const HALF_WIDTH: f32 = 40.0;
const SPACING: f32 = 0.5;
const SIGMA0: f32 = 3.0;
/// Cells squared per second; the explicit limit is `1 / (4 D)` = 0.25 s.
const D: f32 = 1.0;
const DURATION: f32 = 20.0;

fn body() -> Particles {
    let centre = GRID as f32 * 0.5;
    let mut particles = Vec::new();
    let n = (2.0 * HALF_WIDTH / SPACING) as usize;
    for i in 0..n {
        for j in 0..n {
            let x = Vec2::new(
                centre - HALF_WIDTH + (i as f32 + 0.5) * SPACING,
                centre - HALF_WIDTH + (j as f32 + 0.5) * SPACING,
            );
            let r2 = (x - Vec2::splat(centre)).length_squared();
            particles.push(Particle {
                x,
                mass: SPACING * SPACING,
                initial_volume: SPACING * SPACING,
                volume: SPACING * SPACING,
                density: 1.0,
                temperature: (-r2 / (2.0 * SIGMA0 * SIGMA0)).exp(),
                ..Particle::zeroed()
            });
        }
    }
    Particles::from(particles)
}

/// Amount-weighted moments of the scalar carried in `temperature`: total,
/// variance along x and y, largest and smallest value.
struct Moments {
    total: f64,
    variance: (f64, f64),
    max: f32,
    min: f32,
}

fn moments(particles: &Particles) -> Moments {
    let (mut total, mut mx, mut my) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..particles.len() {
        let a = (particles.mass[i] * particles.temperature[i]) as f64;
        total += a;
        mx += a * particles.x[i].x as f64;
        my += a * particles.x[i].y as f64;
    }
    let (cx, cy) = (mx / total, my / total);
    let (mut vx, mut vy) = (0.0f64, 0.0f64);
    for i in 0..particles.len() {
        let a = (particles.mass[i] * particles.temperature[i]) as f64;
        vx += a * (particles.x[i].x as f64 - cx).powi(2);
        vy += a * (particles.x[i].y as f64 - cy).powi(2);
    }
    let max = particles
        .temperature
        .iter()
        .copied()
        .fold(f32::MIN, f32::max);
    let min = particles
        .temperature
        .iter()
        .copied()
        .fold(f32::MAX, f32::min);
    Moments {
        total,
        variance: (vx / total, vy / total),
        max,
        min,
    }
}

/// One application of a diffusing operator to the particles, for a time.
type Stepper = Box<dyn FnMut(&mut Particles, f32)>;

/// Which of the two diffusing operators a run goes through.
#[derive(Clone, Copy)]
enum Operator {
    Scalar,
    Heat,
}

impl Operator {
    /// One application, the way `Simulation` applies it once per frame:
    /// diffusivity `d`, decay (or Newton cooling) `lambda`, time `dt`.
    fn stepper(self, d: f32, lambda: f32) -> Stepper {
        match self {
            Operator::Scalar => {
                let mut field = ScalarDiffusionField::new(
                    ScalarDiffusionConfig {
                        diffusivity: d,
                        decay_rate: lambda,
                        ambient: 0.0,
                    },
                    |p| p.temperature,
                    |p, delta| p.temperature += delta,
                    GRID,
                );
                field.blend = 1.0;
                let registry =
                    MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(1.0, 1.0)));
                Box::new(move |particles, dt| field.apply(particles, dt, &registry))
            }
            Operator::Heat => {
                // alpha = k / (rho c dx^2) = d with rho = c = dx = 1.
                let mut thermal = ThermalDiffusion::new(
                    ThermalConfig {
                        conductivity: d,
                        heat_capacity: 1.0,
                        density: 1.0,
                        ambient: 0.0,
                        grid_cell_size: 1.0,
                        cooling_rate: lambda,
                        ..Default::default()
                    },
                    GRID,
                );
                Box::new(move |particles, dt| thermal.apply(particles, dt))
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Operator::Scalar => "scalar field",
            Operator::Heat => "heat",
        }
    }
}

/// Runs one diffusion case and returns every missed criterion.
fn diffusion_case(op: Operator, frame: f32, failures: &mut Vec<String>) {
    let label = format!("{} at frame {frame} s", op.name());
    let start = moments(&body());
    let peak0 = start.max as f64;
    let sigma0_2 = start.variance.0;

    // Physical diffusion, no decay.
    let mut particles = body();
    let mut step = op.stepper(D, 0.0);
    let frames = (DURATION / frame).round() as usize;
    let (mut worst_var, mut worst_peak, mut worst_total, mut lowest) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for f in 1..=frames {
        step(&mut particles, frame);
        let t = (f as f32 * frame) as f64;
        let m = moments(&particles);
        let expected = sigma0_2 + 2.0 * D as f64 * t;
        for v in [m.variance.0, m.variance.1] {
            worst_var = worst_var.max((v / expected - 1.0).abs());
        }
        let peak = peak0 * sigma0_2 / expected;
        worst_peak = worst_peak.max((m.max as f64 / peak - 1.0).abs());
        worst_total = worst_total.max((m.total / start.total - 1.0).abs());
        lowest = lowest.min(m.min as f64 / peak0);
    }
    println!(
        "{label}: variance off {:.4}, peak off {:.4}, total off {:.2e}, lowest {:.4} of the peak",
        worst_var, worst_peak, worst_total, lowest
    );
    let mut check = |ok: bool, what: String| {
        if !ok {
            println!("  FAIL {what}");
            failures.push(what);
        }
    };
    check(
        worst_var <= 0.05,
        format!("{label}: variance off by {worst_var:.4}"),
    );
    check(
        worst_total <= 1e-3,
        format!("{label}: total off by {worst_total:.2e}"),
    );
    check(
        worst_peak <= 0.05,
        format!("{label}: peak off by {worst_peak:.4}"),
    );
    check(
        lowest >= -0.01,
        format!("{label}: a value fell to {lowest:.4} of the peak"),
    );

    // Decay at lambda * frame = 3, over four frames.
    let lambda = 3.0 / frame;
    let mut particles = body();
    let mut step = op.stepper(D, lambda);
    let mut worst_decay = 0.0f64;
    for f in 1..=4 {
        step(&mut particles, frame);
        let t = (f as f32 * frame) as f64;
        let expected = (-(lambda as f64) * t).exp();
        let ratio = moments(&particles).total / start.total;
        worst_decay = worst_decay.max((ratio / expected - 1.0).abs());
    }
    println!("{label}: decay at lambda frame = 3 off {worst_decay:.4}");
    check(
        worst_decay <= 5e-3,
        format!("{label}: decay off by {worst_decay:.4}"),
    );
}

/// Gate 1: diffusion of a peak against the analytic, through both
/// operators, at half the explicit limit and at twenty times it.
#[test]
fn gate1_the_diffusion_of_a_peak_matches_the_analytic() {
    let mut failures = Vec::new();
    for op in [Operator::Scalar, Operator::Heat] {
        // The transfers alone, with no physical diffusion.
        let start = moments(&body());
        let mut particles = body();
        let mut step = op.stepper(0.0, 0.0);
        for _ in 0..(DURATION / 0.125).round() as usize {
            step(&mut particles, 0.125);
        }
        let m = moments(&particles);
        let transfer_change = (m.variance.0 - start.variance.0)
            .abs()
            .max((m.variance.1 - start.variance.1).abs());
        let physical = 2.0 * D as f64 * DURATION as f64;
        println!(
            "{}: transfers alone change the variance by {transfer_change:.3e} cells^2, {:.2e} of 2 D T",
            op.name(),
            transfer_change / physical
        );
        assert!(
            transfer_change <= 0.01 * physical,
            "{}: the transfers alone move the variance by more than 1 percent of 2 D T, \
             so the diffusion criteria would not measure diffusion",
            op.name()
        );
        for frame in [0.125, 5.0] {
            diffusion_case(op, frame, &mut failures);
        }
    }
    assert!(
        failures.is_empty(),
        "{} criteria missed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ── Gate 3: rods ──────────────────────────────────────────────────────────

/// Cantilever from x = 12 to 52 cells at 1 cm, clamped by pinning its first
/// two points; the beam runs from point 1 to the tip.
/// 81 points: the discrete rod's equilibrium converges to the analytic as
/// 10.8, 5.2 and 2.5 percent at 21, 41 and 81 points
/// (`probe_cantilever_time_average`).
const ROD_POINTS: usize = 81;
const ROD_START_CELLS: f32 = 12.0;
const ROD_LENGTH_CELLS: f32 = 40.0;
const CELL_M: f32 = 0.01;
const LINEAR_DENSITY: f32 = 0.1; // kg/m
const ROD_EA: f32 = 1000.0; // N
const ROD_EI: f32 = 0.5; // N m^2

/// Runs the cantilever 20 s under real gravity through `Simulation`,
/// explicit or implicit, and returns the tip deflection in metres averaged
/// over the last 10 s (the rod rings around its equilibrium) and the largest
/// number of substeps a frame took.
fn settled_tip_deflection(implicit: bool) -> (f32, usize) {
    let config = SimConfig::earth(64, CELL_M, 1.0 / 60.0);
    let y = 40.0;
    let mut points = build_straight_rod(
        Vec2::new(ROD_START_CELLS, y),
        Vec2::new(ROD_START_CELLS + ROD_LENGTH_CELLS, y),
        ROD_POINTS,
        LINEAR_DENSITY,
        CELL_M,
    );
    points.pinned[0] = 1;
    points.pinned[1] = 1;
    let l0 = ROD_LENGTH_CELLS * CELL_M / (ROD_POINTS - 1) as f32;
    let (axial_damping, bending_damping) =
        RodMaterial::critical_damping(l0, LINEAR_DENSITY * l0, ROD_EA, ROD_EI);
    let mut rod = Rod::new(
        points,
        RodMaterial::new(ROD_EA, ROD_EI, axial_damping, bending_damping),
    );
    rod.use_implicit_integration = implicit;
    let mut sim = Simulation::empty(config).with_rod(rod);
    let tip = ROD_POINTS - 1;
    let (mut sum, mut count, mut substeps) = (0.0f32, 0u32, 0usize);
    for frame in 0..1200 {
        sim.step();
        substeps = substeps.max(sim.last_substeps());
        if frame >= 600 {
            sum += (y - sim.rods()[0].points.x[tip].y) * CELL_M;
            count += 1;
        }
    }
    (sum / count as f32, substeps)
}

/// Gate 3, first part: the same cantilever settles to the same deflection
/// explicit and implicit, and to Euler-Bernoulli's `q L^4 / (8 E I)`.
///
/// Derived: `E I w'''' = q`, with `w(0) = w'(0) = 0` at the clamp and
/// `w''(L) = w'''(L) = 0` at the free end, gives
/// `w(x) = q x^2 (6 L^2 - 4 L x + x^2) / (24 E I)`, so
/// `w(L) = q L^4 / (8 E I)`. `L` runs from the clamp (point 1) to the tip.
#[test]
#[ignore = "gate 3: long (81-point rods, 20 s each); run with --ignored --nocapture"]
fn gate3_a_rod_bends_the_same_explicit_and_implicit() {
    let q = LINEAR_DENSITY * 9.81;
    let l0 = ROD_LENGTH_CELLS * CELL_M / (ROD_POINTS - 1) as f32;
    let beam = ROD_LENGTH_CELLS * CELL_M - l0;
    let analytic = q * beam.powi(4) / (8.0 * ROD_EI);
    assert!(
        analytic < 0.05 * beam,
        "setup must stay in small deflection: {analytic} m over {beam} m"
    );
    let (explicit, explicit_substeps) = settled_tip_deflection(false);
    let (implicit, implicit_substeps) = settled_tip_deflection(true);
    println!(
        "cantilever: analytic {:.3} mm, explicit {:.3} mm ({explicit_substeps} substeps/frame), \
         implicit {:.3} mm ({implicit_substeps} substeps/frame)",
        analytic * 1000.0,
        explicit * 1000.0,
        implicit * 1000.0
    );
    let mut failures = Vec::new();
    let apart = (explicit - implicit).abs() / explicit.abs().max(implicit.abs());
    if apart > 0.01 {
        failures.push(format!(
            "explicit and implicit {:.2} percent apart",
            apart * 100.0
        ));
    }
    for (name, value) in [("explicit", explicit), ("implicit", implicit)] {
        let off = (value / analytic - 1.0).abs();
        if off > 0.05 {
            failures.push(format!(
                "{name} {:.2} percent off the analytic",
                off * 100.0
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// What one frame of the explicit and the implicit cantilever costs, and
/// where the tip is after a few frames. A probe.
#[test]
#[ignore = "diagnostic probe for gate 3: run with --ignored --nocapture"]
fn probe_cantilever_cost() {
    for implicit in [false, true] {
        let config = SimConfig::earth(64, CELL_M, 1.0 / 60.0);
        let y = 40.0;
        let mut points = build_straight_rod(
            Vec2::new(ROD_START_CELLS, y),
            Vec2::new(ROD_START_CELLS + ROD_LENGTH_CELLS, y),
            ROD_POINTS,
            LINEAR_DENSITY,
            CELL_M,
        );
        points.pinned[0] = 1;
        points.pinned[1] = 1;
        let l0 = ROD_LENGTH_CELLS * CELL_M / (ROD_POINTS - 1) as f32;
        let (axial_damping, bending_damping) =
            RodMaterial::critical_damping(l0, LINEAR_DENSITY * l0, ROD_EA, ROD_EI);
        let mut rod = Rod::new(
            points,
            RodMaterial::new(ROD_EA, ROD_EI, axial_damping, bending_damping),
        );
        rod.use_implicit_integration = implicit;
        let mut sim = Simulation::empty(config).with_rod(rod);
        let start = std::time::Instant::now();
        for frame in 0..2 {
            sim.step();
            let rod = &sim.rods()[0];
            let fastest = rod.points.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            let lowest = rod.points.x.iter().fold(f32::INFINITY, |m, x| m.min(x.y));
            println!(
                "implicit={implicit} frame {frame}: {} substeps, dt {:.3e} s, tip y {:.5},                  lowest point y {lowest:.3}, fastest point {fastest:.3e} cells/s, sleeping {},                  {:.1} ms so far",
                sim.last_substeps(),
                sim.effective_dt(),
                rod.points.x[ROD_POINTS - 1].y,
                rod.sleeping,
                start.elapsed().as_secs_f32() * 1000.0
            );
        }
    }
}

/// The explicit cantilever with the grid coupling cut: the rod's own
/// integrator (`step_rod`) at the step `Simulation` gives it, gravity
/// applied directly. If this explodes too, the step is not stable for the
/// rod itself and the coupling is not the cause. A probe.
#[test]
#[ignore = "diagnostic probe for gate 3: run with --ignored --nocapture"]
fn probe_cantilever_coupling_cut() {
    let config = SimConfig::earth(64, CELL_M, 1.0 / 60.0);
    let y = 40.0;
    let mut points = build_straight_rod(
        Vec2::new(ROD_START_CELLS, y),
        Vec2::new(ROD_START_CELLS + ROD_LENGTH_CELLS, y),
        ROD_POINTS,
        LINEAR_DENSITY,
        CELL_M,
    );
    points.pinned[0] = 1;
    points.pinned[1] = 1;
    let l0 = ROD_LENGTH_CELLS * CELL_M / (ROD_POINTS - 1) as f32;
    let (axial_damping, bending_damping) =
        RodMaterial::critical_damping(l0, LINEAR_DENSITY * l0, ROD_EA, ROD_EI);
    let material = RodMaterial::new(ROD_EA, ROD_EI, axial_damping, bending_damping);
    let rod = Rod::new(points, material);
    let mut points = rod.points.clone();
    for coefficient in [config.material_cfl_coefficient, 1.0, 1.25] {
        points.x.clone_from(&rod.points.x);
        points.v.iter_mut().for_each(|v| *v = Vec2::ZERO);
        let dt = rod_cfl_dt(&points, &material, coefficient);
        let mut first_blow = None;
        let mut fastest = 0.0f32;
        for step in 0..20_000 {
            step_rod(
                &mut points,
                &material,
                config.gravity,
                Vec2::ZERO,
                0.0,
                CELL_M,
                dt,
            );
            fastest = points.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            if first_blow.is_none() && (fastest >= 1.0e3 || !fastest.is_finite()) {
                first_blow = Some(step);
            }
        }
        println!(
            "coupling cut, rod coefficient {coefficient}: dt {dt:.3e} s, first step above 1e3 \
             cells/s {first_blow:?}, fastest after 20000 steps {fastest:.3e}, tip y {:.5}",
            points.x[ROD_POINTS - 1].y
        );
    }
}

/// The tip over time, explicit and implicit through `Simulation` and the
/// rod on its own (`step_rod`, the discrete model's own equilibrium), at
/// several point counts: whether the runs settle, and where the discrete
/// model settles against the analytic. A probe.
#[test]
#[ignore = "diagnostic probe for gate 3: run with --ignored --nocapture"]
fn probe_cantilever_settling_and_resolution() {
    let analytic = |beam: f32| LINEAR_DENSITY * 9.81 * beam.powi(4) / (8.0 * ROD_EI);
    let length = ROD_LENGTH_CELLS * CELL_M;
    for points_count in [21usize, 41, 81] {
        let config = SimConfig::earth(64, CELL_M, 1.0 / 60.0);
        let y = 40.0;
        let build = || {
            let mut points = build_straight_rod(
                Vec2::new(ROD_START_CELLS, y),
                Vec2::new(ROD_START_CELLS + ROD_LENGTH_CELLS, y),
                points_count,
                LINEAR_DENSITY,
                CELL_M,
            );
            points.pinned[0] = 1;
            points.pinned[1] = 1;
            let l0 = length / (points_count - 1) as f32;
            let (axial_damping, bending_damping) =
                RodMaterial::critical_damping(l0, LINEAR_DENSITY * l0, ROD_EA, ROD_EI);
            Rod::new(
                points,
                RodMaterial::new(ROD_EA, ROD_EI, axial_damping, bending_damping),
            )
        };
        let l0 = length / (points_count - 1) as f32;
        let tip = points_count - 1;
        // The rod on its own, at its stable step, 8 s.
        let rod = build();
        let mut alone = rod.points.clone();
        let dt = rod_cfl_dt(&alone, &rod.material, 0.5);
        for _ in 0..(8.0 / dt) as usize {
            step_rod(
                &mut alone,
                &rod.material,
                config.gravity,
                Vec2::ZERO,
                0.0,
                CELL_M,
                dt,
            );
        }
        let alone_mm = (y - alone.x[tip].y) * CELL_M * 1000.0;
        let mut series = Vec::new();
        for implicit in [false, true] {
            let mut rod = build();
            rod.use_implicit_integration = implicit;
            let mut sim = Simulation::empty(config).with_rod(rod);
            let mut samples = Vec::new();
            for frame in 1..=1200 {
                sim.step();
                if frame % 240 == 0 {
                    samples.push((y - sim.rods()[0].points.x[tip].y) * CELL_M * 1000.0);
                }
            }
            series.push(samples);
        }
        println!(
            "{points_count} points: analytic {:.3} mm clamped at point 1, {:.3} mm at point 0; \
             rod alone {alone_mm:.3} mm; explicit every 4 s {:?}; implicit every 4 s {:?}",
            analytic(length - l0) * 1000.0,
            analytic(length) * 1000.0,
            series[0],
            series[1]
        );
    }
}

/// Ringing or not: the tip every frame for 20 s, explicit and implicit
/// through `Simulation`, and the rod on its own; mean, lowest and highest
/// deflection over the last 10 s. A probe.
#[test]
#[ignore = "diagnostic probe for gate 3: run with --ignored --nocapture"]
fn probe_cantilever_time_average() {
    let length = ROD_LENGTH_CELLS * CELL_M;
    let analytic = |beam: f32| LINEAR_DENSITY * 9.81 * beam.powi(4) / (8.0 * ROD_EI);
    let summary = |samples: &[f32]| {
        let late = &samples[samples.len() / 2..];
        let mean = late.iter().sum::<f32>() / late.len() as f32;
        let lo = late.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = late.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        format!("mean {mean:.3} mm, range {lo:.3} to {hi:.3}")
    };
    for points_count in [21usize, 41, 81] {
        let config = SimConfig::earth(64, CELL_M, 1.0 / 60.0);
        let y = 40.0;
        let l0 = length / (points_count - 1) as f32;
        let build = || {
            let mut points = build_straight_rod(
                Vec2::new(ROD_START_CELLS, y),
                Vec2::new(ROD_START_CELLS + ROD_LENGTH_CELLS, y),
                points_count,
                LINEAR_DENSITY,
                CELL_M,
            );
            points.pinned[0] = 1;
            points.pinned[1] = 1;
            let (axial_damping, bending_damping) =
                RodMaterial::critical_damping(l0, LINEAR_DENSITY * l0, ROD_EA, ROD_EI);
            Rod::new(
                points,
                RodMaterial::new(ROD_EA, ROD_EI, axial_damping, bending_damping),
            )
        };
        let tip = points_count - 1;
        let deflection_mm = |x: Vec2| (y - x.y) * CELL_M * 1000.0;
        // The rod on its own, sampled every 1/60 s for 20 s.
        let rod = build();
        let mut alone = rod.points.clone();
        let dt = rod_cfl_dt(&alone, &rod.material, 0.5);
        let per_frame = ((1.0 / 60.0) / dt).ceil() as usize;
        let dt = (1.0 / 60.0) / per_frame as f32;
        let mut alone_samples = Vec::new();
        for _ in 0..1200 {
            for _ in 0..per_frame {
                step_rod(
                    &mut alone,
                    &rod.material,
                    config.gravity,
                    Vec2::ZERO,
                    0.0,
                    CELL_M,
                    dt,
                );
            }
            alone_samples.push(deflection_mm(alone.x[tip]));
        }
        let mut lines = vec![format!("rod alone {}", summary(&alone_samples))];
        for implicit in [false, true] {
            let mut rod = build();
            rod.use_implicit_integration = implicit;
            let mut sim = Simulation::empty(config).with_rod(rod);
            let mut samples = Vec::new();
            for _ in 0..1200 {
                sim.step();
                samples.push(deflection_mm(sim.rods()[0].points.x[tip]));
            }
            let name = if implicit { "implicit" } else { "explicit" };
            lines.push(format!("{name} {}", summary(&samples)));
        }
        println!(
            "{points_count} points (analytic {:.3} mm at point 1, {:.3} at point 0): {}",
            analytic(length - l0) * 1000.0,
            analytic(length) * 1000.0,
            lines.join("; ")
        );
    }
}

/// A rod's energy in joules, from the terms `forces::compute_internal_forces`
/// derives its forces from: each edge `EA / (2 l0) (l - l0)^2`, each interior
/// vertex `EI / (2 L_v) (kappa - kappa_rest)^2`, plus `m v^2 / 2` and the
/// potential of `gravity` (cells/s^2). Returns (kinetic, elastic, total).
fn rod_energy(
    points: &RodPoints,
    material: &RodMaterial,
    gravity: Vec2,
    dx: f32,
) -> (f64, f64, f64) {
    let n = points.len();
    let ea = |k: usize| {
        if points.ea.is_empty() {
            material.ea
        } else {
            points.ea[k]
        }
    };
    let ei = |k: usize| {
        if points.ei.is_empty() {
            material.ei
        } else {
            points.ei[k]
        }
    };
    let (mut kinetic, mut elastic, mut potential) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let m = points.mass[i] as f64;
        kinetic += 0.5 * m * ((points.v[i] * dx).length_squared() as f64);
        potential -= m * ((gravity * dx).dot(points.x[i] * dx) as f64);
    }
    for k in 0..n - 1 {
        let l = ((points.x[k + 1] - points.x[k]) * dx).length() as f64;
        let l0 = points.rest_edge_length[k] as f64;
        elastic += 0.5 * ea(k) as f64 / l0 * (l - l0).powi(2);
    }
    for i in 1..n - 1 {
        let kappa =
            discrete_curvature(points.x[i - 1] * dx, points.x[i] * dx, points.x[i + 1] * dx) as f64;
        let voronoi = 0.5 * (points.rest_edge_length[i - 1] + points.rest_edge_length[i]) as f64;
        let bend = kappa - points.rest_curvature[i - 1] as f64;
        elastic += 0.5 * ei(i - 1) as f64 / voronoi * bend * bend;
    }
    (kinetic, elastic, kinetic + elastic + potential)
}

/// Where the coupled explicit rod's ringing gets its energy. Each frame,
/// the rod's state before `Simulation::step` is copied and advanced alone
/// over the same frame (`step_rod`, the same sub-step count and gravity),
/// so the difference of the two end energies is what the grid did beyond
/// gravity. Summed over the last 10 s of 20, beside the ringing's size.
/// Frames the solver split into several substeps are left out and counted.
/// Run while `step_rod` still advanced its positions without compensation,
/// it read a steady gain on the rod's own side: that was the replay freezing
/// (`probe_cantilever_reference_absorption`), not energy. A probe.
#[test]
#[ignore = "diagnostic probe for gate 3: run with --ignored --nocapture"]
fn probe_cantilever_energy_balance() {
    for (points_count, frame_dt) in [(21usize, 1.0 / 60.0), (81, 1.0 / 60.0), (81, 1.0 / 240.0)] {
        let started = std::time::Instant::now();
        let config = SimConfig::earth(64, CELL_M, frame_dt);
        let y = 40.0;
        let l0 = ROD_LENGTH_CELLS * CELL_M / (points_count - 1) as f32;
        let mut points = build_straight_rod(
            Vec2::new(ROD_START_CELLS, y),
            Vec2::new(ROD_START_CELLS + ROD_LENGTH_CELLS, y),
            points_count,
            LINEAR_DENSITY,
            CELL_M,
        );
        points.pinned[0] = 1;
        points.pinned[1] = 1;
        let (axial_damping, bending_damping) =
            RodMaterial::critical_damping(l0, LINEAR_DENSITY * l0, ROD_EA, ROD_EI);
        let material = RodMaterial::new(ROD_EA, ROD_EI, axial_damping, bending_damping);
        let mut sim = Simulation::empty(config).with_rod(Rod::new(points, material));
        let frames = (20.0 / frame_dt).round() as usize;
        let tip = points_count - 1;
        let (mut grid_work, mut grid_work_abs, mut own_change) = (0.0f64, 0.0f64, 0.0f64);
        let (mut positive, mut counted, mut split) = (0usize, 0usize, 0usize);
        let (mut kinetic_sum, mut lo, mut hi) = (0.0f64, f32::INFINITY, f32::NEG_INFINITY);
        let (mut tip_sum, mut tip_samples) = (0.0f64, Vec::new());
        for frame in 0..frames {
            let before = sim.rods()[0].points.clone();
            sim.step();
            if frame < frames / 2 {
                continue;
            }
            let after = &sim.rods()[0].points;
            let deflection = (y - after.x[tip].y) * CELL_M * 1000.0;
            lo = lo.min(deflection);
            hi = hi.max(deflection);
            tip_sum += deflection as f64;
            tip_samples.push(deflection);
            let (kinetic, _, coupled_end) = rod_energy(after, &material, config.gravity, CELL_M);
            kinetic_sum += kinetic;
            if sim.last_substeps() != 1 {
                split += 1;
                continue;
            }
            let (_, _, start) = rod_energy(&before, &material, config.gravity, CELL_M);
            let mut alone = before;
            let stable = rod_cfl_dt(&alone, &material, config.material_cfl_coefficient);
            let sub_steps = (frame_dt / stable).ceil().max(1.0) as usize;
            let h = frame_dt / sub_steps as f32;
            for _ in 0..sub_steps {
                step_rod(
                    &mut alone,
                    &material,
                    config.gravity,
                    Vec2::ZERO,
                    0.0,
                    CELL_M,
                    h,
                );
            }
            let (_, _, alone_end) = rod_energy(&alone, &material, config.gravity, CELL_M);
            let work = coupled_end - alone_end;
            grid_work += work;
            grid_work_abs += work.abs();
            own_change += alone_end - start;
            positive += usize::from(work > 0.0);
            counted += 1;
        }
        let late = (frames - frames / 2) as f64;
        let mean_tip = tip_sum / late;
        let crossings = tip_samples
            .windows(2)
            .filter(|w| ((w[0] as f64 - mean_tip) * (w[1] as f64 - mean_tip)) < 0.0)
            .count();
        println!(
            "{points_count} points, frame {:.4} s: tip mean {mean_tip:.3} mm, range {lo:.3} to {hi:.3}, mean kinetic {:.3e} J, mean-crossing rate {:.2} Hz; over {counted} one-substep frames ({split} split): grid work beyond gravity {grid_work:.3e} J (sum of |.| {grid_work_abs:.3e}, positive in {positive}), the rod's own change alone {own_change:.3e} J; {:.0} s",
            frame_dt,
            kinetic_sum / late,
            crossings as f64 / (2.0 * 10.0),
            started.elapsed().as_secs_f32()
        );
    }
}

/// Whether the "rod alone" reference settles because of its damping or
/// because f32 froze it. Three runs of the same cantilever for 20 s: through
/// `Simulation` (whose `advance_rod` sums positions with compensation),
/// alone through `advance_rod` with gravity as the only external
/// acceleration, and alone through `step_rod`. Before `step_rod` summed its
/// positions with compensation (`integrator::advance_position`), its run
/// froze every point within 5 s while the first mode still rang at 5 mm;
/// now it matches the compensated run. For each,
/// the tip's half-range over 5 to 10 s and over 15 to 20 s, beside the
/// decay the declared damping predicts for the first mode: `critical_damping`
/// sets `c_b` per vertex, a stiffness-proportional `beta = c_b l0 / EI`, so
/// `zeta_1 = beta omega_1 / 2` with `omega_1 = 1.8751^2 sqrt(EI / (mu L^4))`.
/// The coupled run also repeats the energy balance against the compensated
/// replay. A probe.
#[test]
#[ignore = "diagnostic probe for gate 3: run with --ignored --nocapture"]
fn probe_cantilever_reference_absorption() {
    for points_count in [21usize, 81] {
        let started = std::time::Instant::now();
        let frame_dt = 1.0 / 60.0;
        let config = SimConfig::earth(64, CELL_M, frame_dt);
        let y = 40.0;
        let l0 = ROD_LENGTH_CELLS * CELL_M / (points_count - 1) as f32;
        let (axial_damping, bending_damping) =
            RodMaterial::critical_damping(l0, LINEAR_DENSITY * l0, ROD_EA, ROD_EI);
        let material = RodMaterial::new(ROD_EA, ROD_EI, axial_damping, bending_damping);
        let build = || {
            let mut points = build_straight_rod(
                Vec2::new(ROD_START_CELLS, y),
                Vec2::new(ROD_START_CELLS + ROD_LENGTH_CELLS, y),
                points_count,
                LINEAR_DENSITY,
                CELL_M,
            );
            points.pinned[0] = 1;
            points.pinned[1] = 1;
            points
        };
        let beam = ROD_LENGTH_CELLS * CELL_M - l0;
        let omega_1 = 1.8751f32.powi(2) * (ROD_EI / (LINEAR_DENSITY * beam.powi(4))).sqrt();
        let zeta_1 = bending_damping * l0 / ROD_EI * omega_1 / 2.0;
        let predicted = (-zeta_1 * omega_1 * 10.0).exp();
        let frames = (20.0 / frame_dt).round() as usize;
        let tip = points_count - 1;
        let windows = |samples: &[f32]| {
            let half_range = |w: &[f32]| {
                let lo = w.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                0.5 * (hi - lo)
            };
            let q = samples.len() / 4;
            let (early, late) = (
                half_range(&samples[q..2 * q]),
                half_range(&samples[3 * q..]),
            );
            format!(
                "half-range {early:.4} then {late:.4} mm (ratio {:.3})",
                late / early.max(f32::MIN_POSITIVE)
            )
        };
        let deflection = |p: &RodPoints| (y - p.x[tip].y) * CELL_M * 1000.0;
        let gravity_only = vec![config.gravity; points_count];
        let params = || RodForceParams {
            wind_velocity: Vec2::ZERO,
            wind_drag_coeff: 0.0,
            push_center: None,
            push_strength: 0.0,
            push_radius: 0.0,
            dx_meters: CELL_M,
            dt: frame_dt,
            stability_fraction: config.material_cfl_coefficient,
        };

        let mut sim = Simulation::empty(config).with_rod(Rod::new(build(), material));
        let (mut coupled, mut grid_work, mut own_change, mut positive) =
            (Vec::new(), 0.0f64, 0.0f64, 0usize);
        for frame in 0..frames {
            let before = sim.rods()[0].points.clone();
            sim.step();
            let after = &sim.rods()[0].points;
            coupled.push(deflection(after));
            if frame >= frames / 2 && sim.last_substeps() == 1 {
                let (_, _, start) = rod_energy(&before, &material, config.gravity, CELL_M);
                let mut replay = before;
                advance_rod(&mut replay, &material, params(), &gravity_only);
                let (_, _, replay_end) = rod_energy(&replay, &material, config.gravity, CELL_M);
                let (_, _, coupled_end) = rod_energy(after, &material, config.gravity, CELL_M);
                grid_work += coupled_end - replay_end;
                own_change += replay_end - start;
                positive += usize::from(coupled_end > replay_end);
            }
        }

        let mut compensated = build();
        let mut compensated_tip = Vec::new();
        for _ in 0..frames {
            advance_rod(&mut compensated, &material, params(), &gravity_only);
            compensated_tip.push(deflection(&compensated));
        }

        let mut plain = build();
        let stable = rod_cfl_dt(&plain, &material, config.material_cfl_coefficient);
        let sub_steps = (frame_dt / stable).ceil().max(1.0) as usize;
        let h = frame_dt / sub_steps as f32;
        let mut plain_tip = Vec::new();
        for _ in 0..frames {
            for _ in 0..sub_steps {
                step_rod(
                    &mut plain,
                    &material,
                    config.gravity,
                    Vec2::ZERO,
                    0.0,
                    CELL_M,
                    h,
                );
            }
            plain_tip.push(deflection(&plain));
        }
        let frozen = emerge::diagnostics::position_resolution(
            plain.x.iter().copied(),
            plain.v.iter().copied(),
            h,
        );

        println!(
            "{points_count} points: declared zeta_1 {zeta_1:.2e}, omega_1 {omega_1:.1} rad/s, predicted ratio over 10 s {predicted:.3}\n  coupled: {}; energy over the last 10 s, grid beyond gravity {grid_work:.3e} J (positive in {positive} frames), the compensated replay's own change {own_change:.3e} J\n  alone, compensated: {}\n  alone, step_rod: {} (at the end {} moving, {:.3} of them with every increment under half an ulp, which only compensation keeps; sub-step {h:.2e} s)\n  {:.0} s",
            windows(&coupled),
            windows(&compensated_tip),
            windows(&plain_tip),
            frozen.moving,
            frozen.frozen,
            started.elapsed().as_secs_f32()
        );
    }
}
