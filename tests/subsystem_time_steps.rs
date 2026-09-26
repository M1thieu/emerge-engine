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
//! - Explicit rods, advanced inside the mechanics substep: the substep
//!   takes the minimum, as `timestep_bound` does for the materials.
//! - Grains: their contacts are sub-cycled inside the mechanics substep to
//!   their own step, rather than clamping the substep; with a real
//!   mineral's stiffness a minimum would collapse the whole scene's step.
//!   The cost is measured and reported.
//! - Heat and scalar fields, advanced once per frame on their own clock:
//!   they sub-cycle to their own step and no longer clamp the mechanics.
//! - A rod goes implicit by itself when its explicit step would fall below
//!   the rest of the scene's; no flag set by hand. An override remains, so a
//!   test can compare the two integrations.
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
//!   `q L^4 / (8 E I)`, derived in the test.
//! - A stiff rod, whose explicit step is below the scene's, goes implicit by
//!   itself: adding it leaves the scene's substeps per frame unchanged.
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

use emerge::particle::{Particle, Particles};
use emerge::thermodynamics::{
    ScalarDiffusionConfig, ScalarDiffusionField, ThermalConfig, ThermalDiffusion,
};
use emerge::{MaterialRegistry, NeoHookeanMaterial};
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
