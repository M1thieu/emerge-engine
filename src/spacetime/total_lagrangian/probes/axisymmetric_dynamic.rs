//! The static probes' cases through the dynamic step (`step_total_lagrangian`,
//! the midpoint, the hoop, the mirrors of `axisymmetric`), against the same
//! exact answers, with points at the knot spans' Gauss abscissae: the quadrature
//! that integrates the weak form exactly.
//!
//! The compressed disk and the end-loaded cylinder start ON the exact uniform
//! state and must stay on it, the rim or the top's load holding it. The two
//! columns start undeformed under a gravity ramped over ten periods of their
//! first mode, then held for one period, the settlement read as its mean over
//! that period (the midpoint keeps the vibration a ramp leaves, so a mean over a
//! whole period is the static answer's measure). Each case at two resolutions.
//! Massless points carry the cylinder's dead load and read the top on the axis.
//! The density of the disk and the cylinder (1000 kg/m^3) sets only the step: it
//! has no effect on an equilibrium.

use glam::{DMat2, DVec2};

use super::super::axisymmetric as ax;
use super::super::{
    ConservingSettings, ContactSet, Elastic, Estimator, GridLayout, LagrangianCache, Particles,
    Plane, StepControl, controlled_step, start_velocity_field, step_total_lagrangian,
    tolerance_energy,
};
use super::{Layout, bisect, lame};
use crate::matter::materials::solid::neo_hookean::neo_hookean_second_pk_axisymmetric;

/// Columns and rows of grid below the axis and the base, and beyond the body.
const MARGIN: usize = 4;

/// The nearest cell width the grid holds exactly (its layout's `f32`), so that
/// the points' positions and the grid's nodes share one length.
fn cell(h: f64) -> f64 {
    f64::from(h as f32)
}

struct Body {
    layout: GridLayout,
    reference: Vec<DVec2>,
    radii: Vec<f64>,
    v0: Vec<f64>,
    mass: Vec<f64>,
    /// Points with mass; the rest are the massless load and reading points.
    massive: usize,
    h: f64,
}

impl Body {
    /// A cylinder `nr` by `nz` cells of `h` on the axis and the base, of density
    /// `rho`, with `extra` massless points `(R, Z)` after the massive ones.
    fn new((nr, nz, h, rho): (usize, usize, f64, f64), extra: &[(f64, f64)]) -> Self {
        let origin = MARGIN as f64 * h;
        let mut body = Body {
            layout: GridLayout::new(2 * MARGIN + nr + 2, 2 * MARGIN + nz + 2, h as f32),
            reference: Vec::new(),
            radii: Vec::new(),
            v0: Vec::new(),
            mass: Vec::new(),
            massive: 0,
            h,
        };
        for (r, wr) in Layout::KnotGauss.abscissae(nr as i64, h) {
            for (z, wz) in Layout::KnotGauss.abscissae(nz as i64, h) {
                body.reference.push(DVec2::new(origin + r, origin + z));
                body.radii.push(r);
                body.v0.push(wr * wz * r);
                body.mass.push(rho * wr * wz * r);
            }
        }
        body.massive = body.reference.len();
        for &(r, z) in extra {
            body.reference.push(DVec2::new(origin + r, origin + z));
            body.radii.push(r);
            body.v0.push(0.0);
            body.mass.push(0.0);
        }
        body
    }

    /// The state at the uniform stretches `(a, b)`, `F = diag(a, b, a)`.
    fn stretched(&self, a: f64, b: f64) -> (Vec<DVec2>, Vec<DMat2>) {
        let origin = MARGIN as f64 * self.h;
        let x = self
            .reference
            .iter()
            .map(|q| DVec2::new(origin + a * (q.x - origin), origin + b * (q.y - origin)))
            .collect();
        let f = vec![DMat2::from_diagonal(DVec2::new(a, b)); self.reference.len()];
        (x, f)
    }

    /// The hoop stretch of point `p`.
    fn f33(&self, x: &[DVec2], p: usize) -> f64 {
        (x[p].x - MARGIN as f64 * self.h) / self.radii[p]
    }
}

/// One case's layout, with its mirrors: the axis, and each `(normal axis, node,
/// side)` plane; the case's dead loads cleared after it.
fn with_mirrors<T>(body: &Body, planes: &[ax::Mirror], run: impl FnOnce(GridLayout) -> T) -> T {
    let out = run(body
        .layout
        .of_revolution(ax::Revolution::with_planes(MARGIN, planes)));
    ax::set_point_loads(Vec::new());
    out
}

/// How a case steps: the midpoint at the fixed `dt` the cases began with, or the
/// controller at a tight tolerance, with `dt` its longest step.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stepping {
    Fixed,
    Controlled,
}

/// The tight tolerance of the controlled runs, against the scene's own 1e-2.
const TIGHT: f64 = 1e-4;

/// One case's stepping state.
struct Stepper<'a> {
    body: &'a Body,
    layout: GridLayout,
    stepping: Stepping,
    settings: ConservingSettings,
    cache: Option<LagrangianCache>,
    /// No point is pushed: this path's piece has no contact.
    penalty: Vec<f64>,
    dt: f64,
    /// The controller's next step and its energy scale.
    tau: f64,
    scale: f64,
    /// The time covered, and the solves and refusals the controller made.
    t: f64,
    solves: u32,
    rejected: u32,
}

impl<'a> Stepper<'a> {
    /// `stored` the case's stored energy per radian at its start, its gravity
    /// the largest it reaches (`tolerance_energy`).
    fn new(
        body: &'a Body,
        layout: GridLayout,
        stepping: Stepping,
        dt: f64,
        (stored, gravity): (f64, DVec2),
    ) -> Self {
        let mass: f64 = body.mass.iter().sum();
        Self {
            body,
            layout,
            stepping,
            settings: ConservingSettings::default(),
            cache: None,
            penalty: vec![0.0; body.mass.len()],
            dt,
            tau: dt,
            scale: tolerance_energy(stored, gravity, [(mass, &body.reference[..])]).sqrt(),
            t: 0.0,
            solves: 0,
            rejected: 0,
        }
    }

    /// The length of the next step.
    fn next_step(&self) -> f64 {
        match self.stepping {
            Stepping::Fixed => self.dt,
            Stepping::Controlled => self.tau,
        }
    }

    /// One step from `(x, v, f)`; returns its largest Newton count.
    fn step(
        &mut self,
        (x, v, f): (&mut [DVec2], &mut [DVec2], &mut [DMat2]),
        solids: &mut [Elastic],
        gravity: DVec2,
    ) -> u32 {
        let particles = Particles {
            x,
            v,
            f,
            mass: &self.body.mass,
            v0: &self.body.v0,
        };
        match self.stepping {
            Stepping::Fixed => {
                let free = |_: usize, _: usize, _: usize| None;
                let report = step_total_lagrangian(
                    &self.layout,
                    particles,
                    &self.body.reference,
                    solids,
                    (gravity, None, self.dt),
                    (&free, &self.settings),
                    &mut self.cache,
                );
                assert!(report.converged, "at {}: {report:?}", self.t);
                self.t += self.dt;
                report.newton
            }
            Stepping::Controlled => {
                let control = StepControl::klapproth(TIGHT, self.dt, Estimator::Acary);
                let set = ContactSet {
                    planes: &[],
                    pairs: &[],
                    outlines: &[],
                    corners: &[],
                    friction: 0.0,
                };
                let step = controlled_step(
                    &self.layout,
                    particles,
                    (&self.body.reference, solids),
                    (gravity, &mut self.tau),
                    (set, &self.penalty, &self.settings),
                    &mut self.cache,
                    (&control, self.scale),
                );
                assert!(!step.at_floor, "at {}: at the estimator's floor", self.t);
                assert_eq!(step.stagnated, 0, "at {}: Newton stagnated", self.t);
                self.t += step.dt;
                self.solves += step.solves;
                self.rejected += step.rejected;
                step.kept.iter().map(|r| r.newton).max().unwrap_or(0)
            }
        }
    }
}

/// The stored energy per radian of `(x, f)`.
fn stored_energy(body: &Body, solids: &[Elastic], x: &[DVec2], f: &[DMat2]) -> f64 {
    (0..body.massive)
        .map(|p| body.v0[p] * ax::energy_with_stretch(&solids[p], f[p], body.f33(x, p)))
        .sum()
}

/// Uniform radial compression: a disk of radius 10 cells between two slip planes 2 cells
/// apart, started on the exact uniform radial compression with its rim held: for
/// 40 steps every point's `J` and Cauchy stresses stay within 1 % of this law's
/// exact values, at `J` 0.3 and (reported) 0.16.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn a_compressed_disk_stays_on_its_exact_state() {
    compressed_disk(Stepping::Fixed);
}

/// The same through the controller's concurrent tries.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn a_compressed_disk_stays_on_its_exact_state_through_the_controller() {
    compressed_disk(Stepping::Controlled);
}

fn compressed_disk(stepping: Stepping) {
    let (lambda, mu) = lame(2300e6, 0.33);
    let rho = 1000.0;
    for refine in [1usize, 2] {
        for target in [0.3f64, 0.16] {
            let (nr, nz) = (10 * refine, 2 * refine);
            let h = cell(0.01 / nr as f64);
            let body = Body::new((nr, nz, h, rho), &[]);
            let e = target.sqrt() - 1.0;
            let (mut x, mut f) = body.stretched(1.0 + e, 1.0);
            let mut v = vec![DVec2::ZERO; x.len()];
            let mut solids = vec![Elastic::neo_hookean(lambda, mu); x.len()];
            let dt = 1.5 * h / ((lambda + 2.0 * mu) / rho).sqrt();
            let rr = (mu * ((1.0 + e).powi(2) - 1.0) + lambda * target.ln()) / target;
            let zz = lambda * target.ln() / target;
            let size = rr.abs().max(zz.abs());
            let planes = [(1, MARGIN, 1), (1, MARGIN + nz, -1), (0, MARGIN + nr, -1)];
            let stored = stored_energy(&body, &solids, &x, &f);
            let (worst_j, worst_sigma, top_speed) = with_mirrors(&body, &planes, |layout| {
                let mut stepper = Stepper::new(&body, layout, stepping, dt, (stored, DVec2::ZERO));
                let (mut worst_j, mut worst_sigma, mut top_speed) = (0.0f64, 0.0f64, 0.0f64);
                while stepper.t < 40.0 * dt * (1.0 - 1e-9) {
                    stepper.step((&mut x, &mut v, &mut f), &mut solids, DVec2::ZERO);
                    for p in 0..body.massive {
                        let f33 = body.f33(&x, p);
                        let j = f[p].determinant() * f33;
                        worst_j = worst_j.max((j / target - 1.0).abs());
                        let c = f[p].transpose() * f[p];
                        let (s, s33) = neo_hookean_second_pk_axisymmetric(c, f33 * f33, lambda, mu);
                        let sigma = f[p] * s * f[p].transpose() * (1.0 / j);
                        let hoop = f33 * f33 * s33 / j;
                        let error = [
                            sigma.x_axis.x - rr,
                            sigma.y_axis.y - zz,
                            sigma.y_axis.x,
                            hoop - rr,
                        ]
                        .iter()
                        .fold(0.0f64, |m, e| m.max(e.abs()))
                            / size;
                        worst_sigma = worst_sigma.max(error);
                        top_speed = top_speed.max(v[p].length());
                    }
                }
                if stepping == Stepping::Controlled {
                    println!(
                        "controller: {} solves, {} rejected",
                        stepper.solves, stepper.rejected
                    );
                }
                (worst_j, worst_sigma, top_speed)
            });
            println!(
                "compressed disk, {stepping:?}, cells {nr}x{nz} J {target}: worst |J/J_exact - 1| {worst_j:.3e}, worst stress error {worst_sigma:.3e} of {size:.4e} Pa, largest speed {top_speed:.3e} m/s, dt {dt:.3e} s"
            );
            if target == 0.3 {
                assert!(
                    worst_j < 0.01 && worst_sigma < 0.01,
                    "compressed disk at J 0.3"
                );
            }
        }
    }
}

/// An end-loaded cylinder: radius 10 mm, length 40 mm, on a slip base, its top under
/// a dead nominal traction of 600 MPa, started on the exact homogeneous state:
/// for 40 steps every point's `P_zz` stays within 1 % of the traction and the
/// top on the axis within 1 % of its exact displacement.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn an_end_loaded_cylinder_stays_on_its_exact_state() {
    end_loaded_cylinder(Stepping::Fixed);
}

/// The same through the controller (its tries in sequence: the dead loads are
/// this thread's).
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn an_end_loaded_cylinder_stays_on_its_exact_state_through_the_controller() {
    end_loaded_cylinder(Stepping::Controlled);
}

fn end_loaded_cylinder(stepping: Stepping) {
    let (lambda, mu) = lame(2300e6, 0.33);
    let (t0, rho) = (600e6, 1000.0);
    let a2 = |b: f64| b * b - t0 * b / mu;
    let b = bisect((t0 / mu + 1e-9).max(1.0), 3.0, |b| {
        lambda * (a2(b) * b).ln() - mu * (1.0 - a2(b))
    });
    let a = a2(b).sqrt();
    for refine in [1usize, 2] {
        let (nr, nz) = (5 * refine, 20 * refine);
        let h = cell(0.01 / nr as f64);
        // The top's Gauss points carry the load (per radian); a point next to
        // the axis reads the top.
        let top = nz as f64 * h;
        let gauss = Layout::KnotGauss.abscissae(nr as i64, h);
        let mut extra: Vec<(f64, f64)> = gauss.iter().map(|&(r, _)| (r, top)).collect();
        extra.push((1e-3 * h, top));
        let body = Body::new((nr, nz, h, rho), &extra);
        let loads: Vec<(usize, DVec2)> = gauss
            .iter()
            .enumerate()
            .map(|(i, &(r, wr))| (body.massive + i, DVec2::new(0.0, wr * r * t0)))
            .collect();
        let reader = body.reference.len() - 1;
        let (mut x, mut f) = body.stretched(a, b);
        let mut v = vec![DVec2::ZERO; x.len()];
        let mut solids = vec![Elastic::neo_hookean(lambda, mu); x.len()];
        let dt = 1.5 * h / ((lambda + 2.0 * mu) / rho).sqrt();
        let length = nz as f64 * h;
        let top_exact = (b - 1.0) * length;
        let planes = [(1, MARGIN, 1)];
        let stored = stored_energy(&body, &solids, &x, &f);
        let (worst_pzz, worst_top, top_speed) = with_mirrors(&body, &planes, |layout| {
            ax::set_point_loads(loads);
            let mut stepper = Stepper::new(&body, layout, stepping, dt, (stored, DVec2::ZERO));
            let (mut worst_pzz, mut worst_top, mut top_speed) = (0.0f64, 0.0f64, 0.0f64);
            while stepper.t < 40.0 * dt * (1.0 - 1e-9) {
                stepper.step((&mut x, &mut v, &mut f), &mut solids, DVec2::ZERO);
                for p in 0..body.massive {
                    let f33 = body.f33(&x, p);
                    let c = f[p].transpose() * f[p];
                    let (s, _) = neo_hookean_second_pk_axisymmetric(c, f33 * f33, lambda, mu);
                    let first = f[p] * s;
                    worst_pzz = worst_pzz.max((first.y_axis.y / t0 - 1.0).abs());
                    top_speed = top_speed.max(v[p].length());
                }
                let top_now = x[reader].y - MARGIN as f64 * h - length;
                worst_top = worst_top.max((top_now / top_exact - 1.0).abs());
            }
            if stepping == Stepping::Controlled {
                println!(
                    "controller: {} solves, {} rejected",
                    stepper.solves, stepper.rejected
                );
            }
            (worst_pzz, worst_top, top_speed)
        });
        println!(
            "end-loaded cylinder, {stepping:?}, cells {nr}x{nz}: stretches b {b:.6} a {a:.6}; worst |P_zz/T_0 - 1| {worst_pzz:.3e}, worst top error {worst_top:.3e}, largest speed {top_speed:.3e} m/s, dt {dt:.3e} s"
        );
        assert!(worst_pzz < 0.01 && worst_top < 0.01, "end-loaded cylinder");
    }
}

/// One self-weight run: the top on the axis's settlement, its time mean over the
/// last period, the ramp being `ramp_periods` periods of `period`, the step
/// `period / per_period` (the controller's longest). Returns the mean, its spread
/// over that period, the largest Newton count and the steps taken.
fn settle(
    body: &Body,
    solids: &[Elastic],
    planes: &[ax::Mirror],
    (g, period, ramp_periods, per_period): (f64, f64, usize, usize),
    (reader, stepping): (usize, Stepping),
) -> (f64, f64, u32, usize) {
    let mut x = body.reference.clone();
    let mut v = vec![DVec2::ZERO; x.len()];
    let mut f = vec![DMat2::IDENTITY; x.len()];
    let mut solids = solids.to_vec();
    let dt = period / per_period as f64;
    let ramp = ramp_periods as f64 * period;
    with_mirrors(body, planes, |layout| {
        let full = DVec2::new(0.0, -g);
        let mut stepper = Stepper::new(body, layout, stepping, dt, (0.0, full));
        let (mut sum, mut covered, mut low, mut high) =
            (0.0f64, 0.0f64, f64::INFINITY, f64::NEG_INFINITY);
        let (mut newton, mut steps) = (0, 0);
        while stepper.t < (ramp + period) * (1.0 - 1e-9) {
            let share = ((stepper.t + 0.5 * stepper.next_step()) / ramp).min(1.0);
            let begun = stepper.t;
            let count = stepper.step((&mut x, &mut v, &mut f), &mut solids, full * share);
            newton = newton.max(count);
            steps += 1;
            if begun >= ramp * (1.0 - 1e-9) {
                let settlement = x[reader].y - body.reference[reader].y;
                let length = stepper.t - begun;
                sum += settlement * length;
                covered += length;
                low = low.min(settlement);
                high = high.max(settlement);
            }
        }
        (sum / covered, high - low, newton, steps)
    })
}

/// A column of radius 4 cells and height 40 cells on a slip base under its own
/// weight, its rim free (uniaxial stress, strain 1e-4, exact settlement `rho g
/// H^2 / (2 E)`, the slip base adding an end effect of about 0.15 %) or held
/// (uniaxial strain, 5 %, the exact 1D finite-strain
/// settlement): the settlement's mean within 1 %.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn a_column_settles_under_its_own_weight() {
    column(Stepping::Fixed);
}

/// The same through the controller's concurrent tries.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn a_column_settles_under_its_own_weight_through_the_controller() {
    column(Stepping::Controlled);
}

fn column(stepping: Stepping) {
    let (rho, g, height) = (1000.0, 9.81, 0.04);
    for (case, young, held) in [
        ("free-rim column", rho * g * height / 1e-4, false),
        ("held-rim column", 6000.0, true),
    ] {
        let (lambda, mu) = lame(young, 0.33);
        let exact = if held {
            let stretch = |z: f64| {
                let p = -rho * g * (height - z);
                bisect(0.2, 1.0, |f| mu * (f - 1.0 / f) + lambda * f.ln() / f - p)
            };
            let n = 2000;
            let dz = height / f64::from(n);
            let sum: f64 = (0..=n)
                .map(|k| {
                    let w = if k == 0 || k == n {
                        1.0
                    } else if k % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    w * (stretch(f64::from(k) * dz) - 1.0)
                })
                .sum();
            sum * dz / 3.0
        } else {
            -rho * g * height * height / (2.0 * young)
        };
        let speed = if held {
            ((lambda + 2.0 * mu) / rho).sqrt()
        } else {
            (young / rho).sqrt()
        };
        let period = 4.0 * height / speed;
        for refine in [1usize, 2] {
            let (nr, nz) = (4 * refine, 40 * refine);
            let h = cell(height / nz as f64);
            let body = Body::new((nr, nz, h, rho), &[(1e-3 * h, height)]);
            let solids = vec![Elastic::neo_hookean(lambda, mu); body.reference.len()];
            let mut planes = vec![(1, MARGIN, 1)];
            if held {
                planes.push((0, MARGIN + nr, -1));
            }
            let reader = body.reference.len() - 1;
            let clock = std::time::Instant::now();
            let (mean, spread, newton, steps) = settle(
                &body,
                &solids,
                &planes,
                (g, period, 10, 20),
                (reader, stepping),
            );
            println!("{case} {stepping:?}: {steps} steps");
            println!(
                "{case} dynamic cells {nr}x{nz}: E {young:.4e} Pa, settlement mean {mean:.6e} m vs exact {exact:.6e} m ({:+.3e} relative), spread over the last period {:.3e} of it, most Newton iterations {newton}, wall {:.1} s",
                mean / exact - 1.0,
                spread / exact.abs(),
                clock.elapsed().as_secs_f64()
            );
            assert!((mean / exact - 1.0).abs() < 0.01, "{case}");
        }
    }
}

/// The hoop reaches the controller's concurrent tries: a free cylinder started with
/// a radial velocity rings, through the controller at one thread and at four. The
/// two runs agree to the last bit, and each closes the energy with the hoop's
/// stretch within the bound the plane path keeps (1e-6 of its energy), which a
/// try without the hoop's force would miss by the whole of the hoop's energy.
#[test]
fn the_hoop_reaches_the_controllers_tries_at_any_thread_count() {
    let (lambda, mu) = lame(1e6, 0.33);
    let rho = 1000.0;
    let h = cell(0.01 / 4.0);
    let body = Body::new((4, 4, h, rho), &[]);
    let c = ((lambda + 2.0 * mu) / rho).sqrt();
    let dt = 1.5 * h / c;
    let run = || {
        let mut x = body.reference.clone();
        let mut v: Vec<DVec2> = body
            .radii
            .iter()
            .map(|r| DVec2::new(1e-3 * c * r / (4.0 * h), 0.0))
            .collect();
        let mut f = vec![DMat2::IDENTITY; x.len()];
        let mut solids = vec![Elastic::neo_hookean(lambda, mu); x.len()];
        let kinetic = |v: &[DVec2]| -> f64 {
            (0..v.len())
                .map(|p| 0.5 * body.mass[p] * v[p].length_squared())
                .sum()
        };
        let start = kinetic(&v) + stored_energy(&body, &solids, &x, &f);
        let layout = body.layout.of_revolution(ax::Revolution::about(MARGIN));
        let mut stepper = Stepper::new(
            &body,
            layout,
            Stepping::Controlled,
            dt,
            (start, DVec2::ZERO),
        );
        let (mut worst, mut stored_most) = (0.0f64, 0.0f64);
        while stepper.t < 20.0 * dt {
            stepper.step((&mut x, &mut v, &mut f), &mut solids, DVec2::ZERO);
            let stored = stored_energy(&body, &solids, &x, &f);
            stored_most = stored_most.max(stored);
            worst = worst.max((kinetic(&v) + stored - start).abs());
        }
        (x, worst / start, stored_most / start)
    };
    let pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool")
    };
    let (one, worst_one, most) = pool(1).install(run);
    let (four, worst_four, _) = pool(4).install(run);
    println!(
        "ringing cylinder through the controller: energy off by {worst_one:.3e} (1 thread), {worst_four:.3e} (4), stored at most {most:.3} of it"
    );
    assert!(most > 0.05, "the cylinder does not ring: {most}");
    assert!(worst_one <= 1e-6 && worst_four <= 1e-6);
    for (a, b) in one.iter().zip(&four) {
        assert_eq!(
            a.x.to_bits(),
            b.x.to_bits(),
            "x differs across thread counts"
        );
        assert_eq!(
            a.y.to_bits(),
            b.y.to_bits(),
            "y differs across thread counts"
        );
    }
}

/// A cylinder's outline in the meridian plane, counter-clockwise, one point every
/// half cell (`spawn_outline`), from the bottom on the axis, as `(R, Z)` from the
/// base; its side on the axis weighs nothing. And its corners' indices.
fn cylinder_outline(across: usize, along: usize, h: f64) -> (Vec<(f64, f64)>, Vec<usize>) {
    let (radius, length) = (across as f64 * h, along as f64 * h);
    let sides = [
        ((0.0, 0.0), (radius, 0.0), 2 * across),
        ((radius, 0.0), (radius, length), 2 * along),
        ((radius, length), (0.0, length), 2 * across),
        ((0.0, length), (0.0, 0.0), 2 * along),
    ];
    let mut extra = Vec::new();
    let mut corners = Vec::new();
    for &((r0, z0), (r1, z1), count) in &sides {
        corners.push(extra.len());
        extra.extend((0..count).map(|i| {
            let share = i as f64 / count as f64;
            (r0 + (r1 - r0) * share, z0 + (z1 - z0) * share)
        }));
    }
    (extra, corners)
}

/// What a rod's impact on a wall gave.
struct RodImpact {
    /// Contact time over `2 L / c0`, minus one.
    contact_error: f64,
    /// Touch time against the free flight's, in `L / c0`.
    touch_error: f64,
    /// Axial momentum after the contact over the start's, opposed (exact: 1).
    rebound: f64,
    /// The largest energy imbalance over the start's energy, its bound the
    /// steps' own residual and contact work over the same.
    ledger: f64,
    ledger_bound: f64,
    /// The largest axial momentum imbalance against the walls' impulses, over
    /// the start's momentum.
    momentum: f64,
    /// The radial momentum's largest size over the axial start's: no wall
    /// pushes radially, and the axis's mirror holds it at zero by symmetry
    /// only in the sum over a whole ring, which here is its own measure.
    steps: usize,
    solves: u32,
    rejected: u32,
    lowest_gap: f64,
}

/// A rod of `cells` by `cells_across` falling on a frictionless floor at the strain
/// `v0 / c0`, through the controller at tolerance `tolerance`, forecast on.
fn rod_on_a_wall(
    (cells_across, cells, nu): (usize, usize, f64),
    (strain, tolerance): (f64, f64),
) -> RodImpact {
    let (young, rho) = (1e6, 1000.0);
    let (lambda, mu) = lame(young, nu);
    let h = cell(0.001);
    let length = cells as f64 * h;
    let (extra, corners) = cylinder_outline(cells_across, cells, h);
    let body = Body::new((cells_across, cells, h, rho), &extra);
    let layout = body.layout.of_revolution(ax::Revolution::about(MARGIN));
    let outlines = vec![(body.massive..body.reference.len()).collect::<Vec<usize>>()];
    let corners = vec![corners];
    let c0 = (young / rho).sqrt();
    let v0 = strain * c0;
    let settings = ConservingSettings {
        forecast_activation: true,
        ..ConservingSettings::default()
    };
    // A touch 2.37 of the first step's ahead: not on a step's boundary.
    let dt = 1.5 * h / ((lambda + 2.0 * mu) / rho).sqrt();
    let touch = 2.37 * dt;
    let bottom = MARGIN as f64 * h;
    let planes = [Plane {
        axis: 1,
        at: bottom - v0 * touch,
        inward: 1.0,
        speed: 0.0,
    }];
    let massive = body.massive;
    let mut x = body.reference.clone();
    let start: Vec<DVec2> = (0..x.len())
        .map(|p| {
            if p < massive {
                DVec2::new(0.0, -v0)
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let field = start_velocity_field(
        &layout,
        &body.reference,
        (&body.mass, &body.v0, &start),
        &settings,
    );
    let mut v = start;
    v[massive..].copy_from_slice(&field[massive..]);
    let mut f = vec![DMat2::IDENTITY; x.len()];
    let mut solids = vec![Elastic::neo_hookean(lambda, mu); x.len()];
    // The multipliers' penalty, as the path sets it on an outline point.
    let penalty: Vec<f64> = (0..x.len())
        .map(|p| {
            if p < massive {
                0.0
            } else {
                solids[p].bulk + solids[p].mu
            }
        })
        .collect();
    let momentum0: f64 = (0..massive).map(|p| body.mass[p] * v[p].y).sum();
    let energy = |x: &[DVec2], v: &[DVec2], f: &[DMat2], solids: &[Elastic]| -> f64 {
        let kinetic: f64 = (0..massive)
            .map(|p| 0.5 * body.mass[p] * v[p].length_squared())
            .sum();
        kinetic + stored_energy(&body, solids, x, f)
    };
    let energy0 = energy(&x, &v, &f, &solids);
    let total: f64 = body.mass.iter().sum();
    let scale = tolerance_energy(energy0, DVec2::ZERO, [(total, &body.reference[..])]).sqrt();
    let control = StepControl::klapproth(tolerance, dt, Estimator::Acary);
    let exact = 2.0 * length / c0;
    let end = touch + exact + 8.0 * dt;
    let (mut t, mut tau) = (0.0, dt);
    let mut cache: Option<LagrangianCache> = None;
    let (mut first, mut last): (Option<f64>, f64) = (None, 0.0);
    let (mut impulse, mut allowed, mut ledger, mut worst_momentum) = (0.0, 0.0, 0.0f64, 0.0f64);
    let (mut steps, mut solves, mut rejected, mut lowest) = (0, 0, 0, f64::INFINITY);
    while t < end {
        let step = controlled_step(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &body.mass,
                v0: &body.v0,
            },
            (&body.reference, &mut solids),
            (DVec2::ZERO, &mut tau),
            (
                ContactSet {
                    planes: &planes,
                    pairs: &[],
                    outlines: &outlines,
                    corners: &corners,
                    friction: 0.0,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
            (&control, scale),
        );
        assert!(!step.at_floor, "rod, at {t}: at the estimator's floor");
        assert_eq!(step.stagnated, 0, "rod, at {t}: Newton stagnated");
        let piece = step.dt / step.kept.len() as f64;
        for (k, report) in step.kept.iter().enumerate() {
            let begun = t + k as f64 * piece;
            if report.contacts > 0 {
                first.get_or_insert(begun);
                last = begun + piece;
            }
            impulse += report.push * piece;
            allowed += report.residual_work + report.contact_work_bound;
        }
        t += step.dt;
        steps += 1;
        solves += step.solves;
        rejected += step.rejected;
        let round_off = x.len() as f64 * f64::EPSILON * energy0;
        ledger = ledger.max((energy(&x, &v, &f, &solids) - energy0).abs() - round_off);
        let momentum: f64 = (0..massive).map(|p| body.mass[p] * v[p].y).sum();
        worst_momentum = worst_momentum.max((momentum - momentum0 - impulse).abs());
        for point in &x[massive..] {
            lowest = lowest.min(point.y - planes[0].at);
        }
    }
    let momentum: f64 = (0..massive).map(|p| body.mass[p] * v[p].y).sum();
    let first = first.expect("the rod reached the wall");
    RodImpact {
        contact_error: (last - first) / exact - 1.0,
        touch_error: (first - touch) * c0 / length,
        rebound: -momentum / momentum0,
        ledger: ledger / energy0,
        ledger_bound: allowed / energy0,
        momentum: worst_momentum / momentum0.abs(),
        steps,
        solves,
        rejected,
        lowest_gap: lowest,
    }
}

/// A neo-Hookean rod (radius 2 cells, length 32, Poisson's ratio 0.33) falls on a
/// rigid frictionless wall at the strain 1e-3: on the wall for `2 L / c0`, `c0^2 = E /
/// rho` (Doyen, Ern and Piperno 2011, sec. 3.1, for the 1D bar), its energy and its
/// axial momentum kept. Its take-off is reported, not asserted: at `a / L` = 1/16 it
/// keeps 0.957 of its momentum, 0.957 again at twice the cells, 0.973 at 1/32 and 0.984
/// at 1/64 (measured, `a_rods_take_off_against_its_mesh_and_tolerance`), so a thin rod
/// meets the 1D law within 2 % and a thick one keeps vibration of its own, a reading
/// with no source held.
#[test]
fn a_rod_stays_on_a_wall_for_the_time_a_wave_takes_to_cross_it_twice() {
    let got = rod_on_a_wall((2, 32, 0.33), (1e-3, 1e-3));
    println!(
        "rod on a wall, 2 x 32 cells: contact time off by {:+.3e}, touch {:+.3e} L/c0, take-off momentum {:.5} of the start's, energy off by {:.3e} (bound {:.3e}), axial momentum against the impulses {:.3e}, lowest gap {:.3e} m, {} steps, {} solves, {} rejected",
        got.contact_error,
        got.touch_error,
        got.rebound,
        got.ledger,
        got.ledger_bound,
        got.momentum,
        got.lowest_gap,
        got.steps,
        got.solves,
        got.rejected
    );
    assert!(got.contact_error.abs() <= 0.02, "contact time");
    assert!(got.ledger <= got.ledger_bound.max(1e-6), "energy");
    assert!(got.momentum <= 1e-6, "axial momentum");
}

/// The same rod at Poisson's ratio 0, the exact 1D bar: its free sides carry no lateral
/// stress, so it also leaves at `v0`, within 2 % at these cells (0.8 %, then 0.5 % and
/// 0.4 % at 64 and 128 cells).
#[test]
fn a_rod_without_lateral_strain_leaves_a_wall_at_its_speed() {
    let got = rod_on_a_wall((2, 32, 0.0), (1e-3, 1e-3));
    println!(
        "1D rod on a wall, 2 x 32 cells: contact time off by {:+.3e}, take-off momentum {:.5}, energy off by {:.3e} (bound {:.3e}), axial momentum {:.3e}",
        got.contact_error, got.rebound, got.ledger, got.ledger_bound, got.momentum
    );
    assert!(got.contact_error.abs() <= 0.02, "contact time");
    assert!((got.rebound - 1.0).abs() <= 0.02, "take-off");
    assert!(got.ledger <= got.ledger_bound.max(1e-6), "energy");
    assert!(got.momentum <= 1e-6, "axial momentum");
}

/// The rod's take-off against the mesh, the tolerance and Poisson's ratio: whether
/// what is short is the mesh (it goes with the cells at `nu = 0`, the exact 1D bar)
/// or the rod's own vibration (it stays with the ratio `a / L` at `nu > 0`).
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn a_rods_take_off_against_its_mesh_and_tolerance() {
    for (across, cells, nu, tolerance) in [
        (2, 32, 0.33, 1e-2),
        (2, 32, 0.33, 1e-3),
        (2, 32, 0.0, 1e-3),
        (2, 64, 0.0, 1e-3),
        (2, 64, 0.33, 1e-3),
        (2, 128, 0.33, 1e-3),
        (4, 64, 0.33, 1e-3),
        (2, 128, 0.0, 1e-3),
        (4, 128, 0.33, 1e-3),
    ] {
        if let Ok(only) = std::env::var("EMERGE_PROBE_ROD_ONLY") {
            if only != format!("{across},{cells},{nu}") {
                continue;
            }
            println!("knob active: only the rod {only}");
        }
        let clock = std::time::Instant::now();
        let got = rod_on_a_wall((across, cells, nu), (1e-3, tolerance));
        println!(
            "rod {across} x {cells} nu {nu} tol {tolerance:e}: contact time {:+.3e}, take-off momentum {:.5}, energy {:.3e} (bound {:.3e}), momentum {:.3e}, lowest gap {:.3e} m, {} steps, {} solves, {} rejected, wall {:.0} s",
            got.contact_error,
            got.rebound,
            got.ledger,
            got.ledger_bound,
            got.momentum,
            got.lowest_gap,
            got.steps,
            got.solves,
            got.rejected,
            clock.elapsed().as_secs_f64()
        );
    }
}

/// What a pressed cylinder gave.
struct Pressed {
    /// The largest `|sigma / law - 1|` on the true strains 0.15 to 0.40.
    worst: f64,
    /// `sigma / eps` below the strain 0.02, least squares, over `E`.
    slope: f64,
    /// The balance over the press's work, its bound the same.
    ledger: f64,
    ledger_bound: f64,
    /// The plastic heat over the press's work.
    heat: f64,
    /// The largest top-to-bottom spread of the points' `sigma_zz`, over the mean.
    spread: f64,
    /// Kept tries whose Newton stagnated.
    stagnated: u32,
    steps: usize,
    solves: u32,
    rejected: u32,
}

/// Roma Plastilina's record in a free cylinder between two slip planes, its top driven
/// at the constant true strain rate `rate`, to the strain 0.40, through the controller.
fn pressed_cylinder(
    (across, along): (usize, usize),
    (rate, nu, tolerance): (f64, f64, f64),
) -> Pressed {
    let record =
        crate::matter::materials::solid::viscoplastic_records::ViscoplasticRecord::roma_plastilina_hernandez_2023();
    let (young, k) = (record.e_pa, record.law.k);
    let n = record.law.m;
    let m = record.law.hardening.expect("the record hardens").exponent;
    // Its density, from the same papers.
    let rho = 1529.0;
    let (lambda, mu) = lame(young, nu);
    let h = cell(0.002);
    let (extra, corners) = cylinder_outline(across, along, h);
    let body = Body::new((across, along, h, rho), &extra);
    let massive = body.massive;
    let layout = body
        .layout
        .of_revolution(ax::Revolution::with_planes(MARGIN, &[(1, MARGIN, 1)]));
    let outlines = vec![(massive..body.reference.len()).collect::<Vec<usize>>()];
    let corners = vec![corners];
    let settings = ConservingSettings {
        forecast_activation: true,
        ..ConservingSettings::default()
    };
    let base = MARGIN as f64 * h;
    let length0 = along as f64 * h;
    let solid = Elastic::hencky_j2(lambda, mu, 0.0, 0.0, 0.0).with_norton(record.law);
    let mut solids = vec![solid; body.reference.len()];
    // The homogeneous flow at this rate, so no wave starts it.
    let start: Vec<DVec2> = (0..body.reference.len())
        .map(|p| {
            if p < massive {
                let q = body.reference[p];
                DVec2::new(0.5 * rate * body.radii[p], -rate * (q.y - base))
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let field = start_velocity_field(
        &layout,
        &body.reference,
        (&body.mass, &body.v0, &start),
        &settings,
    );
    let mut v = start;
    v[massive..].copy_from_slice(&field[massive..]);
    let mut x = body.reference.clone();
    let mut f = vec![DMat2::IDENTITY; x.len()];
    let penalty: Vec<f64> = (0..x.len())
        .map(|p| {
            if p < massive {
                0.0
            } else {
                solids[p].bulk + solids[p].mu
            }
        })
        .collect();
    let energy = |x: &[DVec2], v: &[DVec2], f: &[DMat2], solids: &[Elastic]| -> f64 {
        let kinetic: f64 = (0..massive)
            .map(|p| 0.5 * body.mass[p] * v[p].length_squared())
            .sum();
        kinetic + stored_energy(&body, solids, x, f)
    };
    let energy0 = energy(&x, &v, &f, &solids);
    let strain_end = 0.4f64;
    let law_at = |strain: f64| k * strain.powf(m) * rate.powf(n);
    // The scale: the elastic energy at the end's stress, as the start holds none.
    let volume: f64 = body.v0[..massive].iter().sum();
    let scale = (0.5 * law_at(strain_end).powi(2) / young * volume).sqrt();
    let end_time = strain_end / rate;
    let control = StepControl::klapproth(tolerance, end_time / 400.0, Estimator::Acary);
    let mut top = Plane {
        axis: 1,
        at: base + length0,
        inward: -1.0,
        speed: 0.0,
    };
    let mut tau = control.largest;
    let mut cache: Option<LagrangianCache> = None;
    let (mut worst, mut spread, mut sum_se, mut sum_ee) = (0.0f64, 0.0f64, 0.0, 0.0);
    let (mut work, mut heat, mut allowed, mut ledger) = (0.0, 0.0, 0.0, 0.0f64);
    let (mut steps, mut solves, mut rejected, mut stagnated) = (0, 0, 0, 0);
    let mut strain = 0.0;
    while strain < strain_end {
        top.speed = -rate * (top.at - base);
        let step = controlled_step(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &body.mass,
                v0: &body.v0,
            },
            (&body.reference, &mut solids),
            (DVec2::ZERO, &mut tau),
            (
                ContactSet {
                    planes: &[top],
                    pairs: &[],
                    outlines: &outlines,
                    corners: &corners,
                    friction: 0.0,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
            (&control, scale),
        );
        assert!(!step.at_floor, "press, at the strain {strain}: the floor");
        stagnated += step.stagnated;
        top = top.after(step.dt);
        for report in &step.kept {
            work += report.work.0;
            heat += report.plastic;
            allowed += report.residual_work + report.contact_work_bound;
        }
        steps += 1;
        solves += step.solves;
        rejected += step.rejected;
        strain = (length0 / (top.at - base)).ln();
        // The true axial stress, compression positive, over the points.
        let (mut sum, mut low, mut high) = (0.0, f64::INFINITY, f64::NEG_INFINITY);
        for p in 0..massive {
            let f33 = body.f33(&x, p);
            let c = f[p].transpose() * f[p];
            let super::super::ElasticLaw::HenckyJ2(law, state) = &solids[p].law else {
                unreachable!("the record's law");
            };
            let (s, _) = law.second_pk_with_stretch(c, f33 * f33, state);
            let j = f[p].determinant() * f33;
            let sigma = -(f[p] * s * f[p].transpose()).y_axis.y / j;
            sum += body.v0[p] * sigma;
            low = low.min(sigma);
            high = high.max(sigma);
        }
        let sigma = sum / volume;
        if strain >= 0.15 {
            worst = worst.max((sigma / law_at(strain.min(strain_end)) - 1.0).abs());
            spread = spread.max((high - low) / sigma);
        }
        if strain <= 0.02 {
            sum_se += sigma * strain;
            sum_ee += strain * strain;
        }
        let balance = energy(&x, &v, &f, &solids) - energy0 + heat - work;
        ledger = ledger.max(balance.abs());
    }
    Pressed {
        worst,
        slope: sum_se / sum_ee / young,
        ledger: ledger / work,
        ledger_bound: allowed / work,
        heat: heat / work,
        spread,
        stagnated,
        steps,
        solves,
        rejected,
    }
}

/// The record's uniaxial compression through the axisymmetric dynamic step: a free
/// cylinder between slip planes, its top driven at 32.8 /s, 100 times the record's
/// fastest measured rate and nearer a drop's (about 157 /s), so this verifies the path
/// carries the law and compares with no measurement, follows the law's formula within the `k` uncertainty (6.4 %) on the true strains 0.15
/// to 0.40, its elastic slope within 1 % of `E` (the criteria of the law-level test,
/// `rp1_uniaxial_compression_follows_its_measured_power_law`), the ledger closed with
/// the plastic work as heat. On finer meshes the worst stays near the law-level test's own
/// 1.6 % (1.5 % at 4 x 8, 1.7 % at 8 x 16, `roma_plastilina_pressed_on_a_finer_mesh`) and
/// the stress's spread across points halves with the mesh.
#[test]
fn roma_plastilina_pressed_in_a_cylinder_follows_its_measured_law() {
    let got = pressed_cylinder((2, 4), (32.8, 0.49, 1e-2));
    println!(
        "pressed cylinder, 2 x 4 cells, 32.8 /s: sigma / law off by at most {:.3e}, slope / E {:.5}, balance {:.3e} of the press's work (bound {:.3e}), heat {:.4} of it, sigma spread {:.3e}, {} stagnated, {} steps, {} solves, {} rejected",
        got.worst,
        got.slope,
        got.ledger,
        got.ledger_bound,
        got.heat,
        got.spread,
        got.stagnated,
        got.steps,
        got.solves,
        got.rejected
    );
    assert!(got.worst <= 0.064, "against the law");
    assert!((got.slope - 1.0).abs() <= 0.01, "slope");
    assert!(got.ledger <= got.ledger_bound.max(1e-6), "ledger");
    assert!(got.heat > 0.0, "no heat");
    assert_eq!(got.stagnated, 0, "stagnated");
}

/// The same on a finer mesh, a tighter tolerance and the other Poisson's ratio. What
/// sets the step here is the held layer's velocity alternation, about its acceleration
/// times the step (measured at 1e-3: 2.4e-7 s, with the kinetic part of the error at
/// 2.7e-3 and the elastic at 1.6e-8).
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn roma_plastilina_pressed_on_a_finer_mesh() {
    for (cells, nu, tolerance) in [
        ((2, 4), 0.49, 1e-2),
        ((4, 8), 0.49, 1e-2),
        ((4, 8), 0.434, 1e-2),
        ((8, 16), 0.49, 1e-2),
        ((4, 8), 0.49, 1e-3),
    ] {
        let clock = std::time::Instant::now();
        let got = pressed_cylinder(cells, (32.8, nu, tolerance));
        println!(
            "pressed cylinder {cells:?} nu {nu} tol {tolerance:e}: off by {:.3e}, slope / E {:.5}, balance {:.3e} (bound {:.3e}), heat {:.4}, spread {:.3e}, {} stagnated, {} steps, {} solves, {} rejected, wall {:.0} s",
            got.worst,
            got.slope,
            got.ledger,
            got.ledger_bound,
            got.heat,
            got.spread,
            got.stagnated,
            got.steps,
            got.solves,
            got.rejected,
            clock.elapsed().as_secs_f64()
        );
    }
}
