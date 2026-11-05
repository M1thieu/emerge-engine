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
use super::super::{ConservingSettings, Elastic, GridLayout, Particles, step_total_lagrangian};
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

/// The step's settings, on the original B-splines: the static probes' space,
/// whose free surface keeps its own nodes. The default occupation measures the
/// same here: its cell fractions read the weights per radian, so no face passes
/// and each falls through to an unknown of its own.
fn original_splines() -> ConservingSettings {
    ConservingSettings {
        occupation: None,
        ..ConservingSettings::default()
    }
}

/// One case's mirrors: the axis, and each `(normal axis, node, side)` plane.
fn with_mirrors<T>(body: &Body, planes: &[(usize, usize, i64)], run: impl FnOnce() -> T) -> T {
    ax::set_axis(Some(MARGIN));
    ax::set_planes(planes.to_vec());
    ax::set_radii(Some(body.radii.clone()));
    let out = run();
    ax::set_axis(None);
    ax::set_planes(Vec::new());
    ax::set_radii(None);
    ax::set_point_loads(Vec::new());
    out
}

/// Uniform radial compression: a disk of radius 10 cells between two slip planes 2 cells
/// apart, started on the exact uniform radial compression with its rim held: for
/// 40 steps every point's `J` and Cauchy stresses stay within 1 % of this law's
/// exact values, at `J` 0.3 and (reported) 0.16.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn a_compressed_disk_stays_on_its_exact_state() {
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
            let solids = vec![Elastic::neo_hookean(lambda, mu); x.len()];
            let dt = 1.5 * h / ((lambda + 2.0 * mu) / rho).sqrt();
            let rr = (mu * ((1.0 + e).powi(2) - 1.0) + lambda * target.ln()) / target;
            let zz = lambda * target.ln() / target;
            let size = rr.abs().max(zz.abs());
            let planes = [(1, MARGIN, 1), (1, MARGIN + nz, -1), (0, MARGIN + nr, -1)];
            let (worst_j, worst_sigma, top_speed) = with_mirrors(&body, &planes, || {
                let settings = original_splines();
                let free = |_: usize, _: usize, _: usize| None;
                let mut cache = None;
                let (mut worst_j, mut worst_sigma, mut top_speed) = (0.0f64, 0.0f64, 0.0f64);
                for k in 0..40 {
                    let report = step_total_lagrangian(
                        &body.layout,
                        Particles {
                            x: &mut x,
                            v: &mut v,
                            f: &mut f,
                            mass: &body.mass,
                            v0: &body.v0,
                        },
                        &body.reference,
                        &solids,
                        (DVec2::ZERO, None, dt),
                        (&free, &settings),
                        &mut cache,
                    );
                    assert!(report.converged, "compressed disk, step {k}: {report:?}");
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
                (worst_j, worst_sigma, top_speed)
            });
            println!(
                "compressed disk, dynamic, cells {nr}x{nz} J {target}: worst |J/J_exact - 1| {worst_j:.3e}, worst stress error {worst_sigma:.3e} of {size:.4e} Pa, largest speed {top_speed:.3e} m/s, dt {dt:.3e} s"
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
        let solids = vec![Elastic::neo_hookean(lambda, mu); x.len()];
        let dt = 1.5 * h / ((lambda + 2.0 * mu) / rho).sqrt();
        let length = nz as f64 * h;
        let top_exact = (b - 1.0) * length;
        let planes = [(1, MARGIN, 1)];
        let (worst_pzz, worst_top, top_speed) = with_mirrors(&body, &planes, || {
            ax::set_point_loads(loads);
            let settings = original_splines();
            let free = |_: usize, _: usize, _: usize| None;
            let mut cache = None;
            let (mut worst_pzz, mut worst_top, mut top_speed) = (0.0f64, 0.0f64, 0.0f64);
            for k in 0..40 {
                let report = step_total_lagrangian(
                    &body.layout,
                    Particles {
                        x: &mut x,
                        v: &mut v,
                        f: &mut f,
                        mass: &body.mass,
                        v0: &body.v0,
                    },
                    &body.reference,
                    &solids,
                    (DVec2::ZERO, None, dt),
                    (&free, &settings),
                    &mut cache,
                );
                assert!(
                    report.converged,
                    "end-loaded cylinder, step {k}: {report:?}"
                );
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
            (worst_pzz, worst_top, top_speed)
        });
        println!(
            "end-loaded cylinder, dynamic, cells {nr}x{nz}: stretches b {b:.6} a {a:.6}; worst |P_zz/T_0 - 1| {worst_pzz:.3e}, worst top error {worst_top:.3e}, largest speed {top_speed:.3e} m/s, dt {dt:.3e} s"
        );
        assert!(worst_pzz < 0.01 && worst_top < 0.01, "end-loaded cylinder");
    }
}

/// One self-weight run: the top on the axis's settlement, its mean over the
/// last period, the ramp being `ramp_periods` periods of `period` at `per_period`
/// steps each. Returns the mean, its spread over that period and the largest
/// Newton count.
fn settle(
    body: &Body,
    solids: &[Elastic],
    planes: &[(usize, usize, i64)],
    (g, period, ramp_periods, per_period): (f64, f64, usize, usize),
    reader: usize,
) -> (f64, f64, u32) {
    let mut x = body.reference.clone();
    let mut v = vec![DVec2::ZERO; x.len()];
    let mut f = vec![DMat2::IDENTITY; x.len()];
    let dt = period / per_period as f64;
    let ramp = ramp_periods * per_period;
    with_mirrors(body, planes, || {
        let settings = original_splines();
        let free = |_: usize, _: usize, _: usize| None;
        let mut cache = None;
        let mut samples = Vec::with_capacity(per_period);
        let mut newton = 0;
        for k in 0..ramp + per_period {
            let share = ((k as f64 + 0.5) / ramp as f64).min(1.0);
            let report = step_total_lagrangian(
                &body.layout,
                Particles {
                    x: &mut x,
                    v: &mut v,
                    f: &mut f,
                    mass: &body.mass,
                    v0: &body.v0,
                },
                &body.reference,
                solids,
                (DVec2::new(0.0, -g * share), None, dt),
                (&free, &settings),
                &mut cache,
            );
            assert!(report.converged, "self-weight step {k}: {report:?}");
            newton = newton.max(report.newton);
            if k >= ramp {
                samples.push(x[reader].y - body.reference[reader].y);
            }
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let low = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let high = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (mean, high - low, newton)
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
            let (mean, spread, newton) =
                settle(&body, &solids, &planes, (g, period, 10, 20), reader);
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
