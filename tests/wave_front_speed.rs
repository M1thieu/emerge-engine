//! The elastic front's speed on the total Lagrangian path, with the detection
//! separated from the speed (criteria written before the run). The scene is `wave_transit_density`'s Riemann problem: a
//! bar with free lateral faces whose left half starts at +V and right half at
//! rest. The plateau behind the front is V/2, so V/4 is the front's midpoint.
//!
//! One arrival at one probe mixes the speed with a constant offset (how the
//! initial step forms on the grid). Here each run times four probes and takes
//! the speed as the slope of the least-squares line through their arrivals,
//! the offset cancelling, and does so at three thresholds, whose spread bounds
//! the detection's own effect on the slope. At `nu = 0` the plane wave is exact
//! at every wavelength; the runs refine the grid at a constant Courant number,
//! and halve the frame alone once, to split time from space. At `nu = 0.3` the
//! long-wave `c_bar` is the limit of thin bars, so a thicker bar on the same
//! grid measures the bar's own dispersion.
//!
//! Measured: at `nu = 0` the V/4 slope is 0.98477, 0.99424 and 0.99684 of `c_bar`
//! at h = 1, 0.5 and 0.25 cm, halving the frame alone changes it by 0.01 %, and
//! the thresholds spread by 1.0 % at the finest grid. At `nu = 0.3` the thresholds
//! disagree by 6 % (2 cm thick) and 19 % (4 cm), so a dispersive bar has no single
//! front speed to compare with `c_bar`: the long-wave speed takes a mode's
//! frequency, not an arrival.

extern crate emerge_engine as emerge;

use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// The probes' distances from the middle, cm: the free end is 24 cm away, and
/// its reflection reaches the farthest probe only after 28 cm of travel.
const PROBES_CM: [f32; 4] = [5.0, 10.0, 15.0, 20.0];
/// The thresholds, as fractions of V.
const THRESHOLDS: [f64; 3] = [0.125, 0.25, 0.375];

struct Run {
    /// Cells per centimetre.
    refine: usize,
    /// The frame's division beyond the refinement.
    frame_division: usize,
    nu: f32,
    thickness_cm: usize,
}

fn bar_speed(props: &Elastic) -> f64 {
    let (e, nu, rho) = (
        f64::from(props.e_pa),
        f64::from(props.nu),
        f64::from(props.rho_kg_m3),
    );
    (e / ((1.0 - nu * nu) * rho)).sqrt()
}

/// Each threshold's slope (m/s) and intercept (s) over the probes.
fn fit(distances: &[f64], arrivals: &[f64]) -> (f64, f64) {
    let n = distances.len() as f64;
    let (mx, mt) = (
        distances.iter().sum::<f64>() / n,
        arrivals.iter().sum::<f64>() / n,
    );
    let (mut xt, mut tt) = (0.0, 0.0);
    for (x, t) in distances.iter().zip(arrivals) {
        xt += (x - mx) * (t - mt);
        tt += (t - mt) * (t - mt);
    }
    let slope = xt / tt;
    (slope, mt - mx / slope)
}

/// The arrival per threshold (rows) and probe (columns), s.
fn arrivals(run: &Run, props: &Elastic) -> Vec<Vec<f64>> {
    let dx = 0.01 / run.refine as f32;
    let frame = 2.5e-5 / (run.refine * run.frame_division) as f32;
    let cells_per_cm = run.refine as f32;
    let mut config = SimConfig::earth(64 * run.refine, dx, frame);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-8);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(props, &config)))
        .0;
    let middle = 32.0 * cells_per_cm;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(
                (48 * run.refine) as i32,
                (run.thickness_cm * run.refine) as i32,
            ))
            .at(Vec2::new(middle, middle))
            .spacing(0.5)
            .material(id)
            .mass_from(props, &config),
    );
    // 1 cm/s, in cells/s.
    let speed = 0.01 / dx;
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        if x.x < middle {
            *v = Vec2::new(speed, 0.0);
        }
    }
    let probe_speeds = |sim: &Simulation| -> Vec<f64> {
        let p = sim.particles();
        PROBES_CM
            .iter()
            .map(|cm| {
                let probe = middle + cm * cells_per_cm;
                let (mut sum, mut count) = (0.0f64, 0.0f64);
                for i in 0..p.len() {
                    if (p.x[i].x - probe).abs() < 0.5 && p.mass[i] > 0.0 {
                        sum += f64::from(p.v[i].x);
                        count += 1.0;
                    }
                }
                sum / count
            })
            .collect()
    };
    let mut found = vec![vec![f64::NAN; PROBES_CM.len()]; THRESHOLDS.len()];
    let mut previous = probe_speeds(&sim);
    let last_frame = 240 * run.refine * run.frame_division;
    for count in 1..=last_frame {
        sim.step();
        let now = probe_speeds(&sim);
        for (row, share) in found.iter_mut().zip(THRESHOLDS) {
            let threshold = share * f64::from(speed);
            for (at, (a, b)) in row.iter_mut().zip(previous.iter().zip(&now)) {
                if at.is_nan() && *b >= threshold {
                    let fraction = (threshold - a) / (b - a);
                    *at = (count as f64 - 1.0 + fraction) * f64::from(frame);
                }
            }
        }
        previous = now;
        if found.iter().flatten().all(|t| !t.is_nan()) {
            return found;
        }
    }
    panic!("a front never reached its probe in {last_frame} frames");
}

/// One run's V/4 slope over `c_bar`, printed with the other thresholds.
fn measure(run: &Run) -> f64 {
    let props = Elastic {
        e_pa: Elastic::ILLUSTRATIVE.e_pa,
        nu: run.nu,
        rho_kg_m3: 1000.0,
    };
    let c_bar = bar_speed(&props);
    let distances: Vec<f64> = PROBES_CM.iter().map(|cm| f64::from(*cm) * 0.01).collect();
    let clock = std::time::Instant::now();
    let found = arrivals(run, &props);
    let mut midpoint = f64::NAN;
    for (share, row) in THRESHOLDS.iter().zip(&found) {
        let (slope, offset) = fit(&distances, row);
        println!(
            "nu {} thickness {} cm, h {:.2} cm, frame / {}: threshold {share} V, slope {slope:.3} m/s = {:.5} c_bar ({c_bar:.3} m/s), offset {:.3e} s = {:.2} frames",
            run.nu,
            run.thickness_cm,
            1.0 / run.refine as f64,
            run.frame_division,
            slope / c_bar,
            offset,
            offset / (2.5e-5 / (run.refine * run.frame_division) as f64),
        );
        if *share == 0.25 {
            midpoint = slope / c_bar;
        }
    }
    println!("  wall {:.1} s", clock.elapsed().as_secs_f64());
    midpoint
}

/// The plane wave: the finest V/4 slope within 1 percent of `c_bar`.
#[test]
#[ignore = "measurement: run with --ignored --nocapture"]
fn the_front_of_a_plane_wave_refined() {
    let mut last = f64::NAN;
    for (refine, frame_division) in [(1, 1), (1, 2), (2, 1), (4, 1)] {
        last = measure(&Run {
            refine,
            frame_division,
            nu: 0.0,
            thickness_cm: 2,
        });
    }
    assert!(
        (last - 1.0).abs() <= 0.01,
        "the finest V/4 slope is {last:.5} of c_bar"
    );
}

/// A finite bar (`nu = 0.3`): the refinement, then a thicker bar on a same grid.
#[test]
#[ignore = "measurement: run with --ignored --nocapture"]
fn the_front_of_a_finite_bar_refined_and_thickened() {
    for refine in [1, 2, 4] {
        measure(&Run {
            refine,
            frame_division: 1,
            nu: 0.3,
            thickness_cm: 2,
        });
    }
    measure(&Run {
        refine: 2,
        frame_division: 1,
        nu: 0.3,
        thickness_cm: 4,
    });
}
