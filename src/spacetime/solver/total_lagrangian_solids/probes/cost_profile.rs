//! Where a kept step's time goes at impact: E2's drop with the block 8 x
//! 4, 16 x 8 and 32 x 16 cells, the path's own `Timing` summed over the
//! impact frames (a contact active) and over the flight frames apart. A
//! probe: measurement, nothing held. The unknowns are twice the grid's
//! active nodes, counted from the built grid.

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::spacetime::total_lagrangian::Timing;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

fn illustrative() -> ElasticProps {
    ElasticProps::ILLUSTRATIVE
}

/// E2's drop with a `width` x `height` block, tolerance 1e-1.
fn drop((width, height): (i32, i32)) -> Simulation {
    let mut config = SimConfig::earth(48, 0.01, 1.0 / 60.0);
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
            .box_of(IVec2::new(width, height))
            .at(Vec2::new(24.0, floor + 10.0 + 0.5 * height as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&illustrative(), &config),
    );
    sim
}

/// A class of frames' sums.
#[derive(Default)]
struct Sums {
    frames: u32,
    wall: f64,
    timing: Timing,
    steps: u64,
    rejected: u64,
    newton: u64,
    gmres: u64,
}

impl Sums {
    fn add(&mut self, f: &TotalLagrangianFrame) {
        self.frames += 1;
        self.wall += f.wall;
        self.timing.add(&f.timing);
        self.steps += u64::from(f.steps);
        self.rejected += u64::from(f.rejected);
        self.newton += u64::from(f.newton);
        self.gmres += u64::from(f.gmres);
    }

    fn print(&self, label: &str) {
        if self.frames == 0 {
            println!("  {label}: no frame");
            return;
        }
        let t = &self.timing;
        let ms = |s: f64| 1e3 * s;
        let top = t.residual
            + t.linearisation
            + t.preconditioner_build
            + t.gmres
            + t.contact
            + t.multipliers
            + t.grid
            + t.finishing
            + t.controller;
        let share = |s: f64| 100.0 * s / self.wall.max(f64::MIN_POSITIVE);
        println!(
            "  {label}: {} frames, {:.1} ms wall per frame; controlled steps {}, rejected {}, Newton {}, GMRES {}, internal-force evaluations {}, factorizations {}",
            self.frames,
            ms(self.wall) / f64::from(self.frames),
            self.steps,
            self.rejected,
            self.newton,
            self.gmres,
            t.internal_evaluations,
            t.factorizations
        );
        println!(
            "    per controlled step {:.2} ms, per Newton iteration {:.1} us, per GMRES iteration {:.1} us, per internal-force evaluation {:.1} us",
            ms(self.wall) / self.steps.max(1) as f64,
            1e6 * self.wall / self.newton.max(1) as f64,
            1e6 * t.gmres / self.gmres.max(1) as f64,
            1e6 * (t.residual + t.jacobian) / t.internal_evaluations.max(1) as f64
        );
        println!(
            "    shares of the wall time: residual {:.1} %, linearisation {:.1} %, preconditioner build {:.1} %, GMRES {:.1} % (of it Jacobian products {:.1} %, push derivative {:.1} %, preconditioner solve {:.1} %, responses' products {:.1} %), responses {:.1} %, contact {:.1} %, multipliers {:.1} %, grid {:.1} %, finishing {:.1} %, controller {:.1} %; unaccounted {:.1} %",
            share(t.residual),
            share(t.linearisation),
            share(t.preconditioner_build),
            share(t.gmres),
            share(t.jacobian),
            share(t.push_derivative),
            share(t.preconditioner_solve),
            share(t.responses_jacobian),
            share(t.responses),
            share(t.contact),
            share(t.multipliers),
            share(t.grid),
            share(t.finishing),
            share(t.controller),
            share(self.wall - top)
        );
    }
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_impact_cost_profile() {
    super::cap::capped("probe_impact_cost_profile", 7200, || {
        println!(
            "threads: {} (RAYON_NUM_THREADS {:?})",
            rayon::current_num_threads(),
            std::env::var("RAYON_NUM_THREADS").ok()
        );
        for size in [(8, 4), (16, 8), (32, 16)] {
            let mut sim = drop(size);
            let (mut flight, mut impact) = (Sums::default(), Sums::default());
            let mut unknowns = 0;
            for _ in 0..12 {
                sim.step();
                let f = sim.total_lagrangian_frame().expect("the path is on");
                if unknowns == 0 {
                    let state = sim.total_lagrangian.as_ref().expect("on");
                    let grid = &state.cache.as_ref().expect("stepped").grid;
                    unknowns = 2 * grid.active.iter().filter(|&&a| a).count();
                }
                if f.contacts > 0 {
                    impact.add(&f);
                } else {
                    flight.add(&f);
                }
            }
            println!(
                "block {} x {} cells: {} particles, {unknowns} unknowns",
                size.0,
                size.1,
                sim.particles().len()
            );
            impact.print("impact frames");
            flight.print("flight frames");
        }
    });
}

/// One internal-force pass taken apart on E2's block after its first
/// frame: per particle, the two grid reads of `F`, the law's step stress
/// and the scatter, each timed alone over `ROUNDS` passes, then the whole
/// pass as the solve makes it (`parallel_scatter`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_internal_force_parts() {
    use std::hint::black_box;
    use std::time::Instant;
    const ROUNDS: usize = 200;
    let mut sim = drop((8, 4));
    sim.step();
    let state = sim.total_lagrangian.as_ref().expect("on");
    let grid = &state.cache.as_ref().expect("stepped").grid;
    let n = state.particles;
    let dt = 1.0 / 60.0;
    // A displacement field small but not zero, so the law takes its
    // general branch.
    let u: Vec<f64> = (0..grid.len)
        .map(|k| 1e-4 * ((k % 7) as f64 - 3.0))
        .collect();
    let deformation = |p: usize| {
        let (_, gu) = grid.read(&u, p, 0);
        let (_, gv) = grid.read(&u, p, 1);
        state.f[p] + DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y))
    };
    let per = |seconds: f64| 1e9 * seconds / (ROUNDS * n) as f64;
    let clock = Instant::now();
    for _ in 0..ROUNDS {
        for p in 0..n {
            black_box(deformation(p));
        }
    }
    let read = per(clock.elapsed().as_secs_f64());
    let ends: Vec<(DMat2, DMat2)> = (0..n)
        .map(|p| {
            let (f0, f1) = (state.f[p], deformation(p));
            (f0.transpose() * f0, f1.transpose() * f1)
        })
        .collect();
    let clock = Instant::now();
    for _ in 0..ROUNDS {
        for (solid, end) in state.solids[..n].iter().zip(&ends) {
            black_box(solid.step_stress(end.0, end.1, dt));
        }
    }
    let law = per(clock.elapsed().as_secs_f64());
    // The law's tangent along a direction, as a Jacobian product asks it.
    let direction = DMat2::from_cols(DVec2::new(1e-5, 2e-6), DVec2::new(2e-6, -1e-5));
    let clock = Instant::now();
    for _ in 0..ROUNDS {
        for (solid, end) in state.solids[..n].iter().zip(&ends) {
            black_box(solid.step_stress_tangent(end.0, end.1, dt, direction));
        }
    }
    let tangent = per(clock.elapsed().as_secs_f64());
    let mut acc = vec![0.0; grid.len];
    let row = DVec2::new(0.3, -0.2);
    let clock = Instant::now();
    for _ in 0..ROUNDS {
        for p in 0..n {
            for a in 0..2 {
                grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
            }
        }
    }
    black_box(&acc);
    let scatter = per(clock.elapsed().as_secs_f64());
    let clock = Instant::now();
    for _ in 0..ROUNDS {
        black_box(crate::spacetime::total_lagrangian::parallel_scatter(
            grid.len,
            n,
            |p, acc| {
                let f1 = deformation(p);
                let f0 = state.f[p];
                let s = state.solids[p].step_stress(f0.transpose() * f0, f1.transpose() * f1, dt);
                let first = state.v0[p] * (0.5 * (f0 + f1)) * s;
                for a in 0..2 {
                    let row = first.row(a);
                    grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
                }
            },
        ));
    }
    let whole = per(clock.elapsed().as_secs_f64());
    // A Jacobian product as `solve::jacobian_at` takes it with the exact
    // linearisation: the consistent mass along `d`, then the internal
    // force's product from precomputed stresses and tangents (the body of
    // `solve::internal_product`, copied).
    let unit = [
        DMat2::from_cols(DVec2::X, DVec2::ZERO),
        DMat2::from_cols(DVec2::ZERO, DVec2::Y),
        DMat2::from_cols(DVec2::Y, DVec2::X),
    ];
    let lin: Vec<(DMat2, DMat2, [DMat2; 3])> = (0..n)
        .map(|p| {
            let (c0, c1) = ends[p];
            let s = state.solids[p].step_stress(c0, c1, dt);
            let ds = unit.map(|e| state.solids[p].step_stress_tangent(c0, c1, dt, e));
            (deformation(p), s, ds)
        })
        .collect();
    let d: Vec<f64> = (0..grid.len)
        .map(|k| 1e-3 * ((k % 5) as f64 - 2.0))
        .collect();
    let clock = Instant::now();
    for _ in 0..ROUNDS {
        black_box(grid.mass_times(&d, &state.mass[..], 0.0));
    }
    let mass = per(clock.elapsed().as_secs_f64());
    let clock = Instant::now();
    for _ in 0..ROUNDS {
        black_box(crate::spacetime::total_lagrangian::parallel_scatter(
            grid.len,
            n,
            |p, acc| {
                let (f1, s, ds) = &lin[p];
                let (_, gu) = grid.read(&d, p, 0);
                let (_, gv) = grid.read(&d, p, 1);
                let df = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
                let dc = df.transpose() * *f1 + f1.transpose() * df;
                let dss = ds[0] * dc.x_axis.x + ds[1] * dc.y_axis.y + ds[2] * dc.y_axis.x;
                let first = state.v0[p] * (0.5 * df * *s + 0.5 * (state.f[p] + *f1) * dss);
                for a in 0..2 {
                    let row = first.row(a);
                    grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
                }
            },
        ));
    }
    let product = per(clock.elapsed().as_secs_f64());
    println!(
        "internal-force pass, {n} particles, {} unknowns, {} threads: per particle, F's two reads {read:.0} ns, the law's step stress {law:.0} ns, its tangent along one direction {tangent:.0} ns, the scatter {scatter:.0} ns; the whole pass {whole:.0} ns; a Jacobian product: the consistent mass {mass:.0} ns, the internal force's {product:.0} ns",
        grid.len,
        rayon::current_num_threads()
    );
}

/// Where a Newton iteration's time goes, per particle, on E2's 8 x 4 block
/// (flight frames and the impact frame apart): the frame's wall time, the
/// solves' (from the temporary counters in `hooks::PARTS`), and inside
/// them the residual evaluations, the linearisations, the Jacobian
/// products (their vector setup, mass, internal force and masking) and the
/// preconditioner's solves; the rest named by difference. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_newton_accounting() {
    use crate::spacetime::total_lagrangian::hooks;
    let mut sim = drop((8, 4));
    let n = sim.particles().len() as f64;
    let mut sums = [[0.0f64; 10]; 2];
    let mut timing = [Timing::default(), Timing::default()];
    let mut wall = [0.0f64; 2];
    hooks::take_parts();
    for _ in 0..12 {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("on");
        let parts = hooks::take_parts();
        let class = usize::from(f.contacts > 0);
        for (s, p) in sums[class].iter_mut().zip(parts) {
            *s += p;
        }
        timing[class].add(&f.timing);
        wall[class] += f.wall;
    }
    for (class, label) in [(0, "flight"), (1, "impact")] {
        let s = &sums[class];
        let t = &timing[class];
        let newton = s[9].max(1.0);
        let us = |seconds: f64| 1e6 * seconds / newton / n;
        let products = s[0];
        println!(
            "{label}: {:.0} Newton iterations in {:.0} solves; per Newton iteration: {:.2} products, {:.2} residuals, {:.2} linearisations",
            s[9],
            s[7],
            products / newton,
            s[6] / newton,
            s[5] / newton
        );
        println!(
            "  us per particle per Newton iteration: frame wall {:.2}, of it the solves {:.2}; residuals {:.2}, linearisations {:.2}, products {:.2} (setup {:.2}, mass {:.2}, internal force {:.2}, masking {:.2}), preconditioner solves {:.2}, preconditioner build {:.2}, GMRES algebra {:.2}; inside the solves unaccounted {:.2}; outside the solves (controller, contact passes, grid, finishing, the step's own) {:.2}",
            us(wall[class]),
            us(s[8]),
            us(t.residual),
            us(t.linearisation),
            us(t.jacobian),
            us(s[1]),
            us(s[2]),
            us(s[3]),
            us(s[4]),
            us(t.preconditioner_solve),
            us(t.preconditioner_build),
            us(t.gmres - t.jacobian - t.preconditioner_solve),
            us(s[8] - t.residual - t.linearisation - t.gmres - t.preconditioner_build),
            us(wall[class] - s[8])
        );
    }
}
