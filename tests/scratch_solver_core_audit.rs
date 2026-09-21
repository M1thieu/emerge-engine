//! Scratch solver-core audit (2026-09-21): profiling, benchmarks and stress
//! tests for the explicit substep loop. Every test is `#[ignore]`d and PRINTS
//! measured numbers; none of them asserts a performance claim. Nothing in
//! `src/` is modified -- the one "fix" measured here (exact lattice volume)
//! is applied to a scene's particles from the outside, after spawn.
//!
//!   cargo test --profile quick --test scratch_solver_core_audit -- \
//!       --ignored --nocapture --test-threads=1 <test_name>
//!
//! `RAYON_NUM_THREADS=1` in front of that measures the single-thread cost.
extern crate emerge_engine as emerge;

use emerge::{
    CorotatedMaterial, DruckerPragerMaterial, Elastic, Fluid, FromSI, GranularProps, MaterialModel,
    MuIRheologyMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, Viscoelastic,
    ViscoelasticMaterial,
};
use glam::{IVec2, Mat2, Vec2};

// ── shared measurement plumbing ─────────────────────────────────────────────

/// Accumulated cost of `frames` calls to `Simulation::step()`.
#[derive(Default, Clone, Copy)]
struct Cost {
    frames: usize,
    substeps: usize,
    wall_us: f64,
    p2g_us: u64,
    grid_us: u64,
    g2p_us: u64,
    cfl_us: u64,
    project_us: u64,
    other_us: u64,
    total_us: u64,
}

impl Cost {
    fn add_frame(&mut self, sim: &Simulation, wall_us: f64) {
        let s = sim.diagnostics_snapshot();
        let t = s.timing;
        self.frames += 1;
        self.substeps += s.substeps_last_step;
        self.wall_us += wall_us;
        self.p2g_us += t.p2g_us;
        self.grid_us += t.grid_update_us;
        self.g2p_us += t.g2p_us;
        self.cfl_us += t.cfl_us;
        self.project_us += t.project_us;
        let named = t.p2g_us + t.grid_update_us + t.g2p_us + t.cfl_us + t.project_us;
        self.other_us += t.total_us.saturating_sub(named);
        self.total_us += t.total_us;
    }

    fn print(&self, label: &str, particles: usize, frame_dt_s: f32) {
        let frames = self.frames.max(1) as f64;
        let substeps = self.substeps.max(1) as f64;
        let wall_ms_frame = self.wall_us / 1000.0 / frames;
        let sim_ms_frame = frame_dt_s as f64 * 1000.0;
        let us_per_substep = self.wall_us / substeps;
        let ns_per_particle_substep = 1000.0 * us_per_substep / particles.max(1) as f64;
        let pct = |v: u64| 100.0 * v as f64 / self.total_us.max(1) as f64;
        println!(
            "{label:<34} N={particles:>6} substeps/frame={:>7.1} wall={wall_ms_frame:>8.2}ms/frame \
             slowdown_vs_real_time={:>6.1}x  {us_per_substep:>7.1}us/substep {ns_per_particle_substep:>6.0}ns/(particle*substep) \
             | p2g {:>4.1}% grid {:>4.1}% g2p {:>4.1}% cfl {:>4.1}% project {:>4.1}% other {:>4.1}%",
            substeps / frames,
            wall_ms_frame / sim_ms_frame,
            pct(self.p2g_us),
            pct(self.grid_us),
            pct(self.g2p_us),
            pct(self.cfl_us),
            pct(self.project_us),
            pct(self.other_us),
        );
    }
}

fn step_timed(sim: &mut Simulation) -> f64 {
    let t = std::time::Instant::now();
    sim.step();
    t.elapsed().as_secs_f64() * 1.0e6
}

/// Replace the spawn-time kernel-density volume estimate with the exact
/// volume every lattice sample represents: `spacing^2` grid cells. Only for
/// particles whose material does not own its own volume state (strict
/// fluids already set `V0 = m / rho0` themselves).
fn apply_exact_lattice_volume(sim: &mut Simulation, spacing: f32, materials: &[u32]) {
    let v0 = spacing * spacing;
    let p = sim.particles_mut();
    for i in 0..p.len() {
        if !materials.contains(&p.material_id[i]) {
            continue;
        }
        let j = p.deformation_gradient[i].determinant();
        p.initial_volume[i] = v0;
        p.volume[i] = v0 * j;
        p.density[i] = p.mass[i] / p.volume[i];
    }
}

/// Spawn-time density census against the exact lattice density `m/spacing^2`.
fn density_census(sim: &Simulation, spacing: f32) -> (f32, f32, f32) {
    let p = sim.particles();
    let mut min = f32::INFINITY;
    let mut sum = 0.0;
    let mut below = 0usize;
    for i in 0..p.len() {
        let exact = p.mass[i] / (spacing * spacing);
        let ratio = p.density[i] / exact;
        min = min.min(ratio);
        sum += ratio;
        if ratio < 0.9 {
            below += 1;
        }
    }
    (
        min,
        sum / p.len().max(1) as f32,
        100.0 * below as f32 / p.len().max(1) as f32,
    )
}

/// Per-particle material timestep bound, evaluated with the SAME trait
/// function the CFL scan calls. Returns (min, median) over particles and the
/// density ratio of the particle that sets the min.
fn material_bound_spread(
    sim: &Simulation,
    materials: &[(u32, &dyn MaterialModel)],
    spacing: f32,
) -> (f32, f32, f32) {
    let config = sim.config();
    let p = sim.particles();
    let mut dts = Vec::with_capacity(p.len());
    let mut worst = (f32::INFINITY, 1.0f32);
    for i in 0..p.len() {
        let Some((_, m)) = materials.iter().find(|(id, _)| *id == p.material_id[i]) else {
            continue;
        };
        let dt = m.timestep_bound(
            p.density[i],
            p.hardening_scale[i],
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        if dt.is_finite() {
            dts.push(dt);
            if dt < worst.0 {
                worst = (dt, p.density[i] / (p.mass[i] / (spacing * spacing)));
            }
        }
    }
    dts.sort_by(f32::total_cmp);
    let median = dts.get(dts.len() / 2).copied().unwrap_or(f32::NAN);
    (worst.0, median, worst.1)
}

fn block(config: &SimConfig, center: Vec2, cells: IVec2, material_id: u32) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: cells,
        box_center: center,
        material_id,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(config)
    }
}

// ── 1. the four rigid demos: profile, baseline vs exact lattice volume ──────

fn corotated_demo(exact_volume: bool) -> (Simulation, Vec<CorotatedMaterial>) {
    const E_PA: [f32; 3] = [5.0e5, 2.0e6, 1.0e7];
    let config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 100_000,
        material_cfl_coefficient: 0.5,
        ..SimConfig::earth(64, 0.01, 0.0005)
    };
    let props = |e: f32| Elastic {
        e_pa: e,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    let spawn = |slot: usize| {
        block(
            &config,
            Vec2::new([14.0, 32.0, 50.0][slot], 40.0),
            IVec2::new(10, 10),
            slot as u32,
        )
        .mass_from(&props(E_PA[slot]), &config)
    };
    let mats: Vec<CorotatedMaterial> = E_PA
        .iter()
        .map(|&e| CorotatedMaterial::from_physical(&props(e), &config))
        .collect();
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(CorotatedMaterial::from_physical(
            &props(E_PA[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(CorotatedMaterial::from_physical(&props(E_PA[1]), &config)),
        )
        .with_material(
            2,
            Box::new(CorotatedMaterial::from_physical(&props(E_PA[2]), &config)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));
    if exact_volume {
        apply_exact_lattice_volume(&mut sim, 0.5, &[0, 1, 2]);
    }
    (sim, mats)
}

fn viscoelastic_demo(exact_volume: bool) -> (Simulation, Vec<ViscoelasticMaterial>) {
    const ETA: [f32; 3] = [0.0, 100.0, 1000.0];
    let config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 100_000,
        material_cfl_coefficient: 0.5,
        ..SimConfig::earth(64, 0.01, 0.0006)
    };
    let props = |eta: f32| Viscoelastic {
        elastic: Elastic {
            e_pa: 2.0e6,
            nu: 0.45,
            rho_kg_m3: 1000.0,
        },
        eta_pa_s: eta,
    };
    let spawn = |slot: usize| {
        block(
            &config,
            Vec2::new([14.0, 32.0, 50.0][slot], 20.0),
            IVec2::new(10, 10),
            slot as u32,
        )
        .mass_from(&props(ETA[slot]), &config)
    };
    let mats: Vec<ViscoelasticMaterial> = ETA
        .iter()
        .map(|&eta| ViscoelasticMaterial::from_physical(&props(eta), &config))
        .collect();
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(ViscoelasticMaterial::from_physical(
            &props(ETA[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(ViscoelasticMaterial::from_physical(&props(ETA[1]), &config)),
        )
        .with_material(
            2,
            Box::new(ViscoelasticMaterial::from_physical(&props(ETA[2]), &config)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));
    if exact_volume {
        apply_exact_lattice_volume(&mut sim, 0.5, &[0, 1, 2]);
    }
    (sim, mats)
}

fn mui_demo(exact_volume: bool) -> (Simulation, Vec<MuIRheologyMaterial>) {
    const Q: [f32; 3] = [5.58, 3.00, 1.12];
    let config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 100_000,
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(64, 0.01, 0.0003)
    };
    let props = GranularProps {
        elastic: Elastic {
            e_pa: 15.0e6,
            nu: 0.3,
            rho_kg_m3: 1600.0,
        },
        friction_angle_deg: 30.0,
        dilatancy_angle_deg: 0.0,
    };
    let build = |q: f32| {
        let mut m = MuIRheologyMaterial::from_physical(&props, &config);
        m.inertial_q = q;
        m
    };
    let cells = IVec2::new(10, 22);
    let spawn = |slot: usize| {
        block(
            &config,
            Vec2::new([14.0, 32.0, 50.0][slot], 4.0 + cells.y as f32 * 0.5),
            cells,
            slot as u32,
        )
        .mass_from(&props, &config)
    };
    let mats: Vec<MuIRheologyMaterial> = Q.iter().map(|&q| build(q)).collect();
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(build(Q[0])))
        .with_material(1, Box::new(build(Q[1])))
        .with_material(2, Box::new(build(Q[2])))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));
    if exact_volume {
        apply_exact_lattice_volume(&mut sim, 0.5, &[0, 1, 2]);
    }
    (sim, mats)
}

fn run_demo<M: MaterialModel>(
    name: &str,
    (mut sim, mats): (Simulation, Vec<M>),
    frames: usize,
    checkpoints: &[usize],
) {
    let frame_dt = sim.config().dt;
    let (dmin, dmean, below) = density_census(&sim, 0.5);
    let refs: Vec<(u32, &dyn MaterialModel)> = mats
        .iter()
        .enumerate()
        .map(|(i, m)| (i as u32, m as &dyn MaterialModel))
        .collect();
    let (bmin, bmedian, worst_ratio) = material_bound_spread(&sim, &refs, 0.5);
    println!(
        "{name}: spawn density/exact min={dmin:.3} mean={dmean:.3} below-0.9={below:.1}% | \
         material dt bound min={bmin:.3e} median={bmedian:.3e} (bulk/worst={:.2}x, worst particle density ratio {worst_ratio:.3})",
        bmedian / bmin
    );
    let n = sim.particles().len();
    let mut cost = Cost::default();
    for f in 0..frames {
        let wall = step_timed(&mut sim);
        cost.add_frame(&sim, wall);
        if checkpoints.contains(&(f + 1)) {
            let mut line = format!("    t={:.4}s", (f + 1) as f32 * frame_dt);
            for id in 0..3u32 {
                let s = sim.material_state(id);
                line += &format!(
                    "  mat{id}: centroid=({:.2},{:.2}) vavg={:.2} detF={:.4}",
                    s.centroid.x, s.centroid.y, s.avg_speed, s.avg_det_f
                );
            }
            println!("{line}");
        }
    }
    let snap = sim.diagnostics_snapshot();
    println!(
        "    end: non_finite={} J range [{:.4}, {:.4}]",
        snap.non_finite_particle_values, snap.min_deformation_j, snap.max_deformation_j
    );
    cost.print(name, n, frame_dt);
}

/// Same scene as `examples/cpu/bingham_cost_probe.rs` (elastoviscoplastic
/// branch, three yield stresses), baseline vs exact lattice volume.
fn bingham_demo(exact_volume: bool) -> (Simulation, Vec<emerge::BinghamFluidMaterial>) {
    use emerge::{BinghamFluidMaterial, BinghamProps};
    const DX_M: f32 = 0.002;
    const RHO: f32 = 1000.0;
    const COLUMN: IVec2 = IVec2::new(4, 20);
    const YIELDS: [f32; 3] = [2.0, 60.0, 400.0];
    let config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 100_000,
        ..SimConfig::earth(64, DX_M, 0.002)
    };
    let v_max = (2.0 * 9.81 * COLUMN.y as f32 * DX_M).sqrt();
    let props = |tau0: f32| BinghamProps {
        rho_kg_m3: RHO,
        eta_pa_s: 0.5,
        bulk_modulus_pa: RHO * (10.0 * v_max).powi(2),
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / 0.05,
    };
    let spawn = |slot: usize| {
        block(
            &config,
            Vec2::new([12.0, 32.0, 52.0][slot], 2.0 + COLUMN.y as f32 * 0.5),
            COLUMN,
            slot as u32,
        )
        .mass_from(&props(YIELDS[slot]), &config)
    };
    let mats: Vec<BinghamFluidMaterial> = YIELDS
        .iter()
        .map(|&t| BinghamFluidMaterial::from_physical(&props(t), &config))
        .collect();
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props(YIELDS[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(YIELDS[1]),
                &config,
            )),
        )
        .with_material(
            2,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(YIELDS[2]),
                &config,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));
    if exact_volume {
        apply_exact_lattice_volume(&mut sim, 0.5, &[0, 1, 2]);
    }
    (sim, mats)
}

/// Number of OTHER particles whose 3x3 quadratic stencil shares at least one
/// grid node with particle `i` (base cells within 2 in both axes).
fn stencil_neighbours(x: &[Vec2], i: usize) -> usize {
    let base = |p: Vec2| (p - Vec2::splat(0.5)).floor();
    let bi = base(x[i]);
    x.iter()
        .enumerate()
        .filter(|&(j, &p)| {
            let d = (base(p) - bi).abs();
            j != i && d.x <= 2.0 && d.y <= 2.0
        })
        .count()
}

/// Open question (b): is "shares none of its 9 nodes with another particle"
/// a sufficient definition of an isolated particle for the Bai & Schroeder
/// 2022 single-particle bound? Fixed-dt stability of tiny clusters of
/// Corotated particles (E = 10 MPa, nu = 0.3, exact volume, no gravity,
/// perturbed F), as a function of how many grid nodes they share.
fn cluster_stability(offsets: &[Vec2], courant: f32, steps: usize) -> f32 {
    let dx_m = 0.01;
    let mut config = SimConfig {
        min_dt: 1.0e-9,
        max_substeps_per_step: 4,
        ..SimConfig::earth(64, dx_m, 1.0e-5)
    };
    config.gravity = Vec2::ZERO;
    let props = Elastic {
        e_pa: 1.0e7,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    let mat = CorotatedMaterial::from_physical(&props, &config);
    let one = |c: Vec2| {
        SpawnRegion {
            spacing: 1.0,
            box_size: IVec2::new(1, 1),
            box_center: c,
            material_id: 0,
            precompute_initial_volumes: false,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config)
    };
    let origin = Vec2::new(32.0, 32.0) + offsets[0];
    let mut sim = Simulation::new(config, one(origin))
        .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    for &d in &offsets[1..] {
        let _ = sim.add_body(one(origin + d));
    }
    apply_exact_lattice_volume(&mut sim, 1.0, &[0]);
    {
        let p = sim.particles_mut();
        for i in 0..p.len() {
            p.deformation_gradient[i] =
                Mat2::from_cols(Vec2::new(1.02, 0.004), Vec2::new(-0.003, 0.985));
            let j = p.deformation_gradient[i].determinant();
            p.volume[i] = p.initial_volume[i] * j;
            p.density[i] = p.mass[i] / p.volume[i];
        }
    }
    let rho = sim.particles().mass[0] / sim.particles().initial_volume[0];
    let c = ((mat.lambda + 2.0 * mat.mu) / rho).sqrt();
    let dt = courant * config.grid_cell_size / c;
    {
        let cfg = sim.config_mut();
        cfg.adaptive_timestep = false;
        cfg.dt = dt;
        cfg.min_dt = dt.min(cfg.min_dt);
    }
    // Energy proxy: kinetic + |F - I| growth; an unstable mode grows both.
    let size = |sim: &Simulation| -> f32 {
        let p = sim.particles();
        (0..p.len())
            .map(|i| {
                let d = p.deformation_gradient[i] - Mat2::IDENTITY;
                d.x_axis.length_squared()
                    + d.y_axis.length_squared()
                    + 1.0e-6 * p.v[i].length_squared()
            })
            .sum()
    };
    let s0 = size(&sim);
    let mut worst = 1.0f32;
    for _ in 0..steps {
        sim.step();
        let s = size(&sim);
        if !s.is_finite() {
            return f32::INFINITY;
        }
        worst = worst.max(s / s0);
        if worst > 1.0e6 {
            break;
        }
    }
    worst
}

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_isolated_particle_definition() {
    // Offsets in cells. Two particles share k nodes of their 3x3 stencils:
    // base cells 3 apart share none; 2 apart (diagonal) share 1; 2 apart
    // (axis) share 3; 1 apart (axis) share 6.
    let o = Vec2::new(0.3, 0.6);
    let configs: [(&str, Vec<Vec2>); 9] = [
        ("single particle at (.3,.6) in cell", vec![o]),
        ("single particle at (.5,.5)", vec![Vec2::new(0.5, 0.5)]),
        ("single particle at (.0,.0)", vec![Vec2::new(0.0, 0.0)]),
        ("single particle at (.0,.5)", vec![Vec2::new(0.0, 0.5)]),
        ("single particle at (.25,.75)", vec![Vec2::new(0.25, 0.75)]),
        (
            "pair, 3 cells apart (0 shared)",
            vec![o, Vec2::new(3.0, 0.0)],
        ),
        (
            "pair, diagonal 2 apart (1 shared)",
            vec![o, Vec2::new(2.0, 2.0)],
        ),
        (
            "pair, 2 apart on x (3 shared)",
            vec![o, Vec2::new(2.0, 0.0)],
        ),
        (
            "pair, 1 apart on x (6 shared)",
            vec![o, Vec2::new(1.0, 0.0)],
        ),
    ];
    for (name, offsets) in configs.iter() {
        let mut line = format!("{name:<36}");
        for courant in [0.5f32, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0] {
            let g = cluster_stability(offsets, courant, 3000);
            let tag = if g > 100.0 { "X" } else { "ok" };
            line += &format!(" f={courant:.2}:{tag}");
        }
        println!("{line}");
    }
}

// ── Pass A: conservation, measured ──────────────────────────────────────

/// Corotated strain energy of one particle, `V0 * (mu*|F-R|^2 + lambda/2*(J-1)^2)`
/// -- the potential whose Kirchhoff stress the engine uses.
fn corotated_energy(f: Mat2, v0: f32, lambda: f32, mu: f32) -> f64 {
    let d = f - polar_r(f);
    let j = f.determinant();
    (v0 * (mu * (d.x_axis.length_squared() + d.y_axis.length_squared())
        + 0.5 * lambda * (j - 1.0) * (j - 1.0))) as f64
}

struct Budget {
    momentum: glam::DVec2,
    angular: f64,
    kinetic: f64,
    elastic: f64,
    potential: f64,
}

/// Particle-side totals, APIC affine part included: with quadratic splines
/// `sum_i w_ip d d^T = dx^2/4 I` (dx = 1 cell), so a particle's affine field
/// carries angular momentum `m/4 (C_yx - C_xy)` and kinetic energy
/// `m/8 |C|_F^2` (Jiang et al. 2015, APIC).
fn budget(sim: &Simulation, lambda: f32, mu: f32) -> Budget {
    let p = sim.particles();
    let g = sim.config().gravity;
    let mut b = Budget {
        momentum: glam::DVec2::ZERO,
        angular: 0.0,
        kinetic: 0.0,
        elastic: 0.0,
        potential: 0.0,
    };
    for i in 0..p.len() {
        let m = p.mass[i] as f64;
        let x = p.x[i].as_dvec2();
        let v = p.v[i].as_dvec2();
        let c = p.velocity_gradient[i];
        b.momentum += m * v;
        b.angular +=
            m * (x.x * v.y - x.y * v.x) + m * 0.25 * (c.x_axis.y as f64 - c.y_axis.x as f64);
        b.kinetic += 0.5 * m * v.length_squared()
            + 0.125 * m * (c.x_axis.length_squared() + c.y_axis.length_squared()) as f64;
        b.elastic += corotated_energy(p.deformation_gradient[i], p.initial_volume[i], lambda, mu);
        b.potential -= m * (g.as_dvec2().dot(x));
    }
    b
}

/// A spinning, vibrating elastic block in zero gravity, never touching a
/// wall: linear and angular momentum must be conserved (APIC), and the total
/// energy may only decrease (transfer dissipation), never grow.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_conservation_free_spinning_block() {
    let mut config = SimConfig {
        min_dt: 1.0e-9,
        max_substeps_per_step: 100_000,
        ..SimConfig::earth(64, 0.01, 0.002)
    };
    config.gravity = Vec2::ZERO;
    let props = Elastic {
        e_pa: 1.0e6,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    let mat = CorotatedMaterial::from_physical(&props, &config);
    let spawn =
        block(&config, Vec2::new(32.0, 32.0), IVec2::new(12, 12), 0).mass_from(&props, &config);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
    {
        // Rigid spin (edge speed ~30 cells/s) plus a breathing mode.
        let p = sim.particles_mut();
        let n = p.len() as f32;
        let centre = p.x.iter().copied().sum::<Vec2>() / n;
        for i in 0..p.len() {
            let r = p.x[i] - centre;
            p.v[i] = 4.0 * Vec2::new(-r.y, r.x) + 1.5 * r;
        }
    }
    let b0 = budget(&sim, mat.lambda, mat.mu);
    let e0 = b0.kinetic + b0.elastic + b0.potential;
    println!(
        "t=0: |P|={:.4e} L={:.6e} E={:.6e} (kinetic {:.4e}, elastic {:.4e})",
        b0.momentum.length(),
        b0.angular,
        e0,
        b0.kinetic,
        b0.elastic
    );
    let mut substeps = 0usize;
    for f in 1..=150 {
        sim.step();
        substeps += sim.diagnostics_snapshot().substeps_last_step;
        if f % 30 == 0 {
            let b = budget(&sim, mat.lambda, mat.mu);
            let e = b.kinetic + b.elastic + b.potential;
            let mass: f64 = sim.particles().mass.iter().map(|&m| m as f64).sum();
            println!(
                "t={:.2}s ({substeps} substeps): dP/(M v_edge)={:.2e}  dL/L0={:+.3e}  E/E0={:.5}  (kinetic {:.4e}, elastic {:.4e})",
                f as f32 * 0.002,
                (b.momentum - b0.momentum).length() / (mass * 30.0),
                (b.angular - b0.angular) / b0.angular,
                e / e0,
                b.kinetic,
                b.elastic
            );
        }
    }
}

/// Friction to heat, closed at the grid: a block slides on a Coulomb floor.
/// The energy the boundary records as dissipated (`sum_i e_i m_i`, read after
/// every single substep) is compared with the mechanical energy the slide
/// actually lost beyond what the same slide loses on a frictionless floor.
fn sliding_block_energy(mu: f32) -> (f64, f64, f64) {
    let dx_m = 0.01;
    let props = Elastic {
        e_pa: 2.0e6,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    let mut config = SimConfig {
        min_dt: 1.0e-9,
        max_substeps_per_step: 4,
        ..SimConfig::earth(64, dx_m, 1.0e-5)
    };
    let mat = CorotatedMaterial::from_physical(&props, &config);
    // One substep per step(): dt at 0.4 of the elastic CFL, well inside it.
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0).sqrt();
    config.adaptive_timestep = false;
    config.dt = 0.4 / c;
    config.min_dt = config.dt;
    let spawn =
        block(&config, Vec2::new(20.0, 2.0 + 3.0), IVec2::new(10, 6), 0).mass_from(&props, &config);
    let boundary: Box<dyn emerge::BoundaryCondition> = if mu > 0.0 {
        Box::new(emerge::FrictionBoundary::new(config.boundary_thickness, mu))
    } else {
        Box::new(SlipBoundary::new(config.boundary_thickness))
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .with_boundary(boundary);
    apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
    // Let it settle on the floor at rest first, then launch it at 1 m/s.
    for _ in 0..400 {
        sim.step();
    }
    {
        let p = sim.particles_mut();
        for i in 0..p.len() {
            p.v[i] = Vec2::new(0.5 / dx_m, 0.0);
            p.velocity_gradient[i] = Mat2::ZERO;
        }
    }
    let b0 = budget(&sim, mat.lambda, mat.mu);
    let e0 = b0.kinetic + b0.elastic + b0.potential;
    let res = sim.config().grid_res as i32;
    let mut heat = 0.0f64;
    for _ in 0..6000 {
        sim.step();
        let g = sim.grid();
        for x in 0..res {
            for y in 0..res {
                let cell = IVec2::new(x, y);
                let e = g.friction_heat_at(cell);
                if e > 0.0 {
                    heat += (e * g.mass_at(cell)) as f64;
                }
            }
        }
    }
    let b1 = budget(&sim, mat.lambda, mat.mu);
    let e1 = b1.kinetic + b1.elastic + b1.potential;
    (e0 - e1, heat, b1.momentum.x / b0.momentum.x)
}

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_friction_heat_energy_balance() {
    let (lost_slip, heat_slip, p_slip) = sliding_block_energy(0.0);
    let (lost_fric, heat_fric, p_fric) = sliding_block_energy(0.3);
    println!(
        "frictionless floor: mechanical energy lost {lost_slip:.4e} (numerical), heat recorded {heat_slip:.4e}, final/initial x-momentum {p_slip:.3}"
    );
    println!(
        "Coulomb floor mu=0.3: mechanical energy lost {lost_fric:.4e}, heat recorded {heat_fric:.4e}, final/initial x-momentum {p_fric:.3}"
    );
    println!(
        "heat recorded / (loss with friction - loss without) = {:.3}",
        heat_fric / (lost_fric - lost_slip)
    );
}

/// Temporal order of accuracy of the explicit loop: the same vibrating free
/// block run at fixed dt, dt/2, dt/4, dt/8 to the same time T; error of the
/// corner particle's position against the finest run.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_temporal_order_of_accuracy() {
    let run = |courant: f32| -> (Vec2, f64) {
        let mut config = SimConfig {
            min_dt: 1.0e-9,
            max_substeps_per_step: 4,
            ..SimConfig::earth(64, 0.01, 1.0e-5)
        };
        config.gravity = Vec2::ZERO;
        let props = Elastic {
            e_pa: 1.0e6,
            nu: 0.3,
            rho_kg_m3: 1000.0,
        };
        let mat = CorotatedMaterial::from_physical(&props, &config);
        let spawn =
            block(&config, Vec2::new(32.0, 32.0), IVec2::new(10, 10), 0).mass_from(&props, &config);
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
        {
            let p = sim.particles_mut();
            let n = p.len() as f32;
            let centre = p.x.iter().copied().sum::<Vec2>() / n;
            for i in 0..p.len() {
                let r = p.x[i] - centre;
                p.v[i] = 3.0 * r + 2.0 * Vec2::new(-r.y, r.x);
            }
        }
        let c = ((mat.lambda + 2.0 * mat.mu) / 1.0).sqrt();
        let dt = courant / c;
        let t_end = 0.4 * 256.0 / c; // same physical time for every dt
        let steps = (t_end / dt).round() as usize;
        {
            let cfg = sim.config_mut();
            cfg.adaptive_timestep = false;
            cfg.dt = dt;
            cfg.min_dt = dt;
        }
        let b0 = budget(&sim, mat.lambda, mat.mu);
        let e0 = b0.kinetic + b0.elastic;
        for _ in 0..steps {
            sim.step();
        }
        let b1 = budget(&sim, mat.lambda, mat.mu);
        (sim.particles().x[0], (b1.kinetic + b1.elastic) / e0)
    };
    let courants = [0.4f32, 0.2, 0.1, 0.05, 0.025];
    let xs: Vec<(Vec2, f64)> = courants.iter().map(|&c| run(c)).collect();
    let reference = xs[xs.len() - 1].0;
    let mut prev: Option<f32> = None;
    for (k, &c) in courants.iter().enumerate() {
        let err = (xs[k].0 - reference).length();
        let ratio = prev.map(|p| p / err).unwrap_or(f32::NAN);
        println!(
            "f={c:.3}: corner position error vs finest {err:.3e} cells (ratio to previous {ratio:.2}) | energy kept at T: {:.4}",
            xs[k].1
        );
        prev = Some(err);
    }
}

/// Same single-particle system, long horizon, printing the actual growth
/// ratio so a slow instability (growth ~0.1% per step) cannot hide under a
/// short run's detection threshold.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_isolated_particle_long_horizon() {
    for courant in [0.55f32, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85] {
        let g = cluster_stability(&[Vec2::new(0.3, 0.6)], courant, 30_000);
        println!("single particle, f={courant:.2}: max growth over 30 000 steps = {g:.3e}");
    }
}

/// How often do the silent safety nets fire in the four delivered rigid
/// demos (baseline construction)? `j_projection_count` counts
/// `project_particle_state_to_admissible` repairs; position clamps are not
/// counted by the engine, so they are detected here as particles sitting
/// exactly on the clamp bound.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_safety_net_activity_in_demos() {
    fn tally(name: &str, mut sim: Simulation, frames: usize) {
        let (mut proj, mut dropped, mut on_bound) = (0usize, 0.0f32, 0usize);
        let res = sim.config().grid_res;
        let t = sim.config().boundary_thickness;
        let (lo, hi) = (t.saturating_sub(1) as f32, (res - t) as f32);
        for _ in 0..frames {
            sim.step();
            let s = sim.diagnostics_snapshot();
            proj += s.j_projection_count;
            dropped += s.sim_time_dropped;
            let p = sim.particles();
            on_bound +=
                p.x.iter()
                    .filter(|x| x.x == lo || x.x == hi || x.y == lo || x.y == hi)
                    .count();
        }
        println!(
            "{name:<14} {frames} frames: J/state repairs {proj}, simulated time dropped {dropped:.2e} s, particle-frames sitting exactly on the clamp bound {on_bound}"
        );
    }
    tally("corotated", corotated_demo(false).0, 700);
    tally("viscoelastic", viscoelastic_demo(false).0, 500);
    tally("mu(I)", mui_demo(false).0, 500);
    tally("bingham", bingham_demo(false).0, 120);
}

/// Open question (a) of the 2026-09-21 report: where does the J = 11.1
/// particle of the baseline Bingham run come from?
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_bingham_j11_investigation() {
    let (mut sim, _) = bingham_demo(false);
    {
        let p = sim.particles();
        for id in 0..3u32 {
            let (mut min, mut sum, mut n) = (f32::INFINITY, 0.0f32, 0usize);
            for i in 0..p.len() {
                if p.material_id[i] != id {
                    continue;
                }
                let r = p.density[i] / (p.mass[i] / 0.25);
                min = min.min(r);
                sum += r;
                n += 1;
            }
            println!(
                "spawn census body/material {id}: density / exact-lattice density min {min:.3} mean {:.3} (n={n})",
                sum / n as f32
            );
        }
    }
    let dx_m = sim.config().dx_meters;
    let mut worst_seen = 1.0f32;
    let mut tracked: Option<usize> = None;
    for f in 0..120 {
        sim.step();
        let p = sim.particles();
        let (mut jmax, mut imax) = (0.0f32, 0usize);
        for i in 0..p.len() {
            let j = p.deformation_gradient[i].determinant();
            if j > jmax {
                jmax = j;
                imax = i;
            }
        }
        let over = (0..p.len())
            .filter(|&i| p.deformation_gradient[i].determinant() > 1.5)
            .count();
        if jmax > worst_seen * 1.5 || f % 20 == 19 {
            worst_seen = worst_seen.max(jmax);
            let x = p.x[imax];
            println!(
                "frame {f:>3}: max J {jmax:>7.3} particle {imax} (mat {}) at ({:.2},{:.2}) v=({:.1},{:.1}) cells/s density/rho0 {:.3} stencil-neighbours {} | particles with J>1.5: {over}",
                p.material_id[imax],
                x.x,
                x.y,
                p.v[imax].x,
                p.v[imax].y,
                p.density[imax] / (p.mass[imax] / 0.25),
                stencil_neighbours(&p.x, imax)
            );
            if jmax > 5.0 && tracked.is_none() {
                tracked = Some(imax);
            }
        }
    }
    if let Some(i) = tracked {
        let p = sim.particles();
        println!(
            "tracked particle {i}: initial_volume {:.4} (exact lattice 0.25), mass {:.4}, final J {:.3}, stencil-neighbours now {}, speed {:.3} m/s",
            p.initial_volume[i],
            p.mass[i],
            p.deformation_gradient[i].determinant(),
            stencil_neighbours(&p.x, i),
            p.v[i].length() * dx_m
        );
    }
}

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_bingham_baseline_vs_exact_lattice_volume() {
    run_demo("bingham BASELINE", bingham_demo(false), 120, &[60, 120]);
    run_demo("bingham EXACT-V0", bingham_demo(true), 120, &[60, 120]);
}

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_rigid_demos_baseline_vs_exact_lattice_volume() {
    // Long enough for every block to land and every column to collapse.
    run_demo(
        "corotated BASELINE",
        corotated_demo(false),
        700,
        &[200, 500, 600, 700],
    );
    run_demo(
        "corotated EXACT-V0",
        corotated_demo(true),
        700,
        &[200, 500, 600, 700],
    );
    run_demo(
        "viscoelastic BASELINE",
        viscoelastic_demo(false),
        500,
        &[250, 350, 425, 500],
    );
    run_demo(
        "viscoelastic EXACT-V0",
        viscoelastic_demo(true),
        500,
        &[250, 350, 425, 500],
    );
    run_demo("mu(I) BASELINE", mui_demo(false), 500, &[100, 250, 500]);
    run_demo("mu(I) EXACT-V0", mui_demo(true), 500, &[100, 250, 500]);
}

// ── 2. stiffness sweep: substeps must scale as sqrt(E) ──────────────────────

fn single_corotated_block(e_pa: f32, dx_m: f32, grid: usize, frame_dt: f32) -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-8,
        max_substeps_per_step: 1_000_000,
        material_cfl_coefficient: 0.5,
        ..SimConfig::earth(grid, dx_m, frame_dt)
    };
    let props = Elastic {
        e_pa,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    // A 0.1 m x 0.1 m block resting 5 cm above the floor, whatever dx is.
    let side_cells = (0.1 / dx_m).round() as i32;
    let center = Vec2::new(
        grid as f32 * 0.5,
        (config.boundary_thickness as f32) + (0.05 / dx_m) + side_cells as f32 * 0.5,
    );
    let spawn = block(&config, center, IVec2::splat(side_cells), 0).mass_from(&props, &config);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
    sim
}

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_stiffness_sweep() {
    let frame_dt = 0.001;
    let mut first: Option<f64> = None;
    for e in [5.0e5f32, 2.0e6, 8.0e6, 3.2e7] {
        let mut sim = single_corotated_block(e, 0.01, 64, frame_dt);
        let n = sim.particles().len();
        let mut cost = Cost::default();
        for _ in 0..60 {
            let wall = step_timed(&mut sim);
            cost.add_frame(&sim, wall);
        }
        let per_frame = cost.substeps as f64 / cost.frames as f64;
        let base = *first.get_or_insert(per_frame);
        println!(
            "E={e:.1e} Pa: substeps/frame {per_frame:.1} (x{:.2} vs E=5e5; sqrt(E) predicts x{:.2})",
            per_frame / base,
            (e / 5.0e5).sqrt()
        );
        cost.print(&format!("  E={e:.1e}"), n, frame_dt);
    }
}

// ── 3. resolution sweep: cost per simulated second ~ dx^-3 in 2D ────────────

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_resolution_sweep() {
    // Same physical scene (0.64 m domain, 0.1 m block, E=2 MPa) at three dx.
    let frame_dt = 0.001;
    for (dx, grid) in [(0.02f32, 32usize), (0.01, 64), (0.005, 128), (0.0025, 256)] {
        let mut sim = single_corotated_block(2.0e6, dx, grid, frame_dt);
        let n = sim.particles().len();
        let mut cost = Cost::default();
        for _ in 0..40 {
            let wall = step_timed(&mut sim);
            cost.add_frame(&sim, wall);
        }
        cost.print(&format!("dx={:.1}mm grid={grid}", dx * 1000.0), n, frame_dt);
    }
}

// ── 4. violent impact: how far does dt collapse, and is it survivable ───────

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_violent_impact_stress() {
    // A stiff block (E=10 MPa) thrown at the floor at 5, 10 and 20 m/s --
    // rockfall-class speeds. Nothing is softened to make it pass.
    for speed_m_s in [5.0f32, 10.0, 20.0] {
        let frame_dt = 0.0005;
        let mut sim = single_corotated_block(1.0e7, 0.01, 64, frame_dt);
        let v = Vec2::new(0.0, -speed_m_s / 0.01);
        {
            let p = sim.particles_mut();
            for i in 0..p.len() {
                p.v[i] = v;
            }
        }
        let n = sim.particles().len();
        let mut cost = Cost::default();
        let mut rest_substeps = 0usize;
        let mut peak_substeps = 0usize;
        let mut worst_j = (f32::INFINITY, f32::NEG_INFINITY);
        let mut non_finite = 0usize;
        let mut dropped = 0.0f32;
        for f in 0..120 {
            let wall = step_timed(&mut sim);
            cost.add_frame(&sim, wall);
            let s = sim.diagnostics_snapshot();
            if f == 0 {
                rest_substeps = s.substeps_last_step;
            }
            peak_substeps = peak_substeps.max(s.substeps_last_step);
            worst_j = (
                worst_j.0.min(s.min_deformation_j),
                worst_j.1.max(s.max_deformation_j),
            );
            non_finite += s.non_finite_particle_values;
            dropped += s.sim_time_dropped;
        }
        println!(
            "impact {speed_m_s:>4.0} m/s: substeps/frame first={rest_substeps} peak={peak_substeps} (x{:.1}) \
             J range [{:.3}, {:.3}] non_finite={non_finite} dropped_sim_time={dropped:.2e}s",
            peak_substeps as f32 / rest_substeps.max(1) as f32,
            worst_j.0,
            worst_j.1,
        );
        cost.print(&format!("  impact {speed_m_s:.0} m/s"), n, frame_dt);
    }
}

// ── 5. mixed scene: what the single global dt costs ─────────────────────────

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_mixed_scene_global_cfl_cost() {
    // Showcase-like mix: a small patch of real sand (E=15 MPa), a large soft
    // elastic body (E=50 kPa) and a water pool whose sound speed is derated
    // to 10x the scene's own peak speed (the WCSPH convention already used
    // by the Bingham demo, density error ~1%).
    let frame_dt = 0.002;
    let config = SimConfig {
        min_dt: 1.0e-8,
        max_substeps_per_step: 1_000_000,
        ..SimConfig::earth(64, 0.01, frame_dt)
    };
    let sand_props = GranularProps {
        elastic: Elastic {
            e_pa: 15.0e6,
            nu: 0.3,
            rho_kg_m3: 1600.0,
        },
        friction_angle_deg: 30.0,
        dilatancy_angle_deg: 0.0,
    };
    let soft_props = Elastic {
        e_pa: 5.0e4,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    let v_max = (2.0f32 * 9.81 * 0.3).sqrt();
    let water_props = Fluid {
        rho_kg_m3: 1000.0,
        eta_pa_s: 1.0e-3,
        bulk_modulus_pa: 1000.0 * (10.0 * v_max).powi(2),
        yield_stress_pa: None,
    };
    let sand = DruckerPragerMaterial::from_physical(&sand_props, &config);
    let soft = CorotatedMaterial::from_physical(&soft_props, &config);
    let water = water_props.material(&config);

    let sand_spawn =
        block(&config, Vec2::new(12.0, 8.0), IVec2::new(8, 8), 0).mass_from(&sand_props, &config);
    let soft_spawn = block(&config, Vec2::new(30.0, 10.0), IVec2::new(16, 12), 1)
        .mass_from(&soft_props, &config);
    let water_spawn = block(&config, Vec2::new(50.0, 8.0), IVec2::new(20, 10), 2)
        .mass_from(&water_props, &config);

    let mut sim = Simulation::new(config, sand_spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::from_physical(
            &sand_props,
            &config,
        )))
        .with_material(
            1,
            Box::new(CorotatedMaterial::from_physical(&soft_props, &config)),
        )
        .with_material(2, water_props.material(&config))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(soft_spawn);
    let _ = sim.add_body(water_spawn);
    apply_exact_lattice_volume(&mut sim, 0.5, &[0, 1]);

    // Per-material bound at each material's own rest state.
    let p = sim.particles();
    let mut count = [0usize; 3];
    let mut dt_min = [f32::INFINITY; 3];
    for i in 0..p.len() {
        let id = p.material_id[i] as usize;
        let m: &dyn MaterialModel = match id {
            0 => &sand,
            1 => &soft,
            _ => water.as_ref(),
        };
        let dt = m.timestep_bound(
            p.density[i],
            p.hardening_scale[i],
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        count[id] += 1;
        if dt.is_finite() {
            dt_min[id] = dt_min[id].min(dt);
        }
    }
    // The gravity bound applies to everyone.
    let g = config.gravity.length();
    let gravity_dt = (config.cfl_coefficient * config.grid_cell_size / g).sqrt();
    let per_material: Vec<f32> = dt_min.iter().map(|d| d.min(gravity_dt)).collect();
    let global = per_material.iter().copied().fold(f32::INFINITY, f32::min);
    let global_work = p.len() as f64 * (frame_dt / global).ceil() as f64;
    let multirate_work: f64 = (0..3)
        .map(|k| count[k] as f64 * (frame_dt / per_material[k]).ceil() as f64)
        .sum();
    for (k, name) in ["sand E=15MPa", "soft elastic E=50kPa", "water (derated)"]
        .iter()
        .enumerate()
    {
        println!(
            "  {name:<22} N={:>5} own dt bound={:.3e}s -> {:>6.0} substeps/frame on its own",
            count[k],
            per_material[k],
            (frame_dt / per_material[k]).ceil()
        );
    }
    println!(
        "  global dt={global:.3e}s: particle-substeps/frame global={global_work:.0} vs \
         ideal per-material rates={multirate_work:.0} -> multi-rate upper bound x{:.1}",
        global_work / multirate_work
    );
    let n = sim.particles().len();
    let mut cost = Cost::default();
    for _ in 0..30 {
        let wall = step_timed(&mut sim);
        cost.add_frame(&sim, wall);
    }
    cost.print("mixed scene (measured)", n, frame_dt);
}

// ── 6. particle-count scaling of the per-substep cost ──────────────────────

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_particle_count_scaling() {
    let frame_dt = 0.0005;
    for side in [10i32, 20, 40, 80] {
        let grid = 128usize;
        let config = SimConfig {
            min_dt: 1.0e-8,
            max_substeps_per_step: 1_000_000,
            material_cfl_coefficient: 0.5,
            ..SimConfig::earth(grid, 0.01, frame_dt)
        };
        let props = Elastic {
            e_pa: 2.0e6,
            nu: 0.3,
            rho_kg_m3: 1000.0,
        };
        let spawn = block(
            &config,
            Vec2::new(64.0, 4.0 + side as f32 * 0.5),
            IVec2::splat(side),
            0,
        )
        .mass_from(&props, &config);
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
        let n = sim.particles().len();
        let mut cost = Cost::default();
        for _ in 0..20 {
            let wall = step_timed(&mut sim);
            cost.add_frame(&sim, wall);
        }
        cost.print(&format!("block {side}x{side} cells"), n, frame_dt);
    }
}

// ── 6b. where is the REAL explicit stability limit? (fixed-dt sweep) ───────

/// A free elastic block (no gravity, no wall contact) with a small random
/// velocity field that excites every grid mode, stepped at a FIXED dt equal
/// to `courant * dx / c_p`. A stable explicit scheme keeps the kinetic
/// energy bounded by the initial total energy; an unstable one grows it
/// exponentially. The engine's own material bound is `courant = 0.5`.
fn fixed_courant_run(courant: f32, e_pa: f32, steps: usize) -> (f32, bool) {
    let dx_m = 0.01;
    let mut config = SimConfig {
        min_dt: 1.0e-9,
        max_substeps_per_step: 4,
        ..SimConfig::earth(64, dx_m, 1.0e-5)
    };
    config.gravity = Vec2::ZERO;
    let props = Elastic {
        e_pa,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    let mat = CorotatedMaterial::from_physical(&props, &config);
    let spawn = SpawnRegion {
        rng_seed: 7,
        initial_velocity_scale: 2.0,
        ..block(&config, Vec2::new(32.0, 32.0), IVec2::new(12, 12), 0)
    }
    .mass_from(&props, &config);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
    let rho = sim.particles().density[0];
    let c = ((mat.lambda + 2.0 * mat.mu) / rho).sqrt();
    let dt = courant * config.grid_cell_size / c;
    {
        let cfg = sim.config_mut();
        cfg.adaptive_timestep = false;
        cfg.dt = dt;
        cfg.min_dt = dt.min(cfg.min_dt);
    }
    let ke = |sim: &Simulation| -> f32 {
        let p = sim.particles();
        (0..p.len())
            .map(|i| 0.5 * p.mass[i] * p.v[i].length_squared())
            .sum()
    };
    let ke0 = ke(&sim);
    let mut worst = 1.0f32;
    let mut finite = true;
    for _ in 0..steps {
        sim.step();
        let k = ke(&sim);
        if !k.is_finite() {
            finite = false;
            break;
        }
        worst = worst.max(k / ke0);
        if worst > 1.0e6 {
            break;
        }
    }
    (worst, finite)
}

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_explicit_stability_limit_sweep() {
    for e in [2.0e6f32, 1.0e7] {
        for courant in [0.5f32, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5] {
            let (growth, finite) = fixed_courant_run(courant, e, 6000);
            let verdict = if !finite || growth > 100.0 {
                "UNSTABLE"
            } else {
                "stable"
            };
            println!(
                "E={e:.0e} Pa  courant={courant:.2} (dt = {courant:.2} dx/c_p): max KE/KE0 over 6000 steps = {growth:>10.3e}  {verdict}"
            );
        }
    }
}

/// Stress test of the material CFL coefficient headroom, adaptive dt ON
/// (the real engine path), exact lattice volume. (a) the corotated demo run
/// through landing and rebound; (b) a stiff block thrown at the floor at
/// 20 m/s, where J reaches ~0.64 and Sun, Shinar & Schroeder 2020 warn that
/// the rest-state sound speed under-estimates the true one.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_material_cfl_coefficient_headroom_stress() {
    for coefficient in [0.5f32, 0.7, 0.8, 0.9, 1.0] {
        let (mut sim, _) = corotated_demo(true);
        sim.config_mut().material_cfl_coefficient = coefficient;
        let n = sim.particles().len();
        let dt = sim.config().dt;
        let mut cost = Cost::default();
        let mut line = String::new();
        for f in 0..700 {
            let wall = step_timed(&mut sim);
            cost.add_frame(&sim, wall);
            if [600usize, 700].contains(&(f + 1)) {
                for id in 0..3u32 {
                    let s = sim.material_state(id);
                    line += &format!(
                        " t={:.2}s mat{id} y={:.2}",
                        (f + 1) as f32 * dt,
                        s.centroid.y
                    );
                }
            }
        }
        let snap = sim.diagnostics_snapshot();
        println!(
            "coef={coefficient:.1} demo:{line} | non_finite={} J=[{:.4},{:.4}]",
            snap.non_finite_particle_values, snap.min_deformation_j, snap.max_deformation_j
        );
        cost.print(&format!("  coef={coefficient:.1} corotated demo"), n, dt);

        let mut impact = single_corotated_block(1.0e7, 0.01, 64, 0.0005);
        impact.config_mut().material_cfl_coefficient = coefficient;
        {
            let p = impact.particles_mut();
            for i in 0..p.len() {
                p.v[i] = Vec2::new(0.0, -20.0 / 0.01);
            }
        }
        let ke = |s: &Simulation| -> f64 {
            let p = s.particles();
            (0..p.len())
                .map(|i| 0.5 * p.mass[i] as f64 * p.v[i].length_squared() as f64)
                .sum()
        };
        let ke0 = ke(&impact);
        let mut worst = 0.0f64;
        let mut jr = (f32::INFINITY, f32::NEG_INFINITY);
        let mut non_finite = 0usize;
        let mut substeps = 0usize;
        for _ in 0..400 {
            impact.step();
            let s = impact.diagnostics_snapshot();
            substeps += s.substeps_last_step;
            non_finite += s.non_finite_particle_values;
            jr = (jr.0.min(s.min_deformation_j), jr.1.max(s.max_deformation_j));
            worst = worst.max(ke(&impact) / ke0);
        }
        println!(
            "  coef={coefficient:.1} 20 m/s impact: substeps/frame={:.1} max KE/KE0={worst:.3} J=[{:.3},{:.3}] non_finite={non_finite}",
            substeps as f64 / 400.0,
            jr.0,
            jr.1
        );
    }
}

// ── 7. lean dense-grid reference kernel: the per-substep cost floor ─────────

/// Minimal 2D MLS-MPM (Hu et al. 2018), same arithmetic per particle as the
/// engine's Corotated path: quadratic B-spline weights, APIC affine momentum,
/// fused MLS stress impulse `-4 dt V0 tau`, Corotated Kirchhoff stress with
/// the same closed-form 2D polar decomposition, exact 2x2 matrix-exponential
/// F update, gravity, 2-cell slip walls, and the velocity CFL max folded
/// into G2P. Dense grid, single thread, no material dispatch.
struct Lean {
    x: Vec<Vec2>,
    v: Vec<Vec2>,
    c: Vec<Mat2>,
    f: Vec<Mat2>,
    mass: f32,
    vol0: f32,
    lambda: f32,
    mu: f32,
    res: usize,
    grid_m: Vec<f32>,
    grid_p: Vec<Vec2>,
    gravity: Vec2,
}

fn polar_r(f: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt().max(1.0e-12);
    let (c, s) = (x / norm, y / norm);
    Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c))
}

fn exp2x2(a: Mat2) -> Mat2 {
    let s = 0.5 * (a.x_axis.x + a.y_axis.y);
    let b = a - Mat2::from_diagonal(Vec2::splat(s));
    let q2 = -b.determinant();
    let (ch, sh) = if q2 > 1.0e-12 {
        let q = q2.sqrt();
        (q.cosh(), q.sinh() / q)
    } else if q2 < -1.0e-12 {
        let q = (-q2).sqrt();
        (q.cos(), q.sin() / q)
    } else {
        (1.0, 1.0)
    };
    s.exp() * (Mat2::from_diagonal(Vec2::splat(ch)) + sh * b)
}

impl Lean {
    fn substep(&mut self, dt: f32) -> f32 {
        let res = self.res;
        self.grid_m.fill(0.0);
        self.grid_p.fill(Vec2::ZERO);
        let stress_scale = -4.0 * dt * self.vol0;
        for p in 0..self.x.len() {
            let xp = self.x[p];
            let base = (xp - Vec2::splat(0.5)).floor();
            let fx = xp - base;
            let w = [
                0.5 * (Vec2::splat(1.5) - fx) * (Vec2::splat(1.5) - fx),
                Vec2::splat(0.75) - (fx - Vec2::ONE) * (fx - Vec2::ONE),
                0.5 * (fx - Vec2::splat(0.5)) * (fx - Vec2::splat(0.5)),
            ];
            let f = self.f[p];
            let j = f.determinant();
            let tau = 2.0 * self.mu * (f - polar_r(f)) * f.transpose()
                + Mat2::from_diagonal(Vec2::splat(self.lambda * (j - 1.0) * j));
            let affine = stress_scale * tau + self.mass * self.c[p];
            let mv = self.mass * self.v[p];
            let (bx, by) = (base.x as usize, base.y as usize);
            for (i, wi) in w.iter().enumerate() {
                for (k, wk) in w.iter().enumerate() {
                    let weight = wi.x * wk.y;
                    let dpos = Vec2::new(i as f32, k as f32) - fx;
                    let idx = (bx + i) * res + (by + k);
                    self.grid_p[idx] += weight * (mv + affine * dpos);
                    self.grid_m[idx] += weight * self.mass;
                }
            }
        }
        let wall = 2usize;
        for gx in 0..res {
            for gy in 0..res {
                let idx = gx * res + gy;
                let m = self.grid_m[idx];
                if m <= 0.0 {
                    continue;
                }
                let mut vel = self.grid_p[idx] / m + dt * self.gravity;
                if (gx < wall && vel.x < 0.0) || (gx >= res - wall && vel.x > 0.0) {
                    vel.x = 0.0;
                }
                if (gy < wall && vel.y < 0.0) || (gy >= res - wall && vel.y > 0.0) {
                    vel.y = 0.0;
                }
                self.grid_p[idx] = vel;
            }
        }
        let mut max_speed = 0.0f32;
        for p in 0..self.x.len() {
            let xp = self.x[p];
            let base = (xp - Vec2::splat(0.5)).floor();
            let fx = xp - base;
            let w = [
                0.5 * (Vec2::splat(1.5) - fx) * (Vec2::splat(1.5) - fx),
                Vec2::splat(0.75) - (fx - Vec2::ONE) * (fx - Vec2::ONE),
                0.5 * (fx - Vec2::splat(0.5)) * (fx - Vec2::splat(0.5)),
            ];
            let (bx, by) = (base.x as usize, base.y as usize);
            let mut nv = Vec2::ZERO;
            let mut b = Mat2::ZERO;
            for (i, wi) in w.iter().enumerate() {
                for (k, wk) in w.iter().enumerate() {
                    let weight = wi.x * wk.y;
                    let dpos = Vec2::new(i as f32, k as f32) - fx;
                    let gv = self.grid_p[(bx + i) * res + (by + k)];
                    nv += weight * gv;
                    b += weight * Mat2::from_cols(gv * dpos.x, gv * dpos.y);
                }
            }
            let c = 4.0 * b;
            self.v[p] = nv;
            self.c[p] = c;
            self.x[p] = (xp + dt * nv).clamp(Vec2::splat(1.0), Vec2::splat(res as f32 - 2.0));
            self.f[p] = exp2x2(dt * c) * self.f[p];
            let grad = (c.x_axis.length_squared() + c.y_axis.length_squared()).sqrt();
            max_speed = max_speed.max(nv.length() + grad * 1.5 * std::f32::consts::SQRT_2);
        }
        max_speed
    }
}

/// Raw grid pointer shared across a colour pass. Sound only because blocks
/// of one colour are 8 cells apart while a 4x4 block's quadratic stencil
/// footprint is 7 cells wide: no two tasks of the same pass touch a node.
#[derive(Clone, Copy)]
struct GridPtr(*mut Vec2, *mut f32);
unsafe impl Send for GridPtr {}
unsafe impl Sync for GridPtr {}

impl GridPtr {
    // Accessed through methods so a closure captures the whole (Sync)
    // struct, not its raw-pointer fields one by one.
    fn momentum(self) -> *mut Vec2 {
        self.0
    }
    fn mass(self) -> *mut f32 {
        self.1
    }
}

impl Lean {
    /// Same arithmetic as `substep`, parallelised the way `tmp/sparkl`
    /// does it (`particle_to_grid.rs`: region colouring, no atomics, no
    /// per-task grids): particles counting-sorted into 4x4-cell blocks,
    /// P2G run one colour (of 4) at a time with direct writes into ONE
    /// dense grid, grid update and G2P as plain parallel loops, the
    /// velocity CFL max folded into the G2P reduction.
    fn substep_parallel(
        &mut self,
        dt: f32,
        order: &mut Vec<u32>,
        offsets: &mut Vec<u32>,
        phase_us: &mut [f64; 4],
    ) -> f32 {
        use rayon::prelude::*;
        let t_sort = std::time::Instant::now();
        const B: usize = 4;
        let res = self.res;
        let nb = res.div_ceil(B);
        // Counting sort by block.
        offsets.clear();
        offsets.resize(nb * nb + 1, 0);
        let block_of = |x: Vec2| -> usize {
            let base = (x - Vec2::splat(0.5)).floor();
            (base.x as usize / B) * nb + (base.y as usize / B)
        };
        for &x in &self.x {
            offsets[block_of(x) + 1] += 1;
        }
        for b in 0..nb * nb {
            offsets[b + 1] += offsets[b];
        }
        order.clear();
        order.resize(self.x.len(), 0);
        {
            let mut cursor = offsets.clone();
            for (p, &x) in self.x.iter().enumerate() {
                let b = block_of(x);
                order[cursor[b] as usize] = p as u32;
                cursor[b] += 1;
            }
        }
        self.grid_m.par_iter_mut().for_each(|m| *m = 0.0);
        self.grid_p.par_iter_mut().for_each(|p| *p = Vec2::ZERO);
        phase_us[0] += t_sort.elapsed().as_secs_f64() * 1.0e6;
        let t_p2g = std::time::Instant::now();
        let stress_scale = -4.0 * dt * self.vol0;
        let ptr = GridPtr(self.grid_p.as_mut_ptr(), self.grid_m.as_mut_ptr());
        let (x, v, c, f) = (&self.x, &self.v, &self.c, &self.f);
        let (mass, mu, lambda) = (self.mass, self.mu, self.lambda);
        for color in 0..4usize {
            (0..nb * nb)
                .into_par_iter()
                .filter(|&b| ((b / nb) % 2) + 2 * ((b % nb) % 2) == color)
                .for_each(|b| {
                    for &p in &order[offsets[b] as usize..offsets[b + 1] as usize] {
                        let p = p as usize;
                        let xp = x[p];
                        let base = (xp - Vec2::splat(0.5)).floor();
                        let fx = xp - base;
                        let w = [
                            0.5 * (Vec2::splat(1.5) - fx) * (Vec2::splat(1.5) - fx),
                            Vec2::splat(0.75) - (fx - Vec2::ONE) * (fx - Vec2::ONE),
                            0.5 * (fx - Vec2::splat(0.5)) * (fx - Vec2::splat(0.5)),
                        ];
                        let fp = f[p];
                        let j = fp.determinant();
                        let tau = 2.0 * mu * (fp - polar_r(fp)) * fp.transpose()
                            + Mat2::from_diagonal(Vec2::splat(lambda * (j - 1.0) * j));
                        let affine = stress_scale * tau + mass * c[p];
                        let mv = mass * v[p];
                        let (bx, by) = (base.x as usize, base.y as usize);
                        for (i, wi) in w.iter().enumerate() {
                            for (k, wk) in w.iter().enumerate() {
                                let weight = wi.x * wk.y;
                                let dpos = Vec2::new(i as f32, k as f32) - fx;
                                let idx = (bx + i) * res + (by + k);
                                // SAFETY: see `GridPtr`.
                                unsafe {
                                    *ptr.momentum().add(idx) += weight * (mv + affine * dpos);
                                    *ptr.mass().add(idx) += weight * mass;
                                }
                            }
                        }
                    }
                });
        }
        phase_us[1] += t_p2g.elapsed().as_secs_f64() * 1.0e6;
        let t_grid = std::time::Instant::now();
        let gravity = self.gravity;
        self.grid_p
            .par_chunks_mut(res)
            .zip(self.grid_m.par_chunks(res))
            .enumerate()
            .for_each(|(gx, (row_p, row_m))| {
                for gy in 0..res {
                    let m = row_m[gy];
                    if m <= 0.0 {
                        continue;
                    }
                    let mut vel = row_p[gy] / m + dt * gravity;
                    if (gx < 2 && vel.x < 0.0) || (gx >= res - 2 && vel.x > 0.0) {
                        vel.x = 0.0;
                    }
                    if (gy < 2 && vel.y < 0.0) || (gy >= res - 2 && vel.y > 0.0) {
                        vel.y = 0.0;
                    }
                    row_p[gy] = vel;
                }
            });
        phase_us[2] += t_grid.elapsed().as_secs_f64() * 1.0e6;
        let t_g2p = std::time::Instant::now();
        let grid_p = &self.grid_p;
        let n = self.x.len();
        let min_len = (n / (rayon::current_num_threads() * 2)).max(64);
        let max_speed = self
            .x
            .par_iter_mut()
            .zip(self.v.par_iter_mut())
            .zip(self.c.par_iter_mut())
            .zip(self.f.par_iter_mut())
            .with_min_len(min_len)
            .map(|(((xp, vp), cp), fp)| {
                let base = (*xp - Vec2::splat(0.5)).floor();
                let fx = *xp - base;
                let w = [
                    0.5 * (Vec2::splat(1.5) - fx) * (Vec2::splat(1.5) - fx),
                    Vec2::splat(0.75) - (fx - Vec2::ONE) * (fx - Vec2::ONE),
                    0.5 * (fx - Vec2::splat(0.5)) * (fx - Vec2::splat(0.5)),
                ];
                let (bx, by) = (base.x as usize, base.y as usize);
                let mut nv = Vec2::ZERO;
                let mut b = Mat2::ZERO;
                for (i, wi) in w.iter().enumerate() {
                    for (k, wk) in w.iter().enumerate() {
                        let weight = wi.x * wk.y;
                        let dpos = Vec2::new(i as f32, k as f32) - fx;
                        let gv = grid_p[(bx + i) * res + (by + k)];
                        nv += weight * gv;
                        b += weight * Mat2::from_cols(gv * dpos.x, gv * dpos.y);
                    }
                }
                let cn = 4.0 * b;
                *vp = nv;
                *cp = cn;
                *xp = (*xp + dt * nv).clamp(Vec2::splat(1.0), Vec2::splat(res as f32 - 2.0));
                *fp = exp2x2(dt * cn) * *fp;
                let grad = (cn.x_axis.length_squared() + cn.y_axis.length_squared()).sqrt();
                nv.length() + grad * 1.5 * std::f32::consts::SQRT_2
            })
            .reduce(|| 0.0f32, f32::max);
        phase_us[3] += t_g2p.elapsed().as_secs_f64() * 1.0e6;
        max_speed
    }
}

/// Sense-reversing spin barrier: a persistent thread team synchronises three
/// times per substep, so an OS-level park/unpark (tens of microseconds on
/// this machine, see `audit_rayon_handoff_cost_inside_vs_outside_pool`) per
/// phase would dominate a ~50 us substep.
struct SpinBarrier {
    count: std::sync::atomic::AtomicUsize,
    generation: std::sync::atomic::AtomicUsize,
    n: usize,
}

impl SpinBarrier {
    fn wait(&self) {
        use std::sync::atomic::Ordering;
        let generation = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            while self.generation.load(Ordering::Acquire) == generation {
                std::hint::spin_loop();
            }
        }
    }
}

/// Persistent-team version of `Lean::substep`, `substeps` steps in one go:
/// each of `threads` workers owns a contiguous particle range and a PRIVATE
/// dense grid (exactly one per thread, allocated once -- not one per rayon
/// task), then workers sum the private grids over their own cell range and
/// apply the grid update there, then gather. Three spin barriers per substep.
fn lean_team_run(lean: &mut Lean, dt: f32, substeps: usize, threads: usize) {
    let res = lean.res;
    let cells = res * res;
    let n = lean.x.len();
    let mut private_p = vec![Vec2::ZERO; cells * threads];
    let mut private_m = vec![0.0f32; cells * threads];
    let barrier = SpinBarrier {
        count: std::sync::atomic::AtomicUsize::new(0),
        generation: std::sync::atomic::AtomicUsize::new(0),
        n: threads,
    };
    let stress_scale = -4.0 * dt * lean.vol0;
    let (mass, mu, lambda, gravity) = (lean.mass, lean.mu, lean.lambda, lean.gravity);
    #[derive(Clone, Copy)]
    struct P(*mut Vec2, *mut f32, *mut Vec2, *mut f32);
    unsafe impl Send for P {}
    unsafe impl Sync for P {}
    #[derive(Clone, Copy)]
    struct Q(*mut Vec2, *mut Vec2, *mut Mat2, *mut Mat2);
    unsafe impl Send for Q {}
    unsafe impl Sync for Q {}
    let grids = P(
        private_p.as_mut_ptr(),
        private_m.as_mut_ptr(),
        lean.grid_p.as_mut_ptr(),
        lean.grid_m.as_mut_ptr(),
    );
    let parts = Q(
        lean.x.as_mut_ptr(),
        lean.v.as_mut_ptr(),
        lean.c.as_mut_ptr(),
        lean.f.as_mut_ptr(),
    );
    let barrier = &barrier;
    std::thread::scope(|scope| {
        for t in 0..threads {
            scope.spawn(move || {
                let (grids, parts) = (grids, parts);
                let p_lo = n * t / threads;
                let p_hi = n * (t + 1) / threads;
                let c_lo = cells * t / threads;
                let c_hi = cells * (t + 1) / threads;
                // SAFETY: every write below goes to memory this worker owns
                // exclusively in the current phase (its own private grid,
                // its own cell range, its own particle range); phases are
                // separated by the barrier.
                unsafe {
                    let my_p = grids.0.add(t * cells);
                    let my_m = grids.1.add(t * cells);
                    for _ in 0..substeps {
                        for p in p_lo..p_hi {
                            let xp = *parts.0.add(p);
                            let base = (xp - Vec2::splat(0.5)).floor();
                            let fx = xp - base;
                            let w = [
                                0.5 * (Vec2::splat(1.5) - fx) * (Vec2::splat(1.5) - fx),
                                Vec2::splat(0.75) - (fx - Vec2::ONE) * (fx - Vec2::ONE),
                                0.5 * (fx - Vec2::splat(0.5)) * (fx - Vec2::splat(0.5)),
                            ];
                            let fp = *parts.3.add(p);
                            let j = fp.determinant();
                            let tau = 2.0 * mu * (fp - polar_r(fp)) * fp.transpose()
                                + Mat2::from_diagonal(Vec2::splat(lambda * (j - 1.0) * j));
                            let affine = stress_scale * tau + mass * *parts.2.add(p);
                            let mv = mass * *parts.1.add(p);
                            let (bx, by) = (base.x as usize, base.y as usize);
                            for (i, wi) in w.iter().enumerate() {
                                for (k, wk) in w.iter().enumerate() {
                                    let weight = wi.x * wk.y;
                                    let dpos = Vec2::new(i as f32, k as f32) - fx;
                                    let idx = (bx + i) * res + (by + k);
                                    *my_p.add(idx) += weight * (mv + affine * dpos);
                                    *my_m.add(idx) += weight * mass;
                                }
                            }
                        }
                        barrier.wait();
                        for idx in c_lo..c_hi {
                            let mut m = 0.0f32;
                            let mut mom = Vec2::ZERO;
                            for s in 0..threads {
                                let o = s * cells + idx;
                                m += *grids.1.add(o);
                                mom += *grids.0.add(o);
                                *grids.1.add(o) = 0.0;
                                *grids.0.add(o) = Vec2::ZERO;
                            }
                            *grids.3.add(idx) = m;
                            if m > 0.0 {
                                let (gx, gy) = (idx / res, idx % res);
                                let mut vel = mom / m + dt * gravity;
                                if (gx < 2 && vel.x < 0.0) || (gx >= res - 2 && vel.x > 0.0) {
                                    vel.x = 0.0;
                                }
                                if (gy < 2 && vel.y < 0.0) || (gy >= res - 2 && vel.y > 0.0) {
                                    vel.y = 0.0;
                                }
                                *grids.2.add(idx) = vel;
                            }
                        }
                        barrier.wait();
                        for p in p_lo..p_hi {
                            let xp = *parts.0.add(p);
                            let base = (xp - Vec2::splat(0.5)).floor();
                            let fx = xp - base;
                            let w = [
                                0.5 * (Vec2::splat(1.5) - fx) * (Vec2::splat(1.5) - fx),
                                Vec2::splat(0.75) - (fx - Vec2::ONE) * (fx - Vec2::ONE),
                                0.5 * (fx - Vec2::splat(0.5)) * (fx - Vec2::splat(0.5)),
                            ];
                            let (bx, by) = (base.x as usize, base.y as usize);
                            let mut nv = Vec2::ZERO;
                            let mut b = Mat2::ZERO;
                            for (i, wi) in w.iter().enumerate() {
                                for (k, wk) in w.iter().enumerate() {
                                    let weight = wi.x * wk.y;
                                    let dpos = Vec2::new(i as f32, k as f32) - fx;
                                    let gv = *grids.2.add((bx + i) * res + (by + k));
                                    nv += weight * gv;
                                    b += weight * Mat2::from_cols(gv * dpos.x, gv * dpos.y);
                                }
                            }
                            let cn = 4.0 * b;
                            *parts.1.add(p) = nv;
                            *parts.2.add(p) = cn;
                            *parts.0.add(p) = (xp + dt * nv)
                                .clamp(Vec2::splat(1.0), Vec2::splat(res as f32 - 2.0));
                            *parts.3.add(p) = exp2x2(dt * cn) * *parts.3.add(p);
                        }
                        barrier.wait();
                    }
                }
            });
        }
    });
}

/// Persistent spin-barrier team vs serial lean kernel vs engine, on the
/// corotated demo geometry and on a larger 25 600-particle block.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_lean_persistent_team() {
    let (fresh, mats) = corotated_demo(true);
    let config = *fresh.config();
    let p = fresh.particles();
    let make_demo = || Lean {
        x: p.x.clone(),
        v: p.v.clone(),
        c: vec![Mat2::ZERO; p.len()],
        f: vec![Mat2::IDENTITY; p.len()],
        mass: p.mass[0],
        vol0: p.initial_volume[0],
        lambda: mats[2].lambda,
        mu: mats[2].mu,
        res: config.grid_res,
        grid_m: vec![0.0; config.grid_res * config.grid_res],
        grid_p: vec![Vec2::ZERO; config.grid_res * config.grid_res],
        gravity: config.gravity,
    };
    let sub_dt = config.dt / 12.0;
    let substeps = 4000usize;
    let n = p.len();
    let mut serial = make_demo();
    let t = std::time::Instant::now();
    for _ in 0..substeps {
        serial.substep(sub_dt);
    }
    let us = t.elapsed().as_secs_f64() * 1.0e6 / substeps as f64;
    println!("demo N={n}: serial lean {us:.1}us/substep");
    for threads in [2usize, 3, 4] {
        let mut team = make_demo();
        let t = std::time::Instant::now();
        lean_team_run(&mut team, sub_dt, substeps, threads);
        let us = t.elapsed().as_secs_f64() * 1.0e6 / substeps as f64;
        let gap = serial
            .x
            .iter()
            .zip(&team.x)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0f32, f32::max);
        println!(
            "demo N={n}: persistent team {threads} thr {us:.1}us/substep ({:.1}ns/particle) max gap vs serial {gap:.1e}",
            1000.0 * us / n as f64
        );
    }
    // Larger block (the 80x80 case of the scaling test).
    let grid = 128usize;
    let cfg = SimConfig {
        min_dt: 1.0e-8,
        max_substeps_per_step: 1_000_000,
        ..SimConfig::earth(grid, 0.01, 0.0005)
    };
    let props = Elastic {
        e_pa: 2.0e6,
        nu: 0.3,
        rho_kg_m3: 1000.0,
    };
    let mut big = Simulation::new(
        cfg,
        block(&cfg, Vec2::new(64.0, 44.0), IVec2::splat(80), 0).mass_from(&props, &cfg),
    )
    .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &cfg)));
    apply_exact_lattice_volume(&mut big, 0.5, &[0]);
    let mat = CorotatedMaterial::from_physical(&props, &cfg);
    let bp = big.particles();
    let make_big = || Lean {
        x: bp.x.clone(),
        v: bp.v.clone(),
        c: vec![Mat2::ZERO; bp.len()],
        f: vec![Mat2::IDENTITY; bp.len()],
        mass: bp.mass[0],
        vol0: bp.initial_volume[0],
        lambda: mat.lambda,
        mu: mat.mu,
        res: grid,
        grid_m: vec![0.0; grid * grid],
        grid_p: vec![Vec2::ZERO; grid * grid],
        gravity: cfg.gravity,
    };
    let nb = bp.len();
    let steps = 300usize;
    let dt_big = 1.8e-5;
    let mut serial = make_big();
    let t = std::time::Instant::now();
    for _ in 0..steps {
        serial.substep(dt_big);
    }
    let us = t.elapsed().as_secs_f64() * 1.0e6 / steps as f64;
    println!(
        "big N={nb}: serial lean {us:.1}us/substep ({:.1}ns/particle)",
        1000.0 * us / nb as f64
    );
    for threads in [2usize, 4] {
        let mut team = make_big();
        let t = std::time::Instant::now();
        lean_team_run(&mut team, dt_big, steps, threads);
        let us = t.elapsed().as_secs_f64() * 1.0e6 / steps as f64;
        println!(
            "big N={nb}: persistent team {threads} thr {us:.1}us/substep ({:.1}ns/particle)",
            1000.0 * us / nb as f64
        );
    }
}

/// Engine vs lean kernel on the EXACT corotated-demo geometry (3 blocks,
/// 1200 particles, 64x64 grid), same substep dt, 1 and N threads.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_lean_vs_engine_on_corotated_demo() {
    let (mut sim, mats) = corotated_demo(true);
    let config = *sim.config();
    let n = sim.particles().len();
    // Engine: per-substep cost over 200 frames.
    let mut cost = Cost::default();
    for _ in 0..200 {
        let wall = step_timed(&mut sim);
        cost.add_frame(&sim, wall);
    }
    cost.print("engine (exact V0)", n, config.dt);
    // Lean kernels start from a fresh copy of the same spawn. The three
    // blocks have different stiffness; use the stiffest for all (cost only).
    let (fresh, _) = corotated_demo(true);
    let p = fresh.particles();
    let make = || Lean {
        x: p.x.clone(),
        v: p.v.clone(),
        c: vec![Mat2::ZERO; p.len()],
        f: vec![Mat2::IDENTITY; p.len()],
        mass: p.mass[0],
        vol0: p.initial_volume[0],
        lambda: mats[2].lambda,
        mu: mats[2].mu,
        res: config.grid_res,
        grid_m: vec![0.0; config.grid_res * config.grid_res],
        grid_p: vec![Vec2::ZERO; config.grid_res * config.grid_res],
        gravity: config.gravity,
    };
    let sub_dt = config.dt / 12.0;
    let substeps = 4000usize;
    let mut serial = make();
    let t = std::time::Instant::now();
    for _ in 0..substeps {
        serial.substep(sub_dt);
    }
    let us = t.elapsed().as_secs_f64() * 1.0e6 / substeps as f64;
    println!(
        "lean dense serial                  N={n:>6} {us:>7.1}us/substep {:>6.1}ns/(particle*substep)",
        1000.0 * us / n as f64
    );
    let mut par = make();
    let (mut order, mut offsets) = (Vec::new(), Vec::new());
    let mut phase_us = [0.0f64; 4];
    let t = std::time::Instant::now();
    for _ in 0..substeps {
        par.substep_parallel(sub_dt, &mut order, &mut offsets, &mut phase_us);
    }
    println!(
        "  coloured-parallel phases per substep: sort+clear {:.1}us  p2g {:.1}us  grid {:.1}us  g2p {:.1}us",
        phase_us[0] / substeps as f64,
        phase_us[1] / substeps as f64,
        phase_us[2] / substeps as f64,
        phase_us[3] / substeps as f64
    );
    let us = t.elapsed().as_secs_f64() * 1.0e6 / substeps as f64;
    let drift = serial
        .x
        .iter()
        .zip(&par.x)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0f32, f32::max);
    println!(
        "lean dense coloured-parallel ({} thr) N={n:>6} {us:>7.1}us/substep {:>6.1}ns/(particle*substep)  (max position gap vs serial {drift:.2e} cells)",
        rayon::current_num_threads(),
        1000.0 * us / n as f64
    );
    // Same parallel kernel, but issued from INSIDE the pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .build()
        .expect("pool");
    let mut par_in = make();
    let mut phase_in = [0.0f64; 4];
    let us = pool.install(|| {
        let t = std::time::Instant::now();
        for _ in 0..substeps {
            par_in.substep_parallel(sub_dt, &mut order, &mut offsets, &mut phase_in);
        }
        t.elapsed().as_secs_f64() * 1.0e6 / substeps as f64
    });
    println!(
        "  inside pool.install phases per substep: sort+clear {:.1}us  p2g {:.1}us  grid {:.1}us  g2p {:.1}us",
        phase_in[0] / substeps as f64,
        phase_in[1] / substeps as f64,
        phase_in[2] / substeps as f64,
        phase_in[3] / substeps as f64
    );
    println!(
        "lean dense coloured-parallel INSIDE pool ({} thr) N={n:>6} {us:>7.1}us/substep {:>6.1}ns/(particle*substep)",
        rayon::current_num_threads(),
        1000.0 * us / n as f64
    );
}

/// Reproducibility: is the engine bit-identical run to run, and across
/// thread counts? Same question for the two parallel reference kernels:
/// region colouring (every node written by one block per colour, in a fixed
/// particle order) should be independent of the thread count; per-thread
/// private grids reduced in a fixed order should be deterministic for a
/// FIXED thread count only.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_determinism_run_to_run_and_across_thread_counts() {
    let max_gap = |a: &[Vec2], b: &[Vec2]| {
        a.iter()
            .zip(b)
            .map(|(p, q)| (*p - *q).length())
            .fold(0.0f32, f32::max)
    };
    let engine_run = |threads: usize| -> Vec<Vec2> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            let (mut sim, _) = corotated_demo(true);
            for _ in 0..300 {
                sim.step();
            }
            sim.particles().x.clone()
        })
    };
    let e8a = engine_run(8);
    let e8b = engine_run(8);
    let e1 = engine_run(1);
    let e3 = engine_run(3);
    println!(
        "engine corotated demo, 300 frames (through landing): 8 thr vs 8 thr gap {:.3e} cells | 8 thr vs 1 thr {:.3e} | 8 thr vs 3 thr {:.3e}",
        max_gap(&e8a, &e8b),
        max_gap(&e8a, &e1),
        max_gap(&e8a, &e3)
    );

    let (fresh, mats) = corotated_demo(true);
    let config = *fresh.config();
    let p = fresh.particles();
    let make = || Lean {
        x: p.x.clone(),
        v: p.v.clone(),
        c: vec![Mat2::ZERO; p.len()],
        f: vec![Mat2::IDENTITY; p.len()],
        mass: p.mass[0],
        vol0: p.initial_volume[0],
        lambda: mats[2].lambda,
        mu: mats[2].mu,
        res: config.grid_res,
        grid_m: vec![0.0; config.grid_res * config.grid_res],
        grid_p: vec![Vec2::ZERO; config.grid_res * config.grid_res],
        gravity: config.gravity,
    };
    let sub_dt = config.dt / 12.0;
    let steps = 3000usize;
    let coloured = |threads: usize| -> Vec<Vec2> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        let mut lean = make();
        let (mut order, mut offsets) = (Vec::new(), Vec::new());
        let mut phases = [0.0f64; 4];
        pool.install(|| {
            for _ in 0..steps {
                lean.substep_parallel(sub_dt, &mut order, &mut offsets, &mut phases);
            }
        });
        lean.x
    };
    let c1 = coloured(1);
    let c8 = coloured(8);
    let c8b = coloured(8);
    println!(
        "coloured kernel, {steps} substeps: 1 thr vs 8 thr gap {:.3e} | 8 vs 8 {:.3e}",
        max_gap(&c1, &c8),
        max_gap(&c8, &c8b)
    );
    let team = |threads: usize| -> Vec<Vec2> {
        let mut lean = make();
        lean_team_run(&mut lean, sub_dt, steps, threads);
        lean.x
    };
    let t2a = team(2);
    let t2b = team(2);
    let t3 = team(3);
    let mut serial = make();
    for _ in 0..steps {
        serial.substep(sub_dt);
    }
    println!(
        "private-grid team, {steps} substeps: 2 thr vs 2 thr gap {:.3e} | 2 thr vs 3 thr {:.3e} | 2 thr vs serial {:.3e}",
        max_gap(&t2a, &t2b),
        max_gap(&t2a, &t3),
        max_gap(&t2a, &serial.x)
    );
}

/// Physics check for the exact-lattice-volume fix, on the SAME analytic
/// scene `physics_correctness::self_weight_strain_is_spacing_independent`
/// uses (NeoHookean column resting on the floor, rho*g*h/2E). The expected
/// ratio for an exact solver is NOT 1.0 but slightly above it: the column
/// body extends from the bottom particle's centre to box_min+10 cells while
/// the measured height h0 stops at the top particle's centre, so the top
/// particle's sag relative to the bottom one is (rho g/E)*int_0^h0 (H-y) dy
/// with H = h0 + spacing, i.e. (H*h0 - h0^2/2)/(h0^2/2) of the analytic --
/// times (1 - nu^2) in plane strain.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_exact_volume_self_weight_analytic_check() {
    use emerge::NeoHookeanMaterial;
    const E_PA: f32 = 1.0e5;
    const NU: f32 = 0.2;
    fn ratio(spacing: f32, rho: f32, exact: bool) -> (f32, f32) {
        let config = SimConfig {
            boundary_thickness: 3,
            max_substeps_per_step: 500,
            ..SimConfig::earth(64, 0.01, 0.005)
        };
        let spawn = SpawnRegion {
            spacing,
            box_size: IVec2::new(6, 10),
            box_center: Vec2::new(32.0, 8.0),
            material_id: 0,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let (lambda, mu) = config.lame_from_si_physical_cfg(E_PA, NU, rho);
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(NeoHookeanMaterial::new(lambda, mu)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        if exact {
            apply_exact_lattice_volume(&mut sim, spacing, &[0]);
        }
        let height = |s: &Simulation| {
            let p = s.particles();
            p.x.iter().map(|x| x.y).fold(f32::MIN, f32::max)
                - p.x.iter().map(|x| x.y).fold(f32::MAX, f32::min)
        };
        let h0 = height(&sim);
        let analytic = rho * 9.81 * (h0 * config.dx_meters) / (2.0 * E_PA);
        let big_h = h0 + spacing;
        let expected = (big_h * h0 - 0.5 * h0 * h0) / (0.5 * h0 * h0) * (1.0 - NU * NU);
        for _ in 0..400 {
            sim.step();
        }
        let (mut acc, mut n) = (0.0f64, 0);
        for _ in 0..400 {
            sim.step();
            acc += ((h0 - height(&sim)) / h0) as f64;
            n += 1;
        }
        ((acc / n as f64) as f32 / analytic, expected)
    }
    for spacing in [0.25f32, 0.5, 1.0] {
        let (base, expected) = ratio(spacing, 1000.0, false);
        let (exact, _) = ratio(spacing, 1000.0, true);
        println!(
            "spacing={spacing:<5} strain/analytic: kernel-estimate V0 {base:.3}x | exact lattice V0 {exact:.3}x | exact-solver expectation {expected:.3}x"
        );
    }
}

/// Is the per-substep cost dominated by rayon's cross-thread hand-off?
/// A `par_iter` issued from a thread OUTSIDE the pool is injected and the
/// caller blocks on an OS latch until a worker finishes it; issued from
/// INSIDE the pool (`ThreadPool::install`), the issuing worker starts on
/// the work itself. Same engine code, same scene, same substeps: only the
/// thread that calls `step()` changes.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_rayon_handoff_cost_inside_vs_outside_pool() {
    let threads = rayon::current_num_threads();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool");

    let (mut outside, _) = corotated_demo(true);
    let n = outside.particles().len();
    let dt = outside.config().dt;
    let mut cost = Cost::default();
    for _ in 0..200 {
        let wall = step_timed(&mut outside);
        cost.add_frame(&outside, wall);
    }
    cost.print("engine step() from main thread", n, dt);

    let (mut inside, _) = corotated_demo(true);
    let mut cost = Cost::default();
    pool.install(|| {
        for _ in 0..200 {
            let wall = step_timed(&mut inside);
            cost.add_frame(&inside, wall);
        }
    });
    cost.print("engine step() inside pool.install", n, dt);
    let gap = outside
        .particles()
        .x
        .iter()
        .zip(&inside.particles().x)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0f32, f32::max);
    println!("  max position gap between the two runs: {gap:.2e} cells");
}

#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_lean_dense_reference_kernel() {
    let frame_dt = 0.0005;
    for side in [10i32, 20, 40, 80] {
        // Same scene as audit_particle_count_scaling, stepped by the lean
        // kernel at the SAME substep dt the engine chooses for it.
        let mut sim = {
            let grid = 128usize;
            let config = SimConfig {
                min_dt: 1.0e-8,
                max_substeps_per_step: 1_000_000,
                material_cfl_coefficient: 0.5,
                ..SimConfig::earth(grid, 0.01, frame_dt)
            };
            let props = Elastic {
                e_pa: 2.0e6,
                nu: 0.3,
                rho_kg_m3: 1000.0,
            };
            let spawn = block(
                &config,
                Vec2::new(64.0, 4.0 + side as f32 * 0.5),
                IVec2::splat(side),
                0,
            )
            .mass_from(&props, &config);
            let mat = CorotatedMaterial::from_physical(&props, &config);
            Simulation::new(config, spawn)
                .with_default_material(Box::new(mat))
                .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        };
        apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
        let config = *sim.config();
        let props = Elastic {
            e_pa: 2.0e6,
            nu: 0.3,
            rho_kg_m3: 1000.0,
        };
        let mat = CorotatedMaterial::from_physical(&props, &config);
        let p = sim.particles();
        let mut lean = Lean {
            x: p.x.clone(),
            v: p.v.clone(),
            c: vec![Mat2::ZERO; p.len()],
            f: vec![Mat2::IDENTITY; p.len()],
            mass: p.mass[0],
            vol0: p.initial_volume[0],
            lambda: mat.lambda,
            mu: mat.mu,
            res: config.grid_res,
            grid_m: vec![0.0; config.grid_res * config.grid_res],
            grid_p: vec![Vec2::ZERO; config.grid_res * config.grid_res],
            gravity: config.gravity,
        };
        // Engine dt for this scene: take one real step to read it.
        sim.step();
        let sub_dt = sim.diagnostics_snapshot().effective_dt;
        let substeps = 2000usize;
        let t = std::time::Instant::now();
        let mut ms = 0.0f32;
        for _ in 0..substeps {
            ms = ms.max(lean.substep(sub_dt));
        }
        let us = t.elapsed().as_secs_f64() * 1.0e6;
        let n = lean.x.len();
        let centroid = lean.x.iter().copied().sum::<Vec2>() / n as f32;
        println!(
            "lean dense 1-thread block {side}x{side}: N={n:>6} {:>7.1}us/substep {:>6.1}ns/(particle*substep) \
             (sub_dt={sub_dt:.3e}, max speed {ms:.1} cells/s, centroid y {:.2})",
            us / substeps as f64,
            1000.0 * us / substeps as f64 / n as f64,
            centroid.y
        );
    }
}

// ── pass B: pressure projection right-hand side on a free-falling droplet ──

/// Divergence of the grid velocity at `pos`, central differences over
/// `2h`. `extrapolate = false` reproduces `Grid::project_fluid_incompressibility`
/// exactly (a missing neighbour reads as velocity zero); `true` replaces a
/// missing neighbour by the node's own velocity (zero-gradient extension).
fn central_divergence(grid: &emerge::Grid, pos: IVec2, h: f32, extrapolate: bool) -> f32 {
    let own = grid.velocity_at(pos);
    let read = |offset: IVec2| {
        let q = pos + offset;
        if extrapolate && grid.mass_at(q) <= 0.0 {
            own
        } else {
            grid.velocity_at(q)
        }
    };
    (read(IVec2::X).x - read(-IVec2::X).x) / (2.0 * h)
        + (read(IVec2::Y).y - read(-IVec2::Y).y) / (2.0 * h)
}

/// Pass B: the scene of `scratch_falling_droplet_pressure_projection_check.rs`
/// (airborne water blob, no wall anywhere near) advanced ONE substep with the
/// projection off, then the projection's right-hand side is rebuilt from the
/// grid. A body in uniform free fall has zero divergence; anything the
/// projection sees is manufactured by how it reads missing nodes.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_pressure_rhs_fake_divergence_on_free_fall() {
    const GRID: usize = 64;
    const DT: f32 = 3.0e-4; // one substep
    for iterations in [0u32, 1] {
        let config = SimConfig {
            min_dt: 1.0e-5,
            max_substeps_per_step: 400,
            material_cfl_coefficient: 0.1,
            cfl_include_affine_speed: false,
            fluid_pressure_iterations: iterations,
            fluid_near_wall_cfl_scale: 20.0,
            fluid_near_wall_compression_threshold: 0.0,
            ..SimConfig::earth(GRID, 0.01, DT)
        };
        const SPACING: f32 = 0.6;
        let spawn = SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(7, 7),
            box_center: Vec2::new(32.0, 32.0),
            material_id: 0,
            initial_velocity_scale: 0.0,
            precompute_initial_volumes: true,
            mass_override: Some(0.1 * SPACING * SPACING),
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(emerge::NewtonianFluidMaterial::low_viscosity(
                0.1, 0.0,
            )))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        sim.step();
        let substeps = sim.diagnostics_snapshot().substeps_last_step;
        let h = config.grid_cell_size;
        let grid = sim.grid();

        let (mut raw_max, mut raw_at_edge, mut ext_max) = (0.0f32, false, 0.0f32);
        let mut v_sum = Vec2::ZERO;
        let mut nodes = 0usize;
        for x in 16..48 {
            for y in 16..48 {
                let pos = IVec2::new(x, y);
                if grid.mass_at(pos) <= 0.0 {
                    continue;
                }
                nodes += 1;
                v_sum += grid.velocity_at(pos);
                let edge = [IVec2::X, -IVec2::X, IVec2::Y, -IVec2::Y]
                    .iter()
                    .any(|&o| grid.mass_at(pos + o) <= 0.0);
                let raw = central_divergence(grid, pos, h, false).abs();
                if raw > raw_max {
                    raw_max = raw;
                    raw_at_edge = edge;
                }
                ext_max = ext_max.max(central_divergence(grid, pos, h, true).abs());
            }
        }
        let v_mean = v_sum / nodes.max(1) as f32;
        let particle_vy =
            sim.particles().v.iter().map(|v| v.y).sum::<f32>() / sim.particles().len() as f32;
        let p = sim.particles();
        let (j_lo, j_hi) = (0..p.len()).fold((f32::MAX, f32::MIN), |(lo, hi), i| {
            let j = p.deformation_gradient[i].determinant();
            (lo.min(j), hi.max(j))
        });
        println!(
            "projection passes {iterations}: {substeps} substep(s), {nodes} grid nodes, mean v_y \
             {:.4} cells/s (g*dt = {:.4})",
            v_mean.y,
            -config.gravity.y.abs() * DT
        );
        println!(
            "  divergence as the projection reads it: max |div| {raw_max:.3} 1/s (at a surface \
             node: {raw_at_edge}); predicted |v|/(2h) = {:.3}",
            v_mean.length() / (2.0 * h)
        );
        println!(
            "  same field, missing neighbour = own velocity: max |div| {ext_max:.2e} 1/s; \
             particle J after the substep [{j_lo:.4}, {j_hi:.4}]; particle mean v_y {particle_vy:.4}"
        );
    }
}

/// Pass B follow-up: the 2026-09-17 attempt swapped `velocity_at` for
/// `velocity_at_or_extrapolated` in the projection's RHS and was reverted
/// (J still clamped). Its code was not kept; this rebuilds the RHS three ways
/// over the WHOLE padded box, exactly where `pressure.rs` evaluates it, and
/// reports fluid cells (mass > 0) and air cells (mass = 0) separately:
///   A. current code: missing neighbour = 0;
///   B. the 09-17 swap as its note and the API imply: absent neighbour (not in
///      the hash map) = centre's own `velocity_at` value, gravity already
///      integrated so no `+ g dt` (its own double-count fix);
///   C. RHS only on fluid cells, massless neighbour = centre velocity.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_pressure_rhs_where_the_0917_swap_puts_the_source() {
    const GRID: usize = 64;
    const DT: f32 = 3.0e-4;
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 0,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    const SPACING: f32 = 0.6;
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(7, 7),
        box_center: Vec2::new(32.0, 32.0),
        material_id: 0,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(0.1 * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(emerge::NewtonianFluidMaterial::low_viscosity(
            0.1, 0.0,
        )))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    sim.step();
    let h = config.grid_cell_size;
    let grid = sim.grid();

    // "dirty" = present in the hash map, as `pressure.rs` uses it.
    let (mut lo, mut hi) = (IVec2::splat(i32::MAX), IVec2::splat(i32::MIN));
    let (mut present, mut present_massless) = (0usize, 0usize);
    for x in 0..GRID as i32 {
        for y in 0..GRID as i32 {
            let p = IVec2::new(x, y);
            if grid.is_extrapolated(p) {
                continue;
            }
            present += 1;
            if grid.mass_at(p) <= 0.0 {
                present_massless += 1;
            }
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    let (lo, hi) = (lo - IVec2::splat(4), hi + IVec2::splat(4));

    let div = |pos: IVec2, read: &dyn Fn(IVec2) -> Vec2| {
        (read(pos + IVec2::X).x - read(pos - IVec2::X).x) / (2.0 * h)
            + (read(pos + IVec2::Y).y - read(pos - IVec2::Y).y) / (2.0 * h)
    };
    let mut max = [[0.0f32; 2]; 3]; // [variant][fluid, air]
    let mut air_cells_hit = [0usize; 3];
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            let pos = IVec2::new(x, y);
            let fluid = grid.mass_at(pos) > 0.0;
            let own = grid.velocity_at(pos);
            let a = div(pos, &|q| grid.velocity_at(q));
            let b = div(pos, &|q| {
                grid.velocity_at_or_extrapolated(q, own, Vec2::ZERO, DT, config.boundary_thickness)
            });
            let c = if fluid {
                div(pos, &|q| if grid.mass_at(q) > 0.0 { grid.velocity_at(q) } else { own })
            } else {
                0.0
            };
            for (k, v) in [a, b, c].into_iter().enumerate() {
                let slot = usize::from(!fluid);
                max[k][slot] = max[k][slot].max(v.abs());
                if !fluid && v.abs() > 1e-6 {
                    air_cells_hit[k] += 1;
                }
            }
        }
    }
    println!(
        "padded box {}x{}; {present} nodes in the hash map, {present_massless} of them massless",
        hi.x - lo.x + 1,
        hi.y - lo.y + 1
    );
    for (k, name) in ["A current", "B 09-17 swap", "C fluid-only"].iter().enumerate() {
        println!(
            "  {name:<14} max |div| fluid cells {:.3}, air cells {:.3} ({} air cells nonzero)",
            max[k][0], max[k][1], air_cells_hit[k]
        );
    }
}

/// Pass C, time-convergence lead: the same vibrating free block as
/// `audit_temporal_order_of_accuracy`, energy kept after the same physical time
/// at decreasing dt, with ASFLIP off (plain APIC) and on. APIC's energy loss
/// grows with the number of transfers; a FLIP-type increment should make the
/// loss per transfer shrink with dt.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_time_convergence_asflip() {
    let run = |courant: f32, asflip: f32| -> f64 {
        let mut config = SimConfig {
            min_dt: 1.0e-9,
            max_substeps_per_step: 4,
            asflip_blend: asflip,
            ..SimConfig::earth(64, 0.01, 1.0e-5)
        };
        config.gravity = Vec2::ZERO;
        let props = Elastic {
            e_pa: 1.0e6,
            nu: 0.3,
            rho_kg_m3: 1000.0,
        };
        let mat = CorotatedMaterial::from_physical(&props, &config);
        let spawn =
            block(&config, Vec2::new(32.0, 32.0), IVec2::new(10, 10), 0).mass_from(&props, &config);
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        apply_exact_lattice_volume(&mut sim, 0.5, &[0]);
        {
            let p = sim.particles_mut();
            let n = p.len() as f32;
            let centre = p.x.iter().copied().sum::<Vec2>() / n;
            for i in 0..p.len() {
                let r = p.x[i] - centre;
                p.v[i] = 3.0 * r + 2.0 * Vec2::new(-r.y, r.x);
            }
        }
        let c = ((mat.lambda + 2.0 * mat.mu) / 1.0).sqrt();
        let dt = courant / c;
        let t_end = 0.4 * 256.0 / c;
        let steps = (t_end / dt).round() as usize;
        {
            let cfg = sim.config_mut();
            cfg.adaptive_timestep = false;
            cfg.dt = dt;
            cfg.min_dt = dt;
        }
        let b0 = budget(&sim, mat.lambda, mat.mu);
        let e0 = b0.kinetic + b0.elastic;
        for _ in 0..steps {
            sim.step();
        }
        let b1 = budget(&sim, mat.lambda, mat.mu);
        (b1.kinetic + b1.elastic) / e0
    };
    for asflip in [0.0f32, 0.5, 0.97] {
        let kept: Vec<String> = [0.4f32, 0.1, 0.025]
            .iter()
            .map(|&c| format!("{:.4}", run(c, asflip)))
            .collect();
        println!(
            "asflip_blend {asflip:.2}: energy kept after the same time at f = 0.4 / 0.1 / 0.025 (256 / 1024 / 4096 steps): {}",
            kept.join(" / ")
        );
    }
}
