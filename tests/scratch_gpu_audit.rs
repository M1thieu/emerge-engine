//! Scratch GPU audit probe (2026-09-21, pass B). Untracked, never committed.
//! Prints measurements only.
//!
//!   cargo test --profile quick --features gpu --test scratch_gpu_audit -- \
//!       --ignored --nocapture --test-threads=1
#![cfg(feature = "gpu")]
extern crate emerge_engine as emerge;

use emerge::gpu::GpuSimulation;
use emerge::{
    CorotatedMaterial, DruckerPragerMaterial, Elastic, FromSI, GranularProps, MaterialModel,
    MaterialRegistry, SimConfig, Simulation, SlipBoundary, SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use pollster::block_on;

fn config() -> SimConfig {
    SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 400,
        ..SimConfig::earth(64, 0.01, 0.002)
    }
}

fn spawn(cfg: &SimConfig) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(10, 10),
        box_center: Vec2::new(32.0, 45.0),
        material_id: 0,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(cfg)
    }
}

fn material(sand: bool, cfg: &SimConfig) -> Box<dyn MaterialModel> {
    let elastic = Elastic {
        e_pa: 15.0e6,
        nu: 0.3,
        rho_kg_m3: 1600.0,
    };
    if sand {
        Box::new(DruckerPragerMaterial::from_physical(
            &GranularProps {
                elastic,
                friction_angle_deg: 30.0,
                dilatancy_angle_deg: 0.0,
            },
            cfg,
        ))
    } else {
        Box::new(CorotatedMaterial::from_physical(&elastic, cfg))
    }
}

/// Mean vertical velocity (cells/s) after `frames` frames of free fall.
fn cpu_fall(sand: bool, frames: usize) -> f32 {
    let cfg = config();
    let mut sim = Simulation::new(cfg, spawn(&cfg))
        .with_default_material(material(sand, &cfg))
        .with_boundary(Box::new(SlipBoundary::new(cfg.boundary_thickness)));
    for _ in 0..frames {
        sim.step();
    }
    let p = sim.particles();
    p.v.iter().map(|v| v.y).sum::<f32>() / p.len() as f32
}

fn gpu_fall(sand: bool, frames: usize) -> Option<(f32, usize)> {
    let cfg = config();
    let particles = build_particles(&cfg, spawn(&cfg));
    let registry = MaterialRegistry::with_default(material(sand, &cfg));
    let mut gpu = block_on(GpuSimulation::new(cfg, particles, registry));
    let mut substeps = 0;
    for _ in 0..frames {
        gpu.step_frame();
        substeps += gpu.diagnostics_snapshot().substeps_last_step;
    }
    gpu.sync_particles_blocking();
    let p = gpu.particles();
    if p.is_empty() {
        return None;
    }
    Some((
        p.iter().map(|q| q.v.y).sum::<f32>() / p.len() as f32,
        substeps,
    ))
}

/// Pass B: does the GPU's per-substep `v *= 0.999` for plasticity models
/// (particles_update.wgsl) change free fall? Prediction: a terminal speed of
/// g*dt/0.001 (~0.44 m/s at E = 15 MPa sand's ~45 us substep).
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_gpu_plastic_damping_free_fall() {
    let frames = 50; // 0.1 s
    let g_t = 9.81 / 0.01 * 0.1; // cells/s after 0.1 s of free fall
    for (name, sand) in [("Corotated (control)", false), ("DruckerPrager sand", true)] {
        let cpu = cpu_fall(sand, frames);
        let Some((gpu, n)) = gpu_fall(sand, frames) else {
            println!("{name}: GPU returned no particles");
            continue;
        };
        // v_{k+1} = 0.999 (v_k + g dt): closed form after n equal substeps.
        let dt = 0.1 / n as f32;
        let damped = 9.81 / 0.01 * dt * 0.999 * (1.0 - 0.999f32.powi(n as i32)) / 0.001;
        println!(
            "{name:<22} mean v_y after 0.1 s: analytic {:.1} cells/s | CPU {cpu:.1} | GPU {gpu:.1}              ({n} GPU substeps; with v *= 0.999 per substep: {:.1})",
            -g_t, -damped
        );
    }
}

fn bingham(shear_modulus_pa: f32, cfg: &SimConfig) -> Box<dyn MaterialModel> {
    let c_ref = 10.0 * (2.0 * 9.81 * 0.1f32).sqrt();
    Box::new(emerge::BinghamFluidMaterial::from_physical(
        &emerge::BinghamProps {
            rho_kg_m3: 1000.0,
            eta_pa_s: 0.5,
            bulk_modulus_pa: 1000.0 * c_ref * c_ref,
            yield_stress_pa: 400.0,
            shear_modulus_pa,
        },
        cfg,
    ))
}

/// Pass B: a material with `needs_cpu_update()` (Bingham's elastoviscoplastic
/// branch) makes `step_frame` re-upload the CPU copy every frame, and that
/// copy is refreshed from an ASYNC readback begun at the end of the previous
/// frame. Free fall is the cleanest witness: no stress, no wall, only time.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_gpu_cpu_update_path_free_fall() {
    let frames = 50; // 0.1 s
    for (name, g_pa) in [
        ("Bingham viscous (control)", 0.0),
        ("Bingham EVP (CPU update)", 8000.0),
    ] {
        let cfg = config();
        let registry = MaterialRegistry::with_default(bingham(g_pa, &cfg));
        let needs_cpu = registry.any_needs_cpu_update();
        let particles = build_particles(&cfg, spawn(&cfg));
        let y0 = particles.iter().map(|q| q.x.y).sum::<f32>() / particles.len() as f32;
        let mut gpu = block_on(GpuSimulation::new(cfg, particles, registry));
        let mut substeps = 0;
        for _ in 0..frames {
            gpu.step_frame();
            substeps += gpu.diagnostics_snapshot().substeps_last_step;
        }
        gpu.sync_particles_blocking();
        let p = gpu.particles();
        let n = p.len().max(1) as f32;
        let v = p.iter().map(|q| q.v.y).sum::<f32>() / n;
        let dy = p.iter().map(|q| q.x.y).sum::<f32>() / n - y0;

        println!(
            "{name:<26} needs_cpu_update {needs_cpu}: GPU v_y {v:.1} cells/s, fall {dy:.2} cells \
             ({substeps} substeps) | analytic v_y -98.1, fall -4.91"
        );
    }
}

/// Pass B: `IdealGasMaterial` has no WGSL arm and, unlike NACC and
/// NoCompression, no constructor guard. A gas block released in vacuum (zero
/// gravity) must expand; mean particle speed after 0.2 ms, CPU vs GPU.
#[test]
#[ignore = "audit probe, prints measurements"]
fn audit_gpu_gas_has_no_pressure() {
    let cfg = SimConfig {
        gravity: Vec2::ZERO,
        min_dt: 1.0e-8,
        max_substeps_per_step: 400,
        ..SimConfig::earth(64, 0.01, 2.0e-4)
    };
    let gas =
        || Box::new(emerge::IdealGasMaterial::air(1.2, 300.0, &cfg)) as Box<dyn MaterialModel>;
    let spawn = SpawnRegion {
        box_center: Vec2::new(32.0, 32.0),
        ..spawn(&cfg)
    };
    let mean_speed = |v: &mut dyn Iterator<Item = Vec2>| {
        let (s, n) = v.fold((0.0f32, 0usize), |(s, n), v| (s + v.length(), n + 1));
        s / n.max(1) as f32
    };

    let mut sim = Simulation::new(cfg, spawn.clone())
        .with_default_material(gas())
        .with_boundary(Box::new(SlipBoundary::new(cfg.boundary_thickness)));
    sim.step();
    let cpu = mean_speed(&mut sim.particles().v.iter().copied());
    let cpu_substeps = sim.diagnostics_snapshot().substeps_last_step;

    let mut gpu = block_on(GpuSimulation::new(
        cfg,
        build_particles(&cfg, spawn),
        MaterialRegistry::with_default(gas()),
    ));
    gpu.step_frame();
    let gpu_substeps = gpu.diagnostics_snapshot().substeps_last_step;
    gpu.sync_particles_blocking();
    let gpu_speed = mean_speed(&mut gpu.particles().iter().map(|p| p.v));
    println!(
        "air block in vacuum after 0.2 ms: mean |v| CPU {cpu:.1} cells/s ({cpu_substeps} substeps) \
         | GPU {gpu_speed:.3} cells/s ({gpu_substeps} substeps)"
    );
}
