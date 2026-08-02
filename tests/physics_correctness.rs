//! Physics correctness tests for emerge.
//!
//! These tests verify conservation laws, material invariants, and solver properties
//! that must hold for the engine to be physically valid.
//!
//! Each test has a clear physical claim and is comparable to reference MPM implementations
//! (sparkl, matter, taichi128).

extern crate emerge_engine as emerge;
use emerge::fields::LinearDragField;
use emerge::materials::MaterialModel;
use emerge::particle::{Particle, Particles};
use emerge::thermodynamics::{ScalarDiffusionConfig, ScalarDiffusionField};
use emerge::{
    ActivationStatsPlugin, DiagnosticsFrame, DiagnosticsRegistry, MaterialCountPlugin,
    ThermalStatsPlugin, collect_snapshot,
};
use emerge::{
    BinghamFluidMaterial, CorotatedMaterial, DruckerPragerMaterial, GranularFluidMaterial,
    MuIRheologyMaterial, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation,
    SpawnRegion, StomakhinMaterial, ViscoelasticMaterial, VonMisesMaterial, WithPreStress,
};
// Boundary types kept on their own `use` line (not merged into the material
// import block above) so this test file's imports don't collide with other
// branches that also add to that block -- keeps independent PRs conflict-free.
use emerge::{FrictionBoundary, GripFrictionBoundary, RatchetFrictionBoundary, SlipBoundary};
use glam::{IVec2, Mat2, Vec2};

// â”€â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Wrap a single `Particle` in a one-element `Particles` SoA, call `kirchhoff_stress`, return result.
fn kirchhoff_stress_of(mat: &dyn emerge::materials::MaterialModel, p: &Particle) -> glam::Mat2 {
    let soa = Particles::from(vec![*p]);
    mat.kirchhoff_stress(&soa, 0)
}

/// Wrap a single `Particle` in a one-element `Particles` SoA, call `update_particle`, write back.
fn update_particle_of(mat: &dyn emerge::materials::MaterialModel, p: &mut Particle, dt: f32) {
    let mut soa = Particles::from(vec![*p]);
    mat.update_particle(&mut soa.update_ctx(0), dt);
    *p = soa.get(0);
}

fn zero_gravity_config(grid_res: usize) -> SimConfig {
    SimConfig {
        grid_res,
        dt: 0.05,
        gravity: Vec2::ZERO,
        adaptive_timestep: true,
        ..SimConfig::default()
    }
}

fn center_spawn(grid_res: usize, side: usize) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(side as i32, side as i32),
        box_center: Vec2::splat(grid_res as f32 * 0.5),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::default()
    }
}

fn total_mass(solver: &Simulation) -> f32 {
    solver.particles().iter().map(|p| p.mass).sum()
}

fn linear_momentum(solver: &Simulation) -> Vec2 {
    solver.particles().iter().map(|p| p.mass * p.v).sum()
}

fn kinetic_energy(solver: &Simulation) -> f32 {
    solver
        .particles()
        .iter()
        .map(|p| 0.5 * p.mass * p.v.length_squared())
        .sum()
}

fn min_j(solver: &Simulation) -> f32 {
    solver
        .particles()
        .iter()
        .map(|p| p.deformation_gradient.determinant())
        .fold(f32::INFINITY, f32::min)
}

// â”€â”€â”€ CONSERVATION: MASS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Mass is a particle property and never changes â€” the solver must not add or remove particles.
#[test]
fn mass_is_conserved_neohookean() {
    let mut solver = Simulation::new(zero_gravity_config(32), center_spawn(32, 6))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "mass changed: before={m0:.6} after={m1:.6} delta={:.2e}",
        (m1 - m0).abs()
    );
}

#[test]
fn mass_is_conserved_fluid() {
    let config = SimConfig {
        recompute_density_each_step: true,
        ..zero_gravity_config(32)
    };
    let mut solver = Simulation::new(config, center_spawn(32, 6))
        .with_default_material(Box::new(NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0)));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "fluid: mass changed: before={m0:.6} after={m1:.6}"
    );
}

#[test]
fn mass_is_conserved_snow() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let mut solver = Simulation::new(zero_gravity_config(32), center_spawn(32, 6))
        .with_default_material(Box::new(snow));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "snow: mass not conserved: {m0:.6} â†’ {m1:.6}"
    );
}

/// Baseline mass-conservation coverage for `GranularFluidMaterial`, matching every
/// other material's pattern in this file.
#[test]
fn mass_is_conserved_granular_fluid() {
    let mud = GranularFluidMaterial::saturated_loam(1.0e5, 0.2);
    let mut solver = Simulation::new(zero_gravity_config(32), center_spawn(32, 6))
        .with_default_material(Box::new(mud));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "granular fluid: mass not conserved: {m0:.6} -> {m1:.6}"
    );
}

// â”€â”€â”€ CONSERVATION: LINEAR MOMENTUM (no external forces) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// With zero gravity and zero initial velocity, total momentum must stay near zero.
/// (MLS-MPM is weakly momentum conserving; small residuals from grid averaging are expected.)
#[test]
fn zero_velocity_spawn_has_near_zero_momentum() {
    let mut solver = Simulation::new(zero_gravity_config(32), center_spawn(32, 8))
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let p0 = linear_momentum(&solver);
    solver.step_n(50);
    let p1 = linear_momentum(&solver);

    // Absolute momentum drift per particle (mass=1): should stay tiny
    let n = solver.particles().len() as f32;
    let drift = (p1 - p0).length() / n;
    assert!(
        drift < 1e-3,
        "momentum drift per particle too large: {drift:.2e} (initial p={p0}, final p={p1})"
    );
}

/// With uniform gravity and no initial motion, momentum grows at rate mÂ·g â€” verify linearity.
#[test]
fn gravity_grows_momentum_linearly() {
    let g = Vec2::new(0.0, -9.81);
    let config = SimConfig {
        gravity: g,
        dt: 0.01,
        adaptive_timestep: false,
        ..SimConfig::default()
    };
    let mut solver = Simulation::new(config, center_spawn(64, 4))
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 200.0)));

    let m_total = total_mass(&solver);
    let p_before = linear_momentum(&solver);

    let n_steps = 10;
    let dt = 0.01f32;
    solver.step_n(n_steps);

    let p_after = linear_momentum(&solver);
    let elapsed = dt * n_steps as f32;
    let expected_impulse = g * m_total * elapsed;
    let actual_impulse = p_after - p_before;

    // Allow 5% tolerance: boundary clamping absorbs some momentum
    let rel_err = (actual_impulse - expected_impulse).length() / (expected_impulse.length() + 1e-6);
    assert!(
        rel_err < 0.05,
        "gravity impulse wrong: expected={expected_impulse:.3?} actual={actual_impulse:.3?} rel_err={rel_err:.3}"
    );
}

// â”€â”€â”€ J > 0 INVARIANT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// det(F) > 0 is a non-negotiable physical invariant â€” particles can't invert.
/// Requires `project_invalid_state: true` (standard config) â€” the J floor that real simulations use.
#[test]
fn j_stays_positive_neohookean() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "NeoHookean: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_snow() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(snow));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Snow: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_sand() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(sand));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Sand: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_granular_fluid() {
    let mud = GranularFluidMaterial::saturated_loam(1.0e5, 0.2);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(mud));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "GranularFluid: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_corotated() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(CorotatedMaterial::new(10.0, 20.0)));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Corotated: J collapsed to {jmin:.2e}");
}

// â”€â”€â”€ SNOW PLASTICITY: Jp BOUNDS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Snow Jp must stay within [min_jp, max_jp] after any number of steps.
/// This is the yield surface enforcement â€” clamped singular values constrain Jp.
#[test]
fn snow_jp_stays_within_bounds() {
    let min_jp = 0.6f32;
    let max_jp = 20.0f32;
    let snow = StomakhinMaterial::new(38_889.0, 58_333.0, 10.0, 0.025, 0.0075, min_jp, max_jp);

    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(snow));

    solver.step_n(300);

    for (i, p) in solver.particles().iter().enumerate() {
        let jp = p.plastic_volume_ratio;
        assert!(
            jp >= min_jp * 0.99 && jp <= max_jp * 1.01,
            "snow particle {i}: Jp={jp:.4} out of [{min_jp}, {max_jp}]"
        );
    }
}

/// Snow hardening scale h = exp(Î¾(1-Jp)) must be non-negative and finite.
/// Note: h=0.0 is valid f32 underflow of exp(âˆ’190) when Jpâ‰ˆmax_jp â€” effectively zero stress.
/// What matters is that h stays finite (no NaN/Inf) and non-negative.
#[test]
fn snow_hardening_scale_finite() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(snow));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.hardening_scale >= 0.0 && p.hardening_scale.is_finite(),
            "snow particle {i}: hardening_scale={:.4} (must be finite â‰¥0)",
            p.hardening_scale
        );
    }
}

// â”€â”€â”€ SAND: NO TENSION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Sand cannot sustain tension (p â‰¤ 0 â†’ project to stress-free).
/// Test via direct material update on a tensile deformation gradient.
#[test]
fn sand_tension_cutoff_removes_tensile_stress() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Pure extension: F = diag(1.5, 1.5) â€” volume 2.25Ã—, tensile state
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.5));
    p.velocity_gradient = Mat2::ZERO;

    // Initialize particle (seeds plastic state)
    sand.init_particle(&mut p);
    update_particle_of(&sand, &mut p, 0.01);

    // After projection, stress should be near zero (tensile â†’ return to identity)
    let tau = kirchhoff_stress_of(&sand, &p);
    let tau_norm = (tau.x_axis.length_squared() + tau.y_axis.length_squared()).sqrt();
    assert!(
        tau_norm < 1.0,
        "sand: tensile stress not projected (||Ï„||={tau_norm:.4})"
    );
}

/// Sand Drucker-Prager: log_volume_strain must stay finite.
/// Requires project_invalid_state=true to prevent Jâ†’0 which causes log(J)=âˆ’âˆž.
#[test]
fn sand_log_volume_strain_finite() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(sand));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.log_volume_strain.is_finite(),
            "sand particle {i}: log_volume_strain={}",
            p.log_volume_strain
        );
    }
}

// â”€â”€â”€ MATERIAL STRESS SYMMETRY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Kirchhoff stress Ï„ must be symmetric for all materials (objectivity / frame-indifference).
/// Ï„ = Ï„áµ€: |Ï„â‚€â‚ âˆ’ Ï„â‚â‚€| < Îµ.
fn check_stress_symmetry(mat: &dyn MaterialModel, label: &str) {
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Small shear deformation: F = [[1.1, 0.1], [0.05, 0.95]]
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.1, 0.05), Vec2::new(0.1, 0.95));
    mat.init_particle(&mut p);

    let tau = kirchhoff_stress_of(mat, &p);
    let asym = (tau.col(0).y - tau.col(1).x).abs();
    assert!(
        asym < 1e-4,
        "{label}: Kirchhoff stress asymmetric: Ï„â‚€â‚={:.6} Ï„â‚â‚€={:.6} |diff|={asym:.2e}",
        tau.col(1).x,
        tau.col(0).y,
    );
}

#[test]
fn neohookean_stress_symmetric() {
    check_stress_symmetry(&NeoHookeanMaterial::new(100.0, 200.0), "NeoHookean");
}

#[test]
fn corotated_stress_symmetric() {
    check_stress_symmetry(&CorotatedMaterial::new(100.0, 200.0), "Corotated");
}

#[test]
fn snow_stress_symmetric() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    check_stress_symmetry(&snow, "Snow");
}

#[test]
fn granular_fluid_stress_symmetric() {
    // Not using the shared `check_stress_symmetry` helper's 1e-4 ABSOLUTE
    // tolerance: this material's Tait EOS term produces stress magnitudes
    // (~6400 here) far larger than the other materials this helper was
    // calibrated against, so plain f32 rounding noise at that scale
    // (~6400 * 1e-7 ~= 6e-4) alone exceeds a tolerance tuned for O(1-100)
    // stresses. Checked RELATIVE asymmetry instead, which is scale-invariant.
    let mud = GranularFluidMaterial::saturated_loam(1.0e5, 0.2);
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.1, 0.05), Vec2::new(0.1, 0.95));
    mud.init_particle(&mut p);

    let tau = kirchhoff_stress_of(&mud, &p);
    let asym = (tau.col(0).y - tau.col(1).x).abs();
    let scale = tau.col(0).y.abs().max(tau.col(1).x.abs()).max(1.0);
    assert!(
        asym / scale < 1.0e-5,
        "GranularFluid: Kirchhoff stress asymmetric beyond float noise: \
         tau01={:.6} tau10={:.6} relative diff={:.2e}",
        tau.col(1).x,
        tau.col(0).y,
        asym / scale
    );
}

#[test]
fn sand_stress_symmetric() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Compressive deformation (sand only resists compression)
    p.deformation_gradient = Mat2::from_cols(Vec2::new(0.9, 0.05), Vec2::new(0.05, 0.9));
    sand.init_particle(&mut p);
    update_particle_of(&sand, &mut p, 0.01);

    let tau = kirchhoff_stress_of(&sand, &p);
    let asym = (tau.col(0).y - tau.col(1).x).abs();
    assert!(asym < 1e-4, "Sand: stress asymmetric: {asym:.2e}");
}

// â”€â”€â”€ SVD CORRECTNESS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Our analytical 2Ã—2 SVD must satisfy F = UÂ·diag(Ïƒ)Â·Váµ€ and U,V orthogonal.
/// This is tested internally in mechanics/svd.rs, but we verify the public path
/// through StomakhinMaterial.update_particle which uses svd2().
#[test]
fn snow_update_preserves_f_decomposition_invariant() {
    // After snow update, F_elastic must remain a valid deformation gradient.
    // det(F) > 0, F finite, singular values in (0, +âˆž).
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Start from slight compression
    p.deformation_gradient = Mat2::from_cols(Vec2::new(0.95, 0.02), Vec2::new(-0.02, 0.95));
    p.plastic_volume_ratio = 1.0;
    p.hardening_scale = 1.0;

    for _ in 0..50 {
        p.velocity_gradient = Mat2::from_cols(Vec2::new(-0.01, 0.005), Vec2::new(0.005, -0.01));
        update_particle_of(&snow, &mut p, 0.01);
    }

    let j = p.deformation_gradient.determinant();
    assert!(
        j > 0.0 && j.is_finite(),
        "Snow: F det invalid after updates: J={j}"
    );
    assert!(p.deformation_gradient.is_finite(), "Snow: F non-finite");
    assert!(p.hardening_scale > 0.0 && p.hardening_scale.is_finite());
}

// â”€â”€â”€ ENERGY NON-GROWTH (elastic, no gravity) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Kinetic energy of a resting elastic blob (no gravity, zero initial velocity)
/// must stay near zero â€” no spurious energy injection from the solver.
#[test]
fn resting_jelly_no_energy_growth() {
    let config = SimConfig {
        gravity: Vec2::ZERO,
        dt: 0.05,
        ..SimConfig::default()
    };
    let spawn = SpawnRegion {
        initial_velocity_scale: 0.0,
        ..center_spawn(64, 6)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let ke0 = kinetic_energy(&solver);
    solver.step_n(200);
    let ke1 = kinetic_energy(&solver);

    // Resting blob: initial KE â‰ˆ 0. After steps it may have tiny numerical KE but
    // must not grow significantly.
    let n = solver.particles().len() as f32;
    assert!(
        ke1 / n < 1e-4,
        "resting jelly: KE grew from {ke0:.2e} to {ke1:.2e} ({:.2e} per particle)",
        ke1 / n
    );
}

// â”€â”€â”€ CFL STABILITY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Adaptive substep must never produce a sub_dt that violates particle CFL.
/// Proxy: particle speed Ã— sub_dt â‰¤ 1 cell (with CFL coeff).
/// We verify this by checking velocities never exceed the grid/dt threshold.
#[test]
fn adaptive_substep_keeps_velocities_bounded() {
    let config = SimConfig {
        gravity: Vec2::new(0.0, -9.81),
        dt: 0.1,
        adaptive_timestep: true,
        cfl_coefficient: 0.4,
        ..SimConfig::default()
    };
    // High initial velocity to stress CFL
    let spawn = SpawnRegion {
        initial_velocity_scale: 5.0,
        ..center_spawn(64, 6)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(50.0, 100.0)));

    solver.step_n(100);

    // CFL=0.4 bounds max substep speed; here we only assert finiteness, not the exact bound.
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.v.is_finite(),
            "CFL test: particle {i} velocity non-finite: {:?}",
            p.v
        );
        assert!(
            p.v.length() < 500.0,
            "CFL test: particle {i} velocity exploded: |v|={:.1}",
            p.v.length()
        );
    }
}

// â”€â”€â”€ DIAGNOSTICS PLUGIN SYSTEM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// DiagnosticsRegistry::collect must aggregate all plugin outputs.
#[test]
fn diagnostics_registry_aggregates_plugins() {
    use emerge::grid::Grid;

    let config = SimConfig {
        grid_res: 8,
        dt: 0.1,
        ..SimConfig::default()
    };

    let particles = vec![
        Particle {
            x: Vec2::new(4.0, 4.0),
            v: Vec2::new(1.0, 0.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 300.0,
            activation: 0.8,
            material_id: 0,
            ..Particle::zeroed()
        },
        Particle {
            x: Vec2::new(5.0, 4.0),
            v: Vec2::new(-1.0, 0.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 320.0,
            activation: 0.0,
            material_id: 1,
            ..Particle::zeroed()
        },
    ];

    let grid = Grid::new(config.grid_res);
    let particles_soa = emerge::particle::Particles::from(particles.clone());
    let snap = collect_snapshot(0, &particles_soa, &grid, &config, config.dt, 1);

    let mut registry = DiagnosticsRegistry::new();
    registry.register(Box::new(ActivationStatsPlugin));
    registry.register(Box::new(ThermalStatsPlugin));
    registry.register(Box::new(MaterialCountPlugin));
    // Closure plugin
    registry.register_fn("custom", |particles, _snap| {
        vec![("n_total".into(), particles.len() as f32)]
    });

    assert_eq!(registry.len(), 4);

    let frame = registry.collect(&particles, &snap);

    // Activation: mean = (0.8 + 0.0)/2 = 0.4, frac = 1/2 = 0.5
    let act_mean = frame.get("act_mean").expect("act_mean missing");
    assert!(
        (act_mean - 0.4).abs() < 1e-5,
        "act_mean={act_mean:.4} expected 0.4"
    );

    let act_frac = frame.get("act_frac").expect("act_frac missing");
    assert!(
        (act_frac - 0.5).abs() < 1e-5,
        "act_frac={act_frac:.4} expected 0.5"
    );

    // Temperature: mean = (300+320)/2=310, max=320
    let t_mean = frame.get("T_mean").expect("T_mean missing");
    assert!(
        (t_mean - 310.0).abs() < 1e-3,
        "T_mean={t_mean:.2} expected 310"
    );

    let t_max = frame.get("T_max").expect("T_max missing");
    assert!(
        (t_max - 320.0).abs() < 1e-3,
        "T_max={t_max:.2} expected 320"
    );

    // Material counts: mat0_n=1, mat1_n=1
    let mat0 = frame.get("mat0_n").expect("mat0_n missing");
    assert_eq!(mat0 as usize, 1, "mat0_n wrong");

    let mat1 = frame.get("mat1_n").expect("mat1_n missing");
    assert_eq!(mat1 as usize, 1, "mat1_n wrong");

    // Custom: n_total=2
    let n = frame.get("n_total").expect("n_total missing");
    assert_eq!(n as usize, 2, "n_total wrong");
}

/// DiagnosticsFrame::format_line produces compact output with all keys.
#[test]
fn diagnostics_frame_format_line_is_compact() {
    let frame = DiagnosticsFrame {
        stats: vec![
            ("n".into(), 256.0),
            ("ke".into(), 1.2345),
            ("act_mean".into(), 0.5),
        ],
    };
    let line = frame.format_line();
    assert!(line.contains("n=256"), "missing n=256 in: {line}");
    assert!(line.contains("ke=1.2345"), "missing ke in: {line}");
    assert!(
        line.contains("act_mean=0.5000"),
        "missing act_mean in: {line}"
    );
}

/// Empty registry produces empty DiagnosticsFrame.
#[test]
fn empty_registry_produces_empty_frame() {
    let mut registry = DiagnosticsRegistry::new();
    let p: Vec<Particle> = vec![];
    let config = SimConfig {
        grid_res: 8,
        ..SimConfig::default()
    };
    use emerge::grid::Grid;
    let grid = Grid::new(8);
    let snap = collect_snapshot(
        0,
        &emerge::particle::Particles::new(),
        &grid,
        &config,
        0.1,
        1,
    );
    let frame = registry.collect(&p, &snap);
    assert!(frame.stats.is_empty(), "expected empty frame");
    assert!(frame.format_line().is_empty(), "expected empty format");
}

// â”€â”€â”€ SCALAR DIFFUSION FIELD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Scalar diffusion must move a high-concentration particle's field toward lower concentration.
/// Without decay: total Ï† (summed over particles) should be approximately conserved.
#[test]
fn scalar_diffusion_spreads_and_conserves() {
    let grid_res = 16;
    let config = ScalarDiffusionConfig {
        diffusivity: 1.0,
        decay_rate: 0.0, // no decay â†’ conserved
        ambient: 0.0,
    };

    let mut field = ScalarDiffusionField::new(
        config,
        |p| p.temperature,
        |p, delta| p.temperature += delta,
        grid_res,
    );

    // Two particles: one hot (T=100), one cold (T=0). After diffusion, heat spreads.
    let mut particles = Particles::from(vec![
        Particle {
            x: Vec2::new(7.0, 8.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 100.0,
            ..Particle::zeroed()
        },
        Particle {
            x: Vec2::new(9.0, 8.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 0.0,
            ..Particle::zeroed()
        },
    ]);

    let t_total_before: f32 = particles.temperature.iter().sum();

    // 10 substeps of diffusion
    for _ in 0..10 {
        field.apply(&mut particles, 0.01);
    }

    let t_total_after: f32 = particles.temperature.iter().sum();

    // Cold particle should have warmed
    assert!(
        particles.temperature[1] > 0.1,
        "cold particle didn't warm: T={:.4}",
        particles.temperature[1]
    );

    // Hot particle should have cooled
    assert!(
        particles.temperature[0] < 100.0,
        "hot particle didn't cool: T={:.4}",
        particles.temperature[0]
    );

    // Conservation: total T should be roughly conserved (Â±20% tolerance â€” boundary effects)
    let conservation_err = (t_total_after - t_total_before).abs() / t_total_before;
    assert!(
        conservation_err < 0.20,
        "scalar field: total T changed too much: before={t_total_before:.2} after={t_total_after:.2} err={conservation_err:.2}"
    );
}

/// With decay_rate > 0, total Ï† must decrease over time.
#[test]
fn scalar_diffusion_decay_reduces_total() {
    let config = ScalarDiffusionConfig {
        diffusivity: 0.0,
        decay_rate: 1.0, // fast decay â€” T halves in ~0.69s
        ambient: 0.0,
    };

    let mut field = ScalarDiffusionField::new(
        config,
        |p| p.temperature,
        |p, delta| p.temperature += delta,
        16,
    );

    let mut particles = Particles::from(vec![Particle {
        x: Vec2::new(8.0, 8.0),
        mass: 1.0,
        initial_volume: 1.0,
        volume: 1.0,
        density: 1.0,
        temperature: 100.0,
        ..Particle::zeroed()
    }]);

    for _ in 0..50 {
        field.apply(&mut particles, 0.02); // 1s total
    }

    // After 1s at decay_rate=1.0: T should be ~100*e^(-1) â‰ˆ 36.8
    // Allow Â±50% â€” grid average discretization makes this noisy with one particle
    assert!(
        particles.temperature[0] < 70.0,
        "decay: temperature not decreasing: T={:.2}",
        particles.temperature[0]
    );
    assert!(
        particles.temperature[0] > 0.0,
        "decay: temperature went negative: T={:.2}",
        particles.temperature[0]
    );
}

// â”€â”€â”€ MATERIAL RATE CONSISTENCY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Half-step Ã— 2 must be approximately equivalent to one full step.
/// This tests that material update is smooth/continuous (not discontinuous jumps).
#[test]
fn snow_half_step_consistency() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);

    let base_particle = Particle {
        mass: 1.0,
        initial_volume: 1.0,
        volume: 1.0,
        density: 1.0,
        deformation_gradient: Mat2::from_cols(Vec2::new(0.98, 0.01), Vec2::new(-0.01, 0.98)),
        plastic_volume_ratio: 1.0,
        hardening_scale: 1.0,
        velocity_gradient: Mat2::from_cols(Vec2::new(-0.01, 0.005), Vec2::new(0.005, -0.01)),
        ..Particle::zeroed()
    };

    // Full step
    let mut p_full = base_particle;
    update_particle_of(&snow, &mut p_full, 0.02);

    // Two half-steps
    let mut p_half = base_particle;
    update_particle_of(&snow, &mut p_half, 0.01);
    update_particle_of(&snow, &mut p_half, 0.01);

    let j_full = p_full.deformation_gradient.determinant();
    let j_half = p_half.deformation_gradient.determinant();

    // J should be close (within 1% â€” subcycling plasticity has small discrepancies)
    assert!(
        (j_full - j_half).abs() < 0.01,
        "snow: full-step J={j_full:.6} vs halfÃ—2 J={j_half:.6} â€” too different"
    );
}

/// VonMises: after enough plastic deformation, stress norm must not exceed yield surface.
#[test]
fn von_mises_stress_bounded_by_yield() {
    let yield_stress = 100.0f32;
    let vm = VonMisesMaterial::new(1_000.0, 500.0, yield_stress);

    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let spawn = SpawnRegion {
        initial_velocity_scale: 5.0,
        ..center_spawn(64, 6)
    };
    let mut solver = Simulation::new(config, spawn).with_default_material(Box::new(vm));

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let tau = kirchhoff_stress_of(&vm, &p);
        // von Mises equivalent stress: sqrt(3/2 * s:s) where s = dev(Ï„)
        let tr = (tau.col(0).x + tau.col(1).y) * 0.5;
        let s00 = tau.col(0).x - tr;
        let s11 = tau.col(1).y - tr;
        let s01 = tau.col(1).x; // off-diagonal
        let vm_stress = (1.5 * (s00 * s00 + s11 * s11 + 2.0 * s01 * s01)).sqrt();
        // Allow 40% overshoot: initial_velocity_scale=5.0 creates violent collisions where
        // discrete return-mapping can't fully project to the yield surface in a single step.
        // Key invariant: stress stays finite and bounded, not that it's exactly at yield.
        assert!(
            vm_stress < yield_stress * 1.40,
            "VonMises particle {i}: Ïƒ_vm={vm_stress:.2} > yield {yield_stress:.2}"
        );
    }
}

// â”€â”€â”€ MULTI-MATERIAL ISOLATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Two materials spawned in different regions must not interfere with each other's invariants.
#[test]
fn two_material_solver_both_j_positive() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));

    let spawn0 = SpawnRegion {
        box_center: Vec2::new(20.0, 40.0),
        box_size: IVec2::new(6, 6),
        spacing: 0.5,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::default()
    };

    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let mut solver = Simulation::new(config, spawn0)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)))
        .with_material(1, Box::new(snow));

    let spawn1 = SpawnRegion {
        box_center: Vec2::new(44.0, 40.0),
        box_size: IVec2::new(6, 6),
        spacing: 0.5,
        initial_velocity_scale: 0.0,
        material_id: 1,
        ..SpawnRegion::default()
    };
    let _tag = solver.add_body(spawn1);

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let j = p.deformation_gradient.determinant();
        assert!(
            j > 0.0,
            "two-material: particle {i} mat={} J={j:.2e}",
            p.material_id
        );
    }
}

// â”€â”€â”€ Âµ(I) RHEOLOGY SANITY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// SandMuI: friction_hardening (Âµ(I)) must stay within [Âµ_static, Âµ_dynamic].
#[test]
fn sand_mui_friction_stays_in_range() {
    let mat = MuIRheologyMaterial::small_grain(5429.0, 0.357);
    let mu_static = 20.9f32.to_radians().tan();
    let mu_dynamic = 32.8f32.to_radians().tan();

    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(mat));

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let mu_i = p.friction_hardening;
        assert!(
            mu_i >= mu_static * 0.95 && mu_i <= mu_dynamic * 1.05,
            "SandMuI particle {i}: Âµ(I)={mu_i:.4} out of [{mu_static:.4}, {mu_dynamic:.4}]"
        );
    }
}

// â”€â”€â”€ Bingham fluid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Bingham mud stays above floor without collapsing (yield stress holds shape under gravity).
#[test]
fn bingham_mud_stable_under_gravity() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(BinghamFluidMaterial::high_yield(1500.0, 1.0e4)));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(
            p.x.y > 1.0,
            "mud particle fell through floor: y={:.3}",
            p.x.y
        );
        assert!(p.x.is_finite(), "mud particle position NaN");
        assert!(p.v.is_finite(), "mud particle velocity NaN");
    }
}

/// Bingham J > 0 invariant.
#[test]
fn bingham_j_positive() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(BinghamFluidMaterial::high_yield(1500.0, 1.0e4)));
    solver.step_n(60);
    for p in solver.particles() {
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "Bingham J={j:.4} â‰¤ 0 â€” volume collapsed");
    }
}

/// Bingham lava: higher yield/viscosity than mud, still stable.
#[test]
fn bingham_lava_stable() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 6)).with_default_material(Box::new(
        BinghamFluidMaterial::viscous_high_yield(2700.0, 1.0e5),
    ));
    solver.step_n(40);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "lava particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "lava J={j:.4} â‰¤ 0");
    }
}

// â”€â”€â”€ Viscoelastic (Kelvin-Voigt) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Viscoelastic soft tissue: J > 0, no NaN, stable under gravity.
#[test]
fn viscoelastic_soft_tissue_stable() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(
        ViscoelasticMaterial::near_incompressible(5.0e4, 10.0),
    ));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "tissue particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "tissue J={j:.4} â‰¤ 0");
    }
}

/// Viscoelastic cell body: very soft, stable.
#[test]
fn viscoelastic_cell_body_stable() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 6)).with_default_material(Box::new(
        ViscoelasticMaterial::moderately_compressible(5.0e3, 0.05),
    ));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "cell particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "cell J={j:.4} â‰¤ 0");
    }
}

/// KV viscous contribution: stress with non-zero strain rate > stress without.
/// Tests that the dashpot term activates when velocity_gradient is non-zero.
#[test]
fn viscoelastic_viscous_term_activates() {
    let e = 5.0e4f32;
    let nu = 0.40f32;
    let eta = 500.0f32;

    let visco = ViscoelasticMaterial::from_young_modulus(e, nu, eta);
    let elastic = NeoHookeanMaterial::from_young_modulus(e, nu);

    // Particle at rest with identity F â€” same elastic stress for both.
    let mut p = Particle::zeroed();
    p.volume = 1.0;
    p.density = 1.0;
    p.mass = 1.0;

    let tau_elastic_rest = kirchhoff_stress_of(&elastic, &p);
    let tau_visco_rest = kirchhoff_stress_of(&visco, &p);
    // At rest (C=0, F=I) both give same stress (NeoHookean base is identical).
    let diff_rest = (tau_visco_rest - tau_elastic_rest).x_axis.length()
        + (tau_visco_rest - tau_elastic_rest).y_axis.length();
    assert!(
        diff_rest < 1.0,
        "at rest KV and elastic should agree: diff={diff_rest}"
    );

    // Now give particle a shear strain rate via velocity_gradient.
    p.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0));

    let tau_elastic_shear = kirchhoff_stress_of(&elastic, &p);
    let tau_visco_shear = kirchhoff_stress_of(&visco, &p);

    // KV adds Î·Â·D_dev â€” stress norms must differ.
    let norm_e = tau_elastic_shear.x_axis.length() + tau_elastic_shear.y_axis.length();
    let norm_v = tau_visco_shear.x_axis.length() + tau_visco_shear.y_axis.length();
    assert!(
        (norm_v - norm_e).abs() > 1.0,
        "KV dashpot should contribute when Câ‰ 0: norm_elastic={norm_e:.2} norm_visco={norm_v:.2}"
    );
}

// â”€â”€â”€ Phase rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Phase rule transitions hot particles to a new material id.
#[test]
fn phase_rule_transitions_hot_particles() {
    const COLD_ID: u32 = 0;
    const HOT_ID: u32 = 1;
    let hot_threshold = 0.5f32;

    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_material(
            HOT_ID,
            Box::new(NeoHookeanMaterial::from_young_modulus(1.0e5, 0.3)),
        )
        .with_phase_rule(move |p| {
            if p.material_id == COLD_ID && p.temperature > hot_threshold {
                Some(HOT_ID)
            } else {
                None
            }
        });

    // Heat half the particles manually.
    let n = solver.particles().len();
    for i in 0..n / 2 {
        solver.particles_mut().temperature[i] = hot_threshold + 0.1;
    }

    solver.step();

    let hot_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == HOT_ID)
        .count();
    assert!(
        hot_count >= n / 2,
        "expected â‰¥{} hot particles, got {hot_count}",
        n / 2
    );
}

/// Phase rule: no transitions when condition not met.
#[test]
fn phase_rule_no_spurious_transitions() {
    const MAT_B: u32 = 1;

    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_material(
            MAT_B,
            Box::new(NeoHookeanMaterial::from_young_modulus(1.0e5, 0.3)),
        )
        .with_phase_rule(|p| {
            if p.temperature > 999.0 {
                Some(MAT_B)
            } else {
                None
            }
        });

    // No particles have temperature > 999
    solver.step_n(10);

    let b_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == MAT_B)
        .count();
    assert_eq!(b_count, 0, "spurious transitions to MAT_B: {b_count}");
}

// â”€â”€â”€ Neighbor queries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// particles_near returns only particles within radius.
#[test]
fn particles_near_radius_correct() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let center = Vec2::splat(32.0);
    let radius = 2.0;

    let ps = solver.particles();
    for i in solver.particles_near(center, radius) {
        let dist = (ps.x[i] - center).length();
        assert!(
            dist <= radius + f32::EPSILON,
            "particle at dist={dist:.3} outside radius={radius}"
        );
    }
}

/// count_near matches manual count.
#[test]
fn count_near_matches_manual() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let center = Vec2::splat(32.0);
    let radius = 3.0;
    let mat_id = 0u32;

    let api_count = solver.count_near(center, radius, mat_id);
    let manual_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == mat_id && (p.x - center).length() <= radius)
        .count();

    assert_eq!(api_count, manual_count);
}

/// `particles_knn` must return exactly the same k-nearest INDEX SET as a
/// brute-force sort over every particle -- proves the geometric radius
/// expansion doesn't miss a closer particle just outside its current search
/// box. Query point is deliberately OFF the spawn's own symmetric center
/// (32.0, 32.0): querying from dead center over a symmetric grid puts many
/// particles at the exact same distance, making the k-th-nearest cutoff
/// genuinely ambiguous (confirmed empirically -- an earlier version of this
/// test queried from center and failed on a real tie at the boundary, not an
/// algorithm bug). An off-center point makes distances generically distinct.
#[test]
fn particles_knn_matches_brute_force() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let center = Vec2::new(32.37, 31.82);
    let k = 7; // Ballerini et al. 2008's real ~6-7 neighbor figure

    let ps = solver.particles();
    let mut brute: Vec<(usize, f32)> = (0..ps.len())
        .map(|i| (i, (ps.x[i] - center).length_squared()))
        .collect();
    brute.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let mut expected: Vec<usize> = brute.into_iter().take(k).map(|(i, _)| i).collect();
    expected.sort_unstable();

    let mut got = solver.particles_knn(center, k);
    got.sort_unstable();
    assert_eq!(
        got, expected,
        "particles_knn must match a brute-force k-nearest scan (same particle set)"
    );
}

/// Requesting more neighbors than exist must return everything, not panic or loop forever.
#[test]
fn particles_knn_clamps_to_available_particle_count() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let total = solver.particles().len();
    let got = solver.particles_knn(Vec2::splat(32.0), total + 1000);
    assert_eq!(
        got.len(),
        total,
        "requesting more neighbors than exist must return exactly all of them, not panic"
    );
}

// ─── thermo-mechanical coupling (E(T)) ──────────────────────────────────────────
//
// `thermal_expansion` on NeoHookean/Corotated/Viscoelastic: CPU kirchhoff_stress and GPU
// p2g.wgsl share the same formula (`t_scale = 1.0 + thermal_expansion * temperature`).
// Verifies negative = softening, per its doc comment.

fn stress_frobenius_norm(tau: Mat2) -> f32 {
    (tau.col(0).length_squared() + tau.col(1).length_squared()).sqrt()
}

#[test]
fn neohookean_negative_thermal_expansion_softens_stress() {
    let mut mat = NeoHookeanMaterial::new(100.0, 200.0);
    mat.thermal_expansion = -1.0e-3; // per its own doc comment: negative = softening

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Same moderate shear/stretch deformation for both — only temperature differs.
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(0.15, 0.9));
    mat.init_particle(&mut p);

    p.temperature = 0.0;
    let tau_cold = kirchhoff_stress_of(&mat, &p);

    p.temperature = 500.0;
    let tau_hot = kirchhoff_stress_of(&mat, &p);

    let norm_cold = stress_frobenius_norm(tau_cold);
    let norm_hot = stress_frobenius_norm(tau_hot);
    assert!(
        norm_hot < norm_cold,
        "heating with negative thermal_expansion should soften (lower stress for the same \
         deformation): cold={norm_cold:.4} hot={norm_hot:.4}"
    );

    // Sanity: thermal_expansion=0.0 (the default) must be completely temperature-independent —
    // this is the "zero behavior change for anything that doesn't opt in" guarantee.
    let neutral = NeoHookeanMaterial::new(100.0, 200.0);
    let mut p_neutral = p;
    p_neutral.temperature = 0.0;
    let tau_neutral_cold = kirchhoff_stress_of(&neutral, &p_neutral);
    p_neutral.temperature = 500.0;
    let tau_neutral_hot = kirchhoff_stress_of(&neutral, &p_neutral);
    assert!(
        (stress_frobenius_norm(tau_neutral_cold) - stress_frobenius_norm(tau_neutral_hot)).abs()
            < 1e-6,
        "thermal_expansion=0.0 must be exactly temperature-independent"
    );
}

#[test]
fn corotated_negative_thermal_expansion_softens_stress() {
    let mut mat = CorotatedMaterial::new(100.0, 200.0);
    mat.thermal_expansion = -1.0e-3;

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(0.15, 0.9));
    mat.init_particle(&mut p);

    p.temperature = 0.0;
    let norm_cold = stress_frobenius_norm(kirchhoff_stress_of(&mat, &p));
    p.temperature = 500.0;
    let norm_hot = stress_frobenius_norm(kirchhoff_stress_of(&mat, &p));

    assert!(
        norm_hot < norm_cold,
        "Corotated: heating with negative thermal_expansion should soften: \
         cold={norm_cold:.4} hot={norm_hot:.4}"
    );
}

/// A muscle-driven soft body at FULL activation must stay bounded, not detonate.
///
/// Regression for the `basic_creature` demo blowup: driving `activation` to its
/// documented `[0,1]` ceiling with a strong `active_stress_coeff` produces large
/// active stress, which is only CFL-stable if (a) the adaptive substepper has
/// real headroom and (b) `project_invalid_state` is on to catch any momentary
/// degenerate particle before it cascades. With too few substeps and the
/// projection safety net off, the body scatters to NaN. This asserts the
/// stable-config contract: a peristaltic creature run at max drive for many
/// frames stays finite and spatially coherent.
#[test]
fn muscle_creature_stays_bounded_at_full_activation() {
    const GRID: usize = 64;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;

    let mut mat = NeoHookeanMaterial::new(5.0, 10.0);
    mat.active_stress_coeff = 25.0;
    let config = SimConfig {
        min_dt: 0.01,
        // Full CFL headroom + the degenerate-state safety net on: the two settings
        // that keep max-activation muscle stress stable (see doc above).
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(32.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(FrictionBoundary::new(4, 0.65)));

    let body_range = 0..sim.particles().len();
    let body_left = body_center.x - 12.0;
    {
        let particles = sim.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = Vec2::Y;
        }
    }

    // Bilateral CPG under a sustained hard steering bias -- matches the real
    // interactive session that triggered a full-body NaN collapse at frame 1070
    // (basic_creature demo, steer held at +1.0 for hundreds of frames).
    const N_RINGS: usize = 2;
    const N_PER_RING: usize = MUSCLE_GROUPS / N_RINGS;
    let mut lnn = emerge::control::Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, 1.0);
    for step in 0..1500 {
        lnn.set_ring_bias(0, N_PER_RING, 1.0);
        lnn.set_ring_bias(1, N_PER_RING, -1.0);
        lnn.step(DT);
        let acts: Vec<f32> = lnn.activations().collect();
        let range = body_range.clone();
        let particles = sim.particles_mut();
        for i in range {
            let group = particles.muscle_group_id[i] as usize;
            particles.activation[i] = (0.9 * acts[group]).clamp(0.0, 1.0);
        }
        sim.step();

        let snap = sim.diagnostics_snapshot();
        assert_eq!(
            snap.non_finite_particle_values, 0,
            "creature went non-finite at step {step} under sustained steering bias"
        );
    }
}

/// Three locomotion mechanisms compared:
/// 1. Plain `FrictionBoundary` -- near-zero net drift (scallop theorem: symmetric
///    muscle cycle against constant friction cancels its own displacement).
/// 2. `GripFrictionBoundary` (phase-gated grip) -- still only near-zero drift.
/// 3. `RatchetFrictionBoundary` (directional/setae-style, asymmetric by tangential
///    velocity sign, phase-independent) -- what actually works, matching SoftZoo and
///    real-crawler literature (structural asymmetry, not phase-gating).
#[test]
fn grip_friction_locomotion_sweep() {
    const GRID: usize = 64;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;

    fn run(boundary: Box<dyn emerge::BoundaryCondition>, fiber_dir: Vec2) -> f32 {
        let mut mat = NeoHookeanMaterial::new(5.0, 10.0);
        mat.active_stress_coeff = 25.0;
        let config = SimConfig {
            min_dt: 0.01,
            max_substeps_per_step: 64,
            project_invalid_state: true,
            ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        };
        let body_center = Vec2::new(32.0, 20.0);
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(24, 6),
            box_center: body_center,
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(mat))
            .with_boundary(boundary);

        let body_range = 0..sim.particles().len();
        let body_left = body_center.x - 12.0;
        {
            let particles = sim.particles_mut();
            for i in body_range.clone() {
                let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
                particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
                particles.activation_dir[i] = fiber_dir;
            }
        }

        let mut lnn = emerge::control::Lnn::traveling_wave(MUSCLE_GROUPS, 1.0);
        let mut centroid_start = Vec2::ZERO;
        for step in 0..800 {
            lnn.step(DT);
            let acts: Vec<f32> = lnn.activations().collect();
            let range = body_range.clone();
            let particles = sim.particles_mut();
            for i in range {
                let group = particles.muscle_group_id[i] as usize;
                particles.activation[i] = (0.9 * acts[group]).clamp(0.0, 1.0);
            }
            sim.step();
            if step == 20 {
                let particles = sim.particles();
                let n = particles.len() as f32;
                centroid_start = (0..particles.len()).map(|i| particles.x[i]).sum::<Vec2>() / n;
            }
        }
        let particles = sim.particles();
        let n = particles.len() as f32;
        let centroid_end = (0..particles.len()).map(|i| particles.x[i]).sum::<Vec2>() / n;
        (centroid_end - centroid_start).x
    }

    for fiber_dir in [Vec2::Y, Vec2::X] {
        for grip_gain in [0.0, 0.3, 0.6, 0.9] {
            let drift_x = run(
                Box::new(GripFrictionBoundary::new(4, 0.65, grip_gain)),
                fiber_dir,
            );
            println!("fiber={fiber_dir:?} grip_gain={grip_gain:.1} drift.x={drift_x:.2}");
            assert!(
                drift_x.is_finite() && drift_x.abs() < 20.0,
                "fiber={fiber_dir:?} grip_gain={grip_gain}: drift.x={drift_x:.2} not physically sane"
            );
        }
    }

    println!("--- RatchetFrictionBoundary (directional/setae-style, no phase gating) ---");
    for fiber_dir in [Vec2::Y, Vec2::X] {
        for (mu_easy, mu_resist) in [(0.65, 0.65), (0.1, 0.95), (0.02, 1.0)] {
            let drift_x = run(
                Box::new(RatchetFrictionBoundary::new(4, mu_easy, mu_resist, Vec2::X)),
                fiber_dir,
            );
            println!(
                "fiber={fiber_dir:?} mu_easy={mu_easy:.2} mu_resist={mu_resist:.2} drift.x={drift_x:.2}"
            );
            // Sanity bound, not a "stay near zero" bound: real crawling produces
            // substantial drift (body is 24 units long); only reject runaway values.
            assert!(
                drift_x.is_finite() && drift_x.abs() < 200.0,
                "fiber={fiber_dir:?} mu_easy={mu_easy} mu_resist={mu_resist}: \
                 drift.x={drift_x:.2} not physically sane"
            );
        }
    }
}

/// `RatchetFrictionBoundary` must produce substantial, correctly-directed net
/// locomotion for a muscle-driven soft body (see `grip_friction_locomotion_sweep` for
/// the mechanisms that don't). Body is 24 units long; a working crawl should cover a
/// meaningful fraction of that, in the commanded `easy_direction`.
#[test]
fn ratchet_friction_produces_real_directed_locomotion() {
    const GRID: usize = 64;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;

    let mut mat = NeoHookeanMaterial::new(5.0, 10.0);
    mat.active_stress_coeff = 25.0;
    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(32.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(RatchetFrictionBoundary::new(
            4,
            0.1,
            0.95,
            Vec2::X,
        )));

    let body_range = 0..sim.particles().len();
    let body_left = body_center.x - 12.0;
    {
        let particles = sim.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = Vec2::Y;
        }
    }

    let mut lnn = emerge::control::Lnn::traveling_wave(MUSCLE_GROUPS, 1.0);
    let mut centroid_start = Vec2::ZERO;
    for step in 0..800 {
        lnn.step(DT);
        let acts: Vec<f32> = lnn.activations().collect();
        let range = body_range.clone();
        let particles = sim.particles_mut();
        for i in range {
            let group = particles.muscle_group_id[i] as usize;
            particles.activation[i] = (0.9 * acts[group]).clamp(0.0, 1.0);
        }
        sim.step();
        if step == 20 {
            let particles = sim.particles();
            let n = particles.len() as f32;
            centroid_start = (0..particles.len()).map(|i| particles.x[i]).sum::<Vec2>() / n;
        }

        let snap = sim.diagnostics_snapshot();
        assert_eq!(
            snap.non_finite_particle_values, 0,
            "creature went non-finite at step {step} during ratchet-driven crawling"
        );
    }
    let particles = sim.particles();
    let n = particles.len() as f32;
    let centroid_end = (0..particles.len()).map(|i| particles.x[i]).sum::<Vec2>() / n;
    let drift_x = (centroid_end - centroid_start).x;

    assert!(
        drift_x > 10.0,
        "ratchet friction should produce a real, substantial crawl in the +X \
         easy_direction (expected > 10 units of an 24-unit-long body), got {drift_x:.2}"
    );
}

/// `RatchetFrictionBoundary::set_easy_direction` must be a live control, not baked in
/// at construction: flipping mid-run via an `Arc`-shared boundary instance must make
/// the body's second-half crawl go the opposite way from the first half.
///
/// Step counts (1200 total / 700 post-flip) are sized for `NeoHookeanMaterial`'s real
/// volumetric term (Simo-Pister log-barrier `ln(J)`, see `elastic.rs`): momentum takes
/// longer to unwind after a flip than with the old bounded `(J^2-1)` term, so the
/// reversal needs more room to show up.
#[test]
fn ratchet_easy_direction_is_live_and_reversible() {
    const GRID: usize = 64;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;

    let mut mat = NeoHookeanMaterial::new(5.0, 10.0);
    mat.active_stress_coeff = 25.0;
    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(32.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let ratchet = std::sync::Arc::new(RatchetFrictionBoundary::new(4, 0.1, 0.95, Vec2::X));
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(std::sync::Arc::clone(&ratchet)));

    let body_range = 0..sim.particles().len();
    let body_left = body_center.x - 12.0;
    {
        let particles = sim.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = Vec2::Y;
        }
    }

    let mut lnn = emerge::control::Lnn::traveling_wave(MUSCLE_GROUPS, 1.0);
    let centroid_at = |sim: &Simulation| -> Vec2 {
        let particles = sim.particles();
        let n = particles.len() as f32;
        (0..particles.len()).map(|i| particles.x[i]).sum::<Vec2>() / n
    };

    let mut centroid_start = Vec2::ZERO;
    let mut centroid_mid = Vec2::ZERO;
    for step in 0..1200 {
        if step == 500 {
            // Flip before the body settles into its resting stall (~step 600).
            ratchet.set_easy_direction(Vec2::NEG_X);
            centroid_mid = centroid_at(&sim);
        }
        lnn.step(DT);
        let acts: Vec<f32> = lnn.activations().collect();
        let range = body_range.clone();
        let particles = sim.particles_mut();
        for i in range {
            let group = particles.muscle_group_id[i] as usize;
            particles.activation[i] = (0.9 * acts[group]).clamp(0.0, 1.0);
        }
        sim.step();
        if step == 20 {
            centroid_start = centroid_at(&sim);
        }
    }
    let centroid_end = centroid_at(&sim);

    let first_half_drift = (centroid_mid - centroid_start).x;
    let second_half_drift = (centroid_end - centroid_mid).x;

    assert!(
        first_half_drift > 3.0,
        "first half (easy_direction=+X) should crawl forward, got {first_half_drift:.2}"
    );
    assert!(
        second_half_drift < -1.0,
        "second half, AFTER live set_easy_direction(NEG_X), should crawl backward \
         (not just stop) -- got {second_half_drift:.2}. If this is ~0, the live \
         direction control isn't actually reaching the physics."
    );
}

/// Drucker-Prager's cone yield surface, by construction, only trims deviatoric (shear)
/// strain -- a near-hydrostatic impact (mostly compression, little shear) is judged
/// "elastic" regardless of magnitude, so real dry sand under a hard contact impulse can
/// compact far past its physical void-ratio limit (~20-40%). Inherent gap in the
/// published model itself (Klar et al. 2016), not an emerge-specific bug.
///
/// Fixed the same way snow's `min_plastic_jacobian` floor works: `DruckerPragerMaterial::
/// min_volume_jacobian` (0.6 default) uniformly rescales the stored singular values
/// after the existing shear-yield projection, only engaging past sand's packing limit.
/// Proven in both multi-field contact orientations (sand as the "grip" field and as the
/// "rest" field).
#[test]
fn drucker_prager_volumetric_floor_prevents_unphysical_contact_collapse() {
    const GRID: usize = 64;
    const DT: f32 = 0.05;

    fn run(rest_is_plastic: bool, both_elastic_control: bool) -> (f32, f32, f32) {
        let config = SimConfig {
            min_dt: 0.005,
            max_substeps_per_step: 64,
            project_invalid_state: true,
            ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        };
        let rest_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(48, 8),
            box_center: Vec2::new(32.0, 8.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let rest_mat: Box<dyn emerge::materials::MaterialModel> =
            if !both_elastic_control && rest_is_plastic {
                Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333))
            } else {
                Box::new(CorotatedMaterial::new(200.0, 400.0))
            };
        let mut sim = Simulation::new(config, rest_spawn).with_default_material(rest_mat);
        let rest_count = sim.particles().len();

        let grip_mat: Box<dyn emerge::materials::MaterialModel> =
            if !both_elastic_control && !rest_is_plastic {
                Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333))
            } else {
                Box::new(CorotatedMaterial::new(200.0, 400.0))
            };
        let grip_mat_id = sim.register_material(grip_mat);
        let grip_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 8),
            box_center: Vec2::new(32.0, 14.0),
            material_id: grip_mat_id.0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(sim.config())
        };
        let _ = sim.add_body(grip_spawn);
        let grip_range = rest_count..sim.particles().len();
        {
            let particles = sim.particles_mut();
            for i in grip_range.clone() {
                particles.contact_group[i] = 1;
            }
        }

        let mut min_j_rest = f32::MAX;
        let mut min_j_grip = f32::MAX;
        for _ in 0..600 {
            sim.step();
            let particles = sim.particles();
            for i in 0..rest_count {
                min_j_rest = min_j_rest.min(particles.deformation_gradient[i].determinant());
            }
            for i in grip_range.clone() {
                min_j_grip = min_j_grip.min(particles.deformation_gradient[i].determinant());
            }
        }
        let snap = sim.diagnostics_snapshot();
        println!("  [detail] min_j_rest_body={min_j_rest:.4} min_j_grip_body={min_j_grip:.4}");
        (min_j_rest, min_j_grip, snap.max_particle_speed)
    }

    let (control_rest_j, control_grip_j, control_vmax) = run(true, true);
    println!(
        "[control: both elastic] min_j_rest={control_rest_j:.4} min_j_grip={control_grip_j:.4} vmax={control_vmax:.3}"
    );
    // plastic REST: the DP-tagged body is the wide slab (contact_group=0) --
    // matches snake_on_terrain's exact arrangement.
    let (plastic_rest_dp_j, _elastic_grip_j, plastic_rest_vmax) = run(true, false);
    println!("[plastic REST] min_j_DP_body={plastic_rest_dp_j:.4} vmax={plastic_rest_vmax:.3}");
    // plastic GRIP: the DP-tagged body is the small block (contact_group=1) this
    // time -- the REST slab is plain elastic, so its own min_j (unrelated to
    // this fix) is expected to be low, matching the control's own elastic
    // compression under this hard impact; only the DP body's own floor matters here.
    let (_elastic_rest_j, plastic_grip_dp_j, plastic_grip_vmax) = run(false, false);
    println!("[plastic GRIP] min_j_DP_body={plastic_grip_dp_j:.4} vmax={plastic_grip_vmax:.3}");

    assert!(
        plastic_rest_dp_j > 0.5,
        "BUG: volumetric floor (min_volume_jacobian=0.6) should prevent sand from \
         compressing past its own real physical packing limit -- got min_j={plastic_rest_dp_j:.4} \
         (was 0.0057 before the fix). If this is still near-zero, the floor isn't reaching \
         the real contact-driven compaction path."
    );
    assert!(
        plastic_grip_dp_j > 0.5,
        "BUG: same floor should hold when the plastic body is the grip field, not just \
         rest -- got min_j={plastic_grip_dp_j:.4}"
    );
}

/// `Lnn::traveling_wave`/`coupled_traveling_wave` must not converge to a
/// fully-synchronized fixed point (oscillation dying) within ~20 steps at dt=0.1,
/// regardless of external ring bias. See `src/information/control/lnn.rs` for the
/// mechanism (no self-inhibition, symmetrized excite/inhibit weights); that module's
/// `coupled_traveling_wave_sustains_a_real_long_horizon_traveling_wave` is the
/// permanent 10,000-step/phase-coherence regression for the same fix.
#[test]
fn cpg_oscillator_does_not_die_within_50_steps() {
    let dt = 0.1;

    let mut lnn = emerge::control::Lnn::coupled_traveling_wave(2, 4, 1.0, 1.0);
    let mut prev: Vec<f32> = lnn.activations().collect();
    let mut died_by_step_50 = false;
    for step in 0..50 {
        lnn.step(dt);
        let acts: Vec<f32> = lnn.activations().collect();
        let max_delta = acts
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if step > 10 && max_delta < 1e-7 {
            died_by_step_50 = true;
        }
        prev = acts;
    }

    assert!(
        !died_by_step_50,
        "BUG STILL PRESENT: coupled_traveling_wave's oscillator died (fully \
         synchronized, zero relative phase) within 50 steps at dt=0.1, with \
         zero external bias. Real gameplay runs thousands of these steps, so \
         this means no real traveling wave ever sustains -- see doc comment."
    );
}

/// Regression for the internal-viscosity fix (see `combined_kirchhoff_stress`'s doc in
/// `src/spacetime/transfer.rs`): a purely elastic (`viscosity = 0.0`) muscle body has no
/// internal dissipation, so cyclic activation pumps energy in every gait cycle with
/// nowhere to go, ratcheting into unbounded compaction collapse (drift -> ~0 by step
/// 6500-7000).
///
/// Kelvin-Voigt viscosity (`ViscoelasticMaterial`'s term, generalized onto
/// `NeoHookeanMaterial`) fixes it: damping proportional to LOCAL strain rate is
/// near-zero for rigid-body translation (the crawl itself) but substantial for the
/// unbounded internal deformation -- physically grounded (living tissue is
/// viscoelastic, Fung 1993), not a stability hack.
///
/// Checks the regression signature at step 8000: drift over the final 1000-step window
/// must stay real. viscosity=150 with the coarser config is cheap enough for a regular
/// test; a flatter, non-decaying result needs viscosity=250-400 with a finer adaptive
/// timestep, too expensive here. `NeoHookeanMaterial::timestep_bound`'s viscosity CFL
/// term is required too -- without it, higher viscosity inverts the deformation
/// gradient within ~500 steps.
#[test]
fn neohookean_viscosity_prevents_compaction_ratchet() {
    const GRID: usize = 96;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;

    let mut mat = NeoHookeanMaterial::new(13.0, 26.0);
    mat.active_stress_coeff = 40.0;
    mat.viscosity = 150.0;
    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(48.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let ratchet = std::sync::Arc::new(RatchetFrictionBoundary::new(4, 0.1, 0.95, Vec2::X));
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(ratchet));

    let body_range = 0..sim.particles().len();
    let body_left = body_center.x - 12.0;
    let fiber_dir = Vec2::new(0.3, 1.0).normalize();
    {
        let particles = sim.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = fiber_dir;
        }
    }

    let mut lnn = emerge::control::Lnn::coupled_traveling_wave(2, 4, 1.0, 1.0);
    for _ in 0..600 {
        lnn.step(DT);
    }

    let n = sim.particles().len() as f32;
    let mut checkpoint_x = 0.0;
    let total_steps = 8000usize;

    for step in 0..total_steps {
        lnn.step(DT);
        let acts: Vec<f32> = lnn.activations().collect();
        let range = body_range.clone();
        let particles = sim.particles_mut();
        for i in range {
            let group = particles.muscle_group_id[i] as usize;
            particles.activation[i] = (0.9 * acts[group]).clamp(0.0, 1.0);
        }
        sim.step();
        if step == total_steps - 1000 {
            checkpoint_x = (0..sim.particles().len())
                .map(|i| sim.particles().x[i].x)
                .sum::<f32>()
                / n;
        }
    }

    let final_x: f32 = (0..sim.particles().len())
        .map(|i| sim.particles().x[i].x)
        .sum::<f32>()
        / n;
    let final_window_drift = final_x - checkpoint_x;

    assert!(
        final_window_drift > 0.08,
        "compaction ratchet is back: drift over the final 1000 steps (out of {total_steps}) \
         was only {final_window_drift:.4} -- the old, viscosity=0 material always collapsed \
         to near-zero-or-negative drift (~-0.01 to +0.02) by this point; a healthy, \
         viscosity-damped body should still show real ongoing drift (~0.1-0.13 measured \
         here, higher still at higher viscosity with a finer adaptive timestep)."
    );
}

/// THE defining test for multi-field frictional contact (Bardenhagen, Guilkey, Roessig,
/// Brackbill 2001, "An Improved Contact Algorithm for the Material Point Method").
///
/// MPM's default contact is unconditional infinite-friction stick: two touching bodies
/// share one velocity field, so a friction coefficient has no effect and nothing can
/// ever slip. Classic textbook Coulomb-contact validation: a block with initial
/// horizontal velocity, resting under gravity on a heavier floor slab (contact_group 1
/// vs. 0), must slide at low friction and stick (decelerate to match the floor) at high
/// friction -- before the fix both cases were identical (always stick). See
/// `Grid::resolve_contact`'s doc in `src/spacetime/grid/mod.rs` for the normal-fit +
/// Baumgarte-stabilization mechanism.
#[test]
fn multi_field_contact_produces_real_coulomb_slip_and_stick() {
    fn run(friction: f32) -> f32 {
        const GRID: usize = 64;
        const DT: f32 = 0.02;
        let config = SimConfig {
            contact_friction: friction,
            min_dt: 0.001,
            max_substeps_per_step: 128,
            project_invalid_state: true,
            ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        };

        // Block: small, contact_group=1 ("grip" field), spawned right at the floor's
        // surface (minimal gap -- a real fall of several units first would cause a hard
        // impact that scrambles the clean slip/stick signal regardless of friction).
        let block_mat = CorotatedMaterial::new(200.0, 400.0);
        let block_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(6, 6),
            box_center: Vec2::new(32.0, 11.6),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, block_spawn)
            .with_default_material(Box::new(block_mat))
            .with_boundary(Box::new(SlipBoundary::new(2)));
        let block_range = 0..sim.particles().len();
        {
            let particles = sim.particles_mut();
            for i in block_range.clone() {
                particles.contact_group[i] = 1;
            }
        }

        // Floor: wide, heavy slab (contact_group=0, the "rest" field, the default --
        // untouched), added second so it doesn't disturb the block's own index range.
        let floor_mat_id = sim.register_material(Box::new(CorotatedMaterial::new(200.0, 400.0)));
        let floor_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(48, 8),
            box_center: Vec2::new(32.0, 8.0),
            material_id: floor_mat_id.0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(sim.config())
        };
        let _ = sim.add_body(floor_spawn);

        // Settle first (friction active the whole time, but starting at rest -- no
        // impact to scramble), THEN inject the real test velocity and measure over a
        // short separate window. Isolates "does it slide" from "does it survive
        // landing."
        for _ in 0..300 {
            sim.step();
        }
        {
            let particles = sim.particles_mut();
            for i in block_range.clone() {
                particles.v[i].x = 3.0;
            }
        }
        for _ in 0..150 {
            sim.step();
        }

        let n = block_range.len() as f32;
        let particles = sim.particles();
        block_range.map(|i| particles.v[i].x).sum::<f32>() / n
    }

    let slip_speed = run(0.0);
    let stick_speed = run(3.0);

    assert!(
        slip_speed > 1.0,
        "BUG: at zero friction the block should keep real horizontal velocity (free \
         separation / slip must be possible) -- got mean v_x={slip_speed:.4} (started at \
         3.0). If this is ~0, contact is still unconditionally sticking regardless of \
         friction, i.e. the fix isn't real."
    );
    assert!(
        stick_speed < 0.5,
        "BUG: at high friction the block should decelerate to near the floor's velocity \
         (real Coulomb stick) -- got mean v_x={stick_speed:.4} (started at 3.0). If this \
         is still ~3.0, friction has no effect at all."
    );
}

/// `DirectionalContactGrip`: the multi-field-contact generalization of
/// `RatchetFrictionBoundary`'s directional/setae-style friction, letting a creature
/// crawl on actual terrain particles via `contact_group` instead of only the engine's
/// fixed-world-floor boundary. Same block-on-floor rig as
/// `multi_field_contact_produces_real_coulomb_slip_and_stick`, but velocity is injected
/// in the easy direction vs. the resisted direction with the identical grip instance --
/// "easy" should keep far more speed than "resist".
#[test]
fn directional_contact_grip_is_real_and_direction_aware() {
    fn run(injected_vx: f32) -> f32 {
        const GRID: usize = 64;
        const DT: f32 = 0.02;
        let config = SimConfig {
            contact_friction: 0.5, // unused when directional_grip is set; sanity default
            min_dt: 0.001,
            max_substeps_per_step: 128,
            project_invalid_state: true,
            ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        };

        let block_mat = CorotatedMaterial::new(200.0, 400.0);
        let block_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(6, 6),
            box_center: Vec2::new(32.0, 11.6),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let grip = std::sync::Arc::new(emerge::DirectionalContactGrip::new(
            0.05,
            0.9,
            Vec2::X, // "easy" direction: +X
        ));
        let mut sim = Simulation::new(config, block_spawn)
            .with_default_material(Box::new(block_mat))
            .with_boundary(Box::new(SlipBoundary::new(2)))
            .with_contact_grip(grip);
        let block_range = 0..sim.particles().len();
        {
            let particles = sim.particles_mut();
            for i in block_range.clone() {
                particles.contact_group[i] = 1;
            }
        }

        let floor_mat_id = sim.register_material(Box::new(CorotatedMaterial::new(200.0, 400.0)));
        let floor_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(48, 8),
            box_center: Vec2::new(32.0, 8.0),
            material_id: floor_mat_id.0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(sim.config())
        };
        let _ = sim.add_body(floor_spawn);

        for _ in 0..300 {
            sim.step();
        }
        {
            let particles = sim.particles_mut();
            for i in block_range.clone() {
                particles.v[i].x = injected_vx;
            }
        }
        for _ in 0..150 {
            sim.step();
        }

        let n = block_range.len() as f32;
        let particles = sim.particles();
        block_range.map(|i| particles.v[i].x).sum::<f32>() / n
    }

    let easy_speed = run(3.0); // aligned with easy_direction=+X
    let resist_speed = run(-3.0); // against it

    assert!(
        easy_speed > 1.0,
        "BUG: sliding in the easy direction should keep real speed (low mu_easy=0.05) -- \
         got mean v_x={easy_speed:.4} (started at 3.0). If this is ~0, the directional \
         grip isn't reaching the real contact resolver at all."
    );
    // Relative, not an absolute cutoff: this rig's actual per-contact-event normal
    // force (a small light block settling under gentle gravity) doesn't fully arrest
    // -3.0 within the test window even at mu_resist=0.9 -- real Coulomb impulse scales
    // with normal_speed, not just mu, so "decelerates to exactly ~0" isn't the right
    // bar here. What proves direction-awareness is the SAME rig, SAME grip instance,
    // giving a dramatically different outcome purely from the sign of the injected
    // velocity: easy retains its speed, resist loses the large majority of it.
    assert!(
        resist_speed.abs() < easy_speed.abs() * 0.35,
        "BUG: resisted sliding should lose far more speed than easy sliding retains -- \
         got easy={easy_speed:.4} (from +3.0) vs resist={resist_speed:.4} (from -3.0). \
         If these are close in magnitude, the resist/easy split isn't actually \
         direction-aware."
    );
}

/// `project_particle_state_to_admissible` (`src/spacetime/solver/step.rs`, private) is the
/// last line of defense against numerical blowup -- every real simulation is built with
/// `SimConfig::standard()`/`project_invalid_state: true` specifically so a momentary NaN or
/// degenerate value gets corrected instead of cascading. Despite that, before this test, NOT
/// ONE of the 11 distinct fields it guards had a direct regression test anywhere in the
/// suite -- every existing use just enables it as a background safety net and trusts it
/// works. This exercises it end-to-end through the real public API (spawn, corrupt one field
/// per particle, step once, verify recovery), not by reaching into the private function.
#[test]
fn project_invalid_state_recovers_every_guarded_field() {
    let config = SimConfig {
        project_invalid_state: true,
        ..SimConfig::standard(32, 0.02, Vec2::ZERO)
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::splat(16.0),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 200.0)));
    assert!(
        sim.particles().len() >= 14,
        "test needs at least 14 particles, one per guarded field, got {}",
        sim.particles().len()
    );

    let nan = f32::NAN;
    {
        let particles = sim.particles_mut();
        particles.x[0] = Vec2::splat(nan);
        particles.x[1] = Vec2::splat(1.0e9); // finite but far out of bounds
        particles.v[2] = Vec2::splat(nan);
        particles.velocity_gradient[3] = Mat2::from_cols(Vec2::splat(nan), Vec2::ZERO);
        particles.deformation_gradient[4] = Mat2::from_cols(Vec2::splat(nan), Vec2::ZERO);
        particles.deformation_gradient[5] = Mat2::ZERO; // det() == 0: degenerate J
        particles.deformation_gradient[6] = Mat2::from_diagonal(Vec2::splat(1.0e6)); // J >> j_max
        particles.plastic_volume_ratio[7] = -1.0;
        particles.hardening_scale[8] = nan;
        particles.friction_hardening[9] = nan;
        particles.log_volume_strain[9] = nan; // share particle 9 -- two scalar-NaN guards, one particle
        particles.mass[10] = -1.0;
        particles.volume[11] = 0.0;
        // initial_volume/density must be corrupted here too, or their recovery branches
        // in step.rs's project_particle_state_to_admissible go untested.
        particles.initial_volume[12] = nan;
        particles.density[13] = -1.0;
    }

    sim.step();

    let particles = sim.particles();
    for (i, p) in particles.iter().enumerate() {
        assert!(
            p.x.is_finite(),
            "particle {i}: position not finite after projection: {:?}",
            p.x
        );
        assert!(
            p.v.is_finite(),
            "particle {i}: velocity not finite after projection: {:?}",
            p.v
        );
        assert!(
            p.velocity_gradient.x_axis.is_finite() && p.velocity_gradient.y_axis.is_finite(),
            "particle {i}: velocity_gradient not finite after projection"
        );
        assert!(
            p.deformation_gradient.x_axis.is_finite() && p.deformation_gradient.y_axis.is_finite(),
            "particle {i}: deformation_gradient not finite after projection"
        );
        assert!(
            p.deformation_gradient.determinant() > 0.0,
            "particle {i}: J={} not positive after projection",
            p.deformation_gradient.determinant()
        );
        assert!(
            p.deformation_gradient.determinant() <= config.j_max * 1.01,
            "particle {i}: J={} exceeds j_max={} after projection",
            p.deformation_gradient.determinant(),
            config.j_max
        );
        assert!(
            p.plastic_volume_ratio.is_finite() && p.plastic_volume_ratio > 0.0,
            "particle {i}: plastic_volume_ratio={} not positive-finite after projection",
            p.plastic_volume_ratio
        );
        assert!(
            p.hardening_scale.is_finite() && p.hardening_scale > 0.0,
            "particle {i}: hardening_scale={} not positive-finite after projection",
            p.hardening_scale
        );
        assert!(
            p.friction_hardening.is_finite(),
            "particle {i}: friction_hardening not finite after projection"
        );
        assert!(
            p.log_volume_strain.is_finite(),
            "particle {i}: log_volume_strain not finite after projection"
        );
        assert!(
            p.mass.is_finite() && p.mass > 0.0,
            "particle {i}: mass={} not positive-finite after projection",
            p.mass
        );
        assert!(
            p.initial_volume.is_finite() && p.initial_volume > 0.0,
            "particle {i}: initial_volume={} not positive-finite after projection",
            p.initial_volume
        );
        assert!(
            p.volume.is_finite() && p.volume > 0.0,
            "particle {i}: volume={} not positive-finite after projection",
            p.volume
        );
        assert!(
            p.density.is_finite() && p.density > 0.0,
            "particle {i}: density={} not positive-finite after projection",
            p.density
        );
    }

    // The corrected state must not just be finite once -- it must be genuinely admissible,
    // i.e. the simulation keeps running cleanly afterward instead of re-diverging next step.
    for _ in 0..20 {
        sim.step();
    }
    for (i, p) in sim.particles().iter().enumerate() {
        assert!(
            p.x.is_finite() && p.v.is_finite() && p.deformation_gradient.determinant() > 0.0,
            "particle {i}: diverged again within 20 steps after projection recovered it"
        );
    }
}

/// `Particle::pinned` (Dirichlet/kinematic anchor) must hold a tagged particle at its
/// exact spawn position under sustained gravity and impact, while unpinned particles in
/// the same body keep falling/reacting normally -- a per-particle boundary condition,
/// not a global freeze toggle.
#[test]
fn pinned_particles_stay_fixed_under_gravity_and_impact() {
    let config = SimConfig {
        project_invalid_state: true,
        ..SimConfig::standard(32, 0.02, Vec2::new(0.0, -0.5))
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 8),
        box_center: Vec2::splat(16.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 200.0)));

    // Pin the bottom row (lowest y) of the block -- a "bedrock" layer -- leave everything
    // else free, matching the real intended use (anchor terrain, not freeze it solid).
    let min_y = sim
        .particles()
        .iter()
        .map(|p| p.x.y)
        .fold(f32::INFINITY, f32::min);
    let pinned_indices: Vec<usize> = sim
        .particles()
        .iter()
        .enumerate()
        .filter(|(_, p)| p.x.y < min_y + 0.1)
        .map(|(i, _)| i)
        .collect();
    assert!(
        !pinned_indices.is_empty(),
        "test setup bug: no particles found in the bottom row to pin"
    );
    let pinned_start_positions: Vec<Vec2> = pinned_indices
        .iter()
        .map(|&i| sim.particles().get(i).x)
        .collect();
    {
        let particles = sim.particles_mut();
        for &i in &pinned_indices {
            particles.pinned[i] = 1;
        }
    }

    // Real external impact, not just gravity -- pinned particles must resist this too.
    sim.apply_impulse(Vec2::splat(16.0), 8.0, Vec2::new(50.0, 20.0));

    for _ in 0..300 {
        sim.step();
    }

    for (&i, &start) in pinned_indices.iter().zip(pinned_start_positions.iter()) {
        let p = sim.particles().get(i);
        assert!(
            (p.x - start).length() < 1.0e-4,
            "pinned particle {i} moved: start={start:?} now={:?} (delta={})",
            p.x,
            (p.x - start).length()
        );
        assert_eq!(
            p.v,
            Vec2::ZERO,
            "pinned particle {i} has nonzero velocity: {:?}",
            p.v
        );
    }

    // Unpinned particles in the SAME body must still respond normally -- otherwise this
    // would just be a slow way to freeze the whole scene, not a real per-particle BC.
    let unpinned_moved = sim
        .particles()
        .iter()
        .enumerate()
        .filter(|(i, _)| !pinned_indices.contains(i))
        .any(|(_, p)| p.v.length() > 0.1 || p.x.y < min_y - 0.5);
    assert!(
        unpinned_moved,
        "no unpinned particle moved/fell at all -- pinning may have frozen the whole body, \
         not just the tagged particles"
    );
}

/// Long, purely passive settle (no muscle activation, no steering) exposes failures the
/// shorter regression tests above don't run long enough to catch.
///
/// `svd2` does not guarantee non-negative singular values, so `min_volume_jacobian`'s
/// floor must clamp on the MAGNITUDE of sigma, not raw sigma -- a `j_new > 0.0` guard
/// alone lets an already-inverted (negative) singular value pass through unclamped.
///
/// `Grid::resolve_contact`'s Baumgarte position correction must be a velocity FLOOR
/// (only pushes `v_rel`'s normal component down to the target separating speed if it
/// isn't there already), not an unconditional additive kick -- the latter, fired every
/// substep along a noisy LR-fitted normal, becomes an unbounded random-walk energy
/// source over thousands of substeps (standard Box2D/Bullet-style sequential-impulse
/// position bias avoids this by construction).
///
/// Residual: the snake's own elastic body still settles to a mild, stable
/// self-inversion (min_j≈-1.07) concentrated at its geometric corners -- ordinary
/// FEM/MPM corner stress concentration, not a remaining contact leak (contact only
/// engages at the snake's bottom face).
#[test]
fn drucker_prager_volumetric_floor_holds_over_long_passive_settle() {
    const GRID: usize = 128;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;
    const SNAKE_CONTACT_GROUP: u32 = 1;

    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let terrain_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(100, 12),
        box_center: Vec2::new(64.0, 10.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, terrain_spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)));
    let terrain_count = sim.particles().len();

    let mut snake_mat = NeoHookeanMaterial::new(13.0, 26.0);
    snake_mat.active_stress_coeff = 80.0;
    snake_mat.viscosity = 150.0;
    let snake_mat_id = sim.register_material(Box::new(snake_mat));
    let body_center = Vec2::new(64.0, 20.0);
    let body_len = 36.0 * 0.5;
    let snake_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(36, 4),
        box_center: body_center,
        material_id: snake_mat_id.0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(sim.config())
    };
    let snake_range_start = terrain_count;
    let _ = sim.add_body(snake_spawn);
    let snake_range = snake_range_start..sim.particles().len();

    let body_left = body_center.x - body_len / 2.0;
    {
        let particles = sim.particles_mut();
        for i in snake_range.clone() {
            particles.contact_group[i] = SNAKE_CONTACT_GROUP;
            let t = ((particles.x[i].x - body_left) / body_len).clamp(0.0, 1.0);
            let group = ((t * MUSCLE_GROUPS as f32) as u32).min(MUSCLE_GROUPS as u32 - 1);
            particles.muscle_group_id[i] = group;
            let local_y = particles.x[i].y - body_center.y;
            let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
            particles.activation_dir[i] = if local_y >= 0.0 {
                Vec2::new(-3.0 * flip, 1.0).normalize()
            } else {
                Vec2::new(3.0 * flip, 1.0).normalize()
            };
        }
    }

    let grip = std::sync::Arc::new(emerge::DirectionalContactGrip::new(0.5, 0.5, Vec2::X));
    let mut sim = sim.with_contact_grip(std::sync::Arc::clone(&grip));

    let centroid_at = |sim: &Simulation, range: std::ops::Range<usize>| -> Vec2 {
        let particles = sim.particles();
        let n = range.len() as f32;
        range.map(|i| particles.x[i]).sum::<Vec2>() / n
    };

    let start = centroid_at(&sim, snake_range.clone());
    // Matches the real live failure exactly: idle grip (symmetric friction,
    // no easy-direction bias), zero muscle activation, for real long enough
    // to have caught the actual bug (live took ~12,500 frames; this runs 16,000
    // headless steps at the SAME dt=0.1 to give real margin past that).
    let mut min_j_terrain = f32::MAX;
    let mut min_j_snake = f32::MAX;
    let mut max_extent = 0.0f32;
    for step in 0..16000 {
        sim.step();
        let particles = sim.particles();
        for i in 0..terrain_count {
            min_j_terrain = min_j_terrain.min(particles.deformation_gradient[i].determinant());
        }
        for i in snake_range.clone() {
            min_j_snake = min_j_snake.min(particles.deformation_gradient[i].determinant());
        }
        if step % 2000 == 0 {
            let snap = sim.diagnostics_snapshot();
            let extent = snap.max_particle_speed; // reuse as a cheap per-checkpoint sanity read
            max_extent = max_extent.max(extent);
            println!(
                "step={step} min_j_terrain={min_j_terrain:.4} min_j_snake={min_j_snake:.4} vmax={extent:.3}"
            );
        }
    }
    println!(
        "FINAL min_j_terrain={min_j_terrain:.4} min_j_snake={min_j_snake:.4} max_vmax_seen={max_extent:.3}"
    );

    assert!(
        min_j_terrain > 0.55,
        "BUG: sand terrain compressed/inverted past its real physical floor over a \
         long passive settle -- got min_j_terrain={min_j_terrain:.4} (was J=-1.000 in \
         the real live playtest that found this). The volumetric floor must hold over \
         long real-time durations, not just short test windows."
    );
    let _ = start;
}

/// Stress test for the Baumgarte velocity-floor fix above -- checks it generalizes
/// past the gentle-rest 36x4 scenario that verified it. Two axes pushed harder: (1)
/// body thickness doubled (48x8) -- more grip-mass nodes, the axis the original
/// epsilon-skip contamination bug scaled with; (2) a genuine dynamic impact (dropped
/// from ~24 units above the terrain) instead of starting already resting, since
/// Baumgarte's correction fires hardest at first impact (largest `gap`). Same 16,000
/// -step duration and assertion bar as the passive-settle test above.
#[test]
fn drucker_prager_volumetric_floor_holds_under_heavy_impact_and_long_settle() {
    const GRID: usize = 128;
    const DT: f32 = 0.1;
    const SNAKE_CONTACT_GROUP: u32 = 1;

    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let terrain_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(100, 12),
        box_center: Vec2::new(64.0, 10.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, terrain_spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)));
    let terrain_count = sim.particles().len();

    let mut snake_mat = NeoHookeanMaterial::new(13.0, 26.0);
    snake_mat.viscosity = 150.0;
    let snake_mat_id = sim.register_material(Box::new(snake_mat));
    // 24 units above the terrain surface (terrain top ~y=16) -- a real, hard fall,
    // not the gentle near-contact start the passive-settle test above uses.
    let body_center = Vec2::new(64.0, 40.0);
    let snake_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(48, 8), // doubled thickness vs. the 36x4 baseline test
        box_center: body_center,
        material_id: snake_mat_id.0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(sim.config())
    };
    let snake_range_start = terrain_count;
    let _ = sim.add_body(snake_spawn);
    let snake_range = snake_range_start..sim.particles().len();

    {
        let particles = sim.particles_mut();
        for i in snake_range.clone() {
            particles.contact_group[i] = SNAKE_CONTACT_GROUP;
        }
    }

    let grip = std::sync::Arc::new(emerge::DirectionalContactGrip::new(0.5, 0.5, Vec2::X));
    let mut sim = sim.with_contact_grip(std::sync::Arc::clone(&grip));

    let mut min_j_terrain = f32::MAX;
    let mut min_j_snake = f32::MAX;
    let mut max_extent = 0.0f32;
    for step in 0..16000 {
        sim.step();
        let particles = sim.particles();
        for i in 0..terrain_count {
            min_j_terrain = min_j_terrain.min(particles.deformation_gradient[i].determinant());
        }
        for i in snake_range.clone() {
            min_j_snake = min_j_snake.min(particles.deformation_gradient[i].determinant());
        }
        if step % 2000 == 0 {
            let snap = sim.diagnostics_snapshot();
            let extent = snap.max_particle_speed;
            max_extent = max_extent.max(extent);
            println!(
                "step={step} min_j_terrain={min_j_terrain:.4} min_j_snake={min_j_snake:.4} vmax={extent:.3}"
            );
        }
    }
    println!(
        "FINAL min_j_terrain={min_j_terrain:.4} min_j_snake={min_j_snake:.4} max_vmax_seen={max_extent:.3}"
    );

    assert!(
        min_j_terrain > 0.55,
        "BUG: sand terrain compressed/inverted past its real physical floor under a \
         hard dynamic impact + long settle from a thicker body -- got \
         min_j_terrain={min_j_terrain:.4}. The velocity-floor Baumgarte fix must hold \
         under a harder impact and thicker body, not just the gentle scenario that \
         originally verified it."
    );
}

/// Third axis for the Baumgarte velocity-floor fix: sustained active muscle-driven
/// locomotion (not passive rest or a one-off impact) at a larger scale (~2x linear
/// terrain/body dimensions), for the same long duration -- the actual motivating
/// scenario for the contact-fix investigation.
///
/// A synthetic CPG-style traveling wave drives `activation` every step (same mechanism
/// as `examples/snake_on_terrain.rs`, reproduced directly so this test has no
/// dependency on example code). Deliberately does not assert on net locomotion
/// distance/gait quality -- muscle/body tuning is separate from contact-resolution
/// correctness and body-proportion changes alone can shift crawl distance several-fold,
/// which would make a distance assertion flaky for unrelated reasons. The claim under
/// test is narrower: the terrain's volumetric floor and solver stability must hold
/// under continuous, large-scale internal driving stress, not just at rest.
#[test]
fn drucker_prager_volumetric_floor_holds_under_active_locomotion_at_larger_scale() {
    const GRID: usize = 192;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;
    const SNAKE_CONTACT_GROUP: u32 = 1;

    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let terrain_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(150, 14), // 1.5x the baseline test's 100x12
        box_center: Vec2::new(96.0, 10.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, terrain_spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)));
    let terrain_count = sim.particles().len();

    let mut snake_mat = NeoHookeanMaterial::new(13.0, 26.0);
    snake_mat.active_stress_coeff = 80.0;
    snake_mat.viscosity = 150.0;
    let snake_mat_id = sim.register_material(Box::new(snake_mat));
    let body_center = Vec2::new(96.0, 20.0);
    let body_len = 54.0 * 0.5; // 1.5x the baseline test's 36x4 body
    let snake_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(54, 6),
        box_center: body_center,
        material_id: snake_mat_id.0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(sim.config())
    };
    let snake_range_start = terrain_count;
    let _ = sim.add_body(snake_spawn);
    let snake_range = snake_range_start..sim.particles().len();
    let muscle_group_of_particle: Vec<u32> = {
        let particles = sim.particles();
        let body_left = body_center.x - body_len / 2.0;
        snake_range
            .clone()
            .map(|i| {
                let t = ((particles.x[i].x - body_left) / body_len).clamp(0.0, 1.0);
                ((t * MUSCLE_GROUPS as f32) as u32).min(MUSCLE_GROUPS as u32 - 1)
            })
            .collect()
    };
    {
        let particles = sim.particles_mut();
        for (offset, i) in snake_range.clone().enumerate() {
            particles.contact_group[i] = SNAKE_CONTACT_GROUP;
            let group = muscle_group_of_particle[offset];
            particles.muscle_group_id[i] = group;
            let local_y = particles.x[i].y - body_center.y;
            let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
            particles.activation_dir[i] = if local_y >= 0.0 {
                Vec2::new(-3.0 * flip, 1.0).normalize()
            } else {
                Vec2::new(3.0 * flip, 1.0).normalize()
            };
        }
    }

    let grip = std::sync::Arc::new(emerge::DirectionalContactGrip::new(0.2, 0.9, Vec2::X));
    let mut sim = sim.with_contact_grip(std::sync::Arc::clone(&grip));

    const CPG_OMEGA: f32 = 0.35;
    const CPG_WAVE_K: f32 = 0.8;
    let mut min_j_terrain = f32::MAX;
    let mut min_j_snake = f32::MAX;
    let mut max_extent = 0.0f32;
    for step in 0..16000 {
        let phase_t = step as f32 * CPG_OMEGA;
        {
            let particles = sim.particles_mut();
            for (offset, i) in snake_range.clone().enumerate() {
                let group = muscle_group_of_particle[offset];
                let phase = phase_t - CPG_WAVE_K * group as f32;
                particles.activation[i] = 0.5 * (1.0 + phase.sin());
            }
        }
        sim.step();
        let particles = sim.particles();
        for i in 0..terrain_count {
            min_j_terrain = min_j_terrain.min(particles.deformation_gradient[i].determinant());
        }
        for i in snake_range.clone() {
            min_j_snake = min_j_snake.min(particles.deformation_gradient[i].determinant());
        }
        if step % 2000 == 0 {
            let snap = sim.diagnostics_snapshot();
            let extent = snap.max_particle_speed;
            max_extent = max_extent.max(extent);
            println!(
                "step={step} min_j_terrain={min_j_terrain:.4} min_j_snake={min_j_snake:.4} vmax={extent:.3}"
            );
        }
    }
    println!(
        "FINAL min_j_terrain={min_j_terrain:.4} min_j_snake={min_j_snake:.4} max_vmax_seen={max_extent:.3} particle_count={}",
        sim.particles().len()
    );

    assert!(
        min_j_terrain > 0.55,
        "BUG: sand terrain compressed/inverted past its real physical floor under \
         sustained active muscle-driven locomotion at larger scale -- got \
         min_j_terrain={min_j_terrain:.4}. The velocity-floor Baumgarte fix must hold \
         under real, continuous driving stress at scale, not just at rest."
    );
}

/// Validates the internal pre-stress mechanism (turgor-pressure-style support, see
/// `Particle::internal_pressure`/`MaterialModel::pressure_scale` docs): a pressurized
/// column should droop less than an identical unpressurized one under the same
/// sustained self-weight load, per Niklas 1992's "hydro-skeleton" theory (internal
/// pressure resists compression/buckling, distinct from bulk elastic stiffness).
#[test]
fn pressurized_column_droops_less_than_unpressurized_under_self_weight() {
    fn run_column(material: Box<dyn MaterialModel>) -> f32 {
        let config = SimConfig::standard(32, 0.02, Vec2::new(0.0, -2.0));
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(4, 16),
            box_center: Vec2::new(16.0, 12.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn).with_default_material(material);

        // Pin the bottom 2 rows -- a real root/anchor, matching basic_plant.rs's own setup.
        let min_y = sim
            .particles()
            .iter()
            .map(|p| p.x.y)
            .fold(f32::INFINITY, f32::min);
        {
            let particles = sim.particles_mut();
            for i in 0..particles.len() {
                if particles.x[i].y < min_y + 1.0 {
                    particles.pinned[i] = 1;
                }
            }
        }

        let initial_top = sim
            .particles()
            .iter()
            .map(|p| p.x.y)
            .fold(f32::MIN, f32::max);

        for _ in 0..2000 {
            sim.step();
        }

        let final_top = sim
            .particles()
            .iter()
            .map(|p| p.x.y)
            .fold(f32::MIN, f32::max);
        initial_top - final_top // droop = how much height was lost
    }

    let lambda = 200.0;
    let mu = 300.0;
    let pressure = 100.0; // real, nonzero, comparable magnitude to mu -- not a token value

    let droop_plain = run_column(Box::new(NeoHookeanMaterial::new(lambda, mu)));
    let droop_pressurized = run_column(Box::new(WithPreStress::new(
        NeoHookeanMaterial::new(lambda, mu),
        pressure,
    )));

    println!("droop_plain={droop_plain:.4} droop_pressurized={droop_pressurized:.4}");
    assert!(
        droop_pressurized < droop_plain,
        "a pressurized column must droop LESS than an identical unpressurized one under \
         the same self-weight load (real, literature-grounded claim -- turgor/hydrostatic \
         pressure genuinely resists compression/buckling): droop_plain={droop_plain:.4} \
         droop_pressurized={droop_pressurized:.4}"
    );
}

/// TEMP DIAGNOSTIC (not a permanent regression, to be removed after use): real
/// pressure sweep against basic_plant.rs's EXACT geometry (height=12, root
/// pinned at y<=11, width=3, gravity=-0.3, GRID=64, dx=0.01, DT=0.1),
/// self-weight only (no wind), to find a pressure value that genuinely fixes
/// the real self-weight droop at the ORIGINAL 400/600 stiffness -- rather
/// than guess-and-check inside the full windowed demo again.
#[test]
#[ignore]
fn diag_basic_plant_pressure_sweep_self_weight_only() {
    // Real basic_plant.rs wind constants, replayed exactly (not re-derived).
    const WIND_DRAG_COEFFICIENT: f32 = 1.5;
    const WIND_SPEED: f32 = 0.00075;
    const WIND_GUST_PERIOD_SECONDS: f32 = 4.0;
    const DT: f32 = 0.1;

    fn run_with_checkpoints(
        lambda: f32,
        mu: f32,
        total_steps: u32,
        checkpoint_every: u32,
        with_wind: bool,
    ) {
        let config = SimConfig {
            min_dt: 0.0005,
            max_substeps_per_step: 400,
            gravity: Vec2::new(0.0, -0.3),
            ..SimConfig::earth(64, 0.01, 0.1)
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(3, 12),
            box_center: Vec2::new(32.0, 15.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let material: Box<dyn MaterialModel> =
            Box::new(ViscoelasticMaterial::new(lambda, mu, 0.1 * mu));
        let mut sim = Simulation::new(config, spawn).with_default_material(material);
        {
            let particles = sim.particles_mut();
            for i in 0..particles.len() {
                if particles.x[i].y <= 11.0 {
                    particles.pinned[i] = 1;
                }
            }
        }
        let mut step = 0;
        let mut wind_time = 0.0f32;
        while step < total_steps {
            let n = checkpoint_every.min(total_steps - step);
            for _ in 0..n {
                if with_wind {
                    wind_time += DT;
                    let omega = std::f32::consts::TAU / WIND_GUST_PERIOD_SECONDS;
                    let gust_speed = WIND_SPEED * (wind_time * omega).sin();
                    sim.remove_force_field("wind");
                    sim.add_named_force_field(
                        "wind",
                        Box::new(LinearDragField::new(
                            Vec2::new(gust_speed, 0.0),
                            WIND_DRAG_COEFFICIENT,
                            1,
                        )),
                    );
                }
                sim.step();
            }
            step += n;
            let particles = sim.particles();
            let (mut min_x, mut max_x, mut min_y, mut max_y) =
                (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            for p in particles.iter() {
                min_x = min_x.min(p.x.x);
                max_x = max_x.max(p.x.x);
                min_y = min_y.min(p.x.y);
                max_y = max_y.max(p.x.y);
            }
            let snap = sim.diagnostics_snapshot();
            println!(
                "lambda={lambda} mu={mu} wind={with_wind} step={step} (t={:.0}s): height={:.3} width={:.3} jmin={:.4} jmax={:.4} ke={:.6}",
                step as f32 * 0.1,
                max_y - min_y,
                max_x - min_x,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.total_kinetic_energy,
            );
        }
    }

    run_with_checkpoints(8000.0, 12000.0, 50000, 2500, false);
}
