use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::boundary::BoundaryCondition;
use crate::materials::registry::MaterialRegistry;
use crate::materials::{ConstitutiveModel, MaterialModel};
use crate::solver::config::KERNEL_D_INVERSE;
use crate::{grid::Grid, grid::kernel::quadratic_weights, particle::Particles};

/// Elastic/plastic Kirchhoff stress plus the active-stress (muscle contraction) term, if any.
///
/// Single source of truth for "what stress does this particle contribute to P2G" — shared by
/// `scatter_particles_to_grid` and tests, so the two can never drift apart. Mirrors the GPU
/// shader's post-switch active-stress block in `p2g.wgsl` exactly: Viscoelastic uses an
/// isotropic contractile term (matches its own Kelvin-Voigt formulation), every other elastic
/// model uses the directional F·(n₀⊗n₀)·Fᵀ fiber form (follows material deformation).
pub(crate) fn combined_kirchhoff_stress(
    material: &dyn MaterialModel,
    particles: &Particles,
    i: usize,
) -> Mat2 {
    let tau = material.kirchhoff_stress(particles, i);
    let coeff = material.activation_scale();
    if particles.activation[i] <= 0.0 || coeff <= 0.0 {
        return tau;
    }
    let isotropic = material.constitutive_model() == ConstitutiveModel::Viscoelastic;
    let tau_active = if isotropic {
        Mat2::from_diagonal(Vec2::splat(particles.activation[i] * coeff))
    } else {
        let n = particles.activation_dir[i];
        let len_sq = n.dot(n);
        if len_sq > f32::EPSILON {
            let n0 = n / len_sq.sqrt();
            let n_outer = Mat2::from_cols(n0 * n0.x, n0 * n0.y);
            let a_mat = n_outer * (particles.activation[i] * coeff);
            let f = particles.deformation_gradient[i];
            f * a_mat * f.transpose()
        } else {
            Mat2::from_diagonal(Vec2::splat(particles.activation[i] * coeff))
        }
    };
    tau + tau_active
}

/// P2G: scatter particle mass, momentum, and stress forces onto the grid (MLS-MPM, Hu 2018 §4).
///
/// Stress is pre-integrated as a momentum impulse so the grid needs one accumulation pass.
/// The APIC affine term conserves angular momentum without a correction step.
///
/// NOT parallelized (unlike G2P below): multiple particles write to the same grid cell (3×3
/// B-spline stencils overlap), so summing their contributions requires either a shared mutable
/// map (unsound across threads — `HashMap::entry()` can trigger a resize) or a thread-local
/// fold/reduce merge. The latter was attempted and reverted 2026-06-20: it's safe and compiles
/// clean, but changes floating-point summation order across particles sharing a cell, and that
/// shifted results enough to break `fluid_spreads_more_than_elastic_under_gravity` (a 600-step
/// chaotic simulation) — confirmed by isolated A/B, not assumed. Reverted rather than accepted
/// the correctness risk for an unmeasured gain.
pub fn scatter_particles_to_grid(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
    active_count: usize,
) {
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);
        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];

        let stress = combined_kirchhoff_stress(material, particles, i);
        let stress_coeff = -material.stress_volume(particles, i) * KERNEL_D_INVERSE * dt;

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let momentum = weight
                    * (mass_i * (v_i + c_i * cell_dist) + stress_coeff * (stress * cell_dist));
                grid.add_mass_momentum(cell_pos, weight * mass_i, momentum);
            }
        }
    }
}

/// Analytic adjoint of P2G's stress→force scatter contribution w.r.t. the
/// particle's own Kirchhoff stress tensor -- the second real piece of
/// differentiable stepping, after `NeoHookeanMaterial::kirchhoff_stress_vjp`.
///
/// SCOPED, not a full P2G adjoint: differentiates only the elastic-force term
/// `weight * stress_coeff * (stress * cell_dist)` inside `scatter_particles_to_grid`,
/// treating the particle's position `x` (and therefore the kernel weights and
/// `cell_dist`) as FIXED. The mass/velocity/affine-C term is untouched here --
/// a separate, much simpler linear adjoint, not yet implemented. Differentiating
/// through the kernel weights' own dependence on `x` (how MOVING the particle
/// changes which cells it deposits to, and by how much) is the real remaining
/// gap in a fully general P2G adjoint -- deliberately deferred, not silently
/// dropped: this covers the actual control-relevant path (muscle activation →
/// stress → grid force) needed to train a controller, without yet handling
/// the harder position-dependence.
///
/// Real derivation: for one particle, cell `c`'s momentum contribution from
/// stress is `y_c = (weight_c * stress_coeff) * (stress * cell_dist_c)` --
/// linear in `stress`, a matrix-vector product `y = M*v` scaled by a fixed
/// scalar. Given the gradient flowing back from each cell's grid momentum,
/// `d_loss_d_momentum[c]` (a Vec2), the standard VJP for `y=Mv` is
/// `dL/dM = outer(dL/dy, v)`, i.e. `dL/dM_kl = dL/dy_k * v_l`. Summed over
/// all 9 stencil cells:
///
///   d_loss_d_stress = sum_c (weight_c * stress_coeff) * outer(d_loss_d_momentum[c], cell_dist_c)
///
/// Returns d_loss_d_stress, ready to feed into e.g.
/// `NeoHookeanMaterial::kirchhoff_stress_vjp` to continue the chain back to F.
/// Verified against central-difference numerical gradients in this module's
/// own tests, same non-negotiable discipline as the stress adjoint itself.
pub fn p2g_stress_vjp(x: Vec2, stress_coeff: f32, d_loss_d_momentum: &[[Vec2; 3]; 3]) -> Mat2 {
    let weights = quadratic_weights(x);
    let mut d_loss_d_stress = Mat2::ZERO;
    for (gx, (wx, momentum_row)) in weights.wx.iter().zip(d_loss_d_momentum.iter()).enumerate() {
        for (gy, (wy, &g)) in weights.wy.iter().zip(momentum_row.iter()).enumerate() {
            let weight = wx * wy;
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            let scalar = weight * stress_coeff;
            // outer(g, cell_dist): column 0 = cell_dist.x * g, column 1 = cell_dist.y * g
            // (matches glam's column-major Mat2, verified against the matrix-vector
            // VJP already proven correct in kirchhoff_stress_vjp).
            d_loss_d_stress += scalar * Mat2::from_cols(cell_dist.x * g, cell_dist.y * g);
        }
    }
    d_loss_d_stress
}

pub struct G2PParams {
    pub vel_limit: f32,
    pub apic_blend: f32,
    pub active_count: usize,
}

/// Analytic adjoint of G2P's velocity gather (`new_v = sum_c weight_c *
/// grid.velocity_at(cell_c)`, see `gather_grid_to_particles`'s Phase 1) w.r.t.
/// the 9 grid velocities in the particle's stencil -- fifth real piece of
/// differentiable stepping, and the mathematical transpose of
/// `p2g_stress_vjp`: same quadratic kernel weights, same 3x3 stencil, but
/// scattering a gradient back out to the grid instead of gathering a value in
/// from it (the well-known P2G/G2P transpose relationship in MPM literature,
/// e.g. Jiang et al. 2016 "The Material Point Method for Simulating
/// Continuum Materials", carries over directly to differentiation).
///
/// SCOPED, matching the P2G adjoint's own scoping: treats particle position
/// `x` (and therefore the kernel weights) as FIXED. Also covers only the new
/// velocity `new_v`, not the APIC affine matrix `b`/`velocity_gradient` G2P
/// computes alongside it (`b = sum_c weight_c * outer(v_grid_c, dist_c)`) --
/// a related, still-open piece: same per-cell structure, needs its own
/// derivation and verification, not silently folded in here. Also doesn't
/// cover the velocity clamp or position boundary-clamp applied after this in
/// the real G2P (piecewise/conditional, same deferred-with-a-name status as
/// grid update's boundary/clamp gap).
///
/// Given the gradient flowing back from the particle's new velocity,
/// `d_loss_d_new_v` (a Vec2), the adjoint of a weighted sum distributes it
/// back to each grid cell by the SAME weight it was gathered with:
///
///   d_loss_d_v_grid[c] = weight_c * d_loss_d_new_v
///
/// Returns the per-cell gradient in the same `[[Vec2; 3]; 3]` shape
/// `p2g_stress_vjp` consumes, so a real trainer can pass this straight
/// through to the P2G side once both meet at the same grid cells. Verified
/// against central-difference numerical gradients in this module's own
/// tests.
pub fn g2p_velocity_vjp(x: Vec2, d_loss_d_new_v: Vec2) -> [[Vec2; 3]; 3] {
    let weights = quadratic_weights(x);
    let mut out = [[Vec2::ZERO; 3]; 3];
    for (row, wx) in out.iter_mut().zip(weights.wx.iter()) {
        for (cell, wy) in row.iter_mut().zip(weights.wy.iter()) {
            *cell = (wx * wy) * d_loss_d_new_v;
        }
    }
    out
}

/// G2P: read grid velocities back into particles, advance state, apply boundaries.
/// Returns the number of particles whose velocity was clamped to `vel_limit`.
pub fn gather_grid_to_particles(
    particles: &mut Particles,
    grid: &Grid,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    materials: &MaterialRegistry,
    params: G2PParams,
) -> usize {
    let G2PParams {
        vel_limit,
        apic_blend,
        active_count,
    } = params;
    let grid_res = grid.resolution();

    // Phase 1 (parallel): grid gather -> v, velocity_gradient, position advance + boundary
    // position clamp. Pure math over read-only grid/boundary state, writing only the calling
    // particle's own x/v/velocity_gradient — no cross-particle data dependency, so disjoint
    // per-field slices can be processed concurrently (gather passes are race-free by
    // construction; see Gao et al. 2018, "GPU Optimization of Material Point Methods").
    let xs = &mut particles.x[..active_count];
    let vs = &mut particles.v[..active_count];
    let vgs = &mut particles.velocity_gradient[..active_count];

    let clamp_count: usize = xs
        .par_iter_mut()
        .zip(vs.par_iter_mut())
        .zip(vgs.par_iter_mut())
        .map(|((x, v), vg)| {
            let weights = quadratic_weights(*x);
            let mut new_v = Vec2::ZERO;
            let mut b = Mat2::ZERO;

            for gx in 0..3 {
                for gy in 0..3 {
                    let weight = weights.wx[gx] * weights.wy[gy];
                    let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                    let dist = cell_pos.as_vec2() - *x + Vec2::splat(0.5);
                    let weighted_velocity = grid.velocity_at(cell_pos) * weight;
                    let term =
                        Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                    b += term;
                    new_v += weighted_velocity;
                }
            }

            // Hard speed cap — CFL in choose_substep_dt is the physics-grounded bound.
            // This fires only when CFL is violated despite the timestep limiter (e.g. first
            // substep of a high-energy spawn). Magnitude clamp preserves direction; no
            // anisotropic bias unlike per-component clamping.
            let spd = new_v.length();
            let clamped = if spd > vel_limit {
                new_v *= vel_limit / spd;
                1
            } else {
                0
            };

            // Apply all boundaries' position clamp (pure function, no particle-struct access).
            let mut new_pos = *x + new_v * dt;
            for boundary in boundaries.iter() {
                new_pos = boundary.clamp_particle_position(new_pos, grid_res);
            }

            *v = new_v;
            *vg = b * KERNEL_D_INVERSE * apic_blend;
            *x = new_pos;
            clamped
        })
        .sum();

    // Phase 2 (sequential): plasticity update + boundary post-hooks need whole-`Particles`
    // mutable access (deformation_gradient, hardening_scale, etc. per material) — not
    // split-borrow-friendly without a larger `MaterialModel` trait redesign, so kept sequential.
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);
        material.update_particle(particles, i, dt);
        for boundary in boundaries.iter() {
            boundary.post_g2p_particle(particles, i, grid_res, dt);
        }
    }

    clamp_count
}

pub fn scatter_particle_mass(particles: &Particles, grid: &mut Grid, active_count: usize) {
    for i in 0..active_count {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                grid.add_mass_momentum(cell_pos, weight * mass, Vec2::ZERO);
            }
        }
    }
}

#[cfg(test)]
mod activation_tests {
    use super::combined_kirchhoff_stress;
    use crate::materials::{NeoHookeanMaterial, ViscoelasticMaterial};
    use crate::particle::{Particle, Particles};
    use glam::{Mat2, Vec2};

    fn particle_at_rest() -> Particle {
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        p.deformation_gradient = Mat2::IDENTITY; // undeformed: passive elastic stress is exactly zero
        p
    }

    /// Directional materials (everything except Viscoelastic): active stress follows the fiber
    /// direction exactly — `activation * coeff` along the fiber axis, zero perpendicular to it.
    #[test]
    fn directional_active_stress_follows_fiber_axis() {
        let mut mat = NeoHookeanMaterial::new(100.0, 200.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 1.0;
        p.activation_dir = Vec2::X;

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);

        assert!(
            (tau.x_axis.x - 10.0).abs() < 1e-5,
            "tau_xx should be activation*coeff=10: {tau:?}"
        );
        assert!(
            tau.y_axis.y.abs() < 1e-5,
            "tau_yy should stay ~0 (perpendicular to fiber): {tau:?}"
        );
    }

    /// Viscoelastic uses an isotropic active term (matches its Kelvin-Voigt formulation and the
    /// GPU shader's `model == 9u` special case) — equal on both diagonal axes, regardless of
    /// `activation_dir`.
    #[test]
    fn viscoelastic_active_stress_is_isotropic() {
        let mut mat = ViscoelasticMaterial::new(100.0, 200.0, 0.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 1.0;
        p.activation_dir = Vec2::X; // must NOT bias the result toward x for this material

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);

        assert!(
            (tau.x_axis.x - 10.0).abs() < 1e-5,
            "tau_xx should be activation*coeff=10: {tau:?}"
        );
        assert!(
            (tau.y_axis.y - 10.0).abs() < 1e-5,
            "tau_yy should equal tau_xx (isotropic, not directional): {tau:?}"
        );
    }

    /// Regression: `ViscoelasticMaterial::kirchhoff_stress` used to add its own isotropic active
    /// term directly AND report a non-zero `activation_scale()`, so the shared P2G path
    /// (`combined_kirchhoff_stress`) added a second active term on top — silently doubling muscle
    /// stress for any Viscoelastic creature body. Pin the total to exactly one contribution.
    #[test]
    fn viscoelastic_active_stress_is_not_double_counted() {
        let mut mat = ViscoelasticMaterial::new(100.0, 200.0, 0.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 1.0;
        p.activation_dir = Vec2::X;

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);
        let expected_single = 10.0; // activation(1.0) * coeff(10.0), applied exactly once
        assert!(
            (tau.x_axis.x - expected_single).abs() < 1e-5,
            "active stress must be applied exactly once, not doubled: tau_xx={}, expected={expected_single}",
            tau.x_axis.x
        );
    }

    #[test]
    fn zero_activation_leaves_stress_unchanged() {
        let mut mat = NeoHookeanMaterial::new(100.0, 200.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 0.0; // off — must be a true no-op regardless of coeff
        p.activation_dir = Vec2::X;

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);
        assert!(
            tau.x_axis.x.abs() < 1e-6 && tau.y_axis.y.abs() < 1e-6,
            "activation=0.0 must produce zero stress on an undeformed particle: {tau:?}"
        );
    }
}

#[cfg(test)]
mod p2g_stress_vjp_tests {
    use super::*;

    /// Recomputes just the stress-scatter contribution `scatter_particles_to_grid`
    /// itself computes for each of the 9 stencil cells, at a given `stress` --
    /// the exact forward formula `p2g_stress_vjp` is the adjoint of, isolated
    /// from mass/velocity/C so the finite-difference check exercises only the
    /// piece being verified.
    fn stress_contributions(x: Vec2, stress_coeff: f32, stress: Mat2) -> [[Vec2; 3]; 3] {
        let weights = quadratic_weights(x);
        let mut out = [[Vec2::ZERO; 3]; 3];
        for (gx, row) in out.iter_mut().enumerate() {
            for (gy, cell) in row.iter_mut().enumerate() {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                *cell = weight * stress_coeff * (stress * cell_dist);
            }
        }
        out
    }

    /// Scalar loss L(stress) = sum_c g_c . contribution_c(stress) -- the
    /// standard way to check a matrix-to-many-vectors function one scalar
    /// component at a time via central differences.
    fn loss(x: Vec2, stress_coeff: f32, stress: Mat2, g: &[[Vec2; 3]; 3]) -> f32 {
        let contributions = stress_contributions(x, stress_coeff, stress);
        let mut total = 0.0;
        for (row, contrib_row) in g.iter().zip(contributions.iter()) {
            for (gv, cv) in row.iter().zip(contrib_row.iter()) {
                total += gv.dot(*cv);
            }
        }
        total
    }

    /// Bundles the fixed context (position, stress_coeff, base stress, incoming
    /// gradients, step size) shared by every one of F's 4 components' checks.
    struct FiniteDiffContext {
        x: Vec2,
        stress_coeff: f32,
        stress: Mat2,
        g: [[Vec2; 3]; 3],
        h: f32,
    }

    fn check_component(
        ctx: &FiniteDiffContext,
        label: &str,
        analytic_val: f32,
        base: f32,
        set: impl Fn(&mut Mat2, f32),
    ) {
        let mut s_plus = ctx.stress;
        set(&mut s_plus, base + ctx.h);
        let mut s_minus = ctx.stress;
        set(&mut s_minus, base - ctx.h);

        let numeric = (loss(ctx.x, ctx.stress_coeff, s_plus, &ctx.g)
            - loss(ctx.x, ctx.stress_coeff, s_minus, &ctx.g))
            / (2.0 * ctx.h);

        let diff = (numeric - analytic_val).abs();
        let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "p2g_stress_vjp mismatch at {label}: analytic={analytic_val:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e} (x={:?})",
            diff / scale,
            ctx.x
        );
    }

    fn check_matches_finite_difference(
        x: Vec2,
        stress_coeff: f32,
        stress: Mat2,
        g: [[Vec2; 3]; 3],
    ) {
        let analytic = p2g_stress_vjp(x, stress_coeff, &g);
        let ctx = FiniteDiffContext {
            x,
            stress_coeff,
            stress,
            g,
            h: 1.0e-3_f32,
        };

        check_component(
            &ctx,
            "F[0][0]",
            analytic.x_axis.x,
            stress.x_axis.x,
            |m, v| m.x_axis.x = v,
        );
        check_component(
            &ctx,
            "F[1][0]",
            analytic.x_axis.y,
            stress.x_axis.y,
            |m, v| m.x_axis.y = v,
        );
        check_component(
            &ctx,
            "F[0][1]",
            analytic.y_axis.x,
            stress.y_axis.x,
            |m, v| m.y_axis.x = v,
        );
        check_component(
            &ctx,
            "F[1][1]",
            analytic.y_axis.y,
            stress.y_axis.y,
            |m, v| m.y_axis.y = v,
        );
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check_matches_finite_difference(
            Vec2::new(10.0, 10.0),
            -0.5,
            Mat2::from_cols(Vec2::new(3.0, 0.5), Vec2::new(0.5, -2.0)),
            [[Vec2::new(1.0, 0.5); 3]; 3],
        );
    }

    #[test]
    fn matches_finite_difference_off_center_with_varied_gradients() {
        // Off-center position (nonzero fractional offset within its cell) and a
        // different, non-uniform incoming gradient per stencil cell -- exercises
        // real per-cell weight/cell_dist variation, not just a symmetric case.
        let g = [
            [
                Vec2::new(0.3, -0.7),
                Vec2::new(1.1, 0.2),
                Vec2::new(-0.4, 0.9),
            ],
            [
                Vec2::new(0.8, 0.1),
                Vec2::new(-0.2, -0.5),
                Vec2::new(0.6, 1.3),
            ],
            [
                Vec2::new(-1.0, 0.4),
                Vec2::new(0.2, -0.9),
                Vec2::new(0.5, 0.5),
            ],
        ];
        check_matches_finite_difference(
            Vec2::new(15.35, 22.78),
            0.8,
            Mat2::from_cols(Vec2::new(-1.5, 2.0), Vec2::new(0.9, 1.2)),
            g,
        );
    }

    #[test]
    fn chains_correctly_into_neohookean_kirchhoff_stress_vjp() {
        // Real end-to-end check: P2G's stress gradient feeds NeoHookean's own
        // F-adjoint, and the composed result still matches a finite-difference
        // taken all the way from F, through stress, through the P2G scatter --
        // proves the two pieces compose correctly, not just individually.
        use crate::materials::NeoHookeanMaterial;
        use crate::particle::{Particle, Particles};

        let mat = NeoHookeanMaterial::new(900.0, 700.0);
        let f = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(-0.05, 0.9));
        let x = Vec2::new(8.4, 12.6);
        let stress_coeff = -0.3;
        let g = [[Vec2::new(0.4, -0.6); 3]; 3];

        let particle_with_f = |f: Mat2| -> Particles {
            let mut particles = Particles::default();
            particles.push(Particle {
                x,
                v: Vec2::ZERO,
                velocity_gradient: Mat2::ZERO,
                deformation_gradient: f,
                mass: 1.0,
                initial_volume: 1.0,
                volume: 1.0,
                density: 1.0,
                material_id: 0,
                plastic_volume_ratio: 1.0,
                hardening_scale: 1.0,
                friction_hardening: 0.0,
                log_volume_strain: 0.0,
                temperature: 0.0,
                user_tag: 0,
                activation: 0.0,
                activation_dir: Vec2::ZERO,
                muscle_group_id: 0,
                sleeping: 0,
            });
            particles
        };

        let end_to_end_loss = |f: Mat2| -> f32 {
            let particles = particle_with_f(f);
            let stress = mat.kirchhoff_stress(&particles, 0);
            let contributions = stress_contributions(x, stress_coeff, stress);
            let mut total = 0.0;
            for (row, contrib_row) in g.iter().zip(contributions.iter()) {
                for (gv, cv) in row.iter().zip(contrib_row.iter()) {
                    total += gv.dot(*cv);
                }
            }
            total
        };

        // Composed analytic gradient: P2G adjoint -> NeoHookean adjoint.
        let particles = particle_with_f(f);
        let stress = mat.kirchhoff_stress(&particles, 0);
        let d_loss_d_stress = p2g_stress_vjp(x, stress_coeff, &g);
        let composed = mat.kirchhoff_stress_vjp(&particles, 0, d_loss_d_stress);
        let _ = stress; // used only to construct d_loss_d_stress's context above

        let h = 1.0e-3_f32;
        let mut f_plus = f;
        f_plus.x_axis.x += h;
        let mut f_minus = f;
        f_minus.x_axis.x -= h;
        let numeric = (end_to_end_loss(f_plus) - end_to_end_loss(f_minus)) / (2.0 * h);

        let diff = (numeric - composed.x_axis.x).abs();
        let scale = numeric.abs().max(composed.x_axis.x.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "composed P2G+NeoHookean adjoint must match end-to-end finite difference: \
             analytic={:.6} numeric={numeric:.6} relative_diff={:.2e}",
            composed.x_axis.x,
            diff / scale
        );
    }
}

#[cfg(test)]
mod g2p_velocity_vjp_tests {
    use super::*;

    /// Forward formula exactly matching G2P's own `new_v` computation (the
    /// weighted sum over the 3x3 stencil), taking the 9 grid velocities
    /// directly as an array instead of reading a real `Grid` -- isolates the
    /// weighted-sum math being verified from grid storage/lookup entirely.
    fn gather_velocity(x: Vec2, v_grid: &[[Vec2; 3]; 3]) -> Vec2 {
        let weights = quadratic_weights(x);
        let mut new_v = Vec2::ZERO;
        for (row, wx) in v_grid.iter().zip(weights.wx.iter()) {
            for (v_cell, wy) in row.iter().zip(weights.wy.iter()) {
                new_v += (wx * wy) * *v_cell;
            }
        }
        new_v
    }

    fn loss(x: Vec2, v_grid: &[[Vec2; 3]; 3], g: Vec2) -> f32 {
        g.dot(gather_velocity(x, v_grid))
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check(Vec2::new(20.0, 20.0), Vec2::new(0.6, -0.4));
    }

    #[test]
    fn matches_finite_difference_off_center() {
        check(Vec2::new(7.35, 41.82), Vec2::new(-1.1, 0.9));
    }

    /// Checks every one of the 9 stencil cells' 2 velocity components (18
    /// scalars total) against central differences -- the full adjoint output,
    /// not just a sample of it.
    fn check(x: Vec2, g: Vec2) {
        let v_grid = [
            [
                Vec2::new(0.3, 0.1),
                Vec2::new(-0.2, 0.5),
                Vec2::new(0.7, -0.6),
            ],
            [
                Vec2::new(-0.4, 0.2),
                Vec2::new(0.1, -0.3),
                Vec2::new(0.5, 0.4),
            ],
            [
                Vec2::new(0.2, -0.5),
                Vec2::new(-0.6, 0.3),
                Vec2::new(0.4, 0.1),
            ],
        ];
        let analytic = g2p_velocity_vjp(x, g);
        let h = 1.0e-3_f32;

        for gx in 0..3 {
            for gy in 0..3 {
                for (axis, label) in [(0, "x"), (1, "y")] {
                    let mut v_plus = v_grid;
                    let mut v_minus = v_grid;
                    if axis == 0 {
                        v_plus[gx][gy].x += h;
                        v_minus[gx][gy].x -= h;
                    } else {
                        v_plus[gx][gy].y += h;
                        v_minus[gx][gy].y -= h;
                    }
                    let numeric = (loss(x, &v_plus, g) - loss(x, &v_minus, g)) / (2.0 * h);
                    let analytic_val = if axis == 0 {
                        analytic[gx][gy].x
                    } else {
                        analytic[gx][gy].y
                    };
                    let diff = (numeric - analytic_val).abs();
                    let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                    assert!(
                        diff / scale < 1.0e-2,
                        "g2p_velocity_vjp mismatch at cell[{gx}][{gy}].{label}: \
                         analytic={analytic_val:.6} numeric={numeric:.6} \
                         relative_diff={:.2e} (x={x:?})",
                        diff / scale
                    );
                }
            }
        }
    }
}
