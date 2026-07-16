use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::boundary::BoundaryCondition;
use crate::materials::registry::MaterialRegistry;
use crate::materials::{ConstitutiveModel, MaterialModel};
use crate::solver::config::KERNEL_D_INVERSE;
use crate::{
    grid::Grid,
    grid::kernel::{axis_weights_derivative, quadratic_weights},
    particle::Particles,
};

/// Elastic/plastic Kirchhoff stress plus the active-stress (muscle contraction) term, if any.
///
/// KNOWN OPEN BUG (found 2026-07-11, still not fixed despite real, repeated effort): a driven
/// creature body settles into a real, unbounded compaction ratchet over long horizons — net
/// drift collapses to ~0 while min(J) keeps falling and never recovers. FIVE distinct real
/// fixes were tried and empirically falsified (each via a real 16,000-20,000-step headless
/// sweep on `basic_creature`'s exact Simulation/RatchetFrictionBoundary/NeoHookeanMaterial
/// setup, not guessed):
///   1. Higher material stiffness — only delays onset (6500 -> 13000 steps), same collapse.
///   2. Lower `apic_blend` (numerical PIC damping) — same, only delays onset.
///   3. Signed [-1,1] activation, naive `2*sigmoid-1` remap — WORSE: real instability
///      (min(J) toward the numerical floor, max(J) past 3.0), because it also doubled the
///      drive amplitude, not a clean test of signedness alone.
///   4. `NeoHookeanMaterial`'s volumetric Kirchhoff term was ALSO a real, separate,
///      independently-worth-fixing bug: it used a bounded `k/2*(J²-1)` (finite ceiling on
///      compression resistance) where Simo & Pister's actual 1984 formulation (which the
///      old doc already cited but didn't implement) uses the log-barrier `k*(ln J)²`
///      potential (τ_vol = k·ln(J), diverges as J→0, genuinely unbounded resistance). Fixed
///      in `kirchhoff_stress`/`kirchhoff_stress_vjp` below. Real, legitimate, kept -- but
///      verified NOT sufficient alone: the same 20,000-step creature sweep still stalls,
///      just with a somewhat different J-trajectory. `MIN_J` (1e-6) was checked and ruled
///      out as an interfering clamp -- min(J) in these runs never gets within three orders
///      of magnitude of it.
///   5. Signed activation retried with amplitude MATCHED to the unsigned case (span 0.9
///      either way, not doubled) -- still stalls (drift ~0 by step ~2500), though without
///      the earlier catastrophic collapse; max(J) still drifts upward over time (up to 3+).
///
/// Separately confirmed via a passive (zero-activation) body: min(J) settles to a FIXED
/// value and velocity decays cleanly to exactly 0 -- the core P2G/G2P/F-update solver is NOT
/// numerically drifting on its own. This is specific to the muscle-driven cyclic-loading +
/// directional-friction interaction, not a general integration artifact. Root cause remains
/// genuinely unsolved; a real fix likely needs rethinking the friction/actuation mechanism
/// itself (e.g. a redesigned contact model, or a controller that never enters the failure
/// regime) rather than another parameter or activation-scheme tweak.
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

/// Analytic adjoint of the DIRECTIONAL active-stress term
/// `combined_kirchhoff_stress` adds on top of a material's passive
/// `kirchhoff_stress` -- `tau_active = activation*coeff*F*A*Fᵀ`, where
/// `A = n0⊗n0` (fiber direction outer product, symmetric, fixed for a given
/// particle). Needed to train `activation` itself via gradient descent (the
/// actual trainable control signal for muscle-driven locomotion), not just
/// F -- `kirchhoff_stress_vjp` only covers the passive term.
///
/// SCOPED to the directional case only (every material except Viscoelastic's
/// isotropic branch) -- matches what a real trained creature body actually
/// uses (fiber-directed contraction); the isotropic branch is a much simpler
/// constant-diagonal term not needed here.
///
/// Since `A` is symmetric and `tau_active` is linear in `activation`, this
/// is the exact same `Y=F*A*Fᵀ` shape as `kirchhoff_stress_vjp`'s own `B=F*Fᵀ`
/// term (that's the `A=I` special case) -- so its adjoint follows the same
/// derivation: `dL/dF = (Ḡ+Ḡᵀ)*F*A` (using `A=Aᵀ` to combine the two `Y=F*A*Fᵀ`
/// product-rule terms into one). `dL/d(activation)` is just the Frobenius
/// inner product against `tau_active/activation` (linear in the scalar, so
/// its own derivative is that same fixed matrix): `dL/d(activation) = coeff *
/// (Ḡ : F*A*Fᵀ)`.
///
/// The SAME `d_loss_d_tau` gradient that feeds `kirchhoff_stress_vjp` feeds
/// this too -- `tau = tau_passive + tau_active` is a plain sum, whose adjoint
/// passes the incoming gradient through to BOTH summands unchanged, so a real
/// trainer calls both functions with the same `g` and adds their F-gradients.
///
/// Verified against central-difference numerical gradients in this module's
/// own tests.
pub fn active_stress_vjp(
    f: Mat2,
    activation: f32,
    coeff: f32,
    fiber_dir: Vec2,
    d_loss_d_tau: Mat2,
) -> (Mat2, f32) {
    let len_sq = fiber_dir.dot(fiber_dir);
    if len_sq <= f32::EPSILON || activation <= 0.0 || coeff <= 0.0 {
        return (Mat2::ZERO, 0.0);
    }
    let n0 = fiber_dir / len_sq.sqrt();
    let a_mat = Mat2::from_cols(n0 * n0.x, n0 * n0.y);

    let g = d_loss_d_tau;
    let k_mat = f * a_mat * f.transpose(); // tau_active / (activation*coeff)
    let d_loss_d_activation = coeff
        * (g.x_axis.x * k_mat.x_axis.x
            + g.x_axis.y * k_mat.x_axis.y
            + g.y_axis.x * k_mat.y_axis.x
            + g.y_axis.y * k_mat.y_axis.y);

    let g_sym = g + g.transpose();
    let d_loss_d_f = (activation * coeff) * (g_sym * f * a_mat);

    (d_loss_d_f, d_loss_d_activation)
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
        let contact_group = particles.contact_group[i];

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
                // Additive second scatter for multi-field contact (Bardenhagen 2001) —
                // see `Particle::contact_group` doc. A no-op call for every particle
                // with contact_group == 0 (the default, i.e. every scene that doesn't
                // use this feature): `Grid::add_grip_mass_momentum` just never gets
                // called, so there's no extra work, not even an empty branch, for the
                // common case.
                if contact_group != 0 {
                    grid.add_grip_mass_momentum(cell_pos, weight * mass_i, momentum);
                }
            }
        }
    }
}

/// Gathers the labeled particle point cloud (`+1.0` grip / `-1.0` rest) that
/// `Grid::resolve_contact`'s logistic-regression normal fit (`fit_contact_normal_lr`)
/// needs, at every node `scatter_particles_to_grid` already marked contact-active.
///
/// Deliberately a SECOND pass over particles, not merged into `scatter_particles_to_grid`
/// above: which nodes are contact-active isn't fully known until that first pass has
/// scattered every grip particle's mass, and `Grid::add_contact_point` only appends to a
/// node that already exists in `contact_cells` (never creates one) — so running this
/// before the first pass completes would silently miss point-cloud data for nodes whose
/// grip contribution hadn't been seen yet. Gated on `grid.has_contact_activity()`: a full
/// no-op, not even a loop iteration, for every scene that never sets
/// `Particle::contact_group` — the same zero-cost-when-unused property as the rest of
/// this feature.
pub fn gather_contact_point_cloud(particles: &Particles, grid: &mut Grid, active_count: usize) {
    if !grid.has_contact_activity() {
        return;
    }
    for i in 0..active_count {
        let x = particles.x[i];
        let label = if particles.contact_group[i] != 0 {
            1.0
        } else {
            -1.0
        };
        let weights = quadratic_weights(x);
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let cell_pos = weights.base_cell + IVec2::new(gx - 1, gy - 1);
                grid.add_contact_point(cell_pos, x, label);
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

/// Analytic adjoint of P2G's FULL forward pass (`scatter_particles_to_grid`)
/// w.r.t. the particle's own position `x` -- the last confirmed-real gap,
/// now closed for P2G. Combines `axis_weights_derivative` (the kernel's own
/// position-sensitivity) with the product rule across the complete momentum
/// AND mass scatter (not just the stress term `p2g_stress_vjp` covers).
///
/// Forward, restated from `scatter_particles_to_grid`: per cell `c`,
///   mass_contrib_c     = weight_c * mass
///   momentum_contrib_c = weight_c * A_c,  A_c = mass*v + M*cell_dist_c
///   M = mass*C + stress_coeff*stress   (constant across cells, for fixed particle state)
///
/// BOTH `weight_c(x)` and `cell_dist_c(x) = cell_pos_c - x + 0.5` depend on
/// `x` (`d(cell_dist)/dx = -I`), so differentiating the product `weight * A`
/// needs the product rule on both factors. Per cell, given the gradients
/// flowing back from that cell's momentum and mass, `d_loss_d_momentum[c]`
/// (Vec2) and `d_loss_d_mass[c]` (f32):
///
///   d_loss_d_x += d(weight_c)/dx * (d_loss_d_momentum[c].A_c + d_loss_d_mass[c]*mass)
///               - weight_c * (Mᵀ * d_loss_d_momentum[c])
///
/// where `d(weight_c)/dx = (dwx[gx]/dx.x * wy[gy], wx[gx] * dwy[gy]/dx.y)`
/// via `axis_weights_derivative`, and the `-weight_c * Mᵀ*d_loss_d_momentum`
/// term comes from `d(A_c)/dx = M * d(cell_dist_c)/dx = -M`.
///
/// Verified against central-difference numerical gradients taken through a
/// forward function reconstructing `scatter_particles_to_grid`'s exact
/// per-cell formula, in this module's own tests.
///
/// Bundles the particle state P2G itself reads (`mass`, `v`, `C`, `stress`,
/// `stress_coeff`) into one struct rather than five separate parameters --
/// this function differentiates the FULL forward pass, so it genuinely needs
/// all of it, but five-plus-position-plus-two-gradient-array parameters
/// crossed the project's own no-`#[allow]` line for argument count.
pub struct P2GParticleState {
    pub mass: f32,
    pub v: Vec2,
    pub c: Mat2,
    pub stress: Mat2,
    pub stress_coeff: f32,
}

pub fn p2g_position_vjp(
    x: Vec2,
    state: &P2GParticleState,
    d_loss_d_momentum: &[[Vec2; 3]; 3],
    d_loss_d_mass: &[[f32; 3]; 3],
) -> Vec2 {
    let weights = quadratic_weights(x);
    let diff = x - weights.base_cell.as_vec2() - Vec2::splat(0.5);
    let dwx = axis_weights_derivative(diff.x);
    let dwy = axis_weights_derivative(diff.y);
    let m = state.mass * state.c + state.stress_coeff * state.stress;

    let mut d_loss_d_x = Vec2::ZERO;
    for gx in 0..3 {
        for gy in 0..3 {
            let wx = weights.wx[gx];
            let wy = weights.wy[gy];
            let weight = wx * wy;
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            let a = state.mass * state.v + m * cell_dist;

            let d_weight_dx = Vec2::new(dwx[gx] * wy, wx * dwy[gy]);
            let g_momentum = d_loss_d_momentum[gx][gy];
            let g_mass = d_loss_d_mass[gx][gy];

            d_loss_d_x += d_weight_dx * (g_momentum.dot(a) + g_mass * state.mass);
            d_loss_d_x -= weight * (m.transpose() * g_momentum);
        }
    }
    d_loss_d_x
}

pub struct G2PParams<'a> {
    pub vel_limit: f32,
    pub apic_blend: f32,
    pub active_count: usize,
    /// ASFLIP blend factor (`SimConfig::asflip_blend`, Fei et al. 2021). 0.0 = disabled,
    /// the exact original G2P formula below (see `pre_force_snapshot`'s doc for the gate).
    pub asflip_blend: f32,
    /// The grid's pre-force velocity snapshot (see `Grid::snapshot_velocities`), or `None`
    /// when ASFLIP is disabled. This, not `asflip_blend` alone, is the real gate: the ASFLIP
    /// correction below only runs when `Some`, so a caller that never opts in (passes `None`)
    /// gets the byte-identical original code path regardless of what `asflip_blend` holds.
    pub pre_force_snapshot: Option<&'a crate::grid::VelocitySnapshot>,
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

/// Analytic adjoint of G2P's APIC affine matrix (`velocity_gradient`)
/// computation w.r.t. the 9 grid velocities -- the piece `g2p_velocity_vjp`
/// deliberately left open, now closed. Real, externally cross-checked: this
/// exact term appears in ChainQueen's own hand-written CUDA backward pass
/// (`backward.cu`, `P2G_backward`'s "(C)" comment) as
/// `invD * N * grad_C_next[alpha][beta] * dpos[beta]` -- confirms both that
/// this term is genuinely needed (not paranoia) and, since it algebraically
/// matches the independently-derived formula below once ChainQueen's `invD`
/// is read as this codebase's `KERNEL_D_INVERSE`, that the derivation is
/// right. `apic_blend` is an emerge-specific extra factor ChainQueen's own
/// formula doesn't have (see `gather_grid_to_particles`'s `vg = b *
/// KERNEL_D_INVERSE * apic_blend`), included here since it's part of
/// emerge's own forward formula.
///
/// Forward (see `gather_grid_to_particles`'s Phase 1): `new_c = scale *
/// sum_c weight_c * outer(v_grid_c, dist_c)`, where `scale =
/// KERNEL_D_INVERSE * apic_blend` and `outer(v,d)` has column 0 = `d.x*v`,
/// column 1 = `d.y*v` (same convention as `p2g_stress_vjp`'s own outer
/// product). Linear in each `v_grid_c`; given the gradient flowing back from
/// the affine matrix, `d_loss_d_new_c` (a Mat2), the VJP of `outer(v,d)`
/// w.r.t. `v` is `M*d` (matrix-vector product, standard result for an outer
/// product's adjoint):
///
///   d_loss_d_v_grid[c] = weight_c * scale * (d_loss_d_new_c * dist_c)
///
/// Callers combine this additively with `g2p_velocity_vjp`'s output (both
/// scatter to the SAME 9 grid cells, since `new_v` and `new_c` are computed
/// from the same stencil in the same G2P pass) to get the true total
/// per-cell gradient. Verified against central-difference numerical
/// gradients in this module's own tests, independently and composed with
/// `g2p_velocity_vjp`.
pub fn g2p_affine_vjp(
    x: Vec2,
    kernel_d_inverse: f32,
    apic_blend: f32,
    d_loss_d_new_c: Mat2,
) -> [[Vec2; 3]; 3] {
    let weights = quadratic_weights(x);
    let scale = kernel_d_inverse * apic_blend;
    let mut out = [[Vec2::ZERO; 3]; 3];
    for (gx, (row, wx)) in out.iter_mut().zip(weights.wx.iter()).enumerate() {
        for (gy, (cell, wy)) in row.iter_mut().zip(weights.wy.iter()).enumerate() {
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            *cell = (wx * wy * scale) * (d_loss_d_new_c * dist);
        }
    }
    out
}

/// Analytic adjoint of the deformation-gradient update `F_new = (I + dt*C) *
/// F_old` w.r.t. both `C` (the APIC affine matrix / velocity_gradient G2P
/// produces) and `F_old` -- sixth real piece of differentiable stepping, and
/// the one that actually CLOSES the loop: `C` comes from G2P, `F_old` is the
/// previous substep's deformation gradient, and this update's own output
/// (`F_new`) is exactly what `kirchhoff_stress_vjp` needs as input for the
/// NEXT substep. Chaining this repeatedly is what backprop-through-multiple-
/// substeps actually means.
///
/// This exact formula is universal MPM kinematics, not any one material's own
/// logic -- confirmed by grep: every material in `matter::materials`
/// (NeoHookean, Corotated, Viscoelastic, and every plastic model's F_trial
/// before its own return-mapping) computes `F_new`/`F_trial` this identical
/// way. Lives here in `spacetime::transfer`, not any material file, for that
/// reason.
///
/// Derivation: let `A = I + dt*C`, so `F_new = A * F_old` -- a plain matrix
/// product. Standard VJP for `Y = A*B`: `dL/dA = Ḡ*Bᵀ`, `dL/dB = Aᵀ*Ḡ`. Since
/// `A` is linear in `C` (`dA/dC = dt` component-wise), `dL/dC = dt * dL/dA`:
///
///   d_loss_d_C     = dt * (d_loss_d_F_new * F_oldᵀ)
///   d_loss_d_F_old = (I + dt*C)ᵀ * d_loss_d_F_new
///
/// Verified against central-difference numerical gradients in this module's
/// own tests, on both outputs independently.
pub fn f_update_vjp(c: Mat2, f_old: Mat2, dt: f32, d_loss_d_f_new: Mat2) -> (Mat2, Mat2) {
    let a = Mat2::IDENTITY + dt * c;
    let d_loss_d_c = dt * (d_loss_d_f_new * f_old.transpose());
    let d_loss_d_f_old = a.transpose() * d_loss_d_f_new;
    (d_loss_d_c, d_loss_d_f_old)
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
        asflip_blend,
        pre_force_snapshot,
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
    let contact_groups = &particles.contact_group[..active_count];
    let pinned_flags = &particles.pinned[..active_count];
    // Gate once, not per particle: when no grip particle ever touched the grid this
    // substep (every scene that doesn't use `Particle::contact_group`), this is false
    // and the loop below takes the exact same path it always has — a plain
    // `grid.velocity_at` lookup, no extra branching cost worth measuring.
    let contact_active = grid.has_contact_activity();

    let clamp_count: usize = xs
        .par_iter_mut()
        .zip(vs.par_iter_mut())
        .zip(vgs.par_iter_mut())
        .zip(contact_groups.par_iter())
        .zip(pinned_flags.par_iter())
        .map(|((((x, v), vg), &contact_group), &pinned)| {
            let v_old = *v;
            let weights = quadratic_weights(*x);
            let mut new_v = Vec2::ZERO;
            let mut b = Mat2::ZERO;

            for gx in 0..3 {
                for gy in 0..3 {
                    let weight = weights.wx[gx] * weights.wy[gy];
                    let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                    let dist = cell_pos.as_vec2() - *x + Vec2::splat(0.5);
                    // Multi-field contact routing (Bardenhagen 2001): a grip particle
                    // reads the resolved grip field, a non-grip particle reads the
                    // resolved rest field, at nodes where contact was ever registered
                    // this substep. Both helpers fall back to the ordinary total
                    // velocity where no contact exists at that node, so this is exact
                    // everywhere, not just near contact.
                    let node_v = if !contact_active {
                        grid.velocity_at(cell_pos)
                    } else if contact_group != 0 {
                        grid.grip_velocity_at(cell_pos)
                    } else {
                        grid.rest_velocity_at(cell_pos)
                    };
                    let weighted_velocity = node_v * weight;
                    let term =
                        Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                    b += term;
                    new_v += weighted_velocity;
                }
            }

            // Dirichlet/kinematic anchor (`Particle::pinned`): force v=0 and
            // velocity_gradient=0 instead of gathering from the grid, so a pinned
            // particle never moves and never accumulates local strain from being
            // dragged — while its own mass/stress still scattered into P2G normally,
            // so it acts as a real, immovable anchor other bodies push against (the
            // standard technique for static/bedrock geometry in deformable-body sims).
            // Checked before the speed cap/position advance so a pinned particle takes
            // neither — position is deliberately left completely untouched, not just
            // re-clamped to itself, avoiding any float drift from a v=0*dt add-then-
            // reclamp round trip.
            if pinned != 0 {
                *v = Vec2::ZERO;
                *vg = Mat2::ZERO;
                return 0;
            }

            // ASFLIP (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in the
            // Material Point Method" -- see `SimConfig::asflip_blend` doc). Reintroduces
            // the classic FLIP residual (`v_p_old - old_v`) on top of the PIC/APIC gather
            // above -- `old_v` is a PIC-style gather against the grid's PRE-FORCE velocity
            // (`pre_force_snapshot`, taken right after P2G's own momentum normalization,
            // before this substep's gravity/boundary/contact modified it), using the SAME
            // stencil weights as `new_v` above. `pre_force_snapshot` being `None` (the
            // default, `asflip_blend=0.0`) is the real gate: `v_store`/`v_position` both
            // stay exactly `new_v`, reproducing the original formula below bit-for-bit.
            //
            // `gamma` (position-correction strength) is 0 while the local velocity
            // gradient indicates compression (`trace(b) < 0` -- two bodies pressing
            // together, e.g. a creature pushing into terrain via multi-field contact, or
            // material pressing against a boundary, since boundary conditions are already
            // baked into `new_v`/`b` by the time G2P reads the grid) and 1 while
            // separating -- exactly the paper's own "easier separation" adaptivity,
            // avoiding injecting extra positional noise while two bodies are in contact.
            let (mut v_store, mut v_position) = (new_v, new_v);
            if let Some(snapshot) = pre_force_snapshot {
                let mut old_v = Vec2::ZERO;
                for gx in 0..3 {
                    for gy in 0..3 {
                        let weight = weights.wx[gx] * weights.wy[gy];
                        let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                        old_v += grid.pre_force_velocity_at(snapshot, cell_pos) * weight;
                    }
                }
                let diff_vel = v_old - old_v;
                let trace_b = b.x_axis.x + b.y_axis.y;
                let gamma = if trace_b < 0.0 { 0.0 } else { 1.0 };
                v_store = new_v + asflip_blend * diff_vel;
                v_position = new_v + gamma * asflip_blend * diff_vel;
            }

            // Hard speed cap — CFL in choose_substep_dt is the physics-grounded bound.
            // This fires only when CFL is violated despite the timestep limiter (e.g. first
            // substep of a high-energy spawn). Magnitude clamp preserves direction; no
            // anisotropic bias unlike per-component clamping. Clamps both `v_store` and
            // `v_position` by the SAME safety ratio (derived from the stored velocity's own
            // magnitude) so they stay mutually consistent -- when ASFLIP is disabled the two
            // are identical (`v_store == v_position == new_v`), so this is byte-identical to
            // the original single-velocity clamp.
            let spd = v_store.length();
            let clamped = if spd > vel_limit {
                let scale = vel_limit / spd;
                v_store *= scale;
                v_position *= scale;
                1
            } else {
                0
            };

            // Apply all boundaries' position clamp (pure function, no particle-struct access).
            let mut new_pos = *x + v_position * dt;
            for boundary in boundaries.iter() {
                new_pos = boundary.clamp_particle_position(new_pos, grid_res);
            }

            *v = v_store;
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
                contact_group: 0,
                sleeping: 0,
                pinned: 0,
                _pad: [0; 2],
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
mod p2g_position_vjp_tests {
    use super::*;

    /// Forward function reconstructing scatter_particles_to_grid's EXACT
    /// per-cell formula (mass_contrib, momentum_contrib), taking the
    /// particle state directly instead of a real Particles/Grid -- isolates
    /// the position-dependence being verified from everything else.
    fn contributions(x: Vec2, state: &P2GParticleState) -> ([[Vec2; 3]; 3], [[f32; 3]; 3]) {
        let weights = quadratic_weights(x);
        let m = state.mass * state.c + state.stress_coeff * state.stress;
        let mut momentum = [[Vec2::ZERO; 3]; 3];
        let mut mass = [[0.0f32; 3]; 3];
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let a = state.mass * state.v + m * cell_dist;
                momentum[gx][gy] = weight * a;
                mass[gx][gy] = weight * state.mass;
            }
        }
        (momentum, mass)
    }

    fn loss(
        x: Vec2,
        state: &P2GParticleState,
        g_momentum: &[[Vec2; 3]; 3],
        g_mass: &[[f32; 3]; 3],
    ) -> f32 {
        let (momentum, mass) = contributions(x, state);
        let mut total = 0.0;
        for gx in 0..3 {
            for gy in 0..3 {
                total += g_momentum[gx][gy].dot(momentum[gx][gy]) + g_mass[gx][gy] * mass[gx][gy];
            }
        }
        total
    }

    fn check(x: Vec2, state: P2GParticleState, g_momentum: [[Vec2; 3]; 3], g_mass: [[f32; 3]; 3]) {
        let analytic = p2g_position_vjp(x, &state, &g_momentum, &g_mass);
        let h = 1.0e-3_f32;

        let numeric_x = (loss(x + Vec2::new(h, 0.0), &state, &g_momentum, &g_mass)
            - loss(x - Vec2::new(h, 0.0), &state, &g_momentum, &g_mass))
            / (2.0 * h);
        let numeric_y = (loss(x + Vec2::new(0.0, h), &state, &g_momentum, &g_mass)
            - loss(x - Vec2::new(0.0, h), &state, &g_momentum, &g_mass))
            / (2.0 * h);

        for (label, analytic_val, numeric) in
            [("x", analytic.x, numeric_x), ("y", analytic.y, numeric_y)]
        {
            let diff = (numeric - analytic_val).abs();
            let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "p2g_position_vjp mismatch at {label}: analytic={analytic_val:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e} (x={x:?})",
                diff / scale
            );
        }
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check(
            Vec2::new(20.0, 20.0),
            P2GParticleState {
                mass: 1.5,
                v: Vec2::new(0.3, -0.2),
                c: Mat2::from_cols(Vec2::new(0.1, 0.05), Vec2::new(-0.05, 0.1)),
                stress: Mat2::from_cols(Vec2::new(3.0, 0.5), Vec2::new(0.5, -2.0)),
                stress_coeff: -0.4,
            },
            [[Vec2::new(0.6, -0.3); 3]; 3],
            [[0.2; 3]; 3],
        );
    }

    #[test]
    fn matches_finite_difference_off_center_with_varied_gradients() {
        let g_momentum = [
            [
                Vec2::new(0.4, -0.5),
                Vec2::new(0.9, 0.1),
                Vec2::new(-0.3, 0.6),
            ],
            [
                Vec2::new(0.6, 0.2),
                Vec2::new(-0.1, -0.4),
                Vec2::new(0.5, 0.9),
            ],
            [
                Vec2::new(-0.8, 0.3),
                Vec2::new(0.2, -0.7),
                Vec2::new(0.4, 0.4),
            ],
        ];
        let g_mass = [[0.3, -0.2, 0.5], [-0.4, 0.6, 0.1], [0.2, -0.3, 0.4]];
        check(
            Vec2::new(9.35, 21.78),
            P2GParticleState {
                mass: 0.8,
                v: Vec2::new(-0.5, 0.4),
                c: Mat2::from_cols(Vec2::new(-0.2, 0.15), Vec2::new(0.1, -0.25)),
                stress: Mat2::from_cols(Vec2::new(-1.5, 2.0), Vec2::new(0.9, 1.2)),
                stress_coeff: 0.6,
            },
            g_momentum,
            g_mass,
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

#[cfg(test)]
mod f_update_vjp_tests {
    use super::*;

    fn f_new(c: Mat2, f_old: Mat2, dt: f32) -> Mat2 {
        (Mat2::IDENTITY + dt * c) * f_old
    }

    fn loss(c: Mat2, f_old: Mat2, dt: f32, g: Mat2) -> f32 {
        let fnew = f_new(c, f_old, dt);
        g.x_axis.x * fnew.x_axis.x
            + g.x_axis.y * fnew.x_axis.y
            + g.y_axis.x * fnew.y_axis.x
            + g.y_axis.y * fnew.y_axis.y
    }

    /// Bundles the fixed context shared by every component check.
    struct FUpdateContext {
        c: Mat2,
        f_old: Mat2,
        dt: f32,
        g: Mat2,
        h: f32,
    }

    /// Central-difference check on one scalar component of either C or
    /// F_old (whichever `set`/`get` target), holding the other input fixed.
    fn check_one_component(
        ctx: &FUpdateContext,
        label: &str,
        analytic_val: f32,
        vary_c: bool,
        set: impl Fn(&mut Mat2, f32),
        get: impl Fn(Mat2) -> f32,
    ) {
        let (mut c_plus, mut f_plus) = (ctx.c, ctx.f_old);
        let (mut c_minus, mut f_minus) = (ctx.c, ctx.f_old);
        if vary_c {
            let base = get(ctx.c);
            set(&mut c_plus, base + ctx.h);
            set(&mut c_minus, base - ctx.h);
        } else {
            let base = get(ctx.f_old);
            set(&mut f_plus, base + ctx.h);
            set(&mut f_minus, base - ctx.h);
        }
        let numeric = (loss(c_plus, f_plus, ctx.dt, ctx.g) - loss(c_minus, f_minus, ctx.dt, ctx.g))
            / (2.0 * ctx.h);
        let diff = (numeric - analytic_val).abs();
        let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "f_update_vjp mismatch at {label}: analytic={analytic_val:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            diff / scale
        );
    }

    /// Checks all 4 scalar components of one input matrix (either C or
    /// F_old), holding the other fixed -- reused for both outputs of
    /// `f_update_vjp`.
    fn check_matrix_input(ctx: &FUpdateContext, label_prefix: &str, analytic: Mat2, vary_c: bool) {
        check_one_component(
            ctx,
            &format!("{label_prefix}[0][0]"),
            analytic.x_axis.x,
            vary_c,
            |m, v| m.x_axis.x = v,
            |m| m.x_axis.x,
        );
        check_one_component(
            ctx,
            &format!("{label_prefix}[1][0]"),
            analytic.x_axis.y,
            vary_c,
            |m, v| m.x_axis.y = v,
            |m| m.x_axis.y,
        );
        check_one_component(
            ctx,
            &format!("{label_prefix}[0][1]"),
            analytic.y_axis.x,
            vary_c,
            |m, v| m.y_axis.x = v,
            |m| m.y_axis.x,
        );
        check_one_component(
            ctx,
            &format!("{label_prefix}[1][1]"),
            analytic.y_axis.y,
            vary_c,
            |m, v| m.y_axis.y = v,
            |m| m.y_axis.y,
        );
    }

    fn check(c: Mat2, f_old: Mat2, dt: f32, g: Mat2) {
        let (d_loss_d_c, d_loss_d_f_old) = f_update_vjp(c, f_old, dt, g);
        let ctx = FUpdateContext {
            c,
            f_old,
            dt,
            g,
            h: 1.0e-3_f32,
        };
        check_matrix_input(&ctx, "d_loss_d_c", d_loss_d_c, true);
        check_matrix_input(&ctx, "d_loss_d_f_old", d_loss_d_f_old, false);
    }

    #[test]
    fn matches_finite_difference_small_dt() {
        check(
            Mat2::from_cols(Vec2::new(0.2, -0.1), Vec2::new(0.05, 0.15)),
            Mat2::from_cols(Vec2::new(1.1, 0.05), Vec2::new(-0.02, 0.95)),
            0.001,
            Mat2::from_cols(Vec2::new(0.6, -0.3), Vec2::new(0.4, 0.8)),
        );
    }

    #[test]
    fn matches_finite_difference_larger_dt_and_deformation() {
        check(
            Mat2::from_cols(Vec2::new(-0.5, 0.3), Vec2::new(0.2, 0.4)),
            Mat2::from_cols(Vec2::new(1.4, 0.2), Vec2::new(-0.15, 0.8)),
            0.05,
            Mat2::from_cols(Vec2::new(-0.7, 1.1), Vec2::new(0.9, -0.4)),
        );
    }
}

#[cfg(test)]
mod g2p_affine_vjp_tests {
    use super::*;

    /// Forward formula exactly matching G2P's own `new_c`/`velocity_gradient`
    /// computation (the weighted outer-product sum), taking the 9 grid
    /// velocities directly as an array instead of reading a real `Grid`.
    fn gather_affine(x: Vec2, v_grid: &[[Vec2; 3]; 3], scale: f32) -> Mat2 {
        let weights = quadratic_weights(x);
        let mut b = Mat2::ZERO;
        for (gx, (row, wx)) in v_grid.iter().zip(weights.wx.iter()).enumerate() {
            for (gy, (v_cell, wy)) in row.iter().zip(weights.wy.iter()).enumerate() {
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let weighted = *v_cell * (wx * wy);
                b += Mat2::from_cols(weighted * dist.x, weighted * dist.y);
            }
        }
        b * scale
    }

    fn loss(x: Vec2, v_grid: &[[Vec2; 3]; 3], scale: f32, g: Mat2) -> f32 {
        let c = gather_affine(x, v_grid, scale);
        g.x_axis.x * c.x_axis.x
            + g.x_axis.y * c.x_axis.y
            + g.y_axis.x * c.y_axis.x
            + g.y_axis.y * c.y_axis.y
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check(
            Vec2::new(30.0, 30.0),
            4.0,
            0.9,
            Mat2::from_cols(Vec2::new(0.5, -0.3), Vec2::new(0.2, 0.7)),
        );
    }

    #[test]
    fn matches_finite_difference_off_center() {
        check(
            Vec2::new(12.6, 5.9),
            4.0,
            0.75,
            Mat2::from_cols(Vec2::new(-0.4, 0.6), Vec2::new(1.0, -0.2)),
        );
    }

    /// Checks every one of the 9 stencil cells' 2 velocity components (18
    /// scalars total) against central differences.
    fn check(x: Vec2, kernel_d_inverse: f32, apic_blend: f32, g: Mat2) {
        let v_grid = [
            [
                Vec2::new(0.2, -0.4),
                Vec2::new(0.6, 0.1),
                Vec2::new(-0.3, 0.5),
            ],
            [
                Vec2::new(0.4, 0.3),
                Vec2::new(-0.1, -0.6),
                Vec2::new(0.2, 0.4),
            ],
            [
                Vec2::new(-0.5, 0.2),
                Vec2::new(0.3, -0.4),
                Vec2::new(0.1, 0.6),
            ],
        ];
        let scale = kernel_d_inverse * apic_blend;
        let analytic = g2p_affine_vjp(x, kernel_d_inverse, apic_blend, g);
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
                    let numeric =
                        (loss(x, &v_plus, scale, g) - loss(x, &v_minus, scale, g)) / (2.0 * h);
                    let analytic_val = if axis == 0 {
                        analytic[gx][gy].x
                    } else {
                        analytic[gx][gy].y
                    };
                    let diff = (numeric - analytic_val).abs();
                    let scale_denom = numeric.abs().max(analytic_val.abs()).max(1.0);
                    assert!(
                        diff / scale_denom < 1.0e-2,
                        "g2p_affine_vjp mismatch at cell[{gx}][{gy}].{label}: \
                         analytic={analytic_val:.6} numeric={numeric:.6} \
                         relative_diff={:.2e} (x={x:?})",
                        diff / scale_denom
                    );
                }
            }
        }
    }

    /// Real end-to-end check: combines g2p_velocity_vjp and g2p_affine_vjp
    /// (the two halves of G2P's actual joint computation, gathered from the
    /// SAME 9 grid velocities in the same pass) and verifies the SUMMED
    /// gradient matches a finite difference taken through the true combined
    /// loss L = g_v . new_v + g_c : new_c -- proves the two adjoints compose
    /// correctly when G2P's real output (both v and C) feeds a real loss,
    /// not just that each is independently correct in isolation.
    #[test]
    fn composes_correctly_with_g2p_velocity_vjp() {
        let x = Vec2::new(18.3, 9.7);
        let kernel_d_inverse = 4.0;
        let apic_blend = 1.0;
        let scale = kernel_d_inverse * apic_blend;
        let g_v = Vec2::new(0.4, -0.6);
        let g_c = Mat2::from_cols(Vec2::new(0.3, 0.5), Vec2::new(-0.7, 0.2));

        let v_grid = [
            [
                Vec2::new(0.1, 0.2),
                Vec2::new(-0.3, 0.4),
                Vec2::new(0.5, -0.1),
            ],
            [
                Vec2::new(0.2, -0.2),
                Vec2::new(0.4, 0.3),
                Vec2::new(-0.4, 0.1),
            ],
            [
                Vec2::new(-0.1, 0.5),
                Vec2::new(0.2, -0.3),
                Vec2::new(0.3, 0.2),
            ],
        ];

        let combined_loss = |v_grid: &[[Vec2; 3]; 3]| -> f32 {
            let weights = quadratic_weights(x);
            let mut new_v = Vec2::ZERO;
            let mut b = Mat2::ZERO;
            for (gxi, (row, wx)) in v_grid.iter().zip(weights.wx.iter()).enumerate() {
                for (gyi, (v_cell, wy)) in row.iter().zip(weights.wy.iter()).enumerate() {
                    let weight = wx * wy;
                    let cell_pos = weights.base_cell + IVec2::new(gxi as i32 - 1, gyi as i32 - 1);
                    let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                    let weighted = *v_cell * weight;
                    new_v += weighted;
                    b += Mat2::from_cols(weighted * dist.x, weighted * dist.y);
                }
            }
            let new_c = b * scale;
            g_v.dot(new_v)
                + g_c.x_axis.x * new_c.x_axis.x
                + g_c.x_axis.y * new_c.x_axis.y
                + g_c.y_axis.x * new_c.y_axis.x
                + g_c.y_axis.y * new_c.y_axis.y
        };

        let from_v = g2p_velocity_vjp(x, g_v);
        let from_c = g2p_affine_vjp(x, kernel_d_inverse, apic_blend, g_c);

        let h = 1.0e-3_f32;
        // Check cell [1][1] (center of stencil) as a representative sample.
        let mut v_plus = v_grid;
        v_plus[1][1].x += h;
        let mut v_minus = v_grid;
        v_minus[1][1].x -= h;
        let numeric = (combined_loss(&v_plus) - combined_loss(&v_minus)) / (2.0 * h);

        let combined_analytic = from_v[1][1].x + from_c[1][1].x;
        let diff = (numeric - combined_analytic).abs();
        let scale_denom = numeric.abs().max(combined_analytic.abs()).max(1.0);
        assert!(
            diff / scale_denom < 1.0e-2,
            "composed g2p_velocity_vjp + g2p_affine_vjp must match end-to-end finite \
             difference: analytic={combined_analytic:.6} numeric={numeric:.6} \
             relative_diff={:.2e}",
            diff / scale_denom
        );
    }
}

#[cfg(test)]
mod active_stress_vjp_tests {
    use super::*;

    fn tau_active(f: Mat2, activation: f32, coeff: f32, fiber_dir: Vec2) -> Mat2 {
        let len_sq = fiber_dir.dot(fiber_dir);
        if len_sq <= f32::EPSILON || activation <= 0.0 || coeff <= 0.0 {
            return Mat2::ZERO;
        }
        let n0 = fiber_dir / len_sq.sqrt();
        let a_mat = Mat2::from_cols(n0 * n0.x, n0 * n0.y) * (activation * coeff);
        f * a_mat * f.transpose()
    }

    fn loss(f: Mat2, activation: f32, coeff: f32, fiber_dir: Vec2, g: Mat2) -> f32 {
        let tau = tau_active(f, activation, coeff, fiber_dir);
        g.x_axis.x * tau.x_axis.x
            + g.x_axis.y * tau.x_axis.y
            + g.y_axis.x * tau.y_axis.x
            + g.y_axis.y * tau.y_axis.y
    }

    fn check(f: Mat2, activation: f32, coeff: f32, fiber_dir: Vec2, g: Mat2) {
        let (analytic_d_f, analytic_d_activation) =
            active_stress_vjp(f, activation, coeff, fiber_dir, g);
        let h = 1.0e-3_f32;

        // Activation (scalar).
        let numeric_activation = (loss(f, activation + h, coeff, fiber_dir, g)
            - loss(f, activation - h, coeff, fiber_dir, g))
            / (2.0 * h);
        let diff = (numeric_activation - analytic_d_activation).abs();
        let scale = numeric_activation
            .abs()
            .max(analytic_d_activation.abs())
            .max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "active_stress_vjp activation mismatch: analytic={analytic_d_activation:.6} \
             numeric={numeric_activation:.6} relative_diff={:.2e}",
            diff / scale
        );

        // F (4 components).
        let check_f_component =
            |set: fn(&mut Mat2, f32), get: fn(Mat2) -> f32, analytic_val: f32| {
                let mut f_plus = f;
                set(&mut f_plus, get(f) + h);
                let mut f_minus = f;
                set(&mut f_minus, get(f) - h);
                let numeric = (loss(f_plus, activation, coeff, fiber_dir, g)
                    - loss(f_minus, activation, coeff, fiber_dir, g))
                    / (2.0 * h);
                let diff = (numeric - analytic_val).abs();
                let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1.0e-2,
                    "active_stress_vjp F mismatch: analytic={analytic_val:.6} numeric={numeric:.6} \
                 relative_diff={:.2e}",
                    diff / scale
                );
            };
        check_f_component(|m, v| m.x_axis.x = v, |m| m.x_axis.x, analytic_d_f.x_axis.x);
        check_f_component(|m, v| m.x_axis.y = v, |m| m.x_axis.y, analytic_d_f.x_axis.y);
        check_f_component(|m, v| m.y_axis.x = v, |m| m.y_axis.x, analytic_d_f.y_axis.x);
        check_f_component(|m, v| m.y_axis.y = v, |m| m.y_axis.y, analytic_d_f.y_axis.y);
    }

    #[test]
    fn matches_finite_difference_axis_aligned_fiber() {
        check(
            Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(-0.05, 0.9)),
            0.6,
            10.0,
            Vec2::X,
            Mat2::from_cols(Vec2::new(0.4, -0.6), Vec2::new(0.3, 0.5)),
        );
    }

    #[test]
    fn matches_finite_difference_off_axis_fiber() {
        check(
            Mat2::from_cols(Vec2::new(0.95, -0.15), Vec2::new(0.2, 1.1)),
            0.8,
            15.0,
            Vec2::new(0.6, 0.8),
            Mat2::from_cols(Vec2::new(-0.5, 0.9), Vec2::new(0.7, -0.2)),
        );
    }
}

#[cfg(test)]
mod multistep_backprop_tests {
    use super::*;
    use crate::materials::NeoHookeanMaterial;
    use crate::particle::{Particle, Particles};

    fn particle_with_f(f: Mat2) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: Vec2::ZERO,
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
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            _pad: [0; 2],
        });
        particles
    }

    struct SubstepConfig {
        x: Vec2,
        mass: f32,
        stress_coeff: f32,
        dt: f32,
        kernel_d_inverse: f32,
        apic_blend: f32,
    }

    /// One real MLS-MPM substep (P2G scatter -> grid velocity update -> G2P
    /// gather -> F update) for a SINGLE particle at a FIXED position --
    /// position held fixed to match every adjoint above's own scoping
    /// (`p2g_stress_vjp`, `g2p_velocity_vjp`, etc. all defer the
    /// kernel-weight/position-dependence gap for the same reason; only
    /// `p2g_position_vjp` handles it, and isn't exercised here since this
    /// proof targets the OTHER remaining gap -- chaining substeps together).
    /// With one particle and fixed position, the 9-cell stencil can be
    /// tracked as a plain local array instead of a real `Grid`.
    fn substep_forward(
        f_old: Mat2,
        v_old: Vec2,
        c_old: Mat2,
        mat: &NeoHookeanMaterial,
        cfg: &SubstepConfig,
    ) -> (Mat2, Vec2, Mat2) {
        let particles = particle_with_f(f_old);
        let stress = mat.kirchhoff_stress(&particles, 0);
        let weights = quadratic_weights(cfg.x);

        let mut new_v = Vec2::ZERO;
        let mut b = Mat2::ZERO;
        for (gx, wx) in weights.wx.iter().enumerate() {
            for (gy, wy) in weights.wy.iter().enumerate() {
                let weight = wx * wy;
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - cfg.x + Vec2::splat(0.5);
                let momentum = weight
                    * (cfg.mass * (v_old + c_old * cell_dist)
                        + cfg.stress_coeff * (stress * cell_dist));
                let mass_c = weight * cfg.mass;
                let velocity_c = momentum / mass_c;
                let weighted = velocity_c * weight;
                new_v += weighted;
                b += Mat2::from_cols(weighted * cell_dist.x, weighted * cell_dist.y);
            }
        }
        let new_c = b * (cfg.kernel_d_inverse * cfg.apic_blend);
        let new_f = (Mat2::IDENTITY + cfg.dt * new_c) * f_old;
        (new_f, new_v, new_c)
    }

    /// The adjoint of `substep_forward`, built ENTIRELY from functions
    /// already shipped and individually verified above -- no new production
    /// math, just composition. This is the actual proof that
    /// backprop-through-multiple-substeps works: `F_old` feeds forward two
    /// ways (through `stress = kirchhoff_stress(F_old)` AND directly as
    /// `f_update_vjp`'s own `F_old` multiplicand), so its total gradient is a
    /// SUM of both paths -- the standard multivariable chain rule, not
    /// special-cased per path.
    ///
    /// `p2g_stress_vjp` is reused twice: once with the real `stress_coeff`
    /// for the stress->F path, once with `mass` standing in for that same
    /// scalar for the `c_old`->grid path -- both are the identical
    /// `weight*scalar*(tensor*cell_dist)` shape `scatter_particles_to_grid`
    /// computes, so the existing adjoint applies unchanged.
    fn substep_backward(
        f_old: Mat2,
        new_c: Mat2,
        mat: &NeoHookeanMaterial,
        cfg: &SubstepConfig,
        g_f_new: Mat2,
        g_v_new: Vec2,
        g_c_new: Mat2,
    ) -> (Mat2, Vec2, Mat2) {
        // F_new = (I + dt*new_c) * F_old
        let (g_c_from_f, g_f_old_a) = f_update_vjp(new_c, f_old, cfg.dt, g_f_new);
        let g_c_total = g_c_new + g_c_from_f;

        // new_v and new_c are both gathered from the same 9 grid velocities.
        let g_vel_from_c = g2p_affine_vjp(cfg.x, cfg.kernel_d_inverse, cfg.apic_blend, g_c_total);
        let g_vel_from_v = g2p_velocity_vjp(cfg.x, g_v_new);

        let weights = quadratic_weights(cfg.x);
        let mut g_momentum = [[Vec2::ZERO; 3]; 3];
        let mut g_v_old = Vec2::ZERO;
        for (gx, wx) in weights.wx.iter().enumerate() {
            for (gy, wy) in weights.wy.iter().enumerate() {
                let weight = wx * wy;
                let mass_c = weight * cfg.mass;
                let g_v_cell = g_vel_from_c[gx][gy] + g_vel_from_v[gx][gy];
                // update_velocities_vjp's d_loss_d_momentum output doesn't
                // depend on the forward momentum value (only on mass), so
                // the placeholder Vec2::ZERO here is exact, not an
                // approximation -- confirmed against the function's own
                // formula (see `grid/mod.rs`).
                let (g_m, _g_mass) = Grid::update_velocities_vjp(Vec2::ZERO, mass_c, g_v_cell);
                g_momentum[gx][gy] = g_m;
                g_v_old += weight * cfg.mass * g_m;
            }
        }

        let g_stress = p2g_stress_vjp(cfg.x, cfg.stress_coeff, &g_momentum);
        let g_c_old = p2g_stress_vjp(cfg.x, cfg.mass, &g_momentum);

        let particles = particle_with_f(f_old);
        let g_f_old_b = mat.kirchhoff_stress_vjp(&particles, 0, g_stress);

        (g_f_old_a + g_f_old_b, g_v_old, g_c_old)
    }

    /// The actual milestone: chain TWO real substeps forward, then backprop
    /// the whole thing back to the very first `F`, and check the result
    /// against a finite difference taken through the ENTIRE two-substep
    /// forward pass -- not just one isolated function. This is what "diff-MPM
    /// chain must finish first" concretely means: not more individually-
    /// verified pieces, but proof they compose across time.
    #[test]
    fn chains_two_substeps_matches_finite_difference() {
        let mat = NeoHookeanMaterial::new(900.0, 700.0);
        let cfg = SubstepConfig {
            x: Vec2::new(12.4, 7.8),
            mass: 1.0,
            stress_coeff: -0.05,
            dt: 0.01,
            kernel_d_inverse: KERNEL_D_INVERSE,
            apic_blend: 1.0,
        };
        let target = Mat2::from_cols(Vec2::new(1.1, 0.05), Vec2::new(-0.03, 0.95));
        let f0_start = Mat2::from_cols(Vec2::new(1.15, 0.08), Vec2::new(-0.06, 0.9));

        let forward_two = |f0: Mat2| -> Mat2 {
            let (f1, v1, c1) = substep_forward(f0, Vec2::ZERO, Mat2::ZERO, &mat, &cfg);
            let (f2, _v2, _c2) = substep_forward(f1, v1, c1, &mat, &cfg);
            f2
        };

        let loss = |f0: Mat2| -> f32 {
            let d = forward_two(f0) - target;
            0.5 * (d.x_axis.x * d.x_axis.x
                + d.x_axis.y * d.x_axis.y
                + d.y_axis.x * d.y_axis.x
                + d.y_axis.y * d.y_axis.y)
        };

        // Analytic: forward, keeping intermediates, then backward from the
        // final loss all the way to f0_start.
        let (f1, v1, c1) = substep_forward(f0_start, Vec2::ZERO, Mat2::ZERO, &mat, &cfg);
        let (f2, _, c2) = substep_forward(f1, v1, c1, &mat, &cfg);
        let g_f2 = f2 - target; // dL/dF2 for L = 0.5*||F2-target||^2

        let (g_f1, g_v1, g_c1) = substep_backward(f1, c2, &mat, &cfg, g_f2, Vec2::ZERO, Mat2::ZERO);
        let (g_f0, _g_v0, _g_c0) = substep_backward(f0_start, c1, &mat, &cfg, g_f1, g_v1, g_c1);

        let h = 1.0e-3_f32;

        // Central-difference check on one scalar component of f0_start.
        let check_component =
            |label: &str, analytic_val: f32, base: f32, set: fn(&mut Mat2, f32)| {
                let mut f_plus = f0_start;
                set(&mut f_plus, base + h);
                let mut f_minus = f0_start;
                set(&mut f_minus, base - h);
                let numeric = (loss(f_plus) - loss(f_minus)) / (2.0 * h);

                let diff = (numeric - analytic_val).abs();
                let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1.0e-2,
                    "two-substep chained adjoint mismatch at F{label}: analytic={analytic_val:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                    diff / scale
                );
            };

        check_component("[0][0]", g_f0.x_axis.x, f0_start.x_axis.x, |m, v| {
            m.x_axis.x = v
        });
        check_component("[1][0]", g_f0.x_axis.y, f0_start.x_axis.y, |m, v| {
            m.x_axis.y = v
        });
        check_component("[0][1]", g_f0.y_axis.x, f0_start.y_axis.x, |m, v| {
            m.y_axis.x = v
        });
        check_component("[1][1]", g_f0.y_axis.y, f0_start.y_axis.y, |m, v| {
            m.y_axis.y = v
        });
    }

    /// Scales the two-substep proof above to a real rollout length (5
    /// substeps) via a plain loop over the same `substep_forward` /
    /// `substep_backward` functions -- no new math, just more of it. Proves
    /// the chain doesn't silently degrade (error accumulation, sign flips)
    /// over a longer horizon closer to what an actual trainer would run.
    #[test]
    fn chains_five_substeps_matches_finite_difference() {
        let mat = NeoHookeanMaterial::new(900.0, 700.0);
        let cfg = SubstepConfig {
            x: Vec2::new(4.6, 18.2),
            mass: 1.0,
            stress_coeff: -0.05,
            dt: 0.01,
            kernel_d_inverse: KERNEL_D_INVERSE,
            apic_blend: 1.0,
        };
        let target = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(-0.05, 0.85));
        let f0_start = Mat2::from_cols(Vec2::new(1.05, 0.03), Vec2::new(-0.02, 0.97));
        const STEPS: usize = 5;

        let forward_n = |f0: Mat2| -> Mat2 {
            let (mut f, mut v, mut c) = (f0, Vec2::ZERO, Mat2::ZERO);
            for _ in 0..STEPS {
                let (f_new, v_new, c_new) = substep_forward(f, v, c, &mat, &cfg);
                f = f_new;
                v = v_new;
                c = c_new;
            }
            f
        };

        let loss = |f0: Mat2| -> f32 {
            let d = forward_n(f0) - target;
            0.5 * (d.x_axis.x * d.x_axis.x
                + d.x_axis.y * d.x_axis.y
                + d.y_axis.x * d.y_axis.x
                + d.y_axis.y * d.y_axis.y)
        };

        // Forward, keeping every intermediate (f, v, c) so the backward
        // pass has what each substep's own backward call needs.
        let mut states = Vec::with_capacity(STEPS + 1);
        states.push((f0_start, Vec2::ZERO, Mat2::ZERO));
        for _ in 0..STEPS {
            let (f, v, c) = *states.last().unwrap();
            states.push(substep_forward(f, v, c, &mat, &cfg));
        }
        let f_final = states[STEPS].0;
        let mut g_f = f_final - target; // dL/dF_final
        let mut g_v = Vec2::ZERO;
        let mut g_c = Mat2::ZERO;

        for step in (0..STEPS).rev() {
            let (f_old, _, _) = states[step];
            let (_, _, c_new) = states[step + 1];
            let (next_g_f, next_g_v, next_g_c) =
                substep_backward(f_old, c_new, &mat, &cfg, g_f, g_v, g_c);
            g_f = next_g_f;
            g_v = next_g_v;
            g_c = next_g_c;
        }
        let g_f0 = g_f;

        let h = 1.0e-3_f32;
        let check_component =
            |label: &str, analytic_val: f32, base: f32, set: fn(&mut Mat2, f32)| {
                let mut f_plus = f0_start;
                set(&mut f_plus, base + h);
                let mut f_minus = f0_start;
                set(&mut f_minus, base - h);
                let numeric = (loss(f_plus) - loss(f_minus)) / (2.0 * h);

                let diff = (numeric - analytic_val).abs();
                let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1.0e-2,
                    "{STEPS}-substep chained adjoint mismatch at F{label}: \
                     analytic={analytic_val:.6} numeric(central-diff)={numeric:.6} \
                     relative_diff={:.2e}",
                    diff / scale
                );
            };

        check_component("[0][0]", g_f0.x_axis.x, f0_start.x_axis.x, |m, v| {
            m.x_axis.x = v
        });
        check_component("[1][0]", g_f0.x_axis.y, f0_start.x_axis.y, |m, v| {
            m.x_axis.y = v
        });
        check_component("[0][1]", g_f0.y_axis.x, f0_start.y_axis.x, |m, v| {
            m.y_axis.x = v
        });
        check_component("[1][1]", g_f0.y_axis.y, f0_start.y_axis.y, |m, v| {
            m.y_axis.y = v
        });
    }
}
