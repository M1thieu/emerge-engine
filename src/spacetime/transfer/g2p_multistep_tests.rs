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
        scalar_field: 0.0,
        user_tag: 0,
        activation: 0.0,
        activation_dir: Vec2::ZERO,
        muscle_group_id: 0,
        contact_group: 0,
        sleeping: 0,
        pinned: 0,
        internal_pressure: 0.0,
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
/// tracked as a plain local array instead of a `Grid`.
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
/// `p2g_stress_vjp` is reused twice: once with the `stress_coeff`
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
    let check_component = |label: &str, analytic_val: f32, base: f32, set: fn(&mut Mat2, f32)| {
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

/// Scales the two-substep proof above to a rollout length (5
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
    let check_component = |label: &str, analytic_val: f32, base: f32, set: fn(&mut Mat2, f32)| {
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
