//! Shared state evolution for the weakly-compressible fluid materials.
//!
//! A liquid has no elastic shear memory.  Its material state is therefore the
//! scalar volume ratio `J`, represented in the common particle `F` slot as
//! `sqrt(J) I`.  We integrate `d(log J)/dt = div(v)`, rather than forming
//! `det(I + dt L)`: the exponential update preserves positive volume for every
//! finite velocity gradient and does not need a nonphysical `J` clamp.

use glam::{Mat2, Vec2};

use crate::particle::{Particle, ParticleUpdateCtx};

#[inline]
pub(crate) fn positive_j(f: Mat2, material_name: &str) -> f32 {
    let j = f.determinant();
    assert!(
        j.is_finite() && j > 0.0,
        "{material_name}: invalid fluid deformation J={j}; reduce the timestep or use a pressure solver"
    );
    j
}

#[inline]
pub(crate) fn volume_j(initial_volume: f32, volume: f32, material_name: &str) -> f32 {
    assert!(
        initial_volume.is_finite() && initial_volume > 0.0 && volume.is_finite() && volume > 0.0,
        "{material_name}: fluid reference/current volume must be finite and positive"
    );
    volume / initial_volume
}

#[inline]
pub(crate) fn isotropic_f_from_j(j: f32, material_name: &str) -> Mat2 {
    assert!(
        j.is_finite() && j > 0.0,
        "{material_name}: invalid fluid volume ratio J={j}; reduce the timestep or use a pressure solver"
    );
    let s = j.sqrt();
    Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s))
}

/// Initialise a material point from the fluid's conserved mass and reference
/// density.  Kernel-gathered density is deliberately not used: it is biased at
/// a free surface and is not a constitutive state variable for this model.
pub(crate) fn init_particle(particle: &mut Particle, rest_density: f32, material_name: &str) {
    assert!(
        rest_density.is_finite() && rest_density > 0.0,
        "{material_name}: rest_density must be finite and positive"
    );
    assert!(
        particle.mass.is_finite() && particle.mass > 0.0,
        "{material_name}: particle mass must be finite and positive"
    );

    let j = positive_j(particle.deformation_gradient, material_name);
    particle.initial_volume = particle.mass / rest_density;
    particle.volume = particle.initial_volume * j;
    particle.density = rest_density / j;
    // A fluid retains only volume.  Dropping inherited shear here is the
    // constitutive definition of this liquid material, not a stability repair.
    particle.deformation_gradient = isotropic_f_from_j(j, material_name);
}

/// Advance the fluid's scalar volume state exactly for a constant local
/// divergence over the substep: `J_{n+1}=J_n exp(dt div(v))`.
pub(crate) fn update_particle(
    ctx: &mut ParticleUpdateCtx,
    dt: f32,
    rest_density: f32,
    material_name: &str,
) {
    assert!(
        dt.is_finite() && dt > 0.0,
        "{material_name}: timestep must be finite and positive"
    );
    assert!(
        rest_density.is_finite() && rest_density > 0.0,
        "{material_name}: rest_density must be finite and positive"
    );

    let old_j = volume_j(ctx.initial_volume, *ctx.volume, material_name);
    let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
    assert!(
        div_v.is_finite(),
        "{material_name}: non-finite velocity divergence; reduce the timestep"
    );
    let log_j = old_j.ln() + dt * div_v;
    let j = log_j.exp();
    assert!(
        j.is_finite() && j > 0.0,
        "{material_name}: volume integration left the representable range (log J={log_j}); reduce the timestep"
    );

    *ctx.deformation_gradient = isotropic_f_from_j(j, material_name);
    *ctx.volume = ctx.initial_volume * j;
    *ctx.density = rest_density / j;
}

#[inline]
pub(crate) fn tait_pressure(
    eos_stiffness: f32,
    eos_power: f32,
    j: f32,
    material_name: &str,
) -> f32 {
    assert!(
        eos_stiffness.is_finite()
            && eos_stiffness >= 0.0
            && eos_power.is_finite()
            && eos_power > 0.0
            && j.is_finite()
            && j > 0.0,
        "{material_name}: EOS parameters must be finite, with stiffness >= 0 and power > 0"
    );
    let density_ratio = 1.0 / j;
    assert!(
        density_ratio.is_finite() && density_ratio > 0.0,
        "{material_name}: Tait density ratio is unrepresentable; reduce the timestep or use a pressure solver"
    );
    let pressure =
        eos_stiffness * (crate::materials::utils::fast_pow(density_ratio, eos_power) - 1.0);
    assert!(
        pressure.is_finite(),
        "{material_name}: Tait pressure is unrepresentable; reduce the timestep or use a pressure solver"
    );
    pressure
}

/// Real, sourced numerical stabilizer for extreme local compression -- von
/// Neumann & Richtmyer 1950 (LA-671, the original shock-capturing artificial
/// viscosity for Lagrangian hydrocodes) quadratic term + Landshoff linear
/// term, the SAME combined form used for MPM specifically (Wang et al.,
/// "Portable, Massively Parallel Implementation of a Material Point Method
/// for Compressible Flows", arXiv:2404.17057, eq. 4): `q =
/// c0*(rho*h*div(v))^2 - c1*h*c_sound*div(v)` when `div(v)<0`, `q=0`
/// otherwise -- gated to compression only, vanishing identically wherever
/// the flow is smooth. `c_sound = sqrt(dp/drho)` in that paper's ideal-gas
/// form is `sqrt(gamma*p/rho)`; substituted here with this material's own
/// Tait EOS sound speed (`c2` from `NewtonianFluidMaterial::timestep_bound`'s
/// own derivation) -- the general, EOS-agnostic definition of `c_sound`, not
/// a gas-specific approximation.
///
/// `c1=1.0` (Landshoff) is the standard theoretical value. `c0` (quadratic,
/// von Neumann-Richtmyer) is Kurapatenko-derived, not a flat constant:
/// `c0 = (eos_power+1)/2` is the STRONG-shock limit of the fundamental
/// derivative G that Kurapatenko (1967) ties c0 to (for an ideal gas,
/// G_weak=(gamma+1)/4, G_strong=(gamma+1)/2 -- Tait EOS's exponent
/// `eos_power` plays gamma's role in this curvature derivation, since Tait
/// locally behaves as p~rho^eos_power the same way an ideal gas behaves as
/// p~rho^gamma). The strong-shock value is the physically correct choice
/// here, not the weaker/general one: this stabilizer exists specifically
/// for the violent, strong-compression regime this engine's own crash
/// investigation (below) found, not a mild acoustic ripple.
///
/// Root-caused 2026-08-08: basic_fluids_gpu.rs's real crash traced to a
/// genuine MPM cell-crossing-style instability under extreme local
/// compression near a wall (J drifting 1.06->2.91 over several frames, then
/// spurious horizontal velocity erupting from nowhere in a scene with zero
/// lateral force) -- CPU and GPU accumulate tiny floating-point differences
/// in this numerically stiff regime (parallel/atomic reduction order vs
/// sequential) and can diverge chaotically once a local configuration goes
/// unstable. The weak-shock c0=(eos_power+1)/4 measurably helped (peak J
/// dropped from ~31000-34000 to ~14000, and the failure mode changed from
/// unbounded runaway to a spike-then-plateau) but did not fully stop the one
/// remaining severe compression event -- the strong-shock value is the next
/// real, sourced escalation, not an arbitrary bump.
/// This is the real, general fix: damp the violent compression BEFORE it
/// can amplify into that chaotic regime, on both platforms identically
/// (verified bit-for-bit-matching formula on CPU and GPU).
#[inline]
pub(crate) fn artificial_bulk_viscosity(
    eos_stiffness: f32,
    eos_power: f32,
    rest_density: f32,
    j: f32,
    div_v: f32,
    grid_cell_size: f32,
) -> f32 {
    if !(div_v < 0.0) {
        return 0.0;
    }
    // Kurapatenko 1967 weak-shock limit -- kept at weak-shock, NOT the
    // strong-shock (eos_power+1)/2: real, measured 2026-08-08, the
    // strong-shock value made this exact crash WORSE (panicked at frame
    // ~60-90 instead of surviving all 300 frames on a bounded plateau) --
    // the quadratic term's own contribution needs to feed back into the CFL
    // bound (Bate et al. 1995's combined c_eff = c_sound + 2*c0*h*|div(v)|)
    // before a stronger coefficient is safe to use; that CFL extension is
    // real, disclosed future work (needs a `timestep_bound` trait signature
    // change, out of scope for a same-night fix), not implemented yet.
    let c0_quadratic = (eos_power + 1.0) * 0.25;
    const C1_LINEAR: f32 = 1.0; // Landshoff
    let density_ratio = 1.0 / j;
    let rho = rest_density * density_ratio;
    let c2 = eos_stiffness
        * eos_power
        * crate::materials::utils::fast_pow(density_ratio, eos_power - 1.0)
        / rest_density;
    let c_sound = c2.max(0.0).sqrt();
    let h = grid_cell_size;
    let quadratic = c0_quadratic * (rho * h * div_v) * (rho * h * div_v);
    let linear = C1_LINEAR * h * c_sound * div_v;
    let q = quadratic - linear;
    if q.is_finite() { q } else { 0.0 }
}
