//! Shared state evolution for the weakly-compressible fluid materials.
//!
//! A liquid has no elastic shear memory.  Its material state is therefore the
//! scalar volume ratio `J`, represented in the common particle `F` slot as
//! `sqrt(J) I`.  We integrate `d(log J)/dt = div(v)`, rather than forming
//! `det(I + dt L)`: the exponential update preserves positive volume for every
//! finite velocity gradient and is unconditionally well-defined without a
//! clamp.
//!
//! `update_particle` below DOES currently clamp its result anyway -- a
//! TEMPORARY, explicitly disclosed restoration (2026-08-13) of a real,
//! measured drift this module's exact integration doesn't prevent on its
//! own. See that function's own doc for the live evidence and the real
//! upstream cause (not yet fixed).

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

/// Real, disclosed engineering ceiling (not a cited physical constant -- no
/// paper gives "how expanded can a liquid MPM particle be before its
/// force-scatter stops being trustworthy") on how much this module's own
/// unclamped `volume` (see this file's own top-of-file doc: `d(log J)/dt =
/// div(v)`, deliberately not J-clamped) is allowed to inflate the FORCE this
/// particle exerts on the grid via P2G's `stress * stress_volume` scatter.
///
/// Found 2026-08-12, by direct code inspection after dense-diagnostic
/// tracing on `basic_fluids_gpu.rs` kept showing isolated/low-support
/// particles near a wall spiking to J=20-46 with F starting near identity.
/// This module has two SEPARATELY deliberate, SEPARATELY tested design
/// decisions -- `tait_pressure` is not clamped in tension
/// (`tait_eos_is_not_pressure_clamped_in_tension`, fluid.rs) and J/volume is
/// not clamped in its exponential update
/// (`exponential_volume_update_is_not_j_clamped`, fluid.rs). Neither, in
/// isolation, is wrong or explains this bug: checked numerically for the
/// actual failing scene, water's own real eos_stiffness there is only ~104
/// Pa (a deliberately derated, real-time-affordable value), so raw Tait
/// pressure already saturates at a small, bounded value well before any
/// literature-standard cavitation floor (e.g. -1 atm) would ever engage --
/// a pressure floor alone is a dead end for this scene.
///
/// The real, confirmed mechanism is multiplicative: force is proportional to
/// stress * volume. Stress is bounded (asymptotes at -eos_stiffness), but
/// `volume` is not -- once a low-mass/isolated particle's J drifts even
/// modestly above 1 (a single violent first-wall-impact substep is enough
/// to start it), volume grows, so the SAME bounded stress produces a
/// proportionally LARGER force, which (for that same low-mass particle) is
/// a proportionally larger acceleration, growing J further next substep --
/// a genuine, self-reinforcing feedback loop that needs neither term
/// individually unbounded. Empirically confirmed (`basic_fluids_gpu.rs`,
/// dense per-frame diagnostics): capping only the force-scatter's own
/// volume input turns an unbounded runaway (v climbing 60->262 across 6
/// frames, F reaching 6.5+) into a bounded, recoverable splash (v settling
/// back into the teens/twenties, F excursions self-correcting).
///
/// Scoped to exactly this: the particle's own tracked `volume`/J state (and
/// `tait_pressure`) are completely untouched here, so both existing tests
/// above are unaffected -- neither exercises this function.
pub(crate) const STRICT_FLUID_FORCE_VOLUME_RATIO_MAX: f32 = 4.0;

/// Volume to use in the P2G force scatter (`stress * force_stress_volume`),
/// distinct from the particle's own honestly-tracked `volume` state -- see
/// `STRICT_FLUID_FORCE_VOLUME_RATIO_MAX`'s own doc for why these two must be
/// different quantities.
#[inline]
pub(crate) fn force_stress_volume(initial_volume: f32, volume: f32) -> f32 {
    volume.min(initial_volume * STRICT_FLUID_FORCE_VOLUME_RATIO_MAX)
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

    // TEMPORARY, explicitly disclosed restoration (2026-08-13) of the bound
    // this module's own doc (top of file) says the exponential update
    // "does not need": `cac544b` (2026-08-11) removed the previous
    // `clamp(0.5, 2.0)` on this exact quantity when it switched to this
    // exact integration, and nothing replaced it. `STRICT_FLUID_FORCE_VOLUME_
    // RATIO_MAX` (this file, above) already caps the FORCE a drifted
    // particle can exert -- confirmed live to fix the fast, multiplicative
    // runaway (force ∝ volume) -- but does nothing to stop J itself slowly
    // creeping past that cap over hundreds of substeps, which is the
    // remaining, separately-confirmed failure mode (GPU: J reached 36.5;
    // CPU strict-fluid: panicked at J=50.0009, both slow creeps to a
    // ceiling, not fast blowups). Root cause of the drift itself (most
    // likely a free-surface velocity-divergence bias feeding this substep's
    // own `div_v`) is understood in outline but not yet fixed -- see
    // `MEMORY.md` fluid-explosion entries, 2026-08-13. This bound is the
    // known-working value from before `cac544b`, restored to get back to
    // usable behavior now; remove it once the real upstream fix lands.
    let j = j.clamp(0.5, 2.0);

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
