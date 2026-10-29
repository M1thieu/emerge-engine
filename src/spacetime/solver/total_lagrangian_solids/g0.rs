//! G0_PWAVE_ENERGY_PLAN.md sec. 2 (test builds, `EMERGE_PROBE_G0`): the
//! share of the energy the pressure waves carry. Two measures, printed
//! only: (1) per step, the stored energy split by the engine's own
//! isochoric/volumetric form (`Elastic::law_energy`'s fallback, `mu |F^ -
//! R^|^2` with `F^ = J^(-1/2) F`, and `K / 2 (J - 1)^2`, `K` the 2D bulk
//! `lambda + mu`), a DIAGNOSTIC split beside the law's own energy (the
//! corotated law mixes the two); (2) at a flight's start, each mode's
//! energy `|z_i|^2 / 2` (mass-normalized modes) and its volumetric
//! fraction, the 2D small-strain split `lambda/2 (tr e)^2 + mu |e|^2 =
//! (lambda + mu)/2 (tr e)^2 + mu |dev e|^2` (DERIVED from the isotropic law
//! `sigma = lambda tr(e) I + 2 mu e`, Telikicherla and Moutsanidis 2024 eq.
//! 36) over the mode's strain energy `omega_i^2 / 2`, the particles as
//! quadrature.

use std::cell::Cell;

use super::TotalLagrangianState;
use super::flight::Flight;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::matter::kinematics::polar_decomposition::polar_decomposition_2d;
use crate::spacetime::total_lagrangian::Modes;

thread_local! {
    /// Per frame: the largest volumetric share over its steps, the stored
    /// energy at that step, and the steps seen.
    static FRAME: Cell<(f64, f64, u32)> = const { Cell::new((0.0, 0.0, 0)) };
}

/// Whether `EMERGE_PROBE_G0` is set (`steps` also prints every step); says
/// so once, from inside.
pub(super) fn on() -> bool {
    static SAID: std::sync::Once = std::sync::Once::new();
    let on = std::env::var_os("EMERGE_PROBE_G0").is_some();
    if on {
        SAID.call_once(|| println!("knob active: G0 prints (G0_PWAVE_ENERGY_PLAN.md)"));
    }
    on
}

/// The state's stored energy: the law's own, and the diagnostic split's
/// isochoric and volumetric parts, J/m.
fn split(state: &TotalLagrangianState) -> (f64, f64, f64) {
    let (mut law, mut dev, mut vol) = (0.0, 0.0, 0.0);
    for p in 0..state.particles {
        let (solid, f, v0) = (&state.solids[p], state.f[p], state.v0[p]);
        let j = f.determinant();
        let fh = f * j.max(f64::MIN_POSITIVE).sqrt().recip();
        let d = fh - polar_decomposition_2d(fh);
        law += v0 * solid.energy(f);
        dev += v0 * solid.mu * (d.x_axis.length_squared() + d.y_axis.length_squared());
        vol += v0 * 0.5 * solid.bulk * (j - 1.0).powi(2);
    }
    (law, dev, vol)
}

/// After each kept step: the frame's running largest volumetric share.
pub(super) fn after_step(state: &TotalLagrangianState, dt: f64) {
    if !on() {
        return;
    }
    let (law, dev, vol) = split(state);
    let share = vol / (dev + vol).max(f64::MIN_POSITIVE);
    if std::env::var("EMERGE_PROBE_G0").is_ok_and(|v| v == "steps") {
        println!(
            "G0 step {dt:.4e} s: stored (law) {law:.6e} J/m, split isochoric {dev:.6e} + volumetric {vol:.6e} J/m, volumetric share {share:.4e}"
        );
    }
    FRAME.with(|c| {
        let (worst, at, steps) = c.get();
        c.set(if share > worst && law > 0.0 {
            (share, law, steps + 1)
        } else {
            (worst, at, steps + 1)
        });
    });
}

/// At a frame's end: the frame's largest volumetric share over its steps
/// and the end state's split.
pub(super) fn frame_end(state: &TotalLagrangianState, frame: u64) {
    if !on() {
        return;
    }
    let (worst, at, steps) = FRAME.with(|c| c.replace((0.0, 0.0, 0)));
    let (law, dev, vol) = split(state);
    println!(
        "G0 frame {frame}: {steps} steps, largest volumetric share {worst:.4e} (stored {at:.4e} J/m there); end: stored (law) {law:.6e} J/m, isochoric {dev:.6e}, volumetric {vol:.6e}, share {:.4e}",
        vol / (dev + vol).max(f64::MIN_POSITIVE)
    );
}

/// At a flight's start: per mode its energy and volumetric fraction, and
/// the P-wave-like energy `E_P = sum_i E_i f_i` with its share of the
/// vibration; the fractions' check (the quadrature's strain energy of each
/// mode against `omega_i^2 / 2`).
pub(super) fn at_flight_start(flight: &Flight, modes: &Modes, state: &TotalLagrangianState) {
    if !on() || flight.has_memory() {
        return;
    }
    let sizes = flight.mode_sizes();
    let (mut total, mut pressure, mut worst_check) = (0.0, 0.0, 0.0f64);
    let mut rows = Vec::new();
    for (i, (&omega, gradients)) in modes.omega.iter().zip(&modes.gradients).enumerate() {
        if omega <= 0.0 {
            continue;
        }
        let (mut volumetric, mut strain) = (0.0, 0.0);
        for (p, g) in gradients.iter().enumerate().take(state.particles) {
            let solid = &state.solids[p];
            let lambda = solid.bulk - solid.mu;
            let trace = g.x_axis.x + g.y_axis.y;
            let e = 0.5 * (*g + g.transpose());
            let e2 = e.x_axis.length_squared() + e.y_axis.length_squared();
            volumetric += state.v0[p] * 0.5 * solid.bulk * trace * trace;
            strain += state.v0[p] * (0.5 * lambda * trace * trace + solid.mu * e2);
        }
        let reference = 0.5 * omega * omega;
        worst_check = worst_check.max((strain / reference - 1.0).abs());
        let fraction = volumetric / reference;
        let energy = 0.5 * sizes.get(i).copied().unwrap_or(0.0).powi(2);
        total += energy;
        pressure += energy * fraction;
        rows.push((i, omega / std::f64::consts::TAU, energy, fraction));
    }
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!(
        "G0 flight start t {:.6}: vibration in the modes {total:.6e} J/m, P-wave-like E_P {pressure:.6e} J/m, share {:.4e}; fractions' check: worst |strain / (omega^2/2) - 1| {worst_check:.3e}",
        state.time,
        pressure / total.max(f64::MIN_POSITIVE)
    );
    for (i, hz, energy, fraction) in rows.iter().take(12) {
        println!(
            "  G0 mode {i}: {hz:.1} Hz, energy {energy:.4e} J/m ({:.4e} of it), volumetric fraction {fraction:.4e}",
            energy / total.max(f64::MIN_POSITIVE)
        );
    }
}

/// Part C's series (sec. 3): `EMERGE_PROBE_G0_KG=x` sets the 3D `K/G` to
/// `x` at the record's own shear modulus and density, `nu = (3x - 2) /
/// (2 (3x + 1))`, `E = 2 G (1 + nu)`; says so once, from inside. Unset, the
/// record unchanged.
pub(super) fn series(record: ElasticProps) -> ElasticProps {
    let Some(x) = std::env::var("EMERGE_PROBE_G0_KG")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    else {
        return record;
    };
    // The record is `f32`; the series is formed in `f64` and rounded once
    // (at `K/G` 1000, `1 - 2 nu` = 1e-3 carries a relative 3e-5 from it).
    let g = f64::from(record.e_pa) / (2.0 * (1.0 + f64::from(record.nu)));
    let nu = (3.0 * x - 2.0) / (2.0 * (3.0 * x + 1.0));
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        println!(
            "knob active: G0 series K/G {x} (G {g:.6e} Pa kept, nu {nu:.9}, E {:.6e} Pa)",
            2.0 * g * (1.0 + nu)
        )
    });
    ElasticProps {
        e_pa: (2.0 * g * (1.0 + nu)) as f32,
        nu: nu as f32,
        ..record
    }
}

/// The series' P-wave speed over the record's at the same `G` and `rho`,
/// `sqrt((K/G + 4/3) / (K0/G + 4/3))`, so reference A keeps its Courant
/// number by its own `c_p` (1 without the knob).
pub(super) fn wave_ratio(record: &ElasticProps) -> f64 {
    let Some(x) = std::env::var("EMERGE_PROBE_G0_KG")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    else {
        return 1.0;
    };
    let nu0 = f64::from(record.nu);
    let x0 = 2.0 * (1.0 + nu0) / (3.0 * (1.0 - 2.0 * nu0));
    ((x + 4.0 / 3.0) / (x0 + 4.0 / 3.0)).sqrt()
}
