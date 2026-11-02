//! FAST_MODE_PLAN.md sec. 45.4 (G-1 to G-3), a test-only measurement,
//! `EMERGE_PROBE_IMPACT_PATH`: after each kept step, its end time (the
//! frame's start plus the steps taken in it, so a flown part of the frame
//! is not counted), its length, the contacts held at its end, the largest
//! Green-Lagrange strain over the body (`|E|_F`, `E = (F^T F - I) / 2`)
//! and its most compressive vertical component, and the body's
//! mass-weighted mean velocity. Decides nothing.

use std::cell::Cell;

use glam::{DMat2, DVec2};

use super::TotalLagrangianState;

thread_local! {
    /// The frame's start time and the steps' time taken since it.
    static CLOCK: Cell<(f64, f64)> = const { Cell::new((f64::NAN, 0.0)) };
}

/// Whether `EMERGE_PROBE_IMPACT_PATH` is set; says so once.
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_IMPACT_PATH").is_some();
        if on {
            println!("knob active: impact path printed after each kept step (sec. 45.4)");
        }
        on
    })
}

/// After each kept step of `dt` seconds: one line.
pub(super) fn after_step(state: &TotalLagrangianState, dt: f64) {
    if !on() {
        return;
    }
    let (start, taken) = CLOCK.get();
    let taken = if start == state.time { taken + dt } else { dt };
    CLOCK.set((state.time, taken));
    let held = state.cache.as_ref().map_or(0, |c| c.contacts.len());
    let (mut largest, mut compressive) = (0.0f64, 0.0f64);
    let (mut mass, mut momentum) = (0.0, DVec2::ZERO);
    for p in (0..state.particles).filter(|&p| state.v0[p] > 0.0) {
        let f = state.f[p];
        let e = (f.transpose() * f - DMat2::IDENTITY) * 0.5;
        largest = largest.max((e.x_axis.length_squared() + e.y_axis.length_squared()).sqrt());
        compressive = compressive.min(e.y_axis.y);
        mass += state.mass[p];
        momentum += state.v[p] * state.mass[p];
    }
    let mean = momentum / mass.max(f64::MIN_POSITIVE);
    println!(
        "IMPACT-PATH t {:.9} dt {dt:.4e} held {held} largest |E| {largest:.4e} most compressive E_yy {compressive:+.4e} mean v ({:+.6e}, {:+.6e}) m/s",
        state.time + taken,
        mean.x,
        mean.y
    );
}
