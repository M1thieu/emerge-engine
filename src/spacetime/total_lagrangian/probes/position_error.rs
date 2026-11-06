//! The step error measured on positions only, on every try (test builds,
//! `EMERGE_PROBE_POSITION_ERROR`, default off). A measurement: nothing here is
//! the engine's default.
//!
//! Source: Acary, Appl. Numer. Math. 62 (2012) 1259-1275, sec. 4.2, eq. 61:
//! the velocities' local error has no order at a contact switch, so the error
//! is taken on the coordinates, `q_1/2 - q_1`, at order `p = 0` (exponent 1).
//! The engine already does so on a try where a point reaches a wall; this
//! knob does it on every try. Adaptation, the one the engine declares: the
//! positions' error is the elastic part of the energy norm of the tries'
//! difference.

/// Whether the probe is on.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_POSITION_ERROR").is_some();
        if on {
            println!("knob active: the step error on positions only, every try");
        }
        on
    })
}
