//! FAST_MODE_PLAN.md sec. 50.6, a test-only measurement (knob
//! `EMERGE_PROBE_ONSET_CAP`, default off): when the error controller
//! refuses a try inside which a contact first holds, its next try is capped
//! at the impact's duration, so the controller does not halve its way down
//! from a frame-long try through two decades of refusals.
//!
//! The duration is the law's: `t_1 = 2 H / c`, the contact time of a body of
//! height `H` struck along that height, `c` its fastest P wave (Wang and
//! Hansen 2016, read, FAST_MODE_PLAN.md sec. 45: 0.443 ms for the V0 block).
//! `c = sqrt((K + mu) / rho)` with the code's `bulk + mu` and `rho = m / V0`,
//! as S1's fixed contact step takes it; `H` is the body's extent along the
//! walls' normal. Every kept step still passes the same error test with the
//! same tolerance: a choice of the next try's length, not of what is kept.
//! Adaptations, declared: the walls' first plane gives the normal; with
//! several bodies, the extent of all of them (one body in the measurement).

use glam::DVec2;

use super::contact::Plane;
use super::elastic::Elastic;

/// Whether the knob is set; says so once.
pub(super) fn knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_ONSET_CAP").is_some();
        if on {
            println!("knob active: onset cap, a refused try with a contact onset caps the next at 2 H / c (sec. 50.6)");
        }
        on
    })
}

/// `2 H / c` for the massive points, or `None` without a wall or matter.
pub(super) fn impact_time(
    x: &[DVec2],
    (mass, v0): (&[f64], &[f64]),
    solids: &[Elastic],
    planes: &[Plane],
) -> Option<f64> {
    let axis = planes.first()?.axis;
    let (mut low, mut high, mut speed) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for p in 0..x.len() {
        if mass[p] <= 0.0 || v0[p] <= 0.0 {
            continue;
        }
        low = low.min(x[p][axis]);
        high = high.max(x[p][axis]);
        speed = speed.max(((solids[p].bulk + solids[p].mu) * v0[p] / mass[p]).sqrt());
    }
    (speed > 0.0 && high > low).then(|| 2.0 * (high - low) / speed)
}

/// Whether `EMERGE_PROBE_NOELS_REJECT` asks for Noels et al.'s reduction on a
/// refused try (FAST_MODE_PLAN.md sec. 50.7); says so once.
pub(super) fn noels_knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_NOELS_REJECT").is_some();
        if on {
            println!("knob active: a refused try's next length times (target / (2 err))^(2/3) (sec. 50.7)");
        }
        on
    })
}

/// The factor on a refused try's length at the error `relative` (its error
/// over the target): Noels, Stainier, Ponthot and Bonini 2002, sec. 3.3,
/// eq. 32 with `eta = 3/2`, "at a rapid change in the physical problem
/// (impact...)": `RAT = [0.5 PRCU / e]^(2/3)`, this engine's target standing
/// for PRCU (an adaptation, declared: their error is the acceleration jump,
/// here it is the step-doubling estimate).
pub(super) fn noels_ratio(relative: f64) -> f64 {
    (0.5 / relative).powf(2.0 / 3.0)
}

/// Soderlind and Wang 2006 (J. Comput. Appl. Math. 185, eq. 10, kappa = 1):
/// the smooth limiter `1 + kappa arctan((rho - 1) / kappa)` on the step
/// ratio `rho = (1 / relative)^(1 / (order + 1))` (eqs. 64 and 66), in place
/// of the clamp to [0.5, 5], when `EMERGE_PROBE_SODERLIND_LIMIT` is set
/// (FAST_MODE_PLAN.md sec. 59.2); `None` otherwise. With kappa = 1 the
/// increase is at most `1 + pi / 2` and the reduction at least `1 - pi / 4`.
pub(super) fn soderlind_ratio(relative: f64, order: f64) -> Option<f64> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_SODERLIND_LIMIT").is_some();
        if on {
            println!("knob active: Soderlind and Wang's arctan limiter on the step ratio, kappa 1 (sec. 59.2)");
        }
        on
    });
    on.then(|| {
        let rho = (1.0 / relative.max(f64::MIN_POSITIVE)).powf(1.0 / (order + 1.0));
        1.0 + (rho - 1.0).atan()
    })
}

/// Whether `EMERGE_PROBE_LAND_ON_CONTACT` asks for the step that lands on the
/// predicted time to contact (FAST_MODE_PLAN.md sec. 61); says so once.
pub(super) fn land_knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_LAND_ON_CONTACT").is_some();
        if on {
            println!("knob active: a free step lands on the predicted time to contact (declared policy, sec. 61)");
        }
        on
    })
}

/// The smallest positive time at which a point reaches a wall in free
/// motion: the root of `g + v_n t + a_n t^2 / 2 = 0` over the points and
/// planes, `g` the gap along the plane's inward normal, `v_n` the normal
/// velocity, `a_n` gravity's normal part (FAST_MODE_PLAN.md sec. 61, a
/// declared step-control policy: it chooses one step's length, it does not
/// subdivide a step). `None` when no point approaches a wall.
pub(super) fn time_to_contact(
    x: &[DVec2],
    v: &[DVec2],
    planes: &[Plane],
    gravity: DVec2,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    for plane in planes {
        let a = plane.inward * gravity[plane.axis];
        for (q, w) in x.iter().zip(v) {
            let g = plane.inward * (q[plane.axis] - plane.at);
            if g <= 0.0 {
                continue;
            }
            let u = plane.inward * w[plane.axis];
            let root = if a == 0.0 {
                (u < 0.0).then(|| -g / u)
            } else {
                let disc = u * u - 2.0 * a * g;
                (disc >= 0.0)
                    .then(|| {
                        let s = disc.sqrt();
                        [(-u - s) / a, (-u + s) / a]
                            .into_iter()
                            .filter(|t| *t > 0.0)
                            .fold(f64::INFINITY, f64::min)
                    })
                    .filter(|t| t.is_finite())
            };
            if let Some(t) = root {
                best = Some(best.map_or(t, |b| b.min(t)));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point 0.1 m above a floor falling at 1 m/s under 9.81 m/s^2 reaches
    /// it at the free motion's root; a point rising away does not count.
    #[test]
    fn the_time_to_contact_is_the_free_motion_root() {
        let floor = Plane {
            axis: 1,
            at: 0.0,
            inward: 1.0,
        };
        let g = DVec2::new(0.0, -9.81);
        let t = time_to_contact(
            &[DVec2::new(0.0, 0.1)],
            &[DVec2::new(0.0, -1.0)],
            &[floor],
            g,
        )
        .expect("a root");
        let exact = (-1.0 + (1.0f64 + 2.0 * 9.81 * 0.1).sqrt()) / 9.81;
        assert!((t - exact).abs() < 1e-15, "{t} vs {exact}");
        let up = time_to_contact(
            &[DVec2::new(0.0, 0.1)],
            &[DVec2::new(0.0, 5.0)],
            &[floor],
            DVec2::ZERO,
        );
        assert!(up.is_none());
    }
}
