//! A body's flight between contacts carried by its modes: the exact
//! solution of the linearised elastic problem in the body's frame, its
//! rigid motion exact under gravity, with no step at all. The stepper
//! resolves every contact; a flight starts only when the last step held
//! none and ends at the first frame the body could touch a wall, the
//! stepper taking over from the flight's exact state.
//!
//! The motion splits into its rigid part and its vibration:
//! - rigid: the centre of mass falls with `g` exactly; the rotation is the
//!   polar factor of `sum m (x - c) (X - C)^T`, turning at the spin that
//!   carries the angular momentum, `L / I`, so both are kept;
//! - vibration: the body-frame displacement and velocity projected on the
//!   elastic modes (`total_lagrangian::body_modes`), each advanced exactly
//!   as `z(t) = z(0) e^(i omega t)`, `z = omega q - i q'`, the undamped
//!   linear mode's own solution, its energy kept.
//!
//! Rebuilt into the step's own state, `x = c + R (X - C + u_b)`, `v = v_c +
//! Omega perp(x - c) + R v_b`, `F = R (I + grad u_b)`: the grid stays on the
//! reference, so the modal state is the step's and the hand-over needs no
//! projection.
//!
//! A turning body's frame adds terms these oscillators leave out. In a moving
//! frame the discretised displacement obeys
//! `M u'' + (D + G(Omega)) u' + K u + P(Omega') + N(Omega) u = R(Omega) + F`,
//! with `G = int 2 rho Phi^T Omega Phi`, `N = int rho Phi^T Omega^2 Phi` and
//! `R = -int rho Phi^T (R^T s'' + Omega' x + Omega^2 x)` (Legrand,
//! *Introduction to rotordynamics*, McGill Structural Dynamics and Vibration
//! Laboratory, 2009, slides 10-13, read). In the plane at constant spin,
//! `Omega = Omega J` (`J` the quarter turn), this is
//! `M u'' + 2 Omega M J u' + (K - Omega^2 M) u = Omega^2 M (X - C)`. On the
//! modes:
//! - the Coriolis coupling `2 Omega G q'`, `G_ij = phi_i^T M J phi_j`, at most
//!   `2 Omega / omega_1` of the stiffness term;
//! - the centrifugal softening, `(Omega / omega_1)^2`;
//! - the static centrifugal stretch.
//!
//! The vibration also changes `I`, so a constant spin does not keep `L`
//! exactly (measured: 2e-6 of `L` at 10 rad/s, 5e-4 at 50 rad/s on the
//! clay block). So a frame is flown only if its rebuilt state keeps the
//! flight's angular momentum to the sums' round-off; otherwise it is
//! stepped. In practice only a body that does not turn flies; a turning
//! body's flight needs those terms (not built).
//!
//! The material energy of the rebuilt state differs from the modes'
//! quadratic energy by the linear basis's residual (the law's
//! nonlinearity), reported per frame
//! (`TotalLagrangianFrame::modal_energy_change`). The step resumes from the
//! rebuilt state, so each hand-back keeps that residual of its moment
//! (measured: up to 1.6e-2 J/m, 2e-3 of the clay block's total, per flight).

use crate::spacetime::total_lagrangian::{Modes, Plane, RIGID_MODES};
use glam::{DMat2, DVec2};

use super::TotalLagrangianState;

/// One flight: its start time, the rigid state and each mode's `z` there.
#[derive(Clone, Debug)]
pub(crate) struct Flight {
    t: f64,
    centre: DVec2,
    velocity: DVec2,
    angle: f64,
    spin: f64,
    /// The reference centre of mass, `C`.
    origin: DVec2,
    /// Per mode, `z = omega q - i q'` in the body's frame, as `(re, im)`
    /// (zero for the rigid ones).
    z: Vec<DVec2>,
    /// The angular momentum about the centre at the start, kg m^2/s per
    /// metre.
    angular: f64,
    /// Per contact point, its reference arm `X - C`, and the bound on its
    /// vibration's displacement, `sum_i |phi_i(X)| |z_i| / omega_i`.
    arms: Vec<(DVec2, f64)>,
}

impl Flight {
    /// The flight from the state at time `t`.
    pub(crate) fn start(state: &TotalLagrangianState, modes: &Modes, t: f64) -> Option<Self> {
        let n = state.particles;
        let total: f64 = state.mass[..n].iter().sum();
        if total <= 0.0 {
            return None;
        }
        let (sum_x, sum_v, sum_reference) =
            (0..n).fold((DVec2::ZERO, DVec2::ZERO, DVec2::ZERO), |(a, b, c), p| {
                let m = state.mass[p];
                (
                    a + m * state.x[p],
                    b + m * state.v[p],
                    c + m * state.reference[p],
                )
            });
        let (centre, velocity, origin) = (sum_x / total, sum_v / total, sum_reference / total);
        let (momentum, inertia) = (0..n).fold((0.0, 0.0), |(l, i), p| {
            let arm = state.x[p] - centre;
            (
                l + state.mass[p] * arm.perp_dot(state.v[p] - velocity),
                i + state.mass[p] * arm.length_squared(),
            )
        });
        let spin = momentum / inertia.max(f64::MIN_POSITIVE);
        // The polar factor of `A = sum m (x - c) (X - C)^T`: in the plane,
        // the rotation by `atan2(A_10 - A_01, A_00 + A_11)`.
        let a = (0..n).fold(DMat2::ZERO, |a, p| {
            let (r, s) = (state.x[p] - centre, state.reference[p] - origin);
            a + state.mass[p] * DMat2::from_cols(r * s.x, r * s.y)
        });
        let angle = (a.x_axis.y - a.y_axis.x).atan2(a.x_axis.x + a.y_axis.y);
        let back = DMat2::from_angle(-angle);
        let z: Vec<DVec2> = modes
            .at
            .iter()
            .zip(&modes.omega)
            .enumerate()
            .map(|(i, (at, &w))| {
                if i < RIGID_MODES {
                    return DVec2::ZERO;
                }
                let (q, rate) = (0..n).fold((0.0, 0.0), |(q, rate), p| {
                    let arm = state.x[p] - centre;
                    let u = back * arm - (state.reference[p] - origin);
                    let v = back * (state.v[p] - velocity - spin * arm.perp());
                    (
                        q + state.mass[p] * u.dot(at[p]),
                        rate + state.mass[p] * v.dot(at[p]),
                    )
                });
                DVec2::new(w * q, -rate)
            })
            .collect();
        let arms = state
            .outlines
            .iter()
            .flatten()
            .map(|&o| {
                let bound: f64 = (RIGID_MODES..modes.omega.len())
                    .filter(|&i| modes.omega[i] > 0.0)
                    .map(|i| modes.at[i][o].length() * z[i].length() / modes.omega[i])
                    .sum();
                (state.reference[o] - origin, bound)
            })
            .collect();
        Some(Self {
            t,
            centre,
            velocity,
            angle,
            spin,
            angular: momentum,
            origin,
            z,
            arms,
        })
    }

    /// The centre of mass and its velocity at time `t`.
    fn rigid_at(&self, gravity: DVec2, t: f64) -> (DVec2, DVec2) {
        let tau = t - self.t;
        (
            self.centre + self.velocity * tau + 0.5 * gravity * tau * tau,
            self.velocity + gravity * tau,
        )
    }

    /// Whether every contact point stays clear of every wall over `[t0,
    /// t1]`, by a lower bound on its gap: the centre's exact parabola plus
    /// the turned reference arm, less the vibration's bound (each mode's
    /// displacement never exceeds `|z| / omega`).
    pub(crate) fn clear(&self, gravity: DVec2, planes: &[Plane], (t0, t1): (f64, f64)) -> bool {
        planes.iter().all(|plane| {
            // The least of `inward c_axis(tau)` over the interval: at an end,
            // or at the parabola's vertex when it lies inside.
            let along = |t: f64| plane.inward * self.rigid_at(gravity, t).0[plane.axis];
            let mut lowest = along(t0).min(along(t1));
            let acceleration = plane.inward * gravity[plane.axis];
            if acceleration != 0.0 {
                let vertex = self.t - plane.inward * self.velocity[plane.axis] / acceleration;
                if vertex > t0 && vertex < t1 {
                    lowest = lowest.min(along(vertex));
                }
            }
            // The arm turned to `t0`, and at most `|Omega| |r| (t1 - t0)` from
            // there over the interval (the turning point's speed).
            let turn = DMat2::from_angle(self.angle + self.spin * (t0 - self.t));
            self.arms.iter().all(|&(reference, bound)| {
                let arm = turn * reference;
                let turning = self.spin.abs() * reference.length() * (t1 - t0);
                lowest + plane.inward * arm[plane.axis] - plane.inward * plane.at - bound - turning
                    > 0.0
            })
        })
    }

    /// Whether `state` (rebuilt) keeps the flight's angular momentum to the
    /// round-off of the two sums, `2 n eps sum m |x - c| (|v| + |v_c|)`.
    pub(crate) fn keeps_angular_momentum(&self, state: &TotalLagrangianState) -> bool {
        let n = state.particles;
        let total: f64 = state.mass[..n].iter().sum();
        let (sum_x, sum_v) = (0..n).fold((DVec2::ZERO, DVec2::ZERO), |(a, b), p| {
            (
                a + state.mass[p] * state.x[p],
                b + state.mass[p] * state.v[p],
            )
        });
        let (centre, velocity) = (sum_x / total, sum_v / total);
        let (momentum, size) = (0..n).fold((0.0, 0.0), |(l, s), p| {
            let arm = state.x[p] - centre;
            (
                l + state.mass[p] * arm.perp_dot(state.v[p] - velocity),
                s + state.mass[p] * arm.length() * (state.v[p].length() + velocity.length()),
            )
        });
        (momentum - self.angular).abs() <= 2.0 * n as f64 * f64::EPSILON * size
    }

    /// The flight's state at time `t`, written into the step's state.
    pub(crate) fn rebuild(
        &self,
        modes: &Modes,
        gravity: DVec2,
        t: f64,
        state: &mut TotalLagrangianState,
    ) {
        let tau = t - self.t;
        let (centre, velocity) = self.rigid_at(gravity, t);
        // `q = Re z / omega`, `q' = -Im z`, each `z` turned by `omega tau`.
        let coordinates: Vec<DVec2> = self
            .z
            .iter()
            .zip(&modes.omega)
            .map(|(z0, &w)| {
                if w <= 0.0 {
                    return DVec2::ZERO;
                }
                let zt = DVec2::from_angle(w * tau).rotate(*z0);
                DVec2::new(zt.x / w, -zt.y)
            })
            .collect();
        let turn = DMat2::from_angle(self.angle + self.spin * tau);
        for p in 0..state.x.len() {
            let (mut u, mut v) = (DVec2::ZERO, DVec2::ZERO);
            for (i, c) in coordinates.iter().enumerate().skip(RIGID_MODES) {
                u += modes.at[i][p] * c.x;
                v += modes.at[i][p] * c.y;
            }
            let arm = turn * (state.reference[p] - self.origin + u);
            state.x[p] = centre + arm;
            state.v[p] = velocity + self.spin * arm.perp() + turn * v;
            if p < state.particles {
                let g = coordinates
                    .iter()
                    .enumerate()
                    .skip(RIGID_MODES)
                    .fold(DMat2::ZERO, |g, (i, c)| g + modes.gradients[i][p] * c.x);
                state.f[p] = turn * (DMat2::IDENTITY + g);
            }
        }
    }
}
