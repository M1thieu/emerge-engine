//! A trial, test builds only: the lone body's flight by an exponential
//! discrete-gradient step on the plane floating-frame equations, selected by
//! `EMERGE_PROBE_FLIGHT=exponential` or a probe's `set_knob`. Off, nothing
//! of the path changes.
//!
//! State `y = (theta, q, p_theta, p_q)`, Hamiltonian `H = 1/2 y^T M0 y + U`
//! (Li and Wu, arXiv 2012.13064, eq. 3), `M0 = diag(0, Omega^2, 1 / I0,
//! Id)`, `I0` today's start inertia, `U = (p_theta - k . p_q)^2 / (2 s) -
//! p_theta^2 / (2 I0)`, `k = l - G q`, `s = I(q) - |k|^2`,
//! `I(q) = Theta_u + 2 d . q + |q|^2` (the floating-frame kinetic energy,
//! ROUTE_A_PAPER.md, after Zwolfer and Gerstmayr 2023, eqs. 115-123). The
//! step is Li and Wu's scheme (7), `y1 = exp(V) y0 + h phi(V) J gbar U`,
//! exact in `H` for any `h` (their Theorem 2.3), with Gonzalez's discrete
//! gradient (1996, Prop. 3.6) on `(q, p_theta, p_q)` in the energy
//! coordinates `zeta = (omega q, -p_q)`, so `p_theta` is kept exactly
//! . Per mode the update is
//! `zeta1 = e^(i omega h) zeta0 + (e^(i omega h) - 1) gbar`,
//! solved by a fixed point whose contraction bound, stop rule, energy bound
//! and clearance bound are all computed here and
//! recorded per step (`Record`). Scope (declared): undamped corotated
//! bodies; any other keeps today's flight.

use std::cell::Cell;
use std::time::Instant;

use super::*;
use crate::spacetime::total_lagrangian::{ElasticLaw, RIGID_MODES};

const UNIT: f64 = 0.5 * f64::EPSILON;

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * UNIT;
    ku / (1.0 - ku)
}

/// The trial's settings: the path on, the truncated model (`G = d = l = 0`,
/// `s = I0`: today's flight, for criterion 6), and the scheme's steps per
/// flown interval (`V_fine`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Knob {
    pub(super) on: bool,
    pub(super) truncated: bool,
    pub(super) steps: usize,
    /// The law's own energy in the Hamiltonian (`law`): `EMERGE_PROBE_FLIGHT=law`.
    pub(super) law: bool,
}

thread_local! {
    static KNOB: Cell<Option<Knob>> = const { Cell::new(None) };
}

/// A probe's own settings for this thread (`None`: the environment's).
pub(super) fn set_knob(knob: Option<Knob>) {
    KNOB.with(|k| k.set(knob));
}

pub(super) fn knob() -> Knob {
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    let flight = std::env::var("EMERGE_PROBE_FLIGHT");
    let knob = KNOB.with(|k| k.get()).unwrap_or_else(|| Knob {
        on: matches!(flight.as_deref(), Ok("exponential" | "law")),
        truncated: std::env::var_os("EMERGE_PROBE_FLIGHT_TRUNCATED").is_some(),
        steps: std::env::var("EMERGE_PROBE_FLIGHT_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1),
        law: flight.as_deref() == Ok("law"),
    });
    if knob.on {
        ANNOUNCED.call_once(|| {
            println!(
                "knob active: turning flight (exponential discrete gradient){}, {} scheme steps a frame{}",
                if knob.law { " with the law's own energy" } else { "" },
                knob.steps,
                if knob.truncated {
                    ", truncated model"
                } else {
                    ""
                }
            );
        });
    }
    knob
}

/// How an attempt ended.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) enum Outcome {
    #[default]
    Flown,
    NotClear,
    EndGap,
    Unproven,
    NoConvergence,
    AngularMomentum,
}

/// One step of the scheme, for the probes.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Record {
    pub(super) t0: f64,
    pub(super) t1: f64,
    /// The flight's start time (one flight's steps share it).
    pub(super) start: f64,
    pub(super) outcome: Outcome,
    pub(super) iterations: u32,
    pub(super) k_max: u32,
    pub(super) rho_pred: f64,
    pub(super) rho_obs: f64,
    pub(super) m2: f64,
    /// `H(y1) - H(y0)`, J/m, and its bound `B_H` with its parts `E_stop`,
    /// `E_DG`, `E_upd` (free rotation included), `E_eval`.
    pub(super) delta_h: f64,
    pub(super) bound_h: f64,
    pub(super) parts: [f64; 4],
    pub(super) h_int: f64,
    pub(super) spin: f64,
    pub(super) flops: f64,
    pub(super) wall: f64,
    /// For the largest observed ratio: the two deltas it compares, and the
    /// iteration's noise floor `2 eps_F / (1 - rho_pred)`.
    pub(super) ratio_deltas: (f64, f64),
    pub(super) noise_floor: f64,
    /// With the law's energy, on a flown attempt's last step: the
    /// step's energy change from the attempt's start to its rebuilt end,
    /// J/m, and the round-off bound of the two evaluations (part (a)); on
    /// a flight's first step, the start's projection remainder (part (d));
    /// on every step, `|P - I|_F` (part (c)).
    pub(super) law: bool,
    pub(super) step_energy_change: f64,
    pub(super) step_energy_bound: f64,
    pub(super) start_remainder: Option<f64>,
    pub(super) gram_defect: f64,
}

/// The body's invariants , over its elastic
/// modes of positive frequency.
#[derive(Clone, Debug)]
struct Invariants {
    elastic: Vec<usize>,
    omega: Vec<f64>,
    omega_1: f64,
    theta_u: f64,
    /// `I0 - Theta_u`, a constant of the implemented `U`.
    delta0: f64,
    d: Vec<f64>,
    l: Vec<f64>,
    /// `G`, row-major, and a bound on `||G||_2`: `|x^T G y| = |sum m u x v|
    /// <= sqrt(x^T P x) sqrt(y^T P y)` with `P` the modes' particle Gram
    /// matrix (Cauchy-Schwarz), so `||G||_2 <= lambda_max(P) <= 1 + ||P -
    /// I||_F`, plus `P`'s own rounding (derived).
    g: Vec<f64>,
    g_norm: f64,
    i0: f64,
    truncated: bool,
    /// The law's part of the Hamiltonian (`law`), and `|P - I|_F`.
    law: Option<law::Law>,
    gram_defect: f64,
}

/// `U~` at a point, its error bound and, when asked, its gradient in
/// `zeta` (per mode) and `dU/dp_theta`, with `theta'`.
struct Eval {
    u: f64,
    error: f64,
    grad: Vec<DVec2>,
    grad_theta: f64,
    theta_dot: f64,
    k: Vec<f64>,
    /// The law's term (`law::Term`) when the Hamiltonian carries it.
    law: Option<law::Term>,
}

/// Counts of the operations, for the flops : matrix-vector
/// products with `G`, evaluations, rotations.
#[derive(Clone, Copy, Debug, Default)]
struct Count {
    matvec: f64,
    evals: f64,
    rotations: f64,
}

impl Count {
    fn flops(&self, n: usize) -> f64 {
        let n = n as f64;
        self.matvec * 2.0 * n * n + self.evals * 30.0 * n + self.rotations * 8.0
    }
}

impl Invariants {
    fn n(&self) -> usize {
        self.elastic.len()
    }

    fn times(&self, x: &[f64], count: &mut Count) -> (Vec<f64>, Vec<f64>) {
        let n = self.n();
        count.matvec += 1.0;
        let (mut y, mut a) = (vec![0.0; n], vec![0.0; n]);
        for r in 0..n {
            for (c, &xc) in x.iter().enumerate() {
                y[r] += self.g[r * n + c] * xc;
                a[r] += (self.g[r * n + c] * xc).abs();
            }
        }
        (y, a)
    }

    /// `U~(zeta)` with `p_theta`; the gradient when `grad`.
    fn eval(&self, zeta: &[DVec2], p_theta: f64, grad: bool, count: &mut Count) -> Eval {
        let n = self.n();
        count.evals += 1.0;
        if self.truncated {
            return Eval {
                u: 0.0,
                error: 0.0,
                grad: vec![DVec2::ZERO; n],
                grad_theta: 0.0,
                theta_dot: p_theta / self.i0,
                k: vec![0.0; n],
                law: None,
            };
        }
        let q: Vec<f64> = zeta.iter().zip(&self.omega).map(|(z, w)| z.x / w).collect();
        let p: Vec<f64> = zeta.iter().map(|z| -z.y).collect();
        let (gq, agq) = self.times(&q, count);
        let k: Vec<f64> = (0..n).map(|i| self.l[i] - gq[i]).collect();
        let ak: Vec<f64> = (0..n).map(|i| self.l[i].abs() + agq[i]).collect();
        let (mut kp, mut akp, mut dq, mut adq, mut qq, mut kk, mut akk) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            kp += k[i] * p[i];
            akp += ak[i] * p[i].abs();
            dq += self.d[i] * q[i];
            adq += (self.d[i] * q[i]).abs();
            qq += q[i] * q[i];
            kk += k[i] * k[i];
            akk += ak[i] * ak[i];
        }
        let s = self.theta_u + 2.0 * dq + qq - kk;
        let ims = self.delta0 - 2.0 * dq - qq + kk;
        let a_s = self.theta_u + 2.0 * adq + qq + akk;
        let a_ims = self.delta0.abs() + 2.0 * adq + qq + akk;
        let s_lo = s - gamma(2 * n + 9) * a_s;
        let t1 = -kp * (2.0 * p_theta - kp) / (2.0 * s);
        let t2 = p_theta * p_theta * ims / (2.0 * s * self.i0);
        let u = t1 + t2;
        let a1 = akp * (2.0 * p_theta.abs() + akp) / (2.0 * s_lo);
        let a2 = p_theta * p_theta * a_ims / (2.0 * s_lo * self.i0);
        let error = gamma(2 * n + 16) * (a1 + a2) * (a_s / s_lo) + UNIT * u.abs();
        let theta_dot = (p_theta - kp) / s;
        let term = self.law.as_ref().map(|law| law.at(&q, &self.omega, grad));
        let (u, error) = match &term {
            Some(t) => (u + t.u, error + t.error + UNIT * (u + t.u).abs()),
            None => (u, error),
        };
        if !grad {
            return Eval {
                u,
                error,
                grad: Vec::new(),
                grad_theta: 0.0,
                theta_dot,
                k,
                law: term,
            };
        }
        let grad_theta = (-kp * self.i0 + p_theta * ims) / (s * self.i0);
        let (gp, _) = self.times(&p, count);
        let (gk, _) = self.times(&k, count);
        let grad = (0..n)
            .map(|i| {
                let dq_i =
                    -theta_dot * gp[i] - theta_dot * theta_dot * ((self.d[i] + q[i]) - gk[i]);
                let dp_i = -theta_dot * k[i];
                let dq_i = dq_i + term.as_ref().map_or(0.0, |t| t.grad_q[i]);
                // `zeta = (omega q, -p)`: `dU~/dzeta = (dU/dq / omega, -dU/dp)`.
                DVec2::new(dq_i / self.omega[i], -dp_i)
            })
            .collect();
        Eval {
            u,
            error,
            grad,
            grad_theta,
            theta_dot,
            k,
            law: term,
        }
    }

    /// `U~(z1) - U~(z0)`, the law's term (when carried) from its exact
    /// increments (`law::Law::change`), and that part's round-off bound.
    fn difference(&self, (z0, e0): (&[DVec2], &Eval), (z1, e1): (&[DVec2], &Eval)) -> (f64, f64) {
        match (&self.law, &e0.law, &e1.law) {
            (Some(law), Some(t0), Some(t1)) => {
                let q = |z: &[DVec2]| -> Vec<f64> {
                    z.iter().zip(&self.omega).map(|(z, w)| z.x / w).collect()
                };
                let (change, error) = law.change((t0, t1), (&q(z0), &q(z1)), &self.omega);
                ((e1.u - t1.u) - (e0.u - t0.u) + change, error)
            }
            _ => (e1.u - e0.u, 0.0),
        }
    }

    /// `M2` over `B(0, R)` for `p_theta`, and `s_min` (`None`
    /// when not positive: unproven).
    fn m2(&self, radius: f64, p_theta: f64) -> Option<(f64, f64)> {
        if self.truncated {
            return Some((0.0, self.i0));
        }
        let g = self.g_norm;
        let rq = radius / self.omega_1;
        let norm = |v: &[f64]| v.iter().map(|e| e * e).sum::<f64>().sqrt();
        let (l, d) = (norm(&self.l), norm(&self.d));
        let big_k = l + g * rq;
        let s0 = self.theta_u - l * l;
        // `grad s(0) = 2 d - 2 G l`.
        let mut gl = vec![0.0; self.n()];
        for (r, out) in gl.iter_mut().enumerate() {
            for (c, &lc) in self.l.iter().enumerate() {
                *out += self.g[r * self.n() + c] * lc;
            }
        }
        let grad_s: f64 = self
            .d
            .iter()
            .zip(&gl)
            .map(|(d, gl)| (2.0 * d - 2.0 * gl).powi(2))
            .sum::<f64>()
            .sqrt();
        let s_min = s0 - grad_s * rq - (g * g - 1.0).max(0.0) * rq * rq;
        let s_min = s_min * (1.0 - gamma(4 * self.n()));
        if s_min <= 0.0 {
            return None;
        }
        let th = (p_theta.abs() + big_k * radius) / s_min;
        let w = g * radius + 2.0 * th * (d + rq + g * big_k);
        let w1 = self.omega_1;
        let m2 = (w * w / (w1 * w1) + big_k * big_k) / s_min
            + th * th * (g * g - 1.0).max(1.0) / (w1 * w1)
            + th * g / w1;
        Some((m2 * (1.0 + gamma(16)), s_min))
    }
}

/// The flight under way.
#[derive(Clone, Debug)]
pub(super) struct Exponential {
    t_start: f64,
    /// The time of the state the flight last wrote.
    t: f64,
    centre: DVec2,
    velocity: DVec2,
    origin: DVec2,
    angle: f64,
    spin: f64,
    /// The angle's discrete-gradient corrections, `sum h dU/dp_theta`.
    extra: f64,
    /// `p_theta`, the flight's angular momentum.
    angular: f64,
    /// Per elastic mode, `zeta` in the free flow's frame.
    w: Vec<DVec2>,
    inv: Invariants,
    /// Per contact point: its reference arm, today's vibration bound
    /// (the truncated model's own) and the Cauchy-Schwarz coefficient
    /// `(sum_i |phi_i|^2 / omega_i^2)^(1/2)`.
    arms: Vec<(DVec2, f64, f64)>,
    /// Per mode (all of them), `q'` in the last rebuilt state, and the
    /// modes' leak (today's `keeps_angular_momentum` tolerance).
    rates: Vec<f64>,
    leak: Vec<f64>,
    /// `H_int` at `t`.
    h_int: f64,
    /// With the law's energy: the start's projection remainder, J/m, until
    /// the first attempt records it.
    start_remainder: Option<f64>,
}

/// `e^(i a)` as `(cos, sin)`, its measured norm defect `|c^2 + s^2 - 1|`
/// and a bound on its angle's error: `4 u |a|` (the argument's own
/// rounding and its reduction) plus `atan2(s, c)` against the reduced `a`.
fn turn(a: f64) -> (DVec2, f64, f64) {
    let e = DVec2::from_angle(a);
    let norm = e.x.mul_add(e.x, e.y.mul_add(e.y, -1.0)).abs();
    let reduced = a - std::f64::consts::TAU * (a / std::f64::consts::TAU).round();
    let angle = 4.0 * UNIT * a.abs() + (e.y.atan2(e.x) - reduced).abs();
    (e, norm, angle)
}

fn times(a: DVec2, b: DVec2) -> DVec2 {
    DVec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
}

fn length(v: &[DVec2]) -> f64 {
    v.iter().map(|z| z.length_squared()).sum::<f64>().sqrt()
}

impl Exponential {
    pub(super) fn start(
        state: &TotalLagrangianState,
        modes: &Modes,
        t: f64,
        (truncated, with_law): (bool, bool),
    ) -> Option<Self> {
        let n = state.particles;
        let total: f64 = state.mass[..n].iter().sum();
        if total <= 0.0 {
            return None;
        }
        // Today's start (`Flight::start`), the same operations in order.
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
        let a = (0..n).fold(DMat2::ZERO, |a, p| {
            let (r, s) = (state.x[p] - centre, state.reference[p] - origin);
            a + state.mass[p] * DMat2::from_cols(r * s.x, r * s.y)
        });
        let angle = (a.x_axis.y - a.y_axis.x).atan2(a.x_axis.x + a.y_axis.y);
        let back = DMat2::from_angle(-angle);
        let elastic: Vec<usize> = (RIGID_MODES..modes.omega.len())
            .filter(|&i| modes.omega[i] > 0.0)
            .collect();
        let omega: Vec<f64> = elastic.iter().map(|&i| modes.omega[i]).collect();
        // `zeta = (omega q, -p_q)`: the truncated model's `p_q` is today's
        // rate (the spin removed), the full model's the projection of the
        // body-frame velocity, nothing removed.
        let w: Vec<DVec2> = elastic
            .iter()
            .map(|&i| {
                let at = &modes.at[i];
                let (q, rate) = (0..n).fold((0.0, 0.0), |(q, rate), p| {
                    let arm = state.x[p] - centre;
                    let u = back * arm - (state.reference[p] - origin);
                    let v = if truncated {
                        back * (state.v[p] - velocity - spin * arm.perp())
                    } else {
                        back * (state.v[p] - velocity)
                    };
                    (
                        q + state.mass[p] * u.dot(at[p]),
                        rate + state.mass[p] * v.dot(at[p]),
                    )
                });
                DVec2::new(modes.omega[i] * q, -rate)
            })
            .collect();
        let leak: Vec<f64> = (0..modes.omega.len())
            .map(|i| {
                if i < RIGID_MODES {
                    return 0.0;
                }
                (0..n)
                    .map(|p| state.mass[p] * (state.reference[p] - origin).perp_dot(modes.at[i][p]))
                    .sum()
            })
            .collect();
        let m = elastic.len();
        let reference = |p: usize| state.reference[p] - origin;
        let theta_u: f64 = (0..n)
            .map(|p| state.mass[p] * reference(p).length_squared())
            .sum();
        let (mut d, mut l, mut g) = (vec![0.0; m], vec![0.0; m], vec![0.0; m * m]);
        if !truncated {
            for (a, &i) in elastic.iter().enumerate() {
                for p in 0..n {
                    let mass = state.mass[p];
                    d[a] += mass * reference(p).dot(modes.at[i][p]);
                    l[a] += mass * reference(p).perp_dot(modes.at[i][p]);
                }
                for (b, &j) in elastic.iter().enumerate() {
                    g[a * m + b] = (0..n)
                        .map(|p| state.mass[p] * modes.at[i][p].perp_dot(modes.at[j][p]))
                        .sum();
                }
            }
        }
        let mut gram_defect = 0.0;
        let g_norm = if truncated {
            0.0
        } else {
            let mut defect = 0.0;
            for (a, &i) in elastic.iter().enumerate() {
                for (b, &j) in elastic.iter().enumerate() {
                    let gram: f64 = (0..n)
                        .map(|p| state.mass[p] * modes.at[i][p].dot(modes.at[j][p]))
                        .sum();
                    let e = gram - if a == b { 1.0 } else { 0.0 };
                    defect += e * e;
                }
            }
            gram_defect = defect.sqrt();
            1.0 + defect.sqrt() + gamma(2 * n + 4) * m as f64
        };
        let omega_1 = omega.iter().copied().fold(f64::INFINITY, f64::min);
        let arms = state
            .outlines
            .iter()
            .flatten()
            .map(|&o| {
                // Today's bound, its operations in order (`Flight::start`).
                let today: f64 = (RIGID_MODES..modes.omega.len())
                    .filter(|&i| modes.omega[i] > 0.0)
                    .map(|i| {
                        let k = elastic.iter().position(|&e| e == i).expect("elastic");
                        modes.at[i][o].length() * w[k].length() / modes.omega[i]
                    })
                    .sum();
                let schwarz = elastic
                    .iter()
                    .map(|&i| modes.at[i][o].length_squared() / (modes.omega[i] * modes.omega[i]))
                    .sum::<f64>()
                    .sqrt();
                (state.reference[o] - origin, today, schwarz)
            })
            .collect();
        let law = (with_law && !truncated).then(|| law::Law::new(state, modes, (&elastic, &omega)));
        let inv = Invariants {
            law,
            gram_defect,
            elastic,
            omega,
            omega_1,
            theta_u,
            delta0: inertia - theta_u,
            d,
            l,
            g,
            g_norm,
            i0: inertia,
            truncated,
        };
        let mut count = Count::default();
        let h_int = 0.5 * length(&w).powi(2)
            + momentum * momentum / (2.0 * inertia)
            + inv.eval(&w, momentum, false, &mut count).u;
        Some(Self {
            t_start: t,
            t,
            centre,
            velocity,
            origin,
            angle,
            spin,
            extra: 0.0,
            angular: momentum,
            w,
            inv,
            arms,
            rates: Vec::new(),
            leak,
            h_int,
            start_remainder: None,
        })
    }

    fn rigid_at(&self, gravity: DVec2, t: f64) -> (DVec2, DVec2) {
        let tau = t - self.t_start;
        (
            self.centre + self.velocity * tau + 0.5 * gravity * tau * tau,
            self.velocity + gravity * tau,
        )
    }

    fn angle_at(&self, t: f64) -> f64 {
        let a = self.angle + self.spin * (t - self.t_start);
        if self.extra == 0.0 { a } else { a + self.extra }
    }

    /// `zeta` at time `t` from the free-frame state.
    fn zeta_at(&self, t: f64) -> Vec<DVec2> {
        let tau = t - self.t_start;
        self.w
            .iter()
            .zip(&self.inv.omega)
            .map(|(w, &o)| DVec2::from_angle(o * tau).rotate(*w))
            .collect()
    }

    /// Whether every contact point stays clear over `[t0, t1]`, `t0` the
    /// flight's own time: the truncated model with today's bound, the full
    /// one with the clearance bound.
    fn clear(&self, gravity: DVec2, planes: &[Plane], (t0, t1): (f64, f64)) -> bool {
        let inv = &self.inv;
        if let Some(law) = &inv.law {
            return self.clear_with_law(law, gravity, planes, (t0, t1));
        }
        let (spin_bound, vibration) = if inv.truncated {
            (self.spin.abs(), None)
        } else {
            let rq0 = (2.0 * self.h_int).sqrt() / inv.omega_1;
            let d: f64 = inv.d.iter().map(|e| e * e).sum::<f64>().sqrt();
            let i_max = inv.theta_u + 2.0 * d * rq0 + rq0 * rq0;
            let e_v = (self.h_int - self.angular * self.angular / (2.0 * i_max)).max(0.0);
            let e_v = e_v + gamma(4 * inv.n()) * self.h_int;
            let radius = (2.0 * e_v).sqrt();
            let Some((_, s_min)) = inv.m2(radius, self.angular) else {
                return false;
            };
            let l: f64 = inv.l.iter().map(|e| e * e).sum::<f64>().sqrt();
            let k_c = l + inv.g_norm * radius / inv.omega_1;
            let p = (2.0 * self.h_int).sqrt();
            ((self.angular.abs() + k_c * p) / s_min, Some(radius))
        };
        planes.iter().all(|plane| {
            let along = |t: f64| plane.inward * self.rigid_at(gravity, t).0[plane.axis];
            let mut lowest = along(t0).min(along(t1));
            let acceleration = plane.inward * gravity[plane.axis];
            if acceleration != 0.0 {
                let vertex = self.t_start - plane.inward * self.velocity[plane.axis] / acceleration;
                if vertex > t0 && vertex < t1 {
                    lowest = lowest.min(along(vertex));
                }
            }
            let turned = DMat2::from_angle(self.angle_at(t0));
            self.arms.iter().all(|&(reference, today, schwarz)| {
                let arm = turned * reference;
                let bound = vibration.map_or(today, |r| r * schwarz);
                let turning = spin_bound * reference.length() * (t1 - t0);
                lowest + plane.inward * arm[plane.axis] - plane.inward * plane.at - bound - turning
                    > 0.0
            })
        })
    }

    /// One step of the scheme from `self.t` to `t1`; the record, or why it
    /// was refused (the flight unchanged then).
    pub(super) fn step(&mut self, t1: f64) -> Result<Record, Outcome> {
        let clock = Instant::now();
        let mut count = Count::default();
        let (ta, h) = (self.t, t1 - self.t);
        let (tau_a, tau_b) = (ta - self.t_start, t1 - self.t_start);
        let inv = self.inv.clone();
        let n = inv.n();
        let p_theta = self.angular;
        let rot_a: Vec<(DVec2, f64, f64)> = inv.omega.iter().map(|&o| turn(o * tau_a)).collect();
        let rot_b: Vec<(DVec2, f64, f64)> = inv.omega.iter().map(|&o| turn(o * tau_b)).collect();
        let rot_h: Vec<(DVec2, f64, f64)> = inv.omega.iter().map(|&o| turn(o * h)).collect();
        count.rotations += 3.0 * n as f64;
        let zeta0: Vec<DVec2> = (0..n).map(|i| rot_a[i].0.rotate(self.w[i])).collect();
        let free: Vec<DVec2> = (0..n).map(|i| rot_b[i].0.rotate(self.w[i])).collect();
        let eh: Vec<DVec2> = rot_h.iter().map(|r| r.0 - DVec2::X).collect();
        let sigma = eh.iter().map(|e| 0.5 * e.length()).fold(0.0, f64::max) * (1.0 + gamma(4));
        let e0 = inv.eval(&zeta0, p_theta, false, &mut count);
        let norm0 = length(&zeta0);
        // The discrete gradient at `(zeta0, z)`.
        let gradient = |z: &[DVec2], m2: f64, count: &mut Count| {
            let mid: Vec<DVec2> = (0..n).map(|i| 0.5 * (zeta0[i] + z[i])).collect();
            let at_mid = inv.eval(&mid, p_theta, true, count);
            let e1 = inv.eval(z, p_theta, false, count);
            let delta: Vec<DVec2> = (0..n).map(|i| z[i] - zeta0[i]).collect();
            let (mut dot, mut adot, mut squared) = (0.0, 0.0, 0.0);
            for (g, d) in at_mid.grad.iter().zip(&delta) {
                dot += g.dot(*d);
                adot += (g.x * d.x).abs() + (g.y * d.y).abs();
                squared += d.length_squared();
            }
            // With the law's term, its difference from the law's exact
            // increments (`law::Law::change`).
            let (difference, change_error) = inv.difference((&zeta0, &e0), (z, &e1));
            let r = difference - dot;
            let r_error = e1.error
                + e0.error
                + change_error
                + gamma(2 * n + 2) * adot
                + 3.0 * UNIT * (e1.u.abs() + e0.u.abs());
            let mut g = at_mid.grad.clone();
            let mut dropped = 0.0;
            // With the law's term `M2` does not bound its curvature: the quotient is formed whenever `r` exceeds its own
            // round-off, and dropped (`|r|` booked) otherwise.
            let formed = if inv.law.is_some() {
                squared > 0.0 && r.abs() > r_error
            } else {
                squared > 0.0 && 0.25 * m2 * squared > r_error
            };
            if formed {
                let c = r / squared;
                for i in 0..n {
                    g[i] += c * delta[i];
                }
            } else if inv.law.is_some() {
                dropped = r.abs();
            } else {
                dropped = 0.25 * m2 * squared;
            }
            let defect = r_error + gamma(2 * n + 4) * r.abs() + dropped;
            (g, at_mid.grad_theta, defect)
        };
        let update =
            |g: &[DVec2]| -> Vec<DVec2> { (0..n).map(|i| free[i] + times(eh[i], g[i])).collect() };
        // The first iterate from the free flow, `delta_1`, then the ball's
        // `M2` and `rho_pred`.
        let m2_start = inv.m2(norm0, p_theta).map_or(f64::INFINITY, |m| m.0);
        let (g_first, g_theta_first, defect_first) = gradient(&free, m2_start, &mut count);
        let next = update(&g_first);
        let delta_1 = (0..n)
            .map(|i| (next[i] - free[i]).length_squared())
            .sum::<f64>()
            .sqrt();
        let Some((m2, _)) = inv.m2(norm0 + 2.0 * delta_1, p_theta) else {
            handoff::trace(|| {
                format!("t {ta:.6}: exponential flight: unproven, s_min not positive")
            });
            return Err(Outcome::Unproven);
        };
        let rho = 3.5 * sigma * m2;
        if rho > 0.5 {
            handoff::trace(|| {
                format!("t {ta:.6}: exponential flight: unproven, rho_pred {rho:.3e}")
            });
            return Err(Outcome::Unproven);
        }
        // `eps_F`: one evaluation's round-off in `|.|_E` : the
        // free rotation's and the increment's per mode.
        let eps_f = {
            let mut sum = 0.0;
            for i in 0..n {
                let inc = times(eh[i], g_first[i]).length();
                let e = gamma(4) * (self.w[i].length() + inc)
                    + g_first[i].length() * (rot_h[i].1 + rot_h[i].2 + 2.0 * UNIT);
                sum += e * e;
            }
            // With the law's term, its gradient's own round-off (absolute
            // near the reference: `law::Law::at`), per mode through the
            // update `(e^(i omega h) - 1) gbar`.
            let law_noise = inv.law.as_ref().map_or(0.0, |law| {
                let q: Vec<f64> = (0..n)
                    .map(|i| 0.5 * (zeta0[i].x + free[i].x) / inv.omega[i])
                    .collect();
                law.at(&q, &inv.omega, true)
                    .noise
                    .iter()
                    .zip(&eh)
                    .zip(&inv.omega)
                    .map(|((e, h), w)| (h.length() * e / w).powi(2))
                    .sum::<f64>()
                    .sqrt()
            });
            sum.sqrt() + gamma(2 * n + 30) * length(&g_first) + law_noise
        };
        let stop = |rho: f64| -> (f64, u32) {
            let tau = 3.0 * eps_f / (1.0 - rho);
            let k_max = if delta_1 <= tau || rho == 0.0 {
                1
            } else {
                1 + ((eps_f / ((1.0 - rho) * delta_1)).ln() / rho.ln())
                    .ceil()
                    .max(0.0) as u32
                    + 2
            };
            (tau, k_max)
        };
        let (mut tau, mut k_max) = stop(rho);
        let (mut previous, mut zeta) = (free.clone(), next);
        let (mut g, mut g_theta, mut defect) = (g_first, g_theta_first, defect_first);
        let (mut iterations, mut last_delta, mut rho_obs) = (1u32, delta_1, 0.0f64);
        let mut ratio_deltas = (0.0, 0.0);
        while last_delta > tau {
            if iterations >= k_max {
                return Err(Outcome::NoConvergence);
            }
            let (g_new, g_theta_new, defect_new) = gradient(&zeta, m2, &mut count);
            let next = update(&g_new);
            let delta = (0..n)
                .map(|i| (next[i] - zeta[i]).length_squared())
                .sum::<f64>()
                .sqrt();
            if last_delta > 0.0 && delta / last_delta > rho_obs {
                rho_obs = delta / last_delta;
                ratio_deltas = (last_delta, delta);
                // The law's curvature is not in `M2`: the stop rule
                // and `K_max` from the observed ratio when it is larger.
                if inv.law.is_some() && rho_obs > rho && delta > 2.0 * eps_f / (1.0 - rho) {
                    if rho_obs > 0.5 {
                        handoff::trace(|| {
                            format!(
                                "t {ta:.6}: exponential flight: unproven, observed ratio {rho_obs:.3e} (deltas {last_delta:.3e} then {delta:.3e}, noise floor {:.3e}, iteration {iterations})",
                                2.0 * eps_f / (1.0 - rho)
                            )
                        });
                        return Err(Outcome::Unproven);
                    }
                    (tau, k_max) = stop(rho_obs);
                }
            }
            previous = std::mem::replace(&mut zeta, next);
            (g, g_theta, defect) = (g_new, g_theta_new, defect_new);
            last_delta = delta;
            iterations += 1;
        }
        // The flight's new state, in the free frame ; the
        // truncated model's `gbar` is exactly zero, its state unchanged.
        let w_new: Vec<DVec2> = if inv.truncated {
            self.w.clone()
        } else {
            (0..n)
                .map(|i| {
                    let inc = times(eh[i], g[i]);
                    let back = DVec2::new(rot_b[i].0.x, -rot_b[i].0.y);
                    self.w[i] + back.rotate(inc)
                })
                .collect()
        };
        let zeta1: Vec<DVec2> = (0..n).map(|i| rot_b[i].0.rotate(w_new[i])).collect();
        let e1 = inv.eval(&zeta1, p_theta, true, &mut count);
        let k0: f64 = 0.5 * zeta0.iter().map(|z| z.length_squared()).sum::<f64>();
        let k1: f64 = 0.5 * zeta1.iter().map(|z| z.length_squared()).sum::<f64>();
        let delta_h = (k1 - k0) + inv.difference((&zeta0, &e0), (&zeta1, &e1)).0;
        // `B_H`.
        let step_delta = (0..n)
            .map(|i| (zeta1[i] - previous[i]).length_squared())
            .sum::<f64>()
            .sqrt()
            * (1.0 + 2.0 * UNIT);
        let e_stop = (length(&e1.grad) + m2 * step_delta + length(&g)) * step_delta;
        let mut e_upd = 0.0;
        for i in 0..n {
            let inc = times(eh[i], g[i]).length();
            let w = self.w[i].length();
            let (norm_a, angle_a) = (rot_a[i].1, rot_a[i].2);
            let (norm_b, angle_b) = (rot_b[i].1, rot_b[i].2);
            let (norm_h, angle_h) = (rot_h[i].1, rot_h[i].2);
            let gi = g[i].length();
            let free_norm =
                w * w * (norm_a + norm_b) * 0.5 * (1.0 + gamma(4)) + gamma(3) * 2.0 * w * w;
            let angles =
                (inc + gi) * w * (angle_a + angle_b + 3.0 * UNIT * (inv.omega[i] * tau_b).abs());
            let rounding = std::f64::consts::SQRT_2 * UNIT * w_new[i].length()
                + inc * (norm_b + 2.0 * angle_b + gamma(4))
                + gi * (gamma(4) * eh[i].length() + angle_h + norm_h + 2.0 * UNIT);
            e_upd += free_norm + angles + (zeta1[i].length() + gi) * rounding;
        }
        let e_eval = gamma(2 * n + 1) * (k0 + k1)
            + e0.error
            + e1.error
            + 2.0 * UNIT * ((k1 - k0).abs() + (e1.u - e0.u).abs() + delta_h.abs());
        let bound_h = e_stop + defect + e_upd + e_eval;
        let rec = Record {
            t0: ta,
            t1,
            start: self.t_start,
            outcome: Outcome::Flown,
            iterations,
            k_max,
            rho_pred: rho,
            rho_obs,
            m2,
            delta_h,
            bound_h,
            parts: [e_stop, defect, e_upd, e_eval],
            h_int: self.h_int + delta_h,
            spin: e1.theta_dot,
            flops: count.flops(n),
            wall: clock.elapsed().as_secs_f64(),
            ratio_deltas,
            noise_floor: 2.0 * eps_f / (1.0 - rho),
            ..Default::default()
        };
        self.w = w_new;
        if g_theta != 0.0 {
            self.extra += h * g_theta;
        }
        self.h_int += delta_h;
        self.t = t1;
        Ok(rec)
    }

    /// The flight's state at its time `self.t`, written into the step's.
    pub(super) fn rebuild(
        &mut self,
        modes: &Modes,
        gravity: DVec2,
        state: &mut TotalLagrangianState,
    ) {
        let t = self.t;
        let (centre, velocity) = self.rigid_at(gravity, t);
        let zeta = self.zeta_at(t);
        let mut count = Count::default();
        let at = self.inv.eval(&zeta, self.angular, false, &mut count);
        let mut coordinates = vec![DVec2::ZERO; modes.omega.len()];
        for (k, &i) in self.inv.elastic.iter().enumerate() {
            let (q, p) = (zeta[k].x / modes.omega[i], -zeta[k].y);
            let rate = if self.inv.truncated {
                p
            } else {
                p - at.k[k] * at.theta_dot
            };
            coordinates[i] = DVec2::new(q, rate);
        }
        let spin = if self.inv.truncated {
            self.spin
        } else {
            at.theta_dot
        };
        self.rates = coordinates.iter().map(|c| c.y).collect();
        let turned = DMat2::from_angle(self.angle_at(t));
        for p in 0..state.x.len() {
            let (mut u, mut v) = (DVec2::ZERO, DVec2::ZERO);
            for (i, c) in coordinates.iter().enumerate().skip(RIGID_MODES) {
                u += modes.at[i][p] * c.x;
                v += modes.at[i][p] * c.y;
            }
            let arm = turned * (state.reference[p] - self.origin + u);
            state.x[p] = centre + arm;
            state.v[p] = velocity + spin * arm.perp() + turned * v;
            if p < state.particles {
                let g = coordinates
                    .iter()
                    .enumerate()
                    .skip(RIGID_MODES)
                    .fold(DMat2::ZERO, |g, (i, c)| g + modes.gradients[i][p] * c.x);
                state.f[p] = turned * (DMat2::IDENTITY + g);
            }
        }
    }

    /// Today's check (`Flight::keeps_angular_momentum`), its tolerance
    /// unchanged.
    fn keeps_angular_momentum(&self, state: &TotalLagrangianState) -> bool {
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
        let carried: f64 = self
            .leak
            .iter()
            .zip(&self.rates)
            .map(|(l, r)| (l * r).abs())
            .sum();
        (momentum - self.angular).abs() <= 2.0 * n as f64 * f64::EPSILON * size + carried
    }
}

#[path = "exponential_flight_driver.rs"]
mod driver;
#[path = "exponential_flight_law.rs"]
mod law;
pub(super) use driver::fly_frame;
pub(super) use law::lemma::report as lemma_report;
pub(super) use law::proof_report;
