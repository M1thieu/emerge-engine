//! The turning flight with the law's own energy (FAST_MODE_PLAN.md sec. 26),
//! selected by `EMERGE_PROBE_FLIGHT=law`: the flight's Hamiltonian is the
//! floating frame's kinetic energy plus the step's own stored energy,
//! `sum_p V0_p W(C_p(q))`, `C_p = (I + G_p)^T (I + G_p)`, `G_p = sum_i q_i
//! grad phi_i(X_p)`, read through `Elastic` (the law's one definition). In
//! Li and Wu's split (arXiv 2012.13064 eq. 3) it is the part `U_e = sum_p
//! V0_p W(C_p) - 1/2 sum omega_i^2 q_i^2` of `U`. The basis is complete
//! (`body_modes`: as many modes as the step's unknowns), so `H` is the
//! step's energy of the rebuilt state; the clearance bound is rederived for
//! it (`W >= 0`, sec. 26).
//!
//! Declared, not exact: the kinetic form takes the modes' mass Gram matrix
//! as the identity (`|P - I|_F` at most 6.1e-14 measured), and one scheme
//! step a frame leaves the trajectory pre-asymptotic in the coupled fast
//! terms (LI_WU_TRIAL_RESULT.md 2.1); the energy is kept either way.

use super::*;
use crate::matter::kinematics::polar_decomposition::polar_decomposition_2d;
use crate::spacetime::total_lagrangian::Elastic;

// KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: the kinetic form takes the modes' mass Gram matrix as the identity (`|P - I|_F` at most 6.1e-14 measured); its share of each hand-back's energy change is within the "left" part counted in sec. 26a (at most 9.5e-14 J/m) | source: FAST_MODE_PLAN.md sec. 26, 26a | replace when: the kinetic form carries `P` | state: measured | gap: none
// KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: one scheme step a frame keeps the energy and the angular momentum but leaves the trajectory pre-asymptotic in the coupled fast terms | source: LI_WU_TRIAL_RESULT.md 2.1 | replace when: an `omega h`-uniform error bound or a step count from one is derived | state: measured on the trial | gap: none
fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

/// The law's part of the flight: each particle's solid and reference
/// volume, each elastic mode's material gradient at every particle, and
/// per contact point (the outlines' order) each elastic mode's value there
/// with `sigma_o = (sum_i |phi_i(X_o)|^2)^(1/2)`.
#[derive(Clone, Debug)]
pub(super) struct Law {
    solids: Vec<Elastic>,
    volume: Vec<f64>,
    gradients: Vec<Vec<DMat2>>,
    points: Vec<Vec<DVec2>>,
    sigma: Vec<f64>,
}

/// `U_e` at a point: its value, its round-off bound, its gradient in `q`
/// with each component's round-off bound (when asked) and each particle's
/// `C`, for the exact increments.
pub(super) struct Term {
    pub(super) u: f64,
    pub(super) error: f64,
    pub(super) grad_q: Vec<f64>,
    pub(super) noise: Vec<f64>,
    pub(super) c: Vec<DMat2>,
}

impl Law {
    pub(super) fn new(state: &TotalLagrangianState, modes: &Modes, elastic: &[usize]) -> Self {
        let n = state.particles;
        let points: Vec<Vec<DVec2>> = state
            .outlines
            .iter()
            .flatten()
            .map(|&o| elastic.iter().map(|&i| modes.at[i][o]).collect())
            .collect();
        let sigma = points
            .iter()
            .map(|phi: &Vec<DVec2>| phi.iter().map(|f| f.length_squared()).sum::<f64>().sqrt())
            .collect();
        Self {
            solids: state.solids[..n].to_vec(),
            volume: state.v0[..n].to_vec(),
            gradients: elastic
                .iter()
                .map(|&i| modes.gradients[i][..n].to_vec())
                .collect(),
            points,
            sigma,
        }
    }

    fn deformation(&self, q: &[f64], p: usize) -> DMat2 {
        q.iter()
            .zip(&self.gradients)
            .fold(DMat2::IDENTITY, |f, (&qi, g)| f + g[p] * qi)
    }

    /// `U_e(q)`, its round-off bound (Jeannerod and Rump 2013, Prop. 3.1 on
    /// each sum, `density_error` on each term) and, when `grad`, `dU_e/dq_i
    /// = sum_p V0_p (F_p S_p) : grad phi_i(X_p) - omega_i^2 q_i`.
    pub(super) fn at(&self, q: &[f64], omega: &[f64], grad: bool) -> Term {
        let unit = 0.5 * f64::EPSILON;
        let (mut stored, mut stored_size, mut error) = (0.0, 0.0, 0.0);
        let mut c = Vec::with_capacity(self.volume.len());
        let width = if grad { q.len() } else { 0 };
        let (mut grad_q, mut noise, mut sizes) =
            (vec![0.0; width], vec![0.0; width], vec![0.0; width]);
        for (p, (solid, &v0)) in self.solids.iter().zip(&self.volume).enumerate() {
            let f = self.deformation(q, p);
            let w = solid.energy(f);
            let (lambda, mu) = solid.lame();
            stored += v0 * w;
            stored_size += (v0 * w).abs();
            error += v0 * density_error(f, lambda, mu, w) + unit * (v0 * w).abs();
            let cp = f.transpose() * f;
            if grad {
                let s = solid.second_pk(cp);
                let piola = f * s;
                // `F`'s own round-off, an `n`-term sum per entry (Jeannerod
                // and Rump 2013, Prop. 3.1 with the identity's term).
                let size_f: f64 = q
                    .iter()
                    .zip(&self.gradients)
                    .map(|(qi, g)| (qi * colon(g[p], g[p]).sqrt()).abs())
                    .sum();
                let e_f = (q.len() as f64 + 1.0) * unit * (std::f64::consts::SQRT_2 + size_f);
                let norm_f = colon(f, f).sqrt();
                // `C = F^T F`: `F`'s error, and its own 2-term products.
                let e_c = 2.0 * norm_f * e_f + e_f * e_f + 2.0 * unit * norm_f * norm_f;
                let e_s = stress_error(cp, s, (lambda, mu), e_c);
                let norm_s = colon(s, s).sqrt();
                let e_piola = norm_f * e_s + norm_s * e_f + 2.0 * unit * norm_f * norm_s;
                let norm_piola = colon(piola, piola).sqrt();
                for (k, g) in self.gradients.iter().enumerate() {
                    let term = v0 * colon(piola, g[p]);
                    grad_q[k] += term;
                    sizes[k] += term.abs();
                    let norm_g = colon(g[p], g[p]).sqrt();
                    noise[k] += v0 * (e_piola + 5.0 * unit * norm_piola) * norm_g;
                }
            }
            c.push(cp);
        }
        let (mut quadratic, mut quadratic_size) = (0.0, 0.0);
        for (i, (&qi, &w)) in q.iter().zip(omega).enumerate() {
            let e = 0.5 * w * w * qi * qi;
            quadratic += e;
            quadratic_size += e.abs();
            if grad {
                grad_q[i] -= w * w * qi;
                noise[i] +=
                    self.volume.len() as f64 * unit * sizes[i] + 3.0 * unit * (w * w * qi).abs();
            }
        }
        let terms = (self.volume.len() + q.len()) as f64;
        Term {
            u: stored - quadratic,
            error: error
                + terms * unit * (stored_size + quadratic_size)
                + 4.0 * unit * quadratic_size,
            grad_q,
            noise,
            c,
        }
    }

    /// `U_e(q1) - U_e(q0)` from the law's exact increments
    /// (`Elastic::energy_increment`, its round-off relative to the change)
    /// and `omega^2 (q1 - q0) (q1 + q0) / 2`, with the sums' round-off
    /// bound (the increments' own is not bounded here: declared).
    pub(super) fn change(
        &self,
        (t0, t1): (&Term, &Term),
        (q0, q1): (&[f64], &[f64]),
        omega: &[f64],
    ) -> (f64, f64) {
        let unit = 0.5 * f64::EPSILON;
        let (mut change, mut size) = (0.0, 0.0);
        for ((solid, &v0), (c0, c1)) in self
            .solids
            .iter()
            .zip(&self.volume)
            .zip(t0.c.iter().zip(&t1.c))
        {
            let e = v0 * solid.energy_increment(*c0, *c1);
            change += e;
            size += e.abs();
        }
        for ((&a, &b), &w) in q0.iter().zip(q1).zip(omega) {
            let e = 0.5 * w * w * (b - a) * (b + a);
            change -= e;
            size += e.abs();
        }
        let terms = (self.volume.len() + q0.len()) as f64;
        (change, (terms + 4.0) * unit * size)
    }
}

// KNOWN-LIMIT[DISCRETIZATION/NO-SOURCE]: the flight's fixed point is stopped at its noise floor with this first-order bound of the law gradient's round-off; its contraction is observed, not proven (the trial's `M2` does not hold the law's curvature), and the scheme's own `Delta H`, within its bound `B_H`, exceeds the energy evaluation's round-off bound on 104 of 390 flown attempts of the long clay run (largest 15 times, 3.7e-12 J/m) | source: FAST_MODE_PLAN.md sec. 26 addendum 2, 26a | replace when: a derived bound of the law's Hessian over the ball enters `M2` and the stop reaches the energy's own round-off | state: measured | gap: none
/// A first-order bound on the computed corotated `S(C)`'s round-off
/// (`corotated_second_pk`: `S = 2 mu (I - U^-1) + lambda (1 - 1/J) adj(C)`,
/// `U^-1 = ((tr C + J) I - C) / (t J)`), plus the change an error `e_c` in
/// `C` makes. Derived on the page: each entry of `S` passes through at most
/// 16 rounded operations (`det`, two square roots, the sums, `t J`, its
/// reciprocal, the products, `I - U^-1`, `1 - 1/J`, the last sum), each
/// within `u` of the size of the term it forms, the terms at most `2 mu
/// (|I|_F + |U^-1|_F)` and `lambda (1 + 1/J) |adj C|_F`; the error in `C`
/// enters through the law's own tangent (`corotated_second_pk_tangent`),
/// whose operator norm is at most its Frobenius norm over the three unit
/// symmetric directions. Corotated bodies only (the flight's scope).
fn stress_error(c: DMat2, s: DMat2, (lambda, mu): (f64, f64), e_c: f64) -> f64 {
    use crate::matter::materials::solid::corotated::stress::corotated_second_pk_tangent;
    let unit = 0.5 * f64::EPSILON;
    let j = c.determinant().max(0.0).sqrt();
    if j <= f64::from(crate::matter::kinematics::volume_ratio::MIN_J) {
        return 0.0;
    }
    let tr = c.x_axis.x + c.y_axis.y;
    let t = (tr + 2.0 * j).sqrt();
    let u_inv = (DMat2::IDENTITY * (tr + j) - c) * (t * j).recip();
    let adj = DMat2::IDENTITY * tr - c;
    let sizes = 2.0 * mu * (std::f64::consts::SQRT_2 + colon(u_inv, u_inv).sqrt())
        + lambda * (1.0 + j.recip()) * colon(adj, adj).sqrt()
        + colon(s, s).sqrt();
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let tangent: f64 = [
        DMat2::from_cols(DVec2::X, DVec2::ZERO),
        DMat2::from_cols(DVec2::ZERO, DVec2::Y),
        DMat2::from_cols(DVec2::new(0.0, half), DVec2::new(half, 0.0)),
    ]
    .iter()
    .map(|&d| {
        let ds = corotated_second_pk_tangent(c, d, lambda, mu);
        colon(ds, ds)
    })
    .sum::<f64>()
    .sqrt();
    16.0 * unit * sizes + tangent * e_c
}

/// A bound on the computed corotated density's round-off at `F`, derived on
/// the page from `corotated_elastic_energy_density` (`Psi = mu |F - R|^2 +
/// lambda/2 (J - 1)^2`, `R` from `polar_decomposition_2d`): each entry of
/// the computed `R` within `5 u` (two sums, a square root, a division), so
/// `|R^ - R|_F <= 8 u`; `F - R` adds `u |F - R|`; `|F - R|^2` a 4-term
/// inner product (`4 u`, Jeannerod and Rump 2013, Thm. 4.2); `J` two
/// products and a difference (`3 u` of `|F00 F11| + |F01 F10|`); the last
/// operations `u |Psi|` each. Corotated bodies only (the flight's scope).
fn density_error(f: DMat2, lambda: f64, mu: f64, psi: f64) -> f64 {
    let unit = 0.5 * f64::EPSILON;
    let dev = f - polar_decomposition_2d(f);
    let d = colon(dev, dev).sqrt();
    let e_dev = 8.0 * unit + unit * d;
    let shear = mu * (2.0 * d * e_dev + e_dev * e_dev + 4.0 * unit * d * d);
    let j = f.determinant();
    let e_j = 3.0 * unit * ((f.x_axis.x * f.y_axis.y).abs() + (f.y_axis.x * f.x_axis.y).abs())
        + unit * (j - 1.0).abs();
    let volume = 0.5 * lambda * (2.0 * (j - 1.0).abs() * e_j + e_j * e_j)
        + 2.0 * unit * 0.5 * lambda * (j - 1.0) * (j - 1.0);
    shear + volume + 3.0 * unit * psi.abs()
}

/// The step's energy (`frame::particle_energy`, the one definition) and the
/// round-off bound of its evaluation, J/m: per particle the kinetic term
/// within `4 u` of itself, the gravity term within `4 u m (|g_x x| + |g_y
/// y|)`, the stored one within `V0 density_error + u |V0 W|`, their sum's
/// `2 u` of the three; the sum over particles `(N - 1) u sum |e_p|`
/// (Jeannerod and Rump 2013, Prop. 3.1).
pub(super) fn step_energy(state: &TotalLagrangianState, gravity: DVec2) -> (f64, f64) {
    let unit = 0.5 * f64::EPSILON;
    let n = state.particles;
    let (mut energy, mut size, mut error) = (0.0, 0.0, 0.0);
    for p in 0..n {
        let (kinetic, stored, potential) = super::super::frame::particle_energy(state, gravity, p);
        let e = kinetic + stored - potential;
        energy += e;
        size += e.abs();
        let (lambda, mu) = state.solids[p].lame();
        let w = state.solids[p].energy(state.f[p]);
        let x = state.x[p];
        error += 4.0 * unit * kinetic.abs()
            + 4.0 * unit * state.mass[p] * ((gravity.x * x.x).abs() + (gravity.y * x.y).abs())
            + state.v0[p] * density_error(state.f[p], lambda, mu, w)
            + unit * stored.abs()
            + 2.0 * unit * (kinetic.abs() + stored.abs() + potential.abs());
    }
    (energy, error + n.saturating_sub(1) as f64 * unit * size)
}

impl Exponential {
    // KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: this bound moves each contact point by a speed bound times the interval, about `omega_1 dt` (some 60) times looser over a frame than the quadratic flight's amplitude bound, so most frames are cut early and the rest is taken at the fixed contact step in the air (long clay run: 0 frames flown whole and 298 stepped without contact after frame 100; row P3: 7 of 30 frames flown whole against 25) | source: FAST_MODE_PLAN.md sec. 26a | replace when: an amplitude bound from the law's energy (a derived coercivity of the corotated energy against the modes' quadratic energy) | state: measured, not adopted | gap: the long run's higher hops not isolated between this and the fixed step
    /// The clearance bound with the law's energy (sec. 26): over `[t0, t1]`,
    /// `t0` the flight's own time, `T <= H` (`W >= 0`), so `|p| <= sqrt(2
    /// H)`, `|Omega| <= sqrt(2 H / s)`, `|q'| <= V(Q) = sqrt(2 H) (1 + K(Q)
    /// / sqrt(s_min(Q)))` while `|q| < Q`; a `Q` with `|q(t0)| + (t1 - t0)
    /// V(Q) < Q` bounds `|q|` over the whole interval, and each contact
    /// point's offset from the centre moves by at most `(t1 - t0) [Omega_max
    /// (|X_o - C| + sigma_o Q) + sigma_o V(Q)]` from its position at `t0`.
    pub(super) fn clear_with_law(
        &self,
        law: &Law,
        gravity: DVec2,
        planes: &[Plane],
        (t0, t1): (f64, f64),
    ) -> bool {
        let inv = &self.inv;
        let h = self.h_int.max(0.0) * (1.0 + gamma(4 * inv.n()));
        let dt = t1 - t0;
        let q: Vec<f64> = self
            .zeta_at(t0)
            .iter()
            .zip(&inv.omega)
            .map(|(z, w)| z.x / w)
            .collect();
        let q0 = q.iter().map(|e| e * e).sum::<f64>().sqrt();
        let l = inv.l.iter().map(|e| e * e).sum::<f64>().sqrt();
        let speed = |big_q: f64| -> Option<(f64, f64)> {
            let (_, s_min) = inv.m2(big_q * inv.omega_1, self.angular)?;
            let spin = (2.0 * h / s_min).sqrt();
            Some(((2.0 * h).sqrt() + (l + inv.g_norm * big_q) * spin, spin))
        };
        // The monotone iteration to its floating-point fixed point, then a
        // candidate above it, checked strictly.
        let mut big_q = q0;
        loop {
            let Some((v, _)) = speed(big_q) else {
                return false;
            };
            let next = q0 + dt * v;
            if !next.is_finite() {
                return false;
            }
            if next <= big_q {
                break;
            }
            big_q = next;
        }
        let candidate = 2.0 * big_q - q0;
        let Some((v, spin)) = speed(candidate) else {
            return false;
        };
        if !(v.is_finite() && spin.is_finite()) || (dt * v > 0.0 && q0 + dt * v >= candidate) {
            return false;
        }
        let turned = DMat2::from_angle(self.angle_at(t0));
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
            self.arms.iter().zip(&law.points).zip(&law.sigma).all(
                |((&(reference, _, _), phi), &sigma)| {
                    let u = phi
                        .iter()
                        .zip(&q)
                        .fold(DVec2::ZERO, |u, (f, &qi)| u + *f * qi);
                    let arm = turned * (reference + u);
                    let movement = dt
                        * (spin * (reference.length() + sigma * candidate) + sigma * v)
                        * (1.0 + gamma(8));
                    lowest + plane.inward * arm[plane.axis] - plane.inward * plane.at - movement
                        > 0.0
                },
            )
        })
    }
}
