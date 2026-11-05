//! The turning flight with the law's own energy,
//! selected by `EMERGE_PROBE_FLIGHT=law`: the flight's Hamiltonian is the
//! floating frame's kinetic energy plus the step's own stored energy,
//! `sum_p V0_p W(C_p(q))`, `C_p = (I + G_p)^T (I + G_p)`, `G_p = sum_i q_i
//! grad phi_i(X_p)`, read through `Elastic` (the law's one definition). In
//! Li and Wu's split (arXiv 2012.13064 eq. 3) it is the part `U_e = sum_p
//! V0_p W(C_p) - 1/2 sum omega_i^2 q_i^2` of `U`. The basis is complete
//! (`body_modes`: as many modes as the step's unknowns), so `H` is the
//! step's energy of the rebuilt state; the clearance bound is rederived for
//! it (`W >= 0`).
//!
//! Declared, not exact: the kinetic form takes the modes' mass Gram matrix
//! as the identity (`|P - I|_F` at most 6.1e-14 measured), and one scheme
//! step a frame leaves the trajectory pre-asymptotic in the coupled fast
//! terms (LI_WU_TRIAL_RESULT.md 2.1); the energy is kept either way.

use super::*;
use crate::matter::kinematics::polar_decomposition::polar_decomposition_2d;
use crate::spacetime::total_lagrangian::Elastic;

#[path = "exponential_flight_lemma.rs"]
pub(super) mod lemma;

// KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: the kinetic form takes the modes' mass Gram matrix as the identity (`|P - I|_F` at most 6.1e-14 measured); its share of each hand-back's energy change is within the "left" part of its hand-back count (at most 9.5e-14 J/m) | source: measured on this path | replace when: the kinetic form carries `P` | state: measured | gap: none
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
    /// For the amplitude bound: per particle
    /// `g_p = (sum_i |grad phi_i(X_p)|^2 / omega_i^2)^(1/2)`; `e_K = |Omega^-1
    /// A Omega^-1 - I|_F` and `beta = |Omega^-1 B Omega^-1|_F` with their
    /// computed entries' round-off added; the largest `(lambda, mu)`.
    reach: Vec<f64>,
    e_k: f64,
    beta: f64,
    moduli: (f64, f64),
    /// The sharper lemma's constants, when it is measured.
    lemma: Option<lemma::Lemma>,
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
    pub(super) fn new(
        state: &TotalLagrangianState,
        modes: &Modes,
        (elastic, omega): (&[usize], &[f64]),
    ) -> Self {
        let n = state.particles;
        let unit = 0.5 * f64::EPSILON;
        // Sec. 27.2 (ii): `A_ij = sum_p V0 (2 mu sym g_i : sym g_j + lambda tr
        // g_i tr g_j)` and `B_ij = sum_p V0 g_i : g_j` in energy coordinates,
        // each entry's sum within `(n + 4) u` of its terms' magnitudes
        // (Jeannerod and Rump 2013, Prop. 3.1, with the products).
        let m = elastic.len();
        let trace = |g: DMat2| g.x_axis.x + g.y_axis.y;
        let (mut e_k, mut beta) = (0.0, 0.0);
        for a in 0..m {
            for b in 0..m {
                let (ga, gb) = (&modes.gradients[elastic[a]], &modes.gradients[elastic[b]]);
                let (mut sa, mut size_a, mut sb, mut size_b) = (0.0, 0.0, 0.0, 0.0);
                for p in 0..n {
                    let (lambda, mu) = state.solids[p].lame();
                    let (x, y) = (ga[p], gb[p]);
                    let (sx, sy) = ((x + x.transpose()) * 0.5, (y + y.transpose()) * 0.5);
                    let ta =
                        state.v0[p] * (2.0 * mu * colon(sx, sy) + lambda * trace(x) * trace(y));
                    let tb = state.v0[p] * colon(x, y);
                    sa += ta;
                    size_a += ta.abs();
                    sb += tb;
                    size_b += tb.abs();
                }
                let scale = omega[a] * omega[b];
                let identity = if a == b { 1.0 } else { 0.0 };
                let round = (n as f64 + 4.0) * unit;
                let ea = (sa / scale - identity).abs() + round * size_a / scale;
                let eb = sb.abs() / scale + round * size_b / scale;
                e_k += ea * ea;
                beta += eb * eb;
            }
        }
        let reach = (0..n)
            .map(|p| {
                elastic
                    .iter()
                    .zip(omega)
                    .map(|(&i, w)| {
                        let g = modes.gradients[i][p];
                        colon(g, g) / (w * w)
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        let moduli = state.solids[..n]
            .iter()
            .map(|s| s.lame())
            .fold((0.0f64, 0.0f64), |(l, u), (a, b)| (l.max(a), u.max(b)));
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
            reach,
            e_k: e_k.sqrt() * (1.0 + 4.0 * unit),
            beta: beta.sqrt() * (1.0 + 4.0 * unit),
            moduli,
            lemma: lemma::on().then(|| lemma::Lemma::new(state, modes, (elastic, omega))),
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

// KNOWN-LIMIT[DISCRETIZATION/NO-SOURCE]: the flight's fixed point is stopped at its noise floor with this first-order bound of the law gradient's round-off; its contraction is observed, not proven (the trial's `M2` does not hold the law's curvature), and the scheme's own `Delta H`, within its bound `B_H`, exceeds the energy evaluation's round-off bound on 104 of 390 flown attempts of the long clay run (largest 15 times, 3.7e-12 J/m) | source: measured on this path | replace when: a derived bound of the law's Hessian over the ball enters `M2` and the stop reaches the energy's own round-off | state: measured | gap: none
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
    // KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: this bound moves each contact point by a speed bound times the interval, about `omega_1 dt` (some 60) times looser over a frame than the quadratic flight's amplitude bound, so most frames are cut early and the rest is stepped (measured with the former fixed contact step, now removed: long clay run: 0 frames flown whole and 298 stepped without contact after frame 100; a 30-frame run: 7 of 30 frames flown whole against 25) | source: measured on this path | replace when: an amplitude bound from the law's energy (a derived coercivity of the corotated energy against the modes' quadratic energy) | state: measured, not adopted | gap: the long run's higher hops are the fixed step's unresolved vibration in the air (the fixed step only while a contact is held brings the hop to 0.97)
    /// The clearance bound with the law's energy: over `[t0, t1]`,
    /// `t0` the flight's own time, `T <= H` (`W >= 0`), so `|p| <= sqrt(2
    /// H)`, `|Omega| <= sqrt(2 H / s)`, `|q'| <= V(Q) = sqrt(2 H) (1 + K(Q)
    /// / sqrt(s_min(Q)))` while `|q| < Q`; a `Q` with `|q(t0)| + (t1 - t0)
    /// V(Q) < Q` bounds `|q|` over the whole interval, and each contact
    /// point's offset from the centre moves by at most `(t1 - t0) [Omega_max
    /// (|X_o - C| + sigma_o Q) + sigma_o V(Q)]` from its position at `t0`.
    fn clear_by_speed(
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

    /// The lower bound of the centre's position over `[t0, t1]` along
    /// `plane`'s inward normal (the parabola's ends and vertex).
    fn lowest_centre(&self, gravity: DVec2, plane: &Plane, (t0, t1): (f64, f64)) -> f64 {
        let along = |t: f64| plane.inward * self.rigid_at(gravity, t).0[plane.axis];
        let mut lowest = along(t0).min(along(t1));
        let acceleration = plane.inward * gravity[plane.axis];
        if acceleration != 0.0 {
            let vertex = self.t_start - plane.inward * self.velocity[plane.axis] / acceleration;
            if vertex > t0 && vertex < t1 {
                lowest = lowest.min(along(vertex));
            }
        }
        lowest
    }

    // KNOWN-LIMIT[DISCRETIZATION/TECH-LIMIT]: the bound is proven but admissible only before the first impact on the hopping clay block (56 of 18052 clearance evaluations): its step (iii) fails, the energy's worst-case strain bound by Cauchy-Schwarz (at least 6.4e-2 at c = 1) exceeding the coercivity's range (r < 4.4e-2, set by the gradient Gram bound beta = 1.47e-6 1/Pa); the speed bound then decides, so frames are still cut early | source: measured on this path | replace when: a sharper coercivity (the cubic term bounded per particle, not through beta) or a sharper strain bound than Cauchy-Schwarz over the whole energy | state: measured, not loosened | gap: none
    /// The amplitude bound from the law's energy: `W >= c(r) 1/2 sum omega_i^2 q_i^2` while every `|G_p| <= r`,
    /// `c(r) = 1 - e_K - 2 r k(r) beta`, `k(r) = sqrt(2) mu / (2 - sqrt(2) r) +
    /// lambda / sqrt(2)`; `W <= H` then bounds `sum omega_i^2 q_i^2` by `2 H /
    /// c(r)`, each `|G_p|` by `g_p` times its root, and each contact point's
    /// vibration by `A_o = s_o (2 H / c(r))^(1/2)` over the whole flight, once
    /// `r` holds itself (`max_p g_p (2 H / c(r))^(1/2) < r`) and holds at
    /// `t0`. `None` when no admissible `r` exists; else whether the frame is
    /// clear, with `(c, r)`.
    fn clear_by_amplitude(
        &self,
        law: &Law,
        gravity: DVec2,
        planes: &[Plane],
        (t0, t1): (f64, f64),
    ) -> Option<(bool, f64, f64)> {
        let inv = &self.inv;
        let h = self.h_int.max(0.0) * (1.0 + gamma(4 * inv.n()));
        let q: Vec<f64> = self
            .zeta_at(t0)
            .iter()
            .zip(&inv.omega)
            .map(|(z, w)| z.x / w)
            .collect();
        let start = (0..law.volume.len())
            .map(|p| {
                let g = law.deformation(&q, p) - DMat2::IDENTITY;
                colon(g, g).sqrt()
            })
            .fold(0.0, f64::max)
            * (1.0 + gamma(q.len() + 4));
        let (lambda, mu) = law.moduli;
        let (root2, min_j) = (
            std::f64::consts::SQRT_2,
            f64::from(crate::matter::kinematics::volume_ratio::MIN_J),
        );
        let reach = law.reach.iter().copied().fold(0.0, f64::max);
        // `c(r)` where `r` is admissible, and the
        // bound `max_p g_p (2 H / c(r))^(1/2)` on `|G|` it implies.
        let implied = |r: f64| -> Option<(f64, f64)> {
            if r >= root2 || 1.0 - root2 * r - 0.5 * r * r <= min_j {
                return None;
            }
            let k = root2 * mu / (2.0 - root2 * r) + lambda / root2;
            let c = (1.0 - law.e_k - 2.0 * r * k * law.beta) * (1.0 - gamma(8));
            (c > 0.0).then(|| (c, reach * (2.0 * h / c).sqrt() * (1.0 + gamma(4))))
        };
        // Why a bound is not admissible, for the probes: the
        // start's own strain outside the law's range, or the energy's strain
        // bound at `c = 1`, `max_p g_p (2 H)^(1/2)`, against the largest `r`
        // with `c(r) > 0`.
        let refuse = |at_start: bool| {
            PROOFS.with(|cell| {
                let mut p = cell.get();
                if at_start {
                    p.start_outside += 1;
                } else {
                    p.iteration_outside += 1;
                }
                p.energy_strain = p.energy_strain.min(reach * (2.0 * h).sqrt());
                cell.set(p);
            });
        };
        if implied(start).is_none() {
            refuse(true);
            return None;
        }
        let mut r = start;
        loop {
            let Some((_, bound)) = implied(r) else {
                refuse(false);
                return None;
            };
            let next = start.max(bound);
            if !next.is_finite() {
                return None;
            }
            if next <= r {
                break;
            }
            r = next;
        }
        let candidate = 2.0 * r - start;
        let Some((c, bound)) = implied(candidate) else {
            refuse(false);
            return None;
        };
        if !(bound < candidate && start < candidate) {
            refuse(false);
            return None;
        }
        let clear = self.clear_within((2.0 * h / c).sqrt(), (gravity, planes), (t0, t1))?;
        Some((clear, c, candidate))
    }

    /// Whether `[t0, t1]` is clear when the vibration's energy coordinates
    /// stay within `amplitude` (`|xi| <= amplitude`): each contact point's
    /// vibration within `s_o amplitude`, its turning within `Omega_max dt
    /// (|X_o - C| + s_o amplitude)`. `None` without a spin bound there.
    fn clear_within(
        &self,
        amplitude: f64,
        (gravity, planes): (DVec2, &[Plane]),
        (t0, t1): (f64, f64),
    ) -> Option<bool> {
        let h = self.h_int.max(0.0) * (1.0 + gamma(4 * self.inv.n()));
        let (_, s_min) = self.inv.m2(amplitude, self.angular)?;
        let spin = (2.0 * h / s_min).sqrt();
        let dt = t1 - t0;
        let turned = DMat2::from_angle(self.angle_at(t0));
        let clear = planes.iter().all(|plane| {
            let lowest = self.lowest_centre(gravity, plane, (t0, t1));
            self.arms.iter().all(|&(reference, _, schwarz)| {
                let arm = turned * reference;
                let vibration = schwarz * amplitude;
                let turning = spin * dt * (reference.length() + vibration);
                lowest + plane.inward * arm[plane.axis]
                    - plane.inward * plane.at
                    - (vibration + turning) * (1.0 + gamma(8))
                    > 0.0
            })
        });
        Some(clear)
    }

    /// Whether `[t0, t1]` is clear by either of the law's two bounds (each a
    /// proof), counted for the probes (`proof_report`).
    pub(super) fn clear_with_law(
        &self,
        law: &Law,
        gravity: DVec2,
        planes: &[Plane],
        span: (f64, f64),
    ) -> bool {
        let amplitude = self.clear_by_amplitude(law, gravity, planes, span);
        let by_amplitude = amplitude.is_some_and(|a| a.0);
        let by_speed = !by_amplitude && self.clear_by_speed(law, gravity, planes, span);
        if let Some(lemma) = &law.lemma {
            self.measure_lemma(
                (law, lemma),
                (gravity, planes),
                span,
                by_amplitude || by_speed,
            );
        }
        PROOFS.with(|cell| {
            let mut p = cell.get();
            match amplitude {
                None => p.inadmissible += 1,
                Some((_, c, r)) => {
                    p.admissible += 1;
                    p.c = (p.c.0.min(c), p.c.1.max(c));
                    p.r = (p.r.0.min(r), p.r.1.max(r));
                }
            }
            p.by_amplitude += u64::from(by_amplitude);
            p.by_speed += u64::from(by_speed);
            p.e_k = p.e_k.max(law.e_k);
            p.beta = p.beta.max(law.beta);
            // The largest `r` with `c(r) > 0` to first order, `(1 - e_K) / (2
            // k(0) beta)`.
            let (lambda, mu) = law.moduli;
            let k0 = std::f64::consts::SQRT_2 * mu / 2.0 + lambda / std::f64::consts::SQRT_2;
            p.range = p.range.max((1.0 - law.e_k) / (2.0 * k0 * law.beta));
            p.calls += 1;
            cell.set(p);
        });
        by_amplitude || by_speed
    }
}

/// The clearance proofs' record, over every evaluation of
/// `clear_with_law` on this thread.
#[derive(Clone, Copy, Debug)]
struct Proofs {
    calls: u64,
    admissible: u64,
    inadmissible: u64,
    by_amplitude: u64,
    by_speed: u64,
    c: (f64, f64),
    r: (f64, f64),
    e_k: f64,
    beta: f64,
    start_outside: u64,
    iteration_outside: u64,
    energy_strain: f64,
    range: f64,
}

thread_local! {
    static PROOFS: std::cell::Cell<Proofs> = const {
        std::cell::Cell::new(Proofs {
            calls: 0,
            admissible: 0,
            inadmissible: 0,
            by_amplitude: 0,
            by_speed: 0,
            c: (f64::INFINITY, 0.0),
            r: (f64::INFINITY, 0.0),
            e_k: 0.0,
            beta: 0.0,
            start_outside: 0,
            iteration_outside: 0,
            energy_strain: f64::INFINITY,
            range: 0.0,
        })
    };
}

/// One line: the clearance proofs counted on this thread.
pub(in super::super) fn proof_report() -> String {
    let p = PROOFS.with(|p| p.get());
    format!(
        "clearance evaluations {}: the amplitude bound admissible {} (not {}: the start outside the range {}, the iteration leaving it {}), c from {:.6} to {:.6}, r from {:.3e} to {:.3e}; clear by the amplitude bound {}, by the speed bound only {}; e_K largest {:.3e}, beta largest {:.3e} 1/Pa; the range r < {:.4e} (first order), the energy's strain bound at c = 1 at least {:.4e}",
        p.calls,
        p.admissible,
        p.inadmissible,
        p.start_outside,
        p.iteration_outside,
        p.c.0,
        p.c.1,
        p.r.0,
        p.r.1,
        p.by_amplitude,
        p.by_speed,
        p.e_k,
        p.beta,
        p.range,
        p.energy_strain
    )
}
