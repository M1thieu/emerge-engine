//! FAST_MODE_PLAN.md sec. 48 (`theta_step`), a test-only measurement: each
//! Maxwell branch `eta_b dE/dt = S_P(C) - mu_b E` advanced by the theta rule,
//! `eta_b (E_n+1 - E_n) / dt = S_P(C_theta) - mu_b E_theta`, `E_theta = E_n +
//! theta (E_n+1 - E_n)`, so `E_n+1 - E_n = (S_P(C_theta) - mu_b E_n) / (eta_b
//! / dt + theta mu_b)` and the conjugate at `theta` is `Q_theta = (1 - r)
//! (S_P(C_theta) - mu_b E_n)`, `r = theta mu_b / (eta_b / dt + theta mu_b)`;
//! the branch's stress `S_P'(C_theta)[Q_theta] / mu_b` (eq. 2.8 at one
//! point). The heat booked is `eta_b |dE|^2 / dt` (the dissipation rate at the
//! step's mean rate, over the step); the step's whole energy residual is
//! booked as the scheme's heat beside it, so the ledger's total does not
//! depend on this split. An adaptation, declared (`theta_step`).

use super::*;

impl Relaxation {
    /// The branches' stress at `C_theta` by the theta rule.
    pub(crate) fn theta_stress(
        &self,
        c_theta: DMat2,
        dt: f64,
        theta: f64,
        moduli: (f64, f64),
    ) -> DMat2 {
        self.each(moduli)
            .map(|(b, (l, m))| {
                let Branch {
                    modulus: mu_b,
                    viscosity,
                } = b.branch;
                let keep = 1.0 - theta * mu_b / (viscosity / dt + theta * mu_b);
                let q = (corotated_second_pk(c_theta, l, m) - b.strain * mu_b) * keep;
                corotated_second_pk_tangent(c_theta, q, l, m) * mu_b.recip()
            })
            .fold(DMat2::ZERO, |sum, s| sum + s)
    }

    /// The branches after a kept theta step, and what it did.
    pub(crate) fn theta_committed(
        &self,
        (c_theta, c1): (DMat2, DMat2),
        dt: f64,
        theta: f64,
        (lambda, mu): (f64, f64),
    ) -> (Self, Relaxed) {
        let mut next = *self;
        let mut heat = 0.0;
        for pair in next.branches[..self.count].iter_mut() {
            for (b, (l, m)) in pair
                .iter_mut()
                .zip([(0.0, mu), (lambda, 0.0)])
                .filter_map(|(b, moduli)| b.as_mut().map(|b| (b, moduli)))
            {
                let Branch {
                    modulus: mu_b,
                    viscosity,
                } = b.branch;
                let change = (corotated_second_pk(c_theta, l, m) - b.strain * mu_b)
                    * (viscosity / dt + theta * mu_b).recip();
                heat += viscosity * colon(change, change) / dt;
                b.strain += change;
            }
        }
        let green = (c1 - DMat2::IDENTITY) * 0.5;
        let strain = colon(green, green).sqrt();
        (
            next,
            Relaxed {
                heat,
                strain,
                beyond: strain > self.strain_limit,
            },
        )
    }
}
