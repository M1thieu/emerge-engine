//! The elastic solid the step moves: its stored energy, its stresses, and the
//! discrete gradient that makes a step's stress carry exactly the energy's
//! change.

use super::*;

/// The solid: the shear modulus and the bulk modulus (`lambda + mu` in
/// plane strain), which stored energy they enter, and the Kelvin-Voigt
/// dashpot in parallel (shear and bulk viscosity; zero, none).
#[derive(Clone, Copy, Debug)]
pub struct Elastic {
    pub mu: f64,
    pub bulk: f64,
    pub law: ElasticLaw,
    pub shear_viscosity: f64,
    pub bulk_viscosity: f64,
}

/// The stored energy the step differentiates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElasticLaw {
    /// A research law, the mac gates' solid only: the isochoric part's
    /// corotated shear and `K / 2 (J - 1)^2`. No material of `matter` has
    /// it, and a constitutive law does not belong in `spacetime`: kept for
    /// those gates' measured history, to be retired or moved to `matter`
    /// when the folders are brought in line. The simulation path never
    /// uses it.
    Isochoric,
    /// The corotated material's own law, `matter`'s
    /// `corotated_elastic_energy_density_f64` and its stress, with
    /// `lambda = bulk - mu` (Stomakhin et al. 2013).
    Corotated,
    /// J2 (von Mises) plasticity on Hencky elasticity, `matter`'s
    /// `HenckyJ2`, with the particle's own plastic state; its step's stress
    /// is Love and Sulsky's algorithmic stress
    /// (`algorithmic_stress_over`).
    HenckyJ2(HenckyJ2, PlasticMetric),
}

impl Elastic {
    /// The research law with these moduli.
    pub fn isochoric(mu: f64, bulk: f64) -> Self {
        Self {
            mu,
            bulk,
            law: ElasticLaw::Isochoric,
            shear_viscosity: 0.0,
            bulk_viscosity: 0.0,
        }
    }

    /// J2 plasticity on Hencky elasticity with the Lame moduli, the
    /// uniaxial yield stress, the linear hardening modulus and the
    /// viscoplastic viscosity (zero: rate independent), the material
    /// unstrained and unyielded.
    pub fn hencky_j2(
        lambda: f64,
        mu: f64,
        yield_stress: f64,
        hardening: f64,
        viscosity: f64,
    ) -> Self {
        Self {
            mu,
            bulk: lambda + mu,
            law: ElasticLaw::HenckyJ2(
                HenckyJ2 {
                    shear: mu,
                    bulk: lambda + 2.0 * mu / 3.0,
                    yield_stress,
                    hardening,
                    viscosity,
                },
                PlasticMetric::unstrained(),
            ),
            shear_viscosity: 0.0,
            bulk_viscosity: 0.0,
        }
    }

    /// The corotated material's law with its Lame moduli.
    pub fn corotated(lambda: f64, mu: f64) -> Self {
        Self {
            mu,
            bulk: lambda + mu,
            law: ElasticLaw::Corotated,
            shear_viscosity: 0.0,
            bulk_viscosity: 0.0,
        }
    }

    /// The same solid with a Kelvin-Voigt dashpot of shear viscosity `eta`
    /// and bulk viscosity `zeta` in parallel (`matter`'s
    /// `newtonian_viscous_stress`).
    pub fn with_viscosity(self, eta: f64, zeta: f64) -> Self {
        Self {
            shear_viscosity: eta,
            bulk_viscosity: zeta,
            ..self
        }
    }

    /// Whether the solid has a dashpot.
    pub(crate) fn is_viscous(&self) -> bool {
        self.shear_viscosity != 0.0 || self.bulk_viscosity != 0.0
    }

    /// The dashpot's second Piola-Kirchhoff stress over a step of length
    /// `dt` from `C_n` to `C_{n+1}`: the Newtonian law on the Green strain
    /// rate `E_dot = Z / dt`, `Z = (C_{n+1} - C_n) / 2` (Barbic eq. 2.34's
    /// Lagrangian form). With `F_{n+1/2} = (F_n + F_{n+1}) / 2`, `sym(F^T dF)
    /// = Z` exactly, so its work over the step is `Z : S_D = dt D(E_dot)`,
    /// the law's dissipation (`viscous_dissipation`), never negative.
    /// Adaptation, derived: the rate at the step's midpoint as the strain
    /// increment over the step.
    pub(crate) fn viscous_stress_over(&self, c0: DMat2, c1: DMat2, dt: f64) -> DMat2 {
        let rate = (c1 - c0) * (0.5 / dt);
        newtonian_viscous_stress(rate, self.shear_viscosity, self.bulk_viscosity)
    }

    /// The energy per reference volume the dashpot dissipates over that
    /// step, `dt D(E_dot)` (J/m^3 in SI).
    pub(crate) fn viscous_dissipation_over(&self, c0: DMat2, c1: DMat2, dt: f64) -> f64 {
        let rate = (c1 - c0) * (0.5 / dt);
        dt * newtonian_dissipation(rate, self.shear_viscosity, self.bulk_viscosity)
    }
}

/// `sqrt(C)` of a symmetric positive definite `C`.
pub(crate) fn sqrt_spd(c: DMat2) -> DMat2 {
    let s = c.determinant().max(0.0).sqrt();
    let t = (c.x_axis.x + c.y_axis.y + 2.0 * s)
        .max(f64::MIN_POSITIVE)
        .sqrt();
    (c + DMat2::from_diagonal(DVec2::splat(s))) * t.recip()
}

pub(crate) fn rotation(f: DMat2) -> DMat2 {
    crate::matter::kinematics::polar_decomposition::polar_decomposition_2d(f)
}

impl Elastic {
    /// Stored energy per unit reference volume (per unit density),
    /// `mu |F^ - R^|^2 + K / 2 (J - 1)^2`, as `elastic::shear_energy` and
    /// the gates' volume energy write it.
    pub fn energy(&self, f: DMat2) -> f64 {
        if let ElasticLaw::HenckyJ2(law, state) = &self.law {
            return law.energy_at(f.transpose() * f, state);
        }
        if self.law == ElasticLaw::Corotated {
            return crate::matter::materials::solid::corotated::stress::corotated_elastic_energy_density(
                f,
                self.bulk - self.mu,
                self.mu,
            );
        }
        let j = f.determinant();
        let fh = f * j.max(f64::MIN_POSITIVE).sqrt().recip();
        let d = fh - rotation(fh);
        self.mu * (d.x_axis.length_squared() + d.y_axis.length_squared())
            + 0.5 * self.bulk * (j - 1.0).powi(2)
    }

    /// Kirchhoff stress `tau = dpsi/dF F^T`, as `elastic::shear_kirchhoff`
    /// and the gates' `K J (J - 1) I` write it.
    pub(crate) fn kirchhoff(&self, f: DMat2) -> DMat2 {
        if let ElasticLaw::HenckyJ2(law, state) = &self.law {
            return f * law.second_pk_at(f.transpose() * f, state).0 * f.transpose();
        }
        if self.law == ElasticLaw::Corotated {
            return crate::matter::materials::solid::corotated::stress::corotated_elastic_stress(
                f,
                self.bulk - self.mu,
                self.mu,
            );
        }
        let j = f.determinant();
        let fh = f * j.max(f64::MIN_POSITIVE).sqrt().recip();
        let tau = 2.0 * self.mu * (fh - rotation(fh)) * fh.transpose();
        let mean = 0.5 * (tau.x_axis.x + tau.y_axis.y);
        tau - DMat2::from_diagonal(DVec2::splat(mean))
            + DMat2::from_diagonal(DVec2::splat(self.bulk * j * (j - 1.0)))
    }

    /// The first Piola-Kirchhoff stress `P = tau F^-T` the step's stress
    /// derives from (`kirchhoff`), for the probes' exact solutions.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn first_pk(&self, f: DMat2) -> DMat2 {
        self.kirchhoff(f) * f.inverse().transpose()
    }

    /// The algorithmic stress of Gonzalez, CMAME 190 (2000), eqs. 32-33,
    /// for the separable energy `W = W_dev(C) + U(J)` (his eqs. 54-55):
    /// `S = 2 dW_dev(C_n, C_n+1) + 2 p dG(C_n, C_n+1)` with `G = J - 1`, `p =
    /// (U(J_n+1) - U(J_n)) / (J_n+1 - J_n)` his `dW_C(Theta_n, Theta_n+1)`, both
    /// discrete derivatives his eq. 12 quotient form. Adapted: `Theta = J`
    /// at each point (his smooth solutions have `Theta = zeta`, p. 1772; no
    /// `Theta` field is projected), no incompressibility multiplier (`lambda
    /// = 0`, `U` the material's own bulk energy, not a penalty), and the
    /// bases at the midpoint strain or, with `trapezoid`, the two ends'
    /// average. Then `1/2 S : (C_n+1 - C_n) = W_dev(C_n+1) - W_dev(C_n) +
    /// U(J_n+1) - U(J_n)` exactly.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn split_stress(&self, c0: DMat2, c1: DMat2, trapezoid: bool) -> DMat2 {
        let jacobian = |c: DMat2| c.determinant().max(0.0).sqrt();
        let volume = |j: f64| 0.5 * self.bulk * (j - 1.0).powi(2);
        // S_vol = 2 U'(J) dJ/dC = U'(J) J C^-1; 2 dG/dC = J C^-1.
        let two_dg = |c: DMat2| jacobian(c) * c.inverse();
        let deviatoric_stress =
            |c: DMat2| self.second_pk(c) - self.bulk * (jacobian(c) - 1.0) * two_dg(c);
        let deviatoric_energy = |c: DMat2| self.energy_of_c(c) - volume(jacobian(c));
        let contract = |a: DMat2, b: DMat2| a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis);
        let mid = 0.5 * (c0 + c1);
        let dc = c1 - c0;
        let norm2 = dc.x_axis.length_squared() + dc.y_axis.length_squared();
        let base = |f: &dyn Fn(DMat2) -> DMat2| {
            if trapezoid {
                0.5 * (f(c0) + f(c1))
            } else {
                f(mid)
            }
        };
        let dev_base = base(&deviatoric_stress);
        let g_base = match hooks::volumetric_base() {
            1 => two_dg(mid),
            2 => 0.5 * (two_dg(c0) + two_dg(c1)),
            _ => base(&two_dg),
        };
        let (j0, j1) = (jacobian(c0), jacobian(c1));
        let pressure = if (j1 - j0).abs() > 1e-14 {
            (volume(j1) - volume(j0)) / (j1 - j0)
        } else {
            self.bulk * (0.5 * (j0 + j1) - 1.0)
        };
        if norm2 <= 1e-28 {
            return dev_base + pressure * g_base;
        }
        let dev = dev_base
            + (2.0
                * (deviatoric_energy(c1) - deviatoric_energy(c0) - 0.5 * contract(dev_base, dc))
                / norm2)
                * dc;
        let dg = g_base + (2.0 * (j1 - j0 - 0.5 * contract(g_base, dc)) / norm2) * dc;
        dev + pressure * dg
    }

    /// A Kelvin-Voigt viscous second Piola-Kirchhoff stress, the material's
    /// own small-strain elasticity times `beta` acting on the Green strain
    /// rate: `S_v = beta (lambda tr(E_dot) I + 2 mu E_dot)`, `E_dot = sym(F^T
    /// F_dot)`, so its dissipation `S_v : E_dot = beta (lambda tr(E_dot)^2 +
    /// 2 mu |E_dot|^2)` is never negative. Stiffness-proportional (Rayleigh
    /// `beta`) damping: a mode of frequency `w` gets the damping ratio `beta w
    /// / 2`. A probe's material term, test-only.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn viscous_stress(&self, f: DMat2, f_dot: DMat2, beta: f64) -> DMat2 {
        let rate = f.transpose() * f_dot;
        let e_dot = 0.5 * (rate + rate.transpose());
        let lambda = self.bulk - self.mu;
        let trace = e_dot.x_axis.x + e_dot.y_axis.y;
        beta * (lambda * trace * DMat2::IDENTITY + 2.0 * self.mu * e_dot)
    }

    /// The deviatoric part of the stored energy, `W(F) - U(J)` with `U(J) =
    /// K / 2 (J - 1)^2` (the probes' energy ledger with a `Theta` field).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn deviatoric_energy(&self, f: DMat2) -> f64 {
        self.energy(f) - 0.5 * self.bulk * (f.determinant() - 1.0).powi(2)
    }

    /// The stress of Gonzalez 2000 eq. 32 with `Theta` constant per cell
    /// (his sec. 4.2 with `m_dis = 1`, eq. 53: the cell's mean `J`) and
    /// `lambda = 0`: the deviatoric discrete derivative of `split_stress`
    /// plus `2 p dG` with the CELL's pressure `p` (the secant of `U` over
    /// the cell's `Theta_n`, `Theta_n+1`). Then `sum_p V0 1/2 S : dC = sum_p
    /// V0 dW_dev + sum_cells V p dTheta = dE` with the stored energy `E =
    /// sum_p V0 W_dev + sum_cells V U(Theta)` exactly.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn mixed_stress(
        &self,
        c0: DMat2,
        c1: DMat2,
        pressure: f64,
        trapezoid: bool,
    ) -> DMat2 {
        let jacobian = |c: DMat2| c.determinant().max(0.0).sqrt();
        let two_dg = |c: DMat2| jacobian(c) * c.inverse();
        let deviatoric_stress =
            |c: DMat2| self.second_pk(c) - self.bulk * (jacobian(c) - 1.0) * two_dg(c);
        let deviatoric_energy =
            |c: DMat2| self.energy_of_c(c) - 0.5 * self.bulk * (jacobian(c) - 1.0).powi(2);
        let contract = |a: DMat2, b: DMat2| a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis);
        let mid = 0.5 * (c0 + c1);
        let dc = c1 - c0;
        let norm2 = dc.x_axis.length_squared() + dc.y_axis.length_squared();
        let base = |f: &dyn Fn(DMat2) -> DMat2| {
            if trapezoid {
                0.5 * (f(c0) + f(c1))
            } else {
                f(mid)
            }
        };
        let dev_base = base(&deviatoric_stress);
        let g_base = base(&two_dg);
        if norm2 <= 1e-28 {
            return dev_base + pressure * g_base;
        }
        let dev = dev_base
            + (2.0
                * (deviatoric_energy(c1) - deviatoric_energy(c0) - 0.5 * contract(dev_base, dc))
                / norm2)
                * dc;
        let (j0, j1) = (jacobian(c0), jacobian(c1));
        let dg = g_base + (2.0 * (j1 - j0 - 0.5 * contract(g_base, dc)) / norm2) * dc;
        dev + pressure * dg
    }

    /// The split stress with its volumetric part in `P`: `F_mid S_dev + p
    /// cof(F_mid)`, `S_dev` the deviatoric discrete derivative of
    /// `mixed_stress`, `p` the secant of `U` between the ends' `J`. In 2D
    /// `J = det F` is quadratic in `F`, so `cof(F_mid) : (F_1 - F_0) = J_1 -
    /// J_0` exactly and the energy identity needs no correction; `P
    /// F_mid^T = F_mid S_dev F_mid^T + p J(F_mid) I` is symmetric, so angular
    /// momentum is kept too. A probe's diagnostic of the volumetric term.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn cofactor_volumetric(&self, f0: DMat2, f1: DMat2, trapezoid: bool) -> DMat2 {
        let mid = 0.5 * (f0 + f1);
        let deviatoric =
            self.mixed_stress(f0.transpose() * f0, f1.transpose() * f1, 0.0, trapezoid);
        let (j0, j1) = (f0.determinant(), f1.determinant());
        let pressure = self.bulk * (0.5 * (j0 + j1) - 1.0);
        let cofactor = DMat2::from_cols(
            DVec2::new(mid.y_axis.y, -mid.y_axis.x),
            DVec2::new(-mid.x_axis.y, mid.x_axis.x),
        );
        mid * deviatoric + pressure * cofactor
    }

    /// A discrete gradient of `W(F)` in `F` itself: the two ends' `P`
    /// averaged, then corrected along `F_1 - F_0` so that `P : dF = dW`
    /// exactly (Gonzalez 2000 eq. 12's quotient correction, in `F`). A
    /// probe's diagnostic: it keeps the energy but not angular momentum
    /// exactly (his eq. 16 needs `F_{n+1/2} S` with a symmetric `S`).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn trapezoid_first_pk(&self, f0: DMat2, f1: DMat2) -> DMat2 {
        let base = 0.5 * (self.first_pk(f0) + self.first_pk(f1));
        let df = f1 - f0;
        let norm2 = df.x_axis.length_squared() + df.y_axis.length_squared();
        if norm2 <= 1e-28 {
            return base;
        }
        let contracted = base.x_axis.dot(df.x_axis) + base.y_axis.dot(df.y_axis);
        let gap = self.energy(f1) - self.energy(f0) - contracted;
        base + (gap / norm2) * df
    }

    /// `W(C)` through `F = sqrt(C)`: the energy depends on `F` through `C`
    /// alone.
    pub(crate) fn energy_of_c(&self, c: DMat2) -> f64 {
        self.energy(sqrt_spd(c))
    }

    /// The stored energy's change from `C_n` to `C_{n+1}`, computed from the
    /// increment `dC` itself: the part of the law the discrete gradient's
    /// quotient divides by `|dC|^2`, so its round-off must be relative to the
    /// change, not to the energies (`algorithmic_stress`). Each law supplies
    /// its own exact form; a law without one must not use the quotient. The
    /// isochoric research law (the mac gates' only, to be retired) is the
    /// one exception: the energies' difference, declared.
    pub(crate) fn energy_increment(&self, c0: DMat2, c1: DMat2) -> f64 {
        match self.law {
            ElasticLaw::Corotated => {
                crate::matter::materials::solid::corotated::stress::corotated_energy_increment(
                    c0,
                    c1,
                    self.bulk - self.mu,
                    self.mu,
                )
            }
            ElasticLaw::Isochoric => self.energy_of_c(c1) - self.energy_of_c(c0),
            // Its own exact increment, the return's included (the step's
            // length enters only through the viscoplastic rate).
            ElasticLaw::HenckyJ2(law, state) => law.step(c0, c1, &state, 0.0).energy_change,
        }
    }

    /// The step's stress between `C_n` and `C_{n+1}` over a step of length
    /// `dt`: for the plastic law, Love and Sulsky's algorithmic stress
    /// (their eqs. 81-83), `S_bar + [2 (dW + dD) - S_bar : dC] / |dC|^2 dC`
    /// with `S_bar` the mean of the two ends' pulled-back stresses (eq. 82,
    /// the backward Euler flow's) and `dD` the step's dissipation (eq. 74),
    /// so `1/2 S : dC = dW + dD` exactly; `dW` from the increment
    /// (`HenckyJ2::step`). Adaptation, declared: their `S_bar` pulls back
    /// the deviatoric stress and the volume is a separate field (their
    /// three-field form); here the whole stress, the step displacement-only.
    /// Every other law: `algorithmic_stress`.
    pub(crate) fn algorithmic_stress_over(&self, c0: DMat2, c1: DMat2, dt: f64) -> DMat2 {
        let ElasticLaw::HenckyJ2(law, state) = &self.law else {
            return self.algorithmic_stress(c0, c1);
        };
        let stepped = law.step(c0, c1, state, dt);
        let mean = 0.5 * (law.second_pk_at(c0, state).0 + law.second_pk_at(c1, &stepped.state).0);
        let dc = c1 - c0;
        let norm2 = dc.x_axis.length_squared() + dc.y_axis.length_squared();
        if norm2 == 0.0 {
            return mean;
        }
        let contracted = mean.x_axis.dot(dc.x_axis) + mean.y_axis.dot(dc.y_axis);
        let gap = stepped.energy_change + stepped.dissipated - 0.5 * contracted;
        mean + (2.0 * gap / norm2) * dc
    }

    /// The solid after a kept step from `C_n` to `C_{n+1}` of length `dt`,
    /// and the energy the step dissipated per reference volume: the plastic
    /// law's state advanced by its return map; every other law unchanged,
    /// none.
    pub(crate) fn committed(&self, c0: DMat2, c1: DMat2, dt: f64) -> (Self, f64) {
        let ElasticLaw::HenckyJ2(law, state) = &self.law else {
            return (*self, 0.0);
        };
        let stepped = law.step(c0, c1, state, dt);
        let mut next = *self;
        next.law = ElasticLaw::HenckyJ2(*law, stepped.state);
        (next, stepped.dissipated)
    }

    /// Whether the law carries a state its steps advance.
    pub(crate) fn is_plastic(&self) -> bool {
        matches!(self.law, ElasticLaw::HenckyJ2(..))
    }

    /// The second Piola-Kirchhoff stress `S(C) = F^-1 tau F^-T` with `F =
    /// sqrt(C)`.
    pub(crate) fn second_pk(&self, c: DMat2) -> DMat2 {
        let f = sqrt_spd(c);
        let inv = f.inverse();
        inv * self.kirchhoff(f) * inv.transpose()
    }

    /// The discrete gradient stress of eq. 54 between `C_n` and `C_{n+1}`.
    /// With the probes' switch (`hooks::set_trapezoid`) the base stress the
    /// quotient correction completes is the average of the two ends'
    /// instead of the midpoint strain's: Gonzalez, CMAME 190 (2000), eq. 12
    /// with that base, an adaptation of its base only. His energy and
    /// angular momentum proofs (eqs. 16, 19-21) use only the correction and
    /// the stress's symmetry, which the average keeps, and his eq. 22 holds
    /// for it as the average differs from `DW(C_{n+1/2})` by `O(|M|^2)`.
    pub(crate) fn algorithmic_stress(&self, c0: DMat2, c1: DMat2) -> DMat2 {
        #[cfg(all(test, feature = "experimental"))]
        if hooks::split() {
            return self.split_stress(c0, c1, hooks::trapezoid());
        }
        let stress = self.second_pk(0.5 * (c0 + c1));
        #[cfg(all(test, feature = "experimental"))]
        let stress = if hooks::trapezoid() {
            0.5 * (self.second_pk(c0) + self.second_pk(c1))
        } else {
            stress
        };
        let dc = c1 - c0;
        let norm2 = dc.x_axis.length_squared() + dc.y_axis.length_squared();
        if norm2 <= 1e-28 {
            return stress;
        }
        let contracted = stress.x_axis.dot(dc.x_axis) + stress.y_axis.dot(dc.y_axis);
        // The energy's change from the increment itself (the corotated law's
        // exact identity, `corotated_energy_increment`), not as the
        // difference of two energies: theirs carries `C = F^T F`'s round-off,
        // about `eps |S| |C|`, whatever `dC` is, and divided by `|dC|` at a
        // particle near its turning point (its `dC` near zero, its
        // neighbours' not) it was a force of 1e-9, the residual's noise Newton
        // stalled on (measured: none of a vibrating bar's 5 stalls with the
        // correction off). Computed from `dC`, the numerator's round-off is
        // relative to the change and the correction's stays near `eps |S|`.
        // Liu & Guan, arXiv 2305.15812, sec. 4.3, name the quotient's
        // fragility at small increments.
        let gap = self.energy_increment(c0, c1) - 0.5 * contracted;
        stress + (2.0 * gap / norm2) * dc
    }
}
