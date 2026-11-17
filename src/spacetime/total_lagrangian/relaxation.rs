//! A solid's relaxation branches (`materials::relaxation`) on the step: each
//! mechanism's Maxwell branch on each part of the corotated law, its
//! internal variable advanced by the exponential recurrence of Taylor,
//! Pister and Goudreau (1970), as given by Simo and Hughes (Computational
//! Inelasticity, 1998, sec. 10.3.2, eq. 10.3.16, read), with an
//! energy-consistent step stress.
//!
//! Per branch of modulus `mu_b` and viscosity `eta_b` on a part whose second
//! Piola-Kirchhoff stress is `S_P(C)` (`corotated_second_pk` with that
//! part's modulus alone), with the internal strain held as `E = Gamma - I`:
//! - the conjugate `Q(C, E) = S_P(C) - mu_b E` and the branch energy
//!   `Ups = |Q|^2 / (4 mu_b)` (Liu and Guan, arXiv 2305.15812, eq. 2.2 with
//!   `S_0 = 0`: the body starts unstrained, `C = I`, `Gamma = I`);
//! - the flow `eta_b dE/dt = Q`, so `Q' + theta Q = S_P'`, `theta = mu_b /
//!   eta_b`; with `S_P` linear in time over the step it is solved exactly
//!   (Simo and Hughes eq. 10.3.16): `Q_n+1 = e^(-h) Q_n + (1 - e^(-h)) / h
//!   dS`, `h = theta dt`, `dS = S_P(C_n+1) - S_P(C_n)`. Its eigenvalue is
//!   exactly `e^(-h)` at every step length: a branch faster than the step
//!   relaxes, a mode faster than the step is damped as the branch damps it
//!   (the midpoint's `(1 - h/2)/(1 + h/2)` tends to -1 and kept such modes
//!   flipping, measured: a set-down block rang 150 times longer than its
//!   record's damping);
//! - the heat, that path's exact dissipation `eta_b int |dE/dt|^2 dt`, never
//!   negative;
//! - the step's stress: the branch force at the step's mean conjugate
//!   `Q_bar = phi1 Q_n + phi2 dS` (`phi1 = (1 - e^(-h)) / h`, `phi2 = (1 -
//!   phi1) / h`, the mean of the exact `Q(t)`), `S_neq = (1 / mu_b)
//!   dS_P/dC [Q_bar]` at `C_n+1/2` (Liu and Guan's eq. 2.8 form), projected
//!   along `Z = dC / 2` so that `Z : S = Ups(C_n+1, E_n+1) - Ups(C_n, E_n) +
//!   heat` exactly (the discrete-gradient structure of their eqs.
//!   4.25-4.26).
//!
//! Adaptations, declared:
//! - the step stress is derived here, not taken from a source: Simo and
//!   Hughes give the recurrence, Liu and Guan the energy-consistent form for
//!   the midpoint update; no read source pairs the exponential recurrence
//!   with a discrete-gradient stress (Muller, Gross and Betsch 2009 and Gross
//!   and Betsch 2005 keep a midpoint or time-finite-element evolution).
//!   `Q_bar` is the exact mean force in the linear limit; at finite strain
//!   the chord's error is second order in smooth motion. Measured: the
//!   Zener gates (`tests/zener_*`) against the exact solution, and the impact
//!   bench converging monotonically in the steps (light_on_heavy: 0.76,
//!   0.13, 0.12, 0.007 % at 86, 185, 370, 1179 steps);
//! - theirs is three-dimensional, incompressible and isochoric (`C~`); here
//!   the plane strain `C` itself, each part of the compressible corotated law
//!   carrying its own branches;
//! - the energy difference in the quotient is formed from the increment
//!   `S_P(C_n+1) - S_P(C_n)` (`corotated_second_pk_increment`), so its
//!   round-off is relative to the change, as for the elastic law
//!   (`Elastic::algorithmic_stress`); and the quotient is formed above this
//!   engine's own `|dC|^2 > 1e-28` (Liu and Guan's sec. 6, item 6, uses `|Z|
//!   > 1e-10`).

use super::*;
use crate::matter::materials::relaxation::{
    Branch, MAX_MECHANISMS, Mechanism, RelaxationError, RelaxationSpectrum,
};
use crate::matter::materials::solid::corotated::stress::{
    corotated_second_pk, corotated_second_pk_increment, corotated_second_pk_tangent,
};

#[cfg(all(test, feature = "experimental"))]
#[path = "probes/branch_update_probes.rs"]
mod branch_update;

/// One branch on the step: its spring and dashpot, and its internal strain
/// `E = Gamma - I`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BranchState {
    branch: Branch,
    strain: DMat2,
}

/// A particle's relaxation branches: per mechanism, one on the corotated
/// law's shear part and one on its volume part (`None` where that part
/// stores nothing or takes no weight). Empty: no relaxation, and no
/// arithmetic on any path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Relaxation {
    count: usize,
    branches: [[Option<BranchState>; 2]; MAX_MECHANISMS],
    /// The spectrum's mechanisms the branches were built from.
    mechanisms: [Mechanism; MAX_MECHANISMS],
    /// The record's validity limit on the strain (`RelaxationSpectrum::
    /// strain_limit`).
    strain_limit: f64,
}

/// One mechanism's memory across a modal frame (`Relaxation::carried`):
/// `e^(-theta dt)`, its weight `kappa`, and the memory strains `eps_l =
/// sum_i sym(grad phi_i) s_il` at the frame's start and end.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CarriedMemory {
    pub(crate) decay: f64,
    pub(crate) kappa: f64,
    pub(crate) start: DMat2,
    pub(crate) end: DMat2,
}

/// What a kept step did to a particle's branches.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Relaxed {
    /// The heat the branches made over the step, per reference volume.
    pub(crate) heat: f64,
    /// The Green strain's size `|C_n+1 - I| / 2` at the step's end.
    pub(crate) strain: f64,
    /// Whether that strain is beyond the record's validity limit.
    pub(crate) beyond: bool,
}

impl Relaxation {
    pub(crate) const fn none() -> Self {
        Self {
            count: 0,
            branches: [[None; 2]; MAX_MECHANISMS],
            mechanisms: [Mechanism {
                theta: 0.0,
                kappa_shear: 0.0,
                kappa_p: 0.0,
            }; MAX_MECHANISMS],
            strain_limit: 0.0,
        }
    }

    /// The branches of `spectrum` on the corotated law of relaxed Lame
    /// moduli `(lambda, mu)`, unstrained.
    pub(crate) fn corotated(
        spectrum: &RelaxationSpectrum,
        lambda: f64,
        mu: f64,
    ) -> Result<Self, RelaxationError> {
        let weights = spectrum.part_weights(lambda, mu)?;
        let mut out = Self::none();
        out.count = spectrum.mechanisms().len();
        out.mechanisms[..out.count].copy_from_slice(spectrum.mechanisms());
        out.strain_limit = spectrum.strain_limit;
        for (l, m) in spectrum.mechanisms().iter().enumerate() {
            let (shear, volume) = weights[l];
            let state = |modulus: f64, kappa: f64| {
                (modulus > 0.0 && kappa > 0.0).then(|| BranchState {
                    branch: Branch::of_part(modulus, kappa, m.theta),
                    strain: DMat2::ZERO,
                })
            };
            out.branches[l] = [state(mu, shear), state(lambda, volume)];
        }
        Ok(out)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// `1 + sum kappa_p`: the P-wave modulus's unrelaxed over relaxed ratio,
    /// the instantaneous stiffness a fast event meets (every branch adding
    /// its weight before it relaxes).
    pub(crate) fn unrelaxed_p_ratio(&self) -> f64 {
        1.0 + self.mechanisms[..self.count]
            .iter()
            .map(|m| m.kappa_p)
            .sum::<f64>()
    }

    /// The mechanisms and the strain limit, when every mechanism weighs the
    /// shear and the P wave alike (`kappa_shear == kappa_p`): a proportional
    /// record, whose part weights are then `(kappa, kappa)` on any law, so
    /// the branches act through the law's own stiffness (`modes`).
    pub(crate) fn proportional(&self) -> Option<(&[Mechanism], f64)> {
        let mechanisms = &self.mechanisms[..self.count];
        mechanisms
            .iter()
            .all(|m| m.kappa_shear == m.kappa_p)
            .then_some((mechanisms, self.strain_limit))
    }

    /// Per mechanism, the branches' stress linearised at rest: `sum_P
    /// dS_P/dC|_I [Q_P] / mu_b` with `Q_P = S_P(C) - mu_b E` at `C`, the
    /// step's `S_neq` (eq. 2.8) at `C = I`. Its contraction with a mode's
    /// gradient is that mode's share of the mechanism's force (`flight`).
    pub(crate) fn linear_stress(&self, c: DMat2, moduli: (f64, f64)) -> [DMat2; MAX_MECHANISMS] {
        let mut out = [DMat2::ZERO; MAX_MECHANISMS];
        for (l, pair) in self.branches[..self.count].iter().enumerate() {
            for (b, (lambda, mu)) in pair
                .iter()
                .zip([(0.0, moduli.1), (moduli.0, 0.0)])
                .filter_map(|(b, part)| b.as_ref().map(|b| (b, part)))
            {
                let q = corotated_second_pk(c, lambda, mu) - b.strain * b.branch.modulus;
                out[l] += corotated_second_pk_tangent(DMat2::IDENTITY, q, lambda, mu)
                    * b.branch.modulus.recip();
            }
        }
        out
    }

    /// The branches carried across a modal frame from `C_0` to `C_1`, in the
    /// linear limit, exactly; returns the heat their part outside the modes
    /// made, per reference volume.
    ///
    /// Derived on the page. Linearised at rest, a branch's conjugate obeys
    /// `Q' + theta Q = S_P,lin(eps')` (`eta_b E' = Q`, `mu_b / eta_b =
    /// theta`), so over the frame `Q_1 = e^(-theta dt) Q_0 + S_P,lin(int_0^dt
    /// e^(-theta (dt - t)) eps'(t) dt)`. With the body-frame strain `eps =
    /// sum_i sym(grad phi_i) q_i` and the modal memory `s_il' + theta s_il =
    /// kappa q_i'`, the integral is `sum_i sym(grad phi_i) (s_il(dt) -
    /// e^(-theta dt) s_il(0)) / kappa`, i.e. `(end - decay start) / kappa`
    /// in `memory`. The internal strain follows from its definition, `E_1 =
    /// (S_P(C_1) - Q_1) / mu_b`.
    ///
    /// `Q_0` splits into the modal part `S_P,lin(start) / kappa`, whose
    /// energy the modes carry, and the rest, which no mode sees: its
    /// linearised force is zero, so it is orthogonal to the modal part in
    /// the branch energy `|Q|^2 / (4 mu_b)` and decays alone as `e^(-theta
    /// t)`, its energy becoming heat, `(1 - e^(-2 theta dt))` of it.
    pub(crate) fn carried(
        &mut self,
        (c0, c1): (DMat2, DMat2),
        (lambda, mu): (f64, f64),
        memory: &[CarriedMemory],
    ) -> f64 {
        let mut heat = 0.0;
        for (pair, m) in self.branches[..self.count].iter_mut().zip(memory) {
            for (b, (l, u)) in pair
                .iter_mut()
                .zip([(0.0, mu), (lambda, 0.0)])
                .filter_map(|(b, part)| b.as_mut().map(|b| (b, part)))
            {
                let mu_b = b.branch.modulus;
                let linear = |eps: DMat2| {
                    corotated_second_pk_tangent(DMat2::IDENTITY, eps * 2.0, l, u) * m.kappa.recip()
                };
                let q0 = corotated_second_pk(c0, l, u) - b.strain * mu_b;
                let outside = q0 - linear(m.start);
                heat += colon(outside, outside) / (4.0 * mu_b) * (1.0 - m.decay * m.decay);
                let q1 = q0 * m.decay + linear(m.end - m.start * m.decay);
                b.strain = (corotated_second_pk(c1, l, u) - q1) * mu_b.recip();
            }
        }
        heat
    }

    /// Each branch with its part's moduli, `(0, mu)` (shear) or `(lambda,
    /// 0)` (volume).
    fn each(&self, (lambda, mu): (f64, f64)) -> impl Iterator<Item = (&BranchState, (f64, f64))> {
        self.branches[..self.count].iter().flat_map(move |pair| {
            pair[0]
                .iter()
                .map(move |b| (b, (0.0, mu)))
                .chain(pair[1].iter().map(move |b| (b, (lambda, 0.0))))
        })
    }

    /// The branches' stored energy at `C`, per reference volume.
    pub(crate) fn energy(&self, c: DMat2, moduli: (f64, f64)) -> f64 {
        self.each(moduli)
            .map(|(b, (l, m))| {
                let q = corotated_second_pk(c, l, m) - b.strain * b.branch.modulus;
                colon(q, q) / (4.0 * b.branch.modulus)
            })
            .sum()
    }

    /// The branches' stress at `C` in their present state, `2 dPsi/dC` of
    /// `energy`: per branch `S_P'(C)[q] / mu_b`, `q = S_P(C) - mu_b E` (the
    /// step stress's `s_neq` at one point). For the error estimate's
    /// accelerations (`estimate`).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn stress(&self, c: DMat2, moduli: (f64, f64)) -> DMat2 {
        self.each(moduli)
            .map(|(b, (l, m))| {
                let q = corotated_second_pk(c, l, m) - b.strain * b.branch.modulus;
                corotated_second_pk_tangent(c, q, l, m) * b.branch.modulus.recip()
            })
            .fold(DMat2::ZERO, |sum, s| sum + s)
    }

    /// The branches' step stress between `C_n` and `C_{n+1}` over `dt`.
    pub(crate) fn step_stress(&self, c0: DMat2, c1: DMat2, dt: f64, moduli: (f64, f64)) -> DMat2 {
        let dc = c1 - c0;
        let z = dc * 0.5;
        let norm2 = colon(z, z);
        let mid = 0.5 * (c0 + c1);
        // A part's stresses at both ends and at the midpoint, and its
        // increment, are the same for every branch on that part: formed once
        // per part, the same expressions on the same inputs, so the sum is
        // bitwise the per-branch one.
        let (lambda, mu) = moduli;
        let on = |part: usize| {
            self.branches[..self.count]
                .iter()
                .any(|b| b[part].is_some())
        };
        let part = |(l, m): (f64, f64)| {
            (
                corotated_second_pk(c0, l, m),
                corotated_second_pk(c1, l, m),
                corotated_second_pk_increment(c0, c1, l, m),
            )
        };
        let parts = [(0.0, mu), (lambda, 0.0)];
        let formed = [on(0).then(|| part(parts[0])), on(1).then(|| part(parts[1]))];
        // `each`'s order: per mechanism, its shear branch then its volume one.
        self.branches[..self.count]
            .iter()
            .flat_map(|pair| (0..2).filter_map(move |k| pair[k].as_ref().map(|b| (b, k))))
            .map(|(b, k)| {
                let (l, m) = parts[k];
                let (s0, s1, ds) = formed[k].expect("formed for each part a branch is on");
                #[cfg(all(test, feature = "experimental"))]
                if let Some(update) = branch_update::active() {
                    return branch_update::stress(b, (s0, s1), (c0, c1, dt), (l, m), update);
                }
                branch_step_stress(b, (s0, s1, ds), (mid, z, norm2), dt, (l, m))
            })
            .fold(DMat2::ZERO, |sum, s| sum + s)
    }

    /// Every branch relaxed at `C`: its internal strain `E = S_P(C) / mu_b`,
    /// so its conjugate `Q = S_P(C) - mu_b E` is zero and the branch stores
    /// and pushes nothing (`energy`, `step_stress` at `C_n = C_{n+1} = C`).
    /// The state a branch reaches held at `C` long enough: Liu and Guan's
    /// eq. 4.23 at constant `C` decays `Q` as `e^(-theta t)`. For the static
    /// mode's end state (`Elastic::static_stress`), an initial condition.
    pub(crate) fn relax_at(&mut self, c: DMat2, (lambda, mu): (f64, f64)) {
        for pair in self.branches[..self.count].iter_mut() {
            for (b, (l, m)) in pair
                .iter_mut()
                .zip([(0.0, mu), (lambda, 0.0)])
                .filter_map(|(b, moduli)| b.as_mut().map(|b| (b, moduli)))
            {
                b.strain = corotated_second_pk(c, l, m) * b.branch.modulus.recip();
            }
        }
    }

    /// The branches after a kept step from `C_n` to `C_{n+1}` of length
    /// `dt`, and what the step did (the heat, the strain against the
    /// record's limit).
    pub(crate) fn committed(
        &self,
        c0: DMat2,
        c1: DMat2,
        dt: f64,
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
                let (s0, s1) = (corotated_second_pk(c0, l, m), corotated_second_pk(c1, l, m));
                #[cfg(all(test, feature = "experimental"))]
                if let Some(update) = branch_update::active() {
                    let (end, branch_heat) = branch_update::advance(b, (s0, s1), dt, update);
                    branch_update::count_negative(branch_heat);
                    heat += branch_heat;
                    b.strain = end;
                    continue;
                }
                let (end, made) = exponential_step(b, (s0, s1), dt);
                heat += made;
                b.strain = end;
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

/// The branch over a step of length `dt` from `S_P(C_n) = s0` to
/// `S_P(C_n+1) = s1`, `S_P` linear in time over the step, solved exactly
/// (Simo and Hughes 1998, eq. 10.3.16): returns `E_n+1` and the step's heat.
/// With `h = theta dt` and the target `E* = S_P / mu_b`,
/// `E_n+1 = e^(-h) E_n + (1 - e^(-h)) E*_n + (1 - (1 - e^(-h)) / h)
/// (E*_n+1 - E*_n)`; the heat is `mu_b / 2 int_0^h |e(s)|^2 ds` (`s` in units
/// of `1 / theta`), with the lag behind the target
/// `e(s) = (E_n - E*_n) e^(-s) + (E*_n+1 - E*_n) expm1(-s) / h`, in closed
/// form above `h = 0.1` and by 6-point Gauss-Legendre below (no
/// cancellation).
fn exponential_step(b: &BranchState, (s0, s1): (DMat2, DMat2), dt: f64) -> (DMat2, f64) {
    let Branch {
        modulus: mu,
        viscosity: eta,
    } = b.branch;
    let h = dt * mu / eta;
    let (target0, target1) = (s0 * mu.recip(), s1 * mu.recip());
    let e0 = b.strain;
    let decay = (-h).exp();
    let rise = target1 - target0;
    // `1 - (1 - e^-h) / h`, the lag behind a moving target.
    let lag = 1.0 + (-h).exp_m1() / h;
    let e1 = e0 * decay + target0 * (1.0 - decay) + rise * lag;
    let e00 = e0 - target0;
    let heat = if h > 0.1 {
        // Closed form with `u = e00 + rise / h`, `v = rise / h`.
        let v = rise * h.recip();
        let u = e00 + v;
        0.5 * mu
            * (0.5 * colon(u, u) * (1.0 - decay * decay) - 2.0 * colon(u, v) * (1.0 - decay)
                + colon(v, v) * h)
    } else {
        let integral: f64 = GAUSS_NODES
            .iter()
            .zip(GAUSS_WEIGHTS)
            .map(|(&x, w)| {
                let s = 0.5 * h * (x + 1.0);
                let e = e00 * (-s).exp() + rise * ((-s).exp_m1() / h);
                w * colon(e, e)
            })
            .sum::<f64>()
            * 0.5
            * h;
        0.5 * mu * integral
    };
    (e1, heat)
}

/// 6-point Gauss-Legendre nodes and weights on `[-1, 1]`.
const GAUSS_NODES: [f64; 6] = [
    -0.932_469_514_203_152,
    -0.661_209_386_466_265,
    -0.238_619_186_083_197,
    0.238_619_186_083_197,
    0.661_209_386_466_265,
    0.932_469_514_203_152,
];
const GAUSS_WEIGHTS: [f64; 6] = [
    0.171_324_492_379_170,
    0.360_761_573_048_139,
    0.467_913_934_572_691,
    0.467_913_934_572_691,
    0.360_761_573_048_139,
    0.171_324_492_379_170,
];

/// `(phi1, phi2)`: `(1 - e^-h) / h` and
/// `(1 - phi1) / h = int_0^1 s e^(-h (1 - s)) ds`, the weights of the step's
/// mean conjugate `Q_bar = phi1 Q_n + phi2 dS`; by their closed forms above
/// `h = 0.1`, by Gauss-Legendre below.
fn mean_weights(h: f64) -> (f64, f64) {
    if h > 0.1 {
        let phi1 = -(-h).exp_m1() / h;
        (phi1, (1.0 - phi1) / h)
    } else {
        GAUSS_NODES
            .iter()
            .zip(GAUSS_WEIGHTS)
            .fold((0.0, 0.0), |(p1, p2), (&x, w)| {
                let s = 0.5 * (x + 1.0);
                let decay = (-h * (1.0 - s)).exp();
                (p1 + 0.5 * w * decay, p2 + 0.5 * w * s * decay)
            })
    }
}

/// `Ups(C, E2) - Ups(C, E1)` at the part stress `s = S_P(C)`, from the change.
fn ups_change(s: DMat2, (e1, e2): (DMat2, DMat2), mu: f64) -> f64 {
    colon(e1 - e2, s * 2.0 - (e1 + e2) * mu) / 4.0
}

/// One branch's step stress (the module's header): the force at `Q_bar`
/// at the midpoint `mid`, projected along `z` (`norm2 = |z|^2`) so that `Z :
/// S` is `Ups`'s change over the step plus the exponential step's heat.
fn branch_step_stress(
    b: &BranchState,
    (s0, s1, ds): (DMat2, DMat2, DMat2),
    (mid, z, norm2): (DMat2, DMat2, f64),
    dt: f64,
    (l, m): (f64, f64),
) -> DMat2 {
    let mu_b = b.branch.modulus;
    let (phi1, phi2) = mean_weights(dt * mu_b / b.branch.viscosity);
    let q_bar = (s0 - b.strain * mu_b) * phi1 + (s1 - s0) * phi2;
    let s_neq = corotated_second_pk_tangent(mid, q_bar, l, m) * mu_b.recip();
    if 4.0 * norm2 <= 1e-28 {
        return s_neq;
    }
    let (e1, heat) = exponential_step(b, (s0, s1), dt);
    // `Ups(C_n+1, E_n+1) - Ups(C_n, E_n+1)` from the increment, plus
    // `Ups(C_n, E_n+1) - Ups(C_n, E_n)`, plus the heat.
    let work = colon(ds, s0 + s1 - e1 * (2.0 * mu_b)) / (4.0 * mu_b)
        + ups_change(s0, (b.strain, e1), mu_b)
        + heat;
    s_neq + z * ((work - colon(s_neq, z)) / norm2)
}

fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::materials::relaxation::Mechanism;

    fn branches(kappa_p: f64) -> Relaxation {
        let spectrum = RelaxationSpectrum::new(
            &[
                Mechanism {
                    theta: 650.0,
                    kappa_shear: 0.38,
                    kappa_p,
                },
                Mechanism {
                    theta: 25000.0,
                    kappa_shear: 0.75,
                    kappa_p: 2.0 * kappa_p,
                },
            ],
            1e-5,
        )
        .expect("positive");
        Relaxation::corotated(&spectrum, 2.0e6, 1.3e6).expect("positive part weights")
    }

    /// A proportional record's branches (shear and P weights alike).
    fn proportional() -> Relaxation {
        let spectrum = RelaxationSpectrum::new(
            &[
                Mechanism {
                    theta: 650.0,
                    kappa_shear: 0.38,
                    kappa_p: 0.38,
                },
                Mechanism {
                    theta: 25000.0,
                    kappa_shear: 0.75,
                    kappa_p: 0.75,
                },
            ],
            1e-5,
        )
        .expect("positive");
        Relaxation::corotated(&spectrum, 2.0e6, 1.3e6).expect("positive part weights")
    }

    /// `carried` against the step itself in the linear limit: a strain
    /// history `eps(t) = eps_a sin(omega t)` of size 1e-7 from rest over one
    /// frame, stepped by the exponential update in 20000 steps, and carried
    /// at once with the memory strain `kappa int e^(-theta (T - t)) eps'(t)
    /// dt` in closed form. Criteria fixed before measuring: each branch's
    /// internal strain within 1e-5 of the stepped one's size (the steps'
    /// second-order error and the law's nonlinearity at 1e-7 are both far
    /// below).
    #[test]
    fn carried_branches_are_the_stepped_ones_in_the_linear_limit() {
        let moduli = (2.0e6, 1.3e6);
        let amplitude = DMat2::from_cols(DVec2::new(1.0e-7, 0.4e-7), DVec2::new(0.4e-7, -0.6e-7));
        let (omega, frame, steps) = (2000.0, 1e-3, 20_000);
        let c = |t: f64| DMat2::IDENTITY + amplitude * (2.0 * (omega * t).sin());
        let mut stepped = proportional();
        let h = frame / steps as f64;
        for k in 0..steps {
            let (t0, t1) = (k as f64 * h, (k + 1) as f64 * h);
            stepped = stepped.committed(c(t0), c(t1), h, moduli).0;
        }
        let mut carried = proportional();
        let memory: Vec<CarriedMemory> = carried
            .proportional()
            .expect("proportional")
            .0
            .iter()
            .map(|m| {
                let theta = m.theta;
                // `int_0^T e^(-theta (T - t)) omega cos(omega t) dt`.
                let integral = omega
                    * (theta * (omega * frame).cos() + omega * (omega * frame).sin()
                        - theta * (-theta * frame).exp())
                    / (theta * theta + omega * omega);
                CarriedMemory {
                    decay: (-theta * frame).exp(),
                    kappa: m.kappa_shear,
                    start: DMat2::ZERO,
                    end: amplitude * (m.kappa_shear * integral),
                }
            })
            .collect();
        let null = carried.carried((DMat2::IDENTITY, c(frame)), moduli, &memory);
        assert_eq!(null, 0.0, "unstrained branches: nothing outside the modes");
        for (a, b) in carried
            .branches
            .iter()
            .flatten()
            .zip(stepped.branches.iter().flatten())
        {
            let (Some(a), Some(b)) = (a, b) else {
                assert!(a.is_none() && b.is_none());
                continue;
            };
            let off = colon(a.strain - b.strain, a.strain - b.strain).sqrt();
            let size = colon(b.strain, b.strain).sqrt();
            assert!(off <= 1e-5 * size, "off by {off:e} of {size:e}");
        }
    }

    /// The branch stress no mode sees decays alone: with no motion and no
    /// memory, `Q_1 = e^(-theta dt) Q_0` and its energy's loss is the heat,
    /// to round-off.
    #[test]
    fn branch_stress_outside_the_modes_decays_into_heat() {
        let moduli = (2.0e6, 1.3e6);
        let c0 = DMat2::from_cols(DVec2::new(1.0 + 2e-6, 1e-6), DVec2::new(1e-6, 1.0 - 1e-6));
        let mut branches = proportional();
        branches = branches.committed(DMat2::IDENTITY, c0, 1e-4, moduli).0;
        let before = branches.energy(c0, moduli);
        let dt = 1e-4;
        let memory: Vec<CarriedMemory> = branches
            .proportional()
            .expect("proportional")
            .0
            .iter()
            .map(|m| CarriedMemory {
                decay: (-m.theta * dt).exp(),
                kappa: m.kappa_shear,
                start: DMat2::ZERO,
                end: DMat2::ZERO,
            })
            .collect();
        let heat = branches.carried((c0, c0), moduli, &memory);
        let after = branches.energy(c0, moduli);
        assert!(heat > 0.0);
        assert!(
            (before - after - heat).abs() <= 1e-12 * before,
            "energy {before:e} -> {after:e}, heat {heat:e}"
        );
    }

    /// The energy identity the step stress is built to (the module's header,
    /// after Liu and Guan's Lemma 4 with eqs. 4.25-4.26) on every branch: the
    /// step's stress contracted with `Z` is the branches' energy change plus
    /// the heat, to round-off, on large and small steps, from a strained state
    /// with strained branches.
    #[test]
    fn the_step_stress_carries_the_energy_change_and_the_heat_exactly() {
        let moduli = (2.0e6, 1.3e6);
        let c0 = DMat2::from_cols(DVec2::new(1.08, 0.03), DVec2::new(0.03, 0.95));
        let mut start = branches(0.5);
        // Strain the branches: one kept step from rest to `c0`.
        start = start.committed(DMat2::IDENTITY, c0, 1e-3, moduli).0;
        let direction = DMat2::from_cols(DVec2::new(0.3, -0.5), DVec2::new(-0.5, 0.8));
        for (h, dt) in [(5e-2, 1e-3), (1e-4, 1e-5), (1e-9, 1e-7)] {
            let c1 = c0 + direction * h;
            let z = (c1 - c0) * 0.5;
            let stress = start.step_stress(c0, c1, dt, moduli);
            let (next, relaxed) = start.committed(c0, c1, dt, moduli);
            let change = next.energy(c1, moduli) - start.energy(c0, moduli);
            let work = colon(stress, z);
            let scale = start.energy(c0, moduli) * f64::EPSILON * 64.0
                + (colon(stress, stress) * colon(z, z)).sqrt() * 1e-12;
            assert!(
                (work - change - relaxed.heat).abs() <= scale,
                "step {h}: work {work:e}, energy change {change:e} + heat {:e}",
                relaxed.heat
            );
            assert!(relaxed.heat >= 0.0);
        }
    }

    /// The step stress as each branch formed its part's stresses itself (the
    /// earlier form): the oracle the shared form must match bit for bit.
    fn step_stress_per_branch(
        r: &Relaxation,
        c0: DMat2,
        c1: DMat2,
        dt: f64,
        moduli: (f64, f64),
    ) -> DMat2 {
        let z = (c1 - c0) * 0.5;
        let norm2 = colon(z, z);
        let mid = 0.5 * (c0 + c1);
        r.each(moduli)
            .map(|(b, (l, m))| {
                let (s0, s1, ds) = (
                    corotated_second_pk(c0, l, m),
                    corotated_second_pk(c1, l, m),
                    corotated_second_pk_increment(c0, c1, l, m),
                );
                branch_step_stress(b, (s0, s1, ds), (mid, z, norm2), dt, (l, m))
            })
            .fold(DMat2::ZERO, |sum, s| sum + s)
    }

    /// The part-shared step stress is bit
    /// for bit the per-branch one, over strained branches (both parts, the
    /// shear part alone, a proportional record, Crandall's), random `C_n`, `C_n+1`
    /// (steps from 1e-12 to 1e-1, the quotient's cut at 0 included) and step
    /// lengths from 1e-7 to 1e-2 s.
    #[test]
    fn the_part_shared_step_stress_is_bitwise_the_per_branch_one() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut uniform = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut checked = 0;
        let crandall = RelaxationSpectrum::plasticine_crandall_1970();
        let records = [
            (branches(0.5), (2.0e6, 1.3e6)),
            (proportional(), (2.0e6, 1.3e6)),
            (
                Relaxation::corotated(&crandall, 2.0e6, 1.3e6).expect("positive part weights"),
                (2.0e6, 1.3e6),
            ),
            // No volume part (`lambda = 0`): the shear branches alone.
            (
                Relaxation::corotated(&crandall, 0.0, 1.3e6).expect("positive part weights"),
                (0.0, 1.3e6),
            ),
        ];
        for (record, moduli) in records {
            for _ in 0..400 {
                let gradient = |u: &mut dyn FnMut() -> f64, size: f64| {
                    DMat2::IDENTITY
                        + DMat2::from_cols(DVec2::new(u(), u()), DVec2::new(u(), u())) * size
                };
                let f0 = gradient(&mut uniform, 0.2);
                let c0 = f0.transpose() * f0;
                let step = 10f64.powf(-12.0 + 11.0 * (0.5 + 0.5 * uniform()));
                let f1 = f0
                    + DMat2::from_cols(
                        DVec2::new(uniform(), uniform()),
                        DVec2::new(uniform(), uniform()),
                    ) * step;
                let c1 = if uniform() > 0.9 {
                    c0
                } else {
                    f1.transpose() * f1
                };
                let dt = 10f64.powf(-7.0 + 5.0 * (0.5 + 0.5 * uniform()));
                let strained = record.committed(DMat2::IDENTITY, c0, 1e-3, moduli).0;
                let shared = strained.step_stress(c0, c1, dt, moduli);
                let oracle = step_stress_per_branch(&strained, c0, c1, dt, moduli);
                let bits = |m: DMat2| m.to_cols_array().map(f64::to_bits);
                assert_eq!(
                    bits(shared),
                    bits(oracle),
                    "C_n {c0:?}, C_n+1 {c1:?}, dt {dt:e}"
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 1600);
    }
}
