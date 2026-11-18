//! Active stress of a muscle fiber: the Hill-type law in the form of
//! Millard, Uchida, Seth and Delp, "Flexing computational muscle", J.
//! Biomech. Eng. 135(2) 021005 (2013), read in full (PMC3705831):
//!
//! - eq. (4), the fiber force `f^M = f_o^M (a f_L(l~) f_V(v~) + f_PE(l~))`;
//! - eqs. (1)-(3), the activation state `a` driven by the excitation `u`:
//!   `a^ = (a - a_min) / (1 - a_min)`, `da/dt = (u - a^) / tau`, `tau =
//!   tau_A (0.5 + 1.5 a^)` while `u > a^`, `tau_D / (0.5 + 1.5 a^)`
//!   otherwise (their defaults 10 ms and 40 ms, from Thelen 2003, their
//!   ref. 39, for human muscle).
//!
//! Adaptations, declared:
//! - The 1D law read as a stress along the fiber of a continuum: the fiber
//!   Cauchy stress `sigma_f = f^M / A`, so `f_o^M` becomes the maximal
//!   isometric stress `sigma_o = f_o^M / A` (`A` the physiological cross
//!   section), the fiber's current direction `n = F n_0 / |F n_0|`, its
//!   stretch `lambda = |F n_0|` standing for `l~`. The passive term `f_PE`
//!   is left to the tissue's own elastic law, which already resists
//!   stretch.
//! - `a_min` exists in the source only to avoid a singularity of their
//!   equilibrium tendon model (their eq. 6); a continuum carrying the fiber
//!   stress directly has no such singularity, so `a_min = 0` is allowed.
//! - The source's curves `f_L` and `f_V` are quintic Bezier fits to data
//!   whose control points the paper does not print: they are inputs here
//!   (`Curves`), and `Curves::PLATEAU_ISOMETRIC` (both 1) is valid only at
//!   the optimal length and zero fiber speed. Kind: scope; why: no curve
//!   read with its numbers; removal: a read source giving `f_L` and `f_V`
//!   in closed form or with their control points.
//!
//! Every coefficient is the record's (`HillMuscle`), never a constant here.

use glam::{DMat2, DVec2};

/// A muscle's measured active properties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HillMuscle {
    /// Maximal isometric fiber stress `sigma_o`, Pa.
    pub max_isometric_stress_pa: f64,
    /// Activation time constant `tau_A`, s.
    pub activation_time_s: f64,
    /// Deactivation time constant `tau_D`, s.
    pub deactivation_time_s: f64,
    /// The activation's lower bound `a_min` (eq. 1), in `[0, 1)`.
    pub min_activation: f64,
}

/// The normalized force-length and force-velocity curves `f_L(lambda)` and
/// `f_V(rate)`, `rate` the fiber stretch rate in optimal lengths per
/// maximal contraction time (Millard's `v~`).
#[derive(Clone, Copy, Debug)]
pub struct Curves {
    pub force_length: fn(f64) -> f64,
    pub force_velocity: fn(f64) -> f64,
}

impl Curves {
    /// Both curves at 1: the optimal length, the fiber at rest. Valid only
    /// there (module doc).
    pub const PLATEAU_ISOMETRIC: Self = Self {
        force_length: |_| 1.0,
        force_velocity: |_| 1.0,
    };
}

impl HillMuscle {
    /// Checks the record: positive stress and time constants, `a_min` in
    /// `[0, 1)`.
    pub fn new(
        max_isometric_stress_pa: f64,
        activation_time_s: f64,
        deactivation_time_s: f64,
        min_activation: f64,
    ) -> Self {
        assert!(
            max_isometric_stress_pa.is_finite() && max_isometric_stress_pa > 0.0,
            "the maximal isometric stress must be positive, got {max_isometric_stress_pa}"
        );
        assert!(
            activation_time_s > 0.0 && deactivation_time_s > 0.0,
            "the activation time constants must be positive"
        );
        assert!(
            (0.0..1.0).contains(&min_activation),
            "a_min must lie in [0, 1), got {min_activation}"
        );
        Self {
            max_isometric_stress_pa,
            activation_time_s,
            deactivation_time_s,
            min_activation,
        }
    }

    /// `da/dt` for excitation `u` in `[0, 1]` at activation `a` (eqs. 1-3).
    /// The time integration belongs to the solver.
    pub fn activation_rate(&self, excitation: f64, activation: f64) -> f64 {
        let scaled = (activation - self.min_activation) / (1.0 - self.min_activation);
        let tau = if excitation > scaled {
            self.activation_time_s * (0.5 + 1.5 * scaled)
        } else {
            self.deactivation_time_s / (0.5 + 1.5 * scaled)
        };
        (excitation - scaled) / tau
    }

    /// The fiber's active Kirchhoff stress `tau = J sigma_f n (x) n` for
    /// deformation gradient `f`, reference fiber direction `fiber` (unit),
    /// activation `a` and fiber stretch rate `rate` (module doc), with
    /// `sigma_f = a sigma_o f_L(lambda) f_V(rate)`: a tension along the
    /// fiber, zero across it.
    pub fn active_kirchhoff_stress(
        &self,
        f: DMat2,
        fiber: DVec2,
        activation: f64,
        rate: f64,
        curves: Curves,
    ) -> DMat2 {
        let along = f * fiber;
        let stretch = along.length();
        let n = along / stretch;
        let sigma = activation
            * self.max_isometric_stress_pa
            * (curves.force_length)(stretch)
            * (curves.force_velocity)(rate);
        let tau = f.determinant() * sigma;
        DMat2::from_cols(tau * n.x * n, tau * n.y * n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The time constants of the source's defaults (human, Thelen 2003 via
    /// Millard eq. 3); the stress is any positive value, the tests being
    /// relative to it.
    fn muscle() -> HillMuscle {
        HillMuscle::new(1.0e5, 0.010, 0.040, 0.0)
    }

    /// At the optimal length, at rest, fully active: the fiber carries
    /// exactly `sigma_o` along itself and nothing across.
    #[test]
    fn a_fully_active_fiber_at_rest_carries_its_isometric_stress() {
        let m = muscle();
        let fiber = DVec2::new(0.6, 0.8);
        let tau =
            m.active_kirchhoff_stress(DMat2::IDENTITY, fiber, 1.0, 0.0, Curves::PLATEAU_ISOMETRIC);
        assert!((fiber.dot(tau * fiber) - m.max_isometric_stress_pa).abs() < 1e-9);
        assert!(fiber.perp().dot(tau * fiber.perp()).abs() < 1e-9);
        assert_eq!(tau, tau.transpose());
    }

    /// A rotation turns the stress with the fiber (objectivity), and no
    /// activation gives no stress.
    #[test]
    fn the_stress_turns_with_the_fiber_and_vanishes_without_activation() {
        let m = muscle();
        let fiber = DVec2::X;
        let r = DMat2::from_angle(0.7);
        let at_rest =
            m.active_kirchhoff_stress(DMat2::IDENTITY, fiber, 0.5, 0.0, Curves::PLATEAU_ISOMETRIC);
        let turned = m.active_kirchhoff_stress(r, fiber, 0.5, 0.0, Curves::PLATEAU_ISOMETRIC);
        let expected = r * at_rest * r.transpose();
        assert!((turned - expected).abs_diff_eq(DMat2::ZERO, 1e-9));
        let off = m.active_kirchhoff_stress(r, fiber, 0.0, 0.0, Curves::PLATEAU_ISOMETRIC);
        assert_eq!(off, DMat2::ZERO);
    }

    /// Eqs. (1)-(3) at their ends: switching on from rest rises at
    /// `1 / (0.5 tau_A)`, switching off from full activation falls at
    /// `2 / tau_D`, and an activation equal to its excitation stays.
    #[test]
    fn the_activation_rises_and_falls_at_the_source_rates() {
        let m = muscle();
        assert!((m.activation_rate(1.0, 0.0) - 1.0 / (0.5 * 0.010)).abs() < 1e-9);
        assert!((m.activation_rate(0.0, 1.0) + 2.0 / 0.040).abs() < 1e-9);
        assert_eq!(m.activation_rate(0.3, 0.3), 0.0);
    }
}
