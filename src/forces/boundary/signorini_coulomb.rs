//! The law of a solid touching a wall or another solid: it may push but
//! never pull (Signorini), and while it pushes, friction resists sliding up
//! to a limit (Coulomb).
//!
//! Signorini: the gap `g` between the two surfaces stays at or above zero,
//! the normal force `N` stays at or above zero (contact pushes, never
//! pulls), and one of the two is zero at any time (`g N = 0`: no force at a
//! distance, no gap under load).
//! Source: Dabaghi, Krejci, Petrov, Pousin and Renard, arXiv 1601.00778
//! (2016), eq. 2.3.
//!
//! Coulomb: the tangential force `T` stays within `mu N`. Inside that
//! bound the surfaces stick (no sliding); on it they slide, with `T`
//! opposing the sliding. Written on the sliding velocity at the middle of
//! a time step, friction can then only turn motion into heat, never the
//! reverse (its work per step is `-mu N |slide|`).
//! Source: Acary and Collins-Craft, JTCAM 2025, doi 10.46298/jtcam.13480,
//! eqs. 56-57 and 73.
//!
//! The functions below state both laws as one projection that a solver can
//! iterate on: the forces are the trial values `lambda - k g` (the last
//! multipliers less a penalty times the gaps), clamped onto what the law
//! allows. At the solution the clamp holds exactly the conditions above,
//! whatever the penalties. The normal and the tangential penalty are two
//! parameters (`k_N`, `k_T`): Simo and Laursen, Comput. Struct. 42 (1992),
//! eqs. 3.9.1 and 3.9.3 (`epsilon_N`, `epsilon_T`), 3.16-3.19.
//! Source: Hueber, Stadler and Wohlmuth, SIAM J. Sci. Comput. 30 (2008),
//! eq. 3.14 (the tangential projection) and sec. 7, eq. 7.1 (Coulomb's
//! bound taken from the normal force of the same iterate). Adaptation:
//! written in two dimensions, where the tangent is one direction and the
//! projection onto the disc is a clamp onto `[-mu N, mu N]`.
//!
//! Friction may resist the two senses of sliding differently, as a snake's
//! scales or an earthworm's setae do: Hu, Nirody, Scott and Shelley, PNAS
//! 106 (2009) 10081, eq. 2, the force along the body's axis `s` while
//! sliding at velocity `u`, `-nu_g [mu_f H(u.s) + mu_b (1 - H(u.s))] (u.s)
//! s`, `H(x) = (1 + sgn x) / 2`. The tangential force then lies in
//! `[-mu_- N, mu_+ N]` (`CoulombBounds`), the interval of each sense.
//! Adaptations, declared: Hu's law is written for sliding only; the same
//! interval is taken at sticking, a convex set on which the projection
//! above and its dissipation hold unchanged (Studer, Leine and Glocker,
//! IJNME 76 (2008), eq. 6, any convex set of admissible forces). In two
//! dimensions the tangent is one line, so only the sense of `s` along it
//! counts; Hu's transverse coefficient has no direction to act in.

/// Which branch of the law a contact is on, held fixed while a semismooth
/// Newton method differentiates it: whether the surfaces press, and if so
/// whether they stick (`slip` 0) or slide (`slip` the sign of the
/// tangential force).
/// Source: Hueber, Stadler and Wohlmuth 2008, eq. 3.2 (the generalized
/// derivative of `max`, branch by branch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactBranch {
    pub pressed: bool,
    pub slip: f64,
}

/// The admissible tangential force per unit normal force: `T` lies in
/// `[-negative N, positive N]`, `T` along the contact's tangent. Both
/// bounds are Coulomb coefficients, never negative; equal, isotropic
/// Coulomb friction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoulombBounds {
    /// Resists sliding along the tangent (`T` negative).
    pub negative: f64,
    /// Resists sliding against the tangent (`T` positive).
    pub positive: f64,
}

impl CoulombBounds {
    /// The same coefficient both ways.
    pub fn isotropic(mu: f64) -> Self {
        Self {
            negative: mu,
            positive: mu,
        }
    }

    /// The bounds of a surface whose friction is `forward` when it slides
    /// along its axis and `backward` against it (Hu et al. eq. 2), for a
    /// tangent whose dot with the axis is `along`, the surface's motion
    /// along the tangent being the one `T` resists when negative. At
    /// `along` 0 the source's `H(0) = 1/2`: the mean both ways.
    pub fn directional(forward: f64, backward: f64, along: f64) -> Self {
        if along > 0.0 {
            Self {
                negative: forward,
                positive: backward,
            }
        } else if along < 0.0 {
            Self {
                negative: backward,
                positive: forward,
            }
        } else {
            Self::isotropic(0.5 * (forward + backward))
        }
    }

    /// The same bounds seen from the other surface of the pair: its
    /// tangential force is the opposite one.
    pub fn reversed(self) -> Self {
        Self {
            negative: self.positive,
            positive: self.negative,
        }
    }

    /// Whether either sense has friction.
    pub fn frictional(self) -> bool {
        self.negative > 0.0 || self.positive > 0.0
    }

    /// The coefficient of the bound a slide of sign `slip` (the sign of `T`
    /// on it) reaches, signed: `slip` times that bound's coefficient.
    pub fn on_slip(self, slip: f64) -> f64 {
        if slip > 0.0 {
            slip * self.positive
        } else {
            slip * self.negative
        }
    }

    /// Whether `along` lies strictly inside the bounds for normal force
    /// `normal`: the surfaces stick.
    pub fn sticks(self, along: f64, normal: f64) -> bool {
        -self.negative * normal < along && along < self.positive * normal
    }
}

/// The normal and tangential forces of one contact. `multipliers` are the
/// last normal and tangential forces `(lambda_N, lambda_T)`, `gaps` the
/// normal and tangential relative displacements over the step `(g_N,
/// g_T)`, `(k, k_t)` the normal and tangential penalties and `friction`
/// Coulomb's coefficient `mu`.
/// Normal: `N = max(0, lambda_N - k g_N)`. Tangential: `lambda_T - k_t g_T`
/// clamped onto `[-mu N, mu N]`. With `frozen`, the same with the branch
/// held (each clamp replaced by the side it took), a smooth function of
/// the gaps.
pub fn contact_force(
    (normal_multiplier, tangential_multiplier): (f64, f64),
    (normal_gap, tangential_gap): (f64, f64),
    ((k, k_t), friction): ((f64, f64), CoulombBounds),
    frozen: Option<ContactBranch>,
) -> (f64, f64) {
    let normal = normal_multiplier - k * normal_gap;
    let trial = tangential_multiplier - k_t * tangential_gap;
    match frozen {
        None => {
            let normal = normal.max(0.0);
            (
                normal,
                trial.clamp(-friction.negative * normal, friction.positive * normal),
            )
        }
        Some(ContactBranch { pressed: false, .. }) => (0.0, 0.0),
        Some(ContactBranch { slip: 0.0, .. }) => {
            (normal, if friction.frictional() { trial } else { 0.0 })
        }
        Some(ContactBranch { slip, .. }) => (normal, friction.on_slip(slip) * normal),
    }
}

/// The branch `contact_force` takes at these multipliers and gaps.
pub fn contact_branch(
    (normal_multiplier, tangential_multiplier): (f64, f64),
    (normal_gap, tangential_gap): (f64, f64),
    ((k, k_t), friction): ((f64, f64), CoulombBounds),
) -> ContactBranch {
    let normal = normal_multiplier - k * normal_gap;
    let trial = tangential_multiplier - k_t * tangential_gap;
    ContactBranch {
        pressed: normal > 0.0,
        slip: if friction.sticks(trial, normal) {
            0.0
        } else {
            trial.signum()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_contact_pushes_but_never_pulls() {
        // Surfaces moving apart over the step (positive gap): no force.
        assert_eq!(
            contact_force(
                (0.0, 0.0),
                (0.5, 0.0),
                ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
                None
            ),
            (0.0, 0.0)
        );
        // Moving into each other: a push.
        let (normal, _) = contact_force(
            (0.0, 0.0),
            (-0.5, 0.0),
            ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
            None,
        );
        assert_eq!(normal, 5.0);
    }

    #[test]
    fn friction_sticks_inside_the_cone_and_slides_on_it() {
        // A small tangential pull is held in full: sticking.
        let (normal, along) = contact_force(
            (10.0, 0.0),
            (0.0, 0.1),
            ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
            None,
        );
        assert_eq!((normal, along), (10.0, -1.0));
        // A large one is capped at mu N, against the slide: sliding.
        let (normal, along) = contact_force(
            (10.0, 0.0),
            (0.0, 1.0),
            ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
            None,
        );
        assert_eq!(normal, 10.0);
        assert!((along + 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_frozen_branch_matches_the_law_where_it_was_taken() {
        let cases = [
            ((10.0, 0.0), (0.0, 0.1)),
            ((10.0, 0.0), (0.0, 1.0)),
            ((10.0, 0.0), (0.0, -1.0)),
            ((0.0, 0.0), (0.5, 0.0)),
        ];
        for (multipliers, gaps) in cases {
            let branch = contact_branch(
                multipliers,
                gaps,
                ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
            );
            assert_eq!(
                contact_force(
                    multipliers,
                    gaps,
                    ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
                    Some(branch)
                ),
                contact_force(
                    multipliers,
                    gaps,
                    ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
                    None
                ),
            );
        }
    }

    #[test]
    fn friction_never_does_positive_work() {
        // The tangential force times the slide is never positive, sticking
        // or sliding, whichever way the slide goes.
        for slide in [-2.0, -0.5, -0.01, 0.0, 0.01, 0.5, 2.0] {
            let (_, along) = contact_force(
                (10.0, 0.0),
                (0.0, slide),
                ((10.0, 10.0), CoulombBounds::isotropic(0.3)),
                None,
            );
            assert!(along * slide <= 0.0, "slide {slide}: force {along}");
        }
    }

    /// A surface easy forward and hard backward: each sense slides at its
    /// own bound, and the frozen branch still matches the law.
    #[test]
    fn a_directional_surface_slides_at_each_senses_bound() {
        let bounds = CoulombBounds::directional(0.1, 0.4, 1.0);
        let law = ((10.0, 10.0), bounds);
        let (_, forward) = contact_force((10.0, 0.0), (0.0, 1.0), law, None);
        let (_, backward) = contact_force((10.0, 0.0), (0.0, -1.0), law, None);
        assert!((forward + 1.0).abs() < 1e-12, "forward {forward}");
        assert!((backward - 4.0).abs() < 1e-12, "backward {backward}");
        for gaps in [(0.0, 1.0), (0.0, -1.0), (0.0, 0.05)] {
            let branch = contact_branch((10.0, 0.0), gaps, law);
            assert_eq!(
                contact_force((10.0, 0.0), gaps, law, Some(branch)),
                contact_force((10.0, 0.0), gaps, law, None)
            );
        }
        // The axis turned against the tangent swaps the senses; across it,
        // the source's H(0) = 1/2.
        assert_eq!(
            CoulombBounds::directional(0.1, 0.4, -1.0),
            bounds.reversed()
        );
        assert_eq!(
            CoulombBounds::directional(0.1, 0.4, 0.0),
            CoulombBounds::isotropic(0.25)
        );
    }
}
