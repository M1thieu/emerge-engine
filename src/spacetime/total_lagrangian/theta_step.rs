//! FAST_MODE_PLAN.md sec. 48, a test-only measurement: the step by Moreau and
//! Jean's theta method (Acary, INRIA RR-7151, 2010, eqs. 16a-e: positions
//! advanced by `v_k+theta`, forces at `k + theta`) instead of the conserving
//! midpoint, its high-frequency dissipation set by a STATED spectral radius
//! at infinity, `EMERGE_PROBE_RHO_INF = rho_inf`, `theta = 1 / (1 + rho_inf)`
//! (the linear one-degree-of-freedom amplification tends to `(1 - theta) /
//! theta`, derived in sec. 48.1). Default off: the midpoint, unchanged.
//!
//! On the step (one step, one interior point, as the midpoint's `theta =
//! 1/2`; not a fraction of the step):
//! - the inertia `M (u / dt - v_n) / (theta dt)` and the end velocity `(u /
//!   dt - (1 - theta) v_n) / theta`;
//! - the stress at `C_theta = (1 - theta) C_n + theta C_n+1`, carried by
//!   `F_theta = (1 - theta) F_n + theta F_n+1` (the midpoint's own convention
//!   at `theta = 1/2`, F4's at `theta = 1`);
//! - the relaxation branches by the same theta rule (`Relaxation::
//!   theta_stress`, `theta_committed`);
//! - the energy the scheme removes booked as heat (`hooks::book_scheme_heat`),
//!   per particle the step's energy residual.
//!
//! Adaptations, declared: the theta rule on the branches' internal variables
//! (Acary applies it to the mechanical state); the stress's tangent by
//! central differences in `C_theta`; solids with a plastic law or a dashpot
//! are refused (not measured).

// KNOWN-LIMIT[DISCRETIZATION/ARTIFACT]: a declared first-order theta step (theta = 1 / (1 + rho_inf)) dissipates every unresolved mode by rho_inf per step, the impact's included, so the rebound scales with rho_inf, a numerical property, not a material one | source: Acary RR-7151 eq. 16; FAST_MODE_PLAN.md sec. 48 | replace when: the impact is resolved (the user's ruling on an in-step event or a finer step at contact), or a law-based in-step closure passes its counts | state: measured only, default off | gap: rho_inf is stated, not derived; choosing it to match a rebound would be calibration

use glam::DMat2;

use super::elastic::Elastic;

/// The theta `EMERGE_PROBE_RHO_INF` states, if set; says so once.
pub(crate) fn theta() -> Option<f64> {
    static THETA: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *THETA.get_or_init(|| {
        let rho: f64 = std::env::var("EMERGE_PROBE_RHO_INF").ok()?.parse().ok()?;
        assert!(
            (0.0..=1.0).contains(&rho),
            "EMERGE_PROBE_RHO_INF is a spectral radius in [0, 1], not {rho}"
        );
        let theta = 1.0 / (1.0 + rho);
        println!("knob active: theta step, stated rho_inf {rho}, theta {theta}");
        Some(theta)
    })
}

/// `(1 - theta) a + theta b`.
pub(crate) fn at(a: DMat2, b: DMat2, theta: f64) -> DMat2 {
    a * (1.0 - theta) + b * theta
}

/// One particle's energy residual over a theta step, J/m: with `M (v_n+1 -
/// v_n) / dt` the force and `dt v_theta` the displacement, the inertia's work
/// is the kinetic change plus `(theta - 1/2) m |dv|^2`, and the stress's
/// work `V0 P_theta : dF` exceeds the stored change (branches included, at
/// their committed state) by what the branches dissipate plus the scheme's
/// own loss. Returns that loss (F4's at `theta = 1`).
pub(crate) fn residual(
    solid: &Elastic,
    (f0, g, dt, theta): (DMat2, DMat2, f64, f64),
    (v0, mass, dv2): (f64, f64, f64),
) -> f64 {
    let f1 = f0 + g;
    let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
    let first = at(f0, f1, theta) * solid.theta_stress(at(c0, c1, theta), dt, theta);
    let work = first.x_axis.dot(g.x_axis) + first.y_axis.dot(g.y_axis);
    let next = solid.committed(c0, c1, dt);
    v0 * (solid.energy(f0) - next.solid.energy(f1) + work - next.relaxed.heat)
        + (theta - 0.5) * mass * dv2
}

impl Elastic {
    /// The theta step's stress at `C_theta`: the law's `S(C_theta)` and the
    /// branches' by the theta rule.
    pub(crate) fn theta_stress(&self, c_theta: DMat2, dt: f64, theta: f64) -> DMat2 {
        assert!(
            !self.is_plastic() && !self.is_viscous(),
            "the theta step is measured on solids without a plastic law or a dashpot"
        );
        let s = self.second_pk(c_theta);
        if self.is_relaxing() {
            s + self
                .relaxation
                .theta_stress(c_theta, dt, theta, self.lame())
        } else {
            s
        }
    }

    /// `theta_stress`'s derivative in `C_theta` along `dc`, by central
    /// differences (the step `step_stress_tangent_by_differences` takes).
    pub(crate) fn theta_stress_tangent(
        &self,
        c_theta: DMat2,
        dt: f64,
        theta: f64,
        dc: DMat2,
    ) -> DMat2 {
        let size = |m: DMat2| {
            m.x_axis
                .x
                .abs()
                .max(m.x_axis.y.abs())
                .max(m.y_axis.x.abs())
                .max(m.y_axis.y.abs())
        };
        let along = size(dc);
        if along == 0.0 {
            return DMat2::ZERO;
        }
        let h = 1e-6 * size(c_theta).max(1.0) / along;
        (self.theta_stress(c_theta + dc * h, dt, theta)
            - self.theta_stress(c_theta - dc * h, dt, theta))
            * (0.5 / h)
    }
}
