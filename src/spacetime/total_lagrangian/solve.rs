//! One step's nonlinear solve: the residual (inertia, internal force, contact
//! push, weight) and Newton's method with a Jacobian-free GMRES.

use super::*;
use std::cell::Cell;
use std::time::Instant;

/// Adds the time since `start` to `clock` (`Timing`).
fn timed(clock: &Cell<f64>, start: Instant) {
    clock.set(clock.get() + start.elapsed().as_secs_f64());
}

/// Whether `EMERGE_PROBE_PUSH_CHECK` is set (read once): the exact plane
/// push derivative compared with central differences on every product. A
/// probe knob, in test builds only.
#[cfg(all(test, feature = "experimental"))]
pub(super) fn push_check() -> bool {
    static SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("EMERGE_PROBE_PUSH_CHECK").is_some())
}

/// A plane contact point's part of its contact's push along a direction
/// `d` (`plane_push_product`): its weight, normal, spread over particles,
/// and its spread's relative motion along `d`.
type PlaneShare = (f64, DVec2, [(usize, f64); 3], DVec2);

/// The grid velocity of the points' velocities `v`: the consistent-mass
/// projection `M_eps v_n = sum_p N(x_p) m_p v_p` (Love and Sulsky's eq. 28,
/// with the effective mass of their eq. 26), on the unknowns with some mass
/// (`Grid::massive`), by conjugate gradients preconditioned with the band
/// factor `factor` of `M_eps`. Returns `v_n` and the right-hand side.
pub(crate) fn grid_velocity(
    grid: &Grid,
    (mass, v): (&[f64], &[DVec2]),
    eps: f64,
    factor: &Band,
) -> (Vec<f64>, Vec<f64>) {
    let massive = grid.massive();
    let mut rhs = vec![0.0; grid.len];
    for (p, (&m, vp)) in mass.iter().zip(v.iter()).enumerate() {
        for a in 0..2 {
            let s = m * vp[a];
            grid.visit(p, a, |k, w, _| rhs[k] += w * s);
        }
    }
    for (value, &on) in rhs.iter_mut().zip(&massive) {
        if !on {
            *value = 0.0;
        }
    }
    let v_n = preconditioned_cg(
        &|y: &[f64]| grid.mass_times(y, mass, eps),
        &rhs,
        &|r: &[f64]| factor.solve(r),
        &massive,
    );
    (v_n, rhs)
}

/// One step's solve on a grid with its current active set: the grid's
/// change `u`, its velocity before `v_n` and the report.
pub(crate) struct Solution {
    pub(crate) u: Vec<f64>,
    pub(crate) v_n: Vec<f64>,
    pub(crate) report: StepReport,
    /// With `Inputs::responses`, the converged step's displacement per
    /// unit of each active multiplier (`Response`); empty otherwise.
    pub(crate) responses: Vec<Response>,
    /// The energy each particle's dashpot dissipated over the step, `V_0
    /// dt D(E_dot)` (`Elastic::viscous_dissipation_over`), at the step's
    /// solution; empty when no solid is viscous.
    pub(crate) dissipated: Vec<f64>,
    /// The friction's augmentation parameter the solve used
    /// (`tangential_penalty`); `None` where each contact's own penalty
    /// served.
    pub(crate) tangential_penalty: Option<f64>,
}

/// How the converged step's displacement answers one active multiplier:
/// `du = J^-1 b`, `b` the push one unit of contact `contact`'s normal (or,
/// sticking, tangential) multiplier adds, `J` Newton's Jacobian with the
/// contact sets frozen at the solution. The derivative of the end gaps in
/// that multiplier, for a semismooth Newton step on the multipliers
/// (`step_total_lagrangian_contact`).
pub(crate) struct Response {
    pub(crate) contact: usize,
    pub(crate) tangential: bool,
    pub(crate) du: Vec<f64>,
}

/// What a solve reads besides the grid.
pub(crate) struct Inputs<'a> {
    pub(crate) x: &'a [DVec2],
    pub(crate) v: &'a [DVec2],
    pub(crate) f: &'a [DMat2],
    pub(crate) mass: &'a [f64],
    pub(crate) v0: &'a [f64],
    pub(crate) solids: &'a [Elastic],
    pub(crate) gravity: DVec2,
    pub(crate) dt: f64,
    pub(crate) settings: &'a ConservingSettings,
    /// The walls' planes and each particle's penalty for holding it behind
    /// them (`step`).
    pub(crate) planes: &'a [Plane],
    pub(crate) penalty: &'a [f64],
    /// The grid's basis is the reference configuration's and `F` grows by
    /// the displacement's material gradient (`step_total_lagrangian`), not
    /// the step's start, where `F` is multiplied by `I + grad_n u`.
    pub(crate) lagrangian: bool,
    /// Each particle's body force per unit mass, cells/s^2, at the step's
    /// midpoint, in place of the uniform `gravity` (the probes'
    /// manufactured solutions); `None`, the gravity.
    pub(crate) body: Option<&'a [DVec2]>,
    /// The contacts' Coulomb friction coefficient (`contact_force`).
    pub(crate) friction: f64,
    /// Whether the converged solve also returns its `Response`s.
    pub(crate) responses: bool,
    /// The friction's augmentation parameter held below Alart's bound
    /// (`tangential_penalty`) instead of each contact's own penalty: set
    /// for the re-solve of a frictional solve that did not converge.
    pub(crate) bounded_friction: bool,
    /// The static mode of the same solve (an initial-condition tool, not a
    /// law: `settle_total_lagrangian_at_rest`): the inertia's coefficient 0
    /// and `v = 0`, the stress the law's own at the end state through `F_1`
    /// (`Elastic::static_stress`) with its matching tangent, so the residual
    /// is `f_int(u) - f_ext + f_contact(u)`, the balance at rest. Every probe
    /// path that changes the force outside the law's stress refuses it.
    pub(crate) statics: bool,
    /// In the static mode, the axis of a free rigid translation the load has
    /// no component on (a body on frictionless walls): its neutral motion is
    /// fixed by a gauge, the body's mass-weighted mean displacement along it
    /// held at zero (`solve`). `None` everywhere else.
    pub(crate) gauge: Option<usize>,
}

/// The band factors a step's solves share across the walls' passes
/// (`step`) while the grid stays the same, and the last responses: all are
/// dropped when a face is let go or held again, and the Jacobian's factor
/// is rebuilt when the held particles change, as their penalties are in
/// it.
#[derive(Clone, Default)]
pub(crate) struct Factors {
    pub(crate) mass: Option<Band>,
    /// The Jacobian's band factors by the system each factors, its
    /// inertia coefficient `2 / dt^2` and its held springs, with the
    /// `clock` reading of its last use, the most recent first: one per try
    /// length the step controller makes in a step (`JACOBIAN_FACTORS`), so
    /// a run of equal steps factors each once instead of at every try
    /// (`solve`).
    pub(crate) jacobian: Vec<(f64, Vec<Spring>, Band, u64)>,
    /// Counts the factors' uses (`solve`), so that `absorb` keeps the
    /// most recently used whatever cache they come from.
    pub(crate) clock: u64,
    /// The last responses computed, by contact key and direction: each
    /// next response's GMRES starts from its own (`gmres_from`).
    pub(crate) responses: Vec<(ContactKey, bool, Vec<f64>)>,
    /// The parts of a solve that depend on the step's start alone
    /// (`StartParts`), kept over one step's contact passes: `None` outside
    /// them (nothing kept), `Some(None)` at their start (`step.rs` sets it,
    /// the first pass's solve fills it), then reused by the later passes.
    pub(crate) start: Option<Option<StartParts>>,
}

/// What a solve computes from the step's start alone: the grid velocity
/// and its right-hand side, the projection's residual, the external force,
/// the residual scale's inertial and internal terms, and the round-off
/// floor. The contact passes of one step solve with the same start (`x`,
/// `v`, `f`, the masses, the solids, `dt`) and change only the contacts,
/// their multipliers and the friction's bound, which none of these read,
/// so a later pass reuses them as computed: the same numbers, bit for bit.
#[derive(Clone)]
pub(crate) struct StartParts {
    v_n: Vec<f64>,
    rhs: Vec<f64>,
    projection: f64,
    external: Vec<f64>,
    inertial: f64,
    internal: f64,
    floor: f64,
    force_round_off: f64,
}

impl Factors {
    /// Takes in `other`'s Jacobian factors, keeping the `JACOBIAN_FACTORS`
    /// most recently used of both: the step controller keeps one try's
    /// cache, and the other tries' factors (their own lengths') would
    /// otherwise be lost and rebuilt at the next try. Every try of a step
    /// starts from the step's cache, so both caches' clocks start from the
    /// same reading and a factor used in any of the step's tries is more
    /// recent than one none of them used; those are the ones evicted
    /// (measured before: a full cache of factors no try used, with the
    /// impact's springs, kept a block's whole steps building a factor at
    /// every try, `probe_factor_builds_traced`).
    pub(crate) fn absorb(&mut self, other: &Factors) {
        for (c, springs, band, used) in &other.jacobian {
            match self
                .jacobian
                .iter_mut()
                .find(|(mine, held, _, _)| mine == c && held == springs)
            {
                Some(entry) => entry.3 = entry.3.max(*used),
                None => self
                    .jacobian
                    .push((*c, springs.clone(), band.clone(), *used)),
            }
        }
        self.clock = self.clock.max(other.clock);
        // Stable: on a tie this cache's own entries stay first.
        self.jacobian
            .sort_by_key(|entry| std::cmp::Reverse(entry.3));
        self.jacobian.truncate(JACOBIAN_FACTORS);
    }
}

/// How many Jacobian factors `Factors` keeps: the step controller's try
/// lengths per step (the whole step, its halves and its thirds,
/// `StepControl::substeps`), times the two held sets each try's passes
/// solve with (none held at the first pass, then the contacts held). With
/// three, a block at rest evicted and rebuilt two factors a frame
/// (measured, `hard_frame::probe_rest_factorizations`).
const JACOBIAN_FACTORS: usize = 2 * KLAPPROTH_SUBSTEPS.len();

/// The step of a central difference of the forces along a direction of
/// largest entry `size`, from displacements of largest entry `base`, on a
/// grid of cell `cell`: `eps^(1/3) (cell + base) / size`, relative to the
/// displacements' scale (one cell when they are zero).
/// The relative step `eps^(1/3)` balances the central difference's two
/// errors (Walter, Numerical Methods and Optimization (2014), Sec. 6.6.1,
/// eqs. 6.101-6.102: method error O(dx^2), "a compromise between the
/// rounding and method errors", no value given). Derived here, declared:
/// with `f` evaluated to a relative rounding error `eps`, the difference
/// `(f(x + h) - f(x - h)) / 2h` carries a rounding error of about
/// `eps |f| / h` and a method error `h^2 |f3| / 6` (`f3` the third
/// derivative); their sum is least at `h^3 = 3 eps |f| / |f3|`, i.e. `h =
/// eps^(1/3)` times a scale `(3 |f| / |f3|)^(1/3)` of the argument.
/// Adaptation, declared: that scale is taken as the displacements' own
/// scale `(cell + base)` (the factor `3^(1/3)` and the ratio `f / f3` are
/// not known per direction). Where the function is linear (a gap against a
/// plane) the method error is zero and any step is exact up to rounding.
pub(crate) fn central_difference_step(cell: f64, base: f64, size: f64) -> f64 {
    f64::EPSILON.cbrt() * (cell + base) / size
}

/// A held contact's penalty in the preconditioner
/// (`Grid::approximate_jacobian`).
#[derive(Clone, Debug)]
pub(crate) enum Spring {
    /// A particle held along one lattice: particle, lattice, stiffness.
    Ground(usize, usize, f64),
    /// A body-body condition's penalty on the relative displacement along
    /// `c`: `stiffness (sum_r c_r . u_r)^2`, its members `(particle, c_r)`
    /// (the point and the segment's two ends): the true tangent's coupling
    /// between the two bodies. Kept whole or not at all
    /// (`Grid::approximate_jacobian`): where the band cannot hold every
    /// entry, its ground springs (the third field) stand in, as before.
    Couple(Vec<(usize, DVec2)>, f64, Vec<(usize, usize, f64)>),
}

/// Two springs are the same factor's springs when they hold the same
/// particles at the same stiffness: a coupling's direction moves with the
/// bodies from step to step, and a factor built for a nearby direction is
/// still a preconditioner (it changes GMRES's iterations, never the
/// converged answer), so the factor is reused as before.
impl PartialEq for Spring {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ground(p, a, k), Self::Ground(q, b, l)) => p == q && a == b && k == l,
            (Self::Couple(m, k, _), Self::Couple(n, l, _)) => {
                k == l && m.len() == n.len() && m.iter().zip(n).all(|(x, y)| x.0 == y.0)
            }
            _ => false,
        }
    }
}

/// One particle's linearisation at a Newton iterate: its end deformation
/// `F_{n+1}`, its step stress, and that stress's derivative in `C_{n+1}`
/// along `E_11`, `E_22` and `E_12 + E_21` (`Elastic::step_stress_tangent`).
#[derive(Clone, Copy, Default)]
struct Linearised {
    f1: DMat2,
    s: DMat2,
    ds: [DMat2; 3],
    /// The hoop's part on the axisymmetric path.
    #[cfg(all(test, feature = "experimental"))]
    hoop: Option<super::axisymmetric::HoopLine>,
}

/// The step's nonlinear solve on `grid` (`step`).
pub(crate) fn solve(
    grid: &Grid,
    inputs: &Inputs<'_>,
    contacts: &[Contact],
    factors: &mut Factors,
) -> Solution {
    #[cfg(all(test, feature = "experimental"))]
    let solve_clock = Instant::now();
    let Inputs {
        x,
        v,
        f,
        mass,
        v0,
        solids,
        gravity,
        dt,
        settings,
        planes,
        penalty,
        lagrangian,
        body,
        friction,
        responses: wants_responses,
        bounded_friction,
        statics,
        gauge: gauge_axis,
    } = *inputs;
    assert!(
        statics || gauge_axis.is_none(),
        "the gauge belongs to the static mode"
    );
    let clock = std::time::Instant::now();
    #[cfg(all(test, feature = "experimental"))]
    hooks::begin_solve();
    // The static mode runs on the law's own stress alone: a probe that
    // changes the force or the load outside it would be ignored silently,
    // so each is refused instead.
    #[cfg(all(test, feature = "experimental"))]
    assert!(
        !statics
            || (grid.hoop.is_none()
                && hooks::cells().is_none()
                && hooks::viscosity() == 0.0
                && !hooks::trapezoid_p()
                && !hooks::cofactor()
                && !hooks::split()
                && !hooks::trapezoid()
                && hooks::homotopy() == 0
                && !hooks::lobatto_wanted()
                && !super::residual_split::on()
                && body.is_none()),
        "the static mode refuses the probe paths (hoop, mixed pressures, the viscosity hook, the stress variants, the homotopy, the estimate, manufactured loads)"
    );
    let n = x.len();
    let eps = settings.epsilon;
    // The midpoint's inertia, `2 M (u / dt - v_n) / dt`; none at rest.
    let inertia = if statics { 0.0 } else { 2.0 };
    // The grid velocity: the consistent-mass projection (eq. 28); at rest,
    // zero.
    // The largest entry, NaN when any is (`f64::max` would drop it and a
    // broken step would read as converged).
    let largest = |y: &[f64]| {
        y.iter().fold(0.0f64, |m, e| {
            if e.is_nan() || m.is_nan() {
                f64::NAN
            } else {
                m.max(e.abs())
            }
        })
    };
    // The step's start parts, if a previous pass of this step computed
    // them (`Factors::start`).
    let reused: Option<StartParts> = factors.start.as_ref().and_then(Clone::clone);
    let factor = factors
        .mass
        .get_or_insert_with(|| grid.mass_band(mass, eps));
    let (v_n, rhs) = match &reused {
        Some(kept) => (kept.v_n.clone(), kept.rhs.clone()),
        None if statics => (vec![0.0; grid.len], vec![0.0; grid.len]),
        None => grid_velocity(grid, (mass, v), eps, factor),
    };
    // The friction's augmentation parameter: each contact's own penalty,
    // or, on the re-solve of a solve that did not converge, the bound
    // (`tangential_penalty`, `Inputs::bounded_friction`).
    // The bound is inertial (`tangential_penalty`): it has no meaning at
    // rest, so the static mode never asks for it.
    assert!(
        !(statics && bounded_friction),
        "the static mode takes each contact's own friction penalty (the bound is inertial)"
    );
    let tangential = if bounded_friction && friction > 0.0 && !contacts.is_empty() {
        tangential_penalty(grid, x, planes, contacts, factor, dt, inertia)
    } else {
        None
    };
    let k_t_of = |c: &Contact| tangential.unwrap_or_else(|| contact_penalty(c, penalty));
    // `M v_n`, read by the projection's residual and the residual scale's
    // inertial term.
    let (projection, inertial) = match &reused {
        Some(kept) => (kept.projection, kept.inertial),
        None => {
            let projected = grid.mass_times(&v_n, mass, eps);
            let projection = (projected
                .iter()
                .zip(&rhs)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / rhs
                    .iter()
                    .map(|b| b * b)
                    .sum::<f64>()
                    .max(f64::MIN_POSITIVE))
            .sqrt();
            (projection, largest(&projected))
        }
    };
    let external: Vec<f64> = match &reused {
        Some(kept) => kept.external.clone(),
        None => {
            let external = match body {
                Some(body) => parallel_scatter(grid.len, n, |p, acc| {
                    for (a, component) in body[p].to_array().into_iter().enumerate() {
                        let s = mass[p] * component;
                        grid.visit(p, a, |k, w, _| acc[k] += w * s);
                    }
                }),
                None => (0..grid.len)
                    .map(|k| grid.lumped[k] * gravity[grid.axis_of[k]])
                    .collect(),
            };
            // The axisymmetric cases' dead loads.
            #[cfg(all(test, feature = "experimental"))]
            let external = super::axisymmetric::with_point_loads(grid, external);
            external
        }
    };
    // The static mode's gauge (`Inputs::gauge`): along a free translation
    // the load has no component on, every displaced copy of an equilibrium
    // is one, and the static Jacobian is singular there; the motion is fixed
    // by `c(u) = w . u = 0`, `w_k = lumped_k / M` on that axis (the body's
    // mass-weighted mean displacement: the lumped mass is the particles' mass
    // carried by the basis, which sums to one there), added as `kappa c(u)
    // w` to the residual. The internal forces sum to zero along a
    // translation (the basis's partition of unity) and the load and the
    // frictionless walls' push have no component on it, so at the solution
    // the gauge carries no force and the state does not depend on `kappa`:
    // an iteration parameter, here the sum of the particles' own stiffness
    // scales `(K + mu) V0 / dx^2`, the springs the preconditioner gets for it.
    let gauge: Option<(usize, Vec<f64>, f64)> = gauge_axis.map(|a| {
        let on = |k: usize| grid.axis_of[k] == a && grid.active[k];
        let total: f64 = (0..grid.len)
            .filter(|&k| on(k))
            .map(|k| grid.lumped[k])
            .sum();
        let w: Vec<f64> = (0..grid.len)
            .map(|k| {
                if on(k) {
                    grid.lumped[k] / total.max(f64::MIN_POSITIVE)
                } else {
                    0.0
                }
            })
            .collect();
        let kappa: f64 = (0..n)
            .filter(|&p| mass[p] > 0.0)
            .map(|p| (solids[p].bulk + solids[p].mu) * v0[p] / (grid.cell * grid.cell))
            .sum();
        (a, w, kappa)
    });
    let gauge_force = |y: &[f64], out: &mut [f64]| {
        if let Some((_, w, kappa)) = &gauge {
            let c: f64 = w.iter().zip(y).map(|(a, b)| a * b).sum();
            for (o, wk) in out.iter_mut().zip(w) {
                *o += kappa * c * wk;
            }
        }
    };
    let f_now: &[DMat2] = f;
    let deformation_at = |u: &[f64], p: usize| -> DMat2 {
        let (gu, gv) = (grid.gradient(u, p, 0), grid.gradient(u, p, 1));
        let grad = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
        if lagrangian {
            f_now[p] + grad
        } else {
            (DMat2::IDENTITY + grad) * f_now[p]
        }
    };
    // The axisymmetric path's hoop, when set.
    #[cfg(all(test, feature = "experimental"))]
    let hoop = grid.hoop.as_deref();
    #[cfg(all(test, feature = "experimental"))]
    assert!(
        hoop.is_none() || lagrangian,
        "the axisymmetric path is the total Lagrangian midpoint only"
    );
    // The internal force (eq. 30a) with the algorithmic stress. On the
    // reference basis `F_n^T grad_n N` is `grad_X N` itself.
    // The internal force's evaluations (`Timing::internal_evaluations`).
    let evaluations = Cell::new(0u64);
    // The last internal force's iterate, bit for bit, with each point's
    // `F_n+1` and the law's stress there (before any probe's variant):
    // Newton linearises at the trial it has just accepted, whose force it
    // evaluated, so `linearised_at` reads these instead of computing the
    // same values again.
    // The iterate and, per point, the first Piola stress, `F_n+1` and the law's stress.
    type Kept = Option<(Vec<f64>, Vec<(DMat2, DMat2, DMat2)>)>;
    let last_stresses: std::cell::RefCell<Kept> = std::cell::RefCell::new(None);
    let internal = |u: &[f64]| -> Vec<f64> {
        evaluations.set(evaluations.get() + 1);
        #[cfg(all(test, feature = "experimental"))]
        if super::residual_split::on() && lagrangian {
            let at = super::residual_split::Sample {
                grid,
                f_now,
                v0,
                solids,
                dt,
            };
            super::residual_split::sample(evaluations.get(), &at, |p| deformation_at(u, p));
        }
        // With a `Theta` constant per cell: each cell's pressure, the secant
        // of `U` between its mean `J` at the step's ends.
        #[cfg(all(test, feature = "experimental"))]
        let pressures = hooks::cells().filter(|_| lagrangian).map(|cells| {
            let (of, count) = (&cells.0, cells.1);
            let mut sums = vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64); count];
            for p in 0..n {
                let cell = &mut sums[of[p]];
                cell.0 += v0[p];
                cell.1 += v0[p] * f_now[p].determinant();
                cell.2 += v0[p] * deformation_at(u, p).determinant();
                cell.3 += v0[p] * solids[p].bulk;
            }
            let pressure: Vec<f64> = sums
                .iter()
                .map(|&(volume, j0, j1, bulk)| {
                    if volume > 0.0 {
                        bulk / volume * (0.5 * (j0 + j1) / volume - 1.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            (cells, pressure)
        });
        #[cfg(all(test, feature = "experimental"))]
        if let Some(hoop) = hoop {
            return parallel_scatter(grid.len, n, |p, acc| {
                super::axisymmetric::hoop_force(
                    grid,
                    hoop,
                    (p, u, x[p]),
                    (v0[p], &solids[p]),
                    (f_now[p], deformation_at(u, p), dt),
                    acc,
                );
            });
        }
        // Each point's first Piola stress in a parallel map, its order kept,
        // then scattered in the same chunks and order as before: the same
        // expressions on the same inputs, the same sums, so the force is
        // bitwise the serial one (one chunk
        // holds 64 points, so a small body's law ran on one thread).
        let first_at = |p: usize| -> (DMat2, DMat2, DMat2) {
            let f1 = deformation_at(u, p);
            let f0 = f_now[p];
            let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
            // The law's step stress and the Kelvin-Voigt dashpot's in
            // parallel, on the same `F_n+1/2` (`Elastic::step_stress`); at
            // rest, the law's own stress at the end state, through `F_1`.
            let (s, carrier) = if statics {
                if v0[p] <= 0.0 {
                    return (DMat2::ZERO, DMat2::ZERO, DMat2::ZERO);
                }
                (solids[p].static_stress(c1), f1)
            } else {
                // The stress acts through `F_n+1/2`.
                (solids[p].step_stress(c0, c1, dt), f0 * 0.5 + f1 * 0.5)
            };
            let law_stress = s;
            #[cfg(all(test, feature = "experimental"))]
            let s = match &pressures {
                Some((cells, pressure)) => solids[p].mixed_stress(
                    f0.transpose() * f0,
                    f1.transpose() * f1,
                    pressure[cells.0[p]],
                    hooks::trapezoid(),
                ),
                None => s,
            };
            let first = if lagrangian {
                v0[p] * carrier * s
            } else {
                v0[p] * carrier * s * f0.transpose()
            };
            // The viscous stress at the step's END velocity, `2 u / dt - v_n`
            // on the grid (TL), times `F_n+1/2`: the force acts on the
            // sign-flipping velocity the midpoint average cannot see.
            #[cfg(all(test, feature = "experimental"))]
            let first = if lagrangian && hooks::viscosity() > 0.0 {
                let end: [DVec2; 2] = [0, 1].map(|a| {
                    let (_, gu) = grid.read(u, p, a);
                    let (_, gv) = grid.read(&v_n, p, a);
                    2.0 * gu / dt - gv
                });
                let f_dot = DMat2::from_cols(
                    DVec2::new(end[0].x, end[1].x),
                    DVec2::new(end[0].y, end[1].y),
                );
                first
                    + v0[p]
                        * (0.5 * (f0 + f1))
                        * solids[p].viscous_stress(f1, f_dot, hooks::viscosity())
            } else {
                first
            };
            #[cfg(all(test, feature = "experimental"))]
            let first = if lagrangian && hooks::trapezoid_p() {
                v0[p] * solids[p].trapezoid_first_pk(f0, f1)
            } else if lagrangian && hooks::cofactor() {
                v0[p] * solids[p].cofactor_volumetric(f0, f1, hooks::trapezoid())
            } else {
                first
            };
            (first, f1, law_stress)
        };
        let firsts: Vec<(DMat2, DMat2, DMat2)> = (0..n).into_par_iter().map(first_at).collect();
        let force = parallel_scatter(grid.len, n, |p, acc| {
            for a in 0..2 {
                let row = firsts[p].0.row(a);
                grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
            }
        });
        *last_stresses.borrow_mut() = Some((u.to_vec(), firsts));
        force
    };
    // The walls' push on the particles held on their planes (`step`),
    // `max(0, lambda - k g)` along the inward normal, `g` the particle's
    // distance to the plane at the step's end.
    // The push `max(0, lambda - k g) n` spread over its particles; with
    // `frozen`, the given contacts' `lambda - k g` without the clamp and
    // none on the others (the semismooth Newton's frozen active set).
    // `bump`: one contact's normal (or tangential) multiplier raised by the
    // given amount, for the push's derivative in it (`Response`).
    let push_with = |u: &[f64],
                     frozen: Option<&[ContactBranch]>,
                     bump: Option<(usize, bool, f64)>|
     -> Vec<f64> {
        let mut out = vec![0.0; grid.len];
        for (i, c) in contacts.iter().enumerate() {
            // A frozen set's free contact pushes nothing (`contact_force`):
            // its geometry is not needed.
            if frozen.is_some_and(|set| !set[i].pressed) {
                continue;
            }
            let (mut multiplier, mut tangential) = (c.multiplier, c.tangential);
            match bump {
                Some((j, false, amount)) if j == i => multiplier += amount,
                Some((j, true, amount)) if j == i => tangential += amount,
                _ => {}
            }
            let (g, shares) = contact_geometry(grid, x, planes, u, c);
            let g_t = if friction > 0.0 {
                contact_tangential_gap(grid, u, &shares)
            } else {
                0.0
            };
            let k = contact_penalty(c, penalty);
            let (size, along) = contact_force(
                (multiplier, tangential),
                (g, g_t),
                ((k, k_t_of(c)), friction),
                frozen.map(|set| set[i]),
            );
            if size == 0.0 && along == 0.0 {
                continue;
            }
            for share in &shares {
                let force = share.weight * (size * share.normal + along * share.normal.perp());
                // The spread's three parts, then the friction couple's two
                // when it acts: a fixed array in the same order (no
                // allocation per share).
                let mut parts = [(0usize, DVec2::ZERO); 5];
                parts[0] = (share.spread[0].0, share.spread[0].1 * force);
                parts[1] = (share.spread[1].0, share.spread[1].1 * force);
                parts[2] = (share.spread[2].0, share.spread[2].1 * force);
                let mut used = 3;
                if along != 0.0
                    && let Some((first, second, couple)) =
                        friction_couple(grid, x, u, &share.point, share.normal, &share.spread)
                {
                    let turn = share.weight * couple * along * share.normal;
                    parts[3] = (second, turn);
                    parts[4] = (first, -turn);
                    used = 5;
                }
                for &(r, part) in &parts[..used] {
                    for a in 0..2 {
                        if part[a] != 0.0 {
                            grid.visit(r, a, |k, w, _| out[k] += w * part[a]);
                        }
                    }
                }
            }
        }
        out
    };
    let push = |u: &[f64]| push_with(u, None, None);
    // The internal force less the walls' push: what the solve linearises.
    let forces = |u: &[f64]| -> Vec<f64> {
        let mut out = internal(u);
        if !contacts.is_empty() {
            out.iter_mut().zip(push(u)).for_each(|(o, p)| *o -= p);
        }
        out
    };
    // The residual on every unknown, held ones included: there, the force
    // the wall exerts.
    let raw = |u: &[f64]| -> Vec<f64> {
        let accel: Vec<f64> = (0..grid.len)
            .map(|k| inertia * (u[k] / dt - v_n[k]) / dt)
            .collect();
        let mut out = grid.mass_times_unmasked(&accel, mass, eps);
        out.iter_mut().zip(forces(u)).for_each(|(o, i)| *o += i);
        out.iter_mut().zip(&external).for_each(|(o, e)| *o -= e);
        gauge_force(u, &mut out);
        out
    };
    // The parts' clocks (`Timing`), shared by the closures below.
    let (residual_clock, jacobian_clock, precondition_clock) =
        (Cell::new(0.0), Cell::new(0.0), Cell::new(0.0));
    let residual = |u: &[f64]| -> Vec<f64> {
        #[cfg(all(test, feature = "experimental"))]
        hooks::add_part(6, 1.0);
        let start = Instant::now();
        let mut out = raw(u);
        for (value, &on) in out.iter_mut().zip(&grid.active) {
            if !on {
                *value = 0.0;
            }
        }
        timed(&residual_clock, start);
        out
    };
    // The preconditioner: `2 M_eps / dt^2 + K`, `K` the linear elastic
    // stiffness at the reference shape, factored exactly in band form once
    // per step (`Band`). This engine's choice; the source's solves used
    // PETSc's matrix-free GMRES (their section 4).
    // A pair's penalty couples its two particles; the preconditioner keeps
    // only each particle's own block of it. A body-body condition's penalty
    // is kept as the coupling the true tangent has, on the relative
    // displacement along the segment's normal (and, with friction, its
    // tangent): `k n (u_p - (1 - xi) u_first - xi u_second)`, the segment's
    // normal and the point's projection `xi` at the step's start positions
    // (a direction for the preconditioner only, declared). Before, each of
    // the three particles was held to ground on both axes, the strongest
    // coupling in the system (the penalty between the bodies) left out:
    // Saad, Iterative Methods for Sparse Linear Systems (2003), ch. 10, the
    // preconditioned matrix is near the identity only when the factors'
    // errors are small; Stueben, A review of algebraic multigrid (2001),
    // the two-level theorem's comment: the strong couplings are the ones
    // that matter for convergence.
    let springs: Vec<Spring> = contacts
        .iter()
        .flat_map(|c| -> Vec<Spring> {
            let (k, k_t) = (contact_penalty(c, penalty), k_t_of(c));
            c.points
                .iter()
                .flat_map(|&(point, weight)| -> Vec<Spring> {
                    let (k, k_t) = (k * weight * weight, k_t * weight * weight);
                    match point.segment {
                        Some((first, second)) => {
                            let side = x[second] - x[first];
                            let length = side.length().max(f64::MIN_POSITIVE);
                            let along = side / length;
                            let xi = ((x[point.particle] - x[first]).dot(side) / (length * length))
                                .clamp(0.0, 1.0);
                            // Without the band's room for the coupling, the
                            // three particles held to ground on both axes, the
                            // springs of before; the friction's coupling has
                            // none (none before).
                            let ground: Vec<(usize, usize, f64)> = [point.particle, first, second]
                                .into_iter()
                                .flat_map(|r| [(r, 0, k), (r, 1, k)])
                                .collect();
                            let couple =
                                |d: DVec2, stiffness: f64, fallback: Vec<(usize, usize, f64)>| {
                                    Spring::Couple(
                                        vec![
                                            (point.particle, d),
                                            (first, -(1.0 - xi) * d),
                                            (second, -xi * d),
                                        ],
                                        stiffness,
                                        fallback,
                                    )
                                };
                            let mut held = vec![couple(along.perp(), k, ground)];
                            if friction > 0.0 {
                                held.push(couple(along, k_t, Vec::new()));
                            }
                            held
                        }
                        // With friction both components are held: the
                        // plane's axis by the normal penalty, the other by
                        // the tangential one.
                        None if friction > 0.0 => {
                            let axis = planes[point.plane].axis;
                            std::iter::once(point.particle)
                                .chain(point.partner)
                                .flat_map(|r| {
                                    [Spring::Ground(r, axis, k), Spring::Ground(r, 1 - axis, k_t)]
                                })
                                .collect()
                        }
                        None => {
                            let axis = planes[point.plane].axis;
                            std::iter::once(Spring::Ground(point.particle, axis, k))
                                .chain(point.partner.map(|q| Spring::Ground(q, axis, k)))
                                .collect()
                        }
                    }
                })
                .collect()
        })
        .collect();
    // The gauge's banded stand-in in the preconditioner: each particle's own
    // stiffness scale as a spring along the gauged axis.
    let springs: Vec<Spring> = match &gauge {
        Some((a, _, _)) => springs
            .into_iter()
            .chain((0..n).filter(|&p| mass[p] > 0.0).map(|p| {
                Spring::Ground(
                    p,
                    *a,
                    (solids[p].bulk + solids[p].mu) * v0[p] / (grid.cell * grid.cell),
                )
            }))
            .collect(),
        None => springs,
    };
    let projected_at = clock.elapsed().as_secs_f64();
    // This system's own factor, kept if a recent step built it: a
    // preconditioner changes the iterations to the solution, never the
    // solution (Chen and Chacon eqs. 15-16).
    //
    // Or a kept factor of a coefficient `c'` within `rho = 4` of this one's
    // `c`, the nearest. Derived on the page (elementary, this engine's own):
    // the band is `P(c) = c M_b + K_0` with `M_b` definite and `K_0`
    // semidefinite, so `x^T P(c) x / x^T P(c') x = (c m + k) / (c' m + k)` lies
    // between `min(1, c/c')` and `max(1, c/c')`, and `kappa(P(c')^-1 P(c)) <=
    // max(c/c', c'/c)`; with the step's tangent `A`, `kappa(P(c')^-1 A) <=
    // max(c/c', c'/c) kappa(P(c)^-1 A)`. `rho = 4` is the controller's own
    // structure (a step and its halves, `step_control`): one factor per
    // accepted step size. What it costs in GMRES iterations is measured, not
    // bounded here.
    let coefficient = inertia / (dt * dt);
    let mut factorizations = 0;
    // The probes' replay (`hooks::watch_gmres_replay`): this system's own
    // exact factor, kept below only if the one in use is reused.
    #[cfg(all(test, feature = "experimental"))]
    let exact_candidate = hooks::gmres_replay()
        .then(|| grid.approximate_jacobian((mass, v0, solids), eps, coefficient, &springs));
    let exact = factors
        .jacobian
        .iter()
        .position(|(c, built, _, _)| *c == coefficient && *built == springs);
    let near = || {
        factors
            .jacobian
            .iter()
            .enumerate()
            .filter(|(_, (_, built, _, _))| *built == springs)
            .map(|(i, (c, _, _, _))| (i, (c / coefficient).max(coefficient / c)))
            .filter(|&(_, ratio)| ratio <= 4.0 * (1.0 + 4.0 * f64::EPSILON))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    };
    match exact.or_else(near) {
        Some(0) => {}
        Some(found) => {
            let entry = factors.jacobian.remove(found);
            factors.jacobian.insert(0, entry);
        }
        None => {
            #[cfg(all(test, feature = "experimental"))]
            if hooks::control_trace() {
                let same_springs = factors.jacobian.iter().any(|(_, b, _, _)| *b == springs);
                let same_coefficient = factors
                    .jacobian
                    .iter()
                    .any(|(c, _, _, _)| *c == coefficient);
                let nearest = factors
                    .jacobian
                    .iter()
                    .map(|(c, _, _, _)| (c / coefficient).max(coefficient / c))
                    .fold(f64::INFINITY, f64::min);
                println!(
                    "  factor: coefficient {coefficient:.6e} (seen {same_coefficient}), springs {} (seen {same_springs}), kept {}, nearest kept ratio {nearest:.3e}",
                    springs.len(),
                    factors.jacobian.len()
                );
            }
            let band = grid.approximate_jacobian((mass, v0, solids), eps, coefficient, &springs);
            #[cfg(all(test, feature = "experimental"))]
            hooks::record_factor(coefficient);
            factorizations += 1;
            factors.jacobian.insert(0, (coefficient, springs, band, 0));
            factors.jacobian.truncate(JACOBIAN_FACTORS);
        }
    }
    factors.clock += 1;
    factors.jacobian[0].3 = factors.clock;
    let band = &factors.jacobian[0].2;
    #[cfg(all(test, feature = "experimental"))]
    let exact_band = exact_candidate.filter(|_| factors.jacobian[0].0 != coefficient);
    let banded_at = clock.elapsed().as_secs_f64();
    // No particle inverted: the iterates the line search may take. Li et
    // al. 2020 (section 4.4, after Smith and Schaefer 2015) bound the step
    // by the largest size before an inversion; here the step is halved
    // until none inverts, a simpler search of this engine's.
    // Only matter counts: a point of no volume (a contact outline's) carries
    // no material, and its `F`, read where the displacement is extrapolated
    // past the body, may invert without anything inverting (measured: a
    // sliding block's side and top outline points, inverted at `det F =
    // -0.25`, froze every line search of the step).
    // On the axisymmetric path a point must also stay off the axis: `J = det
    // F F_33`, with `F_33 = r / R`.
    #[cfg(all(test, feature = "experimental"))]
    let through_axis = |u: &[f64], p: usize| {
        hoop.is_some_and(|h| super::axisymmetric::through_axis(grid, h, (p, u, x[p])))
    };
    #[cfg(not(all(test, feature = "experimental")))]
    let through_axis = |_: &[f64], _: usize| false;
    let admissible = |u: &[f64]| {
        (0..n).all(|p| {
            v0[p] <= 0.0 || (deformation_at(u, p).determinant() > 0.0 && !through_axis(u, p))
        })
    };
    let mut u: Vec<f64> = (0..grid.len)
        .map(|k| if grid.active[k] { dt * v_n[k] } else { 0.0 })
        .collect();
    let mut report = StepReport {
        projection,
        ..Default::default()
    };
    report.timing.factorizations = factorizations;
    let mut r = residual(&u);
    // The residual's scale: the forces of the step's own physics, its
    // inertia, its weight and the stress it starts with. Not the residual
    // of a start: a free flight far from the solution made that scale so
    // large that a step still far from balance read as converged (measured:
    // `u . R` at -5.8e5 against 1.3e4 of kinetic energy change, the energy
    // falling 86 percent in that one step).
    let internal_start = match &reused {
        Some(kept) => kept.internal,
        None => largest(&internal(&vec![0.0; grid.len])),
    };
    let scale = (inertia / dt * inertial)
        .max(largest(&external))
        .max(internal_start)
        .max(f64::MIN_POSITIVE);
    // How small round-off lets the residual get. The stress is computed
    // from `F` and `J`, numbers near one, so it carries an absolute error
    // of about `eps` times the stiffness, however small the strain: the
    // internal force's floor is `eps` times the stiffness's size at each
    // unknown, `sum_p V_p (K + mu) |F_p| |grad N|`. Newton cannot go below
    // it (measured: a stiff block sliding to rest stalled at 4.2e-7
    // against 6.3e-7 for this sum times `eps`, while the tolerance's own
    // scale fell with the block's speed).
    // Source: LAPACK Users' Guide, 3rd ed., "Further Details: Error Bounds
    // for Linear Equation Solving": the componentwise backward error
    // `|r| / (|A| |x| + |b|)` is a few `eps` for a stable solve.
    // Adaptation: `|A| |x|` taken as the stiffness times the
    // configuration's size, since the stress's round-off is relative to
    // `F`, not to the stress; the factor 16 is this engine's margin over
    // the measured floor (declared).
    let (floor, force_round_off) = match &reused {
        Some(kept) => (kept.floor, kept.force_round_off),
        None => {
            let rows = parallel_scatter(grid.len, n, |p, acc| {
                let f = f_now[p];
                let size = (f.x_axis.length_squared() + f.y_axis.length_squared()).sqrt();
                let stiffness = v0[p] * (solids[p].bulk + solids[p].mu) * size;
                for a in 0..2 {
                    grid.visit(p, a, |k, _, g| acc[k] += stiffness * g.length());
                }
            });
            (
                16.0 * f64::EPSILON * largest(&rows),
                16.0 * f64::EPSILON * rows.iter().sum::<f64>(),
            )
        }
    };
    if matches!(factors.start, Some(None)) {
        factors.start = Some(Some(StartParts {
            v_n: v_n.clone(),
            rhs: rhs.clone(),
            projection,
            external: external.clone(),
            inertial,
            internal: internal_start,
            floor,
            force_round_off,
        }));
    }
    // Two starts, the better one kept, after Gast et al. 2015 (section 3.1,
    // "we compute both initial guesses and choose the one which produces
    // the smaller value of E"). Adapted: their second start is the
    // explicit predictor `x_n + dt v_n + dt^2 M^-1 f(x_n)` and they compare
    // their objective; here the second is one step of the linear elastic
    // problem from rest, `u = -(2 M / dt^2 + K)^-1 R(0)`, and the residuals
    // are compared, the step having no objective. The free flight leaves
    // the solution far behind once the step is several periods of the
    // body's vibration long.
    let zero = vec![0.0; grid.len];
    let linear: Vec<f64> = band
        .solve(&residual(&zero))
        .iter()
        .zip(&grid.active)
        .map(|(e, &on)| if on { -e } else { 0.0 })
        .collect();
    #[cfg(all(test, feature = "experimental"))]
    let starts = (admissible(&u), admissible(&linear));
    if admissible(&linear) {
        let r_linear = residual(&linear);
        if !admissible(&u) || largest(&r_linear) < largest(&r) {
            (u, r) = (linear, r_linear);
        }
    }
    report.start_residual = largest(&r) / scale;
    report.residual_allowed = (settings.newton_tolerance * scale).max(floor);
    report.grid_speed = largest(&v_n);
    #[cfg(all(test, feature = "experimental"))]
    let least_det = |w: &[f64]| {
        (0..n)
            .map(|p| deformation_at(w, p).determinant())
            .fold(f64::INFINITY, f64::min)
    };
    #[cfg(all(test, feature = "experimental"))]
    let mut homotopy_solution: Option<Vec<f64>> = None;
    #[cfg(all(test, feature = "experimental"))]
    {
        if hooks::on() {
            // Where the start's least det F sits, and how many particles
            // it inverts.
            let dets: Vec<f64> = (0..n)
                .map(|p| deformation_at(&u, p).determinant())
                .collect();
            let at = (0..n).fold(0, |a, p| if dets[p] < dets[a] { p } else { a });
            let inverted = dets.iter().filter(|&&d| d <= 0.0).count();
            hooks::line(format!(
                "start: free flight admissible {}, linear start admissible {}, start residual {:.2e} of scale {scale:.2e}, least det F {:.3e} at particle {at} x ({:.3}, {:.3}), {inverted} particles inverted, largest grid speed {:.2e}",
                starts.0,
                starts.1,
                report.start_residual,
                least_det(&u),
                x[at].x,
                x[at].y,
                report.grid_speed
            ));
        }
        if hooks::homotopy() > 0 {
            let r0 = residual(&zero);
            let precondition = |y: &[f64]| -> Vec<f64> {
                let mut z = band.solve(y);
                for (value, &on) in z.iter_mut().zip(&grid.active) {
                    if !on {
                        *value = 0.0;
                    }
                }
                z
            };
            // The same Jacobian product as the solve's, at `w`.
            let jacobian_at = |w: &[f64], d: &[f64]| -> Vec<f64> {
                let size = d.iter().fold(0.0f64, |m, e| m.max(e.abs()));
                if size == 0.0 {
                    return vec![0.0; grid.len];
                }
                let base = w.iter().fold(0.0f64, |m, e| m.max(e.abs()));
                let h = central_difference_step(grid.cell, base, size);
                let plus: Vec<f64> = w.iter().zip(d).map(|(a, b)| a + h * b).collect();
                let minus: Vec<f64> = w.iter().zip(d).map(|(a, b)| a - h * b).collect();
                let scaled: Vec<f64> = d.iter().map(|e| inertia * e / (dt * dt)).collect();
                let mut out = grid.mass_times(&scaled, mass, eps);
                for (o, (p, m)) in out.iter_mut().zip(forces(&plus).iter().zip(forces(&minus))) {
                    *o += (p - m) / (2.0 * h);
                }
                for (value, &on) in out.iter_mut().zip(&grid.active) {
                    if !on {
                        *value = 0.0;
                    }
                }
                out
            };
            let mut w = zero.clone();
            let mut lambda = 0.0f64;
            let mut increment = 1.0 / f64::from(hooks::homotopy());
            let (mut solves, mut newton_total, mut gmres_total) = (0u32, 0u32, 0u32);
            let mut least = f64::INFINITY;
            let mut outcome = String::from("reached");
            while lambda < 1.0 {
                let target = (lambda + increment).min(1.0);
                let shifted = |y: &[f64]| -> Vec<f64> {
                    residual(y)
                        .iter()
                        .zip(&r0)
                        .map(|(a, b)| a - (1.0 - target) * b)
                        .collect()
                };
                let mut y = w.clone();
                let mut ry = shifted(&y);
                let mut converged = false;
                let mut iterations = 0u32;
                loop {
                    let worst = largest(&ry);
                    if worst / scale <= settings.newton_tolerance {
                        converged = true;
                        break;
                    }
                    if iterations >= settings.max_newton || worst.is_nan() {
                        break;
                    }
                    iterations += 1;
                    let minus: Vec<f64> = ry.iter().map(|e| -e).collect();
                    let (delta, its) = gmres(
                        &|d: &[f64]| jacobian_at(&y, d),
                        &minus,
                        (&precondition, &grid.active),
                        settings.gmres_tolerance,
                        settings,
                    );
                    gmres_total += its;
                    let mut t = 1.0;
                    loop {
                        let trial: Vec<f64> =
                            y.iter().zip(&delta).map(|(a, b)| a + t * b).collect();
                        if !admissible(&trial) {
                            t *= 0.5;
                            if t < 1e-12 {
                                break;
                            }
                            continue;
                        }
                        let rt = shifted(&trial);
                        if largest(&rt) < worst || t < 1e-3 {
                            y = trial;
                            ry = rt;
                            break;
                        }
                        t *= 0.5;
                    }
                }
                solves += 1;
                newton_total += iterations;
                if converged && hooks::fold() {
                    // Power iteration on J for the largest magnitude, on
                    // J^-1 (GMRES solves) for the least.
                    let size = |z: &[f64]| z.iter().map(|e| e * e).sum::<f64>().sqrt();
                    let start: Vec<f64> = (0..grid.len)
                        .map(|k| {
                            if grid.active[k] {
                                ((k * 7919) % 101) as f64 - 50.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let (mut z, mut largest_eig) = (start.clone(), 0.0);
                    for _ in 0..20 {
                        let norm = size(&z).max(f64::MIN_POSITIVE);
                        z.iter_mut().for_each(|e| *e /= norm);
                        z = jacobian_at(&y, &z);
                        largest_eig = size(&z);
                    }
                    let (mut z, mut least_eig) = (start, f64::INFINITY);
                    for _ in 0..12 {
                        let norm = size(&z).max(f64::MIN_POSITIVE);
                        z.iter_mut().for_each(|e| *e /= norm);
                        let (solved, _) = gmres(
                            &|d: &[f64]| jacobian_at(&y, d),
                            &z,
                            (&precondition, &grid.active),
                            settings.gmres_tolerance,
                            settings,
                        );
                        z = solved;
                        least_eig = 1.0 / size(&z).max(f64::MIN_POSITIVE);
                    }
                    hooks::line(format!(
                        "fold watch at lambda {target:.4}: least |eigenvalue| {least_eig:.3e}, largest {largest_eig:.3e}, ratio {:.3e}, least det F {:.3e}",
                        least_eig / largest_eig,
                        least_det(&y)
                    ));
                }
                if converged {
                    lambda = target;
                    least = least.min(least_det(&y));
                    w = y;
                    increment = (2.0 * increment).min(1.0);
                } else {
                    increment *= 0.5;
                    if increment < 1e-4 {
                        outcome = format!(
                            "stalled at lambda {lambda:.4} (the next increment's Newton left residual {:.2e}, least det F there {:.3e})",
                            largest(&ry) / scale,
                            least_det(&y)
                        );
                        break;
                    }
                }
            }
            hooks::line(format!(
                "homotopy from u = 0: {outcome}, lambda {lambda:.4} in {solves} solves, Newton {newton_total}, GMRES {gmres_total}, least det F on the path {least:.3e}"
            ));
            if lambda >= 1.0 {
                homotopy_solution = Some(w);
            }
        }
    }
    // Masked like the solvers: a held face's entries stay zero, or a
    // Jacobian product would move it.
    let precondition = |y: &[f64]| -> Vec<f64> {
        let start = Instant::now();
        let mut z = band.solve(y);
        for (value, &on) in z.iter_mut().zip(&grid.active) {
            if !on {
                *value = 0.0;
            }
        }
        timed(&precondition_clock, start);
        z
    };
    let branches = |u: &[f64]| -> Vec<ContactBranch> {
        contacts
            .iter()
            .map(|c| {
                let (g, shares) = contact_geometry(grid, x, planes, u, c);
                let g_t = if friction > 0.0 {
                    contact_tangential_gap(grid, u, &shares)
                } else {
                    0.0
                };
                let k = contact_penalty(c, penalty);
                // At the tie, `lambda - k g = 0` (every condition at a
                // static increment's start: no multiplier yet, no gap), both
                // sides are elements of `max`'s generalized derivative (Ito
                // and Kunisch, as below). The step takes it inactive, its
                // inertia keeping the Jacobian definite; at rest the
                // stiffness alone is singular on the rigid motions the
                // touching conditions hold, so the static mode takes it
                // pressed and sticking (measured: the inactive tie left the
                // first static solve's GMRES at its cap, the Jacobian 1.6e15
                // off the residual's differences).
                if statics && c.multiplier - k * g == 0.0 {
                    return ContactBranch {
                        pressed: true,
                        slip: 0.0,
                    };
                }
                contact_branch(
                    (c.multiplier, c.tangential),
                    (g, g_t),
                    ((k, k_t_of(c)), friction),
                )
            })
            .collect()
    };
    // The internal force's derivative exactly, through each particle's
    // linearisation (`Linearised`), unless a probe changes the force outside
    // the step stress; then central differences of the whole force.
    #[cfg(all(test, feature = "experimental"))]
    let exact = !(lagrangian
        && (hooks::cells().is_some()
            || hooks::viscosity() > 0.0
            || hooks::trapezoid_p()
            || hooks::cofactor()));
    #[cfg(not(all(test, feature = "experimental")))]
    let exact = true;
    let (linearise_clock, push_clock) = (Cell::new(0.0), Cell::new(0.0));
    // Each particle's linearisation at `u`, once per Newton iterate; `None`
    // when the product is taken by differences.
    let linearised_at = |u: &[f64]| -> Option<Vec<Linearised>> {
        if !exact {
            return None;
        }
        #[cfg(all(test, feature = "experimental"))]
        hooks::add_part(5, 1.0);
        let start = Instant::now();
        let unit = [
            DMat2::from_cols(DVec2::X, DVec2::ZERO),
            DMat2::from_cols(DVec2::ZERO, DVec2::Y),
            DMat2::from_cols(DVec2::Y, DVec2::X),
        ];
        // The internal force's `F_n+1` and stress, when it was evaluated at
        // this very iterate (bit for bit): the same values as computing them.
        let kept = last_stresses.borrow();
        let kept = kept.as_ref().and_then(|(at, points)| {
            (at.len() == u.len() && at.iter().zip(u).all(|(a, b)| a.to_bits() == b.to_bits()))
                .then_some(points)
        });
        let out = (0..n)
            .into_par_iter()
            .map(|p| {
                if v0[p] <= 0.0 {
                    return Linearised::default();
                }
                let (f0, f1) = (
                    f_now[p],
                    kept.map_or_else(|| deformation_at(u, p), |k| k[p].1),
                );
                let law_stress = kept.map(|k| k[p].2);
                let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
                #[cfg(all(test, feature = "experimental"))]
                if let Some(hoop) = hoop {
                    let r = hoop.radii[p];
                    if r <= 0.0 {
                        return Linearised::default();
                    }
                    let (f33_0, f33_1) = (
                        (x[p].x - hoop.axis_x) / r,
                        (x[p].x - hoop.axis_x + grid.read(u, p, 0).0) / r,
                    );
                    let line = solids[p].hoop_linearised((c0, f33_0), (c1, f33_1), dt, unit);
                    return Linearised {
                        f1,
                        s: line.s,
                        ds: line.ds,
                        hoop: Some(line.rest),
                    };
                }
                if statics {
                    return Linearised {
                        f1,
                        s: law_stress.unwrap_or_else(|| solids[p].static_stress(c1)),
                        ds: unit.map(|e| solids[p].static_stress_tangent(c1, e)),
                        #[cfg(all(test, feature = "experimental"))]
                        hoop: None,
                    };
                }
                Linearised {
                    f1,
                    s: law_stress.unwrap_or_else(|| solids[p].step_stress(c0, c1, dt)),
                    ds: solids[p].step_stress_tangents(c0, c1, dt, unit),
                    #[cfg(all(test, feature = "experimental"))]
                    hoop: None,
                }
            })
            .collect();
        timed(&linearise_clock, start);
        Some(out)
    };
    // The internal force's derivative along `d` from the linearisations:
    // `dF = grad d` (the force's `F` is linear in `u`), `dC = dF^T F + F^T
    // dF`, the stress's change from its three directions, and the first
    // Piola stress's `V0 (1/2 dF S + F_n+1/2 dS)`, scattered as the force is.
    let internal_product = |lin: &[Linearised], d: &[f64]| -> Vec<f64> {
        parallel_scatter(grid.len, n, |p, acc| {
            if v0[p] <= 0.0 {
                return;
            }
            let t = &lin[p];
            let (gu, gv) = (grid.gradient(d, p, 0), grid.gradient(d, p, 1));
            let grad = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
            let f0 = f_now[p];
            let df = if lagrangian { grad } else { grad * f0 };
            let dc = df.transpose() * t.f1 + t.f1.transpose() * df;
            #[cfg(all(test, feature = "experimental"))]
            if let (Some(line), Some(hoop)) = (&t.hoop, hoop) {
                super::axisymmetric::hoop_product(
                    grid,
                    (t.f1, t.s, &t.ds, line),
                    (p, d, hoop.radii[p]),
                    (v0[p], f0, df, dc),
                    acc,
                );
                return;
            }
            let ds = t.ds[0] * dc.x_axis.x + t.ds[1] * dc.y_axis.y + t.ds[2] * dc.y_axis.x;
            // `d(F_n+1/2 S) = 1/2 dF S + F_n+1/2 dS`; at rest `d(F_1 S) = dF S +
            // F_1 dS`.
            let first = if statics {
                df * t.s + t.f1 * ds
            } else {
                0.5 * df * t.s + 0.5 * (f0 + t.f1) * ds
            };
            let first = if lagrangian {
                v0[p] * first
            } else {
                v0[p] * first * f0.transpose()
            };
            for a in 0..2 {
                let row = first.row(a);
                grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
            }
        })
    };
    // A contact holding a segment: its normal, parameter and couple move
    // with `u`, so its push's derivative is taken by differences.
    let holds_segment = |c: &Contact| c.points.iter().any(|(p, _)| p.segment.is_some());
    // The plane contacts' push derivative along `d`, the active set frozen,
    // exactly: a plane share's normal and spread do not move with `u`
    // (`plane_share`), so with `dg = sum_s w_s n_s . sum_j c_j dm_j` and
    // `dg_t` the same along `n_s` turned a quarter, `contact_force`'s
    // frozen branch gives `d size = -k dg` and `d along = -k dg_t`
    // (sticking) or `slip mu d size` (slipping), spread as the push is.
    let plane_push_product = |held: &[ContactBranch], d: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; grid.len];
        let moved = |r: usize| DVec2::new(grid.read(d, r, 0).0, grid.read(d, r, 1).0);
        for (c, branch) in contacts.iter().zip(held) {
            if !branch.pressed || holds_segment(c) {
                continue;
            }
            let k = contact_penalty(c, penalty);
            let shares: Vec<PlaneShare> = c
                .points
                .iter()
                .map(|&(point, weight)| {
                    let (normal, spread) = plane_share(planes, &point);
                    let relative: DVec2 = spread
                        .iter()
                        .filter(|(_, share)| *share != 0.0)
                        .map(|&(r, share)| share * moved(r))
                        .sum();
                    (weight, normal, spread, relative)
                })
                .collect();
            let (mut dg, mut dg_t) = (0.0, 0.0);
            for (weight, normal, _, relative) in &shares {
                dg += weight * normal.dot(*relative);
                dg_t += weight * normal.perp().dot(*relative);
            }
            let d_size = -k * dg;
            let d_along = if friction <= 0.0 {
                0.0
            } else if branch.slip == 0.0 {
                -k_t_of(c) * dg_t
            } else {
                branch.slip * friction * d_size
            };
            for (weight, normal, spread, _) in &shares {
                let force = *weight * (d_size * *normal + d_along * normal.perp());
                for &(r, share) in spread {
                    let part = share * force;
                    for a in 0..2 {
                        if part[a] != 0.0 {
                            grid.visit(r, a, |k, w, _| out[k] += w * part[a]);
                        }
                    }
                }
            }
        }
        out
    };
    let jacobian_at = |u: &[f64],
                       (held, lin): (&[ContactBranch], Option<&[Linearised]>),
                       d: &[f64]|
     -> Vec<f64> {
        let start = Instant::now();
        let size = d.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        if size == 0.0 {
            return vec![0.0; grid.len];
        }
        let base = u.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let h = central_difference_step(grid.cell, base, size);
        // The shifted iterates `u +- h d`, built only where a difference
        // reads them (the internal force without linearisations, the
        // segments' push): the exact path with planes only never does.
        let shifted = std::cell::OnceCell::new();
        let shifted = || {
            shifted.get_or_init(|| {
                let plus: Vec<f64> = u.iter().zip(d).map(|(a, b)| a + h * b).collect();
                let minus: Vec<f64> = u.iter().zip(d).map(|(a, b)| a - h * b).collect();
                (plus, minus)
            })
        };
        // The mass exactly; the internal force exactly from the
        // linearisations, or by central differences (forward ones, one
        // evaluation fewer, left Newton short of convergence on 4 of 60
        // frame-long steps and the energy rising 0.3 percent: measured,
        // reverted).
        let scaled: Vec<f64> = d.iter().map(|e| inertia * e / (dt * dt)).collect();
        #[cfg(all(test, feature = "experimental"))]
        let part = Instant::now();
        #[cfg(all(test, feature = "experimental"))]
        {
            hooks::add_part(0, 1.0);
            hooks::add_part(1, part.duration_since(start).as_secs_f64());
        }
        let mut out = grid.mass_times(&scaled, mass, eps);
        #[cfg(all(test, feature = "experimental"))]
        let part = {
            hooks::add_part(2, part.elapsed().as_secs_f64());
            Instant::now()
        };
        match lin {
            Some(lin) => {
                let product = internal_product(lin, d);
                // The exact product against central differences of
                // the force on the same iterate.
                #[cfg(all(test, feature = "experimental"))]
                if hooks::tangent_check() {
                    let size = product.iter().fold(0.0f64, |m, e| m.max(e.abs()));
                    let gap = product
                        .iter()
                        .zip(internal(&shifted().0).iter().zip(internal(&shifted().1)))
                        .fold(0.0f64, |m, (e, (p, q))| {
                            m.max((e - (p - q) / (2.0 * h)).abs())
                        });
                    println!(
                        "tangent check: product {size:.3e}, relative gap {:.3e}",
                        gap / size.max(f64::MIN_POSITIVE)
                    );
                }
                for (o, value) in out.iter_mut().zip(product) {
                    *o += value;
                }
                #[cfg(all(test, feature = "experimental"))]
                hooks::add_part(3, part.elapsed().as_secs_f64());
            }
            None => {
                for (o, (p, m)) in out
                    .iter_mut()
                    .zip(internal(&shifted().0).iter().zip(internal(&shifted().1)))
                {
                    *o += (p - m) / (2.0 * h);
                }
            }
        }
        // Less the push's derivative with the active set frozen: a
        // smooth function then (linear against a plane; the segment's
        // normal and closest point move with `u`), by the same central
        // differences.
        if held.iter().any(|a| a.pressed) {
            let pushing = Instant::now();
            // The contacts holding a segment by central differences of
            // their push alone (the others marked free in a copy of the
            // set); the plane contacts exactly.
            let only = |segments: bool| -> Vec<ContactBranch> {
                held.iter()
                    .zip(contacts)
                    .map(|(b, c)| ContactBranch {
                        pressed: b.pressed && holds_segment(c) == segments,
                        ..*b
                    })
                    .collect()
            };
            let segment_set = only(true);
            if segment_set.iter().any(|b| b.pressed) {
                let (pp, pm) = (
                    push_with(&shifted().0, Some(&segment_set), None),
                    push_with(&shifted().1, Some(&segment_set), None),
                );
                for (o, (p, m)) in out.iter_mut().zip(pp.iter().zip(&pm)) {
                    *o -= (p - m) / (2.0 * h);
                }
            }
            let exact = plane_push_product(held, d);
            #[cfg(all(test, feature = "experimental"))]
            if push_check() {
                let plane_set = only(false);
                let (pp, pm) = (
                    push_with(&shifted().0, Some(&plane_set), None),
                    push_with(&shifted().1, Some(&plane_set), None),
                );
                let size = exact.iter().fold(0.0f64, |m, e| m.max(e.abs()));
                let gap = exact
                    .iter()
                    .zip(pp.iter().zip(&pm))
                    .fold(0.0f64, |m, (e, (p, q))| {
                        m.max((e - (p - q) / (2.0 * h)).abs())
                    });
                println!("push check: exact {size:.3e}, against differences {gap:.3e}");
            }
            for (o, value) in out.iter_mut().zip(exact) {
                *o -= value;
            }
            timed(&push_clock, pushing);
        }
        gauge_force(d, &mut out);
        #[cfg(all(test, feature = "experimental"))]
        let part = Instant::now();
        for (value, &on) in out.iter_mut().zip(&grid.active) {
            if !on {
                *value = 0.0;
            }
        }
        #[cfg(all(test, feature = "experimental"))]
        hooks::add_part(4, part.elapsed().as_secs_f64());
        timed(&jacobian_clock, start);
        out
    };
    // Newton's best residual so far and the iterations since it improved:
    // five without a new best (a converging step takes two to four) is a
    // stagnation, a residual whose own evaluation noise sits above the
    // tolerance, but only where the best residual reached is within that
    // noise: the internal force's round-off (`floor`) or a contact push's,
    // its penalty times its gap's round-off (`8 eps` times the gap's terms,
    // the bound `contact_allowance` already uses). Above it, Newton goes on
    // to `max_newton`. Reported, not hidden (`StepReport::stagnated`),
    // never a hang: the step's equations then hold only to that noise. The
    // one source measured so far, the discrete gradient's energy difference
    // at small increments, is removed at its root
    // (`corotated_energy_increment`); this stays as the guard. This
    // engine's own, declared. A solver exit criterion, no physics: without
    // the noise test it fired at 1e8 times the tolerance on a contact solve
    // that ran on converged in 9 to 11 iterations, and the contact passes
    // then cycled between two multiplier states to their cap (measured, a
    // sphere on a block). Framing, not proof (convex problems only): an
    // augmented Lagrangian's inner solve must be accurate against the
    // multiplier step it feeds (Eckstein and Silva, Math. Program. 141
    // (2013), eq. 12; Rockafellar's criteria).
    let noise = |u: &[f64]| -> f64 {
        contacts.iter().fold(floor, |most, c| {
            let (_, shares) = contact_geometry(grid, x, planes, u, c);
            most.max(
                contact_penalty(c, penalty)
                    * 8.0
                    * f64::EPSILON
                    * contact_round_off_scale(grid, x, planes, u, &shares),
            )
        })
    };
    let (mut best, mut since_best) = (f64::INFINITY, 0u32);
    loop {
        let worst = largest(&r);
        report.residual = worst / scale;
        if report.residual <= settings.newton_tolerance || worst <= floor {
            report.converged = true;
            break;
        }
        if worst < best {
            best = worst;
            since_best = 0;
        } else {
            since_best += 1;
            if since_best >= 5 && best <= noise(&u) {
                report.stagnated = true;
                break;
            }
        }
        if report.newton >= settings.max_newton {
            break;
        }
        report.newton += 1;
        // The walls' push `max(0, lambda - k g) n` is not differentiable
        // where its argument vanishes: a finite difference straddling that
        // kink has a slope set by its own step, not one linear operator
        // (measured: GMRES unable to lower the linear residual, Newton
        // stalled at 3-7e-8 on steps where two bodies touched with no
        // force). Semismooth Newton instead (Ito and Kunisch, M2AN 37
        // (2003), Prop. 1.1 and Remark 2.1, the primal-dual active set):
        // the push's generalized derivative with the active set frozen at
        // this iterate, a contact active iff `lambda - k g > 0` (the tie
        // taken inactive), applied exactly; the internal force alone by
        // finite differences.
        let held = branches(&u);
        let lin = linearised_at(&u);
        let jacobian = |d: &[f64]| jacobian_at(&u, (&held, lin.as_deref()), d);
        #[cfg(all(test, feature = "experimental"))]
        hooks::record_condition(&jacobian, grid, mass);
        #[cfg(all(test, feature = "experimental"))]
        if hooks::preconditioned_wanted() {
            let built = &factors.jacobian[0].1;
            let own = grid.approximate_jacobian((mass, v0, solids), eps, coefficient, built);
            let quarter =
                grid.approximate_jacobian((mass, v0, solids), eps, 0.25 * coefficient, built);
            hooks::record_preconditioned(&jacobian, grid, (&own, &quarter));
        }
        let minus_r: Vec<f64> = r.iter().map(|e| -e).collect();
        // A fixed linear tolerance: Eisenstat and Walker's forcing term
        // (choice 2, capped at 0.1) left Newton short of convergence on 14
        // and 16 of 60 frame-long steps (measured, reverted).
        let start = Instant::now();
        let (delta, iterations) = gmres(
            &jacobian,
            &minus_r,
            (&precondition, &grid.active),
            settings.gmres_tolerance,
            settings,
        );
        report.timing.gmres += start.elapsed().as_secs_f64();
        report.gmres += iterations;
        // The replay: the true residual `|b - J delta| / |b|` at GMRES's exit,
        // and, with a reused factor, the same system solved with the exact one.
        #[cfg(all(test, feature = "experimental"))]
        if hooks::gmres_replay() {
            let size = |y: &[f64]| y.iter().map(|e| e * e).sum::<f64>().sqrt();
            let true_residual = |d: &[f64]| {
                let jd = jacobian(d);
                let gap: Vec<f64> = jd
                    .iter()
                    .zip(&minus_r)
                    .zip(&grid.active)
                    .map(|((a, b), &on)| if on { b - a } else { 0.0 })
                    .collect();
                size(&gap) / size(&minus_r).max(f64::MIN_POSITIVE)
            };
            let used = (iterations, true_residual(&delta));
            match &exact_band {
                Some(exact) => {
                    let precondition_exact = |y: &[f64]| -> Vec<f64> {
                        let mut z = exact.solve(y);
                        for (value, &on) in z.iter_mut().zip(&grid.active) {
                            if !on {
                                *value = 0.0;
                            }
                        }
                        z
                    };
                    let (d2, i2) = gmres(
                        &jacobian,
                        &minus_r,
                        (&precondition_exact, &grid.active),
                        settings.gmres_tolerance,
                        settings,
                    );
                    hooks::record_reused(used, (i2, true_residual(&d2)));
                }
                None => hooks::record_own(used),
            }
        }
        // What GMRES reached: `|J delta + R| / |R|`.
        #[cfg(all(test, feature = "experimental"))]
        let reached = if hooks::on() {
            let size = |y: &[f64]| y.iter().map(|e| e * e).sum::<f64>().sqrt();
            let jd = jacobian(&delta);
            let gap: Vec<f64> = jd.iter().zip(&r).map(|(a, b)| a + b).collect();
            size(&gap) / size(&r).max(f64::MIN_POSITIVE)
        } else {
            0.0
        };
        // Backtrack while the residual grows.
        let mut t = 1.0;
        loop {
            let trial: Vec<f64> = u.iter().zip(&delta).map(|(a, b)| a + t * b).collect();
            // KNOWN-LIMIT[DISCRETIZATION/OPEN-GAP]: the backtracking's floor
            // 1e-12 (inadmissible trials) and its acceptance of any step
            // below 1e-3 are not sourced (not in Walter 2014 nor Coleman
            // 1984). Removal criterion: a sourced or derived backtracking
            // rule with the counts identical or explained.
            if !admissible(&trial) {
                t *= 0.5;
                if t < 1e-12 {
                    break;
                }
                continue;
            }
            let rt = residual(&trial);
            let worst_t = largest(&rt);
            if worst_t < worst || t < 1e-3 {
                u = trial;
                r = rt;
                break;
            }
            t *= 0.5;
        }
        #[cfg(all(test, feature = "experimental"))]
        if hooks::on() {
            hooks::line(format!(
                "newton {}: residual {:.2e} -> {:.2e}, GMRES {iterations} reaching {reached:.1e}, step length {t:.1e}, least det F {:.3e}",
                report.newton,
                worst / scale,
                largest(&r) / scale,
                least_det(&u)
            ));
        }
    }
    report.residual_work = r.iter().zip(&u).map(|(a, b)| (a * b).abs()).sum();
    report.residual_force =
        (0..grid.len)
            .filter(|&k| grid.active[k])
            .fold(DVec2::ZERO, |sum, k| {
                let mut e = DVec2::ZERO;
                e[grid.axis_of[k]] = r[k];
                sum + e
            });
    report.force_round_off = force_round_off;
    // The mirrors' push (`StepReport::mirror_force`): the points' momentum rate
    // is `sum_k L_k a_k` (the lumped mass is the points' mass carried through
    // the same weights), so whatever the rows' sum adds to it beyond the loads
    // on the points is the folding's. The internal force's evaluation here is
    // a reading, not the solve's: its count is kept out.
    #[cfg(all(test, feature = "experimental"))]
    if hoop.is_some() {
        let counted = evaluations.get();
        let rows = raw(&u);
        evaluations.set(counted);
        let pushed = if contacts.is_empty() {
            vec![0.0; grid.len]
        } else {
            push(&u)
        };
        let mut loads = (0..n).fold(DVec2::ZERO, |sum, p| {
            sum + mass[p] * body.map_or(gravity, |body| body[p])
        });
        loads += super::axisymmetric::point_loads_total();
        let (mut reaction, mut net) = (-loads, DVec2::ZERO);
        for k in 0..grid.len {
            let a = grid.axis_of[k];
            let accel = inertia * (u[k] / dt - v_n[k]) / dt;
            reaction[a] += grid.lumped[k] * accel - rows[k];
            net[a] += pushed[k];
        }
        report.mirror_force = reaction;
        report.contact_net_force = net;
    }
    #[cfg(all(test, feature = "experimental"))]
    if hooks::on() {
        let distance = homotopy_solution.as_ref().map_or(String::new(), |w| {
            let gap = u
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let size = w.iter().fold(0.0f64, |m, e| m.max(e.abs()));
            format!(
                ", largest gap to the homotopy's solution {:.2e} of its largest {size:.2e}",
                gap
            )
        });
        hooks::line(format!(
            "end: converged {}, residual {:.2e}, Newton {}, GMRES {}{distance}",
            report.converged, report.residual, report.newton, report.gmres
        ));
    }
    #[cfg(all(test, feature = "experimental"))]
    if lagrangian && hooks::viscosity() > 0.0 {
        let viscous = parallel_scatter(grid.len, n, |p, acc| {
            let f1 = deformation_at(&u, p);
            let f0 = f_now[p];
            let end: [DVec2; 2] = [0, 1].map(|a| {
                let (_, gu) = grid.read(&u, p, a);
                let (_, gv) = grid.read(&v_n, p, a);
                2.0 * gu / dt - gv
            });
            let f_dot = DMat2::from_cols(
                DVec2::new(end[0].x, end[1].x),
                DVec2::new(end[0].y, end[1].y),
            );
            let first =
                v0[p] * (0.5 * (f0 + f1)) * solids[p].viscous_stress(f1, f_dot, hooks::viscosity());
            for a in 0..2 {
                let row = first.row(a);
                grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
            }
        });
        hooks::set_viscous_work(viscous.iter().zip(&u).map(|(f, d)| f * d).sum());
    }
    report.phases = [
        0.0,
        projected_at,
        banded_at - projected_at,
        clock.elapsed().as_secs_f64() - banded_at,
        0.0,
    ];
    // The responses to the active multipliers, with the sets frozen at the
    // solution: each column one more linear solve with Newton's own
    // Jacobian and preconditioner, started from the same contact's last
    // response (`Factors::responses`): to the same tolerance, fewer
    // iterations when a pass moved the displacement little.
    let mut responses = Vec::new();
    let responding = Instant::now();
    let jacobian_before_responses = jacobian_clock.get();
    if wants_responses && report.converged && !contacts.is_empty() {
        let held = branches(&u);
        let lin = linearised_at(&u);
        let jacobian = |d: &[f64]| jacobian_at(&u, (&held, lin.as_deref()), d);
        let base = push_with(&u, Some(&held), None);
        let mut computed = Vec::new();
        for (i, branch) in held.iter().enumerate() {
            if !branch.pressed {
                continue;
            }
            let sticking = friction > 0.0 && branch.slip == 0.0;
            for tangential in [false, true] {
                if tangential && !sticking {
                    continue;
                }
                let mut b = push_with(&u, Some(&held), Some((i, tangential, 1.0)));
                for ((value, &before), &on) in b.iter_mut().zip(&base).zip(&grid.active) {
                    *value = if on { *value - before } else { 0.0 };
                }
                let key = contacts[i].key;
                let last = factors
                    .responses
                    .iter()
                    .find(|(k, t, _)| *k == key && *t == tangential)
                    .map(|(_, _, du)| du.as_slice());
                #[cfg(all(test, feature = "experimental"))]
                let last = last.filter(|_| !hooks::cold_responses());
                let start = Instant::now();
                let (du, iterations) = gmres_from(
                    &jacobian,
                    &b,
                    last,
                    (&precondition, &grid.active),
                    settings.gmres_tolerance,
                    settings,
                );
                report.timing.gmres += start.elapsed().as_secs_f64();
                report.gmres += iterations;
                computed.push((key, tangential, du.clone()));
                responses.push(Response {
                    contact: i,
                    tangential,
                    du,
                });
            }
        }
        factors.responses = computed;
    }
    report.timing.responses = responding.elapsed().as_secs_f64();
    report.timing.residual = residual_clock.get();
    report.timing.jacobian = jacobian_clock.get();
    report.timing.internal_evaluations = evaluations.get();
    report.timing.linearisation = linearise_clock.get();
    report.timing.push_derivative = push_clock.get();
    report.timing.responses_jacobian = jacobian_clock.get() - jacobian_before_responses;
    report.timing.preconditioner_solve = precondition_clock.get();
    report.timing.preconditioner_build = banded_at - projected_at;
    // At rest no dashpot works (`statics`).
    let dissipated = if !statics && solids.iter().any(Elastic::is_viscous) {
        (0..n)
            .map(|p| {
                if !solids[p].is_viscous() || v0[p] <= 0.0 {
                    return 0.0;
                }
                let (f0, f1) = (f_now[p], deformation_at(&u, p));
                v0[p]
                    * solids[p].viscous_dissipation_over(
                        f0.transpose() * f0,
                        f1.transpose() * f1,
                        dt,
                    )
            })
            .collect()
    } else {
        Vec::new()
    };
    // The measured error estimate of this converged step (`estimate`),
    // when a probe samples it; it changes nothing here.
    #[cfg(all(test, feature = "experimental"))]
    if hooks::lobatto_wanted() {
        let mut applied = external.clone();
        if !contacts.is_empty() {
            applied.iter_mut().zip(push(&u)).for_each(|(a, p)| *a += p);
        }
        let mass_factor = factors.mass.as_ref().expect("factored above");
        hooks::record_lobatto(estimate::lobatto_error(&estimate::Converged {
            grid,
            mass,
            v0,
            solids,
            f_start: f_now,
            lagrangian,
            dt,
            eps,
            mass_factor,
            applied,
            u: &u,
            v_n: &v_n,
        }));
    }
    #[cfg(all(test, feature = "experimental"))]
    {
        hooks::add_part(7, 1.0);
        hooks::add_part(8, solve_clock.elapsed().as_secs_f64());
        hooks::add_part(9, f64::from(report.newton));
    }
    Solution {
        u,
        v_n,
        report,
        responses,
        dissipated,
        tangential_penalty: tangential,
    }
}
