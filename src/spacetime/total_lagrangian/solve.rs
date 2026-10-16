//! One step's nonlinear solve: the residual (inertia, internal force, contact
//! push, weight) and Newton's method with a Jacobian-free GMRES.

use super::*;
use std::cell::Cell;
use std::time::Instant;

/// Adds the time since `start` to `clock` (`Timing`).
fn timed(clock: &Cell<f64>, start: Instant) {
    clock.set(clock.get() + start.elapsed().as_secs_f64());
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
}

/// The band factors a step's solves share across the walls' passes
/// (`step`) while the grid stays the same: both are dropped when a face
/// is let go or held again, and the Jacobian's is rebuilt when the held
/// particles change, as their penalties are in it.
#[derive(Clone, Default)]
pub(crate) struct Factors {
    pub(crate) mass: Option<Band>,
    pub(crate) jacobian: Option<(Vec<Spring>, Band)>,
}

/// A held particle's penalty in the preconditioner: particle, lattice,
/// stiffness (`Grid::approximate_jacobian`).
pub(crate) type Spring = (usize, usize, f64);

/// The step's nonlinear solve on `grid` (`step`).
pub(crate) fn solve(
    grid: &Grid,
    inputs: &Inputs<'_>,
    contacts: &[Contact],
    factors: &mut Factors,
) -> Solution {
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
    } = *inputs;
    let clock = std::time::Instant::now();
    #[cfg(all(test, feature = "experimental"))]
    hooks::begin_solve();
    let n = x.len();
    let eps = settings.epsilon;
    // The midpoint's inertia, `2 M (u / dt - v_n) / dt`.
    let inertia = 2.0;
    // The grid velocity: the consistent-mass projection (eq. 28), on the
    // unknowns with some mass (`Grid::massive`).
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
    let factor = factors
        .mass
        .get_or_insert_with(|| grid.mass_band(mass, eps));
    let v_n = preconditioned_cg(
        &|y: &[f64]| grid.mass_times(y, mass, eps),
        &rhs,
        &|r: &[f64]| factor.solve(r),
        &massive,
    );
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
    let external: Vec<f64> = match body {
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
    let f_now: &[DMat2] = f;
    let deformation_at = |u: &[f64], p: usize| -> DMat2 {
        let (_, gu) = grid.read(u, p, 0);
        let (_, gv) = grid.read(u, p, 1);
        let grad = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
        if lagrangian {
            f_now[p] + grad
        } else {
            (DMat2::IDENTITY + grad) * f_now[p]
        }
    };
    // The internal force (eq. 30a) with the algorithmic stress. On the
    // reference basis `F_n^T grad_n N` is `grad_X N` itself.
    // The internal force's evaluations (`Timing::internal_evaluations`).
    let evaluations = Cell::new(0u64);
    let internal = |u: &[f64]| -> Vec<f64> {
        evaluations.set(evaluations.get() + 1);
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
        parallel_scatter(grid.len, n, |p, acc| {
            let f1 = deformation_at(u, p);
            let f0 = f_now[p];
            let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
            let s = solids[p].algorithmic_stress_over(c0, c1, dt);
            // The Kelvin-Voigt dashpot in parallel: its stress over the
            // step (`Elastic::viscous_stress_over`), on the same `F_n+1/2`.
            let s = if solids[p].is_viscous() {
                s + solids[p].viscous_stress_over(c0, c1, dt)
            } else {
                s
            };
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
                v0[p] * (0.5 * (f0 + f1)) * s
            } else {
                v0[p] * (0.5 * (f0 + f1)) * s * f0.transpose()
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
            for a in 0..2 {
                let row = first.row(a);
                grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
            }
        })
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
            let (size, along) = contact_force(
                (multiplier, tangential),
                (g, g_t),
                (contact_penalty(c, penalty), friction),
                frozen.map(|set| set[i]),
            );
            if size == 0.0 && along == 0.0 {
                continue;
            }
            for share in &shares {
                let force = share.weight * (size * share.normal + along * share.normal.perp());
                let mut parts = vec![
                    (share.spread[0].0, share.spread[0].1 * force),
                    (share.spread[1].0, share.spread[1].1 * force),
                    (share.spread[2].0, share.spread[2].1 * force),
                ];
                if along != 0.0
                    && let Some((first, second, couple)) =
                        friction_couple(grid, x, u, &share.point, share.normal, &share.spread)
                {
                    let turn = share.weight * couple * along * share.normal;
                    parts.push((second, turn));
                    parts.push((first, -turn));
                }
                for (r, part) in parts {
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
        out
    };
    // The parts' clocks (`Timing`), shared by the closures below.
    let (residual_clock, jacobian_clock, precondition_clock) =
        (Cell::new(0.0), Cell::new(0.0), Cell::new(0.0));
    let residual = |u: &[f64]| -> Vec<f64> {
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
    // The preconditioner: `2 M_eps / dt^2 + K`, `K` the linear elastic
    // stiffness at the reference shape, factored exactly in band form once
    // per step (`Band`). This engine's choice; the source's solves used
    // PETSc's matrix-free GMRES (their section 4).
    // A pair's penalty couples its two particles; the preconditioner keeps
    // only each particle's own block of it.
    let springs: Vec<Spring> = contacts
        .iter()
        .flat_map(|c| -> Vec<Spring> {
            let k = contact_penalty(c, penalty);
            c.points
                .iter()
                .flat_map(|&(point, weight)| -> Vec<Spring> {
                    let k = k * weight * weight;
                    match point.segment {
                        Some((first, second)) => [point.particle, first, second]
                            .into_iter()
                            .flat_map(|r| [(r, 0, k), (r, 1, k)])
                            .collect(),
                        // With friction both components are held.
                        None if friction > 0.0 => std::iter::once(point.particle)
                            .chain(point.partner)
                            .flat_map(|r| [(r, 0, k), (r, 1, k)])
                            .collect(),
                        None => {
                            let axis = planes[point.plane].axis;
                            std::iter::once((point.particle, axis, k))
                                .chain(point.partner.map(|q| (q, axis, k)))
                                .collect()
                        }
                    }
                })
                .collect()
        })
        .collect();
    let projected_at = clock.elapsed().as_secs_f64();
    if factors
        .jacobian
        .as_ref()
        .is_none_or(|(built, _)| *built != springs)
    {
        let band =
            grid.approximate_jacobian((mass, v0, solids), eps, inertia / (dt * dt), &springs);
        factors.jacobian = Some((springs, band));
    }
    let band = &factors.jacobian.as_ref().expect("built above").1;
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
    let admissible =
        |u: &[f64]| (0..n).all(|p| v0[p] <= 0.0 || deformation_at(u, p).determinant() > 0.0);
    let mut u: Vec<f64> = (0..grid.len)
        .map(|k| if grid.active[k] { dt * v_n[k] } else { 0.0 })
        .collect();
    let mut report = StepReport {
        projection,
        ..Default::default()
    };
    let mut r = residual(&u);
    // The residual's scale: the forces of the step's own physics, its
    // inertia, its weight and the stress it starts with. Not the residual
    // of a start: a free flight far from the solution made that scale so
    // large that a step still far from balance read as converged (measured:
    // `u . R` at -5.8e5 against 1.3e4 of kinetic energy change, the energy
    // falling 86 percent in that one step).
    let scale = (inertia / dt * largest(&grid.mass_times(&v_n, mass, eps)))
        .max(largest(&external))
        .max(largest(&internal(&vec![0.0; grid.len])))
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
    let floor = {
        let rows = parallel_scatter(grid.len, n, |p, acc| {
            let f = f_now[p];
            let size = (f.x_axis.length_squared() + f.y_axis.length_squared()).sqrt();
            let stiffness = v0[p] * (solids[p].bulk + solids[p].mu) * size;
            for a in 0..2 {
                grid.visit(p, a, |k, _, g| acc[k] += stiffness * g.length());
            }
        });
        16.0 * f64::EPSILON * largest(&rows)
    };
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
                let h = 1e-6 * (grid.cell + base) / size;
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
                contact_branch(
                    (c.multiplier, c.tangential),
                    (g, g_t),
                    (contact_penalty(c, penalty), friction),
                )
            })
            .collect()
    };
    let jacobian_at = |u: &[f64], held: &[ContactBranch], d: &[f64]| -> Vec<f64> {
        let start = Instant::now();
        let size = d.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        if size == 0.0 {
            return vec![0.0; grid.len];
        }
        let base = u.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let h = 1e-6 * (grid.cell + base) / size;
        let plus: Vec<f64> = u.iter().zip(d).map(|(a, b)| a + h * b).collect();
        let minus: Vec<f64> = u.iter().zip(d).map(|(a, b)| a - h * b).collect();
        // The mass exactly, the internal force by central
        // differences: forward ones (one evaluation fewer) left Newton
        // short of convergence on 4 of 60 frame-long steps and the
        // energy rising 0.3 percent (measured, reverted).
        let scaled: Vec<f64> = d.iter().map(|e| inertia * e / (dt * dt)).collect();
        let mut out = grid.mass_times(&scaled, mass, eps);
        for (o, (p, m)) in out
            .iter_mut()
            .zip(internal(&plus).iter().zip(internal(&minus)))
        {
            *o += (p - m) / (2.0 * h);
        }
        // Less the push's derivative with the active set frozen: a
        // smooth function then (linear against a plane; the segment's
        // normal and closest point move with `u`), by the same central
        // differences.
        if held.iter().any(|a| a.pressed) {
            let (pp, pm) = (
                push_with(&plus, Some(held), None),
                push_with(&minus, Some(held), None),
            );
            for (o, (p, m)) in out.iter_mut().zip(pp.iter().zip(&pm)) {
                *o -= (p - m) / (2.0 * h);
            }
        }
        for (value, &on) in out.iter_mut().zip(&grid.active) {
            if !on {
                *value = 0.0;
            }
        }
        timed(&jacobian_clock, start);
        out
    };
    // Newton's best residual so far and the iterations since it improved:
    // five without a new best (a converging step takes two to four) is a
    // stagnation, a residual whose own evaluation noise sits above the
    // tolerance. Reported, not hidden (`StepReport::stagnated`), never a
    // hang: the step's equations then hold only to that noise. The one
    // source measured so far, the discrete gradient's energy difference at
    // small increments, is removed at its root
    // (`corotated_energy_increment`); this stays as the guard. This
    // engine's own, declared.
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
            if since_best >= 5 {
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
        let jacobian = |d: &[f64]| jacobian_at(&u, &held, d);
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
    // Jacobian and preconditioner.
    let mut responses = Vec::new();
    let responding = Instant::now();
    if wants_responses && report.converged && !contacts.is_empty() {
        let held = branches(&u);
        let jacobian = |d: &[f64]| jacobian_at(&u, &held, d);
        let base = push_with(&u, Some(&held), None);
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
                let start = Instant::now();
                let (du, iterations) = gmres(
                    &jacobian,
                    &b,
                    (&precondition, &grid.active),
                    settings.gmres_tolerance,
                    settings,
                );
                report.timing.gmres += start.elapsed().as_secs_f64();
                report.gmres += iterations;
                responses.push(Response {
                    contact: i,
                    tangential,
                    du,
                });
            }
        }
    }
    report.timing.responses = responding.elapsed().as_secs_f64();
    report.timing.residual = residual_clock.get();
    report.timing.jacobian = jacobian_clock.get();
    report.timing.internal_evaluations = evaluations.get();
    report.timing.preconditioner_solve = precondition_clock.get();
    report.timing.preconditioner_build = banded_at - projected_at;
    let dissipated = if solids.iter().any(Elastic::is_viscous) {
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
    Solution {
        u,
        v_n,
        report,
        responses,
        dissipated,
    }
}
