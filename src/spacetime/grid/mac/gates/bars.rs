use super::*;

/// The impact of an elastic bar on a rigid ground, the first benchmark of
/// Doyen, Ern and Piperno, *Time-integration schemes for the finite
/// element dynamic Signorini problem*, SIAM J. Sci. Comput. 33 (2011),
/// section 3.1: a bar of length `L`, undeformed, falls at `v0` onto the
/// ground with no gravity, stays on it `2 L / c0` while a shock crosses it
/// twice, and leaves undeformed at `v0` (their eqs. 26 and 27: the
/// displacement, and the contact pressure `E v0 / c0` throughout the
/// contact). Here a column `width` cells wide with Poisson's ratio 0
/// (`lambda = 0`), whose free sides then carry no lateral stress: the 1D
/// bar, uniaxial, with `E = 2 mu` and `c0^2 = E / rho`. At the small strain
/// `v0 / c0` (`EMERGE_PROBE_BAR_STRAIN`, default 1e-3) the engine's
/// nonlinear energy departs from linear elasticity by about that strain, a
/// declared difference.
///
/// The step is the total Lagrangian conserving one with walls in contact
/// (`total_lagrangian::step_total_lagrangian_contact`), Gauss 3x3 per half cell,
/// the ground a wall at the bottom face's contact points (`width + 2` of
/// them: its ends and the middles of its knot spans, as many as the
/// normal component's coefficients there). By default the contact
/// boundary is massless: those coefficients, whose support holds only the
/// bottom knot span, carry no inertia (Doyen, Ern and Piperno section 7);
/// `EMERGE_PROBE_BAR_MASS=standard` keeps the standard mass, their
/// midpoint with implicit contact (Discretization 4.4).
///
/// Measured against the exact solution at every step's end: the
/// displacement's RMS error over the bar, over `v0 tau_w` (the largest
/// contact displacement, `tau_w = L / c0`); the velocity's, over `v0`; the
/// mean push over the contact steps against `E v0 / c0`; the contact's
/// steps against `2 tau_w`; the take-off velocity (momentum over mass)
/// against `v0`; the energy (`1/2 v M* v` plus the stored energy) at the
/// end against the start, the walls' work, and the vibration left
/// (energy less the centre of mass's kinetic energy; exact: none).
/// Realism is one less each relative error.
fn bar_impact_run(courant: Option<f64>, cells: usize) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    // Medium clay's stiffness and density, a 1 m bar.
    let (e, rho) = (
        float("EMERGE_PROBE_BAR_E", 11.25e6),
        float("EMERGE_PROBE_BAR_RHO", 1700.0),
    );
    let length = float("EMERGE_PROBE_BAR_LENGTH", 1.0);
    let width = float("EMERGE_PROBE_BAR_WIDTH", 2.0) as usize;
    let strain = float("EMERGE_PROBE_BAR_STRAIN", 1e-3);
    // `EMERGE_PROBE_BAR_CONTACT=position|velocity|recovered`
    // (`total_lagrangian::ContactMode`); the last two use the standard mass.
    let mode = match std::env::var("EMERGE_PROBE_BAR_CONTACT").as_deref() {
        Ok("velocity") => crate::spacetime::total_lagrangian::ContactMode::Velocity,
        Ok("recovered") => crate::spacetime::total_lagrangian::ContactMode::Recovered,
        _ => crate::spacetime::total_lagrangian::ContactMode::Position,
    };
    let standard = std::env::var("EMERGE_PROBE_BAR_MASS").as_deref() == Ok("standard")
        || mode != crate::spacetime::total_lagrangian::ContactMode::Position;
    let cell_m = length / cells as f64;
    // Per unit density, cells and seconds; `lambda = 0`, so `bulk = mu`.
    // `EMERGE_PROBE_BAR_CONFINED`
    // holds every horizontal coefficient (rollers on the long sides: 1D
    // motion, uniaxial strain), so `EMERGE_PROBE_BAR_NU` (Poisson's ratio,
    // 0 by default) has the exact 1D solution with the P-wave modulus
    // `lambda + 2 mu = E (1 - nu) / ((1 + nu)(1 - 2 nu))`; at `nu = 0` the
    // free bar's own numbers, bit for bit.
    let confined = std::env::var_os("EMERGE_PROBE_BAR_CONFINED").is_some();
    let nu = float("EMERGE_PROBE_BAR_NU", 0.0);
    assert!(
        confined || nu == 0.0,
        "a free bar at nu > 0 has no exact solution: set EMERGE_PROBE_BAR_CONFINED"
    );
    if confined {
        println!("knob active: confined bar, nu {nu}");
    }
    crate::spacetime::total_lagrangian::hooks::set_held_axis(confined.then_some(0));
    let modulus = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let c0 = (modulus / rho).sqrt() / cell_m;
    let mu = 0.5 * c0 * c0 * (1.0 - 2.0 * nu) / (1.0 - nu);
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, mu / (1.0 - 2.0 * nu));
    let side = cells as f64;
    let tau = side / c0;
    let v0 = strain * c0;
    let frame = f64::from(FRAME);
    let dt = courant.map_or(frame, |nu| nu / c0);
    // The bottom face on a knot of the normal component's lattice (its
    // coefficients sit on the faces `y = j`, its knots half way), the
    // first impact inside the third step, at the fraction
    // `EMERGE_PROBE_BAR_PHASE` (default 0.37) of it.
    // `EMERGE_PROBE_BAR_OFFSET=dx,dy`: the whole scene, bar and ground,
    // moved by a fraction of a cell (the answer's dependence on where the
    // bar sits on the grid; the exact one has none).
    let offset = std::env::var("EMERGE_PROBE_BAR_OFFSET")
        .ok()
        .and_then(|v| {
            let mut parts = v.split(',').map(|s| s.trim().parse::<f64>().ok());
            Some((parts.next()??, parts.next()??))
        })
        .unwrap_or((0.0, 0.0));
    let (x0, bottom) = (4.0 + offset.0, 4.5 + offset.1);
    let t1 = (2.0 + float("EMERGE_PROBE_BAR_PHASE", 0.37)) * dt;
    let gap = v0 * t1;
    let floor = bottom - gap;
    let t2 = t1 + 2.0 * tau;
    let end = t2 + tau.max(3.0 * dt);
    let steps = (end / dt).ceil() as usize;
    let layout = MacLayout::new(width + 8, cells + 8, 1.0);
    let (nodes, node_weights) = gauss_legendre(3);
    let (mut reference, mut v0s): (Vec<DVec2>, Vec<f64>) = (0..2 * cells)
        .flat_map(|j| (0..2 * width).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            let (nodes, node_weights) = (&nodes, &node_weights);
            (0..3).flat_map(move |b| {
                (0..3).map(move |a| {
                    (
                        DVec2::new(
                            x0 + (i as f64 + nodes[a]) * 0.5,
                            bottom + (j as f64 + nodes[b]) * 0.5,
                        ),
                        0.25 * node_weights[a] * node_weights[b],
                    )
                })
            })
        })
        .unzip();
    let body = reference.len();
    // The contact points, of no mass and no volume.
    let mut along = vec![x0];
    along.extend((0..width).map(|i| x0 + i as f64 + 0.5));
    along.push(x0 + width as f64);
    for &xc in &along {
        reference.push(DVec2::new(xc, bottom));
        v0s.push(0.0);
    }
    let n = reference.len();
    let mass = v0s.clone();
    let mut solids = vec![solid; n];
    // `EMERGE_PROBE_THETA=cell`: Theta per grid cell (the cells of the
    // reference positions, in cells), as the TL solids path's `theta_cells`.
    if std::env::var("EMERGE_PROBE_THETA").as_deref() == Ok("cell") {
        let mut index = std::collections::HashMap::new();
        let of: Vec<usize> = reference
            .iter()
            .map(|q| {
                let next = index.len();
                *index
                    .entry((q.x.floor() as i64, q.y.floor() as i64))
                    .or_insert(next)
            })
            .collect();
        println!("knob active: Theta per cell, {} cells", index.len());
        crate::spacetime::total_lagrangian::hooks::set_cells(Some((of, index.len())));
    }
    // The normal component's coefficients on the face `y = 4`: support
    // `[2.5, 5.5]`, of which the bar holds only the knot span `[4.5, 5.5]`.
    let row = (bottom - 0.5) as usize;
    // `EMERGE_PROBE_BAR_MASSLESS_ROWS=n`: the normal component's `n` bottom rows
    // massless, with any contact mode; the removed share of the normal
    // mass printed from the plain B-splines before the first step.
    let rows = std::env::var("EMERGE_PROBE_BAR_MASSLESS_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(rows) = rows {
        let spline = |t: f64| match t.abs() {
            a if a < 0.5 => 0.75 - a * a,
            a if a < 1.5 => 0.5 * (1.5 - a).powi(2),
            _ => 0.0,
        };
        let removed: f64 = (0..body)
            .map(|p| {
                mass[p]
                    * (row..row + rows)
                        .map(|j| spline(reference[p].y - j as f64))
                        .sum::<f64>()
            })
            .sum();
        let total: f64 = mass[..body].iter().sum();
        println!(
            "knob active: {rows} massless bottom rows, {:.4} of the normal mass removed",
            removed / total
        );
    }
    let label = match rows {
        Some(rows) => format!("{rows} massless bottom rows"),
        None if standard => "standard".to_string(),
        None => "massless contact boundary".to_string(),
    };
    let rows = rows.unwrap_or(usize::from(!standard));
    crate::spacetime::total_lagrangian::hooks::set_massless((rows > 0).then(|| {
        (x0 as usize - 1..=x0 as usize + width)
            .flat_map(|i| (row..row + rows).map(move |j| (1, i, j)))
            .collect()
    }));
    // `EMERGE_PROBE_CONTACT_TOL`: the contacts' gap tolerance, cells.
    let mut settings = crate::spacetime::total_lagrangian::ConservingSettings::default();
    settings.contact_tolerance = float("EMERGE_PROBE_CONTACT_TOL", settings.contact_tolerance);
    // `EMERGE_PROBE_FORECAST`: hold contacts from the step their gap closes.
    settings.forecast_activation = std::env::var_os("EMERGE_PROBE_FORECAST").is_some();
    let share = crate::spacetime::total_lagrangian::massive_share(
        &layout,
        &reference,
        (&mass, &v0s),
        settings.occupation,
    );
    let mut x = reference.clone();
    // Uniform `-v0` on the massive coefficients.
    let mut vel: Vec<DVec2> = share.iter().map(|s| DVec2::new(0.0, -v0 * s[1])).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let planes = [crate::spacetime::total_lagrangian::Plane {
        axis: 1,
        at: floor,
        inward: 1.0,
    }];
    // The augmented Lagrangian's penalty on the contact points, an
    // iteration parameter: `EMERGE_PROBE_BARS_PENALTY` (default 100) times
    // the column's P-wave stiffness over one cell, so each Uzawa update
    // closes most of the gap left.
    let factor = float("EMERGE_PROBE_BARS_PENALTY", 100.0);
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < body { 0.0 } else { factor * 2.0 * mu })
        .collect();
    let mut cache = None;
    // The exact solution (their eq. 26) at height `s` above the bottom.
    let exact = |s: f64, t: f64| -> (f64, f64) {
        if t <= t1 {
            (gap - v0 * t, -v0)
        } else if t <= t2 {
            let r = t - t1;
            let reach = tau - (r - tau).abs();
            if s / c0 < reach {
                (-v0 * s / c0, 0.0)
            } else {
                (-v0 * reach, if r < tau { -v0 } else { v0 })
            }
        } else {
            (v0 * (t - t2), v0)
        }
    };
    let volume: f64 = v0s.iter().sum();
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum()
    };
    let e0 = energy(&vel, &f);
    // No gravity: the step tolerance's energy is the start's kinetic and
    // stored energy alone.
    let tolerance = crate::spacetime::total_lagrangian::tolerance_energy(
        e0,
        DVec2::ZERO,
        std::iter::empty::<(f64, &[DVec2])>(),
    );
    let (mut worst_u, mut worst_v) = (0.0f64, 0.0f64);
    let (mut first, mut last) = (None::<f64>, None::<f64>);
    let (mut push_sum, mut push_sq, mut pushes) = (0.0f64, 0.0f64, 0usize);
    let (mut work, mut least_work, mut updates) = (0.0f64, 0.0f64, 0.0f64);
    let (mut newton, mut gmres, mut passes, mut most_passes, mut unconverged, mut cost) =
        (0u64, 0u64, 0u64, 0u32, 0u32, 0.0f64);
    // Why a step is unconverged:
    // stagnated, or at Newton's iteration cap, and the worst final residual
    // over the tolerance.
    let (mut stagnated, mut capped, mut worst_residual) = (0u32, 0u32, 0.0f64);
    let mut deepest = 0.0f64;
    let pressure = c0 * v0;
    // `EMERGE_PROBE_BAR_ADAPTIVE=tol`: the step chosen by its error
    // (`total_lagrangian::controlled_step`), from the frame's step at most.
    let control = std::env::var("EMERGE_PROBE_BAR_ADAPTIVE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(
            |tolerance| crate::spacetime::total_lagrangian::StepControl {
                tolerance,
                safety: 0.9,
                growth: 10.0,
                largest: f64::from(FRAME),
                // `EMERGE_PROBE_SUBSTEPS=2,4,6`: the tries' counts, KSD's 1, 2, 3
                // by default.
                substeps: match std::env::var("EMERGE_PROBE_SUBSTEPS").as_deref() {
                    Ok("2,4,6") => [2, 4, 6],
                    _ => [1, 2, 3],
                },
                // `EMERGE_PROBE_ESTIMATOR=acary`: Acary's estimate.
                estimator: match std::env::var("EMERGE_PROBE_ESTIMATOR").as_deref() {
                    Ok("acary") => crate::spacetime::total_lagrangian::Estimator::Acary,
                    _ => crate::spacetime::total_lagrangian::Estimator::Klapproth,
                },
            },
        );
    let contact_set = crate::spacetime::total_lagrangian::ContactSet {
        planes: &planes,
        pairs: &[],
        outlines: &[],
        corners: &[],
        friction: 0.0,
    };
    let (mut step_length, mut accepted, mut rejected, mut solves) = (dt, 0u32, 0u32, 0u32);
    let (mut after_release, mut shortest, mut solved, mut push_time) =
        (0u32, f64::INFINITY, 0usize, 0.0f64);
    let mut t = 0.0;
    let mut taken = 0usize;
    // Chatter: changes of the number of active contacts from one kept
    // substep to the next, and sign flips of the contact points' normal
    // velocity from one kept step to the next while they are beyond the
    // ground.
    let (mut last_active, mut set_changes) = (0usize, 0u32);
    let mut last_sign = vec![0.0f64; n];
    // The gap the bottom face keeps above the ground while the ground
    // pushes it (a forecast contact's standoff).
    let mut standoff = 0.0f64;
    let mut flips = 0u32;
    while if control.is_some() {
        t < end - 1e-12
    } else {
        taken < steps
    } {
        taken += 1;
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0s,
        };
        let (reports, step) = match &control {
            Some(control) => {
                let r = crate::spacetime::total_lagrangian::controlled_step(
                    &layout,
                    particles,
                    (&reference, &mut solids),
                    (DVec2::ZERO, &mut step_length),
                    (contact_set, &penalty, &settings),
                    &mut cache,
                    (control, tolerance.sqrt()),
                );
                accepted += 1;
                rejected += r.rejected;
                solves += r.solves;
                shortest = shortest.min(r.dt / r.kept.len() as f64);
                if !r.contact && first.is_some() {
                    after_release += 1;
                }
                (r.kept, r.dt)
            }
            None => (
                vec![
                    crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
                        &layout,
                        particles,
                        &reference,
                        &solids,
                        (dt, mode, DVec2::ZERO),
                        (contact_set, &penalty, &settings),
                        &mut cache,
                    ),
                ],
                dt,
            ),
        };
        let sub = step / reports.len() as f64;
        let pushed = reports.iter().any(|r| r.push > 0.0);
        for report in reports {
            t += sub;
            solved += 1;
            newton += u64::from(report.newton);
            gmres += u64::from(report.gmres);
            passes += u64::from(report.passes);
            most_passes = most_passes.max(report.passes);
            unconverged += u32::from(!report.converged);
            if !report.converged {
                stagnated += u32::from(report.stagnated);
                capped += u32::from(report.newton >= settings.max_newton);
                worst_residual = worst_residual.max(report.residual / settings.newton_tolerance);
            }
            cost += report.phases.iter().sum::<f64>();
            work += report.work.0;
            updates += report.work.1;
            least_work = least_work.min(report.work.0);
            if report.contacts != last_active {
                set_changes += 1;
                last_active = report.contacts;
            }
            if report.push > 0.0 {
                first.get_or_insert(t - sub);
                last = Some(t);
                let p = report.push / width as f64;
                push_sum += p * sub;
                push_sq += p * p * sub;
                push_time += sub;
                pushes += 1;
            }
        }
        for q in &x[body..] {
            deepest = deepest.max(floor - q.y);
        }
        if pushed {
            let lowest = x[body..].iter().map(|q| q.y).fold(f64::INFINITY, f64::min);
            standoff = standoff.max(lowest - floor);
        }
        for p in body..n {
            let sign = vel[p].y.signum();
            if x[p].y < floor && last_sign[p] != 0.0 && sign != last_sign[p] {
                flips += 1;
            }
            last_sign[p] = sign;
        }
        let (mut du, mut dv) = (0.0, 0.0);
        for p in 0..body {
            let (u, v) = exact(reference[p].y - bottom, t);
            du += v0s[p] * (x[p].y - reference[p].y - (u - gap)).powi(2)
                + v0s[p] * (x[p].x - reference[p].x).powi(2);
            dv += v0s[p] * ((vel[p].y - v).powi(2) + vel[p].x.powi(2));
        }
        worst_u = worst_u.max((du / volume).sqrt() / (v0 * tau));
        worst_v = worst_v.max((dv / volume).sqrt() / v0);
    }
    let momentum: f64 = (0..body).map(|p| mass[p] * vel[p].y).sum();
    // A uniform velocity on the massive coefficients: its momentum and its
    // kinetic energy per unit velocity and squared velocity.
    let carried: f64 = (0..body).map(|p| mass[p] * share[p][1]).sum();
    let rigid: f64 = (0..body).map(|p| mass[p] * share[p][1] * share[p][1]).sum();
    let take_off = momentum / carried;
    let e_end = energy(&vel, &f);
    let vibration = (e_end - 0.5 * rigid * take_off * take_off) / e0;
    let contact = match (first, last) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    };
    let mean_push = push_sum / push_time.max(f64::MIN_POSITIVE);
    let spread = (push_sq / push_time.max(f64::MIN_POSITIVE) - mean_push * mean_push)
        .max(0.0)
        .sqrt();
    let realism = |error: f64| 100.0 * (1.0 - error).max(0.0);
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
    format!(
        "bar impact, {mode:?} contact, {} mass, h = L/{cells} (L {length} m, width {width} cells), {}, c0 {:.1} m/s, strain {strain:e}: contact {:.2} steps exact; {steps} steps to t = {t:.4} s\n  contact {:.4} s for {:.4} exact (realism {:.1} %, resolution one step {:.4} s); push mean {:.4} of E v0/c0 (realism {:.1} %), spread {:.3} of it, over {pushes} steps; take-off {:.5} of v0 (realism {:.1} %)\n  energy end {:.6} of the start (realism {:.2} %), walls' work {:+.3e} of it (least step {:+.2e}), velocity updates {:+.3e}, balance {:+.2e}; vibration left {:.4} of the start\n  worst displacement error {:.3e} of v0 tau_w (realism {:.1} %), worst velocity error {:.3e} of v0 (realism {:.1} %); deepest end below the ground {deepest:.2e} cells, largest standoff while pushed {standoff:.2e} cells\n  Newton {:.2}, GMRES {:.1}, passes {:.2} (at most {most_passes}) per step, unconverged {unconverged}; {:.2} s per simulated s",
        label,
        match courant {
            Some(nu) => format!("Courant {nu} (dt {dt:.3e} s)"),
            None => format!("frame step (Courant {:.1})", c0 * dt),
        },
        c0 * cell_m,
        2.0 * tau / dt,
        contact,
        2.0 * tau,
        realism((contact - 2.0 * tau).abs() / (2.0 * tau)),
        dt,
        mean_push / pressure,
        realism((mean_push / pressure - 1.0).abs()),
        spread / pressure,
        take_off / v0,
        realism((take_off / v0 - 1.0).abs()),
        e_end / e0,
        realism((e_end / e0 - 1.0).abs()),
        work / e0,
        least_work / e0,
        updates / e0,
        (e_end - e0 - work - updates) / e0,
        vibration,
        worst_u,
        realism(worst_u),
        worst_v,
        realism(worst_v),
        newton as f64 / solved as f64,
        gmres as f64 / solved as f64,
        passes as f64 / solved as f64,
        cost / t
    ) + &format!(
        "
  unconverged steps: stagnated {stagnated}, at the iteration cap {capped}, worst final residual {worst_residual:.2e} of the tolerance"
    ) + &match control {
        Some(c) => format!(
            "\n  controlled, tol {} (error energy at most {:.1e} of the start): {accepted} steps kept, {rejected} rejected, {solves} solves, {after_release} after the release, shortest {shortest:.3e} s ({:.1} per contact); active set changed {set_changes} times over the kept substeps, contact points' normal velocity flipped {flips} times beyond the ground",
            c.tolerance,
            (c.safety * c.tolerance).powi(2),
            2.0 * tau / shortest
        ),
        None => String::new(),
    }
}

/// The bar's impact (`bar_impact_run`) over `EMERGE_PROBE_BAR_COURANT`
/// (comma separated Courant numbers `c0 dt / h`, or `frame` for the 60 Hz
/// frame's step; default `1.5,frame`) and `EMERGE_PROBE_BAR_CELLS`
/// (default 16). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_bar_impact() {
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    for cells in list("EMERGE_PROBE_BAR_CELLS", "16") {
        for step in list("EMERGE_PROBE_BAR_COURANT", "1.5,frame") {
            let Ok(cells) = cells.parse() else { continue };
            let courant = if step == "frame" {
                None
            } else {
                match step.parse() {
                    Ok(nu) => Some(nu),
                    Err(_) => continue,
                }
            };
            println!("{}", bar_impact_run(courant, cells));
        }
    }
}

/// Two identical elastic bars in longitudinal impact, the body-body
/// counterpart of `bar_impact_run`, held by the contact condition in
/// velocity between two bodies (Laursen and Chawla, IJNME 40 (1997), eqs.
/// 46-48 and 51-52: the gap rate from both bodies' midpoint velocities;
/// `total_lagrangian::step_total_lagrangian_contact` with pairs). Bar A, above,
/// falls at `v0` onto bar B, below. `EMERGE_PROBE_BARS_CASE`:
/// - `rest` (default): B at rest. Exact for identical bars (Laursen and
///   Chawla sec. 6.1, Cirak and West sec. 6.1): after a contact of `2 L /
///   c0`, A stops and B leaves at `v0`, both undeformed.
/// - `symmetric`: B rises at `v0`. By symmetry the interface stays still,
///   so each bar behaves as `bar_impact_run`'s bar on a wall: the two must
///   agree, a check of the pair contact against the wall contact.
///
/// `EMERGE_PROBE_BARS_RATIO` (default 1) makes A that many times B's length,
/// B at rest. For equal impedances the solution follows from the
/// characteristics (derived, checked against Kakouris, Chatzis and
/// Triantafyllou, arXiv 2403.13534, sec. 4.2, who state the short striker's
/// case citing Wriggers 2006, not read): the interface moves at `v0 / 2`
/// while the push lasts, `2 min(L_A, L_B) / c0`. A longer striker leaves
/// B rigid at `v0` and keeps the mean `v0 (L_A - L_B) / L_A`; a shorter one
/// stops dead and leaves B the mean `v0 L_A / L_B` (the two stay touching
/// without force until `2 L_B / c0`).
///
/// The two bodies' reference configurations are apart (their B-splines
/// share no coefficient), and A starts displaced next to B: on the
/// reference grid they meet only through the contact multipliers. Same
/// material, sizes and units as `bar_impact_run` (`EMERGE_PROBE_BAR_*`),
/// standard mass, the impact inside the third step at
/// `EMERGE_PROBE_BAR_PHASE`.
fn two_bar_impact_run(courant: Option<f64>, cells: usize) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (e, rho) = (
        float("EMERGE_PROBE_BAR_E", 11.25e6),
        float("EMERGE_PROBE_BAR_RHO", 1700.0),
    );
    let length = float("EMERGE_PROBE_BAR_LENGTH", 1.0);
    let width = float("EMERGE_PROBE_BAR_WIDTH", 2.0) as usize;
    let strain = float("EMERGE_PROBE_BAR_STRAIN", 1e-3);
    let symmetric = std::env::var("EMERGE_PROBE_BARS_CASE").as_deref() == Ok("symmetric");
    let ratio = float("EMERGE_PROBE_BARS_RATIO", 1.0);
    assert!(
        !symmetric || ratio == 1.0,
        "the symmetric case needs equal bars"
    );
    let cells_a = (cells as f64 * ratio).round() as usize;
    let cell_m = length / cells as f64;
    let c0 = (e / rho).sqrt() / cell_m;
    let mu = 0.5 * c0 * c0;
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, mu);
    let side = cells as f64;
    let side_a = cells_a as f64;
    // The push lasts `2 min(L_A, L_B) / c0`.
    let tau = side.min(side_a) / c0;
    let v0 = strain * c0;
    let frame = f64::from(FRAME);
    let dt = courant.map_or(frame, |nu| nu / c0);
    // Closing speed: v0, or 2 v0 when B rises too.
    let closing = if symmetric { 2.0 * v0 } else { v0 };
    let t1 = (2.0 + float("EMERGE_PROBE_BAR_PHASE", 0.37)) * dt;
    let gap = closing * t1;
    let end = t1 + 2.0 * side.max(side_a) / c0 + tau.max(3.0 * dt);
    let steps = (end / dt).ceil() as usize;
    // B's reference above the grid's bottom, A's reference `apart` cells
    // above B's; A starts shifted down to `gap` above B.
    let (x0, bottom, apart) = (4.0, 4.5, 6.0);
    let top_b = bottom + side;
    let bottom_a = top_b + apart;
    let layout = MacLayout::new(width + 8, cells + cells_a + apart as usize + 9, 1.0);
    let (nodes, node_weights) = gauss_legendre(3);
    let block = |base: f64, rows: usize| -> Vec<(DVec2, f64)> {
        let mut out = Vec::new();
        for j in 0..2 * rows {
            for i in 0..2 * width {
                for b in 0..3 {
                    for a in 0..3 {
                        out.push((
                            DVec2::new(
                                x0 + (i as f64 + nodes[a]) * 0.5,
                                base + (j as f64 + nodes[b]) * 0.5,
                            ),
                            0.25 * node_weights[a] * node_weights[b],
                        ));
                    }
                }
            }
        }
        out
    };
    let (mut reference, mut v0s): (Vec<DVec2>, Vec<f64>) = block(bottom, cells).into_iter().unzip();
    let body_b = reference.len();
    let (ra, va): (Vec<DVec2>, Vec<f64>) = block(bottom_a, cells_a).into_iter().unzip();
    reference.extend(ra);
    v0s.extend(va);
    let body_a = reference.len();
    // The contact points: B's top face and A's bottom face, at the same
    // abscissae (the bars stay aligned), of no mass and no volume.
    let mut along = vec![x0];
    along.extend((0..width).map(|i| x0 + i as f64 + 0.5));
    along.push(x0 + width as f64);
    let first_b = reference.len();
    for &xc in &along {
        reference.push(DVec2::new(xc, top_b));
        v0s.push(0.0);
    }
    let first_a = reference.len();
    for &xc in &along {
        reference.push(DVec2::new(xc, bottom_a));
        v0s.push(0.0);
    }
    let n = reference.len();
    let in_a = |p: usize| (body_b..body_a).contains(&p) || p >= first_a;
    let shift = DVec2::new(0.0, gap - apart);
    let mut x: Vec<DVec2> = (0..n)
        .map(|p| reference[p] + if in_a(p) { shift } else { DVec2::ZERO })
        .collect();
    let mass = v0s.clone();
    let solids = vec![solid; n];
    let mut vel: Vec<DVec2> = (0..n)
        .map(|p| {
            if in_a(p) {
                DVec2::new(0.0, -v0)
            } else if symmetric {
                DVec2::new(0.0, v0)
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let mut f = vec![DMat2::IDENTITY; n];
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
    let settings = crate::spacetime::total_lagrangian::ConservingSettings::default();
    // A above B: their separation `y_A - y_B` stays non-negative.
    let pairs: Vec<crate::spacetime::total_lagrangian::Pair> = (0..along.len())
        .map(|i| {
            (
                first_a + i,
                first_b + i,
                crate::spacetime::total_lagrangian::Plane {
                    axis: 1,
                    at: 0.0,
                    inward: 1.0,
                },
            )
        })
        .collect();
    // The augmented Lagrangian's penalty, an iteration parameter (the
    // converged multipliers do not depend on it): `EMERGE_PROBE_BARS_PENALTY`
    // times the P-wave stiffness over one cell (default 10: at 100 the
    // Newton floor of the Jacobian-free solve reached the tolerance on 1-2
    // steps, which stopped Uzawa early and left work in the contact;
    // measured, 2026-10-13).
    let factor = float("EMERGE_PROBE_BARS_PENALTY", 10.0);
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < first_b { 0.0 } else { factor * 2.0 * mu })
        .collect();
    let mut cache = None;
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum()
    };
    let momentum = |vel: &[DVec2]| -> DVec2 { (0..n).map(|p| mass[p] * vel[p]).sum() };
    let mean = |vel: &[DVec2], range: std::ops::Range<usize>| -> f64 {
        let total: f64 = range.clone().map(|p| mass[p]).sum();
        range.map(|p| mass[p] * vel[p].y).sum::<f64>() / total
    };
    let e0 = energy(&vel, &f);
    let p0 = momentum(&vel);
    let scale = (0..n).map(|p| mass[p] * vel[p].length()).sum::<f64>();
    let (mut first, mut last) = (None::<f64>, None::<f64>);
    let (mut work, mut deepest, mut newton, mut gmres, mut unconverged, mut cost) =
        (0.0f64, 0.0f64, 0u64, 0u64, 0u32, 0.0f64);
    let mut worst_momentum = 0.0f64;
    let history = std::env::var("EMERGE_PROBE_BARS_HISTORY").is_ok();
    let (mut released, mut recontacts, mut late_impulse) = (false, 0usize, 0.0f64);
    let mut t = 0.0;
    // With `EMERGE_PROBE_BARS_TRACE` an unconverged step's Newton
    // iterations are printed, solve by solve (`total_lagrangian::hooks`).
    let trace = std::env::var("EMERGE_PROBE_BARS_TRACE").is_ok();
    for _ in 0..steps {
        if trace {
            crate::spacetime::total_lagrangian::hooks::start(0);
        }
        let report = crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
            &layout,
            crate::spacetime::total_lagrangian::Particles {
                x: &mut x,
                v: &mut vel,
                f: &mut f,
                mass: &mass,
                v0: &v0s,
            },
            &reference,
            &solids,
            (
                dt,
                crate::spacetime::total_lagrangian::ContactMode::Velocity,
                DVec2::ZERO,
            ),
            (
                crate::spacetime::total_lagrangian::ContactSet {
                    planes: &[],
                    pairs: &pairs,
                    outlines: &[],
                    corners: &[],
                    friction: 0.0,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
        );
        if trace {
            let solves = crate::spacetime::total_lagrangian::hooks::finish();
            if !report.converged {
                for (k, lines) in solves.iter().enumerate() {
                    println!("    solve {k}:");
                    for line in lines {
                        println!("      {line}");
                    }
                }
            }
        }
        if !report.converged {
            println!(
                "  unconverged at t {:.4}: residual {:.2e} after {} Newton, {} GMRES, {} passes, {} contacts, start residual {:.2e}",
                t + dt,
                report.residual,
                report.newton,
                report.gmres,
                report.passes,
                report.contacts,
                report.start_residual
            );
        }
        t += dt;
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        work += report.work.0;
        // The first contiguous run of pushing steps is the impact; any later
        // push is a re-contact, counted apart with its impulse.
        if report.push > 0.0 {
            if first.is_none() {
                first = Some(t);
            }
            if released {
                recontacts += 1;
                late_impulse += report.push * dt;
            } else {
                last = Some(t);
            }
        } else if first.is_some() {
            released = true;
        }
        // With `EMERGE_PROBE_BARS_HISTORY`, every step's push (over the
        // exact `E v0 / (2 c0)` times the width) and the bars' mean speeds.
        if history {
            println!(
                "  t {t:.4}: push {:.4}, A {:+.4} v0, B {:+.4} v0, {} contacts, {} passes",
                report.push / (0.5 * c0 * v0 * width as f64),
                mean(&vel, body_b..body_a) / v0,
                mean(&vel, 0..body_b) / v0,
                report.contacts,
                report.passes
            );
        }
        for &(p, q, _) in &pairs {
            deepest = deepest.max(x[q].y - x[p].y);
        }
        worst_momentum = worst_momentum.max((momentum(&vel) - p0).length() / scale);
    }
    let contact = match (first, last) {
        (Some(a), Some(b)) => b - a + dt,
        _ => 0.0,
    };
    let (va_end, vb_end) = (mean(&vel, body_b..body_a), mean(&vel, 0..body_b));
    let (va_exact, vb_exact) = if symmetric {
        (v0, -v0)
    } else if side_a >= side {
        (-v0 * (side_a - side) / side_a, -v0)
    } else {
        (0.0, -v0 * side_a / side)
    };
    let e_end = energy(&vel, &f);
    let mass_a: f64 = (body_b..body_a).map(|p| mass[p]).sum();
    let mass_b: f64 = (0..body_b).map(|p| mass[p]).sum();
    let rigid = 0.5 * (mass_a * va_end * va_end + mass_b * vb_end * vb_end);
    let realism = |error: f64| 100.0 * (1.0 - error).max(0.0);
    format!(
        "two bars, {} case, L_A/L_B {ratio}, h = L_B/{cells} (L_B {length} m, width {width} cells), {}, c0 {:.1} m/s, strain {strain:e}: push {:.2} steps exact; {steps} steps to t = {t:.4} s\n  push {:.4} s for {:.4} exact (realism {:.1} %); {recontacts} re-contact steps after it, impulse {:.2e} of the exact push's\n  A leaves at {:+.5} v0 (exact {:+.4}), B at {:+.5} v0 (exact {:+.4}); transfer realism {:.1} %\n  energy end {:.6} of the start, contact work {:+.3e} of it, balance {:+.2e}; vibration left {:.4} of the start; momentum error at worst {:.2e} of the total |m v|\n  deepest crossing {deepest:.2e} cells ({:.2e} m); Newton {:.2}, GMRES {:.1} per step, unconverged {unconverged}; {:.2} s per simulated s",
        if symmetric { "symmetric" } else { "rest" },
        match courant {
            Some(nu) => format!("Courant {nu} (dt {dt:.3e} s)"),
            None => format!("frame step (Courant {:.1})", c0 * dt),
        },
        c0 * cell_m,
        2.0 * tau / dt,
        contact,
        2.0 * tau,
        realism((contact - 2.0 * tau).abs() / (2.0 * tau)),
        late_impulse / (0.5 * c0 * v0 * width as f64 * 2.0 * tau),
        va_end / v0,
        va_exact / v0,
        vb_end / v0,
        vb_exact / v0,
        realism(((va_end - va_exact).abs() + (vb_end - vb_exact).abs()) / (2.0 * v0)),
        e_end / e0,
        work / e0,
        (e_end - e0 - work) / e0,
        (e_end - rigid) / e0,
        worst_momentum,
        deepest * cell_m,
        newton as f64 / steps as f64,
        gmres as f64 / steps as f64,
        cost / t
    )
}

/// The two bars' impact (`two_bar_impact_run`) over
/// `EMERGE_PROBE_BAR_COURANT` (default `1.5,frame`) and
/// `EMERGE_PROBE_BAR_CELLS` (default 16). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_two_bar_impact() {
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    for cells in list("EMERGE_PROBE_BAR_CELLS", "16") {
        for step in list("EMERGE_PROBE_BAR_COURANT", "1.5,frame") {
            let Ok(cells) = cells.parse() else { continue };
            let courant = if step == "frame" {
                None
            } else {
                match step.parse() {
                    Ok(nu) => Some(nu),
                    Err(_) => continue,
                }
            };
            println!("{}", two_bar_impact_run(courant, cells));
        }
    }
}
