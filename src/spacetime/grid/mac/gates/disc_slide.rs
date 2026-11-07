use super::*;

/// Two identical elastic discs in a frictionless, off-centre collision,
/// held by the contact condition in velocity against each other's outline
/// (Laursen and Chawla, IJNME 40 (1997), eqs. 40-48: normal and closest
/// point in the step's midpoint configuration, which keeps both momenta;
/// `total_lagrangian::step_total_lagrangian_contact` with outlines). Disc A moves
/// at `v0` along `x` into disc B at rest, its centre `b` above B's line of
/// motion; A may spin (`EMERGE_PROBE_DISC_SPIN`, `omega R / v0`).
///
/// No closed form for a deformable 2D impact is used. What is checked is
/// what frictionless contact between circles implies (the push along the
/// line of centres, no torque about either centre): each disc keeps its
/// velocity component across that line and its spin; both momenta are
/// conserved exactly (the step's own claim). The normal restitution is
/// reported (to compare later with Gerl and Zippelius, arXiv
/// cond-mat/9808258, Fig. 1, which tends to 1 as `v0 / c -> 0`).
///
/// Parameters (`EMERGE_PROBE_DISC_*`): `RADIUS` cells (default 8), `B`
/// the offset over the diameter (default 0.5), `NU` Poisson's ratio
/// (default 0.3), `STRAIN` `v0` over the P-wave speed (default 1e-3),
/// `OUTLINE` points per cell of circumference (default 2), `PENALTY` (as
/// the bars', default 10), `MU` Coulomb's friction coefficient (default
/// 0), `CONTACT` (`off`: no contact, a control).
/// Medium clay's modulus and density, a 0.2 m disc.
fn disc_impact_run(courant: Option<f64>) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let radius = float("EMERGE_PROBE_DISC_RADIUS", 8.0);
    let offset = float("EMERGE_PROBE_DISC_B", 0.5);
    let nu = float("EMERGE_PROBE_DISC_NU", 0.3);
    let strain = float("EMERGE_PROBE_DISC_STRAIN", 1e-3);
    let spin_ratio = float("EMERGE_PROBE_DISC_SPIN", 0.0);
    let per_cell = float("EMERGE_PROBE_DISC_OUTLINE", 2.0);
    let factor = float("EMERGE_PROBE_DISC_PENALTY", 10.0);
    let friction = float("EMERGE_PROBE_DISC_MU", 0.0);
    let (e, rho) = (11.25e6f64, 1700.0f64);
    let cell_m = 0.1 / radius;
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
    let c_p = (lambda + 2.0 * mu).sqrt();
    let v0 = strain * c_p;
    let frame = f64::from(FRAME);
    let dt = courant.map_or(frame, |nu| nu / c_p);
    // Reference centres apart; A starts displaced left of B.
    let margin = 4.0;
    let centre_b = DVec2::splat(radius + margin);
    let centre_a_ref = DVec2::new(3.0 * radius + 3.0 * margin, radius + margin);
    let side = (4.0 * radius + 4.0 * margin).ceil() as usize;
    let layout = MacLayout::new(
        side + 2,
        (2.0 * radius + 2.0 * margin).ceil() as usize + 2,
        1.0,
    );
    let rise = 2.0 * radius * offset;
    let reach = (4.0 * radius * radius - rise * rise).sqrt();
    let t1 = (2.0 + float("EMERGE_PROBE_BAR_PHASE", 0.37)) * dt;
    let centre_a = DVec2::new(centre_b.x - reach - v0 * t1, centre_b.y + rise);
    let shift = centre_a - centre_a_ref;
    let (mut reference, mut v0s) = ring_quadrature(centre_b, (0.0, radius), side, 1);
    let body_b = reference.len();
    let (ra, va) = ring_quadrature(centre_a_ref, (0.0, radius), side, 1);
    reference.extend(ra);
    v0s.extend(va);
    let body_a = reference.len();
    // The outlines: points on each circle, counter-clockwise, of no mass.
    let count = (2.0 * std::f64::consts::PI * radius * per_cell).ceil() as usize;
    // `EMERGE_PROBE_DISC_INSET` cells: the outline drawn inside the circle
    // (a diagnostic, default 0).
    let inset = float("EMERGE_PROBE_DISC_INSET", 0.0);
    let outline = |centre: DVec2, start: usize| -> (Vec<DVec2>, Vec<usize>) {
        (0..count)
            .map(|k| {
                let angle = 2.0 * std::f64::consts::PI * k as f64 / count as f64;
                (
                    centre + (radius - inset) * DVec2::new(angle.cos(), angle.sin()),
                    start + k,
                )
            })
            .unzip()
    };
    let (points_b, outline_b) = outline(centre_b, reference.len());
    reference.extend(points_b);
    let first_outline_a = reference.len();
    let (points_a, outline_a) = outline(centre_a_ref, first_outline_a);
    reference.extend(points_a);
    v0s.resize(reference.len(), 0.0);
    let n = reference.len();
    let in_a = |p: usize| (body_b..body_a).contains(&p) || p >= first_outline_a;
    let mut x: Vec<DVec2> = (0..n)
        .map(|p| reference[p] + if in_a(p) { shift } else { DVec2::ZERO })
        .collect();
    let omega = spin_ratio * v0 / radius;
    // `EMERGE_PROBE_DISC_SHEAR`: A's velocity also carries a nonuniform,
    // asymmetric part (a diagnostic of the step's own angular momentum).
    let shear = float("EMERGE_PROBE_DISC_SHEAR", 0.0);
    let mut vel: Vec<DVec2> = (0..n)
        .map(|p| {
            if in_a(p) {
                let r = x[p] - centre_a;
                DVec2::new(v0, 0.0)
                    + omega * DVec2::new(-r.y, r.x)
                    + shear * v0 * DVec2::new(0.0, r.x * (r.x + r.y) / (radius * radius))
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let mass = v0s.clone();
    let solids = vec![solid; n];
    let mut f = vec![DMat2::IDENTITY; n];
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
    // `NEWTON` the Newton tolerance, `OCCUPATION` the extended B-splines'
    // `C_c` (`none`: the original B-splines), both as the defaults if unset.
    let mut settings = crate::spacetime::total_lagrangian::ConservingSettings::default();
    settings.newton_tolerance = float("EMERGE_PROBE_DISC_NEWTON", settings.newton_tolerance);
    match std::env::var("EMERGE_PROBE_DISC_OCCUPATION").as_deref() {
        Ok("none") => settings.occupation = None,
        Ok(value) => settings.occupation = value.parse().ok().or(settings.occupation),
        Err(_) => {}
    }
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < body_a { 0.0 } else { factor * c_p * c_p })
        .collect();
    // `EMERGE_PROBE_DISC_CONTACT=off`: no outlines, the discs pass
    // through each other (a control: the step's own momenta, no contact).
    let outlines = if std::env::var("EMERGE_PROBE_DISC_CONTACT").as_deref() == Ok("off") {
        Vec::new()
    } else {
        vec![outline_b.clone(), outline_a.clone()]
    };
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum()
    };
    let linear = |vel: &[DVec2]| -> DVec2 { (0..n).map(|p| mass[p] * vel[p]).sum() };
    let angular = |x: &[DVec2], vel: &[DVec2]| -> f64 {
        (0..n).map(|p| mass[p] * x[p].perp_dot(vel[p])).sum()
    };
    // A disc's centre, centre velocity and spin.
    let state =
        |x: &[DVec2], vel: &[DVec2], range: std::ops::Range<usize>| -> (DVec2, DVec2, f64) {
            let total: f64 = range.clone().map(|p| mass[p]).sum();
            let centre: DVec2 = range.clone().map(|p| mass[p] * x[p]).sum::<DVec2>() / total;
            let speed: DVec2 = range.clone().map(|p| mass[p] * vel[p]).sum::<DVec2>() / total;
            let (mut moment, mut inertia) = (0.0, 0.0);
            for p in range {
                let r = x[p] - centre;
                moment += mass[p] * r.perp_dot(vel[p] - speed);
                inertia += mass[p] * r.length_squared();
            }
            (centre, speed, moment / inertia)
        };
    let e0 = energy(&vel, &f);
    let (l0, j0) = (linear(&vel), angular(&x, &vel));
    let disc_mass: f64 = (0..body_b).map(|p| mass[p]).sum();
    let (l_scale, j_scale) = (disc_mass * v0, disc_mass * v0 * radius);
    let (_, _, spin_a0) = state(&x, &vel, body_b..body_a);
    let mut cache = None;
    // `EMERGE_PROBE_DISC_TRACE`: each step's change of the angular momentum
    // and how its solve ended.
    let trace = std::env::var("EMERGE_PROBE_DISC_TRACE").is_ok();
    let mut j_last = j0;
    crate::spacetime::total_lagrangian::hooks::watch_torque(trace);
    let (mut worst_l, mut worst_j, mut work, mut deepest) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut first, mut last) = (None::<f64>, None::<f64>);
    let (mut newton, mut gmres, mut unconverged, mut cost, mut steps) =
        (0u64, 0u64, 0u32, 0.0f64, 0usize);
    let quiet = (4.0 * radius / c_p).max(3.0 * dt);
    let cap = t1 + 60.0 * radius / c_p + 10.0 * dt;
    let mut t = 0.0;
    while t < cap {
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
                    pairs: &[],
                    outlines: &outlines,
                    corners: &[],
                    friction,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
        );
        t += dt;
        steps += 1;
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        work += report.work.0;
        if report.push > 0.0 {
            first.get_or_insert(t);
            last = Some(t);
        }
        worst_l = worst_l.max((linear(&vel) - l0).length() / l_scale);
        worst_j = worst_j.max((angular(&x, &vel) - j0).abs() / j_scale);
        if trace {
            let j_now = angular(&x, &vel);
            let (step_moment, push_moment) =
                crate::spacetime::total_lagrangian::hooks::take_torque().unwrap_or_default();
            println!(
                "  step {steps}: dJ {:+.3e} of M v0 R (from m x_mid x dv {:+.3e}, the push's share {:+.3e}; total {:+.3e}), push {:.3e}, Newton {} residual {:.2e} converged {}, projection {:.2e}, passes {}",
                (j_now - j_last) / j_scale,
                step_moment / j_scale,
                push_moment / j_scale,
                (j_now - j0) / j_scale,
                report.push,
                report.newton,
                report.residual,
                report.converged,
                report.projection,
                report.passes
            );
            j_last = j_now;
        }
        // Overlap: the outline points of A inside B's circle, and back.
        let (cb, _, _) = state(&x, &vel, 0..body_b);
        let (ca, _, _) = state(&x, &vel, body_b..body_a);
        for &p in &outline_a {
            deepest = deepest.max(radius - (x[p] - cb).length());
        }
        for &p in &outline_b {
            deepest = deepest.max(radius - (x[p] - ca).length());
        }
        if last.is_some_and(|l| t - l > quiet) {
            break;
        }
    }
    let normal = DVec2::new(reach, -rise) / (2.0 * radius);
    let tangent = DVec2::new(rise, reach) / (2.0 * radius);
    let (_, va_end, spin_a) = state(&x, &vel, body_b..body_a);
    let (_, vb_end, spin_b) = state(&x, &vel, 0..body_b);
    let restitution = (vb_end - va_end).dot(normal) / v0 / normal.x;
    let e_end = energy(&vel, &f);
    let rigid = 0.5 * disc_mass * (va_end.length_squared() + vb_end.length_squared());
    let contact = match (first, last) {
        (Some(a), Some(b)) => b - a + dt,
        _ => 0.0,
    };
    format!(
        "discs, R {radius} cells, b/2R {offset}, nu {nu}, spin {spin_ratio} v0/R, {}, c_p {:.1} m/s, strain {strain:e}, outline {count} points: {steps} steps to t = {t:.4} s, contact {:.4} s ({:.1} steps)\n  A across the line of centres {:+.5} of its exact {:+.4} v0, B {:+.2e} v0 (exact 0); normal restitution {:.5}; spin A {:+.3e} (start {:+.3e}), B {:+.3e} v0/R\n  linear momentum error at worst {worst_l:.2e} of M v0, angular {worst_j:.2e} of M v0 R; energy end {:.6} of the start, contact work {:+.3e} of it, balance {:+.2e}; vibration left {:.4}\n  deepest overlap {deepest:.2e} cells; Newton {:.2}, GMRES {:.1} per step, unconverged {unconverged}; {:.2} s per simulated s",
        match courant {
            Some(nu) => format!("Courant {nu} (dt {dt:.3e} s)"),
            None => format!("frame step (Courant {:.1})", c_p * dt),
        },
        c_p * cell_m,
        contact,
        contact / dt,
        va_end.dot(tangent) / (v0 * tangent.x),
        tangent.x,
        vb_end.dot(tangent) / v0,
        restitution,
        spin_a * radius / v0,
        spin_a0 * radius / v0,
        spin_b * radius / v0,
        e_end / e0,
        work / e0,
        (e_end - e0 - work) / e0,
        (e_end - rigid - 0.0) / e0,
        newton as f64 / steps.max(1) as f64,
        gmres as f64 / steps.max(1) as f64,
        cost / t
    )
}

/// The discs' collision (`disc_impact_run`) over `EMERGE_PROBE_BAR_COURANT`
/// (default `1.5,frame`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_disc_impact() {
    let steps = std::env::var("EMERGE_PROBE_BAR_COURANT").unwrap_or_else(|_| "1.5,frame".into());
    for step in steps.split(',').map(str::trim) {
        let courant = if step == "frame" {
            None
        } else {
            match step.parse() {
                Ok(nu) => Some(nu),
                Err(_) => continue,
            }
        };
        println!("{}", disc_impact_run(courant));
    }
}

/// A stiff block sliding on a rigid floor under gravity, Coulomb friction
/// on the contact condition in velocity (`total_lagrangian::step_total_lagrangian_contact`
/// with `ContactSet::friction`). The floor may be an incline of slope
/// `TILT` (`tan theta`, default 0), as gravity turned by `theta` against
/// a level floor, down the slope along `+x`. Its rigid limit is exact: the
/// block slows at `d = g (mu cos theta - sin theta)`, `v(t) = v0 - d t`;
/// with `d > 0` it stops at `t_s = v0 / d`, `v0^2 / (2 d)` from its start,
/// and then stays (static friction holds it, `mu >= tan theta`); with `d
/// < 0` it speeds up. The heat friction makes is its force times the path,
/// `mu g cos theta M s` (Acary and Collins-Craft, JTCAM 2025, eq. 73: the
/// friction's work is the mechanical energy lost, exactly); on the level,
/// `1/2 M v0^2`.
///
/// The block is `WIDTH` by `HEIGHT` cells (defaults 8 by 4) of 1 cm, Gauss
/// 3x3 per half cell as the bar's, with massless contact points on its
/// bottom face as the bar's (its ends and the middles of its knot spans);
/// the floor's plane a hair under that face, so it holds from the first
/// step. `E` (default 1 GPa, the rigid limit's stand-in), `RHO` (1700),
/// `NU` (0.3), `MU` the friction coefficient (0.3), `V0` m/s (1),
/// `PENALTY` (times `c_p^2`, default 1) under `EMERGE_PROBE_SLIDE_*`;
/// `STEPS` the steps per frame to run, a list (default `1,4`).
///
/// Measured: the centre's velocity against the exact one at every step
/// (largest error over the reference speed, `v0` or the exact end speed if
/// larger), the stop time against `t_s`, the path against the exact one,
/// the creep after the stop, the heat against the exact heat, the energy
/// balance (kinetic, stored, gravity's, heat) over `1/2 M` the reference
/// speed squared, the steps where friction made energy (its work
/// positive).
fn block_slide_run(per_frame: usize) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let width = float("EMERGE_PROBE_SLIDE_WIDTH", 8.0) as usize;
    let height = float("EMERGE_PROBE_SLIDE_HEIGHT", 4.0) as usize;
    let (e, rho, nu) = (
        float("EMERGE_PROBE_SLIDE_E", 1e9),
        float("EMERGE_PROBE_SLIDE_RHO", 1700.0),
        float("EMERGE_PROBE_SLIDE_NU", 0.3),
    );
    let friction = float("EMERGE_PROBE_SLIDE_MU", 0.3);
    let cell_m = 0.01;
    let g = 9.81 / cell_m;
    let v0 = float("EMERGE_PROBE_SLIDE_V0", 1.0) / cell_m;
    // At 10 GMRES could not go below 1e-3 while the block stuck (the
    // penalty's stiffness over the inertia, measured at 1 GPa); 1 solves.
    let factor = float("EMERGE_PROBE_SLIDE_PENALTY", 1.0);
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    // `EMERGE_PROBE_SLIDE_LAW=corotated`: the corotated material's law.
    let solid = if std::env::var("EMERGE_PROBE_SLIDE_LAW").as_deref() == Ok("corotated") {
        crate::spacetime::total_lagrangian::Elastic::corotated(lambda, mu)
    } else {
        crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu)
    };
    let c_p = (lambda + 2.0 * mu).sqrt();
    let dt = f64::from(FRAME) / per_frame as f64;
    let tilt = float("EMERGE_PROBE_SLIDE_TILT", 0.0);
    let (sine, cosine) = (
        tilt / (1.0 + tilt * tilt).sqrt(),
        1.0 / (1.0 + tilt * tilt).sqrt(),
    );
    let gravity = g * DVec2::new(sine, -cosine);
    let slowing = g * (friction * cosine - sine);
    // With `d > 0` and a start speed the block stops, then stays: a
    // quarter second more of it; otherwise 0.6 s.
    let stops = slowing > 0.0 && v0 > 0.0;
    let stop = if slowing > 0.0 {
        v0 / slowing
    } else {
        f64::INFINITY
    };
    // `DROP` cells of fall onto the floor first (default 0: resting on it
    // from the start) and `END` seconds of run (default as above).
    let drop = float("EMERGE_PROBE_SLIDE_DROP", 0.0);
    let end = float(
        "EMERGE_PROBE_SLIDE_END",
        if stops { stop + 0.25 } else { 0.6 },
    );
    let speed = |t: f64| -> f64 {
        if slowing > 0.0 {
            (v0 - slowing * t).max(0.0)
        } else {
            v0 - slowing * t
        }
    };
    let path = |t: f64| -> f64 {
        let t = t.min(stop);
        v0 * t - 0.5 * slowing * t * t
    };
    // At rest on a slope it holds, the speed it would reach with no
    // friction sets the scale.
    let drop_speed = (2.0 * g * float("EMERGE_PROBE_SLIDE_DROP", 0.0)).sqrt();
    let reference_speed = v0.max(speed(end).abs()).max(g * sine * end).max(drop_speed);
    let travel = path(end);
    let (x0, floor) = (4.0, 4.0);
    let bottom = floor + drop;
    let layout = MacLayout::new(
        (x0 + width as f64 + travel + 8.0).ceil() as usize,
        (height as f64 + drop + 8.0).ceil() as usize,
        1.0,
    );
    let (nodes, node_weights) = gauss_legendre(3);
    let (mut reference, mut v0s): (Vec<DVec2>, Vec<f64>) = (0..2 * height)
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
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
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
    let mut vel: Vec<DVec2> = share.iter().map(|s| DVec2::new(v0 * s[0], 0.0)).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let planes = [crate::spacetime::total_lagrangian::Plane {
        axis: 1,
        at: if drop > 0.0 { floor } else { bottom + 1e-9 },
        inward: 1.0,
        speed: 0.0,
    }];
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < body { 0.0 } else { factor * c_p * c_p })
        .collect();
    let total: f64 = mass.iter().sum();
    let centre = |x: &[DVec2], vel: &[DVec2]| -> (DVec2, DVec2) {
        (
            (0..n).map(|p| mass[p] * x[p]).sum::<DVec2>() / total,
            (0..n).map(|p| mass[p] * vel[p]).sum::<DVec2>() / total,
        )
    };
    let energy = |x: &[DVec2], vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| {
                0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p])
                    - mass[p] * gravity.dot(x[p])
            })
            .sum()
    };
    let gyration = |x: &[DVec2]| -> f64 {
        let (c, _) = centre(x, x);
        ((0..body)
            .map(|p| mass[p] * (x[p] - c).length_squared())
            .sum::<f64>()
            / total)
            .sqrt()
    };
    let gyration0 = gyration(&x);
    let (mut worst_gyration, mut nan) = (0.0f64, false);
    let (mut impact, mut rebound, mut touched) = (0.0f64, 0.0f64, false);
    let (start, start_v) = centre(&x, &vel);
    let start_x = start.x;
    let _ = start_v;
    let kinetic0 = 0.5 * total * reference_speed * reference_speed;
    let e0 = energy(&x, &vel, &f);
    // The step tolerance's energy: the start's kinetic and stored energy
    // and gravity's work over the block's own extent (`tolerance_energy`),
    // never `e0`, whose potential is measured from the layout's corner.
    let tolerance = crate::spacetime::total_lagrangian::tolerance_energy(
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum(),
        gravity,
        [(total, &x[..n])],
    );
    let mut cache = None;
    let (mut heat, mut worst_v, mut made, mut worst_made) = (0.0f64, 0.0f64, 0usize, 0.0f64);
    let (mut stopped, mut stop_x, mut creep) = (None::<f64>, 0.0, 0.0f64);
    let (mut newton, mut unconverged, mut cost, mut steps) = (0u64, 0u32, 0.0f64, 0usize);
    let mut lowest = f64::INFINITY;
    let mut residual = 0.0f64;
    let mut t = 0.0;
    // `EMERGE_PROBE_SLIDE_NEWTON=k`: every Newton iteration of step `k`.
    let watched: Option<usize> = std::env::var("EMERGE_PROBE_SLIDE_NEWTON")
        .ok()
        .and_then(|v| v.parse().ok());
    // `EMERGE_PROBE_SLIDE_ADAPTIVE=tol`: the step chosen by its error
    // (`total_lagrangian::controlled_step`), at most a frame, ending on every
    // frame's time.
    let control = std::env::var("EMERGE_PROBE_SLIDE_ADAPTIVE")
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
    // `EMERGE_PROBE_SLIDE_DUMP=path`: every frame's state as JSON, for a
    // replay (positions in metres).
    let dump = std::env::var("EMERGE_PROBE_SLIDE_DUMP").ok();
    let mut frames: Vec<String> = Vec::new();
    let mut next_frame = f64::from(FRAME);
    let (mut step_length, mut kept, mut rejected, mut shortest) = (dt, 0u32, 0u32, f64::INFINITY);
    let (mut at_floor, mut contact_work) = (0u32, 0.0f64);
    let contact_set = crate::spacetime::total_lagrangian::ContactSet {
        planes: &planes,
        pairs: &[],
        outlines: &[],
        corners: &[],
        friction,
    };
    while t < end - 1e-12 {
        if watched == Some(steps + 1) {
            crate::spacetime::total_lagrangian::hooks::start(0);
        }
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0s,
        };
        let (report, taken) = match &control {
            Some(control) => {
                // A frame's time within round-off counts as reached: no
                // remainder step of a few `eps` (measured: 3.9e-15 s).
                while next_frame - t <= 1e-12 {
                    next_frame += f64::from(FRAME);
                }
                step_length = step_length.min(next_frame - t);
                let r = crate::spacetime::total_lagrangian::controlled_step(
                    &layout,
                    particles,
                    (&reference, &mut solids),
                    (gravity, &mut step_length),
                    (contact_set, &penalty, &settings),
                    &mut cache,
                    (control, tolerance.sqrt()),
                );
                kept += 1;
                rejected += r.rejected;
                shortest = shortest.min(r.dt / r.kept.len() as f64);
                at_floor += u32::from(r.at_floor);
                let work = r.kept.iter().map(|k| k.work.0).sum::<f64>();
                contact_work += work;
                // `EMERGE_PROBE_SLIDE_WORK=threshold`: every kept step whose
                // contact work exceeds `threshold` of 1/2 M v^2.
                if let Some(threshold) = std::env::var("EMERGE_PROBE_SLIDE_WORK")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    && (work / kinetic0).abs() > threshold
                {
                    println!(
                        "  t {t:.6}: dt {:.3e}, contact work {:+.3e}, reaching {}, at the floor {}, error {:.2e}, rejected {}",
                        r.dt,
                        work / kinetic0,
                        r.reaching,
                        r.at_floor,
                        r.error,
                        r.rejected
                    );
                }
                // The kept substeps as one report: their sums, their
                // worst, whether any pushed.
                let mut all = crate::spacetime::total_lagrangian::StepReport {
                    converged: true,
                    ..Default::default()
                };
                for k in &r.kept {
                    all.newton += k.newton;
                    all.gmres += k.gmres;
                    all.converged &= k.converged;
                    all.residual = all.residual.max(k.residual);
                    all.friction += k.friction;
                    all.push = all.push.max(k.push);
                    all.passes += k.passes;
                    all.contacts = all.contacts.max(k.contacts);
                    for (a, b) in all.phases.iter_mut().zip(k.phases) {
                        *a += b;
                    }
                }
                (all, r.dt)
            }
            None => (
                crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
                    &layout,
                    particles,
                    &reference,
                    &solids,
                    (
                        dt,
                        crate::spacetime::total_lagrangian::ContactMode::Velocity,
                        gravity,
                    ),
                    (contact_set, &penalty, &settings),
                    &mut cache,
                ),
                dt,
            ),
        };
        if watched == Some(steps + 1) {
            for (solve, lines) in crate::spacetime::total_lagrangian::hooks::finish()
                .iter()
                .enumerate()
            {
                for line in lines {
                    println!("  solve {solve}: {line}");
                }
            }
        }
        t += taken;
        steps += 1;
        newton += u64::from(report.newton);
        unconverged += u32::from(!report.converged);
        residual = residual.max(report.residual);
        cost += report.phases.iter().sum::<f64>();
        heat += report.friction;
        if report.friction < 0.0 {
            made += 1;
            worst_made = worst_made.max(-report.friction / kinetic0);
        }
        let (c, v) = centre(&x, &vel);
        let exact = speed(t);
        if std::env::var_os("EMERGE_PROBE_SLIDE_TRACE").is_some() {
            println!(
                "  t {t:.4}: v {:+.5} (exact {:+.5}) v0, heat {:+.3e}, push {:.3e}, Newton {} residual {:.2e} converged {}, passes {}, contacts {}",
                v.x / reference_speed,
                exact / reference_speed,
                report.friction / kinetic0,
                report.push,
                report.newton,
                report.residual,
                report.converged,
                report.passes,
                report.contacts
            );
        }
        worst_v = worst_v.max((v.x - exact).abs() / reference_speed);
        if stops && stopped.is_none() && v.x <= 1e-3 * v0 {
            stopped = Some(t);
            stop_x = c.x;
        }
        if stopped.is_some() {
            creep = creep.max((c.x - stop_x).abs());
        }
        for q in &x[body..] {
            lowest = lowest.min(q.y - floor);
        }
        nan |= x.iter().chain(&vel).any(|q| !q.is_finite());
        worst_gyration = worst_gyration.max((gyration(&x) / gyration0 - 1.0).abs());
        if report.push > 0.0 && !touched {
            touched = true;
        } else if !touched {
            impact = v.y;
        }
        if touched {
            rebound = rebound.max(v.y);
        }
        if dump.is_some() && t >= next_frame - 1e-9 {
            next_frame += f64::from(FRAME);
            let metres = |q: DVec2| format!("[{:.6},{:.6}]", q.x * cell_m, q.y * cell_m);
            let body_points: Vec<String> = x[..body].iter().map(|&q| metres(q)).collect();
            let speeds: Vec<String> = vel[..body]
                .iter()
                .map(|q| format!("{:.5}", q.length() * cell_m))
                .collect();
            let outline: Vec<String> = x[body..].iter().map(|&q| metres(q)).collect();
            let now = energy(&x, &vel, &f);
            frames.push(format!(
                "{{\"time\":{t:.6},\"positions\":[{}],\"speed_m_s\":[{}],\"contact_points\":[{}],\"energy\":{:.9e},\"balance\":{:.3e}}}",
                body_points.join(","),
                speeds.join(","),
                outline.join(","),
                now,
                (now + heat - e0) / kinetic0
            ));
        }
    }
    if let Some(path) = &dump {
        let header = format!(
            "{{\"dx_m\":{cell_m},\"material\":\"elastic solid (corotated, Gonzalez discrete gradient)\",\"E_pa\":{e},\"nu\":{nu},\"rho_kg_m3\":{rho},\"friction\":{friction},\"drop_m\":{:.4},\"impact_speed_m_s\":{:.4},\"floor_y_m\":{:.4},\"floor_x_m\":[0,{:.4}],\"step\":\"{}\",\"precomputed\":true}}",
            drop * cell_m,
            (2.0 * g * drop).sqrt() * cell_m,
            floor * cell_m,
            f64::from(layout.nx as u32) * cell_m,
            match &control {
                Some(c) => format!(
                    "controlled, tol {} (Klapproth, Schiela, Deuflhard 2011, mean-velocity estimator)",
                    c.tolerance
                ),
                None => format!("fixed, {per_frame} per frame (route C)"),
            }
        );
        let text = format!(
            "{{\"header\":{header},\"frames\":[\n{}\n]}}\n",
            frames.join(",\n")
        );
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).expect("dump folder");
        }
        std::fs::write(path, text).expect("dump written");
    }
    let (c, v) = centre(&x, &vel);
    let e_end = energy(&x, &vel, &f);
    let bounce = if drop > 0.0 {
        format!(
            "
  drop {drop} cells: impact {:+.4} m/s, highest rebound {:+.4} m/s (ratio {:.4}), gyration change at worst {worst_gyration:.2e}, NaN {nan}",
            impact * cell_m,
            rebound * cell_m,
            rebound / impact.abs().max(f64::MIN_POSITIVE)
        )
    } else {
        String::new()
    };
    let moved = c.x - start_x;
    let exact_heat = friction * g * cosine * total * travel;
    format!(
        "block {width} x {height} cells of 1 cm, E {e:e} Pa, nu {nu}, mu {friction}, slope {tilt}, v0 {:.2} m/s, c_p {:.0} m/s, {per_frame} steps per frame (dt {dt:.3e} s, Courant {:.0}): {steps} steps to {t:.3} s
  centre velocity error at worst {worst_v:.2e} of {:.3} m/s; stop at {} s (exact {}); path {:.5} m (exact {:.5}, realism {:.4}); creep after the stop {:.2e} m; end speed {:+.3e} m/s (exact {:+.3e})
  heat {:.6} of the exact; energy balance (kinetic, stored, gravity, heat) {:+.2e} of 1/2 M v^2; friction made energy on {made} steps, at worst {worst_made:.2e}; lowest contact point {lowest:+.2e} cells
  Newton {:.2} per step, unconverged {unconverged} (worst residual {residual:.1e} of the tolerance's scale); {:.2} s per simulated s{}",
        v0 * cell_m,
        c_p * cell_m,
        c_p * dt,
        reference_speed * cell_m,
        stopped.map_or("never".into(), |s| format!("{s:.4}")),
        if stops { format!("{stop:.4}") } else { "never".into() },
        moved * cell_m,
        travel * cell_m,
        1.0 - (moved - travel).abs()
            / if travel > 0.0 {
                travel
            } else {
                0.5 * g * sine * end * end
            },
        creep * cell_m,
        v.x * cell_m,
        speed(t) * cell_m,
        if exact_heat > 0.0 { heat / exact_heat } else { heat / kinetic0 },
        (e_end + heat - e0) / kinetic0,
        newton as f64 / steps.max(1) as f64,
        cost / t,
        match &control {
            Some(c) => format!("\n  controlled, tol {}: {kept} steps kept, {rejected} rejected, {at_floor} at the estimator's floor, shortest {shortest:.3e} s, contact work {:+.3e} of 1/2 M v^2", c.tolerance, contact_work / kinetic0),
            None => String::new(),
        }
    ) + &bounce
}

/// The block's slide (`block_slide_run`) at `EMERGE_PROBE_SLIDE_STEPS`
/// steps per frame (default `1,4`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_block_slide() {
    let steps = std::env::var("EMERGE_PROBE_SLIDE_STEPS").unwrap_or_else(|_| "1,4".into());
    for per_frame in steps.split(',').filter_map(|s| s.trim().parse().ok()) {
        println!("{}", block_slide_run(per_frame));
    }
}
