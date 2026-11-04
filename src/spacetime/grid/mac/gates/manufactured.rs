use super::*;

/// One run of the manufactured solution (`probe_manufactured_solution`).
#[derive(Clone, Copy)]
struct Manufactured {
    /// The grid kept on the reference configuration
    /// (`total_lagrangian::step_total_lagrangian`), or rebuilt at every step
    /// (`total_lagrangian::step_dirichlet`).
    lagrangian: bool,
    substep: Substep,
    /// The body's side `L` in cells (`L = 10 cm`, so `h = L / cells`).
    cells: usize,
    /// `G / L`.
    amplitude: f64,
    /// `omega dt` at one frame's step, the same `omega` at every substep.
    omega_frame: f64,
    newton_tolerance: f64,
    /// The longest a P wave step run simulates, s.
    cap: f64,
    /// `sin^2` in place of `sin` along each axis: zero with a zero slope
    /// on the boundary, so the held faces beyond it are what the exact
    /// field's spline needs there to second order.
    squared: bool,
}

/// The manufactured solution of Gong (PhD, UNM 2015, section 4.7, eq.
/// 4.41) after Sadeghirad et al. 2011, with its frequency free: on the
/// reference square `[0, L]^2`, `U = (G sin(pi X / L) sin(w t), G sin(pi Y
/// / L) sin(w t))`. The body force making it exact is derived here for this
/// engine's energy (`total_lagrangian::Elastic`): `B = A - Div P` per unit
/// reference mass (density 1 in the step's units), `A = -w^2 U`, `P = tau
/// F^-T` the step's own stress (`Elastic::first_pk`), its divergence by a
/// central difference along `X` and `Y` of the exact `P` (`P` is diagonal
/// for this diagonal `F`, so only `dP11/dX` and `dP22/dY` remain). The
/// rollers `v1 = 0` at `X = 0, L` and `v2 = 0` at `Y = 0, L` are exact for
/// it; the faces on and beyond them are held at zero both ways. Errors are
/// the relative L2 norms over the particles of displacement, velocity and
/// `P`, each over the norm of its exact field at its peak phase, at the end
/// and at worst over the frames' ends; realism is one less the error.
fn manufactured_run(case: Manufactured) -> String {
    use glam::{DMat2, DVec2};
    let Manufactured {
        lagrangian,
        substep,
        cells,
        amplitude,
        omega_frame,
        newton_tolerance,
        cap,
        squared,
    } = case;
    let side = cells as f64;
    let cell_m = 0.1 / side;
    // Medium clay; `EMERGE_PROBE_MS_NU` overrides its Poisson ratio.
    let nu: f64 = std::env::var("EMERGE_PROBE_MS_NU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.45);
    let (e, rho) = (11.25e6f64, 1700.0f64);
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
    let c_p = (solid.bulk + solid.mu).sqrt();
    // The body's corner on the grid: on a face (default) or between faces.
    let corner: f64 = std::env::var("EMERGE_PROBE_MS_CORNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4.0);
    let layout = MacLayout::new(cells + 8, cells + 8, 1.0);
    let omega = omega_frame / f64::from(FRAME);
    let g = amplitude * side;
    let k = std::f64::consts::PI / side;
    let period = 2.0 * std::f64::consts::PI / omega;
    let frame = f64::from(FRAME);
    // `EMERGE_PROBE_MS_PERIODS` periods (default one).
    let periods: f64 = std::env::var("EMERGE_PROBE_MS_PERIODS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let frames = (periods * period / frame).round().max(1.0) as usize;
    let per_frame = match substep {
        Substep::Frame => 1,
        _ => (frame / f64::from(substep.length(1.0, c_p as f32))).ceil() as usize,
    };
    let dt = frame / per_frame as f64;
    let mut steps = frames * per_frame;
    if !matches!(substep, Substep::Frame) {
        steps = steps.min((cap / dt).round().max(1.0) as usize);
    }
    // The profile along one axis and its slope.
    let profile = |s: f64| {
        let a = k * (s - corner);
        if squared {
            (a.sin() * a.sin(), k * (2.0 * a).sin())
        } else {
            (a.sin(), k * a.cos())
        }
    };
    // The exact fields at a reference point and a time.
    let u =
        |q: DVec2, t: f64| DVec2::new(g * profile(q.x).0, g * profile(q.y).0) * (omega * t).sin();
    let v = |q: DVec2, t: f64| {
        DVec2::new(g * profile(q.x).0, g * profile(q.y).0) * omega * (omega * t).cos()
    };
    let grad = |q: DVec2, t: f64| {
        DMat2::from_diagonal(DVec2::new(
            1.0 + g * profile(q.x).1 * (omega * t).sin(),
            1.0 + g * profile(q.y).1 * (omega * t).sin(),
        ))
    };
    let stress = |q: DVec2, t: f64| solid.first_pk(grad(q, t));
    // With `EMERGE_PROBE_ZETA` a Kelvin-Voigt viscous stress
    // (`total_lagrangian::Elastic::viscous_stress`, stiffness-proportional) whose
    // damping ratio is `zeta` at 1 kHz, `beta = 2 zeta / (2 pi 1000 Hz)`:
    // the clay's small-strain damping, measured in the geotechnical band
    // (0.05-80 Hz) and by bender elements from 1 kHz, used here at the
    // body's unresolved modes (an extrapolation, declared). The step
    // evaluates it at the end-of-step velocity; the exact body force
    // includes it, so the manufactured motion stays exact.
    let zeta: f64 = std::env::var("EMERGE_PROBE_ZETA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let beta = 2.0 * zeta / (2.0 * std::f64::consts::PI * 1000.0);
    crate::spacetime::total_lagrangian::hooks::set_viscosity(beta);
    let grad_rate = |q: DVec2, t: f64| {
        DMat2::from_diagonal(DVec2::new(
            g * profile(q.x).1 * omega * (omega * t).cos(),
            g * profile(q.y).1 * omega * (omega * t).cos(),
        ))
    };
    let total_stress = |q: DVec2, t: f64| {
        let f = grad(q, t);
        if beta > 0.0 {
            solid.first_pk(f) + f * solid.viscous_stress(f, grad_rate(q, t), beta)
        } else {
            solid.first_pk(f)
        }
    };
    let divergence = |q: DVec2, t: f64, delta: f64| {
        let dx = DVec2::new(delta, 0.0);
        let dy = DVec2::new(0.0, delta);
        DVec2::new(
            (total_stress(q + dx, t).x_axis.x - total_stress(q - dx, t).x_axis.x) / (2.0 * delta),
            (total_stress(q + dy, t).y_axis.y - total_stress(q - dy, t).y_axis.y) / (2.0 * delta),
        )
    };
    let delta = 1e-3;
    let body = |q: DVec2, t: f64| -omega * omega * u(q, t) - divergence(q, t, delta);
    // The quadrature points of the reference block: particles two by two
    // per cell, or with `EMERGE_PROBE_MS_QUADRATURE=gauss` the reference
    // domain's own integrals, a Gauss-Legendre rule of
    // `EMERGE_PROBE_MS_GAUSS` points (default 3) per direction on every
    // half cell, where both lattices' B-splines are polynomials.
    let gauss = std::env::var("EMERGE_PROBE_MS_QUADRATURE").as_deref() == Ok("gauss");
    let order: usize = std::env::var("EMERGE_PROBE_MS_GAUSS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let (nodes, node_weights) = if gauss {
        gauss_legendre(order)
    } else {
        (vec![0.5], vec![1.0])
    };
    let (reference, v0): (Vec<DVec2>, Vec<f64>) = (0..2 * cells)
        .flat_map(|j| (0..2 * cells).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            let (nodes, node_weights) = (&nodes, &node_weights);
            (0..nodes.len()).flat_map(move |b| {
                (0..nodes.len()).map(move |a| {
                    (
                        DVec2::new(
                            corner + (i as f64 + nodes[a]) * 0.5,
                            corner + (j as f64 + nodes[b]) * 0.5,
                        ),
                        0.25 * node_weights[a] * node_weights[b],
                    )
                })
            })
        })
        .unzip();
    let n = reference.len();
    // Density 1 in the step's units.
    let mass = v0.clone();
    let solids = vec![solid; n];
    let mut x = reference.clone();
    let mut vel: Vec<DVec2> = reference.iter().map(|&q| v(q, 0.0)).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let (lo, hi) = (corner, corner + side);
    let held = |a: usize, i: usize, j: usize| -> Option<f64> {
        let at = if a == 0 { i as f64 } else { j as f64 };
        if at <= lo + 1e-9 {
            Some(1.0)
        } else if at >= hi - 1e-9 {
            Some(-1.0)
        } else {
            None
        }
    };
    let settings = crate::spacetime::total_lagrangian::ConservingSettings {
        newton_tolerance,
        occupation: match std::env::var("EMERGE_PROBE_OCCUPATION").as_deref() {
            Ok("none") => None,
            _ => crate::spacetime::total_lagrangian::ConservingSettings::default().occupation,
        },
        ..Default::default()
    };
    // L2 norms over the quadrature points, each weighted by its volume
    // (equal weights for the particles: the plain sums).
    let norm = |values: &mut dyn Iterator<Item = f64>| values.sum::<f64>().sqrt();
    let peak = 0.5 * std::f64::consts::PI / omega;
    let u_scale = norm(&mut (0..n).map(|p| v0[p] * u(reference[p], peak).length_squared()));
    let v_scale = norm(&mut (0..n).map(|p| v0[p] * v(reference[p], 0.0).length_squared()));
    let p_scale = norm(&mut (0..n).map(|p| {
        let s = stress(reference[p], peak);
        v0[p] * (s.x_axis.length_squared() + s.y_axis.length_squared())
    }));
    // With `EMERGE_PROBE_THETA=cell`, `Theta` constant per grid cell (the
    // cell's mean J, Gonzalez 2000 eq. 53): the step's volumetric stress
    // takes the cell's pressure, and the stored energy is `sum V0 W_dev +
    // sum_cells V U(Theta)` (`total_lagrangian::Elastic::mixed_stress`).
    let cell_theta = std::env::var("EMERGE_PROBE_THETA").as_deref() == Ok("cell");
    let cell_of: Vec<usize> = reference
        .iter()
        .map(|q| q.x.floor() as usize + layout.nx * q.y.floor() as usize)
        .collect();
    let cell_count = layout.nx * layout.ny;
    crate::spacetime::total_lagrangian::hooks::set_cells(
        cell_theta.then(|| (cell_of.clone(), cell_count)),
    );
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        let kinetic: f64 = (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared())
            .sum();
        if cell_theta {
            let mut sums = vec![(0.0f64, 0.0f64); cell_count];
            for p in 0..n {
                sums[cell_of[p]].0 += v0[p];
                sums[cell_of[p]].1 += v0[p] * f[p].determinant();
            }
            let volumetric: f64 = sums
                .iter()
                .filter(|s| s.0 > 0.0)
                .map(|&(volume, j)| volume * 0.5 * solid.bulk * (j / volume - 1.0).powi(2))
                .sum();
            kinetic
                + (0..n)
                    .map(|p| v0[p] * solid.deviatoric_energy(f[p]))
                    .sum::<f64>()
                + volumetric
        } else {
            kinetic + (0..n).map(|p| v0[p] * solid.energy(f[p])).sum::<f64>()
        }
    };
    // Every particle, and the interior: at least two cells from the
    // boundary, out of reach of the held faces beyond it.
    let inside: Vec<bool> = reference
        .iter()
        .map(|q| {
            let d = (q.x - corner)
                .min(corner + side - q.x)
                .min(q.y - corner)
                .min(corner + side - q.y);
            d >= 2.0
        })
        .collect();
    let scales = |set: &dyn Fn(usize) -> bool| -> [f64; 3] {
        [
            norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * u(reference[p], peak).length_squared()),
            ),
            norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * v(reference[p], 0.0).length_squared()),
            ),
            norm(&mut (0..n).filter(|&p| set(p)).map(|p| {
                let s = stress(reference[p], peak);
                v0[p] * (s.x_axis.length_squared() + s.y_axis.length_squared())
            })),
        ]
    };
    let all_scales = [u_scale, v_scale, p_scale];
    // Every frame's errors, and `F`'s own, when asked.
    let history = std::env::var("EMERGE_PROBE_MS_HISTORY").is_ok();
    let f_scale = norm(&mut (0..n).map(|p| {
        let d = grad(reference[p], peak) - DMat2::IDENTITY;
        v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
    }));
    let inner_scales = scales(&|p| inside[p]);
    let edge_scales = scales(&|p| !inside[p]);
    let errors = |x: &[DVec2], vel: &[DVec2], f: &[DMat2], t: f64| -> [f64; 9] {
        let part = |set: &dyn Fn(usize) -> bool, scale: [f64; 3]| -> [f64; 3] {
            let du = norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * (x[p] - reference[p] - u(reference[p], t)).length_squared()),
            );
            let dv = norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * (vel[p] - v(reference[p], t)).length_squared()),
            );
            let dp = norm(&mut (0..n).filter(|&p| set(p)).map(|p| {
                let d = solid.first_pk(f[p]) - stress(reference[p], t);
                v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
            }));
            [du / scale[0], dv / scale[1], dp / scale[2]]
        };
        let all = part(&|_| true, all_scales);
        let inner = part(&|p| inside[p], inner_scales);
        let edge = part(&|p| !inside[p], edge_scales);
        [
            all[0], all[1], all[2], inner[0], inner[1], inner[2], edge[0], edge[1], edge[2],
        ]
    };
    // The divergence's difference step against half of it, at the peak.
    let step_check = reference
        .iter()
        .step_by(7)
        .map(|&q| {
            let a = divergence(q, peak, delta);
            let b = divergence(q, peak, 0.5 * delta);
            (a - b).length() / b.length().max(f64::MIN_POSITIVE)
        })
        .fold(0.0f64, f64::max);
    let e0 = energy(&vel, &f);
    let (mut work, mut largest_energy) = (0.0f64, e0);
    let mut worst = [0.0f64; 9];
    let (mut newton, mut gmres, mut most, mut unconverged, mut cost) =
        (0u64, 0u64, 0u32, 0u32, 0.0f64);
    let mut strain = 0.0f64;
    // With `EMERGE_PROBE_MS_BOUNDARY=web` the rollers are the weight of
    // the web-splines (`LagrangianCache::weighted`) instead of held faces:
    // per component, the product of the distances to the two planes the
    // held faces hold it on, over the side (slope one at each plane).
    let web = std::env::var("EMERGE_PROBE_MS_BOUNDARY").as_deref() == Ok("web");
    if web && !lagrangian {
        return "grid rebuilt: skipped, the weighted basis is built on the reference grid".into();
    }
    let weight = |a: usize, q: DVec2| -> (f64, DVec2) {
        let s = if a == 0 { q.x } else { q.y };
        let slope = (hi + lo - 2.0 * s) / side;
        let along = if a == 0 { DVec2::X } else { DVec2::Y };
        ((s - lo) * (hi - s) / side, slope * along)
    };
    let free = |_: usize, _: usize, _: usize| -> Option<f64> { None };
    let walls: crate::spacetime::total_lagrangian::Held<'_> = if web { &free } else { &held };
    let mut cache = web.then(|| {
        crate::spacetime::total_lagrangian::LagrangianCache::weighted(
            &layout,
            &reference,
            (&mass, &v0),
            settings.occupation,
            &weight,
        )
    });
    // With `EMERGE_PROBE_MS_TIME=eg2` the step in time is Gross's eG(2)
    // (`total_lagrangian::step_total_lagrangian_eg2`), on the reference grid only.
    let eg2 = std::env::var("EMERGE_PROBE_MS_TIME").as_deref() == Ok("eg2");
    if eg2 && !lagrangian {
        return "grid rebuilt: skipped, eG(2) is built on the reference grid".into();
    }
    let gauss_times = [0.5 - 3.0f64.sqrt() / 6.0, 0.5 + 3.0f64.sqrt() / 6.0];
    // With `EMERGE_PROBE_MS_STOP` an unconverged step is not committed:
    // the run stops there and says so.
    let stop = std::env::var("EMERGE_PROBE_MS_STOP").is_ok();
    let averaged = std::env::var("EMERGE_PROBE_MS_LOAD").as_deref() == Ok("average");
    // With `EMERGE_PROBE_DG=trapezoid` the discrete gradient's base stress
    // is the two ends' average (`total_lagrangian::hooks::set_trapezoid`).
    let gradient = std::env::var("EMERGE_PROBE_DG").unwrap_or_default();
    let trapezoid = !gradient.is_empty();
    crate::spacetime::total_lagrangian::hooks::set_trapezoid(
        gradient == "trapezoid"
            || gradient == "split_trapezoid"
            || gradient == "cofactor_trapezoid",
    );
    crate::spacetime::total_lagrangian::hooks::set_cofactor(gradient.starts_with("cofactor"));
    // `EMERGE_PROBE_VOLUMETRIC_BASE=midpoint|trapezoid` overrides the
    // split's volumetric base alone.
    crate::spacetime::total_lagrangian::hooks::set_volumetric_base(
        match std::env::var("EMERGE_PROBE_VOLUMETRIC_BASE").as_deref() {
            Ok("midpoint") => 1,
            Ok("trapezoid") => 2,
            _ => 0,
        },
    );
    crate::spacetime::total_lagrangian::hooks::set_trapezoid_p(gradient == "trapezoid_p");
    crate::spacetime::total_lagrangian::hooks::set_split(gradient.starts_with("split"));
    let mut stopped = None;
    // The heat the viscous stress books, in total and its least step.
    let (mut heat, mut least_heat) = (0.0f64, f64::INFINITY);
    // With `EMERGE_PROBE_MS_PATTERN` every frame's velocity error is also
    // described: its correlation with the previous frame's, the share of
    // its half-cell averages in the checkerboard and in the two stripe
    // patterns, its share left after a four-neighbour smoothing, and its
    // RMS by distance to the boundary in fractions of the side.
    let pattern = std::env::var("EMERGE_PROBE_MS_PATTERN").is_ok();
    let mut previous: Option<Vec<DVec2>> = None;
    let mut t = 0.0;
    // With `EMERGE_PROBE_MS_RESET` every step starts from the exact state
    // (positions, velocities, deformation gradients): each step's error is
    // then that step's own, its local truncation error.
    let reset = std::env::var("EMERGE_PROBE_MS_RESET").is_ok();
    for step in 0..steps {
        if reset {
            for p in 0..n {
                x[p] = reference[p] + u(reference[p], t);
                vel[p] = v(reference[p], t);
                f[p] = grad(reference[p], t);
            }
        }
        // The load over the step: sampled at its midpoint, or with
        // `EMERGE_PROBE_MS_LOAD=average` the mean of its two ends, as
        // Gonzalez, CMAME 190 (2000), eq. 10 writes `b_{n+1/2}` (the work
        // stays the load times the step's displacement).
        let b: Vec<DVec2> = if averaged {
            reference
                .iter()
                .map(|&q| 0.5 * (body(q, t) + body(q, t + dt)))
                .collect()
        } else {
            reference.iter().map(|&q| body(q, t + 0.5 * dt)).collect()
        };
        let before = x.clone();
        let kept = stop.then(|| (vel.clone(), f.clone()));
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0,
        };
        let mut step_work = None;
        let report = if eg2 {
            let at = gauss_times.map(|xi| {
                reference
                    .iter()
                    .map(|&q| body(q, t + xi * dt))
                    .collect::<Vec<_>>()
            });
            let (report, done) = crate::spacetime::total_lagrangian::step_total_lagrangian_eg2(
                &layout,
                particles,
                &reference,
                &solids,
                ([&at[0], &at[1]], dt),
                (walls, &settings),
                &mut cache,
            );
            step_work = Some(done);
            report
        } else if lagrangian {
            crate::spacetime::total_lagrangian::step_total_lagrangian(
                &layout,
                particles,
                &reference,
                &solids,
                (DVec2::ZERO, Some(&b), dt),
                (walls, &settings),
                &mut cache,
            )
        } else {
            crate::spacetime::total_lagrangian::step_dirichlet(
                &layout,
                particles,
                &solids,
                (&b, dt),
                (&held, &settings),
            )
        };
        if let Some((kept_v, kept_f)) = kept.filter(|_| !report.converged) {
            x = before;
            vel = kept_v;
            f = kept_f;
            newton += u64::from(report.newton);
            gmres += u64::from(report.gmres);
            most = most.max(report.gmres);
            unconverged += 1;
            cost += report.phases.iter().sum::<f64>();
            stopped = Some((
                step + 1,
                report.residual,
                report.newton,
                report.start_residual,
            ));
            break;
        }
        t += dt;
        if beta > 0.0 {
            let booked = crate::spacetime::total_lagrangian::hooks::viscous_work();
            heat += booked;
            least_heat = least_heat.min(booked);
        }
        work += step_work.unwrap_or_else(|| {
            (0..n)
                .map(|p| mass[p] * b[p].dot(x[p] - before[p]))
                .sum::<f64>()
        });
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        most = most.max(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        if (step + 1) % per_frame == 0 || step + 1 == steps {
            let now = errors(&x, &vel, &f, t);
            if pattern {
                let error: Vec<DVec2> = (0..n).map(|p| vel[p] - v(reference[p], t)).collect();
                let size = |e: &[DVec2]| e.iter().map(|d| d.length_squared()).sum::<f64>().sqrt();
                let correlation = previous.as_ref().map_or(f64::NAN, |old: &Vec<DVec2>| {
                    let dot: f64 = error.iter().zip(old).map(|(a, b)| a.dot(*b)).sum();
                    dot / (size(&error) * size(old)).max(f64::MIN_POSITIVE)
                });
                let halves = 2 * cells;
                let mut cell = vec![DVec2::ZERO; halves * halves];
                let mut count = vec![0usize; halves * halves];
                for (p, e) in error.iter().enumerate() {
                    let q = reference[p] - DVec2::splat(corner);
                    let (i, j) = (
                        ((q.x / 0.5).floor() as usize).min(halves - 1),
                        ((q.y / 0.5).floor() as usize).min(halves - 1),
                    );
                    cell[i + halves * j] += *e;
                    count[i + halves * j] += 1;
                }
                for (c, &k) in cell.iter_mut().zip(&count) {
                    *c /= k.max(1) as f64;
                }
                let total: f64 = cell.iter().map(|c| c.length_squared()).sum();
                let share = |sign: &dyn Fn(usize, usize) -> f64| {
                    let mut sum = DVec2::ZERO;
                    for j in 0..halves {
                        for i in 0..halves {
                            sum += sign(i, j) * cell[i + halves * j];
                        }
                    }
                    sum.length_squared() / (halves * halves) as f64 / total.max(f64::MIN_POSITIVE)
                };
                let parity = |k: usize| if k.is_multiple_of(2) { 1.0 } else { -1.0 };
                let checker = share(&|i, j| parity(i + j));
                let stripes = (share(&|i, _| parity(i)), share(&|_, j| parity(j)));
                let mut rough = 0.0;
                for j in 0..halves {
                    for i in 0..halves {
                        let mut mean = DVec2::ZERO;
                        let mut k = 0.0;
                        for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                            let (a, c) = (i as i64 + di, j as i64 + dj);
                            if a >= 0 && c >= 0 && (a as usize) < halves && (c as usize) < halves {
                                mean += cell[a as usize + halves * c as usize];
                                k += 1.0;
                            }
                        }
                        rough += (cell[i + halves * j] - mean / k).length_squared();
                    }
                }
                let bands = [0.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, f64::INFINITY];
                let band = |lo: f64, hi: f64| {
                    let (mut sum, mut k) = (0.0, 0.0);
                    for (p, e) in error.iter().enumerate() {
                        let q = reference[p] - DVec2::splat(corner);
                        let d = q.x.min(side - q.x).min(q.y).min(side - q.y) / side;
                        if d >= lo && d < hi {
                            sum += v0[p] * e.length_squared();
                            k += v0[p];
                        }
                    }
                    (sum / f64::max(k, f64::MIN_POSITIVE)).sqrt()
                        / (v_scale / v0.iter().sum::<f64>().sqrt())
                };
                println!(
                    "  pattern t {t:.4}: vel error {:.3e}, correlation with the last frame {correlation:+.3}, half-cell checkerboard {:.3}, stripes x {:.3} y {:.3}, left after smoothing {:.3}; RMS by distance to the boundary (side fractions 0-1/16, 1/16-1/8, 1/8-1/4, >1/4) {:.3e} {:.3e} {:.3e} {:.3e}",
                    now[1],
                    checker,
                    stripes.0,
                    stripes.1,
                    rough / total.max(f64::MIN_POSITIVE),
                    band(bands[0], bands[1]),
                    band(bands[1], bands[2]),
                    band(bands[2], bands[3]),
                    band(bands[3], bands[4])
                );
                previous = Some(error);
            }
            if history {
                let f_error = norm(&mut (0..n).map(|p| {
                    let d = f[p] - grad(reference[p], t);
                    v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
                })) / f_scale;
                let mean = (0..n)
                    .map(|p| x[p] - reference[p] - u(reference[p], t))
                    .sum::<DVec2>()
                    / n as f64;
                // The first column's error along x, against the middle
                // column's: the slip at the boundary carried inside.
                let column = |at: f64| {
                    let set: Vec<usize> = (0..n)
                        .filter(|&p| (reference[p].x - corner - at).abs() < 0.1)
                        .collect();
                    set.iter()
                        .map(|&p| (x[p] - reference[p] - u(reference[p], t)).x)
                        .sum::<f64>()
                        / set.len().max(1) as f64
                };
                let (edge, middle) = (column(0.25), column(0.5 * side - 0.25));
                println!(
                    "  t {t:.4} (w t {:.3}): disp {:.3e} vel {:.3e} P {:.3e} F {f_error:.3e}, interior disp {:.3e} vel {:.3e}; mean displacement error ({:+.2e}, {:+.2e}) of the scale {:.2e}; x error first column {edge:+.2e}, middle column {middle:+.2e}",
                    omega * t,
                    now[0],
                    now[1],
                    now[2],
                    now[3],
                    now[4],
                    mean.x,
                    mean.y,
                    u_scale / (n as f64).sqrt()
                );
            }
            for (w, e) in worst.iter_mut().zip(now) {
                *w = if e.is_nan() || w.is_nan() {
                    f64::NAN
                } else {
                    w.max(e)
                };
            }
            largest_energy = largest_energy.max(energy(&vel, &f));
            for m in &f {
                let d = *m - DMat2::IDENTITY;
                strain = strain.max((d.x_axis.length_squared() + d.y_axis.length_squared()).sqrt());
            }
        }
    }
    let end = errors(&x, &vel, &f, t);
    let balance = (energy(&vel, &f) - e0 - work + heat) / largest_energy;
    let realism = |e: f64| if e.is_nan() { 0.0 } else { (1.0 - e).max(0.0) };
    let taken = stopped.map_or(steps, |(k, ..)| k);
    let per = taken.max(1) as f64;
    let ran = match stopped {
        Some((k, residual, newton, start)) => format!(
            "STOPPED at step {k} of {steps} (unconverged, not committed: residual {residual:.2e} after {newton} Newton from {start:.2e}), {} committed",
            k - 1
        ),
        None => format!("{steps} steps"),
    };
    let quadrature = if gauss {
        let (nodes, node_weights) = gauss_legendre(order);
        // The rule against the monomials it must integrate exactly.
        let check = (0..2 * order as i32)
            .map(|d| {
                let sum: f64 = nodes
                    .iter()
                    .zip(&node_weights)
                    .map(|(t, w)| w * t.powi(d))
                    .sum();
                (sum - 1.0 / f64::from(d + 1)).abs()
            })
            .fold(0.0f64, f64::max);
        format!(", Gauss {order}x{order} per half cell ({n} points, rule check {check:.1e})")
    } else {
        format!(", particles 2x2 per cell ({n})")
    };
    let quadrature = if beta > 0.0 {
        format!(
            "{quadrature}, Kelvin-Voigt zeta {zeta} at 1 kHz (heat {:.3e} of the largest energy, least step {:+.2e})",
            heat / largest_energy,
            least_heat / largest_energy
        )
    } else {
        quadrature
    };
    format!(
        "{}{}{}{}{quadrature} at {substep}, h = L/{cells}, G/L {amplitude}, w dt_frame {omega_frame} (divergence step check {step_check:.1e}): {ran} to t = {t:.4} s ({:.2} periods); error end disp {:.2e} vel {:.2e} P {:.2e}, worst disp {:.2e} vel {:.2e} P {:.2e}; realism at worst disp {:.1} % vel {:.1} % P {:.1} %; interior (2 cells from the boundary) error end disp {:.2e} vel {:.2e} P {:.2e}, realism at worst disp {:.1} % vel {:.1} % P {:.1} %; edge (within 2 cells) realism at worst disp {:.1} % vel {:.1} % P {:.1} %; energy balance {balance:+.2e}; Newton {:.2} per step, GMRES {:.1} per step and at most {most}, unconverged {unconverged}; {:.1} s per simulated s; largest |F - I| {strain:.3}",
        if lagrangian {
            "reference grid"
        } else {
            "grid rebuilt"
        },
        match (squared, cell_theta) {
            (true, true) => ", sin^2 profile, Theta per cell",
            (true, false) => ", sin^2 profile",
            (false, true) => ", Theta per cell",
            (false, false) => "",
        },
        if web {
            ", weighted walls"
        } else {
            ", held walls"
        },
        if eg2 {
            ", eG(2) in time"
        } else {
            match (averaged, trapezoid) {
                (true, true) if gradient == "trapezoid_p" => {
                    ", load averaged over the step's ends, trapezoid gradient in P"
                }
                (true, true) if gradient == "split" => {
                    ", load averaged over the step's ends, split gradient"
                }
                (true, true) if gradient == "split_trapezoid" => {
                    ", load averaged over the step's ends, split trapezoid gradient"
                }
                (true, true) if gradient == "cofactor" => {
                    ", load averaged over the step's ends, split cofactor gradient"
                }
                (true, true) if gradient == "cofactor_trapezoid" => {
                    ", load averaged over the step's ends, split cofactor trapezoid gradient"
                }
                (true, true) => ", load averaged over the step's ends, trapezoid gradient",
                (true, false) => ", load averaged over the step's ends",
                (false, true) => ", trapezoid gradient",
                (false, false) => "",
            }
        },
        t / period,
        end[0],
        end[1],
        end[2],
        worst[0],
        worst[1],
        worst[2],
        100.0 * realism(worst[0]),
        100.0 * realism(worst[1]),
        100.0 * realism(worst[2]),
        end[3],
        end[4],
        end[5],
        100.0 * realism(worst[3]),
        100.0 * realism(worst[4]),
        100.0 * realism(worst[5]),
        100.0 * realism(worst[6]),
        100.0 * realism(worst[7]),
        100.0 * realism(worst[8]),
        newton as f64 / per,
        gmres as f64 / per,
        cost / t
    )
}

/// The manufactured solution (`manufactured_run`) over its sweep, a probe:
/// `EMERGE_PROBE_MS_METHODS` (`rebuilt,reference`), `_STEPS` (`frame,p`),
/// `_CELLS` (`8,16,32`), `_AMPLITUDES` (`0.001,0.05,0.1`), `_OMEGAS`
/// (`0.1,0.3,1,3,10`, `omega dt` at one frame), `_CAP` (the longest a P
/// wave run simulates, s, default 0.2), `_PROFILE=square` for the `sin^2`
/// profile, `_HISTORY` to print every frame's errors, `_CORNER` the body's
/// corner in cells (default 4, on a face), `_QUADRATURE=gauss` and
/// `_GAUSS` (points per direction per half cell, default 3) for the
/// reference domain's own integrals, `_BOUNDARY=web` for the rollers as
/// web-spline weights, `_STOP` to stop at an unconverged step without
/// committing it, `_TIME=eg2` for the eG(2) step in time, `_LOAD=average`
/// for the load averaged over each step's ends, `_PATTERN` to describe the
/// velocity error's pattern in space and time, `_PERIODS` the periods run
/// (default 1); `EMERGE_PROBE_DG=trapezoid` for the trapezoid base of the
/// discrete gradient (`trapezoid_p` the same in `P`, a diagnostic; `split`
/// and `split_trapezoid` Gonzalez's deviatoric/volumetric split, `cofactor`
/// and `cofactor_trapezoid` the same with the volumetric part `p cof(F_mid)`);
/// `EMERGE_PROBE_MS_NU` the Poisson ratio (default 0.45);
/// `EMERGE_PROBE_THETA=cell` for `Theta` constant per cell;
/// `EMERGE_PROBE_MS_RESET` to start every step from the exact state;
/// `EMERGE_PROBE_VOLUMETRIC_BASE` the split's volumetric base on its own;
/// `EMERGE_PROBE_ZETA` a Kelvin-Voigt damping ratio at 1 kHz;
/// `EMERGE_PROBE_NEWTON_TOLERANCE` (default the step's). First, the checks
/// of the body force: `P` against the energy's gradient by central
/// differences, and the divergence's difference step against half of it.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_manufactured_solution() {
    use glam::{DMat2, DVec2};
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    let numbers = |name: &str, default: &str| -> Vec<f64> {
        list(name, default)
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect()
    };
    // The stress against the energy's gradient, on the strained states the
    // sweep reaches (medium clay at 1 cm: the stress scales, the check does
    // not depend on it).
    {
        let (e, nu, rho, cell_m) = (11.25e6f64, 0.45f64, 1700.0f64, 0.01f64);
        let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
        let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
        let mut worst = 0.0f64;
        for (a, b) in [(1.3, 0.8), (0.75, 1.2), (1.05, 1.31), (0.69, 0.71)] {
            let f = DMat2::from_diagonal(DVec2::new(a, b));
            let p = solid.first_pk(f);
            let eta = 1e-6;
            for col in 0..2 {
                for row in 0..2 {
                    let mut plus = f;
                    let mut minus = f;
                    plus.col_mut(col)[row] += eta;
                    minus.col_mut(col)[row] -= eta;
                    let fd = (solid.energy(plus) - solid.energy(minus)) / (2.0 * eta);
                    let scale = p.x_axis.length().max(p.y_axis.length());
                    worst = worst.max((fd - p.col(col)[row]).abs() / scale);
                }
            }
        }
        println!("check: P = dW/dF by central differences, largest relative gap {worst:.1e}");
    }
    let tolerance: f64 = std::env::var("EMERGE_PROBE_NEWTON_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(
            crate::spacetime::total_lagrangian::ConservingSettings::default().newton_tolerance,
        );
    let cap = numbers("EMERGE_PROBE_MS_CAP", "0.2")[0];
    let squared = std::env::var("EMERGE_PROBE_MS_PROFILE").as_deref() == Ok("square");
    for method in list("EMERGE_PROBE_MS_METHODS", "rebuilt,reference") {
        for step in list("EMERGE_PROBE_MS_STEPS", "frame,p") {
            for cells in numbers("EMERGE_PROBE_MS_CELLS", "8,16,32") {
                for amplitude in numbers("EMERGE_PROBE_MS_AMPLITUDES", "0.001,0.05,0.1") {
                    for omega_frame in numbers("EMERGE_PROBE_MS_OMEGAS", "0.1,0.3,1,3,10") {
                        println!(
                            "{}",
                            manufactured_run(Manufactured {
                                lagrangian: method == "reference",
                                substep: Substep::parse(&step).unwrap_or(Substep::Frame),
                                cells: cells as usize,
                                amplitude,
                                omega_frame,
                                newton_tolerance: tolerance,
                                cap,
                                squared,
                            })
                        );
                    }
                }
            }
        }
    }
}

/// The manufactured solution's failing steps, traced (`total_lagrangian::hooks`):
/// for each case `cells:G/L:w dt_frame` of `EMERGE_PROBE_FAIL_CASES`
/// (default `8:0.1:1,8:0.05:3,8:0.05:10`, the reference grid at one
/// frame), every Newton iteration of the first failing solve and of the
/// solve before it (residual, what GMRES reached, the line search's step,
/// the least `det F`), and the homotopy from `u = 0` that tells whether the
/// step's system has an admissible solution; `EMERGE_PROBE_HOMOTOPY` sets
/// its first load increments (default 4), `EMERGE_PROBE_FOLD` also watches
/// the Jacobian's least eigenvalue along it.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_failing_step() {
    let increments: u32 = std::env::var("EMERGE_PROBE_HOMOTOPY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let cases = std::env::var("EMERGE_PROBE_FAIL_CASES")
        .unwrap_or_else(|_| "8:0.1:1,8:0.05:3,8:0.05:10".into());
    for case in cases.split(',') {
        let parts: Vec<f64> = case
            .split(':')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let [cells, amplitude, omega_frame] = parts[..] else {
            continue;
        };
        crate::spacetime::total_lagrangian::hooks::start(increments);
        crate::spacetime::total_lagrangian::hooks::watch_fold(
            std::env::var("EMERGE_PROBE_FOLD").is_ok(),
        );
        let line = manufactured_run(Manufactured {
            lagrangian: true,
            substep: Substep::Frame,
            cells: cells as usize,
            amplitude,
            omega_frame,
            newton_tolerance: crate::spacetime::total_lagrangian::ConservingSettings::default()
                .newton_tolerance,
            cap: 0.2,
            squared: false,
        });
        let solves = crate::spacetime::total_lagrangian::hooks::finish();
        println!("{line}");
        let failed = |s: &Vec<String>| {
            s.last()
                .is_some_and(|l| l.starts_with("end: converged false"))
        };
        let first = solves.iter().position(failed);
        for (k, solve) in solves.iter().enumerate() {
            let shown = match first {
                Some(f) => k + 1 == f || k == f || (k > f && failed(solve) && k <= f + 1),
                None => k == 0,
            };
            if shown {
                println!("  solve {} of {}:", k + 1, solves.len());
                for l in solve {
                    println!("    {l}");
                }
            } else {
                let homotopy = solve.iter().find(|l| l.starts_with("homotopy"));
                let end = solve.last();
                println!(
                    "  solve {}: {} | {}",
                    k + 1,
                    homotopy.map_or("", String::as_str),
                    end.map_or("", String::as_str)
                );
            }
        }
    }
}
