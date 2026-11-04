use super::*;

/// One run of the generalized vortex (`probe_generalized_vortex`).
#[derive(Clone, Copy)]
struct Vortex {
    /// The grid kept on the reference configuration
    /// (`total_lagrangian::step_total_lagrangian`), or rebuilt at every step
    /// (`total_lagrangian::step_dirichlet`).
    lagrangian: bool,
    substep: Substep,
    /// The ring's width `Ro - Ri` in cells.
    cells: usize,
    /// `T0`, s: the run lasts `T0`.
    period: f64,
    newton_tolerance: f64,
}

/// The generalized vortex as the TLMPM review states it (de Vaucorbeil,
/// Nguyen, Sinaie, Wu, Adv. Appl. Mech. 53 (2020), section 6.3, eqs.
/// 6.13-6.16, after Kamojjala et al. 2015): the ring `Ri = 0.75 m <= R <=
/// Ro = 1.25 m` turns by `alpha = g(t) h(R)` about its centre, `x = Q(alpha)
/// X`, `h = 1 - 8 s^2 + 16 s^4` with `s = (R - R_bar) / (Ri - Ro)`, `g = G
/// sin(pi t / T0)`, `G = 1`, `E = 1e3 Pa`, `nu = 0.3`, `rho0 = 1000`. The
/// motion is isochoric, `h` and `h'` vanish on both circles (fixed and free
/// of traction), and the faces outside the ring are held at zero both ways,
/// as the review sets the nodes outside the ring. The review's body force
/// is for its material; this engine's is derived the same way as the
/// manufactured solution's (`manufactured_run`): `B = A - Div P` with the
/// step's own `P` (`Elastic::first_pk`), the full divergence by central
/// differences. The review also states a zero initial velocity, which its
/// own `g` contradicts (`g'(0) = pi G / T0`): the run starts from the
/// motion's velocity, the only start the motion is exact for. Errors as in
/// the manufactured solution, over the norms of the exact fields at their
/// peak phase (`t = T0 / 2` for displacement and `P`, `t = 0` for velocity).
fn vortex_run(case: Vortex) -> String {
    use glam::{DMat2, DVec2};
    let Vortex {
        lagrangian,
        substep,
        cells,
        period,
        newton_tolerance,
    } = case;
    let cell_m = 0.5 / cells as f64;
    let (e, nu, rho) = (1e3f64, 0.3f64, 1000.0f64);
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
    let c_p = (solid.bulk + solid.mu).sqrt();
    let (inner, outer) = (0.75 / cell_m, 1.25 / cell_m);
    let middle = 0.5 * (inner + outer);
    let side = (2.0 * outer).ceil() as usize + 8;
    let layout = MacLayout::new(side, side, 1.0);
    let centre = DVec2::splat(0.5 * side as f64);
    let pi = std::f64::consts::PI;
    let frame = f64::from(FRAME);
    let frames = (period / frame).round().max(1.0) as usize;
    let per_frame = match substep {
        Substep::Frame => 1,
        _ => (frame / f64::from(substep.length(1.0, c_p as f32))).ceil() as usize,
    };
    let dt = period / (frames * per_frame) as f64;
    let steps = frames * per_frame;
    let rate = pi / period;
    // g = G sin(pi t / T0) and its first two time derivatives;
    // `EMERGE_PROBE_VORTEX_G` sets G (default 1, the review's).
    let amplitude: f64 = std::env::var("EMERGE_PROBE_VORTEX_G")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let g = |t: f64| amplitude * (rate * t).sin();
    let g1 = |t: f64| amplitude * rate * (rate * t).cos();
    let g2 = |t: f64| -amplitude * rate * rate * (rate * t).sin();
    // h and h' at a radius.
    let shape = |r: f64| {
        let s = (r - middle) / (inner - outer);
        let w = 1.0 - 4.0 * s * s;
        (w * w, -16.0 * s * w / (inner - outer))
    };
    let turn =
        |a: f64| DMat2::from_cols(DVec2::new(a.cos(), a.sin()), DVec2::new(-a.sin(), a.cos()));
    let perp = |d: DVec2| DVec2::new(-d.y, d.x);
    let u = |q: DVec2, t: f64| {
        let r = q - centre;
        turn(g(t) * shape(r.length()).0) * r - r
    };
    let v = |q: DVec2, t: f64| {
        let r = q - centre;
        let h = shape(r.length()).0;
        g1(t) * h * perp(turn(g(t) * h) * r)
    };
    let acceleration = |q: DVec2, t: f64| {
        let r = q - centre;
        let h = shape(r.length()).0;
        let turned = turn(g(t) * h) * r;
        g2(t) * h * perp(turned) - (g1(t) * h).powi(2) * turned
    };
    let grad = |q: DVec2, t: f64| {
        let r = q - centre;
        let length = r.length();
        let (h, dh) = shape(length);
        let q_turn = turn(g(t) * h);
        let a = perp(q_turn * r);
        let b = r * (g(t) * dh / length);
        q_turn + DMat2::from_cols(a * b.x, a * b.y)
    };
    let stress = |q: DVec2, t: f64| solid.first_pk(grad(q, t));
    let divergence = |q: DVec2, t: f64, delta: f64| {
        let dx = DVec2::new(delta, 0.0);
        let dy = DVec2::new(0.0, delta);
        (stress(q + dx, t).x_axis - stress(q - dx, t).x_axis + stress(q + dy, t).y_axis
            - stress(q - dy, t).y_axis)
            / (2.0 * delta)
    };
    let delta = 1e-3;
    let body = |q: DVec2, t: f64| acceleration(q, t) - divergence(q, t, delta);
    // Every step's solve, when asked.
    let progress = std::env::var("EMERGE_PROBE_VORTEX_PROGRESS").is_ok();
    // No face held: the motion is fixed and free of traction on both
    // circles, so it is exact for a free ring too.
    // `EMERGE_PROBE_VORTEX_BOUNDARY`: `held` (default, the faces outside
    // the ring held), `free` (none held) or `web` (both components weighted
    // by `w = (R - Ri)(Ro - R)/(Ro - Ri)`, zero on both circles, the
    // web-splines of Hollig and Reif 2003 for the clamped ring; no face held).
    let boundary = std::env::var("EMERGE_PROBE_VORTEX_BOUNDARY").unwrap_or_else(|_| {
        if std::env::var("EMERGE_PROBE_VORTEX_FREE").is_ok() {
            "free".into()
        } else {
            "held".into()
        }
    });
    let free = boundary != "held";
    let web = boundary == "web";
    // Quadrature points: particles two by two per cell (those in the ring),
    // or with `EMERGE_PROBE_VORTEX_QUADRATURE=gauss` the ring's own
    // integrals (`ring_quadrature`, refinement depth
    // `EMERGE_PROBE_VORTEX_DEPTH`, default 4).
    let gauss = std::env::var("EMERGE_PROBE_VORTEX_QUADRATURE").as_deref() == Ok("gauss");
    let depth: u32 = std::env::var("EMERGE_PROBE_VORTEX_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let (reference, v0): (Vec<DVec2>, Vec<f64>) = if gauss {
        ring_quadrature(centre, (inner, outer), side, depth)
    } else {
        (0..2 * side)
            .flat_map(|j| {
                (0..2 * side)
                    .map(move |i| DVec2::new((i as f64 + 0.5) * 0.5, (j as f64 + 0.5) * 0.5))
            })
            .filter(|&q| {
                let r = (q - centre).length();
                (inner..=outer).contains(&r)
            })
            .map(|q| (q, 0.25))
            .unzip()
    };
    let n = reference.len();
    // The ring's area against the quadrature's: the geometric error.
    let area_error =
        v0.iter().sum::<f64>() / (std::f64::consts::PI * (outer * outer - inner * inner)) - 1.0;
    let mass = v0.clone();
    let solids = vec![solid; n];
    let mut x = reference.clone();
    let mut vel: Vec<DVec2> = reference.iter().map(|&q| v(q, 0.0)).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let held = |a: usize, i: usize, j: usize| -> Option<f64> {
        let at =
            DVec2::new(i as f64, j as f64) + crate::spacetime::total_lagrangian::lattice_offset(a);
        let r = (at - centre).length();
        (!free && (r < inner || r > outer)).then_some(1.0)
    };
    let settings = crate::spacetime::total_lagrangian::ConservingSettings {
        newton_tolerance,
        occupation: match std::env::var("EMERGE_PROBE_OCCUPATION").as_deref() {
            Ok("none") => None,
            _ => crate::spacetime::total_lagrangian::ConservingSettings::default().occupation,
        },
        ..Default::default()
    };
    // L2 norms weighted by each point's volume.
    let norm = |values: &mut dyn Iterator<Item = f64>| values.sum::<f64>().sqrt();
    let peak = 0.5 * period;
    let u_scale = norm(&mut (0..n).map(|p| v0[p] * u(reference[p], peak).length_squared()));
    let v_scale = norm(&mut (0..n).map(|p| v0[p] * v(reference[p], 0.0).length_squared()));
    let p_scale = norm(&mut (0..n).map(|p| {
        let s = stress(reference[p], peak);
        v0[p] * (s.x_axis.length_squared() + s.y_axis.length_squared())
    }));
    let exact_strain = reference
        .iter()
        .map(|&q| {
            let d = grad(q, peak) - DMat2::IDENTITY;
            (d.x_axis.length_squared() + d.y_axis.length_squared()).sqrt()
        })
        .fold(0.0f64, f64::max);
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0[p] * solid.energy(f[p]))
            .sum()
    };
    let errors = |x: &[DVec2], vel: &[DVec2], f: &[DMat2], t: f64| -> [f64; 3] {
        let du = norm(
            &mut (0..n)
                .map(|p| v0[p] * (x[p] - reference[p] - u(reference[p], t)).length_squared()),
        );
        let dv = norm(&mut (0..n).map(|p| v0[p] * (vel[p] - v(reference[p], t)).length_squared()));
        let dp = norm(&mut (0..n).map(|p| {
            let d = solid.first_pk(f[p]) - stress(reference[p], t);
            v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
        }));
        [du / u_scale, dv / v_scale, dp / p_scale]
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
    let mut worst = [0.0f64; 3];
    let (mut newton, mut gmres, mut most, mut unconverged, mut cost) =
        (0u64, 0u64, 0u32, 0u32, 0.0f64);
    let mut strain = 0.0f64;
    let mut cache = web.then(|| {
        crate::spacetime::total_lagrangian::LagrangianCache::weighted(
            &layout,
            &reference,
            (&mass, &v0),
            settings.occupation,
            &|_, q| {
                let r = q - centre;
                let length = r.length();
                let w = (length - inner) * (outer - length) / (outer - inner);
                let slope = (outer + inner - 2.0 * length) / (outer - inner);
                (w, r * (slope / length.max(f64::MIN_POSITIVE)))
            },
        )
    });
    // The least det F over the run, the whole ring's and the middle half's
    // (|s| <= 1/4), and an unconverged step stops the run uncommitted with
    // `EMERGE_PROBE_MS_STOP`.
    let (mut least_all, mut least_middle) = (f64::INFINITY, f64::INFINITY);
    // The least det F within a tenth of the width of each circle, and the
    // filled fraction of each point's half cell (its points' volume over
    // the half cell's): where a cut cell sits against the defect.
    let (mut least_inner, mut least_outer) = (f64::INFINITY, f64::INFINITY);
    let half_cell =
        |q: DVec2| (q.x / 0.5).floor() as usize + 2 * side * (q.y / 0.5).floor() as usize;
    let mut filled = vec![0.0f64; 4 * side * side];
    for p in 0..n {
        filled[half_cell(reference[p])] += v0[p] / 0.25;
    }
    let stop = std::env::var("EMERGE_PROBE_MS_STOP").is_ok();
    let mut stopped = None;
    let mut t = 0.0;
    for step in 0..steps {
        let b: Vec<DVec2> = reference.iter().map(|&q| body(q, t + 0.5 * dt)).collect();
        let before = x.clone();
        let kept = stop.then(|| (vel.clone(), f.clone()));
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0,
        };
        let report = if lagrangian {
            crate::spacetime::total_lagrangian::step_total_lagrangian(
                &layout,
                particles,
                &reference,
                &solids,
                (DVec2::ZERO, Some(&b), dt),
                (
                    if web {
                        &|_: usize, _: usize, _: usize| -> Option<f64> { None }
                    } else {
                        &held
                    },
                    &settings,
                ),
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
            unconverged += 1;
            stopped = Some(step + 1);
            break;
        }
        t += dt;
        for p in 0..n {
            let det = f[p].determinant();
            least_all = least_all.min(det);
            let s_p = ((reference[p] - centre).length() - middle) / (outer - inner);
            if s_p <= -0.4 {
                least_inner = least_inner.min(det);
            }
            if s_p >= 0.4 {
                least_outer = least_outer.min(det);
            }
            if s_p.abs() <= 0.25 {
                least_middle = least_middle.min(det);
            }
        }
        work += (0..n)
            .map(|p| mass[p] * b[p].dot(x[p] - before[p]))
            .sum::<f64>();
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        most = most.max(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        if progress {
            let (least, at) = f
                .iter()
                .enumerate()
                .map(|(p, m)| (m.determinant(), p))
                .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
            let s_at = ((reference[at] - centre).length() - middle) / (outer - inner);
            // The least det F in the ring's middle half, |s| <= 1/4, the
            // same band of the body at every resolution, and where it sits.
            let (inside, inside_at) = (0..n)
                .filter(|&p| {
                    let r = (reference[p] - centre).length();
                    ((r - middle) / (outer - inner)).abs() <= 0.25
                })
                .map(|p| (f[p].determinant(), p))
                .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
            let s_inside = ((reference[inside_at] - centre).length() - middle) / (outer - inner);
            let edge = |near: &dyn Fn(f64) -> bool| {
                (0..n)
                    .filter(|&p| {
                        near(((reference[p] - centre).length() - middle) / (outer - inner))
                    })
                    .map(|p| f[p].determinant())
                    .fold(f64::INFINITY, f64::min)
            };
            let (edge_in, edge_out) = (edge(&|s| s <= -0.4), edge(&|s| s >= 0.4));
            let fill = filled[half_cell(reference[at])];
            let offset = reference[at] - centre;
            let angle = offset.y.atan2(offset.x).to_degrees();
            println!(
                "  step {} t {t:.4}: Newton {} GMRES {} converged {} residual {:.2e} start residual {:.2e}, least det F {least:.4} at s = {s_at:+.3} (the circles at -0.5 and +0.5), {inside:.4} at s = {s_inside:+.3} in the middle half; edges inner {edge_in:.4} outer {edge_out:.4}; least point's half cell filled {fill:.3}, angle {angle:.0} deg",
                step + 1,
                report.newton,
                report.gmres,
                report.converged,
                report.residual,
                report.start_residual
            );
        }
        if (step + 1) % per_frame == 0 || step + 1 == steps {
            let now = errors(&x, &vel, &f, t);
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
    let balance = (energy(&vel, &f) - e0 - work) / largest_energy;
    let realism = |e: f64| if e.is_nan() { 0.0 } else { (1.0 - e).max(0.0) };
    let per = steps.max(1) as f64;
    let boundary_label = format!(", {boundary} boundary");
    let quadrature_label = if gauss {
        format!(", ring quadrature depth {depth} ({n} points, area error {area_error:+.1e})")
    } else {
        format!(", particles ({n}, area error {area_error:+.1e})")
    };
    let stopped_label = stopped.map_or(String::new(), |k| {
        format!(" STOPPED at step {k} (unconverged, not committed);")
    });
    format!(
        "vortex G {amplitude}, {}{boundary_label}{quadrature_label} at {substep}, h = width/{cells}, T0 {period} s (w dt {:.3}, {n} particles, divergence step check {step_check:.1e}): {steps} steps to t = {t:.4} s; error end disp {:.2e} vel {:.2e} P {:.2e}, worst disp {:.2e} vel {:.2e} P {:.2e}; realism at worst disp {:.1} % vel {:.1} % P {:.1} %; energy balance {balance:+.2e}; Newton {:.2} per step, GMRES {:.1} per step and at most {most}, unconverged {unconverged}; {:.1} s per simulated s; largest |F - I| {strain:.3} (exact {exact_strain:.3}); least det F over the run {least_all:.4}, middle half {least_middle:.4}, inner edge {least_inner:.4}, outer edge {least_outer:.4} (exact 1);{stopped_label}",
        if lagrangian {
            "reference grid"
        } else {
            "grid rebuilt"
        },
        rate * dt,
        end[0],
        end[1],
        end[2],
        worst[0],
        worst[1],
        worst[2],
        100.0 * realism(worst[0]),
        100.0 * realism(worst[1]),
        100.0 * realism(worst[2]),
        newton as f64 / per,
        gmres as f64 / per,
        cost / t
    )
}

/// The generalized vortex (`vortex_run`) over its sweep, a probe:
/// `EMERGE_PROBE_VORTEX_METHODS` (`reference,rebuilt`), `_STEPS` (`frame`),
/// `_CELLS` (`8,16,32`, cells across the ring), `_PERIODS` (`1,0.1,0.05`,
/// `T0` in s, whole frames), `_PROGRESS` to print every step's solve,
/// `_FREE` to hold no face;
/// `EMERGE_PROBE_NEWTON_TOLERANCE` (default the step's);
/// `EMERGE_PROBE_OCCUPATION=none` for the basis without the occupation
/// threshold; `EMERGE_PROBE_VORTEX_QUADRATURE=gauss` (`_DEPTH`) for the ring's
/// own integrals, `EMERGE_PROBE_VORTEX_BOUNDARY=held|free|web`,
/// `EMERGE_PROBE_VORTEX_G` the amplitude G (default 1),
/// `EMERGE_PROBE_MS_STOP` to stop at an unconverged step.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_generalized_vortex() {
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    let tolerance: f64 = std::env::var("EMERGE_PROBE_NEWTON_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(
            crate::spacetime::total_lagrangian::ConservingSettings::default().newton_tolerance,
        );
    for method in list("EMERGE_PROBE_VORTEX_METHODS", "reference,rebuilt") {
        for step in list("EMERGE_PROBE_VORTEX_STEPS", "frame") {
            for cells in list("EMERGE_PROBE_VORTEX_CELLS", "8,16,32") {
                for period in list("EMERGE_PROBE_VORTEX_PERIODS", "1,0.1,0.05") {
                    let (Ok(cells), Ok(period)) = (cells.parse(), period.parse()) else {
                        continue;
                    };
                    println!(
                        "{}",
                        vortex_run(Vortex {
                            lagrangian: method == "reference",
                            substep: Substep::parse(&step).unwrap_or(Substep::Frame),
                            cells,
                            period,
                            newton_tolerance: tolerance,
                        })
                    );
                }
            }
        }
    }
}
