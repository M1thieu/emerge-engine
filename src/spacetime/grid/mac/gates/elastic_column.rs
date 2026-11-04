use super::*;

#[test]
#[ignore = "staggered elastic solid gate, confined column at rest: run with --ignored --nocapture"]
fn gate_elastic_column() {
    let mut report = Report::new();
    let nu = 0.3f32;
    // (lambda + mu) / rho = E / rho * (nu / ((1 + nu)(1 - 2 nu)) + 1 / (2 (1 + nu))).
    let stiffness_per_e = nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) + 0.5 / (1.0 + nu);
    let e_over_rho = 100.0 / stiffness_per_e;
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let h0 = floor + depth as f32;
    let x = seed_body(
        true,
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_elastic(e_over_rho, nu);
    let g = scene.g();
    let cp2 = scene.shear.unwrap_or(0.0) + scene.sound_speed2.unwrap_or(0.0);
    let centre_i = nx / 2;
    let frames = (5.0 / FRAME).round() as u32;
    let last_second = frames - (1.0 / FRAME).round() as u32;
    let bands = depth / 5;
    let mut band_j = vec![(0.0f64, 0u64); bands];
    let (mut h_sum, mut h_n, mut nan, mut outside) = (0.0f32, 0u32, false, 0usize);
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    for f in 1..=frames {
        let frame = scene.frame();
        cost.add(&frame);
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        outside = outside.max(
            (0..scene.x.len())
                .filter(|&p| !scene.inside_tank(p))
                .count(),
        );
        if f > last_second {
            if let Some(h) = surface_height(&frame.phi, centre_i, wall) {
                h_sum += h;
                h_n += 1;
            }
            for p in 0..scene.x.len() {
                let k = ((scene.x[p].y - floor) / 5.0).floor();
                if k >= 0.0 && (k as usize) < bands {
                    let entry = &mut band_j[k as usize];
                    entry.0 += f64::from(scene.j(p));
                    entry.1 += 1;
                }
            }
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let h = h_sum / h_n.max(1) as f32;
    report.check(!nan, || "elastic column: NaN".to_string());
    report.check(outside == 0, || {
        format!("elastic column: {outside} particles out")
    });
    for (k, &(sum, n)) in band_j.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let mean = (sum / n as f64) as f32;
        let y = floor + 5.0 * k as f32 + 2.5;
        let expected = 1.0 - g * (h - y) / cp2;
        println!("  elastic band {k}: mean J {mean:.5}, expected {expected:.5}");
        report.check((mean - expected).abs() <= 0.005, || {
            format!("elastic column: band {k} mean J {mean:.5}, expected {expected:.5}")
        });
    }
    scene.check_reflections("elastic column", &mut report);
    cost.print("elastic column", wall_s, 5.0);
    report.finish();
}

#[test]
#[ignore = "staggered elastic solid gate, plane shear wave: run with --ignored --nocapture"]
fn gate_shear_wave() {
    let mut report = Report::new();
    let (nu, c_s) = (0.3f32, 5.0f32);
    let (length, height) = (100usize, 208usize);
    let (nx, ny) = (length + 8, height + 8);
    let (x0, y0) = (4.0f32, 4.0f32);
    let tank = (
        Vec2::splat(1.0),
        Vec2::new(nx as f32 - 1.0, ny as f32 - 1.0),
    );
    let mut rng = LcgRng::new(13);
    let top = y0 + height as f32;
    let right = x0 + length as f32;
    let x = seed_body(
        true,
        (
            IVec2::new(x0 as i32, y0 as i32),
            IVec2::new(right as i32, top as i32),
        ),
        |_| true,
        |q| {
            let d = [
                (q.y - y0, Vec2::NEG_Y),
                (top - q.y, Vec2::Y),
                (q.x - x0, Vec2::NEG_X),
                (right - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    let c = c_s / scene.cell_m;
    scene.fixed_dt = Some(0.5 * scene.layout.dx / c);
    // Amended with the free body (module doc): the solid's particles carry the
    // P wave now, so the explicit step follows it, as `plane_wave` sets it.
    let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
    scene.fixed_dt = Some(0.5 * scene.layout.dx / c_p.max(c));
    let kick = 0.05 / scene.cell_m;
    for p in 0..scene.x.len() {
        if scene.x[p].x < x0 + 10.0 {
            scene.v[p].y = kick;
        }
    }
    let mid = y0 + 0.5 * height as f32;
    scene.probe.watch_x = Some((x0 + 50.0, mid - 4.0, mid + 4.0));
    let expected = 40.0 / c;
    // The window: the front's plateau ends at 50 dx / c_s, the edges'
    // disturbance arrives at 53 (criteria).
    let window = 50.0 / c;
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    while scene.time < window {
        let frame = scene.frame();
        cost.add(&frame);
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let history = &scene.probe.watch_x_history;
    let half = 0.25 * kick;
    let arrival = history.windows(2).find_map(|pair| {
        let ((t0, v0), (t1, v1)) = (pair[0], pair[1]);
        (v0 < half && v1 >= half).then(|| t0 + (t1 - t0) * (half - v0) / (v1 - v0))
    });
    let plateau = history
        .iter()
        .filter(|(t, _)| *t > expected && *t < window)
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let error = arrival.map_or(f32::INFINITY, |t| (t - expected).abs() / expected);
    println!(
        "shear wave: v0/4 reached at t={:?}s, expected {expected:.4}s, off by {:.1} %; largest velocity after it {:.3} of v0 (d'Alembert: 0.5)",
        arrival,
        100.0 * error,
        plateau / kick
    );
    report.check(error <= 0.05, || {
        format!("shear wave: arrival off by {:.1} %", 100.0 * error)
    });
    scene.check_reflections("shear wave", &mut report);
    cost.print("shear wave", wall_s, scene.time);
    report.finish();
}

#[test]
#[ignore = "staggered elastic solid gate, dropped block: run with --ignored --nocapture"]
fn gate_elastic_drop() {
    let mut report = Report::new();
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed_body(
        true,
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    let gyration = |scene: &Scene| {
        let (com, _) = scene.centre_of_mass();
        let n = scene.x.len() as f32;
        (scene
            .x
            .iter()
            .map(|q| (*q - com).length_squared())
            .sum::<f32>()
            / n)
            .sqrt()
    };
    let (energy_start, gyration_start) = (scene.energy(), gyration(&scene));
    let (mut worst_energy, mut nan, mut outside) = (f32::NEG_INFINITY, false, 0usize);
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    for f in 1..=(2.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        outside = outside.max(
            (0..scene.x.len())
                .filter(|&p| !scene.inside_tank(p))
                .count(),
        );
        worst_energy = worst_energy.max(scene.energy() / energy_start - 1.0);
        if f % 12 == 0 {
            println!(
                "  elastic drop t={:.2}s: energy {:+.4}, gyration {:.3} (start {gyration_start:.3}), lowest y {:.2}",
                scene.time,
                scene.energy() / energy_start - 1.0,
                gyration(&scene),
                scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min)
            );
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let change = (gyration(&scene) / gyration_start - 1.0).abs();
    println!(
        "elastic drop: NaN {nan}, most outside {outside}, energy rise {worst_energy:+.4}, gyration change {change:.4}"
    );
    report.check(!nan, || "elastic drop: NaN".to_string());
    report.check(outside == 0, || {
        format!("elastic drop: {outside} particles out")
    });
    report.check(worst_energy <= 0.01, || {
        format!("elastic drop: energy rose {worst_energy:+.4}")
    });
    report.check(change <= 0.05, || {
        format!("elastic drop: gyration changed {change:.4}")
    });
    scene.check_reflections("elastic drop", &mut report);
    cost.print("elastic drop", wall_s, 2.0);
    report.finish();
}

/// Diagnosis of the dropped elastic block: its energy split into its
/// parts every frame. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_elastic_drop_energy_parts() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    // EMERGE_PROBE_SCENARIO: "drop" (the dropped block), "rest" (on the floor,
    // at rest), "spin" (no gravity, no contact, spinning at 2 rad/s).
    let scenario = std::env::var("EMERGE_PROBE_SCENARIO").unwrap_or_else(|_| "drop".into());
    let base = if scenario == "rest" { 4.0 } else { 24.0 };
    let scenario = scenario.as_str();
    let (lo, hi) = (Vec2::new(31.0, base), Vec2::new(41.0, base + 10.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let spinning = scenario.starts_with("spin");
    let g_fraction = if spinning { 0.0 } else { 1.0 };
    let mut scene = Scene::new((72, 72), 0.01, g_fraction, tank, x);
    if scenario != "spin_liquid" {
        scene = scene.with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    }
    if spinning {
        let centre = 0.5 * (lo + hi);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            let w = std::env::var("EMERGE_PROBE_OMEGA")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(2.0);
            if std::env::var_os("EMERGE_PROBE_TRANSLATE").is_some() {
                scene.v[p] = Vec2::new(w * 5.0, 0.0);
            } else {
                scene.v[p] = w * Vec2::new(-r.y, r.x);
                scene.c[p] = Mat2::from_cols(Vec2::new(0.0, w), Vec2::new(-w, 0.0));
            }
        }
    }
    let (mu, lambda) = (
        scene.shear.unwrap_or(0.0),
        scene.sound_speed2.unwrap_or(0.0),
    );
    let parts = |s: &Scene| {
        let floor = s.tank.0.y;
        let mut out = [0.0f64; 5];
        for p in 0..s.x.len() {
            out[0] += f64::from(s.mass[p] * 0.5 * s.v[p].length_squared());
            out[1] += f64::from(s.mass[p] * s.g() * (s.x[p].y - floor));
            out[2] += f64::from(s.v0[p] * shear_energy(s.deformation[p], mu));
            out[3] += f64::from(s.v0[p] * 0.5 * lambda * (s.j(p) - 1.0).powi(2));
            out[4] += f64::from(s.deformation[p].determinant() / s.j(p) - 1.0).abs();
        }
        out[4] /= s.x.len() as f64;
        out
    };
    let start = parts(&scene);
    let total0 = (start[0] + start[1] + start[2] + start[3]).max(1.0);
    for f in 1..=72u32 {
        scene.frame();
        if f <= 2 && spinning {
            for p in [0usize, scene.x.len() / 2] {
                let fm = scene.deformation[p];
                let r = super::elastic::rotation(fm);
                let fh = fm * fm.determinant().sqrt().recip();
                println!(
                    "  particle {p} at {:?}: F {:?} (det {:.5}), angle of R {:.4}, |F^ - R| {:.5}, C {:?}",
                    scene.x[p],
                    fm,
                    fm.determinant(),
                    r.x_axis.y.atan2(r.x_axis.x),
                    (fh - r).x_axis.length() + (fh - r).y_axis.length(),
                    scene.c[p]
                );
            }
            // Strain rate |sym C| of the particles within one cell of the
            // block's edge against those deeper in, after this frame.
            let (lo_now, hi_now) = scene.x.iter().fold(
                (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
                |(a, b), q| (a.min(*q), b.max(*q)),
            );
            let (mut edge, mut ne, mut inner, mut ni) = (0.0f32, 0u32, 0.0f32, 0u32);
            for p in 0..scene.x.len() {
                let q = scene.x[p];
                let d = (q - lo_now).min(hi_now - q).min_element();
                let c = scene.c[p];
                let sym = 0.5 * (c + c.transpose());
                let rate = (sym.x_axis.length_squared() + sym.y_axis.length_squared()).sqrt();
                if d < 1.0 {
                    edge += rate;
                    ne += 1;
                } else if d > 2.0 {
                    inner += rate;
                    ni += 1;
                }
            }
            println!(
                "  strain rate after frame {f}: edge {:.3} 1/s ({ne} particles), interior {:.3} 1/s ({ni}); rotation rate 2",
                edge / ne.max(1) as f32,
                inner / ni.max(1) as f32
            );
        }
        let e = parts(&scene);
        let total = e[0] + e[1] + e[2] + e[3];
        println!(
            "f{f:3} t={:.3}: total {:+.4} | kinetic {:.4} potential {:.4} shear {:.4} volume {:.4} (fractions of start) | mean |det F / J - 1| {:.5}",
            scene.time,
            total / total0 - 1.0,
            e[0] / total0,
            e[1] / total0,
            e[2] / total0,
            e[3] / total0,
            e[4]
        );
    }
}

/// Diagnosis of the staggered elastic solid: one substep of a rigidly spinning
/// liquid block, no gravity, and the velocity gradient at an edge particle
/// after each stage. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_rigid_spin_one_substep() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 0.0, tank, x);
    let centre = 0.5 * (lo + hi);
    for p in 0..scene.x.len() {
        let r = scene.x[p] - centre;
        scene.v[p] = 2.0 * Vec2::new(-r.y, r.x);
        scene.c[p] = Mat2::from_cols(Vec2::new(0.0, 2.0), Vec2::new(-2.0, 0.0));
    }
    let edge = (0..scene.x.len())
        .min_by(|&a, &b| scene.x[a].y.total_cmp(&scene.x[b].y))
        .unwrap_or(0);
    let dt = 1.0e-3;
    let (mut vel, face_mass) =
        particles_to_faces(&scene.layout, &scene.x, &scene.v, &scene.c, &scene.mass);
    let read = |vel: &super::field::MacVelocity| {
        let mut v = [Vec2::ZERO];
        let mut c = [Mat2::ZERO];
        faces_to_particles(&scene.layout, vel, &[scene.x[edge]], &mut v, &mut c);
        (v[0], c[0])
    };
    println!(
        "edge particle at {:?}, exact v {:?}",
        scene.x[edge], scene.v[edge]
    );
    println!("after P2G: {:?}", read(&vel));
    let phi = scene.phi();
    for j in 21..27usize {
        let i = 38usize;
        let q = scene.layout.u_position(i, j);
        let g = phi.bilinear_gradient(q - Vec2::splat(0.5));
        println!(
            "  u face ({i},{j}) at y={:.1}: u {:.3} (exact {:.3}), phi left/right {:.3}/{:.3}, grad phi {:?}",
            q.y,
            vel.u.get(i, j),
            -2.0 * (q.y - centre.y),
            phi.get(i - 1, j),
            phi.get(i, j),
            g
        );
    }
    let image = box_container_image(tank.0, tank.1);
    let mut projected = vel.clone();
    let solution = project(
        &scene.layout,
        dt,
        &mut projected,
        Walls {
            weights: &scene.weights,
            image: &image,
            open: &scene.open_centres(),
        },
        Liquid {
            phi: &phi,
            face_mass: &face_mass,
        },
        &scene.settings,
        None,
    );
    println!(
        "after projection + extension: {:?}, largest pressure {}",
        read(&projected),
        solution.pressure.iter().fold(0.0f32, |m, p| m.max(p.abs()))
    );
    vel = projected;
    let _ = vel;
}

/// Diagnosis of the staggered elastic solid: the transfer round trip alone
/// (scatter, projection, gather, particles held still, no gravity) on a resting
/// column with small random velocities. Kinetic energy must never grow. A
/// probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_transfer_round_trip_energy() {
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let h = (wall + depth) as f32;
    let tank = (
        Vec2::new(wall as f32, wall as f32),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x);
    let mut noise = LcgRng::new(3);
    for p in 0..scene.x.len() {
        scene.v[p] = Vec2::new(noise.next_f32() - 0.5, noise.next_f32() - 0.5);
    }
    let phi = scene.phi();
    let image = box_container_image(scene.tank.0, scene.tank.1);
    let kinetic = |s: &Scene| -> f64 {
        (0..s.x.len())
            .map(|p| f64::from(0.5 * s.mass[p] * s.v[p].length_squared()))
            .sum()
    };
    let mut previous = kinetic(&scene);
    let mut rises = 0;
    for round in 1..=200 {
        let (mut vel, face_mass) =
            particles_to_faces(&scene.layout, &scene.x, &scene.v, &scene.c, &scene.mass);
        project(
            &scene.layout,
            1.0e-3,
            &mut vel,
            Walls {
                weights: &scene.weights,
                image: &image,
                open: &scene.open_centres(),
            },
            Liquid {
                phi: &phi,
                face_mass: &face_mass,
            },
            &scene.settings,
            None,
        );
        faces_to_particles(&scene.layout, &vel, &scene.x, &mut scene.v, &mut scene.c);
        let now = kinetic(&scene);
        if now > previous * (1.0 + 1.0e-6) {
            rises += 1;
        }
        if round <= 5 || round % 40 == 0 {
            println!(
                "round {round}: kinetic {now:.6e} ({:+.3e} of the round before)",
                now / previous - 1.0
            );
        }
        previous = now;
    }
    println!("rounds where kinetic energy rose: {rises} of 200");
}

/// Diagnosis of the dropped elastic block, which, resting on the floor at
/// 1 s, went to NaN before 1.2 s. Substep by substep from
/// 0.95 s: the largest `|C|`, speed, `|F^ - R^|` and J range, until the
/// first value that is not finite. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_elastic_drop_first_nan() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    while scene.time < 0.95 {
        scene.frame();
    }
    let mu = scene.shear.unwrap_or(0.0);
    let dt = 0.5 * scene.layout.dx / mu.sqrt();
    let mut frame = scene.frame();
    for step in 0..20000 {
        scene.substep(dt, &mut frame);
        scene.time += dt;
        let bad: Vec<usize> = (0..scene.x.len())
            .filter(|&p| {
                !(scene.x[p].is_finite()
                    && scene.v[p].is_finite()
                    && scene.c[p].is_finite()
                    && scene.deformation[p].is_finite()
                    && scene.log_j[p].is_finite())
            })
            .collect();
        let (mut c_max, mut c_at, mut speed, mut dev, mut j_lo, mut j_hi) =
            (0.0f32, 0usize, 0.0f32, 0.0f32, f32::INFINITY, 0.0f32);
        for p in 0..scene.x.len() {
            let c = scene.c[p]
                .to_cols_array()
                .iter()
                .fold(0.0f32, |m, e| m.max(e.abs()));
            if c > c_max {
                c_max = c;
                c_at = p;
            }
            speed = speed.max(scene.v[p].length());
            dev = dev.max(shear_energy(scene.deformation[p], 1.0).sqrt());
            j_lo = j_lo.min(scene.j(p));
            j_hi = j_hi.max(scene.j(p));
        }
        if step % 200 == 0 || !bad.is_empty() || c_max > 500.0 {
            println!(
                "t={:.5} step {step}: |C| max {c_max:.3} at {:?} (F det {:.4}, J {:.4}), speed {speed:.3}, |F^-R^| max {dev:.4}, J {j_lo:.4}..{j_hi:.4}",
                scene.time,
                scene.x[c_at],
                scene.deformation[c_at].determinant(),
                scene.j(c_at)
            );
        }
        if !bad.is_empty() {
            for &p in bad.iter().take(5) {
                println!(
                    "  bad {p}: x {:?} v {:?} F {:?} log J {}",
                    scene.x[p], scene.v[p], scene.deformation[p], scene.log_j[p]
                );
            }
            break;
        }
    }
}

/// Diagnosis of the dropped elastic block: when do its particles lose
/// every neighbour within one cell (an
/// elastic body must never shed a piece)? A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_elastic_drop_detached() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    for f in 1..=66u32 {
        scene.frame();
        let n = scene.x.len();
        let mut lonely = Vec::new();
        for p in 0..n {
            let nearest = (0..n)
                .filter(|&q| q != p)
                .map(|q| (scene.x[q] - scene.x[p]).length())
                .fold(f32::INFINITY, f32::min);
            if nearest > 1.0 {
                lonely.push((p, nearest));
            }
        }
        let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
        let speed = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        println!(
            "f{f} t={:.3}: {} particles with no neighbour within 1 cell, lowest y {lowest:.2}, top speed {speed:.1}",
            scene.time,
            lonely.len()
        );
        for &(p, d) in lonely.iter().take(3) {
            println!(
                "   {p} at {:?}, nearest {d:.2}, v {:?}, F det {:.3}, J {:.3}",
                scene.x[p],
                scene.v[p],
                scene.deformation[p].determinant(),
                scene.j(p)
            );
        }
    }
}
