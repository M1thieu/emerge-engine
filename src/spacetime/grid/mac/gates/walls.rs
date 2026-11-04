use super::*;

/// Diagnosis of the dropped elastic block: the mirror wall stands for
/// an image body, so a block hitting the floor must move exactly as one of
/// two identical blocks hitting each other head on, with no wall near.
/// No gravity; kinetic plus elastic energy and the top speed of each case
/// (`EMERGE_PROBE_WALL_CASE` = wall or pair). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_wall_against_image_pair() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let case = std::env::var("EMERGE_PROBE_WALL_CASE").unwrap_or_else(|_| "wall".into());
    let tank = (Vec2::splat(4.0), Vec2::new(68.0, 106.0));
    let block = |lo: Vec2, rng: &mut LcgRng| {
        let hi = lo + Vec2::splat(10.0);
        seed(
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
            rng,
        )
    };
    let mut rng = LcgRng::new(17);
    // Wall: base 6 cells above the floor (y = 4). Pair: the same gap to the
    // plane y = 46 on both sides.
    let (mut x, below) = if case == "pair" {
        let lower = block(Vec2::new(31.0, 30.0), &mut rng);
        let n = lower.len();
        let upper: Vec<Vec2> = lower.iter().map(|q| Vec2::new(q.x, 92.0 - q.y)).collect();
        let mut all = upper;
        all.extend(lower);
        (all, n)
    } else {
        (block(Vec2::new(31.0, 10.0), &mut rng), 0)
    };
    x.shrink_to_fit();
    let mut scene = Scene::new((72, 110), 0.01, 0.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    let speed0 = 2.0 / scene.cell_m;
    let n = scene.x.len();
    for p in 0..n {
        // The upper block (all of the wall case, the first half of the
        // pair) moves down, the lower block of the pair up.
        let down = case != "pair" || p < n - below;
        scene.v[p].y = if down { -speed0 } else { speed0 };
    }
    let (mu, lambda) = (
        scene.shear.unwrap_or(0.0),
        scene.sound_speed2.unwrap_or(0.0),
    );
    let energy = |s: &Scene| -> f64 {
        (0..s.x.len())
            .map(|p| {
                f64::from(
                    s.mass[p] * 0.5 * s.v[p].length_squared()
                        + s.v0[p]
                            * (shear_energy(s.deformation[p], mu)
                                + 0.5 * lambda * (s.j(p) - 1.0).powi(2)),
                )
            })
            .sum::<f64>()
            / s.x.len() as f64
    };
    let e0 = energy(&scene);
    if std::env::var_os("EMERGE_PROBE_TRACE").is_some() {
        let start: f32 = std::env::var("EMERGE_PROBE_TRACE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.25);
        while scene.time < start {
            scene.frame();
        }
        let mut frame = scene.frame();
        for step in 0..3000 {
            // The frame's own choice of substep (`Scene::frame`).
            let speed = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            let dt = (0.5 * scene.layout.dx / mu.sqrt()).min(travel_limited_dt(
                speed,
                scene.g(),
                scene.layout.dx,
                scene.settings.max_cells_per_substep,
            ));
            scene.substep(dt, &mut frame);
            scene.time += dt;
            let (mut c_max, mut at) = (0.0f32, 0usize);
            for p in 0..scene.x.len() {
                let c = scene.c[p].to_cols_array().iter().fold(0.0f32, |m, e| {
                    if e.is_finite() {
                        m.max(e.abs())
                    } else {
                        f32::NAN
                    }
                });
                if c.is_nan() {
                    c_max = c;
                    at = p;
                    break;
                }
                if c > c_max {
                    c_max = c;
                    at = p;
                }
            }
            let f = scene.deformation[at];
            let fh = f / f.determinant().abs().sqrt();
            let tau = shear_kirchhoff(f, mu);
            let phi = scene.phi();
            let cell = scene.x[at].floor().as_ivec2();
            let near = (0..scene.x.len())
                .filter(|&q| (scene.x[q] - scene.x[at]).length() < 1.0)
                .count();
            if step % 5 == 0 || c_max > 300.0 || !c_max.is_finite() {
                println!(
                    "trace t={:.4}: |C| {c_max:.1} at {:?} phi {:.2}, {near} within 1 cell, det F {:.3}, J {:.3}, |F^| cols {:.3} {:.3}, |tau|/mu {:.3}, C {:?}",
                    scene.time,
                    scene.x[at],
                    phi.get_signed(cell.x, cell.y).unwrap_or(f32::NAN),
                    f.determinant(),
                    scene.j(at),
                    fh.x_axis.length(),
                    fh.y_axis.length(),
                    (tau.x_axis.length() + tau.y_axis.length()) / mu,
                    scene.c[at]
                );
            }
            if !c_max.is_finite() {
                break;
            }
        }
        return;
    }
    for f in 1..=180u32 {
        scene.frame();
        let top = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        if f % 3 == 0 || !top.is_finite() {
            let (com, _) = scene.centre_of_mass();
            let dx2 = scene.layout.dx * scene.layout.dx;
            let (mut l_orbit, mut l_spin, mut p_tot) = (0.0f64, 0.0f64, Vec2::ZERO);
            let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
            for q in 0..scene.x.len() {
                let r = scene.x[q] - com;
                l_orbit += f64::from(scene.mass[q] * (r.x * scene.v[q].y - r.y * scene.v[q].x));
                l_spin += f64::from(
                    scene.mass[q] * 0.25 * dx2 * (scene.c[q].x_axis.y - scene.c[q].y_axis.x),
                );
                p_tot += scene.mass[q] * scene.v[q];
                lo = lo.min(scene.x[q]);
                hi = hi.max(scene.x[q]);
            }
            let m: f32 = scene.mass.iter().sum();
            println!(
                "{case} f{f} t={:.3}: energy {:+.4}, top speed {:.2} of impact, L orbit {:.3e} spin {:.3e} (scale {:.3e}), momentum {:?} of impact, extent {:?}..{:?}",
                scene.time,
                energy(&scene) / e0 - 1.0,
                top / speed0,
                l_orbit,
                l_spin,
                f64::from(m * speed0 * 5.0),
                p_tot / (m * speed0),
                lo,
                hi
            );
        }
        if !top.is_finite() {
            break;
        }
    }
}

/// Where the drop into a pool loses a particle through a wall, after the
/// staggered elastic solid's transfer changes. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_pool_crossings() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, drop_into_pool_seed());
    let mut deepest = (0.0f32, 0usize, 0u32);
    for f in 0..(2.0 / FRAME).round() as u32 {
        scene.frame();
        for p in 0..scene.x.len() {
            let q = scene.x[p];
            let beyond = (tank.0 - q).max(q - tank.1).max_element();
            if beyond >= 0.0 && beyond >= deepest.0 {
                deepest = (beyond, p, f);
            }
        }
        let out = (0..scene.x.len())
            .filter(|&p| !scene.inside_tank(p))
            .count();
        if out > 0 {
            println!("frame {f}: {out} out");
        }
    }
    println!(
        "deepest beyond a wall: {:e} cell (particle {}, frame {})",
        deepest.0, deepest.1, deepest.2
    );
    for &(t, before, v, gap) in &scene.probe.crossings {
        println!(
            "crossing t={t:.4}s from ({:.7},{:.4}) v=({:e},{:.1}) gap {gap:e}",
            before.x, before.y, v.x, v.y
        );
    }
}

/// Where the dam break (and the real water one with
/// `EMERGE_PROBE_WATER`) loses particles after the ghost pressure: how far
/// beyond which wall, when, and how fast. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_dam_break_crossings() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, dam_break_seed());
    if std::env::var_os("EMERGE_PROBE_WATER").is_some() {
        scene = scene.with_sound_speed(1483.0);
    }
    let mut seen = std::collections::BTreeMap::new();
    for f in 0..(2.0 / FRAME).round() as u32 {
        scene.frame();
        for p in 0..scene.x.len() {
            let q = scene.x[p];
            let beyond = (tank.0 - q).max(q - tank.1).max_element();
            if beyond > 0.0 {
                let entry = seen.entry(p).or_insert((f, 0.0f32, q, scene.v[p], 0u32));
                entry.4 += 1;
                if beyond > entry.1 {
                    entry.1 = beyond;
                    entry.2 = q;
                    entry.3 = scene.v[p];
                }
            }
        }
    }
    println!("{} particles ever beyond a wall", seen.len());
    println!(
        "{} reflections, deepest {:.2} ulps",
        scene.probe.reflections, scene.probe.deepest_reflection_ulps
    );
    for (p, (first, depth, q, v, frames)) in &seen {
        println!(
            "particle {p}: first frame {first}, {frames} frames out, deepest {depth:e} cell at ({:.4},{:.4}), v ({:.1},{:.1})",
            q.x, q.y, v.x, v.y
        );
    }
}

/// Particle 478 of the real-water dam break, the first to pass through the left
/// wall with an earlier version of the staggered solid's transfer: its position
/// and the velocity it gathers every substep around the moment it enters the
/// wall. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_dam_break_particle_through_wall() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene =
        Scene::new((72, 72), 0.01, 1.0, tank, dam_break_seed()).with_sound_speed(1483.0);
    let p = 478usize;
    for f in 0..48u32 {
        if f < 40 {
            scene.frame();
            continue;
        }
        let before = scene.x[p];
        scene.frame();
        println!(
            "frame {f}: x ({:.6},{:.6}) -> ({:.6},{:.6}), v ({:.3},{:.3}), C {:?}",
            before.x, before.y, scene.x[p].x, scene.x[p].y, scene.v[p].x, scene.v[p].y, scene.c[p]
        );
    }
}

/// Where the projection's time goes on the dropped elastic block's explicit
/// control: each piece of `project` timed alone, 200 times, on the state at 0.5
/// s. A probe.
#[test]
#[ignore = "cost probe for real time: run with --ignored --nocapture"]
fn probe_projection_pieces() {
    use super::{extrapolate, pcg, pressure};
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    while scene.time < 0.5 {
        scene.frame();
    }
    let dt = 1.0e-3;
    let (vel, face_mass) =
        particles_to_faces(&scene.layout, &scene.x, &scene.v, &scene.c, &scene.mass);
    let phi = scene.phi();
    let image = box_container_image(scene.tank.0, scene.tank.1);
    let open = scene.open_centres();
    let surface = pressure::Surface {
        phi: &phi,
        theta_floor: scene.settings.theta_floor,
    };
    let reps = 200u32;
    let time = |name: &str, f: &mut dyn FnMut()| {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        println!(
            "PIECE {name:<22} {:8.1} us",
            t.elapsed().as_secs_f64() * 1e6 / f64::from(reps)
        );
    };
    let system = pressure::assemble(&scene.layout, dt, &vel, &scene.weights, surface);
    time("assemble", &mut || {
        std::hint::black_box(pressure::assemble(
            &scene.layout,
            dt,
            &vel,
            &scene.weights,
            surface,
        ));
    });
    let solution = pcg::solve(&system, &scene.settings.solver);
    println!(
        "unknowns {}, iterations {}",
        system.unknowns(),
        solution.iterations
    );
    time("pcg solve", &mut || {
        std::hint::black_box(pcg::solve(&system, &scene.settings.solver));
    });
    let mut mic = scene.settings.solver;
    mic.preconditioner = super::pcg::Preconditioner::Mic0;
    let with_mic = pcg::solve(&system, &mic);
    println!("MIC(0): iterations {}", with_mic.iterations);
    time("pcg solve, MIC(0)", &mut || {
        std::hint::black_box(pcg::solve(&system, &mic));
    });
    let mut v = vel.clone();
    time("clone velocity", &mut || {
        v = std::hint::black_box(vel.clone());
    });
    let mut valid = pressure::apply_pressure(
        &scene.layout,
        dt,
        &solution.pressure,
        &scene.weights,
        surface,
        &mut v,
    );
    time("apply pressure", &mut || {
        let mut w = vel.clone();
        std::hint::black_box(pressure::apply_pressure(
            &scene.layout,
            dt,
            &solution.pressure,
            &scene.weights,
            surface,
            &mut w,
        ));
    });
    let ghost = extrapolate::ghost_pressure(
        &scene.layout,
        dt,
        &phi,
        &solution.pressure,
        &open,
        (
            scene.settings.theta_floor,
            scene.settings.extrapolation_layers,
        ),
    );
    time("ghost pressure", &mut || {
        std::hint::black_box(extrapolate::ghost_pressure(
            &scene.layout,
            dt,
            &phi,
            &solution.pressure,
            &open,
            (
                scene.settings.theta_floor,
                scene.settings.extrapolation_layers,
            ),
        ));
    });
    time("keep reached faces", &mut || {
        let (mut w, mut f) = (v.clone(), valid.clone());
        extrapolate::keep_reached_faces(&mut w, &vel, &mut f, &face_mass, &scene.weights, &ghost);
    });
    extrapolate::keep_reached_faces(&mut v, &vel, &mut valid, &face_mass, &scene.weights, &ghost);
    time("extrapolate velocity", &mut || {
        let (mut w, mut f) = (v.clone(), valid.clone());
        extrapolate::extrapolate_velocity(&mut w, &mut f, scene.settings.extrapolation_layers);
    });
    time("constrain to solids", &mut || {
        let mut w = v.clone();
        extrapolate::constrain_to_solids(&scene.layout, &mut w, &scene.weights, &image);
    });
    time("level set", &mut || {
        std::hint::black_box(scene.phi());
    });
}
