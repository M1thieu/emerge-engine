use super::*;

/// The column at rest at `k` cells per centimetre and one gravity. Returns the
/// largest pressure error, in metres of head.
fn column_at_rest(k: usize, g_fraction: f32, report: &mut Report) -> f32 {
    let (wall, width, depth) = (4 * k, 40 * k, 30 * k);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18 * k);
    let floor = wall as f32;
    let h = floor + depth as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let water = (
        IVec2::new(wall as i32, wall as i32),
        IVec2::new((wall + width) as i32, (wall + depth) as i32),
    );
    let x = seed(
        water,
        |_| true,
        |q| box_free_sides(q, Some(h), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01 / k as f32, g_fraction, tank, x);
    let start = scene.x.clone();
    let centre_i = nx / 2;
    let surface_start = surface_height(&scene.phi(), centre_i, wall).expect("no surface");
    let g = scene.g();
    let label = format!("column {} cm, {g_fraction} g", 1.0 / k as f32);
    let mut cost = Cost::default();
    let (mut worst_p, mut worst_move, mut worst_surface) = (0.0f32, 0.0f32, 0.0f32);
    let wall_clock = Instant::now();
    for _ in 0..(5.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        for j in wall..ny {
            let c = centre_i + nx * j;
            let y = j as f32 + 0.5;
            if y >= h || !frame.active[c] || scene.solid_centres.get(centre_i, j) <= 0.0 {
                continue;
            }
            worst_p = worst_p.max((frame.pressure[c] - g * (h - y)).abs());
        }
        for (a, b) in scene.x.iter().zip(&start) {
            worst_move = worst_move.max((*a - *b).length());
        }
        if let Some(s) = surface_height(&frame.phi, centre_i, wall) {
            worst_surface = worst_surface.max((s - surface_start).abs());
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    scene.print_map(&label);
    let head_m = worst_p / g * scene.cell_m;
    println!(
        "{label}: pressure error {:.3} cell of head ({:.2} mm), largest move {:.3} cell, \
         surface drift {:.3} cell (start {:.3}, true {h}), J at bounds {}",
        worst_p / g,
        head_m * 1000.0,
        worst_move,
        worst_surface,
        surface_start,
        scene.j_at_bounds
    );
    scene.check_reflections(&label, report);
    cost.print(&label, wall_s, 5.0);
    report.check(worst_p <= 0.5 * g, || {
        format!("{label}: pressure off by {:.3} cell of head", worst_p / g)
    });
    report.check(worst_move <= 1.0, || {
        format!("{label}: a particle moved {worst_move:.3} cell")
    });
    report.check(worst_surface <= 0.5, || {
        format!("{label}: surface drifted {worst_surface:.3} cell")
    });
    report.check(scene.j_at_bounds == 0, || {
        format!("{label}: {} J at bounds", scene.j_at_bounds)
    });
    head_m
}

#[test]
#[ignore = "projection gate, column at rest: long; run with --ignored --nocapture"]
fn gate_column_at_rest() {
    let mut report = Report::new();
    let mut coarse = 0.0;
    for &g in &GRAVITIES {
        let head = column_at_rest(1, g, &mut report);
        if g == 1.0 {
            coarse = head;
        }
    }
    let fine = column_at_rest(2, 1.0, &mut report);
    println!(
        "pressure error at 1 g: {:.2} mm at 1 cm, {:.2} mm at 0.5 cm",
        coarse * 1000.0,
        fine * 1000.0
    );
    report.check(fine <= coarse, || {
        format!(
            "finer cells: error {:.2} mm, not below {:.2} mm",
            fine * 1000.0,
            coarse * 1000.0
        )
    });
    report.finish();
}

fn droplet_in_free_fall(g_fraction: f32, report: &mut Report) {
    let (nx, ny) = (32, 144);
    let (centre, radius) = (Vec2::new(16.0, 136.0), 5.0);
    let tank = (Vec2::ZERO, Vec2::new(nx as f32, ny as f32));
    let mut rng = LcgRng::new(11);
    let x = seed(
        (IVec2::new(10, 130), IVec2::new(22, 142)),
        |q| (q - centre).length() < radius,
        |q| disc_side(q, centre, radius),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, g_fraction, tank, x);
    let g = scene.g();
    let label = format!("droplet {g_fraction} g");
    let gyration = |s: &Scene| -> f32 {
        let (com, _) = s.centre_of_mass();
        let total: f32 = s.mass.iter().sum();
        let second: f32 = (0..s.x.len())
            .map(|p| s.mass[p] * (s.x[p] - com).length_squared())
            .sum();
        (second / total).sqrt()
    };
    let gyration_start = gyration(&scene);
    let mut cost = Cost::default();
    let mut worst = [0.0f32; 5];
    let mut v_before = scene.centre_of_mass().1;
    let wall_clock = Instant::now();
    for _ in 0..(0.3 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        for (c, &p) in frame.pressure.iter().enumerate() {
            if frame.active[c] {
                worst[0] = worst[0].max(p.abs() / g);
            }
        }
        let (_, v_com) = scene.centre_of_mass();
        let accel = (v_com - v_before) / FRAME;
        worst[1] = worst[1].max((accel - scene.gravity).length() / g);
        v_before = v_com;
        for p in 0..scene.x.len() {
            worst[2] = worst[2].max((scene.v[p] - v_com).length() / (g * scene.time));
            worst[4] = worst[4].max((scene.j(p) - 1.0).abs());
        }
        worst[3] = worst[3].max((gyration(&scene) / gyration_start - 1.0).abs());
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    scene.print_map(&label);
    let lowest = scene.x.iter().fold(f32::INFINITY, |m, q| m.min(q.y));
    println!(
        "{label}: |p| {:.2e} cell of head, acceleration error {:.2e} of g, relative speed \
         {:.2e} of g t, gyration {:.2e}, |J-1| {:.2e}, lowest particle y={lowest:.1}",
        worst[0], worst[1], worst[2], worst[3], worst[4]
    );
    scene.check_reflections(&label, report);
    cost.print(&label, wall_s, 0.3);
    report.check(lowest > 1.0, || {
        format!("{label}: the droplet reached the floor")
    });
    let limits = [0.01, 1e-3, 1e-3, 5e-3, 1e-3];
    let names = [
        "pressure, cells of head",
        "acceleration error, of g",
        "relative speed, of g t",
        "gyration change",
        "|J - 1|",
    ];
    for k in 0..5 {
        report.check(worst[k] <= limits[k], || {
            format!(
                "{label}: {} {:.3e} over {:.0e}",
                names[k], worst[k], limits[k]
            )
        });
    }
}

#[test]
#[ignore = "projection gate, droplet in free fall: run with --ignored --nocapture"]
fn gate_droplet_in_free_fall() {
    let mut report = Report::new();
    for &g in &GRAVITIES {
        droplet_in_free_fall(g, &mut report);
    }
    report.finish();
}

/// Scenes 3 and 4 share their tank, their duration and their criteria.
fn violent_scene(label: &str, x: Vec<Vec2>, report: &mut Report) -> f32 {
    violent_scene_at(label, x, None, report)
}

/// The dam break and the drop into a pool, and real water with
/// `sound_speed` in m/s. Returns the mean
/// substeps per frame.
fn violent_scene_at(
    label: &str,
    x: Vec<Vec2>,
    sound_speed: Option<f32>,
    report: &mut Report,
) -> f32 {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    if let Some(c) = sound_speed {
        scene = scene.with_sound_speed(c);
    }
    let (volume_start, energy_start) = (scene.volume(), scene.energy());
    let mut cost = Cost::default();
    let (mut worst_volume, mut worst_energy) = (0.0f32, f32::NEG_INFINITY);
    let (mut outside, mut nan) = (0usize, false);
    scene.print_map(label);
    let wall_clock = Instant::now();
    for f in 1..=(2.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        let out = (0..scene.x.len())
            .filter(|&p| !scene.inside_tank(p))
            .count();
        outside = outside.max(out);
        worst_volume = worst_volume.max((scene.volume() / volume_start - 1.0).abs());
        worst_energy = worst_energy.max(scene.energy() / energy_start - 1.0);
        if f % 12 == 0 {
            let front = scene
                .x
                .iter()
                .filter(|q| q.y < tank.0.y + 2.0)
                .fold(f32::NEG_INFINITY, |m, q| m.max(q.x));
            println!(
                "  {label} t={:.2}s front {:.1} cells, volume {:+.4}, energy {:+.4}, \
                 {:.2} particles per interior cell",
                scene.time,
                front - tank.0.x,
                scene.volume() / volume_start - 1.0,
                scene.energy() / energy_start - 1.0,
                scene.interior_particles_per_cell()
            );
        }
        if f % 30 == 0 {
            scene.print_map(label);
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    println!(
        "{label}: NaN {nan}, most outside {outside}, volume change {worst_volume:.4}, \
         energy rise {worst_energy:+.4}, J at bounds {}",
        scene.j_at_bounds
    );
    scene.check_reflections(label, report);
    cost.print(label, wall_s, 2.0);
    report.check(!nan, || format!("{label}: NaN"));
    report.check(outside == 0, || {
        format!("{label}: {outside} particles left the tank")
    });
    report.check(worst_volume <= 0.02, || {
        format!("{label}: volume off by {worst_volume:.4}")
    });
    report.check(worst_energy <= 0.01, || {
        format!("{label}: energy rose {worst_energy:+.4}")
    });
    report.check(scene.j_at_bounds == 0, || {
        format!("{label}: {} J at bounds", scene.j_at_bounds)
    });
    cost.substeps as f32 / cost.frames.max(1) as f32
}

#[test]
#[ignore = "projection gate, dam break: run with --ignored --nocapture"]
fn gate_dam_break() {
    let mut report = Report::new();
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    violent_scene("dam break", x, &mut report);
    report.finish();
}

#[test]
#[ignore = "projection gate, drop into a pool: run with --ignored --nocapture"]
fn gate_drop_into_pool() {
    let mut report = Report::new();
    let mut rng = LcgRng::new(5);
    let mut x = seed(
        (IVec2::new(4, 4), IVec2::new(68, 19)),
        |_| true,
        |q| box_free_sides(q, Some(19.0), None),
        &mut rng,
    );
    let (centre, radius) = (Vec2::new(36.0, 44.0), 5.0);
    x.extend(seed(
        (IVec2::new(30, 38), IVec2::new(42, 50)),
        |q| (q - centre).length() < radius,
        |q| disc_side(q, centre, radius),
        &mut rng,
    ));
    violent_scene("drop into pool", x, &mut report);
    report.finish();
}

/// Where the dam break loses particles through its walls and where J
/// reaches its bounds. A probe, kept for the record; no criterion.
#[test]
#[ignore = "diagnostic probe for the first gate failure: run with --ignored --nocapture"]
fn probe_dam_break_walls_and_bounds() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    for _ in 0..(2.0 / FRAME).round() as u32 {
        scene.frame();
    }
    let [liquid, air, wall] = scene.probe.j_at_bounds_by_place;
    println!("J at bounds: {liquid} in liquid cells, {air} in air cells, {wall} inside walls");
    let crossings = &scene.probe.crossings;
    println!("{} crossings", crossings.len());
    let first = crossings.first().map_or(0.0, |c| c.0);
    println!("first at t={first:.3}s");
    let mut by_side = [0usize; 4];
    for &(_, before, _, _) in crossings {
        let (lo, hi) = (before - tank.0, tank.1 - before);
        let d = [lo.x, hi.x, lo.y, hi.y];
        let k = (0..4).min_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap_or(0);
        by_side[k] += 1;
    }
    println!(
        "by wall: left {}, right {}, floor {}, lid {}",
        by_side[0], by_side[1], by_side[2], by_side[3]
    );
    let n = crossings.len().max(1) as f32;
    let mean_gap = crossings.iter().map(|c| c.3).sum::<f32>() / n;
    let widest_gap = crossings.iter().fold(0.0f32, |m, c| m.max(c.3));
    println!(
        "distance to the wall just before: mean {mean_gap:.3} cell, widest {widest_gap:.3} cell"
    );
    for &(t, before, v, gap) in crossings.iter().take(8) {
        println!(
            "  t={t:.3}s at ({:.2},{:.2}) v=({:.1},{:.1}) gap {gap:.3}",
            before.x, before.y, v.x, v.y
        );
    }
    let bins = [0.1f32, 0.5, 1.0, 1.5, 2.0];
    for w in bins.windows(2) {
        let count = crossings
            .iter()
            .filter(|c| c.3 >= w[0] && c.3 < w[1])
            .count();
        println!("  gap in [{}, {}): {count}", w[0], w[1]);
    }
    let close = crossings.iter().filter(|c| c.3 < 0.1).count();
    println!("  gap below 0.1: {close}");
    // Within two cells of a second wall as well: a tank corner.
    let corner = crossings
        .iter()
        .filter(|&&(_, before, _, _)| {
            let (lo, hi) = (before - tank.0, tank.1 - before);
            let mut d = [lo.x, hi.x, lo.y, hi.y];
            d.sort_by(f32::total_cmp);
            d[1] < 2.0
        })
        .count();
    println!(
        "  within two cells of a corner: {corner} of {}",
        crossings.len()
    );
}

/// Whether the volume change and the packing come from the particles that
/// touched a wall: J and bound hits split by that history. A probe.
#[test]
#[ignore = "diagnostic probe for the first gate failure: run with --ignored --nocapture"]
fn probe_dam_break_volume_away_from_walls() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    let n = scene.x.len();
    let mut touched = vec![false; n];
    let near_wall = |q: Vec2| {
        let (lo, hi) = (q - tank.0, tank.1 - q);
        lo.x.min(lo.y).min(hi.x).min(hi.y) < 1.0
    };
    for f in 1..=(2.0 / FRAME).round() as u32 {
        scene.frame();
        for (hit, &q) in touched.iter_mut().zip(&scene.x) {
            *hit |= near_wall(q);
        }
        if f % 24 == 0 {
            let split = |want: bool| {
                let ids: Vec<usize> = (0..n).filter(|&p| touched[p] == want).collect();
                let mean = ids.iter().map(|&p| scene.j(p)).sum::<f32>() / ids.len().max(1) as f32;
                let low = ids.iter().filter(|&&p| scene.j(p) < 0.9).count();
                let high = ids.iter().filter(|&&p| scene.j(p) > 1.1).count();
                (ids.len(), mean, low, high)
            };
            let (a, b) = (split(false), split(true));
            println!(
                "t={:.2}s never near a wall: {} particles, mean J {:.4}, {} below 0.9, {} above 1.1 |                  touched a wall: {}, mean J {:.4}, {} below 0.9, {} above 1.1 | {:.2} per interior cell",
                scene.time,
                a.0,
                a.1,
                a.2,
                a.3,
                b.0,
                b.1,
                b.2,
                b.3,
                scene.interior_particles_per_cell()
            );
        }
    }
}

/// Deep in the liquid, is the divergence J reads (`tr C` of the gather in
/// use) the divergence the grid keeps after the solve, or larger? Beside
/// it, the cells' divergence interpolated bilinearly, which is what a
/// gather quadratic along each component and linear across it would read.
/// A probe: it measured the first gate run's quadratic gather at 1000 to
/// 3000 times the grid's divergence, which is why J integrates the cells'
/// divergence.
#[test]
#[ignore = "diagnostic probe for the first gate failure: run with --ignored --nocapture"]
fn probe_dam_break_bulk_divergence() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    for f in 1..=60u32 {
        let frame = scene.frame();
        if f % 10 != 0 {
            continue;
        }
        let (nx, ny) = (72i32, 72i32);
        let liquid = |i: i32, j: i32| {
            i >= 0
                && j >= 0
                && i < nx
                && j < ny
                && frame.phi.get(i as usize, j as usize) < 0.0
                && scene.solid_centres.get(i as usize, j as usize) > 0.0
        };
        let cell_div = |i: i32, j: i32| {
            let (i, j) = (i as usize, j as usize);
            frame.vel.u.get(i + 1, j) - frame.vel.u.get(i, j) + frame.vel.v.get(i, j + 1)
                - frame.vel.v.get(i, j)
        };
        let (mut n, mut quad, mut conforming, mut kept) = (0usize, 0.0f32, 0.0f32, 0.0f32);
        for p in 0..scene.x.len() {
            let q = scene.x[p];
            let c = q.floor().as_ivec2();
            if !(-2..=2).all(|dj| (-2..=2).all(|di| liquid(c.x + di, c.y + dj))) {
                continue;
            }
            n += 1;
            quad += (scene.c[p].x_axis.x + scene.c[p].y_axis.y).abs();
            // Bilinear interpolation of the cell divergence at the particle.
            let t = q - Vec2::splat(0.5);
            let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
            let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
            let d = cell_div(i0, j0) * (1.0 - fx) * (1.0 - fy)
                + cell_div(i0 + 1, j0) * fx * (1.0 - fy)
                + cell_div(i0, j0 + 1) * (1.0 - fx) * fy
                + cell_div(i0 + 1, j0 + 1) * fx * fy;
            conforming += d.abs();
            let mut local = 0.0f32;
            for dj in -1..=1 {
                for di in -1..=1 {
                    local = local.max(cell_div(c.x + di, c.y + dj).abs());
                }
            }
            kept += local;
        }
        let m = n.max(1) as f32;
        println!(
            "t={:.2}s {n} bulk particles: |tr C| {:.3e} 1/s, bilinear cell divergence              {:.3e} 1/s, largest cell divergence nearby {:.3e} 1/s",
            scene.time,
            quad / m,
            conforming / m,
            kept / m
        );
    }
}

/// After the second gate run: the particles that never came near a wall
/// still lost volume. Split them by whether they ever sat within two cells
/// of an air cell, to see whether the loss is at the free surface, where
/// the gather reads velocities extrapolated into the air. A probe.
#[test]
#[ignore = "diagnostic probe for the second gate failure: run with --ignored --nocapture"]
fn probe_dam_break_volume_at_the_free_surface() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    let n = scene.x.len();
    let (mut near_wall, mut near_air) = (vec![false; n], vec![false; n]);
    for f in 1..=(2.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        for p in 0..n {
            let q = scene.x[p];
            let (lo, hi) = (q - tank.0, tank.1 - q);
            near_wall[p] |= lo.x.min(lo.y).min(hi.x).min(hi.y) < 1.0;
            let c = q.floor().as_ivec2();
            near_air[p] |= (-2..=2).any(|dj| {
                (-2..=2).any(|di| {
                    let (i, j) = (c.x + di, c.y + dj);
                    let inside = scene
                        .solid_centres
                        .get_signed(i, j)
                        .is_some_and(|s| s > 0.0);
                    inside && frame.phi.get_signed(i, j).is_some_and(|f| f >= 0.0)
                })
            });
        }
        if f % 30 == 0 {
            let group = |wall: bool, air: bool| {
                let ids: Vec<usize> = (0..n)
                    .filter(|&p| near_wall[p] == wall && near_air[p] == air)
                    .collect();
                let m = ids.len().max(1) as f32;
                let mean = ids.iter().map(|&p| scene.j(p)).sum::<f32>() / m;
                (ids.len(), mean)
            };
            let deep = group(false, false);
            let surface = group(false, true);
            println!(
                "t={:.2}s away from walls: never near air {} particles, mean J {:.4} | \
                 near air at some point {} particles, mean J {:.4}",
                scene.time, deep.0, deep.1, surface.0, surface.1
            );
        }
    }
}

/// The dam break and the drop into a pool, their particle positions every
/// fourth frame written as JSON to the path in `EMERGE_GATE_DUMP` (tenths
/// of a cell), and the time each phase of the substep took. A probe.
#[test]
#[ignore = "visual dump and phase timings: run with --ignored --nocapture"]
fn probe_dump_and_phase_costs() {
    let path = std::env::var("EMERGE_GATE_DUMP").ok();
    let mut json = String::from("{\"scenes\":[");
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    for (k, name) in ["dam break", "drop into pool"].into_iter().enumerate() {
        let mut rng = LcgRng::new(if k == 0 { 3 } else { 5 });
        let x = if k == 0 {
            seed(
                (IVec2::new(4, 4), IVec2::new(24, 44)),
                |_| true,
                |q| box_free_sides(q, Some(44.0), Some(24.0)),
                &mut rng,
            )
        } else {
            let mut x = seed(
                (IVec2::new(4, 4), IVec2::new(68, 19)),
                |_| true,
                |q| box_free_sides(q, Some(19.0), None),
                &mut rng,
            );
            let (centre, radius) = (Vec2::new(36.0, 44.0), 5.0);
            x.extend(seed(
                (IVec2::new(30, 38), IVec2::new(42, 50)),
                |q| (q - centre).length() < radius,
                |q| disc_side(q, centre, radius),
                &mut rng,
            ));
            x
        };
        let n = x.len();
        let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
        if k > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"name\":\"{name}\",\"grid\":72,\"frame_dt\":{FRAME},\"every\":4,\"boxes\":[],\"tags\":[{}],\"frames\":[",
            vec!["0"; n].join(",")
        ));
        let frames = (2.0 / FRAME).round() as u32;
        let wall = Instant::now();
        for f in 0..=frames {
            if f % 4 == 0 {
                if f > 0 {
                    json.push(',');
                }
                let xs: Vec<String> = scene
                    .x
                    .iter()
                    .flat_map(|q| [(q.x * 10.0).round() as i32, (q.y * 10.0).round() as i32])
                    .map(|v| v.to_string())
                    .collect();
                json.push('[');
                json.push_str(&xs.join(","));
                json.push(']');
            }
            if f < frames {
                scene.frame();
            }
        }
        json.push_str("]}");
        let total = wall.elapsed().as_micros().max(1) as f64;
        let names = [
            "P2G",
            "level set",
            "assembly (record)",
            "projection",
            "two gathers",
            "particle update",
        ];
        println!(
            "{name}: {} particles, {:.0} ms for 2 s simulated",
            n,
            total / 1e3
        );
        for (label, us) in names.iter().zip(scene.probe.phase_us) {
            println!(
                "  {label:<18} {:6.0} ms  {:5.1} %",
                us as f64 / 1e3,
                100.0 * us as f64 / total
            );
        }
    }
    json.push_str("]}");
    if let Some(path) = path {
        std::fs::write(path, json).expect("write dump");
    }
}

#[test]
#[ignore = "compressible projection gate, column at rest: run with --ignored --nocapture"]
fn gate_compressible_column() {
    let mut report = Report::new();
    let c_m_s = 10.0;
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let h0 = floor + depth as f32;
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_sound_speed(c_m_s);
    let c2 = scene.sound_speed2.unwrap_or(0.0);
    let g = scene.g();
    let centre_i = nx / 2;
    let surface_start = surface_height(&scene.phi(), centre_i, wall).expect("no surface");
    let expected_drop = g * (depth as f32).powi(2) / (2.0 * c2);
    let frames = (5.0 / FRAME).round() as u32;
    let last_second = frames - (1.0 / FRAME).round() as u32;
    let (mut worst_p, mut surface_sum, mut surface_n) = (0.0f32, 0.0f32, 0u32);
    let bands = depth / 5;
    let mut band_j = vec![(0.0f64, 0u64); bands];
    let mut band_h = 0.0f32;
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    for f in 1..=frames {
        let frame = scene.frame();
        cost.add(&frame);
        let Some(h) = surface_height(&frame.phi, centre_i, wall) else {
            report.check(false, || "column: no surface".to_string());
            break;
        };
        for j in wall..ny {
            let c = centre_i + nx * j;
            let y = j as f32 + 0.5;
            if scene.time < 1.0
                || y >= h - 0.5
                || !frame.active[c]
                || scene.solid_centres.get(centre_i, j) <= 0.0
            {
                continue;
            }
            worst_p = worst_p.max((frame.pressure[c] - g * (h - y)).abs());
        }
        if f > last_second {
            surface_sum += h;
            surface_n += 1;
            band_h += h;
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
    let surface = surface_sum / surface_n.max(1) as f32;
    let h_mean = band_h / surface_n.max(1) as f32;
    let drop = surface_start - surface;
    println!(
        "compressible column: pressure error {:.3} cell of head, surface drop {drop:.3} cell against {expected_drop:.3}",
        worst_p / g
    );
    report.check(worst_p <= 0.5 * g, || {
        format!("column: pressure off by {:.3} cell of head", worst_p / g)
    });
    report.check((drop - expected_drop).abs() <= 0.15, || {
        format!("column: surface dropped {drop:.3} cell, expected {expected_drop:.3}")
    });
    for (k, &(sum, n)) in band_j.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let mean = (sum / n as f64) as f32;
        let y = floor + 5.0 * k as f32 + 2.5;
        let expected = 1.0 - g * (h_mean - y) / c2;
        println!("  band {k}: mean J {mean:.5}, expected {expected:.5}");
        report.check((mean - expected).abs() <= 0.005, || {
            format!("column: band {k} mean J {mean:.5}, expected {expected:.5}")
        });
    }
    scene.check_reflections("compressible column", &mut report);
    cost.print("compressible column", wall_s, 5.0);
    report.finish();
}

#[test]
#[ignore = "compressible projection gate, sound speed: run with --ignored --nocapture"]
fn gate_sound_speed() {
    let mut report = Report::new();
    let c_m_s = 10.0;
    let (wall, length, height) = (4usize, 200usize, 8usize);
    let (nx, ny) = (length + 2 * wall, height + 2 * wall);
    let tank = (
        Vec2::splat(wall as f32),
        Vec2::new((wall + length) as f32, (wall + height) as f32),
    );
    let mut rng = LcgRng::new(11);
    let x = seed(
        (
            IVec2::splat(wall as i32),
            IVec2::new((wall + length) as i32, (wall + height) as i32),
        ),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x).with_sound_speed(c_m_s);
    let c = c_m_s / scene.cell_m;
    scene.fixed_dt = Some(scene.layout.dx / c);
    for p in 0..scene.x.len() {
        if scene.x[p].x < wall as f32 + 10.0 {
            scene.log_j[p] = 0.99f32.ln();
        }
    }
    let watch_i = wall + 150;
    scene.probe.watch = Some(watch_i + nx * (wall + height / 2));
    let expected = (150.0 - 5.0) / c;
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    while scene.time < 1.5 * expected {
        let frame = scene.frame();
        cost.add(&frame);
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let (arrival, peak) = scene
        .probe
        .watch_history
        .iter()
        .filter(|(t, _)| *t < 1.5 * expected)
        .fold((0.0f32, f32::NEG_INFINITY), |m, &(t, q)| {
            if q > m.1 { (t, q) } else { m }
        });
    let error = (arrival - expected).abs() / expected;
    println!(
        "sound speed: peak {peak:.1} at t={arrival:.4}s, expected {expected:.4}s, off by {:.1} %",
        100.0 * error
    );
    report.check(error <= 0.05, || {
        format!("sound speed: arrival off by {:.1} %", 100.0 * error)
    });
    scene.check_reflections("sound speed", &mut report);
    cost.print("sound speed", wall_s, scene.time);
    report.finish();
}

#[test]
#[ignore = "compressible projection gate, real water: run with --ignored --nocapture"]
fn gate_real_water() {
    let mut report = Report::new();
    let water = 1483.0;
    for (name, seed_fn) in [
        ("dam break", dam_break_seed as fn() -> Vec<Vec2>),
        ("drop into pool", drop_into_pool_seed as fn() -> Vec<Vec2>),
    ] {
        let mut quiet = Report::new();
        let incompressible = violent_scene_at(name, seed_fn(), None, &mut quiet);
        let label = format!("{name}, real water");
        let compressible = violent_scene_at(&label, seed_fn(), Some(water), &mut report);
        println!(
            "{label}: {compressible:.2} substeps/frame against {incompressible:.2} incompressible"
        );
        report.check(compressible <= 1.1 * incompressible, || {
            format!("{label}: {compressible:.2} substeps/frame, over 1.1 x {incompressible:.2}")
        });
    }
    report.finish();
}

/// Diagnosis of the compressible projection: the sound-speed arrival at
/// substeps of `dx / c`, half and a quarter of it (a time-discretisation error
/// shrinks with the step; a wrong `c` does not), and the compressible column's
/// largest pressure error per half second. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_sound_speed_arrival() {
    let c_m_s = 10.0;
    for fraction in [1.0f32, 0.5, 0.25] {
        let (wall, length, height) = (4usize, 200usize, 8usize);
        let (nx, ny) = (length + 2 * wall, height + 2 * wall);
        let tank = (
            Vec2::splat(wall as f32),
            Vec2::new((wall + length) as f32, (wall + height) as f32),
        );
        let mut rng = LcgRng::new(11);
        let x = seed(
            (
                IVec2::splat(wall as i32),
                IVec2::new((wall + length) as i32, (wall + height) as i32),
            ),
            |_| true,
            |_| (f32::INFINITY, Vec2::Y),
            &mut rng,
        );
        let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x).with_sound_speed(c_m_s);
        let c = c_m_s / scene.cell_m;
        scene.fixed_dt = Some(fraction * scene.layout.dx / c);
        for p in 0..scene.x.len() {
            if scene.x[p].x < wall as f32 + 10.0 {
                scene.log_j[p] = 0.99f32.ln();
            }
        }
        scene.probe.watch = Some(wall + 150 + nx * (wall + height / 2));
        let expected = (150.0 - 5.0) / c;
        while scene.time < 1.5 * expected {
            scene.frame();
        }
        let (arrival, peak) = scene
            .probe
            .watch_history
            .iter()
            .filter(|(t, _)| *t < 1.5 * expected)
            .fold((0.0f32, f32::NEG_INFINITY), |m, &(t, q)| {
                if q > m.1 { (t, q) } else { m }
            });
        println!(
            "dt = {fraction} dx/c: peak {peak:.1} at {arrival:.4} s, expected {expected:.4} s ({:+.1} %)",
            100.0 * (arrival - expected) / expected
        );
    }
    // Column: largest pressure error per half second.
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let h0 = floor + depth as f32;
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_sound_speed(c_m_s);
    let g = scene.g();
    let centre_i = nx / 2;
    let mut worst = 0.0f32;
    let mut worst_y = 0.0f32;
    for f in 1..=(5.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        let h = surface_height(&frame.phi, centre_i, wall).unwrap_or(0.0);
        for j in wall..ny {
            let c = centre_i + nx * j;
            let y = j as f32 + 0.5;
            if y >= h - 0.5 || !frame.active[c] || scene.solid_centres.get(centre_i, j) <= 0.0 {
                continue;
            }
            let e = (frame.pressure[c] - g * (h - y)).abs();
            if e > worst {
                worst = e;
                worst_y = y;
            }
        }
        if f % 30 == 0 {
            let outside = (0..scene.x.len())
                .filter(|&p| !scene.inside_tank(p))
                .count();
            let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
            let geometric = scene.interior_particles_per_cell();
            let mean_j = (0..scene.x.len()).map(|p| scene.j(p)).sum::<f32>() / scene.x.len() as f32;
            println!(
                "column t={:.1}s: largest pressure error {:.3} cell of head (at y={worst_y:.1}), surface {h:.3}, outside {outside}, lowest y {lowest:.3}, {geometric:.3} per interior cell, mean J {mean_j:.5}",
                scene.time,
                worst / g
            );
            worst = 0.0;
        }
    }
}
