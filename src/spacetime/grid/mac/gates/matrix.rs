use super::*;

/// One row of the material matrix: measured coefficients only.
struct MaterialRow {
    name: &'static str,
    /// kg/m^3.
    rho: f32,
    /// Plane-strain Lame coefficients, Pa.
    lambda: f32,
    mu: f32,
}

impl MaterialRow {
    fn from_young(name: &'static str, rho: f32, e: f32, nu: f32) -> Self {
        Self {
            name,
            rho,
            lambda: e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
            mu: e / (2.0 * (1.0 + nu)),
        }
    }

    fn c_p(&self) -> f32 {
        ((self.lambda + 2.0 * self.mu) / self.rho).sqrt()
    }

    fn c_s(&self) -> f32 {
        (self.mu / self.rho).sqrt()
    }
}

/// The rows (criteria: the material matrix).
fn material_rows() -> Vec<MaterialRow> {
    let clay = |name, q_u: f32| MaterialRow::from_young(name, 1700.0, 300.0 * q_u / 2.0, 0.45);
    vec![
        MaterialRow {
            name: "water",
            rho: 1000.0,
            lambda: 2.2e9,
            mu: 0.0,
        },
        clay("very soft clay", 20.0e3),
        clay("medium clay", 75.0e3),
    ]
}

/// A plane wave through a free block of the row's material: longitudinal
/// (P) or transverse (S). Returns the arrival's relative error and the wall
/// time per simulated second, in ms.
fn plane_wave(row: &MaterialRow, longitudinal: bool, report: &mut Report) -> (f32, f32) {
    let kind = if longitudinal { "P" } else { "S" };
    let label = format!("{}, {kind} wave", row.name);
    let speed = if longitudinal { row.c_p() } else { row.c_s() };
    // Edges' disturbances travel at most `c_p`; the window ends when the
    // kick's back has passed the probe, 50 cells, and they must arrive
    // from half the height later.
    let height = (2.0 * (50.0 * row.c_p() / speed + 4.0)).ceil() as usize + 4;
    let length = 100usize;
    let (nx, ny) = (length + 8, height + 8);
    let (x0, y0) = (4.0f32, 4.0f32);
    let tank = (
        Vec2::splat(1.0),
        Vec2::new(nx as f32 - 1.0, ny as f32 - 1.0),
    );
    let mut rng = LcgRng::new(13);
    let (top, right) = (y0 + height as f32, x0 + length as f32);
    let x = seed_body(
        row.mu > 0.0,
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
    let mut scene =
        Scene::new((nx, ny), 0.01, 0.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let c = speed / scene.cell_m;
    scene.fixed_dt = Some(0.5 * scene.layout.dx / (row.c_p().max(row.c_s()) / scene.cell_m));
    let kick = 0.05 / scene.cell_m;
    for p in 0..scene.x.len() {
        if scene.x[p].x < x0 + 10.0 {
            if longitudinal {
                scene.v[p].x = kick;
            } else {
                scene.v[p].y = kick;
            }
        }
    }
    let mid = y0 + 0.5 * height as f32;
    scene.probe.watch_x = Some((x0 + 50.0, mid - 4.0, mid + 4.0));
    scene.probe.watch_longitudinal = longitudinal;
    let expected = 40.0 / c;
    let window = 50.0 / c;
    let mut nan = false;
    let wall_clock = Instant::now();
    scene.run_until(window);
    nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let history = &scene.probe.watch_x_history;
    let half = 0.25 * kick;
    let arrival = history.windows(2).find_map(|pair| {
        let ((t0, v0), (t1, v1)) = (pair[0], pair[1]);
        (v0 < half && v1 >= half).then(|| t0 + (t1 - t0) * (half - v0) / (v1 - v0))
    });
    let error = arrival.map_or(f32::INFINITY, |t| (t - expected) / expected);
    let ms = 1000.0 * wall_s / scene.time.max(f32::MIN_POSITIVE);
    println!(
        "{label}: {} m/s expected, arrival {:+.1} % ({} particles, {height} cells high), {ms:.0} ms per simulated s",
        speed,
        100.0 * error,
        scene.x.len()
    );
    report.check(!nan, || format!("{label}: NaN"));
    report.check(error.abs() <= 0.05, || {
        format!("{label}: arrival off by {:+.1} %", 100.0 * error)
    });
    scene.check_reflections(&label, report);
    (error, ms)
}

#[test]
#[ignore = "material matrix, P waves: run with --ignored --nocapture"]
fn gate_matrix_p_waves() {
    let mut report = Report::new();
    for row in material_rows() {
        plane_wave(&row, true, &mut report);
    }
    report.finish();
}

#[test]
#[ignore = "material matrix, S waves: run with --ignored --nocapture"]
fn gate_matrix_s_waves() {
    let mut report = Report::new();
    for row in material_rows().iter().filter(|r| r.mu > 0.0) {
        plane_wave(row, false, &mut report);
    }
    report.finish();
}

/// The rest scene for one row (criteria: the material matrix). Returns the wall
/// time per simulated second, in ms.
fn rest_column(row: &MaterialRow, report: &mut Report) -> f32 {
    let label = format!("{}, at rest", row.name);
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
        row.mu > 0.0,
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene =
        Scene::new((nx, ny), 0.01, 1.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let g = scene.g();
    let m_over_rho = (row.lambda + 2.0 * row.mu) / row.rho / (scene.cell_m * scene.cell_m);
    let seconds = 2.0f32;
    let frames = (seconds / FRAME).round() as u32;
    let last_second = frames - (1.0 / FRAME).round() as u32;
    let bands = depth / 5;
    let mut band_j = vec![(0.0f64, 0u64); bands];
    let mut nan = false;
    let wall_clock = Instant::now();
    for f in 1..=frames {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        if f > last_second {
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
    let ms = 1000.0 * wall_clock.elapsed().as_secs_f32() / seconds;
    let drift = scene.settings.solver.absolute_tolerance * seconds;
    let full_strain = g * depth as f32 / m_over_rho;
    let mut worst = 0.0f32;
    for (k, &(sum, n)) in band_j.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let mean = (sum / n as f64) as f32;
        let y = floor + 5.0 * k as f32 + 2.5;
        let expected = 1.0 - g * (h0 - y) / m_over_rho;
        let bound = 0.05 * full_strain + drift;
        worst = worst.max((mean - expected).abs() / bound);
        report.check((mean - expected).abs() <= bound, || {
            format!("{label}: band {k} mean J {mean:.6}, expected {expected:.6}, bound {bound:.1e}")
        });
    }
    println!(
        "{label}: strain at the floor {full_strain:.2e}, worst band {worst:.2} of its bound, {ms:.0} ms per simulated s ({:.2}x real time)",
        1000.0 / ms
    );
    report.check(!nan, || format!("{label}: NaN"));
    scene.check_reflections(&label, report);
    ms
}

/// The drop for one row (criteria: the material matrix). Returns the wall time
/// per simulated second, in ms.
fn drop_block(row: &MaterialRow, report: &mut Report) -> f32 {
    let label = format!("{}, dropped", row.name);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed_body(
        row.mu > 0.0,
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
    let mut scene =
        Scene::new((72, 72), 0.01, 1.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
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
    let (energy_start, gyration_start, volume_start) =
        (row_energy(&scene), gyration(&scene), scene.volume());
    let seconds = 1.0f32;
    let (mut worst_energy, mut nan) = (f64::NEG_INFINITY, false);
    let wall_clock = Instant::now();
    for _ in 1..=(seconds / FRAME).round() as u32 {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        worst_energy = worst_energy.max(row_energy(&scene) / energy_start - 1.0);
    }
    let ms = 1000.0 * wall_clock.elapsed().as_secs_f32() / seconds;
    let change = (gyration(&scene) / gyration_start - 1.0).abs();
    let volume_change = (scene.volume() / volume_start - 1.0).abs();
    // Reported, not judged yet: what the run dissipated, a realism defect
    // the criteria do not bound (the very soft clay lost 42 percent in 1 s
    // seeded as a solid, `seed_body`, 57 with the surface pull,
    // bouncing).
    let energy_end = row_energy(&scene) / energy_start - 1.0;
    println!(
        "{label}: highest energy {worst_energy:+.4} of the start, at the end {energy_end:+.4}, gyration change {change:.4}, volume change {volume_change:.4}, {ms:.0} ms per simulated s ({:.2}x real time)",
        1000.0 / ms
    );
    report.check(!nan, || format!("{label}: NaN"));
    report.check(worst_energy <= 0.01, || {
        format!("{label}: energy rose {worst_energy:+.4}")
    });
    if row.mu > 0.0 {
        report.check(change <= 0.05, || {
            format!("{label}: gyration changed {change:.4}")
        });
    } else {
        report.check(volume_change <= 0.02, || {
            format!("{label}: volume changed {volume_change:.4}")
        });
    }
    scene.check_reflections(&label, report);
    ms
}

/// The conserving step's bounce for one solid row (criteria and their
/// amendments, module doc): the matrix's drop, 2 s, one frame per substep, the
/// tank's walls held by contact (`Scene::conserving_substep`). Lift-off is reported, not judged. Returns the
/// wall time per simulated second, in ms.
fn conserving_bounce(row: &MaterialRow, report: &mut Report) -> f32 {
    let label = format!("{}, bounce", row.name);
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
    // The block's surface, a contact point every half cell around it.
    let mut outline = Vec::new();
    let steps = (2.0 * (hi.x - lo.x)) as usize;
    for k in 0..steps {
        let t = k as f32 / steps as f32;
        outline.push(lo + t * Vec2::new(hi.x - lo.x, 0.0));
        outline.push(Vec2::new(hi.x, lo.y) + t * Vec2::new(0.0, hi.y - lo.y));
        outline.push(hi - t * Vec2::new(hi.x - lo.x, 0.0));
        outline.push(Vec2::new(lo.x, hi.y) - t * Vec2::new(0.0, hi.y - lo.y));
    }
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_lame(row.lambda / row.rho, row.mu / row.rho)
        .with_contact_outline(outline);
    scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
        ..Default::default()
    });
    scene.fixed_dt = Some(FRAME);
    // Over the particles with mass: the contact points carry none.
    let gyration = |scene: &Scene| {
        let (com, _) = scene.centre_of_mass();
        let total: f32 = scene.mass.iter().sum();
        (scene
            .x
            .iter()
            .zip(&scene.mass)
            .map(|(q, m)| m * (*q - com).length_squared())
            .sum::<f32>()
            / total)
            .sqrt()
    };
    let floor = tank.0.y;
    let contact_time = 2.0 * (hi.y - lo.y) * scene.cell_m / row.c_p();
    let (energy_start, gyration_start) = (row_energy(&scene), gyration(&scene));
    let seconds = 2.0f32;
    let (mut worst_energy, mut nan) = (0.0f64, false);
    let (mut touched, mut lifted): (Option<f32>, Option<(f32, f32)>) = (None, None);
    let wall_clock = Instant::now();
    for _ in 1..=(seconds / FRAME).round() as u32 {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        let change = row_energy(&scene) / energy_start - 1.0;
        nan |= !change.is_finite();
        worst_energy = worst_energy.max(change.abs());
        let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
        if touched.is_none() && lowest <= floor + 1.0 {
            touched = Some(scene.time);
        }
        if let (Some(t0), None) = (touched, lifted)
            && scene.time >= t0 + 2.0 * contact_time
        {
            let (_, v) = scene.centre_of_mass();
            lifted = Some((lowest - floor, v.y * scene.cell_m));
        }
    }
    let ms = 1000.0 * wall_clock.elapsed().as_secs_f32() / seconds;
    let change = (gyration(&scene) / gyration_start - 1.0).abs();
    let s = scene.probe.conserving;
    println!(
        "{label}: worst energy change {worst_energy:.6} of the start, gyration change {change:.4}, lift-off {lifted:?} (cells above the floor, m/s; contact time {:.2} ms), Newton {} GMRES {} unconverged {} rejected {}, {ms:.0} ms per simulated s ({:.4}x real time)",
        1000.0 * contact_time,
        s.0,
        s.1,
        s.4,
        scene.probe.rejected,
        1000.0 / ms
    );
    report.check(!nan, || format!("{label}: NaN"));
    report.check(worst_energy <= 0.01, || {
        format!("{label}: energy changed {worst_energy:.6}")
    });
    report.check(change <= 0.05, || {
        format!("{label}: gyration changed {change:.4}")
    });
    scene.check_reflections(&label, report);
    ms
}

#[test]
#[ignore = "conserving step, bounce: run with --ignored --nocapture"]
fn gate_conserving_bounce() {
    let mut report = Report::new();
    for row in material_rows().iter().filter(|r| r.mu > 0.0) {
        conserving_bounce(row, &mut report);
    }
    report.finish();
}

#[test]
#[ignore = "material matrix, rest: run with --ignored --nocapture"]
fn gate_matrix_rest() {
    let mut report = Report::new();
    for row in material_rows() {
        rest_column(&row, &mut report);
    }
    report.finish();
}

#[test]
#[ignore = "material matrix, drop: run with --ignored --nocapture"]
fn gate_matrix_drop() {
    let mut report = Report::new();
    for row in material_rows() {
        drop_block(&row, &mut report);
    }
    report.finish();
}

/// Check of the matrix's drop: the very soft clay's drop kept its energy
/// to 1e-4. Does it reach the floor and bounce? Lowest particle, energy
/// and centre speed every tenth of a second. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_soft_clay_drop_trajectory() {
    let row = &material_rows()[std::env::var("EMERGE_PROBE_ROW")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(1)];
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed_body(
        true,
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene =
        Scene::new((72, 72), 0.01, 1.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let e0 = row_energy(&scene);
    let frames = std::env::var("EMERGE_PROBE_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(60u32);
    for f in 1..=frames {
        scene.frame();
        let every = if scene.conserving.is_some() { 1 } else { 6 };
        if f % every == 0 {
            let (_, v) = scene.centre_of_mass();
            let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
            let s = scene.probe.conserving;
            println!(
                "{} t={:.3}: lowest y {lowest:.3}, centre vy {:.2} m/s, energy {:+.5}, newton {} gmres {} worst residual {:.1e} unconverged {} reflections {} rejected {}, stabilisation took {:.5}",
                row.name,
                scene.time,
                v.y * scene.cell_m,
                row_energy(&scene) / e0 - 1.0,
                s.0,
                s.1,
                s.3,
                s.4,
                scene.probe.reflections,
                scene.probe.rejected,
                scene.probe.stabilisation_loss / e0
            );
        }
    }
}

/// The conserving step's bounce with the floor a body: the drop scene's clay
/// block falls on a slab of an elastic solid `EMERGE_PROBE_FLOOR` times stiffer
/// than the medium clay (default 1000, same density), 8 cells thick on the
/// tank's floor, all of it moved by the conserving step at one frame per
/// substep (`EMERGE_PROBE_CONSERVING=frame`, walls `held`). Energy every frame,
/// kinetic, weight and stored, against the block's drop energy onto the slab. A
/// probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_conserving_drop_on_slab() {
    use crate::spacetime::total_lagrangian::Elastic;
    let stiffer: f32 = std::env::var("EMERGE_PROBE_FLOOR")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(1000.0);
    let frames: u32 = std::env::var("EMERGE_PROBE_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(60);
    let rows = material_rows();
    let floor = &rows[2];
    for row in &rows[1..] {
        let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
        let mut rng = LcgRng::new(17);
        let top = 12.0f32;
        let slab = seed_body(
            true,
            (IVec2::new(4, 4), IVec2::new(68, top as i32)),
            |_| true,
            |q| (top - q.y, Vec2::Y),
            &mut rng,
        );
        let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
        let block = seed_body(
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
        let (n_slab, n_block) = (slab.len(), block.len());
        let x: Vec<Vec2> = slab.into_iter().chain(block).collect();
        let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
            .with_lame(row.lambda / row.rho, row.mu / row.rho);
        let cells2 = 1.0 / f64::from(scene.cell_m).powi(2);
        let elastic = |r: &MaterialRow, k: f32| {
            Elastic::isochoric(
                f64::from(k * r.mu / r.rho) * cells2,
                f64::from(k * (r.lambda + r.mu) / r.rho) * cells2,
            )
        };
        let solids: Vec<Elastic> = (0..n_slab)
            .map(|_| elastic(floor, stiffer))
            .chain((0..n_block).map(|_| elastic(row, 1.0)))
            .collect();
        scene.solids = Some(solids.clone());
        let energy = |scene: &Scene| -> (f64, f64) {
            let g = f64::from(scene.g());
            let mut parts = (0.0, 0.0);
            for (p, solid) in solids.iter().enumerate() {
                let m = f64::from(scene.mass[p]);
                let e = m
                    * (0.5 * f64::from(scene.v[p].length_squared()) + g * f64::from(scene.x[p].y))
                    + f64::from(scene.v0[p]) * solid.energy(scene.deformation[p].as_dmat2());
                if p < n_slab {
                    parts.0 += e;
                } else {
                    parts.1 += e;
                }
            }
            parts
        };
        let start = energy(&scene);
        let block_mass: f64 = scene.mass[n_slab..].iter().map(|&m| f64::from(m)).sum();
        let drop = block_mass * f64::from(scene.g()) * f64::from(lo.y - top);
        let clock = std::time::Instant::now();
        let mut worst = 0.0f64;
        for _ in 0..frames {
            scene.frame();
            let now = energy(&scene);
            let change = (now.0 + now.1 - start.0 - start.1) / drop;
            worst = worst.max(change.abs());
            let lowest = scene.x[n_slab..]
                .iter()
                .map(|q| q.y)
                .fold(f32::INFINITY, f32::min);
            let vy = scene.v[n_slab..].iter().map(|v| v.y).sum::<f32>() / n_block as f32;
            let s = scene.probe.conserving;
            println!(
                "{} on slab x{stiffer} t={:.3}: block lowest {lowest:.3}, block vy {:.3} m/s, energy change {change:+.5} of the drop (slab {:+.5}, block {:+.5}), newton {} gmres {} unconverged {} worst residual {:.1e} rejected {}",
                row.name,
                scene.time,
                vy * scene.cell_m,
                (now.0 - start.0) / drop,
                (now.1 - start.1) / drop,
                s.0,
                s.1,
                s.4,
                s.3,
                scene.probe.rejected
            );
        }
        println!(
            "{} on slab x{stiffer}: worst energy change {worst:.5} of the drop, {:.1} s wall for {:.2} s simulated",
            row.name,
            clock.elapsed().as_secs_f64(),
            scene.time
        );
    }
}

/// The free body for one row at `k` cells per centimetre.
fn free_body(row: &MaterialRow, k: usize, report: &mut Report) {
    free_body_at(row, k, None, report);
}

/// The free body, moved by the conserving step at `conserving`'s substep
/// when it names one, and then judged by the conserving step's bounds: the
/// energy within 1 percent either way.
fn free_body_at(row: &MaterialRow, k: usize, conserving: Option<Substep>, report: &mut Report) {
    let kf = k as f32;
    let label = match conserving {
        None => format!("{}, free at {} cm", row.name, 1.0 / kf),
        Some(step) => format!(
            "{}, conserving free at {} cm, substep {step}",
            row.name,
            1.0 / kf
        ),
    };
    let tank = (Vec2::splat(4.0 * kf), Vec2::splat(68.0 * kf));
    let mut rng = LcgRng::new(17);
    // Without shear stiffness the block translates (criteria amendment).
    let translate = row.mu == 0.0;
    let corner = if translate { 8.0 } else { 31.0 };
    let (lo, hi) = (Vec2::splat(corner) * kf, Vec2::splat(corner + 10.0) * kf);
    let x = seed_body(
        row.mu > 0.0,
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
    let mut scene = Scene::new((72 * k, 72 * k), 0.01 / kf, 0.0, tank, x)
        .with_lame(row.lambda / row.rho, row.mu / row.rho);
    if let Some(step) = conserving {
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings::default());
        let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
        scene.fixed_dt = Some(step.length(scene.layout.dx, c_p));
    }
    let centre = 0.5 * (lo + hi);
    // Largest speed 0.5 m/s at the block's corners, `s |r|_max`.
    let s = 0.5 / scene.cell_m / (5.0 * kf * std::f32::consts::SQRT_2);
    for p in 0..scene.x.len() {
        let r = scene.x[p] - centre;
        if translate {
            scene.v[p] = 0.5 / scene.cell_m * Vec2::ONE.normalize();
        } else {
            scene.v[p] = s * Vec2::new(r.y, r.x);
            scene.c[p] = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
        }
    }
    let largest = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
    let (com0, vcom0) = scene.centre_of_mass();
    let inertia: f32 = (0..scene.x.len())
        .map(|p| scene.mass[p] * (scene.x[p] - com0).length_squared())
        .sum();
    let angular = |scene: &Scene| -> f32 {
        let (com, vcom) = scene.centre_of_mass();
        (0..scene.x.len())
            .map(|p| {
                let (r, v) = (scene.x[p] - com, scene.v[p] - vcom);
                scene.mass[p] * (r.x * v.y - r.y * v.x)
            })
            .sum()
    };
    let (l0, e0) = (angular(&scene), row_energy(&scene));
    let (mut worst_v, mut worst_l, mut worst_e, mut nan) =
        (0.0f32, 0.0f32, f64::NEG_INFINITY, false);
    let mut lowest_e = f64::INFINITY;
    let clock = Instant::now();
    for _ in 0..(1.0 / FRAME).round() as u32 {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        let (_, vcom) = scene.centre_of_mass();
        worst_v = worst_v.max((vcom - vcom0).length() / largest);
        worst_l = worst_l.max((angular(&scene) - l0).abs() / (inertia * s));
        let change = row_energy(&scene) / e0 - 1.0;
        worst_e = worst_e.max(change);
        lowest_e = lowest_e.min(change);
    }
    let st = scene.probe.conserving;
    println!(
        "{label}: centre speed up to {worst_v:.4} of the largest start speed, angular momentum off by {worst_l:.4} of I s, energy from {lowest_e:+.4} to {worst_e:+.4}{}",
        if conserving.is_some() {
            format!(
                ", Newton {} unconverged {}, {:.1} s wall",
                st.0,
                st.4,
                clock.elapsed().as_secs_f32()
            )
        } else {
            String::new()
        }
    );
    if conserving.is_some() {
        report.check(lowest_e >= -0.01, || {
            format!("{label}: energy fell {lowest_e:+.4}")
        });
    }
    report.check(!nan, || format!("{label}: NaN"));
    report.check(worst_v <= 0.01, || {
        format!("{label}: centre reached {worst_v:.4} of the start speed")
    });
    report.check(worst_l <= 0.01, || {
        format!("{label}: angular momentum off by {worst_l:.4}")
    });
    report.check(worst_e <= 0.01, || {
        format!("{label}: energy rose {worst_e:+.4}")
    });
}

/// The conserving step's free vibration (criteria in the module doc):
/// every solid row, the free body's block, 1 and 0.5 cm, at the explicit
/// P wave limit, four times it and one frame; the energy within 1 percent
/// either way at every frame, the free body's momentum bounds, no NaN.
#[test]
#[ignore = "conserving step, free vibration: run with --ignored --nocapture"]
fn gate_conserving_free_vibration() {
    let mut report = Report::new();
    for row in material_rows().iter().filter(|r| r.mu > 0.0) {
        for k in [1usize, 2] {
            for step in [Substep::PWave, Substep::FourPWaves, Substep::Frame] {
                free_body_at(row, k, Some(step), &mut report);
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "material matrix, free body: run with --ignored --nocapture"]
fn gate_matrix_free_body() {
    let mut report = Report::new();
    for row in material_rows() {
        for k in [1usize, 2] {
            free_body(&row, k, &mut report);
        }
    }
    report.finish();
}

/// The free body frame by frame for each row: substeps, largest
/// speed, the centre's speed and the energy against the start, stopping
/// after two minutes of wall time. The resolution is `EMERGE_PROBE_K`
/// cells per centimetre. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_free_body_frames() {
    let k: usize = std::env::var("EMERGE_PROBE_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let kf = k as f32;
    let only = std::env::var("EMERGE_PROBE_ROW").ok();
    for row in material_rows()
        .into_iter()
        .filter(|r| only.as_deref().is_none_or(|o| r.name == o))
    {
        let tank = (Vec2::splat(4.0 * kf), Vec2::splat(68.0 * kf));
        let mut rng = LcgRng::new(17);
        let (lo, hi) = (Vec2::new(31.0, 31.0) * kf, Vec2::new(41.0, 41.0) * kf);
        let x = seed(
            (lo.as_ivec2(), hi.as_ivec2()),
            |_| true,
            |_| (f32::INFINITY, Vec2::Y),
            &mut rng,
        );
        let mut scene = Scene::new((72 * k, 72 * k), 0.01 / kf, 0.0, tank, x)
            .with_lame(row.lambda / row.rho, row.mu / row.rho);
        let centre = 0.5 * (lo + hi);
        let s = 0.5 / scene.cell_m / (5.0 * kf * std::f32::consts::SQRT_2);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            scene.v[p] = s * Vec2::new(r.y, r.x);
            scene.c[p] = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
        }
        let e0 = row_energy(&scene);
        let clock = Instant::now();
        for f in 1..=60u32 {
            let frame = scene.frame();
            let top = scene.v.iter().map(|v| v.length()).fold(0.0f32, |m, s| {
                if s.is_nan() || m.is_nan() {
                    f32::NAN
                } else {
                    m.max(s)
                }
            });
            let (_, vcom) = scene.centre_of_mass();
            let late = clock.elapsed().as_secs() > 120;
            if f <= 3 || f % 6 == 0 || !top.is_finite() || late {
                println!(
                    "{} frame {f}: {} substeps, largest speed {:.3} m/s, centre {:.4} m/s, energy {:.4} of the start, {:.1} s wall",
                    row.name,
                    frame.substeps,
                    top * scene.cell_m,
                    vcom.length() * scene.cell_m,
                    row_energy(&scene) / e0,
                    clock.elapsed().as_secs_f32()
                );
            }
            if !top.is_finite() || late {
                break;
            }
        }
    }
}

/// The column at rest at 1 cm and one gravity, frame by frame for the first
/// second: largest particle speed, largest pressure error in cells of head and
/// where, largest move. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_column_frames() {
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
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
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x);
    let start = scene.x.clone();
    let g = scene.g();
    let clock = Instant::now();
    for f in 1..=60u32 {
        let frame = scene.frame();
        let top = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        let moved = scene
            .x
            .iter()
            .zip(&start)
            .fold(0.0f32, |m, (a, b)| m.max((*a - *b).length()));
        let mut worst = (0.0f32, 0usize, 0usize);
        for j in wall..ny {
            for i in wall..wall + width {
                let c = i + nx * j;
                let y = j as f32 + 0.5;
                if y >= h || !frame.active[c] {
                    continue;
                }
                let e = (frame.pressure[c] - g * (h - y)).abs() / g;
                if e > worst.0 {
                    worst = (e, i, j);
                }
            }
        }
        if f <= 6 || f % 6 == 0 {
            println!(
                "frame {f}: {} substeps, largest speed {:.3} cells/s, pressure off by {:.3} cell of head at ({}, {}), largest move {:.3} cell",
                frame.substeps, top, worst.0, worst.1, worst.2, moved
            );
        }
    }
    println!("{:.2} s wall for 1 s", clock.elapsed().as_secs_f32());
}

/// The free body's block of each solid row moved by the conserving
/// implicit step, at the substep `EMERGE_PROBE_STEP` (`p`, `4p` or `frame`,
/// `Substep::parse`), frame
/// by frame: energy against the start, the centre's speed, Newton and
/// GMRES work, wall time. Stops after five minutes. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_conserving_free_vibration() {
    let which = std::env::var("EMERGE_PROBE_STEP").unwrap_or_else(|_| "frame".into());
    for row in material_rows().into_iter().filter(|r| r.mu > 0.0) {
        let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
        let mut rng = LcgRng::new(17);
        let (lo, hi) = (Vec2::new(31.0, 31.0), Vec2::new(41.0, 41.0));
        let x = seed(
            (lo.as_ivec2(), hi.as_ivec2()),
            |_| true,
            |_| (f32::INFINITY, Vec2::Y),
            &mut rng,
        );
        let mut scene = Scene::new((72, 72), 0.01, 0.0, tank, x)
            .with_lame(row.lambda / row.rho, row.mu / row.rho);
        let epsilon = std::env::var("EMERGE_PROBE_EPSILON")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
            epsilon,
            ..Default::default()
        });
        let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
        let substep = Substep::parse(&which).unwrap_or(Substep::Frame);
        scene.fixed_dt = Some(substep.length(scene.layout.dx, c_p));
        let centre = 0.5 * (lo + hi);
        let s = 0.5 / scene.cell_m / (5.0 * std::f32::consts::SQRT_2);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            scene.v[p] = s * Vec2::new(r.y, r.x);
        }
        let e0 = row_energy(&scene);
        let clock = Instant::now();
        for f in 1..=60u32 {
            let frame = scene.frame();
            let (_, vcom) = scene.centre_of_mass();
            let late = clock.elapsed().as_secs() > 900;
            let e = row_energy(&scene) / e0;
            if f <= 3 || f % 6 == 0 || !e.is_finite() || late {
                let st = scene.probe.conserving;
                println!(
                    "{} [{which}] frame {f}: {} substeps, energy {:.6} of the start, centre {:.5} m/s, Newton {} GMRES {} (max {}), residual {:.1e}, {} unconverged, {:.1} s wall, stabilisation took {:.6}",
                    row.name,
                    frame.substeps,
                    e,
                    vcom.length() * scene.cell_m,
                    st.0,
                    st.1,
                    st.2,
                    st.3,
                    st.4,
                    clock.elapsed().as_secs_f32(),
                    scene.probe.stabilisation_loss / e0
                );
            }
            if !e.is_finite() || late {
                break;
            }
        }
    }
}
