use super::*;

/// Per-body steps, a first measured test, nothing built into the engine: two
/// medium clay blocks in one tank, 22 cells apart so no grid node is shared, A
/// dropped onto the floor at 2 m/s and B thrown up at 2 m/s while vibrating
/// (the free body's shear start, corners at 2 m/s), out of every wall's reach
/// for the whole 0.4 s. Run together with the contact refinement as it is,
/// global (`Scene::contact_substep` refines the whole scene), then each block
/// alone in its own scene with its own refinement. A takes the same steps in
/// both, so A together against A alone is what the bodies' shared solve
/// couples; B alone keeps frame steps where together it followed A's
/// refinement. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_per_body_steps() {
    use std::ops::Range;
    let (e, nu, rho, side) = (11.25e6f32, 0.45f32, 1700.0f32, 10.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let block = |centre: Vec2, seed_value: u32| clay_block(centre, side, seed_value);
    let (centre_a, centre_b) = (
        Vec2::new(20.0, 4.0 + 2.0 + 0.5 * side),
        Vec2::new(52.0, 36.0),
    );
    let (seed_a, seed_b) = (block(centre_a, 17), block(centre_b, 23));
    // Each body's start: positions, velocities, APIC matrices.
    let start = |body: usize| -> (Vec<Vec2>, Vec<Vec2>, Vec<Mat2>) {
        if body == 0 {
            let n = seed_a.len();
            (
                seed_a.clone(),
                vec![Vec2::new(0.0, -200.0); n],
                vec![Mat2::ZERO; n],
            )
        } else {
            let s = 200.0 / (0.5 * side * std::f32::consts::SQRT_2);
            let v = seed_b
                .iter()
                .map(|&q| {
                    let r = q - centre_b;
                    s * Vec2::new(r.y, r.x) + Vec2::new(0.0, 200.0)
                })
                .collect();
            let c = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
            (seed_b.clone(), v, vec![c; seed_b.len()])
        }
    };
    // The scene holding `bodies`, with each one's particle range.
    let build = |bodies: &[usize], tolerance: f64| -> (Scene, Vec<Range<usize>>) {
        let (mut x, mut v, mut c, mut ranges) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &b in bodies {
            let (xb, vb, cb) = start(b);
            ranges.push(x.len()..x.len() + xb.len());
            x.extend(xb);
            v.extend(vb);
            c.extend(cb);
        }
        let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x).with_elastic(e / rho, nu);
        scene.v = v;
        scene.c = c;
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
            occupation: Some(0.75),
            epsilon: 0.0,
            newton_tolerance: tolerance,
            ..Default::default()
        });
        scene.refine_contact = Some(1.0);
        scene.fixed_dt = Some(FRAME);
        (scene, ranges)
    };
    let measure = |scene: &Scene, r: &Range<usize>| body_measure(scene, r.clone());
    let frames = 24;
    // One run: per body and frame, the measures and the positions; the
    // substeps and frames refined; the wall time.
    let run = |label: &str, bodies: &[usize], tolerance: f64| {
        let (mut scene, ranges) = build(bodies, tolerance);
        let mut series: Vec<Vec<[f64; 4]>> = vec![Vec::new(); ranges.len()];
        let mut positions: Vec<Vec<Vec<Vec2>>> = vec![Vec::new(); ranges.len()];
        let (mut substeps, mut refined) = (0u32, 0u32);
        let clock = Instant::now();
        for _ in 0..frames {
            refined += u32::from(scene.contact_substep().is_some());
            substeps += scene.frame().substeps;
            for (b, r) in ranges.iter().enumerate() {
                series[b].push(measure(&scene, r));
                positions[b].push(scene.x[r.clone()].to_vec());
            }
        }
        let wall = clock.elapsed().as_secs_f64();
        let st = scene.probe.conserving;
        println!(
            "{label}: {substeps} substeps, {refined} of {frames} frames refined, {:.1} s wall ({:.1} s per simulated s), unconverged {} of {} Newton, worst residual {:.1e}",
            wall,
            wall / (f64::from(frames) * f64::from(FRAME)),
            st.4,
            st.0,
            st.3
        );
        (series, positions, wall)
    };
    // One body in two runs: the measures' worst gaps, and the particles'
    // largest gap at a few frames.
    let compare = |label: &str,
                   s: &[[f64; 4]],
                   q: &[[f64; 4]],
                   xs: &[Vec<Vec2>],
                   xq: &[Vec<Vec2>]| {
        let e0 = q[0][0];
        let (mut energy, mut kinetic, mut height, mut speed) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut drift_s, mut drift_q) = (0.0f64, 0.0f64);
        for (a, b) in s.iter().zip(q) {
            energy = energy.max((a[0] - b[0]).abs() / e0);
            kinetic = kinetic.max((a[1] - b[1]).abs() / e0);
            height = height.max((a[2] - b[2]).abs());
            speed = speed.max((a[3] - b[3]).abs());
            drift_s = drift_s.max((a[0] / e0 - 1.0).abs());
            drift_q = drift_q.max((b[0] / e0 - 1.0).abs());
        }
        let gap = |f: usize| {
            xs[f]
                .iter()
                .zip(&xq[f])
                .fold(0.0f32, |m, (a, b)| m.max((*a - *b).length()))
        };
        let by_frame: Vec<String> = [1, 2, 3, 4, 8, 16, 24]
            .iter()
            .map(|&f| format!("{f}: {:.1e}", gap(f - 1)))
            .collect();
        println!(
            "  {label}: energy off by {energy:.2e} of its start at worst (energy change {drift_s:.2e} and {drift_q:.2e}), kinetic off by {kinetic:.2e}, centre height off by {:.4} mm, centre speed off by {:.2e} m/s; particles off by at most, in cells, after frame {}",
            1000.0 * height,
            speed,
            by_frame.join(", ")
        );
    };
    let default =
        crate::spacetime::total_lagrangian::ConservingSettings::default().newton_tolerance;
    let (together, together_x, together_wall) =
        run("together, refinement global", &[0, 1], default);
    let (a_alone, a_x, a_wall) = run("body A alone", &[0], default);
    compare(
        "A together against alone",
        &together[0],
        &a_alone[0],
        &together_x[0],
        &a_x[0],
    );
    let (b_alone, b_x, b_wall) = run("body B alone", &[1], default);
    compare(
        "B together against alone",
        &together[1],
        &b_alone[0],
        &together_x[1],
        &b_x[0],
    );
    println!(
        "wall time: together {together_wall:.1} s, the two alone {:.1} s ({:.2}x)",
        a_wall + b_wall,
        together_wall / (a_wall + b_wall)
    );
    // The control: A alone again with a Newton tolerance a hundred times
    // tighter, the size of perturbation the shared solve brings.
    let (a_tight, a_tight_x, _) = run("body A alone, Newton tolerance 1e-12", &[0], 1e-12);
    compare(
        "A tight against default, both alone",
        &a_tight[0],
        &a_alone[0],
        &a_tight_x[0],
        &a_x[0],
    );
    compare(
        "A together against A alone with the tight tolerance",
        &together[0],
        &a_tight[0],
        &together_x[0],
        &a_tight_x[0],
    );
}

/// One group of the per-body stress probe: bodies stepped together, the
/// scene holding them and each one's particles.
struct Group {
    scene: Scene,
    bodies: Vec<usize>,
    ranges: Vec<std::ops::Range<usize>>,
}

impl Group {
    /// Body `i`'s positions and velocities.
    fn motion(&self, i: usize) -> (&[Vec2], &[Vec2]) {
        let r = self.ranges[i].clone();
        (&self.scene.x[r.clone()], &self.scene.v[r])
    }
}

/// Whether two bodies, positions and velocities, come within `reach` cells
/// of each other along both axes at once (the max norm, a quadratic
/// B-spline's support being square) at the frame's start or at its
/// predicted end, each particle on its own trajectory with the weight
/// (`Scene::contact_substep`'s prediction).
fn bodies_within(a: (&[Vec2], &[Vec2]), b: (&[Vec2], &[Vec2]), gravity: Vec2, reach: f32) -> bool {
    let ends = |(x, v): (&[Vec2], &[Vec2])| -> Vec<(Vec2, Vec2)> {
        x.iter()
            .zip(v)
            .map(|(&q, &w)| (q, q + w * FRAME + 0.5 * gravity * FRAME * FRAME))
            .collect()
    };
    let (ea, eb) = (ends(a), ends(b));
    let bounds = |e: &[(Vec2, Vec2)]| {
        e.iter().fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(lo, hi), &(q, w)| (lo.min(q).min(w), hi.max(q).max(w)),
        )
    };
    let ((alo, ahi), (blo, bhi)) = (bounds(&ea), bounds(&eb));
    if (alo - Vec2::splat(reach)).cmpgt(bhi).any() || (blo - Vec2::splat(reach)).cmpgt(ahi).any() {
        return false;
    }
    let apart = |p: Vec2, q: Vec2| (p - q).abs().max_element();
    ea.iter().any(|&(qa, wa)| {
        eb.iter()
            .any(|&(qb, wb)| apart(qa, qb) < reach || apart(wa, wb) < reach)
    })
}

/// Per-body steps under stress, a probe, nothing built into the engine: up
/// to six medium clay blocks in a 1.36 m tank over 0.6 s
/// (`EMERGE_PROBE_BODIES`, default `0,1,2,3,4,5`; `EMERGE_PROBE_FRAMES`,
/// default 36). 0 is dropped at 2 m/s, 1 and 2 fall from rest from two
/// heights, 3 is thrown up at 2 m/s while vibrating, 4 and 5 fly at each
/// other at 1 m/s each and meet in the air. Run once as one scene, refined
/// whenever any body nears a wall or another body (the contact refinement,
/// global), then as groups that each take their own steps: a group refines
/// when one of its bodies nears a wall or another of its bodies, and two
/// groups merge as soon as their bodies come within 5 cells along both
/// axes, before they could share a node (a particle's support reaches 1.5
/// cells each way). A guard counts the frames ending with two groups within
/// 2 cells of a common cell. The approach between bodies refining is this
/// probe's: the refinement as it is only knows the walls.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_per_body_stress() {
    // The largest of two, NaN when either is: `f64::max` would drop a NaN and
    // a broken run would read as a small gap.
    let worst = |m: f64, x: f64| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else {
            m.max(x)
        }
    };
    let (e, nu, rho, side) = (11.25e6f32, 0.45f32, 1700.0f32, 10.0f32);
    let (nx, ny) = (144usize, 72usize);
    let tank = (Vec2::splat(4.0), Vec2::new(140.0, 68.0));
    let reach = 5.0f32;
    // `EMERGE_PROBE_CONDITIONING`: each group of several bodies measures its
    // mass conditioning every step, the union's and each body's own.
    let conditioning = std::env::var_os("EMERGE_PROBE_CONDITIONING").is_some();
    // `EMERGE_PROBE_EPSILON`: the effective mass's `eps` (Love and Sulsky eq.
    // 26), 0 the consistent mass; `EMERGE_PROBE_NO_REFINE`: every substep
    // one frame, no refinement near walls or bodies; `EMERGE_PROBE_BASIS`:
    // the conditioning below which a body's basis is dissected.
    let epsilon: f64 = std::env::var("EMERGE_PROBE_EPSILON")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let no_refine = std::env::var_os("EMERGE_PROBE_NO_REFINE").is_some();
    let basis_threshold: f64 = std::env::var("EMERGE_PROBE_BASIS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let chosen: Vec<usize> = std::env::var("EMERGE_PROBE_BODIES")
        .unwrap_or_else(|_| "0,1,2,3,4,5".into())
        .split(',')
        .filter_map(|b| b.trim().parse().ok())
        .collect();
    let frames: usize = std::env::var("EMERGE_PROBE_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(36);
    // Each body's start: positions, velocities, APIC matrices.
    let start = |body: usize| -> (Vec<Vec2>, Vec<Vec2>, Vec<Mat2>) {
        let shake = 200.0 / (0.5 * side * std::f32::consts::SQRT_2);
        let (centre, push, vibrating) = match body {
            0 => (Vec2::new(13.0, 11.0), Vec2::new(0.0, -200.0), false),
            1 => (Vec2::new(31.0, 30.0), Vec2::ZERO, false),
            2 => (Vec2::new(49.0, 50.0), Vec2::ZERO, false),
            3 => (Vec2::new(67.0, 26.0), Vec2::new(0.0, 200.0), true),
            4 => (Vec2::new(87.0, 56.0), Vec2::new(100.0, 0.0), false),
            _ => (Vec2::new(127.0, 56.0), Vec2::new(-100.0, 0.0), false),
        };
        let x = clay_block(centre, side, 17 + body as u32);
        let n = x.len();
        if vibrating {
            let v = x
                .iter()
                .map(|&q| {
                    let r = q - centre;
                    shake * Vec2::new(r.y, r.x) + push
                })
                .collect();
            let c = Mat2::from_cols(Vec2::new(0.0, shake), Vec2::new(shake, 0.0));
            (x, v, vec![c; n])
        } else {
            (x, vec![push; n], vec![Mat2::ZERO; n])
        }
    };
    let starts: Vec<_> = (0..6).map(start).collect();
    // The scene holding `bodies` at their start, with each one's particles.
    let build = |bodies: &[usize]| -> Group {
        let (mut x, mut v, mut c, mut ranges) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &b in bodies {
            let (xb, vb, cb) = &starts[b];
            ranges.push(x.len()..x.len() + xb.len());
            x.extend_from_slice(xb);
            v.extend_from_slice(vb);
            c.extend_from_slice(cb);
        }
        let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_elastic(e / rho, nu);
        scene.v = v;
        scene.c = c;
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
            occupation: Some(0.75),
            epsilon,
            ..Default::default()
        });
        scene.refine_contact = (!no_refine).then_some(1.0);
        scene.fixed_dt = Some(FRAME);
        Group {
            scene,
            bodies: bodies.to_vec(),
            ranges,
        }
    };
    // Whether any two of a group's bodies near, or a body of each of two.
    let near = |a: &Group, b: Option<&Group>| -> bool {
        let other = b.unwrap_or(a);
        (0..a.bodies.len()).any(|i| {
            let from = if b.is_some() { 0 } else { i + 1 };
            (from..other.bodies.len())
                .any(|j| bodies_within(a.motion(i), other.motion(j), a.scene.gravity, reach))
        })
    };
    // The cells within 2 of a group's particles.
    let cells = |g: &Group| -> Vec<bool> {
        let mut out = vec![false; nx * ny];
        for q in &g.scene.x {
            let (i0, j0) = (
                (q.x - 2.0).floor().max(0.0) as usize,
                (q.y - 2.0).floor().max(0.0) as usize,
            );
            let (i1, j1) = (
                ((q.x + 2.0).floor() as usize).min(nx - 1),
                ((q.y + 2.0).floor() as usize).min(ny - 1),
            );
            for j in j0..=j1 {
                for i in i0..=i1 {
                    out[i + nx * j] = true;
                }
            }
        }
        out
    };
    let c_p = {
        let g = build(&chosen[..1]);
        (g.scene.sound_speed2.unwrap_or(0.0) + g.scene.shear.unwrap_or(0.0)).sqrt()
    };
    let p_step = 0.5 / c_p;
    // One run, per body: the measures and positions each frame, the frames
    // it was refined; and the run's totals.
    let run = |label: &str, per_body: bool| {
        let mut groups: Vec<Group> = if per_body {
            chosen.iter().map(|&b| build(&[b])).collect()
        } else {
            vec![build(&chosen)]
        };
        let index = |b: usize| chosen.iter().position(|&c| c == b).expect("chosen");
        let mut series = vec![Vec::new(); chosen.len()];
        let mut positions = vec![Vec::new(); chosen.len()];
        let mut refined = vec![0u32; chosen.len()];
        let (mut particle_steps, mut flags) = (0u64, 0u32);
        let mut merges = Vec::new();
        // Newton, unconverged, worst residual and phases of groups merged
        // away.
        let mut carried = (0u64, 0u32, 0.0f64, [0.0f64; 5]);
        let e0: f64 = groups
            .iter()
            .flat_map(|g| {
                g.ranges
                    .iter()
                    .map(|r| body_measure(&g.scene, r.clone())[0])
            })
            .sum();
        let mut drift = 0.0f64;
        let mut first_broken: Option<usize> = None;
        let clock = Instant::now();
        for frame in 0..frames {
            // Groups merge before their bodies could share a node.
            while let Some((a, b)) = per_body
                .then(|| {
                    (0..groups.len())
                        .flat_map(|a| (a + 1..groups.len()).map(move |b| (a, b)))
                        .find(|&(a, b)| near(&groups[a], Some(&groups[b])))
                })
                .flatten()
            {
                let second = groups.remove(b);
                let first = groups.remove(a);
                let bodies: Vec<usize> =
                    first.bodies.iter().chain(&second.bodies).copied().collect();
                let mut merged = build(&bodies);
                for (old, offset) in [(&first, 0), (&second, first.bodies.len())] {
                    for (i, r) in old.ranges.iter().enumerate() {
                        let to = merged.ranges[offset + i].clone();
                        let (s, m) = (&old.scene, &mut merged.scene);
                        m.x[to.clone()].copy_from_slice(&s.x[r.clone()]);
                        m.v[to.clone()].copy_from_slice(&s.v[r.clone()]);
                        m.c[to.clone()].copy_from_slice(&s.c[r.clone()]);
                        m.deformation[to.clone()].copy_from_slice(&s.deformation[r.clone()]);
                        m.log_j[to].copy_from_slice(&s.log_j[r.clone()]);
                    }
                    let st = old.scene.probe.conserving;
                    carried.0 += st.0;
                    carried.1 += st.4;
                    carried.2 = carried.2.max(st.3);
                    for (t, part) in carried.3.iter_mut().zip(old.scene.probe.phases) {
                        *t += part;
                    }
                }
                merged.scene.time = first.scene.time;
                merges.push(format!("{bodies:?} at frame {}", frame + 1));
                groups.push(merged);
            }
            let mut stepped = Vec::new();
            for g in &mut groups {
                let bodies_near = !no_refine && near(g, None);
                g.scene.fixed_dt = Some(if bodies_near { p_step } else { FRAME });
                if bodies_near || g.scene.contact_substep().is_some() {
                    for &b in &g.bodies {
                        refined[index(b)] += 1;
                    }
                }
                let measure = conditioning && g.bodies.len() > 1;
                if measure {
                    g.scene.probe.conditioning_bodies = g.ranges.clone();
                    g.scene.probe.conditioning = (f64::INFINITY, Vec::new());
                    g.scene.probe.basis_threshold = basis_threshold;
                    g.scene.probe.basis_worst = None;
                }
                g.scene.probe.grid_speed_ratio = 0.0;
                g.scene.probe.start_residual = 0.0;
                let substeps = g.scene.frame().substeps;
                particle_steps += u64::from(substeps) * g.scene.x.len() as u64;
                let st = g.scene.probe.conserving;
                let measured = if measure {
                    let (union, each) = &g.scene.probe.conditioning;
                    let each: Vec<String> = each.iter().map(|c| format!("{c:.1e}")).collect();
                    format!(
                        ", mass conditioning union {union:.1e}, each body {}",
                        each.join(" ")
                    )
                } else {
                    String::new()
                };
                stepped.push(format!(
                    "{:?} {substeps} (solves {}, held at most {}, Newton {}, GMRES {} and at most {} in a step, worst residual {:.1e}, unconverged {}{measured}, grid speed over the fastest particle {:.2}, Newton start {:.1e}, mass stabilisation's loss so far {:.2e} of the run's start)",
                    g.bodies,
                    g.scene.probe.passes,
                    g.scene.probe.contacts,
                    st.0,
                    st.1,
                    st.2,
                    st.3,
                    st.4,
                    g.scene.probe.grid_speed_ratio,
                    g.scene.probe.start_residual,
                    g.scene.probe.stabilisation_loss / e0
                ));
                if let Some((c, report)) = &g.scene.probe.basis_worst {
                    stepped.push(format!("\n      basis at conditioning {c:.1e}, {report}"));
                }
            }
            println!(
                "  {label} frame {}: substeps by group {}, {:.0} s",
                frame + 1,
                stepped.join(", "),
                clock.elapsed().as_secs_f64()
            );
            if per_body {
                let marks: Vec<Vec<bool>> = groups.iter().map(cells).collect();
                let shared = (0..marks.len()).any(|a| {
                    (a + 1..marks.len())
                        .any(|b| marks[a].iter().zip(&marks[b]).any(|(&p, &q)| p && q))
                });
                flags += u32::from(shared);
            }
            let mut total = 0.0;
            for g in &groups {
                for (i, &b) in g.bodies.iter().enumerate() {
                    let m = body_measure(&g.scene, g.ranges[i].clone());
                    total += m[0];
                    series[index(b)].push(m);
                    positions[index(b)].push(g.scene.x[g.ranges[i].clone()].to_vec());
                }
            }
            drift = worst(drift, (total / e0 - 1.0).abs());
            if first_broken.is_none()
                && groups
                    .iter()
                    .any(|g| g.scene.x.iter().any(|q| !q.is_finite()))
            {
                first_broken = Some(frame + 1);
            }
        }
        let wall = clock.elapsed().as_secs_f64();
        let (mut newton, mut unconverged, mut residual, mut phases) = carried;
        for g in &groups {
            let st = g.scene.probe.conserving;
            newton += st.0;
            unconverged += st.4;
            residual = residual.max(st.3);
            for (t, part) in phases.iter_mut().zip(g.scene.probe.phases) {
                *t += part;
            }
        }
        println!(
            "{label}: {} body-frames refined of {}, {:.2}e6 particle-steps, {wall:.1} s wall ({:.1} s per simulated s), phases grid {:.1} s, projection {:.1}, preconditioner {:.1}, Newton {:.1}, update {:.1}; unconverged {unconverged} of {newton} Newton, worst residual {residual:.1e}; total energy change {drift:.2e} at worst; first frame with a position not finite {first_broken:?}; merges {merges:?}; frames ending with groups within reach of a common cell {flags}",
            refined.iter().sum::<u32>(),
            chosen.len() * frames,
            particle_steps as f64 * 1e-6,
            wall / (frames as f64 * f64::from(FRAME)),
            phases[0],
            phases[1],
            phases[2],
            phases[3],
            phases[4]
        );
        (series, positions, refined, wall)
    };
    let (one, one_x, one_refined, one_wall) = run("one scene, refinement global", false);
    let (each, each_x, each_refined, each_wall) = run("groups, each its own steps", true);
    println!(
        "wall time: one scene {one_wall:.1} s, groups {each_wall:.1} s ({:.2}x)",
        one_wall / each_wall
    );
    for (i, &b) in chosen.iter().enumerate() {
        let (s, q) = (&one[i], &each[i]);
        let e0 = q[0][0];
        let (mut kinetic, mut height, mut speed) = (0.0f64, 0.0f64, 0.0f64);
        for (a, c) in s.iter().zip(q) {
            kinetic = worst(kinetic, (a[1] - c[1]).abs() / e0);
            height = worst(height, (a[2] - c[2]).abs());
            speed = worst(speed, (a[3] - c[3]).abs());
        }
        let up = |r: &[[f64; 4]]| r.iter().fold(f64::NEG_INFINITY, |m, f| worst(m, f[3]));
        let gap = |f: usize| {
            one_x[i][f]
                .iter()
                .zip(&each_x[i][f])
                .fold(0.0f64, |m, (a, c)| worst(m, f64::from((*a - *c).length())))
        };
        let marks: Vec<String> = [1, frames / 4, frames / 2, 3 * frames / 4, frames]
            .iter()
            .map(|&f| format!("{f}: {:.1e}", gap(f.max(1) - 1)))
            .collect();
        println!(
            "  body {b}: refined {} frames in one scene, {} alone or grouped; centre height off by {:.3} mm at worst, centre speed off by {speed:.2e} m/s, kinetic off by {kinetic:.2e} of its start; fastest upward centre speed {:.3} and {:.3} m/s; particles off by at most, in cells, after frame {}",
            one_refined[i],
            each_refined[i],
            1000.0 * height,
            up(s),
            up(q),
            marks.join(", ")
        );
    }
}
