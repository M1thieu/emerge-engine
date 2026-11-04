use super::*;

/// How a conserving run steps: always at one substep, or at one frame per
/// substep refined to a multiple of the P wave step while near a wall
/// (`Scene::contact_substep`).
#[derive(Clone, Copy, Debug)]
enum Stepping {
    Fixed(Substep),
    RefinedAtContact(f32),
}

impl std::fmt::Display for Stepping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(substep) => write!(f, "at a substep of {substep}"),
            Self::RefinedAtContact(multiple) => write!(
                f,
                "at one frame per substep, refined at contact to {multiple} P wave steps"
            ),
        }
    }
}

/// One configuration of the conserving step's spread
/// (`probe_conserving_spread`): a square elastic block of `size` cells at `k`
/// cells per centimetre, Young's modulus `e` (Pa), Poisson's ratio `nu`,
/// density `rho`, either vibrating freely (no gravity, the free body's shear
/// start with corner speed `speed`) or dropped (1 g, starting 2 cells above the
/// floor at `speed` downwards), moved at one frame per substep with the
/// extended B-splines and the walls holding only while they push. Prints the
/// energy, the convergence, lift-off and the cost; returns, per frame, the
/// energy's change, the kinetic energy over the start's total, the centre of
/// mass's height (m) and vertical speed (m/s).
#[derive(Clone, Copy)]
struct SpreadCase {
    drop: bool,
    e: f32,
    nu: f32,
    rho: f32,
    size: f32,
    speed: f32,
    k: usize,
    seconds: f32,
    stepping: Stepping,
    /// Prints the error estimate of each frame (`Probe::estimator_reference`,
    /// the block's side as `u_ref`).
    estimate: bool,
}

fn spread_case(case: SpreadCase) -> Vec<[f64; 4]> {
    let SpreadCase {
        drop,
        e,
        nu,
        rho,
        size,
        speed,
        k,
        seconds,
        stepping,
        estimate,
    } = case;
    let kf = k as f32;
    let n = 72 * k;
    let tank = (Vec2::splat(4.0 * kf), Vec2::splat(68.0 * kf));
    let mut rng = LcgRng::new(17);
    let side = size * kf;
    let centre = Vec2::new(
        36.0 * kf,
        if drop {
            4.0 * kf + 2.0 * kf + 0.5 * side
        } else {
            36.0 * kf
        },
    );
    let (lo, hi) = (
        centre - Vec2::splat(0.5 * side),
        centre + Vec2::splat(0.5 * side),
    );
    let x = seed_body(
        true,
        (lo.floor().as_ivec2(), hi.ceil().as_ivec2()),
        |q| q.cmpge(lo).all() && q.cmple(hi).all(),
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
    let g = if drop { 1.0 } else { 0.0 };
    let mut scene = Scene::new((n, n), 0.01 / kf, g, tank, x).with_elastic(e / rho, nu);
    // `EMERGE_PROBE_OCCUPATION=none` keeps the original B-splines.
    let occupation = std::env::var("EMERGE_PROBE_OCCUPATION").ok();
    scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
        occupation: match occupation.as_deref() {
            Some("none") => None,
            other => Some(other.and_then(|c| c.parse().ok()).unwrap_or(0.75)),
        },
        // `eps` of the effective mass (`ConservingSettings::epsilon`): 1 is
        // the lumped mass.
        epsilon: std::env::var("EMERGE_PROBE_EPSILON")
            .ok()
            .and_then(|c| c.parse().ok())
            .unwrap_or(0.0),
        ..Default::default()
    });
    let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
    let (substep, refine) = match stepping {
        Stepping::Fixed(substep) => (substep, None),
        Stepping::RefinedAtContact(multiple) => (Substep::Frame, Some(multiple)),
    };
    scene.refine_contact = refine;
    scene.fixed_dt = Some(substep.length(scene.layout.dx, c_p));
    let cells = speed / scene.cell_m;
    if drop {
        scene.v.iter_mut().for_each(|v| *v = Vec2::new(0.0, -cells));
    } else {
        let s = cells / (0.5 * side * std::f32::consts::SQRT_2);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            scene.v[p] = s * Vec2::new(r.y, r.x);
            scene.c[p] = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
        }
    }
    let e0 = row_energy(&scene);
    let floor = tank.0.y;
    let (mut worst, mut nan) = (0.0f64, false);
    let (mut touched, mut lift): (bool, Option<f32>) = (false, None);
    let mut series = Vec::new();
    let clock = Instant::now();
    let verbose = std::env::var("EMERGE_PROBE_VERBOSE").is_ok();
    let mut substeps = 0u32;
    if estimate {
        scene.probe.estimator_reference = size * kf;
    }
    // `EMERGE_PROBE_CONDITIONING`: the body's mass conditioning every step.
    if std::env::var_os("EMERGE_PROBE_CONDITIONING").is_some() {
        scene.probe.conditioning_bodies = std::iter::once(0..scene.x.len()).collect();
        scene.probe.conditioning = (f64::INFINITY, Vec::new());
    }
    let mut estimates = Vec::new();
    for _ in 0..(seconds / FRAME).round() as u32 {
        substeps += scene.frame().substeps;
        estimates.push(std::mem::take(&mut scene.probe.largest_estimate));
        let change = row_energy(&scene) / e0 - 1.0;
        let kinetic: f64 = (0..scene.x.len())
            .map(|p| f64::from(0.5 * scene.mass[p] * scene.v[p].length_squared()))
            .sum();
        let (com, com_v) = scene.centre_of_mass();
        series.push([
            change,
            kinetic / e0,
            f64::from(com.y * scene.cell_m),
            f64::from(com_v.y * scene.cell_m),
        ]);
        if verbose {
            let st = scene.probe.conserving;
            let j = (0..scene.x.len())
                .map(|p| scene.j(p))
                .fold(f32::INFINITY, f32::min);
            println!(
                "  t={:.3}: energy {change:+.2e}, Newton {} GMRES {} worst residual {:.1e} unconverged {}, smallest J {j:.4}, ledger particles {:+.3e} grid {:+.3e} stored {:+.3e}, work contact {:+.3e} weight {:+.3e}",
                scene.time,
                st.0,
                st.1,
                st.3,
                st.4,
                scene.probe.ledger.0,
                scene.probe.ledger.1,
                scene.probe.ledger.2,
                scene.probe.work.0,
                scene.probe.work.1
            );
        }
        nan |= !change.is_finite() || scene.x.iter().any(|q| !q.is_finite());
        if nan {
            break;
        }
        worst = worst.max(change.abs());
        if drop {
            let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
            touched |= lowest <= floor + kf;
            let (_, v) = scene.centre_of_mass();
            if touched && lowest > floor + kf && v.y > 0.0 {
                lift = Some(lift.unwrap_or(0.0).max(v.y * scene.cell_m));
            }
        }
    }
    let end = row_energy(&scene) / e0 - 1.0;
    let st = scene.probe.conserving;
    let wall = clock.elapsed().as_secs_f32() / seconds;
    println!(
        "{} {stepping}: E {e:.2e} nu {nu} size {size} cm speed {speed} m/s at {:.1} cm: worst energy {worst:.2e}, end {end:+.2e}{}, unconverged {} of {} Newton, worst residual {:.1e}, projection {:.1e}, reflections {}, held on a plane at most {}, {} substeps, {}{wall:.1} s per simulated s",
        if drop { "drop" } else { "vibration" },
        1.0 / kf,
        if nan { " NaN" } else { "" },
        st.4,
        st.0,
        st.3,
        scene.probe.worst_projection,
        scene.probe.reflections,
        scene.probe.contacts,
        substeps,
        if drop {
            format!("lift-off {:?} m/s, ", lift)
        } else {
            String::new()
        }
    );
    let per = |k: usize| 1000.0 * scene.probe.phases[k] / f64::from(scene.probe.steps.max(1));
    println!(
        "  per conserving step ({} steps, {:.2} solves, {:.2} Newton, {:.1} GMRES): grid {:.2} ms, projection {:.2}, preconditioner {:.2}, Newton {:.2}, update {:.2}",
        scene.probe.steps,
        scene.probe.passes as f64 / f64::from(scene.probe.steps.max(1)),
        scene.probe.conserving.0 as f64 / f64::from(scene.probe.steps.max(1)),
        scene.probe.conserving.1 as f64 / f64::from(scene.probe.steps.max(1)),
        per(0),
        per(1),
        per(2),
        per(3),
        per(4)
    );
    println!(
        "  mass stabilisation's loss {:.2e} of the start, grid speed over the fastest particle at most {:.2}, Newton start at worst {:.1e}{}",
        scene.probe.stabilisation_loss / e0,
        scene.probe.grid_speed_ratio,
        scene.probe.start_residual,
        if scene.probe.conditioning_bodies.is_empty() {
            String::new()
        } else {
            format!(
                ", mass conditioning at worst {:.1e}",
                scene.probe.conditioning.0
            )
        }
    );
    if estimate {
        let by_frame: Vec<String> = estimates.iter().map(|e| format!("{e:.1e}")).collect();
        println!("  error estimate by frame: {}", by_frame.join(" "));
    }
    series
}

/// The conserving step across states: stiffness, size, speed, Poisson's
/// ratio and resolution, one at a time around a base (medium clay's 11.25
/// MPa, 0.45, 1700 kg/m^3, 10 cm, 2 m/s, 1 cm), for free vibration
/// (`EMERGE_PROBE_SPREAD=vibration`) or drops (`drop`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_conserving_spread() {
    let drop = std::env::var("EMERGE_PROBE_SPREAD").as_deref() == Ok("drop");
    let seconds = if drop { 0.5 } else { 1.0 };
    let base = (11.25e6f32, 0.45f32, 1700.0f32, 10.0f32, 2.0f32, 1usize);
    let mut cases = vec![base];
    for e in [3.0e6, 1.0e8, 1.0e9, 1.0e10] {
        cases.push((e, base.1, base.2, base.3, base.4, base.5));
    }
    for size in [5.0, 20.0] {
        cases.push((base.0, base.1, base.2, size, base.4, base.5));
    }
    for speed in [0.2, 5.0] {
        cases.push((base.0, base.1, base.2, base.3, speed, base.5));
    }
    for nu in [0.3, 0.49] {
        cases.push((base.0, nu, base.2, base.3, base.4, base.5));
    }
    cases.push((base.0, base.1, base.2, base.3, base.4, 2));
    let only: Option<usize> = std::env::var("EMERGE_PROBE_CASE")
        .ok()
        .and_then(|c| c.parse().ok());
    for (i, &(e, nu, rho, size, speed, k)) in cases.iter().enumerate() {
        if only.is_none_or(|c| c == i) {
            spread_case(SpreadCase {
                drop,
                e,
                nu,
                rho,
                size,
                speed,
                k,
                seconds,
                stepping: Stepping::Fixed(Substep::Frame),
                estimate: false,
            });
        }
    }
}

/// The conserving step at several steppings on the spread's base block
/// (medium clay, 11.25 MPa, 0.45, 10 cm at 1 cm), a probe, not a gate.
/// Each run of `EMERGE_PROBE_RUNS` (default `p,frame,refined:1`, the first
/// the reference: `p`, `4p` or `frame` a fixed substep, `refined:m` one
/// frame per substep refined at contact to `m` P wave steps,
/// `Scene::contact_substep`) is compared frame by frame with the first: the
/// kinetic energy's share (the vibration's phase and amplitude) and the
/// centre of mass's height. `EMERGE_PROBE_WHICH`: `vibration` (the free
/// body's shear start, corner speed `EMERGE_PROBE_SPEED` or 2 m/s, 1 s) or a
/// drop speed in m/s (0.5 s).
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_contact_step_convergence() {
    let which = std::env::var("EMERGE_PROBE_WHICH").unwrap_or_else(|_| "vibration".into());
    let drop_speed: Option<f32> = which.parse().ok();
    let runs = std::env::var("EMERGE_PROBE_RUNS").unwrap_or_else(|_| "p,frame,refined:1".into());
    let mut reference: Option<Vec<[f64; 4]>> = None;
    for run in runs.split(',') {
        let stepping = match run.split_once(':') {
            Some(("refined", multiple)) => {
                Stepping::RefinedAtContact(multiple.parse().unwrap_or(1.0))
            }
            _ => {
                Stepping::Fixed(Substep::parse(run).unwrap_or_else(|| panic!("unknown run {run}")))
            }
        };
        let series = spread_case(SpreadCase {
            drop: drop_speed.is_some(),
            e: 11.25e6,
            nu: 0.45,
            rho: 1700.0,
            size: 10.0,
            speed: drop_speed.unwrap_or_else(|| {
                std::env::var("EMERGE_PROBE_SPEED")
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .unwrap_or(2.0)
            }),
            k: 1,
            seconds: if drop_speed.is_some() { 0.5 } else { 1.0 },
            stepping,
            estimate: false,
        });
        let finite = series
            .iter()
            .take_while(|s| s.iter().all(|e| e.is_finite()))
            .count();
        let (low, high) = series[..finite]
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), s| {
                (l.min(s[0]), h.max(s[0]))
            });
        let against = reference.as_ref().map(|r| {
            let frames = finite.min(r.len());
            let (mut square, mut kinetic, mut height) = (0.0f64, 0.0f64, 0.0f64);
            for (s, q) in series.iter().zip(r).take(frames) {
                square += (s[1] - q[1]).powi(2);
                kinetic = kinetic.max((s[1] - q[1]).abs());
                height = height.max((s[2] - q[2]).abs());
            }
            format!(
                ", against the reference over {frames} frames: kinetic share off by {:.4} rms, {kinetic:.4} at worst; centre height off by {:.3} mm at worst",
                (square / frames.max(1) as f64).sqrt(),
                1000.0 * height
            )
        });
        // The centre of mass after its lowest point: how high it rises
        // again and how fast it moves up (the rebound).
        let lowest = (0..finite).min_by(|&a, &b| series[a][2].total_cmp(&series[b][2]));
        let rebound = lowest.map(|k| {
            let after = &series[k..finite];
            let rise = after.iter().fold(f64::NEG_INFINITY, |m, s| m.max(s[2])) - series[k][2];
            let up = after.iter().fold(f64::NEG_INFINITY, |m, s| m.max(s[3]));
            format!(
                ", centre lowest {:.2} mm at frame {}, then rises {:.2} mm, up to {:.3} m/s upward",
                1000.0 * series[k][2],
                k + 1,
                1000.0 * rise,
                up
            )
        });
        println!(
            "  {run}: {finite} of {} frames finite, energy from {low:+.2e} to {high:+.2e}{}{}",
            series.len(),
            against.unwrap_or_default(),
            rebound.unwrap_or_default()
        );
        if reference.is_none() {
            reference = Some(series[..finite].to_vec());
        }
    }
}

/// The error estimate of Zienkiewicz and Xie (`Probe::estimator_reference`)
/// on the conserving step, frame by frame: the spread's base block (medium
/// clay, 10 cm at 1 cm) vibrating freely at one frame per substep and at
/// the P wave step, and dropped at 0.66 and 2 m/s with the contact
/// refinement, so that its values at an impact stand against those of a
/// vibration a frame's step does not resolve. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_error_estimator() {
    let case = |drop: bool, speed: f32, seconds: f32, stepping: Stepping| SpreadCase {
        drop,
        e: 11.25e6,
        nu: 0.45,
        rho: 1700.0,
        size: 10.0,
        speed,
        k: 1,
        seconds,
        stepping,
        estimate: true,
    };
    spread_case(case(false, 2.0, 1.0, Stepping::Fixed(Substep::Frame)));
    spread_case(case(false, 2.0, 0.05, Stepping::Fixed(Substep::PWave)));
    spread_case(case(true, 0.2, 0.1, Stepping::RefinedAtContact(1.0)));
    spread_case(case(true, 2.0, 0.1, Stepping::RefinedAtContact(1.0)));
}
