//! The real-time ladder's harness: tests/replay_ledger.rs's clay block and replay writer, copied
//! unchanged except for the rung (`EMERGE_RT_CONFIG`: `stepped`, `modal` flight,
//! `handoff` sub-frame) and the header's build label
//! (`EMERGE_REPLAY_BUILD`). In-crate because its knobs exist only in the
//! library's test build.

use super::TotalLagrangianConfig;
use crate::materials::MaterialModel;
use crate::materials::physical_props::Elastic;
use crate::materials::relaxation::ViscoelasticRecord;
use crate::materials::solid::viscoplastic_records::ViscoplasticRecord;
use crate::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// The 8 x 4 cm block of the material `made` builds, `props`' density,
/// dropped `lift` cells, at the step tolerance `tolerance`, its free frames
/// flown by its modes when `modal`.
fn block(
    (tolerance, rung): (f64, &str),
    lift: f32,
    made: &dyn Fn(&SimConfig) -> Box<dyn MaterialModel>,
    props: &Elastic,
) -> Simulation {
    // N cells through the block's 4 cm, h = 4/n
    // cm, the physical scene unchanged; n = 4 is today's block bit for bit.
    let n = cells_across();
    let cell_m = match n {
        1 => 0.04,
        2 => 0.02,
        4 => 0.01,
        8 => 0.005,
        _ => 0.04 / n as f32,
    };
    let lift = lift * n as f32 / 4.0;
    let mut config = SimConfig::earth(10 * n as usize, cell_m, 1.0 / 60.0);
    let path = TotalLagrangianConfig::new(tolerance);
    // The engine's modal rest asked for on
    // the rung (`EMERGE_RT_MODAL_REST`), nothing else changed.
    let mut path = if std::env::var_os("EMERGE_RT_MODAL_REST").is_some() {
        println!("knob active: modal rest");
        path.with_modal_rest()
    } else {
        path
    };
    // Acary's forecast activation (RR-7151, eqs.
    // 16e and 18, gamma 1) asked for (`EMERGE_RT_FORECAST`).
    if std::env::var_os("EMERGE_RT_FORECAST").is_some() {
        println!("knob active: forecast activation (gamma 1)");
        path.settings.forecast_activation = true;
    }
    config.total_lagrangian = Some(match rung {
        "stepped" => path,
        "modal" => path.with_modal_flight(),
        "handoff" => path.with_subframe_handoff(),
        other => panic!("EMERGE_RT_CONFIG {other}: stepped, modal or handoff"),
    });
    let mut sim = Simulation::empty(config);
    // A Coulomb floor (`EMERGE_PROBE_FLOOR_FRICTION`, its coefficient; default
    // the frictionless walls): every wall then takes it, as the path takes one
    // coefficient for every contact.
    if let Some(mu) = std::env::var("EMERGE_PROBE_FLOOR_FRICTION")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        println!("knob active: Coulomb walls, friction coefficient {mu}");
        sim.add_boundary_condition(Box::new(crate::boundary::FrictionBoundary::new(
            config.boundary_thickness,
            mu,
        )));
    }
    let id = sim.register_material(made(&config)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(2 * n, n))
            .at(Vec2::new(5.0 * n as f32, floor + lift + n as f32 / 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(props, &config),
    );
    sim
}

/// One frame of the replay: the format of `total_lagrangian_path.rs`
/// (time, positions m, speeds m/s, energy, heat, balance J/m, contacts) and
/// the ledger, J/m: the energy's parts (kinetic, stored, gravity's), this
/// frame's heat by origin (plastic work, relaxation branches, dashpots;
/// friction's work), the walls' normal work, the bound the solves allow on
/// the balance accumulated to this frame; the steps and rejected tries,
/// whether the frame was flown by the modes, and its wall time, ms.
fn frame_json(sim: &Simulation, time: f64, allowed: f64) -> String {
    let dx = f64::from(sim.config().dx_meters);
    let p = sim.particles();
    let f = sim.total_lagrangian_frame().expect("the path is on");
    let positions: Vec<String> =
        p.x.iter()
            .map(|q| format!("[{:.6},{:.6}]", f64::from(q.x) * dx, f64::from(q.y) * dx))
            .collect();
    let speeds: Vec<String> =
        p.v.iter()
            .map(|v| format!("{:.5}", f64::from(v.length()) * dx))
            .collect();
    let ledger = format!(
        "{{\"kinetic\":{:.9e},\"stored\":{:.9e},\"gravity\":{:.9e},\"plastic_heat\":{:.6e},\"relaxation_heat\":{:.6e},\"dashpot_heat\":{:.6e},\"friction_work\":{:.6e},\"contact_work\":{:.6e},\"allowed\":{:.3e}}}",
        f.parts[0],
        f.parts[1],
        f.parts[2],
        f.plastic_work,
        f.relaxation_heat,
        f.dissipated,
        f.friction_work,
        f.contact_work,
        allowed
    );
    format!(
        "{{\"time\":{time:.6},\"positions\":[{}],\"speed_m_s\":[{}],\"energy\":{:.9e},\"heat\":{:.9e},\"balance\":{:.3e},\"contacts\":{},\"ledger\":{ledger},\"steps\":{},\"rejected\":{},\"modal\":{},\"wall_ms\":{:.3}}}",
        positions.join(","),
        speeds.join(","),
        f.energy,
        f.heat,
        f.balance,
        f.contacts,
        f.steps,
        f.rejected,
        f.modal,
        1e3 * f.wall
    )
}

/// The particles' layout at the start, measured, not assumed: the distinct
/// columns and rows of their positions, and whether consecutive particles
/// run down a column (`column-major`) or along a row.
fn layout(sim: &Simulation) -> String {
    let x = &sim.particles().x;
    let distinct = |mut values: Vec<f32>| {
        values.sort_by(f32::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        values.len()
    };
    let columns = distinct(x.iter().map(|q| q.x).collect());
    let rows = distinct(x.iter().map(|q| q.y).collect());
    let order = if x.len() > 1 && (x[1].x - x[0].x).abs() < 1e-4 {
        "column-major"
    } else {
        "row-major"
    };
    format!("{{\"columns\":{columns},\"rows\":{rows},\"order\":\"{order}\"}}")
}

/// What a scene's header says of it: the tolerance, the material, and what
/// the controller was measured to honour of the tolerance.
struct Labels<'a> {
    tolerance: f64,
    material: &'a str,
    honoured: &'a str,
}

/// Runs `frames` frames and writes `name.json` in `EMERGE_REPLAY_DIR`; the
/// header carries the scene, its labels, the particles' layout, whether
/// frames may fly by the body's modes and how many did, the frame's
/// budget, the wall times and the machine label.
fn record(name: &str, mut sim: Simulation, frames: usize, labels: &Labels<'_>) {
    let frame_s = f64::from(sim.config().dt);
    let grid = layout(&sim);
    let modal = sim
        .config()
        .total_lagrangian
        .is_some_and(|path| path.modal_flight);
    let mut out = Vec::with_capacity(frames);
    let mut allowed = 0.0;
    let mut walls = Vec::with_capacity(frames);
    let mut breakdown = Vec::with_capacity(frames);
    let mut ledger_breaches = 0;
    crate::spacetime::total_lagrangian::hooks::take_parts();
    let rest = REST.with(|r| r.get());
    for k in 1..=frames {
        sim.step();
        if super::handoff::tracing() {
            vibration_count(&sim, k);
        }
        if rest || super::handoff::tracing() {
            rest_sample(&sim, k);
        }
        let f = sim.total_lagrangian_frame().expect("the path is on");
        allowed += f.residual_work + f.contact_work_bound;
        ledger_breaches += f.ledger_breaches;
        if f.ledger_breaches > 0 {
            println!(
                "frame {k}: {} kept steps broke their energy identity at the floor, balance {:+.4e} J/m",
                f.ledger_breaches, f.balance
            );
        }
        walls.push(1e3 * f.wall);
        out.push(frame_json(&sim, frame_s * k as f64, allowed));
        breakdown.push(breakdown_json(&f));
        // The controller's tries by class and
        // outcome, per frame (`EMERGE_PROBE_TRIES`).
        if tries_knob() {
            let t = crate::spacetime::total_lagrangian::hooks::take_tries();
            if t.iter().any(|&c| c > 0) {
                println!(
                    "TRIES frame {k}: kept {}/{}, rejected on the error {}/{}, unconverged {}/{} (smooth/switching)",
                    t[0], t[1], t[2], t[3], t[4], t[5]
                );
            }
            let r = crate::spacetime::total_lagrangian::hooks::take_reach_lengths();
            if r[0] > 0 {
                println!(
                    "REACH frame {k}: {} lists, mean expanded {:.1}, mean merged {:.1}",
                    r[0],
                    r[1] as f64 / r[0] as f64,
                    r[2] as f64 / r[0] as f64
                );
            }
            let s = crate::spacetime::total_lagrangian::residual_split::take();
            if s[0] > 0.0 {
                let us = |t: f64| 1e6 * t / s[0];
                println!(
                    "RESIDUAL frame {k}: {} samples, {:.0} points; per serial evaluation {:.1} us reads, {:.1} us base law, {:.1} us branches, {:.1} us step stress, {:.1} us scatter, {:.1} us whole",
                    s[0],
                    s[1] / s[0],
                    us(s[2]),
                    us(s[3]),
                    us(s[4]),
                    us(s[5]),
                    us(s[6]),
                    us(s[7])
                );
            }
            let b = crate::spacetime::total_lagrangian::hooks::take_band_builds();
            if b[0] > 0.0 {
                println!(
                    "BANDS frame {k}: {} builds, mean n {:.0}, mean width {:.0}; per build {:.3} ms ordering, {:.3} ms assembly, {:.3} ms factor",
                    b[0],
                    b[1] / b[0],
                    b[2] / b[0],
                    1e3 * b[3] / b[0],
                    1e3 * b[4] / b[0],
                    1e3 * b[5] / b[0]
                );
            }
        }
    }
    if rest {
        rest_report();
    }
    // The turning flight's own accounting.
    if super::exponential_flight::knob().on {
        let state = sim.total_lagrangian.as_ref().expect("the path is on");
        super::exponential_flight_probes::summary(name, &state.exponential_log);
    }
    // The cost breakdown: a sidecar file, so the replay itself is unchanged.
    if let Some(dir) = std::env::var_os("EMERGE_REPLAY_DIR") {
        let path = std::path::Path::new(&dir).join(format!("{name}_breakdown.json"));
        std::fs::write(
            &path,
            format!(
                "[
{}
]
",
                breakdown.join(
                    ",
"
                )
            ),
        )
        .expect("the replay directory is writable");
    }
    if crate::spacetime::total_lagrangian::observed_order::on() {
        println!(
            "{name}: observed order: {}",
            crate::spacetime::total_lagrangian::observed_order::report()
        );
    }
    if crate::spacetime::total_lagrangian::end_gap_refund::refund_on() {
        println!(
            "{name}: end-gap refund: {}",
            crate::spacetime::total_lagrangian::end_gap_refund::report()
        );
    }
    if crate::spacetime::total_lagrangian::ledger_guard::on() {
        let balance = sim.total_lagrangian_frame().map_or(0.0, |f| f.balance);
        println!(
            "{name}: ledger guard: {ledger_breaches} kept steps broke their energy identity at the floor; balance {balance:+.4e} J/m; the checks took {:.1} ms of {:.1} ms",
            1e3 * crate::spacetime::total_lagrangian::ledger_guard::spent(),
            walls.iter().sum::<f64>()
        );
    }
    let flown = sim.total_lagrangian_frame().map_or(0, |f| f.modal_frames);
    let mut sorted = walls.clone();
    sorted.sort_by(f64::total_cmp);
    let budget = 1e3 * frame_s;
    let over = walls.iter().filter(|&&w| w > budget).count();
    let config = sim.config();
    let header = format!(
        "{{\"scene\":\"{name}\",\"material\":\"{}\",\"dx_m\":{},\"frame_s\":{},\"gravity_m_s2\":{},\"floor_y_m\":{},\"path\":\"total Lagrangian, weighted-gap contact, Acary step doubling\",\"tolerance\":{:e},\"tolerance_honoured\":\"{}\",\"grid\":{grid},\"modal_flight\":{modal},\"modal_frames\":{flown},\"budget_ms\":{budget:.3},\"wall_ms_median\":{:.3},\"wall_ms_largest\":{:.3},\"frames_over_budget\":{over},\"build\":\"{}\",\"machine\":\"{}\",\"precomputed\":true}}",
        labels.material,
        config.dx_meters,
        config.dt,
        config.gravity.length() * config.dx_meters,
        config.boundary_thickness as f32 * config.dx_meters,
        labels.tolerance,
        labels.honoured,
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1],
        std::env::var("EMERGE_REPLAY_BUILD")
            .unwrap_or_else(|_| "test profile (optimized + debuginfo)".into()),
        std::env::var("EMERGE_REPLAY_LOAD").unwrap_or_else(|_| "not labelled".into())
    );
    let dir = std::env::var_os("EMERGE_REPLAY_DIR")
        .expect("EMERGE_REPLAY_DIR names the output directory");
    let path = std::path::Path::new(&dir).join(format!("{name}.json"));
    let body = format!(
        "{{\"header\":{header},\"frames\":[\n{}\n]}}\n",
        out.join(",\n")
    );
    std::fs::write(&path, body).expect("the replay directory is writable");
    println!(
        "{name}: {frames} frames written to {}; {flown} flown by the modes; wall median {:.1} ms, largest {:.1} ms, {over} of {frames} frames over the {budget:.1} ms budget",
        path.display(),
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1]
    );
}

/// Whether `EMERGE_PROBE_TRIES` asks for the tries' split; says so once.
fn tries_knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_TRIES").is_some();
        if on {
            crate::spacetime::total_lagrangian::hooks::take_tries();
            println!("knob active: the controller's tries by class and outcome per frame");
        }
        on
    })
}

/// One frame's cost breakdown: the frame report's timing parts (s), its
/// kept steps, rejected tries, kept Newton iterations and GMRES iterations,
/// and the TEMPORARY solve counters of the snapshot summed over every solve
/// of the frame (discarded and rejected tries included): Newton iterations,
/// solves, Jacobian products, linearisations, residual evaluations, the
/// solves' wall time.
fn breakdown_json(f: &super::TotalLagrangianFrame) -> String {
    let t = &f.timing;
    let p = crate::spacetime::total_lagrangian::hooks::take_parts();
    format!(
        "{{\"wall_ms\":{:.3},\"steps\":{},\"rejected\":{},\"newton_kept\":{},\"gmres_kept\":{},\"modal\":{},\"contacts\":{},\"newton_all\":{},\"solves_all\":{},\"products_all\":{},\"linearisations_all\":{},\"residuals_all\":{},\"solve_wall_ms\":{:.3},\"ms\":{{\"residual\":{:.3},\"jacobian\":{:.3},\"linearisation\":{:.3},\"push_derivative\":{:.3},\"responses_jacobian\":{:.3},\"preconditioner_build\":{:.3},\"preconditioner_solve\":{:.3},\"gmres\":{:.3},\"responses\":{:.3},\"contact\":{:.3},\"multipliers\":{:.3},\"grid\":{:.3},\"finishing\":{:.3},\"controller\":{:.3}}},\"internal_evaluations\":{},\"factorizations\":{},\"relaxation_heat\":{:.9e},\"relaxation_strain\":{:.3e},\"relaxation_beyond\":{}}}",
        1e3 * f.wall,
        f.steps,
        f.rejected,
        f.newton,
        f.gmres,
        f.modal,
        f.contacts,
        p[9],
        p[7],
        p[0],
        p[5],
        p[6],
        1e3 * p[8],
        1e3 * t.residual,
        1e3 * t.jacobian,
        1e3 * t.linearisation,
        1e3 * t.push_derivative,
        1e3 * t.responses_jacobian,
        1e3 * t.preconditioner_build,
        1e3 * t.preconditioner_solve,
        1e3 * t.gmres,
        1e3 * t.responses,
        1e3 * t.contact,
        1e3 * t.multipliers,
        1e3 * t.grid,
        1e3 * t.finishing,
        1e3 * t.controller,
        t.internal_evaluations,
        t.factorizations,
        f.relaxation_heat,
        f.relaxation_strain,
        f.relaxation_beyond
    )
}

/// `EMERGE_RT_CELLS_ACROSS` (default 4): the cells through the block's
/// thickness.
fn cells_across() -> i32 {
    static N: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("EMERGE_RT_CELLS_ACROSS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        println!("knob active: cells across {n}");
        n
    })
}

#[test]
#[ignore = "real-time ladder harness: run with --ignored --nocapture and EMERGE_REPLAY_DIR, EMERGE_RT_CONFIG set"]
fn realtime_ladder() {
    ladder();
}

/// The same block resting from `F = I` (no
/// lift), the ladder's rungs, cells, materials and tolerance; per frame
/// the centroid's sag and the volume-mean `sigma_yy` against the exact
/// static answer (`rest_sample`), and over frames 31-60 their means and
/// spreads.
#[test]
#[ignore = "rest harness: run with --ignored --nocapture and EMERGE_REPLAY_DIR, EMERGE_RT_CONFIG set"]
fn realtime_rest() {
    REST.with(|r| r.set(true));
    ladder();
}

thread_local! {
    /// The rest scene (`realtime_rest`) and its samples `(sag, sigma_yy)` over the
    /// exact ones, frame by frame.
    static REST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SAMPLES: std::cell::RefCell<Vec<(f64, f64)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The block's lift above the floor in quarter-thickness cells: the drop's
/// 10, none in the rest scene.
fn lift() -> f32 {
    if REST.with(|r| r.get()) { 0.0 } else { 10.0 }
}

/// After frame `k`, the centroid's sag (from its
/// start in the rest scene, from the floor plus half the thickness
/// otherwise) and the volume-mean Cauchy `sigma_yy` (`F S F^T / J`, `S` the
/// solid's own `second_pk`), each over the exact static answer (the rest check's,
/// derived there): `s = rho g H^2 / (3 E'')`, `E'' = 4 mu (lambda + mu) /
/// (lambda + 2 mu)`, mean `sigma_yy = -rho g H / 2`, from the run's own
/// moduli, density, gravity and thickness.
fn rest_sample(sim: &Simulation, k: usize) {
    let Some(state) = sim.total_lagrangian.as_ref() else {
        return;
    };
    let n = state.particles;
    let mass: f64 = state.mass[..n].iter().sum();
    let volume: f64 = state.v0[..n].iter().sum();
    let rho = mass / volume;
    let g = sim.total_lagrangian_gravity().y.abs();
    let (lambda, mu) = state.solids[0].lame();
    let thickness = f64::from(cells_across()) * f64::from(sim.config().dx_meters);
    let plane_modulus = 4.0 * mu * (lambda + mu) / (lambda + 2.0 * mu);
    let exact_sag = rho * g * thickness * thickness / (3.0 * plane_modulus);
    let exact_sigma = -rho * g * thickness / 2.0;
    let centroid = (0..n).map(|p| state.mass[p] * state.x[p].y).sum::<f64>() / mass;
    let floor = sim
        .total_lagrangian_walls()
        .0
        .iter()
        .filter(|plane| plane.axis == 1 && plane.inward > 0.0)
        .map(|plane| plane.at)
        .fold(f64::NEG_INFINITY, f64::max);
    // The rest scene's sag is from the start's own centroid:
    // the reference configuration is the start.
    let reference = if REST.with(|r| r.get()) {
        (0..n)
            .map(|p| state.mass[p] * state.reference[p].y)
            .sum::<f64>()
            / mass
    } else {
        floor + 0.5 * thickness
    };
    let (mut kirchhoff, mut current) = (0.0, 0.0);
    for p in 0..n {
        let f = state.f[p];
        let s = state.solids[p].second_pk(f.transpose() * f);
        kirchhoff += state.v0[p] * (f * s * f.transpose()).y_axis.y;
        current += state.v0[p] * f.determinant();
    }
    let sag = (reference - centroid) / exact_sag;
    let sigma = kirchhoff / current / exact_sigma;
    // Beside: the sag from the floor plus half the thickness, and the start's
    // own offset from it, both over the exact sag.
    let from_floor = (floor + 0.5 * thickness - centroid) / exact_sag;
    let start = (0..n)
        .map(|p| state.mass[p] * state.reference[p].y)
        .sum::<f64>()
        / mass;
    let start_offset = (start - floor - 0.5 * thickness) / exact_sag;
    println!(
        "REST frame {k}: sag over the exact {sag:+.4e} (exact {exact_sag:.4e} m), mean sigma_yy over the exact {sigma:+.4e} (exact {exact_sigma:.4e} Pa); from the floor {from_floor:+.4e}, the start's offset {start_offset:+.4e}"
    );
    SAMPLES.with(|v| v.borrow_mut().push((sag, sigma)));
}

/// The rest scene's frames 31-60: each ratio's mean and the standard
/// deviation of its samples.
fn rest_report() {
    SAMPLES.with(|v| {
        let v = v.borrow();
        let window: Vec<&(f64, f64)> = v.iter().skip(30).take(30).collect();
        if window.is_empty() {
            return;
        }
        let stats = |f: &dyn Fn(&(f64, f64)) -> f64| {
            let n = window.len() as f64;
            let mean = window.iter().map(|w| f(w)).sum::<f64>() / n;
            let spread = (window.iter().map(|w| (f(w) - mean).powi(2)).sum::<f64>() / n).sqrt();
            (mean, spread)
        };
        let (sag, sag_spread) = stats(&|w| w.0);
        let (sigma, sigma_spread) = stats(&|w| w.1);
        println!(
            "REST frames 31-60: sag over the exact mean {sag:.4e}, standard deviation {sag_spread:.4e}; mean sigma_yy over the exact mean {sigma:.4e}, standard deviation {sigma_spread:.4e}"
        );
    });
}

fn ladder() {
    let rung = std::env::var("EMERGE_RT_CONFIG").expect("EMERGE_RT_CONFIG names the rung");
    println!("knob active: real-time ladder rung {rung}");
    let clay = Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    // The rung's step tolerance (`EMERGE_RT_TOLERANCE`,
    // default 1e-1, the ladder's own) for the damped reference.
    let tolerance = std::env::var("EMERGE_RT_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-1);
    if tolerance != 1e-1 {
        println!("knob active: step tolerance {tolerance:e}");
    }
    let clay_labels = Labels {
        tolerance,
        material: "medium clay, corotated, E 11.25 MPa, nu 0.3, 1700 kg/m3 (elastic: no heat)",
        honoured: "kept local error about 2.31 times the tolerance (median; 95th 3.26, largest 4.72): probe_estimator_effectivity, 2026-10-20",
    };
    let name = match rung.as_str() {
        "stepped" => "clay_drop_e2",
        "modal" => "clay_drop_e2_modal",
        _ => "clay_drop_e2_handoff",
    };
    // The long run (`EMERGE_RT_FRAMES`, default
    // 40) and the dissipative clay (`EMERGE_RT_MATERIAL=norton`: the
    // plasticine record of tests/norton.rs at E 30 MPa, its stand-in
    // density) in the same cell.
    let frames = std::env::var("EMERGE_RT_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);
    if frames != 40 {
        println!("knob active: {frames} frames");
    }
    if std::env::var("EMERGE_RT_MATERIAL").as_deref() == Ok("norton") {
        println!("knob active: Norton plasticine record");
        let plasticine = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
        let (e_pa, rho) = (30.0e6, 119.0 * 0.453_592_37 / (0.3048 * 0.3048 * 0.3048));
        let props = Elastic {
            e_pa: e_pa as f32,
            nu: plasticine.nu as f32,
            rho_kg_m3: rho as f32,
        };
        let labels = Labels {
            tolerance,
            material: "technical plasticine, Norton record (Sapozhnikov 2014), E 30 MPa, stand-in density",
            honoured: "not measured for this record",
        };
        record(
            &format!("{name}_norton"),
            block(
                (labels.tolerance, rung.as_str()),
                lift(),
                &|config| Box::new(plasticine.material((e_pa, rho), config)),
                &props,
            ),
            frames,
            &labels,
        );
        return;
    }
    // Crandall et al.'s Plasticine, the record whole
    // (`ViscoelasticRecord::plasticine_crandall_1970_fresh`: its relaxed
    // moduli, Poisson's ratio, density and relaxation spectrum), the same
    // drop.
    if std::env::var("EMERGE_RT_MATERIAL").as_deref() == Ok("plasticine") {
        println!("knob active: Crandall Plasticine record (fresh)");
        let plasticine = ViscoelasticRecord::plasticine_crandall_1970_fresh();
        let labels = Labels {
            tolerance,
            material: "Harbutt's Plasticine, Crandall et al. 1970 record, fresh state: relaxed E 20.2 MPa, nu 0.434, 1906 kg/m3, three Zener mechanisms (loss factor 0.225, 100-3000 Hz)",
            honoured: "not measured for this record",
        };
        record(
            &format!("{name}_plasticine"),
            block(
                (labels.tolerance, rung.as_str()),
                lift(),
                &|config| {
                    let mut material =
                        CorotatedMaterial::from_physical(&plasticine.relaxed, config);
                    material.relaxation = Some(plasticine.spectrum);
                    Box::new(material)
                },
                &plasticine.relaxed,
            ),
            frames,
            &labels,
        );
        return;
    }
    record(
        name,
        block(
            (clay_labels.tolerance, rung.as_str()),
            lift(),
            &|config| Box::new(CorotatedMaterial::from_physical(&clay, config)),
            &clay,
        ),
        frames,
        &clay_labels,
    );
}

thread_local! {
    /// The body's free modes, built once (the grid stays on the reference).
    static MODES: std::cell::RefCell<Option<crate::spacetime::total_lagrangian::Modes>> =
        const { std::cell::RefCell::new(None) };
}

/// Each free elastic mode's kinetic
/// energy in the body frame after frame `k` (`q_i' = sum_p m_p v_p.
/// phi_i(X_p)`, the rigid translation and the spin `L / I` removed as
/// `Flight::start` does, `q_i'^2 / 2`, J/m), and once, each mode's
/// frequency with the loss factor the solid's relaxation spectrum gives
/// there (`M / M_r = a + i b`, `a = 1 + sum kappa omega^2 / (theta^2 +
/// omega^2)`, `b = sum kappa omega theta / (theta^2 + omega^2)`, `eta = b /
/// a`: the record's modulus, Blanc et al. eq. 10). Printed only; nothing
/// is changed.
fn vibration_count(sim: &Simulation, k: usize) {
    use crate::spacetime::total_lagrangian::{RIGID_MODES, body_modes};
    use glam::{DMat2, DVec2};
    let Some(state) = sim.total_lagrangian.as_ref() else {
        return;
    };
    let Some(cache) = state.cache.as_ref() else {
        return;
    };
    let n = state.particles;
    MODES.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = body_modes(
                &cache.grid,
                (&state.mass, &state.v0, &state.solids),
                (n, state.x.len()),
            );
            if let Some(modes) = slot.as_ref() {
                let spectrum = state.solids[0].relaxation.proportional();
                let line: Vec<String> = modes
                    .omega
                    .iter()
                    .enumerate()
                    .skip(RIGID_MODES)
                    .map(|(i, &w)| {
                        let eta = spectrum.map_or(0.0, |(mechanisms, _)| {
                            let (a, b) = mechanisms.iter().fold((1.0, 0.0), |(a, b), m| {
                                let d = m.theta * m.theta + w * w;
                                (
                                    a + m.kappa_shear * w * w / d,
                                    b + m.kappa_shear * w * m.theta / d,
                                )
                            });
                            b / a
                        });
                        format!("{i}:{:.1}Hz/eta{eta:.4}", w / (2.0 * std::f64::consts::PI))
                    })
                    .collect();
                println!("VIB modes {}", line.join(" "));
            }
        }
        let Some(modes) = slot.as_ref() else {
            return;
        };
        let total: f64 = state.mass[..n].iter().sum();
        let (sum_x, sum_v) = (0..n).fold((DVec2::ZERO, DVec2::ZERO), |(a, b), p| {
            (
                a + state.mass[p] * state.x[p],
                b + state.mass[p] * state.v[p],
            )
        });
        let (centre, velocity) = (sum_x / total, sum_v / total);
        let origin =
            (0..n).fold(DVec2::ZERO, |a, p| a + state.mass[p] * state.reference[p]) / total;
        let (momentum, inertia) = (0..n).fold((0.0, 0.0), |(l, i), p| {
            let arm = state.x[p] - centre;
            (
                l + state.mass[p] * arm.perp_dot(state.v[p] - velocity),
                i + state.mass[p] * arm.length_squared(),
            )
        });
        let spin = momentum / inertia.max(f64::MIN_POSITIVE);
        let a = (0..n).fold(DMat2::ZERO, |a, p| {
            let (r, s) = (state.x[p] - centre, state.reference[p] - origin);
            a + state.mass[p] * DMat2::from_cols(r * s.x, r * s.y)
        });
        let back = DMat2::from_angle(-(a.x_axis.y - a.y_axis.x).atan2(a.x_axis.x + a.y_axis.y));
        let energies: Vec<f64> = modes
            .at
            .iter()
            .skip(RIGID_MODES)
            .map(|at| {
                let rate: f64 = (0..n)
                    .map(|p| {
                        let arm = state.x[p] - centre;
                        let v = back * (state.v[p] - velocity - spin * arm.perp());
                        state.mass[p] * v.dot(at[p])
                    })
                    .sum();
                0.5 * rate * rate
            })
            .collect();
        let line: Vec<String> = energies.iter().map(|e| format!("{e:.3e}")).collect();
        println!(
            "VIB frame {k}: kinetic {:.6e} J/m [{}]",
            energies.iter().sum::<f64>(),
            line.join(" ")
        );
    });
}
