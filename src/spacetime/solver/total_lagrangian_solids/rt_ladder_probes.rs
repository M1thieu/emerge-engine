//! The real-time lever ladder's harness (tmp/papers/dl_realtime/
//! RT_LEVER_MEASUREMENT_PLAN.md, measurement worktree only, never committed
//! to dev): tests/replay_ledger.rs's clay block and replay writer, copied
//! unchanged except for the rung (`EMERGE_RT_CONFIG`: A0 stepped, A1 modal
//! flight, A2 sub-frame handoff, A3 handoff plus the fixed contact step,
//! which also needs `EMERGE_PROBE_CONTACT_FIXED_STEP`) and the header's
//! build label (`EMERGE_REPLAY_BUILD`). In-crate because the fixed-step knob
//! exists only in the library's test build.

use super::TotalLagrangianConfig;
use crate::materials::MaterialModel;
use crate::materials::physical_props::Elastic;
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
    // RT_RESOLUTION_SWEEP_PLAN.md: n cells through the block's 4 cm, h = 4/n
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
    config.total_lagrangian = Some(match rung {
        "A0" => path,
        "A1" => path.with_modal_flight(),
        "A2" | "A3" => path.with_subframe_handoff(),
        other => panic!("EMERGE_RT_CONFIG {other}: A0, A1, A2 or A3"),
    });
    let mut sim = Simulation::empty(config);
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
    let mut single_solve_fallbacks = 0;
    crate::spacetime::total_lagrangian::hooks::take_parts();
    for k in 1..=frames {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        allowed += f.residual_work + f.contact_work_bound;
        walls.push(1e3 * f.wall);
        out.push(frame_json(&sim, frame_s * k as f64, allowed));
        let (line, fallbacks) = breakdown_json(&f);
        breakdown.push(line);
        if fallbacks > 0 {
            println!("frame {k}: single-solve fallbacks {fallbacks}");
        }
        single_solve_fallbacks += fallbacks;
    }
    println!("single-solve fallbacks over the run: {single_solve_fallbacks}");
    // The fast-mode plan's cost breakdown (RT_LEVER_MEASUREMENT_PLAN.md's
    // harness; test-only): a sidecar file, so the replay itself is unchanged.
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

/// One frame's cost breakdown: the frame report's timing parts (s), its
/// kept steps, rejected tries, kept Newton iterations and GMRES iterations,
/// and the TEMPORARY solve counters of the snapshot summed over every solve
/// of the frame (discarded and rejected tries included): Newton iterations,
/// solves, Jacobian products, linearisations, residual evaluations, the
/// solves' wall time; the single-solve steps that fell back to the pair
/// (returned beside the line); and the heat backward Euler's steps booked
/// (FAST_MODE_PLAN.md sec. 13): total J/m, steps, the lowest step total,
/// the particles' negative terms (count, sum).
fn breakdown_json(f: &super::TotalLagrangianFrame) -> (String, u32) {
    let t = &f.timing;
    let p = crate::spacetime::total_lagrangian::hooks::take_parts();
    let fallbacks = crate::spacetime::total_lagrangian::take_single_solve_fallbacks();
    let scheme = crate::spacetime::total_lagrangian::hooks::take_scheme_heat();
    let line = format!(
        "{{\"wall_ms\":{:.3},\"steps\":{},\"rejected\":{},\"newton_kept\":{},\"gmres_kept\":{},\"modal\":{},\"contacts\":{},\"newton_all\":{},\"solves_all\":{},\"products_all\":{},\"linearisations_all\":{},\"residuals_all\":{},\"solve_wall_ms\":{:.3},\"ms\":{{\"residual\":{:.3},\"jacobian\":{:.3},\"linearisation\":{:.3},\"push_derivative\":{:.3},\"responses_jacobian\":{:.3},\"preconditioner_build\":{:.3},\"preconditioner_solve\":{:.3},\"gmres\":{:.3},\"responses\":{:.3},\"contact\":{:.3},\"multipliers\":{:.3},\"grid\":{:.3},\"finishing\":{:.3},\"controller\":{:.3}}},\"internal_evaluations\":{},\"factorizations\":{},\"single_solve_fallbacks\":{},\"scheme_heat\":{:.9e},\"scheme_steps\":{},\"scheme_lowest_step\":{:.3e},\"scheme_negative\":{},\"scheme_negative_sum\":{:.3e}}}",
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
        fallbacks,
        scheme.total,
        scheme.steps,
        if scheme.steps > 0 {
            scheme.lowest_step
        } else {
            0.0
        },
        scheme.negative,
        scheme.negative_sum
    );
    (line, fallbacks)
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
fn rt_ladder() {
    let rung = std::env::var("EMERGE_RT_CONFIG").expect("EMERGE_RT_CONFIG names the rung");
    println!("knob active: real-time ladder rung {rung}");
    let clay = Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let clay_labels = Labels {
        tolerance: 1e-1,
        material: "medium clay, corotated, E 11.25 MPa, nu 0.3, 1700 kg/m3 (elastic: no heat)",
        honoured: "kept local error about 2.31 times the tolerance (median; 95th 3.26, largest 4.72): probe_estimator_effectivity, 2026-10-20",
    };
    let name = match rung.as_str() {
        "A0" => "clay_drop_e2",
        "A1" => "clay_drop_e2_modal",
        "A2" => "clay_drop_e2_handoff",
        _ => "clay_drop_e2_fixed_step",
    };
    record(
        name,
        block(
            (clay_labels.tolerance, rung.as_str()),
            10.0,
            &|config| Box::new(CorotatedMaterial::from_physical(&clay, config)),
            &clay,
        ),
        40,
        &clay_labels,
    );
}
