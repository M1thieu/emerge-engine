use super::*;

/// Control: plain DP (ngf_enabled = false, no `GranularFluidityField`)
/// forced to take NGF's substep count (~30 per step, through a tighter
/// `material_cfl_coefficient`). If its runout also drops toward NGF's
/// 0.47x, the undershoot is a substep-count artifact of DP's per-substep
/// pipeline, not an NGF effect. The baseline's mean dt was 0.004251 s at
/// `material_cfl_coefficient = 0.5`, NGF's 0.000167 s (~25.5x smaller),
/// so the coefficient is scaled by the same ~25.5x.
#[test]
// Diagnostic: runs full sand column collapses to print a measurement.
// Minutes to hours even in release, which is why it does not belong in
// the default suite. Kept as the reproducible record; run it with
// `cargo test --release -- --ignored --nocapture`.
#[ignore]
fn diag_forced_small_substeps_baseline_reproduces_ngf_undershoot() {
    const GRID: usize = 96;
    const FLOOR: f32 = 0.05;
    const R0_CELLS: f32 = 4.0;
    const H0_CELLS: f32 = 16.0;
    let aspect_ratio = H0_CELLS / R0_CELLS;
    let predicted_r_inf_cells = R0_CELLS * (1.0 + 2.0 * aspect_ratio.sqrt());

    fn run(material_cfl_coefficient: f32) -> (f32, u64, f32, f32, f32) {
        let config = SimConfig {
            max_substeps_per_step: 4000,
            material_cfl_coefficient,
            ..SimConfig::earth(GRID, CELL_M, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR / CELL_M + 8.0),
            material_id: 0,
            ..SpawnRegion::for_sim(&config)
        };
        let sand = DruckerPragerMaterial::from_physical(
            &GranularProps {
                elastic: Elastic {
                    e_pa: YOUNG_MODULUS_PA,
                    nu: POISSON_RATIO,
                    rho_kg_m3: BULK_DENSITY_KG_M3,
                },
                friction_angle_deg: 35.0,
                dilatancy_angle_deg: 0.0,
            },
            &config,
        );
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let mut total_substeps = 0u64;
        for _ in 0..200 {
            solver.step_n(1);
            total_substeps += solver.diagnostics_snapshot().substeps_last_step as u64;
        }
        let xs = &solver.particles().x;
        let n = xs.len() as f32;
        let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
        let r_inf = xs
            .iter()
            .map(|p| (p.x - center_x).abs())
            .fold(0.0f32, f32::max);
        let qs = &solver.particles().friction_hardening;
        let q_mean = qs.iter().sum::<f32>() / qs.len() as f32;
        let q_max = qs.iter().cloned().fold(0.0f32, f32::max);
        (
            r_inf,
            total_substeps,
            total_substeps as f32 / 200.0,
            q_mean,
            q_max,
        )
    }

    println!("── Forced-small-substep baseline DP, real control experiment ──");
    for &coeff in &[0.5f32, 0.1, 0.0196, 0.01] {
        let (r_inf, total_substeps, avg_substeps, q_mean, q_max) = run(coeff);
        println!(
            "  material_cfl_coefficient={coeff:.4} -> r_inf={r_inf:.2} ratio={:.3}x \
             total_substeps={total_substeps} avg_substeps_per_step={avg_substeps:.1} \
             q_mean={q_mean:.4} q_max={q_max:.4}",
            r_inf / predicted_r_inf_cells
        );
    }
}

/// The rate-limiter cap almost never binds (0.003% of yield checks, see
/// the trajectory diagnostic above), so it does not explain the 0.47x
/// undershoot. Remaining candidate: a `GranularFluidityField` adds its
/// own CFL bound (`granular_fluidity_dt_bound`); if tighter than DP's,
/// NGF runs take smaller substeps throughout. Direct A/B on total substep
/// count for the same collapse.
#[test]
// Diagnostic: runs full sand column collapses to print a measurement.
// Minutes to hours even in release, which is why it does not belong in
// the default suite. Kept as the reproducible record; run it with
// `cargo test --release -- --ignored --nocapture`.
#[ignore]
fn diag_ngf_substep_count_vs_baseline() {
    const GRID: usize = 96;
    const FLOOR: f32 = 0.05;

    fn run(ngf_enabled: bool) -> (u64, f32, f32) {
        let config = SimConfig {
            max_substeps_per_step: 4000,
            ..SimConfig::earth(GRID, CELL_M, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR / CELL_M + 8.0),
            material_id: 0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sand = DruckerPragerMaterial::from_physical(
            &GranularProps {
                elastic: Elastic {
                    e_pa: YOUNG_MODULUS_PA,
                    nu: POISSON_RATIO,
                    rho_kg_m3: BULK_DENSITY_KG_M3,
                },
                friction_angle_deg: 35.0,
                dilatancy_angle_deg: 0.0,
            },
            &config,
        );
        sand.ngf_enabled = ngf_enabled;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        if ngf_enabled {
            let field = GranularFluidityField::new(ngf_config(), ngf_pressure_and_ratio, GRID);
            solver = solver.with_granular_fluidity(field);
        }
        let mut total_substeps = 0u64;
        let mut min_dt = f32::INFINITY;
        let mut sum_dt = 0.0f32;
        let mut total_dropped = 0.0f64;
        for _ in 0..200 {
            solver.step_n(1);
            let snap = solver.diagnostics_snapshot();
            total_substeps += snap.substeps_last_step as u64;
            min_dt = min_dt.min(snap.effective_dt);
            sum_dt += snap.effective_dt;
            total_dropped += snap.sim_time_dropped as f64;
        }
        println!(
            "  [{}] total_dropped_sim_time={total_dropped:.6}s (of {:.2}s intended = {:.2}%)",
            if ngf_enabled {
                "ngf=true "
            } else {
                "ngf=false"
            },
            200.0 * 0.01,
            100.0 * total_dropped / (200.0 * 0.01)
        );
        (total_substeps, min_dt, sum_dt / 200.0)
    }

    let (baseline_substeps, baseline_min_dt, baseline_mean_dt) = run(false);
    let (ngf_substeps, ngf_min_dt, ngf_mean_dt) = run(true);
    println!("── NGF vs baseline substep count, identical real collapse ──");
    println!(
        "  baseline (ngf=false): total_substeps={baseline_substeps} min_dt={baseline_min_dt:.6} mean_dt={baseline_mean_dt:.6}"
    );
    println!(
        "  ngf_enabled=true:     total_substeps={ngf_substeps} min_dt={ngf_min_dt:.6} mean_dt={ngf_mean_dt:.6}"
    );
    println!(
        "  ratio (ngf/baseline substeps) = {:.3}x",
        ngf_substeps as f64 / baseline_substeps.max(1) as f64
    );
}

#[test]
fn ngf_lajeunesse_runout_diagnostic() {
    let (baseline_r, predicted) = run_column_collapse(false, 1, 200);
    let (ngf_r, predicted2) = run_column_collapse(true, 1, 200);
    assert!((predicted - predicted2).abs() < 1e-6);

    println!("── NGF LAJEUNESSE RUNOUT, real SI throughout (diagnostic) ──");
    println!("  predicted R_inf (Lajeunesse 2004)      = {predicted:.2} cells");
    println!(
        "  measured R_inf, cohesionless (fresh baseline) = {baseline_r:.2} cells, ratio={:.2}x",
        baseline_r / predicted
    );
    println!(
        "  measured R_inf, ngf_enabled=true               = {ngf_r:.2} cells, ratio={:.2}x",
        ngf_r / predicted
    );

    assert!(
        baseline_r.is_finite() && ngf_r.is_finite(),
        "non-finite result: baseline_r={baseline_r} ngf_r={ngf_r}"
    );
}

/// Resolution-independence check (Phase 3 of the NGF plan): the same
/// real physical scene, at 2x grid resolution -- a fix must not
/// be a resolution-specific fluke, same discipline already used for
/// the earlier angle-of-repose fix (`sand_preshaped_pile_at_30deg_
/// holds_its_slope`, confirmed at 2x height/2x particle density before
/// being trusted).
#[test]
fn ngf_lajeunesse_runout_resolution_independence() {
    let (ngf_r_1x, predicted_1x) = run_column_collapse(true, 1, 200);
    let (ngf_r_2x, predicted_2x) = run_column_collapse(true, 2, 200);
    let ratio_1x = ngf_r_1x / predicted_1x;
    let ratio_2x = ngf_r_2x / predicted_2x;

    println!("── NGF RESOLUTION-INDEPENDENCE CHECK ──");
    println!("  ratio at 1x resolution (GRID=96)  = {ratio_1x:.3}x");
    println!("  ratio at 2x resolution (GRID=192) = {ratio_2x:.3}x");

    assert!(
        ratio_1x.is_finite() && ratio_2x.is_finite() && ratio_1x > 0.0 && ratio_2x > 0.0,
        "non-finite or zero result: ratio_1x={ratio_1x} ratio_2x={ratio_2x}"
    );
}

/// Does the 200-step measurement above actually reflect a SETTLED pile,
/// or is it a snapshot mid-creep? `ngf_repose_angle_shape_diagnostic`
/// (different scene: same column, no wall-distance headroom, 400 steps)
/// found the pile fully flattened wall-to-wall with near-zero height at
/// 400 steps -- this checks whether the *Lajeunesse* geometry (which
/// gives the pile far more lateral room before it can hit a wall) does
/// the same thing over a longer window, per the plan's own "run far
/// longer than settling takes" requirement (never actually applied to
/// this real-SI scene until now -- the diagnostic/resolution-
/// independence tests above only ever ran 200 steps).
#[test]
// Exploratory: prints a measurement, asserts nothing -- a research
// log kept as a reproducible record, not a regression guard. Runs 3 000 solver steps
// in debug, which is hours, so it must not sit in the default suite
// (this is what made a plain `cargo test` never finish). Run it with
// `cargo test --release -- --ignored --nocapture` when the question
// it answers comes up again.
#[ignore]
fn ngf_lajeunesse_runout_long_duration_creep_check() {
    for &steps in &[200usize, 800, 2000] {
        let (baseline_r, predicted) = run_column_collapse(false, 1, steps);
        let (ngf_r, _) = run_column_collapse(true, 1, steps);
        println!(
            "steps={steps:5}: baseline ratio={:.3}x  ngf ratio={:.3}x  (predicted={predicted:.2} cells)",
            baseline_r / predicted,
            ngf_r / predicted
        );
    }
}

/// Does NGF change whether a pile HOLDS long-term after collapse
/// settles, not just how far it initially spreads? `tests/accuracy.rs`'s
/// own `sand_collapse_relaxation_long_horizon_plateau_check` (arbitrary-
/// unit scene) found baseline DP does NOT: even under the proven
/// "holding" damping recipe (apic_blend=0.05, cundall_damping=1.0), a
/// dynamically-collapsed pile creeps from 29.6deg (t=1500) down to
/// 10.8deg (t=101500), monotonic, never plateaus. NGF exists precisely
/// to give marginal, near-yield flow a length-scale-aware arrest
/// instead of the pointwise Coulomb model's "any nonzero shear ratio
/// can flow forever" behavior -- this is the motivated test of
/// whether it does that specific job, distinct from
/// `ngf_lajeunesse_runout_diagnostic` (which already showed NGF only
/// narrows initial runout ~3%, a different question: how far it gets
/// before settling, not whether "settled" actually holds).
///
/// Real-SI scene required (same reason as every other test in this
/// module). Same column/material as `run_column_collapse`, but adds the
/// proven holding-damping switch (`set_apic_blend`/`set_cundall_
/// damping`, the same real API `tests/accuracy.rs` uses) after the
/// initial collapse settles, then holds for real physical TIME (not an
/// arbitrary step count): 30+ real seconds is already enormously long
/// for a granular pile whose actual dynamic collapse takes well under
/// 1 real second.
#[test]
// Exploratory: prints a measurement, asserts nothing -- a research
// log kept as a reproducible record, not a regression guard. Runs 3 000 solver steps
// in debug, which is hours, so it must not sit in the default suite
// (this is what made a plain `cargo test` never finish). Run it with
// `cargo test --release -- --ignored --nocapture` when the question
// it answers comes up again.
#[ignore]
fn ngf_long_horizon_hold_arrests_creep_vs_baseline() {
    fn run(ngf_enabled: bool) -> Vec<(f32, f32, f32, f32)> {
        let grid: usize = 96;
        let cell_m: f32 = CELL_M;
        const FLOOR_M: f32 = 0.05;
        let config = SimConfig {
            max_substeps_per_step: 4000,
            // apic_blend=1.0 (this config's own base default) is
            // numerically unstable for a violent dynamic
            // collapse, independent of NGF -- 0.6 is the bounded
            // value for the collapse phase itself, distinct from the
            // 0.05 "holding" value applied after settling below.
            apic_blend: 0.6,
            ..SimConfig::earth(grid, cell_m, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(grid as f32 * 0.5, FLOOR_M / cell_m + 8.0),
            material_id: 0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sand = DruckerPragerMaterial::from_physical(
            &GranularProps {
                elastic: Elastic {
                    e_pa: YOUNG_MODULUS_PA,
                    nu: POISSON_RATIO,
                    rho_kg_m3: BULK_DENSITY_KG_M3,
                },
                friction_angle_deg: 35.0,
                dilatancy_angle_deg: 0.0,
            },
            &config,
        );
        sand.ngf_enabled = ngf_enabled;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        if ngf_enabled {
            let field = GranularFluidityField::new(ngf_config(), ngf_pressure_and_ratio, grid);
            solver = solver.with_granular_fluidity(field);
        }

        // Collapse dynamics settle in well under 1 s.
        solver.step_n(200);
        solver.set_apic_blend(0.05);
        solver.set_cundall_damping(1.0);

        let mut results = Vec::new();
        let mut elapsed = 200usize;
        for &extra in &[200usize, 800, 3000] {
            solver.step_n(extra);
            elapsed += extra;
            let xs = &solver.particles().x;
            let vs = &solver.particles().v;
            // p99, not raw max: a single particle flung by the initial
            // violent corner-impact (expected in MPM column
            // collapse) can sit at an outlier position for a long time
            // even once bulk velocity has died down, dragging a raw
            // min/max metric far from the pile's real bulk shape --
            // same outlier-vs-bulk confound already solved once this
            // session (see `sand_collapse_with_phase_gated_relaxation_
            // after_dynamics`'s own percentile check).
            fn p99(mut v: Vec<f32>) -> f32 {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((v.len() as f32 - 1.0) * 0.99).round() as usize;
                v[idx.min(v.len() - 1)]
            }
            fn p01(mut v: Vec<f32>) -> f32 {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((v.len() as f32 - 1.0) * 0.01).round() as usize;
                v[idx.min(v.len() - 1)]
            }
            let ys: Vec<f32> = xs.iter().map(|p| p.y).collect();
            let top_y = p99(ys.clone());
            let bottom_y = p01(ys);
            let n = xs.len() as f32;
            let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
            let half_w = p99(xs.iter().map(|p| (p.x - center_x).abs()).collect());
            let max_speed = vs.iter().map(|v| v.length()).fold(0.0f32, f32::max);
            results.push((elapsed as f32 * 0.01, top_y - bottom_y, half_w, max_speed));
        }
        results
    }

    let baseline = run(false);
    let ngf = run(true);

    println!("── NGF LONG-HORIZON HOLD, real seconds ──");
    for ((t, h, hw, vmax), (_, h2, hw2, vmax2)) in baseline.iter().zip(ngf.iter()) {
        let angle_baseline = (h / hw).atan().to_degrees();
        let angle_ngf = (h2 / hw2).atan().to_degrees();
        println!(
            "t={t:6.2}s  baseline: height={h:.2} half-w={hw:.2} angle={angle_baseline:.1}deg vmax={vmax:.3}   ngf: height={h2:.2} half-w={hw2:.2} angle={angle_ngf:.1}deg vmax={vmax2:.3}"
        );
    }
}

/// Does the static/kinetic Coulomb hysteresis (`static_friction_
/// boost`) actually arrest the long-horizon holding creep, where
/// baseline (this exact scene, see `ngf_long_horizon_hold_arrests_
/// creep_vs_baseline`) does not? This is the structural candidate
/// found by direct inspection of `alpha`'s own math (see
/// `static_friction_boost`'s doc): baseline DP's hardening law
/// asymptotes back to the SAME friction angle for large q regardless of
/// history, giving marginal states zero safety margin. Distinct from
/// NGF (a spatial cooperativity length scale) and from every earlier
/// falsified hypothesis (internal-state reset, packing regularity).
/// `rest_rate_scale`/`static_friction_boost` are both new, uncalibrated
/// knobs -- this sweeps a few real values rather than trusting a single
/// guess.
#[test]
// Exploratory: prints a measurement, asserts nothing -- a research
// log kept as a reproducible record, not a regression guard. Runs 3 000 solver steps
// in debug, which is hours, so it must not sit in the default suite
// (this is what made a plain `cargo test` never finish). Run it with
// `cargo test --release -- --ignored --nocapture` when the question
// it answers comes up again.
#[ignore]
fn static_kinetic_hysteresis_long_horizon_hold_arrests_creep() {
    fn p99(mut v: Vec<f32>) -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((v.len() as f32 - 1.0) * 0.99).round() as usize;
        v[idx.min(v.len() - 1)]
    }
    fn p01(mut v: Vec<f32>) -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((v.len() as f32 - 1.0) * 0.01).round() as usize;
        v[idx.min(v.len() - 1)]
    }

    fn run(static_boost_deg: f32, rest_rate_scale: f32) -> Vec<(f32, f32, f32, f32)> {
        let grid: usize = 96;
        let cell_m: f32 = CELL_M;
        const FLOOR_M: f32 = 0.05;
        let config = SimConfig {
            max_substeps_per_step: 4000,
            apic_blend: 0.6,
            ..SimConfig::earth(grid, cell_m, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(grid as f32 * 0.5, FLOOR_M / cell_m + 8.0),
            material_id: 0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sand = DruckerPragerMaterial::from_physical(
            &GranularProps {
                elastic: Elastic {
                    e_pa: YOUNG_MODULUS_PA,
                    nu: POISSON_RATIO,
                    rho_kg_m3: BULK_DENSITY_KG_M3,
                },
                friction_angle_deg: 35.0,
                dilatancy_angle_deg: 0.0,
            },
            &config,
        );
        sand.static_friction_boost = static_boost_deg.to_radians();
        sand.rest_rate_scale = rest_rate_scale;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

        solver.step_n(200);
        solver.set_apic_blend(0.05);
        solver.set_cundall_damping(1.0);

        let mut results = Vec::new();
        let mut elapsed = 200usize;
        for &extra in &[200usize, 800, 3000] {
            solver.step_n(extra);
            elapsed += extra;
            let xs = &solver.particles().x;
            let ys: Vec<f32> = xs.iter().map(|p| p.y).collect();
            let top_y = p99(ys.clone());
            let bottom_y = p01(ys);
            let n = xs.len() as f32;
            let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
            let half_w = p99(xs.iter().map(|p| (p.x - center_x).abs()).collect());
            let height = top_y - bottom_y;
            let angle = (height / half_w).atan().to_degrees();
            results.push((elapsed as f32 * 0.01, height, half_w, angle));
        }
        results
    }

    let baseline = run(0.0, 1.0);
    let boost5 = run(5.0, 0.05);
    let boost10 = run(10.0, 0.05);
    let boost20 = run(20.0, 0.05);

    println!("── STATIC/KINETIC HYSTERESIS LONG-HORIZON HOLD, real seconds ──");
    for i in 0..baseline.len() {
        println!(
            "t={:6.2}s  base: h={:.2} hw={:.2} a={:.1}deg | +5deg: h={:.2} hw={:.2} a={:.1}deg | +10deg: h={:.2} hw={:.2} a={:.1}deg | +20deg: h={:.2} hw={:.2} a={:.1}deg",
            baseline[i].0,
            baseline[i].1,
            baseline[i].2,
            baseline[i].3,
            boost5[i].1,
            boost5[i].2,
            boost5[i].3,
            boost10[i].1,
            boost10[i].2,
            boost10[i].3,
            boost20[i].1,
            boost20[i].2,
            boost20[i].3,
        );
    }
}
