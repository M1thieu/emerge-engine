//! Nonlocal Granular Fluidity (NGF) tests and diagnostics for
//! `DruckerPragerMaterial`: the repose-angle undershoot, substep-count
//! sensitivity and long-horizon hold and creep, kept apart from the
//! material's core suite (`sand_tests.rs`).
//!
//! Every NGF figure quoted in this file before the fix of issue #51 (the
//! 0.47x Lajeunesse undershoot and the diagnostics built to explain it) was
//! measured with the reaction term's pressure factor inverted, which made
//! the fluidity grow without bound where the pressure is lowest. With the
//! term as the source writes it, the 200-step Lajeunesse column gives the
//! same runout with NGF on as off (14.56 cells, 0.73x at 1x resolution;
//! 0.787x at 2x), so those diagnostics describe the old term, not the
//! current field.

use super::*;

#[cfg(test)]
mod ngf_verification_tests {
    use super::*;
    use crate::materials::physical_props::Elastic;
    use crate::thermodynamics::{GranularFluidityConfig, GranularFluidityField};
    use crate::{FrictionBoundary, SimConfig, Simulation, SpawnRegion};
    use glam::IVec2;

    const YOUNG_MODULUS_PA: f32 = 15.0e6; // Haeri & Skonieczny 2022 Table 1, Excavation
    const POISSON_RATIO: f32 = 0.3; // cross-checked from their own E/B via K=E/(3(1-2nu))
    const BULK_DENSITY_KG_M3: f32 = 1600.0; // real loose dry sand bulk density
    const CELL_M: f32 = 0.01;

    // The pressure fed to the g-field uses real SI lambda/mu
    // (`lame_from_young` on real Pa values), not grid-scaled ones
    // (`scale_lame`) -- the reaction term's `sqrt(P/rho_s)*d` needs REAL
    // Pascals to combine sensibly with `rho_s` (real kg/m3) and `d` (real
    // meters). `mu_ratio` itself is a dimensionless ratio (q_trial/p_trial),
    // so the same scale factor in numerator and denominator would cancel
    // regardless of unit system -- only the standalone pressure (needed by
    // the reaction term in its own right) actually requires real units, so
    // computing everything in real SI is simpler AND correct, not merely
    // "close enough".
    fn ngf_pressure_and_ratio(p: &Particle) -> (f32, f32) {
        let (lambda, mu) = lame_from_young(YOUNG_MODULUS_PA, POISSON_RATIO);
        let (_, sigma, _) = svd2(p.deformation_gradient);
        let sigma = sigma.abs().max(Vec2::splat(LOG_CLAMP));
        let eps = Vec2::new(
            sigma.x.ln() + p.log_volume_strain * 0.5,
            sigma.y.ln() + p.log_volume_strain * 0.5,
        );
        let trace = eps.x + eps.y;
        let dev = eps - Vec2::splat(trace * 0.5);
        let dev_norm = dev.length();
        // Same formula as `MuIRheologyMaterial::update_particle`
        // (`sand_mui.rs`): p_trial = -(lambda+mu)*trace, q_trial =
        // sqrt(2)*mu*dev_norm (STRESS-space deviator -- the `*mu` converts
        // the strain deviator into stress via the elastic shear modulus;
        // without it the result is a strain-space quantity off by a factor
        // of `mu`, ~3600x too small at this scene's SI-to-grid scaling, and
        // `max_mu` stays pinned at the empty-cell fallback `mu_s`).
        // mu_ratio = q_trial/p_trial, a stress ratio, comparable to `mu_s`
        // on the same footing.
        let p_trial = -(lambda + mu) * trace;
        let mu_ratio = if p_trial > 1.0e-6 {
            std::f32::consts::SQRT_2 * mu * dev_norm / p_trial
        } else {
            0.0
        };
        (p_trial.max(0.0), mu_ratio)
    }

    fn ngf_config() -> GranularFluidityConfig {
        // `d` is 8 mm, not the literal 0.3 mm grain (GRAIN_DIAMETER_M). The
        // fluidity diffusion now carries its `1/dx_meters^2` (as
        // `ThermalConfig::alpha_grid` and the Cosserat field do); without it
        // its reach in meters changed with the cell size (a ~7.7x swing
        // between 1x and 2x resolution). And `cfl_bound` treats every
        // material and diffusion bound as an upper bound, so
        // `with_granular_fluidity` needs no hidden configuration rewrite.
        //
        // An 8-40 mm sweep at both resolutions (200-step Lajeunesse column,
        // `ngf_lajeunesse_runout_resolution_independence`) gave the tightest
        // 1x/2x agreement at 8 mm (ratio_1x = 0.474x, ratio_2x = 0.482x, swing
        // 1.02x; 9 mm next at 1.03x; 2-6 mm swung 1.1-2.6x, the 200-step
        // measurement sitting mid-collapse, see
        // `ngf_lajeunesse_runout_long_duration_creep_check`). `mu_s`, `A`, `b`
        // keep their cited, dimensionless values; `d`'s magnitude is a
        // simulation-scale calibration, not a grain size (like `cohesion`).
        //
        // That sweep, and the ~0.47-0.48x ratio it converged to, were measured
        // with the inverted reaction term of issue #51 (see the module doc).
        // After the fix, NGF on gives the baseline's runout (0.728x at 1x,
        // 0.787x at 2x), so this `d` no longer has the basis it was chosen
        // on; recalibrating it is open work.
        const EFFECTIVE_GRAIN_DIAMETER_M: f32 = 0.008;
        const GRAIN_DENSITY_KG_M3: f32 = 2583.0;
        GranularFluidityConfig {
            mu_s: 0.70, // = tan(35 deg), matches this material's own friction_angle
            grain_diameter_m: EFFECTIVE_GRAIN_DIAMETER_M,
            grain_density_kg_m3: GRAIN_DENSITY_KG_M3,
            nonlocal_amplitude: 0.48,
            b: 0.278,
            t0_s: 1.0e-4, // cited value; the exact reaction integration (see `GranularFluidityField::apply`) needs no recalibration
        }
    }

    /// `resolution_scale=1` matches the original scene (GRID=96,
    /// CELL_M=0.01, 8x16-cell column). `resolution_scale=2` doubles the
    /// grid resolution (half the cell size, double the cell counts)
    /// while keeping the PHYSICAL column size identical -- the same
    /// resolution-independence discipline already used for the earlier
    /// angle-of-repose fix (confirmed at 2x resolution before trusting it).
    fn run_column_collapse(ngf_enabled: bool, resolution_scale: usize, steps: usize) -> (f32, f32) {
        let grid: usize = 96 * resolution_scale;
        let cell_m: f32 = CELL_M / resolution_scale as f32;
        const FLOOR: f32 = 0.05; // meters
        let r0_cells = 4.0_f32 * resolution_scale as f32;
        let h0_cells = 16.0_f32 * resolution_scale as f32;
        let aspect_ratio = h0_cells / r0_cells;
        let predicted_r_inf_cells = r0_cells * (1.0 + 2.0 * aspect_ratio.sqrt());

        let config = SimConfig {
            max_substeps_per_step: 4000,
            ..SimConfig::earth(grid, cell_m, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8 * resolution_scale as i32, 16 * resolution_scale as i32),
            box_center: Vec2::new(
                grid as f32 * 0.5,
                FLOOR / cell_m + 8.0 * resolution_scale as f32,
            ),
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

        solver.step_n(steps);

        let xs: Vec<Vec2> = solver.particles().x.clone();
        let n = xs.len() as f32;
        let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
        let measured_r_inf_cells = xs
            .iter()
            .map(|p| (p.x - center_x).abs())
            .fold(0.0f32, f32::max);
        (measured_r_inf_cells, predicted_r_inf_cells)
    }

    /// Same real scene as `run_column_collapse`, but with `grain_diameter_m`
    /// exposed as a swept parameter instead of pinned at `ngf_config`'s
    /// own 8mm -- temporary sweep helper, not a replacement for
    /// `run_column_collapse`/`ngf_config` (those stay the single source of
    /// truth for the shipped calibration).
    fn run_column_collapse_with_d(
        grain_diameter_m: f32,
        resolution_scale: usize,
        steps: usize,
    ) -> (f32, f32) {
        let grid: usize = 96 * resolution_scale;
        let cell_m: f32 = CELL_M / resolution_scale as f32;
        const FLOOR: f32 = 0.05;
        let r0_cells = 4.0_f32 * resolution_scale as f32;
        let h0_cells = 16.0_f32 * resolution_scale as f32;
        let aspect_ratio = h0_cells / r0_cells;
        let predicted_r_inf_cells = r0_cells * (1.0 + 2.0 * aspect_ratio.sqrt());

        let config = SimConfig {
            max_substeps_per_step: 4000,
            ..SimConfig::earth(grid, cell_m, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8 * resolution_scale as i32, 16 * resolution_scale as i32),
            box_center: Vec2::new(
                grid as f32 * 0.5,
                FLOOR / cell_m + 8.0 * resolution_scale as f32,
            ),
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
        sand.ngf_enabled = true;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let mut cfg = ngf_config();
        cfg.grain_diameter_m = grain_diameter_m;
        let field = GranularFluidityField::new(cfg, ngf_pressure_and_ratio, grid);
        solver = solver.with_granular_fluidity(field);

        solver.step_n(steps);

        let xs: Vec<Vec2> = solver.particles().x.clone();
        let n = xs.len() as f32;
        let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
        let measured_r_inf_cells = xs
            .iter()
            .map(|p| (p.x - center_x).abs())
            .fold(0.0f32, f32::max);
        (measured_r_inf_cells, predicted_r_inf_cells)
    }

    /// Parameter sweep: does any `grain_diameter_m` in a plausible range
    /// close the 0.47x undershoot toward 1.0x without losing resolution
    /// independence (swing near 1.02x, not the old 7.7x)? Every candidate is
    /// checked at both resolutions: 2-6 mm looked tighter at one resolution
    /// but swung 1.1-2.6x across the two.
    #[test]
    // Exploratory: prints a sweep, asserts nothing. Runs 20 column
    // collapses (10 diameters x 2 resolutions, 200 steps each) and does
    // not finish in a bounded time even in release -- this was the last
    // test keeping `cargo test` from ever returning. Kept as the
    // reproducible record of the sweep; run it deliberately with
    // `cargo test --release -- --ignored --nocapture`.
    #[ignore]
    fn ngf_grain_diameter_sweep_accuracy_and_resolution_independence() {
        println!("── NGF GRAIN-DIAMETER SWEEP (accuracy vs resolution-independence) ──");
        for &d_mm in &[4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 16.0, 20.0] {
            let d = d_mm * 1.0e-3;
            let (r1, p1) = run_column_collapse_with_d(d, 1, 200);
            let (r2, p2) = run_column_collapse_with_d(d, 2, 200);
            let ratio1 = r1 / p1;
            let ratio2 = r2 / p2;
            let swing = ratio2.max(ratio1) / ratio1.min(ratio2).max(1e-9);
            println!(
                "  d={d_mm:5.1}mm -> ratio_1x={ratio1:.3}x ratio_2x={ratio2:.3}x swing={swing:.3}x"
            );
        }
    }

    /// Does the Cosserat rolling-resistance coupling (`cosserat_modulus_pa`,
    /// see that field for the citation and the SVD-space adaptation) change
    /// the Lajeunesse collapse this file's NGF diagnostic measures? Same
    /// scene and predicted R_inf, only the coupling toggled.
    fn run_column_collapse_cosserat(
        cosserat_enabled: bool,
        alpha_multiplier: f32,
        steps: usize,
    ) -> (f32, f32, f32) {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
        const R0_CELLS: f32 = 4.0;
        const H0_CELLS: f32 = 16.0;
        let aspect_ratio = H0_CELLS / R0_CELLS;
        let predicted_r_inf_cells = R0_CELLS * (1.0 + 2.0 * aspect_ratio.sqrt());

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
        // No sourced value exists for this coupling modulus at this grid
        // scaling (see `cosserat_modulus_pa`), so it is a multiple of the
        // material's grid-scaled `mu`, dimensionally consistent, with
        // `alpha_multiplier` swept for the regime where the coupling matters.
        // `cosserat_length_scale_m = CELL_M`: the grid resolution, not the
        // sub-mm grain diameter (see that field).
        sand.cosserat_modulus_pa = if cosserat_enabled {
            sand.mu * alpha_multiplier
        } else {
            0.0
        };
        sand.cosserat_length_scale_m = CELL_M;
        let cosserat_modulus_pa = sand.cosserat_modulus_pa;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        if cosserat_enabled {
            let field = crate::thermodynamics::CosseratField::new(
                crate::thermodynamics::CosseratConfig {
                    coupling_modulus_pa: cosserat_modulus_pa,
                    grain_diameter_m: CELL_M,
                    micro_inertia_coefficient: 0.1,
                    rotational_damping_rate_per_s: 0.0,
                },
                GRID,
            );
            solver = solver.with_cosserat_field(field);
        }

        solver.step_n(steps);

        let xs: Vec<Vec2> = solver.particles().x.clone();
        let n = xs.len() as f32;
        let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
        let center_y = xs.iter().map(|p| p.y).sum::<f32>() / n;
        let measured_r_inf_cells = xs
            .iter()
            .map(|p| (p.x - center_x).abs())
            .fold(0.0f32, f32::max);
        (measured_r_inf_cells, predicted_r_inf_cells, center_y)
    }

    /// Does `Simulation::cosserat_curvature()` become nonzero during this
    /// collapse, and how does its magnitude compare to `dev_norm`/
    /// `cohesion_term` in the yield check?
    #[test]
    fn diag_cosserat_curvature_actual_magnitude_during_collapse() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
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
        sand.cosserat_modulus_pa = sand.mu;
        sand.cosserat_length_scale_m = GRAIN_DIAMETER_M;
        let coupling_modulus_pa = sand.cosserat_modulus_pa;
        let mu = sand.mu;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let field = crate::thermodynamics::CosseratField::new(
            crate::thermodynamics::CosseratConfig {
                coupling_modulus_pa,
                grain_diameter_m: GRAIN_DIAMETER_M,
                micro_inertia_coefficient: 0.1,
                rotational_damping_rate_per_s: 0.0,
            },
            GRID,
        );
        solver = solver.with_cosserat_field(field);

        println!(
            "── REAL COSSERAT CURVATURE MAGNITUDE, coupling_modulus_pa={coupling_modulus_pa:.4e} mu={mu:.4e} ──"
        );
        let mut cumulative = 0usize;
        for &checkpoint in &[50usize, 150, 300, 600, 1000] {
            solver.step_n(checkpoint - cumulative);
            cumulative = checkpoint;
            let curv = solver.cosserat_curvature();
            let mut mags: Vec<f32> = curv.iter().map(|k| k.length()).collect();
            mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = mags.len();
            let couple_stress_p90 = if n > 0 {
                let kappa = mags[(n as f32 * 0.9) as usize];
                let m = crate::materials::granular::cosserat::elastic_couple_stress_2d(
                    Vec2::new(kappa, 0.0),
                    coupling_modulus_pa,
                    GRAIN_DIAMETER_M,
                );
                m.length() / (2.0 * mu)
            } else {
                0.0
            };
            println!(
                "  step={checkpoint:5}: |kappa| p50={:.6} p90={:.6} max={:.6}  -> couple_stress_term(p90)={:.6e}",
                mags.get(n / 2).copied().unwrap_or(0.0),
                mags.get((n as f32 * 0.9) as usize).copied().unwrap_or(0.0),
                mags.last().copied().unwrap_or(0.0),
                couple_stress_p90
            );
        }
    }

    /// Isolation: does the same instability (max_speed runaway, the column
    /// collapsing to one y near the friction boundary) reproduce with a huge
    /// `cohesion` instead of Cosserat, no Cosserat code at all, only shear
    /// yield suppressed another way? If so it is an engine bug in the
    /// suppressed-shear-yield + volumetric-floor + friction-boundary
    /// interaction that Cosserat merely triggered first.
    #[test]
    // Diagnostic: runs full sand column collapses to print a measurement.
    // Minutes to hours even in release, which is why it does not belong in
    // the default suite. Kept as the reproducible record; run it with
    // `cargo test --release -- --ignored --nocapture`.
    #[ignore]
    fn diag_high_cohesion_reproduces_same_instability_no_cosserat() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
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
        // A huge cohesion shifts the yield threshold enough to suppress shear
        // yield almost entirely, the effect a high cosserat_modulus_pa had,
        // through `sand.rs`'s existing `cohesion_term` in the same yield check.
        sand.cohesion = sand.mu * 100.0;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

        println!("── HIGH COHESION (100*mu), NO COSSERAT -- ISOLATION TEST ──");
        for step in 1..=20 {
            solver.step_n(1);
            let particles = solver.particles();
            let ys: Vec<f32> = particles.x.iter().map(|p| p.y).collect();
            let y_min = ys.iter().cloned().fold(f32::MAX, f32::min);
            let y_max = ys.iter().cloned().fold(f32::MIN, f32::max);
            let snap = solver.diagnostics_snapshot();
            println!(
                "  step={step:3}: y=[{y_min:.4},{y_max:.4}] max_speed={:.4} min_j={:.4}",
                snap.max_particle_speed, snap.min_deformation_j
            );
        }
    }

    /// The previous test gave bit-for-bit the same result with Cosserat on
    /// and off, which suggests the coupling never engages rather than being
    /// weak. Measures whether local vorticity (macro spin, the antisymmetric
    /// velocity-gradient part the coupling is driven by) is present in this
    /// collapse at all.
    #[test]
    // Diagnostic: runs full sand column collapses to print a measurement.
    // Minutes to hours even in release, which is why it does not belong in
    // the default suite. Kept as the reproducible record; run it with
    // `cargo test --release -- --ignored --nocapture`.
    #[ignore]
    fn diag_macro_spin_magnitude_during_collapse() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
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

        println!("── MACRO SPIN (vorticity) MAGNITUDE DURING REAL COLLAPSE ──");
        let mut cumulative = 0usize;
        for &checkpoint in &[50usize, 150, 300] {
            solver.step_n(checkpoint - cumulative);
            cumulative = checkpoint;
            let particles = solver.particles();
            let mut spins: Vec<f32> = particles
                .velocity_gradient
                .iter()
                .take(particles.len())
                .map(|l| 0.5 * (l.x_axis.y - l.y_axis.x))
                .map(|s: f32| s.abs())
                .collect();
            spins.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = spins.len();
            println!(
                "  after {checkpoint:4} steps: |spin| p50={:.6} p90={:.6} max={:.6}",
                spins[n / 2],
                spins[(n as f32 * 0.9) as usize],
                spins[n - 1]
            );
        }
    }

    /// Step-by-step trace: at alpha = 100*mu all particles collapsed to the same
    /// (x, y) point, a degenerate bug, not the rigidity suppressed yield should
    /// cause. Watches it frame by frame to find where it starts.
    #[test]
    // Diagnostic: runs full sand column collapses to print a measurement.
    // Minutes to hours even in release, which is why it does not belong in
    // the default suite. Kept as the reproducible record; run it with
    // `cargo test --release -- --ignored --nocapture`.
    #[ignore]
    fn diag_cosserat_high_alpha_collapse_trace() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
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
        sand.cosserat_modulus_pa = sand.mu * 100.0;
        sand.cosserat_length_scale_m = CELL_M;
        let cosserat_modulus_pa = sand.cosserat_modulus_pa;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let field = crate::thermodynamics::CosseratField::new(
            crate::thermodynamics::CosseratConfig {
                coupling_modulus_pa: cosserat_modulus_pa,
                grain_diameter_m: CELL_M,
                micro_inertia_coefficient: 0.1,
                rotational_damping_rate_per_s: 0.0,
            },
            GRID,
        );
        solver = solver.with_cosserat_field(field);

        println!("── HIGH-ALPHA COLLAPSE TRACE ──");
        for step in 1..=20 {
            solver.step_n(1);
            let particles = solver.particles();
            let xs: Vec<f32> = particles.x.iter().map(|p| p.x).collect();
            let ys: Vec<f32> = particles.x.iter().map(|p| p.y).collect();
            let x_min = xs.iter().cloned().fold(f32::MAX, f32::min);
            let x_max = xs.iter().cloned().fold(f32::MIN, f32::max);
            let y_min = ys.iter().cloned().fold(f32::MAX, f32::min);
            let y_max = ys.iter().cloned().fold(f32::MIN, f32::max);
            let snap = solver.diagnostics_snapshot();
            println!(
                "  step={step:3}: x=[{x_min:.4},{x_max:.4}] y=[{y_min:.4},{y_max:.4}] max_speed={:.4} min_j={:.4} j_proj={} nonfinite={}",
                snap.max_particle_speed,
                snap.min_deformation_j,
                snap.j_projection_count,
                snap.non_finite_particle_values
            );
        }
    }

    /// `cosserat_lajeunesse_runout_alpha_sweep` gave R = 0.00 exactly from
    /// alpha = 100*mu up, an abrupt jump: are the particles finite (elastic
    /// lockup) or NaN/degenerate (a numerical bug)?
    #[test]
    // Diagnostic: runs full sand column collapses to print a measurement.
    // Minutes to hours even in release, which is why it does not belong in
    // the default suite. Kept as the reproducible record; run it with
    // `cargo test --release -- --ignored --nocapture`.
    #[ignore]
    fn diag_cosserat_high_alpha_health_check() {
        let (_, _, _) = run_column_collapse_cosserat(true, 100.0, 200);
        // Re-run with direct access to check health, since the helper only
        // returns the spread metric.
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
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
        sand.cosserat_modulus_pa = sand.mu * 100.0;
        sand.cosserat_length_scale_m = CELL_M;
        let cosserat_modulus_pa = sand.cosserat_modulus_pa;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let field = crate::thermodynamics::CosseratField::new(
            crate::thermodynamics::CosseratConfig {
                coupling_modulus_pa: cosserat_modulus_pa,
                grain_diameter_m: CELL_M,
                micro_inertia_coefficient: 0.1,
                rotational_damping_rate_per_s: 0.0,
            },
            GRID,
        );
        solver = solver.with_cosserat_field(field);
        solver.step_n(200);

        let particles = solver.particles();
        let all_finite = particles
            .x
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite());
        let snap = solver.diagnostics_snapshot();
        let ys: Vec<f32> = particles.x.iter().map(|p| p.y).collect();
        let y_min = ys.iter().cloned().fold(f32::MAX, f32::min);
        let y_max = ys.iter().cloned().fold(f32::MIN, f32::max);
        println!("── HIGH-ALPHA (100*mu) HEALTH CHECK ──");
        println!(
            "  all_finite={all_finite}  non_finite_count={}  invalid_physical_count={}",
            snap.non_finite_particle_values, snap.invalid_physical_particle_values
        );
        println!("  y range: [{y_min:.4}, {y_max:.4}] (column started spanning ~16 cells tall)");
        println!("  max_speed={:.6}", snap.max_particle_speed);
        assert!(
            all_finite,
            "particles went non-finite at alpha=100*mu -- real numerical bug, not elastic lockup"
        );
    }

    /// Sensitivity sweep: with the length scale at `CELL_M` (see
    /// `cosserat_length_scale_m`), sweeps the coupling modulus across orders
    /// of magnitude relative to `mu`: does any defensible value change the
    /// Lajeunesse collapse noticeably?
    #[test]
    fn cosserat_lajeunesse_runout_alpha_sweep() {
        let (baseline_r, predicted, _) = run_column_collapse_cosserat(false, 1.0, 200);
        println!("── COSSERAT ALPHA SWEEP, real SI throughout, l=CELL_M ──");
        println!("  predicted R_inf (Lajeunesse 2004) = {predicted:.2} cells");
        println!(
            "  baseline (no Cosserat)             = {baseline_r:.2} cells, ratio={:.2}x",
            baseline_r / predicted
        );
        for &alpha_multiplier in &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 1000.0, 10000.0,
        ] {
            let (cosserat_r, predicted2, _) =
                run_column_collapse_cosserat(true, alpha_multiplier, 200);
            assert!((predicted - predicted2).abs() < 1e-6);
            println!(
                "  alpha={alpha_multiplier:6.0}*mu -> R={cosserat_r:.2} cells, ratio={:.2}x",
                cosserat_r / predicted
            );
        }
    }

    /// Does a contact split arrest the collapse where every rate-dependent
    /// mechanism (Cundall damping, KE-peak triggers, Cosserat curvature)
    /// fails because its restraining signal fades to zero at rest?
    /// Bardenhagen multi-field contact (`Particle::contact_group`) resolves
    /// with a geometry-fitted normal (`fit_contact_normal_lr`) and Coulomb
    /// friction, neither of which fades at rest. Splits the collapsing column
    /// into two contact groups (left and right halves) with the existing
    /// mechanism only.
    #[test]
    // Exploratory: prints a measurement, asserts nothing -- a research
    // log kept as a reproducible record, not a regression guard. Runs 30 000 solver steps on a 96-cell sand column
    // in debug, which is hours, so it must not sit in the default suite
    // (this is what made a plain `cargo test` never finish). Run it with
    // `cargo test --release -- --ignored --nocapture` when the question
    // it answers comes up again.
    #[ignore]
    fn contact_group_split_long_horizon_arrest_check() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
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
        let center_x = GRID as f32 * 0.5;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        {
            let particles = solver.particles_mut();
            let n = particles.len();
            for i in 0..n {
                if particles.x[i].x < center_x {
                    particles.contact_group[i] = 1;
                }
            }
        }

        let predicted_r_inf = {
            const R0_CELLS: f32 = 4.0;
            const H0_CELLS: f32 = 16.0;
            let aspect_ratio = H0_CELLS / R0_CELLS;
            R0_CELLS * (1.0 + 2.0 * aspect_ratio.sqrt())
        };
        println!("── CONTACT-GROUP-SPLIT LONG-HORIZON ARREST CHECK (real, zero new code) ──");
        let mut cumulative = 0usize;
        for &steps in &[200usize, 1000, 3000, 10000, 30000] {
            solver.step_n(steps - cumulative);
            cumulative = steps;
            let xs: Vec<Vec2> = solver.particles().x.clone();
            let n = xs.len() as f32;
            let cx = xs.iter().map(|p| p.x).sum::<f32>() / n;
            let cy = xs.iter().map(|p| p.y).sum::<f32>() / n;
            let r_measured = xs.iter().map(|p| (p.x - cx).abs()).fold(0.0f32, f32::max);
            println!(
                "  steps={steps:6}: R={r_measured:.2} ({:.2}x predicted) center=({cx:.2},{cy:.2})",
                r_measured / predicted_r_inf
            );
        }
    }

    /// Does the pile hold longer or steeper over a long horizon, beyond
    /// narrowing the initial spread? Same scene, run far past initial
    /// settling.
    #[test]
    // Exploratory: prints a measurement, asserts nothing -- a research
    // log kept as a reproducible record, not a regression guard. Runs 30 000 solver steps on a 96-cell sand column
    // in debug, which is hours, so it must not sit in the default suite
    // (this is what made a plain `cargo test` never finish). Run it with
    // `cargo test --release -- --ignored --nocapture` when the question
    // it answers comes up again.
    #[ignore]
    fn cosserat_long_horizon_arrest_check() {
        // From `cosserat_lajeunesse_runout_alpha_sweep`'s fine sweep:
        // alpha = 5*mu gave ratio 1.01x (R = 20.27 cells against 20.00), close
        // to Lajeunesse et al. 2004, a modest multiple of the shear modulus.
        const ALPHA_MULTIPLIER: f32 = 5.0;
        println!("── COSSERAT LONG-HORIZON ARREST CHECK, alpha={ALPHA_MULTIPLIER}*mu ──");
        for &steps in &[200usize, 1000, 3000, 10000, 30000] {
            let (baseline_r, predicted, baseline_y) =
                run_column_collapse_cosserat(false, 1.0, steps);
            let (cosserat_r, _, cosserat_y) =
                run_column_collapse_cosserat(true, ALPHA_MULTIPLIER, steps);
            println!(
                "  steps={steps:5}: baseline R={:.2} ({:.2}x) center_y={:.2}   cosserat R={:.2} ({:.2}x) center_y={:.2}",
                baseline_r,
                baseline_r / predicted,
                baseline_y,
                cosserat_r,
                cosserat_r / predicted,
                cosserat_y
            );
        }
    }

    /// Does `g` stay small and narrow through the Lajeunesse collapse (the
    /// "spatial cooperation too slow for the moving front" explanation of the
    /// 0.47x undershoot), or spread quickly and stay (pointing instead at the
    /// translation of the paper's rate-explicit coupling into this
    /// quasi-static return mapping)? Same scene as `run_column_collapse`, SI
    /// throughout, `g_stats()` sampled at checkpoints.
    #[test]
    fn diag_ngf_g_field_trajectory_during_real_collapse() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
        const R0_CELLS: f32 = 4.0;
        const H0_CELLS: f32 = 16.0;

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
        sand.ngf_enabled = true;
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let field = GranularFluidityField::new(ngf_config(), ngf_pressure_and_ratio, GRID);
        solver = solver.with_granular_fluidity(field);

        println!("── NGF g-field trajectory during real Lajeunesse collapse ──");
        // Run this test alone (exact name filter): these are process-wide
        // statics, and another ngf_enabled test running in parallel would
        // pollute the count.
        NGF_CAP_TOTAL_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
        NGF_CAP_BINDING_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
        NGF_CAP_SEVERITY_SUM_X1E6.store(0, std::sync::atomic::Ordering::Relaxed);
        let aspect_ratio = H0_CELLS / R0_CELLS;
        let predicted_r_inf_cells = R0_CELLS * (1.0 + 2.0 * aspect_ratio.sqrt());
        for step in 0..200 {
            solver.step_n(1);
            if step % 10 == 0 || step == 199 {
                let (g_min, g_mean, g_max, g_count) = solver.granular_fluidity().unwrap().g_stats();
                let xs = &solver.particles().x;
                let n = xs.len() as f32;
                let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
                let r_now = xs
                    .iter()
                    .map(|p| (p.x - center_x).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "  step={step:3} r={r_now:6.2} (target={predicted_r_inf_cells:.2}) \
                     g_min={g_min:.4} g_mean={g_mean:.4} g_max={g_max:.4} g_nonzero_cells={g_count}"
                );
            }
        }
        let total = NGF_CAP_TOTAL_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let binding = NGF_CAP_BINDING_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let sev_sum = NGF_CAP_SEVERITY_SUM_X1E6.load(std::sync::atomic::Ordering::Relaxed);
        let bind_fraction = binding as f64 / total.max(1) as f64;
        let avg_severity_when_binding = if binding > 0 {
            (sev_sum as f64 / 1.0e6) / binding as f64
        } else {
            1.0
        };
        println!(
            "  NGF cap: total_yield_checks={total} binding={binding} \
             bind_fraction={bind_fraction:.4} avg_gamma_kept_fraction_when_binding={avg_severity_when_binding:.6}"
        );
    }

    mod long_horizon;
}
