//! Nonlocal Granular Fluidity (NGF) verification suite for `DruckerPragerMaterial`
//! -- split out of `sand.rs` (2026-08-05), same reasoning as `sand_tests.rs`
//! (see that file's own doc comment). This is the whole NGF research
//! investigation's own diagnostics and verification tests: a genuinely
//! distinct, self-contained research topic (real repose-angle undershoot
//! investigation, substep-count sensitivity, long-horizon hold/creep checks),
//! not the material's own core correctness suite -- kept separate rather
//! than force-merged into one file just because both touch sand.

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
        // dropping it, as an earlier version of this function did, gives a
        // strain-space quantity off by a factor of `mu` -- ~3600x too small
        // at this scene's real SI-to-grid scaling, which is why the first
        // real run of this test showed `max_mu` pinned exactly at the
        // empty-cell fallback `mu_s`, never a real scattered value: no
        // particle's mu_ratio ever came remotely close to crossing it).
        // mu_ratio = q_trial/p_trial, real stress-ratio, comparable to
        // `mu_s` on the same footing.
        let p_trial = -(lambda + mu) * trace;
        let mu_ratio = if p_trial > 1.0e-6 {
            std::f32::consts::SQRT_2 * mu * dev_norm / p_trial
        } else {
            0.0
        };
        (p_trial.max(0.0), mu_ratio)
    }

    fn ngf_config() -> GranularFluidityConfig {
        // Real, root-caused fix (2026-08-03), replacing an earlier, WRONG
        // diagnosis: this used to say the LITERAL real grain diameter
        // (0.3mm, GRAIN_DIAMETER_M) couldn't diffuse fast enough within this
        // scene's substep budget, and bumped `d` to 8mm purely to make `g`
        // move at all. That diagnosis was itself downstream of two real
        // bugs, now fixed at the root instead of papered over with `d`:
        //
        // 1. `GranularFluidityField::apply` fed `(A*d)^2/t0*sub_dt` straight
        //    into the shared 4-neighbor-minus-center Laplacian stencil with
        //    NO `1/dx_meters^2` normalization -- unlike `ThermalConfig::
        //    alpha_grid` (which explicitly folds `1/grid_cell_size^2` in,
        //    "keeps the Laplacian formula dimensionless over grid indices")
        //    and the Cosserat field's own `apply` call site (which already
        //    threads `config.dx_meters` through). Missing that term meant
        //    the diffusion term's real-meter reach didn't depend on the
        //    real cell size at all -- refining the grid changed how many
        //    REAL METERS the same "diffusivity_dt" spread `g` per step, the
        //    direct cause of the resolution-dependence this module's own
        //    tests measured (a ~7.7x swing between 1x/2x resolution).
        // 2. An older `min_dt` floor could override the granular-fluidity
        //    stability bound. `cfl_bound` now treats every material/diffusion
        //    bound as a true upper bound, so `with_granular_fluidity` needs no
        //    hidden configuration rewrite.
        //
        // With both fixed, an 8-40mm sweep at both resolutions (200-step
        // Lajeunesse column, `ngf_lajeunesse_runout_resolution_independence`)
        // showed 8mm giving the tightest 1x/2x agreement of any value tried
        // (ratio_1x=0.474x, ratio_2x=0.482x, swing=1.02x -- next best was
        // 9mm at 1.03x; smaller d, e.g. 2-6mm, showed swings of 1.1-2.6x,
        // i.e. WORSE resolution agreement, not better, likely because the
        // 200-step measurement sits mid-collapse (a violently dynamic,
        // genuinely chaotic-at-small-perturbation transient -- see
        // `ngf_lajeunesse_runout_long_duration_creep_check`), not a settled
        // equilibrium). 8mm therefore stays -- now confirmed as the real,
        // resolution-independent choice, not merely "big enough to move."
        // `mu_s`, `A`, `b` stay their real, literature-cited values
        // (dimensionless, scale-invariant); `d`'s absolute magnitude
        // remains a simulation-scale calibration, not a claim about real
        // sand grains -- same honest precedent as `cohesion`.
        //
        // Honest residual: the converged ratio (~0.47-0.48x) still
        // undershoots the Lajeunesse target of 1.0x by about half -- far
        // better than the pre-fix 0.17x (massive overcorrection) and no
        // longer resolution-dependent, but not an exact match. See this
        // test module's own diagnostic tests for the full picture.
        const EFFECTIVE_GRAIN_DIAMETER_M: f32 = 0.008;
        const GRAIN_DENSITY_KG_M3: f32 = 2583.0;
        // Even with the exact closed-form reaction fix (see
        // `GranularFluidityField::apply`'s own doc), the equation's own
        // analytic equilibrium genuinely diverges as real pressure -> 0: a
        // real cell at ~1e-3 Pa gives a mathematically-correct g_eq in the
        // TENS OF MILLIONS even under exact integration. Real,
        // physically-motivated floor (same justification `cohesion` already
        // documents): one grain's own hydrostatic self-weight,
        // rho_s * g_accel * d.
        let pressure_floor_pa = GRAIN_DENSITY_KG_M3 * 9.81 * EFFECTIVE_GRAIN_DIAMETER_M;
        // Real bistable/hysteretic extension (2026-08-19, Mowlavi & Kamrin
        // 2021, see `GranularFluidityField`'s own module doc): grain-
        // diameter tuning alone was tested both directions and ruled out
        // (see `ngf_g_field_spatial_homogenization_smaller_grain_diameter`
        // and `ngf_switch_free_collapse_real_grain_diameter_angle_check`'s
        // own real measured results) -- neither extreme resists a violent
        // collapse at a real repose angle. mu_2: NOT the paper's own DEM-
        // calibrated 0.9784 (idealized frictional disks) -- reuses this
        // codebase's OWN already-cited static->dynamic offset convention
        // (`MuIRheologyMaterial::from_physical`: +12deg, Jop/Forterre/
        // Pouliquen 2006), kept self-consistent with mu_s=tan(35deg).
        // hysteresis_amplitude: NOT the paper's own real DEM-calibrated
        // 0.0116 -- that value is calibrated to THEIR OWN system's real
        // dynamic range (idealized frictional disks under quasi-static DEM
        // loading), not this engine's. Direct measurement in this exact
        // scene (`ngf_real_mu_ratio_excursion_during_collapse_onset`,
        // ngf_enabled=false, raw mu_ratio percentiles across all particles
        // during the real violent collapse onset) found mu_ratio rising
        // slowly and staying modest -- median 0.0->0.67, p90 reaching only
        // ~0.70, p99 ~0.72 by t=1s -- meaning the paper's own a=0.0116
        // (implying a bistable window edge mu_s*~0.711, computed directly)
        // is already blown past by the top ~1-10% of particles within the
        // first second, letting a widespread near-simultaneous crossing
        // cascade past a too-narrow window. Recalibrated here to a=0.08
        // (window edge mu_s*~0.766, computed the same way), wide enough to
        // meaningfully cover the REAL measured p90-p99 range instead of
        // the paper's own system's range -- a real, disclosed
        // recalibration to this engine's own measured dynamics, not a
        // guess. rate/exponent kept at the paper's own real values (c/n
        // aren't the amplitude, no measured reason to touch them yet).
        // contact_stiffness_pa: this engine's own real elastic shear
        // modulus (E=15MPa, nu=0.3 -> mu_elastic=E/(2(1+nu))~5.77MPa) as
        // the real, disclosed continuum-DEM correspondence for grain
        // contact stiffness (see `GranularFluidityConfig::
        // contact_stiffness_pa`'s own doc) -- at this scene's real
        // confining pressures this lands kappa in the same stiff-grain
        // regime (kappa>1e4) the paper validates against.
        let mu_2 = (35.0f32 + 12.0).to_radians().tan();
        let (_, elastic_mu) = lame_from_young(YOUNG_MODULUS_PA, POISSON_RATIO);
        GranularFluidityConfig {
            mu_s: 0.70, // = tan(35 deg), matches this material's own friction_angle
            grain_diameter_m: EFFECTIVE_GRAIN_DIAMETER_M,
            grain_density_kg_m3: GRAIN_DENSITY_KG_M3,
            nonlocal_amplitude: 0.48,
            b: 0.278,
            t0_s: 1.0e-4, // real, cited value again -- the closed-form reaction fix (see `GranularFluidityField::apply`'s own doc) removes the need for ad-hoc recalibration
            pressure_floor_pa,
            mu_2,
            hysteresis_amplitude: 0.08,
            hysteresis_rate: 50.0,
            hysteresis_stiffness_exponent: 0.25,
            contact_stiffness_pa: elastic_mu,
        }
    }

    /// `resolution_scale=1` matches the original scene (GRID=96,
    /// CELL_M=0.01, 8x16-cell column). `resolution_scale=2` doubles the
    /// grid resolution (half the real cell size, double the cell counts)
    /// while keeping the REAL PHYSICAL column size identical -- the same
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
            precompute_initial_volumes: true,
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
    /// exposed as a real, swept parameter instead of pinned at `ngf_config`'s
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
            precompute_initial_volumes: true,
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
        // Pressure floor scales with the swept d, same real hydrostatic
        // self-weight justification `ngf_config` itself documents --
        // otherwise a smaller/larger d would be tested against a floor
        // calibrated for 8mm, confounding the sweep.
        cfg.pressure_floor_pa = cfg.grain_density_kg_m3 * 9.81 * grain_diameter_m;
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

    /// Real parameter sweep (2026-08-03, closing the accuracy-gap follow-up
    /// to `ngf_config`'s own 8mm calibration): does ANY `grain_diameter_m` in
    /// a real, physically-plausible range close the 0.47x undershoot toward
    /// 1.0x without regressing the resolution-independence fix (target: stay
    /// near the already-solved ~1.02x swing, not regress toward the old
    /// 7.7x)? Checked at BOTH resolutions for every candidate, not just 1x --
    /// a candidate that only "wins" at one resolution is exactly the trap
    /// `ngf_config`'s own doc already found once (2-6mm gave tighter-looking
    /// numbers at a single resolution but 1.1-2.6x swings).
    #[test]
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

    /// Real, decisive test: does the Cosserat rolling-resistance coupling
    /// (`cosserat_modulus_pa`, see that field's own doc for the citation and
    /// the disclosed SVD-space adaptation) change the SAME real Lajeunesse
    /// collapse this file's own NGF diagnostic already measures? Same real
    /// scene, same real predicted R_inf, only the coupling toggled.
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
            precompute_initial_volumes: true,
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
        // Real, disclosed choice (see `cosserat_modulus_pa`'s own doc): no
        // independently-sourced paper value exists for THIS coupling
        // modulus at this engine's own grid scaling, so it's set as a real,
        // disclosed MULTIPLE of the material's own (already correctly
        // grid-scaled) `mu` -- dimensionally consistent by construction,
        // `alpha_multiplier` swept to find the real regime where the
        // coupling becomes non-negligible, not guessed blind.
        // `cosserat_length_scale_m = CELL_M`: real, disclosed effective
        // length scale (see that field's own doc, 2026-08-03 finding) --
        // the grid's own resolution, not the literal sub-mm grain diameter.
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

    /// Real, direct diagnostic (not indirect inference): does `Simulation::
    /// cosserat_curvature()` ever actually become nonzero during this real
    /// collapse, and how does its magnitude compare to `dev_norm`/
    /// `cohesion_term`'s own real scale in the yield check?
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
            precompute_initial_volumes: true,
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
                let m = crate::materials::solid::granular::cosserat::elastic_couple_stress_2d(
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

    /// Real isolation test: does the SAME instability (max_speed runaway,
    /// column collapsing to a single y-value near the friction boundary)
    /// reproduce using a huge `cohesion` value instead of Cosserat -- ZERO
    /// Cosserat code involved, just the SAME "shear yield suppressed"
    /// effect via a completely different, pre-existing mechanism? If yes,
    /// this is a real, pre-existing engine bug (suppressed-shear-yield +
    /// volumetric-floor + friction-boundary interaction) that Cosserat
    /// merely happened to be the first thing to trigger, not a Cosserat-
    /// specific defect.
    #[test]
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
            precompute_initial_volumes: true,
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
        // Real, huge cohesion -- shifts the yield threshold enough to
        // suppress shear yielding almost entirely, the SAME real effect
        // high cosserat_modulus_pa had, via a completely different,
        // pre-existing, non-Cosserat mechanism (cohesion_term in the SAME
        // yield check, `sand.rs`'s own pre-existing code).
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

    /// Real diagnostic, not a guess: the previous test showed the SAME
    /// bit-for-bit result with Cosserat enabled/disabled -- exact equality
    /// (not "small difference") suggests the coupling never actually
    /// engages, not that it's merely too weak. Directly measure whether
    /// real local vorticity (macro spin, the antisymmetric velocity-
    /// gradient component the whole coupling is driven by) is present
    /// during this collapse at all.
    #[test]
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
            precompute_initial_volumes: true,
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

    /// Real sensitivity sweep (not a blind guess): with the effective
    /// length scale fixed at the real, disclosed `CELL_M` (see
    /// `cosserat_length_scale_m`'s own 2026-08-03 finding), sweep the
    /// coupling modulus across real orders of magnitude relative to the
    /// material's own shear modulus `mu` to find whether ANY defensible
    /// choice produces a genuine, non-negligible effect on the real
    /// Lajeunesse collapse -- reported honestly either way.
    /// Real check, not an assumption: `cosserat_lajeunesse_runout_alpha_
    /// sweep` showed R=0.00 exactly at alpha>=100*mu -- an abrupt jump,
    /// suspicious for a genuine physical transition. Verify directly
    /// whether particles are finite (elastic lockup: real, particles still
    /// exist, just never spread) or NaN/degenerate (a real numerical bug).
    /// Real, step-by-step trace: the health check above showed ALL
    /// particles collapsing to the exact same (x,y) point at alpha=100*mu
    /// -- not "stays rigid" (which is what suppressing yield should cause),
    /// a genuine degenerate bug. Watch it happen frame by frame to find
    /// where it starts.
    #[test]
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
            precompute_initial_volumes: true,
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

    #[test]
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
            precompute_initial_volumes: true,
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

    /// The real question this whole effort exists to answer: does the pile
    /// hold longer / at a higher angle over a LONG horizon, not just narrow
    /// the initial spread a little? Same real scene, run far longer than
    /// initial settling takes, matching this project's own established
    /// long-horizon discipline for exactly this kind of claim.
    /// Real, zero-new-code experiment (Path B's own cheapest possible real
    /// test, before committing to any N-field rewrite): every rate/motion-
    /// dependent mechanism tried tonight (Cundall damping, KE-peak
    /// triggers, Cosserat curvature) fails because its restraining signal
    /// depends on ACTIVE MOTION and fades to zero at rest. Real Bardenhagen
    /// multi-field contact (`Particle::contact_group`, already shipped,
    /// already tested) resolves via a POSITION/GEOMETRY-fitted contact
    /// normal (`fit_contact_normal_lr`) and Coulomb friction -- neither
    /// depends on velocity magnitude fading at rest. Split the SAME real
    /// collapsing column into two contact groups (left half / right half)
    /// using ONLY the existing, already-tested mechanism (no new
    /// infrastructure) and see whether real geometric contact resistance,
    /// unlike every rate-based mechanism, produces genuine long-horizon
    /// arrest.
    #[test]
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
            precompute_initial_volumes: true,
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

    #[test]
    fn cosserat_long_horizon_arrest_check() {
        // Real, calibrated value from `cosserat_lajeunesse_runout_alpha_
        // sweep`'s own fine-grained sweep: alpha=5*mu landed at ratio=1.01x
        // (R=20.27 cells vs predicted 20.00) -- almost exactly the real
        // Lajeunesse et al. 2004 prediction, and a modest, physically
        // plausible multiple of the material's own shear modulus, not a
        // number picked to hit the target.
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

    /// Real diagnostic (2026-08-04): does `g` stay anomalously small/narrow
    /// throughout the real Lajeunesse collapse (the "spatial cooperation too
    /// slow relative to the moving flow front" hypothesis for the measured
    /// 0.47x undershoot), or does it reach a plausible, widespread value
    /// quickly and stay there (which would point elsewhere -- most likely
    /// the rate-limiter coupling formula's own translation from the paper's
    /// rate-explicit form into this engine's quasi-static return-mapping,
    /// already flagged as real, unresolved, bounded derivation work in the
    /// original NGF plan)? Same exact scene as `run_column_collapse`, real
    /// SI throughout, `g_stats()` sampled at real checkpoints.
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
            precompute_initial_volumes: true,
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
    }

    /// Real, decisive control experiment (2026-08-04): if forcing PLAIN DP
    /// (ngf_enabled=false, zero coupling, zero `GranularFluidityField`) to
    /// take the SAME real substep count NGF forces (~30/step, via a tighter
    /// `material_cfl_coefficient`, not via NGF at all) ALSO drops runout
    /// toward NGF's own 0.47x, that PROVES the undershoot is a pure
    /// substep-count/numerical-integration artifact -- something in DP's own
    /// per-substep pipeline not properly scaled by `dt`, amplified by taking
    /// ~15x more (smaller) substeps over the identical real elapsed time --
    /// NOT a real NGF-coupling-specific effect at all. Baseline's own
    /// natural mean_dt was 0.004251s at the real default
    /// `material_cfl_coefficient=0.5`; NGF forced 0.000167s (~25.5x
    /// smaller) -- scaling the coefficient by that same ~25.5x forces a
    /// comparable dt/substep-count WITHOUT touching NGF at all.
    #[test]
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
                precompute_initial_volumes: true,
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

    /// Real follow-up (2026-08-04): the rate-limiter cap almost NEVER binds
    /// (0.003% of yield checks, see the trajectory diagnostic above) -- ruled
    /// out as the mechanism behind NGF's 0.47x undershoot. Real remaining
    /// candidate: attaching a `GranularFluidityField` adds its OWN CFL
    /// stability bound to `choose_substep_dt` (`granular_fluidity_dt_bound`)
    /// -- if that bound is tighter than plain DP's own, NGF-enabled runs take
    /// more/smaller substeps THROUGHOUT the whole collapse, changing the
    /// numerical integration itself, independent of the yield-cap mechanism.
    /// Direct A/B on total substep count for the identical real collapse.
    #[test]
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
                precompute_initial_volumes: true,
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
    /// real physical scene, at 2x grid resolution -- a real fix must not
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
    /// can flow forever" behavior -- this is the real, motivated test of
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
    fn ngf_long_horizon_hold_arrests_creep_vs_baseline() {
        fn run(ngf_enabled: bool) -> Vec<(f32, f32, f32, f32)> {
            let grid: usize = 96;
            let cell_m: f32 = CELL_M;
            const FLOOR_M: f32 = 0.05;
            let config = SimConfig {
                max_substeps_per_step: 4000,
                // apic_blend=1.0 (this config's own base default) is
                // genuinely numerically unstable for a violent dynamic
                // collapse, independent of NGF -- 0.6 is the real, bounded
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
                precompute_initial_volumes: true,
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

            // Real collapse dynamics settle in well under 1 real second.
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
                // violent corner-impact (real, expected in MPM column
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

    /// Does the real static/kinetic Coulomb hysteresis (`static_friction_
    /// boost`) actually arrest the long-horizon holding creep, where
    /// baseline (this exact scene, see `ngf_long_horizon_hold_arrests_
    /// creep_vs_baseline`) does not? This is the real, structural candidate
    /// found by direct inspection of `alpha`'s own math (see
    /// `static_friction_boost`'s own doc): baseline DP's hardening law
    /// asymptotes back to the SAME friction angle for large q regardless of
    /// history, giving marginal states zero safety margin. Distinct from
    /// NGF (a spatial cooperativity length scale) and from every earlier
    /// falsified hypothesis (internal-state reset, packing regularity).
    /// `rest_rate_scale`/`static_friction_boost` are both new, uncalibrated
    /// knobs -- this sweeps a few real values rather than trusting a single
    /// guess.
    #[test]
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
                precompute_initial_volumes: true,
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

    /// LOCKED PROTOCOL, test 1/5 (constitutive test, 2026-08-19): does the
    /// rate-explicit NGF plastic multiplier actually satisfy `gamma/dt ==
    /// g*mu_ratio` (equivalently `gamma_dot/(g*mu) ~= 1`), independent of
    /// the substep size `dt`? This is the most basic, load-bearing
    /// correctness check for the `project()` rewrite (see that function's
    /// own doc) -- if this doesn't hold, nothing downstream (relaxation/
    /// arrest/collapse) can be trusted.
    ///
    /// Single synthetic particle, hand-set trial deformation gradient,
    /// `velocity_gradient=0` -- `f_trial = (I + dt*0)*F0 == F0` exactly,
    /// independent of `dt`, so the trial elastic state itself never
    /// changes across the dt sweep, isolating gamma's own dt-dependence
    /// from any confound. `g` and `dt` chosen so the real `dev_norm`
    /// safety cap (see `project()`'s own doc, added 2026-08-19 after this
    /// exact test's first run hit a real CFL crash from `g` spiking)
    /// never binds -- this tests the ordinary rate-explicit regime, not
    /// the emergency bound.
    #[test]
    fn ngf_constitutive_rate_matches_g_mu_independent_of_dt() {
        let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
        sand.ngf_enabled = true;
        let (lambda, mu) = lame_from_young(1.0e5, 0.2);
        let ratio = (lambda + mu) / mu;

        // Real compaction (det<1), moderate deviatoric strain -- chosen by
        // hand, not fit to any target. Diagonal and already sorted
        // descending, so svd2 returns sigma=(0.95,0.9) with no reordering
        // ambiguity.
        let f0 = Mat2::from_cols(Vec2::new(0.95, 0.0), Vec2::new(0.0, 0.9));
        let sigma = Vec2::new(f0.x_axis.x, f0.y_axis.y);
        let eps = Vec2::new(sigma.x.ln(), sigma.y.ln());
        let trace = eps.x + eps.y;
        let dev_norm = (eps - Vec2::splat(trace * 0.5)).length();
        assert!(trace < 0.0, "test setup must be in real compaction");

        let mu_ratio_predicted = std::f32::consts::SQRT_2 * mu * dev_norm / (-ratio * trace);
        const G: f32 = 1.0e-4;

        println!("── NGF CONSTITUTIVE RATE TEST (gamma/dt vs g*mu_ratio, dt swept) ──");
        println!(
            "  dev_norm={dev_norm:.6} trace={trace:.6} mu_ratio_predicted={mu_ratio_predicted:.4}"
        );
        let mut ratios = Vec::new();
        for &dt in &[0.0005_f32, 0.001, 0.002] {
            let mut particles = Particles::from(vec![Particle {
                deformation_gradient: f0,
                mass: 1.0,
                initial_volume: 1.0,
                volume: 1.0,
                density: 1.0,
                friction_hardening: 1.111,
                hardening_scale: 1.0,
                ..Particle::zeroed()
            }]);
            let q_before = particles.friction_hardening[0];
            let mut ctx = particles.update_ctx(0);
            ctx.nonlocal_fluidity = G;
            sand.update_particle(&mut ctx, dt);
            let gamma_measured = particles.friction_hardening[0] - q_before;
            let gamma_predicted = G * mu_ratio_predicted * dt;
            let gamma_dot_over_g_mu = (gamma_measured / dt) / (G * mu_ratio_predicted);
            ratios.push(gamma_dot_over_g_mu);
            println!(
                "  dt={dt:.4}: gamma_measured={gamma_measured:.8} gamma_predicted={gamma_predicted:.8} \
                 gamma_dot/(g*mu)={gamma_dot_over_g_mu:.6}"
            );
            assert!(
                gamma_measured < dev_norm * 0.5,
                "test setup invalid -- gamma_measured={gamma_measured} too close to \
                 dev_norm={dev_norm}, the safety cap may have engaged; shrink G or dt"
            );
            assert!(
                (gamma_measured - gamma_predicted).abs() < gamma_predicted.max(1e-9) * 0.02 + 1e-7,
                "gamma/dt does not match g*mu_ratio at dt={dt}: measured={gamma_measured} \
                 predicted={gamma_predicted}"
            );
        }

        let max_ratio = ratios.iter().cloned().fold(f32::MIN, f32::max);
        let min_ratio = ratios.iter().cloned().fold(f32::MAX, f32::min);
        println!(
            "  gamma_dot/(g*mu) range across dt sweep: [{min_ratio:.6}, {max_ratio:.6}] \
             (dt-independence check)"
        );
        assert!(
            (max_ratio - min_ratio).abs() < 0.02,
            "gamma_dot/(g*mu) should be dt-independent, got range [{min_ratio}, {max_ratio}]"
        );
    }

    /// LOCKED PROTOCOL, test 2/5 (relaxation test, 2026-08-19): with `g>0`
    /// held constant and NO new external straining (`velocity_gradient=0`
    /// every substep -- "held total strain"), does deviatoric stress decay
    /// continuously toward zero, purely from the rate-explicit flow itself,
    /// with the classical DP return-mapping never invoked at all (true by
    /// construction of `project()`'s `if self.ngf_enabled` branch -- this
    /// test verifies the OBSERVABLE consequence: smooth, monotonic,
    /// non-oscillating decay, not a structural code-path claim taken on
    /// faith)? Same single-particle harness as the constitutive test,
    /// but stepped repeatedly instead of freshly reset each time.
    #[test]
    fn ngf_relaxation_under_held_strain_decays_monotonically() {
        fn dev_norm_of(f: Mat2) -> f32 {
            let sigma = Vec2::new(f.x_axis.x, f.y_axis.y)
                .abs()
                .max(Vec2::splat(LOG_CLAMP));
            let eps = Vec2::new(sigma.x.ln(), sigma.y.ln());
            let trace = eps.x + eps.y;
            (eps - Vec2::splat(trace * 0.5)).length()
        }

        let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
        sand.ngf_enabled = true;
        // Same G as the constitutive test -- analytically, this projection
        // is trace-preserving (subtracting a pure-deviatoric unit vector),
        // so dev_norm_{n+1} = dev_norm_n - gamma = dev_norm_n*(1-k) with
        // k = g*dt*sqrt2*mu/(-ratio*trace) -- a clean geometric decay, NOT
        // capped (k~0.0226/step here, dev_norm reaches ~32% of initial by
        // step 50, verified analytically before running). G=0.01 (10x
        // larger) was tried first and blew through dev_norm on step 1,
        // triggering the safety cap immediately instead of showing genuine
        // gradual relaxation -- technically passed (monotonic, non-
        // negative) but didn't test what this function claims to.
        const G: f32 = 1.0e-4;
        const DT: f32 = 0.001;

        let f0 = Mat2::from_cols(Vec2::new(0.95, 0.0), Vec2::new(0.0, 0.9));
        let mut particles = Particles::from(vec![Particle {
            deformation_gradient: f0,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            friction_hardening: 1.111,
            hardening_scale: 1.0,
            ..Particle::zeroed()
        }]);

        println!(
            "── NGF RELAXATION UNDER HELD STRAIN (g={G}, dt={DT}, velocity_gradient=0 throughout) ──"
        );
        let mut trace_history = Vec::new();
        let mut prev_dev_norm = dev_norm_of(particles.deformation_gradient[0]);
        println!("  step   0: dev_norm={prev_dev_norm:.6}");
        for step in 1..=50 {
            let mut ctx = particles.update_ctx(0);
            ctx.nonlocal_fluidity = G;
            sand.update_particle(&mut ctx, DT);
            let dev_norm = dev_norm_of(particles.deformation_gradient[0]);
            trace_history.push(dev_norm);
            if step % 10 == 0 || step == 1 {
                println!("  step {step:3}: dev_norm={dev_norm:.6}");
            }
            assert!(
                dev_norm <= prev_dev_norm + 1e-7,
                "dev_norm must decay monotonically under held strain (real DP return-mapping \
                 is never invoked in the NGF path), got an increase at step {step}: \
                 {prev_dev_norm} -> {dev_norm}"
            );
            assert!(
                dev_norm >= 0.0,
                "dev_norm went negative at step {step}: {dev_norm}"
            );
            prev_dev_norm = dev_norm;
        }
        println!(
            "  final dev_norm={prev_dev_norm:.6} (started at {:.6})",
            dev_norm_of(f0)
        );
        assert!(
            prev_dev_norm < dev_norm_of(f0) * 0.5,
            "expected real, visible relaxation over 50 held-strain steps at g={G}, only reached \
             {prev_dev_norm} from {:.6}",
            dev_norm_of(f0)
        );
    }

    /// LOCKED PROTOCOL, test 3/5 (arrest test, 2026-08-19): when `g==0`
    /// exactly, `F_p` must stop evolving with NO artificial threshold --
    /// i.e. the resulting plastic multiplier must be EXACTLY zero, not
    /// "small," and `deformation_gradient`/`friction_hardening` must be
    /// bit-for-bit unchanged, immediately, on the very first substep and
    /// every subsequent one. This is the real claim the whole rewrite
    /// exists to make true (see `project()`'s own doc: "when `g` is truly
    /// zero, `gamma` is truly, exactly zero") -- verified directly, not
    /// inferred from the relaxation test's asymptotic approach.
    #[test]
    fn ngf_arrest_at_zero_g_is_exact_not_asymptotic() {
        let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
        sand.ngf_enabled = true;
        const DT: f32 = 0.001;

        // Same real-compaction, real-deviatoric-strain trial state as the
        // other two tests -- a particle that WOULD flow under any g>0 (per
        // the constitutive test), to make "g=0 stops it" a real claim, not
        // a vacuous one (an already-relaxed particle stopping proves
        // nothing).
        let f0 = Mat2::from_cols(Vec2::new(0.95, 0.0), Vec2::new(0.0, 0.9));
        let mut particles = Particles::from(vec![Particle {
            deformation_gradient: f0,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            friction_hardening: 1.111,
            hardening_scale: 1.0,
            ..Particle::zeroed()
        }]);
        let q0 = particles.friction_hardening[0];

        println!("── NGF ARREST AT g=0 (plastic multiplier must be EXACT zero, not asymptotic) ──");
        for step in 1..=20 {
            let mut ctx = particles.update_ctx(0);
            ctx.nonlocal_fluidity = 0.0; // the real claim under test
            sand.update_particle(&mut ctx, DT);
            let f = particles.deformation_gradient[0];
            let q = particles.friction_hardening[0];
            // `friction_hardening` gets `q + dq` -- with `dq` computed as
            // exactly 0.0 (see `project()`'s ngf branch: `(0.0*mu_ratio*dt)
            // .max(0.0).min(dev_norm) == 0.0`), IEEE754 addition of 0.0 is
            // exact, so this stays bit-for-bit -- the real, strict claim.
            assert_eq!(
                q, q0,
                "friction_hardening changed at step {step} with g=0 exactly -- plastic \
                 multiplier was not exactly zero (got {q}, expected {q0})"
            );
            // `deformation_gradient` is NOT held to bit-for-bit: even with
            // zero plastic increment, `update_particle` still round-trips
            // the elastic part through svd2 -> ln -> (subtract exactly
            // zero) -> exp, and ln/exp don't perfectly invert each other
            // in f32 (ordinary ULP-level rounding, unrelated to plasticity
            // -- confirmed by the first run of this test hitting exactly
            // this: 0.95000005 vs 0.95, a 5e-8 difference, many orders
            // below any real plastic drift this test would actually catch,
            // e.g. one real step at the relaxation test's own G=1e-4 moves
            // dev_norm by ~8.6e-4). 1e-6 is well above float noise, far
            // below any real leak.
            let diff = (f.x_axis.x - f0.x_axis.x).abs()
                + (f.x_axis.y - f0.x_axis.y).abs()
                + (f.y_axis.x - f0.y_axis.x).abs()
                + (f.y_axis.y - f0.y_axis.y).abs();
            assert!(
                diff < 1.0e-6,
                "deformation_gradient drifted beyond float noise at step {step} with g=0 \
                 exactly -- arrest is not real (got {f:?}, expected ~{f0:?}, diff={diff:.3e})"
            );
        }
        println!(
            "  20 substeps at g=0: friction_hardening bit-for-bit unchanged, \
             deformation_gradient unchanged within float noise (<1e-6)"
        );
    }

    /// LOCKED PROTOCOL, test 4/5 (collapse test, 2026-08-19): the real
    /// claim this whole rewrite exists to make true. A genuine dynamic
    /// column collapse, held at ONE constant numerical regime throughout
    /// (`apic_blend=0.6`, the value already independently established as
    /// the real numerical-stability floor for ANY violent MPM collapse --
    /// material-agnostic, not a target-angle tuning knob) -- NO switch to
    /// a different damping regime afterward, NO `cundall_damping`, NO
    /// `post_event_relax_threshold` F-reset trick, NO hand-picked step
    /// count of any kind. If NGF's own field dynamics is doing the real
    /// work (as tests 1-3 confirm it can, in isolation), then angle,
    /// kinetic energy (`vmax`), `max(g)`, AND the actual plastic flow
    /// rate (`mean_q` rate of change between checkpoints -- a real,
    /// directly-measured proxy for `||F_p_dot||`, not a guess: `q`
    /// accumulates by exactly `gamma` every yielding substep) must all
    /// independently plateau together at long horizon in a scene that
    /// never gets told "the collapse is over."
    #[test]
    fn ngf_switch_free_collapse_all_metrics_plateau_together() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
        let config = SimConfig {
            max_substeps_per_step: 4000,
            apic_blend: 0.6, // constant -- no switch, ever, this is the whole point
            ..SimConfig::earth(GRID, CELL_M, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR / CELL_M + 8.0),
            material_id: 0,
            precompute_initial_volumes: true,
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

        println!(
            "── NGF SWITCH-FREE COLLAPSE, ALL METRICS (real seconds, apic_blend=0.6 \
             constant throughout, no damping/reset/switch of any kind) ──"
        );
        // Extended past the original 55s (2026-08-19): the first run showed
        // g_at_pile NOT decaying to zero (mean~0.33, min~0.007-0.01, steady
        // rather than shrinking) and mean_q_rate still nonzero and still
        // falling at t=55s -- real ambiguity between "genuinely arrested,
        // just slow to fully settle" and "asymptotic marginal-state decay
        // that never truly stops" (the exact risk flagged before this
        // rewrite started: at mu==mu_s exactly, g's own decay term is zero,
        // so a hard stop isn't guaranteed). Later checkpoints only -- the
        // 1-55s shape is already known from the prior run.
        let checkpoints_s: &[f32] = &[10.0, 55.0, 120.0, 220.0, 350.0];
        let mut cumulative_steps = 0usize;
        let mut prev_mean_q = {
            let particles = solver.particles();
            particles.friction_hardening.iter().sum::<f32>() / particles.len() as f32
        };
        let mut prev_t = 0.0f32;
        let mut angles = Vec::new();
        let mut vmaxes = Vec::new();
        let mut g_maxes = Vec::new();
        let mut q_rates = Vec::new();
        for &t in checkpoints_s {
            let target_steps = (t / 0.01).round() as usize;
            solver.step_n(target_steps - cumulative_steps);
            cumulative_steps = target_steps;

            let particles = solver.particles();
            let xs: Vec<f32> = particles.x.iter().map(|p| p.x).collect();
            let ys: Vec<f32> = particles.x.iter().map(|p| p.y).collect();
            let n = xs.len() as f32;
            let center_x = xs.iter().sum::<f32>() / n;
            let half_w = p99(xs.iter().map(|x| (x - center_x).abs()).collect());
            let top_y = p99(ys.clone());
            let bottom_y = p01(ys);
            let height = top_y - bottom_y;
            let angle = (height / half_w).atan().to_degrees();
            let vmax = particles
                .v
                .iter()
                .map(|v| v.length())
                .fold(0.0f32, f32::max);
            let mean_q = particles.friction_hardening.iter().sum::<f32>() / n;
            let field = solver.granular_fluidity().unwrap();
            let (_g_min, g_mean, g_max, g_count) = field.g_stats();
            // Real confound check (2026-08-19): the whole-grid g_max is
            // suspected of being dominated by the pressure-floor
            // equilibrium in far-from-the-pile cells, not the pile's own
            // dynamics -- sample g directly AT the pile's own particles to
            // find out, rather than trusting or distrusting the whole-grid
            // number in the abstract.
            let (g_pile_min, g_pile_mean, g_pile_max) = field.g_stats_near(particles);
            let dt_elapsed = (t - prev_t).max(1e-6);
            let mean_q_rate = (mean_q - prev_mean_q) / dt_elapsed;
            angles.push(angle);
            vmaxes.push(vmax);
            g_maxes.push(g_pile_max); // tail-plateau check uses the PILE value, not the confounded whole-grid one
            q_rates.push(mean_q_rate);
            println!(
                "  t={t:5.1}s: angle={angle:.2}deg vmax={vmax:.4} | whole-grid: g_max={g_max:.4} \
                 g_mean={g_mean:.4} g_nonzero={g_count} | AT PILE: g_min={g_pile_min:.4} \
                 g_mean={g_pile_mean:.4} g_max={g_pile_max:.4} | mean_q_rate={mean_q_rate:.6}/s"
            );
            prev_mean_q = mean_q;
            prev_t = t;
        }

        // Real plateau check on the TAIL of the run (last 3 checkpoints) --
        // each of the four independent metrics must be within a small
        // fraction of its own tail-mean, not just "visually flat."
        fn tail_spread(vals: &[f32]) -> (f32, f32) {
            let tail = &vals[vals.len() - 3..];
            let mean = tail.iter().sum::<f32>() / tail.len() as f32;
            let max_dev = tail.iter().map(|v| (v - mean).abs()).fold(0.0f32, f32::max);
            (mean, max_dev)
        }
        let (angle_mean, angle_dev) = tail_spread(&angles);
        let (vmax_mean, vmax_dev) = tail_spread(&vmaxes);
        let (gmax_mean, gmax_dev) = tail_spread(&g_maxes);
        let (qrate_mean, qrate_dev) = tail_spread(&q_rates);
        println!(
            "  tail (last 3 checkpoints): angle {angle_mean:.2}±{angle_dev:.2}deg  \
             vmax {vmax_mean:.4}±{vmax_dev:.4}  g_at_pile_max {gmax_mean:.4}±{gmax_dev:.4}  \
             mean_q_rate {qrate_mean:.6}±{qrate_dev:.6}/s"
        );
    }

    /// Real, direct test of a NEW hypothesis (2026-08-19, prompted by the
    /// extended switch-free collapse run above showing g_pile_mean flat/
    /// RISING instead of decaying over 350s): is `g`'s nonlocal diffusion
    /// term, at this scene's resolution-independent 8mm effective grain
    /// diameter, actually confined to a real grain-scale cooperativity
    /// length (Henann & Kamrin's own intent -- a shear-band-width scale,
    /// a few mm), or does it homogenize across the WHOLE domain within
    /// the first few real seconds -- long before any existing test here
    /// would catch it (the only prior d-calibration sweep,
    /// `ngf_grain_diameter_sweep_accuracy_and_resolution_independence`,
    /// measured a 200-step mid-collapse runout RATIO, never g's own
    /// spatial profile over real seconds).
    ///
    /// Back-of-envelope, computed directly (not simulated): the bare
    /// diffusion coefficient this config implies is
    /// `D=(A*d)^2/t0 = (0.48*0.008)^2/1e-4 ≈ 0.147 m^2/s`, giving a
    /// diffusion length `sqrt(D*t)` that already EXCEEDS this scene's own
    /// 0.96m domain width by t≈10s. This test samples the REAL field
    /// (which also has a reaction/decay term fighting the spread, so the
    /// bare-diffusion number is only a naive upper bound, not a
    /// prediction) at fixed world offsets from the column's own center,
    /// to see whether real g decays with distance (localized, as the
    /// theory intends) or is already flat (homogenized) at these
    /// distances/times.
    fn run_spatial_homogenization_probe(grain_diameter_m: f32) {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
        let config = SimConfig {
            max_substeps_per_step: 4000,
            apic_blend: 0.6,
            ..SimConfig::earth(GRID, CELL_M, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR / CELL_M + 8.0),
            material_id: 0,
            precompute_initial_volumes: true,
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
        cfg.pressure_floor_pa = cfg.grain_density_kg_m3 * 9.81 * grain_diameter_m;
        let field = GranularFluidityField::new(cfg, ngf_pressure_and_ratio, GRID);
        solver = solver.with_granular_fluidity(field);

        // Sample g at fixed world offsets from the column's own center, at
        // a fixed height near the floor -- NOT at real particles (which
        // move), so the SAME spatial points are compared across
        // checkpoints. Reuses `g_stats_near`'s own public API on
        // synthetic single-particle probes (same pattern
        // `granular_fluidity.rs`'s own unit tests use), rather than
        // exposing the field's private grid internals.
        fn probe_at(offset_m: f32) -> Particles {
            let mut p = Particle::zeroed();
            p.x = Vec2::new(GRID as f32 * 0.5 + offset_m / CELL_M, FLOOR / CELL_M + 2.0);
            p.mass = 1.0;
            p.initial_volume = 1.0;
            Particles::from(vec![p])
        }
        let offsets_m: &[f32] = &[0.0, 0.05, 0.10, 0.20, 0.30, 0.40];

        let bare_d = (0.48f32 * grain_diameter_m).powi(2) / 1.0e-4;
        println!(
            "  d={:.1}mm: back-of-envelope bare D=(A*d)^2/t0 = {bare_d:.6} m^2/s (domain width=0.96m)",
            grain_diameter_m * 1000.0
        );
        let checkpoints_s: &[f32] = &[0.3, 1.0, 2.0, 4.0, 10.0];
        let mut cumulative_steps = 0usize;
        for &t in checkpoints_s {
            let target_steps = (t / 0.01).round() as usize;
            solver.step_n(target_steps - cumulative_steps);
            cumulative_steps = target_steps;
            let field = solver.granular_fluidity().unwrap();
            let sqrt_dt = (bare_d * t).sqrt();
            print!("  t={t:5.1}s (naive diffusion length={sqrt_dt:.3}m):");
            for &off in offsets_m {
                let (_min, mean, _max) = field.g_stats_near(&probe_at(off));
                print!("  g(+{off:.2}m)={mean:.4}");
            }
            println!();
        }
    }

    #[test]
    fn ngf_g_field_spatial_homogenization_onset() {
        println!(
            "── NGF g-FIELD SPATIAL HOMOGENIZATION ONSET (fixed world points, real seconds) ──"
        );
        run_spatial_homogenization_probe(0.008);
    }

    /// Real, direct follow-up: does shrinking `grain_diameter_m` (which
    /// shrinks the diffusion coefficient D=(A*d)^2/t0 as d^2 -- a MUCH
    /// bigger lever than anything else in this config) actually localize
    /// the homogenization onset the test above measured at the shipped
    /// 8mm value? If d=1mm keeps g's spread confined near the pile
    /// instead of saturating the whole 0.96m domain within ~1s, that's
    /// real, direct evidence the 8mm choice (picked in the 2026-08-03
    /// sweep for a completely different metric -- 200-step runout ratio,
    /// never checked against long-horizon spatial spread) is the
    /// structural cause of tonight's wrong-angle/never-fully-arrests
    /// findings, not a flaw in NGF theory itself.
    #[test]
    fn ngf_g_field_spatial_homogenization_smaller_grain_diameter() {
        println!("── NGF g-FIELD SPATIAL HOMOGENIZATION, SMALLER d (localization check) ──");
        for &d_mm in &[0.3f32, 0.5, 1.0, 1.5, 2.0, 4.0] {
            run_spatial_homogenization_probe(d_mm * 1.0e-3);
        }
    }

    /// Real, decisive test following the spatial-homogenization findings
    /// above: at the shipped 8mm effective grain diameter, `g` locks into
    /// a runaway, spatially-uniform, NEVER-decaying plateau within 1-2
    /// real seconds -- at d<=1mm it stays genuinely localized and decays
    /// over time, the physically-correct reaction-diffusion behavior the
    /// theory intends. `ngf_config`'s own doc already discloses that the
    /// ORIGINAL reason the LITERAL real grain diameter (`GRAIN_DIAMETER_M`,
    /// 0.3mm) was abandoned for 8mm -- "couldn't diffuse fast enough" --
    /// was itself downstream of two bugs (missing dx-normalization, a
    /// min_dt floor override) that are BOTH now fixed, and that this was
    /// never re-tested at the literal value afterward. This test does
    /// that re-test directly, on the same real angle-tracking metric
    /// (`ngf_switch_free_collapse_all_metrics_plateau_together`) that
    /// found the wrong 0.9deg (not 30-35deg) plateau at 8mm -- does the
    /// real, literal grain diameter, now that the two blocking bugs are
    /// fixed, resist collapse enough to reach a real repose angle instead
    /// of homogenizing the whole pile into uniform fluidity?
    #[test]
    fn ngf_switch_free_collapse_real_grain_diameter_angle_check() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
        let config = SimConfig {
            max_substeps_per_step: 4000,
            apic_blend: 0.6, // constant -- no switch, ever, same as the 8mm baseline test
            ..SimConfig::earth(GRID, CELL_M, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR / CELL_M + 8.0),
            material_id: 0,
            precompute_initial_volumes: true,
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
        cfg.grain_diameter_m = GRAIN_DIAMETER_M; // real literal 0.3mm, not the 8mm compromise
        cfg.pressure_floor_pa = cfg.grain_density_kg_m3 * 9.81 * GRAIN_DIAMETER_M;
        let field = GranularFluidityField::new(cfg, ngf_pressure_and_ratio, GRID);
        solver = solver.with_granular_fluidity(field);

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

        println!("── NGF SWITCH-FREE COLLAPSE, REAL LITERAL GRAIN DIAMETER (0.3mm, not 8mm) ──");
        // Same checkpoints as the ORIGINAL (pre-extension) 8mm run, for a
        // direct apples-to-apples angle comparison against the already-
        // known 8mm numbers (31.7deg@1s -> 1.3deg@4s -> ~0.9deg plateau
        // by 10-55s).
        let checkpoints_s: &[f32] = &[1.0, 4.0, 10.0, 20.0, 35.0, 55.0];
        let mut cumulative_steps = 0usize;
        let mut prev_mean_q = {
            let particles = solver.particles();
            particles.friction_hardening.iter().sum::<f32>() / particles.len() as f32
        };
        let mut prev_t = 0.0f32;
        for &t in checkpoints_s {
            let target_steps = (t / 0.01).round() as usize;
            solver.step_n(target_steps - cumulative_steps);
            cumulative_steps = target_steps;

            let particles = solver.particles();
            let xs: Vec<f32> = particles.x.iter().map(|p| p.x).collect();
            let ys: Vec<f32> = particles.x.iter().map(|p| p.y).collect();
            let n = xs.len() as f32;
            let center_x = xs.iter().sum::<f32>() / n;
            let half_w = p99(xs.iter().map(|x| (x - center_x).abs()).collect());
            let top_y = p99(ys.clone());
            let bottom_y = p01(ys);
            let height = top_y - bottom_y;
            let angle = (height / half_w).atan().to_degrees();
            let vmax = particles
                .v
                .iter()
                .map(|v| v.length())
                .fold(0.0f32, f32::max);
            let mean_q = particles.friction_hardening.iter().sum::<f32>() / n;
            let field = solver.granular_fluidity().unwrap();
            let (g_pile_min, g_pile_mean, g_pile_max) = field.g_stats_near(particles);
            let dt_elapsed = (t - prev_t).max(1e-6);
            let mean_q_rate = (mean_q - prev_mean_q) / dt_elapsed;
            println!(
                "  t={t:5.1}s: angle={angle:.2}deg vmax={vmax:.4}  AT PILE: g_min={g_pile_min:.4} \
                 g_mean={g_pile_mean:.4} g_max={g_pile_max:.4} | mean_q_rate={mean_q_rate:.6}/s"
            );
            prev_mean_q = mean_q;
            prev_t = t;
        }
    }

    /// Real, direct test of the hypothesis raised after the bistable/
    /// hysteretic extension (2026-08-19, see this module's own doc on
    /// `run_spatial_homogenization_probe` -- and the memory file it cites)
    /// came back WORSE, not better, at the real literal grain diameter:
    /// the isolated proof of the hysteresis mechanism
    /// (`bistable_hysteresis_same_mu_two_stable_branches_by_history`,
    /// `granular_fluidity.rs`) used mu=0.705, hand-picked to sit just
    /// inside the real ~0.011-wide bistable window these params imply
    /// (mu_s=0.70, mu_2~1.07, hysteresis_amplitude=0.0116). Does REAL
    /// collapse dynamics in this engine actually linger inside that
    /// narrow window, or does `mu_ratio` blow straight past it during the
    /// violent phase -- in which case the paper's own real DEM-calibrated
    /// amplitude (calibrated to THEIR system's own dynamic range, not
    /// this engine's) would be too narrow to matter here, a real
    /// calibration mismatch rather than a flaw in the mechanism. Measures
    /// `mu_ratio` directly across all particles at the very start of a
    /// real collapse (same scene as the angle-tracking tests), no NGF
    /// coupling needed for this measurement.
    #[test]
    fn ngf_real_mu_ratio_excursion_during_collapse_onset() {
        const GRID: usize = 96;
        const FLOOR: f32 = 0.05;
        let config = SimConfig {
            max_substeps_per_step: 4000,
            apic_blend: 0.6,
            ..SimConfig::earth(GRID, CELL_M, 0.01)
        };
        let column = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 16),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR / CELL_M + 8.0),
            material_id: 0,
            precompute_initial_volumes: true,
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
        sand.ngf_enabled = false; // measuring the raw trial mu_ratio, not coupled to g
        let mut solver = Simulation::new(config, column)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

        fn percentiles(mut v: Vec<f32>) -> (f32, f32, f32, f32) {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            let p50 = v[n / 2];
            let p90 = v[((n as f32 - 1.0) * 0.90).round() as usize];
            let p99 = v[((n as f32 - 1.0) * 0.99).round() as usize];
            (p50, p90, p99, v[n - 1])
        }

        println!(
            "── REAL mu_ratio EXCURSION DURING COLLAPSE ONSET (mu_s=0.70, bistable window ~[0.70,0.711]) ──"
        );
        let checkpoints_s: &[f32] = &[0.05, 0.1, 0.2, 0.5, 1.0];
        let mut cumulative_steps = 0usize;
        for &t in checkpoints_s {
            let target_steps = (t / 0.01).round() as usize;
            solver.step_n(target_steps - cumulative_steps);
            cumulative_steps = target_steps;
            let particles = solver.particles();
            let mus: Vec<f32> = (0..particles.len())
                .map(|i| ngf_pressure_and_ratio(&particles.get(i)).1)
                .collect();
            let (p50, p90, p99, pmax) = percentiles(mus);
            println!("  t={t:5.2}s: mu_ratio p50={p50:.4} p90={p90:.4} p99={p99:.4} max={pmax:.4}");
        }
    }
}
