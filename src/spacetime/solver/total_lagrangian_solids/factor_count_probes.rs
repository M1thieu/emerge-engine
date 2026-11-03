//! The band factor's builds counted, nothing changed (`tmp/papers/
//! dl_realtime/FACTOR_REUSE_PLAN.md`, criterion 1): run D's impact frame
//! and the Norton block's first contact, every build with its cause from
//! the control trace's factor line. The same frame also carries M1's
//! contact trace (`CHATTER_TRACE_PLANS.md`, `chatter_trace_probes`), one
//! run for both. A probe: measurement, nothing held.

use super::impact_local::{FRAME_S, drop};
use super::*;
use crate::materials::solid::viscoplastic_records::ViscoplasticRecord;
use crate::spacetime::total_lagrangian::hooks;
use crate::{SimConfig, SpawnRegion as Spawn};
use glam::IVec2;

#[path = "chatter_trace_probes.rs"]
mod chatter_trace_probes;

const RHO_STAND_IN: f64 = 119.0 * 0.453_592_37 / (0.3048 * 0.3048 * 0.3048);
const NORTON_E_PA: f64 = 30e6;

/// `tests/norton.rs`'s drop at 30 MPa, tolerance 1e-2.
fn norton() -> Simulation {
    norton_scene(FRAME_S, TotalLagrangianConfig::new(1e-2))
}

/// The same drop, frames of `frame_s`, on `path`.
fn norton_scene(frame_s: f64, path: TotalLagrangianConfig) -> Simulation {
    let e_pa = NORTON_E_PA;
    let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
    let mut config = SimConfig::earth(40, 0.01, frame_s as f32);
    config.min_dt = config.min_dt.min(frame_s as f32);
    config.total_lagrangian = Some(path);
    let material = record.material((e_pa, RHO_STAND_IN), &config);
    let props = crate::materials::physical_props::Elastic {
        e_pa: e_pa as f32,
        nu: record.nu as f32,
        rho_kg_m3: RHO_STAND_IN as f32,
    };
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        Spawn::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 12.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    sim
}

/// The scene run `before` frames, then one frame with the control trace
/// on (its factor lines are the causes, parsed from the log) and the
/// builds recorded.
fn count(label: &str, mut sim: Simulation, before: usize) {
    for _ in 0..before {
        sim.step();
    }
    hooks::watch_factors(true);
    hooks::watch_contact_rows(true);
    hooks::set_control_trace(true);
    println!("{label} FRAME START");
    sim.step();
    println!("{label} FRAME END");
    hooks::set_control_trace(false);
    let built = hooks::take_factors();
    hooks::watch_factors(false);
    let traced = hooks::take_contact_rows();
    hooks::watch_contact_rows(false);
    chatter_trace_probes::report_m1(label, &traced);
    let f = sim.total_lagrangian_frame().expect("on");
    println!(
        "{label} frame {before}: factor builds recorded {}, Timing::factorizations {}; steps {} rejected {} newton {} gmres {} wall {:.3} s",
        built.len(),
        f.timing.factorizations,
        f.steps,
        f.rejected,
        f.newton,
        f.gmres,
        f.wall
    );
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_factor_builds_counted() {
    super::cap::capped("probe_factor_builds_counted", 3600, || {
        count(
            "run D",
            drop(
                FRAME_S,
                TotalLagrangianConfig::new(1e-2).with_subframe_handoff(),
            ),
            8,
        );
        count("Norton", norton(), 9);
    });
}

/// S1' on the Norton block (`tmp/papers/dl_realtime/S1_FIXED_CONTACT_STEP_PLAN.md`
/// sec. 10), information: the reference N_A (fixed steps of at most `h /
/// (4 c)`, `c` the record's P wave, infinite tolerance, no handoff), the
/// baseline N_B (handoff, tol 1e-2) and the candidate N_S (N_B with
/// `EMERGE_PROBE_CONTACT_FIXED_STEP`, read by the path), selected by
/// `EMERGE_PROBE_NORTON_RUN` = `A`, `B` or `S`; per 1/60 s frame the
/// centre's height and vertical velocity, the cumulative plastic work and
/// the energy. A probe: measurement, nothing held.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_s1_norton() {
    super::cap::capped("probe_s1_norton", 4 * 3600, || {
        let run = std::env::var("EMERGE_PROBE_NORTON_RUN").unwrap_or_else(|_| "B".into());
        let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
        let nu = record.nu;
        let modulus = NORTON_E_PA * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let c = (modulus / RHO_STAND_IN).sqrt();
        let per = if run == "A" {
            (FRAME_S * 2.0 * c / 0.01).ceil() as usize
        } else {
            1
        };
        let mut sim = if run == "A" {
            norton_scene(
                FRAME_S / per as f64,
                TotalLagrangianConfig::new(f64::INFINITY),
            )
        } else {
            // `F` (ROW15_FORECAST_PLAN.md): the baseline with the contacts
            // held from the step their free-flight gap closes.
            let mut path = TotalLagrangianConfig::new(1e-2).with_subframe_handoff();
            path.settings.forecast_activation = run == "F";
            if let Some(tol) = std::env::var("EMERGE_PROBE_GMRES_TOL")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                path.settings.gmres_tolerance = tol;
            }
            println!(
                "knob active: forecast_activation {}, gmres_tolerance {:e}",
                path.settings.forecast_activation, path.settings.gmres_tolerance
            );
            let mut sim = norton_scene(FRAME_S, path);
            super::impact_local::ulp_perturb(&mut sim);
            sim
        };
        println!(
            "Norton {run}: c {c:.4} m/s (E {NORTON_E_PA:e} Pa, nu {nu}, rho {RHO_STAND_IN:.2} kg/m^3); {per} path frames per 1/60 s, kept halves {:.4e} s",
            FRAME_S / per as f64 / 2.0
        );
        let floor = sim.config().boundary_thickness as f64 * f64::from(sim.config().dx_meters);
        let mut plastic = 0.0;
        let count = sim.particles().len() as f64;
        let (mut allowed, mut worst) = (0.0f64, 0.0f64);
        let frames = std::env::var("EMERGE_PROBE_NORTON_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12);
        // MODE_FREE_HANDOFF_PLAN.md M1: each frame end's mode-free lower gap
        // bounds over the next frame, checked against the gaps measured at
        // that frame's end.
        let mut m1_previous: Option<(Vec<f64>, Vec<f64>)> = None;
        for frame in 0..frames {
            let clock = std::time::Instant::now();
            let (mut steps, mut rejected, mut contact) = (0u32, 0u32, false);
            for _ in 0..per {
                sim.step();
                let f = sim.total_lagrangian_frame().expect("on");
                plastic += f.plastic_work;
                allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
                let round_off = count
                    * f64::EPSILON
                    * (f.parts.iter().map(|e| e.abs()).sum::<f64>()
                        + f.heat.abs()
                        + f.start_energy.abs());
                worst = worst.max(f.balance.abs() / (allowed + round_off));
                steps += f.steps;
                rejected += f.rejected;
                contact |= f.contacts > 0;
            }
            let state = sim.total_lagrangian.as_ref().expect("on");
            let n = state.particles;
            let mass: f64 = state.mass[..n].iter().sum();
            let (y, vy) = (0..n).fold((0.0, 0.0), |(y, vy), p| {
                (
                    y + state.mass[p] * state.x[p].y,
                    vy + state.mass[p] * state.v[p].y,
                )
            });
            let f = sim.total_lagrangian_frame().expect("on");
            // The largest depth of an outline point beyond the floor plane.
            let penetration = state.outlines[0]
                .iter()
                .map(|&q| (floor - state.x[q].y).max(0.0))
                .fold(0.0, f64::max);
            if super::handoff::mode_free_probes::on() {
                use super::handoff::mode_free_probes::{clear_until, free, lower_gaps, solve_term};
                let (planes, gravity) = (
                    sim.total_lagrangian_walls().0,
                    sim.total_lagrangian_gravity(),
                );
                let points: Vec<usize> = state.outlines.iter().flatten().copied().collect();
                let measured: Vec<f64> = points
                    .iter()
                    .map(|&o| {
                        planes
                            .iter()
                            .map(|w| w.inward * (state.x[o][w.axis] - w.at))
                            .fold(f64::INFINITY, f64::min)
                    })
                    .collect();
                if let Some((bounds, terms)) = &m1_previous {
                    let least = measured
                        .iter()
                        .zip(bounds)
                        .map(|(g, b)| g - b)
                        .fold(f64::INFINITY, f64::min);
                    // M1' (sec. 10): the same check with the solve's term.
                    let least_with = measured
                        .iter()
                        .zip(bounds)
                        .zip(terms)
                        .map(|((g, b), d)| g - (b - d))
                        .fold(f64::INFINITY, f64::min);
                    let term = terms.iter().copied().fold(0.0, f64::max);
                    println!(
                        "M1' check frame {frame}: least (measured gap - (bound - D_p)) {least_with:+.4e} m, largest D_p {term:.4e} m, measured deficit over D_p {:.3e}{}",
                        (-least).max(0.0) / term,
                        if least_with < 0.0 { " VIOLATED" } else { "" }
                    );
                    println!(
                        "M1 check frame {frame}: least (measured gap - last frame end's bound) {least:+.4e} m, least measured gap {:+.4e} m{}",
                        measured.iter().copied().fold(f64::INFINITY, f64::min),
                        if least < 0.0 { " VIOLATED" } else { "" }
                    );
                }
                let body = free(state);
                let t = state.time;
                // The path's own frame length (`config.dt`, f32), not `FRAME_S`: the
                // first M1 run used `FRAME_S` and measured the 8.7e-10 s between them.
                let next = (t, t + f64::from(sim.config.dt));
                let bounds = lower_gaps(state, &body, (&planes, gravity), next);
                let until = clear_until(state, &body, (&planes, gravity), next);
                let tolerance = sim
                    .config
                    .total_lagrangian
                    .as_ref()
                    .expect("on")
                    .settings
                    .newton_tolerance;
                let (terms, allowed, lambda_min) =
                    solve_term(state, gravity, (next.1 - next.0, tolerance));
                let with: Vec<f64> = bounds.iter().zip(&terms).map(|(b, d)| b - d).collect();
                println!(
                    "M1' frame {frame}: R_a {allowed:.4e}, lambda_min(M) {lambda_min:.4e}, least bound with D_p {:+.4e} m",
                    with.iter().copied().fold(f64::INFINITY, f64::min)
                );
                println!(
                    "M1 frame {frame} end t {t:.6}: least lower gap bound over the next frame {:+.4e} m, clear until {until:.6}",
                    bounds.iter().copied().fold(f64::INFINITY, f64::min)
                );
                m1_previous = Some((bounds, terms));
            }
            println!(
                "Norton {run} frame {frame}: steps {steps} rejected {rejected} contact {contact} wall {:.3} s; height {:.6e} m, vy {:+.6e} m/s, plastic work {plastic:.6e} J/m, energy {:+.6e} J/m, penetration {penetration:.4e} m; ledger worst so far {worst:.3} of its bound",
                clock.elapsed().as_secs_f64(),
                y / mass - floor,
                vy / mass,
                f.energy
            );
        }
    });
}

/// T1 (`tmp/papers/dl_realtime/T1_MASSLESS_BOUNDARY_PLAN.md`): the inert set
/// of DEP eq. 97 carried to the block's four sides, computed on a throwaway
/// grid of the same scene before the run: per wall axis, the normal
/// component's coefficients whose support meets the body only within one
/// cell of that side (the particles each coefficient reaches all lie in the
/// body's outer cell layer on that side; adaptation declared: the body's
/// sides are its outline's reference extents, the knot span the cell-wide
/// layer inside them). Returns the faces `(lattice, i, j)` for
/// `hooks::set_massless` and prints the removed lumped mass per component
/// against the plan's derivation (sec. 6.1: `dx / (3 H)` = 8.3 % of the y
/// mass, `dx / (3 W)` = 4.2 % of the x mass for E2's block).
fn t1_inert_faces(state: &TotalLagrangianState) -> Vec<(usize, usize, usize)> {
    let grid = &state.cache.as_ref().expect("built by the first frame").grid;
    let cell = grid.cell;
    let (lo, hi) = state.outlines[0].iter().fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(lo, hi), &q| (lo.min(state.reference[q]), hi.max(state.reference[q])),
    );
    let mut touched: Vec<Vec<DVec2>> = vec![Vec::new(); grid.len];
    for p in 0..state.particles {
        if state.mass[p] <= 0.0 {
            continue;
        }
        for a in 0..2 {
            grid.visit(p, a, |k, w, _| {
                if w != 0.0 {
                    touched[k].push(state.reference[p]);
                }
            });
        }
    }
    let inert: Vec<bool> = (0..grid.len)
        .map(|k| {
            let t = &touched[k];
            let a = grid.axis_of[k];
            !t.is_empty()
                && (t.iter().all(|x| x[a] < lo[a] + cell) || t.iter().all(|x| x[a] > hi[a] - cell))
        })
        .collect();
    let mut faces = Vec::new();
    for (a, l) in grid.lattices.iter().enumerate() {
        for j in 0..l.nj {
            for i in 0..l.ni {
                let u = grid.slot[l.start + i + l.ni * j];
                if u != usize::MAX && inert[u] {
                    faces.push((a, i, j));
                }
            }
        }
    }
    for (a, name, derived) in [(0, "x", 1.0 / 3.0 / 8.0), (1, "y", 1.0 / 3.0 / 4.0)] {
        let (mut removed, mut all) = (0.0, 0.0);
        for (k, &held) in inert.iter().enumerate().take(grid.len) {
            if grid.axis_of[k] == a {
                all += grid.lumped[k];
                if held {
                    removed += grid.lumped[k];
                }
            }
        }
        println!(
            "T1 inert set, {name} component: {} coefficients, removed lumped mass {:.4} of the component's (derived in the continuum: {derived:.4})",
            (0..grid.len)
                .filter(|&k| inert[k] && grid.axis_of[k] == a)
                .count(),
            removed / all.max(f64::MIN_POSITIVE)
        );
    }
    faces
}

/// T1 on run D (`T1_MASSLESS_BOUNDARY_PLAN.md` sec. 5-6): `EMERGE_PROBE_T1_RUN`
/// = `base` (no inert set: today's counts in the same probe), `T1` (the
/// inert set, run D's path) or `Aprime` (the inert set at reference A's
/// fixed steps). Per frame the steps, rejected tries, contact, wall, vy,
/// vibration (in the standard-mass modes: the massless mass is singular)
/// and the ledger's worst over its bound; on frame 8 (run D's paths) M1's
/// chatter counts, pushes and free runs with their durations. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_t1_massless() {
    use super::impact_local::{FINE, sample};
    use super::vibration::modes;
    super::cap::capped("probe_t1_massless", 4 * 3600, || {
        let run = std::env::var("EMERGE_PROBE_T1_RUN").unwrap_or_else(|_| "T1".into());
        let handoff = TotalLagrangianConfig::new(1e-2).with_subframe_handoff();
        let mut throwaway = drop(FRAME_S, handoff);
        throwaway.step();
        let basis = modes(throwaway.total_lagrangian.as_ref().expect("on"));
        let faces = t1_inert_faces(throwaway.total_lagrangian.as_ref().expect("on"));
        hooks::set_massless((run != "base").then(|| faces.clone()));
        let (mut sim, per) = if run == "Aprime" {
            (
                drop(
                    FRAME_S / FINE as f64,
                    TotalLagrangianConfig::new(f64::INFINITY),
                ),
                FINE,
            )
        } else {
            (drop(FRAME_S, handoff), 1)
        };
        super::impact_local::ulp_perturb(&mut sim);
        let count = sim.particles().len() as f64;
        let (mut allowed, mut worst) = (0.0f64, 0.0f64);
        for frame in 0..10 {
            let clock = std::time::Instant::now();
            let trace = frame == 8 && per == 1;
            if trace {
                hooks::watch_contact_rows(true);
            }
            let (mut steps, mut rejected, mut contact) = (0u32, 0u32, false);
            for _ in 0..per {
                sim.step();
                let f = sim.total_lagrangian_frame().expect("on");
                allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
                let round_off = count
                    * f64::EPSILON
                    * (f.parts.iter().map(|e| e.abs()).sum::<f64>()
                        + f.heat.abs()
                        + f.start_energy.abs());
                worst = worst.max(f.balance.abs() / (allowed + round_off));
                steps += f.steps;
                rejected += f.rejected;
                contact |= f.contacts > 0;
            }
            if trace {
                let traced = hooks::take_contact_rows();
                hooks::watch_contact_rows(false);
                chatter_trace_probes::report_m1(&format!("T1 {run}"), &traced);
            }
            let (height, vy, vibration, energy) = sample(&sim, &basis);
            println!(
                "T1 {run} frame {frame}: steps {steps} rejected {rejected} contact {contact} wall {:.3} s; height {height:.6e} m, vy {vy:+.7e} m/s, vibration {vibration:.5e} J/m, energy {energy:+.6e} J/m; ledger worst so far {worst:.3} of its bound",
                clock.elapsed().as_secs_f64()
            );
        }
        hooks::set_massless(None);
    });
}
