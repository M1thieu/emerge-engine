//! Several bodies on one shared step: 1, 4 and 16 of E2's clay blocks far apart, each
//! flying, resting or impacting, 30 frames; per frame the step's counts,
//! wall and ledger, per body its energy and largest velocity change. A
//! probe: measurement, nothing held.

use super::impact_local::{FRAME_S, clay};
use super::*;
use crate::spacetime::total_lagrangian::hooks;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Role {
    Flying,
    Resting,
    Impacting,
}

/// 30 frames, or `EMERGE_PROBE_MULTI_FRAMES`.
fn frames() -> usize {
    std::env::var("EMERGE_PROBE_MULTI_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30)
}

/// The blocks side by side, 16 cells apart (8 cells between any two),
/// each spawned by its role: flying high enough to fall the whole run
/// (1.23 m in 0.5 s), resting on the floor, impacting from 10 cm.
fn scene(roles: &[Role]) -> (Simulation, Vec<u32>) {
    // `EMERGE_PROBE_REFINE` / `EMERGE_PROBE_SPACING` (`impact_local::refinement`):
    // the same physical scene on cells `r` times finer, as the fixed-step reference's.
    let (r, spacing) = super::impact_local::refinement();
    let res = (16 * roles.len() + 16).max(160) * r;
    let mut config = SimConfig::earth(res, 0.01 / r as f32, FRAME_S as f32);
    config.min_dt = config.min_dt.min(FRAME_S as f32);
    // The step tolerance, 1e-2, or `EMERGE_PROBE_MULTI_TOLERANCE` (the
    // shared-step candidate's test).
    let tolerance = std::env::var("EMERGE_PROBE_MULTI_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-2);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(tolerance));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    // `EMERGE_PROBE_MULTI_REVERSE`
    // spawns the bodies in reverse order at the same positions; the tags are
    // put back in the roles' order.
    let reverse = std::env::var_os("EMERGE_PROBE_MULTI_REVERSE").is_some();
    if reverse {
        println!("knob active: bodies spawned in reverse order");
    }
    let mut order: Vec<(usize, &Role)> = roles.iter().enumerate().collect();
    if reverse {
        order.reverse();
    }
    let mut tags: Vec<u32> = order
        .into_iter()
        .map(|(k, role)| {
            let bottom = match role {
                Role::Flying => 130.0,
                Role::Resting => 0.0,
                Role::Impacting => 10.0,
            };
            sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(8 * r as i32, 4 * r as i32))
                    .at(Vec2::new(
                        (12.0 + 16.0 * k as f32) * r as f32,
                        floor + (bottom + 2.0) * r as f32,
                    ))
                    .spacing(spacing)
                    .material(id)
                    .mass_from(&clay(), &config),
            )
        })
        .collect();
    if reverse {
        tags.reverse();
    }
    (sim, tags)
}

/// Each body's energy (kinetic, stored, gravity's), J/m, and a bound on its
/// evaluation's round-off, `n eps sum |terms|`.
fn energies(sim: &Simulation, owner: &[usize], bodies: usize) -> Vec<(f64, f64)> {
    let state = sim.total_lagrangian.as_ref().expect("on");
    let gravity = sim.total_lagrangian_gravity();
    let mut out = vec![(0.0, 0.0); bodies];
    for p in 0..state.particles {
        let terms = [
            0.5 * state.mass[p] * state.v[p].length_squared(),
            state.v0[p] * state.solids[p].energy(state.f[p]),
            -state.mass[p] * gravity.dot(state.x[p]),
        ];
        let b = &mut out[owner[p]];
        b.0 += terms.iter().sum::<f64>();
        b.1 += terms.iter().map(|t| t.abs()).sum::<f64>();
    }
    let n = state.particles as f64;
    out.iter()
        .map(|&(e, a)| (e, 4.0 * n * f64::EPSILON * a))
        .collect()
}

/// E4 of `tests/realtime_profile.rs`: two 6 by 4 clay blocks on the floor
/// (centres 8 and 18 cells), the left one closing at 0.25 m/s, tolerance
/// 1e-1, grid 40; the profile's own scene, for its per-frame trace.
fn e4_scene() -> (Simulation, Vec<u32>) {
    let mut config = SimConfig::earth(40, 0.01, FRAME_S as f32);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let tags: Vec<u32> = [8.0f32, 18.0]
        .iter()
        .map(|&x| {
            sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(6, 4))
                    .at(Vec2::new(x, floor + 2.0))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&clay(), &config),
            )
        })
        .collect();
    let speed = 0.25 / 0.01;
    let left = tags[0];
    let moving: Vec<bool> = sim
        .particles()
        .user_tag
        .iter()
        .map(|&t| t == left)
        .collect();
    for (v, &m) in sim.particles_mut().v.iter_mut().zip(&moving) {
        if m {
            *v = Vec2::new(speed, 0.0);
        }
    }
    (sim, tags)
}

/// The previous frame's per-body energies (value, round-off bound) and the
/// particles' velocities.
type Snapshot = (Vec<(f64, f64)>, Vec<DVec2>);

fn run(label: &str, roles: &[Role]) {
    run_scene(label, scene(roles), roles);
}

fn run_scene(label: &str, (mut sim, tags): (Simulation, Vec<u32>), roles: &[Role]) {
    let bodies = roles.len();
    // `EMERGE_PROBE_MULTI_ULP=p`: particle `p` (the first spawned body's
    // when `p` is within it) raised by one ulp, the ensemble rule's kind (i).
    if let Some(p) = std::env::var("EMERGE_PROBE_MULTI_ULP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let y = &mut sim.particles_mut().x[p].y;
        *y = f32::from_bits(y.to_bits() + 1);
        println!("knob active: one ulp on particle {p}'s height");
    }
    // A start line from inside the run (an empty log is then visible at
    // once), and `EMERGE_PROBE_MULTI_FRAME_CAP_S`: a
    // frame longer than it ends the run after its print.
    println!(
        "{label} start: {bodies} bodies {roles:?}, {} frames",
        frames()
    );
    let cap = std::env::var("EMERGE_PROBE_MULTI_FRAME_CAP_S")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());
    if let Some(cap) = cap {
        println!("knob active: frame cap {cap} s");
    }
    let clock = std::time::Instant::now();
    let mut m2_previous: Vec<M2Pair> = Vec::new();
    let mut allowed = 0.0f64;
    let mut owner: Vec<usize> = Vec::new();
    let mut before: Option<Snapshot> = None;
    let mut totals = [0u64; 4];
    let mut wall_total = 0.0;
    let mut flying_worst = 0.0f64;
    let mut flying_against_work = 0.0f64;
    // The body-body conditions (`weighted_gap::reaching_conditions`, key
    // kind 4) the kept steps held, from the contact trace.
    hooks::watch_contact_rows(true);
    let owners = |sim: &Simulation| -> Vec<usize> {
        (0..sim.total_lagrangian.as_ref().expect("on").particles)
            .map(|p| {
                tags.iter()
                    .position(|&t| t == sim.particles.user_tag[p])
                    .expect("a body's particle")
            })
            .collect()
    };
    for n in 0..frames() {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("on");
        if owner.is_empty() {
            owner = owners(&sim);
        }
        // Attribution: the owners read again from the tags.
        let moved_owner = owners(&sim)
            .iter()
            .zip(&owner)
            .filter(|(a, b)| a != b)
            .count();
        let body_body = body_body_conditions(&sim, &owner);
        allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
        let count = sim.particles().len() as f64;
        let round_off = count
            * f64::EPSILON
            * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
        let ledger = f.balance.abs() / (allowed + round_off);
        let now = energies(&sim, &owner, bodies);
        let v = sim.total_lagrangian.as_ref().expect("on").v[..owner.len()].to_vec();
        let mut per_body = Vec::new();
        let mut flying_change = 0.0;
        if let Some((e0, v0)) = &before {
            for b in 0..bodies {
                let de = now[b].0 - e0[b].0;
                let dv = (0..owner.len())
                    .filter(|&p| owner[p] == b)
                    .map(|p| (v[p] - v0[p]).length())
                    .fold(0.0, f64::max);
                if roles[b] == Role::Flying {
                    flying_worst = flying_worst.max(de.abs() / (now[b].1 + e0[b].1));
                    flying_change += de.abs();
                }
                // The body's extent, m: whether two bodies touch, and
                // whether the particles' owners still hold (a body's
                // extent jumping would say they do not).
                let x = &sim.total_lagrangian.as_ref().expect("on").x;
                let (low, high) = (0..owner.len()).filter(|&p| owner[p] == b).fold(
                    (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
                    |(l, h), p| (l.min(x[p]), h.max(x[p])),
                );
                let (mass, momentum) = (0..owner.len()).filter(|&p| owner[p] == b).fold(
                    (0.0, DVec2::ZERO),
                    |(m, q), p| {
                        let state = sim.total_lagrangian.as_ref().expect("on");
                        (m + state.mass[p], q + state.mass[p] * state.v[p])
                    },
                );
                let mean = momentum / mass;
                // The centre's specific
                // energy `1/2 |v_c|^2 + |g| y_c` and the vibration (kinetic
                // about the rigid motion plus stored), both constant in free
                // flight, so comparable across runs whose impacts register in
                // different frames.
                let state = sim.total_lagrangian.as_ref().expect("on");
                let mine: Vec<usize> = (0..owner.len()).filter(|&p| owner[p] == b).collect();
                let centre = mine
                    .iter()
                    .fold(DVec2::ZERO, |c, &p| c + state.mass[p] * state.x[p])
                    / mass;
                let (spin_l, inertia, relative, stored) =
                    mine.iter().fold((0.0, 0.0, 0.0, 0.0), |(l, i, k, w), &p| {
                        let (r, u) = (state.x[p] - centre, state.v[p] - mean);
                        (
                            l + state.mass[p] * r.perp_dot(u),
                            i + state.mass[p] * r.length_squared(),
                            k + 0.5 * state.mass[p] * u.length_squared(),
                            w + state.v0[p] * state.solids[p].energy(state.f[p]),
                        )
                    });
                let vibration = relative - 0.5 * spin_l * spin_l / inertia + stored;
                let g = sim.total_lagrangian_gravity();
                let specific = 0.5 * mean.length_squared() - g.dot(centre);
                per_body.push(format!(
                    "{:?} dE {de:+.3e} J/m dv {dv:.3e} m/s mean v ({:+.4}, {:+.4}) m/s x [{:.4}, {:.4}] y [{:.4}, {:.4}] m e_c {specific:.9e} J/kg vib {vibration:.6e} J/m",
                    roles[b], mean.x, mean.y, low.x, high.x, low.y, high.y
                ));
            }
        }
        // The flying bodies' summed energy change against the step's own
        // booked terms (residual work, contact bound) plus round-off: the
        // per-body ledger's yardstick (the round-off bound alone kept above
        // for transparency).
        if before.is_some() {
            flying_against_work = flying_against_work
                .max(flying_change / (f.residual_work + f.contact_work_bound + round_off));
        }
        before = Some((now, v));
        totals[0] += u64::from(f.steps);
        totals[1] += u64::from(f.rejected);
        totals[2] += u64::from(f.newton);
        totals[3] += u64::from(f.gmres);
        wall_total += f.wall;
        println!(
            "{label} frame {n}: steps {} rejected {} newton {} gmres {} contacts {} reaching {} wall {:.3} s ledger {ledger:.3} of its bound; residual work {:.3e} J/m, contact bound {:.3e} J/m; owners changed {moved_owner}; {body_body}; bodies: {}",
            f.steps,
            f.rejected,
            f.newton,
            f.gmres,
            f.contacts,
            f.reaching,
            f.wall,
            f.residual_work,
            f.contact_work_bound,
            per_body.join("; ")
        );
        pair_bound_check(&sim, (&owner, roles), n, &mut m2_previous);
        if cap.is_some_and(|cap| f.wall > cap) {
            println!("{label}: frame {n} over the frame cap, run ended (capped, not a verdict)");
            break;
        }
    }
    hooks::watch_contact_rows(false);
    println!(
        "{label} total: {bodies} bodies {:?}; steps {} rejected {} newton {} gmres {}; frames' wall {wall_total:.1} s (run {:.1} s); flying bodies' largest |dE| per frame over their round-off bound {flying_worst:.3e} (not the yardstick), their summed |dE| per frame over the frame's residual work + contact bound + round-off {flying_against_work:.3e}",
        roles,
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        clock.elapsed().as_secs_f64()
    );
}

/// The frame's kept steps' body-body conditions (key kind 4): those
/// formed, those held with a positive multiplier, their summed impulse
/// `sum lambda dt`, and the body pairs the held ones join (a point's body
/// by its owner, an outline point's by its outline's first particle
/// nearest).
fn body_body_conditions(sim: &Simulation, owner: &[usize]) -> String {
    let state = sim.total_lagrangian.as_ref().expect("on");
    let outline_body: Vec<usize> = state
        .outlines
        .iter()
        .map(|o| {
            let at = state.x[o[0]];
            (0..owner.len())
                .min_by(|&a, &b| {
                    (state.x[a] - at)
                        .length()
                        .total_cmp(&(state.x[b] - at).length())
                })
                .map_or(usize::MAX, |p| owner[p])
        })
        .collect();
    let body_of = |i: usize| -> usize {
        if i < owner.len() {
            owner[i]
        } else {
            state
                .outlines
                .iter()
                .position(|o| o.contains(&i))
                .map_or(usize::MAX, |o| outline_body[o])
        }
    };
    let (mut formed, mut held, mut impulse) = (0u32, 0u32, 0.0f64);
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (dt, rows) in hooks::take_contact_rows() {
        for r in rows.iter().filter(|r| r.key.0 == 4) {
            formed += 1;
            if r.multiplier <= 0.0 {
                continue;
            }
            held += 1;
            impulse += r.multiplier * dt;
            let mut touched: Vec<usize> = r.push.iter().map(|&(i, _)| body_of(i)).collect();
            touched.sort_unstable();
            touched.dedup();
            if let [a, b, ..] = touched[..]
                && !pairs.contains(&(a, b))
            {
                pairs.push((a, b));
            }
        }
    }
    format!(
        "body-body conditions formed {formed}, held {held}, impulse {impulse:.3e}, pairs {pairs:?}"
    )
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_multi_body() {
    super::cap::capped("probe_multi_body", 4 * 3600, || {
        use Role::*;
        let only = std::env::var("EMERGE_PROBE_MULTI").unwrap_or_default();
        let runs: Vec<(&str, Vec<Role>)> = vec![
            ("N1 flying", vec![Flying]),
            ("N1 resting", vec![Resting]),
            ("N1 impacting", vec![Impacting]),
            ("N4 quiet", vec![Flying, Resting, Flying, Resting]),
            ("N4 one impact", vec![Impacting, Resting, Flying, Resting]),
            ("16 blocks, quiet", [Flying, Resting].repeat(8)),
            (
                "16 blocks, one impact",
                std::iter::once(Impacting)
                    .chain([Resting, Flying].repeat(8).into_iter().skip(1))
                    .collect(),
            ),
        ];
        for (label, roles) in runs {
            if !only.is_empty() && !label.starts_with(&only) {
                continue;
            }
            run(label, &roles);
        }
        // E4's scene, traced, only when asked by name.
        if only == "E4" {
            run_scene("E4", e4_scene(), &[Role::Impacting, Role::Resting]);
        }
    });
}

/// One pair's bound record from a frame end: the bodies, their outlines'
/// points, the best direction and its lower bound over the next frame.
struct M2Pair {
    bodies: (usize, usize),
    points: (Vec<usize>, Vec<usize>),
    n: DVec2,
    lower: f64,
    /// Neither body can reach a wall over the interval by
    /// the mode-free bound less its solve term (so neither holds one).
    in_scope: bool,
}

/// The pair bound at a frame end (`EMERGE_PROBE_MODE_FREE`):
/// the last frame end's pair bounds checked against the measured
/// separations, with each body's solve term over the frame's `S` equal
/// steps, `S^2` times one step's (`sum_s (1 + 2 (S - s)) =
/// S^2`); then this frame end's bounds.
fn pair_bound_check(
    sim: &Simulation,
    (owner, roles): (&[usize], &[Role]),
    frame: usize,
    previous: &mut Vec<M2Pair>,
) {
    use super::handoff::mode_free_probes::{
        directions, free_of, lower_gaps, on, pair_lower, separation, solve_term_of,
    };
    if !on() || owner.is_empty() {
        return;
    }
    let state = sim.total_lagrangian.as_ref().expect("on");
    let length = f64::from(sim.config.dt);
    let tolerance = sim
        .config
        .total_lagrangian
        .as_ref()
        .expect("on")
        .settings
        .newton_tolerance;
    let gravity = sim.total_lagrangian_gravity();
    let steps = sim.total_lagrangian_frame().expect("on").steps.max(1);
    let bodies = roles.len();
    let particles: Vec<Vec<usize>> = (0..bodies)
        .map(|b| (0..owner.len()).filter(|&p| owner[p] == b).collect())
        .collect();
    // Each body's outline: the one whose first point's nearest particle the
    // body owns.
    let outline_of: Vec<usize> = (0..bodies)
        .map(|b| {
            state
                .outlines
                .iter()
                .position(|o| {
                    let first = state.reference[o[0]];
                    let nearest = (0..owner.len())
                        .min_by(|&p, &q| {
                            (state.reference[p] - first)
                                .length_squared()
                                .total_cmp(&(state.reference[q] - first).length_squared())
                        })
                        .expect("particles");
                    owner[nearest] == b
                })
                .expect("each body has an outline")
        })
        .collect();
    let step = length / f64::from(steps);
    for pair in previous.iter() {
        let (a, b) = pair.bodies;
        let term = |body: usize, points: &[usize]| {
            let (terms, _, _) = solve_term_of(
                state,
                gravity,
                (step, tolerance),
                (&particles[body], points),
            );
            terms.iter().copied().fold(0.0, f64::max) * f64::from(steps).powi(2)
        };
        let deviation = term(a, &pair.points.0) + term(b, &pair.points.1);
        let measured = separation(state, (&pair.points.0, &pair.points.1), pair.n);
        let resting = roles[a] == Role::Resting || roles[b] == Role::Resting;
        println!(
            "pair bound check, frame {frame} pair {a}-{b} ({}, wall scope {}): measured separation {measured:+.4e} m, bound {:+.4e} m, solve term {deviation:.3e} m, margin {:+.4e} m{}",
            if resting {
                "with a resting body: measurement only"
            } else {
                "free bodies"
            },
            if pair.in_scope {
                "in scope"
            } else {
                "out of scope"
            },
            pair.lower,
            measured - (pair.lower - deviation),
            if measured < pair.lower - deviation {
                " VIOLATED"
            } else {
                ""
            }
        );
    }
    previous.clear();
    let free: Vec<_> = (0..bodies)
        .map(|b| free_of(state, &particles[b], &state.outlines[outline_of[b]]))
        .collect();
    let next = (state.time, state.time + length);
    // The scope per body: every outline point's wall bound over the next
    // frame, less the body's one-step solve term, stays positive.
    let planes = sim.total_lagrangian_walls().0;
    let scope: Vec<bool> = (0..bodies)
        .map(|b| {
            let points = &state.outlines[outline_of[b]];
            let (terms, _, _) =
                solve_term_of(state, gravity, (length, tolerance), (&particles[b], points));
            let term = terms.iter().copied().fold(0.0, f64::max);
            lower_gaps(state, &free[b], (&planes, gravity), next)
                .iter()
                .all(|&g| g - term > 0.0)
        })
        .collect();
    for a in 0..bodies {
        for b in a + 1..bodies {
            let outlines = [
                &state.outlines[outline_of[a]][..],
                &state.outlines[outline_of[b]][..],
            ];
            let (n, lower) = directions(state, outlines, (&free[a], &free[b]))
                .into_iter()
                .map(|n| (n, pair_lower(state, (&free[a], &free[b]), n, next)))
                .max_by(|x, y| x.1.total_cmp(&y.1))
                .expect("a direction");
            previous.push(M2Pair {
                bodies: (a, b),
                points: (outlines[0].to_vec(), outlines[1].to_vec()),
                n,
                lower,
                in_scope: scope[a] && scope[b],
            });
        }
    }
}

/// With `EMERGE_PROBE_TOL_PER_BODY`, each
/// particle's body (the outline's tag) and each body's own tolerance
/// energy at the start, its kinetic and stored energy plus `m_b |g| L_b`
/// over its outline's extent along gravity (`tolerance_energy`'s terms, per
/// body), handed to the controller (`hooks::set_body_scales`).
pub(super) fn set_body_scales(
    sim: &Simulation,
    state: &TotalLagrangianState,
    bodies: &[(f64, std::ops::Range<usize>)],
) {
    if std::env::var_os("EMERGE_PROBE_TOL_PER_BODY").is_none() {
        hooks::set_body_scales(None);
        return;
    }
    let gravity = sim.total_lagrangian_gravity();
    let g = gravity.length();
    let along = if g > 0.0 { gravity / g } else { DVec2::ZERO };
    let mut body_of = vec![0usize; state.x.len()];
    let mut e0 = vec![0.0f64; bodies.len()];
    for (b, (spawned, (body_mass, points))) in
        sim.total_lagrangian_outlines.iter().zip(bodies).enumerate()
    {
        for (p, owner) in body_of.iter_mut().enumerate().take(state.particles) {
            if sim.particles.user_tag[p] == spawned.tag {
                *owner = b;
                e0[b] += 0.5 * state.mass[p] * state.v[p].length_squared()
                    + state.v0[p] * state.solids[p].energy(state.f[p]);
            }
        }
        for o in points.clone() {
            body_of[o] = b;
        }
        let (low, high) = points
            .clone()
            .map(|o| along.dot(state.reference[o]))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), d| {
                (l.min(d), h.max(d))
            });
        if high >= low {
            e0[b] += body_mass * g * (high - low);
        }
    }
    println!("knob active: per-body tolerance energies {e0:?}");
    hooks::set_body_scales(Some((body_of, e0)));
}
