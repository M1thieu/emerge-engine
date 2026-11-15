//! A rigid body on a body of revolution: the same body with its coefficients
//! tied to the one rigid motion the symmetry admits, the axial translation
//! (`axisymmetric::Revolution::with_tied_rows`). No new law and no new contact:
//! its surface is an outline of massless points, its mass one point, and the
//! step's own body-body conditions carry the push. A declared limit of a stiff
//! body, whose cost is one unknown; untying its coefficients gives the real body.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use glam::{DMat2, DVec2};

use super::super::axisymmetric as ax;
use super::super::{
    ConservingSettings, ContactSet, Elastic, Estimator, GridLayout, LagrangianCache, Particles,
    StepControl, controlled_step, start_velocity_field, tolerance_energy,
};
use super::dynamic::{
    Body, MARGIN, RodImpact, cell, cylinder_outline, rod_on_a_wall, stored_energy,
};
use super::lame;

/// A half disc's outline about `(0, centre)`, counter-clockwise: the arc from the
/// lower pole to the upper one in `arcs` chords, then back down the axis one point
/// a cell. Its two poles are the corners.
fn half_disc(radius: f64, centre: f64, arcs: usize, h: f64) -> (Vec<(f64, f64)>, Vec<usize>) {
    let mut points: Vec<(f64, f64)> = (0..arcs)
        .map(|i| {
            let angle = -FRAC_PI_2 + PI * i as f64 / arcs as f64;
            (radius * angle.cos(), centre + radius * angle.sin())
        })
        .collect();
    let upper = points.len();
    let down = (2.0 * radius / h).ceil() as usize;
    points
        .extend((0..down).map(|i| (0.0, centre + radius - 2.0 * radius * i as f64 / down as f64)));
    (points, vec![0, upper])
}

/// A tied body in free flight under gravity, through the controller: on the exact
/// parabola, its coefficients giving one translation and no strain, the ledger in
/// the solves' bound, and the same bits at any thread count.
#[test]
fn a_tied_body_falls_on_its_parabola_at_any_thread_count() {
    let h = cell(1e-3);
    let (lambda, mu) = lame(1e6, 0.3);
    let radius = 10.0 * h;
    let origin = MARGIN as f64 * h;
    let (polygon, corners) = half_disc(radius, radius + 2.0 * h, 64, h);
    let mut reference: Vec<DVec2> = polygon
        .iter()
        .map(|&(r, z)| DVec2::new(origin + r, origin + z))
        .collect();
    let outlines = vec![(0..reference.len()).collect::<Vec<usize>>()];
    let corners = vec![corners];
    reference.push(DVec2::new(origin, origin + radius + 2.0 * h));
    let centre = reference.len() - 1;
    let n = reference.len();
    let total = 1.0452 / TAU;
    let mut mass = vec![0.0; n];
    mass[centre] = total;
    let v0 = vec![0.0; n];
    let top = 2 * MARGIN + 2 * (radius / h) as usize + 6;
    let layout = GridLayout::new(2 * MARGIN + 12, top, h as f32)
        .of_revolution(ax::Revolution::about(MARGIN).with_tied_rows((MARGIN, top)));
    let settings = ConservingSettings::default();
    let gravity = DVec2::new(0.0, -9.81);
    let speed = -1.0;
    let end = 20e-3;
    let run = || {
        let start: Vec<DVec2> = (0..n)
            .map(|p| {
                if p == centre {
                    DVec2::new(0.0, speed)
                } else {
                    DVec2::ZERO
                }
            })
            .collect();
        let mut v = start_velocity_field(&layout, &reference, (&mass, &v0, &start), &settings);
        v[centre] = start[centre];
        let mut x = reference.clone();
        let mut f = vec![DMat2::IDENTITY; n];
        let mut solids = vec![Elastic::neo_hookean(lambda, mu); n];
        let penalty = vec![0.0; n];
        let energy0 = 0.5 * total * speed * speed;
        let scale = tolerance_energy(energy0, gravity, [(total, &reference[..])]).sqrt();
        let control = StepControl::klapproth(1e-3, 1e-3, Estimator::Acary);
        let mut cache: Option<LagrangianCache> = None;
        let (mut t, mut tau) = (0.0, 1e-3);
        let mut allowed = 0.0;
        let mut worst = 0.0f64;
        while t < end {
            let step = controlled_step(
                &layout,
                Particles {
                    x: &mut x,
                    v: &mut v,
                    f: &mut f,
                    mass: &mass,
                    v0: &v0,
                },
                (&reference, &mut solids),
                (gravity, &mut tau),
                (
                    ContactSet {
                        planes: &[],
                        pairs: &[],
                        outlines: &outlines,
                        corners: &corners,
                        friction: 0.0,
                    },
                    &penalty,
                    &settings,
                ),
                &mut cache,
                (&control, scale),
            );
            assert!(
                !step.at_floor,
                "tied body, at {t}: at the estimator's floor"
            );
            t += step.dt;
            allowed += step
                .kept
                .iter()
                .map(|r| r.residual_work + r.contact_work_bound)
                .sum::<f64>();
            let energy = 0.5 * total * v[centre].y * v[centre].y
                + total * gravity.y * (reference[centre].y - x[centre].y);
            worst = worst.max((energy - energy0).abs());
        }
        (t, x, v, f, worst, allowed)
    };
    let pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a thread pool")
    };
    let (t, x, v, f, worst, allowed) = pool(1).install(run);
    let (_, four, _, _, _, _) = pool(4).install(run);
    let fall = speed * t + 0.5 * gravity.y * t * t;
    let speed_at = speed + gravity.y * t;
    let position_off = (x[centre].y - reference[centre].y - fall).abs() / fall.abs();
    let speed_off = (v[centre].y - speed_at).abs() / speed_at.abs();
    let strain = f
        .iter()
        .map(|f| (*f - DMat2::IDENTITY).to_cols_array().map(f64::abs))
        .fold(0.0f64, |w, a| a.into_iter().fold(w, f64::max));
    let apart = x
        .iter()
        .zip(&reference)
        .map(|(q, r)| (q.x - r.x).abs())
        .fold(0.0f64, f64::max);
    let together = x
        .iter()
        .zip(&reference)
        .map(|(q, r)| (q.y - r.y - fall).abs())
        .fold(0.0f64, f64::max);
    let identical = x
        .iter()
        .zip(&four)
        .all(|(a, b)| a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits());
    println!(
        "tied body in flight, t {t:.4e} s: position off {position_off:.2e}, speed off {speed_off:.2e}, strain {strain:.2e}, radial motion {apart:.2e} m, spread on the fall {together:.2e} m, energy off {worst:.2e} J/rad (bound {allowed:.2e}), 1 and 4 threads identical {identical}"
    );
    assert!(position_off <= 1e-12, "position off {position_off:e}");
    assert!(speed_off <= 1e-12, "speed off {speed_off:e}");
    assert!(strain <= 1e-12, "strain {strain:e}");
    assert_eq!(apart, 0.0, "a tied point moved radially");
    assert!(together <= 1e-12 * fall.abs(), "spread {together:e}");
    assert!(
        worst <= allowed + 64.0 * f64::EPSILON * (0.5 * total * speed_at * speed_at),
        "energy off {worst:e}, bound {allowed:e}"
    );
    assert!(identical, "the thread count changed the bits");
}

/// What a rod's impact on a tied floor gave, beside the wall's own measures.
struct OnTiedFloor {
    impact: RodImpact,
    /// The tied body's momentum at the end against the rod's lost, over the start's.
    momentum_off: f64,
}

/// The rod of `dynamic::rod_on_a_wall` on a flat tied body of `ratio` times its mass,
/// at rest, whose surface carries the conditions (its normal exact, its rim beyond
/// the rod's). The rod's own numbers against a wall are the reference.
fn rod_on_a_tied_floor(
    (cells_across, cells, nu): (usize, usize, f64),
    (strain, tolerance): (f64, f64),
    ratio: f64,
) -> OnTiedFloor {
    let (young, rho) = (1e6, 1000.0);
    let (lambda, mu) = lame(young, nu);
    let h = cell(0.001);
    let length = cells as f64 * h;
    let floor_across = 3 * cells_across;
    let floor_along = 2;
    // The tied rows begin clear of the rod's, the floor two rows inside them.
    let first = MARGIN + cells + 5;
    let lift = (first - MARGIN + 2) as f64 * h;
    let (rod_outline, rod_corners) = cylinder_outline(cells_across, cells, h);
    let (floor_outline, floor_corners) = cylinder_outline(floor_across, floor_along, h);
    let mut extra = rod_outline.clone();
    extra.extend(floor_outline.iter().map(|&(r, z)| (r, z + lift)));
    extra.push((0.0, lift + 0.5 * floor_along as f64 * h));
    let mut body = Body::new((cells_across, cells, h, rho), &extra);
    let top = first + floor_along + 4 + MARGIN;
    body.layout = GridLayout::new(2 * MARGIN + floor_across + 2, top, h as f32);
    let layout = body
        .layout
        .of_revolution(ax::Revolution::about(MARGIN).with_tied_rows((first, top)));
    let massive = body.massive;
    let floor_start = massive + rod_outline.len();
    let centre = floor_start + floor_outline.len();
    let rod_mass: f64 = body.mass.iter().sum();
    body.mass[centre] = ratio * rod_mass;
    let outlines = vec![
        (floor_start..centre).collect::<Vec<usize>>(),
        (massive..floor_start).collect::<Vec<usize>>(),
    ];
    let corners = vec![floor_corners, rod_corners];
    let c0 = (young / rho).sqrt();
    let v0 = strain * c0;
    let settings = ConservingSettings {
        forecast_activation: true,
        ..ConservingSettings::default()
    };
    let dt = 1.5 * h / ((lambda + 2.0 * mu) / rho).sqrt();
    let touch = 2.37 * dt;
    let bottom = MARGIN as f64 * h;
    // The floor's top starts where the wall's plane stood.
    let floor_top = bottom + lift + floor_along as f64 * h;
    let shift = bottom - v0 * touch - floor_top;
    let mut x = body.reference.clone();
    for q in &mut x[floor_start..] {
        q.y += shift;
    }
    let start: Vec<DVec2> = (0..x.len())
        .map(|p| {
            if p < massive {
                DVec2::new(0.0, -v0)
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let field = start_velocity_field(
        &layout,
        &body.reference,
        (&body.mass, &body.v0, &start),
        &settings,
    );
    let mut v = start;
    v[massive..].copy_from_slice(&field[massive..]);
    let mut f = vec![DMat2::IDENTITY; x.len()];
    let mut solids = vec![Elastic::neo_hookean(lambda, mu); x.len()];
    let penalty: Vec<f64> = (0..x.len())
        .map(|p| {
            if p < massive {
                0.0
            } else {
                solids[p].bulk + solids[p].mu
            }
        })
        .collect();
    let momentum_of = |v: &[DVec2]| -> f64 { (0..massive).map(|p| body.mass[p] * v[p].y).sum() };
    let momentum0 = momentum_of(&v);
    let tied = body.mass[centre];
    let energy = |x: &[DVec2], v: &[DVec2], f: &[DMat2], solids: &[Elastic]| -> f64 {
        let kinetic: f64 = (0..massive)
            .map(|p| 0.5 * body.mass[p] * v[p].length_squared())
            .sum();
        kinetic + 0.5 * tied * v[centre].length_squared() + stored_energy(&body, solids, x, f)
    };
    let energy0 = energy(&x, &v, &f, &solids);
    let scale = tolerance_energy(
        energy0,
        DVec2::ZERO,
        [(rod_mass, &body.reference[..massive])],
    )
    .sqrt();
    let control = StepControl::klapproth(tolerance, dt, Estimator::Acary);
    let exact = 2.0 * length / c0;
    let end = touch + exact + 8.0 * dt;
    println!("tied rod: touch {touch:.4e} s, end {end:.4e} s, dt {dt:.4e} s");
    let (mut t, mut tau) = (0.0, dt);
    let mut cache: Option<LagrangianCache> = None;
    let (mut first_touch, mut last): (Option<f64>, f64) = (None, 0.0);
    let (mut allowed, mut ledger, mut worst_momentum) = (0.0, 0.0f64, 0.0f64);
    let (mut steps, mut solves, mut rejected) = (0, 0, 0);
    while t < end {
        let step = controlled_step(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &body.mass,
                v0: &body.v0,
            },
            (&body.reference, &mut solids),
            (DVec2::ZERO, &mut tau),
            (
                ContactSet {
                    planes: &[],
                    pairs: &[],
                    outlines: &outlines,
                    corners: &corners,
                    friction: 0.0,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
            (&control, scale),
        );
        assert!(!step.at_floor, "rod, at {t}: at the estimator's floor");
        assert_eq!(step.stagnated, 0, "rod, at {t}: Newton stagnated");
        let piece = step.dt / step.kept.len() as f64;
        for (k, report) in step.kept.iter().enumerate() {
            let begun = t + k as f64 * piece;
            if report.contacts > 0 {
                first_touch.get_or_insert(begun);
                last = begun + piece;
            }
            allowed += report.residual_work + report.contact_work_bound;
        }
        t += step.dt;
        steps += 1;
        solves += step.solves;
        rejected += step.rejected;
        let round_off = x.len() as f64 * f64::EPSILON * energy0;
        ledger = ledger.max((energy(&x, &v, &f, &solids) - energy0).abs() - round_off);
        let closed = momentum_of(&v) + tied * v[centre].y;
        worst_momentum = worst_momentum.max((closed - momentum0).abs());
    }
    let momentum = momentum_of(&v);
    let first_touch = first_touch.expect("the rod reached the floor");
    OnTiedFloor {
        impact: RodImpact {
            contact_error: (last - first_touch) / exact - 1.0,
            touch_error: (first_touch - touch) * c0 / length,
            rebound: -momentum / momentum0,
            ledger: ledger / energy0,
            ledger_bound: allowed / energy0,
            momentum: worst_momentum / momentum0.abs(),
            steps,
            solves,
            rejected,
            lowest_gap: 0.0,
        },
        momentum_off: (tied * v[centre].y + momentum - momentum0).abs() / momentum0.abs(),
    }
}

/// The same rod through the body-body route, on a tied floor 1e4 times its mass:
/// the wall's contact time and take-off within 0.5 %, the tied body gaining what
/// the rod loses within 1e-9 of its momentum, the ledger in bound.
#[test]
fn a_rod_leaves_a_heavy_tied_floor_as_it_leaves_a_wall() {
    let shape = (2, 32, 0.0);
    let load = (1e-3, 1e-3);
    let wall = rod_on_a_wall(shape, load);
    let tied = rod_on_a_tied_floor(shape, load, 1e4);
    println!(
        "rod at nu 0, wall: contact {:+.3e}, take-off {:.5}, steps {}, {} solves, {} rejected; tied floor: contact {:+.3e}, take-off {:.5}, momentum off {:.2e} (per step {:.2e}), ledger {:.2e} (bound {:.2e}), steps {}, {} solves, {} rejected",
        wall.contact_error,
        wall.rebound,
        wall.steps,
        wall.solves,
        wall.rejected,
        tied.impact.contact_error,
        tied.impact.rebound,
        tied.momentum_off,
        tied.impact.momentum,
        tied.impact.ledger,
        tied.impact.ledger_bound,
        tied.impact.steps,
        tied.impact.solves,
        tied.impact.rejected,
    );
    assert!(
        (tied.impact.contact_error - wall.contact_error).abs() <= 0.005,
        "contact time {:e} against {:e}",
        tied.impact.contact_error,
        wall.contact_error
    );
    assert!(
        (tied.impact.rebound - wall.rebound).abs() <= 0.005,
        "take-off {} against {}",
        tied.impact.rebound,
        wall.rebound
    );
    assert!(
        tied.momentum_off <= 1e-9,
        "momentum {:e}",
        tied.momentum_off
    );
    assert!(
        tied.impact.ledger <= tied.impact.ledger_bound,
        "ledger {:e}, bound {:e}",
        tied.impact.ledger,
        tied.impact.ledger_bound
    );
}

/// A sphere on a block: the block's size in cells of `h`, its law and density.
struct Block {
    across: usize,
    depth: usize,
    h: f64,
    solid: Elastic,
    rho: f64,
}

/// The sphere: radius, whole mass, speed at the touch.
struct Sphere {
    radius: f64,
    mass: f64,
    speed: f64,
}

/// One step's reading of a drop.
struct DropSample {
    t: f64,
    /// The sphere's lowest point below the block's first surface.
    depth: f64,
    speed: f64,
    /// The block's push on the whole sphere, from its momentum.
    force: f64,
}

/// What a drop gave.
struct Dropped {
    samples: Vec<DropSample>,
    first: f64,
    last: f64,
    ledger: f64,
    ledger_bound: f64,
    energy0: f64,
    plastic: f64,
    friction: f64,
    /// Axial momentum against gravity's, the mirrors' (`StepReport::mirror_force`)
    /// and the residual's impulses over the whole run, the worst over the start's.
    momentum: f64,
    /// The former reading, declared beside it: against gravity's impulse alone,
    /// until a wave from the first touch can reach a mirror (the PDE's front; the
    /// discrete step's front leads it), over the start's.
    momentum_window: f64,
    /// The largest net push of the contacts over the unknowns, over the start's
    /// momentum per second.
    contact_net: f64,
    /// The deepest any point of the block's outline entered the circle, in cells.
    crossing: f64,
    steps: usize,
    solves: u32,
    rejected: u32,
    stagnated: u32,
}

/// A sphere on a block held by mirrors at its base and its rim, the block's outline
/// carrying the conditions, through the controller with its longest step at the
/// crossing of a cell by the block's fastest wave. Stops at `stop(sample)`.
fn sphere_on_a_block(
    block: &Block,
    sphere: &Sphere,
    (gravity, friction, tolerance): (f64, f64, f64),
    stop: impl Fn(&DropSample) -> bool,
) -> Dropped {
    let Block {
        across,
        depth,
        h,
        solid,
        rho,
    } = block;
    let (across, depth, h, rho) = (*across, *depth, *h, *rho);
    let Sphere {
        radius,
        mass,
        speed,
    } = *sphere;
    let (outline, outline_corners) = cylinder_outline(across, depth, h);
    // The tied rows begin clear of the block's, the sphere two rows inside them.
    let first = MARGIN + depth + 5;
    let lowest = (first - MARGIN + 2) as f64 * h;
    let centre_z = lowest + radius;
    let arcs = (PI * radius / (0.25 * h)).ceil() as usize;
    let (polygon, polygon_corners) = half_disc(radius, centre_z, arcs, h);
    let mut extra = outline.clone();
    extra.extend_from_slice(&polygon);
    extra.push((0.0, centre_z));
    let mut body = Body::new((across, depth, h, rho), &extra);
    let columns = across.max((radius / h).ceil() as usize);
    let top = first + 2 + (2.0 * radius / h).ceil() as usize + 2 + MARGIN;
    body.layout = GridLayout::new(2 * MARGIN + columns + 2, top, h as f32);
    let layout = body.layout.of_revolution(
        ax::Revolution::with_planes(MARGIN, &[(1, MARGIN, 1), (0, MARGIN + across, -1)])
            .with_tied_rows((first, top)),
    );
    let massive = body.massive;
    let sphere_start = massive + outline.len();
    let centre = sphere_start + polygon.len();
    let tied = mass / TAU;
    body.mass[centre] = tied;
    let outlines = vec![
        (massive..sphere_start).collect::<Vec<usize>>(),
        (sphere_start..centre).collect::<Vec<usize>>(),
    ];
    let corners = vec![outline_corners, polygon_corners];
    let settings = ConservingSettings {
        forecast_activation: true,
        ..ConservingSettings::default()
    };
    let wave = ((solid.bulk + solid.mu) / rho).sqrt();
    let dt = 1.5 * h / wave;
    let surface = (MARGIN + depth) as f64 * h;
    // A touch 2.37 of the first step ahead: not on a step's boundary.
    let touch = 2.37 * dt;
    let origin = MARGIN as f64 * h;
    let shift = surface + speed * touch - (origin + lowest);
    let mut x = body.reference.clone();
    for q in &mut x[sphere_start..] {
        q.y += shift;
    }
    let centre_start = x[centre].y;
    let start: Vec<DVec2> = (0..x.len())
        .map(|p| {
            if p == centre {
                DVec2::new(0.0, -speed)
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let field = start_velocity_field(
        &layout,
        &body.reference,
        (&body.mass, &body.v0, &start),
        &settings,
    );
    let mut v = start;
    v[massive..].copy_from_slice(&field[massive..]);
    let mut f = vec![DMat2::IDENTITY; x.len()];
    let mut solids = vec![*solid; x.len()];
    let penalty: Vec<f64> = (0..x.len())
        .map(|p| {
            if p < massive {
                0.0
            } else {
                solids[p].bulk + solids[p].mu
            }
        })
        .collect();
    let down = DVec2::new(0.0, -gravity);
    let momentum_of = |v: &[DVec2]| -> f64 {
        (0..massive).map(|p| body.mass[p] * v[p].y).sum::<f64>() + tied * v[centre].y
    };
    let momentum0 = momentum_of(&v);
    // The stored energy, the kinetic and gravity's against the start.
    let energy = |x: &[DVec2], v: &[DVec2], f: &[DMat2], solids: &[Elastic]| -> f64 {
        let kinetic: f64 = (0..massive)
            .map(|p| 0.5 * body.mass[p] * v[p].length_squared())
            .sum();
        let potential: f64 = (0..massive)
            .map(|p| -body.mass[p] * down.dot(x[p] - body.reference[p]))
            .sum::<f64>()
            - tied * down.y * (x[centre].y - centre_start);
        kinetic
            + 0.5 * tied * v[centre].length_squared()
            + potential
            + stored_energy(&body, solids, x, f)
    };
    let energy0 = energy(&x, &v, &f, &solids);
    let total: f64 = body.mass.iter().sum();
    let scale = tolerance_energy(energy0, down, [(total, &body.reference[..massive])]).sqrt();
    let control = StepControl::klapproth(tolerance, dt, Estimator::Acary);
    let wave_reaches = depth as f64 * h / wave;
    let (mut t, mut tau) = (0.0, dt);
    let mut cache: Option<LagrangianCache> = None;
    let mut samples = Vec::new();
    let (mut first_touch, mut last): (Option<f64>, f64) = (None, 0.0);
    let (mut plastic, mut heat_friction, mut allowed, mut ledger) = (0.0, 0.0, 0.0, 0.0f64);
    let (mut worst_momentum, mut crossing) = (0.0f64, 0.0f64);
    let (mut mirror_impulse, mut whole_residual, mut worst_closed, mut contact_net) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut steps, mut solves, mut rejected, mut stagnated) = (0, 0, 0, 0);
    let mut before = v[centre].y;
    let timing = std::env::var_os("EMERGE_PROBE_SPHERE_TIMING").is_some();
    loop {
        let clock = std::time::Instant::now();
        let step = controlled_step(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &body.mass,
                v0: &body.v0,
            },
            (&body.reference, &mut solids),
            (down, &mut tau),
            (
                ContactSet {
                    planes: &[],
                    pairs: &[],
                    outlines: &outlines,
                    corners: &corners,
                    friction,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
            (&control, scale),
        );
        let controlled = clock.elapsed().as_secs_f64();
        assert!(!step.at_floor, "drop, at {t}: at the estimator's floor");
        stagnated += step.stagnated;
        let piece = step.dt / step.kept.len() as f64;
        for (k, report) in step.kept.iter().enumerate() {
            let begun = t + k as f64 * piece;
            if report.contacts > 0 {
                first_touch.get_or_insert(begun);
                last = begun + piece;
            }
            plastic += report.plastic;
            heat_friction += report.friction;
            allowed += report.residual_work + report.contact_work_bound;
            mirror_impulse += report.mirror_force.y * piece;
            whole_residual += report.residual_force.y * piece;
            contact_net = contact_net.max(report.contact_net_force.y.abs());
        }
        t += step.dt;
        steps += 1;
        solves += step.solves;
        rejected += step.rejected;
        let balance = energy(&x, &v, &f, &solids) - energy0 + plastic + heat_friction;
        ledger = ledger.max(balance.abs());
        let gravity_impulse = -t * gravity * total;
        worst_closed = worst_closed.max(
            (momentum_of(&v) - momentum0 - gravity_impulse - mirror_impulse - whole_residual).abs(),
        );
        // Until a wave from the first touch can reach a mirror, which pushes.
        if first_touch.is_none_or(|touched| t <= touched + wave_reaches) {
            let impulse = gravity_impulse;
            worst_momentum = worst_momentum.max((momentum_of(&v) - momentum0 - impulse).abs());
        }
        let centre_now = x[centre];
        for &p in &outlines[0] {
            crossing = crossing.max(radius - (x[p] - centre_now).length());
        }
        let sample = DropSample {
            t,
            depth: centre_start - centre_now.y - speed * touch,
            speed: v[centre].y,
            force: TAU * tied * ((v[centre].y - before) / step.dt + gravity),
        };
        before = v[centre].y;
        let done = stop(&sample);
        if timing {
            println!(
                "  timing at step {steps}: controlled step {:.3} s, the probe's own reading {:.3} s, {} solves",
                controlled,
                clock.elapsed().as_secs_f64() - controlled,
                step.solves
            );
        }
        if steps % 50 == 0 {
            println!(
                "  at step {steps}: t {:.4e} s, depth {:+.4e} m, speed {:+.4e} m/s, force {:.3e} N, {solves} solves, {rejected} rejected, ledger {ledger:.2e} of {allowed:.2e}",
                sample.t, sample.depth, sample.speed, sample.force
            );
        }
        samples.push(sample);
        if done {
            break;
        }
    }
    Dropped {
        samples,
        first: first_touch.unwrap_or(f64::NAN),
        last,
        ledger,
        ledger_bound: allowed,
        energy0,
        plastic,
        friction: heat_friction,
        momentum: worst_closed / momentum0.abs(),
        momentum_window: worst_momentum / momentum0.abs(),
        contact_net: contact_net / momentum0.abs(),
        crossing: crossing / h,
        steps,
        solves,
        rejected,
        stagnated,
    }
}

/// `2 int_0^1 dy / sqrt(1 - y^(5/2))`, the contact time of a force going as the
/// depth to the power 3/2, over `d_m / v` (derived; the substitution `y = 1 - s^2`
/// clears the end's singularity).
fn hertz_time_factor() -> f64 {
    let panels = 4096;
    let sum: f64 = (0..panels)
        .map(|i| {
            let s = (i as f64 + 0.5) / panels as f64;
            let y: f64 = 1.0 - s * s;
            2.0 * s / (1.0 - y.powf(2.5)).sqrt()
        })
        .sum();
    2.0 * sum / panels as f64
}

/// A rigid sphere's slow drop on a neo-Hookean block against Hertz's law (Hale 1999,
/// Appendix C.1, eqs. C.1, C.3, C.7; Brilliantov et al. 1996, eq. A12). Deepest point
/// and contact time within 10 %, the force's power in the depth within 0.1 of 3/2 on
/// the loading path, the rebound within 5 %. `EMERGE_PROBE_SPHERE_PER_A`: cells in the
/// contact radius (default 4).
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn a_slow_sphere_follows_hertz() {
    let per_a: f64 = std::env::var("EMERGE_PROBE_SPHERE_PER_A")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4.0);
    let tolerance: f64 = std::env::var("EMERGE_PROBE_SPHERE_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    let block_radii: f64 = std::env::var("EMERGE_PROBE_SPHERE_BLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5.0);
    println!(
        "knob active: {per_a} cells in the contact radius, tolerance {tolerance:e}, block {block_radii} contact radii"
    );
    let radius: f64 = 31.75e-3;
    let mass = 1.0452;
    let speed = 0.141;
    let (young, nu, rho) = (1.7e6, 0.33, 1529.0);
    let (lambda, mu) = lame(young, nu);
    // Hertz, against a rigid sphere.
    let contact_modulus = young / (1.0 - nu * nu);
    let k = 4.0 / 3.0 * contact_modulus * radius.sqrt();
    let deepest = (5.0 * mass * speed * speed / (4.0 * k)).powf(0.4);
    let time = hertz_time_factor() * deepest / speed;
    let a = (radius * deepest).sqrt();
    let h = cell(a / per_a);
    let cells = (block_radii * a / h).ceil() as usize;
    println!(
        "Hertz: deepest {deepest:.4e} m ({:.3} R), contact radius {a:.4e} m ({:.3} R), contact time {time:.4e} s; h {h:.4e} m, block {cells} x {cells}",
        deepest / radius,
        a / radius
    );
    let block = Block {
        across: cells,
        depth: cells,
        h,
        solid: Elastic::neo_hookean(lambda, mu),
        rho,
    };
    let sphere = Sphere {
        radius,
        mass,
        speed,
    };
    let run = sphere_on_a_block(&block, &sphere, (0.0, 0.0, tolerance), |s| {
        s.speed > 0.0 && s.depth <= 0.0
    });
    let most = run
        .samples
        .iter()
        .map(|s| s.depth)
        .fold(f64::NEG_INFINITY, f64::max);
    // The force's power on the loading path, from a quarter of the depth to all of it.
    let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for s in &run.samples {
        if s.speed >= 0.0 {
            break;
        }
        if s.depth >= 0.25 * most && s.force > 0.0 {
            let (lx, ly) = (s.depth.ln(), s.force.ln());
            n += 1.0;
            sx += lx;
            sy += ly;
            sxx += lx * lx;
            sxy += lx * ly;
        }
    }
    let power = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let leaves = run.samples.last().expect("a step").speed;
    let free_flight = run.samples.first().expect("a step");
    let touch = free_flight.t - free_flight.depth / speed;
    println!(
        "dissipated: plastic {:.3e}, friction {:.3e} J/rad of the start's {:.3e}",
        run.plastic, run.friction, run.energy0
    );
    println!(
        "contact time from depth 0 to depth 0: {:+.3} % of Hertz",
        100.0 * ((run.samples.last().expect("a step").t - touch) / time - 1.0)
    );
    println!(
        "sphere on an elastic block: deepest {:+.3} % of Hertz, contact time {:+.3} %, power {power:.4}, rebound {:.5}, ledger {:.2e} J/rad of {:.2e} (bound {:.2e}), momentum against gravity, mirrors and residual {:.2e} (the former window's reading {:.2e}), contacts' net push {:.2e}, crossing {:.3e} cells, {} steps, {} solves, {} rejected, {} stagnated",
        100.0 * (most / deepest - 1.0),
        100.0 * ((run.last - run.first) / time - 1.0),
        leaves / speed,
        run.ledger,
        run.energy0,
        run.ledger_bound,
        run.momentum,
        run.momentum_window,
        run.contact_net,
        run.crossing,
        run.steps,
        run.solves,
        run.rejected,
        run.stagnated,
    );
    for s in run.samples.iter().step_by(run.samples.len().div_ceil(40)) {
        println!(
            "t {:.4e} depth {:+.4e} speed {:+.4e} force {:.4e} hertz {:.4e}",
            s.t,
            s.depth,
            s.speed,
            s.force,
            k * s.depth.max(0.0).powf(1.5)
        );
    }
    // The balances first: a red shape row must not hide them.
    assert!(run.ledger <= run.ledger_bound, "ledger {:e}", run.ledger);
    assert!(run.momentum <= 1e-9, "momentum {:e}", run.momentum);
    assert!((most / deepest - 1.0).abs() <= 0.10, "deepest {most:e}");
    assert!(
        ((run.last - run.first) / time - 1.0).abs() <= 0.10,
        "contact time {:e}",
        run.last - run.first
    );
    assert!((power - 1.5).abs() <= 0.1, "power {power}");
    assert!((leaves / speed - 1.0).abs() <= 0.05, "rebound {leaves}");
    assert_eq!(run.stagnated, 0, "Newton stagnated");
}
