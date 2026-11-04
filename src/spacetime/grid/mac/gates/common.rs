use super::*;

pub(super) const FRAME: f32 = 1.0 / 60.0;

/// The fluid material's own J bounds (`NewtonianFluidMaterial::update_particle`).
pub(super) const J_BOUNDS: (f32, f32) = (0.5, 2.0);

pub(super) const GRAVITIES: [f32; 3] = [0.38, 1.0, 2.5];

/// Where things happen, for diagnosis only: no criterion reads it.
#[derive(Default)]
pub(super) struct Probe {
    /// J reaching its bounds, by where the particle was: in a liquid cell
    /// of the tank, in an air cell of the tank, inside a wall.
    pub(super) j_at_bounds_by_place: [usize; 3],
    /// Particles moved back from beyond a wall (`substep`), and the
    /// deepest of them before the move, in ulps of the wall's coordinate.
    pub(super) reflections: u32,
    pub(super) deepest_reflection_ulps: f32,
    /// The largest reflection's depth over its budget (criteria, second
    /// amendment), and the budget's solver term of the current substep,
    /// `4 dt dx r`.
    pub(super) worst_reflection_ratio: f32,
    pub(super) solve_drift: f32,
    /// Particles leaving the tank: time, position before, velocity,
    /// distance to the nearest wall before.
    pub(super) crossings: Vec<(f32, Vec2, Vec2, f32)>,
    /// Wall-clock time per phase of the substep, in microseconds: P2G and
    /// gravity, level set, assembly for the frame record, projection
    /// (assembly, solve, update, extrapolation, walls), the two gathers,
    /// the particle update.
    pub(super) phase_us: [u128; 6],
    /// Cell whose pressure every substep records, with the time (the
    /// sound-speed tube).
    pub(super) watch: Option<usize>,
    pub(super) watch_history: Vec<(f32, f32)>,
    /// Particles within one cell of this x and between these two heights:
    /// their mean vertical velocity every substep (the plane shear wave).
    pub(super) watch_x: Option<(f32, f32, f32)>,
    /// The watch reads the x velocity instead of y (a P wave along x).
    pub(super) watch_longitudinal: bool,
    pub(super) watch_x_history: Vec<(f32, f32)>,
    /// Where and when the deepest reflection happened, and the particle's
    /// position before the step that took it there.
    pub(super) deepest_reflection_at: Option<(f32, Vec2)>,
    /// The conserving step's implicit steps: Newton and GMRES iterations
    /// summed, the most GMRES iterations in one step, the worst final residual
    /// and the steps that did not converge.
    pub(super) conserving: (u64, u64, u32, f64, u32),
    /// Conserving steps rejected and halved (step-size control).
    pub(super) rejected: u32,
    /// The stabilised mass's energy loss summed over the conserving steps.
    pub(super) stabilisation_loss: f64,
    /// The worst relative residual of the conserving steps' projection.
    pub(super) worst_projection: f64,
    /// The last conserving step's energy ledger and work
    /// (`total_lagrangian::StepReport`).
    pub(super) ledger: (f64, f64, f64),
    pub(super) work: (f64, f64),
    /// The most particles the conserving step held on a wall's plane at
    /// once.
    pub(super) contacts: usize,
    /// The conserving steps' wall time by phase (`total_lagrangian::StepReport`)
    /// and how many steps.
    pub(super) phases: [f64; 5],
    pub(super) steps: u32,
    /// The conserving steps' solves summed (the walls' passes).
    pub(super) passes: u64,
    /// The local error estimate of Zienkiewicz and Xie (1991), as Vetter,
    /// Wittel, Stoop and Herrmann write it for the Newmark family (arXiv
    /// 1111.5128, eq. 64), `|beta - 1/6| dt^2 |a_{n+1} - a_n|_inf / u_ref`,
    /// here with the midpoint's `beta = 1/4` and each step's average
    /// acceleration `(v_{n+1} - v_n) / dt` per particle in place of their
    /// predicted one, an adaptation of this engine's. Measured when
    /// `estimator_reference` (`u_ref`, cells) is positive: the last step's
    /// accelerations and the largest estimate since it was last reset.
    pub(super) estimator_reference: f32,
    pub(super) last_acceleration: Vec<Vec2>,
    pub(super) largest_estimate: f32,
    /// The bodies whose mass conditioning each conserving step measures
    /// before it moves (`total_lagrangian::mass_conditioning`): the union's and
    /// each body's own, the smallest since last reset; empty, none.
    pub(super) conditioning_bodies: Vec<std::ops::Range<usize>>,
    pub(super) conditioning: (f64, Vec<f64>),
    /// Below this conditioning a body's basis is dissected
    /// (`total_lagrangian::basis_report`), the worst since last reset kept with
    /// its conditioning; zero, never.
    pub(super) basis_threshold: f64,
    pub(super) basis_worst: Option<(f64, String)>,
    /// Since last reset: the largest grid velocity of a projection over the
    /// fastest particle's speed, and Newton's worst start
    /// (`total_lagrangian::StepReport`).
    pub(super) grid_speed_ratio: f64,
    pub(super) start_residual: f64,
}

/// Where the level set crosses zero going up column `i` from the tank
/// floor, by linear interpolation between cell centres.
pub(super) fn surface_height(phi: &Field2, i: usize, floor_cell: usize) -> Option<f32> {
    (floor_cell + 1..phi.nj()).find_map(|j| {
        let (below, above) = (phi.get(i, j - 1), phi.get(i, j));
        (below < 0.0 && above >= 0.0).then(|| (j as f32 - 0.5) + below / (below - above))
    })
}

/// Four particles per cell over `cells`, each at a random point of its
/// quarter of the cell, kept where `inside` holds; those within one cell
/// of a free surface (`surface` gives the distance and outward normal)
/// moved along the normal to the particle spacing from it (Zhu and
/// Bridson 4.2.1).
pub(super) fn seed(
    cells: (IVec2, IVec2),
    inside: impl Fn(Vec2) -> bool,
    surface: impl Fn(Vec2) -> (f32, Vec2),
    rng: &mut LcgRng,
) -> Vec<Vec2> {
    let spacing = 0.5;
    let mut x = Vec::new();
    for j in cells.0.y..cells.1.y {
        for i in cells.0.x..cells.1.x {
            for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let quarter = Vec2::new(0.25 + 0.5 * a as f32, 0.25 + 0.5 * b as f32);
                let jitter = Vec2::new(rng.next_f32() - 0.5, rng.next_f32() - 0.5) * spacing;
                let mut q = Vec2::new(i as f32, j as f32) + quarter + jitter;
                if !inside(q) {
                    continue;
                }
                let (distance, normal) = surface(q);
                if distance < 1.0 {
                    q += normal * (distance - spacing);
                }
                x.push(q);
            }
        }
    }
    x
}

/// `seed` for a body: a liquid's particles pulled to the particle spacing
/// from its free surface (Zhu and Bridson's technique, `seed`, which
/// smooths the surface their level set rebuilds), a solid's left where they
/// fall. A solid's particles are the quadrature points of its volume, and
/// the pull lays the two outer rows of a box onto one line (the particles a
/// quarter and three quarters of a spacing from a face both end half a
/// spacing from it), a degenerate quadrature (measured: at one frame per
/// substep Newton fails on it within four frames and a particle inverts).
pub(super) fn seed_body(
    solid: bool,
    cells: (IVec2, IVec2),
    inside: impl Fn(Vec2) -> bool,
    surface: impl Fn(Vec2) -> (f32, Vec2),
    rng: &mut LcgRng,
) -> Vec<Vec2> {
    if solid {
        seed(cells, inside, |_| (f32::INFINITY, Vec2::Y), rng)
    } else {
        seed(cells, inside, surface, rng)
    }
}

/// Distance to the nearest of a box's free sides and that side's normal.
pub(super) fn box_free_sides(q: Vec2, top: Option<f32>, right: Option<f32>) -> (f32, Vec2) {
    let mut best = (f32::INFINITY, Vec2::ZERO);
    if let Some(y) = top
        && y - q.y < best.0
    {
        best = (y - q.y, Vec2::Y);
    }
    if let Some(x) = right
        && x - q.x < best.0
    {
        best = (x - q.x, Vec2::X);
    }
    best
}

pub(super) fn disc_side(q: Vec2, centre: Vec2, radius: f32) -> (f32, Vec2) {
    let d = q - centre;
    let r = d.length();
    (radius - r, if r > 0.0 { d / r } else { Vec2::Y })
}

/// Every missed criterion of a run, printed as it is found and asserted
/// together at the end, so one run shows them all.
pub(super) struct Report {
    pub(super) failures: Vec<String>,
}

impl Report {
    pub(super) fn new() -> Self {
        Self {
            failures: Vec::new(),
        }
    }

    pub(super) fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        if !ok {
            let line = what();
            println!("  FAIL {line}");
            self.failures.push(line);
        }
    }

    pub(super) fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} criteria missed:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

/// Totals over a run, for the cost lines.
#[derive(Default)]
pub(super) struct Cost {
    pub(super) frames: u32,
    pub(super) substeps: u32,
    pub(super) max_substeps: u32,
    pub(super) iterations: u64,
    pub(super) solves: u64,
    pub(super) max_iterations: u32,
    pub(super) capped: u32,
    pub(super) largest_travel: f32,
}

impl Cost {
    pub(super) fn add(&mut self, frame: &Frame) {
        self.frames += 1;
        self.substeps += frame.substeps;
        self.max_substeps = self.max_substeps.max(frame.substeps);
        self.iterations += frame.iterations.iter().map(|&i| i as u64).sum::<u64>();
        self.solves += frame.iterations.len() as u64;
        let most = frame.iterations.iter().copied().max().unwrap_or(0);
        self.max_iterations = self.max_iterations.max(most);
        self.capped += frame.capped;
        self.largest_travel = self.largest_travel.max(frame.largest_travel);
    }

    pub(super) fn print(&self, label: &str, wall_s: f32, simulated_s: f32) {
        println!(
            "  cost {label}: {:.1} substeps/frame (max {}), CG {:.1} it/solve (max {}), \
             {} capped, largest travel {:.3} cell, {:.0} ms per simulated s ({})",
            self.substeps as f32 / self.frames.max(1) as f32,
            self.max_substeps,
            self.iterations as f32 / self.solves.max(1) as f32,
            self.max_iterations,
            self.capped,
            self.largest_travel,
            1000.0 * wall_s / simulated_s,
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
        );
    }
}

/// The seeds of the dam break and the drop into a pool.
pub(super) fn dam_break_seed() -> Vec<Vec2> {
    let mut rng = LcgRng::new(3);
    seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    )
}

pub(super) fn drop_into_pool_seed() -> Vec<Vec2> {
    let mut rng = LcgRng::new(5);
    let mut x = seed(
        (IVec2::new(4, 4), IVec2::new(68, 19)),
        |_| true,
        |q| box_free_sides(q, Some(19.0), None),
        &mut rng,
    );
    let (centre, radius) = (Vec2::new(36.0, 44.0), 5.0);
    x.extend(seed(
        (IVec2::new(30, 38), IVec2::new(42, 50)),
        |q| (q - centre).length() < radius,
        |q| disc_side(q, centre, radius),
        &mut rng,
    ));
    x
}

/// `E / rho` (m^2/s^2) giving plane-strain shear speed `c_s` at `nu`.
pub(super) fn e_over_rho_for_shear_speed(c_s: f32, nu: f32) -> f32 {
    2.0 * (1.0 + nu) * c_s * c_s
}

/// Kinetic, potential, shear and volume energy, the floor as zero: the
/// drop's measure for any row.
pub(super) fn row_energy(scene: &Scene) -> f64 {
    let floor = scene.tank.0.y;
    (0..scene.x.len())
        .map(|p| {
            let shear = scene
                .shear
                .map_or(0.0, |mu| shear_energy(scene.deformation[p], mu));
            let volume = scene
                .sound_speed2
                .map_or(0.0, |kappa| 0.5 * kappa * (scene.j(p) - 1.0).powi(2));
            f64::from(
                scene.mass[p]
                    * (0.5 * scene.v[p].length_squared() + scene.g() * (scene.x[p].y - floor))
                    + scene.v0[p] * (shear + volume),
            )
        })
        .sum()
}

/// A conserving run's fixed substep: the explicit P wave limit `0.5 dx /
/// c_p`, four times it, or one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Substep {
    PWave,
    FourPWaves,
    Frame,
}

impl Substep {
    /// From a probe's setting: `p`, `4p` or `frame`.
    pub(super) fn parse(name: &str) -> Option<Self> {
        match name {
            "p" => Some(Self::PWave),
            "4p" => Some(Self::FourPWaves),
            "frame" => Some(Self::Frame),
            _ => None,
        }
    }

    /// Its length in seconds on cells of `dx` for a P wave of `c_p`.
    pub(super) fn length(self, dx: f32, c_p: f32) -> f32 {
        match self {
            Self::PWave => 0.5 * dx / c_p,
            Self::FourPWaves => 2.0 * dx / c_p,
            Self::Frame => FRAME,
        }
    }
}

impl std::fmt::Display for Substep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PWave => "0.5 dx / c_p",
            Self::FourPWaves => "2 dx / c_p",
            Self::Frame => "one frame",
        })
    }
}

/// A square solid block of `side` cells centred on `centre`, seeded
/// (`seed_body`) from `seed_value`: the per-body probes' bodies.
pub(super) fn clay_block(centre: Vec2, side: f32, seed_value: u32) -> Vec<Vec2> {
    let (lo, hi) = (
        centre - Vec2::splat(0.5 * side),
        centre + Vec2::splat(0.5 * side),
    );
    seed_body(
        true,
        (lo.floor().as_ivec2(), hi.ceil().as_ivec2()),
        |q| q.cmpge(lo).all() && q.cmple(hi).all(),
        |_| (f32::INFINITY, Vec2::Y),
        &mut LcgRng::new(seed_value),
    )
}

/// One body's energy (`row_energy` over its particles `r`), kinetic
/// energy, centre height in m and vertical centre speed in m/s.
pub(super) fn body_measure(scene: &Scene, r: std::ops::Range<usize>) -> [f64; 4] {
    let floor = scene.tank.0.y;
    let mu = scene.shear.unwrap_or(0.0);
    let kappa = scene.sound_speed2.unwrap_or(0.0);
    let (mut energy, mut kinetic, mut mass) = (0.0f64, 0.0f64, 0.0f64);
    let (mut y, mut vy) = (0.0f64, 0.0f64);
    for p in r {
        let m = f64::from(scene.mass[p]);
        let k = 0.5 * m * f64::from(scene.v[p].length_squared());
        let stored = f64::from(
            scene.v0[p]
                * (shear_energy(scene.deformation[p], mu)
                    + 0.5 * kappa * (scene.j(p) - 1.0).powi(2)),
        );
        energy += k + m * f64::from(scene.g() * (scene.x[p].y - floor)) + stored;
        kinetic += k;
        mass += m;
        y += m * f64::from(scene.x[p].y);
        vy += m * f64::from(scene.v[p].y);
    }
    let cell = f64::from(scene.cell_m);
    [energy, kinetic, y / mass * cell, vy / mass * cell]
}

/// Quadrature points of the ring `Ri <= R <= Ro` about `centre`: every half
/// cell of the grid (both lattices' B-splines are polynomials on it) wholly
/// inside the ring takes a tensor Gauss-Legendre rule (3x3 by default,
/// `EMERGE_PROBE_VORTEX_GAUSS`); a half
/// cell the circles cut is split in four, down to `depth` levels, and the
/// rule's points of the last level are kept where they lie in the ring.
/// Composite Gauss quadrature over an adaptive subdivision of the cut
/// cells; its geometric error is the run's measured area error.
pub(super) fn ring_quadrature(
    centre: glam::DVec2,
    (inner, outer): (f64, f64),
    side: usize,
    depth: u32,
) -> (Vec<glam::DVec2>, Vec<f64>) {
    use glam::DVec2;
    // `EMERGE_PROBE_VORTEX_GAUSS` points per direction (default 3).
    let order: usize = std::env::var("EMERGE_PROBE_VORTEX_GAUSS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let (nodes, weights) = gauss_legendre(order);
    let (mut points, mut volumes) = (Vec::new(), Vec::new());
    let mut stack: Vec<(DVec2, f64, u32)> = (0..2 * side)
        .flat_map(|j| (0..2 * side).map(move |i| (DVec2::new(i as f64, j as f64) * 0.5, 0.5, 0)))
        .collect();
    while let Some((lo, size, level)) = stack.pop() {
        let nearest = DVec2::new(
            centre.x.clamp(lo.x, lo.x + size),
            centre.y.clamp(lo.y, lo.y + size),
        );
        let near = (nearest - centre).length();
        let far = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .iter()
            .map(|&(a, b)| (lo + size * DVec2::new(a, b) - centre).length())
            .fold(0.0f64, f64::max);
        if far <= inner || near >= outer {
            continue;
        }
        let whole = near >= inner && far <= outer;
        if whole || level >= depth {
            for (b, wb) in nodes.iter().zip(&weights) {
                for (a, wa) in nodes.iter().zip(&weights) {
                    let q = lo + size * DVec2::new(*a, *b);
                    let r = (q - centre).length();
                    if whole || (inner..=outer).contains(&r) {
                        points.push(q);
                        volumes.push(size * size * wa * wb);
                    }
                }
            }
        } else {
            let half = 0.5 * size;
            for (a, b) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                stack.push((lo + half * DVec2::new(a, b), half, level + 1));
            }
        }
    }
    (points, volumes)
}
