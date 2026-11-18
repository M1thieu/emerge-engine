pub mod body_state;
#[cfg(any(test, feature = "research-diagnostics"))]
mod boundary_diagnostics;
pub(crate) mod cfl;
pub mod config;
pub mod density;
pub(crate) mod grid_units;
pub mod handle;
mod implicit_corotated;
mod lifecycle;
mod particles;
mod projection;
mod queries;
pub mod spatial_hash;
mod step;
pub mod total_lagrangian_solids;

pub use body_state::{BodyState, body_state_of, region_body_state_of};
#[cfg(any(test, feature = "research-diagnostics"))]
pub use boundary_diagnostics::{
    AcceptedBoundaryImpulseLedger, BoundaryImpulseExperiment, BoundaryImpulseReport,
    BoundaryNodeImpulseLedger,
};
pub use config::{SimConfig, SpawnRegion};
pub use density::compute_density_grid;
pub use handle::{MaterialHandle, ParticleGroup};
pub use particles::{HydrostaticState, hydrostatic_state};
// Only consumed by systems::gpu's own CFL scan -- unused (and correctly
// warned about) in a build without that feature.
#[cfg(feature = "gpu")]
pub(crate) use cfl::{
    affine_cfl_speed_contribution, cfl_bound, deformation_gradient_ode_dt_bound, is_near_wall,
    shock_viscosity_dt_bound, single_particle_instability_dt_bound,
};

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use spatial_hash::SpatialHash;

use glam::{Mat2, Vec2};

use crate::rod::Rod;
use crate::thermodynamics::{
    CosseratField, GranularFluidityField, ScalarDiffusionField, ThermalDiffusion,
};
use crate::{boundary::BoundaryCondition, fields::Field, materials::registry::MaterialRegistry};
use crate::{
    grid::Grid,
    particle::{Particle, Particles},
};

type PhaseRule = Box<dyn Fn(&Particle) -> Option<u32> + Send + Sync>;

/// The CPU MLS-MPM solver -- the crate's central type. Owns every particle,
/// the shared background grid, the material registry, boundary conditions,
/// force fields, and optional rod/grain/thermal subsystems, and advances
/// all of them together one frame at a time.
///
/// Construct with [`Simulation::new`] (a [`SimConfig`] plus an initial
/// [`SpawnRegion`] of particles) or [`Simulation::empty`] (no initial
/// particles, add bodies afterward via [`Simulation::add_body`]). Advance
/// simulated time with [`Simulation::step`], which always advances exactly
/// `config.dt` regardless of how many adaptive CFL substeps that requires
/// internally. Read state back via [`Simulation::particles`]/
/// `particles_mut`, [`Simulation::material_state`]/`region_state`, or
/// [`Simulation::particles_near`].
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::{SimConfig, Simulation, SpawnRegion};
///
/// let config = SimConfig::standard(64, 0.05, glam::Vec2::new(0.0, -0.05));
/// let mut sim = Simulation::new(config, SpawnRegion::for_sim(&config));
/// sim.step();
/// ```
pub struct Simulation {
    config: SimConfig,
    particles: Particles,
    /// Partition boundary: particles[0..active_count] are active, [active_count..N] sleeping.
    /// P2G / G2P only visit [0..active_count]. Maintained by sleep_particle / wake_particle.
    active_count: usize,
    /// Maps user_tag → physical indices of all particles with that tag.
    /// HashSet gives O(1) insert/remove on every sleep/wake swap.
    tag_index: HashMap<u32, HashSet<usize>>,
    /// Monotonically increasing counter -- next tag issued by add_body.
    next_tag: u32,
    grid: Grid,
    materials: MaterialRegistry,
    /// Per-particle banked frictional heating, kelvin -- see
    /// `spacetime::transfer::friction_heat` for why a rise this small has to
    /// be accumulated before it can be added to an `f32` temperature.
    /// Empty and untouched in every scene where nothing rubs.
    friction_heat_debt: Vec<f32>,
    /// The initial body's lattice spacing and material when it was spawned
    /// without a stated mass: its particles take their material's own
    /// density once that material is set (`apply_declared_density`), as
    /// `add_body` gives a later body at spawn.
    initial_mass_pending: Option<(config::SpawnRegion, u32)>,
    boundaries: Vec<Box<dyn BoundaryCondition>>,
    /// True while `boundaries` still holds only the default `SlipBoundary`
    /// inserted at construction (see `empty`/`new`). The first
    /// `add_boundary_condition` call clears it instead of stacking on top:
    /// the default's zero-friction no-penetration correction runs first every
    /// substep and zeroes the into-wall velocity before a user's
    /// `FrictionBoundary` sees it, which silently cancels that friction (a
    /// grain coasted at constant velocity for 500,000+ steps on a
    /// `FrictionBoundary(2, 0.7)` floor).
    boundaries_are_default: bool,
    /// Optional directional (setae-style) friction for the multi-field contact
    /// "grip" field -- see `DirectionalContactGrip`'s doc. `None` (default) keeps
    /// the existing plain symmetric `contact_friction` behavior; every scene that
    /// never opts in is completely unaffected.
    contact_grip: Option<std::sync::Arc<crate::boundary::DirectionalContactGrip>>,
    /// Friction coefficients set per pair of surfaces
    /// (`set_surface_friction`), read by the total Lagrangian path.
    surface_friction: Vec<(crate::boundary::Surface, crate::boundary::Surface, f64)>,
    /// Pairs whose friction depends on the sense of sliding along a
    /// material's axis (`set_surface_setae`), read by the total Lagrangian
    /// path.
    surface_setae: Vec<SurfaceSetae>,
    force_fields: Vec<(String, Box<dyn Field>)>,
    thermal: Option<ThermalDiffusion>,
    /// Scalar diffusion fields (pheromone, nutrients, morphogen) -- run automatically each substep.
    scalar_fields: Vec<ScalarDiffusionField>,
    /// Simulation time accumulated across this `step()`'s substeps, waiting to
    /// be handed to the diffusion operators in ONE application -- see
    /// `do_substep`'s own note for the stability derivation that makes this
    /// correct (and why per-substep sub-cycling was ~31,000x redundant).
    pending_diffusion_dt: f32,
    /// Index of the substep currently running within this `step()`. Lets
    /// per-FRAME work (validation scans, phase rules) run once instead of
    /// once per substep -- see `do_substep`'s own notes.
    substep_index_in_frame: u32,
    /// Nonlocal Granular Fluidity field (see `energy::thermodynamics::
    /// granular_fluidity` module doc) -- `None` (default) for every scene
    /// that doesn't opt in, same zero-cost-when-unused property `thermal`
    /// already has.
    granular_fluidity: Option<GranularFluidityField>,
    /// Persistent per-particle gathered `g`, indexed by particle -- read by
    /// G2P (`G2PParams::nonlocal_fluidity`), written by the granular-
    /// fluidity pass at the end of the SAME substep (one-substep lag, same
    /// convention thermal/scalar diffusion already use). Empty when
    /// `granular_fluidity` is `None`.
    granular_fluidity_g: Vec<f32>,
    /// Cosserat micro-rotation field (see `energy::thermodynamics::
    /// cosserat_field` module doc) -- `None` (default) for every scene that
    /// doesn't opt in, same zero-cost-when-unused property `granular_fluidity`
    /// already has. Real grid-level angular-momentum channel, added to close
    /// the loop `matter::materials::granular::cosserat`'s kinematics module leaves open.
    cosserat: Option<CosseratField>,
    /// Persistent per-particle gathered micro-rotation, one-substep-lag
    /// convention matching `granular_fluidity_g` exactly.
    cosserat_omega: Vec<f32>,
    /// Persistent per-particle gathered micro-curvature (kappa = grad(omega_c)),
    /// same one-substep-lag convention, read by G2P (`G2PParams::
    /// cosserat_curvature`), written by the Cosserat pass at the end of the
    /// SAME substep.
    cosserat_curvature: Vec<glam::Vec2>,
    frame_index: u64,
    /// Fine-substep hold for `SimConfig::fluid_step_retry_enabled`:
    /// `(held_dt, substeps_remaining)`. A one-off retry (see
    /// `do_substep_with_retry`) is not enough: the next substep re-derives its
    /// size from the CFL scan and forgets the rejection, so a sustained
    /// near-wall compression repeats a reject-shrink-forget cycle every
    /// substep (a per-substep retry swept across several thresholds never
    /// matched a globally tightened CFL coefficient). Once a retry fires,
    /// `choose_substep_dt` is capped to `held_dt` for `substeps_remaining`
    /// further substeps. `None` = no hold active (always, in scenes that
    /// never enable the feature).
    fluid_sticky_fine_dt: Option<(f32, u32)>,
    /// Research diagnostic, `EMERGE_TRACK_BOUNDARY_BIAS` (see
    /// `transfer::diagnose_particle_divergence_decomposition`). Set by
    /// `do_substep` right after P2G with the `dt` that scatter used, filled
    /// with the final `tr(C)` at the end of the same call, then printed by
    /// `do_substep_with_retry` after its retry loop exits, so it reflects the
    /// accepted attempt. `None` unless that switch is set in a
    /// `research-diagnostics` build (see `research_switch`). Fields:
    /// `(tracked_index, trace_translation, trace_affine, trace_stress,
    /// trace_final, dt_used)`.
    pending_divergence_diagnostic: Option<(usize, f32, f32, f32, f32, f32)>,
    /// Structural wall-bounce impulse ledger (`boundary_diagnostics`), set by
    /// `enable_boundary_impulse_diagnostic`; absent from a default build.
    #[cfg(any(test, feature = "research-diagnostics"))]
    boundary_impulse_diagnostic: Option<boundary_diagnostics::BoundaryImpulseDiagnostic>,
    /// Max particle speed measured by the previous `choose_substep_dt` call
    /// (lagged one substep: a substep's max speed is known only after its CFL
    /// fold). Feeds the near-wall gate's Mach-relative compression threshold
    /// (`SimConfig::fluid_near_wall_compression_mach_margin`). `0.0` at
    /// construction, so the first substep's gate is maximally sensitive
    /// (threshold 0.0): cautious for one substep, corrected as soon as the
    /// first fold measures a speed.
    last_max_particle_speed: f32,
    last_step_dt: f32,
    last_substeps: usize,
    last_j_projection_count: usize,
    last_sim_time_dropped: f32,
    last_timing: crate::diagnostics::StepTiming,
    /// `SimConfig::spatial_sort_enabled` cache: computed ONCE per outer
    /// `step()` call (not per substep -- measured: recomputing this
    /// O(N log N) sort every substep cost MORE than the P2G cache-locality
    /// win it was meant to provide, see `spatial_sort_order`'s doc)
    /// and reused across every substep within that call. Particle positions
    /// shift only slightly substep-to-substep, so a step-stale order still
    /// captures most of the locality benefit. Empty when the feature
    /// is off -- zero allocation cost in the default case.
    cached_spatial_sort_order: Vec<usize>,
    /// Automatic phase transition rules, evaluated every substep.
    phase_rules: Vec<PhaseRule>,
    /// Spatial hash over active particles. Turns O(N) radius queries into
    /// O(candidates_in_neighborhood) for `particles_near`/`count_near`/
    /// `particles_knn`/`region_state`.
    ///
    /// Lazily rebuilt: `step()` only marks it dirty (`spatial_hash_dirty`).
    /// Rebuilding every step measured at 16.4% of a step's time even in
    /// scenes that never query it; `GpuSimulation` rebuilds lazily the same
    /// way. `RefCell` because the four queries take `&self` (public API LP
    /// uses) but may trigger a rebuild: a read-only, lazily computed cache.
    /// Mutations outside `step()` that change particle count or positions
    /// (`add_body`, `remove_where`, `split_particles`, construction) rebuild
    /// eagerly and clear the flag, so a query between two `step()` calls
    /// always sees fresh data.
    spatial_hash: RefCell<SpatialHash>,
    /// See `spatial_hash`'s doc. `true` right after `step()` runs a
    /// substep loop (positions moved, hash is stale); cleared by
    /// `ensure_spatial_hash_fresh` the first time any query method is
    /// actually called. Never true after an eager rebuild (spawn/remove/
    /// split), since those clear it immediately after rebuilding.
    spatial_hash_dirty: Cell<bool>,
    /// Discrete elastic rods (Cosserat-rod family, `spacetime::rod`) sharing
    /// this simulation's own MPM grid -- see `step.rs`'s `do_substep` for the
    /// real scatter/gather insertion points. Empty for every scene that
    /// never calls `add_rod`/`with_rod` (zero-cost: 0-iteration loops).
    rods: Vec<Rod>,
    /// Discrete-element grain populations (`spacetime::grains`) sharing this
    /// simulation's own MPM grid -- cited elastic-plastic rolling
    /// resistance (Cundall & Strack 1979 / Luding 2008 / Ai et al. 2011),
    /// see `grains::coupling` for the scatter/gather insertion points
    /// (mirroring `rods` above). Empty for every scene that never
    /// calls `add_grain_population`/`with_grain_population` (zero-cost:
    /// 0-iteration loops). Nothing adds grains automatically: a caller
    /// decides where, as with `add_rod`, optionally guided by the
    /// `grains::oracle` packing-fraction signal and
    /// `enrich_region_into_grain`.
    grain_populations: Vec<crate::grains::population::GrainPopulation>,
    /// The total Lagrangian solid path's state (`SimConfig::total_lagrangian`).
    total_lagrangian: Option<total_lagrangian_solids::TotalLagrangianState>,
    /// Each body's contact outline (points and the indices of its corners,
    /// `spawn_outline`) and material, recorded at its spawn for that path
    /// (empty when it is off).
    total_lagrangian_outlines: Vec<total_lagrangian_solids::SpawnedBody>,
    /// Scratch buffer for wake/sleep candidates -- pre-allocated once, cleared per substep.
    /// Pattern from ziran2020 MpmSimulationBase: scratch_xp/scratch_vp member fields.
    scratch_indices: Vec<usize>,
}

impl std::fmt::Debug for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Simulation")
            .field("grid_res", &self.config.grid_res)
            .field("particles_total", &self.particles.len())
            .field("active", &self.active_count)
            .field("frame", &self.frame_index)
            .finish_non_exhaustive()
    }
}

pub(crate) fn initialize_particles(
    config: &SimConfig,
    spawn: SpawnRegion,
    rng: &mut LcgRng,
) -> Vec<Particle> {
    use crate::solver::config::SpawnShape;
    // Mass per particle from mass per CELL AREA: a lattice at `spacing` cells
    // packs `1/spacing^2` particles into each cell, so the grid sees
    // `grid_density` regardless of how finely the region is discretized.
    // Without the `spacing^2`, refining a scene silently multiplied its grid
    // density -- and therefore gravity relative to stiffness -- by
    // `1/spacing^2`. See `SimConfig::grid_density`.
    let mass = spawn
        .mass_override
        .unwrap_or(config.grid_density * spawn.spacing * spawn.spacing);
    let mut particles = Vec::new();
    let half = spawn.box_size.as_vec2() * 0.5;
    let min = spawn.box_center - half;
    let max = spawn.box_center + half;

    // The precise path (`SimConfig::total_lagrangian`) centres the lattice
    // at the sub-cells' centres, `min + s/2 + k s` (issue #42): each
    // particle's own area `s^2` tiles the requested box exactly, so the
    // body's mass, its volume and its outline describe the same region,
    // centred where it was asked. Laid from `min` itself, every body sat
    // half a spacing below and left of its centre; a flat block dropped
    // flat on that path then took the floor's impulse a quarter cell off its
    // centre of mass and left the bounce turning (measured: 1.2e-4 then
    // 1.8e-2 kg m^2/s per m).
    //
    // The explicit path keeps its original lattice, `min + k s`: it is
    // maintenance only, and its scenes and tests were measured on that
    // lattice. Centring is not neutral there: at an odd particle count per
    // cell axis a centred lattice lands on grid nodes, and the explicit
    // path then under-predicts a column's sag under its own weight (0.352
    // of analytic at spacing 1, measured in issue #42).
    let (x0, y0) = if config.total_lagrangian.is_some() {
        (min.x + 0.5 * spawn.spacing, min.y + 0.5 * spawn.spacing)
    } else {
        (min.x, min.y)
    };
    let mut points = Vec::new();
    let mut i = x0;
    while i < max.x {
        let mut j = y0;
        while j < max.y {
            let pos = Vec2::new(i, j);

            // Apply shape mask -- skip particles outside the disk if disk shape is active.
            let inside = match spawn.shape {
                SpawnShape::Box => true,
                SpawnShape::Disk { radius } => (pos - spawn.box_center).length() <= radius,
            };

            if inside {
                points.push((pos, mass));
            }

            j += spawn.spacing;
        }
        i += spawn.spacing;
    }
    #[cfg(all(test, feature = "experimental"))]
    let points = knot_gauss_points(&spawn, (min, max), mass / (spawn.spacing * spawn.spacing))
        .unwrap_or(points);

    for (pos, mass) in points {
        let jitter_mag = spawn.position_jitter * spawn.spacing;
        let jx = (rng.next_f32() - 0.5) * 2.0 * jitter_mag;
        let jy = (rng.next_f32() - 0.5) * 2.0 * jitter_mag;
        let jittered_pos = pos + Vec2::new(jx, jy);
        let random = Vec2::new(rng.next_f32(), rng.next_f32());
        let velocity = (random - Vec2::splat(0.5)) * spawn.initial_velocity_scale;
        particles.push(Particle {
            x: jittered_pos,
            v: velocity,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient: spawn.initial_deformation_gradient,
            mass,
            initial_volume: config.default_initial_volume,
            volume: config.default_initial_volume,
            density: mass / config.default_initial_volume,
            material_id: spawn.material_id,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening: 0.0,
            log_volume_strain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            scalar_field: 0.0,
            internal_pressure: 0.0,
        });
    }

    particles
}

/// A test-only knob (`EMERGE_PROBE_KNOT_GAUSS`):
/// a box body's particles at two Gauss points per piece of its interval cut
/// at the quadratic B-splines' knots (`k + 1/2` in grid units, the total
/// Lagrangian basis having its nodes at the integers), each carrying the
/// region's mass per cell area times its weight area. The weak form is then
/// integrated exactly where the body cuts the basis (the lattice's
/// quarter points leave a first-cell error at free surfaces and on an axis
/// that does not shrink with the cell). `None`: the lattice.
#[cfg(all(test, feature = "experimental"))]
fn knot_gauss_points(
    spawn: &SpawnRegion,
    (min, max): (Vec2, Vec2),
    mass_per_area: f32,
) -> Option<Vec<(Vec2, f32)>> {
    use crate::solver::config::SpawnShape;
    if std::env::var_os("EMERGE_PROBE_KNOT_GAUSS").is_none()
        || !matches!(spawn.shape, SpawnShape::Box)
    {
        return None;
    }
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| println!("knob active: knot-span Gauss points at spawn"));
    Some(knot_gauss_layout((min, max), mass_per_area))
}

/// The points and masses of `knot_gauss_points` for the box `[min, max]`.
#[cfg(all(test, feature = "experimental"))]
fn knot_gauss_layout((min, max): (Vec2, Vec2), mass_per_area: f32) -> Vec<(Vec2, f32)> {
    let along = |a: f32, b: f32| -> Vec<(f64, f64)> {
        let (a, b) = (f64::from(a), f64::from(b));
        let mut cuts = vec![a];
        let mut k = (a - 0.5).floor() + 1.0;
        while k + 0.5 < b {
            if k + 0.5 > a {
                cuts.push(k + 0.5);
            }
            k += 1.0;
        }
        cuts.push(b);
        cuts.windows(2)
            .flat_map(|w| {
                let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
                [-1.0, 1.0].map(|s| (mid + s * half / 3f64.sqrt(), half))
            })
            .collect()
    };
    let (xs, ys) = (along(min.x, max.x), along(min.y, max.y));
    xs.iter()
        .flat_map(|&(x, wx)| {
            ys.iter().map(move |&(y, wy)| {
                (
                    Vec2::new(x as f32, y as f32),
                    mass_per_area * (wx * wy) as f32,
                )
            })
        })
        .collect()
}

#[cfg(all(test, feature = "experimental"))]
mod knot_gauss_tests {
    use super::*;

    /// The layout's mass and area are the
    /// box's, for a box cut by the knots and one starting on a knot; every
    /// point lies inside it.
    #[test]
    fn the_knot_gauss_layout_carries_the_box_exactly() {
        for (min, max) in [
            (Vec2::new(10.0, 3.0), Vec2::new(18.0, 7.0)),
            (Vec2::new(10.5, 3.25), Vec2::new(14.5, 5.75)),
        ] {
            let points = knot_gauss_layout((min, max), 2.0);
            let area = (max - min).x * (max - min).y;
            let mass: f32 = points.iter().map(|p| p.1).sum();
            assert!(
                (mass / (2.0 * area) - 1.0).abs() < 1e-6,
                "{mass} vs {}",
                2.0 * area
            );
            assert!(
                points
                    .iter()
                    .all(|(x, _)| x.cmpgt(min).all() && x.cmplt(max).all())
            );
        }
    }
}

/// One pair set by `Simulation::set_surface_setae`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SurfaceSetae {
    pub(crate) material: u32,
    pub(crate) other: crate::boundary::Surface,
    pub(crate) forward: f64,
    pub(crate) backward: f64,
    /// Unit, in the material's reference configuration.
    pub(crate) axis: glam::DVec2,
}

#[derive(Debug)]
pub(crate) struct LcgRng {
    state: u32,
}

impl LcgRng {
    pub(crate) const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub(crate) const fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }
}

#[cfg(test)]
mod spawn_lattice_tests {
    use super::total_lagrangian_solids::TotalLagrangianConfig;
    use super::{LcgRng, SimConfig, SpawnRegion, initialize_particles};
    use glam::{IVec2, Vec2};

    fn spawns(config: &SimConfig, spacing: f32) -> [SpawnRegion; 2] {
        let base = SpawnRegion::for_sim(config)
            .at(Vec2::new(32.0, 20.0))
            .spacing(spacing);
        [base.box_of(IVec2::new(12, 6)), base.disk(5.5)]
    }

    fn positions(config: &SimConfig, spawn: SpawnRegion) -> Vec<Vec2> {
        let mut rng = LcgRng::new(spawn.rng_seed);
        initialize_particles(config, spawn, &mut rng)
            .iter()
            .map(|p| p.x)
            .collect()
    }

    /// Criterion: with `total_lagrangian` `None` (the explicit path) the
    /// spawn positions are exactly the original lattice
    /// `x = min + a s`, `y = min + b s` for every integer `a, b >= 0` with
    /// the point below `max`, column by column, masked by the shape; with
    /// `total_lagrangian` `Some` (the precise path) the lattice is centred,
    /// `min + s/2 + k s`, and the mean position equals the box centre to
    /// 1e-6 (issue #42's criterion). Box and Disk at spacing 0.5, 0.25 and
    /// 1.0, chosen so every lattice point is exact in `f32`.
    #[test]
    fn spawn_lattice_is_centred_only_on_the_precise_path() {
        let explicit = SimConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3);
        let mut precise = explicit;
        precise.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
        for spacing in [0.5_f32, 0.25, 1.0] {
            for spawn in spawns(&explicit, spacing) {
                let half = spawn.box_size.as_vec2() * 0.5;
                let (min, max) = (spawn.box_center - half, spawn.box_center + half);
                let mut expected = Vec::new();
                let mut a = 0_u32;
                while min.x + a as f32 * spacing < max.x {
                    let mut b = 0_u32;
                    while min.y + b as f32 * spacing < max.y {
                        let p = min + Vec2::new(a as f32, b as f32) * spacing;
                        let inside = match spawn.shape {
                            crate::solver::config::SpawnShape::Box => true,
                            crate::solver::config::SpawnShape::Disk { radius } => {
                                (p - spawn.box_center).length() <= radius
                            }
                        };
                        if inside {
                            expected.push(p);
                        }
                        b += 1;
                    }
                    a += 1;
                }
                assert!(!expected.is_empty());
                assert_eq!(
                    positions(&explicit, spawn),
                    expected,
                    "explicit path, spacing {spacing}, {:?}: the original lattice",
                    spawn.shape
                );

                let centred = positions(&precise, spawn);
                assert!(!centred.is_empty());
                let n = centred.len() as f64;
                let mean_x = centred.iter().map(|p| f64::from(p.x)).sum::<f64>() / n;
                let mean_y = centred.iter().map(|p| f64::from(p.y)).sum::<f64>() / n;
                assert!(
                    (mean_x - f64::from(spawn.box_center.x)).abs() <= 1e-6
                        && (mean_y - f64::from(spawn.box_center.y)).abs() <= 1e-6,
                    "precise path, spacing {spacing}, {:?}: mean ({mean_x}, {mean_y}) \
                     against the box centre {}",
                    spawn.shape,
                    spawn.box_center
                );
            }
        }
    }
}
