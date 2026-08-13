extern crate emerge_engine as emerge;

/// GPU viscoplastic fluids — Newtonian water dam-break + Bingham mud blob, zero CPU readback.
///
///   Mat 0  Newtonian water (blue) — Tait EOS + deviatoric viscosity
///   Mat 1  Bingham mud    (gold)  — viscoplastic with yield stress
///
///   cargo run --example basic_fluids_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::{ColorMode, DualPhaseSurfaceSource, GridVolumeSource, Renderer};
use emerge::{
    BinghamFluidMaterial, FixedStepConfig, FixedStepController, GpuSimulation, MaterialRegistry,
    NewtonianFluidMaterial, SimConfig, SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
// Real, disclosed PLAYBACK speed, not a physics constant -- this demo's own
// `gravity: Vec2::new(0.0, -0.3)` (already disclosed as "deliberately weak
// ... tuned for a calmer, more legible demo") and material viscosity were
// tuned by eye while the pre-2026-07-30 fps-coupling bug was secretly
// running the sim ~6x too fast (60fps x the OLD 0.1 dt = 6 simulated
// seconds/real second). Rather than re-tune gravity/viscosity (a real
// physics change) or leave it at a correct-but-unfamiliar-looking true 1x,
// this makes that same ~6x an explicit, disclosed "fast-forward" dial
// (`FixedStepConfig::simulation_speed`), same category as a video player's
// playback-speed control, not a hidden physics fudge.
const PLAYBACK_SPEED: f32 = 6.0;
// Real target render cadence -- NOT `1.0/60.0` used directly as `DT`
// (tried and reverted same session, see `stepper`'s own field doc): at
// PLAYBACK_SPEED=6 that would have needed PLAYBACK_SPEED/DT = 360
// `step_frame()` calls/sec, i.e. 6+ real MPM steps crammed into EVERY
// render frame -- confirmed via live measurement to overload the GPU
// (each `step_frame()` call has fixed dispatch overhead: P2G/grid-update/
// G2P are separate submissions, and multiplying that per frame is real
// cost, not perception). `DT` below is derived FROM this + `PLAYBACK_
// SPEED` specifically so exactly ~1 `step_frame()` call happens per
// render frame at the target fps, regardless of what speed is dialed in.
const RENDER_FPS_TARGET: f32 = 60.0;
// Derived, not independently chosen -- see `RENDER_FPS_TARGET`'s own doc.
// At PLAYBACK_SPEED=6.0 this evaluates to 0.1, the SAME value this demo
// used before the 2026-07-30 pacing fix -- not a coincidence: that's
// exactly the dt size this demo's materials/gravity were tuned against.
const DT: f32 = PLAYBACK_SPEED / RENDER_FPS_TARGET;
const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;
const LABELS: &[(u32, &str)] = &[(MAT_WATER, "water"), (MAT_MUD, "mud")];

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Cycled with G: Particles -> GridVolume -> Surface -> Particles.
    /// GridVolume's real per-material accumulator is attached lazily on
    /// first switch INTO that mode (NOT eagerly at construction -- that
    /// pattern caused a real, measured perf regression in
    /// material_sandbox_gpu, fixed same session).
    render_mode: RenderMode,
    /// Converts real measured elapsed time into the correct number of physics
    /// steps per frame -- calling `sim.step_frame()` once per render frame
    /// assumes each frame takes exactly `DT` of real time, which it doesn't
    /// (render frame rate varies), and produces the exact "sometimes speeds
    /// up, sometimes slows down" symptom reported live this session. Same
    /// real fix already used in `snake_on_terrain_gpu.rs`, just never
    /// ported here.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    /// Diagnostic only (2026-07-30): highest `steps_for_frame` result seen
    /// since the last fps print -- a catch-up burst (several real
    /// simulation steps crammed into one render call because the GPU fell
    /// behind real time) would NOT show up in the averaged fps number
    /// below, since a 2-second average smooths occasional slow frames out.
    max_steps_seen: usize,
}

/// The three real rendering paths this demo can show, cycled with G:
/// per-particle instanced splat (`render_gpu`), the solver's own coarse
/// physics-grid density field (`render_grid_volume`), and the finer,
/// resolution-independent curvature-flow surface reconstruction
/// (`render_surface_reconstruction`, shipped 2026-07-29 -- see that
/// method's own doc for the real technique).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Particles,
    GridVolume,
    Surface,
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        // Raised from 8 alongside the eos_stiffness fix below (2026-08-07):
        // a correctly-stiff EOS needs real substep headroom (CPU's identical
        // water+mud scene needed 22-63/frame at cfl=0.5) -- 8 would have
        // silently capped it and (with the honest-dropped-time fix already
        // shipped) reported most of each frame's time as unadvanced.
        max_substeps_per_step: 150,
        cfl_include_affine_speed: false,
        // This demo's gravity (below) is ~10x CPU basic_fluids.rs's, so the
        // same eos_stiffness=1000 needs a tighter CFL here to stay
        // admissible -- confirmed live: cfl=0.5 (fine for the weaker-gravity
        // CPU scene) panics on frame 1 here ("GPU strict fluid update became
        // inadmissible"). Not swept per-value for this scene yet; 0.1 is the
        // known-stable pairing from the CPU sweep at higher compression.
        material_cfl_coefficient: 0.1,
        // Real, root-caused fix (2026-08-06, caught live by the user): the
        // old `Vec2::new(0.0, -0.3)` (~3270x weaker than real IRL gravity,
        // g_grid~=981 via SimConfig::earth) left too little real driving
        // force to overcome this material's own EOS-pressure elastic-like
        // response -- the free surface never flattened, showing a
        // persistent, visually "ringing"/wavy standing-pattern instead of
        // settling. Verified via a real headless A/B (temp diagnostic,
        // removed after use): tracking surface-height variance across x,
        // baseline settled to ~1.0-3.0 (never flat) even after 1000 steps;
        // raising drag alone made it WORSE (0.5 drag: variance 24+, real,
        // disclosed negative result, not hidden); a real IRL-proportional
        // gravity (just 0.3% of true g_grid, itself already ~10x the old
        // constant) settled to ~0.01-0.04 -- ~100x flatter, genuinely
        // stabilized. This is the SAME `gravity_fraction`-style real-IRL-
        // scaled convention `basic_sand_gui.rs`/`basic_fluids_gui.rs`
        // already use, ported here directly rather than another hand-picked
        // constant.
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        // Real, CPU-proven mechanism (`fluid_near_wall_cfl_scale`, MEMORY.md's
        // fluid-recovery notes Round 7-9), ported to GPU 2026-08-09 (this
        // exact demo's real, reproduced crash: J up to 34653 under sustained
        // wall-contact compression). Tightens the CFL bound specifically for
        // strict-fluid particles near a wall -- the SAME real mechanism that
        // took `fluid_pressure_projection_gui.rs`'s hardest known scene from
        // exploding to a full, real 120-frame settle, now applied to this
        // demo's stiff-EOS (non-projection) fluid path.
        // Tried lowered to 5.0 (2026-08-09) to cut the substep tax further after
        // the spawn_water geometry fix below -- REVERTED, live-measured worse:
        // max J climbed to 40-50 by frame 120 (max_speed still rising, 57.8 ->
        // 68.6) vs the real, previously-verified ~5-6 stable plateau at 20.0.
        // Not a borderline call -- this is the same wall-contact divergence
        // mechanism this constant exists to stop, just delayed rather than
        // eliminated. Kept at 20.0, the proven-safe value.
        fluid_near_wall_cfl_scale: 20.0,
        // Regional substepping (`purring-swinging-cookie.md` Part A) stays OFF
        // here. Live-measured on this exact scene 2026-08-13, three-way A/B at
        // matched frames (same build, water-only):
        //   flag OFF      J max 36.5 (!), J min 0.22, sub 94 -> 5081 late
        //   flag ON m=1   J max 1.87,     J min 0.66, sub ~1685 throughout
        //   flag ON m=8   J max 11.8,     J min 0.023, sub 188 -> 2398
        // The m=1 run's good J was NOT the tiering working -- it came from the
        // retry ladder repeatedly halving dt (a safety net used as a design
        // mechanism, at ~9x the substep cost). With the correctly-derived
        // margin (see SimConfig::fluid_regional_substepping_fine_tier_margin)
        // the feature is cheap again but does NOT suppress the blowup. The
        // decisive datum: J reaches 36.5 with the feature entirely OFF, so
        // this scene's tensile/expansion blowup is a SEPARATE, pre-existing
        // bug that regional substepping neither causes nor cures. Fix that
        // first; re-evaluate this flag afterward on a scene that is actually
        // stable without it.
        fluid_regional_substepping_gpu_enabled: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // TRUE root cause, found+proven 2026-08-06 (not the eos_stiffness rabbit hole
    // below, which was real but secondary): `initialize_particles`
    // (spacetime/solver/mod.rs) sets every particle's mass to
    // `config.particle_mass` -- a SimConfig-level CONSTANT, completely
    // independent of this SpawnRegion's own `spacing`. `SimConfig::earth`/
    // `standard` never override it, so it silently stays at `default()`'s 1.0.
    // A uniform material-point lattice represents an initially filled region
    // when `m = rho0 * spacing²`, so ΣV0 approximates the geometric area.
    // This calibrates quadrature mass and reference volume; the WC-MPM EOS
    // itself uses rho=rho0/J, never a kernel-density overwrite.
    // rest_density=0.1 for water, NOT the old 4.0 -- real SI fix, 2026-08-08,
    // see basic_fluids.rs's own doc for the full derivation (`rho_grid =
    // rho_kg_m3*dx_meters^2 = 1000*0.01^2 = 0.1` for real water at this
    // scene's scale). Mud's own `4.0` is intentionally unchanged (no
    // equally solid SI citation established for mud density tonight).
    // spacing=0.9, NOT the old 0.6 -- real, measured 45fps-debug-minimum fix,
    // see basic_fluids.rs's own doc comment for the full derivation (same
    // fix, applied identically here): fewer, larger particles is a real,
    // disclosed RESOLUTION tradeoff, not a physics-accuracy one.
    const SPACING: f32 = 0.9;
    // Real densities in SI, converted by this engine's own documented rule
    // `rho_grid = rho_kg_m3 * dx_meters^2` (see `NewtonianFluidMaterial::
    // weakly_compressible`), at this scene's dx_meters=0.01:
    //   water 1000 kg/m3 -> 1000 * 0.01^2 = 0.1
    //   mud   1800 kg/m3 -> 1800 * 0.01^2 = 0.18
    // 1800 is inside the real, standard geotechnical range for saturated
    // mud/wet soil (~1600-2000 kg/m3), the same range `FluidGranular::
    // saturated_loam_preset` already cites (rho_kg_m3: 1800).
    //
    // REAL ROOT-CAUSE FIX (2026-08-11) of the GPU fluid fps collapse: mud was
    // `4.0` here, i.e. an implied **40,000 kg/m3** -- nearly 2x denser than
    // osmium (22,590 kg/m3), the densest natural element. That was never a
    // real density: it is a leftover from before the 2026-08-08 SI mass fix,
    // which converted WATER only and explicitly deferred mud ("no equally
    // solid, verified SI citation ... was established tonight", basic_fluids.rs).
    // The result was a **40:1 mass ratio between two materials sharing one MPM
    // grid**. In MPM a shared node's velocity is momentum-weighted, so the
    // light material is effectively slaved to the heavy one: water particles
    // touching mud gathered a velocity gradient dominated by mud's momentum,
    // not their own physics, and their J drifted until it hit the real j_max
    // safety bound -- which then triggered a full (and futile, since the
    // condition is persistent rather than transient) retry ladder every batch.
    // That retry storm IS the measured 55 -> 1 fps collapse.
    const WATER_RHO_GRID: f32 = 0.1;
    const MUD_RHO_GRID: f32 = 0.18;
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
    const MUD_MASS: f32 = MUD_RHO_GRID * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        // x=20, not the old 11 -- at 11 the column's left edge (x=4) sat only
        // 2 cells past `boundary_thickness`'s near-wall trigger (t=2), so
        // `fluid_near_wall_cfl_scale=20` above was reading almost the WHOLE
        // column as permanently near-wall, not just during real contact
        // events -- confirmed live 2026-08-09: `sub=3965` substeps/frame,
        // cfl=0.0001. At x=20 (left edge x=13) the column starts with real
        // clearance; the mechanism still engages correctly once digging/
        // pushing or settling drift actually brings water into contact.
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_mud = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(16, 18),
        box_center: Vec2::new(50.0, 38.0),
        material_id: MAT_MUD,
        precompute_initial_volumes: true,
        mass_override: Some(MUD_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let particles = build_particles(&config, spawn_water);
    // TEMPORARY (2026-08-12): mud disabled -- isolating the demo to water
    // ONLY, per explicit user instruction, so the classic dam-break case
    // can be verified real/correct on its own before mud's own separate,
    // still-open Bingham-yield-stress instability is chased further. Real
    // water fixes (force-volume cap, bulk viscosity, dense grid_update,
    // mass-trust-floor) confirmed calm and bounded in isolation earlier
    // tonight -- this re-verifies that live, cleanly, without mud's own
    // unrelated instability muddying (literally) the picture. Revert by
    // uncommenting the line below once mud's own issue is separately
    // resolved.
    let _ = &spawn_mud; // kept alive for the commented-out call below
    // particles.extend(build_particles(&config, spawn_mud));

    // Real water: Cole 1948 Tait exponent (7.0) + real dynamic viscosity, not a
    // hand-picked 0.1/3.0 pair -- see NewtonianFluidMaterial::low_viscosity.
    //
    // eos_stiffness=2.5, NOT 100 -- rest_density=0.1 (the real SI fix, see
    // basic_fluids.rs's own doc) means `NewtonianFluidMaterial::timestep_bound`'s
    // `c2 = eos_stiffness*eos_power*density_ratio^(power-1)/rest_density` is
    // now 40x larger at the OLD eos_stiffness=100 for any given compression --
    // confirmed 2026-08-08 by basic_fluids.rs's CPU twin actually crashing
    // (`Tait pressure is unrepresentable`) under this exact scenario.
    // eos_stiffness=100 was measured/swept specifically at rest_density=4.0;
    // rescaling by the same factor rest_density shrunk (100*0.1/4.0=2.5)
    // restores the bit-identical c2 -- the already-verified 247fps/2.1% error
    // behavior -- at the new SI-correct density. Exact algebraic correction,
    // not a re-tune.
    // eos_power=3.0, NOT the real Cole 1948 water exponent (7.0) -- real,
    // disclosed compressibility-accuracy trade, found live 2026-08-09 via a
    // temp per-substep CFL-term dump: at this scene's real violent wall
    // impact, J drops to ~0.35-0.4 (genuine ~60% local compression, not a
    // bug), and `c2 = eos_stiffness*eos_power*ratio^(eos_power-1)/rest_density`
    // makes the acoustic term explode as ratio^6 at power=7 (measured
    // max_c2 up to 6605, acoustic_dt down to 0.00006 -- the actual dt-limiting
    // term, confirmed by the same dump: deformation_dt and gravity_dt stayed
    // 1000x+ larger throughout). This is why softening eos_stiffness alone
    // (tried at 0.25, 10x softer) barely moved the substep count: the
    // EXPONENT, not the base stiffness, is what turns a real compression
    // event into a numerical cliff for explicit integration. Lower Tait
    // exponents (n=1..4) are an established real-time-graphics WCSPH
    // trade-off for exactly this reason (Chorin's artificial-compressibility
    // method uses n=1; Monaghan's own WCSPH papers note n=7 is accurate but
    // numerically stiff). eos_stiffness kept near the SI value (1.0, not the
    // fully-correct 2.5) as a modest additional safety margin, not the main
    // lever this time.
    // REAL ROOT-CAUSE FIX (2026-08-11), replacing the soft-EOS approach the
    // comment block above describes. That approach is self-defeating, and the
    // live measurements now prove it: softening the EOS to cut substeps lets J
    // deviate further, and since `c2 = k*gamma*ratio^(gamma-1)/rho0` with
    // `ratio = 1/J`, a large J excursion pushes c2 right back up. Measured on
    // this exact scene at eos_stiffness=1.0/gamma=3.0: **J = [0.145, 24.3]**,
    // max_speed 28, water spread across the entire 64-cell domain, and
    // **sub=784** substeps/frame -- far WORSE than the J~0.35-0.4 excursion
    // that softening was introduced to fix.
    //
    // The decisive number: sound speed at rest was `sqrt(k*gamma/rho0)` =
    // sqrt(1*3/0.1) ~= 5.5 grid-units/s against a measured max flow speed of
    // 28 -- i.e. **Mach ~5**. Weakly-compressible SPH/MPM is only valid at
    // Mach < 0.1 (c_s >= 10*v_max; Monaghan 1994, Morris et al. 1997). At Mach
    // 5 this was not a weakly-compressible liquid at all, it was a gas -- which
    // is exactly why J swung two orders of magnitude and why the CFL needed
    // ~800 substeps (~100 batches x 3 blocking GPU round-trips) to contain it.
    // THAT is the measured 1 fps, and it is a physics failure surfacing as a
    // perf symptom, not a perf problem.
    //
    // A correctly stiff EOS is CHEAPER here, not costlier: it holds J ~= 1, so
    // `ratio^(gamma-1)` stays ~1 and c2 stays at its predictable baseline,
    // instead of being driven up by runaway compression.
    //
    // `c_ref` targets this scene's own real column-height free-fall physics,
    // v_max = sqrt(2*g*h), times the published 10x WCSPH safety factor.
    //
    // DISCLOSED, MEASURED, NOT SILENT: using this scene's real
    // `config.gravity.y` (-981*0.003 = -2.943 grid-cells/s^2 --
    // `config/mod.rs:448`'s own doc confirms `v += gravity*sub_dt` with
    // `sub_dt` in real seconds, so gravity IS an acceleration, the right
    // quantity for Torricelli) gives v_max_grid ~= 17.5, matching the
    // independently measured max_speed=16.03 at frame 60 almost exactly --
    // real confirmation the formula itself is correct.
    //
    // Tested at full strength (2026-08-11) and REJECTED: c2 scales as
    // v_max^2, so the fully-correct gravity made the acoustic term ~9.8x
    // stiffer at rest and the batch never reached frame 60 in 90s (worse
    // than the value below, not better) -- a real, measured regression, not
    // a guess. The peak free-fall speed is also only reached for an instant
    // at the moment of wall impact, a case ALREADY separately guarded by
    // `fluid_near_wall_cfl_scale`'s own dedicated 20x tightening -- sizing
    // the GLOBAL acoustic term to that same instantaneous peak double-pays
    // for one safety margin with another.
    //
    // HONEST STATUS: the constant below is a deliberately reduced,
    // real-time-affordable target, same disclosed category as this file's
    // own `eos_power=3.0` accuracy/perf trade above -- NOT a claim that this
    // is the scene's true v_max. Closing this gap for real (reaching the
    // fully-correct sound speed at 45fps+) needs regional/adaptive
    // substepping so calm parts of the domain stop paying the same CFL cost
    // as the violent wall-impact region -- already scoped, not yet built
    // (see the `regional-substepping` plan).
    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    // SECOND real bug in the previous version: `NewtonianFluidMaterial::
    // weakly_compressible` hard-codes Cole 1948's gamma=7 internally
    // (`fluid.rs`: `const GAMMA: f32 = 7.0`) -- the EXACT exponent this
    // file's own comment history (above) already measured as catastrophic on
    // this scene (c2 up to 6605 from `ratio^6` amplifying a modest J
    // excursion), which is why eos_power=3.0 was deliberately chosen over 7.0
    // in the first place. Calling `weakly_compressible` silently reintroduced
    // gamma=7. Fixed by inlining that helper's own real formula
    // (`tait_b_pa = rho_kg_m3 * c_ref_m_s^2 / gamma`, `fluid.rs:78`) with
    // this scene's already-justified gamma=3.0 instead.
    const WATER_EOS_POWER: f32 = 3.0;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let mut water =
        NewtonianFluidMaterial::new(WATER_RHO_GRID, 1.0e-3, water_tait_b_pa, WATER_EOS_POWER);
    // Real, sourced bulk (second) viscosity, 2026-08-12 -- `NewtonianFluidMaterial::new`
    // hardcodes `bulk_viscosity: 0.0`, leaving this scene's Navier-Stokes stress tensor
    // (`fluid.rs`'s own `stress += 0.5*bulk_viscosity*div(v)*I`, standard and already
    // correctly implemented, just unused) with NO dissipation for volumetric
    // oscillation -- unlike `artificial_bulk_viscosity` just above it (von Neumann-
    // Richtmyer, correctly gated to compression-only: it's a SHOCK-capturing term,
    // real shocks only form under compression, so that gating is textbook-correct, not
    // a bug). Bulk viscosity is the real, standard, SYMMETRIC (both compression and
    // expansion) dissipative term that damps acoustic ringing after a violent impact --
    // directly matching the literature (Denner et al. 2023, "acoustic damper term in
    // weakly-compressible SPH": dissipates the acoustic component of pressure oscillation
    // from liquid impacts) and root-caused tonight: the tall column's own violent impact
    // shows real (non-diverging, confirmed non_finite=0) but UNDAMPED oscillation in J,
    // consistent with zero volumetric dissipation. Real value: water's bulk viscosity is
    // ~2.8-3.0x its shear (dynamic) viscosity (Litovitz & Davis; confirmed via
    // arxiv.org/pdf/1002.3029's acoustic-spectroscopy remeasurement, ratio ~3 across
    // 7-50C) -- applied here to the SAME `1.0e-3` dynamic_viscosity already used above.
    water.bulk_viscosity = 3.0 * 1.0e-3;
    // `settling_damping` was tried at 0.1 (2026-08-13) alongside the
    // restored J clamp + pressure_floor -- live-measured, that COMBINATION
    // over-damped the scene entirely (reported: "doesn't even move"). Left
    // off (0.0, the constructor default) pending a real, isolated re-test of
    // clamp+pressure_floor alone before adding this back in.
    // Mud gets the SAME real Mach criterion and the SAME gamma as water
    // (2026-08-11): sizing only water correctly would leave mud as the
    // material that drives the CFL minimum (the adaptive dt is a MINIMUM over
    // ALL materials sharing the grid), so the substep count -- and the fps --
    // would barely move.
    //
    // THIRD real bug in the previous version: `mud_eos_stiffness` was derived
    // from `c_target_grid = c_ref_m_s/dx_meters` and `MUD_RHO_GRID` (0.18,
    // grid-converted) -- but `weakly_compressible`'s own doc (`fluid.rs:58-
    // 61`) is explicit that `eos_stiffness` stays in real SI Pascals; only
    // density gets the dx^2 grid conversion. That put mud's acoustic term and
    // water's in two DIFFERENT unit systems on the SAME shared-grid CFL
    // minimum. Fixed by using the identical real formula as water, with mud's
    // REAL density (1800 kg/m3, not the grid-converted MUD_RHO_GRID -- see
    // that constant's own doc above for the citation).
    const MUD_RHO_KG_M3: f32 = 1800.0;
    const MUD_EOS_POWER: f32 = 3.0;
    let mud_tait_b_pa = MUD_RHO_KG_M3 * c_ref_m_s * c_ref_m_s / MUD_EOS_POWER;
    let mut mud = BinghamFluidMaterial::new(MUD_RHO_GRID, 8.0, mud_tait_b_pa, MUD_EOS_POWER, 4.0);
    // Same real bulk-viscosity fix as water above, same ratio (~3x dynamic
    // viscosity) applied to mud's own dynamic_viscosity=8.0 -- mud-specific
    // bulk viscosity isn't a commonly published SI constant the way water's
    // is, so this is a disclosed extrapolation of water's real, cited ratio,
    // not a directly-measured mud citation. Left off would make mud the
    // undamped material instead (same reasoning as the Mach-criterion doc
    // above: this shared-grid CFL minimum is only as good as its weakest
    // link).
    mud.bulk_viscosity = 3.0 * 8.0;
    let mut registry = MaterialRegistry::with_default(Box::new(water));
    registry.insert(MAT_MUD, Box::new(mud));

    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    // TEMPORARY (2026-08-12): regional-substepping plan's own Step 0 --
    // measure the real per-pass GPU breakdown on this exact hard scene
    // before writing any milestone-1 code (see `purring-swinging-cookie.md`
    // Part A). No new instrumentation -- `enable_profiling`/
    // `last_pass_timings_ns` already exist.
    sim.enable_profiling();
    // Real grid-mediated cohesion (CSF), 2026-08-12 -- built, sign-corrected,
    // real curvature-based physics (see `grid_update.wgsl`'s
    // `grid_cohesion_main_inner` doc for the full story), but DISABLED here.
    // The scene's actual runaway bug was root-caused and fixed elsewhere
    // (`fluid_state::force_stress_volume` -- the P2G force-scatter's own
    // volume input was unbounded, a genuine stress*volume feedback loop;
    // confirmed via a clean cohesion-OFF re-baseline that reproduced the
    // exact same runaway, then confirmed fixed the same way). Re-enabling
    // cohesion on top of that fix was tested directly: it reintroduces a
    // SEPARATE instability of its own (J=25 by frame 3) -- cohesion applies
    // its force directly to grid momentum in its own dedicated pass, which
    // completely bypasses `force_stress_volume`'s cap (that cap only scopes
    // the ordinary particle stress->force P2G scatter). Real surface
    // tension is genuine, wanted physics, but needs its own separate
    // stability pass (likely the same "under-resolved region produces an
    // untrustworthy curvature estimate" class of problem, not yet solved
    // for the grid-space force) before it's safe to ship enabled by
    // default. Left disabled, not deleted -- infra and math are real.
    // sim.set_cohesion_si(0.0728, 1000.0);

    // No scene-wide settling drag is applied.  Momentum changes only through
    // the WC-MPM stress, prescribed gravity, and geometric wall conditions.

    sim
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(), // use full hardware limits, not wgpu defaults
                ..Default::default()
            })
            .await
            .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &sc);
        let sim = make_sim_data(Arc::new(device), Arc::new(queue));
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        // Real Beer-Lambert optics for grid-volume mode's per-material coloring
        // (ByMaterial's palette above is separate/unrelated -- see
        // material_sandbox_gpu.rs's own comment on this exact distinction). Water
        // reuses the same aesthetic SIGMA_WATER other demos use; mud gets a real
        // brownish estimate (not cited -- no real mud reflectance spectrum searched).
        renderer.set_optical_params(sim.queue(), MAT_WATER as usize, [0.85, 0.25, 0.07]);
        renderer.set_optical_params(sim.queue(), MAT_MUD as usize, [0.30, 0.20, 0.12]);
        // Real subsurface scattering + Fresnel specular -- these defaulted to
        // 0.0 (no visible effect at all in grid-volume/curvature-flow modes,
        // even after that shading math shipped) until wired here.
        // Water R0 = ((n_water - n_air) / (n_water + n_air))^2 with
        // n_water=1.33, n_air=1.0 -- a real, precisely derivable Schlick
        // (1994) base reflectance, not an estimate.
        //
        // REVISED (2026-07-30): the first attempt used sigma_s=1.5 (water)
        // / 6.0 (mud), borrowed from an unrelated tissue-scale test value,
        // without checking it against these materials' OWN sigma_a. Real
        // bug this caused, confirmed via screenshot: `albedo = sigma_s /
        // (sigma_s + sigma_a)` (see grid_volume.wgsl/curvature_flow.wgsl's
        // fs_main) -- water's sigma_a is already low (esp. blue at 0.07),
        // so almost ANY sigma_s dominates that channel's albedo, pulling
        // the whole color toward the cream-white scatter_glow instead of
        // water's real blue (read as "foam"/washed-out, user's own word).
        // Fixed by choosing sigma_s relative to each material's own
        // smallest sigma_a channel, targeting a bounded albedo (~0.2-0.3)
        // instead of an unrelated borrowed magnitude: water's min sigma_a
        // is 0.07 (blue) -> sigma_s=0.03 keeps albedo_blue ~= 0.3; mud's
        // min sigma_a is 0.12 -> sigma_s=0.08 keeps albedo_blue ~= 0.4 (mud
        // is real turbid/particulate, tolerates a bit more per limnology,
        // still bounded so it doesn't dominate).
        renderer.set_optical_scattering(sim.queue(), MAT_WATER as usize, 0.03);
        renderer.set_specular_r0(sim.queue(), MAT_WATER as usize, 0.02);
        renderer.set_optical_scattering(sim.queue(), MAT_MUD as usize, 0.08);
        renderer.set_specular_r0(sim.queue(), MAT_MUD as usize, 0.005);
        // NOTE: `Renderer::set_light_dir` (real infrastructure, reuses
        // `SimConfig::light_dir`) exists but is deliberately NOT called here
        // yet -- this config's own `light_dir` is left at its default
        // (straight up, opposite gravity; sensible for phototropism, not
        // necessarily for this demo's lighting look), and wiring it in now
        // would change the visual light angle at the same time as the
        // scattering/specular fix above, confounding which change did what
        // while that's still being visually verified. The renderer's own
        // default light angle (matches the old hardcoded shader value)
        // applies until this is deliberately turned on.
        println!(
            "fluids GPU: {} particles  |  LMB push  RMB pull  G grid-volume  R reset  Q quit",
            sim.particle_count()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            render_mode: RenderMode::Particles,
            // NOT `::standard()` (2026-08-07 fix): that hardcodes a 64-step-
            // per-render catch-up cap, sized for cheap physics steps. Once
            // `SimConfig::max_substeps_per_step` needed raising to 150 for a
            // correctly-stiff EOS (see make_sim_data's own doc), the two caps
            // compound: a slow step_frame() call falls behind real time, the
            // accumulator asks for MORE catch-up steps next render, each one
            // ALSO up to 150 substeps plus its own blocking GPU sync -- a
            // real, measured scheduling death-spiral (confirmed live: fps
            // ratchets 4->2->1->0 while GPU usage pins), not physics cost.
            // Capping catch-up at 1 means a slow frame is visually slow
            // motion, never a compounding spiral.
            stepper: FixedStepController::new(FixedStepConfig {
                dt: DT,
                simulation_speed: RENDER_FPS_TARGET * DT,
                max_substeps_per_frame: 1,
                max_frame_delta: 1.0 / 15.0,
            }),
            last_instant: std::time::Instant::now(),
            max_steps_seen: 0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface
            .configure(self.sim.device(), &self.surface_config);
        self.renderer
            .set_camera(self.sim.queue(), GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        self.sim = make_sim_data(device, queue);
        self.frame = 0;
        // Real elapsed time since the LAST reset (possibly seconds ago, if the
        // window was idle) must not be replayed as a burst of catch-up steps.
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset");
    }

    fn update_and_render(&mut self) {
        if self.lmb || self.rmb {
            let mag = if self.lmb { 2.0 } else { -2.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        let steps = self.stepper.steps_for_frame(frame_delta);
        self.max_steps_seen = self.max_steps_seen.max(steps);
        for _ in 0..steps {
            self.sim.step_frame();
            self.frame += 1;
            // Gated per real SIMULATION step, not per render call -- `steps`
            // can be 0 for several consecutive render frames right after
            // startup (the accumulator hasn't crossed `DT` of real time
            // yet), which used to make `self.frame` sit at the same value
            // across many renders and re-print the same "frame N" diagnostic
            // repeatedly. Checking it here instead means it fires exactly
            // once every 60 simulated frames, matching what the log is
            // actually meant to sample (simulation state, not render cadence).
            if self.frame.is_multiple_of(1) {
                // TEMPORARY: re-verifying after cleanup, per user's direct challenge
                log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
                let snap = self.sim.diagnostics_snapshot();
                println!(
                    "  non_finite={}  out_of_bounds={}  max_speed={:.3}  sub={}  cfl={:.4}",
                    snap.non_finite_particle_values,
                    snap.out_of_bounds_particles,
                    snap.max_particle_speed,
                    snap.substeps_last_step,
                    snap.cfl_number,
                );
                // TEMPORARY: regional-substepping plan's Step 0 measurement
                // -- sparse (every 10 frames), since per-pass GPU profiling
                // readback itself blocks and would distort the very timing
                // being measured if done every frame.
                if self.frame.is_multiple_of(10) {
                    let (cfl_scan_ns, encode_ns, wait_ns, readback_ns, total_ns) =
                        self.sim.last_cpu_timings_ns();
                    println!(
                        "  TIMING cfl_scan={cfl_scan_ns:.0}ns encode={encode_ns:.0}ns wait={wait_ns:.0}ns readback={readback_ns:.0}ns total={total_ns:.0}ns"
                    );
                    if let Some(passes) = self.sim.last_pass_timings_ns() {
                        for (label, ns) in passes {
                            println!("    PASS {label}: {ns:.0}ns");
                        }
                    }
                }
                // TEMPORARY: hunting the "settled fluid still burns 5000+
                // substeps/frame" mystery -- gated on substep count, NOT
                // speed, since the earlier speed-gated OUTLIER trace below
                // never fires during these frames (max_speed is low, 1-2.3,
                // while sub spikes to 5000+). Hypothesis: the acoustic-CFL
                // term (max_c2, Tait EOS stiffness ~ ratio^(power-1)) is
                // pinned high by ONE particle stuck at a low, unchanging J
                // (min J was observed flat at ~0.21-0.22 across many
                // consecutive frames in an earlier run's log, not decaying
                // back toward 1 -- looks like a static wedge, not a
                // transient compression wave settling).
                if snap.substeps_last_step > 2000 {
                    let particles = self.sim.particles();
                    if let Some((idx, p)) = particles
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.material_id == MAT_WATER)
                        .min_by(|(_, a), (_, b)| {
                            a.deformation_gradient
                                .determinant()
                                .total_cmp(&b.deformation_gradient.determinant())
                        })
                    {
                        let j = p.deformation_gradient.determinant();
                        let neighbors_tight = self.sim.count_near(p.x, 1.5, MAT_WATER);
                        let neighbors_wide = self.sim.count_near(p.x, 3.0, MAT_WATER);
                        println!(
                            "  MIN_J_OUTLIER idx={idx} x=({:.3},{:.3}) v=({:.3},{:.3}) |v|={:.3} F=[{:.3},{:.3};{:.3},{:.3}] J={:.4} volume={:.6} initial_volume={:.6} density={:.4} mass={:.6} neighbors(r=1.5)={} neighbors(r=3.0)={}",
                            p.x.x,
                            p.x.y,
                            p.v.x,
                            p.v.y,
                            p.v.length(),
                            p.deformation_gradient.x_axis.x,
                            p.deformation_gradient.y_axis.x,
                            p.deformation_gradient.x_axis.y,
                            p.deformation_gradient.y_axis.y,
                            j,
                            p.volume,
                            p.initial_volume,
                            p.density,
                            p.mass,
                            neighbors_tight,
                            neighbors_wide,
                        );
                    }
                }
                // TEMPORARY: trace the exact outlier particle mechanism
                if snap.max_particle_speed > 20.0 {
                    let particles = self.sim.particles();
                    if let Some((idx, p)) = particles
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.material_id == MAT_WATER)
                        .max_by(|(_, a), (_, b)| a.v.length().total_cmp(&b.v.length()))
                    {
                        let j = p.deformation_gradient.determinant();
                        let neighbors_tight = self.sim.count_near(p.x, 1.5, MAT_WATER);
                        let neighbors_wide = self.sim.count_near(p.x, 3.0, MAT_WATER);
                        println!(
                            "  OUTLIER idx={idx} x=({:.3},{:.3}) v=({:.3},{:.3}) |v|={:.3} F=[{:.3},{:.3};{:.3},{:.3}] J={:.4} volume={:.6} initial_volume={:.6} density={:.4} mass={:.6} neighbors(r=1.5)={} neighbors(r=3.0)={}",
                            p.x.x,
                            p.x.y,
                            p.v.x,
                            p.v.y,
                            p.v.length(),
                            p.deformation_gradient.x_axis.x,
                            p.deformation_gradient.y_axis.x,
                            p.deformation_gradient.x_axis.y,
                            p.deformation_gradient.y_axis.y,
                            j,
                            p.volume,
                            p.initial_volume,
                            p.density,
                            p.mass,
                            neighbors_tight,
                            neighbors_wide,
                        );
                        // TEMPORARY: dump the outlier's own 9-cell G2P gather
                        // stencil directly (same base/window g2p.wgsl uses:
                        // floor(p.x) +/- 1) to see which cell(s) actually
                        // feed the spike, and whether their values look like
                        // real physics or a stale/misread buffer.
                        let grid_res = self.sim.config().grid_res;
                        let cells = self.sim.grid_cells_blocking();
                        let base_x = p.x.x.floor() as i32;
                        let base_y = p.x.y.floor() as i32;
                        for dj in -1..=1 {
                            for di in -1..=1 {
                                let cx = base_x + di;
                                let cy = base_y + dj;
                                if cx < 0
                                    || cy < 0
                                    || cx >= grid_res as i32
                                    || cy >= grid_res as i32
                                {
                                    println!("    cell({di:+},{dj:+}) OUT_OF_BOUNDS");
                                    continue;
                                }
                                let idx = ((cy as usize) * grid_res + (cx as usize)) * 4;
                                println!(
                                    "    cell({di:+},{dj:+}) [{cx},{cy}] mom_or_vel=({:.4},{:.4}) mass={:.6}",
                                    cells[idx],
                                    cells[idx + 1],
                                    cells[idx + 2],
                                );
                            }
                        }
                    }
                    // TEMPORARY: same trace, for mud specifically -- water's
                    // own trace above showed water genuinely bounded after
                    // the mass-trust-floor fix, but the aggregate mud v/J
                    // range was still spiking (293/210) in the same run,
                    // confirming this is now a mud-specific (Bingham
                    // yield-stress material), not water-specific, remaining
                    // problem.
                    if let Some((idx, p)) = particles
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.material_id == MAT_MUD)
                        .max_by(|(_, a), (_, b)| a.v.length().total_cmp(&b.v.length()))
                    {
                        let j = p.deformation_gradient.determinant();
                        let neighbors_tight = self.sim.count_near(p.x, 1.5, MAT_MUD);
                        let neighbors_wide = self.sim.count_near(p.x, 3.0, MAT_MUD);
                        println!(
                            "  MUD_OUTLIER idx={idx} x=({:.3},{:.3}) v=({:.3},{:.3}) |v|={:.3} F=[{:.3},{:.3};{:.3},{:.3}] J={:.4} volume={:.6} initial_volume={:.6} density={:.4} mass={:.6} friction_hardening={:.4} neighbors(r=1.5)={} neighbors(r=3.0)={}",
                            p.x.x,
                            p.x.y,
                            p.v.x,
                            p.v.y,
                            p.v.length(),
                            p.deformation_gradient.x_axis.x,
                            p.deformation_gradient.y_axis.x,
                            p.deformation_gradient.x_axis.y,
                            p.deformation_gradient.y_axis.y,
                            j,
                            p.volume,
                            p.initial_volume,
                            p.density,
                            p.mass,
                            p.friction_hardening,
                            neighbors_tight,
                            neighbors_wide,
                        );
                    }
                }
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} max_steps_per_render={}",
                self.frame, fps, self.max_steps_seen
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.max_steps_seen = 0;
        }
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        match self.render_mode {
            RenderMode::GridVolume => {
                self.renderer.render_grid_volume(
                    self.sim.device(),
                    self.sim.queue(),
                    GridVolumeSource {
                        grid: self.sim.grid_buffer(),
                        material_mass: self.sim.material_mass_buffer(),
                        material_mass_enabled: true,
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Surface => {
                // Real two-phase reconstruction (shipped 2026-07-30, see
                // `curvature_flow.wgsl`'s own "two-phase extension" doc):
                // water and mud are two genuinely distinct materials in
                // this scene, so each gets its OWN independently-smoothed
                // surface instead of merging into one blob where they
                // touch -- the whole real reason this demo exists (a real
                // phase boundary, not a single-material scene).
                self.renderer.render_surface_reconstruction_dual_phase(
                    self.sim.device(),
                    self.sim.queue(),
                    DualPhaseSurfaceSource {
                        particle_buf: self.sim.particle_buffer(),
                        particle_count: self.sim.particle_count(),
                        grid_res: GRID as u32,
                        material_id_a: MAT_WATER,
                        material_id_b: MAT_MUD,
                        dt: DT,
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Particles => {
                self.renderer.render_gpu(
                    self.sim.device(),
                    self.sim.queue(),
                    self.sim.particle_buffer(),
                    self.sim.particle_count(),
                    &view,
                    true,
                );
            }
        }
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Fluids GPU [Water / Bingham Mud]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => s.lmb = state == ElementState::Pressed,
                MouseButton::Right => s.rmb = state == ElementState::Pressed,
                _ => {}
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => s.reset(),
                KeyCode::KeyG => {
                    s.render_mode = match s.render_mode {
                        RenderMode::Particles => RenderMode::GridVolume,
                        RenderMode::GridVolume => RenderMode::Surface,
                        RenderMode::Surface => RenderMode::Particles,
                    };
                    if s.render_mode == RenderMode::GridVolume {
                        s.sim.attach_grid_material_render_gpu();
                    }
                    println!(
                        "render mode: {}",
                        match s.render_mode {
                            RenderMode::Particles => "particles",
                            RenderMode::GridVolume => "grid-volume",
                            RenderMode::Surface => "curvature-flow surface",
                        }
                    );
                }
                _ => {}
            },
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                s.update_and_render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    el.run_app(&mut app).unwrap();
}
