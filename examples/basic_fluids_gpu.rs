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
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    const MUD_MASS: f32 = 4.0 * SPACING * SPACING;
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
    let mut particles = build_particles(&config, spawn_water);
    particles.extend(build_particles(&config, spawn_mud));

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
    let water = NewtonianFluidMaterial::new(0.1, 1.0e-3, 1.0, 3.0);
    let mud = BinghamFluidMaterial::new(4.0, 8.0, 100.0, 3.0, 4.0);
    let mut registry = MaterialRegistry::with_default(Box::new(water));
    registry.insert(MAT_MUD, Box::new(mud));

    let sim = GpuSimulation::with_device(device, queue, config, particles, registry);

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
            if self.frame.is_multiple_of(60) {
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
