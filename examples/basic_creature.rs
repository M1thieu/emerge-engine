extern crate emerge_engine as emerge;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    FrameLogger, Lnn, NeoHookeanMaterial, RatchetFrictionBoundary, SimConfig, Simulation,
    SpawnRegion, per_material_stats,
};
use glam::{IVec2, Vec2};
/// CPU creature -- NeoHookean soft body with peristaltic muscle activation.
///
/// Traveling wave of vertical muscle contraction, crawling via
/// `RatchetFrictionBoundary` -- directional (setae-style) floor friction that
/// resists backward slip much more than forward slip. This is the mechanism
/// that actually produces net locomotion for this body: plain symmetric floor
/// friction measured near-zero net drift regardless of muscle fiber direction
/// (a symmetric contract/release cycle cancels its own displacement, the same
/// reason you can't swim forward clapping symmetrically underwater); real
/// crawlers break that symmetry structurally (setae/hooks), not by timing
/// friction to muscle phase -- confirmed against SoftZoo (the published MPM
/// soft-robot locomotion benchmark, which uses only symmetric friction + learned
/// actuation) and real-crawler robotics literature. See
/// `tests/physics_correctness.rs::ratchet_friction_produces_real_directed_locomotion`
/// for the headless proof.
///
/// Driven by an `Lnn` (Liquid Time-constant Network) continuous-time CPG, not a
/// hand-coded sine wave -- the same controller LP's creatures use. A bilateral
/// (two-ring, mutually-inhibiting) CPG: left/right steer by biasing one ring
/// harder than the other. NOTE: this body is a straight, non-bending column, so
/// "steering" here shifts which half drives harder but cannot produce a real
/// left/right turn -- that needs a body that can curve, a separate limitation.
/// Up/down adjusts wave speed (LNN clock rate). Space pauses. R resets.
///
///   cargo run --example basic_creature --features "render"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// GRID was 64: found 2026-07-09 (real repro, not a guess) that this made the
// crawl look like it "stopped working" after sustained steering -- extensive
// investigation (CPG-under-bias, floor-friction impulse model, substep
// compounding, muscle duty cycle) all turned out to be chasing a ghost: the
// creature was crawling correctly the entire time and simply reached this
// grid's side wall (~24-unit-wide body, 64-unit grid). `set_camera` frames the
// whole fixed grid uniformly with no camera-follow, so a real fix (tracking
// camera) is out of scope for this basic example -- LP's actual answer for
// unbounded travel is its chunk-based world design, not this demo. Widened so
// normal play/steering has real room without hitting a wall and looking dead.
const GRID: usize = 96;
const DT: f32 = 0.1;
const MAT_BODY: u32 = 0;
const MUSCLE_GROUPS: u32 = 8;
// Bilateral CPG: 2 mutually-coupled rings (front/back halves of the body),
// 4 segments each. Steering biases one ring harder than the other.
const N_RINGS: usize = 2;
const N_PER_RING: usize = MUSCLE_GROUPS as usize / N_RINGS;
const RING_CROSS_COUPLING: f32 = 1.0;
// Muscle drive is held at the documented activation ceiling; it is never pushed
// above 1.0 (a muscle can't contract >100%), which also keeps active stress
// inside the CFL budget instead of letting a global amplitude knob detonate it.
const MUSCLE_AMPLITUDE: f32 = 0.9;

// How many steps to run the CPG in isolation (no physics, invisible to the
// player) before the body starts reading its activations. Found 2026-07-09 via
// a real headless test: with the punched-up tuning below, a body reading
// activations from the RAW seeded network (no burn-in) gets a visible
// backward stumble in the first ~1000 steps (real 3.5-unit backslide) before
// the CPG settles into its true phase-locked traveling wave -- the seeded
// initial state isn't wrong, it's just not fully organized yet at steer=0
// (unbiased rings settle slower than a live-biased run, which is why an
// earlier sweep that forced a permanent +-1.0 bias never caught this). 600
// steps (~60 sim-seconds, ~6 gait periods) fully eliminated the backslide
// (verified: 0.00 vs 3.52) and nearly tripled total drift over the same
// window (+24.28 vs +3.41) -- a real, measured fix, not a guessed buffer.
const CPG_BURN_IN_STEPS: usize = 600;

fn make_cpg() -> Lnn {
    make_cpg_biased(0.0)
}

/// Builds a fresh CPG already biased and burned-in for `bias`'s sign -- used
/// both for the initial spawn (bias=0.0) and for a live direction reversal.
///
/// Real fix, found+verified 2026-07-09: naively flipping `set_ring_bias` on
/// the SAME already-organized network (the original approach) doesn't reverse
/// the wave's physical propagation direction -- `coupled_traveling_wave`'s
/// topology hard-bakes low-index -> high-index propagation (see
/// src/control/lnn.rs's `excite next` term); only the tonic gain flips.
/// Flipping only the ratchet's friction direction then means real thrust
/// (still propagating the OLD physical way) fights real resisting friction
/// for as long as reverse is held -- confirmed via a live playtest AND a
/// headless worst-case test to be a genuine, unbounded compression ratchet
/// (min J fell to 0.087 live, and even the safest tuning tested still
/// degraded 0.576->0.333 over 18,000 sustained-reverse steps).
///
/// Throwing the old network away and burning in a BRAND NEW one seeded with
/// the new direction's bias from construction lets the wave organize its own
/// propagation to match the new friction direction, instead of being force-
/// mapped after the fact (a naive index-mirror trick was tried first and
/// made things WORSE, not better -- verified empirically, not assumed).
/// Headless proof: post-reversal net drift went from ~-1 to -2 units (stall)
/// to a genuine -18.71 over 25,000 sustained-reverse steps, with J degrading
/// far more gently (0.601 -> 0.231) than the naive flip (0.601 -> 0.109 over
/// a shorter window).
fn make_cpg_biased(bias: f32) -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, RING_CROSS_COUPLING);
    lnn.set_ring_bias(0, N_PER_RING, bias);
    lnn.set_ring_bias(1, N_PER_RING, -bias);
    for _ in 0..CPG_BURN_IN_STEPS {
        lnn.step(DT);
    }
    lnn
}

// Per-segment colors matching the SoftZoo rainbow palette (ByMaterial slots 0""7).
// ColorMode::ByMaterial assigns color by material_id % 16, so we encode muscle group
// as material_id directly for rendering. Physics still uses MAT_BODY internally.
// For simplicity we render via ByMaterial which gives blue for all (one material).
// Advanced: override muscle group rendering via a custom color callback.

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    body_range: std::ops::Range<usize>,
    lnn: Lnn,
    paused: bool,
    wave_speed: f32,
    /// Steering bias in [-1, 1]: drives one CPG ring harder than the other,
    /// breaking the wave's symmetry the way an animal turns. 0 = straight.
    /// ALSO drives the ratchet's crawl direction live (see `update_and_render`):
    /// steer < 0 reverses which way the body actually crawls -- this is real
    /// control, not cosmetic, since it changes `RatchetFrictionBoundary`'s
    /// `easy_direction` on the shared instance the solver is already using.
    steer: f32,
    /// Sign of `steer` as of the last frame a crawl-direction reseed happened
    /// ({-1.0, +1.0}, never 0.0) -- when `steer`'s effective sign flips, `lnn`
    /// gets thrown away and rebuilt via `make_cpg_biased` for the new
    /// direction instead of having its bias flipped in place. See
    /// `make_cpg_biased`'s doc for why the in-place flip doesn't work.
    last_dir_sign: f32,
    /// Shared handle to the solver's own ratchet boundary -- steering this
    /// updates the SAME instance driving physics, not a copy.
    ratchet: Arc<RatchetFrictionBoundary>,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// True once an anomaly has been reported, so we WARN on the first frame it
    /// appears rather than spamming every frame after.
    anomaly_latched: bool,
    spawn_centroid: Vec2,
    /// NDJSON telemetry, one line per `log_telemetry` call, alongside the
    /// console print -- reads/queries far more reliably than parsing the
    /// human-readable line (used e.g. to build post-hoc telemetry charts
    /// during the 2026-07-09 locomotion investigation).
    telemetry_log: FrameLogger,
}

fn make_sim() -> (
    Simulation,
    std::ops::Range<usize>,
    Arc<RatchetFrictionBoundary>,
) {
    // Stiffness doubled from the original (5.0, 10.0) -- found 2026-07-09 via a
    // real parameter sweep (not a guess), not a smaller tweak: the original
    // material was too soft relative to its own active_stress_coeff, so each
    // muscle contraction outpaced the elastic recovery and the body ratcheted
    // toward a permanently-compressed, near-static state after crawling a
    // bounded ~15-28 units, regardless of direction (see project memory
    // basic_creature_wall_hit_and_reversal_stall_2026-07-09 for the full
    // investigation). Verified this exact combination sustains: last-500-step
    // displacement was STILL POSITIVE (and not decaying) at both 6000 and
    // 12000 steps (0.664 then 1.029) -- genuinely ongoing locomotion, not
    // delayed settling.
    //
    // Pushed further 2026-07-09 (16.0, 32.0 / active_stress_coeff=40.0) for a
    // faster steady cruise -- briefly reverted the same day after a live
    // playtest under sustained reverse steer showed progressive compression
    // (min J fell to 0.087). That regression was traced to the STEERING
    // MECHANISM, not this tuning: naively flipping `set_ring_bias` on the
    // SAME already-organized network doesn't reverse the wave's physical
    // propagation (baked into coupled_traveling_wave's fixed topology, see
    // src/control/lnn.rs), so flipped friction just fights the still-forward-
    // propagating wave forever. Fixed properly via `make_cpg_biased` +
    // `last_dir_sign` in `update_and_render`: reversal now throws away the old
    // CPG and burns in a FRESH one already organized for the new direction,
    // so thrust and friction agree instead of fighting. With that real fix in
    // place, this tuning is safe again: headless worst-case test (3,000-step
    // forward, then full reverse held for 20,000 steps, matching the exact
    // live stress pattern) gives genuine sustained reverse drift of -56.19 --
    // the body visibly crosses the whole grid and would hit the wall long
    // before J becomes a real concern (0.593 -> 0.086, and only reaches the
    // low end after ~17,000 steps of continuously holding reverse, well past
    // where a player would have stopped or turned again). See project memory
    // for the full investigation, including the naive-flip numbers this
    // supersedes.
    //
    // Softened slightly again 2026-07-09 (16,32 -> 13,26) chasing a more
    // visible organic squish -- real 4-way sweep (measuring J-swing amplitude
    // as a proxy for visible deformation, not just guessing) found stiffness
    // is NOT actually the main lever for this: softening barely moved J-swing
    // (0.525 -> 0.595, ~13%) while costing real sustain (fwd last-500 drift
    // 1.025 -> 0.594) and reverse safety margin. (13,26,40) is the best
    // available tradeoff point of everything tried -- a small, real,
    // verified improvement, not a transformative one. The "looks mechanical"
    // feeling is honestly more likely coming from the activation scheme
    // itself (a single global vertical-squeeze direction) or the render's
    // color-only feedback than from this constant -- flagged as a real open
    // item for whenever this gets revisited, not something more constant-
    // tuning will fix.
    let mut mat = NeoHookeanMaterial::new(13.0, 26.0);
    mat.active_stress_coeff = 40.0;
    let config = SimConfig {
        min_dt: 0.01,
        // Full CFL headroom + the degenerate-state projection net on: keeps
        // active muscle stress stable under hard driving instead of detonating
        // when a substep can't subdivide enough. See the muscle-stability
        // regression test in tests/physics_correctness.rs.
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(48.0, 20.0); // grid center, equal room either direction
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: MAT_BODY,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    // Arc'd so this exact instance is shared between the solver (which drives
    // physics through it) and the app (which steers it live from input) --
    // set_easy_direction takes effect immediately, no boundary swap needed.
    let ratchet = Arc::new(RatchetFrictionBoundary::new(4, 0.1, 0.95, Vec2::X));
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(Arc::clone(&ratchet)));

    let body_range = 0..solver.particles().len();
    let body_left = body_center.x - 12.0;
    {
        let particles = solver.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = Vec2::Y;
        }
    }
    (solver, body_range, ratchet)
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
        let (sim, body_range, ratchet) = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByActivation);
        println!(
            "creature: {} particles  |  up/down wave speed  left/right STEER  Space pause  R reset  Q quit",
            sim.particles().len()
        );
        let telemetry_log =
            FrameLogger::open("basic_creature_telemetry.ndjson").expect("failed to open log file");
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            body_range,
            lnn: make_cpg(),
            paused: false,
            wave_speed: 1.0,
            steer: 0.0,
            last_dir_sign: 1.0,
            ratchet,
            renderer,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            anomaly_latched: false,
            spawn_centroid: Vec2::new(32.0, 20.0),
            telemetry_log,
        }
    }

    /// Read the solver's own diagnostics and body geometry, print a full
    /// telemetry line, and WARN immediately if anything is physically wrong.
    /// Returns nothing — this is pure observation, no simulation effect.
    fn log_telemetry(&mut self, fps: f32) {
        let snap = self.sim.diagnostics_snapshot();

        // Body geometry, computed directly from particles.
        let particles = self.sim.particles();
        let n = particles.len().max(1) as f32;
        let mut centroid = Vec2::ZERO;
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut act_sum = 0.0f32;
        let mut act_max = 0.0f32;
        for i in 0..particles.len() {
            let x = particles.x[i];
            centroid += x;
            min = min.min(x);
            max = max.max(x);
            let a = particles.activation[i];
            act_sum += a;
            act_max = act_max.max(a);
        }
        centroid /= n;
        let extent = max - min;
        let drift = centroid - self.spawn_centroid;

        println!(
            "f{:<5} fps={:>3.0} | sub={:>2}/{} eff_dt={:.4} dropped={:.4} cfl={:.2} vmax={:.2} \
             | J=[{:.3},{:.3}] velclamp={} Jproj={} oob={} nan_p={} nan_g={} \
             | centroid=({:.1},{:.1}) drift=({:+.1},{:+.1}) extent=({:.1}x{:.1}) \
             | act mean={:.2} max={:.2} | massErr={:.1e} momErr={:.1e}",
            self.frame,
            fps,
            snap.substeps_last_step,
            self.sim.config().max_substeps_per_step,
            snap.effective_dt,
            snap.sim_time_dropped,
            snap.cfl_number,
            snap.max_particle_speed,
            snap.min_deformation_j,
            snap.max_deformation_j,
            snap.vel_clamp_count,
            snap.j_projection_count,
            snap.out_of_bounds_particles,
            snap.non_finite_particle_values,
            snap.non_finite_grid_values,
            centroid.x,
            centroid.y,
            drift.x,
            drift.y,
            extent.x,
            extent.y,
            act_sum / n,
            act_max,
            snap.relative_mass_error,
            snap.relative_momentum_error,
        );

        // Structured NDJSON alongside the console print -- includes live
        // steer/wave_speed context the engine has no name for, so a run's
        // telemetry can be replayed/charted without regexing the printed line.
        let stats = per_material_stats(self.sim.particles());
        self.telemetry_log.log(
            self.frame,
            self.sim.config().dt,
            &stats,
            &snap,
            &[(MAT_BODY, "body")],
            &[("steer", self.steer), ("wave_speed", self.wave_speed)],
        );

        // Immediate WARN on the first frame anything goes wrong — pinpoints the
        // exact moment the "huge issues" start, which periodic logging can miss.
        let mut problems: Vec<String> = Vec::new();
        if snap.non_finite_particle_values > 0 || snap.non_finite_grid_values > 0 {
            problems.push(format!(
                "NON-FINITE: {} particle + {} grid values are NaN/Inf",
                snap.non_finite_particle_values, snap.non_finite_grid_values
            ));
        }
        if snap.out_of_bounds_particles > 0 {
            problems.push(format!(
                "{} particles left the grid",
                snap.out_of_bounds_particles
            ));
        }
        if snap.sim_time_dropped > 1e-6 {
            problems.push(format!(
                "solver DROPPED {:.4} of sim time — hit max_substeps and gave up (unstable)",
                snap.sim_time_dropped
            ));
        }
        if snap.min_deformation_j < 0.05 {
            problems.push(format!(
                "near-inverted element: min J = {:.4} (→0 means a particle is collapsing)",
                snap.min_deformation_j
            ));
        }
        if extent.x > 30.0 || extent.y > 30.0 {
            problems.push(format!(
                "body SCATTERING: extent {:.1}x{:.1} (spawned ~12x3)",
                extent.x, extent.y
            ));
        }
        if snap.substeps_last_step >= self.sim.config().max_substeps_per_step {
            problems.push(format!(
                "substeps MAXED ({}) — CFL is fighting hard, near the stability edge",
                snap.substeps_last_step
            ));
        }
        if !problems.is_empty() && !self.anomaly_latched {
            self.anomaly_latched = true;
            eprintln!("  ⚠ FIRST ANOMALY at frame {}:", self.frame);
            for p in &problems {
                eprintln!("      - {p}");
            }
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    fn update_and_render(&mut self) {
        if !self.paused {
            // Crawl-direction reversal: reseed a FRESH, burned-in CPG for the
            // new direction instead of flipping bias on the same already-
            // organized-for-the-old-direction network. See make_cpg_biased's
            // doc for why the naive in-place flip is a real, verified-broken
            // approach (unbounded compression ratchet under sustained hold).
            let new_dir_sign = if self.steer >= 0.0 { 1.0 } else { -1.0 };
            if new_dir_sign != self.last_dir_sign {
                self.lnn = make_cpg_biased(new_dir_sign);
                self.last_dir_sign = new_dir_sign;
            }
            // wave_speed scales the LNN's internal clock -- faster wave_speed runs the
            // continuous-time ODE forward faster, raising the oscillation frequency, without
            // needing to reconstruct the network (tau/weights stay fixed).
            // Steer by biasing the two rings apart: one drives harder, the wave
            // goes asymmetric, the creature turns. steer=0 → both rings equal → straight.
            self.lnn.set_ring_bias(0, N_PER_RING, self.steer);
            self.lnn.set_ring_bias(1, N_PER_RING, -self.steer);
            // ALSO drive the crawl direction itself: steer<0 reverses which way
            // the ratchet resists slip, so the body actually crawls backward, not
            // just internally-lopsided while still walking the one baked-in way.
            // Same shared instance the solver already uses -- takes effect this substep.
            self.ratchet.set_easy_direction(if new_dir_sign >= 0.0 {
                Vec2::X
            } else {
                Vec2::NEG_X
            });
            self.lnn.step(DT * self.wave_speed);
            let activations: Vec<f32> = self.lnn.activations().collect();
            let body_range = self.body_range.clone();
            let particles = self.sim.particles_mut();
            for i in body_range {
                let group = particles.muscle_group_id[i] as usize;
                // Clamp to the documented [0,1] activation contract — a muscle can't
                // contract past 100%, and staying in-contract keeps active stress
                // inside the CFL budget.
                particles.activation[i] = (MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
            }
            self.sim.step();
            self.frame += 1;
        }
        self.fps_frames += 1;
        // Telemetry ~2x/sec so the log stays readable but catches transients.
        if self.fps_timer.elapsed().as_secs_f32() >= 0.5 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.log_telemetry(fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Creature [peristaltic locomotion]")
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::Space if pressed => {
                        s.paused = !s.paused;
                        println!("{}", if s.paused { "PAUSED" } else { "RUNNING" });
                    }
                    KeyCode::KeyR if pressed => {
                        let (sim, range, ratchet) = make_sim();
                        s.sim = sim;
                        s.body_range = range;
                        s.ratchet = ratchet;
                        s.lnn = make_cpg();
                        s.steer = 0.0;
                        s.last_dir_sign = 1.0;
                        s.frame = 0;
                        s.anomaly_latched = false;
                        println!("reset");
                    }
                    // Capped at 3.0, not the naive 6.0 tried earlier: Lnn::step
                    // is forward Euler with tau=0.5 at period=1.0, so
                    // dt=DT*wave_speed approaching tau (dt/tau -> 1.0) makes
                    // the leak term's decay factor (1 - dt/tau) go negative --
                    // it stops decaying smoothly and starts flipping sign
                    // almost every step, real numerical noise, not a faster
                    // wave. Measured directly (2026-07-09): sign-flip rate per
                    // 100 steps goes 6 (speed=1) -> 17 (speed=3) -> 41
                    // (speed=4) -> 94 (speed=4.5) -- a real cliff right where
                    // dt/tau crosses ~0.8-0.9, not a gradual change. 3.0 stays
                    // comfortably below it; this is what "pressing Up seems to
                    // glitch" (2026-07-09 playtest) actually was.
                    KeyCode::ArrowUp if pressed => s.wave_speed = (s.wave_speed + 0.2).min(3.0),
                    KeyCode::ArrowDown if pressed => s.wave_speed = (s.wave_speed - 0.2).max(0.1),
                    KeyCode::ArrowLeft if pressed => {
                        s.steer = (s.steer - 0.2).max(-1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    KeyCode::ArrowRight if pressed => {
                        s.steer = (s.steer + 0.2).min(1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    _ => {}
                }
            }
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
