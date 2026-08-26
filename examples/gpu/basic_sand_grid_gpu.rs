extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

/// GPU Drucker-Prager sand, with all three real rendering paths this engine
/// offers (see `RenderMode`'s own doc): per-particle, the solver's own P2G
/// mass field (`Renderer::render_grid_volume` -- real, already-shipped
/// MPM-native technique, see `render-pipeline-plan` memory), and real
/// curvature-flow surface reconstruction (`Renderer::
/// render_surface_reconstruction`, van der Laan et al. 2009 -- the SAME
/// technique fluid demos use, wired into a granular scene for the first
/// time 2026-08-26; the technique's own API was always material-agnostic,
/// nothing here needed to change to support it, just nobody had called it
/// on sand before). G cycles all three for direct A/B.
///
///   cargo run --example basic_sand_grid_gpu --features "gpu render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::materials::MaterialModel;
use emerge::render::{ColorMode, GridVolumeSource, Renderer, SurfaceReconstructionSource};
use emerge::{
    DruckerPragerMaterial, FixedStepController, GpuSimulation, MaterialRegistry, SimConfig,
    SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;
const LABELS: &[(u32, &str)] = &[(MAT_LOOSE, "loose"), (MAT_DENSE, "dense")];
// Real scene spacing, shared between the spawn config below and the renderer
// setup in `State::new` (`Renderer::set_particle_spacing_cells`) -- one
// source of truth so the two can't drift out of sync.
const PARTICLE_SPACING_CELLS: f32 = 0.5;
// Real measured sand absorption (Sherman & Waite 1985, iron-oxide quartz sand).
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

/// The three real rendering paths, cycled with G -- same modes/order as
/// `basic_fluids.rs`/`basic_fluids_gpu.rs`. `Surface` (real curvature-flow,
/// van der Laan et al. 2009) was NEVER wired into a granular demo before
/// 2026-08-26 -- this engine's most advanced surface technique existed and
/// worked, just was never given a sand scene to run on. `material_mass_
/// enabled=true` (`SurfaceReconstructionSource`) colors each reconstructed
/// cell from its own real per-material mass, so loose/dense sand stay
/// visually distinct instead of collapsing to one material slot -- same
/// real mechanism `basic_fluids.rs` uses to keep water/mud/ice distinct.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RenderMode {
    Particles,
    GridVolume,
    Surface,
}

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
    render_mode: RenderMode,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Real-time-decoupled stepping -- see `basic_fluids_gpu.rs`'s own field
    /// doc for the full real bug/fix writeup.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    max_steps_seen: usize,
}

fn make_sand(lambda: f32, mu: f32, phi_deg: f32) -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> (GpuSimulation, f32) {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 12,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        // Same already-validated CFL margin as basic_sand.rs's CPU DP-sand.
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = |c: Vec2, mat: u32, seed: u32| SpawnRegion {
        spacing: PARTICLE_SPACING_CELLS,
        box_size: IVec2::new(18, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        rng_seed: seed,
        position_jitter: 0.5,
        ..SpawnRegion::for_sim(&config)
    };
    let mut particles = build_particles(&config, spawn(Vec2::new(17.0, 40.0), MAT_LOOSE, 11));
    particles.extend(build_particles(
        &config,
        spawn(Vec2::new(47.0, 40.0), MAT_DENSE, 22),
    ));
    // Real, derived: this scene's own actual per-particle mass (post grid-
    // density fix -- see project memory "GRID DENSITY ROOT FIX" -- particle
    // mass now comes from real grid density, not a global constant) times
    // real particles-per-cell from this scene's own `spacing` above
    // (particles sit `spacing` cells apart in both axes, so 1/spacing^2 of
    // them tile one cell). NOT a guessed/copied number -- read directly off
    // an actually-constructed particle, so it can't drift out of sync with
    // whatever this scene's material/spawn parameters happen to be.
    let particles_per_cell = (1.0 / PARTICLE_SPACING_CELLS).powi(2);
    let grid_reference_cell_mass = particles[0].mass * particles_per_cell;
    let mut registry = MaterialRegistry::with_default(Box::new(make_sand(2000.0, 3000.0, 20.0)));
    registry.insert(MAT_DENSE, Box::new(make_sand(2000.0, 3000.0, 40.0)));
    let sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    (sim, grid_reference_cell_mass)
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
                required_limits: adapter.limits(),
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
        let (mut sim, grid_reference_cell_mass) = make_sim_data(Arc::new(device), Arc::new(queue));
        // Real per-cell material tracking, needed by the grid-volume path to
        // pick each cell's dominant material (loose vs dense) -- see
        // grid_volume.wgsl's own doc.
        sim.attach_grid_material_render_gpu();
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(sim.queue(), MAT_LOOSE as usize, SIGMA_SAND);
        renderer.set_optical_params(sim.queue(), MAT_DENSE as usize, SIGMA_SAND);
        // Real bug, same one `basic_fluids.rs`/`basic_fluids_gpu.rs` already hit and
        // fixed once (see those files' own comments): `grid_reference_cell_mass`
        // defaults to 1.0, an old "cells weigh order 0.5-4" convention. This
        // scene's real full-cell mass sits far below that default, so
        // GridVolume/Surface's mass_floor (a FRACTION of this value) was being
        // measured against the wrong scale -- discarding almost everything,
        // including a single isolated particle, whose own splat peak never had a
        // real chance to cross the wrongly-scaled floor. Never set here before
        // 2026-08-26 -- this demo simply never got the same fix fluids did.
        renderer.set_grid_reference_cell_mass(grid_reference_cell_mass);
        // TRIED live 2026-08-26, REVERTED: `Renderer::set_particle_spacing_cells`
        // (widens the splat kernel to hit a target effective-neighbor count --
        // see that fn's own doc) was meant to fix grain/flicker/depop on sparse
        // particles, matching the exact symptom `basic_fluids.rs` also hit once.
        // Live result here was worse, not better: this scene ALSO has real
        // velocity-based anisotropic stretch active (every material gets it,
        // not opt-in -- see `curvature_flow.wgsl`'s "Real velocity-stretch
        // extension" doc), and the two compose multiplicatively on splat
        // radius (`radius = BSPLINE_OUTER_LIMIT * scale * max_stretch *
        // splat_w` in `splat_density_main`) -- widening `splat_w` on top of an
        // already motion-elongated kernel merged genuinely separate, isolated
        // particles into a single wrong blob/streak (live-confirmed via
        // screenshot: Particles mode showed correctly scattered dots; Surface
        // mode showed one blob plus a comet-tailed smear) AND cost measurably
        // more per frame, worse at higher push/pull speed (bigger
        // `max_stretch`). Exactly the same "don't stack independently-
        // justified changes blind" lesson `basic_fluids.rs`'s own comment
        // already recorded for a different pair of changes. Reverted pending
        // a real fix that accounts for the composition (e.g. capping combined
        // radius growth, or making the two share one budget) rather than
        // widening the base kernel independently of what motion-stretch is
        // already doing to it.
        // Real, measured (2026-08-26): the engine's default surface_res_
        // multiplier=6 costs 47.6ms/call on this exact scene (headless GPU
        // timing, `diag_surface_reconstruction_real_cost_vs_grid_volume_
        // and_particles`) -- render alone caps ~21fps, before physics. This
        // is the documented "single biggest quality/cost dial" (cost scales
        // with its SQUARE), NOT `curvature_iterations` (measured flat,
        // 45-47ms whether 4 or 8 -- not the real lever here). mult=3 costs
        // 8.95ms (5.3x cheaper) while the surface grid (64*3=192) is still
        // 3x finer than the raw 64-cell physics grid -- a real, disclosed
        // resolution trade, not a free lunch, but a favorable one.
        renderer.set_surface_res_multiplier(3);
        // Real, derived (not left silent): a settled granular pile has no
        // physical mechanism to propagate the Surface path's free-surface
        // wave PDE the way a real fluid does -- `owns_deformation_volume_
        // state()` is the same real property `basic_fluids.rs` uses to
        // opt IN to it. Sand's own type answers `false` here, so this
        // stays at the engine's real inert default (0.0) -- explicit, not
        // an accident of never having been set.
        if make_sand(1.0, 1.0, 30.0).owns_deformation_volume_state() {
            renderer.set_wave_force_coeff(0.35);
        }
        println!(
            "sand grid-volume GPU: {} particles  |  LMB push  RMB pull  G cycle particles/grid/surface view  R reset  Q quit",
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
            // Default to the real curvature-flow Surface mode -- this
            // example exists specifically to show the continuous-surface
            // look; G cycles Particles/GridVolume/Surface for direct A/B.
            render_mode: RenderMode::Surface,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            stepper: FixedStepController::standard(DT, 60.0),
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
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.surface_config.width,
            self.surface_config.height,
            GRID,
        )
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        // Same scene/materials as `State::new`, so `grid_reference_cell_mass`
        // (already set on `self.renderer` once) doesn't need re-deriving here.
        (self.sim, _) = make_sim_data(device, queue);
        self.sim.attach_grid_material_render_gpu();
        self.frame = 0;
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset");
    }

    fn update_and_render(&mut self) {
        if self.lmb || self.rmb {
            let mag = if self.lmb { 12.0 } else { -12.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
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
                "frame={} fps={:.0} render_mode={:?} max_steps_per_render={}",
                self.frame, fps, self.render_mode, self.max_steps_seen
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
            RenderMode::Surface => {
                self.renderer.render_surface_reconstruction(
                    self.sim.device(),
                    self.sim.queue(),
                    SurfaceReconstructionSource {
                        particle_buf: self.sim.particle_buffer(),
                        particle_count: self.sim.particle_count(),
                        grid_res: GRID as u32,
                        material_slot: MAT_LOOSE,
                        material_mass_enabled: true,
                        dt: DT,
                    },
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
                    .with_title("emerge -- Sand, grid-volume render [G: toggle particle view]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
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
                    println!("render mode: {:?}", s.render_mode);
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
