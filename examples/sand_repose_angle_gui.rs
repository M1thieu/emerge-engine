extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// Merged replacement for the former `sand_pile_stability_gui.rs` and
/// `sand_collapse_true_repose_gui.rs` -- both were really the same real
/// question (does sand reach a genuine ~30 deg angle of repose?) asked
/// from two different starting conditions, sharing almost all of their
/// GUI boilerplate. One demo, real mode switch between them.
///
/// **Pre-shaped mode**: a pile built ALREADY at 30 deg (zero initial
/// velocity) -- tests whether it HOLDS. Real, tested recipe
/// (`tests/accuracy.rs::unconfined_pile_with_cundall_damping_reaches_
/// real_repose_angle`, found 2026-07-26): the engine's own self-consistent
/// Drucker-Prager return mapping (unconditional default) + `apic_blend=
/// 0.05` + `cundall_damping=1.0` (Cundall 1982, production use in Anura3D
/// geotechnical MPM) holds flat for 100,000+ steps headless.
///
/// **Collapse mode**: a tall column that genuinely TOPPLES and settles
/// near the real target (`tests/accuracy.rs::sand_collapse_with_phase_
/// gated_relaxation_after_dynamics`, 2026-08-01). Real root cause found
/// that session: `SimConfig::standard`'s own default `apic_blend=1.0`
/// (full APIC, zero numerical dissipation) is genuinely UNSTABLE for a
/// violent collapse -- confirmed via a widened-domain test where reach
/// grew roughly in proportion to whatever domain size was given. `apic_
/// blend=0.6` is stable without over-damping the real collapse motion
/// (0.05 over-damps it, landing too steep at 44-51 deg). Press H once it
/// settles to switch to the pre-shaped mode's own holding recipe -- watch
/// the SAME real "excess creep" mechanism the patient-pour investigation
/// found: continued relaxation drifts the angle back DOWN past the real
/// target, it does not hold it steady the way the pre-shaped case does
/// (open question as of this session: whether it ever truly plateaus
/// given a long enough horizon, or keeps drifting toward flat -- see
/// `sand_collapse_relaxation_long_horizon_plateau_check`).
///
///   cargo run --example sand_repose_angle_gui --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, FrameLogger, FrictionBoundary, SimConfig, Simulation, SpawnRegion,
    per_material_stats,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// Same push/pull cursor convention as every other sand example (basic_sand_gui.rs):
// LMB push, RMB pull, `apply_radial_impulse` at a fixed real radius.
//
// REAL BUG FOUND AND FIXED: 7.0 (basic_sand_gui.rs's own value) was copied
// without checking it against THIS demo's own much smaller pile -- that
// scene's sand mass spans a wide multi-body area, so 7 cells is a small
// local nudge there. This pile is only 12 cells tall / ~41 wide (see
// PRESHAPED_HEIGHT_CELLS), so a 7-cell-radius push centered mid-pile covers
// MORE than its full height and a third of its width -- looks like "the
// whole pile moves together," not because of any real coupling bug (`apply_
// radial_impulse` is a plain per-particle radial-falloff kick, verified
// local-only), just a radius picked for the wrong scene. 2.5 keeps a push
// a real local nudge (a few grains near the cursor), not a bulk shove.
const PUSH_RADIUS_CELLS: f32 = 2.5;

const GRID: usize = 128;
const FLOOR: f32 = 2.0;
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

// Pre-shaped mode's own real, tested geometry.
const PRESHAPED_DT: f32 = 0.016;
const TARGET_ANGLE_DEG: f32 = 30.0;
const PRESHAPED_HEIGHT_CELLS: f32 = 12.0;

// Collapse mode's own real, tested geometry (matches
// `sand_angle_of_repose_is_physical`'s column, different DT).
const COLLAPSE_DT: f32 = 0.1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    PreShaped,
    Collapse,
}

fn measure_angle_deg(xs: &[Vec2]) -> (f32, f32, f32) {
    let n = xs.len() as f32;
    if n == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
    // REAL BUG FOUND AND FIXED: after enough interactive pushing/pulling
    // spreads the pile wide enough, NO particle can fall within +-2 cells
    // of center_x -- the old `fold(f32::MIN, f32::max)` then silently
    // returned its untouched sentinel (f32::MIN, not a real height),
    // observed live via the NDJSON log (`height=-3.4e38`, `angle_deg=-90`).
    // `max_by`'s `Option` return makes "nothing matched" explicit; falling
    // back to the pile's own overall max y (a real, sensible answer: the
    // tallest particle anywhere) instead of a fixed band's sentinel.
    let height = xs
        .iter()
        .filter(|p| (p.x - center_x).abs() < 2.0)
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let height = if height.is_finite() {
        height
    } else {
        xs.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max)
    } - FLOOR;
    let base_half_width = xs
        .iter()
        .filter(|p| p.y < FLOOR + 1.5)
        .map(|p| (p.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let angle = (height / base_half_width.max(0.1)).atan().to_degrees();
    (height, base_half_width, angle)
}

fn make_sim(mode: Mode) -> Simulation {
    match mode {
        Mode::PreShaped => {
            // Exact real, tested recipe -- `tests/accuracy.rs::
            // unconfined_pile_with_cundall_damping_reaches_real_repose_
            // angle`, reproduced parameter-for-parameter.
            let config = SimConfig {
                max_substeps_per_step: 64,
                apic_blend: 0.05,
                cundall_damping: 1.0,
                ..SimConfig::standard(GRID, PRESHAPED_DT, Vec2::new(0.0, -0.3))
            };
            let cx = GRID as f32 * 0.5;
            let hb = PRESHAPED_HEIGHT_CELLS / TARGET_ANGLE_DEG.to_radians().tan();
            let spawn = SpawnRegion {
                spacing: 0.25,
                box_size: IVec2::new(
                    (2.0 * hb).ceil() as i32 + 4,
                    PRESHAPED_HEIGHT_CELLS.ceil() as i32 + 4,
                ),
                box_center: Vec2::new(cx, FLOOR + 2.0 + PRESHAPED_HEIGHT_CELLS * 0.5),
                material_id: 0,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            };
            let sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
            let mut solver = Simulation::new(config, spawn)
                .with_default_material(Box::new(sand))
                .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
            // Carve the spawned bounding box down to a triangular
            // cross-section at exactly TARGET_ANGLE_DEG -- starts already
            // in its final shape (zero initial velocity): this mode is
            // about whether it HOLDS, not whether a collapse settles there.
            solver.retain_particles(|p| {
                let dy = p.x.y - FLOOR;
                let dx = (p.x.x - cx).abs();
                (0.0..=PRESHAPED_HEIGHT_CELLS).contains(&dy)
                    && dx <= hb * (1.0 - dy / PRESHAPED_HEIGHT_CELLS).max(0.0)
            });
            solver
        }
        Mode::Collapse => {
            // Real, swept, confirmed value -- see this file's own top doc.
            let config = SimConfig {
                max_substeps_per_step: 64,
                apic_blend: 0.6,
                cundall_damping: 0.0,
                ..SimConfig::standard(GRID, COLLAPSE_DT, Vec2::new(0.0, -0.3))
            };
            let column = SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(8, 16),
                box_center: Vec2::new(GRID as f32 * 0.5, FLOOR + 8.0),
                material_id: 0,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            };
            let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
            // Real, calibrated tonight (2026-08-02): edge-triggered elastic-strain
            // reset, grounded in Cundall 1982's kinetic-damping peak-reset --
            // fires once per particle on a real strain-rate falling edge, matching
            // the proven F-only-reset target bit-for-bit (29.6deg) instead of the
            // unarrested creep this scene showed before.
            sand.post_event_relax_threshold = 0.001;
            Simulation::new(config, column)
                .with_default_material(Box::new(sand))
                .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)))
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    mode: Mode,
    holding: bool,
    paused: bool,
    step: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    push_strength: f32,
    logger: FrameLogger,
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
        let mode = Mode::PreShaped;
        let sim = make_sim(mode);
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.7, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&queue, 0, SIGMA_SAND);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window.as_ref(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            fmt,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        let log_path = std::env::temp_dir().join("emerge_sand_repose_angle_gui.ndjson");
        let logger = FrameLogger::open(&log_path).unwrap();
        println!(
            "sand_repose_angle_gui: {} particles  |  M=toggle mode  H=toggle holding (collapse mode only)  LMB=push RMB=pull  SPACE=pause  R=reset  Q=quit",
            sim.particles().len()
        );
        println!("per-frame diagnostics log: {}", log_path.display());
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
            mode,
            holding: false,
            paused: false,
            step: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            // Lower than basic_sand_gui.rs's own 12.0 default -- real,
            // measured live: holding the button re-applies this force EVERY
            // frame with no decay (same convention every sand demo uses), so
            // it's the DURATION held, not the radius, that determines how
            // fast particles end up going. 4.0 keeps a brief tap a gentle
            // nudge; the slider still reaches 40 for a deliberate shove.
            push_strength: 4.0,
            logger,
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
            .set_camera(&self.queue, GRID as u32, w, h, 0.7, true);
    }

    /// Real bug found live: this window is 720x480 (NOT square), and
    /// `Renderer::set_camera` computes a non-uniform `(sx,tx,sy,ty)` for any
    /// non-square viewport (see its own doc/impl) -- the simple "screen
    /// fraction * GRID" formula (correct only when width==height, the
    /// convention every OTHER sand demo's square window happens to satisfy)
    /// silently mis-locates the cursor here. Mirrors `set_camera`'s own
    /// exact math, then inverts it, instead of guessing a corrective factor.
    fn cursor_grid(&self) -> Vec2 {
        let w = self.surface_config.width.max(1) as f32;
        let h = self.surface_config.height.max(1) as f32;
        let aspect = w / h;
        let gr = GRID as f32;
        let (sx, tx, sy, ty) = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        let ndc_x = 2.0 * (self.cursor_pos[0] / w) - 1.0;
        let ndc_y = 1.0 - 2.0 * (self.cursor_pos[1] / h);
        Vec2::new((ndc_x - tx) / sx, (ndc_y - ty) / sy)
    }

    fn reset(&mut self) {
        self.sim = make_sim(self.mode);
        self.holding = false;
        self.step = 0;
        println!("reset: mode={:?}", self.mode);
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::PreShaped => Mode::Collapse,
            Mode::Collapse => Mode::PreShaped,
        };
        self.reset();
    }

    fn toggle_holding(&mut self) {
        if self.mode == Mode::PreShaped {
            println!(
                "already holding in pre-shaped mode -- press M to switch to collapse mode first"
            );
            return;
        }
        self.holding = !self.holding;
        if self.holding {
            self.sim.set_apic_blend(0.05);
            self.sim.set_cundall_damping(1.0);
            println!(
                "holding mode ON (apic_blend=0.05, cundall_damping=1.0) -- watch it creep past the target"
            );
        } else {
            self.sim.set_apic_blend(0.6);
            self.sim.set_cundall_damping(0.0);
            println!(
                "holding mode OFF (apic_blend=0.6, cundall_damping=0.0) -- real collapse dynamics"
            );
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim
                .apply_radial_impulse(self.cursor_grid(), PUSH_RADIUS_CELLS, mag);
        }
        if !self.paused {
            self.sim.step();
            self.step += 1;
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        // Real per-frame diagnostics -- so pushing/pulling the pile and
        // watching whether it re-settles at a real stable angle (or keeps
        // sliding) can be verified after the fact from the log, not just
        // eyeballed live. `is_pushing`/`is_pulling` and the cursor's own
        // grid position ride in `extra` (app-specific context the generic
        // snapshot has no name for), same slot `rod_blade_of_grass_gui.rs`
        // already uses for its own steer input.
        let (height, half_w, angle) = measure_angle_deg(&self.sim.particles().x);
        let max_speed = self
            .sim
            .particles()
            .v
            .iter()
            .fold(0.0f32, |m, v| m.max(v.length()));
        let snap = self.sim.diagnostics_snapshot();
        let stats = per_material_stats(self.sim.particles());
        let cursor = self.cursor_grid();
        self.logger.log(
            self.step,
            snap.effective_dt,
            &stats,
            &snap,
            &[],
            &[
                ("height", height),
                ("half_width", half_w),
                ("angle_deg", angle),
                ("max_speed", max_speed),
                ("holding", if self.holding { 1.0 } else { 0.0 }),
                ("is_pushing", if self.lmb { 1.0 } else { 0.0 }),
                ("is_pulling", if self.rmb { 1.0 } else { 0.0 }),
                ("cursor_x", cursor.x),
                ("cursor_y", cursor.y),
                ("fps", self.last_fps),
            ],
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);

        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let step = self.step;
        let mode = self.mode;
        let holding = self.holding;
        let mut paused = self.paused;
        let mut push_strength = self.push_strength;
        let mut do_reset = false;
        let mut do_toggle_mode = false;
        let mut do_toggle_holding = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Sand: angle of repose")
                .default_pos([10.0, 10.0])
                .default_width(320.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  step={step}"));
                    ui.separator();
                    let mode_label = match mode {
                        Mode::PreShaped => "PRE-SHAPED (apic=0.05, cundall=1.0, always holding)",
                        Mode::Collapse if holding => {
                            "COLLAPSE, holding ON (apic=0.05, cundall=1.0)"
                        }
                        Mode::Collapse => "COLLAPSE (apic=0.6, cundall=0.0)",
                    };
                    ui.label(format!("mode = {mode_label}"));
                    ui.label(format!("height={height:.2}  half-width={half_w:.2}"));
                    ui.label(format!("current angle = {angle:.1} deg"));
                    ui.label("real dry sand IRL = 30-35 deg");
                    ui.separator();
                    match mode {
                        Mode::PreShaped => {
                            ui.label("Real, tested: holds flat 100,000+ steps headless.");
                            ui.label("Watch it NOT slide away.");
                        }
                        Mode::Collapse => {
                            ui.label("apic_blend=1.0 (engine default) is UNSTABLE here --");
                            ui.label("spreads without bound. 0.6 is stable AND accurate.");
                            ui.label("Toggle H once it settles: holding mode creeps the");
                            ui.label("angle back DOWN past target -- don't over-relax it.");
                        }
                    }
                    ui.separator();
                    ui.label("Push/pull strength (LMB push, RMB pull):");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=40.0));
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused (or SPACE)");
                    ui.horizontal(|ui| {
                        if ui.button("M: toggle mode").clicked() {
                            do_toggle_mode = true;
                        }
                        if ui.button("H: toggle holding").clicked() {
                            do_toggle_holding = true;
                        }
                        if ui.button("Reset").clicked() {
                            do_reset = true;
                        }
                    });
                    ui.label("Q quit");
                });
        });
        self.paused = paused;
        self.push_strength = push_strength;
        if do_toggle_mode {
            self.toggle_mode();
        } else if do_toggle_holding {
            self.toggle_holding();
        } else if do_reset {
            self.reset();
        }

        self.egui_state
            .handle_platform_output(window, full_output.platform_output);
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let sd = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let cmd = {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut enc, &tris, &sd);
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut rp, &tris, &sd);
            drop(rp);
            enc.finish()
        };
        self.queue.submit(std::iter::once(cmd));
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        output.present();
    }
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Sand: Angle of Repose (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(720u32, 480u32)),
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
        if let Some(w) = &self.window {
            let resp = s.egui_state.on_window_event(w, &event);
            if resp.consumed {
                return;
            }
        }
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
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == ElementState::Pressed;
                if pressed {
                    match key {
                        KeyCode::KeyM => s.toggle_mode(),
                        KeyCode::KeyH => s.toggle_holding(),
                        KeyCode::KeyR => s.reset(),
                        KeyCode::Space => s.paused = !s.paused,
                        KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                if let Some(w) = &self.window {
                    let w = w.clone();
                    s.update_and_render(&w);
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
