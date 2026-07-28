extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// `basic_fluids.rs` (Newtonian water dam-break + Bingham mud blob) with a real,
/// live egui panel -- same pattern as `basic_sand_gui.rs`/`basic_snow_gui.rs`:
/// real gravity slider (1.0 = genuine IRL 9.81 m/s²), push/pull, and directional-
/// drag digging (the SAME proven mechanism from `basic_sand_gui.rs`: a per-particle
/// velocity nudge along the cursor's own movement, no second body, no contact_group
/// tuning -- mass-conserving by construction). Materials and the dam-break setup
/// are unchanged from `basic_fluids.rs`.
///
/// Real phase range: water below 273K freezes into ice
/// (`NeoHookeanMaterial` wrapped in `WithLatentHeat(-334_000.0)`, same real
/// exothermic value `latent_heat.rs` already uses, via
/// `Simulation::thermal_config_mut()` -- a real, already-existing engine
/// hook, not new plumbing). Exposed as a discrete Warm/Cold toggle, not a
/// continuous slider -- a continuously-tunable gravity/temperature slider
/// pair turns into a hunt-for-the-right-value loop; a two-state toggle proves
/// the same real phase transition without that.
///
/// Real gravity default: 0.01 (same checkpoint already validated for sand and
/// snow at this identical grid scale). If you push the slider toward 1.0
/// (full IRL), run with `--release`: the debug build has a stutter around
/// collision moments from unoptimized bounds-checks/allocator overhead, not a
/// physics cost.
///
///   cargo run --example basic_fluids_gui --features render
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    BinghamFluidMaterial, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion, WithLatentHeat,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;
const MAT_ICE: u32 = 2;
const FREEZING_POINT: f32 = 273.0;
const ICE_LATENT_HEAT: f32 = -334_000.0; // exothermic: freezing releases energy (real water: 334 kJ/kg)
const WARM_AMBIENT: f32 = 300.0;
const COLD_AMBIENT: f32 = 250.0;
// 1/s, Newton cooling rate for the "Cold ambient" toggle -- models convective
// heat loss to surrounding cold air (a freezer), not bulk Fourier conduction
// through the water's own interior (too slow to visibly freeze this scene's
// real ~0.6m water column within any reasonable play session). Chosen
// empirically for a reasonable interactive wait (ice appears within ~2
// minutes).
const FREEZER_COOLING_RATE: f32 = 0.08;
// Radius of the directional dig nudge, grid cells -- matches basic_sand_gui.rs.
const DIG_RADIUS: f32 = 4.0;

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-3,
        max_substeps_per_step: 8,
        recompute_density_each_step: true,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Real water: Cole 1948 Tait exponent (7.0) + real dynamic viscosity, not a
    // hand-picked pair -- see NewtonianFluidMaterial::low_viscosity.
    let water = NewtonianFluidMaterial::low_viscosity(4.0, 10.0);
    let mud = BinghamFluidMaterial::new(4.0, 8.0, 5.0, 3.0, 4.0);
    let ice = WithLatentHeat::new(NeoHookeanMaterial::new(4.0, 8.0), ICE_LATENT_HEAT);
    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6,
            heat_capacity: 4182.0,
            density: 1000.0, // kg/m^3, real water -- see ThermalConfig::density's own doc
            ambient: WARM_AMBIENT,
            // Must match the sim's real dx_meters -- ThermalConfig::grid_cell_size
            // requires this, else alpha_grid() is mis-scaled by orders of
            // magnitude.
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );
    let spawn_water = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(11.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_mud = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(16, 18),
        box_center: Vec2::new(50.0, 38.0),
        material_id: MAT_MUD,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_material(MAT_MUD, Box::new(mud))
        .with_material(MAT_ICE, Box::new(ice))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_thermal(thermal)
        .with_phase_rule(|p| {
            if p.material_id == MAT_WATER && p.temperature < FREEZING_POINT {
                Some(MAT_ICE)
            } else {
                None
            }
        });
    // `add_body` appends particles synchronously, so it must run BEFORE this
    // temperature-init loop -- otherwise mud particles are left at
    // `initialize_particles`'s default (effectively 0K), not WARM_AMBIENT.
    let _ = solver.add_body(spawn_mud);
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = WARM_AMBIENT;
    }
    solver
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
    cursor_pos: [f32; 2],
    last_cursor_grid: Vec2,
    lmb: bool,
    rmb: bool,
    digging: bool,
    push_strength: f32,
    dig_strength: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    cold: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
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
        let sim = make_sim();
        let real_gravity = sim.config().gravity;
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);

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

        println!(
            "basic_fluids_gui: {} particles  |  LMB push  RMB pull  D toggle dig  R reset  Q quit",
            sim.particles().len()
        );
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
            cursor_pos: [0.0; 2],
            last_cursor_grid: Vec2::ZERO,
            lmb: false,
            rmb: false,
            digging: false,
            push_strength: 5.0,
            dig_strength: 18.0,
            real_gravity,
            // 0.01, matching the already-validated sand/snow checkpoint at this
            // same grid scale -- not re-guessed live.
            gravity_fraction: 0.01,
            cold: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
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

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if let Some(cfg) = self.sim.thermal_config_mut() {
            cfg.ambient = if self.cold {
                COLD_AMBIENT
            } else {
                WARM_AMBIENT
            };
            // Convective (Newton) cooling, not bulk conduction -- see
            // FREEZER_COOLING_RATE's own doc; bulk conduction alone is too slow
            // to ever visibly freeze in a play session.
            cfg.cooling_rate = if self.cold { FREEZER_COOLING_RATE } else { 0.0 };
        }
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }
        // Digging: nudges nearby particles along the cursor's OWN movement
        // direction (a furrow/stir), not radially like push/pull -- same
        // proven mechanism as basic_sand_gui.rs, applies just as validly to
        // a fluid (a local directional velocity nudge IS a real stir/drag).
        let cursor = self.cursor_grid();
        if self.digging {
            let delta = cursor - self.last_cursor_grid;
            if delta.length_squared() > 1.0e-8 {
                let dir = delta.normalize();
                let particles = self.sim.particles_mut();
                for i in 0..particles.len() {
                    if (particles.x[i] - cursor).length() < DIG_RADIUS {
                        particles.v[i] += dir * self.dig_strength * DT;
                    }
                }
            }
        }
        self.last_cursor_grid = cursor;
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
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

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let mut push_strength = self.push_strength;
        let mut dig_strength = self.dig_strength;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let mut cold = self.cold;
        let water_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_WATER)
            .count();
        let mud_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_MUD)
            .count();
        let ice_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_ICE)
            .count();
        let mut reset = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Fluids")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}"));
                    ui.label(format!("water={water_n}  mud={mud_n}  ice={ice_n}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s², use --release above ~0.1):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=20.0));
                    ui.checkbox(&mut digging, "Digging/stirring active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.checkbox(&mut cold, "Cold ambient (water freezes below 273K)");
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.dig_strength = dig_strength;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        self.cold = cold;
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
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
                    .with_title("emerge -- Fluids (GUI)")
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
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
                        println!("reset");
                    }
                    _ => {}
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
