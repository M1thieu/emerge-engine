extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Real, live demo of the moisture/cohesion coupling built this session:
/// water poured onto a loose sand pile diffuses through the shared grid
/// (`ScalarDiffusionField`, PIC/FLIP-blended -- see that field's own doc)
/// and sand's `cohesion_bonus_pa` hook responds with real, cited apparent
/// cohesion (the "sandcastle effect" -- Hornbaker et al. 1997; Halsey &
/// Levine 1998). The sand starts loose enough to genuinely slump under its
/// own gravity; wetted regions should visibly hold together while dry
/// regions keep flowing.
///
/// Water is classified as a moisture source by a REAL PHYSICAL PROPERTY
/// (`MaterialModel::owns_deformation_volume_state()` -- the same condition
/// the engine already uses to mean "behaves like a strict fluid"), not by
/// checking a specific material ID -- see `moisture_source`'s own doc.
///
/// `ColorMode::ByScalarField` (already generic, not built for this demo)
/// renders each particle's own moisture level directly -- dry sand stays
/// its normal color, wet sand lights up, so the diffusion itself is
/// visible, not just its downstream mechanical effect.
///
///   cargo run --example sand_water_saturation --features render
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{ScalarDiffusionConfig, ScalarDiffusionField};
use emerge::{
    DruckerPragerMaterial, MaterialModel, NewtonianFluidMaterial, Particle, SimConfig,
    Simulation, SlipBoundary, SpawnRegion,
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
const MAT_SAND: u32 = 0;
const MAT_WATER: u32 = 1;
// Real, disclosed cap on how much poured water can add beyond the initial
// sand pile -- same reason `basic_sand.rs`'s own POUR_BUDGET exists: the
// renderer's wgpu instance buffer is allocated once, not resizable live.
const POUR_BUDGET: usize = 800;
const POUR_SPACING: f32 = 0.5;
const POUR_BOX: IVec2 = IVec2::new(2, 1);

/// Real apparent-cohesion source, per the pendular-regime capillary model
/// `DruckerPragerMaterial::cohesion_bonus_pa` implements -- see this
/// module's own doc for the citations. Classifies by a REAL PROPERTY
/// (`owns_deformation_volume_state`, true for strict fluids like
/// `NewtonianFluidMaterial`), not a material-ID check -- any future fluid
/// material added to this scene would automatically qualify too.
fn moisture_source(_p: &Particle, phi: f32, material: &dyn MaterialModel) -> f32 {
    const RATE: f32 = 3.0; // phi/s
    if material.owns_deformation_volume_state() && phi < 1.0 {
        RATE
    } else {
        0.0
    }
}

fn make_sand() -> DruckerPragerMaterial {
    DruckerPragerMaterial {
        friction_angle: 27.0_f32.to_radians(), // loose enough to genuinely slump
        saturation_cohesion_coeff: 6.0e4,
        pendular_regime_ceiling: 0.3,
        ..DruckerPragerMaterial::new(2000.0, 3000.0)
    }
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        // 12 (basic_sand.rs's own value) panics: that config was tuned for
        // sand alone, no fluid material in the scene. Strict WC-MPM water
        // has real, tighter CFL/stability requirements -- see
        // basic_fluids.rs's own identical fix, same real precedented value,
        // matching basic_fluids_gpu.rs's own.
        max_substeps_per_step: 150,
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let sand_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(30, 16),
        box_center: Vec2::new(32.0, 38.0),
        material_id: MAT_SAND,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: 11,
        position_jitter: 0.5,
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, sand_spawn)
        .with_default_material(Box::new(make_sand()))
        .with_material(MAT_WATER, Box::new(NewtonianFluidMaterial::low_viscosity(4.0, 10.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

fn make_moisture_field(grid_res: usize) -> ScalarDiffusionField {
    let mut field = ScalarDiffusionField::new(
        ScalarDiffusionConfig {
            diffusivity: 0.5,
            decay_rate: 0.0,
            ambient: 0.0,
        },
        |p| p.scalar_field,
        |p, delta| p.scalar_field += delta,
        grid_res,
    );
    field.source = Some(moisture_source);
    // Real, disclosed PIC-leaning blend -- sand is a purely passive reader
    // here (no source of its own), the exact case FLIP's nullspace-noise
    // failure targets. See `ScalarDiffusionField::blend`'s own doc.
    field.blend = 0.3;
    field
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    pouring: bool,
    poured_count: usize,
    push_strength: f32,
    pour_seed: u32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let mut sim = make_sim();
        sim.attach_scalar_field(make_moisture_field(GRID));
        let real_gravity = sim.config().gravity;
        let render_capacity = sim.particles().len() + POUR_BUDGET;
        let mut renderer = Renderer::new(&gfx.device, render_capacity, gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.9, true);
        // ByScalarField, not ByPhysics -- the whole point of this demo is
        // watching moisture actually spread, not just the material split.
        renderer.set_color_mode(ColorMode::ByScalarField);

        println!(
            "sand_water_saturation: {} particles  |  LMB push  RMB pull  hold P to pour water  R reset  Q quit",
            sim.particles().len()
        );
        println!("Color = moisture level (ByScalarField): dry sand stays dark, wet sand lights up.");
        Self {
            gfx,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            pouring: false,
            poured_count: 0,
            push_strength: 12.0,
            pour_seed: 1000,
            real_gravity,
            gravity_fraction: 0.05,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.9, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            GRID,
        )
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }

        if self.pouring && self.poured_count < POUR_BUDGET {
            let config = self.sim.config();
            let half = POUR_BOX.as_vec2() * 0.5;
            let domain_min = Vec2::splat(config.boundary_thickness as f32) + half;
            let domain_max =
                Vec2::splat((config.grid_res - config.boundary_thickness) as f32) - half;
            let cursor = self
                .cursor_grid()
                .clamp(domain_min, domain_max.max(domain_min));
            self.pour_seed += 1;
            let spawn = SpawnRegion {
                spacing: POUR_SPACING,
                box_size: POUR_BOX,
                box_center: cursor,
                material_id: MAT_WATER,
                precompute_initial_volumes: true,
                initial_velocity_scale: 0.0,
                rng_seed: self.pour_seed,
                position_jitter: 0.3,
                ..SpawnRegion::for_sim(self.sim.config())
            };
            let before = self.sim.particles().len();
            let _ = self.sim.add_body(spawn);
            self.poured_count += self.sim.particles().len() - before;
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            println!(
                "frame={} fps={:.1} particles={}",
                self.frame,
                self.last_fps,
                self.sim.particles().len()
            );
        }

        let output = match self.gfx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(
            &self.gfx.device,
            &self.gfx.queue,
            self.sim.particles(),
            &view,
            true,
        );

        let fps = self.last_fps;
        let mut push_strength = self.push_strength;
        let mut gravity_fraction = self.gravity_fraction;
        let n_particles = self.sim.particles().len();
        let poured = self.poured_count;
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Sand + Water Saturation")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=40.0));
                    ui.separator();
                    ui.label(format!("Water poured: {poured}/{POUR_BUDGET}"));
                    ui.add(
                        egui::ProgressBar::new(poured as f32 / POUR_BUDGET as f32)
                            .desired_width(200.0),
                    );
                    ui.separator();
                    ui.label("Dry sand slumps freely. Pour water (P) on part of");
                    ui.label("the pile -- the wet region should hold its shape");
                    ui.label("while the dry region keeps flowing.");
                    ui.separator();
                    ui.label("LMB push  RMB pull  hold P to pour  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.gravity_fraction = gravity_fraction;
        if reset {
            let mut sim = make_sim();
            sim.attach_scalar_field(make_moisture_field(GRID));
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
            self.poured_count = 0;
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
                    .with_title("emerge -- Sand + Water Saturation")
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
            let resp = s.gfx.egui_state.on_window_event(w, &event);
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
                    KeyCode::KeyP => s.pouring = pressed,
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyR if pressed => {
                        let mut sim = make_sim();
                        sim.attach_scalar_field(make_moisture_field(GRID));
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
                        s.poured_count = 0;
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
