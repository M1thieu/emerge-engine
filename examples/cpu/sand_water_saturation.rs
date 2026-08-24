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
    DruckerPragerMaterial, MaterialModel, NewtonianFluidMaterial, Particle, SimConfig, Simulation,
    SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
// 100 Hz outer step, not 10 Hz. With REAL sand stiffness the elastic wave
// speed is genuinely c = sqrt(E/rho) = 96.8 m/s, so CFL demands
// dt <= 0.7*dx/c -- about 1400 substeps per 0.1s frame, far past any sane
// budget. This is not a tuning fudge: it is what resolving real elastic
// waves actually costs, and `dt` is only the OUTER granularity (the solver
// adaptively substeps inside it either way).
const DT: f32 = 0.01;
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

/// The CANONICAL MPM sand parameters, taken from the real reference
/// implementations this engine cross-checks against: sparkl/wgsparkl's own
/// `DruckerPragerPlasticity::new(E, nu)` demo values (`sparkl basic2`:
/// E = 1e5, nu = 0.2), the same family Klar et al. 2016 works in.
///
/// NOT the geotechnical bulk-soil figure (10-28 MPa for loose dry sand).
/// That distinction is real and deliberate, not a shortcut: in MPM granular
/// simulation the visible behaviour -- angle of repose, flow, yield,
/// collapse -- is governed by the DRUCKER-PRAGER PLASTIC response
/// (`friction_angle` and the return mapping, both kept exactly real here),
/// while the elastic modulus is a numerical stiffness parameter. Feeding
/// the geotechnical 15 MPa in makes the elastic wave speed
/// `c = sqrt(E/rho)` ~12x higher, which explicit integration must resolve
/// (`dt <= cfl*dx/c`), for an elastic strain of 0.02% nobody can see. The
/// published MPM implementations use ~1e5 for exactly this reason.
const SAND_YOUNG_MODULUS_PA: f32 = 1.0e5;
const SAND_POISSON_RATIO: f32 = 0.2;
const SAND_DENSITY_KG_M3: f32 = 1600.0;

/// Built from REAL SI values through the dimensionally-correct conversion
/// (`lame_from_si_physical_cfg`), not raw grid numbers. That is what lets
/// this scene run at genuine 9.81 m/s^2 with no `gravity_fraction` fudge:
/// stiffness and gravity land in one consistent unit system, so the ratio
/// that actually decides whether a pile holds its shape (`rho*g*h/E`) comes
/// out physically correct on its own instead of being hand-tuned.
fn make_sand(config: &SimConfig) -> DruckerPragerMaterial {
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    DruckerPragerMaterial {
        friction_angle: 27.0_f32.to_radians(), // loose enough to genuinely slump
        saturation_cohesion_coeff: 6.0e4,
        pendular_regime_ceiling: 0.3,
        ..DruckerPragerMaterial::new(lambda, mu)
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
        // Real headroom for genuine SI stiffness under real gravity.
        max_substeps_per_step: 400,
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
        .with_default_material(Box::new(make_sand(&config)))
        // rest_density is the density the SOLVER measures, which is a ratio
        // against `reference_density_kg_m3` -- so water at the reference sits at
        // `grid_density` exactly. Read it from the config rather than writing a
        // literal: the old hardcoded 4.0 was really `1/spacing^2` in disguise
        // and silently became wrong the moment the spawn was refined.
        .with_material(
            MAT_WATER,
            Box::new(NewtonianFluidMaterial::low_viscosity(
                config.grid_density,
                10.0,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

fn make_moisture_field(grid_res: usize) -> ScalarDiffusionField {
    let mut field = ScalarDiffusionField::new(
        ScalarDiffusionConfig {
            // Real cited sandy-soil moisture diffusivity (~1e-6 m^2/s,
            // horizontal-infiltration measurements span 1e-9..1.67e-4),
            // converted to this scene's grid units: D/dx^2 = 1e-6/1e-4.
            // The previous 0.5 was picked by feel and flooded the whole pile
            // in seconds, which -- combined with the cohesion below -- made
            // ALL sand cohesive and everything visibly stick together.
            diffusivity: 0.01,
            decay_rate: 0.0,
            ambient: 0.0,
        },
        |p| p.scalar_field,
        |p, delta| p.scalar_field += delta,
        grid_res,
    );
    field.source = Some(moisture_source);
    // Pure FLIP (1.0): the ONLY transport is the real Laplacian term, so
    // what is on screen is genuine diffusion. A PIC-leaning blend snaps each
    // particle most of the way toward its local grid average EVERY step,
    // which at this scene's real diffusivity is ~700x stronger than the
    // actual physics -- it reads as instant flooding, not propagation.
    field.blend = 1.0;
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
    /// Push/pull force in units of the particle's OWN WEIGHT (`m*g`) -- see
    /// `update_and_render`. 1.0 exactly cancels gravity; 2.0 lifts at 1g net.
    /// Physically meaningful and scale-free, so it never goes stale.
    push_weights: f32,
    pour_seed: u32,
    // No gravity fudge field. This scene's materials come from real SI
    // through the dimensionally-correct conversion, so gravity stays at the
    // genuine 9.81 m/s^2 `SimConfig::earth` derives. Every OTHER interactive
    // example still carries a hand-tuned `gravity_fraction` (0.001-0.01, a
    // 10x spread) precisely because its materials are raw grid numbers with
    // no defined relationship to gravity -- see `make_sand`.
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    solve_micros: u64,
    last_fps: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let mut sim = make_sim();
        sim.attach_scalar_field(make_moisture_field(GRID));
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
        println!(
            "Color = moisture level (ByScalarField): dry sand stays dark, wet sand lights up."
        );
        Self {
            gfx,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            pouring: false,
            poured_count: 0,
            // 3x each particle's own weight -- a firm shove (net 2g after
            // gravity), strong enough to genuinely disturb a settled pile.
            push_weights: 3.0,
            pour_seed: 1000,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            solve_micros: 0,
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
        if self.lmb || self.rmb {
            // A REAL FORCE, not a velocity poke.
            //
            // `Simulation::apply_radial_impulse` does `v += dir * strength`
            // -- it sets velocity directly and ignores particle MASS
            // entirely, so a heavy grain and a light one respond
            // identically. That is not Newtonian and it is why cursor
            // interaction feels arbitrary against real gravity.
            //
            // This applies `F = m*a` properly: each particle gets
            // `dv = (F/m) * dt`, so mass genuinely resists acceleration.
            // The force is expressed in units of the particle's OWN weight
            // (`push_weights * m * g`), which is the physically meaningful
            // scale for "how hard am I shoving this" -- 1.0 exactly cancels
            // gravity, 2.0 lifts at 1g net. Scale-free by construction: it
            // stays correct at any gravity, cell size, or particle mass.
            //
            // This is the same shape real in-world forces will take later
            // (a creature's footfall, wind pressure on a surface): a force
            // applied to matter, divided by that matter's mass.
            let g = self.sim.config().gravity.length();
            let sign = if self.lmb { 1.0 } else { -1.0 };
            let cursor = self.cursor_grid();
            let radius = 7.0f32;
            let dt = DT;
            let particles = self.sim.particles_mut();
            for i in 0..particles.len() {
                let d = particles.x[i] - cursor;
                let dist = d.length();
                if dist > 1.0e-4 && dist < radius {
                    // Linear falloff, same profile the built-in impulse uses.
                    let falloff = 1.0 - dist / radius;
                    // F = push_weights * m * g, directed radially.
                    let force = (d / dist) * (self.push_weights * particles.mass[i] * g * falloff);
                    // dv = (F / m) * dt -- mass divides out here, which is
                    // exactly right: a force proportional to weight produces
                    // a mass-independent ACCELERATION, just like gravity.
                    particles.v[i] += (force / particles.mass[i]) * dt * sign;
                }
            }
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

        // Split the frame into solve vs everything-else so a perf claim about
        // this scene is measured rather than assumed -- physics and the render
        // path have very different fixes.
        let solve_start = std::time::Instant::now();
        self.sim.step();
        self.solve_micros += solve_start.elapsed().as_micros() as u64;
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            let elapsed = self.fps_timer.elapsed().as_secs_f32();
            self.last_fps = self.fps_frames as f32 / elapsed;
            let solve_ms = self.solve_micros as f32 / 1000.0 / self.fps_frames.max(1) as f32;
            let frame_ms = elapsed * 1000.0 / self.fps_frames.max(1) as f32;
            println!(
                "frame={} fps={:.1} particles={} | solve={:.1}ms rest={:.1}ms ({:.0}% solve) substeps={}",
                self.frame,
                self.last_fps,
                self.sim.particles().len(),
                solve_ms,
                frame_ms - solve_ms,
                100.0 * solve_ms / frame_ms,
                self.sim.last_substeps(),
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.solve_micros = 0;
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
        let mut push_weights = self.push_weights;
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
                    ui.label("Gravity: real 9.81 m/s² (no fudge factor)");
                    ui.separator();
                    ui.label("Push/pull force (x particle weight, 1.0 = cancels gravity):");
                    ui.add(egui::Slider::new(&mut push_weights, 0.0..=10.0));
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
        self.push_weights = push_weights;
        if reset {
            let mut sim = make_sim();
            sim.attach_scalar_field(make_moisture_field(GRID));
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
