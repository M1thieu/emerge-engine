extern crate emerge_engine as emerge;

use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
/// CPU three-material showcase -- sand terrain, fluid pool, elastic blob.
///
///   Mat 0  NeoHookean elastic (blue)  -- creature body, arrow-key drive
///   Mat 1  Sand Drucker-Prager (gold) -- terrain
///   Mat 2  Newtonian fluid  (cyan)    -- water pool
///
///   ^v<>  drive elastic blob  |  LMB push  RMB pull  |  R reset  Q quit
///   cargo run --example basic_showcase --features "render"
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const GRID: usize = 64;
const DT: f32 = 0.1;
const ELASTIC_ID: u32 = 0;
const SAND_ID: u32 = 1;
const FLUID_ID: u32 = 2;
const SPACING: f32 = 0.7;

struct State {
    sim: Simulation,
    renderer: Renderer,
    cursor_frac: [f32; 2],
    lmb: bool,
    rmb: bool,
    arrow_up: bool,
    arrow_down: bool,
    arrow_left: bool,
    arrow_right: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 0.005,
        max_substeps_per_step: 16,
        recompute_density_each_step: true,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let elastic = NeoHookeanMaterial::new(40.0, 80.0);
    let sand = DruckerPragerMaterial::new(400.0, 200.0);
    // Real water: Cole 1948 Tait exponent (7.0) + real dynamic viscosity, not a
    // hand-picked 0.1/4.0 pair -- see NewtonianFluidMaterial::low_viscosity.
    // rest_density=0.1, NOT the old 4.0 -- real SI fix, 2026-08-08, see
    // basic_fluids.rs's own doc for the full derivation.
    // eos_stiffness=0.25, NOT 10 -- rest_density shrinking 40x makes
    // `timestep_bound`'s c2 (sound-speed-squared) 40x larger at the old
    // stiffness for the same compression; confirmed by a real crash in
    // basic_fluids.rs's CPU twin. Rescaling stiffness by the same factor
    // (10*0.1/4.0=0.25) restores the original, already-stable c2 -- see
    // basic_fluids.rs's own doc for the full derivation.
    let fluid = NewtonianFluidMaterial::low_viscosity(0.1, 0.25);

    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(elastic))
        .with_material(SAND_ID, Box::new(sand))
        .with_material(FLUID_ID, Box::new(fluid))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(19.0, 9.0),
        material_id: SAND_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(45.0, 9.0),
        material_id: FLUID_ID,
        precompute_initial_volumes: true,
        // Without this, mass falls back to `config.particle_mass` (1.0),
        // completely decoupled from the material's own rest_density=0.1
        // -- a real, separate gap found 2026-08-08 alongside the SI fix
        // (see basic_fluids.rs's doc). m = rho0*spacing^2, same
        // derivation used everywhere else.
        mass_override: Some(0.1 * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(12, 12),
        box_center: Vec2::new(32.0, 46.0),
        material_id: ELASTIC_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    });
    solver
}

impl State {
    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_frac[0] * GRID as f32,
            (1.0 - self.cursor_frac[1]) * GRID as f32,
        )
    }

    fn set_arrow(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::ArrowUp => self.arrow_up = pressed,
            KeyCode::ArrowDown => self.arrow_down = pressed,
            KeyCode::ArrowLeft => self.arrow_left = pressed,
            KeyCode::ArrowRight => self.arrow_right = pressed,
            _ => {}
        }
    }
}

impl DemoApp for State {
    const TITLE: &'static str = "emerge -- Showcase [Sand / Fluid / Elastic]";

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let sim = make_sim();
        let mut renderer = Renderer::new(device, sim.particles().len(), format);
        renderer.set_camera(queue, GRID as u32, width, height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "showcase: {} particles  |  ^v<> drive blob  LMB push  RMB pull  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            sim,
            renderer,
            cursor_frac: [0.0; 2],
            lmb: false,
            rmb: false,
            arrow_up: false,
            arrow_down: false,
            arrow_left: false,
            arrow_right: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
        }
    }

    fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.renderer
            .set_camera(queue, GRID as u32, width, height, 0.6, true);
    }

    fn update_and_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        // Arrow-key drive: find elastic centroid, apply impulse.
        let mut dir = Vec2::ZERO;
        if self.arrow_up {
            dir.y += 1.0;
        }
        if self.arrow_down {
            dir.y -= 1.0;
        }
        if self.arrow_left {
            dir.x -= 1.0;
        }
        if self.arrow_right {
            dir.x += 1.0;
        }
        if dir != Vec2::ZERO {
            let particles = self.sim.particles();
            let (sum, n) = particles
                .indices()
                .filter(|&i| particles.material_id[i] == ELASTIC_ID)
                .fold((Vec2::ZERO, 0usize), |(s, n), i| {
                    (s + particles.x[i], n + 1)
                });
            if n > 0 {
                let centroid = sum / n as f32;
                let impulse = dir.normalize() * 10.0;
                self.sim.apply_impulse(centroid, 12.0, impulse);
            }
        }

        if self.lmb || self.rmb {
            let mag = if self.lmb { 2.0 } else { -2.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!("frame={} fps={:.0}", self.frame, fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        self.renderer
            .render(device, queue, self.sim.particles(), view, true);
    }

    fn cursor_moved(&mut self, x_frac: f32, y_frac: f32) {
        self.cursor_frac = [x_frac, y_frac];
    }

    fn mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.lmb = pressed,
            MouseButton::Right => self.rmb = pressed,
            _ => {}
        }
    }

    fn key_pressed(&mut self, key: KeyCode) {
        if key == KeyCode::KeyR {
            self.sim = make_sim();
            self.frame = 0;
            println!("reset");
        }
        self.set_arrow(key, true);
    }

    fn key_released(&mut self, key: KeyCode) {
        self.set_arrow(key, false);
    }
}

fn main() {
    run_demo::<State>();
}
