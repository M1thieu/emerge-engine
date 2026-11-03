//! The precise path (`SimConfig::total_lagrangian`) live in a window, in the
//! one configuration that passes every contact-law test at two cells through
//! the block (FAST_MODE_PLAN.md secs. 52.4, 53): Acary's forecast activation
//! with the error controller throughout, tolerance 1e-2, the sub-frame
//! handoff with modal flight. An 8 x 4 cm block of Harbutt's Plasticine
//! (Crandall et al. 1970, fresh: relaxed E 20.2 MPa, nu 0.434, 1906 kg/m3,
//! three measured Zener mechanisms) dropped 10 cm on the floor, one
//! `Simulation::step` per 1/60 s frame.
//!
//! Measured on this configuration (`rt_conv/runs/forecast_ctrl_1e-2`, 420
//! frames): the first touch is resolved (its deepest contact point 0.06 mm
//! below the floor), the block rests on the floor (within 0.001 mm), the
//! energy ledger closes (1.1e-8 J/m), the rebound converges in the tolerance.
//! Its declared limits: the rebound is the two-cell mesh's (0.058, against
//! the mesh-converged 0.046-0.049); the first-touch frame takes about 9 s of
//! compute; the rest of the run costs 14-16 ms per frame.
//!
//! The physics runs on its own thread, so the window keeps drawing while a
//! step is computed: the first-touch frame is shown as it happens, the panel
//! counting its compute time live. Nothing is precomputed or replayed; the
//! simulated time advances one frame per finished step and does not catch
//! up after a late frame.
//!
//! What is drawn: each particle's reference cell pushed by its deformation
//! gradient, the floor as a line, one neutral tint (no colour encodes a
//! quantity; the clay has no measured optical record).
//!
//!   cargo run --release --example tl_window_lawful --features render
//!   cargo run --release --example tl_window_lawful --features render -- --smoke 60
//!
//! `--smoke N` runs N frames without a window and prints each frame's
//! numbers.

extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use emerge::materials::relaxation::ViscoelasticRecord;
use emerge::render::{ColorMode, Renderer};
use emerge::spacetime::solver::total_lagrangian_solids::{
    TotalLagrangianConfig, TotalLagrangianFrame,
};
use emerge::{CorotatedMaterial, FromSI, Particle, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Cells through the block's thickness.
const ACROSS: i32 = 2;
/// The block's thickness, m, and the drop, m.
const THICKNESS_M: f32 = 0.04;
const DROP_M: f32 = 0.10;
const FRAME_S: f32 = 1.0 / 60.0;
/// The step tolerance (sec. 52.4: converged between 1e-2 and 1e-3).
const TOLERANCE: f64 = 1e-2;
/// The renderer's grey palette entry: the only neutral one.
const NEUTRAL_ID: u32 = 14;

/// The scene: the block, its bottom `DROP_M` above the floor.
fn scene() -> Simulation {
    let cell_m = THICKNESS_M / ACROSS as f32;
    let mut config = SimConfig::earth(10 * ACROSS as usize, cell_m, FRAME_S);
    let mut path = TotalLagrangianConfig::new(TOLERANCE);
    path.settings.forecast_activation = true;
    config.total_lagrangian = Some(path.with_subframe_handoff());
    let mut sim = Simulation::empty(config);
    let record = ViscoelasticRecord::plasticine_crandall_1970_fresh();
    let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
    material.relaxation = Some(record.spectrum);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let lift = DROP_M / cell_m;
    let _ = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(2 * ACROSS, ACROSS))
            .at(Vec2::new(
                5.0 * ACROSS as f32,
                floor + lift + ACROSS as f32 / 2.0,
            ))
            .spacing(0.5)
            .material(id)
            .mass_from(&record.relaxed, &config),
    );
    sim
}

/// The mass-weighted centroid height above the floor, m.
fn centroid_height_m(particles: &[Particle], config: &SimConfig) -> f64 {
    let (mut mass, mut y) = (0.0f64, 0.0f64);
    for p in particles {
        mass += f64::from(p.mass);
        y += f64::from(p.mass) * f64::from(p.x.y);
    }
    (y / mass - config.boundary_thickness as f64) * f64::from(config.dx_meters)
}

/// One finished frame, as the physics thread hands it to the window.
struct Snapshot {
    particles: Vec<Particle>,
    frame: Option<TotalLagrangianFrame>,
    index: u64,
    wall_ms: f64,
}

/// What the window asks of the physics thread.
enum Command {
    Pause(bool),
    Reset,
}

/// The physics thread: one step per 1/60 s deadline (no catch-up), each
/// finished frame sent; `busy` holds the running step's start.
fn physics(out: Sender<Snapshot>, commands: Receiver<Command>, busy: Arc<Mutex<Option<Instant>>>) {
    let mut sim = scene();
    let (mut index, mut paused, mut deadline) = (0u64, false, Instant::now());
    let budget = Duration::from_secs_f64(f64::from(FRAME_S));
    loop {
        for command in commands.try_iter() {
            match command {
                Command::Pause(p) => paused = p,
                Command::Reset => {
                    sim = scene();
                    index = 0;
                    deadline = Instant::now();
                }
            }
        }
        let now = Instant::now();
        if paused || now < deadline {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        *busy.lock().expect("the busy lock") = Some(now);
        sim.step();
        let wall_ms = 1e3 * now.elapsed().as_secs_f64();
        *busy.lock().expect("the busy lock") = None;
        deadline = (deadline + budget).max(Instant::now());
        index += 1;
        let snapshot = Snapshot {
            particles: sim.particles().iter().collect(),
            frame: sim.total_lagrangian_frame(),
            index,
            wall_ms,
        };
        if out.send(snapshot).is_err() {
            return;
        }
    }
}

/// `--smoke N`: N frames without a window.
fn smoke(frames: usize) {
    let mut sim = scene();
    let config = *sim.config();
    println!(
        "tl_window_lawful smoke | forecast activation, controller throughout, tolerance {TOLERANCE:e} | {} particles | {} threads",
        sim.particles().len(),
        rayon::current_num_threads()
    );
    for n in 0..frames {
        let clock = Instant::now();
        sim.step();
        let wall = 1e3 * clock.elapsed().as_secs_f64();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        let particles: Vec<Particle> = sim.particles().iter().collect();
        println!(
            "frame {n} | step {wall:.2} ms | kept steps {} rejected {} modal {} contacts {} | balance {:+.3e} J/m | height {:.9} m",
            f.steps,
            f.rejected,
            f.modal,
            f.contacts,
            f.balance,
            centroid_height_m(&particles, &config)
        );
    }
}

struct State {
    gfx: gui_common::Gfx,
    renderer: Renderer,
    config: SimConfig,
    snapshots: Receiver<Snapshot>,
    commands: Sender<Command>,
    busy: Arc<Mutex<Option<Instant>>>,
    last: Option<Snapshot>,
    /// Finished frames' wall times, ms: all of them, and the rest phase's
    /// (frames after the 100th).
    walls: Vec<f64>,
    paused: bool,
    /// The cursor, window pixels.
    cursor_pos: [f32; 2],
}

fn grid_cells() -> u32 {
    10 * ACROSS as u32
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let probe = scene();
        let config = *probe.config();
        let mut renderer = Renderer::new(&gfx.device, probe.particles().len(), gfx.format);
        renderer.set_camera(
            &gfx.queue,
            grid_cells(),
            size.width,
            size.height,
            0.5,
            false,
        );
        renderer.set_color_mode(ColorMode::ByMaterial);
        let (snapshot_tx, snapshots) = channel();
        let (commands, command_rx) = channel();
        let busy = Arc::new(Mutex::new(None));
        let shared = busy.clone();
        std::thread::spawn(move || physics(snapshot_tx, command_rx, shared));
        Self {
            gfx,
            renderer,
            config,
            snapshots,
            commands,
            busy,
            last: Some(Snapshot {
                particles: probe.particles().iter().collect(),
                frame: None,
                index: 0,
                wall_ms: 0.0,
            }),
            walls: Vec::new(),
            paused: false,
            cursor_pos: [0.0; 2],
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w > 0 && h > 0 {
            self.renderer
                .set_camera(&self.gfx.queue, grid_cells(), w, h, 0.5, false);
        }
    }

    fn reset(&mut self) {
        let _ = self.commands.send(Command::Reset);
        self.walls.clear();
    }

    fn update_and_render(&mut self, window: &Window) {
        for snapshot in self.snapshots.try_iter() {
            self.walls.push(snapshot.wall_ms);
            self.last = Some(snapshot);
        }
        let output = match self.gfx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let Some(last) = &self.last else {
            return;
        };
        let drawn: Vec<Particle> = last
            .particles
            .iter()
            .map(|p| Particle {
                material_id: NEUTRAL_ID,
                ..*p
            })
            .collect();
        self.renderer
            .render_slice(&self.gfx.device, &self.gfx.queue, &drawn, &view, true);
        let (w, h) = (
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
        );
        let ppp = self.gfx.egui_ctx.pixels_per_point();
        let floor = self.config.boundary_thickness as f32;
        let (x0, y0) = self.renderer.grid_to_screen_points(0.0, floor, w, h, ppp);
        let (x1, y1) = self
            .renderer
            .grid_to_screen_points(grid_cells() as f32, floor, w, h, ppp);
        let running =
            (*self.busy.lock().expect("the busy lock")).map(|t| t.elapsed().as_secs_f64());
        let height_mm = 1e3 * centroid_height_m(&last.particles, &self.config);
        let (index, wall_ms, frame) = (last.index, last.wall_ms, last.frame);
        let over = self
            .walls
            .iter()
            .filter(|&&w| w > 1e3 * f64::from(FRAME_S))
            .count();
        let worst = self.walls.iter().copied().fold(0.0f64, f64::max);
        let rest = if self.walls.len() > 100 {
            Some(median(&self.walls[100..]))
        } else {
            None
        };
        let mut paused = self.paused;
        let mut reset = false;
        // The cursor in cm from the floor line, through the renderer's own
        // projection.
        let cursor = gui_common::cursor_to_grid(self.cursor_pos, w, h, grid_cells() as usize);
        let cursor_cm = (
            100.0 * cursor.x * self.config.dx_meters,
            100.0 * (cursor.y - floor) * self.config.dx_meters,
        );
        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            ctx.layer_painter(egui::LayerId::background()).line_segment(
                [egui::pos2(x0, y0), egui::pos2(x1, y1)],
                egui::Stroke::new(2.0_f32, egui::Color32::from_gray(200)),
            );
            egui::Window::new("Precise path: the lawful contact, live")
                .default_pos([10.0, 10.0])
                .default_width(400.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("block 8 x 4 cm of Plasticine (Crandall 1970, fresh), dropped 10 cm");
                    ui.label("contact: Acary's forecast activation, error controller throughout, tolerance 1e-2");
                    ui.label("measured: first touch resolved (0.06 mm), rests on the floor, ledger closed, converged in the tolerance");
                    ui.label("declared limits: 2 cm cells (the rebound is that mesh's, 0.058, against 0.046-0.049 converged); the first-touch frame takes about 9 s of compute; no catch-up after a late frame");
                    ui.separator();
                    match running {
                        Some(s) if s > f64::from(FRAME_S) => {
                            ui.colored_label(
                                egui::Color32::LIGHT_YELLOW,
                                format!("impact step in progress: {s:.1} s of compute so far (the simulation waits for it; shown live, nothing precomputed)"),
                            );
                        }
                        _ => {
                            ui.label("stepping in real time");
                        }
                    }
                    ui.label(format!("simulated time {:.3} s (frame {index})", index as f64 / 60.0));
                    ui.label(format!(
                        "last frame's compute {wall_ms:.1} ms (budget 16.7) | worst {worst:.0} ms | frames over budget {over}"
                    ));
                    match rest {
                        Some(m) => ui.label(format!("rest phase (frames after 100): median {m:.1} ms per frame")),
                        None => ui.label("rest phase median: after frame 100"),
                    };
                    if let Some(f) = frame {
                        ui.label(format!(
                            "kept steps {} | refused tries {} | flown by modes: {} | contacts {}",
                            f.steps, f.rejected, f.modal, f.contacts
                        ));
                        ui.label(format!("energy ledger balance {:+.2e} J/m", f.balance));
                    }
                    ui.label(format!("centroid height above the floor {height_mm:.3} mm (rest 20.000)"));
                    ui.separator();
                    ui.label("tint: neutral, no colour encodes a quantity");
                    if ui.button(if paused { "Resume" } else { "Pause" }).clicked() {
                        paused = !paused;
                    }
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                    ui.label(format!(
                        "cursor: x {:.1} cm from the domain's left, {:.1} cm above the floor",
                        cursor_cm.0, cursor_cm.1
                    ));
                    ui.label("Space pause  R reset  Q quit");
                });
        });
        if paused != self.paused {
            self.paused = paused;
            let _ = self.commands.send(Command::Pause(paused));
        }
        if reset {
            self.reset();
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
                    .with_title("emerge -- precise path, the lawful contact")
                    .with_inner_size(winit::dpi::LogicalSize::new(800u32, 800u32)),
            )
            .expect("a window"),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        if let Some(w) = &self.window
            && s.gfx.egui_state.on_window_event(w, &event).consumed
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: key_state,
                        ..
                    },
                ..
            } if key_state.is_pressed() => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => s.reset(),
                KeyCode::Space => {
                    s.paused = !s.paused;
                    let _ = s.commands.send(Command::Pause(s.paused));
                }
                _ => {}
            },
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
    let mut args = std::env::args().skip(1);
    let mut smoke_frames = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--smoke" => {
                smoke_frames = Some(
                    args.next()
                        .and_then(|v| v.parse().ok())
                        .filter(|&n: &usize| n > 0)
                        .unwrap_or_else(|| {
                            eprintln!("--smoke needs a positive frame count");
                            std::process::exit(2);
                        }),
                );
            }
            other => {
                eprintln!("unknown argument {other}; usage: tl_window_lawful [--smoke N]");
                std::process::exit(2);
            }
        }
    }
    if let Some(frames) = smoke_frames {
        smoke(frames);
        return;
    }
    let el = EventLoop::new().expect("an event loop");
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    el.run_app(&mut app).expect("the event loop");
}
