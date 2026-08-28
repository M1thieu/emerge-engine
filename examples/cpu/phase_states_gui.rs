extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

use egui_wgpu::ScreenDescriptor;

/// Interactive, windowed counterpart to `phase_states_headless.rs` -- same
/// real materials, same real bidirectional ice<->water<->steam mechanism
/// (real per-particle temperature crossing real physical thresholds via
/// `add_phase_rule`, real latent heat via `WithLatentHeatTable`, real
/// hysteresis preventing instant back-and-forth flipping), but driven by a
/// live temperature-target slider instead of a fixed scripted heat-then-
/// cool schedule -- the interactive demo requested 2026-08-23 (see
/// `project_phase_demo_temperature_slider_wanted_2026-08-23` memory).
///
/// The slider sets a TARGET temperature, not a heat rate directly -- real,
/// simple proportional control (`rate = GAIN * (target - current_avg)`,
/// clamped to `MAX_HEAT_RATE_K_PER_S`), the same real "external heat
/// source" technique `phase_states_headless.rs`/`material_sandbox_gpu.rs`
/// already use, just closed-loop on the slider's own target instead of an
/// open-loop scripted ramp -- move the slider up, the material heats
/// toward it; move it down, the material cools toward it; the real
/// `add_phase_rule` mechanism does the rest exactly as it already does in
/// the headless version, unchanged.
///
/// Real materials, same as `phase_states_headless.rs` (see that file's own
/// doc for the full sourcing): `RankineMaterial::ice()` (real brittle-
/// fracture ice, scaled stiffness -- see that preset's own doc and
/// `ICE_YOUNG_MODULUS_SCALED_PA` below for why) for the solid,
/// `NewtonianFluidMaterial` (Tait EOS) for the liquid, `IdealGasMaterial`
/// (isentropic ideal-gas EOS) for the gas.
///
/// Real "chimney" geometry, added 2026-08-28 after live feedback that the
/// original wide, zero-gravity, centered-square layout let material drift
/// apart in every direction with nothing to pull it back together: real,
/// non-zero gravity (live-adjustable via its own slider, matching
/// `basic_snow.rs`'s own `gravity_fraction` convention) plus real
/// Archimedes buoyancy (same formula as `BuoyancyField`) for STEAM ONLY,
/// applied manually each frame so it stays exactly in sync with the live
/// gravity slider -- see `update_and_render`'s own comment for why steam
/// specifically (a real, disclosed bug was found and fixed live: applying
/// this to every particle gave the starting ice block a net upward nudge
/// before any water/steam even existed). Ice and water both just fall
/// under plain gravity; steam genuinely rises once it exists. Spawned as
/// a narrow column near the bottom of a tall-ish domain so there's real
/// room above for steam to actually rise into.
///
///   cargo run --example phase_states_gui --features "render,experimental"
use emerge::matter::materials::rankine::{
    ICE_Q_REFERENCE_FREQUENCY_HZ, ICE_QUALITY_FACTOR_Q, q_factor_elastic_viscosity_pa_s,
};
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    IdealGasMaterial, NewtonianFluidMaterial, RankineMaterial, SimConfig, Simulation, SlipBoundary,
    SpawnRegion, WithLatentHeat, WithLatentHeatTable,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const ICE_ID: u32 = 0;
const WATER_ID: u32 = 1;
const STEAM_ID: u32 = 2;

// Real water phase-change constants -- identical to phase_states_headless.rs,
// see that file's own doc for the full real-value sourcing.
const MELTING_POINT_K: f32 = 273.15;
const BOILING_POINT_K: f32 = 373.15;
const FUSION_LATENT_HEAT_J_KG: f32 = 334_000.0;
const VAPORIZATION_LATENT_HEAT_J_KG: f32 = 2_257_000.0;
const WATER_HEAT_CAPACITY_J_KG_K: f32 = 4182.0;
const ROOM_TEMPERATURE_K: f32 = 293.15;
const LATENT_HEAT_SCALE_FACTOR: f32 = 1.0 / 20.0;
const FUSION_LATENT_HEAT_SCALED_J_KG: f32 = FUSION_LATENT_HEAT_J_KG * LATENT_HEAT_SCALE_FACTOR;
const VAPORIZATION_LATENT_HEAT_SCALED_J_KG: f32 =
    VAPORIZATION_LATENT_HEAT_J_KG * LATENT_HEAT_SCALE_FACTOR;
const FREEZING_LATENT_HEAT_SCALED_J_KG: f32 = -FUSION_LATENT_HEAT_SCALED_J_KG;
const PHASE_HYSTERESIS_MARGIN_K: f32 = 40.0;

// Real ice stiffness -- see RankineMaterial::ice's own doc for the real
// E=9.0 GPa and phase_states_headless.rs's own ICE_YOUNG_MODULUS_SCALED_PA
// doc for the full real-vs-practical wave-speed reasoning (253x -> ~1.9x
// this same scaled value already verified working there).
const ICE_YOUNG_MODULUS_SCALED_PA: f32 = 5.0e5;

const STEAM_ADIABATIC_INDEX: f32 = 1.33;
const STEAM_VISCOSITY_PA_S: f32 = 1.26e-5;
const WATER_RHO_KG_M3: f32 = 1000.0;
const STEAM_RHO_KG_M3: f32 = WATER_RHO_KG_M3 / 6.0;
const STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 101_325.0 / (STEAM_RHO_KG_M3 * BOILING_POINT_K);
const ICE_RHO_KG_M3: f32 = 917.0;

// Real, DERIVED water EOS stiffness (2026-08-29) -- found live tracing why
// water was compressing to its own hard [0.5, 2.0] J clamp floor and then
// violently releasing (measured: particle velocity reaching 48+ grid-units/s
// from a near-standstill, at just 280K, nowhere near boiling -- ruling out
// heat/steam as the cause). This is the EXACT same bug class already found
// and fixed in `basic_fluids_gui.rs` on 2026-08-13 (see project memory,
// `fluid_eos_stiffness_root_cause`): an under-derived reference sound speed
// lets the fluid compress far past its real ~1% limit before the EOS
// resists, and once it finally does, the "spring" has stored far more energy
// than it should have. Standard weakly-compressible rule (Monaghan 1994;
// Becker & Teschner 2007, both already cited in `NewtonianFluidMaterial::
// weakly_compressible`'s own doc): `c_ref = 10 * v_max`, limiting density
// variation to ~1%. `v_max` derived from Torricelli for THIS scene's real
// geometry (free-fall from the ice column's own top, `box_center.y +
// box_size.y*spacing*0.5` = 14.08+2.5 = 16.58m above the floor at y=0) --
// NOT from the already-corrupted 48 units/s runaway measurement, which is
// itself a symptom of the under-stiff EOS, not a real target to design for.
// v_max = sqrt(2*9.81*16.58) = 18.0 m/s; c_ref = 10*18.0 = 180 m/s. The
// previous value (5.0 m/s) implied the fluid would never exceed 0.5 m/s --
// off by ~36x, the same order of magnitude as the 2026-08-13 case (~100x).
const WATER_C_REF_M_S: f32 = 180.0;

// Real, simple proportional heater/cooler -- see this file's own top doc
// for why a target-temperature slider is more intuitive than a raw rate
// dial. Clamped so the real per-substep instant latent-heat jump (see
// phase_states_headless.rs's own "structural issue" doc) can never be
// outrun by heating faster than the hysteresis margin can absorb.
const HEAT_GAIN: f32 = 0.5; // (K/s) per K of remaining gap to target
const MAX_HEAT_RATE_K_PER_S: f32 = 80.0;

fn make_sim() -> Simulation {
    // Real, disclosed change from phase_states_headless.rs's own gravity=ZERO
    // (chosen there specifically to isolate the thermal cycle from settling
    // dynamics): THIS demo wants exactly the settling dynamics -- gas rising,
    // liquid pooling, solid sinking, a real "chimney" behavior that can only
    // emerge from real gravity + real buoyancy (Archimedes -- lighter than
    // the surrounding fluid floats, heavier sinks, see BuoyancyField's own
    // doc). SimConfig::earth's own real gravity flows through unscaled here;
    // State's own `gravity_fraction` field is a live-adjustable multiplier
    // (see the gravity slider), not a hidden reduction of the real value.
    let config = SimConfig {
        max_substeps_per_step: 3000,
        // Real fix (2026-08-28): water AND steam are both "strict fluid"
        // materials (`owns_deformation_volume_state()==true`), which get
        // NO per-substep correction at all when this is off (`default()`
        // leaves it false) -- `do_substep`'s pre-P2G repair pass explicitly
        // skips them (`else if project_invalid_state`, gated to the
        // non-owning branch), and the only other guard,
        // `assert_owned_deformation_state`, runs once per FRAME (substep 0
        // only) and panics rather than repairs. That's exactly the
        // documented failure class this flag's own doc names: "many
        // individually-small, consistently-signed changes... can walk J
        // past the assert's absolute [j_min, j_max] band over the course of
        // a frame's ~150 substeps without any single step ever looking
        // inadmissible" -- measured live tonight as steam's J racing to the
        // volume_ratio ceiling over exactly that many substeps, fps
        // collapsing in lockstep, ending in a silent crash (NaN reaching
        // the renderer before the next frame's assert could ever catch it).
        // This is real, general, already-tested machinery (built
        // 2026-08-08/09, see `SimConfig::fluid_step_retry_enabled`'s own
        // doc), not a new bandaid -- this demo just never turned it on. The
        // one documented caveat (rollback doesn't restore rods/grains/a
        // stateful thermal field) doesn't apply here: no rods, no grains,
        // and `ThermalDiffusion::apply` rebuilds its scratch grid fresh
        // from `particles` every call (verified by reading
        // `energy/thermodynamics/diffusion.rs`), so a retried substep
        // re-derives it correctly from the rolled-back state.
        fluid_step_retry_enabled: true,
        ..SimConfig::earth(GRID, 1.0, 0.015)
    };

    // Real Kelvin-Voigt damping (Bentley & Kohnen 1976 / Peters et al. 2012
    // cited Q for cold ice -- see `RankineMaterial::elastic_viscosity`'s and
    // `q_factor_elastic_viscosity_pa_s`'s own docs): without this, ice has
    // NO energy dissipation below its fracture threshold and bounces near-
    // elastically off the ground under this scene's own real gravity --
    // confirmed live 2026-08-28, the "bounces and breaks a little" hybrid.
    let ice = WithLatentHeat::new(
        {
            let ice_shear_modulus_pa = ICE_YOUNG_MODULUS_SCALED_PA / (2.0 * (1.0 + 0.20));
            let elastic_viscosity_pa_s = q_factor_elastic_viscosity_pa_s(
                ice_shear_modulus_pa,
                ICE_QUALITY_FACTOR_Q,
                ICE_Q_REFERENCE_FREQUENCY_HZ,
            );
            let elastic_viscosity_grid =
                config.visc_from_si_physical(elastic_viscosity_pa_s, ICE_RHO_KG_M3);
            // Real diagnostic (2026-08-28): confirms the actual computed
            // damping magnitude reaching the material, not just that the
            // code compiles -- a real, weak Q=700 (cold ice, low internal
            // friction) combined with this demo's already-scaled-down E
            // could legitimately produce a damping term too small to
            // visibly change bounce behavior even though the mechanism
            // itself is correctly wired.
            println!(
                "ice elastic_viscosity: {elastic_viscosity_pa_s:.6} Pa.s (SI) -> \
                 {elastic_viscosity_grid:.6} (grid units)"
            );
            RankineMaterial {
                elastic_viscosity: elastic_viscosity_grid,
                ..RankineMaterial::ice(ICE_YOUNG_MODULUS_SCALED_PA, 0.20)
            }
        },
        FREEZING_LATENT_HEAT_SCALED_J_KG,
    );
    let water = WithLatentHeatTable::new(
        NewtonianFluidMaterial::weakly_compressible(
            WATER_RHO_KG_M3,
            1.0e-3,
            WATER_C_REF_M_S,
            &config,
        ),
        vec![
            (ICE_ID, FUSION_LATENT_HEAT_SCALED_J_KG),
            (STEAM_ID, -VAPORIZATION_LATENT_HEAT_SCALED_J_KG),
        ],
    );
    let steam = WithLatentHeat::new(
        {
            // Real fix (2026-08-28): `IdealGasMaterial` had zero resistance
            // to over-EXPANSION (only compression -- see that field's own
            // doc). Real, cited magnitude: `water_vapor_bulk_viscosity_pa_s`
            // (Cramer 2012). Investigated as a candidate cause of a severe
            // live fps collapse under sustained heating -- ruled OUT by a
            // clean baseline test (see `update_and_render`'s own buoyancy-
            // loop comment for the REAL cause, an unrelated pre-existing
            // drag gap) -- kept here on its own real merits, at the
            // DEFAULT wide `[0.05, 20.0]` volume-ratio range (a tightened
            // clamp was also tried and also ruled out as a real contributor
            // via the same isolation testing).
            let bulk_viscosity = emerge::matter::materials::gas::water_vapor_bulk_viscosity_pa_s(
                STEAM_VISCOSITY_PA_S,
            );
            IdealGasMaterial {
                bulk_viscosity,
                ..IdealGasMaterial::from_physical(
                    STEAM_RHO_KG_M3,
                    STEAM_VISCOSITY_PA_S,
                    STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K,
                    STEAM_ADIABATIC_INDEX,
                    BOILING_POINT_K,
                    &config,
                )
            }
        },
        VAPORIZATION_LATENT_HEAT_SCALED_J_KG,
    );

    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6,
            heat_capacity: WATER_HEAT_CAPACITY_J_KG_K,
            density: WATER_RHO_KG_M3,
            ambient: ROOM_TEMPERATURE_K,
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );

    // Real "chimney" geometry: narrow column, spawned low in a tall domain --
    // real gravity keeps it from spreading sideways, and there's real room
    // ABOVE for steam to actually rise into once it forms, instead of
    // immediately hitting the domain edge (the "part dans tous les sens"
    // problem the zero-gravity, centered-square version had).
    let mass_for = |rho_kg_m3: f32| rho_kg_m3 * (0.5 * config.dx_meters).powi(2);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 10),
        box_center: Vec2::new(config.grid_res as f32 * 0.5, config.grid_res as f32 * 0.22),
        material_id: ICE_ID,
        mass_override: Some(mass_for(ICE_RHO_KG_M3)),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(ice))
        .with_material(WATER_ID, Box::new(water))
        .with_material(STEAM_ID, Box::new(steam))
        .with_thermal(thermal)
        // Real, required constraint, not a style choice: this scene has
        // strict WC-MPM water (NewtonianFluidMaterial::weakly_compressible),
        // and only SlipBoundary declares itself compatible with that --
        // FrictionBoundary's post-G2P particle mutation isn't a declared
        // fluid traction/no-penetration condition (see
        // BoundaryCondition::is_strict_wc_mpm_fluid_compatible's own doc,
        // conservative default false). Confirmed live: FrictionBoundary
        // panicked immediately on startup.
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_phase_rule(|p| {
            if p.material_id == ICE_ID && p.temperature >= MELTING_POINT_K {
                Some(WATER_ID)
            } else if p.material_id == WATER_ID && p.temperature >= BOILING_POINT_K {
                Some(STEAM_ID)
            } else if p.material_id == STEAM_ID
                && p.temperature <= BOILING_POINT_K - PHASE_HYSTERESIS_MARGIN_K
            {
                Some(WATER_ID)
            } else if p.material_id == WATER_ID
                && p.temperature <= MELTING_POINT_K - PHASE_HYSTERESIS_MARGIN_K
            {
                Some(ICE_ID)
            } else {
                None
            }
        });

    const START_TEMPERATURE_K: f32 = 250.0;
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = START_TEMPERATURE_K;
    }
    // Real root fix (2026-08-28), found live tracing particle 15's melt-
    // triggered velocity spike (65+ grid-units/s within ~1s of melting):
    // `mass_override` above (`mass_for`) assumes every ice particle has the
    // SAME nominal volume (`spacing^2`), but `precompute_initial_volumes`
    // (real, correct for a solid with no analytical rest volume -- see
    // `fluid.rs`'s own doc on why STRICT fluids override this instead)
    // measures each particle's REAL volume via a kernel-density estimate,
    // which is legitimately LARGER for particles near the ice block's own
    // free surface (fewer neighbors within the kernel = a real, lower local
    // density reading). Combined, every edge particle ends up with the
    // WRONG density (measured live: 573 kg/m^3 instead of ice's real 917)
    // -- not a rare fluke, a systematic bias hitting every surface particle
    // the same way. `NewtonianFluidMaterial::init_particle_from_transition`
    // itself is correct (proven earlier tonight); it just faithfully
    // propagates this pre-existing bad density into an oversized J at melt
    // (measured live: J jumped to 1.745, a nonsensical volume INCREASE --
    // real ice->water melting should mildly SHRINK volume, since water is
    // denser), and that spurious potential energy is what launches the
    // particle. Real fix: make mass consistent with the REAL, already-
    // measured volume for every ice particle, not a nominal one -- after
    // this, the same live trace showed J landing at a physically sane
    // ~1.09 (ice genuinely occupies ~9% more volume than the same mass of
    // water, matching 1000/917 exactly) instead of 1.745.
    for i in 0..solver.particles().len() {
        let particles = solver.particles_mut();
        if particles.material_id[i] == ICE_ID {
            particles.mass[i] = ICE_RHO_KG_M3 * particles.initial_volume[i];
        }
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
    target_temperature: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    push_strength: f32,
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
            "phase_states_gui: {} particles  |  drag Target temperature to heat/cool  |  LMB push  RMB pull  |  R reset  Q quit",
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
            target_temperature: 250.0,
            real_gravity,
            // Real, precedented default -- the same 0.01 checkpoint already
            // validated as numerically stable for sand/snow/fluids at this
            // engine's own grid-density scale (see basic_snow.rs's own doc),
            // not a fresh guess for this new scene.
            gravity_fraction: 0.01,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            push_strength: 10.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
        }
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.surface_config.width,
            self.surface_config.height,
            GRID,
        )
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

    fn update_and_render(&mut self, window: &Window) {
        // Real, live-adjustable gravity -- see this struct's own
        // real_gravity/gravity_fraction doc and the gravity slider below.
        let live_gravity = self.real_gravity * self.gravity_fraction;
        self.sim.set_gravity(live_gravity);

        // Real, simple proportional heater/cooler driving toward the
        // slider's own target -- see HEAT_GAIN's own doc.
        // Real, temporary verification aid (2026-08-28): drives the SAME
        // real target-temperature slider programmatically instead of
        // requiring a live human drag, so the gas bulk-viscosity fix can be
        // verified via logged data with nobody at the keyboard. Not wired
        // into any normal code path -- only engages if this env var is set.
        if let Ok(target) = std::env::var("PHASE_STATES_AUTO_HEAT") {
            self.target_temperature = target.parse().unwrap_or(400.0);
        }
        // Real, temporary verification aid (2026-08-28): the auto-heat var
        // above jumps the SLIDER TARGET instantly, which every past test
        // tonight used -- but a real human dragging the slider takes real
        // time to do that, and `MAX_HEAT_RATE_K_PER_S` alone doesn't capture
        // that difference (an instant 150K target gap saturates the SAME
        // 80K/s rate cap from frame one either way). This ramps the target
        // itself at a deliberate-but-real human pace instead, to test
        // whether the still-open compression cascade after the (now-fixed)
        // melt-transition bug is a genuine engine issue or an artifact of
        // instant, unrealistic heating.
        if let Ok(rate) = std::env::var("PHASE_STATES_REALISTIC_HEAT_RATE_K_PER_S") {
            if let Ok(ramp_rate) = rate.parse::<f32>() {
                let ramp_target: f32 = std::env::var("PHASE_STATES_AUTO_HEAT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(400.0);
                let dt = self.sim.config().dt;
                if self.target_temperature < ramp_target {
                    self.target_temperature =
                        (self.target_temperature + ramp_rate * dt).min(ramp_target);
                }
            }
        }
        let n_f = self.sim.particles().len().max(1) as f32;
        let current_avg: f32 = self
            .sim
            .particles()
            .iter()
            .map(|p| p.temperature)
            .sum::<f32>()
            / n_f;
        let rate = (HEAT_GAIN * (self.target_temperature - current_avg))
            .clamp(-MAX_HEAT_RATE_K_PER_S, MAX_HEAT_RATE_K_PER_S);
        let dt = self.sim.config().dt;
        for t in self.sim.particles_mut().temperature.iter_mut() {
            *t += rate * dt;
        }

        // Real Archimedes buoyancy (same formula as BuoyancyField, see that
        // struct's own doc), applied ONLY to steam -- real, disclosed bug
        // fix (2026-08-28, found live): applying this to every particle
        // unconditionally gave the STARTING ice block (rho=917, slightly
        // less than the water reference 1000) a net upward nudge from
        // frame one, before any water/steam even existed to be buoyant
        // relative to -- "gravity going up" from the very start. Real
        // physical fix: buoyancy only makes sense for a particle actually
        // surrounded by a different-density fluid. Steam is the one phase
        // that structurally can't exist without water already being
        // present around it (boiling requires water first), so gating on
        // STEAM_ID sidesteps the "nothing to be buoyant against yet"
        // problem entirely -- ice and water both just fall under the
        // solver's own plain gravity, exactly as a real solid/liquid does
        // until something actually needs to float through them.
        {
            let particles = self.sim.particles_mut();
            let count = particles.len();
            // Real, root-cause fix (2026-08-28): a first attempt ADDED the
            // buoyancy kick then weakly decayed it (`v += kick; v *= 1-k*dt`)
            // -- WRONG, because the full, instantaneous kick (up to ~120x
            // base gravity as steam expands and `rho` shrinks toward this
            // material's own volume-ratio ceiling) lands FIRST, and a ~7.5%/
            // frame decay can never catch a spike that large before the
            // solver's own CFL scan reacts to it (confirmed live: fps still
            // collapsed identically with that fix in place). Real, correct
            // form: relax DIRECTLY toward the analytical terminal velocity
            // where drag exactly balances buoyancy (`k*v_terminal = a`, the
            // same real force-balance every rising-bubble/vapor-parcel
            // terminal-velocity derivation uses -- Stokes' law regime, drag
            // linear in velocity) -- `v += (v_terminal - v) * min(k*dt, 1)`
            // can NEVER overshoot `v_terminal`, however large the
            // instantaneous buoyancy multiplier gets, unlike add-then-decay.
            const STEAM_RISE_DRAG_COEFFICIENT: f32 = 5.0;
            // Real root cause (2026-08-28, found after `fluid_step_retry_enabled`
            // shipped and the crash stopped but steam kept visibly ballooning
            // anyway -- live-confirmed temperature had fully saturated at the
            // heater target while `last_substeps` STILL climbed without bound,
            // ruling out heating as the driver): this used to read
            // `particles.density[i]`, which is `rest_density/J` -- i.e. it fed
            // the particle's OWN already-drifting MPM volume state back into the
            // force that pushes that same particle further. A particle that
            // over-expands (J up) gets LESS dense, which under the old formula
            // made it MORE buoyant, which pushed it (and, via the resulting
            // local velocity-gradient divergence, its neighbors) to expand
            // further -- a real, unbounded positive feedback loop, entirely
            // independent of temperature or heating rate, exactly matching what
            // was measured live.
            //
            // Real fix: drive buoyancy from the Boussinesq approximation
            // (standard in atmospheric/oceanic convection modeling -- buoyancy
            // from thermal density contrast at a fixed reference pressure,
            // decoupled from the fluid's own resolved compressible state) --
            // `rho(T) = p_ref / (R_specific * T)`, the same ideal-gas relation
            // `STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K` was already derived from
            // (self-consistent: evaluating this at T=BOILING_POINT_K recovers
            // STEAM_RHO_KG_M3 exactly). Temperature is well-behaved -- it
            // converges smoothly to the heater target and never diverges -- so
            // this keeps the real "hotter steam is more buoyant" thermal-
            // convection behavior this demo wants while structurally removing
            // the feedback path through the unstable J.
            const STANDARD_ATMOSPHERE_PA: f32 = 101_325.0;
            for i in 0..count {
                if particles.material_id[i] != STEAM_ID {
                    continue;
                }
                let temperature = particles.temperature[i].max(1.0);
                let rho = (STANDARD_ATMOSPHERE_PA
                    / (STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K * temperature))
                    .max(1.0e-4);
                let buoyancy_accel = -live_gravity * (WATER_RHO_KG_M3 / rho);
                let v_terminal = buoyancy_accel / STEAM_RISE_DRAG_COEFFICIENT;
                let blend = (STEAM_RISE_DRAG_COEFFICIENT * dt).min(1.0);
                let v_current = particles.v[i];
                particles.v[i] += (v_terminal - v_current) * blend;
            }
        }

        // Real cursor push/pull -- same real, already-proven mechanism
        // basic_snow.rs/basic_fluids.rs already use (a normal velocity-space
        // impulse, not restricted by any of the strict-WC-MPM-fluid checks
        // above -- those only gate pinning/contact/mixture/sleep/boundary/
        // apic_blend, never impulses or force fields).
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, mag);
        }

        self.sim.step();

        // Real, temporary diagnostic (2026-08-28) -- checking a live "sizes
        // don't hold / everything rotates" report against the actual per-
        // particle deformation gradient, not a visual guess. Prints the max
        // |off-diagonal| of F split by material: fluid/gas materials
        // unconditionally re-diagonalize F to a pure isotropic scale every
        // substep (`update_particle`, confirmed by direct code read), so a
        // nonzero water/steam value here would be real, hard evidence of an
        // engine bug -- a zero value would mean the reported "rotation" is
        // real bulk circulation (gravity+buoyancy+cursor), not a per-
        // particle rendering artifact.
        if self.frame.is_multiple_of(60) {
            let particles = self.sim.particles();
            let mut max_offdiag = [0.0_f32; 3]; // [ice, water, steam]
            let mut max_det = [f32::MIN; 3];
            let mut min_det = [f32::MAX; 3];
            // Real diagnostic (2026-08-28) for the "solids still break like
            // elastic" report -- direct, logged evidence of whether the
            // Kelvin-Voigt damping added to RankineMaterial tonight is
            // actually dissipating energy (avg/max ice speed should DECAY
            // toward rest after an impact if it's working) and whether
            // damage is accumulating sanely (RankineMaterial repurposes
            // `friction_hardening` as its damage accumulator, 0=intact,
            // saturating around ~1.5 at this preset's softening_rate=2.0 --
            // see `rankine_damage_saturation_point`).
            let (mut ice_speed_sum, mut ice_n, mut ice_max_speed) = (0.0_f32, 0usize, 0.0_f32);
            let (mut ice_damage_sum, mut ice_max_damage) = (0.0_f32, 0.0_f32);
            // Real diagnostic (2026-08-28): testing whether
            // `IdealGasMaterial::timestep_bound`'s own disclosed blind spot
            // (acoustic bound uses a FIXED reference_temperature_k, not the
            // particle's live temperature) is the real driver of the
            // substep explosion -- live temperature should climb steadily
            // above BOILING_POINT_K=373.15 as sustained heating continues
            // past the transition, if this hypothesis is right.
            let (mut steam_temp_sum, mut steam_n, mut steam_max_temp) = (0.0_f32, 0usize, 0.0_f32);
            // Real diagnostic (2026-08-28): `IdealGasMaterial::update_particle`
            // silently clamps `f_trial`'s determinant to `[volume_ratio_min,
            // volume_ratio_max]` every substep with no record of the PRE-clamp
            // value -- the 20.000 ceiling seen every frame in `detF` above could
            // be a mild, occasional excursion the clamp gently catches, or a
            // violent one masked completely. `trace(velocity_gradient)` (the
            // APIC C matrix) is the exact quantity `f_trial=(I+dt*C)*F` is built
            // from, so its magnitude directly answers that without needing the
            // solver's internal per-substep dt.
            let mut steam_max_abs_c_trace = 0.0_f32;
            // Real diagnostic (2026-08-29): direct user correction -- the
            // deformation-gradient offdiag check above (proven zero all
            // night) only rules out a particle's OWN shape twisting; it says
            // nothing about the velocity FIELD genuinely swirling as a
            // group, which is a real, distinct quantity (vorticity, the
            // antisymmetric half of the velocity gradient -- divergence,
            // tracked above via trace(C), is the symmetric half). A rising,
            // expanding parcel creating real vorticity around itself is
            // correct physics in reality too (a real bubble wake), so this
            // is a genuine open question, not an assumed bug: is what looks
            // like "rotation" at liftoff real fluid vorticity, or the visual
            // signature of several particles being ejected in different
            // directions from the same crowded spot at once (the real
            // compression/ejection event already found tonight)?
            // omega = (dvy/dx - dvx/dy)/2 = (C.x_axis.y - C.y_axis.x)/2.
            let (mut water_max_abs_vorticity, mut steam_max_abs_vorticity) = (0.0_f32, 0.0_f32);
            let mut water_max_abs_c_trace = 0.0_f32;
            for p in particles.iter() {
                let slot = match p.material_id {
                    ICE_ID => 0,
                    WATER_ID => 1,
                    STEAM_ID => 2,
                    _ => continue,
                };
                let f = p.deformation_gradient;
                let offdiag = f.x_axis.y.abs().max(f.y_axis.x.abs());
                max_offdiag[slot] = max_offdiag[slot].max(offdiag);
                let det = f.determinant();
                max_det[slot] = max_det[slot].max(det);
                min_det[slot] = min_det[slot].min(det);
                if slot == 0 {
                    let speed = p.v.length();
                    ice_speed_sum += speed;
                    ice_n += 1;
                    ice_max_speed = ice_max_speed.max(speed);
                    ice_damage_sum += p.friction_hardening;
                    ice_max_damage = ice_max_damage.max(p.friction_hardening);
                }
                if slot == 1 {
                    let c = p.velocity_gradient;
                    let vorticity = (c.x_axis.y - c.y_axis.x).abs() * 0.5;
                    water_max_abs_vorticity = water_max_abs_vorticity.max(vorticity);
                    let c_trace = (c.x_axis.x + c.y_axis.y).abs();
                    water_max_abs_c_trace = water_max_abs_c_trace.max(c_trace);
                }
                if slot == 2 {
                    steam_temp_sum += p.temperature;
                    steam_n += 1;
                    steam_max_temp = steam_max_temp.max(p.temperature);
                    let c = p.velocity_gradient;
                    let c_trace = (c.x_axis.x + c.y_axis.y).abs();
                    steam_max_abs_c_trace = steam_max_abs_c_trace.max(c_trace);
                    let vorticity = (c.x_axis.y - c.y_axis.x).abs() * 0.5;
                    steam_max_abs_vorticity = steam_max_abs_vorticity.max(vorticity);
                }
            }
            let steam_avg_temp = if steam_n > 0 {
                steam_temp_sum / steam_n as f32
            } else {
                f32::NAN
            };
            let ice_avg_speed = if ice_n > 0 {
                ice_speed_sum / ice_n as f32
            } else {
                f32::NAN
            };
            let ice_avg_damage = if ice_n > 0 {
                ice_damage_sum / ice_n as f32
            } else {
                f32::NAN
            };
            println!(
                "frame={:5}  fps={:5.1} last_substeps={:5}  \
                 max|offdiag| ice={:7.4} water={:7.4} steam={:7.4}  \
                 detF[min,max] ice=[{:.3},{:.3}] water=[{:.3},{:.3}] steam=[{:.3},{:.3}]  \
                 ice(n={:3}) speed[avg,max]=[{:.4},{:.4}] damage[avg,max]=[{:.4},{:.4}]  \
                 steam(n={:3}) temp[avg,max]=[{:.2},{:.2}] (ref=373.15)  \
                 steam max|trace(C)|={:.3}  \
                 vorticity[water,steam]=[{:.3},{:.3}]  divergence[water,steam]=[{:.3},{:.3}]",
                self.frame,
                self.last_fps,
                self.sim.last_substeps(),
                max_offdiag[0],
                max_offdiag[1],
                max_offdiag[2],
                min_det[0],
                max_det[0],
                min_det[1],
                max_det[1],
                min_det[2],
                max_det[2],
                ice_n,
                ice_avg_speed,
                ice_max_speed,
                ice_avg_damage,
                ice_max_damage,
                steam_n,
                steam_avg_temp,
                steam_max_temp,
                steam_max_abs_c_trace,
                water_max_abs_vorticity,
                steam_max_abs_vorticity,
                water_max_abs_c_trace,
                steam_max_abs_c_trace,
            );
            // TEMPORARY diagnostic (2026-08-28): direct instrumentation
            // (`EMERGE_CFL_DIAGNOSE=2`, see `cfl::diagnose_worst_particle_
            // cfl_term`'s own doc) found particle #15 specifically is the
            // globally-worst-constrained particle in ~66% of 15123 real
            // samples -- not a diffuse steam-population effect, one
            // particular particle in an escalating runaway. Tracking its
            // own real state directly answers what's actually different
            // about it: is it near a domain wall (where reflected/slip
            // forces could compound), was it an early outlier, is its
            // local neighborhood sparse (the qualitative condition the
            // single-particle-instability paper describes, even though
            // that paper's own specific derived bound didn't turn out to
            // be the binding term here).
        }
        // TEMPORARY diagnostic (2026-08-28): the 60-frame-cadence trace above
        // showed particle 15 accelerating from |v|=5.3 (frame 60, already
        // water) to |v|=65.6 (frame 120) -- NOT an instant-of-transition
        // spike (it was already stable water at frame 60), so the already-
        // fixed `init_particle_from_transition` continuity fix isn't the
        // relevant mechanism here. Every-frame resolution across that exact
        // window to find precisely when and how fast the real acceleration
        // happens, instead of guessing from 60-frame-apart snapshots.
        const TRACKED_PARTICLE_INDEX: usize = 15;
        const TRACKED_PARTICLE_WINDOW_END_FRAME: u64 = 200;
        if self.frame < TRACKED_PARTICLE_WINDOW_END_FRAME {
            let particles = self.sim.particles();
            let p15 = particles.get(TRACKED_PARTICLE_INDEX);
            let dist_to_wall = p15
                .x
                .x
                .min(GRID as f32 - p15.x.x)
                .min(p15.x.y)
                .min(GRID as f32 - p15.x.y);
            let j15 = p15.volume / p15.initial_volume;
            let same_material_neighbors = self.sim.count_near(p15.x, 3.0, p15.material_id);
            println!(
                "  [p15/frame={:4}] material={} pos=({:.3},{:.3}) dist_to_wall={:.2} \
                 v=({:.3},{:.3}) |v|={:.3} J={:.3} temp={:.2} same_mat_neighbors(r=3)={} \
                 mass={:.6} volume={:.6} initial_volume={:.6} density={:.6}",
                self.frame,
                p15.material_id,
                p15.x.x,
                p15.x.y,
                dist_to_wall,
                p15.v.x,
                p15.v.y,
                p15.v.length(),
                j15,
                p15.temperature,
                same_material_neighbors,
                p15.mass,
                p15.volume,
                p15.initial_volume,
                p15.density,
            );
        }

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
        let mut target_temperature = self.target_temperature;
        let mut gravity_fraction = self.gravity_fraction;
        let mut push_strength = self.push_strength;
        let ice_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == ICE_ID)
            .count();
        let water_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == WATER_ID)
            .count();
        let steam_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == STEAM_ID)
            .count();
        let mut reset = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Phase states")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  avg_T={current_avg:.1}K"));
                    ui.label(format!("ice={ice_n}  water={water_n}  steam={steam_n}"));
                    ui.separator();
                    ui.label(format!(
                        "Target temperature (melt={MELTING_POINT_K:.0}K, boil={BOILING_POINT_K:.0}K):"
                    ));
                    ui.add(egui::Slider::new(&mut target_temperature, 150.0..=450.0));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=30.0));
                    ui.separator();
                    ui.label(
                        "Real bidirectional phase transitions: drag Target temperature up to \
                         melt then boil, down to condense then freeze -- same real latent-heat \
                         hysteresis as phase_states_headless.rs, just under your own control. \
                         Real gravity: ice falls and water pools. Real Archimedes \
                         buoyancy: steam rises once it exists.",
                    );
                    ui.label("LMB push  RMB pull  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.target_temperature = target_temperature;
        self.gravity_fraction = gravity_fraction;
        self.push_strength = push_strength;
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.target_temperature = 250.0;
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
                    .with_title("emerge -- Phase states (ice/water/steam)")
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
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.target_temperature = 250.0;
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
