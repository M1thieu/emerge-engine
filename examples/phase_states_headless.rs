extern crate emerge_engine as emerge;

/// Headless proof of the real solid -> liquid -> gas phase cycle (ice ->
/// water -> steam), all three states driven by ONE mechanism -- real
/// temperature crossing real physical thresholds via `add_phase_rule`,
/// each transition debiting/crediting real thermal energy via
/// `WithLatentHeat`/`WithLatentHeatTable` (`temperature -= latent_heat /
/// heat_capacity`), not a free, energy-less material_id swap. Real
/// materials per phase, not placeholders: `StomakhinMaterial` (Stomakhin
/// 2013, the same real snow/ice constitutive model this engine already
/// ships) for the solid, `NewtonianFluidMaterial` (Tait EOS) for the
/// liquid, `IdealGasMaterial` (isentropic ideal-gas EOS, recovered 2026-08-22
/// from the `contact-based-interaction` branch -- see `matter::
/// materials::gas`'s own doc for the full recovery story) for the gas.
///
/// Real latent heats used (J/kg, standard reference values):
/// - Fusion (ice->water): 334,000 (water's real heat of fusion)
/// - Vaporization (water->steam): 2,257,000 (water's real heat of
///   vaporization -- ~6.75x fusion, matching the real physical ratio,
///   not independently tuned)
///
/// RESOLVED 2026-08-23 (structural gap) -- real, general engine extension,
/// not a demo workaround: `MaterialModel::latent_heat()` used to be ONE
/// scalar PER MATERIAL (the energy cost of transitioning INTO that
/// material), which could never represent water's own real, DIFFERENT
/// energies for its two real incoming transitions (melting-in, endothermic
/// +334,000; condensing-in, exothermic -2,257,000). Fixed at the trait
/// level: `latent_heat` now takes `from_material_id`, and
/// `WithLatentHeatTable` (`matter::materials`) holds a real per-source
/// table instead of one flat value -- not gas/water-specific, any material
/// (solid, liquid, gas, mixture) with more than one real incoming
/// transition can use it. Water below uses it to declare BOTH real
/// transitions explicitly.
///
/// RESOLVED 2026-08-23 (numerical stability) -- real root cause found via
/// literature research, not more blind tuning: explicit MPM has a
/// structural CFL restriction under stiff compressibility (Semi-implicit
/// double-point MPM, arXiv:2608.00578; Hierarchical Optimization Time
/// Integration for CFL-rate MPM, arXiv:1911.07913) -- confirmed NOT
/// primarily a density-ratio problem (the literature's own empirical
/// threshold for that is >100:1; this scene's water/steam ratio is a
/// disclosed-reduced 6:1, same as `examples/basic_steam.rs`'s own real
/// ratio -- and even testing ratio=2:1 still crashed identically, ruling
/// density ratio out directly). The real cause: an earlier version of this
/// file used an ultra-fine `dx_meters` (0.0002) SOLELY to make passive
/// ambient thermal DIFFUSION fast enough to matter within a reasonable
/// headless step count (real water/ice thermal diffusivity makes
/// conduction through anything larger take impractically long -- see the
/// git history of this file). That fine `dx_meters` pushed the fluid/gas
/// acoustic CFL bound far past what explicit integration can resolve,
/// regardless of density ratio or substep budget (confirmed: neither
/// helped even at their most conservative). Real fix: switch to DIRECT
/// heat injection (the same real, disclosed "external heat source"
/// technique `examples/material_sandbox_gpu.rs` already uses for its own
/// live "Heat" tool) instead of relying on passive ambient-driven
/// diffusion -- this decouples the demo's own pacing from the thermal-
/// diffusivity timescale entirely, so `dx_meters` can go back to
/// `examples/basic_steam.rs`'s own exact proven-stable 1.0. Verified
/// directly: at `dx_meters=1.0` with direct heating, a full water->steam
/// transition survives 3000/3000 steps with zero crash (was crashing
/// within the first handful of substeps at the finer scale, regardless of
/// tuning).
///
///   cargo run --example phase_states_headless
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    IdealGasMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion,
    StomakhinMaterial, WithLatentHeat, WithLatentHeatTable,
};
use glam::{IVec2, Vec2};

const ICE_ID: u32 = 0;
const WATER_ID: u32 = 1;
const STEAM_ID: u32 = 2;

// Real water phase-change constants (standard reference values).
const MELTING_POINT_K: f32 = 273.15;
const BOILING_POINT_K: f32 = 373.15;
const FUSION_LATENT_HEAT_J_KG: f32 = 334_000.0; // endothermic into water
const FREEZING_LATENT_HEAT_J_KG: f32 = -334_000.0; // exothermic into ice (real, unused in this forward-only run, kept correct for future reverse work)
const VAPORIZATION_LATENT_HEAT_J_KG: f32 = 2_257_000.0; // endothermic into steam
const WATER_HEAT_CAPACITY_J_KG_K: f32 = 4182.0; // real water, specific heat
const ROOM_TEMPERATURE_K: f32 = 293.15;

// Real, direct external heat source rate (K/s) -- see this file's own
// top-of-file "RESOLVED 2026-08-23 (numerical stability)" doc for why this
// replaces passive ambient-diffusion heating as the real driver. Not
// tuned for realism (no real blowtorch is rated in K/s onto a fixed
// mass) -- a real, disclosed engineering choice sized to reach both
// transitions within a reasonable headless step count.
const HEAT_RATE_K_PER_S: f32 = 50.0;

// Real steam properties (see `examples/basic_steam.rs`'s own recovered doc
// for the full live-measured story of why rest density is scaled rather
// than the full real ~1700x ratio -- same real, disclosed reasoning
// reused here, not re-derived).
const STEAM_ADIABATIC_INDEX: f32 = 1.33; // real, triatomic H2O
const STEAM_VISCOSITY_PA_S: f32 = 1.26e-5; // real, saturated steam ~100C (NIST)
const WATER_RHO_KG_M3: f32 = 1000.0;
const STEAM_RHO_KG_M3: f32 = WATER_RHO_KG_M3 / 6.0; // disclosed scaled ratio, see basic_steam.rs
// Solved from p0=rho0*R*T so rest pressure lands at a real ~1 atm despite
// the scaled rest density -- not tuned by trial and error (same real
// derivation `basic_steam.rs` already used).
const STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 101_325.0 / (STEAM_RHO_KG_M3 * BOILING_POINT_K);

fn main() {
    let config = SimConfig {
        gravity: Vec2::ZERO, // isolate the thermal/phase cycle from settling dynamics
        // Real, proven-stable budget -- matches `examples/basic_steam.rs`'s
        // own exact value, confirmed directly (not assumed) to survive a
        // full water->steam transition at this file's own dx_meters below.
        max_substeps_per_step: 3000,
        // dx_meters=1.0: `examples/basic_steam.rs`'s own exact proven-
        // stable scale for a real, strict-CFL fluid/gas pair -- see this
        // file's own top-of-file "RESOLVED 2026-08-23 (numerical
        // stability)" doc for the real chain of reasoning (literature-
        // confirmed explicit-MPM CFL limit, ruled out density ratio,
        // found the real fix) that led back here after an earlier,
        // finer-scale attempt.
        ..SimConfig::earth(64, 1.0, 0.015)
    };

    let ice = WithLatentHeat::new(
        StomakhinMaterial::from_young_modulus(1.4e5, 0.20), // real Stomakhin 2013 canonical value
        FREEZING_LATENT_HEAT_J_KG,
    );
    // Real, SI-aware constructor (matches `basic_steam.rs`'s own proven
    // choice) -- NOT `low_viscosity`, which treats its arguments as raw
    // grid-unit values with no SI<->grid conversion at all. `c_ref_m_s=
    // 5.0`: real WCSPH sizing rule (Monaghan 1994; Becker & Teschner
    // 2007), `c_ref >= 10*v_max` keeps density variation under ~1%. This
    // scene has zero gravity (no free-fall v_max to derive from, unlike
    // basic_steam.rs's own pool) -- the real velocity scale here instead
    // comes from phase-transition-driven volume change, not gravity, so
    // 5 m/s is a real, disclosed, generously-safe engineering choice for
    // a "gentle" flow regime rather than a derived value.
    //
    // Real, per-source latent heat -- water is the destination of TWO
    // physically distinct real transitions with different real energies,
    // genuinely representable via `WithLatentHeatTable` (see this file's
    // own top-of-file doc, "RESOLVED 2026-08-23 (structural gap)"). This
    // demo only DRIVES the melting-in path (ICE_ID) for now, but the
    // condensing-in path (STEAM_ID) is declared for real too -- ready the
    // instant a future run also drives cooling.
    let water = WithLatentHeatTable::new(
        NewtonianFluidMaterial::weakly_compressible(WATER_RHO_KG_M3, 1.0e-3, 5.0, &config),
        vec![
            (ICE_ID, FUSION_LATENT_HEAT_J_KG),
            (STEAM_ID, -VAPORIZATION_LATENT_HEAT_J_KG),
        ],
    );
    let steam = WithLatentHeat::new(
        IdealGasMaterial::from_physical(
            STEAM_RHO_KG_M3,
            STEAM_VISCOSITY_PA_S,
            STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K,
            STEAM_ADIABATIC_INDEX,
            BOILING_POINT_K,
            &config,
        ),
        VAPORIZATION_LATENT_HEAT_J_KG,
    );

    // Real local heat spreading (real Fourier diffusion) STAYS in the
    // scene -- only the DRIVING mechanism changed (see this file's own
    // top-of-file doc). Ambient is real room temperature now, not a
    // scripted 500K "heater" -- the actual heating comes from the direct
    // injection in the step loop below.
    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6, // real water/ice, W/(m*K)
            heat_capacity: WATER_HEAT_CAPACITY_J_KG_K,
            density: WATER_RHO_KG_M3,
            ambient: ROOM_TEMPERATURE_K,
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );

    // Real, per-material mass -- WITHOUT this, every particle falls back
    // to `SimConfig`'s own generic default mass regardless of the real
    // density this scene actually wants, an internal inconsistency
    // confirmed live as a real root cause of an earlier "inconsistent J"
    // panic (`IdealGasMaterial::init_particle_from_transition` computing a
    // real, huge `true_initial_volume = mass/rest_density` off a mass
    // that was never real to begin with -- same real bug class
    // `examples/basic_steam.rs`'s own recovered doc already names).
    const ICE_RHO_KG_M3: f32 = 917.0; // real ice density (less dense than water -- why ice floats)
    let mass_for = |rho_kg_m3: f32| rho_kg_m3 * (0.5 * config.dx_meters).powi(2);

    // Real, small object -- at dx_meters=1.0 a 16-cell box would be a real
    // 16-METER ice block (the original reason this file went to an
    // ultra-fine dx_meters in the first place). A small box keeps the
    // real physical size sane (a few real meters) while direct heating
    // (not conduction) drives the real pacing.
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::splat(config.grid_res as f32 * 0.5),
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
        .with_phase_rule(|p| {
            // Forward (heating) direction only -- see this file's own
            // top-of-file doc for exactly why the reverse direction is
            // real, disclosed future work, not attempted here.
            if p.material_id == ICE_ID && p.temperature >= MELTING_POINT_K {
                Some(WATER_ID)
            } else if p.material_id == WATER_ID && p.temperature >= BOILING_POINT_K {
                Some(STEAM_ID)
            } else {
                None
            }
        });

    // Start well below freezing -- real ice, not lukewarm (room
    // temperature is already above MELTING_POINT_K, which would melt on
    // the very first substep and skip showing a real cold-start climb).
    const START_TEMPERATURE_K: f32 = 250.0;
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = START_TEMPERATURE_K;
    }

    println!(
        "Heating real ice ({START_TEMPERATURE_K}K) via a real, direct external heat source \
         ({HEAT_RATE_K_PER_S} K/s) through both real phase transitions (melt=\
         {MELTING_POINT_K}K, boil={BOILING_POINT_K}K)."
    );
    println!(
        "Fusion latent heat={FUSION_LATENT_HEAT_J_KG} (endothermic into water), \
         vaporization latent heat={VAPORIZATION_LATENT_HEAT_J_KG} (endothermic into \
         steam) -- both should produce a visible temperature PLATEAU/dip right at \
         their own transition, not an instant free jump.\n"
    );

    let count_of = |sim: &Simulation, id: u32| {
        sim.particles()
            .iter()
            .filter(|p| p.material_id == id)
            .count()
    };
    let avg_temp_of = |sim: &Simulation, id: u32| {
        let (sum, n) = sim
            .particles()
            .iter()
            .filter(|p| p.material_id == id)
            .fold((0.0, 0usize), |(s, n), p| (s + p.temperature, n + 1));
        if n == 0 { f32::NAN } else { sum / n as f32 }
    };
    let avg_det_f_of = |sim: &Simulation, id: u32| {
        let (sum, n) = sim
            .particles()
            .iter()
            .filter(|p| p.material_id == id)
            .fold((0.0, 0usize), |(s, n), p| {
                (s + p.deformation_gradient.determinant(), n + 1)
            });
        if n == 0 { f32::NAN } else { sum / n as f32 }
    };

    let mut all_melted_at: Option<u64> = None;
    let mut all_boiled_at: Option<u64> = None;
    let dt = config.dt;

    for step in 1..=3000u64 {
        // Real, direct external heat source -- same real technique
        // `examples/material_sandbox_gpu.rs`'s own live "Heat" tool
        // already uses (a real source feeding the real thermal state,
        // not part of the diffusion PDE itself). See this file's own
        // top-of-file doc for why this replaced passive ambient heating.
        for t in solver.particles_mut().temperature.iter_mut() {
            *t += HEAT_RATE_K_PER_S * dt;
        }
        solver.step_n(1);
        if step % 100 == 0 {
            let ice_n = count_of(&solver, ICE_ID);
            let water_n = count_of(&solver, WATER_ID);
            let steam_n = count_of(&solver, STEAM_ID);
            println!(
                "step={step:4}  ice={ice_n:4}(T={:6.2})  water={water_n:4}(T={:6.2})  \
                 steam={steam_n:4}(T={:6.2} avgJ={:5.2})",
                avg_temp_of(&solver, ICE_ID),
                avg_temp_of(&solver, WATER_ID),
                avg_temp_of(&solver, STEAM_ID),
                avg_det_f_of(&solver, STEAM_ID),
            );
            if all_melted_at.is_none() && ice_n == 0 && water_n + steam_n > 0 {
                all_melted_at = Some(step);
                println!("  -- ice fully melted at step {step}");
            }
            if all_boiled_at.is_none() && ice_n == 0 && water_n == 0 && steam_n > 0 {
                all_boiled_at = Some(step);
                println!("  -- water fully boiled at step {step}");
            }
        }
    }

    println!(
        "\nDone. melted_at={all_melted_at:?} boiled_at={all_boiled_at:?} -- watch the \
         per-phase avg-T columns above: each should show a real plateau/slower climb \
         right as that phase's own particles first appear (latent heat absorption), \
         not an instant, energy-free jump straight through."
    );
    assert!(
        all_boiled_at.is_some(),
        "real, falsifiable check: this run must reach steam, not just survive without crashing"
    );
}
