use glam::Vec2;

/// D⁻¹ = 4.0 for the quadratic B-spline MLS-MPM kernel (always).
/// Not a tunable parameter — hardcoded from Hu 2018 Table 1.
pub(crate) const KERNEL_D_INVERSE: f32 = 4.0;

/// Parameters that control the physics solver and its runtime behavior.
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub grid_res: usize,
    pub grid_cell_size: f32,
    pub dt: f32,
    pub adaptive_timestep: bool,
    pub cfl_include_affine_speed: bool,
    pub cfl_coefficient: f32,
    pub material_cfl_coefficient: f32,
    pub viscous_timestep_coefficient: f32,
    /// Safety factor for `rod::rod_cfl_dt`'s own bound, folded into
    /// `choose_substep_dt` alongside `material_cfl_coefficient`. Not the same
    /// 0.5 as `material_cfl_coefficient`: `rod_cfl_dt` sums every stiffness/
    /// damping term touching each point (a Gershgorin row-sum bound), which
    /// is real but LOOSE for the rod's geometrically nonlinear dynamics — 0.5
    /// diverges for a long/stiff-EI cantilever at N=30/40; 0.4 is the
    /// bisected, long-horizon-verified safe value across that regime and a
    /// short/soft blade-of-grass regime (see `project_rod_cfl_gershgorin_and_cookbook_2026-07-21` memory).
    pub rod_cfl_coefficient: f32,
    pub min_dt: f32,
    pub project_invalid_state: bool,
    pub projection_min_density: f32,
    pub projection_min_volume: f32,
    pub projection_min_deformation_j: f32,
    /// Gravitational acceleration in grid-coordinate units/s².
    /// Use `Vec2::new(x, y)` for angled or planetary gravity. Typical: `Vec2::new(0.0, -9.81)`.
    pub gravity: Vec2,
    /// Direction light is sensed as coming FROM, for `rod::Phototropism`
    /// (see that struct's own doc). A FIXED, externally-set vector, NOT a
    /// real solar/orbital model — `emerge`/LP work at continuum scale, no
    /// day/night sun-angle system exists (the existing `day_night_thermal_gpu`
    /// demo is a pure scalar ambient-temperature oscillation with no light
    /// direction at all). Default: straight up (`Vec2::new(0.0, 1.0)`,
    /// opposite the default `gravity` direction) — "light from directly
    /// above," the common illustrative case. Zero cost/no behavior change
    /// for any rod that doesn't opt into `Phototropism`.
    pub light_dir: Vec2,
    pub boundary_thickness: usize,
    pub default_initial_volume: f32,
    pub recompute_density_each_step: bool,
    pub particle_mass: f32,
    /// Maximum substeps the adaptive loop may run per step() call.
    /// Prevents stiff materials or fast particles from making a single step() unboundedly expensive.
    /// 64 covers snow at lambda=38889 (c_P≈197, ~50 substeps) with headroom.
    pub max_substeps_per_step: usize,
    /// APIC affine-matrix blend [0, 1].
    /// 1.0 = full APIC (angular-momentum-conserving, taichi default).
    /// 0.0 = pure PIC (maximum numerical dissipation, fastest settling).
    /// Intermediate values blend between the two — equivalent to taichi's `apic_damping`.
    ///
    /// **Correction (2026-07-26, superseding an earlier version of this comment
    /// from the same investigation)**: an EOS fluid's affine matrix C can run
    /// measurably "hot" (real measured C-norm ~1900 vs ~0.3 for an identical
    /// solid scene) -- but this is NOT an intrinsic property of pure APIC or of
    /// EOS-based fluids in general. Root cause, confirmed by direct A/B: it was
    /// a `rest_density` vs. actual spawn density MISCALIBRATION (a scene
    /// declaring `rest_density=1.0` while its `particle_mass`/`spacing`
    /// combination produces a real kernel-estimated density of 4.0 --
    /// `estimate_particle_volumes`'s density is `mass/spacing²` in the bulk).
    /// A stiff (7th-power) EOS reacting to an already-~4x-wrong density
    /// injects real energy from the very first substep (confirmed via total
    /// mechanical energy, KE+PE, spiking to 600-670x its own initial value --
    /// an absolute physical bound, not a suspicious-looking number). Once
    /// `rest_density` is corrected to match the real spawn density, energy is
    /// genuinely conserved and the C matrix stays calm (~4, not ~1900) even at
    /// this field's own default of 1.0 -- no blend tuning required.
    ///
    /// **Practical guidance**: before reaching for a low `apic_blend` on an
    /// unstable `NewtonianFluidMaterial`/`BinghamFluidMaterial` scene, check
    /// `rest_density` against what the spawn's `particle_mass`/`spacing`
    /// actually produces -- that's very likely the real fix. `apic_blend<=0.05`
    /// remains a real, working mitigation for scenes where the calibration
    /// can't be fixed directly, but it treats a symptom, not this cause. See
    /// `tests/accuracy.rs::fluid_energy_conserved_with_correct_rest_density`
    /// (the real fix, demonstrated) and its sibling `#[ignore]`d
    /// `miscalibrated_rest_density_injects_spurious_energy` (the historical
    /// repro of the bug this comment used to misdiagnose as a blend problem).
    pub apic_blend: f32,
    /// Upper bound on volumetric expansion J = det(F).
    /// Particles that expand beyond this are rescaled back. No physical material expands
    /// this many times its initial volume without fracturing or flowing first.
    /// Default 50.0. Set higher for extreme-deformation sims (explosions, impacts).
    pub j_max: f32,
    /// Speed below which a passive (activation == 0) particle becomes eligible for sleep.
    /// 0.0 = sleep disabled. Typical: 0.01–0.05 grid-cells/s.
    /// Sleeping particles skip P2G and G2P entirely; woken by neighbouring active cells.
    pub sleep_threshold: f32,
    /// Speed below which a whole rod (max over ALL its points) becomes eligible
    /// for sleep — separate knob from `sleep_threshold` since a rod's natural
    /// residual-sway speed under wind is a different scale than an MPM
    /// particle's. 0.0 = sleep disabled (default — no existing rod scene's
    /// behavior changes). A rod with an active push (`Rod::push_strength > 0`)
    /// never sleeps regardless of this value. Sleeping rods skip scatter/
    /// gather/internal-force integration AND their own `rod_cfl_dt` term in
    /// `choose_substep_dt` — the real cost driver for many simultaneous rods.
    pub rod_sleep_threshold: f32,
    /// Coulomb friction coefficient for multi-field contact between a `contact_group != 0`
    /// particle and everything else (Bardenhagen 2001 — see `Particle::contact_group` doc).
    /// Only has any effect at all when at least one particle actually sets a nonzero
    /// `contact_group`; otherwise `Grid::resolve_contact` never has anything to resolve,
    /// regardless of this value. 0.0 = frictionless (normal no-penetration only, free
    /// tangential slip). Real dry-material Coulomb coefficients are typically 0.3-0.9.
    pub contact_friction: f32,
    /// ASFLIP blend factor [0, 1] (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in
    /// the Material Point Method: A Scheme for Easier Separation and Less Dissipation", ACM
    /// TOG 40(4)). Reintroduces a FLIP-style velocity/position correction on top of ordinary
    /// APIC, letting granular/debris material separate crisply instead of smearing together.
    /// 0.0 = disabled — byte-identical to plain APIC, the default for every existing scene/
    /// test. ~0.97 matches the paper's own reference implementation (`nepluno/pyasflip`).
    /// Costs nothing when 0.0: no grid-velocity snapshot is taken, G2P takes the exact
    /// original code path.
    pub asflip_blend: f32,
    /// Cundall local non-viscous damping coefficient [0, 1] (Cundall 1982/1987
    /// "dynamic relaxation"; MPM formulation per Beuth, Benz, Vermeer, Coetzee,
    /// Bonnier & van den Berg 2007, "Formulation and Application of a Quasi-
    /// Static Material Point Method," NUMOG X — used in production geotechnical
    /// MPM, e.g. Anura3D). Real, material-agnostic fix for the mismatch an
    /// explicit-dynamic MPM solver has with an inherently quasi-static problem
    /// (a granular pile creeping toward equilibrium): damps the component of
    /// each grid cell's velocity change THIS substep (a real proxy for applied
    /// force, since Δv = F·dt/m at fixed dt/mass) that opposes nothing but its
    /// own oscillation — proportional to the FORCE just applied, not to
    /// velocity itself (that's ordinary viscous damping, a different real
    /// mechanism already available via `ViscoelasticMaterial`). Self-gating by
    /// construction: a cell with zero velocity has nothing to oppose (zero
    /// damping), and steady DIRECTED motion (a creature walking, a fluid
    /// splash) barely engages it — only genuine wobble/settling does. Lives at
    /// the grid level, not inside any one material's constitutive law, so
    /// every material benefits once enabled, not just granular ones.
    /// 0.0 = disabled (default) — no velocity snapshot taken, byte-identical
    /// to every existing scene, same zero-cost convention as `asflip_blend`.
    pub cundall_damping: f32,
    /// Two-phase mixture coupling drag coefficient (Tampubolon et al. 2017,
    /// "Multi-species simulation of porous sand and water mixtures" — Darcy-style
    /// momentum exchange between a `MixturePhase::Solid` and `MixturePhase::Fluid`
    /// material, see `WithMixturePhase`). Units: mass/time (a per-node drag rate,
    /// NOT the paper's own permeability-derived `c_E` directly — this is a first,
    /// simplified scalar-coefficient version; mapping to real soil permeability/
    /// porosity is real, disclosed future work, not attempted yet).
    /// 0.0 = disabled (default) — `Grid::has_mixture_activity()` gates the extra
    /// P2G scatter and the whole resolve pass, zero cost for every scene that
    /// doesn't use `WithMixturePhase`, matching `asflip_blend`'s own convention.
    pub mixture_drag_coefficient: f32,
    /// Jacobi iterations for the mixture incompressibility pressure projection
    /// (`Grid::project_mixture_incompressibility`, see its own doc for the full
    /// derivation and citations). Real fix for a real, root-caused instability:
    /// the drag coupling above conserves momentum but never enforces the
    /// mixture's actual incompressibility constraint, so under sustained/
    /// confined loading (water settled into sand) the violation compounds
    /// silently over hundreds of steps until velocities blow past the CFL
    /// bound. 0 = disabled (default) — byte-identical to the original
    /// momentum-only coupling, matching every other opt-in field's convention.
    /// Real, disclosed caveat: this is an approximate, real-time-affordable
    /// Jacobi solve, not an exact Poisson solve — pick this value by measuring
    /// against your actual scene's long-settle behavior (a settled, confined
    /// liquid is the documented worst case for a low iteration count), not by
    /// assuming a small fixed count is free.
    pub mixture_pressure_iterations: u32,

    // ── Physical unit scaling ──────────────────────────────────────────────────
    // Default 1.0 = simulation units (no scaling). Set these to enable SI-calibrated materials.
    // Use `lame_from_si` / `gravity_to_grid` in `materials::utils` to convert SI values.
    /// Physical length of one grid cell in meters. Default 1.0 (grid units).
    ///
    /// Example: if the simulation domain is 64 cells representing 0.64 m, set `dx_meters = 0.01`.
    pub dx_meters: f32,
    /// Physical duration of one simulation time unit in seconds. Default 1.0.
    ///
    /// Typically set to match `config.dt` in physical seconds.
    /// Gravity: `gravity = Vec2::new(0.0, -9.81) * dt_seconds^2 / dx_meters`.
    pub dt_seconds: f32,
}

impl Default for SimConfig {
    /// Safe production defaults: adaptive timestepping on, state projection on.
    /// Use [`SimConfig::standard`] or [`SimConfig::earth`] in practice — they set the
    /// important physical parameters (grid_res, dt, gravity) from arguments.
    fn default() -> Self {
        Self {
            grid_res: 64,
            grid_cell_size: 1.0,
            dt: 1.0,
            adaptive_timestep: true,
            cfl_include_affine_speed: true,
            cfl_coefficient: 0.9,
            material_cfl_coefficient: 0.5,
            viscous_timestep_coefficient: 0.5,
            rod_cfl_coefficient: 0.4,
            min_dt: 1.0e-3,
            project_invalid_state: true,
            projection_min_density: 1.0e-6,
            projection_min_volume: 1.0e-6,
            projection_min_deformation_j: 1.0e-6,
            gravity: Vec2::new(0.0, -0.05),
            light_dir: Vec2::new(0.0, 1.0),
            boundary_thickness: 2,
            default_initial_volume: 1.0,
            recompute_density_each_step: false,
            particle_mass: 1.0,
            max_substeps_per_step: 64,
            apic_blend: 1.0,
            j_max: 50.0,
            sleep_threshold: 0.0,
            rod_sleep_threshold: 0.0,
            contact_friction: 0.5,
            asflip_blend: 0.0,
            cundall_damping: 0.0,
            mixture_drag_coefficient: 0.0,
            mixture_pressure_iterations: 0,
            dx_meters: 1.0,
            dt_seconds: 1.0,
        }
    }
}

impl SimConfig {
    /// Simulation-ready config: sets the three physical parameters that differ per sim.
    ///
    /// Inherits safe defaults from `Default` (adaptive timestepping, state projection on).
    pub fn standard(grid_res: usize, dt: f32, gravity: Vec2) -> Self {
        Self {
            grid_res,
            dt,
            gravity,
            ..Self::default()
        }
    }

    /// Stripped-down config with adaptive timestepping and state projection disabled.
    ///
    /// Use only for: unit tests that need exact deterministic substeps, benchmarks
    /// where you want to measure a fixed workload, or comparing against an external reference.
    /// Never use for real simulations — J can go negative and NaN-cascade.
    pub fn unsafe_defaults() -> Self {
        Self {
            adaptive_timestep: false,
            project_invalid_state: false,
            ..Self::default()
        }
    }

    /// Earth-scale simulation preset.
    ///
    /// Derives gravity and unit scaling from real physical constants so that
    /// material parameters passed via `lame_from_si` produce correct behaviour.
    ///
    /// # Arguments
    /// * `grid_res`    — number of cells per side
    /// * `cell_m`      — physical size of one grid cell in metres (e.g. `0.01` for 1 cm)
    /// * `dt`          — frame time step in simulation seconds (e.g. `0.05`)
    ///
    /// # Derived values
    /// `gravity_solver = 9.81 / cell_m` cells/s² (downward, −Y).
    ///
    /// # Example
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::SimConfig;
    /// // 64-cell domain, 1 cm/cell → g = 981 cells/s²
    /// let config = SimConfig::earth(64, 0.01, 0.05);
    /// ```
    pub fn earth(grid_res: usize, cell_m: f32, dt: f32) -> Self {
        // g [cells/s²] = 9.81 [m/s²] / cell_m [m/cell]
        // Derived from v += gravity * sub_dt where sub_dt is in real seconds.
        let g_solver = 9.81 / cell_m;
        Self {
            dx_meters: cell_m,
            dt_seconds: dt,
            ..Self::standard(grid_res, dt, Vec2::new(0.0, -g_solver))
        }
    }

    // ── SI conversion helpers ─────────────────────────────────────────────────

    /// Convert SI Young's modulus (Pa) + Poisson ratio to grid-unit Lamé parameters.
    ///
    /// Equivalent to `lame_from_si(e_pa, nu, rho, self.dx_meters, self.dt_seconds)`.
    /// Requires `earth()` or explicit `dx_meters`/`dt_seconds` to be meaningful.
    pub fn lame_from_si_cfg(&self, e_pa: f32, nu: f32, rho_kg_m3: f32) -> (f32, f32) {
        crate::materials::lame_from_si(e_pa, nu, rho_kg_m3, self.dx_meters, self.dt_seconds)
    }

    /// Convert SI stress or pressure (Pa) to grid units.
    ///
    /// Use for: yield stress, tensile strength, eos_stiffness, surface tension.
    /// Scale: `p_grid = p_SI · dt² / (ρ · dx²)`
    pub fn stress_from_si(&self, pa: f32, rho_kg_m3: f32) -> f32 {
        pa * self.dt_seconds * self.dt_seconds / (rho_kg_m3 * self.dx_meters * self.dx_meters)
    }

    /// Convert SI dynamic viscosity (Pa·s) to grid units.
    ///
    /// Viscosity multiplies the velocity gradient (units: 1/step in grid space), so its
    /// non-dimensionalization has one extra factor of dt versus stress:
    /// `η_grid = η_SI · ρ · dx² / dt³`
    pub fn visc_from_si(&self, eta_pa_s: f32, rho_kg_m3: f32) -> f32 {
        eta_pa_s * rho_kg_m3 * self.dx_meters * self.dx_meters
            / (self.dt_seconds * self.dt_seconds * self.dt_seconds)
    }

    /// Validate solver-side numerical and domain constraints.
    pub fn validate(&self) {
        assert!(self.grid_res >= 4, "grid_res must be >= 4");
        assert!(self.grid_cell_size > 0.0, "grid_cell_size must be positive");
        assert!(self.dt > 0.0, "dt must be positive");
        assert!(
            self.cfl_coefficient > 0.0,
            "cfl_coefficient must be positive"
        );
        assert!(
            self.material_cfl_coefficient > 0.0,
            "material_cfl_coefficient must be positive"
        );
        assert!(
            self.rod_cfl_coefficient > 0.0,
            "rod_cfl_coefficient must be positive"
        );
        assert!(
            self.viscous_timestep_coefficient > 0.0,
            "viscous_timestep_coefficient must be positive"
        );
        assert!(self.min_dt > 0.0, "min_dt must be positive");
        assert!(self.min_dt <= self.dt, "min_dt must be <= dt");
        assert!(
            self.projection_min_density > 0.0,
            "projection_min_density must be positive"
        );
        assert!(
            self.projection_min_volume > 0.0,
            "projection_min_volume must be positive"
        );
        assert!(
            self.projection_min_deformation_j > 0.0,
            "projection_min_deformation_j must be positive"
        );
        assert!(self.particle_mass > 0.0, "particle_mass must be positive");
        assert!(
            self.contact_friction >= 0.0,
            "contact_friction must be non-negative"
        );
        assert!(
            self.max_substeps_per_step > 0,
            "max_substeps_per_step must be > 0"
        );
        assert!(
            self.default_initial_volume > 0.0,
            "default_initial_volume must be positive"
        );
        assert!(self.j_max > 1.0, "j_max must be > 1.0");
        assert!(
            (0.0..=1.0).contains(&self.apic_blend),
            "apic_blend must be in [0, 1]"
        );
        assert!(
            self.boundary_thickness > 0 && self.boundary_thickness < self.grid_res - 1,
            "boundary_thickness must be in [1, grid_res-2]"
        );
    }
}

// `SpawnRegion` (initial particle layout) + its `SpawnShape` mask and fluent
// builder methods live in spawn.rs -- see that file's own doc comment.
// Re-exported here so every existing `crate::solver::config::SpawnRegion`/
// `SpawnShape` path (and the crate-root `emerge::SpawnRegion`/`SpawnShape`
// re-export in lib.rs) keeps resolving unchanged.
mod spawn;
pub use spawn::{SpawnRegion, SpawnShape};
