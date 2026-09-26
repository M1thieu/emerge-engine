//! Grid-based Fourier heat diffusion for MPM particles.
//!
//! Implements ∂T/∂t = α·∇²T (Fourier's law) where α = k / (ρ·c_p).
//!
//! # Algorithm (per call)
//! 1. **P2G** -- scatter particle temperatures (mass-weighted) to a temporary grid
//! 2. **Normalize** -- grid_temp = grid_heat / grid_mass (mass-weighted average)
//! 3. **Laplacian** -- explicit Euler FD, T_new = T + α·dt·∇²T; a call
//!    longer than `stability_fraction` of the stable step runs 1 to 4 in
//!    several equal passes
//! 4. **G2P** -- gather temperature delta back to particles
//!
//! Uses the same quadratic B-spline kernel as MPM transfer for consistency.
//!
//! # Stable step
//! The explicit step stays stable up to dt ≤ dx² / (4α)
//! ([`ThermalConfig::stability_dt`]). `apply` splits whatever time it is
//! given into passes within `stability_fraction` of it, so it never
//! depends on the mechanics substep and never clamps it. For water at 1 cm
//! the limit is minutes and `apply` takes one pass; a misconfiguration
//! (`grid_cell_size` passed where `dx_meters` belongs, see that field's own
//! doc) now costs passes instead of a runaway. `Simulation` applies heat
//! once per `step()` with the time the step advanced; that one update used
//! to run past its limit, which folding the limit into the mechanics
//! substep did not prevent.

use glam::IVec2;

use super::transfer::heat_radiation;
use crate::{
    grid::kernel::quadratic_weights, particle::Particles,
    solver::config::DEFAULT_MATERIAL_CFL_COEFFICIENT,
};

/// Configuration for grid-based thermal diffusion.
///
/// `Default::density()` is `0.0` (same as every other field) but `alpha_grid()`
/// panics if `density <= 0.0` rather than silently dividing by it -- every real
/// caller must set it explicitly to a real reference value below; there is no
/// physically sane default to fall back to.
#[derive(Clone, Debug, Default)]
pub struct ThermalConfig {
    /// Thermal conductivity k in W/(m·K).
    ///
    /// Reference values (approximate):
    /// - Air:   0.025 W/(m·K)
    /// - Water: 0.6   W/(m·K)
    /// - Rock:  2.0   W/(m·K)
    /// - Steel: 50    W/(m·K)
    pub conductivity: f32,

    /// Specific heat capacity c_p in J/(kg·K).
    ///
    /// Reference values (approximate):
    /// - Air:   1005 J/(kg·K)
    /// - Water: 4182 J/(kg·K)
    /// - Rock:  840  J/(kg·K)
    /// - Steel: 490  J/(kg·K)
    pub heat_capacity: f32,

    /// Density ρ in kg/m³. Required -- the module's own diffusivity formula
    /// (α = k/(ρ·c_p)) needs it; omitting it silently computes α = k/c_p
    /// instead, ~1000x too fast for water.
    ///
    /// Reference values (approximate):
    /// - Air:   1.225 kg/m³
    /// - Water: 1000  kg/m³
    /// - Rock:  2500  kg/m³
    /// - Steel: 7850  kg/m³
    pub density: f32,

    /// Ambient/boundary temperature in K (or simulation-unit temperature).
    ///
    /// Grid cells with no particle mass (empty cells) are held at this temperature.
    /// Boundary cells equilibrate toward this value.
    pub ambient: f32,

    /// Grid cell physical size in meters -- pass `SimConfig::dx_meters`, NOT
    /// `SimConfig::grid_cell_size` (which is always `1.0`, a grid-unit constant, never a
    /// physical length). Passing `grid_cell_size` here understates the real cell size by
    /// orders of magnitude, inflates `alpha_grid()` to match, and silently blows the
    /// "thermal CFL is never the bottleneck" assumption -- explicit Euler overshoots into
    /// runaway temperatures within a few hundred steps. Verified by reproducing it directly.
    ///
    /// Used to convert conductivity/capacity into grid-unit diffusivity.
    pub grid_cell_size: f32,

    /// Newton cooling rate k_c in 1/s: dT/dt = −k_c·(T − ambient).
    ///
    /// Models convective heat loss to the environment. Linear in ΔT -- understates
    /// loss at high temperature, where real radiative loss (below) dominates
    /// (T⁴ vs T). 0.0 = no cooling (default, adiabatic walls).
    pub cooling_rate: f32,

    /// Surface emissivity ε ∈ [0,1] for Stefan-Boltzmann radiative loss
    /// (`transfer::heat_radiation`, σ·ε·A·(T⁴−T_ambient⁴)). 0.0 = disabled (default).
    ///
    /// Same blanket per-particle approximation `cooling_rate` already makes (every
    /// particle treated as if radiating to ambient, not gated on real free-surface
    /// exposure) -- this is a second, more accurate term for the SAME simplification,
    /// not a new architecture. `A` is the particle's own current `volume` (this
    /// engine's 2D areal-density convention already treats it as a real m² footprint
    /// with implicit unit depth, same convention `Elastic::particle_mass` uses) -- the
    /// face the render emission pass would show, per `heat_radiation`'s own doc
    /// ("physical basis for blackbody glow in the render emission pass").
    pub emissivity: f32,
}

impl ThermalConfig {
    /// Thermal diffusivity α = k / (ρ·c_p·dx²) in grid-units²/s.
    ///
    /// Folding dx² in keeps the Laplacian formula dimensionless over grid indices.
    /// Panics if `density <= 0.0` -- there's no physically sane fallback;
    /// silently dividing by zero would produce infinite/NaN diffusivity with
    /// no error at the point of the actual mistake.
    #[inline]
    pub fn alpha_grid(&self) -> f32 {
        assert!(
            self.density > 0.0,
            "ThermalConfig::density must be set to a real value (kg/m^3) -- \
             the default 0.0 has no physical meaning and would silently make \
             alpha_grid() infinite/NaN"
        );
        // α = k / (ρ·c_p·dx²): units = (m²/s) / m² = 1/s (frequency in grid coords)
        self.conductivity
            / (self.density * self.heat_capacity * self.grid_cell_size * self.grid_cell_size)
    }

    /// Explicit-diffusion stability bound `dt ≤ dx²/(4α)`, in terms of the
    /// already-dx²-folded `alpha_grid()` (so `dt ≤ 1/(4·alpha_grid())`, no
    /// separate `dx` argument needed): `stencil::FIVE_POINT_STABILITY_LIMIT`
    /// over the diffusivity. See this module's own `# Stable step`.
    #[inline]
    pub fn stability_dt(&self) -> f32 {
        1.0 / (4.0 * self.alpha_grid())
    }
}

/// Grid-based Fourier heat diffusion.
///
/// Add to `Simulation` via `solver.with_thermal(ThermalDiffusion::new(config, grid_res))`.
/// Applied once per `step()`, with the time the step advanced.
pub struct ThermalDiffusion {
    pub config: ThermalConfig,
    /// Fraction of the explicit step's stability limit each pass of `apply`
    /// uses: the definition of `SimConfig::material_cfl_coefficient`, which
    /// `Simulation` sets before each application.
    pub stability_fraction: f32,
    grid_res: usize,
    // Preallocated scratch buffers -- no per-substep heap allocation.
    grid_work: Vec<f32>, // dual-use: P2G scatter (Σ w·m·T), then Laplacian output (T_new)
    grid_mass: Vec<f32>, // Σ (w · mass) per cell
    grid_temp: Vec<f32>, // normalized T_old -- needed for G2P delta (T_new − T_old)
}

impl ThermalDiffusion {
    pub fn new(config: ThermalConfig, grid_res: usize) -> Self {
        let n = grid_res * grid_res;
        Self {
            config,
            stability_fraction: DEFAULT_MATERIAL_CFL_COEFFICIENT,
            grid_res,
            grid_work: vec![0.0; n],
            grid_mass: vec![0.0; n],
            grid_temp: vec![0.0; n],
        }
    }

    /// Longest time one explicit step stays stable
    /// ([`ThermalConfig::stability_dt`]); `None` without conduction.
    pub fn stable_step_limit(&self) -> Option<f32> {
        (self.config.alpha_grid() > 0.0).then(|| self.config.stability_dt())
    }

    /// Advances heat by `dt` seconds, of any length, in as many equal
    /// passes as keep each conduction step at `stability_fraction` of its
    /// stable step, each pass the whole way from particles to grid and
    /// back (same reason as `ScalarDiffusionField::apply`); Newton cooling
    /// is exact.
    pub fn apply(&mut self, particles: &mut Particles, dt: f32) {
        let passes = super::stencil::stable_sub_steps(
            self.config.alpha_grid() * dt,
            self.stability_fraction,
        );
        let pass_dt = dt / passes as f32;
        for _ in 0..passes {
            self.apply_pass(particles, pass_dt);
        }
    }

    fn apply_pass(&mut self, particles: &mut Particles, sub_dt: f32) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;

        // --- Clear scratch (grid_work = P2G scatter, grid_mass = weights) ---
        for i in 0..n {
            self.grid_work[i] = 0.0;
            self.grid_mass[i] = 0.0;
        }

        // --- P2G: scatter mass-weighted temperature into grid_work ---
        for pi in 0..particles.len() {
            let x = particles.x[pi];
            let mass = particles.mass[pi];
            let temperature = particles.temperature[pi];
            let w = quadratic_weights(x);
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    let mw = weight * mass;
                    self.grid_work[idx] += mw * temperature;
                    self.grid_mass[idx] += mw;
                }
            }
        }

        // --- Normalize: grid_temp = T_old; empty cells = ambient ---
        // grid_work (P2G scatter) is discarded and reused for Laplacian output below.
        for i in 0..n {
            self.grid_temp[i] = if self.grid_mass[i] > 1e-10 {
                self.grid_work[i] / self.grid_mass[i]
            } else {
                self.config.ambient
            };
        }

        // --- Laplacian: explicit Euler FD, output into grid_work ---
        // grid_temp = T_old (read-only). grid_work = T_new (write). `apply`
        // keeps `sub_dt` within the stable step.
        super::stencil::laplacian_step(
            &self.grid_temp,
            &mut self.grid_work,
            self.grid_res,
            self.config.alpha_grid() * sub_dt,
            self.config.ambient,
        );

        // --- G2P: gather Δ T = (T_new − T_old) back to particles ---
        // Delta scatter preserves per-particle state at sparse/edge regions.
        // grid_work = T_new, grid_temp = T_old.
        for pi in 0..particles.len() {
            let x = particles.x[pi];
            let w = quadratic_weights(x);
            let mut delta = 0.0f32;
            let mut w_sum = 0.0f32;

            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    delta += weight * (self.grid_work[idx] - self.grid_temp[idx]);
                    w_sum += weight;
                }
            }

            if w_sum > 1e-10 {
                particles.temperature[pi] += delta / w_sum;
            }
        }

        // Newton cooling: dT/dt = −k_c·(T − T_ambient), by its exact
        // solution T − ambient ← (T − ambient)·exp(−k_c·dt). The explicit
        // Euler form it replaces overshot ambient past k_c·dt = 1.
        if self.config.cooling_rate > 0.0 {
            let kept = (-self.config.cooling_rate * sub_dt).exp();
            let ambient = self.config.ambient;
            for pi in 0..particles.len() {
                particles.temperature[pi] = ambient + (particles.temperature[pi] - ambient) * kept;
            }
        }

        // Stefan-Boltzmann radiative loss: dT/dt = -q/(m·c_p), q = heat_radiation(...).
        // See `ThermalConfig::emissivity`'s own doc for why area = particle volume and
        // why this is the same blanket-exposure approximation as Newton cooling above.
        if self.config.emissivity > 0.0 {
            let ambient = self.config.ambient;
            let c_p = self.config.heat_capacity;
            for pi in 0..particles.len() {
                let heat_capacity_j_per_k = particles.mass[pi] * c_p;
                if heat_capacity_j_per_k <= 0.0 {
                    continue;
                }
                let area = particles.volume[pi];
                let q_watts = heat_radiation(
                    particles.temperature[pi],
                    ambient,
                    area,
                    self.config.emissivity,
                    1.0,
                );
                particles.temperature[pi] -= q_watts / heat_capacity_j_per_k * sub_dt;
            }
        }
    }
}
