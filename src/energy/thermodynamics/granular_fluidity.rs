//! Nonlocal Granular Fluidity (NGF) field — lets granular flow "cooperate"
//! spatially instead of every point deciding to yield in total local
//! isolation. Real, published mechanism (Kamrin & Koval, PRL 2012; Henann &
//! Kamrin, several follow-ups); the MPM-specific numerical scheme below
//! matches a real, peer-reviewed reference read directly (not from its
//! abstract): Haeri & Skonieczny 2022, *"Three-dimensional granular flow
//! continuum modeling via material point method with hyperelastic nonlocal
//! granular fluidity"* (CMAME, arXiv:2111.01523).
//!
//! # Physics
//! `g` (granular fluidity) relates plastic shear strain rate to the stress
//! ratio: `γ̇ = g·μ`. Governed by (Henann & Kamrin 2014, arXiv:1408.5205,
//! eq. 6 — the dynamical form, verified via `pdftotext` against the real
//! PDF, not recalled from memory):
//! ```text
//! t0 · ∂g/∂t = A²d²·∇²g − (μs−μ)·g − b·√(P/ρs)·d·g·|g|
//! ```
//! `A` = nonlocal amplitude, `d` = grain diameter, `μs` = static friction
//! coefficient, `ρs` = grain density, `t0` = a microscopic grain-inertial
//! relaxation timescale, `b` a rate-dependence constant (same convention as
//! `MuIRheologyMaterial`'s own `b = (μ2−μs)/I0`).
//!
//! # Bistable/hysteretic extension (real gap found + fixed, 2026-08-19)
//! Real, measured finding: the plain single-threshold source above (a
//! literal `μ-μs` linear term) cannot arrest a violent column collapse at a
//! real repose angle at ANY grain-diameter calibration tried (both
//! directions swept with a real spatial-homogenization diagnostic — small
//! `d` decays too fast/locally to resist collapse, large `d` homogenizes
//! the WHOLE domain into a runaway never-decaying plateau; see
//! `sand_ngf_tests.rs`'s own real measured history). This independently
//! reconfirms a 2026-08-03 finding (`project_novel_research_frontier_
//! registry` memory) that plain NGF only narrows runout ~3%, never closes
//! the angle-of-repose gap.
//!
//! Real, cited fix: **Mowlavi & Kamrin (2021), "Interplay between
//! hysteresis and nonlocality during onset and arrest of flow in granular
//! materials", Soft Matter 17(31):7359-7375, arXiv:2105.02288** — verified
//! directly against the paper's own eq. 2b/5b/6/8, not recalled from
//! memory. Adds ONE correction term to the SAME reaction slot (the nonlocal
//! diffusion term is untouched):
//! ```text
//! t0 · ∂g/∂t = A²d²·∇²g − F(μ)·g − b·√(P/ρs)·d·μ·g² − χ(g;μ,P)·g
//! F(μ)       = (μ2−μs)(μs−μ) / (μ2−μ)                        [eq. 8]
//! χ(g;μ,P)   = a·[1 − tanh(c·√(P/ρs)·μ·g·κⁿ)]                 [eq. 6]
//! κ          = kn / P                                         (per-cell, real local pressure)
//! ```
//! `F(μ)` generalizes the plain `μ-μs` term (recovers it exactly as
//! `μ2 -> ∞`) with a real upper/dynamic bound `μ2` the source diverges
//! toward, matching rate-strengthening μ(I) behavior. `χ` is the genuinely
//! NEW physics: a bounded (`∈[0,2a]`), always-non-negative rate-WEAKENING
//! correction that makes the local reaction ODE admit THREE roots for `μ`
//! in a real window `(μ*, μs*)` — `g=0` (arrested, always stable), an
//! unstable nonzero root, and a stable flowing root. Which branch the
//! system occupies depends on HISTORY (already flowing vs. already at
//! rest) — that dependence IS the hysteresis. Real, load-bearing detail
//! confirmed directly from the paper (not assumed): this needs NO new
//! per-cell/per-particle state beyond `g` itself — bistability is purely
//! `g`'s own multivalued steady state, so this plugs into the EXISTING
//! grid-only `g` field with no data-layout change.
//!
//! Numerical scheme: `χ`'s own `tanh(...·g...)` dependence on `g` breaks
//! the `u=1/g` linearization the dominant `F(μ)`/quadratic-in-`g` part
//! below still solves exactly (unchanged). Since `χ` is proven bounded and
//! small (`≤2·a≈0.023`, real DEM-calibrated value) relative to `F(μ)`
//! (order `0.1-1` in this engine's real friction-angle range), it's applied
//! as a semi-implicit operator-split correction AFTER the exact dominant
//! solve, dividing (not subtracting) so even a larger-than-assumed `χ`
//! can't overshoot into negative `g` — the same numerical caution this
//! module's own reaction-step history already required once (see the
//! rejected-approaches list in `apply`'s own doc).
//!
//! # Numerical scheme
//! Explicit finite-difference — matches Haeri & Skonieczny's own verified
//! choice, not an invented shortcut. Same P2G→normalize→Laplacian→G2P shape
//! as [`super::scalar_field::ScalarDiffusionField`]/[`super::diffusion::ThermalDiffusion`]
//! (reuses the shared [`super::stencil::laplacian_step`]), plus a real reaction
//! step for the two extra terms above. Real, quoted stability bound from
//! that same paper: `Δt < Δx²·t0 / (2·A²·d²)` — unlike `ScalarDiffusionField`'s
//! bound (only ever documented, never enforced, since thermal diffusivity is
//! tiny relative to MPM's own elastic-wave CFL), this one is plausibly
//! binding and is exposed via [`GranularFluidityField::stability_dt`] for
//! callers to fold into their own adaptive substep choice.
//!
//! # `g` is grid-only state, not a particle field
//! Unlike temperature/pheromone, `g` has no natural per-particle home:
//! `Particle` is fixed at exactly 128 bytes with no spare padding
//! (`src/matter/particle/mod.rs`). This matches the real physics anyway —
//! nonlocal fluidity is fundamentally a spatial/grid quantity, not a
//! per-particle property. `g` therefore persists as this struct's own grid
//! state between calls to [`GranularFluidityField::apply`], and is
//! *gathered* (not stored) into a caller-supplied buffer each substep for
//! use in a yield check.
//!
//! # Deriving pressure and stress ratio (not stored fields either)
//! Pressure `P` and stress ratio `μ` are computed fresh each substep from a
//! particle's own `deformation_gradient`, reusing the exact same
//! Hencky-trace formula `MuIRheologyMaterial::update_particle`
//! (`src/matter/materials/sand_mui.rs`) already uses — real, cited reuse,
//! not a new formula invented here. The caller supplies
//! `pressure_and_ratio: fn(&Particle) -> (f32, f32)` since only the coupled
//! material knows its own elastic Lamé parameters.

use glam::IVec2;

use crate::{grid::kernel::quadratic_weights, particle::Particles};

/// Real, physical parameters for the NGF PDE above.
#[derive(Clone, Copy, Debug)]
pub struct GranularFluidityConfig {
    /// Static friction coefficient μs (dimensionless) — the same real value
    /// as the coupled material's own `tan(friction_angle)`.
    pub mu_s: f32,
    /// Real grain diameter `d` \[m\] — see e.g.
    /// `DruckerPragerMaterial::GRAIN_DIAMETER_M`.
    pub grain_diameter_m: f32,
    /// Grain density ρs \[kg/m³\].
    pub grain_density_kg_m3: f32,
    /// Nonlocal amplitude `A` (dimensionless) — real, cited value 0.48
    /// (Henann & Kamrin 2013 glass beads; independently reconfirmed for
    /// real sand by Haeri & Skonieczny 2022, same value).
    pub nonlocal_amplitude: f32,
    /// Rate-dependence constant `b` (dimensionless), same convention as
    /// `MuIRheologyMaterial`'s own `b = (μ2−μs)/I0`.
    pub b: f32,
    /// Microscopic grain-inertial relaxation timescale `t0` \[s\]. Real,
    /// cited value 1e-4s (Haeri & Skonieczny 2022, Table 1).
    pub t0_s: f32,
    /// Real, physically-motivated minimum pressure \[Pa\] fed to the
    /// reaction term's `1/sqrt(P)` factor. NOT a numerical fudge: the
    /// equation's own analytic equilibrium
    /// (`g_eq = linear_coeff/(b*sqrt(P/rho_s)*d)`) genuinely diverges as
    /// `P -> 0` -- a real cell at ~1e-3 Pa gives a
    /// mathematically-correct-but-physically-absurd `g_eq` in the tens of
    /// millions, even under an exact (unconditionally stable) closed-form
    /// integration -- this is the equation's own real singularity, not a
    /// numerics bug. Real granular material at ANY free surface still has
    /// some minimum confining pressure from its own weight/interlocking
    /// (the same real justification `DruckerPragerMaterial::cohesion`
    /// already documents) -- a natural, physically-derived floor is one
    /// grain's own hydrostatic self-weight: `rho_s * g_accel * d`.
    pub pressure_floor_pa: f32,
    /// Real upper/dynamic friction bound μ₂ (dimensionless, μ₂>μs) for the
    /// bistable/hysteretic source term below (`hysteresis_amplitude` and
    /// friends) — Mowlavi & Kamrin 2021 eq. 8's own generalized `F(μ)`
    /// diverges as `μ -> μ2`, matching real rate-strengthening μ(I)
    /// behavior at its own upper bound, unlike plain NGF's simple
    /// `μ-μs` source this replaces. Real, disclosed choice: NOT the
    /// paper's own DEM-calibrated μ2=0.9784 (idealized frictional disks,
    /// not this engine's real sand friction angle) — instead reuses this
    /// codebase's OWN already-cited static->dynamic offset convention
    /// (`MuIRheologyMaterial::from_physical`: `μ_dynamic =
    /// tan(friction_angle_deg + 12°)`, Jop/Forterre/Pouliquen 2006), kept
    /// self-consistent with `mu_s` above (typically `tan(35°)`).
    pub mu_2: f32,
    /// Real hysteresis amplitude `a` (dimensionless) — Mowlavi & Kamrin
    /// 2021 eq. 6, their own DEM-calibrated value `a=0.0116`. Bounds the
    /// correction term `χ` to `[0, 2a]`, i.e. always SMALL relative to
    /// the dominant `F(μ)` source term and always non-negative (χ can
    /// only add resistance/damp flow, never spontaneously drive it) —
    /// this is what makes the operator-split integration below
    /// (`GranularFluidityField::apply`'s own doc) both safe and accurate.
    pub hysteresis_amplitude: f32,
    /// Real hysteresis rate constant `c` (dimensionless) inside χ's own
    /// `tanh(c·√(P/ρs)·μ·g·κⁿ)` argument — Mowlavi & Kamrin 2021 eq. 6,
    /// their own DEM-calibrated value `c=50`.
    pub hysteresis_rate: f32,
    /// Real stiffness exponent `n` in χ's `κⁿ` factor — Mowlavi & Kamrin
    /// 2021 eq. 6, cite DeGiuli & Wyart for `n=1/4` as a fixed physical
    /// constant (encodes how hysteresis amplitude scales with grain
    /// stiffness), NOT a free fit parameter — kept here as a real,
    /// disclosed config field rather than hardcoded, for testability.
    pub hysteresis_stiffness_exponent: f32,
    /// Real, physically-motivated grain contact stiffness `kn` \[Pa\] —
    /// feeds `κ=kn/P` (dimensionless, per-substep, using the REAL LOCAL
    /// pressure, not a fixed ratio — physically correct: grains are
    /// effectively stiffer relative to load in low-confinement zones).
    /// Real, disclosed continuum-DEM correspondence adaptation (same
    /// category as `cosserat_length_scale_m`'s own precedent): no literal
    /// per-grain contact stiffness exists in an MPM continuum, so this
    /// uses the coupled material's own real elastic shear modulus as the
    /// physically-analogous "resistance to relative grain deformation"
    /// quantity. The paper validates its own stiff-grain-limit regime at
    /// `κ>1e4`; at this engine's real E=15MPa sand (`μ_elastic≈5.8MPa`
    /// via `E/(2(1+ν))`) and typical scene confining pressures
    /// (O(1e2-1e3) Pa), `κ` lands O(1e3-1e4), the same real regime the
    /// paper's own DEM validation covers.
    pub contact_stiffness_pa: f32,
}

impl GranularFluidityConfig {
    /// Real, quoted Von Neumann stability bound for the explicit scheme
    /// above (Haeri & Skonieczny 2022, §4, verified via `pdftotext` against
    /// the real PDF): `Δt < Δx² · t0 / (2·A²·d²)`.
    ///
    /// `dx_m` is the real physical grid cell size (`SimConfig::dx_meters`).
    pub fn stability_dt(&self, dx_m: f32) -> f32 {
        let denom = 2.0 * self.nonlocal_amplitude.powi(2) * self.grain_diameter_m.powi(2);
        dx_m * dx_m * self.t0_s / denom.max(1e-30)
    }
}

/// A persistent, grid-coupled granular fluidity field.
pub struct GranularFluidityField {
    pub config: GranularFluidityConfig,
    /// Reads a particle's current (pressure, stress ratio) pair, in the
    /// same grid-unit stress convention the coupled material's own yield
    /// check already uses.
    pub pressure_and_ratio: fn(&crate::particle::Particle) -> (f32, f32),

    grid_res: usize,
    grid_mass: Vec<f32>, // Σ(w·mass)              — cleared each step
    grid_p: Vec<f32>,    // scattered pressure       — cleared each step
    grid_mu: Vec<f32>,   // scattered stress ratio   — cleared each step
    grid_g: Vec<f32>,    // persistent PDE state — NOT cleared between steps
    grid_work: Vec<f32>, // scratch: diffusion/reaction output before it replaces grid_g
}

impl GranularFluidityField {
    pub fn new(
        config: GranularFluidityConfig,
        pressure_and_ratio: fn(&crate::particle::Particle) -> (f32, f32),
        grid_res: usize,
    ) -> Self {
        let n = grid_res * grid_res;
        Self {
            config,
            pressure_and_ratio,
            grid_res,
            grid_mass: vec![0.0; n],
            grid_p: vec![0.0; n],
            grid_mu: vec![0.0; n],
            grid_g: vec![0.0; n],
            grid_work: vec![0.0; n],
        }
    }

    /// Advance `g` by one substep and gather it into `out[pi]` for every
    /// particle in `particles`. Real per-substep sequence: P2G-scatter
    /// pressure/stress-ratio → normalize → diffuse (shared stencil) →
    /// react (the two extra NGF terms, explicit Euler) → G2P-gather `g`.
    ///
    /// `sub_dt` must respect [`GranularFluidityConfig::stability_dt`] —
    /// callers are responsible for folding that bound into their own
    /// adaptive substep choice (this field does not clamp `sub_dt` itself,
    /// matching how `ScalarDiffusionField`'s bound is also caller-enforced).
    ///
    /// `dx_meters`: real physical grid cell size (`SimConfig::dx_meters`).
    /// [`super::stencil::laplacian_step`]'s 4-neighbor-minus-center sum is a
    /// bare grid-INDEX-space finite difference (no implicit cell size) — it
    /// must be scaled by `1/dx_meters²` to become the real `∇²g` the PDE
    /// actually calls for, exactly the convention `ThermalConfig::alpha_grid`
    /// already documents ("Folding dx² in keeps the Laplacian formula
    /// dimensionless over grid indices") and `Cosserat` field's own `apply`
    /// call site already threads through. A real, previously-missing
    /// dx-normalization: without it, the diffusion term's magnitude doesn't
    /// depend on the real cell size at all, so refining the grid (same `A`,
    /// `d`, `t0`) changes how many REAL METERS the same "diffusivity_dt"
    /// spreads `g` per step — the direct cause of this module's own
    /// resolution-dependence bug (see `sand.rs`'s `ngf_lajeunesse_runout_
    /// resolution_independence` test history).
    pub fn apply(&mut self, particles: &Particles, sub_dt: f32, dx_meters: f32, out: &mut [f32]) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;

        for i in 0..n {
            self.grid_p[i] = 0.0;
            self.grid_mu[i] = 0.0;
            self.grid_mass[i] = 0.0;
        }

        // --- P2G: scatter mass-weighted (P, μ) into grid_p / grid_mu ---
        for pi in 0..particles.len() {
            let p = particles.get(pi);
            let (pressure, mu) = (self.pressure_and_ratio)(&p);
            let w = quadratic_weights(p.x);
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    let mw = weight * p.mass;
                    self.grid_p[idx] += mw * pressure;
                    self.grid_mu[idx] += mw * mu;
                    self.grid_mass[idx] += mw;
                }
            }
        }

        // --- Normalize: empty cells keep their last real g at rest (no
        // flow information there to update from) ---
        for i in 0..n {
            if self.grid_mass[i] > 1e-10 {
                self.grid_p[i] /= self.grid_mass[i];
                self.grid_mu[i] /= self.grid_mass[i];
            } else {
                self.grid_p[i] = 0.0;
                self.grid_mu[i] = self.config.mu_s;
            }
        }

        // --- Diffuse: shared explicit-Euler 5-point Laplacian, same
        // stencil ScalarDiffusionField/ThermalDiffusion already use ---
        // dx_meters^2 division: see this method's own doc.
        let dx2 = (dx_meters * dx_meters).max(1e-30);
        let diffusivity_dt = (self.config.nonlocal_amplitude * self.config.grain_diameter_m)
            .powi(2)
            / (self.config.t0_s * dx2)
            * sub_dt;
        super::stencil::laplacian_step(
            &self.grid_g,
            &mut self.grid_work,
            self.grid_res,
            diffusivity_dt,
            0.0,
        );

        // --- React: the two extra NGF terms -- EXACT closed-form
        // integration, not explicit Euler (two simpler approaches were
        // tried and rejected -- see below). ---
        // t0*dg/dt = (mu-mu_s)*g - b*sqrt(P/rho_s)*d*g*|g|   (already
        // diffused this step; this is the remaining reaction contribution).
        //
        // Two simpler approaches were rejected:
        // 1. Seeding `g` directly to this ODE's own analytic equilibrium
        //    (g_eq = linear_coeff/(b*sqrt(P/rho_s)*d), Henann & Kamrin 2014
        //    eq. 4's "g_loc") the first time a cell crosses `mu_s`: that
        //    equilibrium itself DIVERGES as P->0 (division by sqrt(P)) --
        //    exactly the real, low-confinement regime this whole mechanism
        //    targets, not an edge case. A cell at real pressure ~1e-3 Pa
        //    produces a seed of ~16 MILLION.
        // 2. Seeding a small epsilon instead, then advancing via EXPLICIT
        //    EULER each substep: correct in principle, but this ODE's own
        //    linear growth rate (`linear_coeff/t0`) is fast relative to a
        //    real substep dt at the literal cited `t0=1e-4s` -- explicit
        //    Euler either understates growth badly (material re-freezes
        //    elastically before `g` can rise fast enough, flinging
        //    particles to a boundary clamp) or, at a smaller `t0`, wildly
        //    OVERSHOOTS (`g` reaching 3.5e16 -- at which point the
        //    rate-limiter's `.min()` against the unlimited self-consistent
        //    gamma always picks one side or the other, silently defeating
        //    the entire coupling).
        //
        // Instead: restricted to g>=0 (the physical domain), this ODE is
        // EXACTLY the logistic equation `dg/dt = r*g*(1-g/g_eq)` with
        // `r=linear_coeff/t0`, `g_eq=r/c`, `c=b*sqrt(P/rho_s)*d/t0` -- which
        // has a real, standard closed-form solution via the substitution
        // `u=1/g` (turns the nonlinear ODE into the LINEAR one `du/dt =
        // -r*u + c`, solved exactly): `u(t)=c/r+(u0-c/r)*exp(-r*t)`. This
        // is UNCONDITIONALLY STABLE -- correct for any `dt`, any `t0`, by
        // construction (it's the exact solution, not a finite-difference
        // approximation), so no further ad-hoc parameter recalibration of
        // `t0` is needed to avoid either failure mode above.
        let cfg = self.config;
        const BOOTSTRAP_SEED: f32 = 1.0e-6; // still needed: u=1/g is singular at g=0
        for i in 0..n {
            let g_prev = self.grid_work[i];
            let mu = self.grid_mu[i];
            let pressure = self.grid_p[i].max(cfg.pressure_floor_pa);

            // Real two-threshold F(mu) (Mowlavi & Kamrin 2021 eq. 8, see
            // this module's own doc) generalizing plain NGF's `mu-mu_s`:
            // clamp mu below mu_2 since it's a real asymptote the source
            // diverges toward, never an attainable value.
            let mu_clamped = mu.min(cfg.mu_2 - 1.0e-4);
            let linear_coeff = (cfg.mu_2 - cfg.mu_s) * (mu_clamped - cfg.mu_s)
                / (cfg.mu_2 - mu_clamped).max(1.0e-6);

            // g=0 is a REAL, STABLE fixed point whenever linear_coeff<=0 --
            // leave it at exactly 0 (matches real physics: nothing to grow
            // from without a source). Only bootstrap the epsilon seed (the
            // `u=1/g` substitution below is singular at g=0) when there IS
            // a real source to grow toward. Unaffected by the hysteresis
            // correction below: chi always multiplies g, so it cannot move
            // g away from an exact 0 fixed point either.
            if linear_coeff <= 0.0 && g_prev <= 0.0 {
                self.grid_work[i] = 0.0;
                continue;
            }
            let g = g_prev.max(BOOTSTRAP_SEED);

            let t0 = cfg.t0_s.max(1e-12);
            let r = linear_coeff / t0;
            let c = cfg.b
                * (pressure / cfg.grain_density_kg_m3.max(1e-6)).sqrt()
                * cfg.grain_diameter_m
                / t0;

            let u0 = 1.0 / g;
            let u_new = if r.abs() > 1e-9 {
                c / r + (u0 - c / r) * (-r * sub_dt).exp()
            } else {
                u0 + c * sub_dt // r=0 special case: du/dt=c exactly, linear in t
            };
            let g_dominant = if u_new > 1e-12 { 1.0 / u_new } else { 0.0 };

            // Real hysteretic correction chi (eq. 6, see this module's own
            // doc): bounded [0,2a] and always non-negative, applied as a
            // semi-implicit (dividing) operator-split correction on the
            // already-solved g_dominant -- see this module's own doc for
            // why this is safe even though chi's own g-dependence breaks
            // the exact closed-form solve above.
            let kappa = (cfg.contact_stiffness_pa / pressure.max(1.0e-6)).max(0.0);
            let chi = cfg.hysteresis_amplitude
                * (1.0
                    - (cfg.hysteresis_rate
                        * (pressure / cfg.grain_density_kg_m3.max(1e-6)).sqrt()
                        * mu_clamped
                        * g_dominant
                        * kappa.powf(cfg.hysteresis_stiffness_exponent))
                    .tanh());
            let g_new = g_dominant / (1.0 + chi * sub_dt / t0);

            self.grid_work[i] = g_new.max(0.0);
        }

        self.grid_g.copy_from_slice(&self.grid_work);

        // --- G2P: gather g back to particles (transient — not stored) ---
        for pi in 0..particles.len().min(out.len()) {
            out[pi] = self.gather_g_at(particles.get(pi).x);
        }
    }

    /// Real B-spline gather of the CURRENT grid `g` at one world position --
    /// the same interpolation `apply`'s own G2P step uses, factored out so
    /// diagnostics can sample `g` exactly where they actually care about it
    /// instead of only ever seeing whole-grid aggregates (see
    /// `g_stats_near`'s own doc for why that distinction turned out to
    /// matter in practice).
    fn gather_g_at(&self, x: glam::Vec2) -> f32 {
        let res = self.grid_res as i32;
        let w = quadratic_weights(x);
        let mut g_sum = 0.0f32;
        let mut w_sum = 0.0f32;
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let weight = w.wx[gx as usize] * w.wy[gy as usize];
                let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                    continue;
                }
                let idx = (cell.x * res + cell.y) as usize;
                g_sum += weight * self.grid_g[idx];
                w_sum += weight;
            }
        }
        if w_sum > 1e-10 { g_sum / w_sum } else { 0.0 }
    }

    pub const fn grid_res(&self) -> usize {
        self.grid_res
    }

    /// Real, permanent diagnostic (2026-08-19): (min, mean, max) of `g`
    /// sampled AT the given particles' own positions, not the whole grid.
    /// Added after `g_stats()`'s whole-grid max stayed pinned at an
    /// identical value for 55 real seconds across a real column collapse,
    /// even once the pile had visibly stopped moving -- a real, flagged
    /// suspicion that the whole-grid max was dominated by the pressure-
    /// floor equilibrium value in near-empty/far-from-the-pile cells, not
    /// tracking the pile's own dynamics at all. This answers that directly:
    /// if `g` at the pile's own particles is ALSO pinned at that same
    /// value, the whole-grid number was real, not a confound; if it's
    /// different, `g_stats()` alone was measuring the wrong thing for this
    /// question.
    pub fn g_stats_near(&self, particles: &Particles) -> (f32, f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = 0.0f32;
        let mut sum = 0.0f32;
        let n = particles.len();
        for i in 0..n {
            let g = self.gather_g_at(particles.get(i).x);
            min = min.min(g);
            max = max.max(g);
            sum += g;
        }
        let mean = if n > 0 { sum / n as f32 } else { 0.0 };
        (if n > 0 { min } else { 0.0 }, mean, max)
    }

    /// Real, permanent diagnostic: (min, mean-over-nonzero, max, count-nonzero)
    /// of the current `g` field. Added 2026-08-04 chasing the real, measured
    /// 0.47x Lajeunesse undershoot -- lets a caller check DIRECTLY whether
    /// `g` is staying anomalously small/narrow throughout a real collapse
    /// (the "cooperation too slow/narrow relative to the moving flow front"
    /// hypothesis) rather than reasoning about it in the abstract.
    pub fn g_stats(&self) -> (f32, f32, f32, usize) {
        let mut min = f32::INFINITY;
        let mut max = 0.0f32;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for &g in &self.grid_g {
            if g > 1e-9 {
                min = min.min(g);
                max = max.max(g);
                sum += g;
                count += 1;
            }
        }
        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        (if count > 0 { min } else { 0.0 }, mean, max, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;
    use glam::Vec2;

    fn test_config() -> GranularFluidityConfig {
        // Real values from Haeri & Skonieczny 2022 Table 1 (Excavation case).
        // Hysteresis term disabled (hysteresis_amplitude=0 -> chi=0
        // identically, mu_2 large enough F(mu)->mu-mu_s exactly, see this
        // module's own doc) -- these existing tests check the plain,
        // pre-bistable source term's own closed-form math directly.
        GranularFluidityConfig {
            mu_s: 0.70,
            grain_diameter_m: 0.3e-3,
            grain_density_kg_m3: 2583.0,
            nonlocal_amplitude: 0.48,
            b: 0.278,
            t0_s: 1.0e-4,
            pressure_floor_pa: 0.0, // these unit tests use large, non-degenerate pressures directly
            mu_2: 1.0e6,
            hysteresis_amplitude: 0.0,
            hysteresis_rate: 50.0,
            hysteresis_stiffness_exponent: 0.25,
            contact_stiffness_pa: 1.0e7,
        }
    }

    fn test_particle_at(pos: Vec2) -> Particle {
        let mut p = Particle::zeroed();
        p.x = pos;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p
    }

    #[test]
    fn stability_dt_matches_the_cited_formula_directly() {
        let cfg = test_config();
        let dx = 0.01f32;
        let expected = dx * dx * cfg.t0_s
            / (2.0 * cfg.nonlocal_amplitude.powi(2) * cfg.grain_diameter_m.powi(2));
        assert!((cfg.stability_dt(dx) - expected).abs() < 1e-12);
        assert!(cfg.stability_dt(dx) > 0.0);
    }

    #[test]
    fn g_stays_zero_with_no_particles() {
        let cfg = test_config();
        let mut field = GranularFluidityField::new(cfg, |_p| (0.0, 0.70), 16);
        let particles = Particles::from(vec![]);
        let mut out = vec![0.0; 0];
        field.apply(&particles, 1.0e-6, 0.01, &mut out);
        assert!(field.grid_g.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn g_stays_zero_when_stress_ratio_never_exceeds_static_friction() {
        // At mu == mu_s everywhere, the linear relaxation term is exactly
        // zero and g has no source to grow from a zero start -- a real,
        // checkable invariant of the equation itself, not just "it runs."
        let cfg = test_config();
        let mut field = GranularFluidityField::new(cfg, |_p| (1000.0, 0.70), 8);
        let particles = Particles::from(vec![
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
        ]);
        let mut out = vec![0.0; 4];
        for _ in 0..50 {
            field.apply(&particles, 1.0e-6, 0.01, &mut out);
        }
        assert!(
            field.grid_g.iter().all(|&g| g.abs() < 1e-9),
            "g should stay ~0 when mu==mu_s everywhere"
        );
    }

    #[test]
    fn g_bootstraps_away_from_zero_once_mu_exceeds_mu_s() {
        // mu=0.90 > mu_s=0.70: g=0 is a real but UNSTABLE fixed point here --
        // without the seeding fix this stays at exactly 0.0 forever under
        // forward-Euler. Confirm it actually grows, and converges toward
        // this reaction ODE's own real equilibrium
        // g_eq = (mu-mu_s) / (b*sqrt(P/rho_s)*d).
        let cfg = test_config();
        let pressure = 1.0e5f32;
        let mu = 0.90f32;
        let mut field = GranularFluidityField::new(cfg, |_p| (1.0e5, 0.90), 8);
        let particles = Particles::from(vec![
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
        ]);
        // Growing from a small epsilon seed (real fix -- see `apply`'s own
        // doc for why a direct jump to g_eq is wrong) takes real elapsed
        // time: dg/dt ~ linear_coeff*g/t0 near g~0 is exponential growth,
        // so t_converge ~ t0/linear_coeff * ln(g_eq/seed). Here that's
        // ~1e-4/0.2 * ln(385/1e-6) ~ 9.9ms -- run comfortably past that
        // (30ms), well inside the real stability bound at this dx/t0/A/d
        // (~0.24s, `stability_dt_matches_the_cited_formula_directly`).
        let mut out = vec![0.0; 4];
        for _ in 0..3000 {
            field.apply(&particles, 1.0e-5, 0.01, &mut out);
        }
        let g_eq = (mu - cfg.mu_s)
            / (cfg.b * (pressure / cfg.grain_density_kg_m3).sqrt() * cfg.grain_diameter_m);
        assert!(out[0] > 0.0, "g should have bootstrapped away from zero");
        let relative_error = (out[0] - g_eq).abs() / g_eq;
        assert!(
            relative_error < 0.2,
            "g={} should have converged near g_eq={g_eq} (rel. error {relative_error})",
            out[0]
        );
    }

    /// Real, decisive test of the bistable/hysteretic extension (Mowlavi &
    /// Kamrin 2021, see this module's own doc): at a SINGLE fixed stress
    /// ratio mu just above mu_s (0.705, inside the real bistable window
    /// this config's own mu_2/hysteresis_amplitude implies -- see the
    /// worked derivation in this test's own commit), the SAME physics must
    /// produce TWO different long-run outcomes depending purely on
    /// history: starting from ~0 (material at rest) it must STAY arrested
    /// (chi's extra resistance near g=0 wins over the weak plain driving
    /// force); starting from an already-large g (material already
    /// flowing) it must converge to a real, macroscopic nonzero steady
    /// value (chi fades at large g, plain rate-strengthening dominates).
    /// This is the literal, checkable definition of the hysteresis this
    /// extension exists to provide -- not just "chi is nonzero somewhere."
    #[test]
    fn bistable_hysteresis_same_mu_two_stable_branches_by_history() {
        let mu_2 = (35.0f32 + 12.0).to_radians().tan(); // same real +12deg offset as sand_ngf_tests.rs
        let cfg = GranularFluidityConfig {
            mu_2,
            hysteresis_amplitude: 0.0116, // real DEM-calibrated a
            hysteresis_rate: 50.0,        // real DEM-calibrated c
            hysteresis_stiffness_exponent: 0.25,
            contact_stiffness_pa: 1.0e7, // stiff-grain regime, same order as real sand E
            ..test_config()
        };
        // pressure=1e5, mu=0.705 baked in as literal constants -- `new`
        // requires a bare, non-capturing fn pointer (same real constraint
        // this file's other tests already work around the same way).
        fn run(cfg: GranularFluidityConfig, seed_g: f32) -> f32 {
            let mut field = GranularFluidityField::new(cfg, |_p| (1.0e5, 0.705), 8);
            field.grid_g.fill(seed_g); // real, direct history injection -- this IS the point of the test
            let particles = Particles::from(vec![
                test_particle_at(Vec2::new(4.0, 4.0)),
                test_particle_at(Vec2::new(4.0, 4.0)),
                test_particle_at(Vec2::new(4.0, 4.0)),
                test_particle_at(Vec2::new(4.0, 4.0)),
            ]);
            let mut out = vec![0.0; 4];
            for _ in 0..5000 {
                field.apply(&particles, 1.0e-5, 0.01, &mut out);
            }
            out[0]
        }

        let g_from_rest = run(cfg, 0.0);
        let g_from_flowing = run(cfg, 5.0);

        assert!(
            g_from_rest < 0.05,
            "starting at rest, g should stay arrested (chi's near-g=0 resistance should win) -- got {g_from_rest}"
        );
        assert!(
            g_from_flowing > 1.0,
            "starting already-flowing, g should settle on a real nonzero branch (chi fades at large g) -- got {g_from_flowing}"
        );
    }
}
