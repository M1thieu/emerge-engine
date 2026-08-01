//! A genuine 1D discrete elastic rod (Cosserat-rod family) — a real,
//! dimensionally-reduced continuum solver for slender (length >> width)
//! bodies, sibling to `spacetime::diff` (a second, narrower, self-contained
//! solver living under `spacetime/`, not shoehorned into `solver/`).
//!
//! # Why this exists
//! Real engineering practice does not use full volumetric FEM/MPM for a
//! fishing rod, a cable, or a blade of grass — it uses beam/rod theory, a
//! rigorous 1D reduction of the SAME continuum elasticity MPM's own 2D
//! materials already are relative to full 3D. Pure 2D volumetric MPM shows
//! genuine self-weight Euler/Greenhill buckling for a slender cantilever —
//! pushing a thin volumetric blade taller destabilizes it. `EI` (bending
//! stiffness) is an emergent property of carved cross-section width in 2D
//! MPM; here it is a direct, exact input parameter — the real advantage of
//! the right dimensional reduction for this class of body.
//!
//! # Real physics: discrete elastic rods, specialized to 2D
//! Bergou, Wardetzky, Robinson, Audoly, Grinspun 2008, SIGGRAPH, "Discrete
//! Elastic Rods" — the modern discrete form of classical Cosserat rod theory
//! (Cosserat brothers, 1909). A rod is `N` control points along a centerline;
//! `N-1` edges carry axial (stretch) elastic energy; `N-2` interior vertices
//! carry bending elastic energy from the discrete curvature between adjacent
//! edges. In 3D the curvature is a vector (the discrete binormal) with a
//! separate twist DOF about the centerline; in 2D the binormal direction is
//! fixed to the plane's own normal, so curvature collapses to a signed
//! scalar and twist has no real DOF at all (2D genuinely has one fewer
//! curvature dimension than 3D — a real dimensional fact, not a cut corner).
//! See `forces.rs` for the actual formulas.
//!
//! # Coupling to the shared MPM grid
//! `Grid` (`spacetime::grid`) is a fully source-agnostic mass/momentum
//! accumulator — every mutator is keyed purely by `IVec2` cell position,
//! nothing in it references `Particle`/`Particles`. `coupling.rs` scatters
//! rod points into that SAME grid using the identical quadratic-B-spline
//! weights `transfer::p2g`/`transfer::g2p` already use, so a rod and ordinary
//! MPM particles (fluid, sand, fire) genuinely exchange momentum through one
//! shared mechanism — not an isolated parallel system. See `Simulation::rods`
//! and its `do_substep` insertion points for the real coupling (Phase 2).
//!
//! `RodPoints` is a new, independent SoA — NOT grafted onto `Particle`
//! (which is `repr(C)`/`Pod`/128-byte GPU-layout-locked, append-only after a
//! real past corruption bug) or `Particles`. Same "independent store,
//! cross-talk only through the shared `Grid`" shape `Particles` itself
//! already is relative to `Grid`.
//!
//! # Scope (explicit, disclosed)
//! CPU-first (matches `ARCHITECTURE.md` §5's own stated rule — GPU port is
//! real future work, not attempted here: WGPU bind-group layouts are already
//! at the 4-group WebGPU baseline limit). 2D only. True branching topology
//! exists via `network::RodNetwork` (a real graph, not a single chain); a
//! plain `Rod` itself stays a single unbranched chain. No twist DOF (none
//! exists in 2D). `SimSnapshot::rods` covers per-rod count/sleeping/speed/tip
//! aggregates; `RodNetwork` isn't wired into that aggregate yet.

pub mod coupling;
pub mod forces;
pub mod gravitropism;
pub mod growth;
pub mod implicit;
pub mod integrator;
pub mod network;
pub mod plasticity;
pub mod secondary_growth;

use glam::Vec2;

pub use coupling::{
    RodForceParams, apply_rod_internal_and_wind_forces, gather_grid_to_rod, scatter_rod_to_grid,
};
pub use forces::{
    RodRestState, compute_internal_forces, discrete_curvature, discrete_curvature_gradient,
};
pub use gravitropism::{
    Gravitropism, GravitropismMode, Phototropism, apply_gravitropism, apply_phototropism,
};
pub use growth::{Growth, GrowthResistance, apply_growth};
pub use implicit::{RodImplicitStepParams, step_rod_implicit};
pub use integrator::{apply_mass_scaling_for_target_dt, rod_cfl_dt, step_rod};
pub use network::{
    NetworkBendingVertex, NetworkEdge, RodNetwork, YBranchSpec, build_y_branch,
    compute_network_internal_forces, network_cfl_dt, step_network,
};
pub use plasticity::{RodPlasticity, apply_bending_plasticity};
pub use secondary_growth::{SecondaryGrowth, apply_secondary_growth};

/// A single rod's centerline state — own SoA, independent of `Particles`.
/// `x`/`v` are in grid-cell units (same convention as `Particle::x`/`v`);
/// `mass` is real, unscaled kilograms (matches `Particle::mass`'s own
/// convention, confirmed via `Elastic::particle_mass` returning real kg).
#[derive(Debug, Clone)]
pub struct RodPoints {
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    /// Real kilograms per point.
    pub mass: Vec<f32>,
    /// Dirichlet anchor — identical semantics to `Particle::pinned`: G2P/the
    /// rod's own gather forces `v=0` for a pinned point instead of gathering,
    /// and its position is left completely untouched (not re-clamped to
    /// itself, avoiding float drift), while it still scatters mass/momentum
    /// normally so other bodies push against a real immovable anchor.
    pub pinned: Vec<u32>,
    /// Rest length of edge i (between points i, i+1). Length N-1. Meters.
    pub rest_edge_length: Vec<f32>,
    /// Rest discrete curvature at interior vertex i (points i, i+1, i+2).
    /// Length N-2. DIMENSIONLESS (collapses to the turning angle for small
    /// bends — see `forces::discrete_curvature`'s own doc) — 0.0 for a
    /// straight rod. NOT 1/meters; the per-length normalization lives in the
    /// bending-force formula's own Voronoi-length division, not here.
    pub rest_curvature: Vec<f32>,
    /// Per-edge axial stiffness `E*A`, Newtons. Length N-1. Real prior art:
    /// `network::NetworkEdge::ea` already does this for a branching
    /// `RodNetwork` (different branches are genuinely different
    /// thicknesses) — this ports the same pattern to a plain single-chain
    /// `Rod`, e.g. a stem stiffer at its base than its growing tip.
    /// Uninitialized (empty) when returned by `build_straight_rod` — filled
    /// with `RodMaterial::ea` at every index by `Rod::new`, so a normal
    /// construction is bit-identical to the prior uniform-material behavior;
    /// only a caller that explicitly overwrites entries after construction
    /// gets real non-uniform stiffness.
    pub ea: Vec<f32>,
    /// Per-vertex bending stiffness `E*I`, N·m². Length N-2. Same fill
    /// convention as `ea` above (filled from `RodMaterial::ei` by
    /// `Rod::new`).
    pub ei: Vec<f32>,
    /// Kahan (compensated) summation residual for `x`'s position integration
    /// in `coupling::gather_grid_to_rod`. Needed because a rod's own
    /// CFL-bound `dt` is extremely small (~1e-6s, set by its axial stiffness)
    /// while `x` sits at an ordinary grid-coordinate magnitude (e.g. 32.0,
    /// offset from the domain origin). Each individual `x[i] += v*dt`
    /// increment (velocity*dt ~ 1e-8 to 1e-9 at typical wind-driven speeds)
    /// falls BELOW f32's representable precision at that magnitude (ULP at
    /// 32.0 is ~3.8e-6) — naive accumulation silently rounds every substep's
    /// contribution away to nothing even though the underlying velocity is
    /// sustained and correct. Kahan summation (Kahan 1965, standard
    /// floating-point technique) fixes this by tracking the rounding error
    /// each addition drops and folding it back in next time, without needing
    /// f64 storage.
    pub position_compensation: Vec<Vec2>,
    /// Real multi-field frictional contact opt-in — identical semantics to
    /// `Particle::contact_group` (Bardenhagen 2001 + Nairn, Hammerquist,
    /// Smith 2020 normal fit): 0 = ordinary (sticks to whatever it touches,
    /// the MPM default), nonzero = a genuine slip/stick interface against
    /// everything else, resolved via `SimConfig::contact_friction`. Real
    /// measured root-soil friction coefficients (McKenzie et al. 2013,
    /// *Plant, Cell & Environment*) span ~0.02-0.31 depending on surface —
    /// set `contact_friction` to a value in that real range for a root
    /// scene, rather than relying on default MPM stick contact.
    pub contact_group: Vec<u32>,
    /// Real accumulated PLASTIC curvature magnitude at interior vertex i
    /// (see `plasticity` module doc) -- length N-2, same shape as
    /// `rest_curvature`. Monotonically non-decreasing: every time
    /// `plasticity::apply_bending_plasticity` absorbs an elastic excess into
    /// `rest_curvature`, that excess's magnitude adds here too. Deliberately
    /// SEPARATE from `rest_curvature` itself -- `rest_curvature` is also
    /// driven by gravitropism/growth for entirely non-mechanical (biological)
    /// reasons, so it alone can't distinguish "reshaped by active growth"
    /// from "permanently deformed by overload"; this field tracks only the
    /// latter. Real prior art: exactly the role `Particle::friction_
    /// hardening` plays for `VonMisesMaterial`'s own isotropic hardening
    /// (`sigma_y(kappa) = yield_stress + H*kappa`) -- hardening state lives
    /// on the thing being deformed, not on the material/law describing how.
    /// Always 0.0 and inert when no `Rod::plasticity` is attached.
    pub accumulated_plastic_curvature: Vec<f32>,
    /// Real per-edge linear mass density, kg/m. Length N-1. Authoritative
    /// source of truth for "how much has this edge's cross-section actually
    /// thickened" -- `secondary_growth::apply_secondary_growth` grows this
    /// in lockstep with `ea`'s own real fractional growth (`EA=E*A`, `E`
    /// held constant, so `d(area)/area = d(ea)/ea` exactly; mass ∝ area at
    /// fixed length/material density, same real derivation, no separate
    /// invented mechanism). `insert_tip_point` (`growth.rs`) reads this
    /// directly instead of re-deriving density from a possibly-already-
    /// non-uniform lumped point mass. Uniform-fill convention matches `ea`/
    /// `ei` -- `build_straight_rod` fills every edge with the same value;
    /// only secondary growth (or a caller building non-uniform density on
    /// purpose) makes it diverge per edge.
    pub linear_density_kg_per_m: Vec<f32>,
}

impl RodPoints {
    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Real SI-unit rod material parameters — `EA`/`EI` use the SAME `E`
/// (Young's modulus) and `I` (second moment of area, `I ~ width^3` for a
/// rectangular section) used in the Greenhill self-buckling analysis
/// (`h_crit = (7.8373*EI/(linear_density*g))^(1/3)`) — direct continuity
/// with that formula, not a new concept.
#[derive(Debug, Clone, Copy)]
pub struct RodMaterial {
    /// Axial (stretch) stiffness `E*A`, Newtons.
    pub ea: f32,
    /// Bending stiffness `E*I`, N·m².
    pub ei: f32,
    /// Kelvin-Voigt axial dashpot (mirrors `ViscoelasticMaterial`'s own
    /// `viscosity` field, 1D-projected onto each edge), N·s/m.
    pub axial_damping: f32,
    /// Rayleigh bending dissipation coefficient, N·m·s. Real, standard
    /// generalized-force construction (Rayleigh 1873) — disclosed as this
    /// plan's own composition of two separately-citable classical-mechanics
    /// results (Kelvin-Voigt + Rayleigh dissipation); Bergou et al. 2008
    /// itself does not define damping at all.
    pub bending_damping: f32,
}

impl RodMaterial {
    pub fn new(ea: f32, ei: f32, axial_damping: f32, bending_damping: f32) -> Self {
        Self {
            ea,
            ei,
            axial_damping,
            bending_damping,
        }
    }

    /// Rectangular cross-section convenience: `A = width*thickness`,
    /// `I = width^3*thickness/12` (bending about the axis perpendicular to
    /// the simulation's own 2D plane — the same `I ~ width^3` relationship
    /// as Wikipedia's "Self-buckling" reference). `thickness_m` is the
    /// engine's own implicit out-of-plane depth (same convention
    /// `Elastic::particle_mass`'s areal
    /// density already assumes) — pass `1.0` unless modeling a real
    /// non-unit depth.
    pub fn from_young_modulus_rectangular(
        young_modulus_pa: f32,
        width_m: f32,
        thickness_m: f32,
        axial_damping: f32,
        bending_damping: f32,
    ) -> Self {
        let area = width_m * thickness_m;
        let i = width_m.powi(3) * thickness_m / 12.0;
        Self::new(
            young_modulus_pa * area,
            young_modulus_pa * i,
            axial_damping,
            bending_damping,
        )
    }

    /// Per-point, LOCAL critical damping `(axial_damping, bending_damping)`
    /// for a rod discretized with uniform segment length `l0_m` and
    /// per-point mass `point_mass_kg`.
    ///
    /// **Honest scope correction (2026-07-26, real bug found via user-
    /// reported "never settles straight")**: this is a LOCAL, single-
    /// segment reference (one point's mass against one segment's own
    /// stiffness) -- it is NOT the true GLOBAL modal critical damping for a
    /// whole rod's actual fundamental bending shape (many points moving
    /// together). Confirmed by direct measurement: for a 20-point cantilever
    /// blade, this function's own "critical" value understates the real
    /// modal critical damping by roughly two to three orders of magnitude --
    /// a rod damped at even 30x THIS function's output still had not
    /// settled after 30 real seconds. Use `modal_critical_damping` below for
    /// "make this rod settle naturally, like a real damped cantilever, in a
    /// physically sensible time" -- the real, common use case (grass
    /// blades, pushable branches, anything a player interacts with). This
    /// function's real, narrower, still-valid purpose is a per-segment
    /// numerical reference (e.g. bounding a single segment's own worst-case
    /// local stiffness/mass ratio), not a substitute for the true modal
    /// value.
    ///
    /// The two outputs are different kinds of quantities —
    /// `axial_damping` [N·s/m] is a real translational dashpot, while
    /// `bending_damping` [N·m·s] is conjugate to the dimensionless discrete
    /// curvature (see `forces::discrete_curvature`) — so naive `c=2*sqrt(k*m)`
    /// with the same translational stiffness is dimensionally wrong for the
    /// bending term. `bending_damping`'s generalized stiffness is `EI/l0`
    /// [N·m] (matching `compute_internal_forces`'s own `coeff`), its
    /// generalized mass `point_mass*l0²` [kg·m²] (from equating kinetic
    /// energies given curvature's own `|d(kappa)/dx| ~ O(1/l0)` gradient) —
    /// `c_crit=2*sqrt(k_gen*m_gen)` then comes out in the correct N·m·s.
    pub fn critical_damping(l0_m: f32, point_mass_kg: f32, ea: f32, ei: f32) -> (f32, f32) {
        let l0_m = l0_m.max(1.0e-9);
        let k_axial = ea / l0_m; // N/m, real translational edge stiffness
        let axial_damping = 2.0 * (k_axial * point_mass_kg).sqrt();

        let k_bend_generalized = ei / l0_m; // N*m, curvature-space stiffness
        let m_bend_generalized = point_mass_kg * l0_m * l0_m; // kg*m^2
        let bending_damping = 2.0 * (k_bend_generalized * m_bend_generalized).sqrt();

        (axial_damping, bending_damping)
    }

    /// Real, GLOBAL modal critical damping for a fixed-free (cantilever)
    /// rod's actual fundamental modes -- root-cause fix for
    /// `critical_damping`'s own disclosed local-reference gap above.
    ///
    /// # Real physics: bending
    /// Uses the SAME real, independently-confirmed fundamental cantilever
    /// eigenvalue `energy::acoustics::modal` uses (`beta_1*L = 1.8751`,
    /// verified via web search against Blevins 1979 / Rao's *Mechanical
    /// Vibrations*, not recalled from memory alone), and the rod's own real
    /// length/mass (`rest_edge_length`/`mass` sums) -- NOT a new,
    /// independently-invented constant. The mode shape itself is derived
    /// directly from the clamped-root/free-tip boundary value problem
    /// (`phi(0)=phi'(0)=0`, `phi''(L)=phi'''(L)=0`), giving the standard
    /// closed form `phi(x) = (cos(bx)-cosh(bx)) + sigma*(sinh(bx)-sin(bx))`,
    /// `sigma = (sinh(bL)-sin(bL))/(cosh(bL)+cos(bL))` -- re-derived here
    /// from the boundary conditions directly (not copied from an uncertain
    /// memory of a textbook constant), independently checked against the
    /// real characteristic equation `cos(bL)*cosh(bL) = -1`
    /// (cos(1.8751)*cosh(1.8751) ≈ -1.0009, confirms the eigenvalue). The
    /// true modal mass `integral(mu*phi(x)^2 dx)/phi(L)^2` is computed by
    /// real numerical integration (2000-sample composite trapezoidal rule)
    /// over the rod's own uniform-mass assumption (same disclosed
    /// simplification `energy::acoustics::modal` already makes) -- not a
    /// memorized modal-mass constant, so there is no new unverified number
    /// here, only re-derived real physics plus real numerical integration.
    /// `c_crit_bending = 2 * m_modal * omega_1`.
    ///
    /// # Real physics: axial
    /// Fixed-free rod longitudinal vibration has an exact elementary
    /// solution (no numerical integration needed): mode shape `sin(pi x /
    /// 2L)`, modal mass exactly `mu*L/2` (elementary integral of
    /// `sin^2(pi x/2L)` over `[0,L]`), `omega_1 = (pi/2L)*sqrt(EA/mu)`.
    /// `c_crit_axial = 2 * (mu*L/2) * omega_1`.
    ///
    /// # Scope: takes ONE `ea`/`ei`, not `RodPoints::ea`/`ei`
    /// The closed-form mode shape above is only exact for a UNIFORM rod.
    /// For a genuinely non-uniform rod (see `RodPoints::ei`'s own doc) this
    /// is a real, disclosed approximation — pass a representative (e.g.
    /// mean, or the caller's own base `RodMaterial`) value; a true
    /// non-uniform modal solution needs a different, not-yet-built method
    /// (e.g. a real Rayleigh-Ritz or FE eigenvalue solve), not attempted
    /// here.
    pub fn modal_critical_damping(points: &RodPoints, ea: f32, ei: f32) -> (f32, f32) {
        let n = points.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        let length_m: f32 = points.rest_edge_length.iter().sum();
        let total_mass_kg: f32 = points.mass.iter().sum();
        if length_m <= 0.0 || total_mass_kg <= 0.0 {
            return (0.0, 0.0);
        }
        let mu = total_mass_kg / length_m; // kg/m, uniform-rod assumption (disclosed above)

        // ── Axial: exact elementary result ──
        let omega_axial = (std::f32::consts::PI / (2.0 * length_m)) * (ea / mu).sqrt();
        let m_modal_axial = mu * length_m / 2.0;
        let axial_damping = 2.0 * m_modal_axial * omega_axial;

        // ── Bending: real mode shape, numerically integrated ──
        const BETA_L: f32 = 1.8751; // same real constant energy::acoustics::modal uses
        let b = BETA_L / length_m;
        let sigma = (BETA_L.sinh() - BETA_L.sin()) / (BETA_L.cosh() + BETA_L.cos());
        let phi = |x: f32| -> f32 {
            let bx = b * x;
            (bx.cos() - bx.cosh()) + sigma * (bx.sinh() - bx.sin())
        };
        let phi_tip = phi(length_m);
        if phi_tip.abs() < 1.0e-9 {
            // Degenerate (shouldn't happen for the real beta_1*L root) --
            // real, disclosed fallback to the local reference above rather
            // than divide by ~zero.
            let l0 = length_m / (n - 1) as f32;
            let point_mass = total_mass_kg / n as f32;
            return Self::critical_damping(l0, point_mass, ea, ei);
        }

        const SAMPLES: usize = 2000;
        let dx = length_m / SAMPLES as f32;
        let mut integral = 0.0_f32;
        for i in 0..SAMPLES {
            let x0 = i as f32 * dx;
            let x1 = x0 + dx;
            let f0 = phi(x0).powi(2);
            let f1 = phi(x1).powi(2);
            integral += 0.5 * (f0 + f1) * dx; // composite trapezoidal rule
        }
        let m_modal_bending = mu * integral / phi_tip.powi(2);

        let omega_bending = (BETA_L * BETA_L) * (ei / (mu * length_m.powi(4))).sqrt();
        let bending_damping = 2.0 * m_modal_bending * omega_bending;

        (axial_damping, bending_damping)
    }

    /// Real fundamental bending-mode period (seconds) for a fixed-free rod —
    /// same real `beta_1*L=1.8751` eigenvalue and omega formula
    /// `modal_critical_damping` and `energy::acoustics::modal` both use, so
    /// this always agrees with them (single source of truth for this
    /// constant, not a second, possibly-drifting copy).
    ///
    /// Real use (2026-07-26 root-cause fix): the rod sleep-scoring test
    /// (`step.rs`) used to require a FIXED 0.5s of sustained low velocity
    /// before sleeping, regardless of the rod's own natural period — for a
    /// soft, slow rod whose own period is comparable to or longer than that
    /// fixed window, a genuine, still-large-amplitude oscillation can dwell
    /// below the speed threshold near a swing peak for that whole window,
    /// freezing the rod mid-swing at a real, wrong, off-rest position (this
    /// is not hypothetical -- it was the direct, measured cause of a real
    /// user-reported bug tonight). Scaling the settle-duration to a real
    /// multiple of THIS rod's own period fixes that at the root instead of
    /// picking a bigger fixed constant that would just move the same
    /// failure mode to an even slower rod.
    ///
    /// Same uniform-rod scope note as `modal_critical_damping` above: pass
    /// one representative `ei`, not a non-uniform rod's per-vertex array.
    pub fn fundamental_period_s(points: &RodPoints, ei: f32) -> f32 {
        let n = points.len();
        if n < 2 {
            return 0.0;
        }
        let length_m: f32 = points.rest_edge_length.iter().sum();
        let total_mass_kg: f32 = points.mass.iter().sum();
        if length_m <= 0.0 || total_mass_kg <= 0.0 || ei <= 0.0 {
            return 0.0;
        }
        let mu = total_mass_kg / length_m;
        const BETA_L: f32 = 1.8751; // same real constant used throughout this module
        let omega = (BETA_L * BETA_L) * (ei / (mu * length_m.powi(4))).sqrt();
        if omega <= 0.0 {
            return 0.0;
        }
        std::f32::consts::TAU / omega
    }

    /// Real Euler/Greenhill self-weight buckling critical height
    /// (`h_crit = (7.8373*EI/(mu*g))^(1/3)`) -- this formula was already
    /// documented in this module's own doc comment (see above, "direct
    /// continuity with the Greenhill self-buckling analysis"), motivating
    /// the ENTIRE rod solver's existence, but was never actually callable
    /// until now. Root-cause fix (2026-07-26): a demo scene silently built
    /// a rod taller than its own real critical height (a genuinely, freely
    /// buckling column, matching this exact real physics), and there was no
    /// way to catch that except hours of live debugging. Real, cited
    /// formula (see `mod.rs`'s own top-level doc, "Wikipedia 'Self-
    /// buckling'"-equivalent relation) -- `mu` here is the SAME linear mass
    /// density (kg/m) used everywhere else in this module.
    ///
    /// Closed-form result for a UNIFORM column (one `ei` for the whole
    /// height) -- a rod with per-vertex `RodPoints::ei` has no single exact
    /// non-uniform generalization here; `Rod::buckling_warning` covers that
    /// real case by calling this with the rod's own WEAKEST `ei` (the
    /// conservative bound — a non-uniform rod buckles first at its most
    /// slender point), not by extending this function itself.
    pub fn greenhill_critical_height_m(
        ei: f32,
        linear_density_kg_per_m: f32,
        gravity_m_s2: f32,
    ) -> f32 {
        if ei <= 0.0 || linear_density_kg_per_m <= 0.0 || gravity_m_s2 <= 0.0 {
            return f32::INFINITY; // no real self-weight buckling risk without all three real inputs
        }
        (7.8373 * ei / (linear_density_kg_per_m * gravity_m_s2)).powf(1.0 / 3.0)
    }
}

/// A rod embedded in a `Simulation` — points plus the material they share.
/// No separate gravity field: gravity comes from the single
/// `SimConfig::gravity` value the whole simulation already shares (a second
/// gravity knob would be exactly the kind of drift-prone duplication
/// `ARCHITECTURE.md` §2's "derive, don't store" spirit warns against). Wind
/// and push are per-rod, mutable state instead — set fresh by the caller
/// before each `step()` call, held constant for that call (which may cover
/// several substeps). Both default to zero/off, so embedding a rod with no
/// wind or push logic is exactly zero-cost.
#[derive(Debug, Clone)]
pub struct Rod {
    pub points: RodPoints,
    pub material: RodMaterial,
    pub wind_velocity: Vec2,
    pub wind_drag_coeff: f32,
    /// Read fresh every substep, same as `wind_velocity` — a one-shot
    /// `rod.points.v[i] +=` applied before `step()` gets fully absorbed by
    /// critical damping within the substep loop before the frame even
    /// returns, since `step()` can run thousands of internal substeps.
    pub push_center: Option<Vec2>,
    pub push_strength: f32,
    pub push_radius: f32,
    /// Mirrors `Particle::sleeping` at rod granularity: a rod's points are
    /// elastically coupled (bending/stretch energy between neighbors), so
    /// sleep is an all-or-nothing property of the whole rod, not individual
    /// points, unlike independent MPM particles. A sleeping rod skips
    /// scatter/gather/internal-force integration AND `rod_cfl_dt` entirely —
    /// the real fix for many-simultaneous-rods cost (a grass field): most
    /// blades settle to near-zero velocity and should stop paying their own
    /// (expensive, stiff) CFL bound every substep once they have. Woken by
    /// the same grid-activity-overlap test `wake_particle` already uses, or
    /// immediately when a caller sets an active push.
    pub sleeping: bool,
    /// Sleep-scoring must NOT fire on the instant `max_speed_sq < threshold_sq`:
    /// a freshly-constructed rod trivially satisfies that (`v = Vec2::ZERO` at
    /// construction) before gravity/grid coupling gets a chance to act within
    /// that first substep's tiny `sub_dt`, and would then skip its own gravity
    /// while "asleep" until a neighboring disturbance woke it — receiving the
    /// full, undamped gravitational transient it should have absorbed gradually
    /// as a large unphysical velocity spike. Every major real-time physics
    /// engine (Box2D, Bullet, PhysX) requires a body to stay below its sleep
    /// threshold for a minimum REAL DURATION, not one instant, to avoid this —
    /// this tracks that duration (real seconds), reset to 0 the moment speed
    /// exceeds the threshold, checked in `step.rs`'s sleep-scoring pass.
    pub below_threshold_time: f32,
    /// Real root gravitropism (Porat, Rivière, Meroz 2024 — see
    /// `gravitropism` module doc). `None` (default) = no gravitropic
    /// response, zero cost — a plain stem/blade doesn't grow toward
    /// gravity, only a root does.
    pub gravitropism: Option<Gravitropism>,
    /// Real phototropism (Cholodny & Went auxin-asymmetry theory — see
    /// `gravitropism` module doc's own "Phototropism reuses the SAME core"
    /// section). `None` (default) = no light-seeking response, zero cost.
    pub phototropism: Option<Phototropism>,
    /// Real elongation growth (see `growth` module doc). `None` (default) =
    /// fixed length, zero cost — most bodies aren't actively growing every
    /// frame of their existence.
    pub growth: Option<Growth>,
    /// Real stress-driven secondary growth / thigmomorphogenesis (Jaffe
    /// 1973, Mattheck & Kübler 1995 — see `secondary_growth` module doc).
    /// `None` (default) = fixed stiffness, zero cost — requires Phase 1's
    /// per-vertex `RodPoints::ea`/`ei` to already be filled (`Rod::new`
    /// does this).
    pub secondary_growth: Option<SecondaryGrowth>,
    /// Real elastic-perfectly-plastic bending (see `plasticity` module doc).
    /// `None` (default) = purely elastic, zero cost — most bodies don't
    /// permanently deform under load; a wire/branch/cable that should stay
    /// bent after enough force opts in.
    pub plasticity: Option<RodPlasticity>,
    /// Implicit (backward Euler) integration, opt-in (Baraff & Witkin 1998;
    /// see `implicit` module doc). `false` (default) = the explicit path,
    /// unchanged. When `true`, solved ONCE per `Simulation::step()` at the
    /// full frame `dt`, unconditionally stable regardless of stiffness
    /// (1298->1 substeps/frame measured on `rod_blade_of_grass_gui.rs`'s
    /// blade), excluded from the substep loop's CFL scan.
    ///
    /// Gravity/wind/push stay INSIDE this solve, not on the shared grid —
    /// an implicit rod has no CFL ceiling, so an explicit `v += g*dt` at the
    /// full frame dt is unconditionally unstable (tried once, reverted).
    /// Scope: not grid-coupled — no contact with sand/particles yet.
    pub use_implicit_integration: bool,
    /// Real, measured, disclosed finding (2026-07-27): backward Euler is
    /// unconditionally STABLE at any `dt`, but at a large `dt` relative to
    /// the rod's own natural bending period, it also introduces real
    /// artificial numerical damping that can swamp the physically-tuned
    /// damping (`RodMaterial::axial_damping`/`bending_damping`), making a
    /// real, underdamped sway look like a smooth, "instant" glide to rest
    /// instead. Verified directly: one 0.02s implicit step/frame gave only
    /// 3 real tip-direction reversals over 3s of a pushed 20-point blade;
    /// splitting that SAME frame `dt` into 16 smaller implicit steps (this
    /// field) gave 9 -- visibly more oscillatory, same physical damping
    /// ratio, same total real time. Default `1` = today's exact prior
    /// behavior (one step at the full frame `dt`), zero change for any rod
    /// that doesn't opt in. Only meaningful when `use_implicit_integration`
    /// is `true`.
    pub implicit_substeps: u32,
}

impl Rod {
    /// Fills `points.ea`/`points.ei` to the correct per-edge/per-vertex
    /// length using `material`'s scalar values whenever they don't already
    /// match (the normal case: `build_straight_rod` leaves them empty) --
    /// makes a uniform-material rod bit-identical to the prior
    /// single-scalar-material behavior. A caller who wants real non-uniform
    /// stiffness fills `points.ea`/`ei` explicitly BEFORE calling `Rod::new`
    /// (matching length), and this fill is skipped.
    pub fn new(points: RodPoints, material: RodMaterial) -> Self {
        let mut points = points;
        let n_edges = points.x.len().saturating_sub(1);
        let n_bend = points.x.len().saturating_sub(2);
        if points.ea.len() != n_edges {
            points.ea = vec![material.ea; n_edges];
        }
        if points.ei.len() != n_bend {
            points.ei = vec![material.ei; n_bend];
        }
        Self {
            points,
            material,
            wind_velocity: Vec2::ZERO,
            wind_drag_coeff: 0.0,
            push_center: None,
            push_strength: 0.0,
            push_radius: 0.0,
            sleeping: false,
            below_threshold_time: 0.0,
            gravitropism: None,
            phototropism: None,
            growth: None,
            secondary_growth: None,
            plasticity: None,
            use_implicit_integration: false,
            implicit_substeps: 1,
        }
    }

    /// Real, immediate self-weight buckling check -- root-cause fix
    /// (2026-07-26) for a real bug that cost hours of live debugging before
    /// being traced to genuine Euler/Greenhill self-weight buckling (see
    /// `RodMaterial::greenhill_critical_height_m`'s own doc). Returns
    /// `Some(real, human-readable message)` if this rod's actual real
    /// length exceeds its own critical height (it will NEVER stand
    /// straight under gravity alone, regardless of damping -- that was the
    /// actual, correct physics all along, not a numerical bug), `None` if
    /// it's safely below. Call this once right after construction and
    /// `eprintln!` the result -- catches this class of mistake in seconds
    /// instead of a multi-hour debugging session.
    /// Real, disclosed simplification for a NON-uniform rod (per-vertex
    /// `points.ei`, see `RodPoints::ei`'s own doc): `greenhill_critical_height_m`
    /// is a closed-form result for a UNIFORM column, so there is no single
    /// exact non-uniform generalization here. Uses the WEAKEST (minimum)
    /// `ei` entry as the real, conservative bound instead — a non-uniform
    /// rod genuinely buckles first at its most slender point, so checking
    /// the whole rod's real length against that point's own critical height
    /// cannot UNDER-warn (it may warn slightly early for a rod that's
    /// stiffer everywhere else, never miss a real risk).
    pub fn buckling_warning(&self, gravity_m_s2: f32) -> Option<String> {
        let length_m: f32 = self.points.rest_edge_length.iter().sum();
        let total_mass_kg: f32 = self.points.mass.iter().sum();
        if length_m <= 0.0 || total_mass_kg <= 0.0 {
            return None;
        }
        let mu = total_mass_kg / length_m;
        // `self.material.ei` is only the FILL VALUE `Rod::new` used at
        // construction -- once `SecondaryGrowth` (or any caller) grows
        // `points.ei` beyond it, `material.ei` itself is never updated and
        // must NOT be folded in here, or a genuinely-stiffened rod would
        // incorrectly keep reporting its own stale original weakness.
        let weakest_ei = if self.points.ei.is_empty() {
            self.material.ei
        } else {
            self.points.ei.iter().copied().fold(f32::INFINITY, f32::min)
        };
        let h_crit = RodMaterial::greenhill_critical_height_m(weakest_ei, mu, gravity_m_s2);
        if length_m > h_crit {
            Some(format!(
                "rod is {length_m:.4}m tall but its own real Euler/Greenhill self-weight \
                 buckling critical height is only {h_crit:.4}m (weakest EI={weakest_ei:.4e} \
                 N*m^2, mu={mu:.4} kg/m, g={gravity_m_s2:.2} m/s^2) -- this rod will \
                 genuinely, physically NOT stand straight under gravity alone, no matter the \
                 damping. Either shorten it below {h_crit:.4}m or stiffen it (increase E or the \
                 cross-section's I)."
            ))
        } else {
            None
        }
    }

    pub fn with_wind(mut self, wind_velocity: Vec2, wind_drag_coeff: f32) -> Self {
        self.wind_velocity = wind_velocity;
        self.wind_drag_coeff = wind_drag_coeff;
        self
    }

    pub fn with_gravitropism(mut self, gravitropism: Gravitropism) -> Self {
        self.gravitropism = Some(gravitropism);
        self
    }

    pub fn with_phototropism(mut self, phototropism: Phototropism) -> Self {
        self.phototropism = Some(phototropism);
        self
    }

    pub fn with_growth(mut self, growth: Growth) -> Self {
        self.growth = Some(growth);
        self
    }

    pub fn with_plasticity(mut self, plasticity: RodPlasticity) -> Self {
        self.plasticity = Some(plasticity);
        self
    }

    /// True while `growth` is still meaningfully lengthening the tip edge —
    /// real guard against a genuine sleep/growth interaction bug: sleep
    /// scoring (`step.rs`) only sees `rod.points.v`, but a critically-damped
    /// rod's elastic response reaches quasi-static equilibrium (near-zero
    /// velocity) on a MUCH faster timescale than logistic growth itself
    /// (milliseconds vs. tens of seconds), so a velocity-only sleep check
    /// would put the rod to sleep mid-growth and silently freeze it there —
    /// once `sleeping=true`, `apply_growth` is skipped entirely alongside
    /// everything else. 99% of `max_segment_length_m` is the real, standard
    /// cutoff for an asymptotic logistic curve that mathematically never
    /// exactly reaches its carrying capacity. `false` (safe to sleep) for a
    /// rod with no `growth` at all.
    pub fn is_growing(&self) -> bool {
        match &self.growth {
            Some(g) => match self.points.rest_edge_length.last() {
                Some(&l) => l < 0.99 * g.max_segment_length_m,
                None => false,
            },
            None => false,
        }
    }

    /// True while `gravitropism` still has a real, meaningful angular
    /// deviation left to correct -- the exact same class of bug
    /// `is_growing`'s own doc describes, found 2026-07-27: sleep scoring
    /// only sees `rod.points.v`, but gravitropism reshapes `rest_curvature`
    /// (not velocity directly), so a rod can settle to near-zero velocity
    /// from its LAST push, go to sleep, and then never wake again -- freezing
    /// gravitropism forever with no external event left to rouse it (a
    /// sleeping rod is skipped entirely at both `apply_gravitropism` call
    /// sites in `step.rs`). `false` (safe to sleep) for a rod with no
    /// `gravitropism` at all, or once it's genuinely converged.
    pub fn is_correcting_gravitropically(&self, gravity: Vec2, grid: &crate::grid::Grid) -> bool {
        self.gravitropism
            .as_ref()
            .is_some_and(|g| gravitropism::still_correcting(&self.points, g, gravity, grid))
    }

    /// See `is_correcting_gravitropically`'s own doc -- the phototropism
    /// analog, same sleep-freeze-prevention purpose.
    pub fn is_correcting_phototropically(&self, light_dir: Vec2, grid: &crate::grid::Grid) -> bool {
        self.phototropism.as_ref().is_some_and(|p| {
            gravitropism::still_correcting_phototropically(&self.points, p, light_dir, grid)
        })
    }
}

/// Build a straight rod from `start` to `end`, `n_points` control points,
/// uniform `linear_density_kg_per_m`, zero rest curvature (straight rest
/// shape). `dx_meters` converts the real-meters spacing into grid-cell units
/// for `x` — same SI-to-grid convention `gravity_to_grid`/`lame_from_si`
/// already use elsewhere in this codebase.
pub fn build_straight_rod(
    start: Vec2,
    end: Vec2,
    n_points: usize,
    linear_density_kg_per_m: f32,
    dx_meters: f32,
) -> RodPoints {
    assert!(n_points >= 2, "a rod needs at least 2 points");
    let total_length_m = (end - start).length() * dx_meters;
    let segment_length_m = total_length_m / (n_points as f32 - 1.0);
    let point_mass = linear_density_kg_per_m * segment_length_m;

    let mut x = Vec::with_capacity(n_points);
    let mut v = Vec::with_capacity(n_points);
    let mut mass = Vec::with_capacity(n_points);
    let mut pinned = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let t = i as f32 / (n_points as f32 - 1.0);
        x.push(start.lerp(end, t));
        v.push(Vec2::ZERO);
        // Endpoints carry half a segment's mass (standard lumped-mass
        // discretization), interior points carry a full segment.
        let m = if i == 0 || i == n_points - 1 {
            point_mass * 0.5
        } else {
            point_mass
        };
        mass.push(m);
        pinned.push(0);
    }

    RodPoints {
        x,
        v,
        mass,
        pinned,
        rest_edge_length: vec![segment_length_m; n_points - 1],
        rest_curvature: vec![0.0; n_points.saturating_sub(2)],
        // Left empty -- `Rod::new` fills these from `RodMaterial::ea`/`ei`
        // (this function has no material to fill them with yet).
        ea: Vec::new(),
        ei: Vec::new(),
        position_compensation: vec![Vec2::ZERO; n_points],
        contact_group: vec![0; n_points],
        accumulated_plastic_curvature: vec![0.0; n_points.saturating_sub(2)],
        linear_density_kg_per_m: vec![linear_density_kg_per_m; n_points - 1],
    }
}

#[cfg(test)]
mod root_cause_fixes_tests {
    use super::*;

    /// Real, permanent regression guard for the 2026-07-26 root-cause fix:
    /// `modal_critical_damping` must give a substantially LARGER bending
    /// value than the old, disclosed-as-too-small `critical_damping` for a
    /// real multi-point cantilever -- confirmed empirically tonight to be a
    /// two-to-three-orders-of-magnitude gap for a 20-point blade.
    #[test]
    fn modal_critical_damping_exceeds_local_reference_substantially() {
        let start = Vec2::new(9.0, 4.0);
        let height_m = 0.10;
        let dx_meters = 0.01;
        let points = build_straight_rod(
            start,
            Vec2::new(start.x, start.y + height_m / dx_meters),
            20,
            0.01,
            dx_meters,
        );
        let young_modulus = 1.0e7_f32;
        let ea = young_modulus * 0.003 * 0.001;
        let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
        let l0 = height_m / 19.0;
        let point_mass = 0.01 * l0;
        let (_, old_bending) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
        let (_, new_bending) = RodMaterial::modal_critical_damping(&points, ea, ei);
        assert!(
            new_bending > old_bending * 100.0,
            "modal_critical_damping ({new_bending:.6e}) should exceed the local reference \
             ({old_bending:.6e}) by at least 100x for a 20-point cantilever -- if this ever \
             shrinks close to 1x, the two formulas may have been (wrongly) unified without \
             re-verifying against tonight's real measurement"
        );
    }

    /// Real cross-check: `fundamental_period_s` must be the exact reciprocal
    /// of the frequency `energy::acoustics::modal::cantilever_rod_modes`
    /// computes -- both use the SAME real beta_1 eigenvalue and omega
    /// formula, so any divergence between them is a real bug in one or the
    /// other, not just numerical noise.
    #[cfg(feature = "experimental")]
    #[test]
    fn fundamental_period_matches_acoustics_module_frequency() {
        let start = Vec2::new(0.0, 0.0);
        let height_m = 0.10;
        let points =
            build_straight_rod(start, Vec2::new(start.x, start.y + height_m), 20, 0.01, 1.0);
        let young_modulus = 1.0e7_f32;
        let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
        let material = RodMaterial::new(young_modulus * 0.003 * 0.001, ei, 0.0, 0.0);
        let rod = Rod::new(points, material);

        let period = RodMaterial::fundamental_period_s(&rod.points, ei);
        let modes = crate::acoustics::cantilever_rod_modes(&rod, 1);
        let freq_hz = modes[0].frequency_hz;

        let rel_err = (period - 1.0 / freq_hz).abs() / (1.0 / freq_hz);
        assert!(
            rel_err < 1.0e-4,
            "fundamental_period_s ({period:.6}s) should be the exact reciprocal of \
             cantilever_rod_modes's own frequency ({freq_hz:.4} Hz -> period {:.6}s) -- \
             rel_err={rel_err:.6}",
            1.0 / freq_hz
        );
    }

    /// Real, permanent regression guard reproducing tonight's own found
    /// bug exactly: blade B's real parameters (E=5e6, height=0.10m) must
    /// trigger a buckling warning; blade A's (E=1e7, same height) must not.
    #[test]
    fn buckling_warning_matches_tonights_real_finding() {
        let start = Vec2::new(9.0, 4.0);
        let height_m = 0.10;
        let dx_meters = 0.01;
        let make_rod = |young_modulus: f32| -> Rod {
            let points = build_straight_rod(
                start,
                Vec2::new(start.x, start.y + height_m / dx_meters),
                20,
                0.01,
                dx_meters,
            );
            let ea = young_modulus * 0.003 * 0.001;
            let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
            Rod::new(points, RodMaterial::new(ea, ei, 0.0, 0.0))
        };

        let blade_a = make_rod(1.0e7);
        let blade_b = make_rod(5.0e6);

        assert!(
            blade_a.buckling_warning(9.81).is_none(),
            "blade A (E=1e7) should be safely below its own critical height -- got a warning: {:?}",
            blade_a.buckling_warning(9.81)
        );
        assert!(
            blade_b.buckling_warning(9.81).is_some(),
            "blade B (E=5e6) is the real, confirmed buckling case found tonight -- should warn"
        );
    }

    /// Real, permanent regression guard for the 2026-07-27 sleep-freeze fix:
    /// a rod with a genuine, uncorrected gravitropic deviation must report
    /// `is_correcting_gravitropically() == true` (blocking sleep), while one
    /// already aligned with its own target must NOT (so ordinary sleep still
    /// works once gravitropism has nothing real left to do).
    #[test]
    fn gravitropism_prevents_premature_sleep_until_converged() {
        let dx_meters = 0.01;
        let make_rod = |tip_offset_x: f32| -> Rod {
            let start = Vec2::new(9.0, 4.0);
            let mut points = build_straight_rod(
                start,
                Vec2::new(start.x, start.y + 0.10 / dx_meters),
                6,
                0.01,
                dx_meters,
            );
            let last = points.x.len() - 1;
            points.x[last].x += tip_offset_x;
            let mut rod = Rod::new(points, RodMaterial::new(1.0, 1.0e-6, 0.0, 0.0));
            rod.gravitropism = Some(Gravitropism::new(0.05, 0.005));
            rod
        };

        let gravity = Vec2::new(0.0, -1.0);
        let grid = crate::grid::Grid::new(16);
        let misaligned = make_rod(2.0); // tip pushed sideways -- real, unconverged deviation
        let aligned = make_rod(0.0); // straight down -- matches GSA=0.0's target exactly

        assert!(
            misaligned.is_correcting_gravitropically(gravity, &grid),
            "a rod with a real, uncorrected angular deviation must report still-correcting"
        );
        assert!(
            !aligned.is_correcting_gravitropically(gravity, &grid),
            "a rod already aligned with its own target must NOT block sleep"
        );
    }

    /// Same sleep-freeze-prevention guard as
    /// `gravitropism_prevents_premature_sleep_until_converged`, for the
    /// phototropism analog -- proves `is_correcting_phototropically` isn't
    /// a no-op stub.
    #[test]
    fn phototropism_prevents_premature_sleep_until_converged() {
        let dx_meters = 0.01;
        let make_rod = |tip_offset_x: f32| -> Rod {
            let start = Vec2::new(9.0, 4.0);
            let mut points = build_straight_rod(
                start,
                Vec2::new(start.x, start.y + 0.10 / dx_meters),
                6,
                0.01,
                dx_meters,
            );
            let last = points.x.len() - 1;
            points.x[last].x += tip_offset_x;
            let mut rod = Rod::new(points, RodMaterial::new(1.0, 1.0e-6, 0.0, 0.0));
            rod.phototropism = Some(Phototropism::new(0.05, 0.005));
            rod
        };

        let light_dir = Vec2::new(0.0, 1.0);
        let grid = crate::grid::Grid::new(16);
        let misaligned = make_rod(2.0);
        let aligned = make_rod(0.0);

        assert!(
            misaligned.is_correcting_phototropically(light_dir, &grid),
            "a rod with a real, uncorrected angular deviation from the light direction \
             must report still-correcting"
        );
        assert!(
            !aligned.is_correcting_phototropically(light_dir, &grid),
            "a rod already aligned with the light direction must NOT block sleep"
        );
    }
}

#[cfg(test)]
mod per_vertex_stiffness_tests {
    use super::*;

    /// Real backward-compat guard: `Rod::new` must fill `points.ea`/`ei` to
    /// the material's OWN scalar at every index -- a uniform-material rod's
    /// internal forces must be bit-identical to what the single-scalar
    /// `RodMaterial` path always produced, not just "close".
    #[test]
    fn rod_new_fills_uniform_stiffness_from_material_bit_identical() {
        let points = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 4.0), 5, 0.01, 1.0);
        let material = RodMaterial::new(123.0, 4.5e-3, 0.0, 0.0);
        let rod = Rod::new(points, material);

        assert_eq!(rod.points.ea.len(), 4, "N-1 edges for 5 points");
        assert_eq!(rod.points.ei.len(), 3, "N-2 bending vertices for 5 points");
        for &ea in &rod.points.ea {
            assert_eq!(
                ea, 123.0,
                "every edge must get the material's own EA exactly"
            );
        }
        for &ei in &rod.points.ei {
            assert_eq!(
                ei, 4.5e-3,
                "every bending vertex must get the material's own EI exactly"
            );
        }
    }

    /// A caller who explicitly sets non-uniform `points.ea`/`ei` BEFORE
    /// `Rod::new` gets real non-uniform stiffness -- `Rod::new` must not
    /// overwrite an already-correctly-sized array.
    #[test]
    fn rod_new_preserves_an_explicitly_set_non_uniform_array() {
        let mut points = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 4.0), 5, 0.01, 1.0);
        points.ea = vec![10.0, 20.0, 30.0, 40.0];
        points.ei = vec![1.0, 2.0, 3.0];
        let rod = Rod::new(points, RodMaterial::new(999.0, 999.0, 0.0, 0.0));

        assert_eq!(rod.points.ea, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(rod.points.ei, vec![1.0, 2.0, 3.0]);
    }

    /// The real, observable effect non-uniform stiffness should produce: a
    /// horizontal cantilever with a SOFT base half must deflect more at its
    /// arc-length midpoint under a real, constant transverse tip load than
    /// an equivalent uniformly-stiff rod carrying the exact same load — no
    /// gravity/buckling involved (a straight rod under pure axial gravity
    /// has zero bending moment by symmetry, real physics, not useful for
    /// this comparison), a real transverse point load instead, same style
    /// `tests/accuracy.rs`'s own cantilever-deflection test already uses.
    #[test]
    fn softer_base_bends_more_than_a_uniformly_stiff_rod() {
        let length_m = 1.0;
        let n_points = 11usize;
        let ea = 1.0e5_f32;
        let ei = 50.0_f32;
        let tip_load_n = 0.02_f32;

        let make_rod = |ei_override: Option<Vec<f32>>| -> Rod {
            let mut points = build_straight_rod(
                Vec2::new(0.0, 0.0),
                Vec2::new(length_m, 0.0),
                n_points,
                0.1,
                1.0,
            );
            points.pinned[0] = 1;
            points.pinned[1] = 1;
            if let Some(ei_arr) = ei_override {
                points.ei = ei_arr;
            }
            let l0 = length_m / (n_points as f32 - 1.0);
            let point_mass = 0.1 * l0;
            let (axial_damping, bending_damping) =
                RodMaterial::critical_damping(l0, point_mass, ea, ei);
            Rod::new(
                points,
                RodMaterial::new(ea, ei, axial_damping, bending_damping),
            )
        };

        let n_bend = n_points - 2;
        let soft_ei: Vec<f32> = (0..n_bend)
            .map(|i| if i < n_bend / 2 { ei * 0.1 } else { ei })
            .collect();

        let mut uniform = make_rod(None);
        let mut soft_base = make_rod(Some(soft_ei));

        let dt = rod_cfl_dt(&uniform.points, &uniform.material, 0.4).min(rod_cfl_dt(
            &soft_base.points,
            &soft_base.material,
            0.4,
        ));
        assert!(dt.is_finite() && dt > 0.0);

        let n = n_points;
        for _ in 0..200_000 {
            for rod in [&mut uniform, &mut soft_base] {
                let mut internal = compute_internal_forces(
                    &rod.points.x,
                    &rod.points.v,
                    RodRestState {
                        rest_edge_length: &rod.points.rest_edge_length,
                        rest_curvature: &rod.points.rest_curvature,
                        ea: &rod.points.ea,
                        ei: &rod.points.ei,
                    },
                    &rod.material,
                    1.0,
                );
                internal[n - 1] += Vec2::new(0.0, tip_load_n);
                for (i, internal_force) in internal.iter().enumerate() {
                    if rod.points.pinned[i] != 0 {
                        rod.points.v[i] = Vec2::ZERO;
                        continue;
                    }
                    let a = *internal_force / rod.points.mass[i].max(1.0e-9);
                    rod.points.v[i] += a * dt;
                }
                for i in 0..n {
                    if rod.points.pinned[i] == 0 {
                        rod.points.x[i] += rod.points.v[i] * dt;
                    }
                }
            }
        }

        // Vertical deflection at the arc-length midpoint -- the real,
        // softer-base rod should have deflected further than the uniform
        // reference under the identical tip load by this point.
        let mid = n / 2;
        let uniform_deflection = (uniform.points.x[mid].y).abs();
        let soft_base_deflection = (soft_base.points.x[mid].y).abs();
        assert!(
            soft_base_deflection > uniform_deflection * 1.2,
            "a softer base should deflect measurably more at the midpoint than a uniform rod \
             under the same tip load: soft_base={soft_base_deflection:.6} uniform={uniform_deflection:.6}"
        );
    }
}

#[cfg(test)]
mod secondary_growth_integration_tests {
    use super::*;

    /// The real, concrete proof this whole chain was previously blocked on:
    /// a rod built ABOVE its own Greenhill critical height (`buckling_warning`
    /// reports genuine risk) that experiences real, sustained bending moment under its own
    /// self-weight (a tiny initial tilt breaks the perfectly-straight
    /// symmetric case, which has zero moment by construction -- same real
    /// lesson as this session's earlier buckling investigation) should,
    /// given `SecondaryGrowth`, genuinely stiffen enough over real time to
    /// raise its own critical height back above its actual height.
    #[test]
    fn sustained_bending_stress_raises_greenhill_height_above_actual_height() {
        let dx_meters = 0.01;
        let height_m = 0.10;
        let start = Vec2::new(0.0, 0.0);
        let end = Vec2::new(0.0, height_m / dx_meters);
        let n_points = 12;
        let young_modulus = 5.0e6_f32; // deliberately over-critical, same order as "blade B"
        let ea = young_modulus * 0.003 * 0.001;
        let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;

        let mut points = build_straight_rod(start, end, n_points, 0.01, dx_meters);
        points.pinned[0] = 1;
        points.pinned[1] = 1;
        // Real, genuinely CURVED shape (quadratic in index, not a rigid
        // linear tilt -- three colinear points have zero discrete
        // curvature regardless of overall angle, so a rigid tilt alone
        // would give secondary growth no real moment to respond to).
        // Modest magnitude (max ~0.15 grid cells = 1.5mm at the tip,
        // comparable to the rod's own ~9mm segment length) -- a real,
        // sustained bend a mature stem could plausibly hold under wind
        // load, not a violent distortion that would itself dominate the
        // dynamics. This test verifies `SecondaryGrowth`'s OWN response to
        // a sustained real moment directly (`x` held fixed, no `step_rod`
        // dynamics) -- the genuine buckling INSTABILITY's own real-seconds-
        // scale dynamics are already covered by
        // `tests/rod_gravitropism_whole_organ.rs`'s negative-control test.
        for (i, p) in points.x.iter_mut().enumerate() {
            let t = i as f32 / (n_points - 1) as f32;
            p.x += t * t * 0.15;
        }
        let material = RodMaterial::new(ea, ei, 0.0, 0.0);
        let rod = Rod::new(points, material);

        assert!(
            rod.buckling_warning(9.81).is_some(),
            "this rod must start genuinely over its own critical height"
        );

        let secondary_growth = SecondaryGrowth::new(1.0e-2, 1.0e-7, 0.0, 1.0e9);
        let dt = 0.05_f32;
        let mut rod = rod;
        let mut raised = false;
        for _ in 0..20_000 {
            apply_secondary_growth(&mut rod.points, &secondary_growth, dx_meters, dt);
            if rod.buckling_warning(9.81).is_none() {
                raised = true;
                break;
            }
        }

        assert!(
            raised,
            "sustained bending stress under SecondaryGrowth must eventually raise this rod's \
             own Greenhill critical height above its actual height -- final weakest ei={:?}",
            rod.points.ei.iter().cloned().fold(f32::INFINITY, f32::min)
        );
    }
}
