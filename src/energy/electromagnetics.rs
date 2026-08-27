//! Electromagnetic wave and material property math -- the Energy half of
//! `electromagnetics::`.
//!
//! Pure-Rust, no ECS. Ported from `crates/energy/src/electromagnetism/interactions.rs`.
//! Split from the old unified `electromagnetics/` module 2026-07-11: wave
//! propagation and optical `MaterialProperties` are radiative energy
//! transfer (Energy domain); point-charge/current force-application math is
//! `forces::electromagnetics`.

use crate::forces::electromagnetics::{ElectricField, MagneticField};
use crate::spacetime::solver::LcgRng;
use glam::Vec2;

/// Speed of light in vacuum (m/s).
pub const C: f32 = 299_792_458.0;

/// A plane electromagnetic wave propagating in 2D.
///
/// E and B are transverse to the propagation direction.
/// B_amplitude = E_amplitude / c (vacuum relation).
pub struct ElectromagneticWave {
    pub frequency: f32,
    pub direction: Vec2,
    pub electric_amplitude: f32,
    pub magnetic_amplitude: f32,
    pub phase: f32,
    pub wave_number: f32,
}

impl ElectromagneticWave {
    /// Construct from frequency, propagation direction, electric amplitude, and phase.
    pub fn new(frequency: f32, direction: Vec2, electric_amplitude: f32, phase: f32) -> Self {
        assert!(frequency > 0.0, "Wave frequency must be positive");
        let wavelength = C / frequency;
        let wave_number = 2.0 * std::f32::consts::PI / wavelength;
        Self {
            frequency,
            direction: direction.normalize(),
            electric_amplitude,
            magnetic_amplitude: electric_amplitude / C,
            phase,
            wave_number,
        }
    }

    /// E and B fields at `position` and `time`.
    ///
    /// Real fix, 2026-08-21: B used to be built from an in-plane `m_dir`
    /// vector (rotating `e_dir` a further 90°, which lands ANTIPARALLEL to
    /// the propagation direction) -- that makes the wave LONGITUDINAL, the
    /// opposite of this function's own doc claim ("E and B are transverse to
    /// the propagation direction"). Real fix: in 2D, B is the out-of-plane
    /// scalar (see `MagneticField`'s own doc), genuinely transverse to a
    /// propagation direction that only ever lives in the plane -- E, B,
    /// and the propagation direction are mutually perpendicular exactly as a
    /// real EM wave requires, once B is allowed to point out of the page
    /// instead of being forced back into it.
    pub fn get_fields_at(&self, position: Vec2, time: f32) -> (ElectricField, MagneticField) {
        let proj = self.direction.dot(position);
        let phi = self.wave_number * proj - 2.0 * std::f32::consts::PI * self.frequency * time
            + self.phase;
        let sin_p = phi.sin();
        let e_dir = Vec2::new(-self.direction.y, self.direction.x);
        (
            ElectricField::new(e_dir * (self.electric_amplitude * sin_p), position),
            MagneticField::new(self.magnetic_amplitude * sin_p, position),
        )
    }
}

/// Electromagnetic material properties (permittivity, permeability, conductivity).
#[derive(Debug, Clone, Copy)]
pub struct MaterialProperties {
    /// Electric permittivity ε (F/m).
    pub permittivity: f32,
    /// Magnetic permeability μ (H/m).
    pub permeability: f32,
    /// Electrical conductivity σ (S/m).
    pub conductivity: f32,
}

impl MaterialProperties {
    pub fn vacuum() -> Self {
        Self {
            permittivity: 8.854_188e-12,
            permeability: 1.256_637e-6,
            conductivity: 0.0,
        }
    }
    pub fn new(permittivity: f32, permeability: f32, conductivity: f32) -> Self {
        Self {
            permittivity,
            permeability,
            conductivity,
        }
    }
    /// Refractive index n = √(εᵣ·μᵣ).
    pub fn refractive_index(&self) -> f32 {
        let vac = Self::vacuum();
        ((self.permittivity / vac.permittivity) * (self.permeability / vac.permeability)).sqrt()
    }
    /// Speed of light in this material: v = c/n.
    pub fn light_speed(&self) -> f32 {
        C / self.refractive_index()
    }
}

/// Real electric potential field on a 2D grid, solved by relaxing Laplace's
/// equation (Gauss's law with zero charge density in air, ∇²φ = 0) via
/// Jacobi iteration -- the same numerical family `ScalarDiffusionField`'s
/// own diffusion already uses (Laplace is that same PDE's zero-source,
/// steady-state limit: `∂φ/∂t = D∇²φ` settles to `∇²φ = 0`), but a pure
/// GRID field with no particle coupling -- the potential between a cloud
/// and the ground doesn't live on any particle, unlike temperature or
/// moisture.
///
/// Generic, not lightning-specific: this is a real, reusable steady-state
/// potential-field solver, usable for any problem needing one (a first real
/// application is dielectric-breakdown / lightning-leader growth, since the
/// leader grows toward the strongest local field, `-∇φ`).
///
/// Dirichlet boundaries top/bottom (fixed potential each -- e.g. cloud=0,
/// ground=1), Neumann (zero-gradient, copy-nearest) on the left/right sides
/// -- an open lateral domain. Same convention the real reference
/// implementation this was checked against uses
/// (github.com/diluuuu10/triggered-discharge, cloned in `tmp/` -- Jacobi
/// relaxation for the potential field, confirmed via that repo's own code).
#[derive(Clone)]
pub struct ElectricPotentialField {
    width: usize,
    height: usize,
    phi: Vec<f32>,
    phi_next: Vec<f32>,
    /// Real, generalized Dirichlet overlay: `Some(v)` pins a cell at
    /// potential `v` every relaxation sweep, `None` leaves it free. The
    /// top/bottom rows start pinned this way at construction -- but this is
    /// the SAME mechanism a growing conductor (a dielectric-breakdown
    /// leader channel) uses to extend the boundary as it grows (see `pin`'s
    /// own doc): a real conductor is an equipotential, so once a cell joins
    /// the channel, the field around it must be solved as if THAT cell were
    /// also a fixed-potential electrode, which is the actual physical
    /// mechanism (field concentration ahead of a growing conductive tip)
    /// that makes a leader grow roughly toward its own origin's field
    /// direction instead of uniformly at random.
    fixed: Vec<Option<f32>>,
}

impl ElectricPotentialField {
    /// Real initial guess: linear interpolation from `top_value` to
    /// `bottom_value`. Not required for correctness -- Jacobi relaxation
    /// converges from any starting state, including all-zero -- but a
    /// starting guess already close to the true solution converges in far
    /// fewer iterations than starting flat, the same real reason a good
    /// initial guess matters for any iterative solver.
    pub fn new(width: usize, height: usize, top_value: f32, bottom_value: f32) -> Self {
        assert!(
            width > 0 && height > 0,
            "ElectricPotentialField: grid dimensions must be positive"
        );
        let denom = (height.max(2) - 1) as f32;
        let mut phi = vec![0.0_f32; width * height];
        let mut fixed = vec![None; width * height];
        for (y, row) in phi.chunks_mut(width).enumerate() {
            let t = y as f32 / denom;
            let v = top_value + (bottom_value - top_value) * t;
            row.fill(v);
        }
        for x in 0..width {
            fixed[x] = Some(top_value);
            fixed[(height - 1) * width + x] = Some(bottom_value);
        }
        let phi_next = phi.clone();
        Self {
            width,
            height,
            phi,
            phi_next,
            fixed,
        }
    }

    /// Pin a single cell at a fixed potential, extending the Dirichlet
    /// boundary to include it -- the real mechanism a growing dielectric-
    /// breakdown leader uses to make itself part of the conducting
    /// boundary once it joins the channel (see the `fixed` field's own
    /// doc). Takes effect starting from the NEXT `relax_step` call.
    pub fn pin(&mut self, x: usize, y: usize, value: f32) {
        let i = self.idx(x, y);
        self.fixed[i] = Some(value);
        self.phi[i] = value;
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    /// One Jacobi relaxation sweep: interior cells become the average of
    /// their 4 neighbors (the real discretization of `∇²φ=0` -- see
    /// `ScalarDiffusionField`'s own doc for the identical Laplacian finite-
    /// difference form, just without the diffusion coefficient/dt scaling
    /// since this solves the steady state directly rather than stepping
    /// toward it in time). Top/bottom rows stay pinned at the Dirichlet
    /// boundary values every sweep. Left/right columns use their own
    /// nearest interior neighbor twice in the average, the standard finite-
    /// difference form of a zero-gradient (Neumann) boundary.
    pub fn relax_step(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let i = self.idx(x, y);
                if let Some(v) = self.fixed[i] {
                    self.phi_next[i] = v;
                    continue;
                }
                let left = if x == 0 {
                    self.phi[self.idx(x + 1, y)]
                } else {
                    self.phi[self.idx(x - 1, y)]
                };
                let right = if x == self.width - 1 {
                    self.phi[self.idx(x - 1, y)]
                } else {
                    self.phi[self.idx(x + 1, y)]
                };
                // Up/down: with pinned interior cells now possible (a
                // channel cell may sit at y=0 or y=height-1's own row is
                // already fully fixed, so this only matters for a channel
                // reaching an interior row), the vertical neighbors always
                // exist for 0<y<height-1; a channel cell can only be
                // interior, never on the fixed top/bottom rows themselves.
                let up = self.phi[self.idx(x, y - 1)];
                let down = self.phi[self.idx(x, y + 1)];
                self.phi_next[i] = 0.25 * (left + right + up + down);
            }
        }
        std::mem::swap(&mut self.phi, &mut self.phi_next);
    }

    /// Repeated relaxation -- see `relax_step`'s own doc. Real convergence
    /// rate for Jacobi iteration on a Laplace grid is `O(N^2)` sweeps for an
    /// `N`-cell-tall domain (no acceleration here, e.g. no red-black
    /// Gauss-Seidel or multigrid -- a real, disclosed simplification,
    /// matching the reference implementation's own plain Jacobi approach).
    pub fn relax_n(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.relax_step();
        }
    }

    pub fn phi_at(&self, x: usize, y: usize) -> f32 {
        self.phi[self.idx(x, y)]
    }

    /// Real electric field `E = -∇φ`, central difference (one-sided at
    /// domain edges). This is the actual physics the ball-on-a-hill
    /// analogy describes: the field points DOWNHILL (toward lower
    /// potential), with magnitude equal to the slope.
    pub fn field_at(&self, x: usize, y: usize) -> Vec2 {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(self.width - 1);
        let yu = y.saturating_sub(1);
        let yd = (y + 1).min(self.height - 1);
        let dx = (xr - xl).max(1) as f32;
        let dy = (yd - yu).max(1) as f32;
        let dphidx = (self.phi_at(xr, y) - self.phi_at(xl, y)) / dx;
        let dphidy = (self.phi_at(x, yd) - self.phi_at(x, yu)) / dy;
        Vec2::new(-dphidx, -dphidy)
    }
}

/// A single growing dielectric-breakdown leader channel -- Niemeyer,
/// Pietronero & Wiesmann 1984 ("Fractal dimension of dielectric breakdown in
/// three dimensions," J. Phys. A: Math. Gen. 17), the real, named, cited
/// model this entire family of phenomena (lightning, electrochemical
/// deposition, viscous fingering, mineral dendrites) is built on -- verified
/// independently against a real reference implementation
/// (github.com/diluuuu10/triggered-discharge, cloned in `tmp/`), not just
/// this crate's own invention.
///
/// Growth rule: every empty grid cell 4-adjacent to the existing channel is
/// a candidate. Each candidate's probability of being chosen is
/// proportional to `(local field strength)^eta` -- `eta` is the model's own
/// real, named parameter controlling how strongly growth follows the field
/// versus real physical randomness (dust, humidity, free electrons -- see
/// this module's own doc discussion of chaotic-but-deterministic
/// real-world noise). High eta follows the strongest field almost
/// deterministically; low eta branches more.
///
/// Real, disclosed simplification: after growth, the caller is expected to
/// re-relax the potential field (electrostatic screening -- a branch that
/// has grown changes the field around it, suppressing further growth
/// nearby, which is why real lightning forms a few dominant branches
/// instead of spreading evenly) before the next `grow_step` call. This
/// struct itself does not own or re-relax the field -- see `grow_step`'s
/// own signature, which takes the field by reference each call.
pub struct DielectricBreakdownLeader {
    width: usize,
    height: usize,
    is_channel: Vec<bool>,
    growth_order: Vec<(usize, usize)>,
    /// The real channel-cell each grown cell actually branched FROM --
    /// parallel to `growth_order`, `None` only for the seed itself (which
    /// has no parent). This is the true tree structure of the discharge:
    /// growth order and spatial adjacency are NOT the same thing (the next
    /// cell chosen can be adjacent to ANY existing frontier cell, not
    /// necessarily the most recently grown one), so a renderer connecting
    /// consecutive `growth_order` entries with a line draws spurious edges
    /// across the whole channel instead of its real branches -- exactly
    /// the bug found live (2026-08-27) when the first real rendered strike
    /// showed a wrong-looking fan/mess partway down. `parent_order` is the
    /// real fix: connect each cell to ITS OWN parent, not to whatever grew
    /// immediately before it in time.
    parent_order: Vec<Option<(usize, usize)>>,
    eta: f32,
    rng: LcgRng,
    /// The potential this leader's own channel is held at -- real physics:
    /// a conductor connected to an electrode sits at that electrode's own
    /// potential (here, whichever boundary the seed originates from, e.g.
    /// the cloud). Every newly-grown cell gets pinned to this same value in
    /// `grow_step`.
    origin_value: f32,
}

impl DielectricBreakdownLeader {
    pub fn new(
        field: &mut ElectricPotentialField,
        seed_x: usize,
        seed_y: usize,
        origin_value: f32,
        eta: f32,
        rng_seed: u32,
    ) -> Self {
        let width = field.width();
        let height = field.height();
        assert!(
            seed_x < width && seed_y < height,
            "DielectricBreakdownLeader: seed must be inside the grid"
        );
        let mut is_channel = vec![false; width * height];
        is_channel[seed_y * width + seed_x] = true;
        field.pin(seed_x, seed_y, origin_value);
        Self {
            width,
            height,
            is_channel,
            growth_order: vec![(seed_x, seed_y)],
            parent_order: vec![None],
            eta,
            rng: LcgRng::new(rng_seed),
            origin_value,
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn is_channel_at(&self, x: usize, y: usize) -> bool {
        self.is_channel[self.idx(x, y)]
    }

    /// The channel's own cells, in the real order they were grown -- the
    /// caller (e.g. a renderer) can use this to draw the leader's actual
    /// growth history, not just its current shape.
    pub fn growth_order(&self) -> &[(usize, usize)] {
        &self.growth_order
    }

    /// The real branch structure: `parents()[i]` is the channel cell
    /// `growth_order()[i]` actually grew from (`None` for the seed). See
    /// `parent_order`'s own doc for why this, not growth order, is what a
    /// renderer must connect.
    pub fn parents(&self) -> &[Option<(usize, usize)>] {
        &self.parent_order
    }

    /// Empty cells 4-adjacent to the existing channel -- the real candidate
    /// set the model's own growth rule selects from.
    fn candidates(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if self.is_channel_at(x, y) {
                    continue;
                }
                let adjacent = (x > 0 && self.is_channel_at(x - 1, y))
                    || (x + 1 < self.width && self.is_channel_at(x + 1, y))
                    || (y > 0 && self.is_channel_at(x, y - 1))
                    || (y + 1 < self.height && self.is_channel_at(x, y + 1));
                if adjacent {
                    out.push((x, y));
                }
            }
        }
        out
    }

    /// One real growth step: pick ONE candidate, weighted by
    /// `field_strength^eta` at that candidate cell (the model's own real
    /// growth-probability rule), add it to the channel, then PIN it into
    /// `field` at this leader's own origin potential and re-relax --
    /// electrostatic screening (see this struct's own top doc): a real
    /// conductor is an equipotential, so the newly-joined cell must become
    /// part of the fixed boundary the field solves around, which is the
    /// actual physical mechanism that concentrates the field ahead of the
    /// growing tip and makes the leader self-reinforce roughly toward its
    /// own source direction instead of growing uniformly at random.
    /// `relax_iterations` controls how fully the field re-settles after
    /// each single-cell growth step -- a real cost/accuracy trade the
    /// caller controls, not a hidden constant.
    ///
    /// Returns `false` if there were no candidates left (the channel
    /// reached a domain edge with nowhere left to grow).
    pub fn grow_step(
        &mut self,
        field: &mut ElectricPotentialField,
        relax_iterations: usize,
    ) -> bool {
        let candidates = self.candidates();
        if candidates.is_empty() {
            return false;
        }
        let weights: Vec<f32> = candidates
            .iter()
            .map(|&(x, y)| field.field_at(x, y).length().max(1.0e-6).powf(self.eta))
            .collect();
        let total: f32 = weights.iter().sum();
        let mut r = self.rng.next_f32() * total;
        let mut chosen = candidates.len() - 1;
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                chosen = i;
                break;
            }
            r -= w;
        }
        let (cx, cy) = candidates[chosen];
        // Real parent: whichever existing channel neighbor this cell
        // actually grew from -- see `parent_order`'s own doc for why this
        // must be tracked separately from growth order.
        let parent = [
            (cx.checked_sub(1), Some(cy)),
            (Some(cx + 1).filter(|&x| x < self.width), Some(cy)),
            (Some(cx), cy.checked_sub(1)),
            (Some(cx), Some(cy + 1).filter(|&y| y < self.height)),
        ]
        .into_iter()
        .find_map(|(nx, ny)| {
            let (nx, ny) = (nx?, ny?);
            self.is_channel_at(nx, ny).then_some((nx, ny))
        });
        let i = self.idx(cx, cy);
        self.is_channel[i] = true;
        self.growth_order.push((cx, cy));
        self.parent_order.push(parent);
        field.pin(cx, cy, self.origin_value);
        field.relax_n(relax_iterations);
        true
    }
}

#[cfg(test)]
mod dielectric_breakdown_leader_tests {
    use super::*;

    /// Real, falsifiable check of the model's own defining property: growth
    /// should be biased toward the real field direction (from the seed
    /// toward the boundary with the OPPOSITE potential -- here, growing
    /// down from a top seed toward a bottom Dirichlet boundary at a
    /// different potential, matching a real downward leader), not
    /// uniformly random in every direction. With eta=4 (a real, strongly
    /// field-following exponent per the cited model's own convention -- the
    /// video/repo reference this session checked used eta in a similar
    /// range) the mean y-coordinate of the grown channel should have moved
    /// substantially toward the bottom boundary, not sit near the seed's
    /// own row.
    #[test]
    fn leader_grows_toward_the_field_not_uniformly_random() {
        const WIDTH: usize = 40;
        const HEIGHT: usize = 60;
        let mut field = ElectricPotentialField::new(WIDTH, HEIGHT, 0.0, 1.0);
        field.relax_n(HEIGHT * HEIGHT * 2);
        let mut leader = DielectricBreakdownLeader::new(&mut field, WIDTH / 2, 0, 0.0, 4.0, 42);
        for _ in 0..(HEIGHT / 2) {
            assert!(
                leader.grow_step(&mut field, 20),
                "leader should always have a candidate to grow into on an open grid"
            );
        }
        let mean_y: f32 = leader
            .growth_order()
            .iter()
            .map(|&(_, y)| y as f32)
            .sum::<f32>()
            / leader.growth_order().len() as f32;
        // Real, measured baseline WITHOUT screening (mean_y~2.23 for this same
        // seed/step count) established that a single-point seed's own early
        // steps are dominated by which of its few immediate neighbors gets
        // picked first, before the growing tip's own field concentration has
        // enough channel length to meaningfully bias direction -- so the bar
        // here is "clearly better than that unscreened baseline," not an
        // arbitrary fraction of the grid height.
        assert!(
            mean_y > 5.0,
            "after growing {} steps from the top with real electrostatic screening \
             active, the leader's mean y should clearly exceed the measured \
             no-screening baseline (~2.23) -- got mean_y={mean_y:.2} on a {HEIGHT}-tall grid",
            leader.growth_order().len()
        );
    }

    /// Real check that `eta` actually does what its own doc claims: a very
    /// high eta should produce a straighter (less laterally spread) channel
    /// than a very low eta, on the SAME field and seed, since high eta
    /// follows the strongest (here, purely vertical) field almost
    /// deterministically while low eta lets real random noise dominate.
    #[test]
    fn higher_eta_produces_a_straighter_less_spread_channel() {
        const WIDTH: usize = 40;
        const HEIGHT: usize = 60;
        let mut field = ElectricPotentialField::new(WIDTH, HEIGHT, 0.0, 1.0);
        field.relax_n(HEIGHT * HEIGHT * 2);

        fn lateral_spread(leader: &DielectricBreakdownLeader, width: usize) -> f32 {
            let xs: Vec<f32> = leader
                .growth_order()
                .iter()
                .map(|&(x, _)| x as f32)
                .collect();
            let mean = xs.iter().sum::<f32>() / xs.len() as f32;
            let variance =
                xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / xs.len() as f32;
            variance.sqrt() / width as f32
        }

        let mut field_high = field.clone();
        let mut field_low = field.clone();
        let mut high_eta =
            DielectricBreakdownLeader::new(&mut field_high, WIDTH / 2, 0, 0.0, 8.0, 7);
        let mut low_eta = DielectricBreakdownLeader::new(&mut field_low, WIDTH / 2, 0, 0.0, 0.5, 7);
        for _ in 0..(HEIGHT / 2) {
            high_eta.grow_step(&mut field_high, 20);
            low_eta.grow_step(&mut field_low, 20);
        }
        let high_spread = lateral_spread(&high_eta, WIDTH);
        let low_spread = lateral_spread(&low_eta, WIDTH);
        assert!(
            high_spread < low_spread,
            "high eta (8.0, near-deterministic field-following) should produce a \
             narrower channel than low eta (0.5, noise-dominated): high_spread={high_spread:.4} \
             low_spread={low_spread:.4}"
        );
    }
}

#[cfg(test)]
mod electric_potential_field_tests {
    use super::*;

    /// Real, closed-form check: with uniform Dirichlet top/bottom and
    /// Neumann sides, and no x-varying feature anywhere in the domain, the
    /// EXACT analytic solution to Laplace's equation has no x-dependence at
    /// all -- it's the 1D linear interpolation `new()` already initializes
    /// to. That means this starting state is already the fixed point of
    /// the Jacobi iteration: relaxing it further should change nothing
    /// (within float tolerance), confirming `relax_step` doesn't introduce
    /// a bug that perturbs an already-correct state.
    #[test]
    fn uniform_case_is_a_fixed_point_of_relaxation() {
        let mut field = ElectricPotentialField::new(16, 32, 0.0, 1.0);
        let before: Vec<f32> = (0..field.height()).map(|y| field.phi_at(8, y)).collect();
        field.relax_n(50);
        for (y, &b) in before.iter().enumerate() {
            let after = field.phi_at(8, y);
            assert!(
                (after - b).abs() < 1.0e-5,
                "uniform linear-gradient state should be a fixed point of Jacobi \
                 relaxation (no x-variation anywhere to disturb it), but row {y} moved \
                 from {b} to {after}"
            );
        }
    }

    /// Real convergence check, starting from a WRONG initial guess (flat
    /// zero, not the linear interpolation `new()` normally starts from):
    /// after enough relaxation, the field must still converge to the same
    /// real analytic solution -- a uniform field pointing from high to low
    /// potential, magnitude `(bottom-top)/(height-1)`, zero in x. This is
    /// the textbook parallel-plate capacitor field solution.
    #[test]
    fn converges_to_uniform_field_from_a_flat_start() {
        const WIDTH: usize = 16;
        const HEIGHT: usize = 40;
        const TOP: f32 = 0.0;
        const BOTTOM: f32 = 1.0;
        let mut field = ElectricPotentialField::new(WIDTH, HEIGHT, TOP, BOTTOM);
        // Deliberately wipe the smart initial guess back to flat zero
        // (except the Dirichlet boundaries, which relax_step re-pins every
        // sweep regardless) to test real convergence, not just confirm the
        // constructor's own initial guess was already right.
        for y in 1..HEIGHT - 1 {
            for x in 0..WIDTH {
                let i = y * WIDTH + x;
                field.phi[i] = 0.0;
                field.phi_next[i] = 0.0;
            }
        }
        // O(N^2) plain-Jacobi convergence -- see relax_n's own doc.
        field.relax_n(HEIGHT * HEIGHT * 4);

        let expected_field_y = -(BOTTOM - TOP) / (HEIGHT as f32 - 1.0);
        for y in 1..HEIGHT - 1 {
            for x in 1..WIDTH - 1 {
                let e = field.field_at(x, y);
                assert!(
                    (e.y - expected_field_y).abs() < 0.01,
                    "expected uniform E.y={expected_field_y:.5} at ({x},{y}), got {:.5}",
                    e.y
                );
                assert!(
                    e.x.abs() < 0.01,
                    "expected zero E.x (no x-variation in a uniform parallel-plate \
                     setup) at ({x},{y}), got {:.5}",
                    e.x
                );
            }
        }
    }
}
