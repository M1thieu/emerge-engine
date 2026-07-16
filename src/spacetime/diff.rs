//! Differentiable mini-solver for offline gait training.
//!
//! A self-contained, differentiable MLS-MPM forward simulation plus its
//! hand-derived reverse pass -- the trainer the whole adjoint chain in
//! `spacetime::transfer` was built toward. Structured to match the canonical
//! open-loop DiffTaichi `diffmpm.py` walker demo (verified against the real
//! cloned source, not from memory), because that is the simplest published
//! setup proven to produce visible trained locomotion:
//!
//! - **Time-varying actuation from a sinusoid basis controller** -- the
//!   trainable parameters are per-muscle-group weights over `n_waves` phase-
//!   shifted sinusoids plus a bias, squashed with tanh. Constant per-particle
//!   activation (the first prototype here) can only learn a static squeeze;
//!   a time-varying signal learns a *gait*.
//! - **Signed actuation** (`tanh` in (-1,1)): muscles both contract and
//!   extend, exactly DiffTaichi's convention (`A = [[0,0],[0,1]] * act`, both
//!   signs). NOTE: the engine's runtime muscle model
//!   (`transfer::combined_kirchhoff_stress`) is contract-only `[0,1]` -- a
//!   trained gait transfers to the runtime by remapping, this module does not
//!   change engine semantics.
//! - **Gravity + a sticky floor** as the locomotion symmetry-breaker.
//!   Verified detail from the real `diffmpm.py` source: its friction-cone
//!   code runs on an already-zeroed velocity, so the canonical walker
//!   actually trains against a *sticky* floor (grid cells at floor level
//!   moving downward get zeroed) -- which is exactly what this module
//!   implements, with the branch decision recorded forward and replayed as a
//!   fixed linear map backward (same "detach the branch" treatment as the
//!   kernel-weight kink documented throughout `spacetime::transfer`).
//! - **Actuator groups**: particles share muscle groups (legs), not one
//!   trainable scalar per particle.
//!
//! Every backward formula is either one of the individually finite-difference-
//! verified adjoints from `spacetime::transfer`/`grid`, or is derived and
//! FD-verified in this module's own tests. The one deliberate scope limit,
//! same as everywhere else in the chain: kernel weights use each step's REAL
//! recorded positions as fixed reference points (the position-dependence of
//! *which cells* a particle touches is not differentiated -- the standard
//! detached treatment; ChainQueen's own backward pass makes the same choice
//! per-step-linearization-wise for branch decisions).
//!
//! Scale/units note: this is a *training tool*, not the runtime solver. It
//! runs a small body (tens of particles) for a short horizon (~100 substeps)
//! thousands of times; the trained controller parameters are the output.

use glam::{IVec2, Mat2, Vec2};
use std::collections::{BTreeMap, BTreeSet};

use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;
use crate::materials::{MaterialModel, NeoHookeanMaterial};
use crate::particle::{Particle, Particles};
use crate::solver::config::KERNEL_D_INVERSE;
use crate::transfer::{f_update_vjp, g2p_affine_vjp, g2p_velocity_vjp, p2g_stress_vjp};

// ── Differentiable materials ──────────────────────────────────────────────────

/// A material whose passive Kirchhoff stress has a known analytic adjoint --
/// what makes it usable inside this trainer. Everything in this module was
/// hardcoded to `NeoHookeanMaterial` specifically until this generalization
/// (requested explicitly: emerge's whole design is one solver for all
/// matter, and the trainer shouldn't be the one place that's tied to a
/// single constitutive model). `NeoHookeanMaterial` is the only
/// implementation today; `CorotatedMaterial` is the concrete next target --
/// ChainQueen's real `Times_Rotated_dP_dF_FixedCorotated` (its own hand-
/// written CUDA backward pass, `linalg.h`) gives the reference formula for
/// its polar-decomposition-based stress, but deriving emerge's actual
/// `kirchhoff_stress = P*F^T` adjoint from it needs an extra product-rule
/// step (P depends on F, AND there's an explicit trailing F^T) that hasn't
/// been carefully derived+FD-verified yet -- real remaining work, not
/// silently skipped.
pub trait DifferentiableMaterial {
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2;
    fn kirchhoff_stress_vjp(&self, particles: &Particles, i: usize, d_loss_d_tau: Mat2) -> Mat2;
}

impl DifferentiableMaterial for NeoHookeanMaterial {
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        MaterialModel::kirchhoff_stress(self, particles, i)
    }
    fn kirchhoff_stress_vjp(&self, particles: &Particles, i: usize, d_loss_d_tau: Mat2) -> Mat2 {
        NeoHookeanMaterial::kirchhoff_stress_vjp(self, particles, i, d_loss_d_tau)
    }
}

// ── Body plan ─────────────────────────────────────────────────────────────────

/// Rest layout of a trainable body: particle positions, which muscle group
/// (if any) each particle belongs to, and each group's own fiber direction.
///
/// `fiber_dir` moved here from a single global config value (`DiffConfig`
/// used to carry one shared `Vec2::Y` for the whole body) after a real
/// diagnostic: the walker's trained gait measurably bounced rather than
/// walked (vertical velocity^2 1.76x horizontal, mean-height swing 0.94
/// grid units over a ~1.5-unit-tall body) -- because EVERY group could only
/// push straight up/down, net horizontal drift could only emerge indirectly
/// through the sticky floor's timing, which is inherently a pogo motion,
/// not a step. Cross-checked against EvoGym's real, published, walking
/// voxel robots (Bhatia et al. 2021, source in `evogym/utils.py`): their
/// voxels come in two actuator types, `H_ACT` (horizontal) and `V_ACT`
/// (vertical) -- real walkers mix both, using vertical actuators for
/// stance/lift and horizontal actuators for push-off (the actual Newton's-
/// third-law mechanism real legged locomotion uses: push the ground
/// backward, the ground pushes the body forward). `signed_active_stress`
/// already took `fiber_dir` as a parameter, so this is a real fix, not new
/// derivation -- no new adjoint math, just plumbing a per-group value
/// through where a global constant was hardcoded before.
pub struct BodyPlan {
    pub positions: Vec<Vec2>,
    /// Muscle group per particle; `None` = passive tissue (torso).
    pub group: Vec<Option<usize>>,
    pub n_groups: usize,
    /// Fiber direction per muscle group (indexed by group id).
    pub fiber_dir: Vec<Vec2>,
}

impl BodyPlan {
    /// DiffTaichi-`robot()`-style walker: a passive torso slab with four
    /// actuated legs hanging under its ends, one muscle group per leg, ALL
    /// vertical fiber -- kept for comparison against `biped` (the reference
    /// walker's own vertical-only convention, real but measurably bouncy).
    /// `origin` is the lower-left corner of the *legs*, in grid coords;
    /// `spacing` the particle spacing.
    pub fn walker(origin: Vec2, spacing: f32) -> Self {
        let mut positions = Vec::new();
        let mut group = Vec::new();

        // Four legs, 2 columns x 3 rows each, at columns {0-1, 2-3, 8-9, 10-11}.
        let leg_cols: [(usize, usize); 4] = [(0, 0), (2, 1), (8, 2), (10, 3)];
        for (col0, g) in leg_cols {
            for c in 0..2 {
                for r in 0..3 {
                    positions.push(origin + Vec2::new((col0 + c) as f32, r as f32) * spacing);
                    group.push(Some(g));
                }
            }
        }

        // Torso: 12 columns x 3 rows sitting on top of the legs, passive.
        for c in 0..12 {
            for r in 0..3 {
                positions.push(origin + Vec2::new(c as f32, (3 + r) as f32) * spacing);
                group.push(None);
            }
        }

        Self {
            positions,
            group,
            n_groups: 4,
            fiber_dir: vec![Vec2::Y; 4],
        }
    }

    /// Two-legged biped, each leg split into a THIGH (upper, vertical fiber
    /// -- lift/stance) and a FOOT (lower, horizontal fiber -- push-off),
    /// mirroring EvoGym's real V_ACT/H_ACT mix. 4 muscle groups total:
    /// left-thigh, left-foot, right-thigh, right-foot. `origin` is the
    /// lower-left corner of the feet.
    pub fn biped(origin: Vec2, spacing: f32) -> Self {
        let mut positions = Vec::new();
        let mut group = Vec::new();

        const LEFT_THIGH: usize = 0;
        const LEFT_FOOT: usize = 1;
        const RIGHT_THIGH: usize = 2;
        const RIGHT_FOOT: usize = 3;

        // Each leg: 2 columns wide. Foot = bottom 2 rows (horizontal fiber),
        // thigh = next 3 rows up (vertical fiber). Legs at columns {0-1} and
        // {6-7}, leaving a gap for a natural stride stance.
        let legs: [(usize, usize, usize); 2] =
            [(0, LEFT_FOOT, LEFT_THIGH), (6, RIGHT_FOOT, RIGHT_THIGH)];
        for (col0, foot_g, thigh_g) in legs {
            for c in 0..2 {
                for r in 0..2 {
                    positions.push(origin + Vec2::new((col0 + c) as f32, r as f32) * spacing);
                    group.push(Some(foot_g));
                }
                for r in 0..3 {
                    positions.push(origin + Vec2::new((col0 + c) as f32, (2 + r) as f32) * spacing);
                    group.push(Some(thigh_g));
                }
            }
        }

        // Torso: spans both legs, sitting on top, passive.
        for c in 0..8 {
            for r in 0..3 {
                positions.push(origin + Vec2::new(c as f32, (5 + r) as f32) * spacing);
                group.push(None);
            }
        }

        Self {
            positions,
            group,
            n_groups: 4,
            fiber_dir: vec![
                Vec2::Y, // left thigh: vertical (stance/lift)
                Vec2::X, // left foot: horizontal (push-off)
                Vec2::Y, // right thigh: vertical
                Vec2::X, // right foot: horizontal
            ],
        }
    }
}

// ── Config / controller / state ───────────────────────────────────────────────

pub struct DiffConfig {
    pub mass: f32,
    /// P2G stress premultiplier: `-V0 * KERNEL_D_INVERSE * dt` in the real
    /// solver; a free constant here.
    pub stress_coeff: f32,
    pub dt: f32,
    pub kernel_d_inverse: f32,
    pub apic_blend: f32,
    /// Downward gravitational acceleration (grid units / s^2).
    pub gravity: f32,
    /// Sticky-floor height: grid cells at `y <= floor_y` moving downward get
    /// their velocity zeroed (the verified real behavior of DiffTaichi's
    /// canonical walker floor).
    pub floor_y: f32,
    /// Active-stress scale (DiffTaichi's `act_strength`).
    pub act_strength: f32,
    /// Sinusoid basis size per group.
    pub n_waves: usize,
    /// Gait angular frequency (rad/s of *simulated* time).
    pub omega: f32,
    /// Training loss averages drift over the LAST this-many states of the
    /// rollout (1 = final state only). A final-state-only loss rewards
    /// ending far right by any means -- including ballistic hops; a window
    /// rewards sustained progress. See `controller_gradient`.
    pub loss_window: usize,
    /// Coefficient penalizing mean squared vertical velocity across the
    /// WHOLE rollout, added to the training loss. Real root cause found
    /// live 2026-07-11: `loss_window` alone rewards "consistently far right
    /// through sustained contact" but does not PENALIZE vertical motion
    /// itself -- if a ballistic hop still covers more ground per unit loss
    /// than a grounded gait, gradient descent takes the hop regardless of
    /// how the actuators are arranged (confirmed: adding horizontal
    /// push-off muscles alone, see `BodyPlan::biped`, improved vy/vx from
    /// 1.76 to 1.46-1.58 but left contact fraction at ~0.22-0.23 once
    /// actuation was strong enough to move any real distance). Direct
    /// penalty on vertical velocity is the standard fix in published
    /// legged-locomotion reward functions (torso-height/vertical-velocity
    /// penalties are near-universal there) -- 0.0 disables it (backward
    /// compatible default).
    pub bounce_penalty: f32,
    /// Coefficient penalizing mean squared activation across the whole
    /// rollout and all groups -- the "torque cost" half of standard
    /// locomotion reward shaping (`bounce_penalty` is the other half).
    /// Discourages the controller from firing muscles harder than needed,
    /// which tends to produce smoother, less erratic gaits as a side
    /// effect. 0.0 disables it (backward compatible default).
    pub control_effort_penalty: f32,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            mass: 1.0,
            stress_coeff: -0.05,
            dt: 0.01,
            kernel_d_inverse: KERNEL_D_INVERSE,
            apic_blend: 1.0,
            gravity: 2.0,
            floor_y: 1.0,
            act_strength: 12.0,
            n_waves: 4,
            omega: 16.0,
            loss_window: 1,
            bounce_penalty: 0.0,
            control_effort_penalty: 0.0,
        }
    }
}

/// Open-loop sinusoid-basis controller (DiffTaichi `compute_actuation`):
/// `act[t, g] = tanh( sum_j w[g,j] * sin(omega * t * dt + 2*pi*j/n_waves) + b[g] )`.
#[derive(Clone)]
pub struct SinusoidController {
    /// Row-major `[group][wave]`.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub n_groups: usize,
    pub n_waves: usize,
    /// Bilateral-symmetry constraint: if `mirror_of[g] = Some(s)`, group `g`
    /// reuses group `s`'s weights/bias (a mirrored, not independent, muscle)
    /// instead of having its own free parameters. `None` = free/trainable.
    ///
    /// Real technique, found live 2026-07-11 after a diagnosed failure: a
    /// trained biped with fully independent left/right controllers found a
    /// ONE-LEGGED HOP (one leg permanently retracted, the other doing all
    /// the work) -- nothing in a pure drift/bounce/effort loss requires the
    /// two legs to alternate, and that degenerate solution is simpler for
    /// gradient descent to find than genuine alternation. Cross-checked
    /// against EvoSoro's real, published soft-robot evolution source
    /// (`evosoro/networks.py`, `enforce_symmetry()`): it mirrors left/right
    /// genome parameters structurally so an asymmetric solution can't even
    /// be represented, rather than hoping a loss term discourages it.
    /// Combined here with `phase_offset` (standard CPG anti-phase coupling
    /// for ALTERNATING, not synchronized, gaits): mirroring alone would make
    /// both legs move identically in phase (a two-legged synchronized hop,
    /// not a walk); the phase offset is what turns that into alternation.
    pub mirror_of: Vec<Option<usize>>,
    /// Extra phase (radians) added to group `g`'s sinusoid argument.
    pub phase_offset: Vec<f32>,
}

impl SinusoidController {
    /// Small deterministic pseudo-random init (DiffTaichi uses N(0, 0.01);
    /// this uses a hash-based equivalent so runs reproduce exactly). No
    /// bilateral symmetry by default (`mirror_of` all `None`, `phase_offset`
    /// all 0) -- fully independent per-group parameters, as before.
    pub fn seeded(n_groups: usize, n_waves: usize) -> Self {
        Self::seeded_with(n_groups, n_waves, 0)
    }

    /// Same as `seeded`, but with an explicit `seed` -- every call to
    /// `seeded` (no seed argument) used the SAME index-only hash for the
    /// entire session, meaning every hyperparameter sweep started from the
    /// literal same initial weights every time. Non-convex training
    /// standardly needs multiple random restarts, not just hyperparameter
    /// search over a single fixed starting point -- real gap, found late
    /// 2026-07-11 after several sweeps converged to different DEGENERATE
    /// solutions (frozen, one-legged hop, monotonic tilt) without ever
    /// trying a different basin of attraction.
    pub fn seeded_with(n_groups: usize, n_waves: usize, seed: u32) -> Self {
        let rand = |i: usize| -> f32 {
            let x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            let x = x ^ (x >> 15);
            let x = x.wrapping_mul(0x27d4_eb2d);
            let x = x ^ (x >> 15);
            ((x % 2000) as f32 / 1000.0 - 1.0) * 0.01
        };
        Self {
            weights: (0..n_groups * n_waves).map(rand).collect(),
            bias: vec![0.0; n_groups],
            n_groups,
            n_waves,
            mirror_of: vec![None; n_groups],
            phase_offset: vec![0.0; n_groups],
        }
    }

    /// Configures bilateral symmetry matching `BodyPlan::biped`'s group
    /// layout (0=left-thigh, 1=left-foot, 2=right-thigh, 3=right-foot):
    /// the right leg's groups mirror the left leg's, anti-phase (half a
    /// gait cycle apart) -- see `mirror_of`'s doc for why both pieces
    /// (mirroring AND the phase offset) are needed together.
    pub fn with_biped_symmetry(mut self) -> Self {
        assert_eq!(self.n_groups, 4, "biped symmetry needs exactly 4 groups");
        self.mirror_of = vec![None, None, Some(0), Some(1)];
        self.phase_offset = vec![0.0, 0.0, std::f32::consts::PI, std::f32::consts::PI];
        self
    }

    fn pre_activation(&self, cfg: &DiffConfig, t: usize, g: usize) -> f32 {
        let src = self.mirror_of[g].unwrap_or(g);
        let time = t as f32 * cfg.dt;
        let extra_phase = self.phase_offset[g];
        let mut pre = self.bias[src];
        for j in 0..self.n_waves {
            let phase = 2.0 * std::f32::consts::PI * j as f32 / self.n_waves as f32;
            pre += self.weights[src * self.n_waves + j]
                * (cfg.omega * time + phase + extra_phase).sin();
        }
        pre
    }

    /// Signed activation in (-1, 1) for group `g` at substep `t`.
    pub fn activation(&self, cfg: &DiffConfig, t: usize, g: usize) -> f32 {
        self.pre_activation(cfg, t, g).tanh()
    }

    /// All groups' activations at substep `t`.
    pub fn activations(&self, cfg: &DiffConfig, t: usize) -> Vec<f32> {
        (0..self.n_groups)
            .map(|g| self.activation(cfg, t, g))
            .collect()
    }
}

/// Closed-loop state-feedback controller -- ChainQueen's real `walker_2d.py`
/// design (verified against the real source, `demos/walker_2d.py`): each
/// muscle group's mean position (relative to the body's own centroid, for
/// translation invariance) and mean velocity feed ONE shared linear layer +
/// tanh, producing all groups' activations together (so one group's muscle
/// can depend on ANY group's sensed state, not just a private clock phase).
///
/// Built after `SinusoidController` (open-loop, time-driven) repeatedly
/// collapsed into degenerate gaits -- frozen, one-legged hop, monotonic
/// tip-over -- across a 12-seed search at its best-found hyperparameters,
/// even with bilateral symmetry and anti-phase coupling. That's real
/// evidence the missing piece isn't more tuning: an open-loop clock can't
/// sense and correct for what the body is actually doing; a feedback
/// controller can.
///
/// Per-group feature layout: `[rel_x, rel_y, vel_x, vel_y]`, concatenated
/// group-major -- `feature_len() = n_groups * 4`.
#[derive(Clone)]
pub struct FeedbackController {
    /// Row-major `[output_group][input_feature]`.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub n_groups: usize,
}

impl FeedbackController {
    pub fn feature_len(n_groups: usize) -> usize {
        n_groups * 4
    }

    /// Deterministic pseudo-random init, same hash family as
    /// `SinusoidController::seeded_with` (small magnitude, reproducible).
    pub fn seeded_with(n_groups: usize, seed: u32) -> Self {
        let flen = Self::feature_len(n_groups);
        let rand = |i: usize| -> f32 {
            let x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503).wrapping_add(1));
            let x = x ^ (x >> 15);
            let x = x.wrapping_mul(0x27d4_eb2d);
            let x = x ^ (x >> 15);
            ((x % 2000) as f32 / 1000.0 - 1.0) * 0.01
        };
        Self {
            weights: (0..n_groups * flen).map(rand).collect(),
            bias: vec![0.0; n_groups],
            n_groups,
        }
    }

    /// Per-group mean position (relative to the body centroid) and mean
    /// velocity, flattened group-major. Also returns each group's particle
    /// count (needed by the backward pass to distribute gradient back to
    /// individual particles) and the total particle count (for the
    /// centroid's own gradient).
    fn features(plan: &BodyPlan, state: &DiffState) -> (Vec<f32>, Vec<usize>) {
        let n_groups = plan.n_groups;
        let mut sum_x = vec![Vec2::ZERO; n_groups];
        let mut sum_v = vec![Vec2::ZERO; n_groups];
        let mut count = vec![0usize; n_groups];
        let mut centroid = Vec2::ZERO;
        for (i, group) in plan.group.iter().enumerate() {
            centroid += state.x[i];
            if let Some(g) = *group {
                sum_x[g] += state.x[i];
                sum_v[g] += state.v[i];
                count[g] += 1;
            }
        }
        centroid /= plan.positions.len() as f32;

        let mut feat = vec![0.0f32; n_groups * 4];
        for g in 0..n_groups {
            let n = count[g].max(1) as f32;
            let rel = sum_x[g] / n - centroid;
            let vel = sum_v[g] / n;
            feat[g * 4] = rel.x;
            feat[g * 4 + 1] = rel.y;
            feat[g * 4 + 2] = vel.x;
            feat[g * 4 + 3] = vel.y;
        }
        (feat, count)
    }

    fn activations_from_features(&self, feat: &[f32]) -> Vec<f32> {
        let flen = feat.len();
        (0..self.n_groups)
            .map(|g| {
                let mut pre = self.bias[g];
                for (k, &fk) in feat.iter().enumerate() {
                    pre += self.weights[g * flen + k] * fk;
                }
                pre.tanh()
            })
            .collect()
    }

    /// All groups' activations given the body's CURRENT state (read at the
    /// start of the substep whose stress they'll drive).
    pub fn activations(&self, plan: &BodyPlan, state: &DiffState) -> Vec<f32> {
        let (feat, _) = Self::features(plan, state);
        self.activations_from_features(&feat)
    }

    /// Adjoint of the whole feature-extraction + linear + tanh pipeline.
    /// Given `g_act` (gradient flowing back from each group's activation,
    /// already summed with whatever downstream physics/penalty terms
    /// contribute to it -- same role as `SinusoidController`'s `g_act` in
    /// its own tanh chain), returns the controller parameter gradients
    /// AND, critically, each PARTICLE's gradient contribution for having
    /// been read as an input to this controller -- these must be ADDED to
    /// the position/velocity gradients already flowing from the physics
    /// chain for this same substep's state, not treated as a separate path.
    ///
    /// Derivation, in order:
    /// 1. `d_pre[g] = g_act[g] * (1 - act[g]^2)` (tanh derivative, same as
    ///    `SinusoidController`).
    /// 2. Linear layer adjoint (standard `Y = W*x + b`):
    ///    `d_weights[g,k] = d_pre[g] * feat[k]`, `d_bias[g] = d_pre[g]`,
    ///    `d_feat[k] = sum_g W[g,k] * d_pre[g]` (`W^T * d_pre`).
    /// 3. Unpack `d_feat` per group into `(d_rel, d_vel)`:
    ///    - `d_vel` distributes evenly to every particle in that group:
    ///      `d(v[i]) += d_vel / count[g]`.
    ///    - `rel = mean_x[g] - centroid` is a difference, so its adjoint
    ///      splits two ways: `d(mean_x[g]) += d_rel` (direct term,
    ///      distributed evenly to the group's own particles) AND
    ///      `d(centroid) -= d_rel`, accumulated across EVERY group (since
    ///      centroid feeds every group's `rel` term) then distributed
    ///      EVENLY TO EVERY PARTICLE IN THE BODY (not just one group's --
    ///      centroid is a mean over all particles, passive torso included).
    fn backward(
        &self,
        plan: &BodyPlan,
        feat: &[f32],
        count: &[usize],
        g_act: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<Vec2>, Vec<Vec2>) {
        let flen = feat.len();
        let n_particles = plan.positions.len();
        let mut g_weights = vec![0.0f32; self.weights.len()];
        let mut g_bias = vec![0.0f32; self.n_groups];
        let mut g_feat = vec![0.0f32; flen];

        for g in 0..self.n_groups {
            let act = self.activations_from_features(feat)[g];
            let d_pre = g_act[g] * (1.0 - act * act);
            for (k, &fk) in feat.iter().enumerate() {
                g_weights[g * flen + k] += d_pre * fk;
                g_feat[k] += self.weights[g * flen + k] * d_pre;
            }
            g_bias[g] += d_pre;
        }

        let mut g_x = vec![Vec2::ZERO; n_particles];
        let mut g_v = vec![Vec2::ZERO; n_particles];
        let mut g_centroid = Vec2::ZERO;

        for g in 0..self.n_groups {
            let n = count[g].max(1) as f32;
            let g_rel = Vec2::new(g_feat[g * 4], g_feat[g * 4 + 1]);
            let g_vel = Vec2::new(g_feat[g * 4 + 2], g_feat[g * 4 + 3]);
            g_centroid -= g_rel;
            for (i, group) in plan.group.iter().enumerate() {
                if *group == Some(g) {
                    g_x[i] += g_rel / n;
                    g_v[i] += g_vel / n;
                }
            }
        }

        let per_particle_centroid = g_centroid / n_particles as f32;
        for gx in g_x.iter_mut() {
            *gx += per_particle_centroid;
        }

        (g_weights, g_bias, g_x, g_v)
    }
}

/// Per-particle dynamic state of the mini-sim.
#[derive(Clone)]
pub struct DiffState {
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    pub c: Vec<Mat2>,
    pub f: Vec<Mat2>,
}

impl DiffState {
    pub fn rest(plan: &BodyPlan) -> Self {
        let n = plan.positions.len();
        Self {
            x: plan.positions.clone(),
            v: vec![Vec2::ZERO; n],
            c: vec![Mat2::ZERO; n],
            f: vec![Mat2::IDENTITY; n],
        }
    }

    pub fn mean_x(&self) -> f32 {
        self.x.iter().map(|p| p.x).sum::<f32>() / self.x.len() as f32
    }
}

/// What one forward substep records for its backward pass: the sticky-floor
/// branch decisions (everything else is recomputed from the stored states).
pub struct StepRecord {
    stuck: BTreeSet<(i32, i32)>,
}

// ── Signed active stress ──────────────────────────────────────────────────────

/// Signed directional active stress `act * strength * F * (n0 x n0) * F^T` --
/// DiffTaichi's exact actuation form (`new_F @ A @ new_F.transpose()` with
/// `A = diag(0, act)` for a Y fiber), allowing both contraction and extension.
/// Distinct from the engine's contract-only runtime form on purpose; see the
/// module docs.
fn signed_active_stress(f: Mat2, act: f32, strength: f32, fiber_dir: Vec2) -> Mat2 {
    let len_sq = fiber_dir.dot(fiber_dir);
    if len_sq <= f32::EPSILON {
        return Mat2::ZERO;
    }
    let n0 = fiber_dir / len_sq.sqrt();
    let a_mat = Mat2::from_cols(n0 * n0.x, n0 * n0.y) * (act * strength);
    f * a_mat * f.transpose()
}

/// VJP of `signed_active_stress` w.r.t. `F` and `act`. Same derivation as
/// `transfer::active_stress_vjp` (`Y = F*A*F^T` with symmetric `A`:
/// `dL/dF = (G + G^T) * F * A`; `dL/dact = strength * (G : F*(n0 x n0)*F^T)`),
/// minus the engine's `act <= 0` guard -- signed actuation must keep its
/// gradient on both sides of zero. FD-verified in this module's tests.
fn signed_active_stress_vjp(
    f: Mat2,
    act: f32,
    strength: f32,
    fiber_dir: Vec2,
    g: Mat2,
) -> (Mat2, f32) {
    let len_sq = fiber_dir.dot(fiber_dir);
    if len_sq <= f32::EPSILON {
        return (Mat2::ZERO, 0.0);
    }
    let n0 = fiber_dir / len_sq.sqrt();
    let a_unit = Mat2::from_cols(n0 * n0.x, n0 * n0.y);

    let k_mat = f * a_unit * f.transpose();
    let d_loss_d_act = strength
        * (g.x_axis.x * k_mat.x_axis.x
            + g.x_axis.y * k_mat.x_axis.y
            + g.y_axis.x * k_mat.y_axis.x
            + g.y_axis.y * k_mat.y_axis.y);

    let g_sym = g + g.transpose();
    let d_loss_d_f = (act * strength) * (g_sym * f * a_unit);

    (d_loss_d_f, d_loss_d_act)
}

// ── Stress evaluation via the engine material ─────────────────────────────────

/// Reusable single-particle scratch so per-particle stress/VJP evaluation
/// doesn't allocate a fresh SoA per call (this runs particles x substeps x
/// training-iterations times).
///
/// Holds the material as `Box<dyn DifferentiableMaterial>` (dynamic
/// dispatch), not a generic type parameter -- this trainer's ~20 other
/// functions all take `&mut StressEval` without needing to know or
/// propagate which material is inside, and boxing keeps every one of those
/// signatures unchanged while still letting `new` accept ANY differentiable
/// material, not just `NeoHookeanMaterial`. The per-particle call overhead
/// of dynamic dispatch is irrelevant here (training-time tool, not the
/// real-time solver).
pub struct StressEval {
    scratch: Particles,
    mat: Box<dyn DifferentiableMaterial>,
}

impl StressEval {
    pub fn new(mat: impl DifferentiableMaterial + 'static) -> Self {
        let mut scratch = Particles::default();
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        p.deformation_gradient = Mat2::IDENTITY;
        scratch.push(p);
        Self {
            scratch,
            mat: Box::new(mat),
        }
    }

    fn passive(&mut self, f: Mat2) -> Mat2 {
        self.scratch.deformation_gradient[0] = f;
        self.mat.kirchhoff_stress(&self.scratch, 0)
    }

    fn passive_vjp(&mut self, f: Mat2, g: Mat2) -> Mat2 {
        self.scratch.deformation_gradient[0] = f;
        self.mat.kirchhoff_stress_vjp(&self.scratch, 0, g)
    }
}

// ── Forward ───────────────────────────────────────────────────────────────────

fn total_stress(
    eval: &mut StressEval,
    f: Mat2,
    act: f32,
    fiber_dir: Vec2,
    cfg: &DiffConfig,
) -> Mat2 {
    eval.passive(f) + signed_active_stress(f, act, cfg.act_strength, fiber_dir)
}

/// One full differentiable substep: P2G scatter -> grid update (gravity +
/// sticky floor) -> G2P gather -> position/F update. `acts` is one signed
/// activation per muscle group at this substep.
pub fn forward_substep(
    state: &DiffState,
    plan: &BodyPlan,
    acts: &[f32],
    eval: &mut StressEval,
    cfg: &DiffConfig,
) -> (DiffState, StepRecord) {
    let n = state.x.len();
    let mut momentum_map: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    let mut mass_map: BTreeMap<(i32, i32), f32> = BTreeMap::new();

    for i in 0..n {
        let act = plan.group[i].map_or(0.0, |g| acts[g]);
        let fiber = plan.group[i].map_or(Vec2::Y, |g| plan.fiber_dir[g]);
        let stress = total_stress(eval, state.f[i], act, fiber, cfg);
        let w = quadratic_weights(state.x[i]);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = w.wx[gx] * w.wy[gy];
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - state.x[i] + Vec2::splat(0.5);
                let momentum = weight
                    * (cfg.mass * (state.v[i] + state.c[i] * cell_dist)
                        + cfg.stress_coeff * (stress * cell_dist));
                let key = (cell_pos.x, cell_pos.y);
                *momentum_map.entry(key).or_insert(Vec2::ZERO) += momentum;
                *mass_map.entry(key).or_insert(0.0) += weight * cfg.mass;
            }
        }
    }

    let mut vel_map: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    let mut stuck = BTreeSet::new();
    for (&cell, &mass_c) in mass_map.iter() {
        // Zero-weight guard, same as the real `Grid::update_velocities`.
        if mass_c <= 0.0 {
            continue;
        }
        let mut v = momentum_map[&cell] / mass_c;
        v.y -= cfg.gravity * cfg.dt;
        // Sticky floor (verified-real DiffTaichi walker behavior): a floor
        // cell moving downward loses its velocity entirely.
        if (cell.1 as f32) <= cfg.floor_y && v.y < 0.0 {
            v = Vec2::ZERO;
            stuck.insert(cell);
        }
        vel_map.insert(cell, v);
    }

    let mut next = DiffState {
        x: Vec::with_capacity(n),
        v: Vec::with_capacity(n),
        c: Vec::with_capacity(n),
        f: Vec::with_capacity(n),
    };
    for (&x, &f) in state.x.iter().zip(state.f.iter()) {
        let w = quadratic_weights(x);
        let mut new_v = Vec2::ZERO;
        let mut b = Mat2::ZERO;
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = w.wx[gx] * w.wy[gy];
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let v_cell = *vel_map
                    .get(&(cell_pos.x, cell_pos.y))
                    .unwrap_or(&Vec2::ZERO);
                let weighted = v_cell * weight;
                new_v += weighted;
                b += Mat2::from_cols(weighted * cell_dist.x, weighted * cell_dist.y);
            }
        }
        let new_c = b * (cfg.kernel_d_inverse * cfg.apic_blend);
        next.x.push(x + new_v * cfg.dt);
        next.v.push(new_v);
        next.c.push(new_c);
        next.f.push((Mat2::IDENTITY + cfg.dt * new_c) * f);
    }

    (next, StepRecord { stuck })
}

/// Full rollout: `steps` substeps from the rest state, returning per-step
/// results and the per-step group activations used (cached for backward).
pub fn rollout(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<(DiffState, StepRecord)>, Vec<Vec<f32>>) {
    let mut history = Vec::with_capacity(steps);
    let mut acts_cache = Vec::with_capacity(steps);
    let mut current = DiffState::rest(plan);
    for t in 0..steps {
        let acts = controller.activations(cfg, t);
        let (next, record) = forward_substep(&current, plan, &acts, eval, cfg);
        history.push((next.clone(), record));
        acts_cache.push(acts);
        current = next;
    }
    (history, acts_cache)
}

/// Same rollout, but activation comes from the CURRENT state
/// (`FeedbackController::activations`) instead of a fixed time-based
/// rhythm -- reuses `forward_substep` unchanged (it already takes
/// activations as a plain slice, indifferent to their source).
pub fn rollout_feedback(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<(DiffState, StepRecord)>, Vec<Vec<f32>>) {
    let mut history = Vec::with_capacity(steps);
    let mut acts_cache = Vec::with_capacity(steps);
    let mut current = DiffState::rest(plan);
    for _ in 0..steps {
        let acts = controller.activations(plan, &current);
        let (next, record) = forward_substep(&current, plan, &acts, eval, cfg);
        history.push((next.clone(), record));
        acts_cache.push(acts);
        current = next;
    }
    (history, acts_cache)
}

// ── Backward ──────────────────────────────────────────────────────────────────

/// Gradients flowing INTO one substep from everything after it.
struct IncomingGrad<'a> {
    x: &'a [Vec2],
    v: &'a [Vec2],
    c: &'a [Mat2],
    f: &'a [Mat2],
}

/// Gradients flowing OUT of one substep to the step before it, plus this
/// substep's own per-group activation gradient.
struct OutgoingGrad {
    x: Vec<Vec2>,
    v: Vec<Vec2>,
    c: Vec<Mat2>,
    f: Vec<Mat2>,
    act: Vec<f32>,
}

/// Everything the backward pass needs about ONE recorded forward substep.
struct SubstepCtx<'a> {
    state: &'a DiffState,
    next_state: &'a DiffState,
    record: &'a StepRecord,
    acts: &'a [f32],
}

/// Backward through one substep, built from the individually FD-verified
/// adjoints in `spacetime::transfer`/`grid` plus this module's own
/// FD-verified glue:
/// - position update `x' = x + v'*dt`: `g_v' += g_x'*dt` and (identity term)
///   `g_x += g_x'` -- the chain the constant-activation prototype could skip
///   (its velocity transients died against elastic restoring forces) but a
///   time-varying gait cannot;
/// - gravity: additive constant, gradient passes through unchanged;
/// - sticky floor: cells recorded stuck forward pass NO gradient back through
///   their velocity (their output was the constant zero).
fn backward_substep(
    ctx: &SubstepCtx,
    plan: &BodyPlan,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    incoming: &IncomingGrad,
) -> OutgoingGrad {
    let SubstepCtx {
        state,
        next_state,
        record,
        acts,
    } = *ctx;
    let n = state.x.len();

    // Position/F/C bookkeeping at the output side.
    let mut g_v_total = vec![Vec2::ZERO; n];
    let mut g_c_total = vec![Mat2::ZERO; n];
    let mut g_f_old_running = vec![Mat2::ZERO; n];
    for i in 0..n {
        g_v_total[i] = incoming.v[i] + incoming.x[i] * cfg.dt;
        let (g_c_from_f, g_f_old_a) =
            f_update_vjp(next_state.c[i], state.f[i], cfg.dt, incoming.f[i]);
        g_c_total[i] = incoming.c[i] + g_c_from_f;
        g_f_old_running[i] = g_f_old_a;
    }

    // G2P transpose: per-cell velocity gradient (post-floor).
    let mut g_vel_post: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    for (i, &x) in state.x.iter().enumerate() {
        let g_from_c = g2p_affine_vjp(x, cfg.kernel_d_inverse, cfg.apic_blend, g_c_total[i]);
        let g_from_v = g2p_velocity_vjp(x, g_v_total[i]);
        let w = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                *g_vel_post
                    .entry((cell_pos.x, cell_pos.y))
                    .or_insert(Vec2::ZERO) += g_from_c[gx][gy] + g_from_v[gx][gy];
            }
        }
    }

    // Grid update backward: sticky floor kills gradient; gravity is a
    // constant shift (pass-through); then velocity = momentum/mass.
    let mut mass_map: BTreeMap<(i32, i32), f32> = BTreeMap::new();
    for &x in state.x.iter() {
        let w = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = w.wx[gx] * w.wy[gy];
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                *mass_map.entry((cell_pos.x, cell_pos.y)).or_insert(0.0) += weight * cfg.mass;
            }
        }
    }
    let mut g_momentum_map: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    for (&cell, &g_v_cell) in g_vel_post.iter() {
        if record.stuck.contains(&cell) {
            continue; // forward output was the constant zero
        }
        let mass_c = mass_map[&cell];
        if mass_c <= 0.0 {
            continue; // never produced a real velocity forward
        }
        let (g_m, _g_mass) = Grid::update_velocities_vjp(Vec2::ZERO, mass_c, g_v_cell);
        g_momentum_map.insert(cell, g_m);
    }

    // P2G backward per particle.
    let mut out = OutgoingGrad {
        x: incoming.x.to_vec(), // identity term of x' = x + v'*dt
        v: vec![Vec2::ZERO; n],
        c: vec![Mat2::ZERO; n],
        f: vec![Mat2::ZERO; n],
        act: vec![0.0; plan.n_groups],
    };
    for (i, &x) in state.x.iter().enumerate() {
        let w = quadratic_weights(x);
        let mut g_momentum_local = [[Vec2::ZERO; 3]; 3];
        for (gx, row) in g_momentum_local.iter_mut().enumerate() {
            for (gy, cell) in row.iter_mut().enumerate() {
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                *cell = *g_momentum_map
                    .get(&(cell_pos.x, cell_pos.y))
                    .unwrap_or(&Vec2::ZERO);
            }
        }

        let g_stress = p2g_stress_vjp(x, cfg.stress_coeff, &g_momentum_local);
        let g_c_from_p2g = p2g_stress_vjp(x, cfg.mass, &g_momentum_local);

        let mut g_v_accum = Vec2::ZERO;
        for (gx, wx) in w.wx.iter().enumerate() {
            for (gy, wy) in w.wy.iter().enumerate() {
                g_v_accum += (wx * wy) * cfg.mass * g_momentum_local[gx][gy];
            }
        }

        let g_f_passive = eval.passive_vjp(state.f[i], g_stress);
        let act = plan.group[i].map_or(0.0, |g| acts[g]);
        let fiber = plan.group[i].map_or(Vec2::Y, |g| plan.fiber_dir[g]);
        let (g_f_active, g_act) =
            signed_active_stress_vjp(state.f[i], act, cfg.act_strength, fiber, g_stress);

        out.v[i] = g_v_accum;
        out.c[i] = g_c_from_p2g;
        out.f[i] = g_f_old_running[i] + g_f_passive + g_f_active;
        if let Some(g) = plan.group[i] {
            out.act[g] += g_act;
        }
    }

    out
}

/// Gradient of the locomotion loss `L = -(mean_x(final) - mean_x(rest))`
/// w.r.t. the controller's weights and bias, via full backprop through time.
pub fn controller_gradient(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = plan.positions.len();
    // Average the drift over the last `loss_window` states instead of
    // reading only the final one: a final-state-only loss rewards ending
    // far right by ANY means, and gradient descent found the exploit --
    // ballistic end-of-rollout hops ("goes flying, not walking", observed
    // live 2026-07-11). Averaging over a window rewards being consistently
    // far right through sustained ground contact instead. dL/dx for each
    // windowed state is -1/(n*K); the position-identity chain in
    // `backward_substep` accumulates them correctly across steps.
    let window = cfg.loss_window.clamp(1, steps);
    let per_state = -1.0 / (n as f32 * window as f32);
    // Bounce penalty: mean squared vertical velocity over the WHOLE rollout
    // (not just the drift window) -- bouncing anywhere should be
    // discouraged, not only near the end. dL/dv.y = 2*lambda*v.y/(n*steps).
    // See `DiffConfig::bounce_penalty` for why this exists (a window alone
    // doesn't stop gradient descent from choosing a hop over a walk).
    let bounce_coeff = 2.0 * cfg.bounce_penalty / (n as f32 * steps as f32);
    backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, next_state, seed| {
            if t >= steps - window {
                for g in seed.x.iter_mut() {
                    g.x += per_state;
                }
            }
            if bounce_coeff != 0.0 {
                for (g, s) in seed.v.iter_mut().zip(next_state.v.iter()) {
                    g.y += bounce_coeff * s.y;
                }
            }
        },
    )
}

/// Same backprop through time, but for an arbitrary loss seed: `seed_g_x[i]`
/// = dL/d(final position of particle i). `controller_gradient` is the
/// centroid-drift special case. Exposed separately because losses that
/// aren't pure-centroid are the only way to finite-difference-verify the
/// chain in a contact-free regime -- momentum conservation makes the
/// centroid EXACTLY invariant there (the analytic gradient correctly
/// reports zero), leaving nothing to measure.
pub fn controller_gradient_seeded(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    seed_g_x: &[Vec2],
) -> (Vec<f32>, Vec<f32>) {
    backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, _next_state, seed| {
            if t == steps - 1 {
                for (g, s) in seed.x.iter_mut().zip(seed_g_x.iter()) {
                    *g += *s;
                }
            }
        },
    )
}

/// The two running gradients a loss-seeding closure can add into.
struct GradSeed<'a> {
    x: &'a mut [Vec2],
    v: &'a mut [Vec2],
}

/// Core backprop-through-time loop. `inject_seed(t, next_state, seed)`
/// is called at each substep (in reverse order) BEFORE that substep's
/// backward pass, and adds dL/d(state-after-substep-t's positions/
/// velocities) into the running gradients -- this is how a loss that reads
/// MULTIPLE states along the rollout (an averaged-drift loss, a bounce
/// penalty) seeds its gradient, not just a final-state loss. `next_state` is
/// that substep's own forward result, for losses that need to read it (e.g.
/// the bounce penalty reads `next_state.v`).
fn backprop_through_time(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    inject_seed: &mut dyn FnMut(usize, &DiffState, &mut GradSeed),
) -> (Vec<f32>, Vec<f32>) {
    let (history, acts_cache) = rollout(plan, controller, eval, cfg, steps);
    let n = plan.positions.len();
    let rest = DiffState::rest(plan);

    let mut g_x = vec![Vec2::ZERO; n];
    let mut g_v = vec![Vec2::ZERO; n];
    let mut g_c = vec![Mat2::ZERO; n];
    let mut g_f = vec![Mat2::ZERO; n];

    let mut g_weights = vec![0.0f32; controller.weights.len()];
    let mut g_bias = vec![0.0f32; controller.bias.len()];

    for t in (0..steps).rev() {
        inject_seed(
            t,
            &history[t].0,
            &mut GradSeed {
                x: &mut g_x,
                v: &mut g_v,
            },
        );
        let state = if t == 0 { &rest } else { &history[t - 1].0 };
        let (next_state, record) = &history[t];
        let incoming = IncomingGrad {
            x: &g_x,
            v: &g_v,
            c: &g_c,
            f: &g_f,
        };
        let ctx = SubstepCtx {
            state,
            next_state,
            record,
            acts: &acts_cache[t],
        };
        let out = backward_substep(&ctx, plan, eval, cfg, &incoming);

        // Chain each group's activation gradient into the controller
        // parameters: act = tanh(pre), d(act)/d(pre) = 1 - tanh(pre)^2;
        // pre is linear in weights (sin basis values) and bias. The
        // control-effort penalty (mean act^2 over all groups/substeps) adds
        // DIRECTLY to the gradient w.r.t. `act` itself (same variable the
        // physics gradient `out.act` already targets), before the shared
        // tanh-derivative chain -- not a separate path.
        let time = t as f32 * cfg.dt;
        let effort_coeff =
            2.0 * cfg.control_effort_penalty / (controller.n_groups as f32 * steps as f32);
        for (g, &g_act_physics) in out.act.iter().enumerate() {
            let act = acts_cache[t][g];
            let g_act = g_act_physics + effort_coeff * act;
            let d_pre = g_act * (1.0 - act * act);
            // Mirrored groups reuse another group's weights/bias (see
            // `SinusoidController::mirror_of`'s doc), so their gradient
            // must land on the SAME underlying parameters, at the SAME
            // phase offset the forward pass actually used -- a shared
            // parameter's total gradient is the sum of every use's
            // contribution (standard multivariable chain rule), which
            // summing into `g_weights[src]`/`g_bias[src]` across every
            // group `g` whose `mirror_of` resolves to `src` achieves
            // automatically.
            let src = controller.mirror_of[g].unwrap_or(g);
            let extra_phase = controller.phase_offset[g];
            for j in 0..controller.n_waves {
                let phase = 2.0 * std::f32::consts::PI * j as f32 / controller.n_waves as f32;
                g_weights[src * controller.n_waves + j] +=
                    d_pre * (cfg.omega * time + phase + extra_phase).sin();
            }
            g_bias[src] += d_pre;
        }

        g_x = out.x;
        g_v = out.v;
        g_c = out.c;
        g_f = out.f;
    }

    (g_weights, g_bias)
}

/// Gradient of the same locomotion loss as `controller_gradient`, but for a
/// `FeedbackController`. Same windowed-drift + bounce-penalty objective;
/// the real difference is the activation gradient chains through
/// `FeedbackController::backward` (feature-extraction + linear + tanh)
/// instead of the sinusoid's tanh+weights, and that backward ALSO returns
/// position/velocity gradient contributions (the controller READ this
/// substep's state to decide its own activation) that must be ADDED onto
/// `out.x`/`out.v` -- a real, new path the open-loop controller never had
/// (its activation depended only on `t`, never on the body's own state).
pub fn feedback_controller_gradient(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = plan.positions.len();
    let window = cfg.loss_window.clamp(1, steps);
    let per_state = -1.0 / (n as f32 * window as f32);
    let bounce_coeff = 2.0 * cfg.bounce_penalty / (n as f32 * steps as f32);
    feedback_backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, next_state, seed| {
            if t >= steps - window {
                for g in seed.x.iter_mut() {
                    g.x += per_state;
                }
            }
            if bounce_coeff != 0.0 {
                for (g, s) in seed.v.iter_mut().zip(next_state.v.iter()) {
                    g.y += bounce_coeff * s.y;
                }
            }
        },
    )
}

/// `FeedbackController` analogue of `controller_gradient_seeded` -- same
/// role (arbitrary final-state loss seed, for FD verification in a
/// contact-free regime where the centroid loss's true gradient is exactly
/// zero by momentum conservation).
pub fn feedback_controller_gradient_seeded(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    seed_g_x: &[Vec2],
) -> (Vec<f32>, Vec<f32>) {
    feedback_backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, _next_state, seed| {
            if t == steps - 1 {
                for (g, s) in seed.x.iter_mut().zip(seed_g_x.iter()) {
                    *g += *s;
                }
            }
        },
    )
}

/// `FeedbackController` analogue of `backprop_through_time`. Structurally
/// identical to the sinusoid version (rollout, then reverse-order
/// `backward_substep` calls threading g_x/g_v/g_c/g_f between steps) --
/// the one real difference is chaining `out.act` through
/// `FeedbackController::backward` (recomputing that step's `(feat, count)`
/// from `state`, the same cheap-recompute-in-backward pattern used
/// throughout this module for kernel weights) instead of a fixed tanh+sin
/// formula, and adding that backward's own `(g_x, g_v)` outputs onto
/// `out.x`/`out.v` before they become the next iteration's incoming
/// gradients.
fn feedback_backprop_through_time(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    inject_seed: &mut dyn FnMut(usize, &DiffState, &mut GradSeed),
) -> (Vec<f32>, Vec<f32>) {
    let (history, acts_cache) = rollout_feedback(plan, controller, eval, cfg, steps);
    let n = plan.positions.len();
    let rest = DiffState::rest(plan);

    let mut g_x = vec![Vec2::ZERO; n];
    let mut g_v = vec![Vec2::ZERO; n];
    let mut g_c = vec![Mat2::ZERO; n];
    let mut g_f = vec![Mat2::ZERO; n];
    let mut g_weights = vec![0.0f32; controller.weights.len()];
    let mut g_bias = vec![0.0f32; controller.bias.len()];

    for t in (0..steps).rev() {
        inject_seed(
            t,
            &history[t].0,
            &mut GradSeed {
                x: &mut g_x,
                v: &mut g_v,
            },
        );
        let state = if t == 0 { &rest } else { &history[t - 1].0 };
        let (next_state, record) = &history[t];
        let incoming = IncomingGrad {
            x: &g_x,
            v: &g_v,
            c: &g_c,
            f: &g_f,
        };
        let ctx = SubstepCtx {
            state,
            next_state,
            record,
            acts: &acts_cache[t],
        };
        let out = backward_substep(&ctx, plan, eval, cfg, &incoming);

        let effort_coeff =
            2.0 * cfg.control_effort_penalty / (controller.n_groups as f32 * steps as f32);
        let g_act: Vec<f32> = out
            .act
            .iter()
            .enumerate()
            .map(|(g, &g_act_physics)| g_act_physics + effort_coeff * acts_cache[t][g])
            .collect();

        let (feat, count) = FeedbackController::features(plan, state);
        let (g_w_step, g_b_step, g_x_ctrl, g_v_ctrl) =
            controller.backward(plan, &feat, &count, &g_act);
        for (gw, s) in g_weights.iter_mut().zip(g_w_step.iter()) {
            *gw += s;
        }
        for (gb, s) in g_bias.iter_mut().zip(g_b_step.iter()) {
            *gb += s;
        }

        g_x = out.x;
        g_v = out.v;
        for (gx, extra) in g_x.iter_mut().zip(g_x_ctrl.iter()) {
            *gx += *extra;
        }
        for (gv, extra) in g_v.iter_mut().zip(g_v_ctrl.iter()) {
            *gv += *extra;
        }
        g_c = out.c;
        g_f = out.f;
    }

    (g_weights, g_bias)
}

/// Forward drift (grid units) of the body's mean x over a rollout -- the
/// quantity training maximizes.
pub fn drift(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> f32 {
    let (history, _) = rollout(plan, controller, eval, cfg, steps);
    history[steps - 1].0.mean_x() - DiffState::rest(plan).mean_x()
}

/// `FeedbackController` analogue of `drift`.
pub fn drift_feedback(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> f32 {
    let (history, _) = rollout_feedback(plan, controller, eval, cfg, steps);
    history[steps - 1].0.mean_x() - DiffState::rest(plan).mean_x()
}

/// Gait-quality metrics for judging whether a trained controller WALKS
/// rather than hurls itself -- the difference a drift number alone can't
/// see (observed live: a final-state-drift-trained gait "goes flying").
pub struct GaitMetrics {
    /// Final-state mean-x drift (what `drift` reports).
    pub final_drift: f32,
    /// Drift averaged over the last `cfg.loss_window` states (the windowed
    /// training objective).
    pub windowed_drift: f32,
    /// Fraction of substeps where the body's lowest particle is within one
    /// particle spacing of the floor -- ~1.0 for a grounded walk, small for
    /// ballistic hopping.
    pub contact_fraction: f32,
    /// Highest the body's LOWEST particle ever gets above the floor -- a
    /// direct "how airborne did it go" measure (grid units).
    pub max_clearance: f32,
}

pub fn gait_metrics(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    contact_band: f32,
) -> GaitMetrics {
    let (history, _) = rollout(plan, controller, eval, cfg, steps);
    gait_metrics_from_history(plan, cfg, steps, contact_band, &history)
}

/// `FeedbackController` analogue of `gait_metrics`.
pub fn gait_metrics_feedback(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    contact_band: f32,
) -> GaitMetrics {
    let (history, _) = rollout_feedback(plan, controller, eval, cfg, steps);
    gait_metrics_from_history(plan, cfg, steps, contact_band, &history)
}

fn gait_metrics_from_history(
    plan: &BodyPlan,
    cfg: &DiffConfig,
    steps: usize,
    contact_band: f32,
    history: &[(DiffState, StepRecord)],
) -> GaitMetrics {
    let rest_x = DiffState::rest(plan).mean_x();

    let final_drift = history[steps - 1].0.mean_x() - rest_x;
    let window = cfg.loss_window.clamp(1, steps);
    let windowed_drift = history[steps - window..]
        .iter()
        .map(|(s, _)| s.mean_x() - rest_x)
        .sum::<f32>()
        / window as f32;

    let mut contact_steps = 0usize;
    let mut max_clearance = 0.0f32;
    for (state, _) in history.iter() {
        let lowest = state.x.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let clearance = (lowest - cfg.floor_y).max(0.0);
        if clearance <= contact_band {
            contact_steps += 1;
        }
        max_clearance = max_clearance.max(clearance);
    }

    GaitMetrics {
        final_drift,
        windowed_drift,
        contact_fraction: contact_steps as f32 / steps as f32,
        max_clearance,
    }
}

/// Plain gradient descent on the controller parameters. Returns per-iteration
/// drift so callers can report/plot training progress.
///
/// Keeps the BEST-drift parameters seen, not the last: measured on the real
/// walker, late training oscillates (a 600-substep horizon regressed from a
/// 0.74 best back to 0.06 by the final iteration -- the classic
/// backprop-through-time instability, sharpened here by the contact kinks),
/// so `controller` is restored to its best-scoring snapshot before returning.
/// Standard model selection, not a workaround specific to this trainer.
///
/// Uses Adam (Kingma & Ba 2014), not plain gradient descent: a real, measured
/// symptom found live 2026-07-11 -- with a fixed step size, the same body
/// went from "flies" (contact 0.31) to "frozen" (drift 0.05, near-zero
/// movement) between bounce-penalty values 0.05 and 0.1, a razor-thin usable
/// range. That's the textbook fixed-step-size failure: SGD takes the same
/// size step regardless of how consistent or noisy a parameter's gradient
/// history has been. Adam tracks per-parameter first/second moment
/// estimates and scales each parameter's step by them, damping steps for
/// noisy/spiky gradients and taking confident steps where gradients are
/// small but consistent -- standard fix for exactly this symptom, not a
/// tuning trick specific to this trainer. `lr` here is Adam's own learning
/// rate (typically much smaller than an SGD-tuned one, ~1e-2 to 1e-1 for
/// this problem's scale, not ~1.0).
pub fn train(
    plan: &BodyPlan,
    controller: &mut SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    iterations: usize,
    lr: f32,
) -> Vec<f32> {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1.0e-8;

    let mut drifts = Vec::with_capacity(iterations);
    let mut best_score = f32::NEG_INFINITY;
    let mut best = controller.clone();
    let window = cfg.loss_window.clamp(1, steps);
    let rest_x = DiffState::rest(plan).mean_x();

    let mut m_w = vec![0.0f32; controller.weights.len()];
    let mut v_w = vec![0.0f32; controller.weights.len()];
    let mut m_b = vec![0.0f32; controller.bias.len()];
    let mut v_b = vec![0.0f32; controller.bias.len()];

    for iter in 1..=iterations {
        let (g_w, g_b) = controller_gradient(plan, controller, eval, cfg, steps);
        let bias_correction1 = 1.0 - BETA1.powi(iter as i32);
        let bias_correction2 = 1.0 - BETA2.powi(iter as i32);

        for (((w, g), m), v) in controller
            .weights
            .iter_mut()
            .zip(g_w.iter())
            .zip(m_w.iter_mut())
            .zip(v_w.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *w -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
        for (((b, g), m), v) in controller
            .bias
            .iter_mut()
            .zip(g_b.iter())
            .zip(m_b.iter_mut())
            .zip(v_b.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *b -= lr * m_hat / (v_hat.sqrt() + EPS);
        }

        // Score by the SAME full objective training optimizes (windowed
        // drift, bounce penalty, control-effort penalty), so model
        // selection can't quietly reintroduce an exploit the objective was
        // extended to remove.
        let (history, acts_cache) = rollout(plan, controller, eval, cfg, steps);
        let n = plan.positions.len();
        let windowed_drift = history[steps - window..]
            .iter()
            .map(|(s, _)| s.mean_x() - rest_x)
            .sum::<f32>()
            / window as f32;
        let bounce = history
            .iter()
            .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
            .sum::<f32>()
            / (n as f32 * steps as f32);
        let effort = acts_cache
            .iter()
            .map(|acts| acts.iter().map(|a| a * a).sum::<f32>())
            .sum::<f32>()
            / (controller.n_groups as f32 * steps as f32);
        let score =
            windowed_drift - cfg.bounce_penalty * bounce - cfg.control_effort_penalty * effort;
        if score > best_score {
            best_score = score;
            best = controller.clone();
        }
        drifts.push(history[steps - 1].0.mean_x() - rest_x);
    }
    *controller = best;
    drifts
}

/// `FeedbackController` analogue of `train` -- identical Adam loop and
/// keep-best model selection, only the gradient source and rollout differ.
pub fn train_feedback(
    plan: &BodyPlan,
    controller: &mut FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    iterations: usize,
    lr: f32,
) -> Vec<f32> {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1.0e-8;

    let mut drifts = Vec::with_capacity(iterations);
    let mut best_score = f32::NEG_INFINITY;
    let mut best = controller.clone();
    let window = cfg.loss_window.clamp(1, steps);
    let rest_x = DiffState::rest(plan).mean_x();

    let mut m_w = vec![0.0f32; controller.weights.len()];
    let mut v_w = vec![0.0f32; controller.weights.len()];
    let mut m_b = vec![0.0f32; controller.bias.len()];
    let mut v_b = vec![0.0f32; controller.bias.len()];

    for iter in 1..=iterations {
        let (g_w, g_b) = feedback_controller_gradient(plan, controller, eval, cfg, steps);
        let bias_correction1 = 1.0 - BETA1.powi(iter as i32);
        let bias_correction2 = 1.0 - BETA2.powi(iter as i32);

        for (((w, g), m), v) in controller
            .weights
            .iter_mut()
            .zip(g_w.iter())
            .zip(m_w.iter_mut())
            .zip(v_w.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *w -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
        for (((b, g), m), v) in controller
            .bias
            .iter_mut()
            .zip(g_b.iter())
            .zip(m_b.iter_mut())
            .zip(v_b.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *b -= lr * m_hat / (v_hat.sqrt() + EPS);
        }

        let (history, acts_cache) = rollout_feedback(plan, controller, eval, cfg, steps);
        let n = plan.positions.len();
        let windowed_drift = history[steps - window..]
            .iter()
            .map(|(s, _)| s.mean_x() - rest_x)
            .sum::<f32>()
            / window as f32;
        let bounce = history
            .iter()
            .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
            .sum::<f32>()
            / (n as f32 * steps as f32);
        let effort = acts_cache
            .iter()
            .map(|acts| acts.iter().map(|a| a * a).sum::<f32>())
            .sum::<f32>()
            / (controller.n_groups as f32 * steps as f32);
        let score =
            windowed_drift - cfg.bounce_penalty * bounce - cfg.control_effort_penalty * effort;
        if score > best_score {
            best_score = score;
            best = controller.clone();
        }
        drifts.push(history[steps - 1].0.mean_x() - rest_x);
    }
    *controller = best;
    drifts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (BodyPlan, DiffConfig, StressEval) {
        let plan = BodyPlan::walker(Vec2::new(20.0, 1.3), 0.5);
        let cfg = DiffConfig::default();
        let eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
        (plan, cfg, eval)
    }

    /// Same walker, but placed near the origin so absolute positions (and
    /// with them the f32 quantization floor on an absolute-position loss)
    /// stay small -- FD verification only.
    fn setup_origin() -> (BodyPlan, DiffConfig, StressEval) {
        let plan = BodyPlan::walker(Vec2::new(1.5, 1.3), 0.5);
        let cfg = DiffConfig::default();
        let eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
        (plan, cfg, eval)
    }

    #[test]
    fn signed_active_stress_vjp_matches_finite_difference() {
        let f = Mat2::from_cols(Vec2::new(1.1, 0.08), Vec2::new(-0.06, 0.92));
        let strength = 12.0;
        let fiber = Vec2::Y;
        let g = Mat2::from_cols(Vec2::new(0.4, -0.7), Vec2::new(0.6, 0.3));
        let h = 1.0e-3_f32;

        let loss = |f: Mat2, act: f32| -> f32 {
            let tau = signed_active_stress(f, act, strength, fiber);
            g.x_axis.x * tau.x_axis.x
                + g.x_axis.y * tau.x_axis.y
                + g.y_axis.x * tau.y_axis.x
                + g.y_axis.y * tau.y_axis.y
        };

        // The signed case's whole point: check at a NEGATIVE activation,
        // where the engine variant's guard would kill the gradient.
        for act in [-0.6f32, -0.05, 0.3] {
            let (analytic_f, analytic_act) = signed_active_stress_vjp(f, act, strength, fiber, g);

            let numeric_act = (loss(f, act + h) - loss(f, act - h)) / (2.0 * h);
            let diff = (numeric_act - analytic_act).abs();
            let scale = numeric_act.abs().max(analytic_act.abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "signed act-gradient mismatch at act={act}: analytic={analytic_act:.6} \
                 numeric={numeric_act:.6}"
            );

            let mut f_plus = f;
            f_plus.x_axis.y += h;
            let mut f_minus = f;
            f_minus.x_axis.y -= h;
            let numeric_f = (loss(f_plus, act) - loss(f_minus, act)) / (2.0 * h);
            let diff = (numeric_f - analytic_f.x_axis.y).abs();
            let scale = numeric_f.abs().max(analytic_f.x_axis.y.abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "signed F-gradient mismatch at act={act}: analytic={:.6} numeric={numeric_f:.6}",
                analytic_f.x_axis.y
            );
        }
    }

    /// FD check of the full smooth pipeline -- sinusoid controller, tanh
    /// chain, signed active stress, position-identity chain, multi-particle
    /// shared grid, backprop through time -- with the CONTACT KINK removed
    /// (no gravity, floor placed out of reach). The sticky floor is a
    /// genuine, hard non-differentiability: a finite-difference perturbation
    /// flips stick/unstick branch decisions mid-rollout, so numeric and
    /// analytic (detached-branch subgradient) values legitimately diverge
    /// there -- the standard, documented contact-gradient limitation every
    /// differentiable contact simulator shares, DiffTaichi's canonical
    /// walker included (its floor produces the same kink; its training works
    /// anyway). The contact regime is therefore verified separately by
    /// `contact_gradient_is_a_descent_direction` below with the property
    /// training actually needs, while THIS test pins every smooth piece of
    /// the chain against exact finite differences.
    ///
    /// Uses a per-particle loss (two particles, mixed x/y weights), NOT the
    /// centroid loss: with no external contact, momentum conservation makes
    /// the centroid exactly invariant, so its true gradient is zero and an
    /// FD check of it measures nothing (a fact this module's own
    /// `internal_stress_conserves_centroid` pins down separately).
    #[test]
    fn controller_gradient_matches_finite_difference_smooth_regime() {
        let (plan, mut cfg, mut eval) = setup_origin();
        cfg.gravity = 0.0;
        cfg.floor_y = -100.0; // out of reach: no contact anywhere in the rollout
        // Gentler actuation than the training default, and a body placed
        // near the origin (same trick as DiffTaichi's unit-box domain): the
        // loss reads absolute particle positions, so f32 quantization sets a
        // noise floor of one ULP of |x| per step -- at x~20 that floor
        // swamps small-amplitude gradients, at x~2 it is 16x finer. The
        // remaining systematic disagreement is the chain's one deliberate
        // scope exclusion (kernel-weight position dependence, see module
        // docs): measured at ~5-7% at full training strength, shrinking
        // with actuation amplitude exactly as an excluded motion-
        // proportional term should.
        cfg.act_strength = 4.0;
        let steps = 40;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
        // Non-trivial starting point so gradients aren't at a symmetric zero.
        controller.bias[1] = 0.15;
        controller.weights[2] = 0.2;

        let n = plan.positions.len();
        // Loss L = x_final[3].x - 0.5 * x_final[40].y : deliberately
        // asymmetric, touches both components, sensitive to internal motion.
        let mut seed = vec![Vec2::ZERO; n];
        seed[3] = Vec2::new(1.0, 0.0);
        seed[40] = Vec2::new(0.0, -0.5);

        let (g_w, g_b) =
            controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

        let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
            let (history, _) = rollout(&plan, c, eval, &cfg, steps);
            let fin = &history[steps - 1].0;
            fin.x[3].x - 0.5 * fin.x[40].y
        };

        // h must be large enough that the loss change (~gradient * h) clears
        // f32 quantization: the loss lives at |x| ~ 20 grid units where one
        // ULP is ~2e-6, and gradients here are ~1e-3, so h = 2e-3 would ask
        // the FD numerator to resolve ~3 ULPs -- it reads exactly zero. At
        // h = 5e-2 the signal is ~75 ULPs while tanh/sin curvature error
        // (O(h^2)) stays well inside the tolerance.
        let h = 5.0e-2_f32;
        // A few representative weights across groups/waves, plus one bias.
        for &wi in &[0usize, 2, 5, 9, 14] {
            let mut c_plus = controller.clone();
            c_plus.weights[wi] += h;
            let mut c_minus = controller.clone();
            c_minus.weights[wi] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "controller weight[{wi}] gradient mismatch (smooth regime): analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        {
            let mut c_plus = controller.clone();
            c_plus.bias[1] += h;
            let mut c_minus = controller.clone();
            c_minus.bias[1] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_b[1]).abs();
            let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "controller bias[1] gradient mismatch (smooth regime): analytic={:.6} \
                 numeric={numeric:.6} relative_diff={:.2e}",
                g_b[1],
                diff / scale
            );
        }
    }

    /// The real end-to-end gate for the whole `FeedbackController` chain:
    /// `feedback_controller_backward_matches_finite_difference` above only
    /// checks the controller in isolation (no physics); this backprops
    /// through real P2G/grid-update/G2P/F-update substeps AND the
    /// controller's own state-read backward, together, in the smooth
    /// (contact-free) regime -- same rationale as the sinusoid controller's
    /// own smooth-regime test (non-centroid loss, origin-placed body, gentle
    /// actuation for the O(h^2) FD error margin).
    #[test]
    fn feedback_controller_gradient_matches_finite_difference_smooth_regime() {
        let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
        let cfg = DiffConfig {
            gravity: 0.0,
            floor_y: -100.0,
            act_strength: 4.0,
            ..DiffConfig::default()
        };
        let mut eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
        let steps = 40;
        let mut controller = FeedbackController::seeded_with(plan.n_groups, 7);
        controller.bias[1] = 0.15;
        controller.weights[3] = 0.2;

        let n = plan.positions.len();
        let mut seed = vec![Vec2::ZERO; n];
        seed[2] = Vec2::new(1.0, 0.0);
        seed[n - 3] = Vec2::new(0.0, -0.5);

        let (g_w, g_b) =
            feedback_controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

        let loss_of = |c: &FeedbackController, eval: &mut StressEval| -> f32 {
            let (history, _) = rollout_feedback(&plan, c, eval, &cfg, steps);
            let fin = &history[steps - 1].0;
            fin.x[2].x - 0.5 * fin.x[n - 3].y
        };

        let h = 5.0e-2_f32;
        for &wi in &[0usize, 3, 9, 20] {
            let mut c_plus = controller.clone();
            c_plus.weights[wi] += h;
            let mut c_minus = controller.clone();
            c_minus.weights[wi] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "feedback controller weight[{wi}] gradient mismatch (end-to-end): analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        {
            let mut c_plus = controller.clone();
            c_plus.bias[1] += h;
            let mut c_minus = controller.clone();
            c_minus.bias[1] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_b[1]).abs();
            let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "feedback controller bias[1] gradient mismatch (end-to-end): analytic={:.6} \
                 numeric={numeric:.6} relative_diff={:.2e}",
                g_b[1],
                diff / scale
            );
        }
    }

    /// Every FD test above uses `BodyPlan::walker`, whose 4 groups all share
    /// the SAME fiber direction (`Vec2::Y`) -- moving `fiber_dir` from a
    /// single `DiffConfig` value onto per-group `BodyPlan` storage is a real
    /// behavior change (mixed directions ACROSS a body, the whole point of
    /// `BodyPlan::biped`'s thigh/foot split) that none of them exercise.
    /// Same smooth-regime + origin-placement rationale, `BodyPlan::biped`
    /// instead of `walker`.
    #[test]
    fn mixed_fiber_directions_gradient_matches_finite_difference() {
        let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
        let cfg = DiffConfig {
            gravity: 0.0,
            floor_y: -100.0,
            act_strength: 4.0,
            ..DiffConfig::default()
        };
        let mut eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
        let steps = 40;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
        controller.bias[1] = 0.15; // left foot (horizontal fiber)
        controller.weights[2 * cfg.n_waves] = 0.2; // right thigh (vertical fiber)

        let n = plan.positions.len();
        let mut seed = vec![Vec2::ZERO; n];
        seed[2] = Vec2::new(1.0, 0.0);
        seed[n - 3] = Vec2::new(0.0, -0.5);

        let (g_w, g_b) =
            controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

        let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
            let (history, _) = rollout(&plan, c, eval, &cfg, steps);
            let fin = &history[steps - 1].0;
            fin.x[2].x - 0.5 * fin.x[n - 3].y
        };

        let h = 5.0e-2_f32;
        for &wi in &[
            0usize,
            cfg.n_waves + 1,
            2 * cfg.n_waves,
            3 * cfg.n_waves + 2,
        ] {
            let mut c_plus = controller.clone();
            c_plus.weights[wi] += h;
            let mut c_minus = controller.clone();
            c_minus.weights[wi] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "mixed-fiber weight[{wi}] gradient mismatch: analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        {
            let mut c_plus = controller.clone();
            c_plus.bias[1] += h;
            let mut c_minus = controller.clone();
            c_minus.bias[1] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_b[1]).abs();
            let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "mixed-fiber bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
                 relative_diff={:.2e}",
                g_b[1],
                diff / scale
            );
        }
    }

    /// Physical invariant, and the reason the FD test above needs a
    /// non-centroid loss: with no gravity and no floor, muscle stress is
    /// purely internal, so the body's mass centroid must not move no matter
    /// how hard the controller fires -- Newton's third law flowing through
    /// P2G/G2P intact. Also the physics fact that makes contact necessary
    /// for locomotion at all.
    #[test]
    fn internal_stress_conserves_centroid() {
        let (plan, mut cfg, mut eval) = setup();
        cfg.gravity = 0.0;
        cfg.floor_y = -100.0;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
        // Fire the muscles hard: conservation must hold regardless.
        controller.bias = vec![0.8, -0.6, 0.7, -0.5];
        for w in controller.weights.iter_mut() {
            *w += 0.3;
        }

        let d = drift(&plan, &controller, &mut eval, &cfg, 60);
        assert!(
            d.abs() < 1.0e-3,
            "internal muscle stress must not move the centroid (no contact, no gravity): \
             drift={d:.6}"
        );
    }

    /// Contact-regime verification: exact FD equality is unattainable across
    /// the sticky floor's branch kinks (see the smooth-regime test's doc),
    /// so verify the property gradient descent actually relies on instead --
    /// the analytic gradient must be a real descent direction of the true
    /// (kinked) loss: stepping against it must reduce the loss.
    #[test]
    fn contact_gradient_is_a_descent_direction() {
        let (plan, cfg, mut eval) = setup();
        let steps = 60;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
        controller.bias[1] = 0.15;
        controller.weights[2] = 0.2;

        let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
            -drift(&plan, c, eval, &cfg, steps)
        };
        let loss_base = loss_of(&controller, &mut eval);

        let (g_w, g_b) = controller_gradient(&plan, &controller, &mut eval, &cfg, steps);
        let grad_norm_sq: f32 = g_w.iter().chain(g_b.iter()).map(|g| g * g).sum();
        assert!(
            grad_norm_sq > 0.0,
            "contact-regime gradient must be nonzero to test descent"
        );

        // A real descent step (same order as training's lr) must reduce loss.
        let alpha = 0.5;
        let mut stepped = controller.clone();
        for (w, g) in stepped.weights.iter_mut().zip(g_w.iter()) {
            *w -= alpha * g;
        }
        for (b, g) in stepped.bias.iter_mut().zip(g_b.iter()) {
            *b -= alpha * g;
        }
        let loss_stepped = loss_of(&stepped, &mut eval);

        assert!(
            loss_stepped < loss_base,
            "stepping against the analytic gradient must reduce the true loss: \
             base={loss_base:.6} stepped={loss_stepped:.6}"
        );
    }

    /// FD check of the WINDOWED (multi-state) loss seeding -- the mechanism
    /// `controller_gradient` uses when `loss_window > 1`, where dL/dx is
    /// injected at several substeps and must compose correctly with the
    /// position-identity chain. Same smooth regime + single-particle loss +
    /// origin placement rationale as the single-seed FD test above; loss =
    /// average of particle 3's x over the last K states.
    #[test]
    fn windowed_loss_gradient_matches_finite_difference() {
        let (plan, mut cfg, mut eval) = setup_origin();
        cfg.gravity = 0.0;
        cfg.floor_y = -100.0;
        cfg.act_strength = 4.0;
        let steps = 40;
        let window = 8usize;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
        controller.bias[1] = 0.15;
        controller.weights[2] = 0.2;

        let per_state = 1.0 / window as f32;
        let (g_w, g_b) = backprop_through_time(
            &plan,
            &controller,
            &mut eval,
            &cfg,
            steps,
            &mut |t, _next_state, seed| {
                if t >= steps - window {
                    seed.x[3].x += per_state;
                }
            },
        );

        let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
            let (history, _) = rollout(&plan, c, eval, &cfg, steps);
            history[steps - window..]
                .iter()
                .map(|(s, _)| s.x[3].x)
                .sum::<f32>()
                / window as f32
        };

        let h = 5.0e-2_f32;
        for &wi in &[0usize, 2, 9] {
            let mut c_plus = controller.clone();
            c_plus.weights[wi] += h;
            let mut c_minus = controller.clone();
            c_minus.weights[wi] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "windowed loss weight[{wi}] gradient mismatch: analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        {
            let mut c_plus = controller.clone();
            c_plus.bias[1] += h;
            let mut c_minus = controller.clone();
            c_minus.bias[1] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_b[1]).abs();
            let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "windowed loss bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
                 relative_diff={:.2e}",
                g_b[1],
                diff / scale
            );
        }
    }

    /// FD check of `bounce_penalty`'s own gradient contribution -- the real
    /// fix for the "still flies" gap `loss_window` alone left open (see
    /// `DiffConfig::bounce_penalty`'s doc). Uses `controller_gradient`
    /// directly (the public API a real trainer calls) with both the drift
    /// window AND the penalty active together, so it verifies they compose
    /// correctly, not just that the penalty term is correct in isolation.
    #[test]
    fn bounce_penalty_gradient_matches_finite_difference() {
        let (plan, mut cfg, mut eval) = setup_origin();
        cfg.gravity = 0.0;
        cfg.floor_y = -100.0;
        cfg.act_strength = 4.0;
        cfg.loss_window = 10;
        cfg.bounce_penalty = 2.0;
        let steps = 40;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
        controller.bias[1] = 0.15;
        controller.weights[2] = 0.2;

        let (g_w, g_b) = controller_gradient(&plan, &controller, &mut eval, &cfg, steps);

        let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
            let (history, _) = rollout(&plan, c, eval, &cfg, steps);
            let n = plan.positions.len();
            let window = cfg.loss_window;
            let drift = history[steps - window..]
                .iter()
                .map(|(s, _)| s.mean_x())
                .sum::<f32>()
                / window as f32
                - DiffState::rest(&plan).mean_x();
            let bounce = history
                .iter()
                .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
                .sum::<f32>()
                / (n as f32 * steps as f32);
            -drift + cfg.bounce_penalty * bounce
        };

        let h = 5.0e-2_f32;
        for &wi in &[0usize, 2, 9] {
            let mut c_plus = controller.clone();
            c_plus.weights[wi] += h;
            let mut c_minus = controller.clone();
            c_minus.weights[wi] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "bounce-penalty weight[{wi}] gradient mismatch: analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        {
            let mut c_plus = controller.clone();
            c_plus.bias[1] += h;
            let mut c_minus = controller.clone();
            c_minus.bias[1] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_b[1]).abs();
            let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "bounce-penalty bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
                 relative_diff={:.2e}",
                g_b[1],
                diff / scale
            );
        }
    }

    /// FD check of `control_effort_penalty`'s gradient contribution -- the
    /// "torque cost" half of reward shaping, alongside `bounce_penalty`'s
    /// "don't bounce" half. Both active together, verifying composition
    /// exactly like the bounce-penalty test above.
    #[test]
    fn control_effort_penalty_gradient_matches_finite_difference() {
        let (plan, mut cfg, mut eval) = setup_origin();
        cfg.gravity = 0.0;
        cfg.floor_y = -100.0;
        cfg.act_strength = 4.0;
        cfg.loss_window = 10;
        cfg.bounce_penalty = 1.0;
        cfg.control_effort_penalty = 3.0;
        let steps = 40;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
        controller.bias[1] = 0.15;
        controller.weights[2] = 0.2;

        let (g_w, g_b) = controller_gradient(&plan, &controller, &mut eval, &cfg, steps);

        let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
            let (history, acts_cache) = rollout(&plan, c, eval, &cfg, steps);
            let n = plan.positions.len();
            let window = cfg.loss_window;
            let drift = history[steps - window..]
                .iter()
                .map(|(s, _)| s.mean_x())
                .sum::<f32>()
                / window as f32
                - DiffState::rest(&plan).mean_x();
            let bounce = history
                .iter()
                .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
                .sum::<f32>()
                / (n as f32 * steps as f32);
            let effort = acts_cache
                .iter()
                .map(|acts| acts.iter().map(|a| a * a).sum::<f32>())
                .sum::<f32>()
                / (c.n_groups as f32 * steps as f32);
            -drift + cfg.bounce_penalty * bounce + cfg.control_effort_penalty * effort
        };

        let h = 5.0e-2_f32;
        for &wi in &[0usize, 2, 9] {
            let mut c_plus = controller.clone();
            c_plus.weights[wi] += h;
            let mut c_minus = controller.clone();
            c_minus.weights[wi] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "control-effort weight[{wi}] gradient mismatch: analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        {
            let mut c_plus = controller.clone();
            c_plus.bias[1] += h;
            let mut c_minus = controller.clone();
            c_minus.bias[1] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_b[1]).abs();
            let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "control-effort bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
                 relative_diff={:.2e}",
                g_b[1],
                diff / scale
            );
        }
    }

    /// FD check of the bilateral-symmetry gradient redirection --
    /// mirrored groups (right leg) must accumulate their gradient onto the
    /// SAME underlying weights/bias as their source group (left leg), at
    /// the source's own phase offset (0) reflected through the mirror's
    /// extra phase, not their own dead weight slots. Uses `BodyPlan::biped`
    /// with `with_biped_symmetry()` -- the actual configuration meant to
    /// fix the one-legged-hop degenerate gait.
    #[test]
    fn bilateral_symmetry_gradient_matches_finite_difference() {
        let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
        let cfg = DiffConfig {
            gravity: 0.0,
            floor_y: -100.0,
            act_strength: 4.0,
            ..DiffConfig::default()
        };
        let mut eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
        let steps = 40;
        let mut controller =
            SinusoidController::seeded(plan.n_groups, cfg.n_waves).with_biped_symmetry();
        controller.bias[0] = 0.15; // left thigh (source of group 2's mirror)
        controller.weights[cfg.n_waves] = 0.2; // left foot (source of group 3's mirror)

        let n = plan.positions.len();
        let mut seed = vec![Vec2::ZERO; n];
        seed[2] = Vec2::new(1.0, 0.0);
        seed[n - 3] = Vec2::new(0.0, -0.5);

        let (g_w, g_b) =
            controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

        // Groups 2 and 3 (right leg) are mirrors -- their own weight/bias
        // slots must never receive gradient; only groups 0 and 1 should.
        for j in 0..cfg.n_waves {
            assert_eq!(
                g_w[2 * cfg.n_waves + j],
                0.0,
                "mirrored group's own weight slot must stay untouched (gradient lands on source)"
            );
            assert_eq!(
                g_w[3 * cfg.n_waves + j],
                0.0,
                "mirrored group's own weight slot must stay untouched (gradient lands on source)"
            );
        }

        let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
            let (history, _) = rollout(&plan, c, eval, &cfg, steps);
            let fin = &history[steps - 1].0;
            fin.x[2].x - 0.5 * fin.x[n - 3].y
        };

        let h = 5.0e-2_f32;
        for &wi in &[0usize, cfg.n_waves + 1] {
            let mut c_plus = controller.clone();
            c_plus.weights[wi] += h;
            let mut c_minus = controller.clone();
            c_minus.weights[wi] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "bilateral-symmetry weight[{wi}] gradient mismatch: analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        {
            let mut c_plus = controller.clone();
            c_plus.bias[0] += h;
            let mut c_minus = controller.clone();
            c_minus.bias[0] -= h;
            let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
            let diff = (numeric - g_b[0]).abs();
            let scale = numeric.abs().max(g_b[0].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "bilateral-symmetry bias[0] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
                 relative_diff={:.2e}",
                g_b[0],
                diff / scale
            );
        }
    }

    /// FD check of `FeedbackController::backward` in COMPLETE isolation
    /// from the physics chain (no rollout, no P2G/G2P) -- pins down the
    /// feature-extraction + centroid + linear-layer adjoint on its own
    /// before it's ever trusted wired into a real substep, since this is
    /// the highest-risk new derivation of the whole controller (mean/
    /// centroid backward is exactly the kind of "looks obvious, has a
    /// transpose/sign trap" derivation this project's discipline exists to
    /// catch before it's trusted).
    #[test]
    fn feedback_controller_backward_matches_finite_difference() {
        let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
        let controller = FeedbackController::seeded_with(plan.n_groups, 3);

        // A real, non-rest state: nonzero velocities and perturbed
        // positions, so both `rel` and `vel` features are nonzero.
        let mut state = DiffState::rest(&plan);
        for (i, x) in state.x.iter_mut().enumerate() {
            *x += Vec2::new(0.02 * (i as f32 % 5.0 - 2.0), 0.01 * (i as f32 % 3.0));
        }
        for (i, v) in state.v.iter_mut().enumerate() {
            *v = Vec2::new(0.03 * (i as f32 % 4.0 - 1.5), -0.02 * (i as f32 % 2.0));
        }

        // Loss reads two arbitrary particles' positions and velocities --
        // exercises both the `rel` (position) and `vel` gradient paths,
        // and (via the centroid) every particle's OWN position gradient,
        // not just the group directly queried.
        let g = Vec2::new(0.6, -0.4);
        let loss_of = |state: &DiffState| -> f32 {
            let acts = controller.activations(&plan, state);
            acts.iter()
                .enumerate()
                .map(|(g_id, a)| a * (1.0 + g_id as f32))
                .sum::<f32>()
                + g.dot(state.x[5])
                + g.dot(state.v[20])
        };

        let (feat, count) = FeedbackController::features(&plan, &state);
        let acts = controller.activations_from_features(&feat);
        // d(loss)/d(act[g]) from the sum-of-scaled-activations term.
        let g_act: Vec<f32> = (0..plan.n_groups).map(|g_id| 1.0 + g_id as f32).collect();
        let (g_w, g_b, mut g_x, mut g_v) = controller.backward(&plan, &feat, &count, &g_act);
        // Direct loss contributions bypassing the controller entirely.
        g_x[5] += g;
        g_v[20] += g;
        let _ = acts;

        let h = 1.0e-3_f32;

        // Weight and bias gradients.
        for &wi in &[0usize, 5, 12] {
            let mut c_plus = FeedbackController {
                weights: controller.weights.clone(),
                bias: controller.bias.clone(),
                n_groups: controller.n_groups,
            };
            c_plus.weights[wi] += h;
            let mut c_minus = FeedbackController {
                weights: controller.weights.clone(),
                bias: controller.bias.clone(),
                n_groups: controller.n_groups,
            };
            c_minus.weights[wi] -= h;
            let numeric = (c_plus
                .activations(&plan, &state)
                .iter()
                .enumerate()
                .map(|(g_id, a)| a * (1.0 + g_id as f32))
                .sum::<f32>()
                - c_minus
                    .activations(&plan, &state)
                    .iter()
                    .enumerate()
                    .map(|(g_id, a)| a * (1.0 + g_id as f32))
                    .sum::<f32>())
                / (2.0 * h);
            let diff = (numeric - g_w[wi]).abs();
            let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "feedback controller weight[{wi}] gradient mismatch: analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_w[wi],
                diff / scale
            );
        }
        for (bi, &g_b_bi) in g_b.iter().enumerate() {
            let mut c_plus = FeedbackController {
                weights: controller.weights.clone(),
                bias: controller.bias.clone(),
                n_groups: controller.n_groups,
            };
            c_plus.bias[bi] += h;
            let mut c_minus = FeedbackController {
                weights: controller.weights.clone(),
                bias: controller.bias.clone(),
                n_groups: controller.n_groups,
            };
            c_minus.bias[bi] -= h;
            let numeric = (c_plus
                .activations(&plan, &state)
                .iter()
                .enumerate()
                .map(|(g_id, a)| a * (1.0 + g_id as f32))
                .sum::<f32>()
                - c_minus
                    .activations(&plan, &state)
                    .iter()
                    .enumerate()
                    .map(|(g_id, a)| a * (1.0 + g_id as f32))
                    .sum::<f32>())
                / (2.0 * h);
            let diff = (numeric - g_b_bi).abs();
            let scale = numeric.abs().max(g_b_bi.abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "feedback controller bias[{bi}] gradient mismatch: analytic={:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                g_b_bi,
                diff / scale
            );
        }

        // Position and velocity gradients (the new, real-risk paths):
        // perturb particle 5's x and particle 20's v directly.
        for &(pi, axis) in &[(5usize, 0usize), (5, 1), (20, 0), (20, 1)] {
            let mut s_plus = state.clone();
            let mut s_minus = state.clone();
            if axis == 0 {
                s_plus.x[pi].x += h;
                s_minus.x[pi].x -= h;
            } else {
                s_plus.x[pi].y += h;
                s_minus.x[pi].y -= h;
            }
            let numeric = (loss_of(&s_plus) - loss_of(&s_minus)) / (2.0 * h);
            let analytic = if axis == 0 { g_x[pi].x } else { g_x[pi].y };
            let diff = (numeric - analytic).abs();
            let scale = numeric.abs().max(analytic.abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "feedback controller position[{pi}].{axis} gradient mismatch: analytic={analytic:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                diff / scale
            );
        }
        for &(pi, axis) in &[(20usize, 0usize), (20, 1)] {
            let mut s_plus = state.clone();
            let mut s_minus = state.clone();
            if axis == 0 {
                s_plus.v[pi].x += h;
                s_minus.v[pi].x -= h;
            } else {
                s_plus.v[pi].y += h;
                s_minus.v[pi].y -= h;
            }
            let numeric = (loss_of(&s_plus) - loss_of(&s_minus)) / (2.0 * h);
            let analytic = if axis == 0 { g_v[pi].x } else { g_v[pi].y };
            let diff = (numeric - analytic).abs();
            let scale = numeric.abs().max(analytic.abs()).max(1.0e-3);
            assert!(
                diff / scale < 5.0e-2,
                "feedback controller velocity[{pi}].{axis} gradient mismatch: analytic={analytic:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                diff / scale
            );
        }

        // A particle with NO group (passive torso) must still receive a
        // nonzero position gradient purely through the centroid term.
        let torso_particle = plan
            .group
            .iter()
            .position(|g| g.is_none())
            .expect("body plan must have passive torso particles");
        assert!(
            g_x[torso_particle].length() > 0.0,
            "passive (ungrouped) particles must still get a centroid-mediated position gradient"
        );
    }

    /// The end-to-end claim: gradient descent on the sinusoid controller
    /// finds a gait that beats the (near-zero-drift) untrained start.
    #[test]
    fn training_finds_a_gait() {
        let (plan, cfg, mut eval) = setup();
        let steps = 80;
        let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);

        let drift_before = drift(&plan, &controller, &mut eval, &cfg, steps);
        let drifts = train(&plan, &mut controller, &mut eval, &cfg, steps, 40, 0.5);
        let drift_after = *drifts.last().unwrap();

        println!(
            "training_finds_a_gait: {} particles, {steps} substeps, 40 iterations\n  \
             untrained drift: {drift_before:.4}\n  trained drift:   {drift_after:.4}",
            plan.positions.len()
        );

        assert!(
            drift_after > drift_before && drift_after > 0.0,
            "trained gait should produce real positive drift beyond the untrained start: \
             before={drift_before:.4} after={drift_after:.4}"
        );
    }
}
