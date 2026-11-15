//! Start at equilibrium: a lone body set down gently on the walls, its
//! weight ramped from zero, solved by the static mode of the path's own
//! solve (`solve`'s `Inputs::statics`). An initial-condition tool, not a
//! law: the residual is the step's at rest, `f_int(u) - f_ext + f_contact(u)
//! = 0`, with the law's own stress, the step's contact law and passes, and
//! the step's tolerances.
//!
//! What it gives is ONE equilibrium, that of one declared history: the body
//! placed touching the walls unstrained, its weight raised from 0 to `g` in
//! equal increments, each increment Alart and Curnier's quasi-static
//! frictional problem (CMAME 92(3), 1991: the slip measured from the
//! increment's start, the friction state carried from one increment to the
//! next). On a Coulomb wall the equilibria form a family and the answer
//! depends on the load's path; this is the "set down gently" one, not THE
//! equilibrium. The increments are halved until the end state moves by less
//! than the step controller's own tolerance (its energy norm against `rho
//! tol sqrt(E0)`, `step_control`), never a set count.
//!
//! Declared limits:
//! - plastic laws are refused: their state carries the load's history (the
//!   static mode for them comes after the elastic one);
//! - relaxation branches are left out of the stress and relaxed at the end
//!   state (`Relaxation::relax_at`): the record's long-time limit, the
//!   relaxed moduli (Blanc et al. 2016, `M(0) = M_r`);
//! - a free translation the load has no component on (a slide along
//!   frictionless walls) is neutral: every displaced copy of the equilibrium
//!   is one, and the static mode fixes it by a gauge, the body's mass-weighted
//!   mean displacement along it held at zero (no force at the solution); any
//!   other rigid motion no touching wall holds (none touching, a free
//!   rotation, a loaded free axis) has no equilibrium: refused and reported
//!   before any solve;
//! - an unstable equilibrium (tipping) or any increment whose Newton does
//!   not converge is reported, never forced;
//! - one body only (body-body contact at rest comes after);
//! - the ledger starts from the equilibrium's energy, no heat booked for the
//!   ramp (an initial condition).

use glam::{DMat2, DVec2};

use super::{Simulation, TotalLagrangianState};
use crate::spacetime::total_lagrangian::{
    ContactMode, ContactSet, Estimator, JACOBI_TOLERANCE, Particles, StepControl, contact_step,
    tolerance_energy,
};
use crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite;

/// What the settle did: the counts of its solves and how its result was
/// judged.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtRest {
    /// The increments of the ramp kept (the finest run's), and the load
    /// increments solved over every run of the halving.
    pub increments: u64,
    pub solved: u64,
    /// Newton iterations, contact passes, GMRES iterations and Jacobian
    /// band factorizations over every solve.
    pub newton: u64,
    pub passes: u64,
    pub gmres: u64,
    pub factorizations: u64,
    /// The last increment's residual over its scale, and the tolerance it
    /// was held to (`ConservingSettings::newton_tolerance`).
    pub residual: f64,
    pub tolerance: f64,
    /// The end state's change from the run with half as many increments, in
    /// the step controller's energy norm, J^(1/2)/m^(1/2), and the target it
    /// met (`rho tol sqrt(E0)`).
    pub change: f64,
    pub target: f64,
    /// The deepest contact point beyond its wall at the end, m (zero when
    /// none is), and the contacts the last increment ended holding.
    pub deepest: f64,
    pub held: usize,
    /// The axis of a free translation the load has no component on, fixed by
    /// the static mode's gauge (a body on frictionless walls), if any.
    pub gauged: Option<usize>,
}

/// Why the settle refused, nothing changed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SettleRefused {
    /// The total Lagrangian path is off (`SimConfig::total_lagrangian`).
    PathOff,
    /// The path has already run a frame: a start condition only.
    AlreadyRunning,
    /// Not exactly one body.
    Bodies(usize),
    /// A plastic law (its state carries the load's history).
    Plastic,
    /// A rigid motion of the body that no touching wall holds: its
    /// translation along x and y and its rotation, per unit of each (the
    /// eigenvector of the least held stiffness).
    FreeRigidMode([f64; 3]),
    /// An increment's Newton did not converge: the increments' count of
    /// that run, which increment, and its residual over its scale.
    NotConverged {
        increments: u64,
        increment: u64,
        residual: f64,
    },
    /// The halving reached the load factor's own resolution without the end
    /// state settling.
    NoLimit,
}

impl Simulation {
    /// Places the lone body at rest on the walls under its own weight, from
    /// its spawn (touching the walls, unstrained): the static mode of the
    /// path's solve, the weight ramped from 0 to `g` (this file's head for
    /// the history it declares and its limits). Before the first frame only;
    /// the run then starts from this shape, its ledger from its energy. The
    /// velocities the particles carry are given back after the solve (none
    /// for a body spawned at rest): the shape is the one at rest under the
    /// weight, the motion the caller's.
    pub fn settle_total_lagrangian_at_rest(&mut self) -> Result<AtRest, SettleRefused> {
        let path = self.config.total_lagrangian.ok_or(SettleRefused::PathOff)?;
        if self.total_lagrangian.is_some() || self.frame_index != 0 {
            return Err(SettleRefused::AlreadyRunning);
        }
        self.assert_total_lagrangian_supported();
        // The spawn, at rest: each run of the ramp starts from it.
        let spawn = || {
            let mut s = self.build_total_lagrangian_state((&path.settings, path.modal_flight));
            s.v.iter_mut().for_each(|v| *v = DVec2::ZERO);
            s
        };
        let base = spawn();
        if base.outlines.len() != 1 {
            return Err(SettleRefused::Bodies(base.outlines.len()));
        }
        let n = base.particles;
        if base.solids[..n].iter().any(|s| s.is_plastic()) {
            return Err(SettleRefused::Plastic);
        }
        let (planes, friction) = self.total_lagrangian_walls();
        let gravity = self.total_lagrangian_gravity();
        let gauge = free_rigid_mode(&base, &planes, friction, gravity)
            .map_err(SettleRefused::FreeRigidMode)?;
        let layout = self.total_lagrangian_layout();
        let dt = f64::from(self.config.dt);
        let control = StepControl::klapproth(path.tolerance, dt, Estimator::Acary);
        let target = control.safety * control.tolerance * base.tolerance_energy.sqrt();
        let mut counts = AtRest {
            tolerance: path.settings.newton_tolerance,
            target,
            gauged: gauge,
            ..AtRest::default()
        };
        // One run of the ramp in `increments` equal load increments from the
        // spawn.
        let ramp =
            |increments: u64, counts: &mut AtRest| -> Result<TotalLagrangianState, SettleRefused> {
                let mut s = spawn();
                for i in 1..=increments {
                    let load = gravity * (i as f64 / increments as f64);
                    let report = contact_step(
                        &layout,
                        Particles {
                            x: &mut s.x,
                            v: &mut s.v,
                            f: &mut s.f,
                            mass: &s.mass,
                            v0: &s.v0,
                        },
                        (&s.reference, &s.solids),
                        (dt, ContactMode::Velocity, load, true, gauge),
                        (
                            ContactSet {
                                planes: &planes,
                                pairs: &[],
                                outlines: &s.outlines,
                                corners: &s.corners,
                                friction,
                            },
                            &s.penalty,
                            &path.settings,
                        ),
                        &mut s.cache,
                    );
                    counts.solved += 1;
                    counts.newton += u64::from(report.newton);
                    counts.passes += u64::from(report.passes);
                    counts.gmres += u64::from(report.gmres);
                    counts.factorizations += report.timing.factorizations;
                    counts.residual = report.residual;
                    counts.held = report.contacts;
                    if !report.converged {
                        return Err(SettleRefused::NotConverged {
                            increments,
                            increment: i,
                            residual: report.residual,
                        });
                    }
                }
                Ok(s)
            };
        let mut increments = 1u64;
        let mut coarse = ramp(increments, &mut counts)?;
        let settled = loop {
            // The load factor's own resolution, `4 eps` (the controller's
            // floor on time, `step_control`).
            if 1.0 / ((2 * increments) as f64) < 4.0 * f64::EPSILON {
                return Err(SettleRefused::NoLimit);
            }
            increments *= 2;
            let fine = ramp(increments, &mut counts)?;
            counts.change = change_norm(&coarse, &fine);
            if counts.change <= target {
                break fine;
            }
            coarse = fine;
        };
        counts.increments = increments;
        let mut state = settled;
        // The velocities the particles carry given back, as the path reads
        // them at its start (the contact points' from the body's field,
        // `build_total_lagrangian_state`): zero for a body spawned at rest.
        state.v = self.build_total_lagrangian_state((&path.settings, false)).v;
        for p in 0..n {
            if state.solids[p].is_relaxing() {
                let c = state.f[p].transpose() * state.f[p];
                let moduli = state.solids[p].lame();
                state.solids[p].relaxation.relax_at(c, moduli);
            }
        }
        for plane in &planes {
            for &q in &state.outlines[0] {
                let beyond = -plane.inward * (state.x[q][plane.axis] - plane.at);
                counts.deepest = counts.deepest.max(beyond);
            }
        }
        if let Some(cache) = state.cache.as_mut() {
            // The ramp's friction made no heat that counts (an initial
            // condition), and its factors are of the static system.
            let _ramp_heat = cache.take_heat();
            cache.factors.jacobian.clear();
            cache.factors.responses.clear();
        }
        // The held set the body rests on is the equilibrium's own contacts:
        // recorded as the last held set, the one a rest by the constrained
        // modes starts from (`rest::rest_frame`), as a stepped frame records
        // its own.
        state.last_held = super::rest::held_keys(&state);
        // The ledger from the equilibrium.
        state.start_energy = self.total_lagrangian_energy(&state);
        let kinetic_and_stored: f64 = (0..n)
            .map(|p| {
                0.5 * state.mass[p] * state.v[p].length_squared()
                    + state.v0[p] * state.solids[p].energy(state.f[p])
            })
            .sum();
        let body_mass: f64 = state.mass[..n].iter().sum();
        let outline: Vec<DVec2> = state.outlines[0]
            .iter()
            .map(|&q| state.reference[q])
            .collect();
        state.tolerance_energy =
            tolerance_energy(kinetic_and_stored, gravity, [(body_mass, &outline[..])]);
        state.start_momentum = (0..n).fold(DVec2::ZERO, |sum, p| sum + state.mass[p] * state.v[p]);
        self.write_total_lagrangian_particles(&state);
        self.total_lagrangian = Some(state);
        Ok(counts)
    }
}

/// The rigid motions no touching wall holds: `Ok(None)` when every one is
/// held; `Ok(Some(axis))` when the only free one is a translation along
/// `axis` that the load (gravity) has no component on, a neutral motion the
/// static mode fixes by its gauge (a body on frictionless walls); `Err` with
/// the free motion otherwise (no support, a free rotation, a loaded free
/// axis). Each outline point touching a wall (within its position's
/// round-off, the static mode's test) holds its normal motion and, with
/// friction, its tangential one; the rigid motions (two translations and the
/// rotation about the mass centre, scaled by the body's radius of gyration)
/// are held when the held rows' Gram matrix on them has full rank, judged
/// as the rest's zero modes are (`rest::Basis`, `64 r eps` of the largest).
/// Adaptation, declared: the points' rows stand in for the weighted
/// conditions built from them.
fn free_rigid_mode(
    state: &TotalLagrangianState,
    planes: &[crate::spacetime::total_lagrangian::Plane],
    friction: f64,
    gravity: DVec2,
) -> Result<Option<usize>, [f64; 3]> {
    let n = state.particles;
    let total: f64 = state.mass[..n].iter().sum();
    if total <= 0.0 {
        return Err([1.0, 0.0, 0.0]);
    }
    let centre = (0..n).fold(DVec2::ZERO, |c, p| c + state.mass[p] * state.x[p]) / total;
    let radius = ((0..n)
        .map(|p| state.mass[p] * (state.x[p] - centre).length_squared())
        .sum::<f64>()
        / total)
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let touching = 8.0
        * f64::EPSILON
        * state
            .x
            .iter()
            .map(|q| q.x.abs().max(q.y.abs()))
            .chain(planes.iter().map(|plane| plane.at.abs()))
            .fold(0.0f64, f64::max);
    let mut gram = vec![0.0; 9];
    for &q in &state.outlines[0] {
        let modes = [DVec2::X, DVec2::Y, (state.x[q] - centre).perp() / radius];
        for plane in planes {
            if plane.inward * (state.x[q][plane.axis] - plane.at) > touching {
                continue;
            }
            let mut rows = vec![modes.map(|m| plane.inward * m[plane.axis])];
            if friction > 0.0 {
                rows.push(modes.map(|m| m[1 - plane.axis]));
            }
            for row in rows {
                for i in 0..3 {
                    for j in 0..3 {
                        gram[i * 3 + j] += row[i] * row[j];
                    }
                }
            }
        }
    }
    let (values, vectors) = jacobi_eigen_symmetric_definite(gram.clone(), 3, JACOBI_TOLERANCE);
    let largest = values.iter().fold(0.0f64, |m, &e| m.max(e));
    let floor = 64.0 * 3.0 * f64::EPSILON * largest;
    let least = (0..3)
        .min_by(|&a, &b| values[a].total_cmp(&values[b]))
        .expect("three values");
    if largest > 0.0 && values[least] > floor {
        return Ok(None);
    }
    let free_mode = [0, 1, 2].map(|i| vectors[i * 3 + least]);
    // A single free translation, unloaded, with the two other motions held.
    let free_axes: Vec<usize> = (0..2).filter(|&a| gram[a * 3 + a] <= floor).collect();
    if let [axis] = free_axes[..] {
        let (other, rotation) = (1 - axis, 2);
        let sub = vec![
            gram[other * 3 + other],
            gram[other * 3 + rotation],
            gram[rotation * 3 + other],
            gram[rotation * 3 + rotation],
        ];
        let (sub_values, _) = jacobi_eigen_symmetric_definite(sub, 2, JACOBI_TOLERANCE);
        let held = sub_values.iter().all(|&e| e > floor);
        if held && gravity[axis] == 0.0 {
            return Ok(Some(axis));
        }
    }
    Err(free_mode)
}

/// Two end states' difference in the step controller's energy norm
/// (`controlled_step`'s `stored` part on `F`'s difference; no kinetic part,
/// both at rest).
fn change_norm(a: &TotalLagrangianState, b: &TotalLagrangianState) -> f64 {
    (0..a.particles)
        .map(|p| {
            let g: DMat2 = b.f[p] - a.f[p];
            let shear =
                g.x_axis.x.powi(2) + g.y_axis.y.powi(2) + 0.5 * (g.y_axis.x + g.x_axis.y).powi(2);
            let trace = g.x_axis.x + g.y_axis.y;
            let solid = &a.solids[p];
            let lambda = solid.bulk - solid.mu;
            0.5 * a.v0[p] * (2.0 * solid.mu * shear + lambda * trace * trace)
        })
        .sum::<f64>()
        .max(0.0)
        .sqrt()
}
