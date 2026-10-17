//! A lone body resting on the walls carried by its constrained modes: the
//! exact solution of the linearised elastic problem with its held contacts
//! as bilateral constraints, valid while Signorini's conditions hold as
//! equalities (every held multiplier compressive) and no other point
//! reaches a wall. Every change of the held set, every release and every
//! impact stays with the step.
//!
//! The held conditions are the step's own weighted gaps (`weighted_gap`),
//! `G u + g = 0`. Their Lagrangian, `L = T - U + lambda^T (G u + g)`, gives
//! `M u'' + K u = f + G^T lambda` (Kerstens, *A review of methods for
//! constrained eigenvalue problems*, HERON 50(2), 2005, sec. 2, Dowell's
//! method, eqs. 4-10). On the null space of `G`, `u = u_0 + Z w`, the
//! multipliers drop out: `Z^T M Z w'' + Z^T K Z w = Z^T (f - K u_0)`, a
//! standard symmetric problem, solved for its modes as the free body's are
//! (`total_lagrangian::eigen_pairs`); each mode is a forced undamped
//! oscillator, exact. The multipliers follow from the constraint's second
//! derivative, `G M^-1 G^T lambda = G M^-1 (K u - f)`, linear in the modal
//! state; their least value over the rest is bounded from the modes'
//! amplitudes, so a rest frame is flown only while every held multiplier
//! provably stays compressive and every other contact point provably stays
//! clear. Frictionless walls only (a sticking contact's cone is not built):
//! a friction coefficient hands the body to the step.
//!
//! The material energy of the rebuilt state differs from the modes'
//! quadratic one by the linear basis's residual, reported per frame
//! (`TotalLagrangianFrame::modal_energy_change`). Entering, the held
//! points' normal velocity (the step holds it to its tolerance, not to zero)
//! is left out of the constrained modes: its energy is reported there too.
//!
//! A body with relaxation branches the modes carry (one proportional
//! spectrum, `total_lagrangian::damped_system`) rests the same way, each
//! constrained mode a generalised Zener oscillator about its static offset:
//! mechanism `l`'s force is `f_l(t) = K Z Phi s_l(t) + e^(-theta_l t) r_l`,
//! `s_il' + theta_l s_il = kappa_l q_i'`, with `r_l` the part of the start's
//! branch force no constrained mode carries (it only meets the walls, and
//! decays). The static solution is `q_i = b_i / omega_i^2` with no memory;
//! the motion about it goes by the flight's propagator
//! (`damped_propagator`), its energy never growing, which bounds the
//! multipliers and the free points' gaps as the undamped amplitudes do,
//! and the strain against the record's linear limit. The branches are
//! rebuilt at each frame's end as in flight (`Relaxation::carried`), the
//! heat split the same way (declared there).

use crate::spacetime::total_lagrangian::{
    Band, CarriedMemory, ContactKey, Grid, JACOBI_TOLERANCE, Plane, System, damped_propagator,
    damped_system, eigen_pairs, field_at, symmetrise,
};
use crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite;
use glam::{DMat2, DVec2};

use super::TotalLagrangianState;

/// `a b` for row-major `a` (`rows` by `inner`) and `b` (`inner` by `cols`).
fn times(a: &[f64], (rows, inner): (usize, usize), b: &[f64], cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|ij| {
            let (i, j) = (ij / cols, ij % cols);
            (0..inner).map(|e| a[i * inner + e] * b[e * cols + j]).sum()
        })
        .collect()
}

/// `a^T` for row-major `a` (`rows` by `cols`).
fn transpose(a: &[f64], (rows, cols): (usize, usize)) -> Vec<f64> {
    (0..rows * cols)
        .map(|ij| a[(ij % rows) * cols + ij / rows])
        .collect()
}

/// What a rest needs of its held set alone, built once per set and kept
/// (`TotalLagrangianState`): the constrained modes and the multipliers'
/// response to them.
#[derive(Clone)]
pub(crate) struct Basis {
    pub(crate) keys: Vec<ContactKey>,
    unknowns: Vec<usize>,
    n: usize,
    m: Vec<f64>,
    k: Vec<f64>,
    band: Band,
    /// The constrained modes, ascending, `0` for a zero mode (a slide the
    /// frictionless walls let free): their columns on the unknowns, their
    /// values at every point and their material gradients at the particles.
    omega: Vec<f64>,
    columns: Vec<Vec<f64>>,
    at: Vec<Vec<DVec2>>,
    gradients: Vec<Vec<DMat2>>,
    /// `M^-1 G^T` by held condition, and `S = G M^-1 G^T` decomposed.
    minv_gt: Vec<Vec<f64>>,
    schur: (Vec<f64>, Vec<f64>),
    /// Per mode, the multipliers per unit of its coordinate.
    coupling: Vec<Vec<f64>>,
    /// The (point, wall) pairs the held conditions hold.
    held_pairs: Vec<(usize, usize)>,
    /// The body's relaxation mechanisms as `(theta, kappa)` (empty without)
    /// and its record's strain limit.
    memory: Vec<(f64, f64)>,
    strain_limit: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Basis {
    /// The basis of the held set the last step ended on, or `None` when the
    /// body cannot rest by its modes: a law without modes, friction, a held
    /// condition not a wall's.
    pub(crate) fn build(
        state: &TotalLagrangianState,
        (grid, planes, friction): (&Grid, &[Plane], f64),
    ) -> Option<Self> {
        let cache = state.cache.as_ref()?;
        if friction > 0.0 || cache.contacts.is_empty() || state.outlines.len() != 1 {
            return None;
        }
        let trace = cache.traces.first()?;
        let mut held = Vec::new();
        for contact in &cache.contacts {
            let (kind, function, plane) = contact.key;
            if kind != 3 {
                return None;
            }
            let members = trace.functions.get(function)?;
            let total: f64 = members.iter().map(|&(_, w)| w).sum();
            if total <= 0.0 {
                return None;
            }
            let points: Vec<(usize, f64)> = members
                .iter()
                .map(|&(r, w)| (state.outlines[0][r], w / total))
                .collect();
            held.push((contact.key, points, *planes.get(plane)?));
        }
        held.sort_by_key(|h| h.0);
        let particles = state.particles;
        let points = state.x.len();
        let (System { unknowns, n, m, k }, memory, strain_limit) =
            damped_system(grid, (&state.mass, &state.v0, &state.solids), particles)?;
        let mut slot = vec![usize::MAX; grid.len];
        for (i, &u) in unknowns.iter().enumerate() {
            slot[u] = i;
        }
        // `G`: one row per held condition, `dg/du`.
        let rows = held.len();
        let mut g = vec![0.0; rows * n];
        for (h, (_, members, plane)) in held.iter().enumerate() {
            for &(q, w) in members {
                grid.visit(q, plane.axis, |kk, nn, _| {
                    if slot[kk] != usize::MAX {
                        g[h * n + slot[kk]] += w * plane.inward * nn;
                    }
                });
            }
        }
        // The null space of `G`: the eigenvectors of `G^T G + I` at 1, the
        // `n - rows` least (the weighted gaps' rows are independent by
        // construction, `weighted_gap`, so `G` has rank `rows`).
        let mut normal = times(&transpose(&g, (rows, n)), (n, rows), &g, n);
        for i in 0..n {
            normal[i * n + i] += 1.0;
        }
        symmetrise(&mut normal, n);
        let (values, vectors) = jacobi_eigen_symmetric_definite(normal, n, JACOBI_TOLERANCE);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let r = n.checked_sub(rows).filter(|&r| r > 0)?;
        let z: Vec<f64> = (0..n * r)
            .map(|ij| vectors[(ij / r) * n + order[ij % r]])
            .collect();
        let zt = transpose(&z, (n, r));
        let (mz, kz) = (
            times(&zt, (r, n), &times(&m, (n, n), &z, r), r),
            times(&zt, (r, n), &times(&k, (n, n), &z, r), r),
        );
        let (squared, psi, shift) = eigen_pairs(mz, &kz, r)?;
        // A zero mode: `omega^2` within the eigen-solve's own round-off of
        // its scale (Demmel and Veselic's relative accuracy).
        let floor = 64.0 * r as f64 * f64::EPSILON * shift.abs();
        let omega: Vec<f64> = squared
            .iter()
            .map(|&w2| if w2 <= floor { 0.0 } else { w2.sqrt() })
            .collect();
        let phi = times(&z, (n, r), &psi, r);
        let columns: Vec<Vec<f64>> = (0..r)
            .map(|e| (0..n).map(|i| phi[i * r + e]).collect())
            .collect();
        let band = grid.mass_band(&state.mass, 0.0);
        let inverse = |y: &[f64]| -> Vec<f64> {
            let mut full = vec![0.0; grid.len];
            for (&u, &value) in unknowns.iter().zip(y) {
                full[u] = value;
            }
            let solved = band.solve(&full);
            unknowns.iter().map(|&u| solved[u]).collect()
        };
        let minv_gt: Vec<Vec<f64>> = (0..rows).map(|h| inverse(&g[h * n..(h + 1) * n])).collect();
        let mut s = vec![0.0; rows * rows];
        for a in 0..rows {
            for b in 0..rows {
                s[a * rows + b] = dot(&g[a * n..(a + 1) * n], &minv_gt[b]);
            }
        }
        symmetrise(&mut s, rows);
        let schur = jacobi_eigen_symmetric_definite(s, rows, JACOBI_TOLERANCE);
        if schur.0.iter().any(|&e| e <= 0.0) {
            return None;
        }
        let (at, gradients) = columns
            .iter()
            .map(|c| field_at(grid, (&unknowns, c), (particles, points)))
            .unzip();
        let held_pairs = held
            .iter()
            .flat_map(|(key, members, _)| members.iter().map(move |&(q, _)| (q, key.2)))
            .collect();
        let mut basis = Self {
            keys: held.iter().map(|(key, _, _)| *key).collect(),
            unknowns,
            n,
            m,
            k,
            band,
            omega,
            columns,
            at,
            gradients,
            minv_gt,
            schur,
            coupling: Vec::new(),
            held_pairs,
            memory,
            strain_limit,
        };
        basis.coupling = (0..r)
            .map(|e| basis.multipliers(&times(&basis.k, (n, n), &basis.columns[e], 1)))
            .collect();
        Some(basis)
    }

    /// The multipliers a force `x` on the unknowns calls for, `S^-1 G M^-1
    /// x`.
    fn multipliers(&self, x: &[f64]) -> Vec<f64> {
        let y: Vec<f64> = self.minv_gt.iter().map(|c| dot(c, x)).collect();
        let (sd, sq) = &self.schur;
        let rows = y.len();
        (0..rows)
            .map(|a| {
                (0..rows)
                    .map(|e| {
                        let along: f64 = (0..rows).map(|b| sq[b * rows + e] * y[b]).sum();
                        sq[a * rows + e] * along / sd[e]
                    })
                    .sum()
            })
            .collect()
    }
}

/// One rest: its start time, its basis, and its modal state.
#[derive(Clone)]
pub(crate) struct Rest {
    t: f64,
    pub(crate) basis: Basis,
    /// The displacement at the start: at every point, and its material
    /// gradient at every particle.
    start_at: Vec<DVec2>,
    start_gradients: Vec<DMat2>,
    /// Per mode, the constant forcing `b = Phi^T (f - K u_0)` and the start
    /// rate `q'(0) = Phi^T M v_0`.
    forcing: Vec<f64>,
    rate: Vec<f64>,
    /// Per held condition, its multiplier at the start.
    multiplier: Vec<f64>,
    /// The other contact points against each wall: (point, plane index,
    /// gap at the start).
    free: Vec<(usize, usize, f64)>,
    /// The memory of a body with relaxation branches.
    damped: Option<Damped>,
    /// What the last rest frame made: each particle's heat, J/m, and the
    /// largest Green strain size at its end.
    pub(crate) heat: Vec<f64>,
    pub(crate) strain: f64,
}

/// A damped rest's modes, advanced one frame at a time.
#[derive(Clone)]
struct Damped {
    /// Per mechanism, `e^(-theta length)`, and `length`, s.
    decay: Vec<f64>,
    length: f64,
    /// Per mode, its propagator (empty for a zero mode) and its static
    /// offset `b / omega^2` (zero for a zero mode).
    propagators: Vec<Vec<f64>>,
    offset: Vec<f64>,
    /// Per mode, `y = (omega (q - q_p), q', omega s_l / sqrt(kappa_l))` at
    /// `t`, and at the frame before (empty for a zero mode).
    y: Vec<Vec<f64>>,
    previous: Vec<Vec<f64>>,
    t: f64,
    /// Per mode and mechanism, `s_il` at the rest's start.
    start_memory: Vec<Vec<f64>>,
    /// Per mechanism, the multipliers of `r_l`, and its `theta`.
    outside: Vec<(f64, Vec<f64>)>,
}

/// Per damped mode, `(omega, y at the last frame's start, y at its end)`,
/// and the mechanisms `(theta, kappa)` (`Rest::damped_modes`).
#[cfg(all(test, feature = "experimental"))]
pub(crate) type DampedModes = (Vec<(f64, Vec<f64>, Vec<f64>)>, Vec<(f64, f64)>);

/// Why a rest did not start, with its basis back (boxed, it is large).
pub(crate) struct Refused {
    pub(crate) basis: Box<Basis>,
    /// Whether its strain could pass the record's linear limit (counted;
    /// a multiplier not compressive is not).
    pub(crate) beyond_limit: bool,
}

fn squared(y: &[f64]) -> f64 {
    y.iter().map(|e| e * e).sum()
}

fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

impl Rest {
    /// The rest from the state at time `t` on `basis`, its frames `length`
    /// long, or the basis back when a held multiplier is not compressive or
    /// a damped body's strain could pass its record's limit.
    pub(crate) fn start(
        basis: Basis,
        state: &TotalLagrangianState,
        (grid, planes): (&Grid, &[Plane]),
        (gravity, t, length): (DVec2, f64, f64),
    ) -> Result<Self, Refused> {
        let (particles, points, n) = (state.particles, state.x.len(), basis.n);
        // The state's displacement and velocity as grid coefficients: the
        // consistent-mass projections, exact (the grid stays on the
        // reference, so both lie in the basis).
        let project = |f: &dyn Fn(usize) -> DVec2| -> Vec<f64> {
            let mut b = vec![0.0; grid.len];
            for p in 0..particles {
                let value = state.mass[p] * f(p);
                for a in 0..2 {
                    grid.visit(p, a, |kk, nn, _| b[kk] += nn * value[a]);
                }
            }
            let full = basis.band.solve(&b);
            basis.unknowns.iter().map(|&u| full[u]).collect()
        };
        let u0 = project(&|p| state.x[p] - state.reference[p]);
        let v0 = project(&|p| state.v[p]);
        let mut slot = vec![usize::MAX; grid.len];
        for (i, &u) in basis.unknowns.iter().enumerate() {
            slot[u] = i;
        }
        let mut load = vec![0.0; n];
        for p in 0..particles {
            for a in 0..2 {
                let s = state.mass[p] * gravity[a];
                grid.visit(p, a, |kk, nn, _| {
                    if slot[kk] != usize::MAX {
                        load[slot[kk]] += nn * s;
                    }
                });
            }
        }
        let ku0 = times(&basis.k, (n, n), &u0, 1);
        let mv0 = times(&basis.m, (n, n), &v0, 1);
        let residual: Vec<f64> = load.iter().zip(&ku0).map(|(f, ku)| f - ku).collect();
        let forcing: Vec<f64> = basis.columns.iter().map(|c| dot(c, &residual)).collect();
        let rate: Vec<f64> = basis.columns.iter().map(|c| dot(c, &mv0)).collect();
        let mut start_force: Vec<f64> = ku0.iter().zip(&load).map(|(ku, f)| ku - f).collect();
        // The branches' linearised force per mechanism on the unknowns,
        // `sum_p V0_p sigma_lp : grad N` (the stiffness's own assembly).
        let memory_force: Vec<Vec<f64>> = (0..basis.memory.len())
            .map(|l| {
                let mut out = vec![0.0; n];
                for p in 0..particles {
                    let solid = &state.solids[p];
                    let sigma = solid
                        .relaxation
                        .linear_stress(state.f[p].transpose() * state.f[p], solid.lame())[l];
                    for a in 0..2 {
                        let row = DVec2::new(sigma.x_axis[a], sigma.y_axis[a]);
                        grid.visit(p, a, |kk, _, grad| {
                            if slot[kk] != usize::MAX {
                                out[slot[kk]] += state.v0[p] * row.dot(grad);
                            }
                        });
                    }
                }
                out
            })
            .collect();
        for f in &memory_force {
            for (s, e) in start_force.iter_mut().zip(f) {
                *s += e;
            }
        }
        let multiplier = basis.multipliers(&start_force);
        if multiplier.iter().any(|&l| l <= 0.0) {
            return Err(Refused {
                basis: Box::new(basis),
                beyond_limit: false,
            });
        }
        let (start_at, start_gradients) =
            field_at(grid, (&basis.unknowns, &u0), (particles, points));
        let damped = (!basis.memory.is_empty()).then(|| {
            Damped::start(
                &basis,
                (forcing.as_slice(), rate.as_slice()),
                &memory_force,
                (t, length),
            )
        });
        let free = state.outlines[0]
            .iter()
            .flat_map(|&q| (0..planes.len()).map(move |i| (q, i)))
            .filter(|pair| !basis.held_pairs.contains(pair))
            .map(|(q, i)| {
                let plane = planes[i];
                (q, i, plane.inward * (state.x[q][plane.axis] - plane.at))
            })
            .collect();
        let rest = Self {
            t,
            basis,
            start_at,
            start_gradients,
            forcing,
            rate,
            multiplier,
            free,
            damped,
            heat: Vec::new(),
            strain: 0.0,
        };
        if rest.damped.is_some() && rest.strain_bound(0.0) > rest.basis.strain_limit {
            return Err(Refused {
                basis: Box::new(rest.basis),
                beyond_limit: true,
            });
        }
        Ok(rest)
    }

    /// For the gates: per damped mode, its `omega`, its state at the last
    /// frame's start and end, and the mechanisms `(theta, kappa)`.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn damped_modes(&self) -> Option<DampedModes> {
        let damped = self.damped.as_ref()?;
        let modes = (0..self.basis.omega.len())
            .filter(|&i| !damped.y[i].is_empty())
            .map(|i| {
                (
                    self.basis.omega[i],
                    damped.previous[i].clone(),
                    damped.y[i].clone(),
                )
            })
            .collect();
        Some((modes, self.basis.memory.clone()))
    }

    /// `start` without the strain check, for the probes' `settle`: the
    /// resting state of a body whose current motion is beyond its record's
    /// limit (it is then placed at rest, inside it).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn start_for_settling(
        basis: Basis,
        state: &TotalLagrangianState,
        (grid, planes): (&Grid, &[Plane]),
        (gravity, t, length): (DVec2, f64, f64),
    ) -> Result<Self, Refused> {
        let limit = basis.strain_limit;
        let mut unchecked = basis;
        unchecked.strain_limit = f64::INFINITY;
        let mut rest = Self::start(unchecked, state, (grid, planes), (gravity, t, length))?;
        rest.basis.strain_limit = limit;
        Ok(rest)
    }

    /// A bound on the Green strain size over the rest to `tau1` after its
    /// start (from the current frame on, with memory): `b + b^2 / 2` with
    /// `b = max_p |grad u_static| + sum_i |grad phi_i| a_i`, `a_i` each
    /// mode's reach about its static offset (`|y_i| / omega_i` with memory,
    /// its energy never growing; a zero mode's excursion to `tau1`).
    fn strain_bound(&self, tau1: f64) -> f64 {
        let Some(damped) = &self.damped else {
            return 0.0;
        };
        let reach: Vec<f64> = (0..self.basis.omega.len())
            .map(|i| {
                let w = self.basis.omega[i];
                if w == 0.0 {
                    self.reach(i, tau1).1
                } else {
                    squared(&damped.y[i]).sqrt() / w
                }
            })
            .collect();
        let b = (0..self.start_gradients.len())
            .map(|p| {
                let fixed = self
                    .basis
                    .gradients
                    .iter()
                    .zip(&damped.offset)
                    .fold(self.start_gradients[p], |g, (grad, &q)| g + grad[p] * q);
                colon(fixed, fixed).sqrt()
                    + self
                        .basis
                        .gradients
                        .iter()
                        .zip(&reach)
                        .map(|(grad, &a)| colon(grad[p], grad[p]).sqrt() * a)
                        .sum::<f64>()
            })
            .fold(0.0, f64::max);
        b + 0.5 * b * b
    }

    /// Mode `i`'s coordinate and rate at `tau` after the start: a forced
    /// undamped oscillator from rest at zero displacement, or for a zero
    /// mode a uniformly accelerated slide; with memory, the damped state at
    /// the last frame's end about the static offset.
    fn coordinate(&self, i: usize, tau: f64) -> (f64, f64) {
        let (w, b, q0) = (self.basis.omega[i], self.forcing[i], self.rate[i]);
        if w == 0.0 {
            return (q0 * tau + 0.5 * b * tau * tau, q0 + b * tau);
        }
        if let Some(damped) = &self.damped {
            let y = &damped.y[i];
            return (damped.offset[i] + y[0] / w, y[1]);
        }
        let (s, c) = (w * tau).sin_cos();
        (b / (w * w) * (1.0 - c) + q0 / w * s, b / w * s + q0 * c)
    }

    /// The most a mode's coordinate leaves its centre over `[0, tau1]`: for
    /// an oscillator its amplitude about `b / omega^2`, at any time; for a
    /// zero mode its largest excursion over the interval.
    fn reach(&self, i: usize, tau1: f64) -> (f64, f64) {
        let (w, b, q0) = (self.basis.omega[i], self.forcing[i], self.rate[i]);
        if w == 0.0 {
            let end = (q0 * tau1 + 0.5 * b * tau1 * tau1).abs();
            let vertex = if b != 0.0 { -q0 / b } else { -1.0 };
            let inside = if vertex > 0.0 && vertex < tau1 {
                (q0 * vertex + 0.5 * b * vertex * vertex).abs()
            } else {
                0.0
            };
            return (0.0, end.max(inside));
        }
        let centre = b / (w * w);
        (centre, (centre * centre + (q0 / w) * (q0 / w)).sqrt())
    }

    /// Whether the rest holds to `t1`: every held multiplier stays
    /// compressive and every other contact point clear of every wall, by
    /// bounds from the modes' amplitudes.
    pub(crate) fn holds(&self, planes: &[Plane], t1: f64) -> bool {
        let tau1 = t1 - self.t;
        if let Some(damped) = &self.damped {
            return self.damped_holds(damped, planes, tau1);
        }
        let reaches: Vec<(f64, f64)> = (0..self.basis.omega.len())
            .map(|i| self.reach(i, tau1))
            .collect();
        let compressive = self.multiplier.iter().enumerate().all(|(h, &l0)| {
            let low = reaches
                .iter()
                .zip(&self.basis.coupling)
                .fold(l0, |low, (&(centre, amplitude), c)| {
                    low + c[h] * centre - c[h].abs() * amplitude
                });
            low > 0.0
        });
        compressive
            && self.free.iter().all(|&(q, i, gap)| {
                let plane = planes[i];
                let moved: f64 = reaches
                    .iter()
                    .zip(&self.basis.at)
                    .map(|(&(centre, amplitude), at)| {
                        at[q][plane.axis].abs() * (centre.abs() + amplitude)
                    })
                    .sum();
                gap - moved > 0.0
            })
    }

    /// The resting state, written into the step's state, for the probes: each
    /// mode at its static offset `b / omega^2` and at rest, but for the mode
    /// the load offsets most, given the rate whose amplitude is `kick` of
    /// that offset. A body placed at its resting compression.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn settle(&self, kick: f64, state: &mut TotalLagrangianState) {
        let offset = |i: usize| {
            let w = self.basis.omega[i];
            if w == 0.0 {
                0.0
            } else {
                self.forcing[i] / (w * w)
            }
        };
        let loaded = (0..self.basis.omega.len())
            .max_by(|&i, &j| offset(i).abs().total_cmp(&offset(j).abs()))
            .unwrap_or(0);
        let coordinates: Vec<(f64, f64)> = (0..self.basis.omega.len())
            .map(|i| {
                let rate = if i == loaded {
                    kick * offset(i).abs() * self.basis.omega[i]
                } else {
                    0.0
                };
                (offset(i), rate)
            })
            .collect();
        for p in 0..state.x.len() {
            let (mut u, mut v) = (self.start_at[p], DVec2::ZERO);
            for (at, &(q, rate)) in self.basis.at.iter().zip(&coordinates) {
                u += at[p] * q;
                v += at[p] * rate;
            }
            state.x[p] = state.reference[p] + u;
            state.v[p] = v;
            if p < state.particles {
                let g = self
                    .basis
                    .gradients
                    .iter()
                    .zip(&coordinates)
                    .fold(self.start_gradients[p], |g, (grad, &(q, _))| {
                        g + grad[p] * q
                    });
                state.f[p] = DMat2::IDENTITY + g;
            }
        }
    }

    /// `holds` with memory, from the current frame's start, `tau0 = tau1 -
    /// length`. The multipliers are `lambda_0 + sum_i c_i (q_i + sum_l
    /// (s_il - s_il(0))) + sum_l d_l (e^(-theta_l tau) - 1)`. Each mode's
    /// `|(q_i - q_p) + sum_l s_il|` is at most `sqrt(1 + sum_l kappa_l)
    /// |y_i| / omega_i` (Cauchy-Schwarz on `y`, which never grows), and the
    /// `d_l` term is monotone in `tau` (its least at an end of the frame).
    /// The free points' gaps take `|q_i - q_p| <= |y_i| / omega_i`, and the
    /// strain stays within the record's limit.
    fn damped_holds(&self, damped: &Damped, planes: &[Plane], tau1: f64) -> bool {
        let tau0 = tau1 - damped.length;
        let weight: f64 = (1.0
            + self
                .basis
                .memory
                .iter()
                .map(|&(_, kappa)| kappa)
                .sum::<f64>())
        .sqrt();
        // Per mode: its centre and its reach about it, for the multipliers
        // and for the gaps.
        let modes: Vec<(f64, f64, f64)> = (0..self.basis.omega.len())
            .map(|i| {
                let w = self.basis.omega[i];
                if w == 0.0 {
                    let (centre, amplitude) = self.reach(i, tau1);
                    return (centre, amplitude, amplitude);
                }
                let size = squared(&damped.y[i]).sqrt() / w;
                let memory: f64 = damped.start_memory[i].iter().sum();
                (damped.offset[i] - memory, weight * size, size)
            })
            .collect();
        let compressive = self.multiplier.iter().enumerate().all(|(h, &l0)| {
            let mut low = modes
                .iter()
                .zip(&self.basis.coupling)
                .fold(l0, |low, (&(centre, amplitude, _), c)| {
                    low + c[h] * centre - c[h].abs() * amplitude
                });
            for (theta, d) in &damped.outside {
                let at = |tau: f64| d[h] * ((-theta * tau).exp() - 1.0);
                low += at(tau0.max(0.0)).min(at(tau1));
            }
            low > 0.0
        });
        compressive
            && self.free.iter().all(|&(q, i, gap)| {
                let plane = planes[i];
                let moved: f64 = modes
                    .iter()
                    .zip(&self.basis.at)
                    .zip(&damped.offset)
                    .map(|((&(_, _, amplitude), at), &offset)| {
                        at[q][plane.axis].abs() * (offset.abs() + amplitude)
                    })
                    .sum();
                gap - moved > 0.0
            })
            && self.strain_bound(tau1) <= self.basis.strain_limit
    }

    /// The rest's state at `t`, written into the step's state; with memory,
    /// `t` is one frame after the last, and false (the state untouched) when
    /// a computed propagator lengthened a mode's state (`flight`'s check).
    pub(crate) fn rebuild(&mut self, t: f64, state: &mut TotalLagrangianState) -> bool {
        if let Some(damped) = &mut self.damped {
            assert!(
                (t - damped.t - damped.length).abs()
                    <= 4.0 * f64::EPSILON * t.abs().max(damped.length),
                "a damped rest advances one frame at a time"
            );
            let next: Vec<Vec<f64>> = damped
                .y
                .iter()
                .zip(&damped.propagators)
                .map(|(y, p)| {
                    let k = y.len();
                    (0..k)
                        .map(|r| (0..k).map(|c| p[r * k + c] * y[c]).sum())
                        .collect()
                })
                .collect();
            if next
                .iter()
                .zip(&damped.y)
                .any(|(a, b)| squared(a) > squared(b))
            {
                return false;
            }
            damped.previous = std::mem::replace(&mut damped.y, next);
            damped.t = t;
        }
        let tau = t - self.t;
        let coordinates: Vec<(f64, f64)> = (0..self.basis.omega.len())
            .map(|i| self.coordinate(i, tau))
            .collect();
        for p in 0..state.x.len() {
            let (mut u, mut v) = (self.start_at[p], DVec2::ZERO);
            for (at, &(q, rate)) in self.basis.at.iter().zip(&coordinates) {
                u += at[p] * q;
                v += at[p] * rate;
            }
            state.x[p] = state.reference[p] + u;
            state.v[p] = v;
            if p < state.particles {
                let g = self
                    .basis
                    .gradients
                    .iter()
                    .zip(&coordinates)
                    .fold(self.start_gradients[p], |g, (grad, &(q, _))| {
                        g + grad[p] * q
                    });
                state.f[p] = DMat2::IDENTITY + g;
            }
        }
        true
    }

    /// After a kept damped rest frame (`rebuild`), each particle's branches
    /// rebuilt from the constrained modes' memory (`Relaxation::carried`),
    /// from the deformation gradients `start` at the frame's start; returns
    /// the heat made, J/m, its split kept in `heat` (as in flight) and the
    /// largest strain in `strain`. Without memory, nothing.
    pub(crate) fn carry_branches(
        &mut self,
        state: &mut TotalLagrangianState,
        start: &[DMat2],
    ) -> f64 {
        self.heat.clear();
        self.strain = 0.0;
        let Some(damped) = &self.damped else {
            return 0.0;
        };
        let (omega, memory) = (&self.basis.omega, &self.basis.memory);
        let modal: f64 = damped
            .y
            .iter()
            .zip(&damped.previous)
            .map(|(y1, y0)| 0.5 * (squared(y0) - squared(y1)))
            .sum();
        let s = |y: &[f64], i: usize, l: usize| y[2 + l] * memory[l].1.sqrt() / omega[i];
        let n = state.particles;
        let (mut heat, mut stored) = (vec![0.0; n], vec![0.0; n]);
        for p in 0..n {
            let mut carried: Vec<CarriedMemory> = memory
                .iter()
                .zip(&damped.decay)
                .map(|(&(_, kappa), &decay)| CarriedMemory {
                    decay,
                    kappa,
                    start: DMat2::ZERO,
                    end: DMat2::ZERO,
                })
                .collect();
            for (i, (y1, y0)) in damped.y.iter().zip(&damped.previous).enumerate() {
                if y1.is_empty() {
                    continue;
                }
                let g = self.basis.gradients[i][p];
                let sym = (g + g.transpose()) * 0.5;
                for (l, c) in carried.iter_mut().enumerate() {
                    c.start += sym * s(y0, i, l);
                    c.end += sym * s(y1, i, l);
                }
            }
            let (c0, c1) = (
                start[p].transpose() * start[p],
                state.f[p].transpose() * state.f[p],
            );
            let solid = &mut state.solids[p];
            let moduli = solid.lame();
            let before = solid.relaxation.energy(c0, moduli);
            heat[p] = state.v0[p] * solid.relaxation.carried((c0, c1), moduli, &carried);
            stored[p] = state.v0[p] * (before + solid.relaxation.energy(c1, moduli));
            let green = (c1 - DMat2::IDENTITY) * 0.5;
            self.strain = self.strain.max(colon(green, green).sqrt());
        }
        let total: f64 = stored.iter().sum();
        let volume: f64 = state.v0[..n].iter().sum();
        for p in 0..n {
            heat[p] += modal
                * if total > 0.0 {
                    stored[p] / total
                } else {
                    state.v0[p] / volume
                };
        }
        let made = heat.iter().sum();
        self.heat = heat;
        made
    }
}

impl Damped {
    /// The memory at a rest's start: each constrained mode's `s_il =
    /// c_i^T f_l / omega_i^2` from the branches' force `f_l`, its state
    /// about the static offset (the coordinates start at zero), the
    /// multipliers of the part `r_l = f_l - sum_i K c_i s_il` no mode
    /// carries, and the propagators over `length`.
    fn start(
        basis: &Basis,
        (forcing, rate): (&[f64], &[f64]),
        memory_force: &[Vec<f64>],
        (t, length): (f64, f64),
    ) -> Self {
        let n = basis.n;
        let count = basis.omega.len();
        let start_memory: Vec<Vec<f64>> = (0..count)
            .map(|i| {
                let w = basis.omega[i];
                memory_force
                    .iter()
                    .map(|f| {
                        if w == 0.0 {
                            0.0
                        } else {
                            dot(&basis.columns[i], f) / (w * w)
                        }
                    })
                    .collect()
            })
            .collect();
        let outside = basis
            .memory
            .iter()
            .enumerate()
            .map(|(l, &(theta, _))| {
                let mut r = memory_force[l].clone();
                for (i, column) in basis.columns.iter().enumerate() {
                    if start_memory[i][l] == 0.0 {
                        continue;
                    }
                    let kc = times(&basis.k, (n, n), column, 1);
                    for (e, v) in r.iter_mut().zip(kc) {
                        *e -= v * start_memory[i][l];
                    }
                }
                (theta, basis.multipliers(&r))
            })
            .collect();
        let offset: Vec<f64> = (0..count)
            .map(|i| {
                let w = basis.omega[i];
                if w == 0.0 { 0.0 } else { forcing[i] / (w * w) }
            })
            .collect();
        let y: Vec<Vec<f64>> = (0..count)
            .map(|i| {
                let w = basis.omega[i];
                if w == 0.0 {
                    return Vec::new();
                }
                let mut y = vec![-w * offset[i], rate[i]];
                for (l, &(_, kappa)) in basis.memory.iter().enumerate() {
                    y.push(w * start_memory[i][l] / kappa.sqrt());
                }
                y
            })
            .collect();
        let propagators = y
            .iter()
            .zip(&basis.omega)
            .map(|(y, &w)| {
                if y.is_empty() {
                    Vec::new()
                } else {
                    damped_propagator(w, &basis.memory, length)
                }
            })
            .collect();
        Self {
            decay: basis
                .memory
                .iter()
                .map(|&(theta, _)| (-theta * length).exp())
                .collect(),
            length,
            propagators,
            offset,
            previous: y.clone(),
            y,
            t,
            start_memory,
            outside,
        }
    }
}

/// The keys of the contacts the last step held, sorted.
pub(crate) fn held_keys(state: &TotalLagrangianState) -> Vec<ContactKey> {
    let mut keys: Vec<ContactKey> = state
        .cache
        .as_ref()
        .map(|c| c.contacts.iter().map(|contact| contact.key).collect())
        .unwrap_or_default();
    keys.sort_unstable();
    keys
}

/// One frame of a lone body's rest carried by its constrained modes, when
/// it may rest (`Rest`): the last two stepped frames ended on the same held
/// set, and the bounds hold to the frame's end. False hands the frame to
/// the step, the rest (if any) ended there.
pub(crate) fn rest_frame(
    state: &mut TotalLagrangianState,
    (planes, friction, gravity): (&[Plane], f64, DVec2),
    (t0, length): (f64, f64),
) -> bool {
    let t1 = t0 + length;
    if state.rest.is_none() {
        let keys = held_keys(state);
        if keys.is_empty() || keys != state.last_held || state.flight.is_some() {
            return false;
        }
        let Some(cache) = state.cache.as_ref() else {
            return false;
        };
        // The held set's basis, kept from the last attempt on the same set.
        let kept = state.rest_basis.take();
        let basis = match kept {
            Some(basis) if basis.keys == keys => Some(basis),
            _ => Basis::build(state, (&cache.grid, planes, friction)),
        };
        let Some(basis) = basis else {
            // A body whose branches the constrained modes do not carry (two
            // spectra, a non-proportional record) is counted; friction and
            // a condition not a wall's are not (as before).
            let relaxing = state.solids[..state.particles]
                .iter()
                .any(|s| s.is_relaxing());
            state.modal_rest_refused += u64::from(relaxing && friction == 0.0);
            return false;
        };
        let cache = state.cache.as_ref().expect("checked above");
        match Rest::start(basis, state, (&cache.grid, planes), (gravity, t0, length)) {
            Ok(rest) => state.rest = Some(rest),
            Err(refused) => {
                state.modal_rest_refused += u64::from(refused.beyond_limit);
                state.rest_basis = Some(*refused.basis);
                return false;
            }
        }
    }
    let Some(mut rest) = state.rest.take() else {
        return false;
    };
    if !rest.holds(planes, t1) {
        state.rest_hand_backs += 1;
        state.rest_basis = Some(rest.basis);
        return false;
    }
    let start = state.f.clone();
    if !rest.rebuild(t1, state) {
        state.modal_check_failed += 1;
        state.rest_hand_backs += 1;
        state.rest_basis = Some(rest.basis);
        return false;
    }
    rest.carry_branches(state, &start);
    state.rest = Some(rest);
    state.modal_rest_frames += 1;
    true
}
