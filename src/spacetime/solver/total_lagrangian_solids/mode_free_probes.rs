//! The mode-free clearance bound against the modal one at each flight start
//! (MODE_FREE_HANDOFF_PLAN.md, M0): per contact point, `(2 E tr C_p)^(1/2)`
//! with `C_p` the point's 2 x 2 static compliance on the elastic subspace
//! and `E` the vibration energy, against the flight's `sum_i |phi_i(p)|
//! |z_i| / omega_i`; and the cut each gives. A probe, switched by
//! `EMERGE_PROBE_MODE_FREE`; nothing held.

use super::*;
use crate::matter::materials::solid::corotated::stress::corotated_second_pk_tangent;
use crate::spacetime::total_lagrangian::Plane;

/// Whether `EMERGE_PROBE_MODE_FREE` is set; says so once, from inside.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_MODE_FREE").is_some();
        if on {
            println!("knob active: mode-free clearance bound compared at each flight start");
        }
        on
    })
}

/// Whether `EMERGE_PROBE_MODE_FREE_CUT` is set: the path's flights then cut
/// by the mode-free bound (a non-default probe); says so once.
pub(crate) fn cut_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_MODE_FREE_CUT").is_some();
        if on {
            println!("knob active: flights cut by the mode-free bound (trace form)");
        }
        on
    })
}

/// The lower Cholesky factor of a row-major `n x n` definite matrix.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            l[i * n + j] = if i == j {
                assert!(s > 0.0, "not definite at {i}: {s:e}");
                s.sqrt()
            } else {
                s / l[j * n + j]
            };
        }
    }
    l
}

/// `A^-1 b` from `cholesky(A)`.
fn solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..n {
        y[i] = (y[i] - (0..i).map(|k| l[i * n + k] * y[k]).sum::<f64>()) / l[i * n + i];
    }
    for i in (0..n).rev() {
        y[i] = (y[i] - (i + 1..n).map(|k| l[k * n + i] * y[k]).sum::<f64>()) / l[i * n + i];
    }
    y
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The last time in `[t0, t1]` the flight's clearance bound holds,
/// bisected as `fly_as_far_as_clear` does.
fn reach(flight: &Flight, (planes, gravity): (&[Plane], DVec2), (t0, t1): (f64, f64)) -> f64 {
    if flight.clear(gravity, planes, (t0, t1)) {
        return t1;
    }
    let (mut low, mut high) = (t0, t1);
    while high - low > 4.0 * f64::EPSILON * high.abs().max(t1 - t0) {
        let middle = 0.5 * (low + high);
        if flight.clear(gravity, planes, (t0, middle)) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

/// The mode-free quantities of a body's state: per contact point (the
/// outlines' points in order) its 2 x 2 compliance on the elastic subspace,
/// the vibration energy from the state, and the rigid fit the flight uses.
pub(crate) struct Free {
    points: Vec<usize>,
    compliance: Vec<DMat2>,
    energy: f64,
    kinetic: f64,
    stored: f64,
    rigid_residual: f64,
    unknowns: usize,
    centre: DVec2,
    velocity: DVec2,
    spin: f64,
    angle: f64,
    origin: DVec2,
}

impl Free {
    /// Per contact point, the trace form `(2 E tr C_p)^(1/2)`.
    pub(crate) fn bounds(&self) -> Vec<f64> {
        self.compliance
            .iter()
            .map(|c| (2.0 * self.energy * (c.x_axis.x + c.y_axis.y)).sqrt())
            .collect()
    }
}

/// `Free` from the state, with no eigenvalue: the consistent mass and the
/// stiffness at rest assembled on the unknowns, the rigid motions fitted by
/// the mass, `K + (M R)(M R)^T` factored, one solve per point and axis.
pub(crate) fn free(state: &TotalLagrangianState) -> Free {
    let all: Vec<usize> = (0..state.particles).collect();
    let points: Vec<usize> = state.outlines.iter().flatten().copied().collect();
    free_of(state, &all, &points)
}

/// `free` for one body of several: its particles and its outline's points;
/// its unknowns are those its particles reach (bodies share none,
/// `Grid::empty_band`), so its `M` and `K` are its own blocks.
pub(crate) fn free_of(state: &TotalLagrangianState, particles: &[usize], points: &[usize]) -> Free {
    let grid = &state
        .cache
        .as_ref()
        .expect("a flight starts from a step")
        .grid;
    let mut touched = vec![false; grid.len];
    for &p in particles {
        for a in 0..2 {
            grid.visit(p, a, |k, _, _| touched[k] = true);
        }
    }
    let unknowns: Vec<usize> = (0..grid.len)
        .filter(|&k| grid.active[k] && touched[k])
        .collect();
    let n = unknowns.len();
    let mut slot = vec![usize::MAX; grid.len];
    for (i, &k) in unknowns.iter().enumerate() {
        slot[k] = i;
    }
    // The consistent mass and the stiffness at rest on the unknowns (the
    // corotated tangent `vibration`'s probe uses, the elastic part of any
    // law), assembled column by column: no eigenvalues.
    let (mut m, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (j, &column) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[column] = 1.0;
        let mc = grid.mass_times(&unit, &state.mass, 0.0);
        let mut kc = vec![0.0; grid.len];
        for &p in particles {
            let (solid, volume) = (state.solids[p], state.v0[p]);
            let (_, gu) = grid.read(&unit, p, 0);
            let (_, gv) = grid.read(&unit, p, 1);
            let g = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
            let ds = corotated_second_pk_tangent(
                DMat2::IDENTITY,
                g + g.transpose(),
                solid.bulk - solid.mu,
                solid.mu,
            );
            for a in 0..2 {
                let row = DVec2::new(ds.x_axis[a], ds.y_axis[a]);
                grid.visit(p, a, |q, _, grad| kc[q] += volume * row.dot(grad));
            }
        }
        for (i, &r) in unknowns.iter().enumerate() {
            m[i * n + j] = mc[r];
            k[i * n + j] = kc[r];
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (
                0.5 * (m[i * n + j] + m[j * n + i]),
                0.5 * (k[i * n + j] + k[j * n + i]),
            );
            (m[i * n + j], m[j * n + i], k[i * n + j], k[j * n + i]) = (a, a, b, b);
        }
    }
    let lm = cholesky(&m, n);
    // `b = sum_p m_p N(X_p) f(X_p)`: the mass-weighted load of a field
    // given at the particles; `M^-1 b` its projection on the unknowns
    // (exact for the linear fields the basis reproduces).
    let load = |f: &dyn Fn(usize) -> DVec2| -> Vec<f64> {
        let mut b = vec![0.0; n];
        for &p in particles {
            let value = f(p);
            for a in 0..2 {
                grid.visit(p, a, |q, w, _| {
                    if slot[q] != usize::MAX {
                        b[slot[q]] += state.mass[p] * value[a] * w;
                    }
                });
            }
        }
        b
    };
    let total: f64 = particles.iter().map(|&p| state.mass[p]).sum();
    let origin = particles
        .iter()
        .copied()
        .fold(DVec2::ZERO, |s, p| s + state.mass[p] * state.reference[p])
        / total;
    // The rigid motions, M-orthonormal (Gram-Schmidt in `M`).
    let fields: [&dyn Fn(usize) -> DVec2; 3] = [&|_| DVec2::X, &|_| DVec2::Y, &|p| {
        (state.reference[p] - origin).perp()
    }];
    let mut rigid: Vec<Vec<f64>> = Vec::new();
    for f in fields {
        let mut r = solve(&lm, &load(f), n);
        for s in &rigid {
            let c = dot(s, &mat_vec(&m, &r, n));
            r.iter_mut().zip(s).for_each(|(e, se)| *e -= c * se);
        }
        let norm = dot(&r, &mat_vec(&m, &r, n)).sqrt();
        r.iter_mut().for_each(|e| *e /= norm);
        rigid.push(r);
    }
    let mr: Vec<Vec<f64>> = rigid.iter().map(|r| mat_vec(&m, r, n)).collect();
    let k_scale = (0..n).map(|i| k[i * n + i]).fold(0.0, f64::max);
    let rigid_residual = rigid
        .iter()
        .map(|r| {
            let kr = mat_vec(&k, r, n);
            dot(&kr, &kr).sqrt() / (k_scale * dot(r, r).sqrt())
        })
        .fold(0.0, f64::max);
    // `K + (M R)(M R)^T`: definite, and equal to `K` on the elastic subspace.
    let mut k_reg = k.clone();
    for v in &mr {
        for i in 0..n {
            for j in 0..n {
                k_reg[i * n + j] += v[i] * v[j];
            }
        }
    }
    let lk = cholesky(&k_reg, n);
    // Each contact point's 2 x 2 compliance `C_p = e~^T K^+ e~`, `e~ = (I -
    // M R R^T) e` the point's rows with the rigid part removed.
    let points = points.to_vec();
    let compliance: Vec<DMat2> = points
        .iter()
        .map(|&o| {
            let rows: Vec<Vec<f64>> = (0..2)
                .map(|a| {
                    let mut e = vec![0.0; n];
                    grid.visit(o, a, |q, w, _| {
                        if slot[q] != usize::MAX {
                            e[slot[q]] += w;
                        }
                    });
                    for (r, v) in rigid.iter().zip(&mr) {
                        let c = dot(r, &e);
                        e.iter_mut().zip(v).for_each(|(x, vx)| *x -= c * vx);
                    }
                    e
                })
                .collect();
            let y: Vec<Vec<f64>> = rows.iter().map(|e| solve(&lk, e, n)).collect();
            let c = |a: usize, b: usize| 0.5 * (dot(&rows[a], &y[b]) + dot(&rows[b], &y[a]));
            DMat2::from_cols(DVec2::new(c(0, 0), c(1, 0)), DVec2::new(c(0, 1), c(1, 1)))
        })
        .collect();
    // The vibration energy, from the state (the flight's own rigid fit:
    // centre, velocity, spin, angle), kinetic relative plus stored `1/2
    // d^T K d` of the relative displacement's projection.
    let (sum_x, sum_v) = particles
        .iter()
        .copied()
        .fold((DVec2::ZERO, DVec2::ZERO), |(a, b), p| {
            (
                a + state.mass[p] * state.x[p],
                b + state.mass[p] * state.v[p],
            )
        });
    let (centre, velocity) = (sum_x / total, sum_v / total);
    let (momentum, inertia) = particles.iter().copied().fold((0.0, 0.0), |(l, i), p| {
        let arm = state.x[p] - centre;
        (
            l + state.mass[p] * arm.perp_dot(state.v[p] - velocity),
            i + state.mass[p] * arm.length_squared(),
        )
    });
    let spin = momentum / inertia;
    let a = particles.iter().copied().fold(DMat2::ZERO, |a, p| {
        let (r, s) = (state.x[p] - centre, state.reference[p] - origin);
        a + state.mass[p] * DMat2::from_cols(r * s.x, r * s.y)
    });
    let back = DMat2::from_angle(-(a.x_axis.y - a.y_axis.x).atan2(a.x_axis.x + a.y_axis.y));
    let kinetic: f64 = particles
        .iter()
        .copied()
        .map(|p| {
            let arm = state.x[p] - centre;
            0.5 * state.mass[p]
                * (back * (state.v[p] - velocity - spin * arm.perp())).length_squared()
        })
        .sum();
    let mut d = solve(
        &lm,
        &load(&|p| back * (state.x[p] - centre) - (state.reference[p] - origin)),
        n,
    );
    for (r, v) in rigid.iter().zip(&mr) {
        let c = dot(v, &d);
        d.iter_mut().zip(r).for_each(|(x, rx)| *x -= c * rx);
    }
    let stored = 0.5 * dot(&d, &mat_vec(&k, &d, n));
    Free {
        points,
        compliance,
        energy: kinetic + stored,
        kinetic,
        stored,
        rigid_residual,
        unknowns: n,
        centre,
        velocity,
        spin,
        angle: (a.x_axis.y - a.y_axis.x).atan2(a.x_axis.x + a.y_axis.y),
        origin,
    }
}

/// Per contact point, the mode-free lower bound on its least gap to every
/// wall over `[t0, t1]`: `Flight::clear`'s terms (the centre's parabola, the
/// arm turned at the measured spin, the turning term) with the modal
/// vibration bound replaced by the trace form (MODE_FREE_HANDOFF_PLAN.md
/// sec. 3); the fit taken at the state's time.
pub(crate) fn lower_gaps(
    state: &TotalLagrangianState,
    free: &Free,
    (planes, gravity): (&[Plane], DVec2),
    (t0, t1): (f64, f64),
) -> Vec<f64> {
    let at = |t: f64| {
        let tau = t - state.time;
        free.centre + free.velocity * tau + 0.5 * gravity * tau * tau
    };
    let turn = DMat2::from_angle(free.angle + free.spin * (t0 - state.time));
    free.points
        .iter()
        .zip(free.bounds())
        .map(|(&o, bound)| {
            let reference = state.reference[o] - free.origin;
            let turning = free.spin.abs() * reference.length() * (t1 - t0);
            planes
                .iter()
                .map(|plane| {
                    let along = |t: f64| plane.inward * at(t)[plane.axis];
                    let mut lowest = along(t0).min(along(t1));
                    let acceleration = plane.inward * gravity[plane.axis];
                    if acceleration != 0.0 {
                        let vertex =
                            state.time - plane.inward * free.velocity[plane.axis] / acceleration;
                        if vertex > t0 && vertex < t1 {
                            lowest = lowest.min(along(vertex));
                        }
                    }
                    lowest + plane.inward * (turn * reference)[plane.axis]
                        - plane.inward * plane.at
                        - bound
                        - turning
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// ROW15_REACH_CUT_PLAN.md sec. 7: per contact point, the UPPER bound on
/// its least gap to the walls at the instant `t`: `lower_gaps`' terms with
/// the bound's sign reversed, the rigid fit's gap at `t` (the arm turned at
/// `t0`'s angle) plus the vibration bound and the turning term over
/// `[t0, t]`. The least over the planes of a point's upper bounds bounds its
/// least gap from above.
pub(crate) fn upper_gaps_at(
    state: &TotalLagrangianState,
    free: &Free,
    planes: &[Plane],
    gravity: DVec2,
    (t0, t): (f64, f64),
) -> Vec<f64> {
    let tau = t - state.time;
    let at = free.centre + free.velocity * tau + 0.5 * gravity * tau * tau;
    let turn = DMat2::from_angle(free.angle + free.spin * (t0 - state.time));
    free.points
        .iter()
        .zip(free.bounds())
        .map(|(&o, bound)| {
            let reference = state.reference[o] - free.origin;
            let turning = free.spin.abs() * reference.length() * (t - t0);
            planes
                .iter()
                .map(|plane| {
                    plane.inward * at[plane.axis] + plane.inward * (turn * reference)[plane.axis]
                        - plane.inward * plane.at
                        + bound
                        + turning
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// The last time in `[t0, t1]` up to which every contact point's mode-free
/// lower gap bound stays positive, bisected (the bound is a minimum over the
/// interval, so monotone in its end).
pub(crate) fn clear_until(
    state: &TotalLagrangianState,
    free: &Free,
    walls: (&[Plane], DVec2),
    (t0, t1): (f64, f64),
) -> f64 {
    let clear = |t: f64| {
        lower_gaps(state, free, walls, (t0, t))
            .iter()
            .all(|&g| g > 0.0)
    };
    if clear(t1) {
        return t1;
    }
    let (mut low, mut high) = (t0, t1);
    while high - low > 4.0 * f64::EPSILON * high.abs().max(t1 - t0) {
        let middle = 0.5 * (low + high);
        if clear(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

/// M0 at one flight start: prints both bounds' extremes per point, the
/// Cauchy-Schwarz check with the modal energy, and the cuts.
pub(super) fn compare(
    state: &TotalLagrangianState,
    flight: &Flight,
    walls: (&[Plane], DVec2),
    (t0, t1): (f64, f64),
) {
    if !on() {
        return;
    }
    let Free {
        points,
        compliance,
        energy: free_energy,
        kinetic,
        stored,
        rigid_residual,
        unknowns: n,
        ..
    } = free(state);
    let sizes = flight.mode_sizes();
    let modal_energy = 0.5 * sizes.iter().map(|s| s * s).sum::<f64>();
    let modal = flight.arm_bounds();
    let trace: Vec<f64> = compliance.iter().map(|c| c.x_axis.x + c.y_axis.y).collect();
    let largest: Vec<f64> = compliance
        .iter()
        .map(|c| {
            let (p, q, r) = (c.x_axis.x, c.y_axis.y, c.y_axis.x);
            0.5 * (p + q) + (0.25 * (p - q) * (p - q) + r * r).sqrt()
        })
        .collect();
    // Cauchy-Schwarz with the modal energy: `B_modal <= (2 E_modal tr C)^(1/2)`.
    let check = modal
        .iter()
        .zip(&trace)
        .map(|(&b, &t)| ((2.0 * modal_energy * t).sqrt() - b) / b.max(f64::MIN_POSITIVE))
        .fold(f64::INFINITY, f64::min);
    let free: Vec<f64> = trace
        .iter()
        .map(|&t| (2.0 * free_energy * t).sqrt())
        .collect();
    let free_largest: Vec<f64> = largest
        .iter()
        .map(|&l| (2.0 * free_energy * l).sqrt())
        .collect();
    let ratio = |b: &[f64]| {
        b.iter()
            .zip(&modal)
            .map(|(f, m)| f / m.max(f64::MIN_POSITIVE))
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), r| {
                (lo.min(r), hi.max(r))
            })
    };
    let (lo, hi) = ratio(&free);
    let (lo_l, hi_l) = ratio(&free_largest);
    let cut = [
        reach(flight, walls, (t0, t1)),
        reach(&flight.with_arm_bounds(&free), walls, (t0, t1)),
        reach(&flight.with_arm_bounds(&free_largest), walls, (t0, t1)),
    ];
    println!(
        "M0 flight start t {t0:.6} to {t1:.6}: {n} unknowns, {} contact points, memory {}, rigid fit |K R| {rigid_residual:.1e}; energy modal {modal_energy:.6e}, from the state {free_energy:.6e} (kinetic {kinetic:.3e}, stored {stored:.3e}); C-S check (2 E_modal tr C)^(1/2) / B_modal - 1, least {check:+.3e}; mode-free / modal per point: trace form {lo:.3} to {hi:.3}, largest-eigenvalue form {lo_l:.3} to {hi_l:.3}; cut modal {:.6}, trace form {:.6}, largest-eigenvalue form {:.6} (flight given up {:.3e} s, {:.3e} s)",
        points.len(),
        flight.has_memory(),
        cut[0],
        cut[1],
        cut[2],
        cut[0] - cut[1],
        cut[0] - cut[2]
    );
}

fn mat_vec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| dot(&a[i * n..(i + 1) * n], x)).collect()
}

/// M1' (MODE_FREE_HANDOFF_PLAN.md sec. 10): per contact point, the stepped
/// trajectory's deviation term over one step of `dt`, `D_p = (dt^2/2)
/// (tr(E_p M^-1 E_p^T))^(1/2) (n / lambda_min(M))^(1/2) R_a`, with `R_a =
/// max(tol x scale, floor)` as `solve.rs` forms them: `scale` from the
/// inertial term `2 |M v_n|_inf / dt` (`M v_n` taken as the particles'
/// momentum load, the grid velocity's mass projection) and the weight's
/// `|M g|_inf` (the start stress's term left out: a body in free fall
/// carries none, declared); `floor` 16 eps times the largest stiffness row
/// sum. Also returns `R_a` and `lambda_min(M)`.
pub(crate) fn solve_term(
    state: &TotalLagrangianState,
    gravity: DVec2,
    (dt, tolerance): (f64, f64),
) -> (Vec<f64>, f64, f64) {
    let all: Vec<usize> = (0..state.particles).collect();
    let points: Vec<usize> = state.outlines.iter().flatten().copied().collect();
    solve_term_of(state, gravity, (dt, tolerance), (&all, &points))
}

/// `solve_term` for one body of several: its own `M` block and unknowns,
/// the step's `R_a` over every particle (the stop test is global).
pub(crate) fn solve_term_of(
    state: &TotalLagrangianState,
    gravity: DVec2,
    (dt, tolerance): (f64, f64),
    (particles, points): (&[usize], &[usize]),
) -> (Vec<f64>, f64, f64) {
    let grid = &state.cache.as_ref().expect("built by a step").grid;
    let n_p = state.particles;
    let mut touched = vec![false; grid.len];
    for &p in particles {
        for a in 0..2 {
            grid.visit(p, a, |k, _, _| touched[k] = true);
        }
    }
    let unknowns: Vec<usize> = (0..grid.len)
        .filter(|&k| grid.active[k] && touched[k])
        .collect();
    let n = unknowns.len();
    let mut slot = vec![usize::MAX; grid.len];
    for (i, &k) in unknowns.iter().enumerate() {
        slot[k] = i;
    }
    let mut m = vec![0.0; n * n];
    for (j, &column) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[column] = 1.0;
        let mc = grid.mass_times(&unit, &state.mass, 0.0);
        for (i, &r) in unknowns.iter().enumerate() {
            m[i * n + j] = mc[r];
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let a = 0.5 * (m[i * n + j] + m[j * n + i]);
            (m[i * n + j], m[j * n + i]) = (a, a);
        }
    }
    let (eigen, _) = crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite(
        m.clone(),
        n,
        1e-14,
    );
    let lambda_min = eigen.iter().copied().fold(f64::INFINITY, f64::min);
    let lm = cholesky(&m, n);
    let (mut momentum, mut weight, mut rows) = (
        vec![0.0; grid.len],
        vec![0.0; grid.len],
        vec![0.0; grid.len],
    );
    for p in 0..n_p {
        let f = state.f[p];
        let size = (f.x_axis.length_squared() + f.y_axis.length_squared()).sqrt();
        let stiffness = state.v0[p] * (state.solids[p].bulk + state.solids[p].mu) * size;
        for a in 0..2 {
            grid.visit(p, a, |k, w, g| {
                momentum[k] += state.mass[p] * state.v[p][a] * w;
                weight[k] += state.mass[p] * gravity[a] * w;
                rows[k] += stiffness * g.length();
            });
        }
    }
    let largest = |v: &[f64]| v.iter().fold(0.0f64, |a, &e| a.max(e.abs()));
    let scale = (2.0 / dt * largest(&momentum)).max(largest(&weight));
    let floor = 16.0 * f64::EPSILON * largest(&rows);
    let allowed = (tolerance * scale).max(floor);
    let terms = points
        .iter()
        .map(|&o| {
            let trace: f64 = (0..2)
                .map(|a| {
                    let mut e = vec![0.0; n];
                    grid.visit(o, a, |q, w, _| {
                        if slot[q] != usize::MAX {
                            e[slot[q]] += w;
                        }
                    });
                    dot(&e, &solve(&lm, &e, n))
                })
                .sum();
            0.5 * dt * dt * trace.sqrt() * (n as f64 / lambda_min).sqrt() * allowed
        })
        .collect();
    (terms, allowed, lambda_min)
}

/// M2 (MODE_FREE_HANDOFF_PLAN.md sec. 12): the lower bound on the
/// separation of bodies `a` and `b` along the unit `n` over `[t0, t1]`:
/// the centres' relative motion (linear: a uniform gravity cancels) at the
/// worse end, less each body's support of its turned reference arms, their
/// turning terms and vibration bounds (`free_of`'s trace form).
pub(crate) fn pair_lower(
    state: &TotalLagrangianState,
    (a, b): (&Free, &Free),
    n: DVec2,
    (t0, t1): (f64, f64),
) -> f64 {
    let relative = |t: f64| {
        let tau = t - state.time;
        n.dot((b.centre - a.centre) + (b.velocity - a.velocity) * tau)
    };
    let support = |body: &Free, sign: f64| -> f64 {
        let turn = DMat2::from_angle(body.angle + body.spin * (t0 - state.time));
        body.points
            .iter()
            .zip(body.bounds())
            .map(|(&o, bound)| {
                let arm = state.reference[o] - body.origin;
                sign * n.dot(turn * arm) + body.spin.abs() * arm.length() * (t1 - t0) + bound
            })
            .fold(f64::NEG_INFINITY, f64::max)
    };
    relative(t0).min(relative(t1)) - support(a, 1.0) - support(b, -1.0)
}

/// The measured separation of two point sets along `n`: `min_q n.x_q -
/// max_p n.x_p`, `p` in `a`, `q` in `b`.
pub(crate) fn separation(
    state: &TotalLagrangianState,
    (a, b): (&[usize], &[usize]),
    n: DVec2,
) -> f64 {
    let low = b
        .iter()
        .map(|&q| n.dot(state.x[q]))
        .fold(f64::INFINITY, f64::min);
    let high = a
        .iter()
        .map(|&p| n.dot(state.x[p]))
        .fold(f64::NEG_INFINITY, f64::max);
    low - high
}

/// The candidate directions for a pair: each outline edge's unit normal,
/// both senses, and the centre-to-centre direction.
pub(crate) fn directions(
    state: &TotalLagrangianState,
    outlines: [&[usize]; 2],
    (a, b): (&Free, &Free),
) -> Vec<DVec2> {
    let mut out = vec![(b.centre - a.centre).normalize_or_zero()];
    for outline in outlines {
        for (i, &p) in outline.iter().enumerate() {
            let q = outline[(i + 1) % outline.len()];
            let normal = (state.x[q] - state.x[p]).perp().normalize_or_zero();
            out.push(normal);
            out.push(-normal);
        }
    }
    out.retain(|n| n.length_squared() > 0.0);
    out
}
