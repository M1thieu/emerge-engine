//! A test-only measurement before any build
//! plan: does this engine's basis (quadratic B-splines, nodes at `i h`,
//! 2 x 2 points per cell at the quarter points) carry the axisymmetric
//! weak form, the axis included? A static probe, knob
//! `EMERGE_PROBE_AXISYMMETRIC`, default off.
//!
//! Source: Nairn and Guilkey, *Axisymmetric form of the generalized
//! interpolation material point method*, IJNME 101 (2015) 127-147: masses
//! per radian, the hoop term `P_33 N_i / r` in the internal force (their
//! eq. 3), `v_r = 0` on the axis by a mirrored ghost node (Fig. 6), the
//! classic point limit of their shape functions (sec. 2). The total-
//! Lagrangian form is derived from it (an adaptation,
//! declared): `F_33 = 1 + u_r / R`, weights `A_p R_p`.
//!
//! Adaptations, declared: the static assembly lives here, not in
//! `solve.rs` (a measurement, folded into the path or deleted when the
//! path is built); the Jacobian by central differences; boundaries by
//! reflecting the B-spline coefficients about a node line (the axis and a
//! slip plane: the field's mirror symmetry; a prescribed normal
//! displacement: the affine odd mirror, which also sets the curvature to
//! zero there, an artifact of the imposition); the nodes beyond a free
//! surface keep their own B-splines (not the extended B-splines).

use glam::{DMat2, DVec2};

#[path = "axisymmetric_dynamic.rs"]
mod dynamic;

use super::basis::axis;
use super::multipliers::dense_solve;
use crate::matter::materials::solid::neo_hookean::neo_hookean_second_pk_axisymmetric;

/// Whether `EMERGE_PROBE_AXISYMMETRIC` is set; says so once.
fn knob() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_AXISYMMETRIC").is_some();
        if on {
            println!("knob active: EMERGE_PROBE_AXISYMMETRIC (static probe)");
        }
        on
    })
}

/// How one velocity component's coefficients continue past a boundary
/// node line.
#[derive(Clone, Copy)]
enum Rule {
    /// The nodes beyond are unknowns of their own.
    Free,
    /// `u(b + k) = u(b - k)`.
    Even,
    /// `u(b) = g`, `u(b + k) = 2 g - u(b - k)`, `g` at full load.
    Odd(f64),
}

/// The four sides' rules, per component `[r, z]`.
#[derive(Clone, Copy)]
struct Sides {
    axis: [Rule; 2],
    rim: [Rule; 2],
    base: [Rule; 2],
    top: [Rule; 2],
}

const AXIS: [Rule; 2] = [Rule::Odd(0.0), Rule::Even];
const SLIP_BASE: [Rule; 2] = [Rule::Even, Rule::Odd(0.0)];
const FREE: [Rule; 2] = [Rule::Free, Rule::Free];

/// A static axisymmetric problem on `nr` by `nz` cells of width `h` (SI).
struct Problem {
    nr: i64,
    nz: i64,
    h: f64,
    lambda: f64,
    mu: f64,
    sides: Sides,
    /// Weight per reference volume at full load, along `-z`.
    weight: f64,
    /// Dead nominal traction on the top at full load, along `+z`.
    traction: f64,
    /// Where the quadrature points sit (`Layout`).
    layout: Layout,
    /// Plane strain instead (no weight `R`, no hoop term, `F_33 = 1`), the
    /// axis then a symmetry plane: the same quadrature question asked of
    /// the engine's plane-strain path.
    planar: bool,
}

/// The quadrature: the engine's quarter points,
/// or two diagnostics measured after them: two Gauss points per cell, and
/// two Gauss points per knot span of the quadratic B-splines (the knots
/// sit at the half cells, so a span is `[k - 1/2, k + 1/2] h` cut by the
/// body's edges). Only the last integrates `R N_i` and `R dN_i/dR`
/// exactly, so only it holds a uniform stress as a discrete equilibrium.
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    Quarter,
    CellGauss,
    KnotGauss,
}

impl Layout {
    fn label(self) -> &'static str {
        match self {
            Layout::Quarter => "",
            Layout::CellGauss => " (Gauss per cell, diagnostic)",
            Layout::KnotGauss => " (Gauss per knot span, diagnostic)",
        }
    }

    /// The 1D abscissae over `[0, n h]` and their length weights.
    fn abscissae(self, n: i64, h: f64) -> Vec<(f64, f64)> {
        let g = 0.5 / 3f64.sqrt();
        match self {
            Layout::Quarter => (0..n)
                .flat_map(|k| [0.25, 0.75].map(|t| ((k as f64 + t) * h, 0.5 * h)))
                .collect(),
            Layout::CellGauss => (0..n)
                .flat_map(|k| [0.5 - g, 0.5 + g].map(|t| ((k as f64 + t) * h, 0.5 * h)))
                .collect(),
            Layout::KnotGauss => {
                let mut cuts = vec![0.0];
                cuts.extend((0..n).map(|k| (k as f64 + 0.5) * h));
                cuts.push(n as f64 * h);
                cuts.windows(2)
                    .flat_map(|w| {
                        let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
                        [-1.0, 1.0].map(|s| (mid + s * half / 3f64.sqrt(), half))
                    })
                    .collect()
            }
        }
    }
}

/// A coefficient: `sign x free[k] + constant x load`, or the constant
/// alone.
#[derive(Clone, Copy)]
struct Coefficient {
    free: Option<(usize, f64)>,
    constant: f64,
}

struct Point {
    x: DVec2,
    weight: f64,
    nodes: [(usize, f64, DVec2); 9],
}

struct Discrete {
    map: Vec<[Coefficient; 2]>,
    unknowns: usize,
    points: Vec<Point>,
    /// The external force at full load, per node and component.
    external: Vec<[f64; 2]>,
}

/// One point's state: its reference position, `J`, the Cauchy stress
/// `[rr, zz, rz, thetatheta]` and the nominal `P_zz`, `P_rr`.
struct PointState {
    x: DVec2,
    j: f64,
    sigma: [f64; 4],
    p_zz: f64,
    p_rr: f64,
}

/// The state at one load, and the top's axial displacement on the axis.
struct State {
    points: Vec<PointState>,
    top_axis_uz: f64,
    iterations: usize,
}

impl Problem {
    fn node(&self, i: i64, j: i64) -> usize {
        ((i + 1) * (self.nz + 3) + (j + 1)) as usize
    }

    fn resolve(
        &self,
        i: i64,
        j: i64,
        c: usize,
        own: &mut dyn FnMut(usize) -> usize,
    ) -> Coefficient {
        let fixed = |rule: Rule| match rule {
            Rule::Odd(g) => Some(Coefficient {
                free: None,
                constant: g,
            }),
            _ => None,
        };
        let s = self.sides;
        if let Some(k) = [
            (i == 0, s.axis[c]),
            (i == self.nr, s.rim[c]),
            (j == 0, s.base[c]),
            (j == self.nz, s.top[c]),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .find_map(|(_, rule)| fixed(rule))
        {
            return k;
        }
        let mirror = |rule: Rule, image: Coefficient| match rule {
            Rule::Free => None,
            Rule::Even => Some(image),
            Rule::Odd(g) => Some(Coefficient {
                free: image.free.map(|(k, sign)| (k, -sign)),
                constant: 2.0 * g - image.constant,
            }),
        };
        let beyond = [
            (i == -1, s.axis[c], (1, j)),
            (i == self.nr + 1, s.rim[c], (self.nr - 1, j)),
            (j == -1, s.base[c], (i, 1)),
            (j == self.nz + 1, s.top[c], (i, self.nz - 1)),
        ];
        for (on, rule, (ii, jj)) in beyond {
            if on && !matches!(rule, Rule::Free) {
                let image = self.resolve(ii, jj, c, own);
                return mirror(rule, image).expect("not free");
            }
        }
        Coefficient {
            free: Some((own(self.node(i, j) * 2 + c), 1.0)),
            constant: 0.0,
        }
    }

    fn discretize(&self) -> Discrete {
        let (columns, rows) = (self.nr + 3, self.nz + 3);
        let mut index = vec![usize::MAX; (columns * rows * 2) as usize];
        let mut unknowns = 0;
        let mut own = |slot: usize| {
            if index[slot] == usize::MAX {
                index[slot] = unknowns;
                unknowns += 1;
            }
            index[slot]
        };
        let mut map = vec![
            [Coefficient {
                free: None,
                constant: 0.0
            }; 2];
            (columns * rows) as usize
        ];
        for i in -1..=self.nr + 1 {
            for j in -1..=self.nz + 1 {
                let node = self.node(i, j);
                for (c, slot) in map[node].iter_mut().enumerate() {
                    *slot = self.resolve(i, j, c, &mut own);
                }
            }
        }
        let h = self.h;
        let mut points = Vec::new();
        let mut external = vec![[0.0; 2]; map.len()];
        let along_z = self.layout.abscissae(self.nz, h);
        for &(r, wr) in &self.layout.abscissae(self.nr, h) {
            for &(z, wz) in &along_z {
                {
                    let x = DVec2::new(r, z);
                    let weight = if self.planar { wr * wz } else { wr * wz * r };
                    let (bi, wx, dx) = axis(x.x, 0.0, h);
                    let (bj, wy, dy) = axis(x.y, 0.0, h);
                    let mut nodes = [(0, 0.0, DVec2::ZERO); 9];
                    for (m, node) in nodes.iter_mut().enumerate() {
                        let (p, q) = (m / 3, m % 3);
                        *node = (
                            self.node(bi + p as i64, bj + q as i64),
                            wx[p] * wy[q],
                            DVec2::new(dx[p] * wy[q], wx[p] * dy[q]),
                        );
                        external[node.0][1] -= weight * self.weight * node.1;
                    }
                    points.push(Point { x, weight, nodes });
                }
            }
        }
        if self.traction != 0.0 {
            // Two Gauss points per knot span on the reference top, exact
            // for the piecewise-cubic integrand `N_i(R, H) R` (the first
            // count used two per cell, which straddle the knots).
            let (bj, wy, _) = axis(self.nz as f64 * h, 0.0, h);
            for (r, wr) in Layout::KnotGauss.abscissae(self.nr, h) {
                {
                    let (bi, wx, _) = axis(r, 0.0, h);
                    for (p, wp) in (0i64..).zip(wx) {
                        for (q, wq) in (0i64..).zip(wy) {
                            external[self.node(bi + p, bj + q)][1] +=
                                wr * if self.planar { 1.0 } else { r } * self.traction * wp * wq;
                        }
                    }
                }
            }
        }
        Discrete {
            map,
            unknowns,
            points,
            external,
        }
    }

    /// The coefficients at `free` and load factor `load`.
    fn coefficients(d: &Discrete, free: &[f64], load: f64) -> Vec<DVec2> {
        d.map
            .iter()
            .map(|m| {
                let value = |k: Coefficient| {
                    k.free.map_or(0.0, |(i, sign)| sign * free[i]) + k.constant * load
                };
                DVec2::new(value(m[0]), value(m[1]))
            })
            .collect()
    }

    /// `F_2` and `F_33` at a point (`F_33 = 1` in plane strain).
    fn kinematics(&self, point: &Point, u: &[DVec2]) -> (DMat2, f64) {
        let (f2, f33) = Self::axisymmetric_kinematics(point, u);
        (f2, if self.planar { 1.0 } else { f33 })
    }

    fn axisymmetric_kinematics(point: &Point, u: &[DVec2]) -> (DMat2, f64) {
        let mut grad = DMat2::ZERO;
        let mut ur = 0.0;
        for &(n, w, g) in &point.nodes {
            grad += DMat2::from_cols(u[n] * g.x, u[n] * g.y);
            ur += w * u[n].x;
        }
        (DMat2::IDENTITY + grad, 1.0 + ur / point.x.x)
    }

    /// The residual on the free unknowns and a force scale for its
    /// tolerance; `None` where a point inverts.
    fn residual(&self, d: &Discrete, free: &[f64], load: f64) -> Option<(Vec<f64>, f64)> {
        let u = Self::coefficients(d, free, load);
        let mut force: Vec<[f64; 2]> = d
            .external
            .iter()
            .map(|e| [-e[0] * load, -e[1] * load])
            .collect();
        let mut scale: f64 = d.external.iter().map(|e| e[1].abs()).sum::<f64>() * load;
        for point in &d.points {
            let (f2, f33) = self.kinematics(point, &u);
            if f2.determinant() <= 0.0 || f33 <= 0.0 {
                return None;
            }
            let (s2, s33) = neo_hookean_second_pk_axisymmetric(
                f2.transpose() * f2,
                f33 * f33,
                self.lambda,
                self.mu,
            );
            let (p2, p33) = (f2 * s2, f33 * s33);
            for &(n, w, g) in &point.nodes {
                let planar = p2 * g * point.weight;
                let hoop = if self.planar {
                    0.0
                } else {
                    p33 * w / point.x.x * point.weight
                };
                force[n][0] += planar.x + hoop;
                force[n][1] += planar.y;
                scale += planar.abs().element_sum() + hoop.abs();
            }
        }
        let mut r = vec![0.0; d.unknowns];
        for (m, f) in d.map.iter().zip(&force) {
            for (coefficient, component) in m.iter().zip(f) {
                if let Some((k, sign)) = coefficient.free {
                    r[k] += sign * component;
                }
            }
        }
        Some((r, scale))
    }

    /// Newton on each load in `loads` (increasing), from the last
    /// solution; the state at each.
    fn solve(&self, loads: &[f64]) -> Vec<State> {
        let d = self.discretize();
        let mut free = vec![0.0; d.unknowns];
        let eps = 1e-7 * self.h;
        let mut out = Vec::new();
        for &load in loads {
            let mut iterations = 0;
            loop {
                let (r, scale) = self.residual(&d, &free, load).expect("an inverted point");
                let norm = r.iter().map(|e| e * e).sum::<f64>().sqrt();
                if norm <= 1e-10 * scale {
                    break;
                }
                assert!(
                    iterations < 40,
                    "Newton did not converge at load {load}: {norm:e} of {scale:e}"
                );
                iterations += 1;
                let mut jacobian = vec![vec![0.0; d.unknowns]; d.unknowns];
                for k in 0..d.unknowns {
                    let mut probe = free.clone();
                    probe[k] = free[k] + eps;
                    let (plus, _) = self.residual(&d, &probe, load).expect("an inverted point");
                    probe[k] = free[k] - eps;
                    let (minus, _) = self.residual(&d, &probe, load).expect("an inverted point");
                    for (row, (a, b)) in jacobian.iter_mut().zip(plus.iter().zip(&minus)) {
                        row[k] = (a - b) / (2.0 * eps);
                    }
                }
                let step = dense_solve(jacobian, r.iter().map(|e| -e).collect())
                    .expect("a singular Jacobian");
                let mut length = 1.0;
                loop {
                    let trial: Vec<f64> = free
                        .iter()
                        .zip(&step)
                        .map(|(a, s)| a + length * s)
                        .collect();
                    let better = self
                        .residual(&d, &trial, load)
                        .map(|(t, _)| t.iter().map(|e| e * e).sum::<f64>().sqrt() < norm);
                    if better == Some(true) || length < 1e-3 {
                        free = trial;
                        break;
                    }
                    length *= 0.5;
                }
            }
            out.push(self.state(&d, &free, load, iterations));
        }
        out
    }

    fn state(&self, d: &Discrete, free: &[f64], load: f64, iterations: usize) -> State {
        let u = Self::coefficients(d, free, load);
        let points = d
            .points
            .iter()
            .map(|point| {
                let (f2, f33) = self.kinematics(point, &u);
                let (s2, s33) = neo_hookean_second_pk_axisymmetric(
                    f2.transpose() * f2,
                    f33 * f33,
                    self.lambda,
                    self.mu,
                );
                let (p2, p33) = (f2 * s2, f33 * s33);
                let j = f2.determinant() * f33;
                let sigma = p2 * f2.transpose() * (1.0 / j);
                PointState {
                    x: point.x,
                    j,
                    sigma: [
                        sigma.x_axis.x,
                        sigma.y_axis.y,
                        sigma.y_axis.x,
                        p33 * f33 / j,
                    ],
                    p_zz: p2.y_axis.y,
                    p_rr: p2.x_axis.x,
                }
            })
            .collect();
        let (bi, wx, _) = axis(0.0, 0.0, self.h);
        let (bj, wy, _) = axis(self.nz as f64 * self.h, 0.0, self.h);
        let mut top_axis_uz = 0.0;
        for (p, wp) in (0i64..).zip(wx) {
            for (q, wq) in (0i64..).zip(wy) {
                top_axis_uz += wp * wq * u[self.node(bi + p, bj + q)].y;
            }
        }
        State {
            points,
            top_axis_uz,
            iterations,
        }
    }
}

/// Lame's moduli from Young's modulus and Poisson's ratio.
fn lame(e: f64, nu: f64) -> (f64, f64) {
    (
        e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        e / (2.0 * (1.0 + nu)),
    )
}

/// The root of a monotone `f` on `[a, b]` by bisection.
fn bisect(mut a: f64, mut b: f64, f: impl Fn(f64) -> f64) -> f64 {
    let fa = f(a);
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        if (f(m) > 0.0) == (fa > 0.0) {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

/// Uniform radial compression of a disk between two
/// slip planes, the rim pushed to `u_r = e R_0`.
#[test]
fn uniform_radial_compression_of_a_disk() {
    if !knob() {
        return;
    }
    let (lambda, mu) = lame(2300e6, 0.33);
    let e_final = -0.6;
    let targets = [0.3f64, 0.16];
    for (refine, layout) in [1, 2]
        .into_iter()
        .flat_map(|r| [Layout::Quarter, Layout::CellGauss, Layout::KnotGauss].map(|l| (r, l)))
    {
        let (nr, nz) = (10 * refine, 2 * refine);
        let r0 = 0.01;
        let h = r0 / nr as f64;
        let problem = Problem {
            nr,
            nz,
            h,
            lambda,
            mu,
            sides: Sides {
                axis: AXIS,
                rim: [Rule::Odd(e_final * r0), Rule::Even],
                base: SLIP_BASE,
                top: SLIP_BASE,
            },
            weight: 0.0,
            traction: 0.0,
            layout,
            planar: false,
        };
        let checkpoints: Vec<f64> = targets
            .iter()
            .map(|&j| (j.sqrt() - 1.0) / e_final)
            .collect();
        let mut loads: Vec<f64> = (1..=60).map(|k| f64::from(k) / 60.0).collect();
        loads.extend(&checkpoints);
        loads.sort_by(f64::total_cmp);
        loads.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        let states = problem.solve(&loads);
        for (&target, &load) in targets.iter().zip(&checkpoints) {
            let at = loads
                .iter()
                .position(|&l| (l - load).abs() < 1e-12)
                .expect("a checkpoint");
            let state = &states[at];
            let e = e_final * load;
            let j_exact = (1.0 + e).powi(2);
            let rr = (mu * ((1.0 + e).powi(2) - 1.0) + lambda * j_exact.ln()) / j_exact;
            let zz = lambda * j_exact.ln() / j_exact;
            let size = rr.abs().max(zz.abs());
            let mut worst_j: f64 = 0.0;
            let mut worst_sigma: f64 = 0.0;
            let mut near_axis = Vec::new();
            for point in &state.points {
                let (x, j, s) = (point.x, point.j, point.sigma);
                worst_j = worst_j.max((j / j_exact - 1.0).abs());
                let error = [s[0] - rr, s[1] - zz, s[2], s[3] - rr]
                    .iter()
                    .fold(0.0f64, |m, v| m.max(v.abs()))
                    / size;
                worst_sigma = worst_sigma.max(error);
                if x.x < h {
                    near_axis.push(format!("({:.2}, {:+.3e})", x.x / h, j / j_exact - 1.0));
                }
            }
            println!(
                "compressed disk, cells {nr}x{nz}{} J {target} (exact {j_exact:.6}): worst |J/J_exact - 1| {worst_j:.3e}, worst stress error {worst_sigma:.3e} of {size:.4e} Pa, last Newton iterations {}; nearest the axis (R/h, J error): {}",
                layout.label(),
                state.iterations,
                near_axis.join(" ")
            );
        }
    }
}

/// An end-loaded cylinder (Nairn and Guilkey sec. 4.2
/// geometry), the top under a dead nominal traction.
#[test]
fn end_loaded_cylinder() {
    if !knob() {
        return;
    }
    let (lambda, mu) = lame(2300e6, 0.33);
    let t0 = 600e6;
    // Exact homogeneous uniaxial stress, F = diag(a, b, a): P_rr = 0 gives
    // lambda ln J = mu (1 - a^2), then P_zz = mu (b^2 - a^2) / b = T_0.
    let a2 = |b: f64| b * b - t0 * b / mu;
    let b_low = t0 / mu + 1e-9;
    let b = bisect(b_low.max(1.0), 3.0, |b| {
        lambda * (a2(b) * b).ln() - mu * (1.0 - a2(b))
    });
    let length = 0.04;
    for (refine, layout) in [1, 2]
        .into_iter()
        .flat_map(|r| [Layout::Quarter, Layout::CellGauss, Layout::KnotGauss].map(|l| (r, l)))
    {
        let (nr, nz) = (5 * refine, 20 * refine);
        let h = 0.01 / nr as f64;
        let problem = Problem {
            nr,
            nz,
            h,
            lambda,
            mu,
            sides: Sides {
                axis: AXIS,
                rim: FREE,
                base: SLIP_BASE,
                top: FREE,
            },
            weight: 0.0,
            traction: t0,
            layout,
            planar: false,
        };
        let loads: Vec<f64> = (1..=20).map(|k| f64::from(k) / 20.0).collect();
        let state = problem.solve(&loads).pop().expect("a state");
        let worst_pzz = state
            .points
            .iter()
            .fold(0.0f64, |m, p| m.max((p.p_zz / t0 - 1.0).abs()));
        let worst_prr = state.points.iter().fold(0.0f64, |m, p| m.max(p.p_rr.abs())) / t0;
        let top_exact = (b - 1.0) * length;
        // Where the worst point is, and the worst away from the free
        // surfaces' outermost cells (reported beside the criterion).
        let error = |p: &PointState| (p.p_zz / t0 - 1.0).abs();
        let at = state
            .points
            .iter()
            .max_by(|p, q| error(p).total_cmp(&error(q)))
            .expect("points");
        let inner = |p: &&PointState| p.x.x < (nr - 1) as f64 * h && p.x.y < (nz - 1) as f64 * h;
        let worst_inner = state
            .points
            .iter()
            .filter(inner)
            .fold(0.0f64, |m, p| m.max(error(p)));
        println!(
            "end-loaded cylinder, cells {nr}x{nz}{}: worst P_zz at (R/h, Z/h) ({:.2}, {:.2}); worst away from the free surfaces' outermost cells {worst_inner:.3e}",
            layout.label(),
            at.x.x / h,
            at.x.y / h
        );
        println!(
            "end-loaded cylinder, cells {nr}x{nz}{}: exact stretches b {b:.6} a {:.6}; worst |P_zz/T_0 - 1| {worst_pzz:.3e}, worst |P_rr|/T_0 {worst_prr:.3e}, top on the axis {:.6e} m vs exact {top_exact:.6e} m ({:+.3e} relative), last Newton iterations {}",
            layout.label(),
            a2(b).sqrt(),
            state.top_axis_uz,
            state.top_axis_uz / top_exact - 1.0,
            state.iterations
        );
    }
}

/// A column under its own weight on a slip
/// base, its rim free (uniaxial stress) or held (uniaxial strain).
#[test]
fn column_under_its_own_weight() {
    if !knob() {
        return;
    }
    let (rho, g, height) = (1000.0, 9.81, 0.04);
    for (case, young, held) in [
        ("free-rim column", rho * g * height / 1e-4, false),
        ("held-rim column", 6000.0, true),
    ] {
        let (lambda, mu) = lame(young, 0.33);
        let exact = if held {
            // P(F) = mu (F - 1/F) + lambda ln F / F = -rho g (H - Z);
            // settlement int (F - 1) dZ by Simpson's rule.
            let stretch = |z: f64| {
                let p = -rho * g * (height - z);
                bisect(0.2, 1.0, |f| mu * (f - 1.0 / f) + lambda * f.ln() / f - p)
            };
            let n = 2000;
            let dz = height / f64::from(n);
            let sum: f64 = (0..=n)
                .map(|k| {
                    let w = if k == 0 || k == n {
                        1.0
                    } else if k % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    w * (stretch(f64::from(k) * dz) - 1.0)
                })
                .sum();
            sum * dz / 3.0
        } else {
            -rho * g * height * height / (2.0 * young)
        };
        for (refine, layout) in [1, 2]
            .into_iter()
            .flat_map(|r| [Layout::Quarter, Layout::KnotGauss].map(|l| (r, l)))
        {
            let (nr, nz) = (4 * refine, 40 * refine);
            let problem = Problem {
                nr,
                nz,
                h: height / nz as f64,
                lambda,
                mu,
                sides: Sides {
                    axis: AXIS,
                    rim: if held {
                        [Rule::Odd(0.0), Rule::Even]
                    } else {
                        FREE
                    },
                    base: SLIP_BASE,
                    top: FREE,
                },
                weight: rho * g,
                traction: 0.0,
                layout,
                planar: false,
            };
            let loads: Vec<f64> = if held {
                (1..=4).map(|k| f64::from(k) / 4.0).collect()
            } else {
                vec![1.0]
            };
            let state = problem.solve(&loads).pop().expect("a state");
            println!(
                "{case} cells {nr}x{nz}{}: E {young:.4e} Pa, top settlement on the axis {:.6e} m vs exact {exact:.6e} m ({:+.3e} relative), last Newton iterations {}",
                layout.label(),
                state.top_axis_uz,
                state.top_axis_uz / exact - 1.0,
                state.iterations
            );
        }
    }
}

/// The end-loaded cylinder's question in plane strain: a half block 10 x 40 mm against a symmetry plane, a slip base,
/// the top under a dead nominal traction, the side free. Exact: homogeneous
/// plane-strain uniaxial stress, `F = diag(a, b)`, `P_xx = 0`, `P_yy =
/// T_0`, from this law (`mu (a^2 - 1) + lambda ln(a b) = 0`, `mu (b^2 -
/// a^2) / b = T_0`). The engine's plane-strain path uses the quarter
/// points.
#[test]
fn plane_strain_end_loaded_block() {
    if !knob() {
        return;
    }
    let (lambda, mu) = lame(2300e6, 0.33);
    let t0 = 600e6;
    let a2 = |b: f64| b * b - t0 * b / mu;
    let b = bisect(1.0, 3.0, |b| {
        lambda * (a2(b).sqrt() * b).ln() + mu * (a2(b) - 1.0)
    });
    let length = 0.04;
    for (refine, layout) in [1, 2]
        .into_iter()
        .flat_map(|r| [Layout::Quarter, Layout::KnotGauss].map(|l| (r, l)))
    {
        let (nr, nz) = (5 * refine, 20 * refine);
        let h = 0.01 / nr as f64;
        let problem = Problem {
            nr,
            nz,
            h,
            lambda,
            mu,
            sides: Sides {
                axis: AXIS,
                rim: FREE,
                base: SLIP_BASE,
                top: FREE,
            },
            weight: 0.0,
            traction: t0,
            layout,
            planar: true,
        };
        let loads: Vec<f64> = (1..=20).map(|k| f64::from(k) / 20.0).collect();
        let state = problem.solve(&loads).pop().expect("a state");
        let error = |p: &PointState| (p.p_zz / t0 - 1.0).abs();
        let worst = state.points.iter().fold(0.0f64, |m, p| m.max(error(p)));
        let at = state
            .points
            .iter()
            .max_by(|p, q| error(p).total_cmp(&error(q)))
            .expect("points");
        let inner = |p: &&PointState| p.x.x < (nr - 1) as f64 * h && p.x.y < (nz - 1) as f64 * h;
        let worst_inner = state
            .points
            .iter()
            .filter(inner)
            .fold(0.0f64, |m, p| m.max(error(p)));
        let top_exact = (b - 1.0) * length;
        println!(
            "end-loaded block, plane strain, cells {nr}x{nz}{}: worst |P_yy/T_0 - 1| {worst:.3e} at (x/h, y/h) ({:.2}, {:.2}), away from the free surfaces' outermost cells {worst_inner:.3e}; top at the symmetry plane {:.6e} m vs exact {top_exact:.6e} m ({:+.3e})",
            layout.label(),
            at.x.x / h,
            at.x.y / h,
            state.top_axis_uz,
            state.top_axis_uz / top_exact - 1.0
        );
    }
}
