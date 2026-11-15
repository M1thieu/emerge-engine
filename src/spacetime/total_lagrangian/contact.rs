//! Where two surfaces touch, measured on the grid's unknowns: the gap, the
//! normal, how a contact's force spreads over particles, and the couple that
//! keeps friction from changing the angular momentum. The contact law itself
//! lives in `forces::boundary::signorini_coulomb`.

use super::*;

/// A wall's plane: the material stays on its inward side, `inward (x[axis]
/// - at) >= 0`.
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub axis: usize,
    pub at: f64,
    pub inward: f64,
    /// The plane's speed along its axis, `d at / dt`, constant over a step:
    /// a wall driven at a prescribed motion, as a press's platen; zero for a
    /// wall at rest. `at` is its place at the step's start. The same
    /// Signorini conditions on the relative motion, and the push's work is
    /// the press's.
    pub speed: f64,
}

impl Plane {
    /// The plane `elapsed` later.
    pub(crate) fn after(&self, elapsed: f64) -> Self {
        Self {
            at: self.at + self.speed * elapsed,
            ..*self
        }
    }

    /// The same plane for a condition whose start gap is `gap`: one already
    /// across holds where it is, not pushed back out, so the push does no
    /// positive work (Laursen and Love, IJNME 53 (2002) eq. 74, the positive part
    /// of the last gap, and eq. 71, the condition for their impulse to exist).
    pub(crate) fn at_most_across(self, gap: f64) -> Self {
        Self {
            at: self.at + self.inward * gap.min(0.0),
            ..self
        }
    }
}

/// One point of a contact: a particle held against a wall's plane (or a
/// partner across it), or against another body's segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ContactPoint {
    pub(crate) particle: usize,
    pub(crate) plane: usize,
    /// For a contact between two bodies, the other body's particle
    /// (`step_total_lagrangian_contact`): the plane then measures the two
    /// particles' separation, and the push acts on both, equal and opposite.
    pub(crate) partner: Option<usize>,
    /// For a point against another body's boundary, that boundary's
    /// segment, its two end particles in counter-clockwise order (outward
    /// normal on their right): the plane is then not used
    /// (`point_geometry`).
    pub(crate) segment: Option<(usize, usize)>,
    /// With a segment, the point's two neighbours on its own outline
    /// (previous, next, counter-clockwise): the normal is then the point's
    /// own nodal normal and the closest point is where the line along it
    /// meets the segment (`point_geometry`).
    pub(crate) nodal: Option<(usize, usize)>,
}

/// Which condition a contact is, to find it again from pass to pass: a
/// kind, then two indices (a point and its plane, or a multiplier function
/// and its obstacle, `weighted_gap`).
pub(crate) type ContactKey = (u8, usize, usize);

/// One contact condition held by the augmented Lagrangian (`step`): the
/// weighted mean of its points' gaps, `g = sum_r w_r g_r` (`sum_r w_r =
/// 1`), never negative at the step's end, and its multipliers, the force
/// the obstacle exerts, shared over the points by the same weights. One
/// point of weight 1 is a collocation contact; several, the weighted gap
/// of a multiplier function (`weighted_gap`).
#[derive(Clone, Debug)]
pub(crate) struct Contact {
    pub(crate) points: Box<[(ContactPoint, f64)]>,
    pub(crate) key: ContactKey,
    pub(crate) multiplier: f64,
    /// The friction's multiplier, the tangential force along each point's
    /// normal turned a quarter counter-clockwise (`contact_force`); zero
    /// without friction.
    pub(crate) tangential: f64,
}

impl Contact {
    /// A collocation contact: one point, weight 1.
    pub(crate) fn at(point: ContactPoint, key: ContactKey) -> Self {
        Self {
            points: Box::new([(point, 1.0)]),
            key,
            multiplier: 0.0,
            tangential: 0.0,
        }
    }

    /// The point of a collocation contact.
    pub(crate) fn single(&self) -> &ContactPoint {
        debug_assert_eq!(self.points.len(), 1, "a collocation contact");
        &self.points[0].0
    }
}

/// A contact's augmented-Lagrangian penalty, its points' weighted.
pub(crate) fn contact_penalty(c: &Contact, penalty: &[f64]) -> f64 {
    c.points.iter().map(|(p, w)| w * penalty[p.particle]).sum()
}

/// The friction's augmentation parameter `k_T` for one solve, shared by its
/// contacts: `1 / max_i sum_j |W_ij|`, `W = (dt^2 / 2) B M_eps^-1 B^T`, `B`
/// the contacts' tangential rows (`g_T = B u`). `None` when a row reaches
/// an unknown without mass (no inertia bounds the step there).
///
/// Why: with a fixed multiplier, the solve's friction term has slope
/// `A + k_T B^T B` sticking and `A` slipping (`A` the step's operator
/// without contacts, `2 M_eps / dt^2 + K`). Alart, J. Math. Pures Appl. 76
/// (1997), Thm 6 and eq. 57: a generalized Newton on such a function
/// converges if its slopes stay within a ratio of 2, and otherwise cycles
/// between the two slip regions when the solution sticks (his Fig. 5);
/// eq. 59 carries the bound to `n` unknowns through the Uzawa contraction,
/// `r < 2 / lambda_max(B A^-1 B^T)`. Adaptation, declared: on our nested
/// form (the multipliers fixed inside a solve) the 1-unknown condition
/// reads `k_T < a` (slopes `a + k_T` and `a`), so `k_T lambda_max(B A^-1
/// B^T) < 1` is asked. With `K` positive semidefinite, `A >= 2 M_eps /
/// dt^2`, so `B A^-1 B^T <= W`, and Gershgorin bounds `lambda_max(W)` by
/// its largest absolute row sum: the value returned meets both bounds. A
/// material modulus as `k_T` broke them by 6e3 to 2e4 on a block at rest
/// on a Coulomb floor (measured), every Newton step of a fresh contact's
/// solve then crossing the stick region.
pub(crate) fn tangential_penalty(
    grid: &Grid,
    x: &[DVec2],
    planes: &[Plane],
    contacts: &[Contact],
    mass: &Band,
    dt: f64,
    inertia: f64,
) -> Option<f64> {
    let zero = vec![0.0; grid.len];
    let massive = grid.massive();
    let mut rows = Vec::with_capacity(contacts.len());
    for c in contacts {
        let (_, shares) = contact_geometry(grid, x, planes, &zero, c);
        let mut row = vec![0.0; grid.len];
        for share in &shares {
            let tangent = share.normal.perp();
            for &(r, weight) in &share.spread {
                if weight == 0.0 {
                    continue;
                }
                for a in 0..2 {
                    let part = share.weight * weight * tangent[a];
                    if part != 0.0 {
                        grid.visit(r, a, |k, w, _| row[k] += w * part);
                    }
                }
            }
        }
        for (k, value) in row.iter_mut().enumerate() {
            if !grid.active[k] {
                *value = 0.0;
            } else if !massive[k] && *value != 0.0 {
                return None;
            }
        }
        rows.push(row);
    }
    let solved: Vec<Vec<f64>> = rows.iter().map(|row| mass.solve(row)).collect();
    let scale = dt * dt / inertia;
    let w: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            solved
                .iter()
                .map(|z| scale * row.iter().zip(z).map(|(p, q)| p * q).sum::<f64>())
                .collect()
        })
        .collect();
    let largest = w
        .iter()
        .map(|row| row.iter().map(|e| e.abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    (largest > 0.0).then(|| 1.0 / largest)
}

/// A contact's tolerance on the gap it leaves at the step's end: a
/// fraction (`contact_tolerance`) of its points' relative motion over the
/// step at the step's start velocity `v`, so its work stays that fraction
/// of the work it does at any step length; never below what the step
/// resolves:
/// - the round-off of the gap as computed (`contact_round_off_scale`). The
///   floor was `8 eps (1 + |x|)`: at a step of 4e-15 s (the controller's,
///   at a block's impact) that is 2 percent of the step's motion, and the
///   contact's work kept 5e-4 of the impact energy (measured).
/// - `resolution`, the gap the solve resolves (`gap_resolution`). With its
///   points at rest the motion is zero and the round-off alone asked for
///   a gap finer than Newton's solve resolves (measured, a block at rest on
///   the floor: the gap at 1e-21 to 8e-18 of a cell against an allowance
///   of 1e-25, every try at the passes' cap, the step controller halving
///   to 2e-9 s).
pub(crate) fn contact_allowance(
    grid: &Grid,
    (x, v): (&[DVec2], &[DVec2]),
    planes: &[Plane],
    u: &[f64],
    shares: &[Share],
    (contact_tolerance, dt, resolution): (f64, f64, f64),
) -> f64 {
    let motion: f64 = shares
        .iter()
        .map(|s| {
            let relative: DVec2 = s.spread.iter().map(|&(r, w)| w * v[r]).sum();
            s.weight * relative.length()
        })
        .sum();
    let round_off = contact_round_off_scale(grid, x, planes, u, shares);
    (contact_tolerance * motion * dt)
        .max(8.0 * f64::EPSILON * round_off)
        .max(resolution)
        .max(f64::MIN_POSITIVE)
}

/// For each held condition at a step's
/// end, which of `contact_allowance`'s terms binds (the relative term, the
/// round-off, the resolution) and the relative term's value, m, counted in
/// `hooks::add_allowance_term`. Test-only.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn count_allowance_terms(
    grid: &Grid,
    (x, v, planes, u): (&[DVec2], &[DVec2], &[Plane], &[f64]),
    contacts: &[Contact],
    (contact_tolerance, dt): (f64, f64),
    (penalty, residual): (&[f64], f64),
) {
    for c in contacts {
        let (_, shares) = contact_geometry(grid, x, planes, u, c);
        let motion: f64 = shares
            .iter()
            .map(|s| {
                let relative: DVec2 = s.spread.iter().map(|&(r, w)| w * v[r]).sum();
                s.weight * relative.length()
            })
            .sum();
        let relative = contact_tolerance * motion * dt;
        let round_off = 8.0 * f64::EPSILON * contact_round_off_scale(grid, x, planes, u, &shares);
        let resolution = gap_resolution(c, penalty, residual);
        let binding = if relative >= round_off && relative >= resolution {
            0
        } else if round_off >= resolution {
            1
        } else {
            2
        };
        super::hooks::add_allowance_term(binding, relative);
    }
}

/// The gap the solve resolves at a contact: Newton stops with residual
/// entries up to `residual` (`StepReport::residual_allowed`), and the push
/// `max(0, lambda - k g)` adds the contact's penalty `k` to the Jacobian
/// along its gap's gradient `n`. With `J = A + k n n^T`, `A` the rest
/// (positive definite at rest), `J w = n` is solved by `w = A^-1 n / (1 + k
/// s)`, `s = n^T A^-1 n` (substitute: `A w + k (n^T w) n = n`), so a
/// residual `r` along `n` moves the gap by `n^T J^-1 n r = r s / (1 + k s)
/// < r / k`: the gap is resolved to about `residual / k`, a finer gap asking
/// a finer solve than the step's equations are held to. Measured against
/// the noise it covers: 1e-21 to 8e-18 of a cell at rest, the floor about
/// 8e-17.
/// `k` is the contact's own penalty (`contact_penalty`), the one its push
/// uses.
pub(crate) fn gap_resolution(c: &Contact, penalty: &[f64], residual: f64) -> f64 {
    let k = contact_penalty(c, penalty);
    #[cfg(all(test, feature = "experimental"))]
    if super::hooks::no_resolution_floor() {
        return 0.0;
    }
    if k > 0.0 { residual / k } else { 0.0 }
}

/// One point's part of a contact at the step's displacement: its weight,
/// normal, how its push spreads over particles (`point_geometry`), and the
/// point itself.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Share {
    pub(crate) weight: f64,
    pub(crate) normal: DVec2,
    pub(crate) spread: [(usize, f64); 3],
    pub(crate) point: ContactPoint,
}

/// A contact's gap, the weighted mean of its points' (`point_geometry`),
/// and each point's share.
pub(crate) fn contact_geometry(
    grid: &Grid,
    x: &[DVec2],
    planes: &[Plane],
    u: &[f64],
    c: &Contact,
) -> (f64, Vec<Share>) {
    let mut gap = 0.0;
    let shares = c
        .points
        .iter()
        .map(|&(point, weight)| {
            let (g, normal, spread) = point_geometry(grid, x, planes, u, &point);
            gap += weight * g;
            Share {
                weight,
                normal,
                spread,
                point,
            }
        })
        .collect();
    (gap, shares)
}

/// A contact's tangential gap, its points' weighted (`tangential_gap`).
pub(crate) fn contact_tangential_gap(grid: &Grid, u: &[f64], shares: &[Share]) -> f64 {
    shares
        .iter()
        .map(|s| s.weight * tangential_gap(grid, u, s.normal, &s.spread))
        .sum()
}

/// The tangential motion a contact's friction works against, its points'
/// weighted (`friction_motion`).
pub(crate) fn contact_friction_motion(
    grid: &Grid,
    x: &[DVec2],
    u: &[f64],
    shares: &[Share],
) -> f64 {
    shares
        .iter()
        .map(|s| s.weight * friction_motion(grid, x, u, &s.point, s.normal, &s.spread))
        .sum()
}

/// The magnitudes a contact's gap is summed from, its points' weighted
/// (`gap_round_off_scale`).
pub(crate) fn contact_round_off_scale(
    grid: &Grid,
    x: &[DVec2],
    planes: &[Plane],
    u: &[f64],
    shares: &[Share],
) -> f64 {
    shares
        .iter()
        .map(|s| s.weight * gap_round_off_scale(grid, x, planes, u, &s.point, &s.spread))
        .sum()
}

/// The tangential relative displacement over the step of a contact point
/// whose normal and spread `point_geometry` gave: along the normal turned a
/// quarter counter-clockwise, with the same weights.
pub(crate) fn tangential_gap(
    grid: &Grid,
    u: &[f64],
    normal: DVec2,
    spread: &[(usize, f64); 3],
) -> f64 {
    let tangent = normal.perp();
    spread
        .iter()
        .filter(|(_, weight)| *weight != 0.0)
        .map(|&(r, weight)| {
            weight * tangent.dot(DVec2::new(grid.read(u, r, 0).0, grid.read(u, r, 1).0))
        })
        .sum()
}

/// Friction between two bodies turns them as well as slowing them: the
/// tangential force pushes the point one way and the segment's closest
/// point the other, and while a gap `g` remains between them (along the
/// normal, at the step's midpoint) the pair makes a couple `g f`, which
/// would change the angular momentum. The consistent variation of the
/// closest point's coordinate adds a term that cancels it.
/// Source: Franke, Hesch and Betsch, IJNME 2013, doi 10.1002/nme.4466,
/// eq. 29 (`delta xi = A^-1 [(delta phi1 - delta phi2) . a + g n . delta
/// phi2,xi]`; their remark after eq. 66: without the `g` term angular
/// momentum is kept only at zero gap). On a straight segment of length `L`
/// that term is the force `f g / L` along `n` on its second end and the
/// opposite on its first: a pure couple. Adaptation: written as the
/// couple that cancels the pair's moment, `c = -f (lever x t) / (side x
/// n)` per unit `f`, which is their `g / L` when the normal is the
/// segment's and also holds with the point's nodal normal (Puso et al.
/// 2024, not perpendicular to the segment). Returns the segment's ends and
/// `c`, evaluated at the step's midpoint configuration (their eq. 75).
pub(crate) fn friction_couple(
    grid: &Grid,
    x: &[DVec2],
    u: &[f64],
    c: &ContactPoint,
    normal: DVec2,
    spread: &[(usize, f64); 3],
) -> Option<(usize, usize, f64)> {
    let (first, second) = c.segment?;
    let middle = |r: usize| x[r] + 0.5 * DVec2::new(grid.read(u, r, 0).0, grid.read(u, r, 1).0);
    let (a, b) = (middle(first), middle(second));
    let xi = -spread[2].1;
    let lever = middle(c.particle) - (a + xi * (b - a));
    let turning = (b - a).perp_dot(normal);
    if turning.abs() <= f64::MIN_POSITIVE {
        return None;
    }
    Some((first, second, -lever.perp_dot(normal.perp()) / turning))
}

/// The tangential motion friction works against over the step: the
/// tangential gap, plus the couple's ends' motion along the normal
/// (`friction_couple`), so the friction's work counts the couple too.
pub(crate) fn friction_motion(
    grid: &Grid,
    x: &[DVec2],
    u: &[f64],
    c: &ContactPoint,
    normal: DVec2,
    spread: &[(usize, f64); 3],
) -> f64 {
    let along = tangential_gap(grid, u, normal, spread);
    match friction_couple(grid, x, u, c, normal, spread) {
        Some((first, second, couple)) => {
            let moved = |r: usize| DVec2::new(grid.read(u, r, 0).0, grid.read(u, r, 1).0);
            along + couple * normal.dot(moved(second) - moved(first))
        }
        None => along,
    }
}

/// The line a point's nodal normal is square to, from its previous neighbour to
/// its next, at the positions `at`. On a body of revolution a point on the axis has
/// the axis on one side and, by symmetry, its other neighbour's mirror image there
/// as its neighbour, so the line runs square to the axis and the normal lies along
/// it (derived here).
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn nodal_line(
    grid: &Grid,
    at: impl Fn(usize) -> DVec2,
    point: usize,
    (previous, next): (usize, usize),
) -> DVec2 {
    if let Some(hoop) = grid.hoop.as_ref().filter(|h| h.radii[point] <= 0.0) {
        if hoop.radii[previous] > 0.0 {
            return DVec2::new(-2.0 * (at(previous).x - hoop.axis_x), 0.0);
        }
        if hoop.radii[next] > 0.0 {
            return DVec2::new(2.0 * (at(next).x - hoop.axis_x), 0.0);
        }
    }
    at(next) - at(previous)
}

/// The line a point's nodal normal is square to, from its previous neighbour to
/// its next, at the positions `at`.
#[cfg(not(all(test, feature = "experimental")))]
pub(crate) fn nodal_line(
    _: &Grid,
    at: impl Fn(usize) -> DVec2,
    _: usize,
    (previous, next): (usize, usize),
) -> DVec2 {
    at(next) - at(previous)
}

/// A contact point's gap, its normal and how its push spreads over the
/// particles (particle, weight), for the displacement `u` of the step:
/// - against a plane, or a partner across it, the plane's own gap and
///   normal;
/// - against a segment, the contact condition in velocity of Laursen and
///   Chawla, IJNME 40 (1997), eqs. 46-48: the normal and the closest point
///   taken in the step's midpoint configuration (their sec. 3, which keeps
///   both momenta), the gap the normal relative displacement of the point
///   and the closest point at its parameter `xi`, `n . (u_p - (1 - xi)
///   u_1 - xi u_2)`, the push `+n` on the point and `-(1 - xi) n`, `-xi
///   n` on the ends;
/// - with the point's neighbours (`nodal`), the normal is the point's own
///   nodal normal, from its neighbours (reversed: it points out of the
///   other body), and `xi` is where the line through the point along it
///   meets the segment's line, after Puso, Porter and Slavik, LLNL 2024,
///   sec. 3.1 (eq. 11: the gap vector along the nodal normal, so the push
///   and its lever arm are collinear and the angular momentum is kept,
///   eq. 15). `xi` is not clamped: past an end it extrapolates the
///   segment's line, keeping that collinearity (this engine's choice,
///   declared; a point that slides past an end is held against the line).
pub(crate) fn point_geometry(
    grid: &Grid,
    x: &[DVec2],
    planes: &[Plane],
    u: &[f64],
    c: &ContactPoint,
) -> (f64, DVec2, [(usize, f64); 3]) {
    let moved = |r: usize| DVec2::new(grid.read(u, r, 0).0, grid.read(u, r, 1).0);
    match c.segment {
        Some((first, second)) => {
            let middle = |r: usize| x[r] + 0.5 * moved(r);
            let (a, b) = (middle(first), middle(second));
            let side = b - a;
            let length = side.length().max(f64::MIN_POSITIVE);
            let (xi, normal) = match c.nodal {
                Some((previous, next)) => {
                    let along = nodal_line(grid, middle, c.particle, (previous, next));
                    let normal =
                        -DVec2::new(along.y, -along.x) / along.length().max(f64::MIN_POSITIVE);
                    let denominator = normal.perp_dot(side);
                    let xi = if denominator.abs() > f64::MIN_POSITIVE {
                        normal.perp_dot(middle(c.particle) - a) / denominator
                    } else {
                        0.5
                    };
                    (xi, normal)
                }
                None => (
                    ((middle(c.particle) - a).dot(side) / (length * length)).clamp(0.0, 1.0),
                    DVec2::new(side.y, -side.x) / length,
                ),
            };
            let relative = moved(c.particle) - (1.0 - xi) * moved(first) - xi * moved(second);
            (
                normal.dot(relative),
                normal,
                [(c.particle, 1.0), (first, -(1.0 - xi)), (second, -xi)],
            )
        }
        None => {
            let plane = planes[c.plane];
            // The start separation first, then the step's displacement: the
            // gap is near zero and the penalty large, so the round-off of
            // `x + u` against the plane (`eps |x|`, measured: a residual
            // floor of 1e-9 of its scale under a 10 c_p^2 penalty) must not
            // enter it. Against the plane at the start position the first
            // part is exactly zero.
            let (mut start, mut displacement) = (
                x[c.particle][plane.axis] - plane.at,
                moved(c.particle)[plane.axis],
            );
            if let Some(q) = c.partner {
                start -= x[q][plane.axis];
                displacement -= moved(q)[plane.axis];
            }
            let separation = start + displacement;
            let (normal, spread) = plane_share(planes, c);
            (plane.inward * separation, normal, spread)
        }
    }
}

/// A plane contact point's normal and spread (`point_geometry`'s plane
/// branch): the plane's inward normal, `+1` on the point, `-1` on its
/// partner across the plane. Neither depends on the displacement, so the
/// point's gap is linear in it.
pub(crate) fn plane_share(planes: &[Plane], c: &ContactPoint) -> (DVec2, [(usize, f64); 3]) {
    let plane = planes[c.plane];
    let mut spread = [(c.particle, 1.0), (c.particle, 0.0), (c.particle, 0.0)];
    if let Some(q) = c.partner {
        spread[1] = (q, -1.0);
    }
    let mut normal = DVec2::ZERO;
    normal[plane.axis] = plane.inward;
    (normal, spread)
}

/// The magnitudes the gap `point_geometry` computes is summed from, whose
/// `eps` multiple bounds its round-off: each subtraction of the start
/// separation (a floating subtraction errs by `eps` of its result) and each
/// interpolated displacement's terms `|N_k u_k|`, both components (the
/// tangential slip a sticking contact is held to as well). Against the plane at the
/// start position the start part is exactly zero and the bound is the
/// step's displacement alone, so it shrinks with the step instead of
/// staying at `eps |x|`.
pub(crate) fn gap_round_off_scale(
    grid: &Grid,
    x: &[DVec2],
    planes: &[Plane],
    u: &[f64],
    c: &ContactPoint,
    spread: &[(usize, f64); 3],
) -> f64 {
    let terms = |r: usize, a: usize| -> f64 {
        let mut sum = 0.0;
        grid.visit(r, a, |k, w, _| sum += (w * u[k]).abs());
        sum
    };
    match c.segment {
        Some(_) => spread
            .iter()
            .map(|&(r, w)| w.abs() * (terms(r, 0) + terms(r, 1)))
            .sum(),
        None => {
            let plane = planes[c.plane];
            let first = x[c.particle][plane.axis] - plane.at;
            let start = first - c.partner.map_or(0.0, |q| x[q][plane.axis]);
            first.abs()
                + start.abs()
                + spread
                    .iter()
                    .map(|&(r, w)| w.abs() * (terms(r, 0) + terms(r, 1)))
                    .sum::<f64>()
        }
    }
}

/// What `step_total_lagrangian_contact` holds: rigid planes, pairs of
/// facing points of two bodies (`Pair`), and the bodies' outlines, each a
/// closed list of boundary particles in counter-clockwise order, every
/// point of one held against the other outlines' segments (both ways).
#[derive(Clone, Copy)]
pub struct ContactSet<'a> {
    pub planes: &'a [Plane],
    pub pairs: &'a [Pair],
    pub outlines: &'a [Vec<usize>],
    /// Per outline, the indices (into it) of its corners, where the exact
    /// shape's normal jumps (`weighted_gap`); none given, none.
    pub corners: &'a [Vec<usize>],
    /// Coulomb's friction coefficient between every pair of surfaces in
    /// contact (`contact_force`); 0, frictionless.
    pub friction: f64,
}
