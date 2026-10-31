//! The internal force's derivative at one Newton iterate as a sparse matrix:
//! assembled once from the particles' linearisations, then each Jacobian
//! product is a matrix product instead of a pass over the particles.
//!
//! The derivative along `d` is linear in `d` at a fixed iterate: per particle
//! `grad d` (four numbers, `grad[b][j] = sum_m g_m[j] d_m` over lattice `b`'s
//! stencil) maps linearly to the first Piola stress's change, whose row `a`
//! is scattered as `acc[k] += row_a . g_k` over lattice `a`'s stencil. With
//! `P(E_bj)` that change for the unit gradient, the entry is
//! `K[k, m] = sum_p sum_ij g_k[i] P(E_bj)[a][i] g_m[j]`, `k` on lattice `a`,
//! `m` on lattice `b`: the same sums as the pass, in another order (its
//! round-off only).

use super::*;
use std::sync::{Arc, Mutex};

/// The grid's kept pattern (`Grid::tangent_pattern`).
pub(crate) type KeptPattern = Arc<Mutex<Option<Arc<TangentPattern>>>>;

/// The compressed rows of the internal force's derivative, and per particle
/// the entry each of its stencil pairs adds to, in the order lattice `a`,
/// lattice `b`, the unknowns `visit(p, a)` reaches, then those `visit(p, b)`
/// reaches.
#[derive(Debug, Default)]
pub(crate) struct TangentPattern {
    start: Vec<usize>,
    column: Vec<usize>,
    slots: Vec<Vec<usize>>,
}

impl Grid {
    /// The sparsity of `n` particles' internal-force derivative, built at the
    /// first call and kept with the grid.
    pub(crate) fn tangent_pattern(&self, n: usize) -> Arc<TangentPattern> {
        let mut kept = self
            .tangent_pattern
            .lock()
            .expect("the tangent pattern's lock");
        if let Some(pattern) = kept.as_ref().filter(|t| t.slots.len() == n) {
            return pattern.clone();
        }
        let pairs: Vec<Vec<(usize, usize)>> = (0..n)
            .map(|p| {
                let reached: [Vec<usize>; 2] = [0, 1].map(|a| {
                    let mut list = Vec::new();
                    self.visit(p, a, |k, _, _| list.push(k));
                    list
                });
                let mut own = Vec::new();
                for rows in &reached {
                    for cols in &reached {
                        for &k in rows {
                            own.extend(cols.iter().map(|&m| (k, m)));
                        }
                    }
                }
                own
            })
            .collect();
        let mut entries: Vec<(usize, usize)> = pairs.iter().flatten().copied().collect();
        entries.sort_unstable();
        entries.dedup();
        let mut start = Vec::with_capacity(self.len + 1);
        let mut column = Vec::with_capacity(entries.len());
        let mut next = entries.iter().peekable();
        for k in 0..self.len {
            start.push(column.len());
            while let Some(&(_, m)) = next.next_if(|e| e.0 == k) {
                column.push(m);
            }
        }
        start.push(column.len());
        let slot = |k: usize, m: usize| -> usize {
            let row = &column[start[k]..start[k + 1]];
            start[k] + row.binary_search(&m).expect("a pattern entry")
        };
        let slots = pairs
            .iter()
            .map(|own| own.iter().map(|&(k, m)| slot(k, m)).collect())
            .collect();
        let pattern = Arc::new(TangentPattern {
            start,
            column,
            slots,
        });
        *kept = Some(pattern.clone());
        pattern
    }
}

/// The internal force's derivative at one iterate.
pub(crate) struct Tangent {
    pattern: Arc<TangentPattern>,
    value: Vec<f64>,
    /// The same matrix with every operand of its terms replaced by its
    /// absolute value (`check_tangent`).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) absolute: Option<Box<Tangent>>,
}

/// The unit gradients `E_bj` (`grad[b][j] = 1`) in the order `(b, j)` =
/// (0, 0), (0, 1), (1, 0), (1, 1); `grad`'s column `j` holds `g_0[j], g_1[j]`.
pub(crate) const UNIT_GRADIENTS: [DMat2; 4] = [
    DMat2::from_cols(DVec2::X, DVec2::ZERO),
    DMat2::from_cols(DVec2::ZERO, DVec2::X),
    DMat2::from_cols(DVec2::Y, DVec2::ZERO),
    DMat2::from_cols(DVec2::ZERO, DVec2::Y),
];

impl Tangent {
    /// Assembles the derivative from each particle's stress changes along
    /// the unit gradients (`UNIT_GRADIENTS`), `None` for a particle that adds
    /// nothing. Each particle's entries are computed in parallel and added in
    /// particle order, so the matrix is the same bits whatever the threads.
    pub(crate) fn assemble(
        grid: &Grid,
        n: usize,
        changes: impl Fn(usize) -> Option<[DMat2; 4]> + Sync,
    ) -> Self {
        Self::assemble_signed(grid, n, changes, false)
    }

    /// `assemble` from the stress changes' absolute bound (every operand of
    /// the change's formula by its absolute value) and the weight gradients'
    /// absolute values: the sum of the terms' absolute values per entry.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn assemble_absolute(
        grid: &Grid,
        n: usize,
        changes: impl Fn(usize) -> Option<[DMat2; 4]> + Sync,
    ) -> Self {
        Self::assemble_signed(grid, n, changes, true)
    }

    fn assemble_signed(
        grid: &Grid,
        n: usize,
        changes: impl Fn(usize) -> Option<[DMat2; 4]> + Sync,
        absolute: bool,
    ) -> Self {
        let pattern = grid.tangent_pattern(n);
        let parts: Vec<Option<Vec<f64>>> = (0..n)
            .into_par_iter()
            .map(|p| changes(p).map(|change| particle_entries(grid, p, &change, absolute)))
            .collect();
        let mut value = vec![0.0; pattern.column.len()];
        for (part, slots) in parts.iter().zip(&pattern.slots) {
            if let Some(part) = part {
                for (&s, e) in slots.iter().zip(part) {
                    value[s] += e;
                }
            }
        }
        Self {
            pattern,
            value,
            #[cfg(all(test, feature = "experimental"))]
            absolute: None,
        }
    }

    /// `K d`, each row summed in column order.
    pub(crate) fn times(&self, d: &[f64]) -> Vec<f64> {
        let t = &self.pattern;
        (0..t.start.len() - 1)
            .into_par_iter()
            .map(|k| {
                (t.start[k]..t.start[k + 1])
                    .map(|e| self.value[e] * d[t.column[e]])
                    .sum()
            })
            .collect()
    }
}

/// One particle's entries in its slots' order: per row unknown `k` of
/// lattice `a`, `h[b][j] = P(E_bj)[a] . g_k`, then per column unknown `m` of
/// lattice `b`, `sum_j h[b][j] g_m[j]`.
fn particle_entries(grid: &Grid, p: usize, change: &[DMat2; 4], absolute: bool) -> Vec<f64> {
    let reached: [Vec<DVec2>; 2] = [0, 1].map(|a| {
        let mut list = Vec::new();
        grid.visit(p, a, |_, _, g| {
            list.push(if absolute { g.abs() } else { g })
        });
        list
    });
    let mut out = Vec::new();
    for (a, rows) in reached.iter().enumerate() {
        for (b, cols) in reached.iter().enumerate() {
            for gk in rows {
                let h = DVec2::new(
                    change[2 * b].row(a).dot(*gk),
                    change[2 * b + 1].row(a).dot(*gk),
                );
                out.extend(cols.iter().map(|gm| h.dot(*gm)));
            }
        }
    }
    out
}

/// Whether `EMERGE_PROBE_TANGENT_CHECK` is set (read once): every product
/// of the assembled tangent compared with the pass over the particles. A
/// probe knob, in test builds only.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn tangent_check() -> bool {
    static SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SET.get_or_init(|| {
        let set = std::env::var_os("EMERGE_PROBE_TANGENT_CHECK").is_some();
        if set {
            println!("knob active: tangent check (assembled tangent against the particle pass)");
        }
        set
    })
}

/// The operations on one term's path through the stress change's formula
/// (`solve`'s `first_change`, the Eulerian form, the longer): `dF = grad F0`
/// 2, `dC` 3, `dS` 3, the first Piola stress 4, `V0 (.) F0^T` 3.
#[cfg(all(test, feature = "experimental"))]
const CHANGE_OPERATIONS: usize = 15;

/// `gamma_m = m u / (1 - m u)`, `u = 2^-53`.
#[cfg(all(test, feature = "experimental"))]
fn gamma(m: usize) -> f64 {
    let mu = m as f64 * f64::EPSILON / 2.0;
    mu / (1.0 - mu)
}

/// Compares the assembled tangent's product `product` (`K d`) with the
/// particle pass's `reference` along the same `d`, and prints the worst
/// entry against its round-off bound. The bound, derived here from the
/// standard model `fl(x op y) = (x op y)(1 + delta)`, `|delta| <= u` (eq. 2.2
/// of Hallman, arXiv 2104.06531, citing Higham's Accuracy and Stability of
/// Numerical Algorithms): a value computed as a sum of products, in any
/// order, is the exact sum of its terms each times a product of at most `D`
/// factors `(1 + delta)`, `D` the operations on that term's path, and
/// `|prod (1 + delta) - 1| <= (1 + u)^D - 1 <= gamma_D`; so it is off by at
/// most `gamma_D` times the sum of its terms' absolute values, here `(|K|
/// |d|)_i` (`Tangent::absolute`), and two computations of the same sum
/// differ by at most `gamma_D1 + gamma_D2` times it. The paths' operations:
/// the pass reads `grad d` (`most_reached` terms, each a product, and the
/// basis' extension and web-spline weights 4 more), the change's formula,
/// the row's dot product 2, the scatter's sum (at most `most_terms` terms)
/// and the 64 chunks' sum; the assembly's terms go through the change's
/// formula, `h` 2, `h . g_m` 2, the extension and weights 8 (both
/// gradients), the entry's sum over particles and the product's row sum
/// (each at most `most_terms` terms) and the product 1.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn check_tangent(tangent: &Tangent, product: &[f64], reference: &[f64], d: &[f64]) {
    let Some(absolute) = &tangent.absolute else {
        return;
    };
    let t = &tangent.pattern;
    // The most particle-pair terms one row sums, and a bound on the
    // unknowns one particle reaches on a lattice (both lattices' together:
    // a particle's pairs are their square).
    let mut terms = vec![0usize; t.start.len() - 1];
    for &slot in t.slots.iter().flatten() {
        terms[t.start.partition_point(|&s| s <= slot) - 1] += 1;
    }
    let most_terms = terms.into_iter().max().unwrap_or(0);
    let most_reached = t
        .slots
        .iter()
        .map(|own| own.len().isqrt())
        .max()
        .unwrap_or(0);
    let pass = most_reached + 4 + CHANGE_OPERATIONS + 2 + most_terms + 64;
    let assembly = CHANGE_OPERATIONS + 2 + 2 + 8 + 2 * most_terms + 1;
    let factor = gamma(pass) + gamma(assembly);
    let magnitude: Vec<f64> = d.iter().map(|e| e.abs()).collect();
    let bound = absolute.times(&magnitude);
    // The plan's literal reading, gamma of the terms in an entry, beside it.
    let literal = gamma(most_terms);
    let (mut worst, mut at, mut over, mut over_literal) = (0.0f64, 0, 0usize, 0usize);
    for (i, ((k, r), b)) in product.iter().zip(reference).zip(&bound).enumerate() {
        let gap = (k - r).abs();
        let allowed = factor * b;
        over += usize::from(gap > allowed);
        over_literal += usize::from(gap > literal * b);
        let ratio = if allowed > 0.0 {
            gap / allowed
        } else if gap > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        if ratio > worst {
            (worst, at) = (ratio, i);
        }
    }
    println!(
        "tangent check: worst gap / bound {worst:.3e} at entry {at} (gap {:.3e}, bound {:.3e}), \
         entries over the bound {over} of {}, gamma factor {factor:.3e}; literal gamma_N \
         {literal:.3e} (N {}), entries over it {over_literal}",
        (product[at] - reference[at]).abs(),
        factor * bound[at],
        product.len(),
        most_terms
    );
}
