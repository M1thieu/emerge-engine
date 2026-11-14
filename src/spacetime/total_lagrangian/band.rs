//! A banded matrix and its Cholesky factor: the preconditioner the step's
//! linear solves lean on.

/// A symmetric positive definite matrix in band form, factored `L L^T`
/// in place (Cholesky), its unknowns renumbered by `order` to keep the
/// band narrow. A non-positive pivot is lifted to a small positive one, so
/// the factor stays usable as a preconditioner.
#[derive(Clone)]
pub(crate) struct Band {
    pub(crate) n: usize,
    pub(crate) width: usize,
    /// `L(i, j)` for `i - width <= j <= i`, row by row.
    pub(crate) data: Vec<f64>,
    /// The band's position of each unknown.
    pub(crate) order: Vec<usize>,
    /// `L(k, i)` for `i < k <= i + width`, row `i` by row: the factor's
    /// columns copied contiguous by `factor`, so the backward sweep reads
    /// memory in order. The same values, products and sums as reading `L`
    /// down its columns.
    pub(crate) upper: Vec<f64>,
}

impl Band {
    /// Adds `value` at the unknowns `(k, m)`, kept on the lower triangle.
    pub(crate) fn add(&mut self, k: usize, m: usize, value: f64) {
        let (i, j) = (self.order[k], self.order[m]);
        if i >= j {
            let slot = self.at(i, j);
            self.data[slot] += value;
        }
    }

    /// Whether `add` keeps the entry of the unknowns `(k, m)`: the lower
    /// triangle in the band's order, the upper one being its mirror.
    pub(crate) fn keeps(&self, k: usize, m: usize) -> bool {
        self.order[k] >= self.order[m]
    }

    pub(crate) fn at(&self, i: usize, j: usize) -> usize {
        i * (self.width + 1) + (j + self.width - i)
    }

    pub(crate) fn factor(&mut self) {
        let w = self.width;
        for i in 0..self.n {
            let lo = i.saturating_sub(w);
            for j in lo..=i {
                let mut sum = self.data[self.at(i, j)];
                // `L(i, k)` and `L(j, k)` for `k` from `first` to `j - 1`:
                // contiguous in both rows, subtracted in increasing `k`.
                let first = lo.max(j.saturating_sub(w));
                let len = j - first;
                let (a, b) = (self.at(i, first), self.at(j, first));
                for (x, y) in self.data[a..a + len].iter().zip(&self.data[b..b + len]) {
                    sum -= x * y;
                }
                let slot = self.at(i, j);
                if i == j {
                    let floor = 1e-12 * self.data[slot].abs().max(f64::MIN_POSITIVE);
                    self.data[slot] = sum.max(floor).sqrt();
                } else {
                    self.data[slot] = sum / self.data[self.at(j, j)];
                }
            }
        }
        let mut upper = vec![0.0; self.n * w];
        for i in 0..self.n {
            let hi = (i + w + 1).min(self.n);
            for k in i + 1..hi {
                upper[i * w + (k - i - 1)] = self.data[self.at(k, i)];
            }
        }
        self.upper = upper;
    }

    pub(crate) fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let (n, w) = (self.n, self.width);
        let mut y = vec![0.0; n];
        for (k, &value) in rhs.iter().enumerate() {
            y[self.order[k]] = value;
        }
        for i in 0..n {
            let lo = i.saturating_sub(w);
            let start = self.at(i, lo);
            let row = &self.data[start..start + (i - lo)];
            let sum: f64 = row.iter().zip(&y[lo..i]).map(|(l, e)| l * e).sum();
            y[i] = (y[i] - sum) / self.data[self.at(i, i)];
        }
        for i in (0..n).rev() {
            let hi = (i + w + 1).min(n);
            let column = &self.upper[i * w..i * w + (hi - i - 1)];
            let sum: f64 = column.iter().zip(&y[i + 1..hi]).map(|(l, e)| l * e).sum();
            y[i] = (y[i] - sum) / self.data[self.at(i, i)];
        }
        (0..n).map(|k| y[self.order[k]]).collect()
    }
}
