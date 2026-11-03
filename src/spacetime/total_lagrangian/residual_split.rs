//! FAST_MODE_PLAN.md sec. 63.6: the internal force's micro-count
//! (`EMERGE_PROBE_RESIDUAL_SPLIT`, test-only). One evaluation in `EVERY` is
//! repeated serially, part by part, on the same state: the reads of the
//! deformation gradient, the law's step stress (its base law and its
//! relaxation branches apart), and the scatter of the force rows. The
//! evaluation the solve uses is untouched; the repeat's results are dropped.

use std::hint::black_box;
use std::time::Instant;

use glam::{DMat2, DVec2};

use super::basis::Grid;
use super::elastic::Elastic;

/// One evaluation sampled in this many.
const EVERY: u64 = 25;

/// Whether the knob is set (read once; printed once).
pub(crate) fn on() -> bool {
    static SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SET.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_RESIDUAL_SPLIT").is_some();
        if on {
            println!("knob active: the internal force's micro-count, one evaluation in {EVERY} (sec. 63.6)");
        }
        on
    })
}

/// Sums since the last `take`: samples, points, and seconds in the reads,
/// the base law, the relaxation branches, the whole step stress, the
/// scatter, and the whole serial evaluation.
static SPLIT: std::sync::Mutex<[f64; 8]> = std::sync::Mutex::new([0.0; 8]);

pub(crate) fn take() -> [f64; 8] {
    std::mem::take(&mut *SPLIT.lock().expect("residual split lock"))
}

/// The plain total Lagrangian midpoint's evaluation only (no probe variant
/// of the stress): `carrier` is `F_n+1/2`, the first Piola stress `v0 F S`.
pub(crate) struct Sample<'a> {
    pub grid: &'a Grid,
    pub f_now: &'a [DMat2],
    pub v0: &'a [f64],
    pub solids: &'a [Elastic],
    pub dt: f64,
}

pub(crate) fn sample(evaluation: u64, at: &Sample<'_>, deformation_at: impl Fn(usize) -> DMat2) {
    if evaluation % EVERY != 1 {
        return;
    }
    let Sample {
        grid,
        f_now,
        v0,
        solids,
        dt,
    } = *at;
    let n = f_now.len();
    let whole = Instant::now();
    let clock = Instant::now();
    let f1: Vec<DMat2> = (0..n).map(&deformation_at).collect();
    let reads = clock.elapsed().as_secs_f64();
    let c: Vec<(DMat2, DMat2)> = (0..n)
        .map(|p| (f_now[p].transpose() * f_now[p], f1[p].transpose() * f1[p]))
        .collect();
    let clock = Instant::now();
    let base: Vec<DMat2> = (0..n)
        .map(|p| solids[p].algorithmic_stress_over(c[p].0, c[p].1, dt))
        .collect();
    let base_s = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let branches: Vec<DMat2> = (0..n)
        .filter(|&p| solids[p].is_relaxing())
        .map(|p| {
            solids[p]
                .relaxation
                .step_stress(c[p].0, c[p].1, dt, solids[p].lame())
        })
        .collect();
    let branches_s = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let s: Vec<DMat2> = (0..n)
        .map(|p| solids[p].step_stress(c[p].0, c[p].1, dt))
        .collect();
    let stress_s = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let mut acc = vec![0.0; grid.len];
    for p in 0..n {
        let first = v0[p] * (0.5 * (f_now[p] + f1[p])) * s[p];
        for a in 0..2 {
            let row: DVec2 = first.row(a);
            grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
        }
    }
    let scatter = clock.elapsed().as_secs_f64();
    let whole = whole.elapsed().as_secs_f64();
    black_box((base, branches, acc));
    let mut sums = SPLIT.lock().expect("residual split lock");
    for (sum, part) in sums.iter_mut().zip([
        1.0, n as f64, reads, base_s, branches_s, stress_s, scatter, whole,
    ]) {
        *sum += part;
    }
}
