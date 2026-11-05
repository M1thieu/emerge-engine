//! The sharper clearance lemma, measured only, with
//! `EMERGE_PROBE_CLEARANCE_LEMMA=1`: beside the law flight's two clearance
//! proofs, its loop is computed and recorded at every evaluation, and its
//! inequalities are checked on every rebuilt end state; nothing it computes
//! enters a decision of the flight.
//!
//! The lemma, for `|t| <= tau < 2`, `k0 tau < 1`, `J > MIN_J` (`t = tr G`, `w =
//! G_21 - G_12`, `a = |devsym G|^2`): (P) `W(I + G) >= (1 - k0 tau) W_lin(G) -
//! tau kw(tau) w^2`, `k0 = max(1/2, lambda / (2 mu))`, `kw(tau) = mu / (2 (2 -
//! tau)) + lambda / 4`, from the exact 2 x 2 split `|F - R|^2 = a + (s -
//! sqrt2)^2` (Lewicka and Mueller, arXiv 1501.01917, (5.9), (5.11); the polar
//! identity, Musesti, arXiv 1605.06964) and `det G = (t^2 + w^2)/4 - a/2`;
//! (Global) `sum V0 W >= c(tau) E_w`, `c(tau) = (1 - e_K)(1 - k0 tau) - 2 tau
//! kw(tau) beta_w`; (Flight) `|t_p| <= g^t_p (2 H / c)^(1/2)`, `J_p >= 1 - tau -
//! alpha_p^2 H / c`, closed by the amplitude bound's bootstrap on `tau`.

use super::*;

// KNOWN-LIMIT[DISCRETIZATION/NO-SOURCE]: the lemma is this engine's derivation from the cited exact 2 x 2 identities, not a published theorem; its bootstrap holds for the flight equations' exact solution (the discrete end state checked inside the walls), and it is measured only | source: this engine's derivation, measured on this path | replace when: the lemma is reviewed and something uses it to fly frames | state: measured only | gap: none
/// Whether the lemma is measured (`EMERGE_PROBE_CLEARANCE_LEMMA`),
/// announced once.
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_CLEARANCE_LEMMA").is_some();
        if on {
            println!("knob active: the clearance lemma measured, the flight unchanged");
        }
        on
    })
}

/// The lemma's constants from the body, per flight: per particle `g^t_p =
/// |t^T Ghat_p|` and `alpha_p = |L_a Ghat_p|_2`; `beta_w` with its two
/// bounds; the moduli's extremes (the lemma's constants are monotone in
/// them, so the largest `lambda` and the largest and smallest `mu` give
/// constants valid for every particle).
#[derive(Clone, Debug)]
pub(super) struct Lemma {
    trace: Vec<f64>,
    alpha: Vec<f64>,
    beta_w: f64,
    frobenius: f64,
    rows: f64,
    k0: f64,
    mu_max: f64,
    lambda_max: f64,
}

fn parts(g: DMat2) -> (f64, f64, f64, f64) {
    // `G = [[G11, G12], [G21, G22]]`, glam's columns `(G11, G21)`, `(G12, G22)`.
    let (g11, g21, g12, g22) = (g.x_axis.x, g.x_axis.y, g.y_axis.x, g.y_axis.y);
    let t = g11 + g22;
    let w = g21 - g12;
    let (d1, d2) = (
        (g11 - g22) / std::f64::consts::SQRT_2,
        (g12 + g21) / std::f64::consts::SQRT_2,
    );
    (t, w, d1, d2)
}

impl Lemma {
    pub(super) fn new(
        state: &TotalLagrangianState,
        modes: &Modes,
        (elastic, omega): (&[usize], &[f64]),
    ) -> Self {
        let n = state.particles;
        let unit = 0.5 * f64::EPSILON;
        let m = elastic.len();
        let grad = |i: usize, p: usize| modes.gradients[elastic[i]][p];
        let (mut trace, mut alpha) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for p in 0..n {
            let (mut tt, mut s11, mut s12, mut s22) = (0.0, 0.0, 0.0, 0.0);
            for (i, w) in omega.iter().enumerate().take(m) {
                let (t, _, d1, d2) = parts(grad(i, p));
                let w2 = w * w;
                tt += t * t / w2;
                s11 += d1 * d1 / w2;
                s12 += d1 * d2 / w2;
                s22 += d2 * d2 / w2;
            }
            trace.push(tt.sqrt() * (1.0 + (m as f64 + 4.0) * unit));
            let top = 0.5 * (s11 + s22) + (0.25 * (s11 - s22).powi(2) + s12 * s12).sqrt();
            alpha.push(top.max(0.0).sqrt() * (1.0 + (m as f64 + 8.0) * unit));
        }
        // `Bt_w = sum_p V0 (w^T Ghat_p)^T (w^T Ghat_p)`, each entry within `(n +
        // 4) u` of its terms' magnitudes.
        let (mut frobenius, mut rows) = (0.0f64, 0.0f64);
        for a in 0..m {
            let mut row = 0.0;
            for b in 0..m {
                let (mut s, mut size) = (0.0, 0.0);
                for p in 0..n {
                    let e = state.v0[p] * parts(grad(a, p)).1 * parts(grad(b, p)).1;
                    s += e;
                    size += e.abs();
                }
                let scale = omega[a] * omega[b];
                let entry = s.abs() / scale + (n as f64 + 4.0) * unit * size / scale;
                frobenius += entry * entry;
                row += entry;
            }
            rows = rows.max(row);
        }
        let frobenius = frobenius.sqrt() * (1.0 + 4.0 * unit);
        let rows = rows * (1.0 + (m as f64 + 2.0) * unit);
        let lame: Vec<(f64, f64)> = state.solids[..n].iter().map(|s| s.lame()).collect();
        let lambda_max = lame.iter().map(|l| l.0).fold(0.0, f64::max);
        let mu_max = lame.iter().map(|l| l.1).fold(0.0, f64::max);
        let mu_min = lame.iter().map(|l| l.1).fold(f64::INFINITY, f64::min);
        Self {
            trace,
            alpha,
            beta_w: frobenius.min(rows),
            frobenius,
            rows,
            k0: (0.5f64).max(lambda_max / (2.0 * mu_min)),
            mu_max,
            lambda_max,
        }
    }

    fn kw(&self, tau: f64) -> f64 {
        self.mu_max / (2.0 * (2.0 - tau)) + self.lambda_max / 4.0
    }

    /// `c(tau)`, `None` outside the lemma's range or when not positive.
    fn c(&self, e_k: f64, tau: f64) -> Option<f64> {
        if tau >= 2.0 || self.k0 * tau >= 1.0 {
            return None;
        }
        let c = ((1.0 - e_k) * (1.0 - self.k0 * tau) - 2.0 * tau * self.kw(tau) * self.beta_w)
            * (1.0 - gamma(8));
        (c > 0.0).then_some(c)
    }
}

/// How one evaluation's loop ended.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Closure {
    Closed {
        clear: bool,
    },
    StartOutside,
    /// The monotone iteration left the lemma's range: no fixed point.
    IterationLeaving,
    /// A fixed point `tau*` exists, but the candidate `2 tau* - tau0` is outside the range or fails the strict check.
    Candidate,
    Volume,
}

#[derive(Clone, Copy, Debug)]
struct Evaluation {
    g_old: f64,
    g_t: f64,
    rho: f64,
    tau0: f64,
    after_contact: bool,
    closure: Closure,
    /// The least fixed point `tau*` when the iteration found one.
    fixed: Option<f64>,
    /// Clear by the lemma while neither of today's two proofs is.
    gain: bool,
}

/// The state checks' counts: checked, below zero, beyond the round-off
/// margin, and the least slack over the margin.
#[derive(Clone, Copy, Debug, Default)]
struct Checks {
    checked: u64,
    negative: u64,
    beyond: u64,
    least: Option<f64>,
}

impl Checks {
    fn add(&mut self, slack: f64, margin: f64) {
        self.checked += 1;
        self.negative += u64::from(slack < 0.0);
        self.beyond += u64::from(slack < -margin);
        let r = slack / margin.max(f64::MIN_POSITIVE);
        self.least = Some(self.least.map_or(r, |l| l.min(r)));
    }
}

thread_local! {
    static EVALUATIONS: std::cell::RefCell<Vec<Evaluation>> = const { std::cell::RefCell::new(Vec::new()) };
    static CONTACTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static POINTWISE: std::cell::Cell<Checks> = const { std::cell::Cell::new(Checks { checked: 0, negative: 0, beyond: 0, least: None }) };
    static GLOBAL: std::cell::Cell<Checks> = const { std::cell::Cell::new(Checks { checked: 0, negative: 0, beyond: 0, least: None }) };
    static CONSTANTS: std::cell::Cell<(f64, f64, f64, f64, f64)> = const { std::cell::Cell::new((0.0, 0.0, 0.0, 0.0, 0.0)) };
}

/// A contact has been held (the evaluations after it are "after the first
/// contact").
pub(in super::super) fn contact_held() {
    CONTACTED.with(|c| c.set(true));
}

impl Exponential {
    /// One evaluation of the lemma's loop over `[t0, t1]`,
    /// recorded; `today` is the evaluation's own answer.
    pub(super) fn measure_lemma(
        &self,
        (law, lemma): (&Law, &Lemma),
        (gravity, planes): (DVec2, &[Plane]),
        span: (f64, f64),
        today: bool,
    ) {
        let inv = &self.inv;
        let n = law.volume.len();
        let h = self.h_int.max(0.0) * (1.0 + gamma(4 * inv.n()));
        let q: Vec<f64> = self
            .zeta_at(span.0)
            .iter()
            .zip(&inv.omega)
            .map(|(z, w)| z.x / w)
            .collect();
        let min_j = f64::from(crate::matter::kinematics::volume_ratio::MIN_J);
        let (mut tau0, mut volume_ok) = (0.0f64, true);
        for p in 0..n {
            let f = law.deformation(&q, p);
            tau0 = tau0.max(parts(f - DMat2::IDENTITY).0.abs());
            volume_ok &= f.determinant() > min_j;
        }
        let tau0 = tau0 * (1.0 + gamma(q.len() + 4));
        let reach = law.reach.iter().copied().fold(0.0, f64::max);
        let trace = lemma.trace.iter().copied().fold(0.0, f64::max);
        let alpha = lemma.alpha.iter().copied().fold(0.0, f64::max);
        let root = (2.0 * h).sqrt();
        CONSTANTS.with(|c| {
            let (b, fr, ro, ek, old) = c.get();
            c.set((
                b.max(lemma.beta_w),
                fr.max(lemma.frobenius),
                ro.max(lemma.rows),
                ek.max(law.e_k),
                old.max(law.beta),
            ));
        });
        let implied = |tau: f64| -> Option<(f64, f64)> {
            let c = lemma.c(law.e_k, tau)?;
            Some((c, trace * (2.0 * h / c).sqrt() * (1.0 + gamma(4))))
        };
        let mut fixed = None;
        let closure = (|| {
            if !volume_ok || implied(tau0).is_none() {
                return Closure::StartOutside;
            }
            let mut tau = tau0;
            loop {
                let Some((_, bound)) = implied(tau) else {
                    return Closure::IterationLeaving;
                };
                let next = tau0.max(bound);
                if !next.is_finite() {
                    return Closure::IterationLeaving;
                }
                if next <= tau {
                    break;
                }
                tau = next;
            }
            fixed = Some(tau);
            let candidate = 2.0 * tau - tau0;
            let Some((c, bound)) = implied(candidate) else {
                return Closure::Candidate;
            };
            if !(bound < candidate && tau0 < candidate) {
                return Closure::Candidate;
            }
            if 1.0 - candidate - alpha * alpha * h / c * (1.0 + gamma(4)) <= min_j {
                return Closure::Volume;
            }
            let clear = self
                .clear_within((2.0 * h / c).sqrt(), (gravity, planes), span)
                .unwrap_or(false);
            Closure::Closed { clear }
        })();
        let evaluation = Evaluation {
            g_old: reach * root,
            g_t: trace * root,
            rho: trace / reach.max(f64::MIN_POSITIVE),
            tau0,
            after_contact: CONTACTED.with(|c| c.get()),
            closure,
            fixed,
            gain: closure == Closure::Closed { clear: true } && !today,
        };
        EVALUATIONS.with(|e| e.borrow_mut().push(evaluation));
    }
}

/// The lemma's inequalities on a rebuilt end state: (P) at
/// every particle with `tau = |t_p|`, (Global) with `tau = max_p |t_p|`; each
/// against a round-off margin (`density_error` plus `8 u` of the right side's
/// terms), never adjusted.
pub(in super::super) fn check_state(flight: &Exponential, state: &TotalLagrangianState) {
    let Some(law) = &flight.inv.law else {
        return;
    };
    let Some(lemma) = &law.lemma else {
        return;
    };
    let unit = 0.5 * f64::EPSILON;
    let min_j = f64::from(crate::matter::kinematics::volume_ratio::MIN_J);
    let back = DMat2::from_angle(-flight.angle_at(flight.t));
    let n = state.particles;
    let (mut stored, mut margin_sum, mut tau, mut volume_ok) = (0.0, 0.0, 0.0f64, true);
    for p in 0..n {
        let f = back * state.f[p];
        let g = f - DMat2::IDENTITY;
        let (t, w, d1, d2) = parts(g);
        let a = d1 * d1 + d2 * d2;
        let (lambda, mu) = state.solids[p].lame();
        let energy = state.solids[p].energy(f);
        let margin = density_error(f, lambda, mu, energy) + 8.0 * unit * energy.abs();
        stored += state.v0[p] * energy;
        margin_sum += state.v0[p] * margin;
        tau = tau.max(t.abs());
        volume_ok &= f.determinant() > min_j;
        let tp = t.abs();
        if f.determinant() > min_j && tp < 2.0 && lemma.k0 * tp < 1.0 {
            let linear = mu * (a + 0.5 * t * t) + 0.5 * lambda * t * t;
            let right = (1.0 - lemma.k0 * tp) * linear - tp * lemma.kw(tp) * w * w;
            let m = margin + 8.0 * unit * (linear.abs() + (tp * lemma.kw(tp) * w * w).abs());
            POINTWISE.with(|c| {
                let mut k = c.get();
                k.add(energy - right, m);
                c.set(k);
            });
        }
    }
    if !volume_ok {
        return;
    }
    let Some(c) = lemma.c(law.e_k, tau) else {
        return;
    };
    let e_w: f64 = 0.5
        * flight
            .zeta_at(flight.t)
            .iter()
            .map(|z| z.x * z.x)
            .sum::<f64>();
    let right = c * e_w;
    let m = margin_sum + (n as f64 + 8.0) * unit * (stored.abs() + right.abs());
    GLOBAL.with(|k| {
        let mut g = k.get();
        g.add(stored - right, m);
        k.set(g);
    });
}

fn deciles(mut v: Vec<f64>) -> String {
    if v.is_empty() {
        return "none".into();
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    format!(
        "min {:.3e}, 10 % {:.3e}, 50 % {:.3e}, 90 % {:.3e}, max {:.3e}",
        at(0.0),
        at(0.1),
        at(0.5),
        at(0.9),
        at(1.0)
    )
}

/// The lemma's record on this thread, several lines.
pub(in super::super::super) fn report() -> String {
    let all = EVALUATIONS.with(|e| e.borrow().clone());
    let after: Vec<&Evaluation> = all.iter().filter(|e| e.after_contact).collect();
    let count =
        |v: &[&Evaluation], f: &dyn Fn(&Evaluation) -> bool| v.iter().filter(|e| f(e)).count();
    let every: Vec<&Evaluation> = all.iter().collect();
    let closed = |e: &Evaluation| matches!(e.closure, Closure::Closed { .. });
    let (beta_w, frobenius, rows, e_k, beta) = CONSTANTS.with(|c| c.get());
    let (p, g) = (POINTWISE.with(|c| c.get()), GLOBAL.with(|c| c.get()));
    format!(
        "lemma evaluations {} ({} after the first contact); the lemma's loop closed {} ({} after), start outside {}, iteration leaving the range {} (no fixed point), a fixed point but the candidate failing {} ({} after), volume admissibility failed {}; tau* where found: {}\n\
         lemma: clear by the lemma while neither of today's proofs is: {} ({} after the first contact); closed and clear {}\n\
         lemma: beta_w {:.4e} 1/Pa (Frobenius {:.4e}, row sums {:.4e}); the amplitude bound's beta {:.4e}; e_K {:.3e}\n\
         lemma: G_old = max g_p (2H)^(1/2): {}; after the first contact: {}\n\
         lemma: G_t = max g^t_p (2H)^(1/2): {}; after the first contact: {}\n\
         lemma: rho = max g^t / max g: {}; tau0: {}\n\
         lemma: (P) checked {}, below zero {}, beyond the margin {}, least slack / margin {:.3e}; (Global) checked {}, below zero {}, beyond the margin {}, least slack / margin {:.3e}",
        all.len(),
        after.len(),
        count(&every, &closed),
        count(&after, &closed),
        count(&every, &|e| e.closure == Closure::StartOutside),
        count(&every, &|e| e.closure == Closure::IterationLeaving),
        count(&every, &|e| e.closure == Closure::Candidate),
        count(&after, &|e| e.closure == Closure::Candidate),
        count(&every, &|e| e.closure == Closure::Volume),
        deciles(all.iter().filter_map(|e| e.fixed).collect()),
        count(&every, &|e| e.gain),
        count(&after, &|e| e.gain),
        count(&every, &|e| e.closure == Closure::Closed { clear: true }),
        beta_w,
        frobenius,
        rows,
        beta,
        e_k,
        deciles(all.iter().map(|e| e.g_old).collect()),
        deciles(after.iter().map(|e| e.g_old).collect()),
        deciles(all.iter().map(|e| e.g_t).collect()),
        deciles(after.iter().map(|e| e.g_t).collect()),
        deciles(all.iter().map(|e| e.rho).collect()),
        deciles(all.iter().map(|e| e.tau0).collect()),
        p.checked,
        p.negative,
        p.beyond,
        p.least.unwrap_or(f64::NAN),
        g.checked,
        g.negative,
        g.beyond,
        g.least.unwrap_or(f64::NAN)
    )
}
