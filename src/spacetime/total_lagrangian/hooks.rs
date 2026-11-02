use std::cell::{Cell, RefCell};

/// The discrete gradient's base stress (`Elastic::algorithmic_stress`):
/// the midpoint strain's (off) or the average of the two ends' (on). A
/// process-wide switch, as the stress is evaluated on the worker threads.
static TRAPEZOID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_trapezoid(on: bool) {
    TRAPEZOID.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The multipliers' responses started from zero instead of from their last
/// ones (`Factors::responses`), to compare the warm start: a process-wide
/// switch, set between frames.
static COLD_RESPONSES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_cold_responses(on: bool) {
    COLD_RESPONSES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn cold_responses() -> bool {
    COLD_RESPONSES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Every try of the step controller printed (`controlled_step`), as
/// `EMERGE_PROBE_CONTROL_TRACE` does, switched on for one frame by a probe.
static CONTROL_TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_control_trace(on: bool) {
    CONTROL_TRACE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn control_trace() -> bool {
    CONTROL_TRACE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The contacts' gap tolerance without its resolution floor
/// (`contact::gap_resolution`), as it was before it, to compare: a
/// process-wide switch, set between frames, or on for the whole run when
/// the environment variable `EMERGE_PROBE_NO_RESOLUTION_FLOOR` is set.
static NO_RESOLUTION_FLOOR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_no_resolution_floor(on: bool) {
    NO_RESOLUTION_FLOOR.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn no_resolution_floor() -> bool {
    static FROM_ENVIRONMENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    NO_RESOLUTION_FLOOR.load(std::sync::atomic::Ordering::Relaxed)
        || *FROM_ENVIRONMENT
            .get_or_init(|| std::env::var_os("EMERGE_PROBE_NO_RESOLUTION_FLOOR").is_some())
}

/// FAST_MODE_PLAN.md sec. 36.3 (Part I): the walls held at the step's end.
/// With `EMERGE_PROBE_CONTACT_END_GAP` set, every wall condition (an
/// outline's weighted gap, a contact point) is held against the wall itself
/// instead of a plane at its points' start positions, so the held quantity
/// is the gap at the step's end (Signorini there, Puso, Porter and Slavik
/// eq. 16; against a flat rigid wall their `g^_n` is the gap, eq. 22, so
/// their case (23)1 holds at every activation); every wall condition is a
/// candidate, joining in the passes when its end gap is beyond its
/// tolerance; and the tolerance is the resolution alone (`contact_allowance`
/// without its relative term). Contacts between bodies stay route C's.
/// Announced once.
pub(crate) fn end_gap_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("EMERGE_PROBE_CONTACT_END_GAP").is_some();
        if on {
            println!("knob active: walls held at the step's end (Part I)");
        }
        on
    })
}

/// The MAC grid's staggered lattices instead of the collocated ones
/// (`lattices`), to compare: on when the environment variable
/// `EMERGE_PROBE_STAGGERED` is set.
pub(super) fn staggered() -> bool {
    std::env::var_os("EMERGE_PROBE_STAGGERED").is_some()
}

pub(super) fn trapezoid() -> bool {
    TRAPEZOID.load(std::sync::atomic::Ordering::Relaxed)
}

/// The same base in `P` instead of `S` (`Elastic::trapezoid_first_pk`):
/// a diagnostic of the `F_mid S` product, not momentum-exact.
static TRAPEZOID_P: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_trapezoid_p(on: bool) {
    TRAPEZOID_P.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn trapezoid_p() -> bool {
    TRAPEZOID_P.load(std::sync::atomic::Ordering::Relaxed)
}

/// The stress as Gonzalez's split of the energy into its deviatoric
/// and volumetric parts (`Elastic::split_stress`).
static SPLIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_split(on: bool) {
    SPLIT.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn split() -> bool {
    SPLIT.load(std::sync::atomic::Ordering::Relaxed)
}

/// The split's volumetric part as `p cof(F_n+1/2)` in `P` instead of
/// `F_n+1/2 2 p dG(C_n, C_n+1)` (`Elastic::cofactor_volumetric`).
static COFACTOR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_cofactor(on: bool) {
    COFACTOR.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn cofactor() -> bool {
    COFACTOR.load(std::sync::atomic::Ordering::Relaxed)
}

/// The split's volumetric base on its own: 0 as the deviatoric one, 1
/// the midpoint strain's, 2 the two ends' average.
static VOLUMETRIC_BASE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(crate) fn set_volumetric_base(base: u8) {
    VOLUMETRIC_BASE.store(base, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn volumetric_base() -> u8 {
    VOLUMETRIC_BASE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The Kelvin-Voigt factor `beta` (s) of `Elastic::viscous_stress`, as
/// the bits of an `f64`; zero: no viscous stress.
static VISCOSITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn set_viscosity(beta: f64) {
    VISCOSITY.store(beta.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn viscosity() -> f64 {
    f64::from_bits(VISCOSITY.load(std::sync::atomic::Ordering::Relaxed))
}

/// The viscous work of the last converged solve, `u . f_v` per unit
/// density: the heat the step books (its sign is not guaranteed per
/// step by this discretization, so it is reported, not assumed).
static VISCOUS_WORK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) fn set_viscous_work(work: f64) {
    VISCOUS_WORK.store(work.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn viscous_work() -> f64 {
    f64::from_bits(VISCOUS_WORK.load(std::sync::atomic::Ordering::Relaxed))
}

/// Each particle's cell for a `Theta` constant per cell
/// (`Elastic::mixed_stress`), with the number of cells.
pub(crate) type Cells = std::sync::Arc<(Vec<usize>, usize)>;

/// The cells in use; `None`: off.
static CELLS: std::sync::RwLock<Option<Cells>> = std::sync::RwLock::new(None);

pub(crate) fn set_cells(cells: Option<(Vec<usize>, usize)>) {
    *CELLS.write().expect("cells lock") = cells.map(std::sync::Arc::new);
}

pub(super) fn cells() -> Option<Cells> {
    CELLS.read().expect("cells lock").clone()
}

/// Faces whose unknowns carry no inertia, as (lattice, i, j): their
/// rows and columns of the mass are zero (`Grid::massless`).
pub(crate) type Massless = std::sync::Arc<Vec<(usize, usize, usize)>>;

/// The massless faces in use; `None`: off.
static MASSLESS: std::sync::RwLock<Option<Massless>> = std::sync::RwLock::new(None);

pub(crate) fn set_massless(faces: Option<Vec<(usize, usize, usize)>>) {
    *MASSLESS.write().expect("massless lock") = faces.map(std::sync::Arc::new);
}

pub(super) fn massless() -> Option<Massless> {
    MASSLESS.read().expect("massless lock").clone()
}

thread_local! {
    static TRACE: RefCell<Option<Vec<Vec<String>>>> = const { RefCell::new(None) };
    static HOMOTOPY: Cell<u32> = const { Cell::new(0) };
    static FOLD: Cell<bool> = const { Cell::new(false) };
    static TORQUE: Cell<Option<(f64, f64)>> = const { Cell::new(None) };
}

/// Records, at each `step_total_lagrangian_contact`, the particles'
/// angular momentum change the step gives (`sum m x_mid x dv`) and the
/// share of it the contacts' push alone gives (`dt M^-1 push` in place
/// of `dv`); `take_torque` reads the last.
pub(crate) fn watch_torque(on: bool) {
    TORQUE.with(|t| t.set(on.then_some((0.0, 0.0))));
}

pub(crate) fn take_torque() -> Option<(f64, f64)> {
    TORQUE.with(Cell::get)
}

pub(super) fn torque_on() -> bool {
    TORQUE.with(|t| t.get().is_some())
}

pub(super) fn set_torque(value: (f64, f64)) {
    TORQUE.with(|t| t.set(Some(value)));
}

/// Starts recording; `increments > 0` also runs the homotopy, starting
/// with that many load increments, at every solve.
pub(crate) fn start(increments: u32) {
    TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
    HOMOTOPY.with(|h| h.set(increments));
}

/// Stops recording: the lines of every solve, one list per solve.
pub(crate) fn finish() -> Vec<Vec<String>> {
    HOMOTOPY.with(|h| h.set(0));
    TRACE.with(|t| t.borrow_mut().take().unwrap_or_default())
}

pub(super) fn on() -> bool {
    TRACE.with(|t| t.borrow().is_some())
}

pub(super) fn homotopy() -> u32 {
    HOMOTOPY.with(Cell::get)
}

/// Also estimate, at every point the homotopy reaches, the Jacobian's
/// eigenvalues of least and largest magnitude (a fold of the step's
/// equations makes the least one vanish).
pub(crate) fn watch_fold(on: bool) {
    FOLD.with(|f| f.set(on));
}

pub(super) fn fold() -> bool {
    FOLD.with(Cell::get)
}

pub(super) fn begin_solve() {
    TRACE.with(|t| {
        if let Some(all) = t.borrow_mut().as_mut() {
            all.push(Vec::new());
        }
    });
}

pub(super) fn line(text: String) {
    TRACE.with(|t| {
        if let Some(current) = t.borrow_mut().as_mut().and_then(|all| all.last_mut()) {
            current.push(text);
        }
    });
}

/// The condition probe: per Newton iteration of the next `samples` ones,
/// the active unknowns' count and the infinity-norm condition numbers of
/// the Jacobian and of the consistent mass, `|A|_inf |A^-1|_inf`, each
/// assembled dense column by column from the solve's own products.
/// Process-wide (the tries may run on worker threads).
static CONDITION: std::sync::Mutex<(usize, Vec<ConditionRow>)> =
    std::sync::Mutex::new((0, Vec::new()));

/// One Newton iteration's row: the active unknowns, the Jacobian's and the
/// mass's `kappa_inf`.
pub(crate) type ConditionRow = (usize, f64, f64);

pub(crate) fn watch_condition(samples: usize) {
    *CONDITION.lock().expect("condition lock") = (samples, Vec::new());
}

pub(crate) fn take_condition() -> Vec<ConditionRow> {
    std::mem::take(&mut CONDITION.lock().expect("condition lock").1)
}

/// `|A|_inf |A^-1|_inf` of the dense `n` by `n` row-major `a`, the inverse
/// by Gaussian elimination with partial pivoting; infinite when a pivot
/// vanishes.
fn condition_inf(mut a: Vec<f64>, n: usize) -> f64 {
    let norm = |m: &[f64]| {
        (0..n)
            .map(|i| m[i * n..(i + 1) * n].iter().map(|e| e.abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    let size = norm(&a);
    let mut inverse: Vec<f64> = (0..n * n)
        .map(|k| if k / n == k % n { 1.0 } else { 0.0 })
        .collect();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .expect("a row");
        if a[pivot * n + col] == 0.0 {
            return f64::INFINITY;
        }
        for k in 0..n {
            a.swap(col * n + k, pivot * n + k);
            inverse.swap(col * n + k, pivot * n + k);
        }
        let d = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= d;
            inverse[col * n + k] /= d;
        }
        for row in 0..n {
            let factor = a[row * n + col];
            if row == col || factor == 0.0 {
                continue;
            }
            for k in 0..n {
                a[row * n + k] -= factor * a[col * n + k];
                inverse[row * n + k] -= factor * inverse[col * n + k];
            }
        }
    }
    size * norm(&inverse)
}

/// Records one Newton iteration's condition numbers while the probe asks
/// for samples (`watch_condition`).
pub(super) fn record_condition(
    jacobian: &dyn Fn(&[f64]) -> Vec<f64>,
    grid: &super::Grid,
    mass: &[f64],
) {
    {
        let state = CONDITION.lock().expect("condition lock");
        if state.0 == 0 {
            return;
        }
    }
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    let (mut j, mut m) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (c, &column) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[column] = 1.0;
        let (jc, mc) = (jacobian(&unit), grid.mass_times(&unit, mass, 0.0));
        for (r, &row) in unknowns.iter().enumerate() {
            j[r * n + c] = jc[row];
            m[r * n + c] = mc[row];
        }
    }
    let row = (n, condition_inf(j, n), condition_inf(m, n));
    let mut state = CONDITION.lock().expect("condition lock");
    if state.0 > 0 {
        state.0 -= 1;
        state.1.push(row);
    }
}

/// Where Norton's returns are spent in the step controller (`controlled_step`,
/// Acary's estimator): in the whole step of a kept pair of tries, in its kept
/// halves, and in rejected pairs. Counted from `j2_hencky::returns_seen`
/// while the probes watch the returns.
static RETURN_SPLIT: std::sync::Mutex<[u64; 3]> = std::sync::Mutex::new([0; 3]);

pub(crate) fn add_return_split(whole: u64, halves: u64, kept: bool) {
    let mut split = RETURN_SPLIT.lock().expect("return split lock");
    if kept {
        split[0] += whole;
        split[1] += halves;
    } else {
        split[2] += whole + halves;
    }
}

pub(crate) fn take_return_split() -> [u64; 3] {
    std::mem::take(&mut *RETURN_SPLIT.lock().expect("return split lock"))
}

/// The probes' record of the band factorizations (`solve`): the
/// coefficient `inertia / dt^2` of each factor built, in order.
static FACTORS_BUILT: std::sync::Mutex<Option<Vec<f64>>> = std::sync::Mutex::new(None);

pub(crate) fn watch_factors(on: bool) {
    *FACTORS_BUILT.lock().expect("factors lock") = on.then(Vec::new);
}

pub(super) fn record_factor(coefficient: f64) {
    if let Some(list) = FACTORS_BUILT.lock().expect("factors lock").as_mut() {
        list.push(coefficient);
    }
}

pub(crate) fn take_factors() -> Vec<f64> {
    std::mem::take(
        FACTORS_BUILT
            .lock()
            .expect("factors lock")
            .as_mut()
            .unwrap_or(&mut Vec::new()),
    )
}

/// The probes' replay of Newton's linear solves (`solve`): for a solve whose
/// band factor was reused from another coefficient, the same system solved
/// again with its own exact factor; per solve `(iterations, true residual /
/// |b|)` with the reused factor and with the exact one. And for the solves
/// that used their own factor, `(iterations, true residual / |b|)`.
type Replay = Option<(Vec<(u32, f64, u32, f64)>, Vec<(u32, f64)>)>;

static GMRES_REPLAY: std::sync::Mutex<Replay> = std::sync::Mutex::new(None);

pub(crate) fn watch_gmres_replay(on: bool) {
    *GMRES_REPLAY.lock().expect("replay lock") = on.then(|| (Vec::new(), Vec::new()));
}

pub(super) fn gmres_replay() -> bool {
    GMRES_REPLAY.lock().expect("replay lock").is_some()
}

pub(super) fn record_reused(reused: (u32, f64), exact: (u32, f64)) {
    if let Some((list, _)) = GMRES_REPLAY.lock().expect("replay lock").as_mut() {
        list.push((reused.0, reused.1, exact.0, exact.1));
    }
}

pub(super) fn record_own(own: (u32, f64)) {
    if let Some((_, list)) = GMRES_REPLAY.lock().expect("replay lock").as_mut() {
        list.push(own);
    }
}

pub(crate) fn take_gmres_replay() -> Replay {
    GMRES_REPLAY.lock().expect("replay lock").take()
}

/// The preconditioned condition probe: per Newton iteration of the next
/// `samples` ones, `kappa_inf(P(c)^-1 A)` and `kappa_inf(P(c/4)^-1 A)`, `A` the
/// step's Jacobian and `P(c)` the band preconditioner built for the
/// system's own coefficient `c` or for a quarter of it (the whole step's,
/// reused on the halves), assembled dense column by column.
static PRECONDITIONED: std::sync::Mutex<(usize, Vec<(f64, f64)>)> =
    std::sync::Mutex::new((0, Vec::new()));

pub(crate) fn watch_preconditioned(samples: usize) {
    *PRECONDITIONED.lock().expect("preconditioned lock") = (samples, Vec::new());
}

pub(super) fn preconditioned_wanted() -> bool {
    PRECONDITIONED.lock().expect("preconditioned lock").0 > 0
}

pub(crate) fn take_preconditioned() -> Vec<(f64, f64)> {
    std::mem::take(&mut PRECONDITIONED.lock().expect("preconditioned lock").1)
}

pub(super) fn record_preconditioned(
    jacobian: &dyn Fn(&[f64]) -> Vec<f64>,
    grid: &super::Grid,
    (own, quarter): (&super::Band, &super::Band),
) {
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    let (mut a_own, mut a_quarter) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (c, &column) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[column] = 1.0;
        let a = jacobian(&unit);
        let (z_own, z_quarter) = (own.solve(&a), quarter.solve(&a));
        for (r, &row) in unknowns.iter().enumerate() {
            a_own[r * n + c] = z_own[row];
            a_quarter[r * n + c] = z_quarter[row];
        }
    }
    let row = (condition_inf(a_own, n), condition_inf(a_quarter, n));
    let mut state = PRECONDITIONED.lock().expect("preconditioned lock");
    if state.0 > 0 {
        state.0 -= 1;
        state.1.push(row);
    }
}

/// The probes' split of the step controller's tries (`controlled_step`,
/// Acary's doubling) by class and outcome: smooth (eq. 63) or switching
/// (a point reaching a wall, eqs. 67-70), each kept, rejected on its error,
/// or rejected unconverged; indexed `2 * outcome + switching`.
static TRIES: std::sync::Mutex<[u32; 6]> = std::sync::Mutex::new([0; 6]);

pub(super) fn add_try(switching: bool, outcome: usize) {
    TRIES.lock().expect("tries lock")[2 * outcome + usize::from(switching)] += 1;
}

pub(crate) fn take_tries() -> [u32; 6] {
    std::mem::take(&mut *TRIES.lock().expect("tries lock"))
}

/// Whether a probe is armed that a try's thread or the tries' order would
/// change: the per-thread trace, homotopy, fold and torque records (read on
/// the calling thread), the records kept in call order (condition numbers,
/// factors built, GMRES replays, the sampled whole step's estimate), the
/// viscous work (the last solve's), and the factor prints of the control
/// trace. `controlled_step` then runs its tries one after the other.
pub(super) fn order_sensitive() -> bool {
    on() || homotopy() > 0
        || fold()
        || torque_on()
        || CONDITION.lock().expect("condition lock").0 > 0
        || preconditioned_wanted()
        || FACTORS_BUILT.lock().expect("factors lock").is_some()
        || gmres_replay()
        || lobatto_wanted()
        || viscosity() != 0.0
        || control_trace()
}

/// The error estimate's measurement (`estimate`, `step_control`): every
/// `every`-th pair of the step controller's tries is sampled (0: off; the
/// default read once from `EMERGE_PROBE_EFFECTIVITY`), and every
/// `halving`-th sampled pair also runs the step-halving check (0: never).
static EFFECTIVITY: std::sync::Mutex<Option<(u32, u32, u32)>> = std::sync::Mutex::new(None);

pub(crate) fn set_effectivity(every: u32, halving: u32) {
    *EFFECTIVITY.lock().expect("effectivity lock") = Some((every, halving, 0));
}

/// `Some(halving)` when this pair is sampled.
pub(super) fn effectivity_sample() -> Option<bool> {
    let mut state = EFFECTIVITY.lock().expect("effectivity lock");
    let (every, halving, count) = state.get_or_insert_with(|| {
        let every = std::env::var("EMERGE_PROBE_EFFECTIVITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        (every, 0, 0)
    });
    if *every == 0 {
        return None;
    }
    *count += 1;
    if *count % *every != 0 {
        return None;
    }
    let sampled = *count / *every;
    Some(*halving > 0 && sampled % *halving == 0)
}

/// The last converged solve's estimate, while a sampled whole step runs.
static LOBATTO: std::sync::Mutex<(bool, Option<super::estimate::ParticleError>)> =
    std::sync::Mutex::new((false, None));

pub(super) fn want_lobatto(on: bool) {
    *LOBATTO.lock().expect("lobatto lock") = (on, None);
}

pub(super) fn lobatto_wanted() -> bool {
    LOBATTO.lock().expect("lobatto lock").0
}

pub(super) fn record_lobatto(error: super::estimate::ParticleError) {
    LOBATTO.lock().expect("lobatto lock").1 = Some(error);
}

pub(super) fn take_lobatto() -> Option<super::estimate::ParticleError> {
    let mut state = LOBATTO.lock().expect("lobatto lock");
    state.0 = false;
    state.1.take()
}

/// One sampled pair (`step_control`), every size in the controller's norm:
/// the step `dt`; whether the pair is switching-class, and whether its
/// whole step held a contact; the estimate of the
/// whole step (`estimate`), also with its kinetic part from the velocity
/// error (Lacoma and Romero's eq. 32 form); the whole step's error against
/// the 64-substep reference (both forms); the doubling's estimate of the
/// halves and the halves' error against the reference; and, on halving
/// samples, `(h, estimate, reference error)` at `dt`, `dt/2`, `dt/4`,
/// `dt/8`.
#[derive(Clone, Debug)]
pub(crate) struct EffectivityRow {
    pub(crate) dt: f64,
    pub(crate) switching: bool,
    /// Whether the whole step held a contact (`StepReport::contacts`).
    pub(crate) held: bool,
    /// The whole step's largest particle speeds: start, mean, end
    /// (`estimate::ParticleError`).
    pub(crate) speeds: [f64; 3],
    /// The frame the pair belongs to (`set_effectivity_frame`), the
    /// controller's target and the error it compared with it (eq. 63's or
    /// eq. 67's), and the plastic work of the whole step.
    pub(crate) frame: u32,
    pub(crate) target: f64,
    pub(crate) compared: f64,
    pub(crate) plastic: f64,
    pub(crate) estimate: f64,
    pub(crate) estimate_velocity: f64,
    pub(crate) whole_error: f64,
    pub(crate) whole_error_velocity: f64,
    pub(crate) doubling: f64,
    pub(crate) halves_error: f64,
    pub(crate) halving: Vec<(f64, f64, f64)>,
}

static EFFECTIVITY_ROWS: std::sync::Mutex<Vec<EffectivityRow>> = std::sync::Mutex::new(Vec::new());

/// The frame the next sampled pairs belong to (`EffectivityRow::frame`).
static EFFECTIVITY_FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) fn set_effectivity_frame(frame: u32) {
    EFFECTIVITY_FRAME.store(frame, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn effectivity_frame() -> u32 {
    EFFECTIVITY_FRAME.load(std::sync::atomic::Ordering::Relaxed)
}

pub(super) fn record_effectivity(row: EffectivityRow) {
    EFFECTIVITY_ROWS.lock().expect("rows lock").push(row);
}

pub(crate) fn take_effectivity() -> Vec<EffectivityRow> {
    std::mem::take(&mut *EFFECTIVITY_ROWS.lock().expect("rows lock"))
}

/// TEMPORARY measurement (not to be committed): the solve's parts, summed:
/// 0 Jacobian products, 1 their vector setup s, 2 their mass products s,
/// 3 their internal-force products s, 4 their masking s, 5 linearisations,
/// 6 residual evaluations, 7 solves, 8 the solves' wall time s, 9 Newton
/// iterations.
pub(crate) static PARTS: std::sync::Mutex<[f64; 10]> = std::sync::Mutex::new([0.0; 10]);

pub(crate) fn add_part(i: usize, v: f64) {
    PARTS.lock().expect("parts")[i] += v;
}

pub(crate) fn take_parts() -> [f64; 10] {
    std::mem::take(&mut *PARTS.lock().expect("parts"))
}

/// FAST_MODE_PLAN.md sec. 36.2 (K-2): over the kept and tried steps since
/// the last take, how many held conditions' allowances were bound by the
/// relative term, the round-off and the resolution, and the largest
/// relative term, m.
static ALLOWANCE_TERMS: std::sync::Mutex<[f64; 4]> = std::sync::Mutex::new([0.0; 4]);

pub(crate) fn add_allowance_term(binding: usize, relative: f64) {
    let mut terms = ALLOWANCE_TERMS.lock().expect("allowance terms");
    terms[binding] += 1.0;
    terms[3] = terms[3].max(relative);
}

pub(crate) fn take_allowance_terms() -> [f64; 4] {
    std::mem::take(&mut *ALLOWANCE_TERMS.lock().expect("allowance terms"))
}

/// One held contact of one step in the probes' contact trace
/// (`tmp/papers/dl_realtime/CHATTER_TRACE_PLANS.md`, M1): its key, its
/// multiplier at the step's end, its weighted normal velocity `G v` at the
/// step's start and end (positive separating), the gap allowance it was held
/// to (`contact_allowance`) and that allowance's resolution term
/// (`gap_resolution`), and its push on each point it spreads to, the force
/// the step applies over its length (M2).
#[derive(Clone, Debug)]
pub(crate) struct ContactRow {
    pub(crate) key: super::ContactKey,
    pub(crate) multiplier: f64,
    pub(crate) start: f64,
    pub(crate) end: f64,
    pub(crate) allowance: f64,
    pub(crate) resolution: f64,
    pub(crate) push: Vec<(usize, glam::DVec2)>,
}

/// A step of the contact trace: its length and its contacts' rows.
pub(crate) type TracedStep = (f64, Vec<ContactRow>);

/// The contact trace: every solve's rows by id while a controlled step
/// runs, and the kept steps' rows in order.
type ContactTrace = Option<(u64, Vec<(u64, TracedStep)>, Vec<TracedStep>)>;

static CONTACT_TRACE: std::sync::Mutex<ContactTrace> = std::sync::Mutex::new(None);

pub(crate) fn watch_contact_rows(on: bool) {
    *CONTACT_TRACE.lock().expect("contact trace lock") = on.then(|| (0, Vec::new(), Vec::new()));
}

pub(super) fn contact_rows_on() -> bool {
    CONTACT_TRACE.lock().expect("contact trace lock").is_some()
}

/// Records one solve's rows; returns its id (`StepReport::trace_id`).
pub(super) fn record_contact_rows(dt: f64, rows: Vec<ContactRow>) -> u64 {
    let mut trace = CONTACT_TRACE.lock().expect("contact trace lock");
    let Some((next, pending, _)) = trace.as_mut() else {
        return 0;
    };
    *next += 1;
    pending.push((*next, (dt, rows)));
    *next
}

/// Keeps the rows of a controlled step's kept solves, in order, and drops
/// the others (rejected tries, error estimates).
pub(super) fn keep_contact_rows(kept: &[super::StepReport]) {
    let mut trace = CONTACT_TRACE.lock().expect("contact trace lock");
    let Some((_, pending, steps)) = trace.as_mut() else {
        return;
    };
    for report in kept {
        if let Some(i) = pending.iter().position(|(id, _)| *id == report.trace_id) {
            steps.push(pending.swap_remove(i).1);
        }
    }
    pending.clear();
}

pub(crate) fn take_contact_rows() -> Vec<TracedStep> {
    CONTACT_TRACE
        .lock()
        .expect("contact trace lock")
        .as_mut()
        .map(|(_, _, steps)| std::mem::take(steps))
        .unwrap_or_default()
}

/// A contact's weighted normal velocity `G v`, positive separating, over
/// its shares (`contact_geometry`).
fn normal_velocity(shares: &[super::Share], v: &[glam::DVec2]) -> f64 {
    shares
        .iter()
        .map(|s| {
            let relative: glam::DVec2 = s.spread.iter().map(|&(r, w)| w * v[r]).sum();
            s.weight * s.normal.dot(relative)
        })
        .sum()
}

/// The contact trace's rows at a step's start (`step_total_lagrangian_
/// contact`, before the particles' update), each with its shares at the
/// start positions, as the step measures them; `None` when the trace is off.
pub(super) fn contact_rows_start(
    grid: &super::Grid,
    (x, v, used, u): (&[glam::DVec2], &[glam::DVec2], &[super::Plane], &[f64]),
    contacts: &[super::Contact],
    (penalty, residual): (&[f64], f64),
    (tolerance, dt): (f64, f64),
) -> Option<Vec<(ContactRow, Vec<super::Share>)>> {
    contact_rows_on().then(|| {
        contacts
            .iter()
            .map(|c| {
                let (_, shares) = super::contact_geometry(grid, x, used, u, c);
                let resolution = super::gap_resolution(c, penalty, residual);
                // The force as the step spreads it (`step_total_lagrangian_
                // contact`'s torque probe): the frictionless walls' part.
                let push = shares
                    .iter()
                    .flat_map(|s| {
                        let total =
                            s.weight * (c.multiplier * s.normal + c.tangential * s.normal.perp());
                        s.spread.map(|(r, w)| (r, w * total))
                    })
                    .collect();
                let row = ContactRow {
                    key: c.key,
                    multiplier: c.multiplier,
                    start: normal_velocity(&shares, v),
                    end: 0.0,
                    allowance: super::contact_allowance(
                        grid,
                        (x, v),
                        used,
                        u,
                        &shares,
                        (tolerance, dt, resolution),
                    ),
                    resolution,
                    push,
                };
                (row, shares)
            })
            .collect()
    })
}

/// The rows completed with the end velocities, after the update.
pub(super) fn contact_rows_end(
    traced: Vec<(ContactRow, Vec<super::Share>)>,
    v: &[glam::DVec2],
) -> Vec<ContactRow> {
    traced
        .into_iter()
        .map(|(row, shares)| ContactRow {
            end: normal_velocity(&shares, v),
            ..row
        })
        .collect()
}

/// PER_BODY_TOLERANCE_PLAN.md: each particle's body and each body's own
/// tolerance energy `E0_b`, set at the path's start by the solids layer
/// when `EMERGE_PROBE_TOL_PER_BODY` is on; the controller's error then
/// keeps every body within `tol sqrt(E0_b)` (`step_control`'s `norm`).
pub(crate) type BodyScales = std::rc::Rc<(Vec<usize>, Vec<f64>)>;

thread_local! {
    static BODY_SCALES: RefCell<Option<BodyScales>> = const { RefCell::new(None) };
}

pub(crate) fn set_body_scales(scales: Option<(Vec<usize>, Vec<f64>)>) {
    BODY_SCALES.with(|b| *b.borrow_mut() = scales.map(std::rc::Rc::new));
}

pub(crate) fn body_scales() -> Option<BodyScales> {
    BODY_SCALES.with(|b| b.borrow().clone())
}

/// ROW15_REACH_CUT_PLAN.md: the points the controller watches at its first
/// pair (per state point, whether it belongs to a body free at the step's
/// start), and what it records when one of them reaches a plane there: the
/// pair's length, the estimator's floor and the points reached. The step
/// is then handed back untaken (`controlled_step`).
pub(crate) struct ReachAbort {
    pub(crate) dt: f64,
    pub(crate) floor: f64,
    pub(crate) reached: Vec<usize>,
}

thread_local! {
    static REACH_WATCH: RefCell<Option<std::rc::Rc<Vec<bool>>>> = const { RefCell::new(None) };
    static REACH_ABORT: RefCell<Option<ReachAbort>> = const { RefCell::new(None) };
    static REACH_CRITICAL: Cell<Option<f64>> = const { Cell::new(None) };
    static REACH_LATE: Cell<u32> = const { Cell::new(0) };
}

pub(crate) fn set_reach_watch(watch: Option<Vec<bool>>) {
    REACH_WATCH.with(|w| *w.borrow_mut() = watch.map(std::rc::Rc::new));
}

pub(crate) fn reach_watch() -> Option<std::rc::Rc<Vec<bool>>> {
    REACH_WATCH.with(|w| w.borrow().clone())
}

pub(crate) fn set_reach_abort(abort: ReachAbort) {
    REACH_ABORT.with(|a| *a.borrow_mut() = Some(abort));
}

pub(crate) fn take_reach_abort() -> Option<ReachAbort> {
    REACH_ABORT.with(|a| a.borrow_mut().take())
}

/// The critical step's length for the next `controlled_step` call: one
/// unestimated try of that length.
pub(crate) fn set_reach_critical(length: Option<f64>) {
    REACH_CRITICAL.with(|c| c.set(length));
}

pub(crate) fn take_reach_critical() -> Option<f64> {
    REACH_CRITICAL.with(|c| c.take())
}

/// A watched point reaching a plane after a rejected pair (left as today).
pub(crate) fn add_reach_late() {
    REACH_LATE.with(|c| c.set(c.get() + 1));
}

pub(crate) fn take_reach_late() -> u32 {
    REACH_LATE.with(|c| c.replace(0))
}

/// MIXED_FORMULATION_PLAN.md criterion 6': a grid axis whose every face the
/// contact step's grid holds at zero velocity (`Held`), so a bar's motion is
/// 1D (laterally confined, rollers on its long sides); `usize::MAX` off. A
/// process-wide value: the grid may be built on a worker thread.
static HELD_AXIS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

pub(crate) fn set_held_axis(axis: Option<usize>) {
    HELD_AXIS.store(
        axis.unwrap_or(usize::MAX),
        std::sync::atomic::Ordering::Relaxed,
    );
}

pub(crate) fn held_axis() -> Option<usize> {
    let axis = HELD_AXIS.load(std::sync::atomic::Ordering::Relaxed);
    (axis != usize::MAX).then_some(axis)
}

/// F4 (FAST_MODE_PLAN.md sec. 13): the step solved by backward Euler instead
/// of the conserving midpoint, set by the contact phase's control for each
/// controlled step inside it. A process-wide switch: the step's tries run on
/// the pool's threads.
static BACKWARD_EULER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn set_backward_euler(on: bool) {
    BACKWARD_EULER.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn backward_euler() -> bool {
    BACKWARD_EULER.load(std::sync::atomic::Ordering::Relaxed)
}

/// What the backward Euler steps booked as the scheme's heat since the last
/// `take_scheme_heat`: the total, the steps, the most negative step total,
/// and the particles' negative terms (count and sum). Every finished try
/// books, a discarded one included.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SchemeHeat {
    pub(crate) total: f64,
    pub(crate) steps: u64,
    pub(crate) lowest_step: f64,
    pub(crate) negative: u64,
    pub(crate) negative_sum: f64,
}

static SCHEME_HEAT: std::sync::Mutex<SchemeHeat> = std::sync::Mutex::new(SchemeHeat {
    total: 0.0,
    steps: 0,
    lowest_step: f64::INFINITY,
    negative: 0,
    negative_sum: 0.0,
});

pub(super) fn book_scheme_heat(terms: &[f64]) {
    let mut heat = SCHEME_HEAT.lock().expect("scheme heat lock");
    let step: f64 = terms.iter().sum();
    heat.total += step;
    heat.steps += 1;
    heat.lowest_step = heat.lowest_step.min(step);
    for &d in terms.iter().filter(|d| **d < 0.0) {
        heat.negative += 1;
        heat.negative_sum += d;
    }
}

pub(crate) fn take_scheme_heat() -> SchemeHeat {
    let mut heat = SCHEME_HEAT.lock().expect("scheme heat lock");
    let out = *heat;
    *heat = SchemeHeat {
        total: 0.0,
        steps: 0,
        lowest_step: f64::INFINITY,
        negative: 0,
        negative_sum: 0.0,
    };
    out
}

/// `EMERGE_PROBE_TANGENT_CHECK` set (read once): every exact internal-force
/// product compared with central differences of the force on the same
/// iterate, the largest gap printed relative to the product's size.
pub(super) fn tangent_check() -> bool {
    static SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("EMERGE_PROBE_TANGENT_CHECK").is_some())
}
