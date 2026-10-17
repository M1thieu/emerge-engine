//! The exact 1D generalised Zener solution (`common/zener.rs`) checked
//! against itself and against Blanc et al. 2016, then the engine's solids
//! with relaxation branches measured against it.
//!
//! Two parts, each with its criteria fixed before measuring: the
//! reference's own exactness, and the gates of the path's relaxation
//! branches.

extern crate emerge_engine as emerge;

#[path = "common/zener.rs"]
mod zener;

use std::f64::consts::PI;
use zener::{ModeSolution, ModeState, Zener};

use emerge::materials::physical_props::Elastic;
use emerge::materials::relaxation::{Mechanism, RelaxationSpectrum};
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

// The exact 1D generalised Zener bar (`common/zener.rs`) checked against
// itself and against Blanc et al. 2016, before any engine code uses it.
//
// A free-free bar's mode depends on the material only through `w^2 = E_r
// k^2 / rho` and the mechanisms, so the reference is checked at mode
// frequencies below, inside and above the band the mechanisms were fitted
// over. The mechanisms are the first record: Plasticine, a constant loss
// factor 0.225 over 100-3000 Hz (Crandall, Kurzweil, Nigam and Remington
// 1970) fitted with three mechanisms by Blanc et al.'s eq. 28 (minimax,
// positive weights, their eq. 33): the largest error in the loss factor
// over the band is 0.83 %.
//
// Criteria fixed before measuring:
// (a) every root makes the characteristic polynomial vanish to 1e-12 of
// its terms, the start state is reproduced to 1e-12, and the assembled
// solution satisfies the mode's equations to 1e-12 of their terms at
// every sampled time. (Fixed first as "every eigenvector satisfies `A v
// = p v` to 1e-12": that asks more than a root next to `-theta_l` can
// be held to in f64, 1.8e-12 measured at 30 Hz, so it was replaced by
// the assembled solution's residual, which is the exactness wanted.
// Removal criterion: a root and eigenvector method with a sourced
// conditioning bound, against which the eigenvectors are checked
// instead.)
// (b) Blanc et al.'s energy (Property 1) plus the heat made stays the
// start energy to 1e-12 of it over twenty periods, and the heat never
// decreases;
// (c) the state stays real (imaginary residue below 1e-12 of its size).
//
// Reported, not asserted: each mode's logarithmic decrement against
// `pi Q^-1` at its damped frequency (their eq. 28; equal to first order in
// `Q^-1` only).

/// Plasticine's three mechanisms: weights and relaxation frequencies
/// (rad/s), fitted to Crandall et al.'s loss factor 0.225 over 100-3000 Hz.
fn plasticine(rho: f64, relaxed: f64) -> Zener {
    Zener {
        rho,
        relaxed,
        kappa: vec![0.3807440105974887, 0.3612646181734217, 0.7539982742914667],
        theta: vec![654.8767306216079, 3865.458262006148, 25438.050835431713],
    }
}

#[test]
fn a_zener_mode_rings_down_exactly_and_its_energy_becomes_heat() {
    // A unit density and relaxed modulus: the mode's frequency is then its
    // wavenumber.
    let material = plasticine(1.0, 1.0);
    println!(
        "Plasticine record: unrelaxed over relaxed modulus {:.4}",
        material.unrelaxed() / material.relaxed
    );
    for f_hz in [30.0, 104.0, 530.0, 1500.0, 4049.0, 20000.0] {
        let k = 2.0 * PI * f_hz;
        // Released from rest at zero strain with the velocity of the mode,
        // its memory empty.
        let start = ModeState {
            q: 0.0,
            rate: 1.0,
            memory: vec![0.0; 3],
        };
        let mode = ModeSolution::new(&material, k, &start);
        let (root, reproduced) = mode.checks();
        assert!(
            root < 1e-12 && reproduced < 1e-12,
            "{f_hz} Hz: roots {root:.2e}, start {reproduced:.2e}"
        );
        // The oscillating pair: the root with the largest imaginary part.
        let pair = mode
            .roots
            .iter()
            .copied()
            .max_by(|a, b| a.im.total_cmp(&b.im))
            .expect("roots");
        let (decay, damped) = (-pair.re, pair.im);
        let period = 2.0 * PI / damped;
        let e0 = mode.energy(&start);
        let (mut ledger, mut imaginary, mut last_heat, mut monotone) = (0.0f64, 0.0f64, 0.0, true);
        let mut residual = 0.0f64;
        for i in 0..=2000 {
            let t = 20.0 * period * i as f64 / 2000.0;
            let (state, im) = mode.at(t);
            let heat = mode.heat(t);
            residual = residual.max(mode.residual(t));
            ledger = ledger.max((mode.energy(&state) + heat - e0).abs() / e0);
            // Against the start's velocity amplitude, `sqrt(2 E0)`.
            imaginary = imaginary.max(im / (2.0 * e0).sqrt());
            if heat < last_heat - 1e-15 * e0 {
                monotone = false;
            }
            last_heat = heat;
        }
        let decrement = 2.0 * PI * decay / damped;
        let q_inv = material.inverse_q(damped);
        let frame = mode.energy(&mode.at(1.0 / 60.0).0) / e0;
        println!(
            "{f_hz:>7} Hz: damped {:.4} Hz, decrement {decrement:.5} against pi Q^-1 {:.5} (ratio {:.4}); \
             roots {root:.1e}, start {reproduced:.1e}, equations {residual:.1e}, ledger {ledger:.1e}, imaginary {imaginary:.1e}; \
             energy left after a 1/60 s frame {frame:.3e}",
            damped / (2.0 * PI),
            PI * q_inv,
            decrement / (PI * q_inv)
        );
        assert!(
            ledger < 1e-12,
            "{f_hz} Hz: energy plus heat off the start by {ledger:.2e} of it"
        );
        assert!(monotone, "{f_hz} Hz: the heat decreased");
        assert!(
            residual < 1e-12,
            "{f_hz} Hz: the equations' residual {residual:.2e} of their terms"
        );
        assert!(
            imaginary < 1e-12,
            "{f_hz} Hz: imaginary residue {imaginary:.2e}"
        );
    }
}

/// A Zener rod dropped flat on a rigid wall, with the rod's quarter-wave
/// frequency `c_r / 4L` across the mechanisms' band. Criteria fixed before
/// measuring: without mechanisms the rod leaves at `2L / c` with all its
/// momentum, to the truncation's order (1e-3 at 1024 modes; measured 0.99921,
/// 0.99980, 0.99995 at 256, 1024, 4096); with them, the energy plus the heat
/// stays the truncated series' start energy to 1e-12 at separation. The
/// restitution is reported against the number of modes: the same to 5
/// digits at 256 and 1024 modes up to 1.5 kHz, 4e-4 apart at 4049 Hz,
/// where the rod leaves exactly at the unrelaxed front's return and the
/// separation time converges like the truncation's `1/M`.
///
/// Declared gaps, each with its removal criterion:
/// - a second contact after separation is not followed: the end's flight
///   after separation projected on the free-free modes, with any return to
///   the wall solved as a new contact;
/// - the record's relaxed modulus is a unit here (the mode depends on it
///   only through the quarter-wave frequency): before any scene uses the
///   record, `E_r` is set from Crandall et al.'s measured wave speed at a
///   reference frequency (Blanc et al. eq. 25).
#[test]
fn a_zener_rod_on_a_rigid_wall_leaves_with_the_momentum_its_damping_leaves() {
    let elastic = Zener {
        rho: 1.0,
        relaxed: 1.0,
        kappa: vec![],
        theta: vec![],
    };
    let material = plasticine(1.0, 1.0);
    for f1 in [104.0, 530.0, 1500.0, 4049.0] {
        // c_r = 1: L = c_r / (4 f1).
        let length = 0.25 / f1;
        for modes in [256, 1024] {
            let free = zener::rod_on_wall(&elastic, length, 1.0, modes);
            let damped = zener::rod_on_wall(&material, length, 1.0, modes);
            println!(
                "quarter-wave {f1:>6} Hz, {modes:>5} modes: elastic leaves at {:.5} of 2L/c with {:.5} of its momentum; \
                 Zener leaves at {:.5} of 2L/c_r, restitution {:.5}, energy rebound {:.4} / vibration {:.4} / heat {:.4}, ledger {:.1e}",
                free.separation / (2.0 * length),
                free.restitution,
                damped.separation / (2.0 * length),
                damped.restitution,
                damped.split.0,
                damped.split.1,
                damped.split.2,
                damped.ledger
            );
            assert!(
                damped.ledger < 1e-12,
                "{f1} Hz, {modes} modes: ledger {:.2e}",
                damped.ledger
            );
            if modes == 1024 {
                assert!(
                    (free.separation / (2.0 * length) - 1.0).abs() < 1e-3
                        && (free.restitution - 1.0).abs() < 1e-3,
                    "{f1} Hz: the elastic rod leaves at {:.5} of 2L/c with {:.5} of its momentum",
                    free.separation / (2.0 * length),
                    free.restitution
                );
            }
        }
    }
}

// A solid with relaxation branches on the total Lagrangian path, measured
// against the exact 1D generalised Zener solution (`common/zener.rs`).
//
// The law: the corotated solid with a measured relaxation spectrum on its
// shear and volume parts (`materials::relaxation`), advanced by Liu and
// Guan's energy-consistent update (`total_lagrangian::relaxation`).
//
// Criteria fixed before measuring:
// (a) a free-free bar with `nu = 0` (`lambda = 0`: exactly uniaxial, only the
// shear part relaxing) released in its first axial mode, below the
// record's strain limit: its damped frequency and logarithmic decrement
// within 5 % of the exact Zener mode's at the bar's own elastic
// frequency (the same bar measured without branches), both read the
// same way from frame samples;
// (b) every frame's energy (kinetic, stored, the branches' included) plus
// the heat made equals the start's to 1e-8 of it, and the branches'
// heat is positive;
// (c) no particle-step beyond the record's strain limit (the run stays in
// its linear range), and none reported when the branches are absent.
//
// Records: Plasticine (Crandall et al. 1970, three mechanisms,
// proportional) and a synthetic one, test data and not a material, with
// its P-wave weights twice its shear weights.
const DX_M: f32 = 0.01;
const E_PA: f32 = 1.0e5;
const RHO: f32 = 1000.0;
const LENGTH_CELLS: i32 = 32;
const FRAME_S: f32 = 2.0e-3;
/// The release speed, m/s: a peak axial strain `v / c` of 5e-6, below
/// Plasticine's linear limit of 1e-5.
const SPEED_M_S: f32 = 5.0e-5;

/// The synthetic record: Plasticine's frequencies, its weights on shear and
/// twice them on P waves. Test data, not a material.
fn synthetic() -> RelaxationSpectrum {
    let plasticine = RelaxationSpectrum::plasticine_crandall_1970();
    let mechanisms: Vec<Mechanism> = plasticine
        .mechanisms()
        .iter()
        .map(|m| Mechanism {
            theta: m.theta,
            kappa_shear: m.kappa_shear,
            kappa_p: 2.0 * m.kappa_shear,
        })
        .collect();
    RelaxationSpectrum::new(&mechanisms, plasticine.strain_limit).expect("positive")
}

/// A run's energy ledger, frame by frame.
///
/// - `worst`: the largest `|balance|` over the start energy (the
///   Kelvin-Voigt gate's criterion, 1e-8, kept);
/// - `tightest`: the largest `|balance|` over what the solves allow, the
///   path suite's derived bound (`tests/total_lagrangian_path.rs`): the
///   frames' residual work and contact work bounds accumulated, plus the
///   energy sums' round-off `n eps (sum |parts| + |heat| + |start|)`. At or
///   below 1, the balance is the solves' own residual;
/// - the branches' heat, and the particle-steps beyond the record's limit.
#[derive(Default)]
struct Ledger {
    allowed: f64,
    worst: f64,
    worst_absolute: f64,
    tightest: f64,
    start: f64,
    heat: f64,
    beyond: u32,
    strain: f64,
}

impl Ledger {
    fn frame(&mut self, sim: &Simulation) {
        let f = sim.total_lagrangian_frame().expect("the path is on");
        assert!(f.energy.is_finite());
        self.allowed += f.residual_work + f.contact_work_bound;
        let round_off = sim.particles().len() as f64
            * f64::EPSILON
            * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
        self.tightest = self
            .tightest
            .max(f.balance.abs() / (self.allowed + round_off));
        self.worst = self.worst.max(f.balance.abs() / f.start_energy.abs());
        self.worst_absolute = self.worst_absolute.max(f.balance.abs());
        self.start = f.start_energy;
        self.heat += f.relaxation_heat;
        self.beyond += f.relaxation_beyond;
        self.strain = self.strain.max(f.relaxation_strain);
    }

    fn summary(&self) -> String {
        format!(
            "balance at most {:.2e} J/m ({:.2e} of the start's {:.3e} J/m, {:.3e} of the solves' bound), branches' heat {:.3e} J/m, largest strain {:.2e}, beyond the limit {}",
            self.worst_absolute,
            self.worst,
            self.start,
            self.tightest,
            self.heat,
            self.strain,
            self.beyond
        )
    }

    /// The ledger's criteria: the balance within 1e-8 of the start energy
    /// and within the solves' bound, heat made, nothing beyond the record's
    /// limit.
    fn check(&self, name: &str) {
        assert!(self.worst < 1e-8, "{name}: energy balance {:e}", self.worst);
        assert!(
            self.tightest <= 1.0,
            "{name}: balance {:e} of the solves' bound",
            self.tightest
        );
        assert!(self.heat > 0.0, "{name}: no heat");
        assert_eq!(self.beyond, 0, "{name}: beyond the record's strain limit");
    }
}

/// What a ringing run measured: the modal amplitude per frame and its
/// ledger.
struct Ring {
    samples: Vec<f64>,
    ledger: Ledger,
}

/// A rod scene's scale: cell size, Young's modulus, frame and speed (the
/// release or impact speed). The rod itself is 0.32 m by 0.02 m, `nu = 0`,
/// 1000 kg/m^3, in a 0.48 m domain.
#[derive(Clone, Copy)]
struct Rod {
    dx: f32,
    e_pa: f32,
    frame_s: f32,
    speed_m_s: f32,
    /// The step controller's tolerance.
    tolerance: f64,
}

/// The ringing gate's bar: quarter-wave 7.81 Hz, 1 cm cells.
const BAR: Rod = Rod {
    dx: DX_M,
    e_pa: E_PA,
    frame_s: FRAME_S,
    speed_m_s: SPEED_M_S,
    tolerance: 1e-2,
};

impl Rod {
    fn cells(&self, metres: f32) -> i32 {
        (metres / self.dx).round() as i32
    }

    fn config(&self) -> SimConfig {
        let mut config = SimConfig::earth(self.cells(0.48) as usize, self.dx, self.frame_s);
        config.gravity = Vec2::ZERO;
        config.min_dt = config.min_dt.min(1e-9);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(self.tolerance));
        config
    }

    fn props(&self) -> Elastic {
        Elastic {
            e_pa: self.e_pa,
            nu: 0.0,
            rho_kg_m3: RHO,
        }
    }

    /// The bar's own elastic speed of sound, m/s, from its free-free mode.
    fn sound_speed(&self) -> f64 {
        let free = ring_bar_at(self, None, 100);
        let (period, _) = period_and_decrement(&free.samples, f64::from(self.frame_s));
        2.0 * f64::from(self.cells(0.32) as f32 * self.dx) / period
    }
}

/// The free-free bar released in its first mode, `frames` frames.
fn ring_bar(relaxation: Option<RelaxationSpectrum>, frames: usize) -> Ring {
    ring_bar_at(&BAR, relaxation, frames)
}

fn ring_bar_at(rod: &Rod, relaxation: Option<RelaxationSpectrum>, frames: usize) -> Ring {
    let props = rod.props();
    let config = rod.config();
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let x0 = rod.cells(0.08) as f32;
    let l = rod.cells(0.32) as f32;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(rod.cells(0.32), rod.cells(0.02)))
            .at(Vec2::new(x0 + 0.5 * l, rod.cells(0.24) as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    let speed = rod.speed_m_s / rod.dx;
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        *v = Vec2::new(speed * (std::f32::consts::PI * (x.x - x0) / l).cos(), 0.0);
    }
    let amplitude = move |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let shape = f64::from((std::f32::consts::PI * (p.x[i].x - x0) / l).cos());
            num += f64::from(p.mass[i]) * f64::from(p.v[i].x) * shape;
            den += f64::from(p.mass[i]) * shape * shape;
        }
        num / den
    };
    run(sim, amplitude, frames)
}

/// The damped period and the logarithmic decrement per period from samples
/// of an oscillating signal every `dt`: the peaks of `|signal|` (a parabola
/// through the three samples around each), every half period; over the
/// first five, two periods.
fn period_and_decrement(samples: &[f64], dt: f64) -> (f64, f64) {
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for k in 1..samples.len() - 1 {
        let (a, b, c) = (samples[k - 1].abs(), samples[k].abs(), samples[k + 1].abs());
        if b >= a && b > c {
            let curvature = a - 2.0 * b + c;
            let shift = if curvature != 0.0 {
                0.5 * (a - c) / curvature
            } else {
                0.0
            };
            peaks.push(((k as f64 + shift) * dt, b - 0.25 * (a - c) * shift));
        }
    }
    assert!(peaks.len() >= 5, "too few peaks: {peaks:?}");
    let ((t0, a0), (t4, a4)) = (peaks[0], peaks[4]);
    ((t4 - t0) / 2.0, (a0 / a4).ln() / 2.0)
}

/// The exact Zener mode of relaxed angular frequency `omega` for `record`'s
/// weights `kappa` (the shear's or the P wave's), sampled like the runs.
fn reference(
    record: &RelaxationSpectrum,
    kappa: impl Fn(&Mechanism) -> f64,
    omega: f64,
    frames: usize,
) -> (f64, f64) {
    let material = zener::Zener {
        rho: 1.0,
        relaxed: 1.0,
        kappa: record.mechanisms().iter().map(&kappa).collect(),
        theta: record.mechanisms().iter().map(|m| m.theta).collect(),
    };
    let start = zener::ModeState {
        q: 0.0,
        rate: 1.0,
        memory: vec![0.0; record.mechanisms().len()],
    };
    let mode = zener::ModeSolution::new(&material, omega, &start);
    let (root, reproduced) = mode.checks();
    assert!(
        root < 1e-12 && reproduced < 1e-12,
        "reference: roots {root:e}, start {reproduced:e}"
    );
    let dt = f64::from(FRAME_S);
    let samples: Vec<f64> = (0..=frames)
        .map(|i| mode.at(dt * i as f64).0.rate)
        .collect();
    period_and_decrement(&samples, dt)
}

#[test]
fn a_zener_bar_rings_down_at_the_exact_rate_and_its_energy_becomes_heat() {
    const FRAMES: usize = 100;
    let dt = f64::from(FRAME_S);
    let free = ring_bar(None, FRAMES);
    let (period, elastic_decrement) = period_and_decrement(&free.samples, dt);
    let omega = 2.0 * PI / period;
    assert_eq!(free.ledger.beyond, 0, "no branches, none reported");
    println!(
        "elastic bar: period {period:.5} s (omega {omega:.3} rad/s), decrement {elastic_decrement:.2e}; {}",
        free.ledger.summary()
    );
    // Plasticine only: with `lambda = 0` the law has no volume part, so a
    // record whose P-wave relaxation differs from its shear's is refused on
    // it (`materials::relaxation`'s tests); the synthetic record is measured
    // where the volume part exists (the uniaxial-strain and shear gates).
    let (name, record) = ("Plasticine", RelaxationSpectrum::plasticine_crandall_1970());
    let damped = ring_bar(Some(record), FRAMES);
    let (period_d, decrement) = period_and_decrement(&damped.samples, dt);
    // With `lambda = 0` only the shear part stores energy: the bar
    // relaxes with the shear weights whatever the P ones.
    let (period_ref, decrement_ref) = reference(&record, |m| m.kappa_shear, omega, FRAMES);
    println!(
        "{name}: damped period {period_d:.5} s against the exact {period_ref:.5} (ratio {:.4}), decrement {decrement:.5} against {decrement_ref:.5} (ratio {:.4}); {}",
        period_d / period_ref,
        decrement / decrement_ref,
        damped.ledger.summary()
    );
    damped.ledger.check(name);
    assert!(
        (period_d / period_ref - 1.0).abs() < 0.05,
        "{name}: period {period_d} against {period_ref}"
    );
    assert!(
        (decrement / decrement_ref - 1.0).abs() < 0.05,
        "{name}: decrement {decrement} against {decrement_ref}"
    );
}

/// Runs `frames` frames of `sim`, sampling `amplitude` after each.
fn run(mut sim: Simulation, amplitude: impl Fn(&Simulation) -> f64, frames: usize) -> Ring {
    let mut ring = Ring {
        samples: vec![amplitude(&sim)],
        ledger: Ledger::default(),
    };
    for _ in 0..frames {
        sim.step();
        ring.ledger.frame(&sim);
        ring.samples.push(amplitude(&sim));
    }
    ring
}

/// The pure-shear scene: a free square of side `SQUARE_CELLS` in zero
/// gravity, `nu = 0.3`, released in its Lame mode, `v = V (cos kx sin ky,
/// -sin kx cos ky)`, `k = pi / L` from its corner. That field is exactly
/// isochoric (`div v = 0`) and leaves every edge free of traction
/// (`sigma_xx = 0` at `x = 0, L`, `sigma_yy = 0` at `y = 0, L`, `sigma_xy =
/// 0` everywhere), so it is a mode at `omega = sqrt(2) pi c_s / L` for any
/// `lambda`, loading the shear part alone.
const SQUARE_CELLS: i32 = 8;
/// The release speed, m/s: a peak shear strain `V / (sqrt(2) c_s)` of
/// about 5e-6 (`c_s` 6.2 m/s), below the record's limit.
const SQUARE_SPEED_M_S: f32 = 4.0e-5;

fn ring_square(relaxation: Option<RelaxationSpectrum>, frames: usize) -> Ring {
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.3,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(24, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let l = SQUARE_CELLS as f32;
    let corner = Vec2::splat(12.0 - 0.5 * l);
    let _square = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::splat(SQUARE_CELLS))
            .at(corner + Vec2::splat(0.5 * l))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let k = std::f32::consts::PI / l;
    let shape = move |x: Vec2| -> Vec2 {
        let q = x - corner;
        Vec2::new(
            (k * q.x).cos() * (k * q.y).sin(),
            -(k * q.x).sin() * (k * q.y).cos(),
        )
    };
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        *v = shape(x) * (SQUARE_SPEED_M_S / DX_M);
    }
    let amplitude = move |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let s = shape(p.x[i]);
            num += f64::from(p.mass[i]) * f64::from(p.v[i].dot(s));
            den += f64::from(p.mass[i]) * f64::from(s.dot(s));
        }
        num / den
    };
    run(sim, amplitude, frames)
}

/// The uniaxial-strain scene: a slab spanning the domain's frictionless
/// side walls, on its floor, `nu = 0.3`, set down unstrained under gravity.
/// The walls hold its sides (its weight keeps them pressed, `sigma_xx =
/// nu / (1 - nu) sigma_yy`), so it rings in its fixed-free P-wave mode,
/// `omega = pi c_p / (2 H)`, `c_p^2 = (lambda + 2 mu) / rho`, about its
/// settled compression.
const SLAB_GRID: usize = 16;
/// Stiff enough that its weight strains it by 2.4e-6 (peak about twice
/// that while it rings), below the record's limit.
const SLAB_E_PA: f32 = 3.0e8;
/// A twentieth of the mode's period (0.63 ms at 1.6 kHz).
const SLAB_FRAME_S: f32 = 3.0e-5;

fn ring_slab(relaxation: Option<RelaxationSpectrum>, frames: usize) -> (Ring, f64) {
    let props = Elastic {
        e_pa: SLAB_E_PA,
        nu: 0.3,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(SLAB_GRID, DX_M, SLAB_FRAME_S);
    config.min_dt = config.min_dt.min(1e-9);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let wall = config.boundary_thickness as f32;
    let width = SLAB_GRID as f32 - 2.0 * wall;
    let cells = width.round() as i32;
    let _slab = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::splat(cells))
            .at(Vec2::splat(wall + 0.5 * cells as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let h = cells as f32;
    let shape = move |x: Vec2| -> f32 { (std::f32::consts::PI * (x.y - wall) / (2.0 * h)).sin() };
    let amplitude = move |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let s = f64::from(shape(p.x[i]));
            num += f64::from(p.mass[i]) * f64::from(p.v[i].y) * s;
            den += f64::from(p.mass[i]) * s * s;
        }
        num / den
    };
    (run(sim, amplitude, frames), f64::from(h * DX_M))
}

/// The exact mode released from rest at its static offset's opposite, the
/// step response under a constant load: `q = q_p + q_h` with `q_p` the
/// relaxed static offset (no memory at rest), so the velocity is the
/// homogeneous part's from `q_h = -q_p`, at rest, no memory.
fn reference_step(
    record: &RelaxationSpectrum,
    kappa: impl Fn(&Mechanism) -> f64,
    omega: f64,
    frames: usize,
    dt: f64,
) -> (f64, f64) {
    let material = zener::Zener {
        rho: 1.0,
        relaxed: 1.0,
        kappa: record.mechanisms().iter().map(&kappa).collect(),
        theta: record.mechanisms().iter().map(|m| m.theta).collect(),
    };
    let start = zener::ModeState {
        q: -1.0,
        rate: 0.0,
        memory: vec![0.0; record.mechanisms().len()],
    };
    let mode = zener::ModeSolution::new(&material, omega, &start);
    let (root, reproduced) = mode.checks();
    assert!(
        root < 1e-12 && reproduced < 1e-12,
        "reference: roots {root:e}, start {reproduced:e}"
    );
    let samples: Vec<f64> = (0..=frames)
        .map(|i| mode.at(dt * i as f64).0.rate)
        .collect();
    period_and_decrement(&samples, dt)
}

/// One gate's comparison: the damped run's period and decrement against
/// the exact mode's, with the ledger's criteria.
fn compare(name: &str, damped: &Ring, dt: f64, (period_ref, decrement_ref): (f64, f64)) {
    let (period, decrement) = period_and_decrement(&damped.samples, dt);
    println!(
        "{name}: damped period {period:.6e} s against the exact {period_ref:.6e} (ratio {:.4}), decrement {decrement:.5} against {decrement_ref:.5} (ratio {:.4}); {}",
        period / period_ref,
        decrement / decrement_ref,
        damped.ledger.summary()
    );
    damped.ledger.check(name);
    assert!(
        (period / period_ref - 1.0).abs() < 0.05,
        "{name}: period {period} against {period_ref}"
    );
    assert!(
        (decrement / decrement_ref - 1.0).abs() < 0.05,
        "{name}: decrement {decrement} against {decrement_ref}"
    );
}

#[test]
fn a_zener_square_rings_down_in_pure_shear_at_the_exact_shear_rate() {
    const FRAMES: usize = 100;
    let dt = f64::from(FRAME_S);
    let free = ring_square(None, FRAMES);
    let (period, elastic_decrement) = period_and_decrement(&free.samples, dt);
    let omega = 2.0 * PI / period;
    let c_s = (f64::from(E_PA) / (2.0 * 1.3) / f64::from(RHO)).sqrt();
    println!(
        "elastic square: period {period:.5} s (omega {omega:.3} rad/s, the Lame mode's sqrt(2) pi c_s / L {:.3}), decrement {elastic_decrement:.2e}; {}",
        2f64.sqrt() * PI * c_s / (f64::from(SQUARE_CELLS) * f64::from(DX_M)),
        free.ledger.summary()
    );
    for (name, record) in [
        (
            "square, Plasticine",
            RelaxationSpectrum::plasticine_crandall_1970(),
        ),
        (
            "square, synthetic (P weights twice the shear's)",
            synthetic(),
        ),
    ] {
        let damped = ring_square(Some(record), FRAMES);
        compare(
            name,
            &damped,
            dt,
            reference(&record, |m| m.kappa_shear, omega, FRAMES),
        );
    }
}

#[test]
fn a_confined_zener_slab_rings_down_in_uniaxial_strain_at_the_exact_p_wave_rate() {
    const FRAMES: usize = 90;
    let dt = f64::from(SLAB_FRAME_S);
    let (free, h) = ring_slab(None, FRAMES);
    let (period, elastic_decrement) = period_and_decrement(&free.samples, dt);
    let omega = 2.0 * PI / period;
    let nu = 0.3f64;
    let m = f64::from(SLAB_E_PA) * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    println!(
        "elastic slab: period {period:.6e} s (omega {omega:.1} rad/s, the fixed-free P mode's pi c_p / 2H {:.1}), decrement {elastic_decrement:.2e}; {}",
        PI * (m / f64::from(RHO)).sqrt() / (2.0 * h),
        free.ledger.summary()
    );
    for (name, record) in [
        (
            "slab, Plasticine",
            RelaxationSpectrum::plasticine_crandall_1970(),
        ),
        ("slab, synthetic (P weights twice the shear's)", synthetic()),
    ] {
        let (damped, _) = ring_slab(Some(record), FRAMES);
        // The P-wave modulus relaxes with the record's P weights, its parts'
        // weights `(pi kappa^p - 2 mu kappa^s) / lambda` and `kappa^s` summed.
        compare(
            name,
            &damped,
            dt,
            reference_step(&record, |m| m.kappa_p, omega, FRAMES, dt),
        );
    }
}

/// The impact scene: the `nu = 0` rod standing on its end, in zero
/// gravity, moving at the scene's speed onto the floor from a gap of 1e-4
/// cell. Gravity is left out: over a slow rod's contact (64 ms at 7.81 Hz)
/// its impulse would dwarf an impact slow enough for the record's linear
/// range.
struct Drop {
    /// The centre of mass's speed over the impact's, just after the first
    /// separation and at the run's end.
    first: f64,
    end: f64,
    /// At the first separation, over the start (all kinetic) energy: the
    /// rigid rebound (`first^2`), the rod's vibration (its kinetic and
    /// stored energy beyond the rebound's) and the heat made.
    split: (f64, f64, f64),
    /// The kept steps over the run.
    steps: u32,
    ledger: Ledger,
}

fn drop_rod_at(rod: &Rod, relaxation: Option<RelaxationSpectrum>, frames: usize) -> Drop {
    let props = rod.props();
    let config = rod.config();
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let l = rod.cells(0.32) as f32;
    let _rod = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(rod.cells(0.02), rod.cells(0.32)))
            .at(Vec2::new(rod.cells(0.24) as f32, floor + 1e-4 + 0.5 * l))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let impact = rod.speed_m_s / rod.dx;
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(0.0, -impact);
    }
    let speed = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
        p.v.iter()
            .zip(&p.mass)
            .map(|(v, &m)| f64::from(v.y) * f64::from(m))
            .sum::<f64>()
            / mass
            / f64::from(impact)
    };
    let mut ledger = Ledger::default();
    let (mut touched, mut first, mut steps) = (false, None, 0u32);
    for _ in 0..frames {
        sim.step();
        ledger.frame(&sim);
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        steps += frame.steps;
        if frame.contacts > 0 {
            touched = true;
        } else if touched && first.is_none() {
            let e = speed(&sim);
            let start = frame.start_energy;
            first = Some((
                e,
                (
                    e * e,
                    (frame.parts[0] + frame.parts[1]) / start - e * e,
                    frame.heat / start,
                ),
            ));
        }
    }
    let (first, split) = first.expect("the rod leaves the floor within the run");
    Drop {
        first,
        end: speed(&sim),
        split,
        steps,
        ledger,
    }
}

/// The exact Zener rod on a rigid wall (`common/zener.rs`, 1024 modes) at a
/// rod's own elastic speed of sound `c`.
fn exact_rod(record: &RelaxationSpectrum, rod: &Rod, c: f64) -> zener::RodContact {
    zener::rod_on_wall(
        &zener::Zener {
            rho: f64::from(RHO),
            relaxed: f64::from(RHO) * c * c,
            kappa: record.mechanisms().iter().map(|m| m.kappa_shear).collect(),
            theta: record.mechanisms().iter().map(|m| m.theta).collect(),
        },
        f64::from(rod.cells(0.32) as f32 * rod.dx),
        f64::from(rod.speed_m_s),
        1024,
    )
}

/// One rod at one scale: its elastic and damped drops against the exact
/// restitution and energy split, printed; returns (damped restitution over
/// the exact, the damped drop).
fn rod_row(name: &str, rod: &Rod, frames: usize) -> (f64, Drop) {
    let c = rod.sound_speed();
    let elastic = drop_rod_at(rod, None, frames);
    let record = RelaxationSpectrum::plasticine_crandall_1970();
    let damped = drop_rod_at(rod, Some(record), frames);
    let exact = exact_rod(&record, rod, c);
    let length = f64::from(rod.cells(0.32) as f32 * rod.dx);
    println!(
        "{name} (cell {} m, quarter-wave {:.2} Hz, c {c:.4} m/s): elastic leaves with {:.5} (end {:.5}); Plasticine leaves with {:.5} (end {:.5}) against the exact {:.5} (ratio {:.4}); split rebound / vibration / heat {:.4} / {:.4} / {:.4} against the exact {:.4} / {:.4} / {:.4}; damped {}",
        rod.dx,
        c / (4.0 * length),
        elastic.first,
        elastic.end,
        damped.first,
        damped.end,
        exact.restitution,
        damped.first / exact.restitution,
        damped.split.0,
        damped.split.1,
        damped.split.2,
        exact.split.0,
        exact.split.1,
        exact.split.2,
        damped.ledger.summary()
    );
    (damped.first / exact.restitution, damped)
}

/// Below the record's band: the rod's quarter-wave at 7.81 Hz is twelve
/// times below the 100-3000 Hz its spectrum was fitted over. This gate
/// tests the discretisation against the same record, not the record's
/// physics in its band (the in-band gate below). Criterion fixed before
/// measuring: within 5 % at 1 cm cells; the half cell reported as the
/// convergence check.
///
/// Measured (step tolerance 1e-2):
/// - 1 cm cells: elastic 0.97858, damped 0.84177 against the exact
///   0.83858 (1.0038);
/// - 0.5 cm cells: elastic 0.98496, damped 0.84211 (1.0042).
/// The elastic loss shrinks with the cell; the damped excess does not move
/// with it.
///
/// With the step tolerance instead (`probe_rod_restitution_against_the_step_tolerance`,
/// 1 cm cells), the damped excess falls toward zero: 1.0038, 1.0025, 1.0014 at
/// 1e-2, 1e-3, 1e-4. The elastic rod is not monotone there: 0.97858, 0.98624,
/// 0.98563.
///
/// Declared gap: the energy split is read at the end of the frame after the
/// separation, the exact one at the separation itself, and the branches keep
/// turning vibration into heat in flight, so the two splits are not
/// comparable. Only the rebound (`e^2`) is. Removal criterion: the exact
/// solution followed into flight, sampled at the run's own times.
#[test]
fn a_zener_rod_dropped_on_its_end_leaves_with_the_exact_restitution() {
    const FRAMES: usize = 120;
    let (ratio, damped) = rod_row("rod below the band", &BAR, FRAMES);
    damped.ledger.check("rod below the band");
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "restitution {ratio} of the exact"
    );
    let half = Rod {
        dx: 0.5 * DX_M,
        ..BAR
    };
    let (half_ratio, half_damped) = rod_row("rod below the band, half cell", &half, FRAMES);
    half_damped.ledger.check("rod below the band, half cell");
    println!(
        "convergence: restitution over the exact {ratio:.4} at 1 cm, {half_ratio:.4} at 0.5 cm"
    );
}

/// In the record's band: the rod's quarter-wave near 530 Hz (`E` 4.6e8 Pa,
/// `c` near 678 m/s), the E2 block's own band, against the exact
/// restitution 0.64039 of the reference's table, at 1 cm and 0.5 cm cells.
/// Criterion fixed before measuring: the restitution within 5 % at the
/// finest cell run. The energy split is reported, under the gap declared
/// above (only the rebound `e^2` is comparable).
///
/// Measured: 0.64444 (1 cm, 1.0063) and 0.64379 (0.5 cm, 1.0053); the
/// rebound fraction 0.4145 against the exact 0.4102.
#[test]
fn a_zener_rod_in_its_records_band_leaves_with_the_exact_restitution() {
    const FRAMES: usize = 40;
    let coarse = Rod {
        dx: DX_M,
        e_pa: 4.6e8,
        frame_s: 5.0e-5,
        speed_m_s: 4.0e-3,
        tolerance: 1e-2,
    };
    let (ratio, damped) = rod_row("rod in the band", &coarse, FRAMES);
    damped.ledger.check("rod in the band");
    let half = Rod {
        dx: 0.5 * DX_M,
        ..coarse
    };
    let (half_ratio, half_damped) = rod_row("rod in the band, half cell", &half, FRAMES);
    half_damped.ledger.check("rod in the band, half cell");
    println!("in band: restitution over the exact {ratio:.4} at 1 cm, {half_ratio:.4} at 0.5 cm");
    assert!(
        (half_ratio - 1.0).abs() < 0.05,
        "restitution {half_ratio} of the exact at 0.5 cm"
    );
}

/// The validity ledger reports, never hides: the bar rung at a strain of
/// 1e-3, a hundred times the record's linear limit, counts particle-steps
/// beyond it.
#[test]
fn branches_applied_beyond_their_records_strain_limit_are_reported() {
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.0,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(48, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(LENGTH_CELLS, 2))
            .at(Vec2::new(24.0, 24.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        // 1 cm/s in the first mode's shape: a peak strain of 1e-3.
        let shape = (std::f32::consts::PI * (x.x - 8.0) / LENGTH_CELLS as f32).cos();
        *v = Vec2::new(shape * 0.01 / DX_M, 0.0);
    }
    let mut beyond = 0;
    let mut strain = 0.0f64;
    for _ in 0..10 {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        beyond += frame.relaxation_beyond;
        strain = strain.max(frame.relaxation_strain);
    }
    println!(
        "rung at 1 cm/s: largest strain {strain:.2e}, particle-steps beyond the limit {beyond}"
    );
    assert!(
        strain > 1e-5 && beyond > 0,
        "strain {strain:e}, beyond {beyond}"
    );
}

/// One body of two materials: the ringing bar's right half carries
/// Plasticine's record, its left half none. With modal flight asked for, a
/// free body: its modes refuse the branches, so every frame is stepped and
/// counted as refused; the ledger closes and only the damped half heats.
#[test]
fn a_body_of_two_materials_is_stepped_and_its_damped_half_alone_heats() {
    const FRAMES: usize = 30;
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.0,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(48, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2).with_modal_flight());
    let plain = CorotatedMaterial::from_physical(&props, &config);
    let mut damped = plain;
    damped.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
    let mut sim = Simulation::empty(config);
    let plain_id = sim.register_material(Box::new(plain)).0;
    let damped_id = sim.register_material(Box::new(damped)).0;
    let x0 = 8.0f32;
    let l = LENGTH_CELLS as f32;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(LENGTH_CELLS, 2))
            .at(Vec2::new(x0 + 0.5 * l, 24.0))
            .spacing(0.5)
            .material(plain_id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    for (i, x) in positions.iter().enumerate() {
        if x.x > x0 + 0.5 * l {
            sim.particles_mut().material_id[i] = damped_id;
        }
        sim.particles_mut().v[i] = Vec2::new(
            (SPEED_M_S / DX_M) * (std::f32::consts::PI * (x.x - x0) / l).cos(),
            0.0,
        );
    }
    let (mut ledger, mut flown) = (Ledger::default(), 0u32);
    for _ in 0..FRAMES {
        sim.step();
        ledger.frame(&sim);
        flown += u32::from(sim.total_lagrangian_frame().expect("on").modal);
    }
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let heat = ledger.heat;
    println!(
        "two materials: {FRAMES} frames, flown {flown}, refused flights {}, total heat {:.3e} J/m; {}",
        frame.modal_refused,
        frame.heat,
        ledger.summary()
    );
    assert_eq!(flown, 0, "a body with branches never flies");
    // The first frame is stepped before a flight can be asked (the free
    // body is told by the contact cache its first step builds): every later
    // frame asks, and is refused and counted.
    assert_eq!(
        frame.modal_refused,
        FRAMES as u64 - 1,
        "each refused flight counted"
    );
    ledger.check("two materials");
    assert!((frame.heat - heat).abs() <= 1e-12 * heat.max(frame.heat));
}

/// The cost of the branches (report only, criteria none): Newton
/// iterations, GMRES products and wall time per frame, with and without
/// Plasticine's spectrum, on E2's drop (the clay block dropped 10 cells,
/// 45 frames) and on the settled slab's scene (30 frames). The spectrum on
/// these scenes' moduli is a cost measurement, not a material: the record's
/// relaxed moduli are not set (its declared gap). Wall times mean something
/// only on a quiet machine.
#[test]
#[ignore = "cost probe: a quiet machine, run with --ignored --nocapture"]
fn probe_relaxation_cost() {
    let measure = |name: &str, mut sim: Simulation, frames: usize| {
        let (mut newton, mut gmres, mut steps, mut rejected, mut wall) =
            (0u64, 0u64, 0u64, 0u64, 0.0f64);
        let mut slowest = 0.0f64;
        for _ in 0..frames {
            sim.step();
            let f = sim.total_lagrangian_frame().expect("the path is on");
            newton += u64::from(f.newton);
            gmres += u64::from(f.gmres);
            steps += u64::from(f.steps);
            rejected += u64::from(f.rejected);
            wall += f.wall;
            slowest = slowest.max(f.wall);
        }
        let n = frames as f64;
        println!(
            "| {name} | {:.1} | {:.0} | {:.1} | {:.1} | {:.1} | {:.1} |",
            newton as f64 / n,
            gmres as f64 / n,
            steps as f64 / n,
            rejected as f64 / n,
            1e3 * wall / n,
            1e3 * slowest
        );
    };
    println!(
        "| scene | Newton/frame | GMRES/frame | steps/frame | rejected/frame | ms/frame | slowest ms |"
    );
    println!("|---|---|---|---|---|---|---|");
    let clay = Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    for relaxed in [false, true] {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let mut material = CorotatedMaterial::from_physical(&clay, &config);
        if relaxed {
            material.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
        }
        let mut sim = Simulation::empty(config);
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&clay, &config),
        );
        measure(
            if relaxed {
                "E2 drop, Plasticine spectrum"
            } else {
                "E2 drop"
            },
            sim,
            45,
        );
    }
    let slab = Elastic {
        e_pa: 1.0e5,
        nu: 0.0,
        rho_kg_m3: 1000.0,
    };
    for relaxed in [false, true] {
        let dx = 0.01f32;
        let mut config = SimConfig::earth(48, dx, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let mut material = CorotatedMaterial::from_physical(&slab, &config);
        if relaxed {
            material.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
        }
        let mut sim = Simulation::empty(config);
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _slab = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(30, 12))
                .at(Vec2::new(24.0, floor + 6.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&slab, &config),
        );
        measure(
            if relaxed {
                "settled slab, Plasticine spectrum"
            } else {
                "settled slab"
            },
            sim,
            30,
        );
    }
}

/// Whether the damped rod's small excess restitution over the exact one
/// (+0.4 % below the band at both 1 cm and 0.5 cm cells, while the elastic
/// rod's grid loss halves) comes from the time step: the same rod, 1 cm
/// cells, at step tolerances 1e-2, 1e-3 and 1e-4, elastic and damped.
/// Report only.
///
/// Measured:
/// - damped over exact 1.0038, 1.0025, 1.0014 (125, 166, 345 steps);
/// - elastic 0.97858, 0.98624, 0.98563 (243, 2628, 3859 steps), not
///   monotone.
/// The damped excess is the step's, not the cell's.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_rod_restitution_against_the_step_tolerance() {
    const FRAMES: usize = 120;
    let c = BAR.sound_speed();
    let record = RelaxationSpectrum::plasticine_crandall_1970();
    let exact = exact_rod(&record, &BAR, c);
    for tolerance in [1e-2, 1e-3, 1e-4] {
        let rod = Rod { tolerance, ..BAR };
        let elastic = drop_rod_at(&rod, None, FRAMES);
        let damped = drop_rod_at(&rod, Some(record), FRAMES);
        println!(
            "tolerance {tolerance:.0e}: elastic {:.5} ({} steps), damped {:.5} ({} steps), exact {:.5}, damped over exact {:.4}; damped {}",
            elastic.first,
            elastic.steps,
            damped.first,
            damped.steps,
            exact.restitution,
            damped.first / exact.restitution,
            damped.ledger.summary()
        );
    }
}

/// The Plasticine block: a scene built from one record whole
/// (`ViscoelasticRecord::plasticine_crandall_1970_fresh`: its relaxed
/// moduli, Poisson's ratio, density and spectrum), the E2 drop's geometry
/// (8 by 4 cm, 10 cells above the floor, 1 cm cells, 1/60 s frames).
/// Report only: its impact strains are far beyond the record's linear limit
/// (counted, `relaxation_beyond`), and no plastic law is paired with the
/// record (none read for this material), so the impact is outside what the
/// record covers.
#[test]
#[ignore = "scene probe: run with --ignored --nocapture"]
fn probe_plasticine_block_drop() {
    let record =
        emerge::materials::relaxation::ViscoelasticRecord::plasticine_crandall_1970_fresh();
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
    material.relaxation = Some(record.spectrum);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 12.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&record.relaxed, &config),
    );
    println!(
        "Plasticine block ({}): E_r {:.3e} Pa, nu {}, rho {} kg/m^3",
        record.state, record.relaxed.e_pa, record.relaxed.nu, record.relaxed.rho_kg_m3
    );
    let mut ledger = Ledger::default();
    for k in 1..=45 {
        sim.step();
        ledger.frame(&sim);
        let f = sim.total_lagrangian_frame().expect("the path is on");
        let p = sim.particles();
        let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
        let vy =
            p.v.iter()
                .zip(&p.mass)
                .map(|(v, &m)| f64::from(v.y) * f64::from(m))
                .sum::<f64>()
                / mass
                * f64::from(config.dx_meters);
        println!(
            "  frame {k}: {} steps, {} rejected, contacts {}, centre speed {vy:+.4} m/s, heat {:.4e} J/m, beyond {}, strain {:.2e}, {:.0} ms",
            f.steps,
            f.rejected,
            f.contacts,
            f.heat,
            f.relaxation_beyond,
            f.relaxation_strain,
            1e3 * f.wall
        );
    }
    println!("Plasticine block: {}", ledger.summary());
}
