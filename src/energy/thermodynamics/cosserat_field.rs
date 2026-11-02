//! Cosserat micro-rotation field: the grid-level angular-momentum channel
//! that `matter::materials::granular::cosserat`'s kinematics leave open. Like
//! `GranularFluidityField`, an opt-in, grid-sized scratch field with its own
//! P2G scatter, solve and G2P gather, apart from the `Cell`/momentum
//! machinery, so a scene that never builds a `CosseratField` is unaffected.
//!
//! # What this solves
//! The micropolar (Eringen) / Cosserat rotational balance, scalar in 2D, is
//! second order in the micro-rotation angle. Neff et al. (2022) eq. (3.16) and
//! (3.17), the Eringen form with micro-inertia coefficient `j`:
//! ```text
//! rho * j * d2(theta_c)/dt2 = div(m) + kappa * (curl u - 2 * theta_c)
//! ```
//! where `curl u = 2 * theta_m` is twice the macro rotation (the antisymmetric
//! part of the displacement gradient) and `kappa` is Eringen's coupling
//! modulus in Pa, so the torque is `2 * kappa * (theta_m - theta_c)`. The
//! factor 2 is the relation between two moduli of the SAME paper: Neff et al.
//! (read in full, `tmp/papers/review_cosserat/neff2022.txt`) write the
//! coupling as Eringen's `kappa` in eqs. (3.16)-(3.17) and as the Cosserat
//! couple modulus `mu_c` in their eq. (5.8), `kappa = 2 * mu_c`. The modulus
//! in this module is Eringen's `kappa`, the coefficient of
//! `(curl u - 2 theta_c)`: `coupling_modulus_pa` (`alpha`) is that `kappa`, so
//! pass `alpha = 2 * mu_c` if you hold Neff's couple modulus `mu_c`. (That the
//! de Borst et al. `alpha` is the same quantity is taken from the pre-existing
//! relation in `matter::materials::granular::cosserat` and was not re-read
//! here.) Written for the relative angle
//! `phi = theta_c - theta_m`, with `rho * j` the micro-inertia per unit volume:
//! ```text
//! rho * j * d2(phi)/dt2 = -2 * alpha * phi - eta_r * d(phi)/dt
//!                                  (couple-stress divergence dropped)
//! ```
//! The `eta_r` term is the damping seam, zero by default (see the declared
//! gaps).
//! Dimensions: `rho * j` is kg/m^3 * m^2 = kg/m (2D: per unit depth), `alpha`
//! is Pa = kg/(m s^2), so `Omega^2 = 2 * alpha / (rho * j)` is 1/s^2. The
//! coupling is therefore an elastic restoring torque on the ANGLE, a
//! harmonic oscillator of angular frequency `Omega`, not a first-order
//! relaxation. A first-order relaxation of the rate toward the macro spin would
//! instead need a rotational viscosity in Pa s, a coefficient this module has
//! no source for.
//!
//! On the grid, a node carries the micro-inertia `I_n = sum(w * m * c * d^2)`
//! (kg m^2) and the torque stiffness `K_n = sum(w * 2 * alpha * V)` (N m per
//! rad), scattered from the particles (`V` the particle volume, so `m / V` is
//! the density: no separate density input). `Omega_n^2 = K_n / I_n =
//! 2 * alpha / (rho * c * d^2)`. With the relative rate
//! `w = omega_c - spin` and relative angle `phi`, `phi' = w`, `w' = -Omega^2 phi`
//! has the exact solution used per substep (macro spin held constant over the
//! substep). With the damping rate `g = eta_r / (rho * j)` (1/s, zero by
//! default), `a = g / 2` and `Omega_d^2 = Omega^2 - a^2` (the linear damped
//! oscillator `phi'' + g phi' + Omega^2 phi = 0`):
//! ```text
//! phi(t) = e^(-a t) * (phi0 * cos(Omega_d t) + (w0 + a phi0) * sin(Omega_d t) / Omega_d)
//! w(t)   = e^(-a t) * (w0 * cos(Omega_d t) - (Omega^2 phi0 + a w0) * sin(Omega_d t) / Omega_d)
//! ```
//! (cos and sin become cosh and sinh when `Omega_d^2 < 0`, the overdamped
//! case; for `g = 0` it is the undamped solution
//! `phi0 cos(Omega t) + w0 sin(Omega t)/Omega`.) Derived on the page by
//! substituting `phi = e^(-a t) y`. Exact for the local oscillator, so
//! unconditionally stable;
//! [`CosseratConfig::stability_dt`] is the resolution bound.
//!
//! Grains are treated as rounded for the micro-inertia coefficient (`1/10`,
//! the mass-specific polar moment of a solid sphere of diameter `d`), with
//! the single grain diameter of `GranularFluidityConfig`, not modelling
//! angularity.
//!
//! # Declared gaps (open, with removal criterion)
//! - KNOWN-LIMIT, kind: missing sourced coefficient. No dissipation by
//!   default: the elastic Cosserat coupling alone conserves the
//!   oscillation, so `omega_c` rings about the macro spin at `Omega` and never
//!   settles. When `Omega * dt` is not small (check
//!   `CosseratConfig::coupling_frequency` against the substep) the
//!   oscillation is aliased by the substep sampling and the micro-curvature
//!   (gradient of `omega_c`) is noise rather than a smooth field. No sourced
//!   damping coefficient was found (granular rolling dissipation or a
//!   rotational viscosity `eta_r` in Pa s) in the papers read, so none is
//!   invented. Why it is still open: the law is wired and exact
//!   (`CosseratConfig::rotational_damping_rate_per_s`, the ONE seam, `0.0`
//!   = off) but its value has no source. Replace when: a measured
//!   rotational viscosity or rolling-resistance law for the grain material
//!   is read and converted to `g = eta_r / (rho * j)` there, with its source
//!   in the config doc. `CosseratConfig::coupling_phase_per_substep` reports
//!   the aliasing `Omega * dt` so a scene can check it.
//! - The couple-stress divergence `div(m)` is not in this solve (it was not
//!   before): the field is a local oscillator per node.
//! - The micro-curvature is the gradient of the gathered micro-rotation RATE
//!   field (as before), whereas the cited theory takes the gradient of the
//!   angle. Remove when the absolute angle is carried.
//! - Neff et al. is the 3D small-rotation theory, reduced here to the single
//!   out-of-plane component (adaptation).
//!
//! # Citation
//! Neff, P., Ghiba, I.-D., Madeo, A., Placidi, L., Rosi, G., "Cosserat
//! micropolar elasticity: classical Eringen vs. dislocation form" (2022),
//! arXiv:2206.02473, eqs. (3.16), (3.17), (5.8). Constitutive relation
//! and `alpha`: de Borst, R., Sabet, S.A. and Hageman, T. (2022), "Non-associated
//! Cosserat plasticity", Int. J. Mech. Sci. 230, 107535; see
//! `matter::materials::granular::cosserat`.

use glam::{IVec2, Vec2};

use crate::{
    grid::kernel::quadratic_weights, matter::materials::granular::cosserat::micro_curvature_2d,
};

/// Physical parameters of the Cosserat micro-rotation field.
#[derive(Clone, Copy, Debug)]
pub struct CosseratConfig {
    /// Elastic coupling modulus `alpha` \[Pa\] (see
    /// `matter::materials::granular::cosserat`: `m = alpha * l^2 * kappa`, and
    /// the coupling torque `2*alpha*(omega_macro - omega_c)`).
    pub coupling_modulus_pa: f32,
    /// Grain diameter `l` \[m\], the quantity
    /// `GranularFluidityConfig::grain_diameter_m` uses.
    pub grain_diameter_m: f32,
    /// Dimensionless micro-inertia shape coefficient. `1/10`, a solid
    /// sphere's mass-specific polar moment (`I = (2/5)*r^2 = (1/10)*d^2`), for
    /// rounded grains.
    pub micro_inertia_coefficient: f32,
    /// Damping rate `g = eta_r / (rho * j)` \[1/s\] of the relative
    /// micro-rotation (viscous torque `-eta_r * d(phi)/dt`). `0.0` means no
    /// dissipation. The ONE seam of the missing dissipation law: no sourced
    /// value exists yet (see the module's declared gaps), so every preset sets
    /// `0.0`; replace it when a measured rotational viscosity is read.
    pub rotational_damping_rate_per_s: f32,
}

impl CosseratConfig {
    /// Phase `Omega * dt` \[rad\] the coupling oscillator advances per
    /// substep at a bulk density \[kg/m^3\]. The oscillation is resolved
    /// by the substep only when this is well below `pi`; above it the
    /// sampled `omega_c` is aliased and its gradient (the micro-curvature)
    /// is not a smooth field.
    pub fn coupling_phase_per_substep(&self, density_kg_m3: f32, sub_dt: f32) -> f32 {
        self.coupling_frequency(density_kg_m3) * sub_dt
    }

    /// Micro-inertia of one particle \[kg m^2\]: `m * c * d^2`
    /// (`I = (2/5) r^2 m = d^2 m / 10` for a solid sphere, `d = 2r`).
    pub fn micro_inertia(&self, particle_mass: f32) -> f32 {
        particle_mass
            * self.micro_inertia_coefficient
            * self.grain_diameter_m
            * self.grain_diameter_m
    }

    /// Angular frequency `Omega` \[1/s\] of the elastic coupling at a bulk
    /// density \[kg/m^3\]: `Omega^2 = 2 * alpha / (rho * c * d^2)`. Pa over
    /// (kg/m^3 * m^2) is 1/s^2.
    pub fn coupling_frequency(&self, density_kg_m3: f32) -> f32 {
        let rho_j = density_kg_m3
            * self.micro_inertia_coefficient
            * self.grain_diameter_m
            * self.grain_diameter_m;
        (2.0 * self.coupling_modulus_pa / rho_j.max(f32::MIN_POSITIVE)).sqrt()
    }

    /// Resolution bound \[s\] of the local oscillator, `2 / Omega`. Derived:
    /// central differences on `phi'' = -Omega^2 phi` give
    /// `lambda^2 - (2 - Omega^2 dt^2) lambda + 1 = 0`, whose roots stay on the
    /// unit circle only if `|2 - Omega^2 dt^2| <= 2`, i.e. `dt <= 2 / Omega`.
    /// The solve itself is the exact oscillator solution and does not need it
    /// for stability; it is the step beyond which an explicit scheme of the
    /// same equation would blow up and the oscillation is under-resolved.
    pub fn stability_dt(&self, density_kg_m3: f32) -> f32 {
        2.0 / self
            .coupling_frequency(density_kg_m3)
            .max(f32::MIN_POSITIVE)
    }
}

/// Exact solution after `t` of `phi'' + g phi' + omega2 phi = 0` from
/// `(phi0, w0 = phi'(0))`, returning `(phi(t), phi'(t))`. With `a = g / 2` and
/// `D = omega2 - a^2`: `phi = e^(-a t) (phi0 C + (w0 + a phi0) S)` and
/// `phi' = e^(-a t) (w0 C - (omega2 phi0 + a w0) S)`, where for `D > 0`
/// `C = cos(sqrt(D) t)`, `S = sin(sqrt(D) t) / sqrt(D)`, for `D < 0` the
/// hyperbolic analogues, and for `D = 0` `C = 1`, `S = t`.
fn damped_oscillator(phi0: f32, w0: f32, omega2: f32, g: f32, t: f32) -> (f32, f32) {
    let a = 0.5 * g;
    let d = omega2 - a * a;
    let q = d * t * t; // signed (sqrt(|D|) t)^2
    let (c, s) = if q.abs() < 1.0e-6 {
        (1.0 - 0.5 * q, t * (1.0 - q / 6.0))
    } else if q > 0.0 {
        let r = d.sqrt();
        ((r * t).cos(), (r * t).sin() / r)
    } else {
        let r = (-d).sqrt();
        ((r * t).cosh(), (r * t).sinh() / r)
    };
    let decay = (-a * t).exp();
    (
        decay * (phi0 * c + (w0 + a * phi0) * s),
        decay * (w0 * c - (omega2 * phi0 + a * w0) * s),
    )
}

/// The grid's units against SI for one scene: the cell size
/// (`SimConfig::dx_meters`) and the mass unit per metre of depth
/// (`SimConfig::grid_mass_unit_kg_per_m()`). Engine particles are in these
/// units; the config of this module is in SI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridScale {
    /// A cell, m.
    pub dx_meters: f32,
    /// The mass unit per metre of depth, kg/m.
    pub mass_unit_kg_per_m: f32,
}

impl GridScale {
    /// Particles already in SI (1 m cell, 1 kg/m mass unit).
    pub const SI: Self = Self {
        dx_meters: 1.0,
        mass_unit_kg_per_m: 1.0,
    };
}

/// A persistent, grid-coupled Cosserat micro-rotation field.
pub struct CosseratField {
    pub config: CosseratConfig,
    grid_res: usize,
    grid_mass: Vec<f32>,  // Sigma(w * micro_inertia)         -- cleared each step
    grid_stiff: Vec<f32>, // Sigma(w * 2*alpha*V) torque stiffness -- cleared each step
    grid_l: Vec<f32>,     // Sigma(w * I * omega_c)           -- cleared each step
    grid_spin: Vec<f32>,  // Sigma(w * I * macro spin)        -- cleared each step
    grid_phi: Vec<f32>,   // Sigma(w * I * relative angle)    -- cleared each step
    grid_omega: Vec<f32>, // persistent PDE state -- NOT cleared between steps
    grid_work: Vec<f32>,  // scratch: solve output before it replaces grid_omega
    grid_phi_out: Vec<f32>, // persistent relative angle, gathered back to particles
    /// Relative micro-rotation angle `theta_c - theta_macro` per particle \[rad\].
    rel_angle: Vec<f32>,
}

impl CosseratField {
    pub fn new(config: CosseratConfig, grid_res: usize) -> Self {
        let n = grid_res * grid_res;
        Self {
            config,
            grid_res,
            grid_mass: vec![0.0; n],
            grid_stiff: vec![0.0; n],
            grid_l: vec![0.0; n],
            grid_spin: vec![0.0; n],
            grid_phi: vec![0.0; n],
            grid_omega: vec![0.0; n],
            grid_work: vec![0.0; n],
            grid_phi_out: vec![0.0; n],
            rel_angle: Vec::new(),
        }
    }

    pub const fn grid_res(&self) -> usize {
        self.grid_res
    }

    /// Advance omega_c by one substep and gather it (plus its spatial
    /// gradient, the micro-curvature kappa) into `out_omega`/`out_curvature`
    /// for every particle. `macro_spin` is the particle's own local
    /// antisymmetric velocity-gradient component (`0.5*(dvy/dx - dvx/dy)`,
    /// computed by the CALLER from `ParticleUpdateCtx::velocity_gradient` --
    /// this field has no access to that context, matching
    /// `GranularFluidityField::pressure_and_ratio`'s own caller-supplied
    /// convention).
    ///
    /// Units: the config is SI (Pa, m); engine particles are in grid units
    /// (see `SimConfig`: length `dx_meters`, mass per metre of depth
    /// `grid_mass_unit_kg_per_m`). The conversion happens here, as in
    /// `GranularFluidityField::apply`, from `SimConfig::dx_meters` and
    /// `SimConfig::grid_mass_unit_kg_per_m()`: SI mass = grid mass * unit,
    /// SI area = grid volume * dx^2, so the density is `unit / dx^2` times the
    /// grid density.
    pub fn apply(
        &mut self,
        particles: &crate::particle::Particles,
        sub_dt: f32,
        scale: GridScale,
        macro_spin: fn(&crate::particle::Particle) -> f32,
        out_omega: &mut [f32],
        out_curvature: &mut [Vec2],
    ) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;
        if self.rel_angle.len() < particles.len() {
            self.rel_angle.resize(particles.len(), 0.0);
        }

        for i in 0..n {
            self.grid_mass[i] = 0.0;
            self.grid_stiff[i] = 0.0;
            self.grid_l[i] = 0.0;
            self.grid_spin[i] = 0.0;
            self.grid_phi[i] = 0.0;
        }

        // --- P2G: scatter micro-inertia I, torque stiffness 2*alpha*V and the
        // I-weighted omega_c, macro spin and relative angle ---
        let stiff_per_volume = 2.0 * self.config.coupling_modulus_pa;
        let dx_meters = scale.dx_meters;
        let area_unit_m2 = dx_meters * dx_meters;
        for pi in 0..particles.len() {
            let p = particles.get(pi);
            let i_micro = self.config.micro_inertia(p.mass * scale.mass_unit_kg_per_m);
            let spin = macro_spin(&p);
            let w = quadratic_weights(p.x);
            // The field's own last-gathered values for this particle (last
            // substep's result), 0.0 at rest/first touch.
            let omega_c_particle = out_omega.get(pi).copied().unwrap_or(0.0);
            let phi_particle = self.rel_angle[pi];
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    let mw = weight * i_micro;
                    self.grid_l[idx] += mw * omega_c_particle;
                    self.grid_mass[idx] += mw;
                    self.grid_spin[idx] += mw * spin;
                    self.grid_phi[idx] += mw * phi_particle;
                    self.grid_stiff[idx] += weight * stiff_per_volume * p.volume * area_unit_m2;
                }
            }
        }

        // --- Solve: exact solution of phi' = w, w' = -Omega^2 phi over sub_dt,
        // with w = omega_c - spin, Omega^2 = K/I (module doc). ---
        for i in 0..n {
            if self.grid_mass[i] <= 0.0 {
                self.grid_work[i] = self.grid_omega[i]; // no particle here this step -- hold last value
                continue;
            }
            let inv_i = 1.0 / self.grid_mass[i];
            let spin_avg = self.grid_spin[i] * inv_i;
            let w0 = self.grid_l[i] * inv_i - spin_avg;
            let phi0 = self.grid_phi[i] * inv_i;
            let omega2 = (self.grid_stiff[i] * inv_i).max(0.0);
            let (phi_t, w_t) = damped_oscillator(
                phi0,
                w0,
                omega2,
                self.config.rotational_damping_rate_per_s.max(0.0),
                sub_dt,
            );
            self.grid_work[i] = spin_avg + w_t;
            self.grid_phi_out[i] = phi_t;
        }
        self.grid_omega.copy_from_slice(&self.grid_work);

        // --- G2P: gather omega_c and its spatial gradient (curvature) ---
        for pi in 0..particles.len().min(out_omega.len()) {
            let p = particles.get(pi);
            let w = quadratic_weights(p.x);
            let mut omega_sum = 0.0f32;
            let mut phi_sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    omega_sum += weight * self.grid_omega[idx];
                    phi_sum += weight * self.grid_phi_out[idx];
                    w_sum += weight;
                }
            }
            out_omega[pi] = if w_sum > 1e-10 {
                omega_sum / w_sum
            } else {
                0.0
            };
            self.rel_angle[pi] = if w_sum > 1e-10 { phi_sum / w_sum } else { 0.0 };

            if pi < out_curvature.len() {
                let base = w.base_cell;
                let x = base.x.clamp(0, res - 1) as usize;
                let y = base.y.clamp(0, res - 1) as usize;
                out_curvature[pi] =
                    micro_curvature_2d(&self.grid_omega, self.grid_res, x, y, dx_meters);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{Particle, Particles};

    fn test_config() -> CosseratConfig {
        CosseratConfig {
            coupling_modulus_pa: 1.0e6,
            grain_diameter_m: 0.3e-3,
            micro_inertia_coefficient: 0.1,
            rotational_damping_rate_per_s: 0.0,
        }
    }

    /// A particle of bulk density `rho` (`mass = rho * volume`).
    fn test_particle_at(pos: Vec2, rho: f32, volume: f32) -> Particle {
        let mut p = Particle::zeroed();
        p.x = pos;
        p.mass = rho * volume;
        p.initial_volume = volume;
        p.volume = volume;
        p
    }

    fn four_particles(rho: f32, volume: f32) -> Particles {
        Particles::from(vec![test_particle_at(Vec2::new(8.0, 8.0), rho, volume); 4])
    }

    #[test]
    fn stability_dt_is_positive_and_finite() {
        let cfg = test_config();
        let dt = cfg.stability_dt(1500.0);
        assert!(dt.is_finite() && dt > 0.0);
    }

    #[test]
    fn stability_dt_is_in_seconds() {
        // alpha = 2e6 Pa, d = 1e-2 m, c = 0.1, rho = 1000 kg/m^3:
        // Omega^2 = 2*2e6 / (1000 * 0.1 * 1e-4) = 4e8 1/s^2, Omega = 2e4 1/s,
        // dt = 2/Omega = 1e-4 s.
        let cfg = CosseratConfig {
            coupling_modulus_pa: 2.0e6,
            grain_diameter_m: 1.0e-2,
            micro_inertia_coefficient: 0.1,
            rotational_damping_rate_per_s: 0.0,
        };
        let dt = cfg.stability_dt(1000.0);
        assert!((dt - 1.0e-4).abs() / 1.0e-4 < 1e-5, "dt={dt}");
        // The same physical case in g, cm, s (alpha = 2e7 g/(cm s^2),
        // d = 1 cm, rho = 1 g/cm^3) is the same number of seconds.
        let cgs = CosseratConfig {
            coupling_modulus_pa: 2.0e7,
            grain_diameter_m: 1.0,
            micro_inertia_coefficient: 0.1,
            rotational_damping_rate_per_s: 0.0,
        };
        let dt_cgs = cgs.stability_dt(1.0);
        assert!((dt - dt_cgs).abs() / dt < 1e-5, "si={dt} cgs={dt_cgs}");
    }

    #[test]
    fn omega_stays_zero_with_zero_macro_spin() {
        // With no macro spin anywhere (irrotational flow) nothing drives the
        // micro-rotation from rest: omega_c stays exactly 0.
        let cfg = test_config();
        let mut field = CosseratField::new(cfg, 16);
        let particles = four_particles(1500.0, 1.0e-4);
        let mut out_omega = vec![0.0; 4];
        let mut out_curvature = vec![Vec2::ZERO; 4];
        for _ in 0..50 {
            field.apply(
                &particles,
                1.0e-6,
                GridScale::SI,
                |_p| 0.0,
                &mut out_omega,
                &mut out_curvature,
            );
        }
        assert!(out_omega.iter().all(|&o| o.abs() < 1e-9));
    }

    /// Runs the SI/CGS scene of the unit tests and returns omega_c.
    fn run_oscillator(cfg: CosseratConfig, rho: f32, volume: f32, steps: usize) -> f32 {
        const SPIN: f32 = 3.0;
        let mut field = CosseratField::new(cfg, 16);
        let particles = four_particles(rho, volume);
        let mut out_omega = vec![0.0; 4];
        let mut out_curvature = vec![Vec2::ZERO; 4];
        for _ in 0..steps {
            field.apply(
                &particles,
                2.0e-5,
                GridScale::SI,
                |_p| SPIN,
                &mut out_omega,
                &mut out_curvature,
            );
        }
        out_omega[0]
    }

    fn si_config() -> CosseratConfig {
        CosseratConfig {
            coupling_modulus_pa: 2.0e6,
            grain_diameter_m: 1.0e-2,
            micro_inertia_coefficient: 0.1,
            rotational_damping_rate_per_s: 0.0,
        }
    }

    #[test]
    fn omega_follows_the_exact_oscillator_driven_by_macro_spin() {
        // Released from rest under a constant spin: omega_c(t) =
        // spin * (1 - cos(Omega t)), Omega = 2e4 1/s here (see
        // `stability_dt_is_in_seconds`).
        let steps = 50;
        let omega = run_oscillator(si_config(), 1000.0, 1.0e-4, steps);
        let expected = 3.0 * (1.0 - (2.0e4f32 * 2.0e-5 * steps as f32).cos());
        assert!(
            (omega - expected).abs() < 2.0e-3 * 3.0,
            "omega_c={omega} expected={expected}"
        );
    }

    #[test]
    fn rescaling_density_units_leaves_the_micro_rotation_unchanged() {
        // kg, m, s -> g, cm, s. Pa = kg/(m s^2) -> 10 g/(cm s^2); length x100;
        // mass x1000; area (the 2D particle volume) x1e4; density 1000 kg/m^3
        // -> 1 g/cm^3. Rates and times are unchanged.
        let steps = 50;
        let si = run_oscillator(si_config(), 1000.0, 1.0e-4, steps);
        let cgs_cfg = CosseratConfig {
            coupling_modulus_pa: 2.0e7,
            grain_diameter_m: 1.0,
            micro_inertia_coefficient: 0.1,
            rotational_damping_rate_per_s: 0.0,
        };
        let cgs = run_oscillator(cgs_cfg, 1.0, 1.0, steps);
        assert!(
            (si - cgs).abs() < 1.0e-4 * si.abs().max(1.0),
            "si={si} cgs={cgs}"
        );
    }

    #[test]
    fn damping_seam_settles_omega_to_the_macro_spin() {
        // phi'' + g phi' + Omega^2 phi = 0: with g > 0 the micro-rotation rate
        // relaxes to the macro spin (SPIN) instead of ringing forever.
        let mut cfg = si_config();
        let omega = cfg.coupling_frequency(1000.0);
        cfg.rotational_damping_rate_per_s = 0.5 * omega;
        let undamped = run_oscillator(si_config(), 1000.0, 1.0, 4000);
        let damped = run_oscillator(cfg, 1000.0, 1.0, 4000);
        assert!((damped - 3.0).abs() < 1.0e-3, "damped omega_c={damped}");
        assert!((undamped - 3.0).abs() > (damped - 3.0).abs());
    }

    #[test]
    fn engine_particles_in_grid_units_give_the_same_frequency_as_si() {
        // Through the real call path: particles spawned by the solver (mass
        // and volume in grid units) with a SimConfig that has a cell size.
        // The physical density is mass * unit / (volume * dx^2), so the field
        // must oscillate at `coupling_frequency` of that density, whatever
        // `dx_meters` is.
        use crate::prelude::*;
        for cell_m in [0.01f32, 0.04] {
            let config = SimConfig::earth(32, cell_m, 0.01);
            let mut solver = Simulation::new(config, SpawnRegion::for_sim(&config));
            let cfg = CosseratConfig {
                coupling_modulus_pa: 2.0e6,
                grain_diameter_m: cell_m,
                micro_inertia_coefficient: 0.1,
                rotational_damping_rate_per_s: 0.0,
            };
            // The spawn gives each particle its own density; make it one
            // uniform SI density so the expected frequency is that of RHO_SI.
            const RHO_SI: f32 = 1000.0;
            let unit = config.grid_mass_unit_kg_per_m();
            let n_all = solver.particles().len();
            for i in 0..n_all {
                let v = solver.particles().volume[i];
                solver.particles_mut().mass[i] =
                    RHO_SI * v * config.dx_meters * config.dx_meters / unit;
            }
            let rho_si = RHO_SI;
            let omega = cfg.coupling_frequency(rho_si);
            let sub_dt = 0.5 / omega;
            let steps = 40;
            let particles = solver.particles();
            let n = particles.len();
            assert!(n > 0 && particles.get(0).volume > 0.0);
            let mut field = CosseratField::new(cfg, 32);
            let mut out_omega = vec![0.0; n];
            let mut out_curvature = vec![Vec2::ZERO; n];
            for _ in 0..steps {
                field.apply(
                    particles,
                    sub_dt,
                    GridScale {
                        dx_meters: config.dx_meters,
                        mass_unit_kg_per_m: config.grid_mass_unit_kg_per_m(),
                    },
                    |_p| 3.0,
                    &mut out_omega,
                    &mut out_curvature,
                );
            }
            let expected = 3.0 * (1.0 - (0.5 * steps as f32).cos());
            let mid = out_omega[n / 2];
            assert!(
                (mid - expected).abs() < 5.0e-3 * 3.0,
                "cell={cell_m} omega_c={mid} expected={expected}"
            );
        }
    }

    #[test]
    fn zero_particles_produces_zero_field_and_no_panic() {
        let cfg = test_config();
        let mut field = CosseratField::new(cfg, 8);
        let particles = Particles::from(vec![]);
        let mut out_omega = vec![0.0; 0];
        let mut out_curvature = vec![Vec2::ZERO; 0];
        field.apply(
            &particles,
            1.0e-6,
            GridScale::SI,
            |_p| 0.0,
            &mut out_omega,
            &mut out_curvature,
        );
        assert!(field.grid_omega.iter().all(|&o| o == 0.0));
    }
}
