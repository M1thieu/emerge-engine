//! Beer-Lambert attenuation: what a medium takes out of light passing through.
//!
//! The counterpart of `super::blackbody`, which says what a medium *adds*.
//! Both are energy transport, and both are stated in SI so a coefficient
//! measured in a laboratory can be used unchanged.
//!
//! The coefficients themselves belong to the substance, not to this module
//! and not to the renderer: see `matter::materials::optical` for measured
//! values and for how a material declares its own.

use std::{error::Error, fmt};

/// Per-material absorption/scattering coefficients in inverse metres.
///
/// Scattering is stored as the reduced coefficient `mu_s' = mu_s (1 - g)`
/// together with the asymmetry `g` (mean cosine of the scattering angle), so
/// both the diffusion regime (which sees `mu_s'`) and the direct beam (which
/// sees the scattering coefficient `mu_s = mu_s' / (1 - g)`) are served from
/// one record. The definition of `mu_s'` is Jacques and Prahl, ECE532 class 3,
/// "Reduced scattering coefficient". Scattering stays one visible-band
/// scalar. The unit is explicit; a later spectral transport upgrade can widen
/// it without changing the absorption contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticalCoefficientsSi {
    pub absorption_m_inv: [f32; 3],
    pub reduced_scattering_m_inv: f32,
    /// Mean cosine `g` of the scattering angle, in `[-1, 1)`. 0 is isotropic
    /// scattering, for which the reduced and the plain scattering coefficient
    /// coincide.
    pub scattering_asymmetry: f32,
}

impl OpticalCoefficientsSi {
    /// Isotropic scattering (`g = 0`, so `mu_s = mu_s'`). Use
    /// [`Self::with_asymmetry`] for a medium whose scattering is forward
    /// peaked.
    pub fn new(
        absorption_m_inv: [f32; 3],
        reduced_scattering_m_inv: f32,
    ) -> Result<Self, OpticalCoefficientsError> {
        if absorption_m_inv
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(OpticalCoefficientsError::InvalidAbsorption);
        }
        if !reduced_scattering_m_inv.is_finite() || reduced_scattering_m_inv < 0.0 {
            return Err(OpticalCoefficientsError::InvalidReducedScattering);
        }
        Ok(Self {
            absorption_m_inv,
            reduced_scattering_m_inv,
            scattering_asymmetry: 0.0,
        })
    }

    /// The same coefficients with scattering asymmetry `g`, which must be
    /// finite and in `[-1, 1)` (`g = 1` would make `mu_s = mu_s' / (1 - g)`
    /// unbounded).
    pub fn with_asymmetry(self, g: f32) -> Result<Self, OpticalCoefficientsError> {
        if !g.is_finite() || !(-1.0..1.0).contains(&g) {
            return Err(OpticalCoefficientsError::InvalidAsymmetry);
        }
        Ok(Self {
            scattering_asymmetry: g,
            ..self
        })
    }

    /// Scattering coefficient `mu_s = mu_s' / (1 - g)`, per metre: the rate
    /// at which scattering removes light from the direct beam.
    pub fn scattering_m_inv(&self) -> f32 {
        scattering_from_reduced(self.reduced_scattering_m_inv, self.scattering_asymmetry)
    }

    /// Extinction `sigma_t = sigma_a + sigma_s` per channel, per metre: the
    /// decay rate of the direct (unscattered) beam (pbr-book 3rd ed. sec.
    /// 11.1.3, eq. 11.1; Libois et al. 2013, `sigma_e = sigma_s + sigma_a`).
    pub fn extinction_m_inv(&self) -> [f32; 3] {
        let scattering = self.scattering_m_inv();
        self.absorption_m_inv.map(|a| a + scattering)
    }
}

/// `mu_s = mu_s' / (1 - g)`, the inverse of the reduced-scattering definition
/// `mu_s' = mu_s (1 - g)` (Jacques and Prahl, ECE532 class 3). The divisor is
/// floored at `f32::EPSILON` so a degenerate `g` cannot return infinity.
pub fn scattering_from_reduced(reduced_scattering_m_inv: f32, asymmetry: f32) -> f32 {
    let one_minus_g = (1.0 - asymmetry).max(f32::EPSILON);
    reduced_scattering_m_inv.max(0.0) / one_minus_g
}

/// Exact absorption-only solution for a homogeneous slab.
///
/// `relative_density` is `rho/rho_ref`, so the exponent is dimensionless:
/// `(m^-1) * 1 * m`. This is the reference used by CPU and GPU validation;
/// it does not include scattering, reflection, or emission.
pub fn beer_lambert_transmittance(
    absorption_m_inv: [f32; 3],
    relative_density: f32,
    path_length_meters: f32,
) -> [f32; 3] {
    let column_length = relative_density.max(0.0) * path_length_meters.max(0.0);
    absorption_m_inv.map(|sigma| (-sigma.max(0.0) * column_length).exp())
}

/// Radiance leaving a slab of absorbing, scattering medium, in whatever unit
/// `background` and `incident` are given in.
///
/// Three real terms, in the order light meets them:
///
/// - Fresnel reflection at the front face. What reflects never enters, so it
///   carries the incident radiance straight back and the two interior terms
///   are weighted by `1 - R`.
/// - Direct transmission of whatever is behind the slab, attenuated by the
///   full extinction `sigma_t = sigma_a + sigma_s` (pbr-book 3rd ed. sec.
///   11.1.3, eq. 11.1). Scattering removes light from the direct beam just
///   as absorption does, which is why the exponent is not `sigma_a` alone.
///   `sigma_s` is the scattering coefficient itself, not the reduced one:
///   `mu_s' = mu_s (1 - g)` is a diffusion-regime quantity (Jacques and
///   Prahl, ECE532 class 3) and would understate the direct beam's loss by
///   `1 / (1 - g)`, 5.7 times for snow.
/// - Single scattering. Of everything the slab removed from the beam,
///   `sigma_s / sigma_t` -- the single-scattering albedo -- was scattered
///   rather than absorbed. Lit by the incident radiance, so the glow takes
///   the colour of the actual light source rather than a chosen tint.
///   (Jacques, "Optical properties of biological tissues: a review", Phys.
///   Med. Biol. 58, 2013.)
///
/// `cos_view` is measured from the surface normal; pass 1.0 where no normal
/// exists, which reduces Fresnel to its normal-incidence value.
///
/// Disclosed simplification: single scattering with an isotropic phase
/// function and no multiple-scattering term, so a very dense, highly
/// scattering medium is under-lit. Fixing that means adding the diffusion
/// term, not a constant.
///
/// `scattering_m_inv` is `mu_s`, see
/// [`OpticalCoefficientsSi::scattering_m_inv`].
///
/// KNOWN-LIMIT[HARD-WIRED/TECH-LIMIT]: `systems/render/shaders/radiative_transfer.inc.wgsl`
/// (`slab_radiance`, line 50) still mirrors the previous behaviour and is fed
/// the reduced coefficient | source: pbr-book 3rd ed. eq. 11.1; Jacques and
/// Prahl ECE532 class 3 | replace when: the shader reads `mu_s'` and `g` from
/// the optical table and a GPU adapter test reproduces this function | state:
/// waiting | gap: no GPU adapter test in CI
pub fn slab_radiance(
    background: [f32; 3],
    incident: [f32; 3],
    absorption_m_inv: [f32; 3],
    scattering_m_inv: f32,
    path_length_meters: f32,
    fresnel_r0: f32,
    cos_view: f32,
) -> [f32; 3] {
    let sigma_s = scattering_m_inv.max(0.0);
    let path = path_length_meters.max(0.0);
    let reflectance = super::fresnel::schlick_reflectance(fresnel_r0, cos_view);
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        let sigma_t = absorption_m_inv[channel].max(0.0) + sigma_s;
        let transmitted = (-sigma_t * path).exp();
        let albedo = sigma_s / sigma_t.max(1.0e-9);
        let interior =
            background[channel] * transmitted + incident[channel] * albedo * (1.0 - transmitted);
        out[channel] = interior * (1.0 - reflectance) + incident[channel] * reflectance;
    }
    out
}

/// How fast light dies out with depth inside a medium, per metre, per channel,
/// for its absorption `mu_a`, reduced scattering `mu_s'` and asymmetry `g`.
///
/// Light reaching a point inside matter is the part never scattered, which
/// decays with the full extinction `mu_a + mu_s` (`mu_s = mu_s' / (1 - g)`,
/// the scattering coefficient, not the reduced one), plus the part that
/// scattered and kept going. Where scattering dominates, that diffuse part
/// decays far more slowly, with the effective attenuation of diffusion
/// theory, `1 / delta = sqrt(mu_a / D)` with `D = 1 / (3 (mu_a + mu_s'))`,
/// i.e. `sqrt(3 mu_a (mu_a + mu_s'))` (S. L. Jacques and S. A. Prahl, ECE532
/// Biomedical Optics course notes, Oregon Graduate Institute 1998,
/// "Steady-state diffusion theory" and "Limits of diffusion theory",
/// omlc.org/classroom/ece532/class5). Deep enough, the slower of the two
/// exponentials carries all the light, so the slower rate is returned. In
/// snow it gives light a reach of about 16 cm in red and about 1 m in blue,
/// against 0.7 mm for the unscattered beam alone: snow is white inside, and
/// blue deep down. With no scattering it is `mu_a`, Beer-Lambert. The notes
/// state the diffusion form fails where absorption is not small against
/// scattering; there the unscattered rate is the smaller one anyway.
///
/// The renderer's light pass (`systems/render/shaders/light_pass.wgsl`)
/// marches this rate toward the light.
pub fn penetration_attenuation_m_inv(
    absorption_m_inv: [f32; 3],
    reduced_scattering_m_inv: f32,
    scattering_asymmetry: f32,
) -> [f32; 3] {
    let reduced = reduced_scattering_m_inv.max(0.0);
    let scattering = scattering_from_reduced(reduced, scattering_asymmetry);
    absorption_m_inv.map(|absorption| {
        let absorption = absorption.max(0.0);
        let unscattered = absorption + scattering;
        let diffuse = (3.0 * absorption * (absorption + reduced)).sqrt();
        unscattered.min(diffuse)
    })
}

/// Optical coefficients of a mixture of constituents, weighted by volume
/// fraction.
///
/// Each `(volume_fraction, coefficients)` pair gives the share of the volume
/// a constituent occupies and the coefficients of that constituent on its own.
/// A coefficient is a number density times a cross-section, `sigma = n C`
/// (Libois et al. 2013, eqs. 1 and 2), and in a mixture the number density of
/// a constituent is its pure-state density times its volume fraction. The
/// absorption of a mixture is therefore the sum over constituents of
/// cross-section times number density (Libois eq. 24), i.e.
/// `sigma_mix = sum_i phi_i sigma_i`; the same sum holds for the scattering
/// coefficient. It is not a mass-fraction average: equal masses of a dense and
/// a light constituent occupy different volumes, see
/// [`volume_fractions_from_masses`].
///
/// Adaptation, derived here and not transcribed: the mixture's asymmetry is
/// the scattering-weighted mean cosine, `g = sum_i phi_i sigma_s,i g_i /
/// sum_i phi_i sigma_s,i`, because the mixture's phase function is the
/// scattering-weighted sum of the constituents' phase functions and `g` is
/// the first moment of the phase function. With it the mixture's reduced
/// scattering `sigma_s (1 - g)` equals `sum_i phi_i mu_s',i`.
///
/// KNOWN-LIMIT[HARD-WIRED/TECH-LIMIT]: the renderer's per-cell blends still
/// weight the slots by MASS fraction, not by this volume fraction:
/// `shaders/light_pass.wgsl:88-90` (`cell_attenuation`),
/// `shaders/grid_volume.wgsl:233-241` and `:342-346` (`blended_optical_slot`),
/// `shaders/curvature_flow.wgsl:1391-1403` and `:1529-1534` | source: Libois
/// et al. 2013 eqs. 1, 2, 24 (number density times cross-section) | replace
/// when: the blend weights are `volume_fractions_from_masses` of the per-slot
/// mass and each slot's reference density, and a GPU adapter test matches
/// [`mix_by_volume_fraction`] | state: waiting | gap: no GPU adapter test in
/// CI, and the per-slot reference density is not yet in the optical table
///
/// Fractions summing to less than 1 leave the remainder as void (no optical
/// effect). Returns an error if a fraction is negative or not finite, or the
/// fractions sum to more than 1 (beyond a rounding tolerance of 1e-6).
pub fn mix_by_volume_fraction(
    components: &[(f32, OpticalCoefficientsSi)],
) -> Result<OpticalCoefficientsSi, OpticalCoefficientsError> {
    let mut total_fraction = 0.0f64;
    let mut absorption = [0.0f64; 3];
    let mut reduced = 0.0f64;
    let mut scattering = 0.0f64;
    let mut scattering_times_g = 0.0f64;
    for (fraction, coefficients) in components {
        if !fraction.is_finite() || *fraction < 0.0 {
            return Err(OpticalCoefficientsError::InvalidVolumeFraction);
        }
        let phi = f64::from(*fraction);
        total_fraction += phi;
        for (sum, a) in absorption.iter_mut().zip(coefficients.absorption_m_inv) {
            *sum += phi * f64::from(a);
        }
        let sigma_s = f64::from(coefficients.scattering_m_inv());
        reduced += phi * f64::from(coefficients.reduced_scattering_m_inv);
        scattering += phi * sigma_s;
        scattering_times_g += phi * sigma_s * f64::from(coefficients.scattering_asymmetry);
    }
    if total_fraction > 1.0 + 1.0e-6 {
        return Err(OpticalCoefficientsError::InvalidVolumeFraction);
    }
    let asymmetry = if scattering > 0.0 {
        scattering_times_g / scattering
    } else {
        0.0
    };
    Ok(OpticalCoefficientsSi {
        absorption_m_inv: absorption.map(|a| a as f32),
        reduced_scattering_m_inv: reduced as f32,
        scattering_asymmetry: asymmetry as f32,
    })
}

/// Volume fractions (of the volume the listed constituents occupy together)
/// from their masses and bulk densities: `phi_i = (m_i / rho_i) / sum_j (m_j /
/// rho_j)`. A constituent with a non-positive density or mass contributes no
/// volume. Returns all zeros if no constituent has volume.
pub fn volume_fractions_from_masses(masses_kg: &[f32], densities_kg_m3: &[f32]) -> Vec<f32> {
    let volumes: Vec<f64> = masses_kg
        .iter()
        .zip(densities_kg_m3)
        .map(|(m, rho)| {
            if *m > 0.0 && *rho > 0.0 {
                f64::from(*m) / f64::from(*rho)
            } else {
                0.0
            }
        })
        .collect();
    let total: f64 = volumes.iter().sum();
    volumes
        .into_iter()
        .map(|v| if total > 0.0 { (v / total) as f32 } else { 0.0 })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpticalCoefficientsError {
    InvalidAbsorption,
    InvalidReducedScattering,
    InvalidAsymmetry,
    InvalidVolumeFraction,
}

impl fmt::Display for OpticalCoefficientsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidAbsorption => "absorption coefficients must be finite and non-negative",
            Self::InvalidReducedScattering => "reduced scattering must be finite and non-negative",
            Self::InvalidAsymmetry => "scattering asymmetry must be finite and in [-1, 1)",
            Self::InvalidVolumeFraction => {
                "volume fractions must be finite, non-negative and sum to at most 1"
            }
        };
        f.write_str(message)
    }
}

impl Error for OpticalCoefficientsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penetration_without_scattering_is_beer_lambert() {
        let absorption = [0.5, 1.0, 2.0];
        assert_eq!(
            penetration_attenuation_m_inv(absorption, 0.0, 0.0),
            absorption
        );
        assert_eq!(
            penetration_attenuation_m_inv(absorption, 0.0, 0.8),
            absorption
        );
    }

    /// Criterion A6: where absorption is not small against scattering the
    /// unscattered beam's rate wins, and it is the extinction with the
    /// scattering coefficient `mu_s'/(1-g)`, not the reduced one.
    #[test]
    fn penetration_unscattered_branch_uses_scattering_not_reduced() {
        let rate = penetration_attenuation_m_inv([100.0; 3], 10.0, 0.5);
        for r in rate {
            assert!((r - 120.0).abs() < 1.0e-4, "rate {r}, expected 120");
        }
        // Isotropic: unchanged from the reduced-scattering formula.
        let rate = penetration_attenuation_m_inv([100.0; 3], 10.0, 0.0);
        for r in rate {
            assert!((r - 110.0).abs() < 1.0e-4, "rate {r}, expected 110");
        }
        // Never faster than the direct beam's own decay.
        for g in [0.0f32, 0.5, 0.825] {
            for mu_s_reduced in [0.1f32, 10.0, 1000.0] {
                for mu_a in [0.001f32, 1.0, 50.0] {
                    let bound = mu_a + mu_s_reduced / (1.0 - g);
                    let rate = penetration_attenuation_m_inv([mu_a; 3], mu_s_reduced, g)[0];
                    assert!(rate <= bound * (1.0 + 1.0e-6), "{rate} > {bound}");
                }
            }
        }
    }

    /// Criterion A1: a purely scattering slab in front of a lit background
    /// passes exp(-mu_s L) of it, mu_s = mu_s'/(1-g), not exp(-mu_s' L).
    #[test]
    fn direct_beam_decays_with_the_scattering_coefficient_not_the_reduced_one() {
        let g = 0.825f32;
        let medium = OpticalCoefficientsSi::new([0.0; 3], 100.0)
            .unwrap()
            .with_asymmetry(g)
            .unwrap();
        let sigma_s = f64::from(medium.scattering_m_inv());
        assert!((sigma_s - 100.0 / 0.175).abs() < 1.0e-3);
        for path in [1.0e-3f32, 5.0e-3] {
            let out = slab_radiance(
                [1.0; 3],
                [0.0; 3],
                medium.absorption_m_inv,
                medium.scattering_m_inv(),
                path,
                0.0,
                1.0,
            );
            let expected = (-sigma_s * f64::from(path)).exp();
            let old = (-100.0f64 * f64::from(path)).exp();
            for value in out {
                let relative = (f64::from(value) - expected).abs() / expected;
                assert!(relative < 1.0e-5, "L {path}: got {value}, want {expected}");
            }
            assert!(
                (old - expected).abs() / expected > 0.1,
                "the old and new answers must differ visibly at L {path}"
            );
        }
    }

    /// Criterion A2: without scattering the asymmetry is irrelevant.
    #[test]
    fn zero_scattering_ignores_asymmetry() {
        let sigma = [0.35, 0.033, 0.011];
        let plain = beer_lambert_transmittance(sigma, 1.0, 2.0);
        let medium = OpticalCoefficientsSi::new(sigma, 0.0)
            .unwrap()
            .with_asymmetry(0.9)
            .unwrap();
        assert_eq!(medium.scattering_m_inv(), 0.0);
        let slab = slab_radiance(
            [1.0; 3],
            [1.0; 3],
            sigma,
            medium.scattering_m_inv(),
            2.0,
            0.0,
            1.0,
        );
        // incident = background = 1 and no scattering: only absorption acts.
        for channel in 0..3 {
            assert!((slab[channel] - plain[channel]).abs() < 1.0e-6);
        }
    }

    /// Criterion A3: an isotropic medium (g = 0) reproduces the previous
    /// behaviour exactly, so every existing caller of `new` is unchanged.
    #[test]
    fn isotropic_medium_is_the_previous_behaviour() {
        let medium = OpticalCoefficientsSi::new([0.4, 0.2, 0.1], 7.0).unwrap();
        assert_eq!(medium.scattering_asymmetry, 0.0);
        assert_eq!(medium.scattering_m_inv(), 7.0);
        assert_eq!(medium.extinction_m_inv(), [7.4, 7.2, 7.1]);
    }

    /// Criterion A4: transmitted + absorbed + scattered = incident. With a
    /// uniform field of radiance 1 behind and in front, the slab's output is
    /// 1 minus the absorbed share; the scattered share returns to the field.
    #[test]
    fn slab_conserves_transmitted_absorbed_and_scattered_energy() {
        for (sigma_a, reduced, g, path) in [
            (0.5f32, 100.0f32, 0.825f32, 0.01f32),
            (2.0, 30.0, 0.5, 0.2),
            (0.01, 400.0, 0.9, 0.05),
            (5.0, 0.0, 0.0, 0.3),
            (0.0, 50.0, 0.7, 0.1),
        ] {
            let medium = OpticalCoefficientsSi::new([sigma_a; 3], reduced)
                .unwrap()
                .with_asymmetry(g)
                .unwrap();
            let sigma_s = medium.scattering_m_inv();
            let sigma_t = sigma_a + sigma_s;
            let transmitted = (-sigma_t * path).exp();
            let absorbed = sigma_a / sigma_t * (1.0 - transmitted);
            let scattered = sigma_s / sigma_t * (1.0 - transmitted);
            assert!(
                (transmitted + absorbed + scattered - 1.0).abs() < 1.0e-6,
                "partition does not sum to 1"
            );
            assert!(absorbed >= 0.0 && scattered >= 0.0);
            let out = slab_radiance([1.0; 3], [1.0; 3], [sigma_a; 3], sigma_s, path, 0.0, 1.0);
            for value in out {
                assert!(
                    (value - (1.0 - absorbed)).abs() < 1.0e-6,
                    "got {value}, expected {}",
                    1.0 - absorbed
                );
            }
        }
    }

    /// Criterion A5: snow's extinction against Libois eq. 1 and 4,
    /// `sigma_e = n C_ext = 3 v_i / (2 r)` for spheres of radius r.
    #[test]
    fn snow_extinction_matches_libois_geometric_optics_limit() {
        let (r, v_i) = (242.5e-6f32, 0.465f32);
        let snow = crate::materials::optical::snow(r, v_i);
        let ratio = snow.scattering_m_inv() / snow.reduced_scattering_m_inv;
        let expected_ratio = 1.0 / (1.0 - crate::materials::optical::SNOW_SCATTERING_ASYMMETRY);
        assert!((ratio - expected_ratio).abs() < 1.0e-5 * expected_ratio);
        let libois_sigma_e = 1.5 * v_i / r;
        for extinction in snow.extinction_m_inv() {
            let relative = (extinction - libois_sigma_e).abs() / libois_sigma_e;
            assert!(
                relative < 1.0e-3,
                "sigma_e {extinction} vs {libois_sigma_e}"
            );
        }
    }

    /// Criterion A7.
    #[test]
    fn asymmetry_out_of_range_is_rejected() {
        let base = OpticalCoefficientsSi::new([0.1; 3], 1.0).unwrap();
        for bad in [1.0f32, 1.5, -1.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                base.with_asymmetry(bad),
                Err(OpticalCoefficientsError::InvalidAsymmetry)
            );
        }
        assert!(base.with_asymmetry(-1.0).is_ok());
        assert!(base.with_asymmetry(0.999).is_ok());
    }

    fn coefficients(a: f32, reduced: f32, g: f32) -> OpticalCoefficientsSi {
        OpticalCoefficientsSi::new([a, 2.0 * a, 3.0 * a], reduced)
            .unwrap()
            .with_asymmetry(g)
            .unwrap()
    }

    /// Criteria D1 and D2.
    #[test]
    fn mixing_one_component_is_identity_and_splitting_changes_nothing() {
        let c = coefficients(0.3, 40.0, 0.6);
        let one = mix_by_volume_fraction(&[(1.0, c)]).unwrap();
        assert_eq!(one.absorption_m_inv, c.absorption_m_inv);
        assert!((one.reduced_scattering_m_inv - 40.0).abs() < 1.0e-5);
        assert!((one.scattering_asymmetry - 0.6).abs() < 1.0e-6);
        let split = mix_by_volume_fraction(&[(0.5, c), (0.5, c)]).unwrap();
        for channel in 0..3 {
            assert!((split.absorption_m_inv[channel] - c.absorption_m_inv[channel]).abs() < 1.0e-6);
        }
        assert!((split.reduced_scattering_m_inv - 40.0).abs() < 1.0e-5);
        assert!((split.scattering_asymmetry - 0.6).abs() < 1.0e-6);
    }

    /// Criteria D3 and D4.
    #[test]
    fn mixing_is_linear_in_volume_fraction_and_void_dilutes() {
        let a = coefficients(0.3, 40.0, 0.8);
        let b = coefficients(1.0, 10.0, 0.2);
        let mix = mix_by_volume_fraction(&[(0.25, a), (0.5, b)]).unwrap();
        for channel in 0..3 {
            let expected = 0.25 * a.absorption_m_inv[channel] + 0.5 * b.absorption_m_inv[channel];
            assert!((mix.absorption_m_inv[channel] - expected).abs() < 1.0e-6);
        }
        let reduced = 0.25 * 40.0 + 0.5 * 10.0;
        assert!((mix.reduced_scattering_m_inv - reduced).abs() < 1.0e-4 * reduced);
        let sigma_s = 0.25 * a.scattering_m_inv() + 0.5 * b.scattering_m_inv();
        assert!((mix.scattering_m_inv() - sigma_s).abs() < 1.0e-4 * sigma_s);
        // mu_s' = mu_s (1 - g) must also hold for the mixture itself.
        let reconstructed = mix.scattering_m_inv() * (1.0 - mix.scattering_asymmetry);
        assert!((reconstructed - reduced).abs() < 1.0e-4 * reduced);
        // Half void: half of the single-component coefficients.
        let half = mix_by_volume_fraction(&[(0.5, a)]).unwrap();
        assert!((half.absorption_m_inv[2] - 0.5 * a.absorption_m_inv[2]).abs() < 1.0e-6);
        assert!((half.reduced_scattering_m_inv - 20.0).abs() < 1.0e-5);
        assert!((half.scattering_asymmetry - 0.8).abs() < 1.0e-6);
        // More than the whole volume, or a negative share, is refused.
        assert_eq!(
            mix_by_volume_fraction(&[(0.7, a), (0.7, b)]),
            Err(OpticalCoefficientsError::InvalidVolumeFraction)
        );
        assert_eq!(
            mix_by_volume_fraction(&[(-0.1, a)]),
            Err(OpticalCoefficientsError::InvalidVolumeFraction)
        );
    }

    /// Criterion D5: equal masses of a dense and a light constituent are
    /// not equal shares of the volume, so a mass-fraction blend is a
    /// different (wrong) mixture.
    #[test]
    fn mass_fractions_are_not_volume_fractions() {
        let phi = volume_fractions_from_masses(&[1.0, 1.0], &[1000.0, 500.0]);
        assert!((phi[0] - 1.0 / 3.0).abs() < 1.0e-7);
        assert!((phi[1] - 2.0 / 3.0).abs() < 1.0e-7);
        let absorber = OpticalCoefficientsSi::new([1.0; 3], 0.0).unwrap();
        let clear = OpticalCoefficientsSi::new([0.0; 3], 0.0).unwrap();
        let by_volume = mix_by_volume_fraction(&[(phi[0], absorber), (phi[1], clear)]).unwrap();
        let by_mass = mix_by_volume_fraction(&[(0.5, absorber), (0.5, clear)]).unwrap();
        assert!((by_volume.absorption_m_inv[0] - 1.0 / 3.0).abs() < 1.0e-6);
        assert!((by_mass.absorption_m_inv[0] - 0.5).abs() < 1.0e-6);
        assert!(volume_fractions_from_masses(&[0.0], &[1000.0]) == vec![0.0]);
    }

    /// Snow's measured coefficients (`optical::snow` at a 242.5 um grain
    /// and 200 kg/m^3): scattering dominates, so light penetrates at the
    /// diffusion rate, far slower than the unscattered beam's.
    #[test]
    fn penetration_in_snow_follows_diffusion_theory() {
        let snow = crate::materials::optical::snow(
            242.5e-6,
            200.0 / crate::materials::optical::ICE_DENSITY_KG_M3,
        );
        let got = penetration_attenuation_m_inv(
            snow.absorption_m_inv,
            snow.reduced_scattering_m_inv,
            snow.scattering_asymmetry,
        );
        let mu_s = snow.reduced_scattering_m_inv;
        for (rate, mu_a) in got.into_iter().zip(snow.absorption_m_inv) {
            let diffuse = (3.0 * mu_a * (mu_a + mu_s)).sqrt();
            assert!(diffuse < mu_a + mu_s);
            assert!((rate - diffuse).abs() <= 1.0e-6 * diffuse);
        }
    }

    #[test]
    fn beer_lambert_matches_known_homogeneous_slab() {
        let transmittance = beer_lambert_transmittance([2.0, 1.0, 0.5], 1.0, 0.25);
        let expected = [(-0.5f32).exp(), (-0.25f32).exp(), (-0.125f32).exp()];
        for (got, want) in transmittance.into_iter().zip(expected) {
            assert!((got - want).abs() < 1.0e-6, "got {got}, expected {want}");
        }
    }

    /// With no scattering and no reflection, the slab model must collapse
    /// exactly onto Beer-Lambert -- the general law has to contain the
    /// special case, or one of the two is wrong.
    #[test]
    fn slab_reduces_to_beer_lambert_without_scattering_or_reflection() {
        let sigma = [0.35, 0.033, 0.011];
        let background = [0.7, 0.8, 0.9];
        let plain = beer_lambert_transmittance(sigma, 1.0, 2.0);
        let slab = slab_radiance(background, [1.0; 3], sigma, 0.0, 2.0, 0.0, 1.0);
        for channel in 0..3 {
            let expected = background[channel] * plain[channel];
            assert!(
                (slab[channel] - expected).abs() < 1.0e-6,
                "channel {channel}: slab {} vs Beer-Lambert {expected}",
                slab[channel]
            );
        }
    }

    /// A slab that reflects everything shows only the light falling on it.
    #[test]
    fn total_reflection_hides_whatever_is_behind() {
        let incident = [0.2, 0.4, 0.6];
        let slab = slab_radiance([1.0; 3], incident, [0.5; 3], 1.0, 3.0, 1.0, 1.0);
        for channel in 0..3 {
            assert!((slab[channel] - incident[channel]).abs() < 1.0e-6);
        }
    }

    /// Scattering brightens a medium lit more strongly than its background,
    /// which is the whole reason milk is white and not grey.
    #[test]
    fn scattering_brightens_a_medium_lit_from_the_front() {
        let dark_background = [0.0; 3];
        let incident = [1.0; 3];
        let absorbing = slab_radiance(dark_background, incident, [0.5; 3], 0.0, 2.0, 0.0, 1.0);
        let scattering = slab_radiance(dark_background, incident, [0.5; 3], 5.0, 2.0, 0.0, 1.0);
        for channel in 0..3 {
            assert!(
                scattering[channel] > absorbing[channel] + 0.1,
                "channel {channel}: scattering {} should clearly exceed {}",
                scattering[channel],
                absorbing[channel]
            );
        }
    }

    #[test]
    fn beer_lambert_is_invariant_to_spatial_discretization() {
        let sigma = [0.35, 0.033, 0.011];
        let physical_length = 2.0;
        let analytic = beer_lambert_transmittance(sigma, 1.0, physical_length);

        for cells in [2usize, 20, 200, 2_000] {
            let dx = physical_length / cells as f32;
            let accumulated_tau = sigma.map(|s| (0..cells).map(|_| s * dx).sum::<f32>());
            let discrete = accumulated_tau.map(|tau| (-tau).exp());
            for (got, want) in discrete.into_iter().zip(analytic) {
                assert!(
                    (got - want).abs() < 5.0e-5,
                    "{cells} cells changed transmittance: got {got}, expected {want}"
                );
            }
        }
    }
}
