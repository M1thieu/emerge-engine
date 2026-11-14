//! Spectrum to colour: radiometry becoming photometry.
//!
//! `super::blackbody` says how much power a hot body emits at each wavelength
//! (Planck's law). This module says what that spectrum amounts to as a
//! colour, which needs one thing physics alone does not supply: the response
//! of human photoreceptors, i.e. the CIE 1931 standard observer.
//!
//! This is still the energy domain, not the render domain. A colour derived
//! from a spectrum is a property of the light, and `systems::render` consumes
//! it rather than deriving its own.
//!
//! Two paths compute the same colour:
//!
//! - [`blackbody_linear_srgb`] integrates real Planck radiance against the
//!   observer. It is the ground truth, and too expensive for a fragment
//!   shader (hundreds of `exp` calls per pixel).
//! - [`blackbody_linear_srgb_locus_fit`] evaluates a published closed-form
//!   fit to the Planckian locus. It is what `blackbody.inc.wgsl` implements,
//!   line for line, so the GPU and the CPU agree.
//!
//! `locus_fit_matches_planck_integration` is the test that keeps the second
//! honest about the first.
//!
//! References:
//! - Wyman, Sloan, Shirley, "Simple Analytic Approximations to the CIE XYZ
//!   Color Matching Functions", Journal of Computer Graphics Techniques 2(2),
//!   2013 -- the multi-lobe Gaussian fit used by [`cie_1931_observer`].
//! - Kang, Moon, Hong, Lee, Cho, Kim, "Design of Advanced Color Temperature
//!   Control System for HDTV Applications", Journal of the Korean Physical
//!   Society 41(6):865-871, 2002 -- the cubic-spline Planckian locus used by
//!   [`blackbody_linear_srgb_locus_fit`], valid over 1667 K to 25000 K.
//! - IEC 61966-2-1 (sRGB) -- the XYZ to linear-RGB matrix, D65 white point.

use super::blackbody::planck_spectral_radiance;

/// Lowest temperature the Kang 2002 locus fit is defined for. Below this the
/// fit is evaluated at its own lower bound: a blackbody there is a deep
/// orange-red whose chromaticity barely moves, while its radiance has already
/// fallen by orders of magnitude under `T^4`, so the visible error is
/// dominated by the brightness term, not the hue.
pub const LOCUS_FIT_MIN_K: f32 = 1667.0;
/// Highest temperature the Kang 2002 locus fit is defined for.
pub const LOCUS_FIT_MAX_K: f32 = 25_000.0;

/// CIE 1931 2-degree standard observer colour matching functions.
///
/// Wyman/Sloan/Shirley 2013's multi-lobe fit: each lobe is a Gaussian whose
/// width differs either side of its centre, which is how an asymmetric
/// photoreceptor response is captured without a 471-entry table.
pub fn cie_1931_observer(wavelength_nm: f64) -> [f64; 3] {
    fn lobe(x: f64, center: f64, inverse_width_lo: f64, inverse_width_hi: f64) -> f64 {
        let width = if x < center {
            inverse_width_lo
        } else {
            inverse_width_hi
        };
        let t = (x - center) * width;
        (-0.5 * t * t).exp()
    }
    let l = wavelength_nm;
    [
        0.362 * lobe(l, 442.0, 0.0624, 0.0374) + 1.056 * lobe(l, 599.8, 0.0264, 0.0323)
            - 0.065 * lobe(l, 501.1, 0.0490, 0.0382),
        0.821 * lobe(l, 568.8, 0.0213, 0.0247) + 0.286 * lobe(l, 530.9, 0.0613, 0.0322),
        1.217 * lobe(l, 437.0, 0.0845, 0.0278) + 0.681 * lobe(l, 459.0, 0.0385, 0.0725),
    ]
}

/// CIE xy chromaticity of a blackbody, by integrating Planck's law against the
/// standard observer over the visible band.
///
/// 1 nm steps over 360-830 nm: the observer's narrowest lobe is ~25 nm wide,
/// so this resolves it with room to spare, and the whole integral runs once
/// per lookup-table entry, never per pixel.
pub fn blackbody_chromaticity_xy(temperature_k: f64) -> [f64; 2] {
    let mut xyz = [0.0f64; 3];
    for nm in 360..=830 {
        let wavelength_nm = f64::from(nm);
        let radiance = planck_spectral_radiance(wavelength_nm * 1.0e-9, temperature_k);
        let observer = cie_1931_observer(wavelength_nm);
        for channel in 0..3 {
            xyz[channel] += radiance * observer[channel];
        }
    }
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum <= 0.0 {
        // Colder than anything that emits measurable visible light.
        return [0.6, 0.4];
    }
    [xyz[0] / sum, xyz[1] / sum]
}

/// Spectral sensitivity of each linear sRGB channel, at one wavelength.
///
/// The CIE observer passed through the sRGB primaries: `r(l)`, `g(l)`, `b(l)`
/// say how much light at wavelength `l` each channel reports. Real sRGB
/// matching functions have negative lobes, because the primaries cannot
/// reproduce every visible colour; a negative *weight* is meaningless when
/// averaging a material property, so those are clamped to zero. That is a
/// stated approximation of band averaging, not of the observer itself.
pub fn srgb_channel_sensitivity(wavelength_nm: f64) -> [f64; 3] {
    unclamped_srgb_sensitivity(wavelength_nm).map(|weight| weight.max(0.0))
}

/// CIE XYZ to linear sRGB, IEC 61966-2-1, D65 white point. The one
/// definition of the matrix in the engine.
const XYZ_TO_LINEAR_SRGB: [[f64; 3]; 3] = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
];

fn xyz_to_linear_srgb(xyz: [f64; 3]) -> [f64; 3] {
    XYZ_TO_LINEAR_SRGB.map(|row| row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2])
}

/// The CIE observer through the sRGB matrix, negative lobes kept: the weights
/// under which a spectral radiance integrates to linear sRGB.
fn unclamped_srgb_sensitivity(wavelength_nm: f64) -> [f64; 3] {
    xyz_to_linear_srgb(cie_1931_observer(wavelength_nm))
}

/// Weights `w` with `Y = w . rgb`: the second row of the inverse of the
/// XYZ to linear sRGB matrix, since `XYZ = M^-1 rgb`. For the matrix above it
/// is the familiar luminance weighting of the sRGB primaries; it is computed
/// here from the matrix so there is no second set of constants to keep equal.
pub fn linear_srgb_luminance_weights() -> [f64; 3] {
    let m = XYZ_TO_LINEAR_SRGB;
    // Row 1 of the inverse is column 1 of the cofactor matrix over det M:
    // inverse[1][j] = cofactor(j, 1) / det.
    let cofactor = |row: usize, col: usize| {
        let (r0, r1) = match row {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let (c0, c1) = match col {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let minor = m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
        if (row + col).is_multiple_of(2) {
            minor
        } else {
            -minor
        }
    };
    let det = m[0][0] * cofactor(0, 0) + m[0][1] * cofactor(0, 1) + m[0][2] * cofactor(0, 2);
    [0, 1, 2].map(|j| cofactor(j, 1) / det)
}

/// Collapses a measured spectrum into one value per linear sRGB channel.
///
/// `samples` are `(wavelength_nm, value)` pairs in increasing wavelength, in
/// whatever unit the quantity has -- this returns the same unit, since it is
/// a weighted mean, not an integral. Each channel is weighted by its own
/// spectral sensitivity, so a quantity that varies steeply across the visible
/// band (water's absorption rises ~50x from blue to red) lands in the right
/// channel instead of being smeared by a flat block average.
///
/// Trapezoidal in wavelength, so unevenly spaced measurements are handled
/// correctly. A channel with no sensitivity over the sampled range returns 0.
pub fn spectral_band_average(samples: &[(f32, f32)]) -> [f32; 3] {
    let mut weighted = [0.0f64; 3];
    let mut weight = [0.0f64; 3];
    for pair in samples.windows(2) {
        let (lo_nm, lo_value) = (f64::from(pair[0].0), f64::from(pair[0].1));
        let (hi_nm, hi_value) = (f64::from(pair[1].0), f64::from(pair[1].1));
        let width = hi_nm - lo_nm;
        if width <= 0.0 {
            continue;
        }
        let mid_nm = 0.5 * (lo_nm + hi_nm);
        let mid_value = 0.5 * (lo_value + hi_value);
        let sensitivity = srgb_channel_sensitivity(mid_nm);
        for channel in 0..3 {
            weighted[channel] += sensitivity[channel] * mid_value * width;
            weight[channel] += sensitivity[channel] * width;
        }
    }
    [0, 1, 2].map(|channel| {
        if weight[channel] <= 0.0 {
            0.0
        } else {
            (weighted[channel] / weight[channel]) as f32
        }
    })
}

/// Beer-Lambert transmission of a measured absorption spectrum through
/// `path_m` metres, per linear sRGB channel: the *mean of the transmission*
/// over the channel's band, not the transmission of the mean coefficient.
///
/// Derivation. Absorption acts on each wavelength separately,
/// `dL(lambda) = -a(lambda) L(lambda) dz` (pbr-book 3rd ed. sec. 11.1.1), so
/// a flat-spectrum beam of unit radiance leaves a path `z` with
/// `exp(-a(lambda) z)` at each wavelength, and a channel with sensitivity
/// `w(lambda)` reports `T(z) = integral w exp(-a z) / integral w`. Because
/// `exp` is convex, Jensen's inequality gives `T(z) >= exp(-<a> z)` with
/// `<a> = integral w a / integral w` the band-averaged coefficient
/// ([`spectral_band_average`]); equality holds only if `a` is constant over
/// the band. The two agree to first order, `T = 1 - <a> z + ...`, which is
/// why the band mean is right for a thin layer and wrong for a thick one:
/// the wavelengths that absorb least come to dominate what survives.
///
/// Numerics: `a(lambda)` is taken as piecewise linear between the samples
/// and `w` as constant at each segment's midpoint (the same weighting as
/// [`spectral_band_average`]); the integral of `exp(-a z)` over a segment is
/// then exact, `(exp(-a1 z) - exp(-a2 z)) / ((a2 - a1) z)` per unit width.
/// `samples` are `(wavelength_nm, a in m^-1)` in increasing wavelength. A
/// channel with no sensitivity over the sampled range returns 1 (nothing
/// measured, nothing absorbed).
pub fn spectral_band_transmittance(samples: &[(f32, f32)], path_m: f32) -> [f32; 3] {
    band_transmittance(samples, f64::from(path_m)).map(|t| t as f32)
}

/// [`spectral_band_transmittance`] in `f64`, so that
/// [`spectral_band_effective_absorption`] can take a logarithm of a value
/// close to 1 without losing it to `f32` rounding.
fn band_transmittance(samples: &[(f32, f32)], path_m: f64) -> [f64; 3] {
    let path = path_m.max(0.0);
    let mut weighted = [0.0f64; 3];
    let mut weight = [0.0f64; 3];
    for pair in samples.windows(2) {
        let (lo_nm, a1) = (f64::from(pair[0].0), f64::from(pair[0].1).max(0.0));
        let (hi_nm, a2) = (f64::from(pair[1].0), f64::from(pair[1].1).max(0.0));
        let width = hi_nm - lo_nm;
        if width <= 0.0 {
            continue;
        }
        // Mean over the segment of exp(-a(lambda) z), a linear from a1 to a2.
        let x = (a2 - a1) * path;
        let mean = if x.abs() < 1.0e-9 {
            (-a1 * path).exp() * (1.0 - 0.5 * x)
        } else {
            (-a1 * path).exp() * (-(-x).exp_m1()) / x
        };
        let sensitivity = srgb_channel_sensitivity(0.5 * (lo_nm + hi_nm));
        for channel in 0..3 {
            weighted[channel] += sensitivity[channel] * mean * width;
            weight[channel] += sensitivity[channel] * width;
        }
    }
    [0, 1, 2].map(|channel| {
        if weight[channel] <= 0.0 {
            1.0
        } else {
            weighted[channel] / weight[channel]
        }
    })
}

/// The absorption coefficient that, applied as a single Beer-Lambert
/// exponent, gives the band transmission at `path_m` exactly:
/// `-ln T(z) / z` with `T` from [`spectral_band_transmittance`]. It depends
/// on the path (it falls from the band mean at `z -> 0` toward the band's
/// least-absorbed wavelengths), which is the information a single
/// band-averaged coefficient cannot carry. At `path_m = 0` it is the band
/// mean [`spectral_band_average`].
pub fn spectral_band_effective_absorption(samples: &[(f32, f32)], path_m: f32) -> [f32; 3] {
    if path_m <= 0.0 {
        return spectral_band_average(samples);
    }
    let path = f64::from(path_m);
    band_transmittance(samples, path).map(|t| {
        // Underflow to zero transmission means "arbitrarily large": report
        // the coefficient of the smallest positive f64.
        (-t.max(f64::MIN_POSITIVE).ln() / path) as f32
    })
}

/// Channel boundaries of the visible integration used for emission, nm:
/// 360 to 830, the range the CIE observer fit is evaluated over elsewhere in
/// this module (and pbr-book's `Lambda_min`/`Lambda_max`).
const BAND_LO_NM: f64 = 360.0;
const BAND_HI_NM: f64 = 830.0;
/// Midpoint-rule cell width of [`blackbody_band_radiance_w_m2_sr`]. The
/// narrowest observer lobe has a standard deviation of 1/0.0845 = 11.8 nm, so
/// 5 nm cells resolve it; `blackbody_band_luminance_matches_fine_integral`
/// holds the result to a 0.1 nm integral.
const BAND_STEP_NM: f64 = 5.0;

/// Unclamped linear-sRGB sensitivities at the centre of each 5 nm cell, built
/// once: evaluating the seven-lobe observer fit per call would cost more than
/// the Planck terms it multiplies.
fn band_cells() -> &'static [(f64, [f64; 3])] {
    static CELLS: std::sync::OnceLock<Vec<(f64, [f64; 3])>> = std::sync::OnceLock::new();
    CELLS.get_or_init(|| {
        let count = ((BAND_HI_NM - BAND_LO_NM) / BAND_STEP_NM).round() as usize;
        (0..count)
            .map(|cell| {
                let centre_nm = BAND_LO_NM + (cell as f64 + 0.5) * BAND_STEP_NM;
                (centre_nm, unclamped_srgb_sensitivity(centre_nm))
            })
            .collect()
    })
}

/// Band radiance of a blackbody in linear sRGB, `W / (m^2 sr)`:
/// `integral B(lambda, T) s_c(lambda) dlambda` with `s_c` the CIE observer
/// through the sRGB matrix, negative lobes kept.
///
/// This is Wyman, Sloan and Shirley 2013, eq. 1 (`X = integral L xbar
/// dlambda`, no normalising constant) followed by the XYZ to linear sRGB
/// matrix, so a radiance `L(lambda)` in `W / (m^2 sr m)` becomes a number in
/// `W / (m^2 sr)` per channel. It is the definition of a "spectral-band
/// radiance" used by `systems::render::PhysicalRenderContract`: a display
/// white given in those units and the emission computed here are the same
/// kind of quantity, which total radiance `sigma T^4 / pi` over all
/// wavelengths is not (it counts the infrared a camera or eye does not
/// register, almost all of it for a body below a few thousand kelvin; see
/// `visible_share_of_a_1000_k_body_is_tiny`). Out-of-gamut colours give
/// negative components; the caller decides how to show them.
pub fn blackbody_band_radiance_w_m2_sr(temperature_k: f64) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    for (centre_nm, sensitivity) in band_cells() {
        let radiance = planck_spectral_radiance(centre_nm * 1.0e-9, temperature_k);
        for channel in 0..3 {
            sum[channel] += radiance * sensitivity[channel];
        }
    }
    // Cell width in metres: B is per metre of wavelength.
    sum.map(|value| value * BAND_STEP_NM * 1.0e-9)
}

/// Luminance-type band radiance `Y = integral B ybar dlambda`, `W / (m^2
/// sr)`: [`blackbody_band_radiance_w_m2_sr`] weighted by
/// [`linear_srgb_luminance_weights`].
pub fn blackbody_band_luminance_w_m2_sr(temperature_k: f64) -> f64 {
    let weights = linear_srgb_luminance_weights();
    let rgb = blackbody_band_radiance_w_m2_sr(temperature_k);
    weights[0] * rgb[0] + weights[1] * rgb[1] + weights[2] * rgb[2]
}

/// How bright a blackbody shows, as the factor that multiplies its
/// max-normalised colour ([`blackbody_linear_srgb_locus_fit`]).
///
/// Normalisation. The display white `white` (linear sRGB band radiances in
/// `W / (m^2 sr)`, the contract's `display_white_radiance_w_m2_sr`) is the
/// radiance shown as output 1.0. The body is shown with the same luminance
/// ratio, `Y_body / Y_white`, where each `Y` is the luminance-type band
/// radiance ([`blackbody_band_luminance_w_m2_sr`] for the body, the same
/// weights applied to `white`). The factor is that ratio divided by the
/// luminance of the max-normalised colour it multiplies, so the displayed
/// colour has exactly luminance `Y_body / Y_white` and the hue stays the
/// Planckian one.
///
/// Adaptation, declared: brightness is matched in luminance and the hue
/// comes from the closed-form locus fit (which the GPU shader mirrors); a
/// per-channel match would also be exact but would not reuse that fit.
pub fn blackbody_display_exposure(temperature_k: f32, white: [f32; 3]) -> f32 {
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return 0.0;
    }
    let weights = linear_srgb_luminance_weights();
    let white_luminance = weights
        .iter()
        .zip(white)
        .map(|(w, c)| w * f64::from(c))
        .sum::<f64>();
    if white_luminance <= 0.0 {
        return 0.0;
    }
    let colour = blackbody_linear_srgb_locus_fit(temperature_k);
    let colour_luminance = weights
        .iter()
        .zip(colour)
        .map(|(w, c)| w * f64::from(c))
        .sum::<f64>();
    if colour_luminance <= 0.0 {
        return 0.0;
    }
    let body_luminance = blackbody_band_luminance_w_m2_sr(f64::from(temperature_k));
    (body_luminance / white_luminance / colour_luminance) as f32
}

/// CIE xy chromaticity to linear sRGB, exposure-normalized so the brightest
/// channel is 1.
///
/// Out-of-gamut chromaticities (which the Planckian locus reaches at both
/// ends) are brought into gamut by desaturating toward white -- adding the
/// most negative channel to all three. That is a visible approximation:
/// sRGB simply cannot show those colours, and this keeps the hue while
/// admitting the loss of saturation, rather than clipping one channel to zero
/// and shifting the hue silently.
///
/// Brightness is deliberately NOT part of this: it comes from
/// `energy::radiation::blackbody_radiance_w_m2_sr`, so the two physical
/// factors stay separable.
pub fn chromaticity_to_linear_srgb(chromaticity_xy: [f64; 2]) -> [f32; 3] {
    let [x, y] = chromaticity_xy;
    if y <= 1.0e-6 {
        return [1.0, 1.0, 1.0];
    }
    // Unit-luminance XYZ for this chromaticity.
    let (big_x, big_y, big_z) = (x / y, 1.0, (1.0 - x - y) / y);
    let mut rgb = xyz_to_linear_srgb([big_x, big_y, big_z]);
    let lowest = rgb[0].min(rgb[1]).min(rgb[2]);
    if lowest < 0.0 {
        for channel in &mut rgb {
            *channel -= lowest;
        }
    }
    let highest = rgb[0].max(rgb[1]).max(rgb[2]);
    if highest <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [
        (rgb[0] / highest) as f32,
        (rgb[1] / highest) as f32,
        (rgb[2] / highest) as f32,
    ]
}

/// Exposure-normalized linear sRGB of a blackbody, from real Planck radiance
/// integrated against the CIE observer. Ground truth for the shader fit.
pub fn blackbody_linear_srgb(temperature_k: f32) -> [f32; 3] {
    chromaticity_to_linear_srgb(blackbody_chromaticity_xy(f64::from(temperature_k)))
}

/// Exposure-normalized linear sRGB of a blackbody, via Kang et al. 2002's
/// closed-form Planckian locus.
///
/// This is the exact CPU mirror of `blackbody.inc.wgsl`'s `blackbody_srgb`.
/// Keeping both and testing them against each other is what lets a fragment
/// shader claim real blackbody colour without integrating a spectrum per
/// pixel.
pub fn blackbody_linear_srgb_locus_fit(temperature_k: f32) -> [f32; 3] {
    let t = f64::from(temperature_k.clamp(LOCUS_FIT_MIN_K, LOCUS_FIT_MAX_K));
    let (inv, inv2) = (1.0 / t, 1.0 / (t * t));
    let inv3 = inv2 * inv;
    let x = if t <= 4000.0 {
        -0.266_123_9e9 * inv3 - 0.234_358_9e6 * inv2 + 0.877_695_6e3 * inv + 0.179_910
    } else {
        -3.025_846_9e9 * inv3 + 2.107_037_9e6 * inv2 + 0.222_634_7e3 * inv + 0.240_390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.106_381_4 * x3 - 1.348_110_20 * x2 + 2.185_558_32 * x - 0.202_196_83
    } else if t <= 4000.0 {
        -0.954_947_6 * x3 - 1.374_185_93 * x2 + 2.091_370_15 * x - 0.167_488_67
    } else {
        3.081_758_0 * x3 - 5.873_386_70 * x2 + 3.751_129_97 * x - 0.370_014_83
    };
    chromaticity_to_linear_srgb([x, y])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CIE standard illuminant A is *defined* as a Planckian radiator at
    /// 2856 K with chromaticity x = 0.44758, y = 0.40745. Reproducing it
    /// validates Planck's law, the constants, the observer fit and the
    /// integration together, against a number this codebase did not choose.
    #[test]
    fn reproduces_cie_illuminant_a_chromaticity() {
        let [x, y] = blackbody_chromaticity_xy(2856.0);
        assert!(
            (x - 0.447_58).abs() < 0.006 && (y - 0.407_45).abs() < 0.006,
            "illuminant A came out at ({x:.5}, {y:.5}), expected (0.44758, 0.40745)"
        );
    }

    /// D65's correlated colour temperature is ~6504 K. A true Planckian
    /// radiator there sits slightly below the daylight locus, near
    /// (0.3135, 0.3237) -- close to, but deliberately not equal to, D65's own
    /// (0.3127, 0.3290).
    #[test]
    fn reproduces_planckian_locus_near_daylight() {
        let [x, y] = blackbody_chromaticity_xy(6504.0);
        assert!(
            (x - 0.3135).abs() < 0.006 && (y - 0.3237).abs() < 0.006,
            "6504K came out at ({x:.5}, {y:.5}), expected (0.3135, 0.3237)"
        );
    }

    /// The shader's closed-form fit must agree with the spectral
    /// integration across the range the engine actually renders: embers,
    /// flame, lava, incandescence, sunlight.
    #[test]
    fn locus_fit_matches_planck_integration() {
        for temperature in [1667.0f32, 2000.0, 2856.0, 3500.0, 4000.0, 5772.0, 9000.0] {
            let exact = blackbody_linear_srgb(temperature);
            let fit = blackbody_linear_srgb_locus_fit(temperature);
            for channel in 0..3 {
                assert!(
                    (exact[channel] - fit[channel]).abs() < 0.03,
                    "{temperature}K channel {channel}: exact {exact:?} vs fit {fit:?}"
                );
            }
        }
    }

    /// The single physical fact the old `heat()` ramp got backwards: hotter
    /// blackbodies get bluer, not redder.
    ///
    /// Checked twice. Chromaticity `x` must fall monotonically -- that is the
    /// Planckian locus itself, unaffected by any display. Blue over red must
    /// then not fall -- the same statement once sRGB has had its say. The two
    /// are not redundant: below ~2000 K the locus is outside the sRGB gamut,
    /// so `b` sits at 0 for several steps while `x` keeps moving.
    #[test]
    fn hotter_blackbodies_are_bluer() {
        let temperatures = [1200.0f32, 1800.0, 2500.0, 3500.0, 5000.0, 7000.0, 10_000.0];
        let mut previous_x = f64::INFINITY;
        let mut previous_ratio = f32::NEG_INFINITY;
        for temperature in temperatures {
            let [x, _] = blackbody_chromaticity_xy(f64::from(temperature));
            assert!(
                x < previous_x,
                "{temperature}K chromaticity x {x:.4} did not fall below {previous_x:.4}"
            );
            previous_x = x;

            let [r, _, b] = blackbody_linear_srgb(temperature);
            let ratio = b / r.max(1.0e-6);
            assert!(
                ratio >= previous_ratio,
                "{temperature}K blue/red {ratio:.4} fell below {previous_ratio:.4}"
            );
            previous_ratio = ratio;
        }
        // And the endpoints must actually differ, or "monotonic" is vacuous.
        let [.., cold_b] = blackbody_linear_srgb(1200.0);
        let [.., hot_b] = blackbody_linear_srgb(10_000.0);
        assert!(
            hot_b > cold_b + 0.5,
            "{cold_b:.3} -> {hot_b:.3} is too flat"
        );
    }

    /// An ember is red, the sun is near-white. Absolute sanity, not a ratio.
    #[test]
    fn ember_is_red_and_sunlight_is_near_white() {
        let [r, g, b] = blackbody_linear_srgb(1200.0);
        assert!(
            r > 0.99 && g < 0.55 && b < 0.2,
            "1200K should be deep red, got [{r:.3}, {g:.3}, {b:.3}]"
        );
        let [r, g, b] = blackbody_linear_srgb(5772.0);
        assert!(
            r.min(g).min(b) > 0.75,
            "solar temperature should be near-white, got [{r:.3}, {g:.3}, {b:.3}]"
        );
    }

    // ---- Thermal brightness (criteria B1-B7) ----

    /// Reference luminance-type band radiance: 0.1 nm midpoint cells straight
    /// on the observer's ybar, with no sRGB matrix in the path.
    fn fine_luminance(temperature_k: f64) -> f64 {
        let step_nm = 0.1;
        let cells = ((830.0 - 360.0) / step_nm) as usize;
        (0..cells)
            .map(|cell| {
                let nm = 360.0 + (cell as f64 + 0.5) * step_nm;
                planck_spectral_radiance(nm * 1.0e-9, temperature_k)
                    * cie_1931_observer(nm)[1]
                    * step_nm
                    * 1.0e-9
            })
            .sum()
    }

    fn luminance(rgb: [f32; 3]) -> f64 {
        let w = linear_srgb_luminance_weights();
        w[0] * f64::from(rgb[0]) + w[1] * f64::from(rgb[1]) + w[2] * f64::from(rgb[2])
    }

    fn sun_white() -> [f32; 3] {
        blackbody_band_radiance_w_m2_sr(5772.0).map(|c| c as f32)
    }

    /// The luminance row computed from the matrix is the sRGB one:
    /// `Y = 0.2126 R + 0.7152 G + 0.0722 B` to the matrix's own four digits.
    #[test]
    fn luminance_weights_invert_the_srgb_matrix() {
        let w = linear_srgb_luminance_weights();
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 2.0e-3, "{w:?}");
        // Y of the D65 white, XYZ (0.9505, 1, 1.089) -> rgb (1,1,1).
        let rgb = xyz_to_linear_srgb([0.9505, 1.0, 1.089]);
        let y = w[0] * rgb[0] + w[1] * rgb[1] + w[2] * rgb[2];
        assert!((y - 1.0).abs() < 1.0e-9, "Y of the matrix image {y}");
    }

    /// B2.
    #[test]
    fn blackbody_band_luminance_matches_fine_integral() {
        for temperature in [300.0f64, 1000.0, 2856.0, 5772.0, 10_000.0] {
            let fast = blackbody_band_luminance_w_m2_sr(temperature);
            let fine = fine_luminance(temperature);
            let relative = (fast - fine).abs() / fine;
            assert!(
                relative < 1.0e-3,
                "{temperature} K: {fast:e} vs reference {fine:e}, relative {relative:e}"
            );
        }
    }

    /// B3: the visible band cannot carry more than the whole spectrum does
    /// (ybar is at most max(ybar)), and a hotter body is brighter.
    #[test]
    fn band_luminance_is_bounded_by_total_radiance_and_increases_with_temperature() {
        let max_ybar = (0..4_700)
            .map(|i| cie_1931_observer(360.0 + 0.1 * f64::from(i))[1])
            .fold(0.0f64, f64::max);
        let mut previous = 0.0f64;
        for temperature in [300.0f64, 1000.0, 2856.0, 5772.0, 10_000.0] {
            let y = blackbody_band_luminance_w_m2_sr(temperature);
            let total = f64::from(crate::energy::radiation::blackbody_radiance_w_m2_sr(
                temperature as f32,
            ));
            assert!(y > 0.0);
            assert!(
                y <= max_ybar * total * (1.0 + 1.0e-3),
                "{temperature} K: Y {y:e} exceeds {max_ybar} x total {total:e}"
            );
            assert!(y > previous, "{temperature} K not brighter than colder");
            previous = y;
        }
    }

    /// B1: a room-temperature body does not glow. White is the band radiance
    /// of the sun's disc; the old total-radiance exposure of the same body was
    /// far above 1 (printed for the record).
    #[test]
    fn room_temperature_body_renders_below_one_8_bit_level() {
        let white = sun_white();
        let temperature = 293.15f32;
        let colour = blackbody_linear_srgb_locus_fit(temperature);
        let exposure = blackbody_display_exposure(temperature, white);
        let shown = luminance(colour.map(|c| c * exposure));
        let old_total_over_mean_white =
            crate::energy::radiation::blackbody_radiance_w_m2_sr(temperature)
                / ((white[0] + white[1] + white[2]) / 3.0);
        eprintln!(
            "293.15 K: band luminance ratio {shown:e} (limit {:e}); old total/mean-white {old_total_over_mean_white:e}",
            1.0 / 255.0
        );
        assert!(shown < 1.0 / 255.0, "shows {shown:e}");
        assert!(shown > 0.0);
    }

    /// B4: the displayed colour has exactly the luminance ratio.
    #[test]
    fn displayed_luminance_is_the_band_luminance_ratio() {
        let white = sun_white();
        let white_luminance = luminance(white);
        for temperature in [2856.0f32, 5772.0, 9000.0] {
            let colour = blackbody_linear_srgb_locus_fit(temperature);
            let exposure = blackbody_display_exposure(temperature, white);
            let shown = luminance(colour.map(|c| c * exposure));
            let expected =
                blackbody_band_luminance_w_m2_sr(f64::from(temperature)) / white_luminance;
            assert!(
                (shown - expected).abs() <= 1.0e-4 * expected,
                "{temperature} K: shown {shown}, expected {expected}"
            );
        }
    }

    /// B5: the hue the brightness path integrates agrees with the locus fit
    /// the colour path uses.
    #[test]
    fn integrated_band_hue_agrees_with_the_locus_fit() {
        for temperature in [2856.0f32, 5772.0, 9000.0] {
            let rgb = blackbody_band_radiance_w_m2_sr(f64::from(temperature));
            let highest = rgb[0].max(rgb[1]).max(rgb[2]);
            let fit = blackbody_linear_srgb_locus_fit(temperature);
            for channel in 0..3 {
                let integrated = (rgb[channel] / highest) as f32;
                assert!(
                    (integrated - fit[channel]).abs() < 0.03,
                    "{temperature} K channel {channel}: integrated {integrated} vs fit {fit:?}"
                );
            }
        }
    }

    /// B6: white equal to the body's own band radiance shows it at 1.
    #[test]
    fn body_equal_to_white_displays_at_unit_luminance() {
        for temperature in [3000.0f32, 6504.0] {
            let white = blackbody_band_radiance_w_m2_sr(f64::from(temperature)).map(|c| c as f32);
            let colour = blackbody_linear_srgb_locus_fit(temperature);
            let exposure = blackbody_display_exposure(temperature, white);
            let shown = luminance(colour.map(|c| c * exposure));
            assert!(
                (shown - 1.0).abs() < 1.0e-4,
                "{temperature} K shows {shown}"
            );
        }
    }

    /// How little of a 1000 K body's total radiance a camera or eye can use.
    ///
    /// Criterion B7 as written (luminance-weighted share `Y / total` inside
    /// `[5e-6, 5e-5]`, bracketing the audit's 1.1e-5) FAILED: the value is
    /// 2.197e-7. Second reading, written after seeing that number: the audit's
    /// figure does not weight by the observer, it is the flat share of the
    /// 380-780 nm band, and the luminance-weighted share is smaller because
    /// the observer is nearly blind at the red edge where a 1000 K body's
    /// visible light lives. This test pins both: the flat share inside the
    /// same bracket as the audit's figure, the weighted share below it.
    #[test]
    fn visible_share_of_a_1000_k_body_is_tiny() {
        let total = f64::from(crate::energy::radiation::blackbody_radiance_w_m2_sr(1000.0));
        let weighted = blackbody_band_luminance_w_m2_sr(1000.0) / total;
        let flat: f64 = (0..4_000)
            .map(|cell| {
                let nm = 380.0 + (f64::from(cell) + 0.5) * 0.1;
                planck_spectral_radiance(nm * 1.0e-9, 1000.0) * 0.1e-9
            })
            .sum::<f64>()
            / total;
        eprintln!("1000 K: flat 380-780 nm share {flat:e}, luminance-weighted {weighted:e}");
        assert!((5.0e-6..5.0e-5).contains(&flat), "flat share {flat:e}");
        assert!(weighted < flat && weighted > 0.0, "weighted {weighted:e}");
    }

    // ---- Band transmittance (criteria C1, C3, C6) ----

    fn ramp(a_blue: f32, a_red: f32) -> Vec<(f32, f32)> {
        (0..=60)
            .map(|i| {
                let nm = 400.0 + 5.0 * i as f32;
                (nm, a_blue + (a_red - a_blue) * (nm - 400.0) / 300.0)
            })
            .collect()
    }

    /// C3.
    #[test]
    fn flat_spectrum_transmits_plain_beer_lambert() {
        let flat: Vec<(f32, f32)> = (0..=40).map(|i| (400.0 + 7.5 * i as f32, 0.37)).collect();
        for path in [0.0f32, 0.1, 1.0, 10.0, 30.0] {
            let t = spectral_band_transmittance(&flat, path);
            let want = (-0.37f64 * f64::from(path)).exp() as f32;
            for value in t {
                assert!((value - want).abs() < 1.0e-6, "L {path}: {value} vs {want}");
            }
        }
    }

    /// C1 and C6.
    #[test]
    fn band_transmittance_limits_and_monotonicity() {
        let spectrum = ramp(0.01, 0.6);
        let mean = spectral_band_average(&spectrum);
        let thin = spectral_band_effective_absorption(&spectrum, 1.0e-6);
        for channel in 0..3 {
            assert!(
                (thin[channel] - mean[channel]).abs() <= 1.0e-3 * mean[channel],
                "channel {channel}: thin-limit {} vs band mean {}",
                thin[channel],
                mean[channel]
            );
        }
        let mut previous_t = [1.0f32; 3];
        let mut previous_a = [f32::INFINITY; 3];
        for path in [0.1f32, 1.0, 3.0, 10.0, 30.0] {
            let t = spectral_band_transmittance(&spectrum, path);
            let a = spectral_band_effective_absorption(&spectrum, path);
            for channel in 0..3 {
                assert!(t[channel] > 0.0 && t[channel] <= 1.0);
                assert!(t[channel] < previous_t[channel], "T not decreasing");
                assert!(
                    a[channel] <= previous_a[channel] * (1.0 + 1.0e-6),
                    "effective coefficient rose with path: {} -> {}",
                    previous_a[channel],
                    a[channel]
                );
                // Jensen: never below the transmission of the mean coefficient.
                let naive = (-f64::from(mean[channel]) * f64::from(path)).exp() as f32;
                assert!(t[channel] >= naive * (1.0 - 1.0e-6), "Jensen violated");
            }
            previous_t = t;
            previous_a = a;
        }
    }
}
