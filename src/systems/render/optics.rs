//! Physical optical inputs shared by every renderer path.
//!
//! The simulation is two-dimensional, while optical attenuation needs a
//! three-dimensional path length.  [`PhysicalRenderContract`] makes that
//! missing modelling choice explicit: one grid cell has a stated SI size and
//! the 2-D slice represents a slab with a stated out-of-plane thickness.
//! Nothing in this module silently assumes that a cell or particle is one
//! metre deep.

use std::{error::Error, fmt};

use glam::Vec3;

/// Validated SI inputs needed to turn simulated density into optical depth.
///
/// Radiance channels are linear RGB spectral-band radiances in
/// `W / (m^2 sr)`.  They are deliberately not display-encoded sRGB values.
/// The camera direction is normally perpendicular to the simulated plane;
/// retaining it explicitly prevents future refraction/reflection code from
/// synthesising a view direction inside a shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalRenderContract {
    dx_meters: f32,
    view_thickness_meters: f32,
    incident_radiance_w_m2_sr: [f32; 3],
    background_radiance_w_m2_sr: [f32; 3],
    display_white_radiance_w_m2_sr: [f32; 3],
    camera_direction: Vec3,
    light_direction: Vec3,
}

/// Unvalidated inputs to [`PhysicalRenderContract::new`] -- same real
/// struct-bundling fix already used elsewhere in this codebase
/// (`ContactKinematics`, `SubstepScene`/`SubstepBounds`,
/// `PhasePipelineBuffers`) for a constructor whose parameters are exactly
/// its own output fields, not a workaround for the lint alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicalRenderContractParams {
    pub dx_meters: f32,
    pub view_thickness_meters: f32,
    pub incident_radiance_w_m2_sr: [f32; 3],
    pub background_radiance_w_m2_sr: [f32; 3],
    pub display_white_radiance_w_m2_sr: [f32; 3],
    pub camera_direction: Vec3,
    pub light_direction: Vec3,
}

impl PhysicalRenderContract {
    pub fn new(params: PhysicalRenderContractParams) -> Result<Self, PhysicalRenderContractError> {
        let PhysicalRenderContractParams {
            dx_meters,
            view_thickness_meters,
            incident_radiance_w_m2_sr,
            background_radiance_w_m2_sr,
            display_white_radiance_w_m2_sr,
            camera_direction,
            light_direction,
        } = params;
        if !dx_meters.is_finite() || dx_meters <= 0.0 {
            return Err(PhysicalRenderContractError::NonPositiveDx);
        }
        if !view_thickness_meters.is_finite() || view_thickness_meters <= 0.0 {
            return Err(PhysicalRenderContractError::NonPositiveViewThickness);
        }
        validate_radiance(incident_radiance_w_m2_sr)?;
        validate_radiance(background_radiance_w_m2_sr)?;
        validate_positive_radiance(display_white_radiance_w_m2_sr)?;
        let camera_direction = normalize_direction(
            camera_direction,
            PhysicalRenderContractError::InvalidCameraDirection,
        )?;
        let light_direction = normalize_direction(
            light_direction,
            PhysicalRenderContractError::InvalidLightDirection,
        )?;

        Ok(Self {
            dx_meters,
            view_thickness_meters,
            incident_radiance_w_m2_sr,
            background_radiance_w_m2_sr,
            display_white_radiance_w_m2_sr,
            camera_direction,
            light_direction,
        })
    }

    pub fn dx_meters(self) -> f32 {
        self.dx_meters
    }

    pub fn view_thickness_meters(self) -> f32 {
        self.view_thickness_meters
    }

    pub fn incident_radiance_w_m2_sr(self) -> [f32; 3] {
        self.incident_radiance_w_m2_sr
    }

    pub fn background_radiance_w_m2_sr(self) -> [f32; 3] {
        self.background_radiance_w_m2_sr
    }

    /// Sensor/display radiance represented by linear output value 1.0 in
    /// each spectral band. This closes the dimensional gap between radiance
    /// transport and the renderer's dimensionless floating-point target.
    pub fn display_white_radiance_w_m2_sr(self) -> [f32; 3] {
        self.display_white_radiance_w_m2_sr
    }

    pub fn camera_direction(self) -> Vec3 {
        self.camera_direction
    }

    pub fn light_direction(self) -> Vec3 {
        self.light_direction
    }
}

/// Per-material absorption/scattering coefficients in inverse metres.
///
/// Reduced scattering remains one visible-band scalar for compatibility with
/// the current renderer.  The unit is explicit; a later spectral transport
/// upgrade can widen it without changing the absorption contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticalCoefficientsSi {
    pub absorption_m_inv: [f32; 3],
    pub reduced_scattering_m_inv: f32,
}

impl OpticalCoefficientsSi {
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
        })
    }
}

/// Exact absorption-only solution for a homogeneous slab.
///
/// `relative_density` is `rho/rho_ref`, so the exponent is dimensionless:
/// `(m^-1) * 1 * m`.  This is the reference used by CPU and GPU validation;
/// it does not include scattering, reflection, or emission.
pub fn beer_lambert_transmittance(
    absorption_m_inv: [f32; 3],
    relative_density: f32,
    path_length_meters: f32,
) -> [f32; 3] {
    let column_length = relative_density.max(0.0) * path_length_meters.max(0.0);
    absorption_m_inv.map(|sigma| (-sigma.max(0.0) * column_length).exp())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalRenderContractError {
    NonPositiveDx,
    NonPositiveViewThickness,
    InvalidRadiance,
    InvalidDisplayWhiteRadiance,
    InvalidCameraDirection,
    InvalidLightDirection,
}

impl fmt::Display for PhysicalRenderContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonPositiveDx => "dx_meters must be finite and positive",
            Self::NonPositiveViewThickness => "view_thickness_meters must be finite and positive",
            Self::InvalidRadiance => "radiance channels must be finite and non-negative",
            Self::InvalidDisplayWhiteRadiance => {
                "display-white radiance channels must be finite and positive"
            }
            Self::InvalidCameraDirection => "camera direction must be finite and non-zero",
            Self::InvalidLightDirection => "light direction must be finite and non-zero",
        };
        f.write_str(message)
    }
}

impl Error for PhysicalRenderContractError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpticalCoefficientsError {
    InvalidAbsorption,
    InvalidReducedScattering,
}

impl fmt::Display for OpticalCoefficientsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidAbsorption => "absorption coefficients must be finite and non-negative",
            Self::InvalidReducedScattering => "reduced scattering must be finite and non-negative",
        };
        f.write_str(message)
    }
}

impl Error for OpticalCoefficientsError {}

fn validate_radiance(radiance: [f32; 3]) -> Result<(), PhysicalRenderContractError> {
    if radiance
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PhysicalRenderContractError::InvalidRadiance);
    }
    Ok(())
}

fn validate_positive_radiance(radiance: [f32; 3]) -> Result<(), PhysicalRenderContractError> {
    if radiance
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(PhysicalRenderContractError::InvalidDisplayWhiteRadiance);
    }
    Ok(())
}

fn normalize_direction(
    direction: Vec3,
    error: PhysicalRenderContractError,
) -> Result<Vec3, PhysicalRenderContractError> {
    if !direction.is_finite() || direction.length_squared() <= f32::EPSILON {
        return Err(error);
    }
    Ok(direction.normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_rejects_missing_physical_scale() {
        let result = PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.0,
            view_thickness_meters: 0.01,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.0; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: Vec3::Z,
            light_direction: Vec3::Y,
        });
        assert_eq!(result, Err(PhysicalRenderContractError::NonPositiveDx));
    }

    #[test]
    fn contract_normalizes_real_directions() {
        let contract = PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.01,
            view_thickness_meters: 0.02,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.0; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: Vec3::new(0.0, 0.0, -4.0),
            light_direction: Vec3::new(3.0, 4.0, 0.0),
        })
        .unwrap();
        assert!((contract.camera_direction().length() - 1.0).abs() < 1.0e-6);
        assert!((contract.light_direction().length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn beer_lambert_matches_known_homogeneous_slab() {
        let transmittance = beer_lambert_transmittance([2.0, 1.0, 0.5], 1.0, 0.25);
        let expected = [(-0.5f32).exp(), (-0.25f32).exp(), (-0.125f32).exp()];
        for (got, want) in transmittance.into_iter().zip(expected) {
            assert!((got - want).abs() < 1.0e-6, "got {got}, expected {want}");
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
