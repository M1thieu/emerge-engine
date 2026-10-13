//! Polar decomposition `F = R U`: a deformation split into a pure rotation `R`
//! and a pure stretch `U`.
//!
//! Source: Gurtin, Fried & Anand (2010), ch. 7 (stretch, strain and rotation).

use glam::{Mat2, Vec2};

/// 2D polar decomposition: returns the rotation R such that F = R·S.
///
/// Uses the analytical formula for 2×2 matrices (no SVD needed):
///   x = F₀₀+F₁₁, y = F₁₀−F₀₁, norm = √(x²+y²), R = `[[x,−y],[y,x]]`/norm
/// Returns Mat2::IDENTITY when F is near-singular (norm ≤ ε).
/// Used identically by CorotatedMaterial, StomakhinMaterial, VonMisesMaterial, DruckerPragerMaterial.
pub fn polar_decomposition_2d(f: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    if norm > f32::EPSILON {
        Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
    } else {
        Mat2::IDENTITY
    }
}

/// Analytic Jacobian-vector product of `polar_decomposition_2d`: given `F`
/// and a perturbation direction `dF`, returns `dR`. Closed form for this
/// engine's `R = M/norm`, `M = [[x,-y],[y,x]]`, `x = tr(F)`, `y = F01-F10`
/// (the generic SVD-based polar derivative does not match it):
///   dx=tr(dF), dy=dF01-dF10, dM=[[dx,-dy],[dy,dx]]
///   d(norm)=(x*dx+y*dy)/norm,  dR=dM/norm - R*d(norm)/norm
/// Checked by central finite differences
/// (`polar_decomposition_jvp_matches_finite_difference`): max relative error
/// 0.06% at h = 1e-3. Smaller h is worse here (f32 cancellation on this
/// normalized quantity).
///
/// Used by `corotated_elastic_stress_jvp` below, which
/// `spacetime::solver::implicit_corotated` calls to build its local Hessian.
#[inline]
pub(crate) fn polar_decomposition_2d_jvp(f: Mat2, df: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    if norm <= f32::EPSILON {
        return Mat2::ZERO;
    }
    let dx = df.x_axis.x + df.y_axis.y;
    let dy = df.x_axis.y - df.y_axis.x;
    let d_norm = (x * dx + y * dy) / norm;
    let dm = Mat2::from_cols(Vec2::new(dx, dy), Vec2::new(-dy, dx));
    let r = Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm);
    dm * (1.0 / norm) - r * (d_norm / norm)
}
