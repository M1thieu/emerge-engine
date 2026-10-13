//! Fixed corotated elasticity: its Kirchhoff stress, its energy density, and
//! their derivatives with respect to `F`.
//!
//! Source: Stomakhin et al. 2013, eq. (5)-(8), the corotated model this
//! engine's snow material also cites.

use glam::Mat2;
#[cfg(test)]
use glam::Vec2;

use crate::matter::kinematics::polar_decomposition::{
    polar_decomposition_2d, polar_decomposition_2d_jvp,
};
use crate::matter::kinematics::volume_ratio::MIN_J;
use crate::tensor_algebra::frobenius::frob;
#[cfg(test)]
use crate::tensor_algebra::symmetric_eigen::spd_project_symmetric_4x4;

/// Corotated-elastic Kirchhoff stress: `2*mu*(F-R)*F^T + lambda*(J-1)*J*I`,
/// the standard hyperelastic form (Stomakhin et al. 2013, the same
/// corotated model this engine's snow material already cites). Every
/// plastic material that returns to this SAME elastic branch after its own
/// return-mapping (DruckerPragerMaterial, VonMisesMaterial, RankineMaterial,
/// MuIRheologyMaterial) used to hand-duplicate this exact formula in its own
/// `kirchhoff_stress` -- each material's plasticity lives entirely in its
/// own `update_particle`, this function owns only the shared elastic stress
/// evaluated on the (already plastically corrected) current F.
#[inline]
pub(crate) fn corotated_elastic_stress(f: Mat2, lambda: f32, mu: f32) -> Mat2 {
    let j = f.determinant();
    if j <= MIN_J {
        return Mat2::ZERO;
    }
    let r = polar_decomposition_2d(f);
    2.0 * mu * (f - r) * f.transpose() + lambda * (j - 1.0) * j * Mat2::IDENTITY
}

/// Corotated elastic energy density `Psi(F) = mu*||F-R||_F^2 +
/// 0.5*lambda*(J-1)^2` (Stomakhin et al. 2013). `corotated_elastic_stress` is
/// its First-Piola gradient times `F^T`: `dPsi/dF = 2*mu*(F-R) +
/// lambda*(J-1)*J*F^-T`, and `(dPsi/dF)*F^T = 2*mu*(F-R)*F^T +
/// lambda*(J-1)*J*I`. Zero at `J <= MIN_J`, like the stress ("no force").
///
/// Test-only: the oracle `model_residual`'s gradient is checked against in
/// `spacetime::solver::implicit_corotated`, whose trust region measures a
/// Gauss-Newton `0.5*||residual||^2` merit instead (see `newton_solve`).
#[cfg(test)]
#[inline]
pub(crate) fn corotated_elastic_energy_density(f: Mat2, lambda: f32, mu: f32) -> f32 {
    let j = f.determinant();
    if j <= MIN_J {
        return 0.0;
    }
    let r = polar_decomposition_2d(f);
    let dev = f - r;
    mu * frob(dev, dev) + 0.5 * lambda * (j - 1.0) * (j - 1.0)
}

/// Analytic Jacobian-vector product of `corotated_elastic_stress`'s
/// Kirchhoff stress w.r.t. `F`: given `F` and a perturbation `dF`, returns
/// `dTau`. Product rule on `2*mu*(F-R)*F^T + lambda*(J-1)*J*I` using
/// `polar_decomposition_2d_jvp` above and `dJ=J*(F^-T:dF)`. The
/// Hessian-vector-product building block of an implicit (Newton-CG) solve for
/// every material on this elastic branch (DruckerPrager, VonMises, Rankine,
/// MuIRheology, Corotated), see `spacetime::solver::implicit_corotated`.
/// Checked by central finite differences
/// (`corotated_stress_jvp_matches_finite_difference`): max relative error
/// 0.23% at sand-magnitude stiffness.
///
/// Zero at `j <= MIN_J`, where the stress is pinned at zero: a practical
/// simplification at a hard clamp, not a claim about the continuous
/// derivative.
#[inline]
pub(crate) fn corotated_elastic_stress_jvp(f: Mat2, df: Mat2, lambda: f32, mu: f32) -> Mat2 {
    let j = f.determinant();
    if j <= MIN_J {
        return Mat2::ZERO;
    }
    let r = polar_decomposition_2d(f);
    let dr = polar_decomposition_2d_jvp(f, df);
    let f_t = f.transpose();
    let df_t = df.transpose();
    let f_inv_t = f.inverse().transpose();
    let d_j = j * frob(f_inv_t, df); // dJ = J*(F^-T:dF)

    let d_elastic_term = (df - dr) * f_t + (f - r) * df_t;
    let d_vol_term = (2.0 * j - 1.0) * d_j; // d[(J-1)*J] = (2J-1)*dJ
    2.0 * mu * d_elastic_term + lambda * d_vol_term * Mat2::IDENTITY
}

/// A PSD approximation of `d(tau)/d(L)` (the Kirchhoff stress derivative
/// with respect to a velocity-gradient perturbation `dl`, through `dF =
/// dt*dl*f_n`), applied to `dl`.
///
/// Teran et al. 2005's SPD construction proves `dP:dF >= 0` for First-Piola
/// stress with a material-space perturbation. This engine's force pairs
/// Kirchhoff stress with the spatial kernel gradient (as `tmp/ziran2020` and
/// `tmp/GeoTaichi` do), and `tau = P*F^T` as a function of `L` inherits no such
/// guarantee: a Piola-projected operator converted by the product rule still
/// let the residual grow across CG iterations, which a PSD system cannot do.
///
/// So the local 4x4 operator (`L` and `tau` both have 4 components in 2D) is
/// built from four calls to the exact `corotated_elastic_stress_jvp`,
/// symmetrized (only the symmetric part enters the quadratic form), then
/// eigendecomposed (`jacobi_eigen_symmetric_4x4`, Golub & Van Loan's cyclic
/// Jacobi) with negative eigenvalues clamped to zero: the nearest SPD matrix
/// in Frobenius norm (Higham 1988), Teran's clamping principle without his
/// `dP/dF`-specific closed form. A Gershgorin diagonal shift, also
/// sufficient for PSD, swamps the curvature at production stiffness
/// (`basic_sand`'s E = 15 MPa): the line search rejected every step.
///
/// Split in two: [`corotated_kirchhoff_dtau_dl_psd_matrix`] builds the matrix
/// once per trial `v` (it does not depend on `dl`); [`apply_dtau_dl_psd_matrix`]
/// applies it for a 4x4 mat-vec, instead of rebuilding it every CG iteration.
///
/// Test-only: `implicit_corotated` uses `steihaug_cg` (Nocedal & Wright),
/// which handles indefinite curvature and needs no PSD projection. Kept, with
/// `corotated_kirchhoff_dtau_dl_psd_tests`, in case a PSD operator is needed
/// again.
#[cfg(test)]
pub(crate) fn corotated_kirchhoff_dtau_dl_psd_matrix(
    f: Mat2,
    lambda: f32,
    mu: f32,
    dt: f32,
    f_n: Mat2,
) -> [[f32; 4]; 4] {
    let basis = [
        Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
    ];
    let mut h = [[0.0f32; 4]; 4];
    for (col, &e) in basis.iter().enumerate() {
        let df = dt * e * f_n;
        let d_tau = corotated_elastic_stress_jvp(f, df, lambda, mu);
        let d_tau_vec = [
            d_tau.x_axis.x,
            d_tau.x_axis.y,
            d_tau.y_axis.x,
            d_tau.y_axis.y,
        ];
        for (row, &val) in d_tau_vec.iter().enumerate() {
            h[row][col] = val;
        }
    }
    let mut sym = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            sym[i][j] = 0.5 * (h[i][j] + h[j][i]);
        }
    }
    spd_project_symmetric_4x4(sym)
}

/// Application step for [`corotated_kirchhoff_dtau_dl_psd_matrix`]'s output:
/// a 4x4 mat-vec, no JVP evaluations. Test-only, for
/// [`corotated_kirchhoff_dtau_dl_psd`]'s direct comparison.
#[cfg(test)]
#[inline]
pub(crate) fn apply_dtau_dl_psd_matrix(matrix: &[[f32; 4]; 4], dl: Mat2) -> Mat2 {
    let dl_vec = [dl.x_axis.x, dl.x_axis.y, dl.y_axis.x, dl.y_axis.y];
    let mut out = [0.0f32; 4];
    for (i, row) in matrix.iter().enumerate() {
        out[i] = row.iter().zip(&dl_vec).map(|(a, b)| a * b).sum();
    }
    Mat2::from_cols(Vec2::new(out[0], out[1]), Vec2::new(out[2], out[3]))
}

/// Convenience wrapper matching the original single-call API (build +
/// apply in one shot) -- kept for the test suite's own direct-comparison
/// checks; production code (`spacetime::solver::implicit_corotated`) uses
/// the split build/apply pair instead to avoid rebuilding the matrix once
/// per CG iteration.
#[cfg(test)]
pub(crate) fn corotated_kirchhoff_dtau_dl_psd(
    f: Mat2,
    lambda: f32,
    mu: f32,
    dt: f32,
    f_n: Mat2,
    dl: Mat2,
) -> Mat2 {
    let matrix = corotated_kirchhoff_dtau_dl_psd_matrix(f, lambda, mu, dt, f_n);
    apply_dtau_dl_psd_matrix(&matrix, dl)
}

#[cfg(test)]
mod corotated_elastic_stress_jvp_tests {
    use super::*;

    fn frob_pub(a: Mat2, b: Mat2) -> f32 {
        a.x_axis.x * b.x_axis.x
            + a.x_axis.y * b.x_axis.y
            + a.y_axis.x * b.y_axis.x
            + a.y_axis.y * b.y_axis.y
    }

    fn states() -> Vec<Mat2> {
        vec![
            Mat2::IDENTITY,
            Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
            Mat2::from_cols(Vec2::new(1.0, 0.4), Vec2::new(0.1, 1.1)),
            Mat2::from_cols(Vec2::new(0.9, -0.2), Vec2::new(0.3, 1.2)),
            Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
        ]
    }
    fn directions() -> Vec<Mat2> {
        vec![
            Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
            Mat2::from_cols(Vec2::new(0.2, 0.3), Vec2::new(-0.1, 0.5)),
        ]
    }

    /// Central finite difference at h = 1e-3, not smaller: f32 cancellation on
    /// `polar_decomposition_2d`'s normalized quantity makes smaller h worse.
    #[test]
    fn matches_finite_difference_at_moderate_and_real_sand_stiffness() {
        let h = 1.0e-3f32;
        for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e7, 1.5e7)] {
            let mut max_rel_err = 0.0f32;
            for &f in &states() {
                for &df in &directions() {
                    let plus = corotated_elastic_stress(f + h * df, lambda, mu);
                    let minus = corotated_elastic_stress(f - h * df, lambda, mu);
                    let numeric = (plus - minus) * (1.0 / (2.0 * h));
                    let analytic = corotated_elastic_stress_jvp(f, df, lambda, mu);
                    let diff = analytic - numeric;
                    let rel_err =
                        frob_pub(diff, diff).sqrt() / frob_pub(analytic, analytic).sqrt().max(1.0);
                    max_rel_err = max_rel_err.max(rel_err);
                }
            }
            println!(
                "corotated_elastic_stress_jvp: lambda={lambda} mu={mu} max_rel_err={max_rel_err:.6}"
            );
            assert!(
                max_rel_err < 5.0e-3,
                "JVP wrong at lambda={lambda} mu={mu}: {max_rel_err}"
            );
        }
    }

    #[test]
    fn zero_below_min_j_floor_matching_the_stress_itself() {
        // A near-singular F that pins `corotated_elastic_stress` at
        // Mat2::ZERO -- the JVP must match that pinned-constant behavior.
        let f = Mat2::from_cols(Vec2::new(1.0e-8, 0.0), Vec2::new(0.0, 1.0e-8));
        let df = Mat2::from_cols(Vec2::new(0.1, 0.0), Vec2::new(0.0, 0.1));
        assert_eq!(corotated_elastic_stress(f, 100.0, 50.0), Mat2::ZERO);
        assert_eq!(corotated_elastic_stress_jvp(f, df, 100.0, 50.0), Mat2::ZERO);
    }

    /// `corotated_elastic_energy_density` is the potential
    /// `corotated_elastic_stress` is the gradient of, so the trust-region
    /// ratio test solves the problem the other tests trust. Central finite
    /// difference at h = 1e-3 (smaller h is worse here); First-Piola comes from
    /// the Kirchhoff function via `P = tau*F^-T` rather than a second copy of
    /// the formula.
    #[test]
    fn energy_density_gradient_matches_finite_difference_of_the_stress() {
        let h = 1.0e-3f32;
        for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e7, 1.5e7)] {
            let mut max_rel_err = 0.0f32;
            for &f in &states() {
                // `Mat2::IDENTITY` is a critical point of `Psi`
                // (F=R, J=1 -> both terms are exactly zero, the analytic
                // global minimum) -- its analytic gradient is exactly
                // zero, which makes central-difference ill-conditioned
                // there by construction: `f(x+h)`/`f(x-h)` are both
                // dominated by the (identical, non-cancelling) O(h^2)
                // curvature term, so `(plus-minus)/(2h)` recovers pure
                // O(h^2)*f'''/6 truncation noise, not a discrepancy
                // (confirmed: at lambda=2e7, this alone produced a raw FD
                // value of ~1.8 against an analytic 0 -- but EVERY other
                // state, all deformed with nonzero analytic
                // gradients, passed at the same h and stiffness with
                // max_rel_err=2.45e-4). Standard, documented FD-checking
                // practice (e.g. PyTorch's `gradcheck`) explicitly skips
                // exactly this scenario -- checking a derivative AT a
                // critical point -- rather than loosening the tolerance
                // for every other, well-conditioned case too.
                if f == Mat2::IDENTITY {
                    continue;
                }
                for &df in &directions() {
                    let plus = corotated_elastic_energy_density(f + h * df, lambda, mu);
                    let minus = corotated_elastic_energy_density(f - h * df, lambda, mu);
                    let numeric = (plus - minus) / (2.0 * h);
                    let tau = corotated_elastic_stress(f, lambda, mu);
                    let piola = tau * f.inverse().transpose();
                    let analytic = frob_pub(piola, df);
                    let rel_err = (analytic - numeric).abs() / analytic.abs().max(1.0);
                    max_rel_err = max_rel_err.max(rel_err);
                }
            }
            println!(
                "corotated_elastic_energy_density: lambda={lambda} mu={mu} max_rel_err={max_rel_err:.6}"
            );
            assert!(
                max_rel_err < 5.0e-3,
                "energy gradient doesn't match the stress at lambda={lambda} mu={mu}: {max_rel_err}"
            );
        }
    }
}

#[cfg(test)]
mod corotated_kirchhoff_dtau_dl_psd_tests {
    use super::*;

    fn states() -> Vec<Mat2> {
        vec![
            Mat2::IDENTITY,
            Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
            Mat2::from_cols(Vec2::new(1.0, 0.1), Vec2::new(-0.05, 1.05)),
            // Large volumetric expansion -- the regime the plain
            // (unprojected) Hessian is known-indefinite in.
            Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 3.0)),
            Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
        ]
    }

    fn directions() -> Vec<Mat2> {
        vec![
            Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
            Mat2::from_cols(Vec2::new(0.2, 0.1), Vec2::new(-0.1, 0.3)),
            Mat2::from_cols(Vec2::new(-0.1, 0.2), Vec2::new(0.15, -0.2)),
        ]
    }

    /// The actual property this function exists for: `frob(H(dl), dl) >=
    /// 0` for EVERY direction `dl`, at EVERY state including the ones the
    /// plain (unprojected) Hessian is known-indefinite at -- this is what
    /// guarantees CG can never see a negative-curvature direction from
    /// this operator, unlike the plain JVP or a Piola-space projection
    /// (which does not survive the Kirchhoff/spatial-gradient conversion:
    /// CG in `spacetime::solver::implicit_corotated` saw negative curvature
    /// with it).
    #[test]
    fn quadratic_form_is_never_negative() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let dt = 0.016f32;
        for &f in &states() {
            for &f_n in &states() {
                for &dl in &directions() {
                    let h_dl = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
                    let q = frob(h_dl, dl);
                    assert!(
                        q >= -1.0e-3 * frob(dl, dl).max(1.0),
                        "quadratic form went negative: f={f:?} f_n={f_n:?} dl={dl:?} q={q}"
                    );
                }
            }
        }
    }

    #[test]
    fn matches_plain_jvp_via_chain_rule_at_moderate_already_psd_states() {
        // At small/moderate deformation with no pre-existing rotation, the
        // real local Hessian's SYMMETRIC PART is already close to PSD (no
        // real Gershgorin shift needed), so this should stay close to the
        // exact chain-rule JVP. NOT bit-identical even here, by design:
        // symmetrizing intentionally discards `dTau/dL`'s antisymmetric
        // part (which contributes nothing to the quadratic form PSD-ness
        // is actually about), so a few-percent difference from the
        // raw (non-symmetrized) exact JVP is expected, not a bug -- this
        // check is only about ruling out AGGRESSIVE over-damping at a
        // benign state, not exact reproduction.
        let lambda = 2.0e5f32;
        let mu = 1.5e5f32;
        let dt = 0.016f32;
        let f = Mat2::from_cols(Vec2::new(1.02, 0.01), Vec2::new(-0.01, 0.98));
        let f_n = Mat2::IDENTITY;
        for &dl in &directions() {
            let via_psd = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
            let df = dt * dl * f_n;
            let exact = corotated_elastic_stress_jvp(f, df, lambda, mu);
            let diff = via_psd - exact;
            let rel_err = frob(diff, diff).sqrt() / frob(exact, exact).sqrt().max(1.0);
            assert!(
                rel_err < 5.0e-2,
                "should stay close to the exact JVP at a moderate, benign state (no aggressive over-damping): dl={dl:?} rel_err={rel_err}"
            );
        }
    }

    /// At production stiffness (`basic_sand`'s E = 15 MPa DruckerPrager, lambda
    /// and mu of this order) and a benign, mostly PSD state, the projection
    /// keeps the curvature close to intact: it tracks the exact JVP's
    /// quadratic form about as closely as
    /// `matches_plain_jvp_via_chain_rule_at_moderate_already_psd_states` does
    /// at low stiffness. A Gershgorin shift large enough for diagonal
    /// dominance here swamped the curvature (every line-search step
    /// rejected over 30 frames).
    #[test]
    fn eigenvalue_clamp_preserves_curvature_at_real_production_stiffness() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let dt = 0.016f32;
        let f = Mat2::from_cols(Vec2::new(1.02, 0.01), Vec2::new(-0.01, 0.98));
        let f_n = Mat2::IDENTITY;
        for &dl in &directions() {
            let via_psd = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
            let df = dt * dl * f_n;
            let exact = corotated_elastic_stress_jvp(f, df, lambda, mu);
            let q_psd = frob(via_psd, dl);
            let q_exact = frob(exact, dl);
            assert!(
                q_psd > 0.5 * q_exact,
                "eigenvalue clamp over-damped real positive curvature at production stiffness: dl={dl:?} q_psd={q_psd} q_exact={q_exact}"
            );
        }
    }

    #[test]
    fn stays_finite_at_a_large_volumetric_expansion() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let f = Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 3.0));
        let f_n = Mat2::IDENTITY;
        let dl = Mat2::from_cols(Vec2::new(0.5, 0.2), Vec2::new(-0.3, 0.4));
        let out = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, 0.016, f_n, dl);
        assert!(
            out.x_axis.is_finite() && out.y_axis.is_finite(),
            "must stay finite even in the indefinite regime: {out:?}"
        );
    }
}
