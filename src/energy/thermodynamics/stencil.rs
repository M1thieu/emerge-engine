//! Shared 5-point explicit-Euler Laplacian diffusion stencil.
//!
//! The one piece of math identical between [`super::diffusion`]
//! and [`super::scalar_field`] -- both scatter a particle scalar to the grid,
//! run this stencil, then gather the delta back. What differs between them
//! (direct-field access vs. runtime fn-pointer access, decay-to-zero vs.
//! Newton-cooling-to-ambient) is not accidental duplication -- see
//! each module's docs. Only the stencil itself was hand-copied.

/// Largest `D dt / dx^2` one explicit step of the 5-point scheme stays
/// stable at. A Fourier mode of wavenumbers `(kx, ky)` is multiplied each
/// step by `g = 1 - 4 r (sin^2(kx dx / 2) + sin^2(ky dx / 2))`, `r = D dt /
/// dx^2`, which ranges over `[1 - 8 r, 1]`; `|g| <= 1` for every mode exactly
/// when `r <= 1/4`. Derived from the scheme, not quoted.
pub(crate) const FIVE_POINT_STABILITY_LIMIT: f32 = 0.25;

/// How many equal explicit steps `diffusivity_dt` (`D dt` in cells
/// squared) takes so that each stays at `fraction` of the scheme's
/// stability limit (`FIVE_POINT_STABILITY_LIMIT`); at least one.
pub(crate) fn stable_sub_steps(diffusivity_dt: f32, fraction: f32) -> u32 {
    let per_step = fraction * FIVE_POINT_STABILITY_LIMIT;
    if diffusivity_dt > 0.0 && per_step > 0.0 {
        (diffusivity_dt / per_step).ceil().max(1.0) as u32
    } else {
        1
    }
}

/// Applies one explicit-Euler diffusion step: `grid_out[c] = grid_in[c] +
/// diffusivity_dt * laplacian(grid_in, c)`.
///
/// Off-grid neighbors (domain edges) are treated as `ambient` -- a Dirichlet
/// boundary condition. Column-major layout: `idx = x * grid_res + y`,
/// matching the mechanics grid.
pub(crate) fn laplacian_step(
    grid_in: &[f32],
    grid_out: &mut [f32],
    grid_res: usize,
    diffusivity_dt: f32,
    ambient: f32,
) {
    for x in 0..grid_res {
        for y in 0..grid_res {
            let c = x * grid_res + y;
            let t_c = grid_in[c];
            let t_xm = if x > 0 {
                grid_in[c - grid_res]
            } else {
                ambient
            };
            let t_xp = if x + 1 < grid_res {
                grid_in[c + grid_res]
            } else {
                ambient
            };
            let t_ym = if y > 0 { grid_in[c - 1] } else { ambient };
            let t_yp = if y + 1 < grid_res {
                grid_in[c + 1]
            } else {
                ambient
            };
            let laplacian = t_xm + t_xp + t_ym + t_yp - 4.0 * t_c;
            grid_out[c] = t_c + diffusivity_dt * laplacian;
        }
    }
}
