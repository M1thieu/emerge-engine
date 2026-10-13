use glam::Mat2;

/// Frobenius inner product of two 2×2 matrices, `A : B = sum_ij a_ij b_ij`:
/// the matrix version of the dot product.
#[inline]
pub(crate) fn frob(a: Mat2, b: Mat2) -> f32 {
    a.x_axis.x * b.x_axis.x
        + a.x_axis.y * b.x_axis.y
        + a.y_axis.x * b.y_axis.x
        + a.y_axis.y * b.y_axis.y
}
