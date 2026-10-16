use super::*;

/// The eG(2) scheme's matrices against Gross's Appendix C.1.2 (C.2):
/// `A^R_m = [0 1; -8 3]`, `A^R_p = [-1; 1]`, `A^p_m = [2 1/2; -8 4]`,
/// `A^p_p = [1/2; -1]`.
#[test]
fn galerkin2_matches_the_thesis_matrices() {
    let s = Galerkin2::new();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(s.a_m.col(0).x, 0.0) && close(s.a_m.col(1).x, 1.0));
    assert!(close(s.a_m.col(0).y, -8.0) && close(s.a_m.col(1).y, 3.0));
    assert!(close(s.a_p.x, -1.0) && close(s.a_p.y, 1.0));
    assert!(close(s.a_v.col(0).x, 2.0) && close(s.a_v.col(1).x, 0.5));
    assert!(close(s.a_v.col(0).y, -8.0) && close(s.a_v.col(1).y, 4.0));
    assert!(close(s.b_v.x, 0.5) && close(s.b_v.y, -1.0));
    assert!(close(s.w[0] + s.w[1], 1.0));
}

/// The discrete gradient makes `S : dC / 2` exactly the energy's
/// change, however large the step.
#[test]
fn the_algorithmic_stress_carries_the_exact_energy_change() {
    let solid = Elastic::isochoric(3.0, 11.0);
    let f0 = DMat2::from_cols(DVec2::new(1.1, 0.05), DVec2::new(-0.2, 0.95));
    let f1 = DMat2::from_cols(DVec2::new(0.8, 0.3), DVec2::new(0.1, 1.25));
    let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
    let s = solid.algorithmic_stress(c0, c1);
    let dc = c1 - c0;
    let work = 0.5 * (s.x_axis.dot(dc.x_axis) + s.y_axis.dot(dc.y_axis));
    let change = solid.energy(f1) - solid.energy(f0);
    assert!(
        (work - change).abs() < 1e-12 * change.abs().max(1.0),
        "{work} vs {change}"
    );
    assert!((s.x_axis.y - s.y_axis.x).abs() < 1e-12, "S symmetric");
}

/// `S(C)` is twice the derivative of `W(C)`, checked by central
/// differences.
#[test]
fn the_stress_is_twice_the_energy_derivative() {
    let solid = Elastic::isochoric(2.0, 7.0);
    let f = DMat2::from_cols(DVec2::new(1.05, 0.1), DVec2::new(0.07, 0.9));
    let c = f.transpose() * f;
    let s = solid.second_pk(c);
    let h = 1e-6;
    for (i, j) in [(0, 0), (1, 1), (0, 1)] {
        let mut e = DMat2::ZERO;
        e.col_mut(j)[i] = 1.0;
        e.col_mut(i)[j] = 1.0;
        let e = if i == j { 0.5 * e } else { e };
        let d = (solid.energy_of_c(c + h * e) - solid.energy_of_c(c - h * e)) / (2.0 * h);
        let s_e = s.x_axis.dot(e.x_axis) + s.y_axis.dot(e.y_axis);
        assert!(
            (2.0 * d - s_e).abs() < 1e-6,
            "({i}, {j}): {} vs {s_e}",
            2.0 * d
        );
    }
}
/// A block whose edges cut the cells: particles at four per cell, the
/// block spanning cells `[3.3, 9.15] x [2.6, 7.2]` of a 14 x 12 grid.
fn cut_block() -> (GridLayout, Vec<DVec2>, Vec<f64>) {
    let layout = GridLayout::new(14, 12, 1.0);
    let mut x = Vec::new();
    let (lo, hi) = (DVec2::new(3.3, 2.6), DVec2::new(9.15, 7.2));
    let mut y = lo.y + 0.25;
    while y < hi.y {
        let mut q = lo.x + 0.25;
        while q < hi.x {
            x.push(DVec2::new(q, y));
            q += 0.5;
        }
        y += 0.5;
    }
    let volume = vec![0.25; x.len()];
    (layout, x, volume)
}

/// The extended basis keeps the partition of unity and reproduces
/// linear fields at every particle, values and gradients: both momenta
/// rest on it.
#[test]
fn the_extended_basis_reproduces_linear_fields() {
    let (layout, x, volume) = cut_block();
    let open = |_: usize, _: usize, _: usize| None;
    let grid = Grid::new(&layout, &x, (&volume, &volume), &open, Some(0.75));
    let extended = grid.extension.iter().filter(|e| e.is_some()).count();
    assert!(extended > 0, "the cut block has degenerate faces");
    // Each unknown's position, from its stacked face.
    let mut position = vec![DVec2::ZERO; grid.len];
    for l in &grid.lattices {
        for j in 0..l.nj {
            for i in 0..l.ni {
                let unknown = grid.slot[l.start + i + l.ni * j];
                if unknown != usize::MAX {
                    position[unknown] =
                        (DVec2::new(i as f64, j as f64) + l.offset) * f64::from(layout.dx);
                }
            }
        }
    }
    for (p, &q) in x.iter().enumerate() {
        for a in 0..2 {
            let (mut one, mut at, mut grad_one) = (0.0, DVec2::ZERO, DVec2::ZERO);
            let mut grad_at = DMat2::ZERO;
            grid.visit(p, a, |k, w, g| {
                one += w;
                at += w * position[k];
                grad_one += g;
                grad_at += DMat2::from_cols(g * position[k].x, g * position[k].y);
            });
            assert!((one - 1.0).abs() < 1e-12, "particle {p}: sum {one}");
            assert!((at - q).length() < 1e-12, "particle {p}: {at} vs {}", q);
            assert!(grad_one.length() < 1e-12, "particle {p}: gradient sum");
            let identity = grad_at - DMat2::IDENTITY;
            assert!(
                identity.x_axis.length() + identity.y_axis.length() < 1e-11,
                "particle {p}: gradient of x is {grad_at}"
            );
        }
    }
}

/// The smallest eigenvalue of the consistent mass over its largest
/// diagonal entry, by inverse iteration: the original B-splines of a
/// cut block make it nearly singular, the extended ones lift it.
fn mass_conditioning(occupation: Option<f64>) -> f64 {
    let (layout, x, volume) = cut_block();
    let open = |_: usize, _: usize, _: usize| None;
    let grid = Grid::new(&layout, &x, (&volume, &volume), &open, occupation);
    let diag = grid.mass_diagonal(&volume, 0.0);
    let largest = diag.iter().fold(0.0f64, |m, &d| m.max(d));
    let apply = |y: &[f64]| grid.mass_times(y, &volume, 0.0);
    let mut y: Vec<f64> = (0..grid.len)
        .map(|k| ((k * 7919) % 101) as f64 - 50.0)
        .collect();
    let mut lambda = 0.0;
    for _ in 0..40 {
        let norm = y.iter().map(|e| e * e).sum::<f64>().sqrt();
        y.iter_mut().for_each(|e| *e /= norm);
        let z = conjugate_gradient(&apply, &y, &diag, &grid.active);
        lambda = 1.0 / z.iter().map(|e| e * e).sum::<f64>().sqrt();
        y = z;
    }
    lambda / largest
}

#[test]
fn the_extended_basis_lifts_the_mass_conditioning() {
    let original = mass_conditioning(None);
    let extended = mass_conditioning(Some(0.75));
    println!(
        "smallest eigenvalue over the largest diagonal: original {original:.3e}, extended {extended:.3e}"
    );
    assert!(
        extended > 10.0 * original,
        "{original:.3e} -> {extended:.3e}"
    );
}

/// Two blocks side by side on the same rows, apart: the band is as wide
/// as one block's, each numbered as a block of its own.
#[test]
fn bodies_apart_keep_the_band_of_one() {
    let block = |left: f64| {
        let mut x = Vec::new();
        let mut y = 2.25;
        while y < 7.0 {
            let mut q = left + 0.25;
            while q < left + 5.0 {
                x.push(DVec2::new(q, y));
                q += 0.5;
            }
            y += 0.5;
        }
        x
    };
    let layout = GridLayout::new(30, 10, 1.0);
    let open = |_: usize, _: usize, _: usize| None;
    let width = |x: &[DVec2]| {
        let volume = vec![0.25; x.len()];
        let grid = Grid::new(&layout, x, (&volume, &volume), &open, Some(0.75));
        grid.empty_band(x.len()).width
    };
    let one = block(3.0);
    let mut two = one.clone();
    two.extend(block(20.0));
    assert_eq!(width(&two), width(&one));
}

/// The band factor solves a symmetric positive definite band system,
/// its unknowns renumbered: `A x = b` back to rounding.
#[test]
fn the_band_factor_solves_its_system() {
    let n = 9;
    let width = 2;
    // A renumbering and a band matrix in the band's numbering.
    let order = vec![3, 0, 7, 1, 8, 2, 6, 4, 5];
    let dense = |i: usize, j: usize| -> f64 {
        match i.abs_diff(j) {
            0 => 6.0 + i as f64 * 0.1,
            1 => -1.5,
            2 => 0.25,
            _ => 0.0,
        }
    };
    let mut band = Band {
        n,
        width,
        data: vec![0.0; n * (width + 1)],
        order: order.clone(),
    };
    for i in 0..n {
        for j in i.saturating_sub(width)..=i {
            let slot = band.at(i, j);
            band.data[slot] = dense(i, j);
        }
    }
    band.factor();
    let b: Vec<f64> = (0..n).map(|k| (k as f64 * 0.7).sin()).collect();
    let x = band.solve(&b);
    for k in 0..n {
        let i = order[k];
        let ax: f64 = (0..n).map(|m| dense(i, order[m]) * x[m]).sum();
        assert!((ax - b[k]).abs() < 1e-12, "row {k}: {ax} vs {}", b[k]);
    }
}
