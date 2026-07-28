//! `rod::gravitropism`: Porat, Rivière, Meroz 2024, J. Exp. Bot. 75(2):620, eq. 2.
//! Proves the tip rotates toward gravity through the full mechanical pipeline (grid
//! coupling, elastic bending, damping), isolated from ordinary self-weight sag via A/B
//! (gravitropism on vs off).

extern crate emerge_engine as emerge;
use emerge::rod::{Gravitropism, GravitropismMode, RodMaterial, build_straight_rod};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation};
use glam::Vec2;

/// A short, stiff, heavily-damped horizontal rod (pinned at the base,
/// pointing due +x -- perpendicular to gravity) so the tip's own alignment
/// with gravity is unambiguous to measure: 1.0 = tip points straight down,
/// 0.0 = tip still horizontal.
fn build_rod(gravitropism: Option<Gravitropism>) -> Simulation {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 400,
        min_dt: 0.0005,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let span = 1.5;
    let n_points = 8usize;
    let mut rod_points = build_straight_rod(
        Vec2::new(10.0, 24.0),
        Vec2::new(10.0 + span, 24.0),
        n_points,
        0.3,
        1.0,
    );
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let l0 = span / (n_points as f32 - 1.0);
    let point_mass = 0.3 * l0;
    // Stiff + heavily damped: minimizes ordinary elastic self-weight sag so
    // the gravitropism-specific signal isn't swamped by it.
    let ea = 5.0e6 * 0.06 * 0.02;
    let ei = 5.0e6 * 0.06_f32.powi(3) * 0.02 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        5.0e6,
        0.06,
        0.02,
        axial_damping,
        bending_damping,
    );
    let mut rod = emerge::rod::Rod::new(rod_points, material);
    if let Some(g) = gravitropism {
        rod = rod.with_gravitropism(g);
    }
    solver.add_rod(rod);
    solver
}

/// 1.0 = tip direction exactly aligned with gravity (straight down), 0.0 =
/// perpendicular (still horizontal), matching the initial condition.
fn tip_gravity_alignment(solver: &Simulation) -> f32 {
    let points = &solver.rods()[0].points;
    let n = points.len();
    let tip_edge = points.x[n - 1] - points.x[n - 2];
    let tip_dir = tip_edge.normalize_or_zero();
    let gravity_dir = solver.config().gravity.normalize_or_zero();
    tip_dir.dot(gravity_dir)
}

#[test]
fn gravitropism_rotates_tip_toward_gravity_alignment_more_than_self_weight_alone() {
    let mut baseline = build_rod(None);
    let mut with_gravitropism = build_rod(Some(Gravitropism::new(0.05, 0.005)));

    let alignment_before = tip_gravity_alignment(&baseline);
    assert!(
        alignment_before.abs() < 0.05,
        "test setup isn't actually horizontal/perpendicular to gravity: alignment={alignment_before:.4}"
    );

    baseline.step_n(6000);
    with_gravitropism.step_n(6000);

    let baseline_alignment = tip_gravity_alignment(&baseline);
    let gravitropism_alignment = tip_gravity_alignment(&with_gravitropism);

    // Gravitropism must produce meaningfully more alignment than self-weight sag alone.
    assert!(
        gravitropism_alignment > baseline_alignment + 0.1,
        "gravitropism should rotate the tip measurably more toward gravity than self-weight \
         sag alone: baseline={baseline_alignment:.4} with_gravitropism={gravitropism_alignment:.4}"
    );
}

#[test]
fn whole_organ_mode_still_rotates_the_tip_toward_gravity_through_the_explicit_substep_path() {
    // This rod never sets `use_implicit_integration`, so it runs through
    // `do_substep`'s CFL-limited call site (step.rs, possibly many
    // `apply_gravitropism` calls per frame at small sub_dt) -- the OTHER
    // real call site from the one every `gravitropism.rs` unit test and the
    // GUI examples exercise (those are all implicit, frame-dt). WholeOrgan
    // must work correctly there too, not just at the implicit call site.
    let mut baseline = build_rod(None);
    let mut with_gravitropism = build_rod(Some(
        Gravitropism::new(0.05, 0.005).with_mode(GravitropismMode::WholeOrgan),
    ));

    baseline.step_n(6000);
    with_gravitropism.step_n(6000);

    let baseline_alignment = tip_gravity_alignment(&baseline);
    let gravitropism_alignment = tip_gravity_alignment(&with_gravitropism);

    assert!(
        gravitropism_alignment > baseline_alignment + 0.1,
        "WholeOrgan gravitropism should rotate the tip measurably more toward gravity than \
         self-weight sag alone through the explicit/substep call site too: \
         baseline={baseline_alignment:.4} with_gravitropism={gravitropism_alignment:.4}"
    );
}
