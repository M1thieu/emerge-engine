use super::*;

/// `cursor_grid()`'s `screen_pos / window_size * grid_res` assumed the grid
/// fills the window, while `set_camera` letterboxes or pillarboxes to keep
/// the aspect ratio (they agreed only for a square window). `screen_to_grid`
/// is the exact inverse; checked against geometric invariants rather than a
/// re-derived formula, which would copy a bug into its test.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn screen_to_grid_is_exact_inverse_of_set_camera_at_any_aspect_ratio() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    let grid_res = 32u32;

    // Exactly square: no letterboxing on either axis, so the four window
    // corners must map to the four exact grid corners.
    r.set_camera(&queue, grid_res, 100, 100, 0.6, true);
    let (gx, gy) = r.screen_to_grid(0.0, 0.0, 100, 100);
    assert!(
        (gx - 0.0).abs() < 1.0e-3 && (gy - grid_res as f32).abs() < 1.0e-3,
        "square window's top-left screen corner must map to the grid's \
         top-left corner (x=0, y=grid_res) -- got ({gx}, {gy})"
    );
    let (gx, gy) = r.screen_to_grid(100.0, 100.0, 100, 100);
    assert!(
        (gx - grid_res as f32).abs() < 1.0e-3 && (gy - 0.0).abs() < 1.0e-3,
        "square window's bottom-right screen corner must map to the grid's \
         bottom-right corner (x=grid_res, y=0) -- got ({gx}, {gy})"
    );

    // A real invariant that holds at ANY aspect ratio: letterbox/pillarbox
    // padding is always symmetric, so the window's own screen-space CENTER
    // must always map to the grid's center, wide or tall or square. This is
    // exactly the case the old naive mapping got wrong away from square.
    for (w, h) in [(100u32, 100u32), (300, 100), (100, 300), (37, 211)] {
        r.set_camera(&queue, grid_res, w, h, 0.6, true);
        let (gx, gy) = r.screen_to_grid(w as f32 / 2.0, h as f32 / 2.0, w, h);
        let expected = grid_res as f32 / 2.0;
        assert!(
            (gx - expected).abs() < 1.0e-2 && (gy - expected).abs() < 1.0e-2,
            "window center must map to grid center at ANY aspect ratio \
             (w={w}, h={h}) -- got ({gx}, {gy}), expected ({expected}, {expected})"
        );
    }

    // A wide window (aspect > 1) pillarboxes on X but fills Y edge to edge
    // -- so its full screen-space Y range must still cover the grid's full
    // Y range exactly, unlike X which is padded.
    r.set_camera(&queue, grid_res, 300, 100, 0.6, true);
    let (_, gy_top) = r.screen_to_grid(150.0, 0.0, 300, 100);
    let (_, gy_bottom) = r.screen_to_grid(150.0, 100.0, 300, 100);
    assert!(
        (gy_top - grid_res as f32).abs() < 1.0e-3 && gy_bottom.abs() < 1.0e-3,
        "a wide window's Y axis is never letterboxed -- top/bottom screen \
         edges must map exactly to grid Y=grid_res/Y=0 -- got top={gy_top} \
         bottom={gy_bottom}"
    );
}

/// `grid_to_screen`/`grid_distance_to_pixels`, the one source for both
/// directions (a hand-derived copy of the projection drifted from what
/// `set_camera` uploaded), round-trip exactly with `screen_to_grid` at several
/// aspect ratios, without re-deriving the formula.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_to_screen_is_exact_inverse_of_screen_to_grid_at_any_aspect_ratio() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    let grid_res = 32u32;

    for (w, h) in [(100u32, 100u32), (300, 100), (100, 300), (37, 211)] {
        r.set_camera(&queue, grid_res, w, h, 0.6, true);
        for (sx, sy) in [
            (0.0, 0.0),
            (w as f32, h as f32),
            (w as f32 * 0.5, h as f32 * 0.5),
            (w as f32 * 0.25, h as f32 * 0.75),
        ] {
            let (gx, gy) = r.screen_to_grid(sx, sy, w, h);
            let (rx, ry) = r.grid_to_screen(gx, gy, w, h);
            assert!(
                (rx - sx).abs() < 1.0e-2 && (ry - sy).abs() < 1.0e-2,
                "grid_to_screen must exactly invert screen_to_grid at w={w} h={h} \
                 -- screen ({sx},{sy}) -> grid ({gx},{gy}) -> screen ({rx},{ry})"
            );
        }
    }

    // A independently-checkable invariant for the radius conversion:
    // the camera is isotropic (grid cells always render as true squares, see
    // `set_camera`'s doc), so a distance measured along the Y axis via
    // `screen_to_grid` must match `grid_distance_to_pixels`'s own conversion
    // of that same real grid distance, exactly.
    r.set_camera(&queue, grid_res, 300, 100, 0.6, true);
    let (_, gy_top) = r.screen_to_grid(0.0, 0.0, 300, 100);
    let (_, gy_mid) = r.screen_to_grid(0.0, 50.0, 300, 100);
    let grid_distance = gy_top - gy_mid;
    let pixel_distance = r.grid_distance_to_pixels(grid_distance, 100);
    assert!(
        (pixel_distance - 50.0).abs() < 1.0e-2,
        "grid_distance_to_pixels must agree with screen_to_grid's own real \
         scale -- expected 50.0 pixels, got {pixel_distance}"
    );
}

/// A UI overlay fed physical pixels into a toolkit (egui) that draws in
/// logical points is off by the display's scale factor.
/// `grid_to_screen_points`/`grid_distance_to_points` are checked at 100%,
/// 125% (the scale that showed it) and 200%.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_to_screen_points_divides_out_the_real_dpi_scale_factor() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    let grid_res = 32u32;
    r.set_camera(&queue, grid_res, 300, 200, 0.6, true);

    let (px_x, px_y) = r.grid_to_screen(16.0, 16.0, 300, 200);
    let px_radius = r.grid_distance_to_pixels(2.0, 200);

    for ppp in [1.0_f32, 1.25, 1.5, 2.0] {
        let (pt_x, pt_y) = r.grid_to_screen_points(16.0, 16.0, 300, 200, ppp);
        let pt_radius = r.grid_distance_to_points(2.0, 200, ppp);
        assert!(
            (pt_x - px_x / ppp).abs() < 1.0e-3 && (pt_y - px_y / ppp).abs() < 1.0e-3,
            "grid_to_screen_points at ppp={ppp} must be exactly the physical \
             result divided by the real scale factor -- got ({pt_x},{pt_y}), \
             expected ({},{})",
            px_x / ppp,
            px_y / ppp
        );
        assert!(
            (pt_radius - px_radius / ppp).abs() < 1.0e-3,
            "grid_distance_to_points at ppp={ppp} must match physical/ppp -- \
             got {pt_radius}, expected {}",
            px_radius / ppp
        );
    }

    // At the real 100% (no scaling) case specifically, points and pixels
    // must be numerically IDENTICAL -- this is the case that silently looked
    // "correct" during development on a 100%-scaled display while actually
    // being wrong at every other real scale factor.
    let (pt_x, pt_y) = r.grid_to_screen_points(16.0, 16.0, 300, 200, 1.0);
    assert!(
        (pt_x - px_x).abs() < 1.0e-6 && (pt_y - px_y).abs() < 1.0e-6,
        "at 100% scaling, points and physical pixels must be identical"
    );
}

/// Pore-fluid index matching (`Renderer::set_refractive_index`) reduces
/// scattering as a particle's `scalar_field` saturates toward 1.0 (the
/// generic wet-material mechanism). With quartz sand's index (~1.5) and
/// saturation from dry to full on one slot, the colour changes and darkens
/// toward the absorption-only colour.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn wetness_darkens_by_physics_color_via_refractive_index() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]); // illustrative sand absorption, unsourced
    r.set_optical_scattering(&queue, 0, 8.0);
    r.set_refractive_index(0, 1.5); // real quartz refractive index (Hecht, "Optics")

    let mut dry = Particle::zeroed();
    dry.material_id = 0;
    dry.deformation_gradient = Mat2::IDENTITY;
    dry.scalar_field = 0.0;
    let mut wet = dry;
    wet.scalar_field = 1.0;

    let c_dry = r.particle_color(&dry, 0);
    let c_wet = r.particle_color(&wet, 0);
    assert_ne!(
        c_dry, c_wet,
        "saturation must actually change ByPhysics color once refractive_index is set"
    );
    let brightness = |c: [f32; 4]| c[0] + c[1] + c[2];
    assert!(
        brightness(c_wet) < brightness(c_dry),
        "wet sand must render DARKER than dry (reduced scattering from \
         pore-fluid index-matching), not brighter -- dry={c_dry:?} wet={c_wet:?}"
    );
}

/// The default `refractive_index` (1.0, same as air) must be fully inert --
/// a material that never calls `set_refractive_index` renders identically
/// regardless of its particles' `scalar_field`, exactly as before this
/// feature existed. Real safety property: this mechanism must never
/// silently activate for materials that never opted in.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn wetness_darkening_is_inert_without_refractive_index_opt_in() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(&queue, 0, 8.0);
    // Deliberately NOT calling set_refractive_index -- default stays 1.0.

    let mut dry = Particle::zeroed();
    dry.material_id = 0;
    dry.deformation_gradient = Mat2::IDENTITY;
    dry.scalar_field = 0.0;
    let mut wet = dry;
    wet.scalar_field = 1.0;

    assert_eq!(
        r.particle_color(&dry, 0),
        r.particle_color(&wet, 0),
        "a material that never calls set_refractive_index must be byte-identical \
         regardless of scalar_field -- this feature must be opt-in, not silently active"
    );
}

/// Diagnostic: the isolated per-call GPU cost of each render path
/// (Particles/GridVolume/Surface) on the same scene, synced with
/// device.poll so it measures completed GPU work. Built when
/// `basic_sand_grid_gpu.rs` dropped to 24-27 fps with
/// `render_surface_reconstruction`, to separate the technique's cost from the
/// wiring.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn diag_surface_reconstruction_real_cost_vs_grid_volume_and_particles() {
    use crate::gpu::GpuSimulation;
    use crate::{DruckerPragerMaterial, MaterialRegistry, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;
    use std::time::Instant;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    const GRID: usize = 64;
    let config = SimConfig::standard(GRID, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(32.0, 20.0))
            .disk(20.0)
            .spacing(0.5)
            .material(0),
    );
    let registry =
        MaterialRegistry::with_default(Box::new(DruckerPragerMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    let particle_count = sim.particle_count();

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, particle_count, fmt);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_camera(&queue, GRID as u32, 800, 600, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("perf_test_target"),
        size: wgpu::Extent3d {
            width: 800,
            height: 600,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    const WARMUP: usize = 3;
    const TIMED: usize = 20;

    macro_rules! measure {
        ($label:expr, $call:expr) => {{
            for _ in 0..WARMUP {
                $call;
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            let start = Instant::now();
            for _ in 0..TIMED {
                $call;
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let elapsed = start.elapsed();
            println!(
                "{}: {:.3}ms/call ({} particles, {GRID}x{GRID} grid)",
                $label,
                elapsed.as_secs_f64() * 1000.0 / TIMED as f64,
                particle_count
            );
        }};
    }

    measure!(
        "render_gpu (Particles)",
        r.render_gpu(
            &device,
            &queue,
            GpuRenderParams {
                particle_buf: sim.particle_buffer(),
                particle_count,
                output_view: &view,
                clear: true,
                interp_alpha: 1.0,
            },
        )
    );
    measure!(
        "render_grid_volume",
        r.render_grid_volume(
            &device,
            &queue,
            GridVolumeSource {
                grid: sim.grid_buffer(),
                material_mass: sim.material_mass_buffer(),
                material_mass_enabled: true,
                grid_res: GRID as u32,
            },
            &view,
            true,
        )
    );
    measure!(
        "render_surface_reconstruction (default mult=6, iters=12)",
        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count,
                grid_res: GRID as u32,
                material_slot: 0,
                material_mass_enabled: true,
                dt: 0.1,
            },
            &view,
            true,
        )
    );

    // Sweep of `surface_res_multiplier` (the biggest quality/cost dial, cost
    // goes with its square) and `curvature_iterations` (van der Laan et al.
    // 2009's "several per frame"), measured at each combination.
    for mult in [2u32, 3, 4] {
        r.set_surface_res_multiplier(mult);
        measure!(
            format!("surface_res_multiplier={mult} (iters=12)"),
            r.render_surface_reconstruction(
                &device,
                &queue,
                SurfaceReconstructionSource {
                    particle_buf: sim.particle_buffer(),
                    particle_count,
                    grid_res: GRID as u32,
                    material_slot: 0,
                    material_mass_enabled: true,
                    dt: 0.1,
                },
                &view,
                true,
            )
        );
    }
    r.set_surface_res_multiplier(6);
    for iters in [4u32, 6, 8] {
        r.set_curvature_iterations(iters);
        measure!(
            format!("curvature_iterations={iters} (mult=6)"),
            r.render_surface_reconstruction(
                &device,
                &queue,
                SurfaceReconstructionSource {
                    particle_buf: sim.particle_buffer(),
                    particle_count,
                    grid_res: GRID as u32,
                    material_slot: 0,
                    material_mass_enabled: true,
                    dt: 0.1,
                },
                &view,
                true,
            )
        );
    }
}

/// Every camera goes through `region_projection`. Framed whole, the grid must
/// come out as `set_camera` drew it before it called that function (the
/// closed form below is that version's, restated); a region must fill the
/// window along one axis, centred, with square pixels. Both window
/// orientations, since the fit switches axis between them.
#[test]
fn region_projection_frames_a_region_and_the_whole_grid() {
    let close = |a: f32, b: f32| (a - b).abs() <= 1.0e-5 * a.abs().max(b.abs()).max(1.0);
    for (w, h) in [(1280u32, 720u32), (720, 1280), (900, 900)] {
        let aspect = w as f32 / h as f32;
        let gr = 160.0f32;
        let before = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        let now = Renderer::region_projection(Vec2::ZERO, Vec2::splat(gr), w, h);
        assert!(
            close(now.0, before.0)
                && close(now.1, before.1)
                && close(now.2, before.2)
                && close(now.3, before.3),
            "{w}x{h}: whole grid {now:?}, set_camera drew {before:?}"
        );

        // A wide strip and a tall one.
        for (min, max) in [
            (Vec2::new(30.0, 2.0), Vec2::new(150.0, 40.0)),
            (Vec2::new(70.0, 2.0), Vec2::new(90.0, 120.0)),
        ] {
            let (sx, tx, sy, ty) = Renderer::region_projection(min, max, w, h);
            let ndc = |p: Vec2| Vec2::new(p.x * sx + tx, p.y * sy + ty);
            let centre = ndc((min + max) * 0.5);
            assert!(centre.length() < 1.0e-5, "{w}x{h}: centre at {centre}");
            let (lo, hi) = (ndc(min), ndc(max));
            assert!(
                lo.cmpge(Vec2::splat(-1.0 - 1.0e-5)).all()
                    && hi.cmple(Vec2::splat(1.0 + 1.0e-5)).all(),
                "{w}x{h}: region {min}..{max} spills out of the window, {lo}..{hi}"
            );
            assert!(
                close(hi.x, 1.0) || close(hi.y, 1.0),
                "{w}x{h}: region {min}..{max} fills neither axis, {lo}..{hi}"
            );
            // Square pixels: a cell spans as many pixels across as up.
            assert!(
                close(sx * w as f32, sy * h as f32),
                "{w}x{h}: cells are not square"
            );
        }
    }
}

/// The round trip the cursor depends on, through a renderer: a point
/// drawn under `set_camera_region` and read back by `screen_to_grid` comes
/// back where it was, and the framed strip's ends land on the window's
/// edges.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn a_cursor_reads_back_the_point_a_region_camera_drew() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Wider than either window, so it fills the width in both.
    let (min, max) = (Vec2::new(30.0, 2.0), Vec2::new(150.0, 40.0));
    for (w, h) in [(1280u32, 720u32), (720, 1280)] {
        r.set_camera_region(&queue, (min, max), w, h, 0.6, true);
        for p in [min, max, (min + max) * 0.5, Vec2::new(41.3, 17.9)] {
            let (x, y) = r.grid_to_screen(p.x, p.y, w, h);
            let (gx, gy) = r.screen_to_grid(x, y, w, h);
            assert!(
                (gx - p.x).abs() < 1.0e-3 && (gy - p.y).abs() < 1.0e-3,
                "{w}x{h}: {p} drawn at ({x}, {y}) reads back as ({gx}, {gy})"
            );
        }
        let (left, _) = r.grid_to_screen(min.x, min.y, w, h);
        let (right, _) = r.grid_to_screen(max.x, min.y, w, h);
        assert!(
            left.abs() < 1.0e-2 && (right - w as f32).abs() < 1.0e-2,
            "{w}x{h}: strip drawn from x = {left} to {right} px"
        );
    }
}

/// `free_surface_cell_step` is the quadratic B-spline's mass inside one cell
/// centred on its peak, for the kernel `grid::kernel::axis_weights`
/// evaluates, stretched by the splat width: checked against a numerical
/// integral of that kernel.
#[test]
fn free_surface_cell_step_is_the_kernels_central_mass() {
    // N(u) from the kernel's own weights: the centre weight within half a
    // cell, an outer weight beyond it.
    let n = |u: f32| {
        let a = u.abs();
        if a <= 0.5 {
            crate::grid::kernel::axis_weights(a)[1]
        } else if a < 1.5 {
            crate::grid::kernel::axis_weights(a - 1.0)[0]
        } else {
            0.0
        }
    };
    for splat_cells in [0.5f32, 1.0, 2.0, 3.0, 4.0] {
        let h = 0.5 / splat_cells;
        let steps = 20_000;
        let du = 2.0 * h / steps as f32;
        let integral: f32 = (0..steps).map(|i| n(-h + (i as f32 + 0.5) * du) * du).sum();
        let step = free_surface_cell_step(splat_cells);
        assert!(
            (step - integral).abs() < 1.0e-4,
            "splat width {splat_cells}: {step} against the kernel's {integral}"
        );
    }
    // The physics grid's own value, `grid_volume.wgsl`'s constant.
    assert!((free_surface_cell_step(1.0) - 2.0 / 3.0).abs() < 1.0e-6);
}
