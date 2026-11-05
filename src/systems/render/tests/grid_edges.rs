use super::*;

/// Hysteresis (Schmitt trigger) visibility (`curvature_flow.wgsl`'s Pass 2c):
/// a cell whose density hovers between the low and high thresholds keeps its
/// state. Dispatches `visibility_step_main` on a synthetic density, since the
/// particle splat cannot be held in that gap for several frames.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn visibility_hysteresis_does_not_flicker_in_the_gap_between_thresholds() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    let grid_res = 32u32;
    r.ensure_surface_capacity(&device, grid_res);
    let surface_res = r.surface_res;
    let cell_count = (surface_res * surface_res) as usize;

    const MASS_FLOOR: f32 = 0.15;

    let dispatch_visibility_step = |density_value: f32| {
        let mut density = vec![0.0f32; cell_count];
        density[0] = density_value;
        queue.write_buffer(&r.surface_a_buf, 0, bytemuck::cast_slice(&density));
        queue.write_buffer(
            &r.visibility_params_buf,
            0,
            bytemuck::bytes_of(&VisibilityParams {
                surface_res,
                mass_floor: MASS_FLOOR,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_visibility_step_bg"),
            layout: &r.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: r.visibility_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&r.visibility_step_pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(surface_res.div_ceil(8), surface_res.div_ceil(8), 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
    };

    let read_visibility_cell0 =
        || -> f32 { readback_f32_blocking(&device, &queue, &r.visibility_buf, cell_count)[0] };

    // Frame 1: clearly above the HIGH threshold (1.3x) -> must turn visible.
    dispatch_visibility_step(MASS_FLOOR * 1.5);
    assert!(
        read_visibility_cell0() > 0.5,
        "mass clearly above the HIGH hysteresis threshold must turn the cell visible"
    );

    // Frame 2: drop into the AMBIGUOUS gap -- below mass_floor itself, but
    // still above the LOW threshold (0.7x). A naive single-threshold test
    // would flip this off; hysteresis must keep it visible since it was
    // already visible and hasn't dropped below LOW.
    dispatch_visibility_step(MASS_FLOOR * 0.9);
    assert!(
        read_visibility_cell0() > 0.5,
        "a cell already visible must NOT turn invisible just because its mass \
         dropped below mass_floor itself, as long as it stays above the LOW \
         hysteresis threshold -- this is the whole point of hysteresis"
    );

    // Frame 3: drop clearly below the LOW threshold -> must finally turn invisible.
    dispatch_visibility_step(MASS_FLOOR * 0.5);
    assert!(
        read_visibility_cell0() <= 0.5,
        "mass clearly below the LOW hysteresis threshold must turn the cell invisible"
    );

    // Frame 4: rise back into the SAME ambiguous gap -- above mass_floor
    // itself, but still below the HIGH threshold. Must stay invisible,
    // proving the same gap is stable in both directions, not just one.
    dispatch_visibility_step(MASS_FLOOR * 1.1);
    assert!(
        read_visibility_cell0() <= 0.5,
        "a cell already invisible must NOT turn visible just because its mass \
         rose above mass_floor itself, as long as it stays below the HIGH \
         hysteresis threshold"
    );
}

/// `grid_peak_main` writes, for each cell, the largest cell mass within two
/// cells of it, the reach `grid_volume.wgsl`'s edge needs: one heavy cell in
/// a uniform field shows in its 5 x 5 neighbourhood and nowhere else.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_peak_is_the_largest_mass_within_two_cells() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    let grid_res = 32u32;
    r.ensure_grid_peak_capacity(&device, grid_res);
    let cell_count = (grid_res * grid_res) as usize;
    let (heavy, background, hx, hy) = (3.0f32, 0.25f32, 10i32, 12i32);
    let mut cells = vec![0u32; cell_count * 4];
    for c in 0..cell_count {
        cells[c * 4 + 2] = background.to_bits();
    }
    cells[(hy as usize * grid_res as usize + hx as usize) * 4 + 2] = heavy.to_bits();
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_grid_peak_density"),
        size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));
    queue.write_buffer(
        &r.grid_peak_params_buf,
        0,
        bytemuck::bytes_of(&GridPeakParams {
            grid_res,
            _pad: [0; 3],
        }),
    );
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test_grid_peak_bg"),
        layout: &r.grid_peak_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: r.grid_peak_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: r.grid_peak_params_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(&r.grid_peak_pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(grid_res.div_ceil(8), grid_res.div_ceil(8), 1);
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let peak = readback_f32_blocking(&device, &queue, &r.grid_peak_buf, cell_count);
    for y in 0..grid_res as i32 {
        for x in 0..grid_res as i32 {
            let near = (x - hx).abs() <= 2 && (y - hy).abs() <= 2;
            let expected = if near { heavy } else { background };
            assert_eq!(
                peak[(y * grid_res as i32 + x) as usize],
                expected,
                "cell ({x},{y})"
            );
        }
    }
}

/// Grid mass of particles at `positions` (in cells), `mass_each` each,
/// scattered with the solver's own quadratic B-spline (`grid::kernel`).
fn scatter_particle_mass(grid_res: u32, positions: &[glam::Vec2], mass_each: f32) -> Vec<f32> {
    use crate::spacetime::grid::kernel::quadratic_weights;
    let mut mass = vec![0.0f32; (grid_res * grid_res) as usize];
    for &x in positions {
        let w = quadratic_weights(x);
        for (gx, wx) in w.wx.iter().enumerate() {
            for (gy, wy) in w.wy.iter().enumerate() {
                let c = w.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                if c.x >= 0 && c.y >= 0 && c.x < grid_res as i32 && c.y < grid_res as i32 {
                    mass[(c.y as u32 * grid_res + c.x as u32) as usize] += mass_each * wx * wy;
                }
            }
        }
    }
    mass
}

/// Renders the grid `mass` (reference cell mass 1, the default) through
/// `render_grid_volume` over a transparent `size` x `size` target, and
/// returns every pixel's alpha, the drawn coverage, row-major from the top.
fn grid_volume_coverage(grid_res: u32, size: u32, mass: &[f32]) -> Vec<u8> {
    let (device, queue) = headless_device();
    let cell_count = (grid_res * grid_res) as usize;
    let mut words = vec![0u32; cell_count * 4];
    for (c, m) in mass.iter().enumerate() {
        words[c * 4 + 2] = m.to_bits();
    }
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_coverage_grid"),
        size: (cell_count * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&words));
    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_coverage_material_mass"),
        size: (cell_count * 16 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    r.set_camera(&queue, grid_res, size, size, 0.6, true);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_coverage_target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    // A transparent target, so each pixel's alpha is the drawn coverage.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("test_coverage_clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    queue.submit(std::iter::once(enc.finish()));
    r.render_grid_volume(
        &device,
        &queue,
        GridVolumeSource {
            grid: &grid_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: false,
            grid_res,
        },
        &view,
        false,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let padded = (size * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_coverage_staging"),
        size: (padded * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    (0..size)
        .flat_map(|y| (0..size).map(move |x| (y, x)))
        .map(|(y, x)| mapped[(y * padded + x * 4 + 3) as usize])
        .collect()
}

/// The grid volume draws matter's edge where the matter ends. A slab of
/// particles filling x < 16.25 (a boundary between cell edges), scattered
/// with the solver's own kernel, must render its edge within 0.1 cell of
/// 16.25: 0.05 is the bilinear interpolation of the sampled field (the
/// crossing of its four cell values is at 16.20), the rest half a pixel at 8
/// pixels per cell. The edge criterion it replaced, a nearest-cell gate at
/// 0.15 of a full cell, put this edge at 16.66.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_edge_sits_where_the_matter_ends() {
    let (grid_res, size) = (32u32, 256u32);
    let px_per_cell = size as f32 / grid_res as f32;
    // Particles at spacing 0.5, a quarter of a full cell's mass each, so the
    // inside holds exactly the reference cell mass.
    let (spacing, last_x) = (0.5f32, 16.0f32);
    let mut positions = Vec::new();
    let mut x = last_x;
    while x > 0.0 {
        let mut y = spacing / 2.0;
        while y < grid_res as f32 {
            positions.push(glam::Vec2::new(x, y));
            y += spacing;
        }
        x -= spacing;
    }
    let true_edge = last_x + spacing / 2.0;
    let mass = scatter_particle_mass(grid_res, &positions, spacing * spacing);
    let coverage = grid_volume_coverage(grid_res, size, &mass);
    // A row through the middle of the slab, away from the domain's walls.
    let row = (size / 2 + 4) as usize;
    let alpha: Vec<f32> = coverage[row * size as usize..(row + 1) * size as usize]
        .iter()
        .map(|&a| a as f32)
        .collect();
    let start = (true_edge * px_per_cell) as usize - 2 * px_per_cell as usize;
    let crossing = (start..size as usize - 1)
        .find(|&c| alpha[c] >= 127.5 && alpha[c + 1] < 127.5)
        .expect("the slab must have an edge on this row");
    let fraction = (alpha[crossing] - 127.5) / (alpha[crossing] - alpha[crossing + 1]);
    let edge = (crossing as f32 + 0.5 + fraction) / px_per_cell;
    assert!(
        (edge - true_edge).abs() < 0.1,
        "edge drawn at x = {edge}, the matter ends at {true_edge}"
    );
}

/// Nothing is drawn where there is no matter. Past a body's edge both the
/// mass and its local peak fade to zero, two cells beyond the kernel's
/// reach; there the edge test must not find a crossing that is not there.
/// (Written as a mass difference, `mass - peak / 2` approaches zero without
/// crossing it, and the anti-aliasing drew a one-pixel ring around every
/// body.) A disk off the grid's lattice puts that fade-out at every offset
/// from the pixel centres.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_draws_nothing_past_the_matter() {
    let (grid_res, size) = (32u32, 256u32);
    let px_per_cell = size as f32 / grid_res as f32;
    let (spacing, centre, radius) = (0.5f32, glam::Vec2::new(16.3, 15.7), 6.0f32);
    let mut positions = Vec::new();
    let mut x = centre.x - radius;
    while x <= centre.x + radius {
        let mut y = centre.y - radius;
        while y <= centre.y + radius {
            if (glam::Vec2::new(x, y) - centre).length() <= radius {
                positions.push(glam::Vec2::new(x, y));
            }
            y += spacing;
        }
        x += spacing;
    }
    let mass = scatter_particle_mass(grid_res, &positions, spacing * spacing);
    let coverage = grid_volume_coverage(grid_res, size, &mass);
    // The matter reaches half a spacing past the outermost particle.
    let matter_radius = radius + spacing / 2.0;
    let mut stray = Vec::new();
    for row in 0..size {
        for col in 0..size {
            let p = glam::Vec2::new(
                (col as f32 + 0.5) / px_per_cell,
                grid_res as f32 - (row as f32 + 0.5) / px_per_cell,
            );
            let a = coverage[(row * size + col) as usize];
            if (p - centre).length() > matter_radius + 1.0 && a > 0 {
                stray.push((col, row, a));
            }
        }
    }
    assert!(
        stray.is_empty(),
        "{} pixels drawn more than a cell past the matter, e.g. {:?}",
        stray.len(),
        &stray[..stray.len().min(5)]
    );
}

/// DETERMINISTIC flicker measurement -- unlike live-demo screenshots
/// (confounded: a different random splash every relaunch, which can swing
/// measured "flicker pixel" counts by 10x+ run to run with no shader
/// change at all), this steps a FIXED-SEED particle scenario forward
/// through many physics + render frames and reads back actual rendered
/// pixels each time. The whole pipeline (particle physics, the wave PDE,
/// hysteresis) has no wall-clock dependency, and the atomic splat scatter
/// is integer (exactly order-independent) -- so the same seed reproduces
/// bit-identical results run to run, making this a re-runnable A/B harness
/// for any future shading change, unlike a live demo screenshot ever could
/// be.
///
/// NOT a strict pass/fail gate yet: the exact acceptable flicker fraction
/// hasn't been established (this is the first time it's been measured
/// this way). Asserts a generous placeholder ceiling so this stays a
/// regression guard against a CATASTROPHIC regression (like the reverted
/// density-persistence attempt, which measured roughly half of all
/// sampled points flickering) without yet claiming the CURRENT baseline
/// itself is "acceptable" -- that judgment is still open, tracked
/// separately, not asserted here as settled.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_reconstruction_does_not_flicker_over_many_deterministic_frames() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0)
            .jitter(0.15)
            .rng_seed(42),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let mut sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deterministic_flicker_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    const FRAMES: usize = 40;
    let mut samples_per_frame: Vec<Vec<f64>> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        sim.step_frame();
        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        samples_per_frame.push(readback_luminance_grid(
            &device, &queue, &texture, 64, 64, 2,
        ));
    }

    let num_points = samples_per_frame[0].len();
    let mut flicker_count = 0usize;
    for p in 0..num_points {
        let mut min_v = f64::MAX;
        let mut max_v = f64::MIN;
        for frame in samples_per_frame.iter().take(FRAMES) {
            let v = frame[p];
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let range = max_v - min_v;
        let mut sign_changes = 0;
        for f in 1..FRAMES - 1 {
            let d1 = samples_per_frame[f][p] - samples_per_frame[f - 1][p];
            let d2 = samples_per_frame[f + 1][p] - samples_per_frame[f][p];
            if (d1 > 3.0 && d2 < -3.0) || (d1 < -3.0 && d2 > 3.0) {
                sign_changes += 1;
            }
        }
        if sign_changes >= 2 && range > 15.0 {
            flicker_count += 1;
        }
    }

    let flicker_fraction = flicker_count as f64 / num_points as f64;
    eprintln!(
        "DETERMINISTIC flicker measurement: {flicker_count}/{num_points} \
         ({:.1}%) sampled points show non-monotonic luminance oscillation \
         across {FRAMES} real, reproducible frames (seed=42)",
        flicker_fraction * 100.0
    );
    assert!(
        flicker_fraction < 0.5,
        "catastrophic flicker regression: {flicker_count}/{num_points} \
         ({:.1}%) sampled points oscillating -- this generous ceiling only \
         guards against a severe regression (like the reverted \
         density-persistence attempt, which hit ~50%); it does NOT yet\
         assert the current baseline is fully acceptable",
        flicker_fraction * 100.0
    );
}
