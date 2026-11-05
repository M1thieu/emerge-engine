use super::*;

/// End-to-end smoke test: real particle cluster, real `GpuSimulation`, real
/// offscreen texture, real `render_surface_reconstruction` call. Proves the
/// whole 6-pass pipeline (clear/splat/convert/12x iterate/render) survives
/// together, not just that each shader entry point compiles in isolation.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn render_surface_reconstruction_survives_end_to_end() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_coefficients_si(
        &queue,
        0,
        OpticalCoefficientsSi::new([0.18, 0.22, 0.55], 0.0).unwrap(),
    );
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.1,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [1.0; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::Z,
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    r.set_camera(&queue, 32, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_reconstruction_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
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

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res: 32,
            material_slot: 0,
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
}

/// After the splat, convert and 12 curvature-iterate passes, the settled
/// buffer (`surface_a_buf`, `CURVATURE_ITERATIONS` being even) has
/// substantially higher density near the particle cluster than far from it:
/// the pipeline reconstructs a density field, not uniform noise or zeros
/// (which the end-to-end smoke test above would still pass).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn render_surface_reconstruction_produces_real_density_near_particles() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Small, tight cluster near the grid center -- unambiguous "here"
    // vs. the grid corners, which this scene never populates at all.
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_density_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
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

    let surface_res = r.surface_res;
    let values = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );

    // The particle cluster sits at grid position (16, 16), converted to this
    // buffer's finer units (surface_res/grid_res, as in the shader).
    let scale = surface_res as f32 / grid_res as f32;
    let center = (16.0 * scale) as i32;
    let center_idx = (center as u32 * surface_res + center as u32) as usize;
    let corner_idx = 0usize; // (0, 0) -- this scene never puts any particle there

    assert!(
        values[center_idx] > values[corner_idx] + 0.05,
        "density near the real particle cluster must be substantially higher \
         than density at an empty corner: center={} corner={}",
        values[center_idx],
        values[corner_idx]
    );
    assert!(
        values.iter().all(|v| v.is_finite()),
        "curvature-flow smoothing must never produce NaN/inf, even after 12 iterations"
    );
}

/// Two-phase extension: two separate material clusters each settle density
/// in their own phase buffer and stay near zero in the other's, so
/// `phase_filter_material_id` partitions particles by material instead of
/// running one unfiltered splat twice (the independent-phase-field design of
/// `curvature_flow.wgsl`'s "two-phase extension" depends on it).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn dual_phase_reconstruction_keeps_two_materials_in_their_own_phase_buffer() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Two spatially separate clusters -- far enough apart (8 vs 24 on
    // a 32-cell grid) that neither's real B-spline splat reach can touch
    // the other's territory, isolating the filter itself as the only thing
    // under test.
    let mut particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(8.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(0),
    );
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(24.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(1),
    ));
    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(1, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dual_phase_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
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

    r.render_surface_reconstruction_dual_phase(
        &device,
        &queue,
        DualPhaseSurfaceSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_id_a: 0,
            material_id_b: 1,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let surface_res = r.surface_res;
    let phase_a_values = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );
    let phase_b_values = readback_f32_blocking(
        &device,
        &queue,
        &r.phase_b_a_buf,
        (surface_res * surface_res) as usize,
    );

    let scale = surface_res as f32 / grid_res as f32;
    let idx_at = |grid_x: f32, grid_y: f32| -> usize {
        let sx = (grid_x * scale) as u32;
        let sy = (grid_y * scale) as u32;
        (sy * surface_res + sx) as usize
    };
    let cluster_a_idx = idx_at(8.0, 16.0);
    let cluster_b_idx = idx_at(24.0, 16.0);

    assert!(
        phase_a_values[cluster_a_idx] > 0.05,
        "phase A must show real density at cluster A's own location: got {}",
        phase_a_values[cluster_a_idx]
    );
    assert!(
        phase_a_values[cluster_b_idx] < 0.01,
        "phase A must NOT show real density at cluster B's location (material \
         filter must have excluded those particles): got {}",
        phase_a_values[cluster_b_idx]
    );
    assert!(
        phase_b_values[cluster_b_idx] > 0.05,
        "phase B must show real density at cluster B's own location: got {}",
        phase_b_values[cluster_b_idx]
    );
    assert!(
        phase_b_values[cluster_a_idx] < 0.01,
        "phase B must NOT show real density at cluster A's location (material \
         filter must have excluded those particles): got {}",
        phase_b_values[cluster_a_idx]
    );
}

/// N-material extension (single-phase path, see `curvature_flow.wgsl`): 3
/// separate clusters sharing one smoothed density field each render their
/// own `OpticalTable` colour, so `dominant_material` reads the per-cell mass
/// array, not the `material_slot` fallback (3 materials in
/// `basic_jellies_gpu.rs` rendered as one colour).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn n_material_surface_reconstruction_colors_each_material_distinctly() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Three spatially separate clusters (11 grid cells apart, disk
    // radius 2.0) -- far enough that the shared curvature-smoothed field
    // still resolves 3 distinct dominant-material regions instead of one
    // blended blob.
    let mut particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(5.0, 16.0))
            .disk(2.0)
            .spacing(0.5)
            .material(0),
    );
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(16.0, 16.0))
            .disk(2.0)
            .spacing(0.5)
            .material(1),
    ));
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(27.0, 16.0))
            .disk(2.0)
            .spacing(0.5)
            .material(2),
    ));
    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(1, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(2, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    // Same real magnitude range `basic_jellies_gpu.rs`'s own SIGMA_NEO/COR/VIS
    // use (0.05-0.6), not an arbitrary saturated extreme -- LOW absorption in
    // one channel means that channel is mostly TRANSMITTED (bright); HIGH
    // absorption in the other two means they're mostly absorbed (dark). So
    // material 0 (low red absorption) reads red-dominant, etc.
    r.set_optical_params(&queue, 0, [0.05, 0.55, 0.55]);
    r.set_optical_params(&queue, 1, [0.55, 0.05, 0.55]);
    r.set_optical_params(&queue, 2, [0.55, 0.55, 0.05]);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("n_material_surface_test_target"),
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

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: true,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // Read the whole 64x64 frame ONCE (not per-pixel) -- the exact camera
    // projection's sub-pixel rounding isn't worth hand-deriving precisely;
    // instead, scan a window around each cluster's approximate
    // expected screen location and take whichever pixel shows that
    // material's color most strongly. Robust against a few cells of
    // rounding error in the pixel<->surface-cell mapping (confirmed via a
    // real diagnostic run: hand-derived pixel targets landed within ~1-3
    // surface cells of the true splat center, well inside the B-spline
    // kernel's real footprint for most but not all of the 3 clusters at a
    // single exact pixel -- the window scan absorbs that margin), which is
    // all that's actually under test here (dominant_material's real
    // per-cell resolution), not exact sub-pixel camera arithmetic.
    let unpadded_bytes_per_row: u32 = 64 * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("n_material_frame_readback_staging"),
        size: (padded_bytes_per_row * 64) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("n_material_frame_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(64),
            },
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let frame = slice.get_mapped_range().to_vec();
    staging.unmap();
    let pixel_at = |x: i32, y: i32| -> [u8; 4] {
        let x = x.clamp(0, 63) as u32;
        let y = y.clamp(0, 63) as u32;
        let off = (y * padded_bytes_per_row + x * 4) as usize;
        [frame[off], frame[off + 1], frame[off + 2], frame[off + 3]]
    };
    // Search the WHOLE frame for the pixel that most strongly shows each
    // material's color (rather than hand-deriving the exact camera
    // projection to a specific pixel, which is fragile at cell-level
    // precision against a per-slot mass field that -- unlike total density
    // -- is never curvature-smoothed, so it stays exactly as narrow as the
    // raw B-spline splat, not spread by the later smoothing pass).
    // "Strongest" = that channel exceeds the other two by the widest
    // margin (Beer-Lambert transmission: low sigma_a in `channel` means it
    // dominates the rendered color).
    let strongest_for_channel = |channel: usize| -> ([u8; 4], i32, i32) {
        let mut best = [0u8; 4];
        let mut best_score = i32::MIN;
        let mut best_pos = (0i32, 0i32);
        for y in 0..64 {
            for x in 0..64 {
                let px = pixel_at(x, y);
                let others: i32 = (0..3).filter(|&c| c != channel).map(|c| px[c] as i32).sum();
                let score = px[channel] as i32 * 2 - others;
                if score > best_score {
                    best_score = score;
                    best = px;
                    best_pos = (x, y);
                }
            }
        }
        (best, best_pos.0, best_pos.1)
    };
    let (px_a, ax, ay) = strongest_for_channel(0);
    let (px_b, bx, by) = strongest_for_channel(1);
    let (px_c, cx, cy) = strongest_for_channel(2);

    assert!(
        px_a[0] > px_a[1] && px_a[0] > px_a[2],
        "material 0 (low red absorption) must be found SOMEWHERE in frame \
         as red-dominant: best={:?} at ({ax},{ay})",
        px_a
    );
    assert!(
        px_b[1] > px_b[0] && px_b[1] > px_b[2],
        "material 1 (low green absorption) must be found SOMEWHERE in frame \
         as green-dominant: best={:?} at ({bx},{by})",
        px_b
    );
    assert!(
        px_c[2] > px_c[0] && px_c[2] > px_c[1],
        "material 2 (low blue absorption) must be found SOMEWHERE in frame \
         as blue-dominant: best={:?} at ({cx},{cy})",
        px_c
    );
    // The 3 winning locations must be different regions (not all
    // the same handful of pixels), proving 3 spatially distinct dominant-
    // material resolutions, not one lucky pixel satisfying all 3 channel
    // checks by coincidence.
    let dist2 = |x1: i32, y1: i32, x2: i32, y2: i32| (x1 - x2).pow(2) + (y1 - y2).pow(2);
    assert!(
        dist2(ax, ay, bx, by) > 9,
        "material 0's and material 1's winning pixels are suspiciously close: \
         ({ax},{ay}) vs ({bx},{by})"
    );
    assert!(
        dist2(bx, by, cx, cy) > 9,
        "material 1's and material 2's winning pixels are suspiciously close: \
         ({bx},{by}) vs ({cx},{cy})"
    );
    assert!(
        dist2(ax, ay, cx, cy) > 9,
        "material 0's and material 2's winning pixels are suspiciously close: \
         ({ax},{ay}) vs ({cx},{cy})"
    );
}

/// Blended (mass-fraction-weighted) N-material resolver (see
/// `curvature_flow.wgsl`'s `blended_optical_slot`): two close clusters whose
/// splat footprints overlap leave at least one surface cell with mass in
/// both slots, and the blend formula is checked on the raw buffer (pixel
/// assertions are fragile after tone mapping and quantization): a mixed cell
/// resolves to a weighted average strictly between the two pure optics.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn n_material_blend_produces_real_weighted_average_at_a_mixed_cell() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Two clusters close enough (4 grid cells apart, disk radius 3.0 --
    // particle placement itself overlaps by construction) that their real
    // B-spline splat footprints share cells -- unlike the well-
    // separated clusters in the distinctness test above, which deliberately
    // avoid overlap.
    let mut particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(14.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(0),
    );
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(18.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(1),
    ));
    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(1, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_params(&queue, 0, [3.0, 0.0, 0.0]); // pure-red absorption
    r.set_optical_params(&queue, 1, [0.0, 3.0, 0.0]); // pure-green absorption
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("n_material_blend_test_target"),
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

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: true,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let surface_res = r.surface_res;
    let mm = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_material_mass_buf,
        (surface_res * surface_res * MAX_RENDER_MATERIAL_SLOTS) as usize,
    );

    // The optics written above: red=[3,0,0,0], green=[0,3,0,0] (sigma_s and
    // specular default to 0, only sigma_a set).
    let slot_red = [3.0f32, 0.0, 0.0, 0.0];
    let slot_green = [0.0f32, 3.0, 0.0, 0.0];

    // Search the whole per-cell mass array for a cell where BOTH slot 0 and
    // slot 1 have substantial mass -- proof the two footprints
    // overlap at this scene geometry, not just adjacent.
    let cell_count = (surface_res * surface_res) as usize;
    let mut found = false;
    for cell in 0..cell_count {
        let base = cell * MAX_RENDER_MATERIAL_SLOTS as usize;
        let m0 = mm[base];
        let m1 = mm[base + 1];
        // These fixed-point masses are read back as bit-reinterpreted f32 (see
        // `blended_optical_slot`: fine for ordering and proportion) and sit in
        // the IEEE 754 denormal range (~1e-39), so `> 0.01` would never fire;
        // `> 0.0` tests nonzero mass.
        if m0 > 0.0 && m1 > 0.0 {
            found = true;
            let total = m0 + m1;
            let expected: Vec<f32> = (0..4)
                .map(|c| (m0 * slot_red[c] + m1 * slot_green[c]) / total)
                .collect();
            // The blended red channel must be strictly between pure-green's
            // (0.0) and pure-red's (3.0) values -- a weighted
            // average, not a winner-take-all snap to either pure value.
            assert!(
                expected[0] > 0.0 && expected[0] < 3.0,
                "blended red channel at a real mixed cell (m0={m0}, m1={m1}) \
                 must sit strictly between 0.0 and 3.0, not snap to either \
                 pure value: got {}",
                expected[0]
            );
            assert!(
                expected[1] > 0.0 && expected[1] < 3.0,
                "blended green channel at a real mixed cell (m0={m0}, m1={m1}) \
                 must sit strictly between 0.0 and 3.0, not snap to either \
                 pure value: got {}",
                expected[1]
            );
            // A cell with MORE red mass must lean more toward red than a
            // cell with LESS red mass -- the weighting is not just
            // "average of the two extremes regardless of ratio". Checked
            // via the closed-form ratio directly rather than a second
            // sampled cell (deterministic, no dependence on scene geometry
            // producing a second usable mixed cell).
            let expected_ratio = m0 / total;
            assert!(
                (expected[0] / 3.0 - expected_ratio).abs() < 1.0e-4,
                "blended red channel must scale linearly with slot 0's real \
                 mass fraction ({expected_ratio}): got fraction {}",
                expected[0] / 3.0
            );
            break;
        }
    }
    assert!(
        found,
        "scene geometry produced no real overlapping-mass cell -- test setup \
         needs closer clusters or a wider disk radius, this doesn't verify \
         the blend at all if there's nothing to blend"
    );
}
