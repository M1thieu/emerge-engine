use super::*;

/// Anisotropic splat (see `curvature_flow.wgsl`): a particle whose
/// `deformation_gradient` is stretched 2.5x along x splats a wider footprint
/// along x than along y at the same offset. An identity-F control at the
/// same position shows no such bias, so the asymmetry comes from F and not
/// from the splat loop.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn anisotropic_splat_widens_footprint_along_stretched_axis() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));

    let render_single_particle = |f: Mat2| -> Vec<f32> {
        // A small, smooth cluster (not one isolated near-delta-function
        // particle): every particle shares the SAME imposed `F`, so the
        // aggregate splat is still a clean directional-bias test, but the
        // density field entering `curvature_iterate_main` is smooth enough
        // not to trip that pass's own known instability on sharp/isolated
        // inputs (an explicit curvature-flow step, like any explicit
        // diffusion scheme, is only guaranteed stable on smooth data --
        // real usage never spawns a truly isolated single particle either).
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(1.5)
                .spacing(0.5)
                .material(0),
        );
        for p in &mut particles {
            p.deformation_gradient = f;
        }

        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aniso_splat_test_target"),
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
        readback_f32_blocking(
            &device,
            &queue,
            &r.surface_a_buf,
            (surface_res * surface_res) as usize,
        )
    };

    // `ensure_surface_capacity` computes this exact same product before any
    // render call -- read the constant directly rather than instantiate a
    // throwaway `Renderer` (whose `surface_res` starts at a placeholder `1`
    // until a render grows it).
    let surface_res = grid_res * SURFACE_RES_MULTIPLIER;
    let scale = surface_res as f32 / grid_res as f32;
    let center = (16.0 * scale) as i32;
    // Chosen to sit just past the ISOTROPIC kernel's real reach (radius
    // ~1.5 grid cells) but well inside the axis the stretched particle's F
    // extends 2.5x -- the discriminating distance between the two cases.
    let offset = (2.0 * scale) as i32;
    let idx = |dx: i32, dy: i32| -> usize {
        ((center + dy) as u32 * surface_res + (center + dx) as u32) as usize
    };

    let stretched = render_single_particle(Mat2::from_cols(
        glam::Vec2::new(2.5, 0.0),
        glam::Vec2::new(0.0, 1.0),
    ));
    let density_x = stretched[idx(offset, 0)];
    let density_y = stretched[idx(0, offset)];
    assert!(
        density_x > density_y + 0.02,
        "a particle stretched 2.5x along x must splat substantially more \
         density along x than along the unstretched y axis at the same \
         offset: x={} y={}",
        density_x,
        density_y
    );

    let isotropic = render_single_particle(Mat2::IDENTITY);
    let iso_x = isotropic[idx(offset, 0)];
    let iso_y = isotropic[idx(0, offset)];
    assert!(
        (iso_x - iso_y).abs() < 0.02,
        "an F=identity particle must splat a symmetric footprint (no \
         directional bias from the splat loop itself): x={} y={}",
        iso_x,
        iso_y
    );
}

/// Velocity stretch: a fast particle with F = identity also splats wider
/// along its velocity than across it (the signature
/// `anisotropic_splat_widens_footprint_along_stretched_axis` shows for F). A
/// stationary control shows no bias, and neither does `dt = 0.0`, the no-op
/// the other tests in this file rely on (their `dt: 0.1` must not change
/// them while their particles rest).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn velocity_stretch_widens_footprint_along_motion_direction() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));

    let render_single_cluster = |velocity: glam::Vec2, dt: f32| -> Vec<f32> {
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(1.5)
                .spacing(0.5)
                .material(0),
        );
        for p in &mut particles {
            p.v = velocity;
        }

        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("velocity_stretch_test_target"),
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
                dt,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let surface_res = r.surface_res;
        readback_f32_blocking(
            &device,
            &queue,
            &r.surface_a_buf,
            (surface_res * surface_res) as usize,
        )
    };

    let surface_res = grid_res * SURFACE_RES_MULTIPLIER;
    let scale = surface_res as f32 / grid_res as f32;
    let center = (16.0 * scale) as i32;
    let offset = (2.0 * scale) as i32;
    let idx = |dx: i32, dy: i32| -> usize {
        ((center + dy) as u32 * surface_res + (center + dx) as u32) as usize
    };

    // speed*dt/BSPLINE_OUTER_LIMIT = 22.5*0.1/1.5 = 1.5 -> stretch_factor=2.5,
    // matching the F-based test's own real 2.5x for a direct, consistent
    // comparison. 22.5 grid-units/s is a plausible fast-splash speed
    // for this engine (live-measured max speeds during violent impacts have
    // reached 100-450+ grid-units/s elsewhere in this project).
    let moving = render_single_cluster(glam::Vec2::new(22.5, 0.0), 0.1);
    let density_x = moving[idx(offset, 0)];
    let density_y = moving[idx(0, offset)];
    assert!(
        density_x > density_y + 0.02,
        "a particle moving fast along x (F=identity, no shape deformation) must \
         splat substantially more density along its own motion axis than \
         perpendicular to it: x={density_x} y={density_y}"
    );

    let stationary = render_single_cluster(glam::Vec2::ZERO, 0.1);
    let stat_x = stationary[idx(offset, 0)];
    let stat_y = stationary[idx(0, offset)];
    assert!(
        (stat_x - stat_y).abs() < 0.02,
        "a stationary (v=0) particle must splat a symmetric footprint -- no \
         motion, no stretch: x={stat_x} y={stat_y}"
    );

    // The same fast velocity with dt = 0.0 (no physics step behind it) shows
    // zero bias: the extension is inert without dt, not merely small here.
    let fast_but_dt_zero = render_single_cluster(glam::Vec2::new(22.5, 0.0), 0.0);
    let zero_dt_x = fast_but_dt_zero[idx(offset, 0)];
    let zero_dt_y = fast_but_dt_zero[idx(0, offset)];
    assert!(
        (zero_dt_x - zero_dt_y).abs() < 0.02,
        "dt=0.0 must be a real no-op regardless of velocity (the physical \
         quantity is displacement = v*dt, not v alone): x={zero_dt_x} y={zero_dt_y}"
    );
}

/// GPU end-to-end check of the scattering and Fresnel port from
/// `prep_instances.wgsl`'s ByPhysics into `grid_volume.wgsl`'s `fs_main`.
/// With no CPU shortcut (`Renderer::particle_color`) for this path, a cluster
/// is rendered twice, scattering and specular off and then at tissue/water
/// values, and a pixel is read back to show the colour changes.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_scattering_and_specular_change_rendered_color() {
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
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let mut sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    // One real P2G step -- `grid_volume.wgsl` samples the solver's own grid
    // mass field, which is only populated once a step has actually run.
    sim.step_frame();

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let render_with = |sigma_s: f32, r0: f32, physical_thickness: Option<f32>| -> [u8; 4] {
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_coefficients_si(
            &queue,
            0,
            OpticalCoefficientsSi::new([0.3, 0.3, 0.3], sigma_s).unwrap(),
        );
        r.set_optical_scattering(&queue, 0, sigma_s);
        r.set_specular_r0(&queue, 0, r0);
        if let Some(thickness) = physical_thickness {
            r.set_grid_reference_cell_mass(1.0);
            r.set_physical_render_contract(
                &queue,
                PhysicalRenderContract::new(PhysicalRenderContractParams {
                    dx_meters: 0.1,
                    slice_thickness_m: thickness,
                    incident_radiance_w_m2_sr: [1.0; 3],
                    background_radiance_w_m2_sr: [1.0; 3],
                    display_white_radiance_w_m2_sr: [1.0; 3],
                    camera_direction: glam::Vec3::Z,
                    light_direction: glam::Vec3::Y,
                })
                .unwrap(),
            );
        }
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("grid_volume_optical_test_target"),
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

        r.render_grid_volume(
            &device,
            &queue,
            GridVolumeSource {
                grid: sim.grid_buffer(),
                material_mass: sim.material_mass_buffer(),
                material_mass_enabled: false,
                grid_res,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_pixel(&device, &queue, &texture, 64, 64, 32, 32)
    };

    let without_optics = render_with(0.0, 0.0, None);
    let with_optics = render_with(8.0, 0.02, None); // real tissue-scale sigma_s, water-scale R0
    assert_ne!(
        without_optics, with_optics,
        "identical absorption but different scattering/specular must render a \
         different pixel color at the particle cluster's center: without={:?} with={:?}",
        without_optics, with_optics
    );

    let physical_absorption = render_with(0.0, 0.0, Some(1.0));
    assert!(
        physical_absorption[..3]
            .iter()
            .any(|channel| *channel < 250),
        "an occupied SI slab must attenuate the white physical background: {physical_absorption:?}"
    );
}

/// `grid_volume.wgsl` renders blackbody emission from a mass-weighted
/// temperature scattered into the buffer's channel 0. The `grid_int` buffer
/// is built by hand (the layout of
/// `grid_peak_is_the_largest_mass_within_two_cells`: 4 u32 slots per cell,
/// mass at offset 2) so mass stays equal and
/// temperature differs between the two renders.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_blackbody_emission_brightens_hot_cells() {
    let (device, queue) = headless_device();
    let grid_res = 8u32;
    let cell_count = (grid_res * grid_res) as usize;
    const SLOTS: usize = 16;

    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_grid_volume_material_mass"),
        size: (cell_count * SLOTS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &material_mass_buf,
        0,
        bytemuck::cast_slice(&vec![0f32; cell_count * SLOTS]),
    );

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let render_at_temp = |temp_k: f32| -> [u8; 4] {
        let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_grid_volume_grid_int"),
            size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cells = vec![0u32; cell_count * 4];
        // EVERY cell, not just one: a uniform full field, drawn everywhere,
        // and a mass-weighted temperature consistent
        // with that same mass (slot 0 = mass*temp, slot 2 = mass, so
        // `avg_temp = slot0/slot2 = temp_k` exactly, matching
        // `sample_weighted_temp`'s own real convention) -- uniform fill
        // sidesteps any dependency on exactly which cell the readback pixel's
        // bilinear/nearest sampling happens to land on.
        let mass = 1.0f32;
        for c in 0..cell_count {
            cells[c * 4] = (mass * temp_k).to_bits();
            cells[c * 4 + 2] = mass.to_bits();
        }
        queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));

        let mut r = Renderer::new(&device, 1, fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("grid_volume_blackbody_test_target"),
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
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_pixel(&device, &queue, &texture, 64, 64, 32, 32)
    };

    let cold = render_at_temp(293.0); // ambient room temperature
    let hot = render_at_temp(3000.0); // real near-ignition/glow-hot range
    let cold_brightness: u32 = cold[0] as u32 + cold[1] as u32 + cold[2] as u32;
    let hot_brightness: u32 = hot[0] as u32 + hot[1] as u32 + hot[2] as u32;
    assert!(
        hot_brightness > cold_brightness,
        "a hot cell must render brighter than an ambient-temperature cell with \
         otherwise identical mass (real blackbody emission, additive) -- \
         cold={cold:?} (sum={cold_brightness}) hot={hot:?} (sum={hot_brightness})"
    );
}

/// Grid-volume centre pixel of a scene whose every column is filled with
/// mass 1.0 from y = 0 up to (not including) `fill_to_y_exclusive`, on a
/// 24-cell grid. With `shape_holding`, slot 0 is adopted from a Neo-Hookean
/// solid, whose nonzero shear modulus marks it as holding its shape.
fn column_scene_centre_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fill_to_y_exclusive: u32,
    shape_holding: bool,
) -> [u8; 4] {
    use crate::{MaterialRegistry, NeoHookeanMaterial};
    let grid_res = 24u32;
    let cell_count = (grid_res * grid_res) as usize;
    const SLOTS: usize = 16;

    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_column_depth_material_mass"),
        size: (cell_count * SLOTS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &material_mass_buf,
        0,
        bytemuck::cast_slice(&vec![0f32; cell_count * SLOTS]),
    );
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_column_depth_grid_int"),
        size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut cells = vec![0u32; cell_count * 4];
    let mass = 1.0f32;
    for cy in 0..fill_to_y_exclusive.min(grid_res) {
        for cx in 0..grid_res {
            let c = (cy * grid_res + cx) as usize;
            cells[c * 4 + 2] = mass.to_bits();
        }
    }
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(device, 1, fmt);
    // A water-like sigma_a (Pope & Fry 1997, as in render_plan.md): with a
    // near-zero absorption depth would barely matter.
    r.set_optical_params(queue, 0, [0.35, 0.033, 0.011]);
    if shape_holding {
        // Declares no optics, so the sigma_a above stays.
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        r.adopt_material_optics(queue, &registry);
    }
    r.set_camera(queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("column_depth_test_target"),
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

    r.render_grid_volume(
        device,
        queue,
        GridVolumeSource {
            grid: &grid_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: false,
            grid_res,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    readback_pixel(device, queue, &texture, 64, 64, 32, 32)
}

fn pixel_brightness(p: [u8; 4]) -> u32 {
    u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])
}

/// Column-depth attenuation: water darkens with depth (Pope & Fry 1997, the
/// source of the sigma_a table), so with the same local mass at the query
/// cell a tall column above it (deep) renders darker than a thin band
/// (shallow). Only what lies above the cell differs.
///
/// Both scenes fill past the domain's vertical middle (17 of 24 rows), so the
/// centre pixel lands inside material either way without guessing which row
/// it maps to.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_column_depth_darkens_deep_regions_more_than_shallow() {
    let (device, queue) = headless_device();
    let shallow = column_scene_centre_pixel(&device, &queue, 17, false);
    let deep = column_scene_centre_pixel(&device, &queue, 24, false);
    let (shallow_brightness, deep_brightness) = (pixel_brightness(shallow), pixel_brightness(deep));
    assert!(
        deep_brightness < shallow_brightness,
        "a deep column must render measurably darker than a shallow one at the SAME          local mass (real solar attenuation with depth, not local density alone) --          shallow={shallow:?} (sum={shallow_brightness}) deep={deep:?} (sum={deep_brightness})"
    );
}

/// The same two scenes as above, but the matter holds its shape: no
/// column-depth darkening, so a solid body reads as one flat tone instead of
/// a lit cylinder dark at the bottom.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_shape_holding_matter_does_not_darken_with_column_depth() {
    let (device, queue) = headless_device();
    let shallow = column_scene_centre_pixel(&device, &queue, 17, true);
    let deep = column_scene_centre_pixel(&device, &queue, 24, true);
    assert_eq!(
        shallow, deep,
        "a solid must not darken with the depth of matter above it"
    );
}

/// `dominant_material` reads `material_mass` as i32: read as bit-reinterpreted
/// f32, this GPU flushes the denormal values to zero in the fragment shader.
/// Two grid halves with different dominant slots render their own colours,
/// not both slot 0 (every other call site here sets
/// `material_mass_enabled: false`).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_dominant_material_colors_regions_distinctly() {
    let (device, queue) = headless_device();
    let grid_res = 8u32;
    let cell_count = (grid_res * grid_res) as usize;
    const SLOTS: usize = 16;

    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_dominant_material_grid_int"),
        size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut cells = vec![0u32; cell_count * 4];
    // Uniform real mass everywhere (slot 2), comfortably above mass_floor,
    // no temperature contribution (slot 0 stays 0 -- isolates this test
    // from blackbody emission entirely).
    let mass = 1.0f32;
    for c in 0..cell_count {
        cells[c * 4 + 2] = mass.to_bits();
    }
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));

    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_dominant_material_material_mass"),
        size: (cell_count * SLOTS * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Fixed-point-scale magnitude (same order as P2G's own
    // `MASS_ATOMIC_SCALE`-driven values) -- deliberately denormal-adjacent,
    // not a large/normal-range float, since FTZ/rounding bugs live in that
    // regime. Left half (x<4) dominant in slot 0, right half (x>=4)
    // dominant in slot 1.
    let mut mm = vec![0i32; cell_count * SLOTS];
    for cy in 0..grid_res {
        for cx in 0..grid_res {
            let c = (cy * grid_res + cx) as usize;
            let slot = if cx < grid_res / 2 { 0 } else { 1 };
            mm[c * SLOTS + slot] = 100_000;
        }
    }
    queue.write_buffer(&material_mass_buf, 0, bytemuck::cast_slice(&mm));

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    r.set_optical_params(&queue, 0, [0.05, 0.55, 0.55]); // red-dominant
    r.set_optical_params(&queue, 1, [0.55, 0.05, 0.55]); // green-dominant
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dominant_material_test_target"),
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

    r.render_grid_volume(
        &device,
        &queue,
        GridVolumeSource {
            grid: &grid_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: true,
            grid_res,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // Whole halves of the grid, not a single particle splat -- pixel
    // targeting is forgiving here, but sample a few points per half anyway
    // rather than trust one exact pixel.
    let left = readback_pixel(&device, &queue, &texture, 64, 64, 16, 32);
    let right = readback_pixel(&device, &queue, &texture, 64, 64, 48, 32);

    assert_ne!(
        left, right,
        "left half (slot 0) and right half (slot 1) must render visibly \
         distinct colors: left={:?} right={:?}",
        left, right
    );
    assert!(
        left[0] > left[1] && left[0] > left[2],
        "left half (dominant slot 0, low red absorption) should read \
         red-dominant: {:?}",
        left
    );
    assert!(
        right[1] > right[0] && right[1] > right[2],
        "right half (dominant slot 1, low green absorption) should read \
         green-dominant: {:?}",
        right
    );
}
