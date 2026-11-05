use super::*;

/// Same real end-to-end check as
/// `grid_volume_scattering_and_specular_change_rendered_color`, for
/// `curvature_flow.wgsl`'s single-phase `fs_main` -- the other fragment
/// shader that just received the same optical-parity port.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_scattering_and_specular_change_rendered_color() {
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
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let render_with = |sigma_s: f32, r0: f32| -> [u8; 4] {
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_optical_scattering(&queue, 0, sigma_s);
        r.set_specular_r0(&queue, 0, r0);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("curvature_flow_optical_test_target"),
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
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_pixel(&device, &queue, &texture, 64, 64, 32, 32)
    };

    let without_optics = render_with(0.0, 0.0);
    let with_optics = render_with(8.0, 0.02);
    assert_ne!(
        without_optics, with_optics,
        "identical absorption but different scattering/specular must render a \
         different pixel color at the particle cluster's center: without={:?} with={:?}",
        without_optics, with_optics
    );
}

/// Diagnostic, not a regression test -- run manually via `cargo test
/// diagnose_curvature_flow_edge_hair_pixels -- --ignored --nocapture` when
/// investigating "hair" fuzz around Surface mode's silhouette edges. Prints
/// an RGBA scanline crossing a cold (ambient-temperature-only, no
/// blackbody emission at all) cluster's own boundary, so the actual pixel
/// numbers can be read directly instead of guessing from a screenshot --
/// isolates whether the artifact depends on temperature/emission at all.
#[test]
#[ignore]
fn diagnose_curvature_flow_edge_hair_pixels() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Cold, ambient-temperature-only cluster -- same SIGMA_NEO/scattering
    // basic_jellies_gpu.rs's own MAT_NEO uses, no heat at all, to isolate
    // whether the reported edge artifact depends on temperature/emission.
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(8.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_params(&queue, 0, [0.05, 0.55, 0.60]);
    r.set_optical_scattering(&queue, 0, 0.02);
    r.set_specular_r0(&queue, 0, 0.01);
    r.set_camera(&queue, grid_res, 128, 128, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("diagnose_edge_hair_target"),
        size: wgpu::Extent3d {
            width: 128,
            height: 128,
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
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // Raw density (`surface_a_buf`, after curvature flow) along the same row,
    // to read the interior depth directly instead of from colour.
    let surface_res = r.surface_res;
    let mass_values = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );
    let scale = surface_res as f32 / grid_res as f32;
    let row = (16.0 * scale) as u32; // grid y=16 -> surface row, matches this scene's camera-space y=64
    for sx in 0..surface_res {
        let m = mass_values[(row * surface_res + sx) as usize];
        if m > 1.0e-4 {
            eprintln!("surface_cell x={sx} mass={m}");
        }
    }

    // Scanline straight through the cluster's own edge (cluster is centered
    // at grid (16,16) with real radius 8, camera maps grid->pixel via
    // set_camera's own 0.6 zoom over a 128px target -- print a wide enough
    // range to see background -> edge -> interior.
    for x in 40..115 {
        let px = readback_pixel(&device, &queue, &texture, 128, 128, x, 64);
        eprintln!("y=64 x={x} rgba={px:?}");
    }
    // Also scan through a more CURVED part of the silhouette (near the
    // disk's top cap, not straight through its widest point) -- the
    // reported "hair" artifact looks worst on curved edges in the
    // screenshot, a flat scan through dead-center might miss it.
    for x in 40..115 {
        let px = readback_pixel(&device, &queue, &texture, 128, 128, x, 40);
        eprintln!("y=40 x={x} rgba={px:?}");
    }
}

/// Surface mode renders blackbody emission from a mass-weighted temperature
/// (`surface_temp_atomic`/`surface_temp_final`, the formula of
/// `grid_volume.wgsl`). Built from particles, not a hand-made buffer: two
/// `GpuSimulation`s from the same spawn region, differing only in
/// `particles.temperature`, so the temperature reaches the pixel through
/// splat, convert and curvature flow.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_blackbody_emission_brightens_hot_cluster() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_at_temp = |temp_k: f32| -> [u8; 4] {
        let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(4.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temp_k;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("curvature_flow_blackbody_test_target"),
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
                material_mass_enabled: false,
                dt: 0.1,
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
        "a hot particle cluster must render brighter than an ambient-temperature \
         one with otherwise identical mass/shape (real blackbody emission, \
         additive, single-phase curvature-flow surface mode) -- \
         cold={cold:?} (sum={cold_brightness}) hot={hot:?} (sum={hot_brightness})"
    );
}

/// Light diffusion (`curvature_flow.wgsl`'s Pass 1e) accumulates across
/// frames: its fluence Phi persists, since diffusion needs time to spread.
/// The same static scene (fixed temperature, zero velocity; the wave field
/// stays flat because its forcing is the density's change) is rendered
/// repeatedly through one `Renderer`. A hot scene's brightness at a pixel
/// rises from frame 1 to frame 30; an ambient scene drifts by only a small
/// fraction of that, ruling out frame-to-frame noise. Not zero: the emission
/// goes as Stefan-Boltzmann's `(T/T_ref)^4`, so 293 K still radiates, ~11000x
/// less than 3000 K at the default 3000 K anchor.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_diffusion_builds_up_real_glow_over_multiple_frames() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_n_frames = |temp_k: f32, n_frames: u32| -> [u8; 4] {
        let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(4.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temp_k;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light_diffusion_test_target"),
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

        let mut last = [0u8; 4];
        for _ in 0..n_frames {
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
            last = readback_pixel(&device, &queue, &texture, 64, 64, 32, 32);
        }
        last
    };

    let hot_frame1 = render_n_frames(3000.0, 1);
    let hot_frame30 = render_n_frames(3000.0, 30);
    let hot_b1: u32 = hot_frame1[0] as u32 + hot_frame1[1] as u32 + hot_frame1[2] as u32;
    let hot_b30: u32 = hot_frame30[0] as u32 + hot_frame30[1] as u32 + hot_frame30[2] as u32;
    assert!(
        hot_b30 > hot_b1,
        "a hot, unchanging scene's own rendered brightness at the SAME pixel must \
         genuinely increase from frame 1 to frame 30 as the real, persistent light-\
         diffusion field builds up -- frame1={hot_frame1:?} (sum={hot_b1}) \
         frame30={hot_frame30:?} (sum={hot_b30})"
    );

    let cold_frame1 = render_n_frames(293.0, 1);
    let cold_frame30 = render_n_frames(293.0, 30);
    let cold_b1: u32 = cold_frame1[0] as u32 + cold_frame1[1] as u32 + cold_frame1[2] as u32;
    let cold_b30: u32 = cold_frame30[0] as u32 + cold_frame30[1] as u32 + cold_frame30[2] as u32;
    let hot_drift = hot_b30.abs_diff(hot_b1);
    let cold_drift = cold_b30.abs_diff(cold_b1);
    assert!(
        cold_drift * 5 <= hot_drift,
        "an ambient (cold) scene emits real light too (`light_diffuse_main`'s own \
         (T/T_ref)^4 term is never exactly zero), but ~11000x weaker than the hot scene \
         at 293K vs 3000K -- its frame1->frame30 drift must stay a small fraction of \
         the hot scene's, not comparable to it (rules out unrelated per-frame noise \
         as the explanation): hot drift={hot_drift} (frame1={hot_frame1:?} \
         frame30={hot_frame30:?}) cold drift={cold_drift} (frame1={cold_frame1:?} \
         frame30={cold_frame30:?})"
    );
}

/// Thermal diffusion (`curvature_flow.wgsl`'s Pass 1c, `temp_avg_main`/
/// `temp_diffuse_main`): a hot (3000 K) and an ambient (293 K) cluster
/// touching, a sharp temperature step, the input an explicit stencil
/// misbehaves on if its Fourier bound (`DIFFUSION_ALPHA*DIFFUSION_DT <=
/// 0.25`) were wrong. Reads `surface_temp_float_buf`: finite everywhere, and
/// the hot side warmer than the cold side.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_thermal_diffusion_stays_finite_and_separates_hot_from_cold() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));

    let mut hot_side = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(12, 12),
            box_center: glam::Vec2::new(10.0, 16.0),
            material_id: 0,
            rng_seed: 1,
            ..SpawnRegion::for_sim(&config)
        },
    );
    for p in hot_side.iter_mut() {
        p.temperature = 3000.0;
    }
    let mut cold_side = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(12, 12),
            box_center: glam::Vec2::new(22.0, 16.0),
            material_id: 0,
            rng_seed: 2,
            ..SpawnRegion::for_sim(&config)
        },
    );
    for p in cold_side.iter_mut() {
        p.temperature = 293.0;
    }
    hot_side.extend(cold_side);
    let particles = hot_side;

    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("curvature_flow_thermal_diffusion_test_target"),
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
    let temps = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_temp_float_buf,
        (surface_res * surface_res) as usize,
    );

    assert!(
        temps.iter().all(|t| t.is_finite()),
        "the real explicit heat-equation step must never produce NaN/inf, even \
         across a genuinely sharp hot/cold boundary"
    );

    let scale = surface_res as f32 / grid_res as f32;
    let row = (16.0 * scale) as u32;
    let hot_col = (10.0 * scale) as u32;
    let cold_col = (22.0 * scale) as u32;
    let hot_val = temps[(row * surface_res + hot_col) as usize];
    let cold_val = temps[(row * surface_res + cold_col) as usize];

    assert!(
        hot_val > cold_val + 500.0,
        "the hot cluster's own settled temperature must stay substantially \
         above the cold cluster's, even after real diffusion spreads some \
         heat toward the boundary -- hot={hot_val} cold={cold_val}"
    );
    assert!(
        hot_val < 3000.0 + 50.0 && cold_val > 293.0 - 50.0,
        "neither side should overshoot its own real input temperature by more \
         than a small margin -- a real sign the explicit stencil is stable, \
         not oscillating -- hot={hot_val} cold={cold_val}"
    );
}
