use super::*;

/// Subsurface scattering changes ByPhysics's output: two materials with the
/// same absorption and different `sigma_s` render differently (see
/// prep_instances.wgsl's ByPhysics branch for the single-scattering albedo).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn scattering_changes_by_physics_color() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 1, [0.3, 0.3, 0.3]);
    r.set_optical_scattering(&queue, 1, 5.0); // real tissue-scale reduced scattering coeff

    let mut p0 = Particle::zeroed();
    p0.material_id = 0;
    p0.deformation_gradient = Mat2::IDENTITY;
    let mut p1 = p0;
    p1.material_id = 1;

    let c0 = r.particle_color(&p0, 0);
    let c1 = r.particle_color(&p1, 0);
    assert_ne!(
        c0, c1,
        "identical absorption but different sigma_s must render differently"
    );
}

/// Specular Fresnel reflectance changes ByPhysics's output, the R0 term.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn specular_r0_changes_by_physics_color() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 1, [0.3, 0.3, 0.3]);
    r.set_specular_r0(&queue, 1, 0.02); // real water-scale Fresnel base reflectance

    let mut p0 = Particle::zeroed();
    p0.material_id = 0;
    p0.deformation_gradient = Mat2::IDENTITY;
    let mut p1 = p0;
    p1.material_id = 1;

    let c0 = r.particle_color(&p0, 0);
    let c1 = r.particle_color(&p1, 0);
    assert_ne!(
        c0, c1,
        "identical absorption but different specular R0 must render differently"
    );
}

/// `Renderer::new` must succeed and the (now auto-uploading) optical setters
/// must not panic with the extended (scattering + specular) `OpticalTable`
/// layout -- a end-to-end check that the WGSL struct and Rust struct
/// stayed in sync (a mismatch here would show up as a wgpu validation panic,
/// not a silent bug).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn renderer_construction_and_optical_upload_survive_extended_table() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(&queue, 0, 8.0);
    r.set_specular_r0(&queue, 0, 0.02);
}

#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn physical_contract_drives_cpu_beer_lambert_in_si() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_coefficients_si(
        &queue,
        0,
        OpticalCoefficientsSi::new([2.0, 1.0, 0.5], 0.0).unwrap(),
    );
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.01,
            slice_thickness_m: 0.25,
            incident_radiance_w_m2_sr: [10.0; 3],
            background_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            display_white_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            camera_direction: glam::Vec3::Z,
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    let mut p = Particle::zeroed();
    p.deformation_gradient = Mat2::IDENTITY;
    let got = r.particle_color(&p, 0);
    let expected = [(-0.5f32).exp(), (-0.25f32).exp(), (-0.125f32).exp()];
    for (channel, want) in got[..3].iter().copied().zip(expected) {
        assert!(
            (channel - want).abs() < 1.0e-6,
            "got {channel}, expected {want}"
        );
    }
}

/// The GPU particle path must consume the same SI slab contract as the CPU
/// path. Read back the compute-generated instance directly, avoiding the
/// unrelated tiny-quad rasterization limitation documented below.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn physical_contract_drives_gpu_particle_beer_lambert_in_si() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let config = SimConfig::standard(8, 0.01, glam::Vec2::ZERO);
    let mut particle = Particle::zeroed();
    particle.x = glam::Vec2::splat(4.0);
    particle.mass = 1.0;
    particle.volume = 1.0;
    particle.deformation_gradient = Mat2::IDENTITY;
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim = GpuSimulation::with_device(
        device.clone(),
        queue.clone(),
        config,
        vec![particle],
        registry,
    );

    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_coefficients_si(
        &queue,
        0,
        OpticalCoefficientsSi::new([2.0, 1.0, 0.5], 0.0).unwrap(),
    );
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.01,
            slice_thickness_m: 0.25,
            incident_radiance_w_m2_sr: [10.0; 3],
            background_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            display_white_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            camera_direction: glam::Vec3::Z,
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    r.set_camera(&queue, 8, 32, 32, 1.0, true);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("physical_gpu_particle_target"),
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    r.render_gpu(
        &device,
        &queue,
        GpuRenderParams {
            particle_buf: sim.particle_buffer(),
            particle_count: 1,
            output_view: &texture.create_view(&wgpu::TextureViewDescriptor::default()),
            clear: true,
            interp_alpha: 1.0,
        },
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let raw = readback_f32_blocking(&device, &queue, &r.storage_instances, 12);
    let expected = [(-0.5f32).exp(), (-0.25f32).exp(), (-0.125f32).exp()];
    for (channel, want) in raw[8..11].iter().copied().zip(expected) {
        assert!(
            (channel - want).abs() < 1.0e-5,
            "got {channel}, expected {want}"
        );
    }
}

/// End-to-end GPU path (the one LP actually uses, `render_gpu`): real
/// particles on a `GpuSimulation`, real compute dispatch through
/// `prep_instances.wgsl` with the extended `OpticalTable`, real render pass to
/// an offscreen texture. Proves the whole pipeline survives, not just that
/// `Renderer::new` compiles the shader in isolation.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn render_gpu_survives_scattering_and_specular_end_to_end() {
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
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(&queue, 0, 8.0);
    r.set_specular_r0(&queue, 0, 0.02);
    r.set_camera(&queue, 32, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_gpu_test_target"),
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

    r.render_gpu(
        &device,
        &queue,
        GpuRenderParams {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            output_view: &view,
            clear: true,
            interp_alpha: 1.0,
        },
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
}

/// `blackbody.inc.wgsl` claims to be a mirror of `energy::radiation`. This is
/// what makes that claim checkable on real hardware rather than by reading
/// both and hoping.
///
/// A scene is rendered in `ByThermal` (emission and nothing else) with the
/// exposure anchored at the scene's own temperature, so the expected pixel is
/// exactly the blackbody colour -- no exposure factor, no lighting, no
/// absorption in the way. The readback is sRGB-encoded, so it is decoded back
/// to linear before comparison against the CPU's own Planck integration.
///
/// Tolerance 0.06 per channel covers three stacked approximations that are
/// each documented where they live: the Kang 2002 locus fit against real
/// Planck (tested separately at 0.03), 8-bit quantization, and the shader's
/// f32 arithmetic against the CPU's f64.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn gpu_blackbody_emission_matches_planck_on_the_cpu() {
    use crate::energy::radiation::blackbody_linear_srgb;
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let srgb_to_linear = |encoded: u8| -> f32 {
        let value = f32::from(encoded) / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };

    for temperature in [2000.0f32, 3500.0, 6000.0] {
        let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(6.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temperature;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_color_mode(ColorMode::ByThermal);
        r.set_camera(&queue, 32, 256, 256, 0.6, true);
        // Exposure anchored at the scene's own temperature: (T/T_ref)^4 = 1.
        r.set_emission_reference_temperature(&queue, temperature);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blackbody_parity_target"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
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
        r.render_gpu(
            &device,
            &queue,
            GpuRenderParams {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                output_view: &view,
                clear: true,
                interp_alpha: 1.0,
            },
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let pixel = readback_brightest_pixel(&device, &queue, &texture, 256, 256);
        let rendered = [
            srgb_to_linear(pixel[0]),
            srgb_to_linear(pixel[1]),
            srgb_to_linear(pixel[2]),
        ];
        let expected = blackbody_linear_srgb(temperature);
        for channel in 0..3 {
            assert!(
                (rendered[channel] - expected[channel]).abs() < 0.06,
                "{temperature}K channel {channel}: GPU rendered {rendered:?} \
                 (raw {pixel:?}) but Planck through the CIE observer says {expected:?}"
            );
        }
    }
}

/// Under a real SI contract the render must still carry scattering and
/// Fresnel, not absorption alone.
///
/// This guards a gap that was real until this was written: the contract
/// branch computed `background * exp(-sigma_a * path)` and nothing else, so
/// turning on measured SI optics silently cost a scene its subsurface glow
/// and its specular. Two materials identical except for `sigma_s`, and two
/// identical except for `R0`, must each render differently.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn si_contract_path_keeps_scattering_and_fresnel() {
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};

    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.01,
            slice_thickness_m: 1.0,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.5; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: glam::Vec3::new(0.0, 1.0, -1.0),
        })
        .unwrap(),
    );
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 1, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 2, [0.3, 0.3, 0.3]);
    r.set_optical_scattering(&queue, 1, 5.0);
    r.set_specular_r0(&queue, 2, 0.2);

    let mut plain = Particle::zeroed();
    plain.material_id = 0;
    plain.deformation_gradient = Mat2::IDENTITY;
    let mut scattering = plain;
    scattering.material_id = 1;
    let mut reflective = plain;
    reflective.material_id = 2;

    let base = r.particle_color(&plain, 0);
    assert_ne!(
        base,
        r.particle_color(&scattering, 0),
        "sigma_s must change the SI-contract color, not be dropped with the rest          of the scattering term"
    );
    assert_ne!(
        base,
        r.particle_color(&reflective, 0),
        "Fresnel R0 must change the SI-contract color"
    );
}

/// Luminescence: matter that glows WITHOUT being hot must actually light
/// the scene, through the same photon-diffusion solver thermal emitters
/// already feed.
///
/// A firefly is not hot. Before this, the only way to emit light was to be
/// at a few thousand kelvin, so a cold glowing thing was impossible to
/// express. The declared source enters `S` in
/// `(1/c) dphi/dt = D grad^2 phi - mu_a phi + S`, weighted by how much
/// emitting matter the cell holds.
///
/// Same cold scene twice, at ambient temperature throughout, differing only
/// by whether the material declares a luminous emission.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn luminescent_material_lights_the_scene_without_being_hot() {
    use crate::gpu::GpuSimulation;
    use crate::materials::registry::MaterialRegistry;
    use crate::{NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    /// A cold material that glows, the way a firefly or a luminous fungus
    /// does. 2000 W/m^3 is a stated scene value, not a measurement of any
    /// particular organism.
    #[derive(Debug)]
    struct Luminescent(NeoHookeanMaterial);

    impl crate::materials::MaterialModel for Luminescent {
        fn constitutive_model(&self) -> crate::materials::ConstitutiveModel {
            self.0.constitutive_model()
        }
        fn kirchhoff_stress(&self, particles: &crate::particle::Particles, i: usize) -> glam::Mat2 {
            self.0.kirchhoff_stress(particles, i)
        }
        fn stress_volume(&self, particles: &crate::particle::Particles, i: usize) -> f32 {
            self.0.stress_volume(particles, i)
        }
        fn params(&self) -> crate::materials::MaterialParams {
            self.0.params()
        }
        fn luminous_emission_w_m3(&self) -> f32 {
            2000.0
        }
    }

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let grid_res = 32u32;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_glow = |luminous: bool| -> u32 {
        let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(4.0)
                .spacing(0.5)
                .material(0),
        );
        // Cold throughout: nothing here may glow by being hot.
        for p in particles.iter_mut() {
            p.temperature = 293.0;
        }
        let base = NeoHookeanMaterial::new(100.0, 50.0);
        let registry: MaterialRegistry = if luminous {
            MaterialRegistry::with_default(Box::new(Luminescent(base)))
        } else {
            MaterialRegistry::with_default(Box::new(base))
        };
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
        r.adopt_material_optics(&queue, sim.registry());

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("luminescence_target"),
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
        // The fluence field is persistent, so it needs a few frames to build.
        for _ in 0..30 {
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
        }
        let p = readback_pixel(&device, &queue, &texture, 64, 64, 32, 32);
        u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])
    };

    let dark = render_glow(false);
    let glowing = render_glow(true);
    assert!(
        glowing > dark,
        "a material declaring a luminous source must light the scene even at          293 K: glowing={glowing} vs non-glowing={dark}"
    );
}

/// The physical fact the previous hand-drawn ramp inverted, checked through
/// the GPU pipeline rather than in isolation: a hotter body renders
/// bluer, not redder.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn gpu_thermal_emission_gets_bluer_with_temperature() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_at = |temperature: f32| -> [u8; 4] {
        let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(6.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temperature;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_color_mode(ColorMode::ByThermal);
        r.set_camera(&queue, 32, 256, 256, 0.6, true);
        r.set_emission_reference_temperature(&queue, temperature);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blackbody_hue_target"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
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
        r.render_gpu(
            &device,
            &queue,
            GpuRenderParams {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                output_view: &view,
                clear: true,
                interp_alpha: 1.0,
            },
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_brightest_pixel(&device, &queue, &texture, 256, 256)
    };

    let ember = render_at(1800.0);
    let hot = render_at(9000.0);
    assert!(
        ember[2] < 80 && hot[2] > 200,
        "1800K must render as a deep red with little blue and 9000K as blue-white: \
         ember={ember:?} hot={hot:?}"
    );
}

/// `render_gpu`'s own compute-shader instance prep (`prep_instances.wgsl`)
/// must actually produce visible particle pixels, not just "not panic" --
/// the test above only proved the pipeline survives, never read back a
/// single pixel. Real particles at a known grid position, `ByMaterial`
/// color mode (material 0's real palette entry, `material_color(0)` in
/// `prep_instances.wgsl` = `vec4(0.35, 0.65, 1.00, 1.0)`, a distinct blue),
/// must show up as that color, not the clear color (0.05, 0.05, 0.08).
///
/// The target matches the 256 x 256 window `set_camera` is given. Until
/// issue #45 it was 64 x 64 and nothing was drawn:
/// `subpixel_particles_light_only_pixels_whose_centre_they_cover` below
/// measures why.
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
#[test]
fn render_gpu_produces_visible_particle_pixels_not_just_clear_color() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = pixel_test_disk(0.0);
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    assert!(sim.particle_count() > 0, "test setup must spawn particles");

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_color_mode(ColorMode::ByMaterial);
    r.set_camera(&queue, 32, 256, 256, 0.6, true);
    let (texture, view) = pixel_test_target(&device, fmt, 256);

    r.render_gpu(
        &device,
        &queue,
        GpuRenderParams {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            output_view: &view,
            clear: true,
            interp_alpha: 1.0,
        },
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // A partly covered pixel is blended toward the background, so the
    // brightest pixel carries the particles' own colour. Clear color
    // (0.05,0.05,0.08 linear) is near-black; material-0 blue
    // (0.35,0.65,1.00) is bright, especially in the blue channel.
    let brightest = readback_brightest_pixel(&device, &queue, &texture, 256, 256);
    assert!(
        brightest[2] > 120 && brightest[0] < 180,
        "render_gpu must produce visible material-0 (blue) particle pixels, \
         not just the clear color; brightest pixel in the target {brightest:?}"
    );
}

/// Control test for the above: SAME scene/camera/texture, but through the
/// CPU `render_slice()` path instead of `render_gpu` -- isolates whether a
/// blank result is specific to the GPU compute-prep path or a shared
/// `draw_pass`/pipeline problem that would affect both.
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
#[test]
fn render_cpu_produces_visible_particle_pixels_control() {
    let (device, queue) = headless_device();
    let particles = pixel_test_disk(0.0);
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, particles.len(), fmt);
    r.set_color_mode(ColorMode::ByMaterial);
    r.set_camera(&queue, 32, 256, 256, 0.6, true);
    let (texture, view) = pixel_test_target(&device, fmt, 256);

    r.render_slice(&device, &queue, &particles, &view, true);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let brightest = readback_brightest_pixel(&device, &queue, &texture, 256, 256);
    assert!(
        brightest[2] > 120 && brightest[0] < 180,
        "control: CPU render_slice() must produce visible particle pixels with the \
         exact same scene/camera/texture params as the GPU test above; brightest \
         pixel in the target {brightest:?}"
    );
}

/// Issue #45, measured. The particle pass point-samples: a fragment exists
/// only where a pixel centre falls inside a particle's quad, and the round
/// clip keeps it only within half the quad's width of the particle. A
/// particle smaller than a pixel therefore lights one pixel or none,
/// depending on where it sits, with no partial coverage.
///
/// The two tests above drew into a 64 x 64 target until this was found: 2
/// pixels per cell, so a 0.6-cell particle is a disc of radius 0.6 px. The
/// spawn lattice, 0.5 cells apart from x = 10.0, puts every particle on a
/// pixel corner, 0.71 px from the nearest pixel centre, and every fragment
/// was discarded: 0 of 4096 pixels lit. A quarter-cell (half-pixel) shift
/// puts each particle on a pixel centre and lights exactly one pixel per
/// particle. On the same 64 x 64 target a 4-cell particle, or a square one,
/// lights the disk.
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
#[test]
fn subpixel_particles_light_only_pixels_whose_centre_they_cover() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let lit_pixels = |shift: f32| -> usize {
        let particles = pixel_test_disk(shift);
        let mut r = Renderer::new(&device, particles.len(), fmt);
        r.set_color_mode(ColorMode::ByMaterial);
        r.set_camera(&queue, 32, 64, 64, 0.6, true);
        let (texture, view) = pixel_test_target(&device, fmt, 64);
        r.render_slice(&device, &queue, &particles, &view, true);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let luminance = readback_luminance_grid(&device, &queue, &texture, 64, 64, 1);
        let background = luminance[0];
        luminance
            .iter()
            .filter(|&&l| (l - background).abs() > 0.5)
            .count()
    };
    let count = pixel_test_disk(0.0).len();
    let on_corners = lit_pixels(0.0);
    let on_centres = lit_pixels(0.25);
    assert_eq!(
        (on_corners, on_centres),
        (0, count),
        "a 0.6 px disc lights no pixel from a pixel corner and exactly one from \
         a pixel centre ({count} particles)"
    );
}
