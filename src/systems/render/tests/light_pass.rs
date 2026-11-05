use super::*;

/// Runs the grid-volume light pass on a grid whose cell `(x, y)` holds
/// `mass(x, y)` (reference cell mass 1), slot 0 with absorption `sigma_a`
/// and reduced scattering `sigma_s`, under a contract of cell size `dx` and light toward
/// `light`; returns the per-cell RGB transmittance, row-major (y * res + x).
fn light_pass_transmittance(
    res: u32,
    dx: f32,
    sigma_a: [f32; 3],
    sigma_s: f32,
    light: glam::Vec3,
    mass: impl Fn(u32, u32) -> f32,
) -> Vec<[f32; 3]> {
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};
    let (device, queue) = headless_device();
    let cells = (res * res) as usize;
    let mut words = vec![0u32; cells * 4];
    for y in 0..res {
        for x in 0..res {
            words[(y * res + x) as usize * 4 + 2] = mass(x, y).to_bits();
        }
    }
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light_pass_grid"),
        size: (cells * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&words));
    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light_pass_material_mass"),
        size: (cells * 16 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    r.set_optical_params(&queue, 0, sigma_a);
    r.set_optical_scattering(&queue, 0, sigma_s);
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: dx,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.25; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: light,
        })
        .unwrap(),
    );
    r.set_camera(&queue, res, 64, 64, 0.6, true);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("light_pass_target"),
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
            grid_res: res,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let raw = readback_f32_blocking(&device, &queue, &r.light_transmittance_buf, cells * 4);
    raw.chunks(4).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Gate (a) of the volumetric-shadow pass: in a uniform medium lit from
/// straight above, the light reaching a cell has crossed every cell above
/// it, so its transmittance is Beer-Lambert's `exp(-sigma * dx * n)` with
/// `n` the number of cells between it and the top edge.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_pass_matches_beer_lambert_in_a_uniform_medium() {
    let (res, dx, sigma) = (24u32, 0.1f32, [0.5f32, 1.0, 2.0]);
    let t = light_pass_transmittance(res, dx, sigma, 0.0, glam::Vec3::Y, |_, _| 1.0);
    for y in 0..res {
        let cells_above = (res - 1 - y) as f32;
        for x in [0u32, 7, res - 1] {
            let got = t[(y * res + x) as usize];
            for ch in 0..3 {
                let expected = (-sigma[ch] * dx * cells_above).exp();
                assert!(
                    (got[ch] - expected).abs() < 1.0e-3,
                    "cell ({x},{y}) channel {ch}: {} against exp(-{} * {dx} * {cells_above}) = {expected}",
                    got[ch],
                    sigma[ch]
                );
            }
        }
    }
}

/// The diffusion regime: in a strongly scattering, weakly absorbing medium
/// (snow's measured coefficients) light reaching a depth has mostly
/// scattered on the way, and it dies out with diffusion theory's effective
/// attenuation sqrt(3 mu_a (mu_a + mu_s')) (Jacques and Prahl, ECE532 notes),
/// far slower than the unscattered beam's mu_a + mu_s'.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_pass_uses_diffusion_attenuation_in_a_scattering_medium() {
    let (res, dx, sigma_a, sigma_s) = (24u32, 0.01f32, [0.055f32, 0.018, 0.0014], 236.0f32);
    let t = light_pass_transmittance(res, dx, sigma_a, sigma_s, glam::Vec3::Y, |_, _| 1.0);
    for y in 0..res {
        let cells_above = (res - 1 - y) as f32;
        let got = t[(y * res + 5) as usize];
        for ch in 0..3 {
            let mu_eff = (3.0 * sigma_a[ch] * (sigma_a[ch] + sigma_s)).sqrt();
            assert!(
                mu_eff < sigma_a[ch] + sigma_s,
                "the diffuse rate must be the slower one"
            );
            let expected = (-mu_eff * dx * cells_above).exp();
            assert!(
                (got[ch] - expected).abs() < 1.0e-3,
                "cell (5,{y}) channel {ch}: {} against exp(-{mu_eff} * {dx} * {cells_above}) = {expected}",
                got[ch]
            );
        }
    }
}

/// Gate (b): an opaque block under oblique light (toward +x, +y) shadows
/// exactly the cells whose ray toward the light crosses it. Cells whose ray
/// passes the block shrunk by one cell must be dark, cells whose ray misses
/// it grown by one cell must be fully lit; the one-cell band between is the
/// march's own discretisation and is not judged.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_pass_shadow_edge_follows_the_ray_geometry() {
    let res = 32u32;
    let t = light_pass_transmittance(
        res,
        0.1,
        [100.0; 3],
        0.0,
        glam::Vec3::new(1.0, 1.0, 0.0),
        |x, y| {
            if in_shadow_test_block(x as f32 + 0.5, y as f32 + 0.5) {
                1.0
            } else {
                0.0
            }
        },
    );
    assert_shadow_follows_the_ray_geometry(&t, res);
}

/// The opaque block of the shadow-geometry tests, in physics-grid cells.
const SHADOW_TEST_BLOCK: (f32, f32, f32, f32) = (12.0, 16.0, 20.0, 24.0);

fn in_shadow_test_block(cx: f32, cy: f32) -> bool {
    let (x0, x1, y0, y1) = SHADOW_TEST_BLOCK;
    cx > x0 && cx < x1 && cy > y0 && cy < y1
}

/// Judges a `res` transmittance field lit toward (1, 1) past
/// `SHADOW_TEST_BLOCK` (see `light_pass_shadow_edge_follows_the_ray_geometry`).
fn assert_shadow_follows_the_ray_geometry(t: &[[f32; 3]], res: u32) {
    let (x0, x1, y0, y1) = SHADOW_TEST_BLOCK;
    // Slab test of the ray from the cell centre along (1, 1)/sqrt(2) against
    // the box grown by `margin` cells.
    let hits = |x: u32, y: u32, margin: f32| {
        let (ox, oy) = (x as f32 + 0.5, y as f32 + 0.5);
        let (lo_x, hi_x) = (x0 - margin, x1 + margin);
        let (lo_y, hi_y) = (y0 - margin, y1 + margin);
        let t_enter = (lo_x - ox).max(lo_y - oy);
        let t_exit = (hi_x - ox).min(hi_y - oy);
        t_exit > t_enter.max(0.0)
    };
    let (mut dark, mut lit) = (0, 0);
    for y in 0..res {
        for x in 0..res {
            if in_shadow_test_block(x as f32 + 0.5, y as f32 + 0.5) {
                continue;
            }
            let got = t[(y * res + x) as usize][0];
            if hits(x, y, -1.0) {
                assert!(got < 0.01, "cell ({x},{y}) behind the block got {got}");
                dark += 1;
            } else if !hits(x, y, 1.0) {
                assert!(got > 0.99, "cell ({x},{y}) clear of the block got {got}");
                lit += 1;
            }
        }
    }
    assert!(
        dark > 20 && lit > 500,
        "too few judged cells: {dark} dark, {lit} lit"
    );
}

/// `render_surface_reconstruction` runs the light pass from its own
/// reconstructed density, with no grid-volume render before it: an opaque
/// disk lit from straight above leaves the cells under it dark and the
/// cells above it, or beside it, fully lit.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_reconstruction_shadows_with_its_own_light_pass() {
    use crate::gpu::GpuSimulation;
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};
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
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
    // 50 per metre over 0.1 m cells: an optical depth of 5 per cell.
    r.set_optical_params(&queue, 0, [50.0; 3]);
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.1,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.25; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_light_pass_target"),
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
    let cells = (grid_res * grid_res) as usize;
    let t = readback_f32_blocking(&device, &queue, &r.light_transmittance_buf, cells * 4);
    let at = |x: u32, y: u32| t[(y * grid_res + x) as usize * 4];
    // The disk spans y = 10..22 along x = 16.
    assert!(at(16, 6) < 0.01, "under the disk: {}", at(16, 6));
    assert!(at(16, 26) > 0.99, "above the disk: {}", at(16, 26));
    assert!(at(3, 6) > 0.99, "beside the disk: {}", at(3, 6));
}

/// Runs the light pass from a curvature-flow surface density: `density(i, j)`
/// on a `surface_res` square covering the same domain as the `res` physics
/// grid (reference cell mass 1), with per-slot masses `slot_mass(i, j, s)`
/// when given (slot 0 throughout otherwise), slot `s` absorbing
/// `sigma_a[s]` and not scattering, under a contract of cell size `dx` and
/// light toward `light`. Returns the per-cell RGB transmittance on the
/// physics grid, row-major (y * res + x).
fn surface_light_pass_transmittance(
    res: u32,
    surface_res: u32,
    dx: f32,
    sigma_a: &[[f32; 3]],
    light: glam::Vec3,
    density: impl Fn(u32, u32) -> f32,
    slot_mass: Option<&dyn Fn(u32, u32, usize) -> i32>,
) -> Vec<[f32; 3]> {
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};
    const SLOTS: usize = 16;
    let (device, queue) = headless_device();
    let surface_cells = (surface_res * surface_res) as usize;
    let mut field = vec![0.0f32; surface_cells];
    let mut masses = vec![0i32; surface_cells * SLOTS];
    for j in 0..surface_res {
        for i in 0..surface_res {
            let idx = (j * surface_res + i) as usize;
            field[idx] = density(i, j);
            if let Some(slot_mass) = slot_mass {
                for s in 0..SLOTS {
                    masses[idx * SLOTS + s] = slot_mass(i, j, s);
                }
            }
        }
    }
    let storage = |label: &str, bytes: &[u8]| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        buf
    };
    let density_buf = storage("surface_light_pass_density", bytemuck::cast_slice(&field));
    let material_mass_buf = storage(
        "surface_light_pass_material_mass",
        bytemuck::cast_slice(&masses),
    );
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    for (slot, sigma) in sigma_a.iter().enumerate() {
        r.set_optical_params(&queue, slot, *sigma);
    }
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: dx,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.25; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: light,
        })
        .unwrap(),
    );
    r.ensure_light_pass_capacity(&device, res);
    r.ensure_light_surface_capacity(&device, surface_res);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("surface_light_pass"),
    });
    r.encode_light_pass(
        &device,
        &queue,
        &mut enc,
        res,
        LightPassSource::Surface {
            density: &density_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: slot_mass.is_some(),
            surface_res,
            material_slot: 0,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let cells = (res * res) as usize;
    let raw = readback_f32_blocking(&device, &queue, &r.light_transmittance_buf, cells * 4);
    raw.chunks(4).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Asserts a transmittance field lit from straight above is Beer-Lambert's
/// `exp(-sigma * dx * n)` through `n` cells of extinction `sigma` per metre.
fn assert_beer_lambert_from_above(t: &[[f32; 3]], res: u32, dx: f32, sigma: [f32; 3], case: &str) {
    for y in 0..res {
        let cells_above = (res - 1 - y) as f32;
        for x in [0u32, 7, res - 1] {
            let got = t[(y * res + x) as usize];
            for ch in 0..3 {
                let expected = (-sigma[ch] * dx * cells_above).exp();
                assert!(
                    (got[ch] - expected).abs() < 1.0e-3,
                    "{case}: cell ({x},{y}) channel {ch}: {} against exp(-{} * {dx} * {cells_above}) = {expected}",
                    got[ch],
                    sigma[ch]
                );
            }
        }
    }
}

/// The surface path marches the same physics grid, from the mean extinction
/// of the surface cells inside each grid cell. A surface whose columns
/// alternate empty and twice the reference density averages to the
/// reference density, so the light through it is Beer-Lambert's through a
/// uniform medium; so is a uniform surface at a resolution that is not a
/// whole multiple of the grid's, whose grid cells hold two or three surface
/// cells in turn.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_light_pass_matches_beer_lambert_through_the_averaged_density() {
    let (res, dx, sigma) = (24u32, 0.1f32, [0.5f32, 1.0, 2.0]);
    let alternating = surface_light_pass_transmittance(
        res,
        6 * res,
        dx,
        &[sigma],
        glam::Vec3::Y,
        |i, _| if i % 2 == 0 { 0.0 } else { 2.0 },
        None,
    );
    assert_beer_lambert_from_above(&alternating, res, dx, sigma, "alternating columns");
    let uneven = surface_light_pass_transmittance(
        res,
        res * 5 / 2,
        dx,
        &[sigma],
        glam::Vec3::Y,
        |_, _| 1.0,
        None,
    );
    assert_beer_lambert_from_above(&uneven, res, dx, sigma, "surface 2.5 times finer");
}

/// The surface's materials blend by mass fraction within each surface cell,
/// then average over the grid cell: even columns hold slot 0 alone, odd
/// columns slot 0 and slot 1 at 1 : 3, so every grid cell attenuates as
/// `0.5 sigma_0 + 0.5 (0.25 sigma_0 + 0.75 sigma_1)`.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_light_pass_blends_slots_by_mass_fraction() {
    let (res, dx) = (16u32, 0.1f32);
    let (sigma_0, sigma_1) = ([0.4f32, 0.8, 1.6], [2.0f32, 1.0, 0.0]);
    let slot_mass = |i: u32, _: u32, s: usize| match (i % 2, s) {
        (_, 0) => 1000,
        (1, 1) => 3000,
        _ => 0,
    };
    let t = surface_light_pass_transmittance(
        res,
        6 * res,
        dx,
        &[sigma_0, sigma_1],
        glam::Vec3::Y,
        |_, _| 1.0,
        Some(&slot_mass),
    );
    let blended: [f32; 3] = std::array::from_fn(|ch| {
        0.5 * sigma_0[ch] + 0.5 * (0.25 * sigma_0[ch] + 0.75 * sigma_1[ch])
    });
    assert_beer_lambert_from_above(&t, res, dx, blended, "mass-fraction blend");
}

/// Gate (b) on the surface path: a block of surface cells filling whole
/// grid cells shadows exactly as the same block on the grid does.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_light_pass_shadow_edge_follows_the_ray_geometry() {
    let (res, multiplier) = (32u32, 6u32);
    let t = surface_light_pass_transmittance(
        res,
        multiplier * res,
        0.1,
        &[[100.0; 3]],
        glam::Vec3::new(1.0, 1.0, 0.0),
        |i, j| {
            let centre = |k: u32| (k as f32 + 0.5) / multiplier as f32;
            if in_shadow_test_block(centre(i), centre(j)) {
                1.0
            } else {
                0.0
            }
        },
        None,
    );
    assert_shadow_follows_the_ray_geometry(&t, res);
}
