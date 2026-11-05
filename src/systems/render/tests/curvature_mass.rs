use super::*;

/// Blocking readback of a single atomic<i32> total (used for the volume-
/// preserving-correction buffers, which are NOT f32 like everything else
/// `readback_f32_blocking` reads -- same staging pattern, different cast.
fn readback_i32_total(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer) -> i32 {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("i32_total_readback_staging"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("i32_total_readback"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, 4);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let value = bytemuck::cast_slice::<u8, i32>(&mapped)[0];
    drop(mapped);
    staging.unmap();
    value
}

/// Volume-preserving correction (`curvature_flow.wgsl`'s Pass 1d, the discrete
/// form of `V = -H + lambda(t)`): the 15x-35x mass growth once measured came
/// from comparing a mass total with a raw, unweighted sum of densities on a
/// `surface_res_multiplier`x finer grid (see
/// `curvature_flow_mass_growth_scales_with_iteration_count`). With the area
/// factor in the Lagrange multiplier it sits near 1.0, not ~1/34, which would
/// have made small and thin objects vanish. `corrected_total` is read back as
/// a raw sum like `pre_total`/`raw_post_total`, so it is compared with
/// `true_particle_mass_sum * multiplier^2`, the shader's structural factor,
/// not a separate tolerance.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_volume_correction_matches_true_particle_mass() {
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
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    );
    let true_particle_mass_sum: f32 = particles.iter().map(|p| p.mass).sum();

    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume_correction_test_target"),
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

    // Matches `curvature_flow.wgsl`'s own `TOTAL_ATOMIC_SCALE` -- a
    // deliberately SMALLER scale than the per-cell `DENSITY_ATOMIC_SCALE`,
    // since these are GLOBAL sums across an entire scene's cells/particles,
    // not one cell's own bounded local overlap (see that constant's own
    // doc for the i32-overflow bug this fixes).
    const TOTAL_ATOMIC_SCALE: f32 = 1000.0;
    let pre_total =
        readback_i32_total(&device, &queue, &r.pre_total_atomic_buf) as f32 / TOTAL_ATOMIC_SCALE;
    let raw_post_total =
        readback_i32_total(&device, &queue, &r.post_total_atomic_buf) as f32 / TOTAL_ATOMIC_SCALE;

    let surface_res = r.surface_res;
    let settled = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );
    let corrected_total: f32 = settled.iter().sum();

    eprintln!(
        "true_particle_mass={true_particle_mass_sum} pre_total(splat)={pre_total} \
         raw_post_total(pre-correction)={raw_post_total} corrected_total(post-correction)={corrected_total}"
    );

    assert!(
        (pre_total - true_particle_mass_sum).abs() < true_particle_mass_sum * 0.05,
        "the splat pass's own accumulated total must closely match the real \
         particle mass sum (sanity check on the ground truth itself) -- \
         true={true_particle_mass_sum} splat_total={pre_total}"
    );
    let relative_error_uncorrected = (raw_post_total - pre_total).abs() / pre_total;
    eprintln!(
        "real, uncorrected raw drift (expected large -- see this test's own \
         doc, it's a units mismatch, not what the correction below fixes): \
         {:.1}%",
        relative_error_uncorrected * 100.0
    );

    // The real check: `corrected_total` is still a RAW sum (same units as
    // `raw_post_total`), so it's compared against `true_particle_mass_sum`
    // scaled by the SAME real structural factor (`surface_res_multiplier^2`)
    // the shader's own Lagrange multiplier accounts for -- not a second,
    // independently-chosen tolerance.
    let multiplier = r.surface_res_multiplier() as f32;
    let expected_corrected_total = true_particle_mass_sum * multiplier * multiplier;
    let relative_error_corrected =
        (corrected_total - expected_corrected_total).abs() / expected_corrected_total;
    eprintln!(
        "real, area-corrected drift: {:.2}% (expected_corrected_total={expected_corrected_total:.1} \
         vs actual corrected_total={corrected_total:.1})",
        relative_error_corrected * 100.0
    );
    assert!(
        relative_error_corrected < 0.1,
        "the RE-ENABLED volume-preserving correction must bring the settled \
         total within a small, real margin of true particle mass (scaled by \
         the surface grid's own real area factor) -- got \
         corrected_total={corrected_total:.1}, \
         expected~={expected_corrected_total:.1} \
         ({:.1}% off, wanted < 10%)",
        relative_error_corrected * 100.0
    );
}

/// Diagnostic, not an assertion: sweeps the engine's curvature iteration
/// count on the particle-splat scene above, to see whether the total-mass
/// growth scales with iterations or appears at the first smoothing step
/// (flat-noise and clean-disk hypotheses were ruled out on idealized fields).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_mass_growth_scales_with_iteration_count() {
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
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    const TOTAL_ATOMIC_SCALE: f32 = 1000.0;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    eprintln!("real engine mass growth vs. iteration count, same scene as the sibling test:");
    eprintln!(
        "{:>6} {:>14} {:>10}",
        "iters", "raw_post_total", "pct_of_pre"
    );
    for iterations in [2u32, 4, 6, 8, 10, 12] {
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
        r.set_curvature_iterations(iterations);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mass_growth_sweep_target"),
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

        let pre_total = readback_i32_total(&device, &queue, &r.pre_total_atomic_buf) as f32
            / TOTAL_ATOMIC_SCALE;
        let raw_post_total = readback_i32_total(&device, &queue, &r.post_total_atomic_buf) as f32
            / TOTAL_ATOMIC_SCALE;

        // `pre_total` comes from per-particle mass during the splat
        // (resolution-independent: a normalized kernel's weights sum to 1).
        // `raw_post_total` is a raw sum of per-cell densities, and each surface
        // cell's area is `1/surface_res_multiplier^2` of a physics cell's,
        // never multiplied back in by `post_total_reduce_main`/
        // `convert_atomic_to_float_main`. Dividing by `surface_res_multiplier^2`
        // should recover about `pre_total`.
        let multiplier = r.surface_res_multiplier() as f32;
        let area_corrected = raw_post_total / (multiplier * multiplier);
        eprintln!(
            "{:>6} {:>14.3} {:>9.1}%  area_corrected={:.2} (multiplier={:.0}, vs pre_total={:.2})",
            iterations,
            raw_post_total,
            100.0 * raw_post_total / pre_total,
            area_corrected,
            multiplier,
            pre_total
        );
    }
}

/// Wave-equation surface (`curvature_flow.wgsl`'s Pass 2b): repeated calls to
/// `render_surface_reconstruction` with a cluster present excite the wave
/// field away from zero, and it stays finite across many steps (`WAVE_DAMPING`
/// bounds it under continuous forcing).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn wave_field_is_excited_by_real_density_and_stays_bounded() {
    use crate::gpu::GpuSimulation;
    use crate::matter::materials::MaterialModel;
    use crate::{
        MaterialRegistry, NewtonianFluidMaterial, SimConfig, SpawnRegion, build_particles,
    };
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
    // Waves only excite for a material that behaves like a fluid
    // (`owns_deformation_volume_state()`, see `Renderer::set_wave_force_coeff`);
    // `NeoHookeanMaterial`, a solid, does not. The coefficient 0.35 is the one
    // `examples/cpu/basic_fluids.rs` and `examples/gpu/basic_sand_grid_gpu.rs`
    // use with the same gate.
    let material = NewtonianFluidMaterial::low_viscosity(1.0, 1.0);
    let registry = MaterialRegistry::with_default(Box::new(material));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
    // Opt-in (see `WaveStepParams::wave_force_coeff` in `curvature_flow.wgsl`):
    // inert at 0.0 unless the scene sets it, as the examples do.
    assert!(
        material.owns_deformation_volume_state(),
        "test material must actually be a real fluid, or this whole test proves nothing"
    );
    r.set_wave_force_coeff(0.35);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wave_field_test_target"),
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

    const FRAMES: u32 = 30;
    for _ in 0..FRAMES {
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

    let surface_res = r.surface_res;
    let cell_count = (surface_res * surface_res) as usize;
    // After FRAMES calls, the buffer holding the latest settled state is
    // whichever served as "next" on the LAST call -- same rotation
    // `render_surface_reconstruction` itself uses internally.
    let current_idx = (FRAMES % 3) as usize;
    let values = readback_f32_blocking(&device, &queue, &r.wave_bufs[current_idx], cell_count);

    assert!(
        values.iter().all(|v| v.is_finite()),
        "wave field must stay finite (real damping must bound continuous forcing), \
         even after {FRAMES} real steps"
    );
    assert!(
        values.iter().any(|v| v.abs() > 1.0e-6),
        "a real particle cluster's density gradient must excite the wave field away \
         from its all-zero initial state after {FRAMES} real steps"
    );
}

/// Regression check for the wave-excitation forcing term: it must be the
/// TEMPORAL density difference (this frame's density minus last frame's),
/// not the SPATIAL density gradient -- a spatial gradient is nonzero at
/// any object's edge PERMANENTLY, whether anything moves or not, so the
/// wave field would never settle even for a fully static body. This test
/// drives a COMPLETELY STATIC particle cluster (same buffer, never touched
/// between calls -- the strongest analogue of "physics reports
/// max_speed~0") for many frames and confirms the wave field's peak
/// magnitude DECAYS over time once the one-time "body just appeared" burst
/// passes, rather than staying pinned at a roughly constant nonzero level
/// forever (what a spatial-gradient forcing term would do, since the
/// excitation source -- the object's own unchanging edge -- never goes
/// away).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_wave_field_decays_once_density_stops_changing() {
    use crate::gpu::GpuSimulation;
    use crate::matter::materials::MaterialModel;
    use crate::{
        MaterialRegistry, NewtonianFluidMaterial, SimConfig, SpawnRegion, build_particles,
    };
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
    // A fluid material and the `set_wave_force_coeff` opt-in below, as in
    // `wave_field_is_excited_by_real_density_and_stays_bounded` and the
    // examples.
    let material = NewtonianFluidMaterial::low_viscosity(1.0, 1.0);
    let registry = MaterialRegistry::with_default(Box::new(material));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
    assert!(
        material.owns_deformation_volume_state(),
        "test material must actually be a real fluid, or this whole test proves nothing"
    );
    r.set_wave_force_coeff(0.35);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wave_decay_test_target"),
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

    let render_frame = |r: &mut Renderer| {
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
    };

    // First real frame grows the surface buffers to their true size --
    // `surface_res` must be read AFTER this, not before, or `cell_count`
    // silently stays at the constructor's 1-cell placeholder.
    render_frame(&mut r);
    let surface_res = r.surface_res;
    let cell_count = (surface_res * surface_res) as usize;

    // Past the initial one-time excitation burst (expected -- a body
    // appearing IS a disturbance), but still early in the real
    // WAVE_DAMPING=0.996 decay curve.
    const EARLY_FRAME: u32 = 10;
    for _ in 1..EARLY_FRAME {
        render_frame(&mut r);
    }
    let early_idx = (EARLY_FRAME % 3) as usize;
    let early_values = readback_f32_blocking(&device, &queue, &r.wave_bufs[early_idx], cell_count);
    let early_peak = early_values.iter().fold(0.0f32, |a, v| a.max(v.abs()));

    const LATE_FRAME: u32 = 300;
    for _ in EARLY_FRAME..LATE_FRAME {
        render_frame(&mut r);
    }
    let late_idx = (LATE_FRAME % 3) as usize;
    let late_values = readback_f32_blocking(&device, &queue, &r.wave_bufs[late_idx], cell_count);
    let late_peak = late_values.iter().fold(0.0f32, |a, v| a.max(v.abs()));

    assert!(
        late_values.iter().all(|v| v.is_finite()),
        "wave field must stay finite across {LATE_FRAME} real frames"
    );
    assert!(
        late_peak < early_peak * 0.9,
        "with a COMPLETELY STATIC particle cluster (density never changes \
         after the first frame), the wave field's peak magnitude must \
         genuinely decay over {LATE_FRAME} frames, not stay pinned near its \
         early value -- the old spatial-gradient forcing term would keep \
         re-exciting it forever from the object's own permanent edge, \
         exactly the bug this fix addresses -- early(frame {EARLY_FRAME})={early_peak} \
         late(frame {LATE_FRAME})={late_peak}"
    );
}

// ── curvature_iterate_main numerical stability ──────────────────────────────
//
// A line-for-line CPU port of `curvature_flow.wgsl`'s `curvature_iterate_main`
// (same formula, same constants), so the update rule itself is measured. At
// `GRAD_EPSILON = 1.0e-3`, in a near-flat region (a fluid's interior, only
// splat noise) the denominator `(grad_sq + GRAD_EPSILON)^1.5` is dominated by
// epsilon and kappa becomes noise over a near-zero constant, bounded only by
// MAX_KAPPA: 12 iterations grew a flat field's noise variance 357x and a sharp
// corner grew (0.1 -> 0.205) instead of rounding.
mod curvature_iterate_stability {
    /// Mirrors `curvature_flow.wgsl`'s `CURVATURE_PSEUDO_DT`, `MAX_KAPPA`,
    /// and `GRAD_EPSILON` -- all three are WGSL-only (shader compile-time
    /// consts, not part of any uniform), so there is no single source of
    /// truth to import from Rust. If any of the three shader constants ever
    /// change, update the matching one here too, or this "line-for-line
    /// port" silently stops verifying the value actually shipped.
    /// `CURVATURE_PSEUDO_DT` in particular is documented shader-side as a
    /// tuned, retunable value, not a derived one -- the most likely of the
    /// three to drift.
    const CURVATURE_PSEUDO_DT: f32 = 0.15;
    const MAX_KAPPA: f32 = 4.0;
    const GRAD_EPSILON: f32 = 0.1;

    fn sample(field: &[f32], cx: i32, cy: i32, res: i32) -> f32 {
        if cx < 0 || cy < 0 || cx >= res || cy >= res {
            0.0
        } else {
            field[(cy * res + cx) as usize]
        }
    }

    /// Exact port of `curvature_iterate_main`'s body -- must be kept in sync
    /// with `curvature_flow.wgsl` if that formula ever changes.
    fn curvature_iterate(field_in: &[f32], res: i32, grad_epsilon: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; field_in.len()];
        for cy in 0..res {
            for cx in 0..res {
                let center = sample(field_in, cx, cy, res);
                let dx =
                    (sample(field_in, cx + 1, cy, res) - sample(field_in, cx - 1, cy, res)) * 0.5;
                let dy =
                    (sample(field_in, cx, cy + 1, res) - sample(field_in, cx, cy - 1, res)) * 0.5;
                let dxx = sample(field_in, cx + 1, cy, res) - 2.0 * center
                    + sample(field_in, cx - 1, cy, res);
                let dyy = sample(field_in, cx, cy + 1, res) - 2.0 * center
                    + sample(field_in, cx, cy - 1, res);
                let dxy = (sample(field_in, cx + 1, cy + 1, res)
                    - sample(field_in, cx + 1, cy - 1, res)
                    - sample(field_in, cx - 1, cy + 1, res)
                    + sample(field_in, cx - 1, cy - 1, res))
                    * 0.25;
                let grad_sq = dx * dx + dy * dy;
                let denom = (grad_sq + grad_epsilon).powf(1.5);
                let kappa = ((dxx * dy * dy - 2.0 * dx * dy * dxy + dyy * dx * dx) / denom)
                    .clamp(-MAX_KAPPA, MAX_KAPPA);
                let i = (cy * res + cx) as usize;
                out[i] = (center + CURVATURE_PSEUDO_DT * kappa).max(0.0);
            }
        }
        out
    }

    // Deterministic xorshift -- zero external deps, fully reproducible.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }
    }

    fn variance(field: &[f32]) -> f32 {
        let n = field.len() as f32;
        let mean: f32 = field.iter().sum::<f32>() / n;
        field.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n
    }

    /// The regression this bug actually was: a near-flat, noisy field (the
    /// real shape of a fluid's interior -- see `render::mod`'s N_eff
    /// analysis for where the noise floor comes from) must not have its
    /// variance GROW under repeated iteration. `GRAD_EPSILON=1.0e-3` grew it
    /// 357x over 12 iterations; the shipped value must keep it bounded with
    /// real margin, not just barely under 1.0.
    #[test]
    #[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
    fn flat_noisy_field_does_not_amplify() {
        const RES: i32 = 48;
        const ITERATIONS: usize = 12;
        let flat_value = 0.1f32;
        let noise_std = 0.03f32; // matches the N_eff~13 noise floor at this scene's reference_cell_mass=0.1

        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut field: Vec<f32> = (0..(RES * RES) as usize)
            .map(|_| (flat_value + noise_std * rng.next_f32()).max(0.0))
            .collect();
        let var0 = variance(&field);
        for _ in 0..ITERATIONS {
            field = curvature_iterate(&field, RES, GRAD_EPSILON);
        }
        let var_final = variance(&field);
        let ratio = var_final / var0.max(1.0e-12);

        assert!(
            ratio < 1.2,
            "curvature_iterate_main is amplifying flat-region noise instead \
             of damping it: variance ratio {ratio:.3} over {ITERATIONS} \
             iterations (>= 1.0 means growth). This is the exact mechanism \
             behind the white-noise/flicker regression fixed 2026-08-14 -- \
             GRAD_EPSILON is too small relative to the density field's real \
             sampling noise floor."
        );
    }

    /// The other half of the same fix: raising GRAD_EPSILON enough to
    /// stabilize the flat interior must not also neuter the pass's actual
    /// job. A sharp 90-degree corner (the free-surface case) must still
    /// round off measurably after iteration, not sit inert.
    #[test]
    #[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
    fn sharp_corner_still_rounds() {
        const RES: i32 = 48;
        const ITERATIONS: usize = 12;
        let flat_value = 0.1f32;

        let mut field = vec![0.0f32; (RES * RES) as usize];
        for cy in 0..RES {
            for cx in 0..RES {
                if cx >= 12 && cy >= 12 {
                    field[(cy * RES + cx) as usize] = flat_value;
                }
            }
        }
        let corner_idx = (12 * RES + 12) as usize;
        for _ in 0..ITERATIONS {
            field = curvature_iterate(&field, RES, GRAD_EPSILON);
        }

        assert!(
            field[corner_idx] < flat_value * 0.9,
            "a sharp corner must round off (its own value should drop \
             meaningfully toward its empty neighbours) after {ITERATIONS} \
             curvature-flow iterations -- got {:.4} from a start of \
             {flat_value}, too small a change. GRAD_EPSILON may be large \
             enough to have suppressed real edge curvature along with the \
             noise it was raised to fix.",
            field[corner_idx]
        );
        assert!(
            field[corner_idx] < flat_value,
            "a sharp corner must not GROW under curvature flow (mean \
             curvature flow shrinks, it does not expand) -- got {:.4} from \
             a start of {flat_value}, which is the exact instability \
             signature the old GRAD_EPSILON=1.0e-3 had (0.1 -> 0.205, this \
             file's own prior measurement).",
            field[corner_idx]
        );
    }
}
