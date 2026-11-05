//! Test suite for `Renderer` -- split out of `mod.rs` (was ~150 of its ~930
//! lines), same pattern as `gpu/solver/device_lost_tests.rs`.

use super::gpu_types::GridPeakParams;
use super::*;
use crate::particle::Particle;
use glam::Mat2;

fn headless_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = crate::systems::gpu::create_wgpu_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no GPU adapter available for render test");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("failed to create device")
}

/// The disk of particles the pixel tests above draw: radius 6 cells at the
/// centre of a 32-cell grid, 0.5 cells apart, material 0, moved by `shift`
/// cells along both axes.
fn pixel_test_disk(shift: f32) -> Vec<crate::Particle> {
    use crate::{SimConfig, SpawnRegion, build_particles};
    let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
    build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0 + shift))
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    )
}

/// A square render target for the pixel tests, with its view.
fn pixel_test_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pixel_test_target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Blocking single-pixel RGBA8 texture readback -- test-only. Same
/// staging-buffer/copy/poll/map_async/poll/read/unmap pattern as
/// `readback_f32_blocking` below, but for a render-target texture instead
/// of a storage buffer, so it must additionally respect wgpu's
/// `COPY_BYTES_PER_ROW_ALIGNMENT` (256-byte) padding requirement on the
/// destination buffer's row stride.
fn readback_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> [u8; 4] {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel_readback_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pixel_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let row_start = (y * padded_bytes_per_row) as usize;
    let px_start = row_start + (x * 4) as usize;
    let pixel = [
        mapped[px_start],
        mapped[px_start + 1],
        mapped[px_start + 2],
        mapped[px_start + 3],
    ];
    drop(mapped);
    staging.unmap();
    pixel
}

/// Blocking readback of the single brightest pixel in the target -- test-only,
/// same staging pattern as `readback_pixel` above.
///
/// For a test that renders emitters on a dark background, this is the right
/// sample: a partially-covered pixel is blended toward the background and so
/// is always dimmer than a fully-covered one, which makes the maximum the
/// pixel that carries the emitter's own colour. Sampling a fixed coordinate
/// instead would depend on exactly where the particle lattice happens to
/// land relative to the pixel grid.
fn readback_brightest_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> [u8; 4] {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brightest_pixel_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("brightest_pixel_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let mut best = ([0u8; 4], -1i32);
    for y in 0..height {
        let row = (y * padded_bytes_per_row) as usize;
        for x in 0..width {
            let at = row + (x * 4) as usize;
            let pixel = [mapped[at], mapped[at + 1], mapped[at + 2], mapped[at + 3]];
            let sum = i32::from(pixel[0]) + i32::from(pixel[1]) + i32::from(pixel[2]);
            if sum > best.1 {
                best = (pixel, sum);
            }
        }
    }
    drop(mapped);
    staging.unmap();
    best.0
}

/// Blocking readback of a SAMPLED GRID of pixel luminances (every
/// `sample_stride`-th pixel in both x/y, row-major) -- test-only, same
/// staging pattern as `readback_pixel` above but reads the whole texture
/// once instead of one pixel, for a per-frame flicker measurement
/// across many frames (see `surface_reconstruction_does_not_flicker_over_
/// many_deterministic_frames` below).
fn readback_luminance_grid(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    sample_stride: u32,
) -> Vec<f64> {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flicker_grid_readback_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flicker_grid_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let mut out = Vec::new();
    let mut y = 0u32;
    while y < height {
        let mut x = 0u32;
        while x < width {
            let off = (y * padded_bytes_per_row + x * 4) as usize;
            let b = mapped[off] as f64;
            let g = mapped[off + 1] as f64;
            let r = mapped[off + 2] as f64;
            out.push(0.299 * r + 0.587 * g + 0.114 * b);
            x += sample_stride;
        }
        y += sample_stride;
    }
    drop(mapped);
    staging.unmap();
    out
}

/// Blocking f32 storage-buffer readback -- test-only, mirrors the real
/// established pattern `gpu::buffers::readback::readback_f32_blocking`
/// already uses (staging buffer, copy, poll, map_async, poll, read,
/// unmap), just inlined here since `surface_a_buf` is a `Renderer`-owned
/// buffer, not a `GpuBuffers` one.
fn readback_f32_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    count: usize,
) -> Vec<f32> {
    let byte_count = (count * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface_readback_staging"),
        size: byte_count,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("surface_readback"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_count);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..byte_count);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    values
}

mod curvature;
mod curvature_mass;
mod grid_edges;
mod grid_volume;
mod light_pass;
mod optics;
mod surface;
mod view;
