/// emerge particle renderer -- physics-driven, no assets.
///
/// # Two rendering paths
///
/// **CPU path** (`render_slice`):
///   Builds `InstanceData` per particle on CPU, uploads via `write_buffer`.
///
/// **GPU path** (`render_gpu`):
///   Runs `prep_instances.wgsl` compute to fill instance buffer directly from
///   the particle storage buffer -- zero CPU readback, zero stall.
///   Pass `sim.particle_buffer()` + `sim.particle_count()`. No `sync_particles_blocking()`.
use std::mem;

use wgpu::util::DeviceExt;

use crate::particle::{Particle, Particles};

const RENDER_SHADER: &str = include_str!("shaders/render_particles.wgsl");
const PREP_SHADER: &str = include_str!("shaders/prep_instances.wgsl");
const GRID_VOLUME_SHADER: &str = include_str!("shaders/grid_volume.wgsl");
const PREP_WG: u32 = 64;

// ── Color mode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    ByMaterial = 0,
    ByVelocity = 1,
    ByVolume = 2,
    ByPhysics = 3,
    ByThermal = 4,
    ByActivation = 5,
    /// Generic second scalar carrier (resource/grass level, pheromone, nutrients).
    /// See `Particle::scalar_field`'s own doc. Distinct wire value (6, not the next
    /// unused slot after ByActivation's implicit WGSL else-branch) so the GPU shader's
    /// existing fallback `else` can keep meaning ByActivation without renumbering it.
    ByScalarField = 6,
}

// GPU-side wire structs (InstanceData/CameraParams/RenderConfig/OpticalTable,
// GridVolumeParams/GridVolumeSource) live in gpu_types.rs -- see that file's doc.
mod gpu_types;
pub use gpu_types::GridVolumeSource;
use gpu_types::{CameraParams, GridVolumeParams, InstanceData, OpticalTable, RenderConfig};

// wgpu pipeline construction (the three build_*_pipeline functions + their
// bind-group-layout helpers) lives in pipelines.rs -- see that file's doc.
mod pipelines;
use pipelines::{build_grid_volume_pipeline, build_particle_pipeline, build_prep_pipeline};

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer, // VERTEX | COPY_DST — drawn as per-instance attributes
    storage_instances: wgpu::Buffer, // STORAGE | COPY_SRC — compute write target (GPU path)
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    max_particles: usize,

    prep_pipeline: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    render_config_buf: wgpu::Buffer,
    optical_table_buf: wgpu::Buffer,

    grid_volume_pipeline: wgpu::RenderPipeline,
    grid_volume_bgl: wgpu::BindGroupLayout,
    grid_volume_params_buf: wgpu::Buffer,
    /// Cached ortho projection + grid_res (set by `set_camera`) -- lets
    /// `render_grid_volume` take just (device, queue, grid_buf, material_mass_buf,
    /// view, clear) instead of repeating width/height/grid_res, keeping it under
    /// clippy's argument-count lint.
    cached_ortho: (f32, f32, f32, f32),
    cached_grid_res: u32,

    scratch: Vec<InstanceData>,
    color_mode: ColorMode,
    vel_scale: f32,
    sigma_a: [[f32; 3]; 16],
    /// Reduced scattering coefficient per material slot (single scalar -- see
    /// `OpticalTable`'s own doc for why this isn't per-channel).
    sigma_s: [f32; 16],
    /// Specular Fresnel base reflectance R0 per material slot (see `OpticalTable`'s
    /// own doc for the real-but-bounded caveat).
    specular_r0: [f32; 16],
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        max_particles: usize,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let cap = max_particles.max(1);

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            // VERTEX for draw; COPY_DST for both the CPU fill path and the GPU compute copy.
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GPU compute write target. Kept distinct from the vertex buffer: wgpu treats a
        // read_write storage buffer as an exclusive usage, so sharing one buffer for both
        // compute-write and vertex-read trips its usage tracker. Copied into instance_buffer.
        let storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances_storage"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_verts"),
            contents: bytemuck::cast_slice::<[f32; 2], u8>(&[
                [-0.5f32, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_idx"),
            contents: bytemuck::cast_slice::<u16, u8>(&[0u16, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_camera"),
            size: mem::size_of::<CameraParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_config"),
            size: mem::size_of::<RenderConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let optical_table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_optics"),
            size: mem::size_of::<OpticalTable>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_volume_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_volume_params"),
            size: mem::size_of::<GridVolumeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (render_pipeline, render_bgl) = build_particle_pipeline(device, output_format);
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let (prep_pipeline, prep_bgl) = build_prep_pipeline(device);
        let (grid_volume_pipeline, grid_volume_bgl) =
            build_grid_volume_pipeline(device, output_format);

        Self {
            render_pipeline,
            render_bind_group,
            instance_buffer,
            storage_instances,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            max_particles: cap,
            prep_pipeline,
            prep_bgl,
            render_config_buf,
            grid_volume_pipeline,
            grid_volume_bgl,
            grid_volume_params_buf,
            cached_ortho: (1.0, 0.0, 1.0, 0.0),
            cached_grid_res: 1,
            optical_table_buf,
            scratch: Vec::with_capacity(cap),
            color_mode: ColorMode::ByMaterial,
            vel_scale: 0.05,
            sigma_a: [[0.3f32; 3]; 16],
            sigma_s: [0.0f32; 16],
            specular_r0: [0.0f32; 16],
        }
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Call at init and on every resize.
    pub fn set_camera(
        &mut self,
        queue: &wgpu::Queue,
        grid_res: u32,
        width: u32,
        height: u32,
        particle_scale: f32,
        round_particles: bool,
    ) {
        let gr = grid_res as f32;
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let (sx, tx, sy, ty) = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        self.cached_ortho = (sx, tx, sy, ty);
        self.cached_grid_res = grid_res;
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraParams {
                view_proj: [
                    sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
                ],
                particle_scale,
                round_particles: round_particles as u32,
                _pad: [0.0; 2],
            }),
        );
    }

    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }
    pub fn set_vel_scale(&mut self, s: f32) {
        self.vel_scale = s;
    }

    /// Sets the Beer-Lambert absorption coefficient for `slot` AND uploads it to
    /// the GPU immediately, not just CPU-side state -- `render()`'s per-particle
    /// `particle_color()` path reads CPU state directly, but `render_grid_volume`'s
    /// GPU shader reads a GPU-resident buffer that would otherwise silently keep
    /// whatever was uploaded last, ignoring every material's real color in
    /// grid-volume mode. The redundant-write cost when setting several slots in a
    /// row is negligible (scene-setup-time only, never a per-frame path).
    pub fn set_optical_params(&mut self, queue: &wgpu::Queue, slot: usize, sigma_a: [f32; 3]) {
        self.sigma_a[slot % 16] = sigma_a;
        self.upload_optical_params(queue);
    }

    /// Reduced scattering coefficient for `slot` -- see `OpticalTable`'s doc for
    /// what this represents physically (real subsurface scattering, single-
    /// scattering approximation) and its real citation (Jacques 2013). Auto-
    /// uploads immediately -- see `set_optical_params`'s own doc for why.
    pub fn set_optical_scattering(&mut self, queue: &wgpu::Queue, slot: usize, sigma_s: f32) {
        self.sigma_s[slot % 16] = sigma_s;
        self.upload_optical_params(queue);
    }

    /// Specular Fresnel base reflectance R0 for `slot` -- see `OpticalTable`'s doc
    /// for the real-but-bounded caveat (constant near-normal reflectance, no
    /// surface-normal-dependent angle term). Auto-uploads immediately -- see
    /// `set_optical_params`'s own doc for why.
    pub fn set_specular_r0(&mut self, queue: &wgpu::Queue, slot: usize, r0: f32) {
        self.specular_r0[slot % 16] = r0;
        self.upload_optical_params(queue);
    }

    fn upload_optical_params(&self, queue: &wgpu::Queue) {
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );
    }

    // ── GPU compute render path ────────────────────────────────────────────────

    /// Zero-readback GPU render. No `sync_particles_blocking()` needed.
    pub fn render_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        if particle_count == 0 {
            return;
        }
        self.ensure_capacity(device, particle_count);

        queue.write_buffer(
            &self.render_config_buf,
            0,
            bytemuck::bytes_of(&RenderConfig {
                mode: self.color_mode as u32,
                particle_count: particle_count as u32,
                vel_scale: self.vel_scale,
                _pad: 0,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prep_bg"),
            layout: &self.prep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.storage_instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.render_config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_gpu"),
        });
        // Compute: fill the storage instance buffer from the particle buffer.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prep_instances"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.prep_pipeline);
            cp.set_bind_group(0, &prep_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(PREP_WG), 1, 1);
        }
        // GPU->GPU copy into the vertex buffer (decouples storage and vertex roles).
        let bytes = (particle_count * mem::size_of::<InstanceData>()) as u64;
        enc.copy_buffer_to_buffer(&self.storage_instances, 0, &self.instance_buffer, 0, bytes);
        // Render: draw instanced quads from the vertex buffer.
        self.draw_pass(&mut enc, output_view, clear, particle_count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Grid-volume render path ────────────────────────────────────────────────

    /// Renders the solver's own grid mass field directly (see `grid_volume.wgsl`'s
    /// own doc for the real technique). Requires `set_camera` to have been called
    /// first (same as `render_gpu` needs for its own bind group) -- reuses the
    /// identical cached orthographic projection/grid_res so both modes line up on
    /// screen without re-deriving them.
    pub fn render_grid_volume(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: GridVolumeSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let (sx, tx, sy, ty) = self.cached_ortho;
        queue.write_buffer(
            &self.grid_volume_params_buf,
            0,
            bytemuck::bytes_of(&GridVolumeParams {
                sx,
                tx,
                sy,
                ty,
                grid_res: self.cached_grid_res,
                // Typical per-particle cell-mass scale here is order 0.5-4 per occupied
                // cell; 0.15 requires non-trivial local density before showing anything,
                // instead of any measurable trace (which combined with bilinear smoothing
                // would overshoot true particle extent).
                mass_floor: 0.15,
                material_mass_enabled: source.material_mass_enabled as u32,
                _pad1: 0.0,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_volume_bg"),
            layout: &self.grid_volume_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_volume_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: source.material_mass.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_grid_volume"),
        });
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            })
        } else {
            wgpu::LoadOp::Load
        };
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_grid_volume"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rp.set_pipeline(&self.grid_volume_pipeline);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── CPU render path ────────────────────────────────────────────────────────

    /// CPU-fill render for the SoA `Particles` store (CPU `Simulation`).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &Particles,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles.iter() {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(&p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_soa"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    pub fn render_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[Particle],
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_cpu"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn ensure_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count > self.max_particles {
            let size = (count * mem::size_of::<InstanceData>()) as u64;
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances_storage"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.max_particles = count;
        }
    }

    fn draw_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: bool,
        count: usize,
    ) {
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            })
        } else {
            wgpu::LoadOp::Load
        };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, &self.render_bind_group, &[]);
        rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rp.set_vertex_buffer(1, self.instance_buffer.slice(..));
        rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..count as u32);
    }
}

// particle_color (the CPU-path per-particle color computation) is split into
// color.rs alongside the rest of the "Color helpers" section below -- see
// that file's own doc comment.
mod color;
use color::write_optical_table;

// Test suite split into its own file -- was ~150 of this file's ~930 lines,
// same pattern as `gpu/solver/device_lost_tests.rs`.
#[cfg(test)]
mod tests;
