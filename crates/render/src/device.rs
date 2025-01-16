//! device.rs - Manages GPU initialization and rendering resources.

use wgpu::{Device, Instance, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration};
use wgpu::util::DeviceExt; // Import DeviceExt for helper methods
use std::sync::Arc;

/// Manages GPU device and queue.
pub struct RenderDevice {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl RenderDevice {
    /// Initialize GPU with wgpu backend.
    pub async fn new(surface: &Surface, config: &SurfaceConfiguration) -> Self {
        // Create GPU instance
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            dx12_shader_compiler: Default::default(),
        });

        // Request adapter (GPU selection)
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            ..Default::default()
        })
        .await
        .expect("Failed to find a compatible GPU.");

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // No trace path
            )
            .await
            .expect("Failed to create GPU device.");

        // Configure the surface
        surface.configure(&device, config);

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }

    /// Create a buffer for vertex or index data.
    pub fn create_buffer(&self, usage: wgpu::BufferUsages, data: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Buffer"),
            contents: data,
            usage,
        })
    }

    /// Create a shader module.
    pub fn create_shader(&self, source: &str) -> wgpu::ShaderModule {
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }
}
