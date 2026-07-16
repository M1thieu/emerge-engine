use super::{GpuProfiling, GpuSimulation, PROFILE_PASS_LABELS};

impl GpuSimulation {
    /// Turns on per-pass GPU timing for `encode_substep`'s 7 labeled passes. Returns false
    /// (no-op) if this device wasn't created with `TIMESTAMP_QUERY` support — `new()`
    /// requests it opportunistically when the adapter supports it; `with_device()` depends
    /// on whatever device the caller already built. Call once after construction; read
    /// results back with `last_pass_timings_ns()` after stepping a few frames.
    pub fn enable_profiling(&mut self) -> bool {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return false;
        }
        let n = PROFILE_PASS_LABELS.len() as u32;
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("emerge_profile_queries"),
            ty: wgpu::QueryType::Timestamp,
            count: n * 2, // begin+end per pass
        });
        let resolve_size = (n * 2) as u64 * 8; // 8 bytes per u64 timestamp
        let resolve_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emerge_profile_resolve"),
            size: resolve_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emerge_profile_readback"),
            size: resolve_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.profiling = Some(GpuProfiling {
            query_set,
            resolve_buf,
            readback_buf,
            timestamp_period_ns: self.queue.get_timestamp_period(),
        });
        true
    }

    /// Reads back the last substep's per-pass GPU timings (label, nanoseconds), in
    /// `encode_substep`'s pass order. Blocks until the GPU work + readback completes — a
    /// diagnostic call, not for the hot path. Returns None if `enable_profiling()` wasn't
    /// called or wasn't supported on this device.
    pub fn last_pass_timings_ns(&mut self) -> Option<Vec<(&'static str, f32)>> {
        let profiling = self.profiling.as_ref()?;
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let slice = profiling.readback_buf.slice(..);
        let flag = std::sync::Arc::new(std::sync::Mutex::new(None));
        let flag2 = flag.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            *flag2.lock().unwrap() = Some(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        flag.lock().unwrap().take()?.ok()?;
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        let period = profiling.timestamp_period_ns;
        let result = PROFILE_PASS_LABELS
            .iter()
            .enumerate()
            .map(|(i, &label)| {
                let begin = timestamps[i * 2];
                let end = timestamps[i * 2 + 1];
                (label, (end.saturating_sub(begin)) as f32 * period)
            })
            .collect();
        drop(data);
        profiling.readback_buf.unmap();
        Some(result)
    }

    /// Builds `ComputePassTimestampWrites` for pass index `i` (in `PROFILE_PASS_LABELS`
    /// order) if profiling is enabled, else `None` — keeps each pass's descriptor a
    /// one-liner regardless of whether profiling is active.
    pub(super) fn profile_writes(&self, i: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.profiling
            .as_ref()
            .map(|p| wgpu::ComputePassTimestampWrites {
                query_set: &p.query_set,
                beginning_of_pass_write_index: Some(i * 2),
                end_of_pass_write_index: Some(i * 2 + 1),
            })
    }
}
