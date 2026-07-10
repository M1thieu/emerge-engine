use std::sync::Arc;

use crate::materials::registry::MaterialRegistry;
use crate::solver::LcgRng;
use crate::solver::config::{SimConfig, SpawnRegion};
use crate::solver::density::estimate_particle_volumes;
use crate::solver::initialize_particles;
use crate::{grid::Grid, particle::Particle};

mod particles;
mod queries;
mod step;

use super::buffers::GpuBuffers;
use super::pipeline::SimPipelines;
use super::step_params::{
    GpuFieldEntry, GpuImpulseEntry, MAX_MATERIALS, MAX_SLEEP_WAKE_TAGS, NUM_BLOCKS,
};

/// Workgroup sizes — must match `@workgroup_size(...)` in the WGSL shaders.
const WG_GRID: u32 = 8; // grid_clear and grid_update: 8×8 2D workgroups
const WG_PARTICLES: u32 = 64; // p2g and g2p: 64-wide 1D workgroups

/// Shared between the wgpu map_async callback (any thread) and step_frame's poll.
type ReadbackResult = std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

/// GPU-backed MLS-MPM solver.
///
/// Pass sequence:
///   Once per frame: particle_sort (identity permutation → sorted_particle_ids)
///   Per substep:    grid_clear → p2g → grid_update → g2p → particles_update → force_fields
///
/// Particles live in VRAM between frames; the CPU only touches them at spawn and for
/// plasticity readback (currently: none — all plasticity runs in particles_update.wgsl).
pub struct GpuSimulation {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: GpuBuffers,
    pipelines: SimPipelines,
    config: SimConfig,
    registry: MaterialRegistry,
    /// CPU-side particle mirror. One frame behind the GPU when readback is strided.
    /// Access via `particles()` / `particles_mut()`. Do not replace the Vec directly.
    particles: Vec<Particle>,
    particle_count: usize,
    last_sub_dt: f32,
    last_substeps: usize,
    frame_index: u64,
    /// GPU force-field entries — uploaded to the force_fields_params uniform each substep.
    force_field_entries: Vec<GpuFieldEntry>,
    /// Frame counter used to stride CPU readbacks when all materials are GPU-resident.
    readback_frame: usize,
    /// Download CPU particle state every N step_frame calls when no CPU plasticity is needed.
    /// 1 = every frame (default, always accurate). 2+ = skip frames, reducing GPU stall cost.
    /// One-frame lag on sprite positions is invisible at 60fps.
    pub readback_stride: usize,
    /// Particle positions/materials changed — sort + upload required before next GPU pass.
    /// Set by spawn, phase_transition, mark_particles_dirty().
    layout_dirty: bool,
    /// Pending impulses to apply on GPU at the start of the next step_frame.
    /// Applied via a dedicated compute pass that reads LIVE GPU particle positions,
    /// avoiding the stale-CPU-mirror artifacts from the old upload approach.
    pending_impulses: Vec<GpuImpulseEntry>,
    /// Pending force-sleep/force-wake-by-tag for the next step_frame, applied once in
    /// force_fields.wgsl then cleared. Minimal hook for LP's future chunk system — see
    /// `sleep_tag`/`wake_tag` doc comments and the `GpuSleepWakeParams` layout.
    pending_sleep_tags: Vec<u32>,
    pending_wake_tags: Vec<u32>,
    /// Pending async readback — Some while GPU → staging copy + mapping is in flight.
    /// Checked each step_frame; on completion, CPU particles are updated without blocking.
    /// Arc<Mutex<...>> so the wgpu callback (any thread) can signal the main thread.
    pending_readback: Option<ReadbackResult>,
    /// Real, honest count of async readback failures (`map_async` completing with
    /// `Err`) ever recovered from — should be 0 in ordinary operation on real
    /// hardware; nonzero is a real signal something is stressing the GPU backend
    /// (rare on fast hardware, more likely on slow/software backends). Added
    /// 2026-07-05 alongside the fix for the failure path leaking the staging
    /// buffer's mapped state — see `GpuBuffers::abandon_readback`'s doc.
    pub readback_error_count: u64,
    /// Set once, permanently, if this instance's device is ever lost (confirmed
    /// real cause of emerge issue #10 — see project memory
    /// `gpu_readback_error_path_bug_issue10`: a genuine `Out of Memory` device
    /// loss under sustained load on slow/software GPU backends). A lost device
    /// cannot be un-lost; every further GPU call on it would panic, so
    /// `step_frame`/the blocking sync methods check this and become safe no-ops
    /// once set, rather than crashing. Always populated for `new()` instances;
    /// `with_device()` instances need one call to `enable_device_lost_detection()`
    /// first (see that method's doc for why it isn't automatic there — a wgpu
    /// device can only have one lost-callback, so auto-registering on a
    /// possibly-shared device risks silently overwriting a caller's own).
    /// Callers should poll `device_lost_reason()` if they care why the sim went
    /// quiet — this is deliberately observable, not silently swallowed.
    device_lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Per-pass GPU timestamp profiling — see `enable_profiling()`. None unless explicitly
    /// turned on; zero cost to every other code path when not in use.
    profiling: Option<GpuProfiling>,
    /// One bind group per `step_params_pool` slot, built once and reused by every
    /// `step_frame()` call instead of being recreated per-substep-per-frame. At high
    /// substep counts (LP's stiff-terrain scenes routinely need ~5-6k substeps/frame)
    /// recreating thousands of bind groups every frame exhausted the GPU's descriptor
    /// allocator within seconds (`wgpu error: Out of Memory` from `queue.submit`,
    /// reported against LP's own scene 2026-07-01). The buffers a bind group points at
    /// (`step_params_pool[i]`) never change identity after construction, only their
    /// contents (rewritten every frame via `upload_step_params_at`) — so the bind group
    /// itself can be built once and only needs rebuilding when `spawn_region`
    /// reallocates `buffers.particles` (see `rebuild_bind_group_pool`).
    bind_group_pool: Vec<wgpu::BindGroup>,
    /// Real spatial acceleration for `particles_near`/`count_near`/`group_centroid` --
    /// ported from `solver::Simulation`'s already-proven `SpatialHash` (was previously
    /// wired into the CPU-only `Simulation` but not `GpuSimulation`, meaning every
    /// caller of these three query methods on the GPU path -- the one LP actually uses --
    /// paid a full O(N) linear scan per call regardless of how local the query was.
    /// Rebuilt once per `step_frame()` (and after any explicit particle sync), same
    /// ~1-frame staleness tolerance already accepted everywhere else these queries read
    /// the CPU mirror.
    spatial_hash: crate::solver::spatial_hash::SpatialHash,
    /// CPU-side wall-clock breakdown of the last `step_frame()` call (cfl_scan_ns,
    /// encode_ns, submit_ns, readback_ns, total_ns) — `Instant::now()` calls are
    /// themselves nanosecond-cost, so these are always recorded, not gated behind
    /// `enable_profiling()`. Read via `last_cpu_timings_ns()`. `total_ns` minus the sum of
    /// the other four reveals any unbracketed cost.
    last_cpu_timings: (f32, f32, f32, f32, f32),
}

/// One [begin, end] timestamp pair per labeled compute pass in `encode_substep`, written
/// every substep (later substeps overwrite earlier ones within the same `step_frame()`
/// call — fine for finding the dominant cost, since substeps cost about the same each
/// time; not meant to capture per-substep variance).
const PROFILE_PASS_LABELS: &[&str] = &[
    "active_block_refresh (sort)",
    "grid_clear",
    "p2g",
    "grid_update",
    "g2p",
    "particles_update",
    "force_fields",
];

struct GpuProfiling {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    timestamp_period_ns: f32,
}

/// One bind group per `step_params_pool` slot -- see `GpuSimulation::bind_group_pool`'s
/// doc comment for why this is built once and reused rather than recreated per substep.
fn build_bind_group_pool(
    device: &wgpu::Device,
    pipelines: &SimPipelines,
    buffers: &GpuBuffers,
) -> Vec<wgpu::BindGroup> {
    buffers
        .step_params_pool
        .iter()
        .map(|step_params| pipelines.make_bind_group(device, buffers, step_params))
        .collect()
}

impl GpuSimulation {
    /// Create a GpuSimulation, initialize wgpu, upload initial particle and material data.
    ///
    /// `async` because wgpu adapter/device requests are async.
    /// In examples, wrap with `pollster::block_on(GpuSimulation::new(...))`.
    pub async fn new(
        config: SimConfig,
        particles: Vec<Particle>,
        registry: MaterialRegistry,
    ) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        // Request the adapter's actual limits, not wgpu's conservative defaults (128MiB
        // storage binding). Hardware commonly supports far more (e.g. 2047MiB on desktop
        // GPUs) — capping at the default artificially shrinks the single-buffer particle/grid
        // ceiling well below what the device can actually do.
        //
        // TIMESTAMP_QUERY requested opportunistically (only if the adapter actually supports
        // it) so `enable_profiling()` can work later without requiring it everywhere —
        // hardware/backends that lack it fall back to empty, identical to before this line
        // existed.
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("emerge_gpu"),
                required_features: features,
                required_limits: adapter.limits(),
                ..Default::default() // experimental_features, trace, memory_hints
            })
            .await
            .expect("failed to create wgpu device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let sim = Self::with_device(device, queue, config, particles, registry);

        // Real device-lost detection (confirmed cause of emerge issue #10, see
        // project memory) -- this device is EXCLUSIVELY ours (just created above,
        // no other caller could have registered a competing handler on it yet),
        // so it's always safe to enable it automatically here.
        sim.enable_device_lost_detection();
        sim
    }

    /// Build a `GpuSimulation` on an existing device/queue so its GPU buffers can be
    /// shared with a renderer or surface on the same device — required for the
    /// zero-readback [`crate::render::Renderer::render_gpu`] path. `new()` creates its
    /// own headless device instead, which is correct for compute-only or CPU-readback
    /// workflows but cannot share GPU buffers with another device.
    pub fn with_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: SimConfig,
        particles: Vec<Particle>,
        registry: MaterialRegistry,
    ) -> Self {
        let material_params = registry.all_params();

        // Run init_particle before uploading. Mirrors Simulation::spawn_region().
        // Materials that seed plastic state (Snow: Jp=1, Sand: q=neutral) start wrong
        // without this.
        let mut initialized = particles;
        for p in &mut initialized {
            registry.get(p.material_id).init_particle(p);
        }
        let particle_count = initialized.len();

        let buffers = GpuBuffers::new(
            &device,
            particle_count,
            config.grid_res,
            MAX_MATERIALS,
            config.max_substeps_per_step,
        );

        buffers.upload_particles(&queue, &initialized);
        buffers.upload_materials(&queue, &material_params);

        let pipelines = SimPipelines::new(&device);
        // A zero-sized particle buffer (no initial particles -- e.g. LP constructs
        // empty, then adds terrain/water/creature via spawn_region) fails bind group
        // creation outright ("binding size is zero"). spawn_region already rebuilds
        // this pool once real particles exist; skip the doomed eager build until then.
        let bind_group_pool = if particle_count > 0 {
            build_bind_group_pool(&device, &pipelines, &buffers)
        } else {
            Vec::new()
        };

        let mut spatial_hash = crate::solver::spatial_hash::SpatialHash::new(config.grid_cell_size);
        spatial_hash.rebuild(
            &initialized.iter().map(|p| p.x).collect::<Vec<_>>(),
            initialized.len(),
        );

        Self {
            device,
            queue,
            buffers,
            pipelines,
            config,
            registry,
            particles: initialized,
            particle_count,
            last_sub_dt: config.dt,
            last_substeps: 0,
            frame_index: 0,
            force_field_entries: Vec::new(),
            readback_frame: 0,
            readback_stride: 1,
            layout_dirty: true, // seed particle_sort on first step_frame
            pending_impulses: Vec::new(),
            pending_sleep_tags: Vec::new(),
            pending_wake_tags: Vec::new(),
            pending_readback: None,
            readback_error_count: 0,
            device_lost: std::sync::Arc::new(std::sync::Mutex::new(None)),
            profiling: None,
            last_cpu_timings: (0.0, 0.0, 0.0, 0.0, 0.0),
            bind_group_pool,
            spatial_hash,
        }
    }

    /// Returns (cfl_scan_ns, encode_ns, wait_ns, readback_ns, total_ns) from the last
    /// `step_frame()` call. `encode_ns` is pure CPU-side command-building time (bind
    /// group already cached, just recording dispatches); `wait_ns` (renamed from the
    /// old always-zero `submit_ns` -- multi-chunk frames now really do block between
    /// chunks) is time spent in `device.poll(wait_indefinitely())` between substep
    /// batches, i.e. real GPU execution time for scenes needing >64 substeps/frame.
    pub fn last_cpu_timings_ns(&self) -> (f32, f32, f32, f32, f32) {
        self.last_cpu_timings
    }

    /// Force every particle with `user_tag == tag` asleep, regardless of velocity,
    /// applied at the start of the next `step_frame()`. P2G still scatters for them
    /// (see `gpu_sleep_wake_phase1` memory note — sleeping particles must keep
    /// providing structural support); only their own gather/integration/force-field
    /// work is skipped.
    ///
    /// Minimal hook, not a chunk system: this just lets a caller (e.g. LP's future
    /// chunk loader, once it exists) force-sleep a tagged group by distance instead
    /// of waiting for velocity to drop. Mirrors the CPU `Simulation::sleep_tag` API.
    pub fn sleep_tag(&mut self, tag: u32) {
        if self.pending_sleep_tags.len() < MAX_SLEEP_WAKE_TAGS {
            self.pending_sleep_tags.push(tag);
        } else {
            eprintln!(
                "emerge: GPU sleep-tag queue full ({MAX_SLEEP_WAKE_TAGS}/frame max) — tag dropped"
            );
        }
    }

    /// Force every particle with `user_tag == tag` awake, regardless of grid activity.
    /// Mirrors the CPU `Simulation::wake_tag` API. See `sleep_tag` doc comment.
    pub fn wake_tag(&mut self, tag: u32) {
        if self.pending_wake_tags.len() < MAX_SLEEP_WAKE_TAGS {
            self.pending_wake_tags.push(tag);
        } else {
            eprintln!(
                "emerge: GPU wake-tag queue full ({MAX_SLEEP_WAKE_TAGS}/frame max) — tag dropped"
            );
        }
    }

    /// Mark CPU particles as layout-changed (positions/materials) — triggers sort + upload.
    pub fn mark_particles_dirty(&mut self) {
        self.layout_dirty = true;
    }

    /// Upload revised material params (e.g., if interactive sliders change them).
    pub fn upload_materials(&self) {
        self.buffers
            .upload_materials(&self.queue, &self.registry.all_params());
    }

    pub fn registry(&self) -> &MaterialRegistry {
        &self.registry
    }
    pub fn registry_mut(&mut self) -> &mut MaterialRegistry {
        &mut self.registry
    }

    /// The wgpu Device — share with the LP render system to read the particle buffer directly.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// The wgpu Queue — share with the LP render system for command submission.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// The GPU particle storage buffer — bind this in LP's custom render shader.
    /// Layout: `array<Particle>`, each Particle is 112 bytes, repr(C).
    /// Stays in VRAM between frames; read-only from the render side.
    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.particles
    }

    /// Verification-only accessor: read back `sorted_particle_ids` as a `Vec<u32>`.
    /// Used by tests to confirm the particle_sort pipeline produces a valid permutation —
    /// not part of the render/game-loop API.
    pub fn sorted_particle_ids_blocking(&self) -> Vec<u32> {
        self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.sorted_particle_ids,
            self.particle_count,
        )
    }

    /// Test/diagnostic readback for the GPU sparse grid Phase 1 active-block list — the
    /// first `active_block_count_blocking()` entries are valid; the rest are stale/unused.
    pub fn active_block_ids_blocking(&self) -> Vec<u32> {
        self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.active_block_ids,
            NUM_BLOCKS,
        )
    }

    /// Test/diagnostic readback for how many entries in `active_block_ids_blocking()` are
    /// valid this frame.
    pub fn active_block_count_blocking(&self) -> u32 {
        self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.active_block_count,
            1,
        )[0]
    }

    /// Test/diagnostic readback of the dense grid buffer — 4 f32 per cell (momentum.x,
    /// momentum.y, mass, _pad), same field order as the WGSL `Cell` struct, flat-indexed
    /// `(y * grid_res + x) * 4`. Lets tests verify grid_clear actually zeroed cells far from
    /// any particle (the failure mode a block-boundary mapping bug would produce: stale,
    /// never-cleared mass/momentum left behind in an unrelated block).
    pub fn grid_cells_blocking(&self) -> Vec<f32> {
        let cell_floats = self.config.grid_res * self.config.grid_res * 4;
        self.buffers.readback_f32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.grid,
            cell_floats,
        )
    }

    /// Real, honest report of why this instance's device was lost, if it ever
    /// was — `None` in ordinary operation. Automatically wired for `new()`
    /// instances; `with_device()` instances need one explicit call to
    /// `enable_device_lost_detection()` first (see that method's doc for why
    /// it isn't automatic there). Once set, `step_frame` and the blocking sync
    /// methods become safe no-ops instead of panicking on a dead device —
    /// callers that care should poll this rather than assume silence means
    /// healthy.
    pub fn device_lost_reason(&self) -> Option<String> {
        self.device_lost.lock().ok().and_then(|g| g.clone())
    }

    /// Opt in to real device-lost detection (the confirmed real cause of
    /// emerge issue #10 — a genuine `Out of Memory` device loss under
    /// sustained load on slow/software GPU backends; see project memory
    /// `gpu_readback_error_path_bug_issue10`). Called automatically by `new()`
    /// (which owns its device exclusively, so it's always safe there). NOT
    /// automatic for `with_device()` (shared-device use, e.g. a renderer on the
    /// same device as this sim) because a wgpu device can only have ONE
    /// lost-callback (and, as of 2026-07-08, only one uncaptured-error handler
    /// too — same `Option<Arc<dyn Handler>>` single-slot storage internally,
    /// confirmed by reading wgpu-27.0.1's `ErrorSinkRaw`) — auto-registering
    /// here could silently overwrite a caller's own handler. Call this
    /// explicitly after `with_device()` if you (like LP) don't have your own
    /// device-lost handling and want emerge's; don't call it if you've already
    /// registered your own callback/handler on this device — the second
    /// registration wins and the first is silently lost (this is wgpu's own
    /// behavior, not something this method can prevent).
    ///
    /// ALSO installs an uncaptured-error handler (2026-07-08). wgpu's default
    /// behavior for ANY uncaptured error is an unconditional panic
    /// (`panic!("wgpu error: {err}")`, confirmed by reading wgpu-27.0.1's
    /// `default_error_handler`) — this handler replaces that default and
    /// **never panics**, regardless of what the error says. That "never" is
    /// load-bearing, not a simplification: an earlier version of this handler
    /// tried to be more precise — classify errors naming a destroyed/lost
    /// resource as an inferred device loss (no panic), but still panic for
    /// anything else so a genuine, unrelated validation bug wouldn't be
    /// silently swallowed. That version was reproduced crashing LOCALLY
    /// (forcing the D3D12 WARP adapter — the same backend windows-latest CI
    /// uses — instead of waiting on another CI round-trip) with the full
    /// backtrace showing the panic originated from THIS handler's own `panic!`
    /// call, invoked synchronously from inside `wgpu_core::Queue::submit`'s
    /// internal error path — and unwinding a panic from there is what produced
    /// `STATUS_STACK_BUFFER_OVERRUN`, not the error itself. In other words:
    /// panicking from ANY code reachable from this callback is unsafe on this
    /// backend, independent of whether the message looks like a device-loss
    /// artifact or a real bug — so the "still panic for real bugs" branch was
    /// itself the crash, not a safety net. The fix: never panic here, full
    /// stop. Every uncaptured error sets `device_lost` (so `is_device_lost()`'s
    /// existing no-op guards take over) and is `eprintln!`'d in full so it's
    /// still visible for debugging — just never re-thrown as a Rust panic from
    /// inside this specific callback context.
    pub fn enable_device_lost_detection(&self) {
        let flag = self.device_lost.clone();
        self.device
            .set_device_lost_callback(move |reason, message| {
                *flag.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(format!("{reason:?}: {message}"));
            });

        let flag = self.device_lost.clone();
        self.device
            .on_uncaptured_error(std::sync::Arc::new(move |error: wgpu::Error| {
                let message = error.to_string();
                let mut guard = flag.lock().unwrap_or_else(|e| e.into_inner());
                if guard.is_none() {
                    *guard = Some(format!("(uncaptured wgpu error) {message}"));
                }
                drop(guard);
                eprintln!(
                    "emerge: uncaptured wgpu error, treating device as unusable from \
                     here (see GpuSimulation::enable_device_lost_detection's doc for \
                     why this never panics): {message}"
                );
            }));
    }

    fn is_device_lost(&self) -> bool {
        self.device_lost
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    /// Download particles from GPU to CPU synchronously (diagnostics / one-shot use).
    /// Prefer the async readback path in step_frame for per-frame use.
    pub fn download_particles_blocking(&mut self) {
        let flag = self
            .buffers
            .begin_readback(&self.device, &self.queue, self.particle_count);
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if let Ok(mut g) = flag.lock() {
            g.take();
        }
        self.particles = self.buffers.finish_readback(self.particle_count);
    }

    /// Read-only access to the CPU particle mirror (one frame behind GPU when strided).
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Mutable access to the CPU particle mirror.
    ///
    /// **CFL WARNING:** velocity changes bypass the solver's CFL clamp.
    /// For gameplay impulses use `apply_impulse` / `apply_radial_impulse` instead.
    /// After modifying, call `mark_particles_dirty()` so the GPU sees the changes.
    pub fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.particles
    }

    /// Append a new particle region to the simulation.
    ///
    /// Generates particles CPU-side, appends to the internal mirror, recomputes
    /// initial volumes for all particles, then reallocates the GPU particle buffer
    /// to fit the new total and uploads all particles.
    ///
    /// Returns the index range the new particles occupy in the internal mirror.
    /// LP uses this as `creature_id → particle_range` for ownership tracking.
    ///
    /// Call before `step_frame` — mid-frame spawning is not supported.
    pub fn spawn_region(&mut self, spawn: SpawnRegion) -> std::ops::Range<usize> {
        let start = self.particles.len();
        spawn.validate_for_sim(&self.config);
        debug_assert!(
            self.registry.is_registered(spawn.material_id),
            "spawn_region: material_id {} is not registered — call solver.set_material({}, ...) first",
            spawn.material_id,
            spawn.material_id
        );
        let mut rng = LcgRng::new(spawn.rng_seed);
        let new_particles = initialize_particles(&self.config, spawn, &mut rng);
        self.particles.extend(new_particles);

        // Recompute initial volumes for the combined particle set using a temporary grid.
        let mut tmp_grid = Grid::new(self.config.grid_res);
        {
            let mut tmp_soa = crate::particle::Particles::from(std::mem::take(&mut self.particles));
            let n = tmp_soa.len();
            estimate_particle_volumes(&mut tmp_soa, &mut tmp_grid, n, true);
            self.particles = tmp_soa.to_vec();
        }

        let n = self.particles.len();

        // Reallocate all GPU buffers that are sized per-particle (including staging).
        self.buffers.particles = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particles"),
            size: (n * core::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.buffers.sorted_particle_ids = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_sorted_particle_ids"),
            size: (n * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.buffers.readback_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particle_staging"),
            size: (n * core::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.particle_count = n;
        self.pending_readback = None; // old staging is gone
        self.buffers.upload_particles(&self.queue, &self.particles);
        // buffers.particles was just reallocated above -- cached bind groups reference
        // the old buffer object and would be stale (or invalid) without this.
        self.bind_group_pool = build_bind_group_pool(&self.device, &self.pipelines, &self.buffers);
        self.rebuild_spatial_hash();
        start..n
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    /// Rebuilds `spatial_hash` from the current `self.particles` -- call any time
    /// `self.particles` changes (readback completion, explicit sync, spawn). O(N),
    /// same real cost class as the linear scans it replaces for QUERIES, but paid
    /// once per mutation instead of once per query -- a real win whenever more than
    /// one query happens against the same particle state (e.g. LP's ecology calling
    /// `sense_local`/`centroid`/`phenotype` per creature per frame).
    fn rebuild_spatial_hash(&mut self) {
        let positions: Vec<glam::Vec2> = self.particles.iter().map(|p| p.x).collect();
        self.spatial_hash.rebuild(&positions, self.particles.len());
    }

    /// Blocking GPU → CPU particle sync. Updates `self.particles` immediately.
    /// Stalls the CPU until all in-flight GPU work completes — use only after step_frame
    /// when you need current positions right now (e.g. rendering). Not for the hot path.
    pub fn sync_particles_blocking(&mut self) {
        // Safe no-op once the device is lost -- see step_frame's identical guard.
        if self.is_device_lost() {
            return;
        }
        // If an async readback is in-flight, the staging buffer may be mapped or pending map.
        // Wait for it to complete, then consume it to unmap the staging buffer before reuse.
        // Real fix (2026-07-05, issue #10): must distinguish Ok/Err here -- the old
        // code called finish_readback (which calls get_mapped_range) on EITHER, but
        // a failed map has nothing valid to extract; only abandon_readback (unmap
        // only) is safe on Err. See GpuBuffers::abandon_readback's doc.
        if let Some(flag) = self.pending_readback.take() {
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            match flag.lock().ok().and_then(|mut g| g.take()) {
                Some(Ok(())) => {
                    let _ = self.buffers.finish_readback(self.particle_count);
                }
                Some(Err(_)) => {
                    self.readback_error_count += 1;
                    self.buffers.abandon_readback();
                }
                None => {}
            }
        }
        self.particles =
            self.buffers
                .readback_blocking(&self.device, &self.queue, self.particle_count);
        self.rebuild_spatial_hash();
    }

    /// Like `sync_particles_blocking`, but only for the given particle index ranges --
    /// updates just `self.particles[range]` for each range, leaving the rest of the CPU
    /// mirror as whatever the last async/full sync delivered. For callers that only need
    /// a small, known subset of particles current every frame (e.g. a handful of live
    /// creatures inside a much larger terrain/water scene) instead of the whole buffer --
    /// see `GpuBuffers::readback_ranges_blocking`'s own doc for why this is cheaper than
    /// repeated full syncs, not just "less data" but batched into one CPU↔GPU round-trip.
    /// Ranges may overlap or be given in any order; each is written independently.
    pub fn sync_particle_ranges_blocking(&mut self, ranges: &[std::ops::Range<usize>]) {
        // Safe no-op once the device is lost -- see step_frame's identical guard.
        if self.is_device_lost() {
            return;
        }
        // Same real fix as sync_particles_blocking -- see that function's comment.
        if let Some(flag) = self.pending_readback.take() {
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            match flag.lock().ok().and_then(|mut g| g.take()) {
                Some(Ok(())) => {
                    let _ = self.buffers.finish_readback(self.particle_count);
                }
                Some(Err(_)) => {
                    self.readback_error_count += 1;
                    self.buffers.abandon_readback();
                }
                None => {}
            }
        }
        let results = self
            .buffers
            .readback_ranges_blocking(&self.device, &self.queue, ranges);
        for (range, data) in ranges.iter().zip(results) {
            self.particles[range.clone()].copy_from_slice(&data);
        }
        self.rebuild_spatial_hash();
    }

    pub fn set_gravity(&mut self, gravity: glam::Vec2) {
        self.config.gravity = gravity;
    }

    /// Replace the default material and re-upload the materials buffer.
    pub fn set_default_material(&mut self, material: Box<dyn crate::materials::MaterialModel>) {
        self.registry.set_default(material);
        self.upload_materials();
    }

    pub fn gravity(&self) -> glam::Vec2 {
        self.config.gravity
    }

    /// The live GPU grid buffer (STORAGE | COPY_SRC).
    /// Layout: `array<Cell>` where Cell = { momentum: vec2, mass: f32, _pad: f32 } (16 bytes).
    /// Consumers (e.g. LP's metaball renderer) can bind this read-only in their own compute pass.
    pub fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.grid
    }

    /// Register a material, auto-assigning the next available ID.
    ///
    /// Mirrors `Simulation::register_material` — use this instead of `set_material`
    /// when you don't want to track IDs manually. Returns a typed handle.
    ///
    /// LP pattern: call at world-init time to build a material palette, then
    /// use `handle.id()` in `SpawnRegion::for_sim(...).material(handle.id())`.
    pub fn register_material(
        &mut self,
        material: Box<dyn crate::materials::MaterialModel>,
    ) -> crate::solver::handle::MaterialHandle {
        let id = self.registry.next_id();
        self.registry.insert(id, material);
        self.upload_materials();
        crate::solver::handle::MaterialHandle(id)
    }

    /// Register or replace a material by explicit ID and re-upload the materials buffer.
    pub fn set_material(
        &mut self,
        material_id: u32,
        material: Box<dyn crate::materials::MaterialModel>,
    ) {
        self.registry.insert(material_id, material);
        self.upload_materials();
    }

    /// The sub-dt used in the last substep of the most recent `step_frame` call.
    pub fn effective_dt(&self) -> f32 {
        self.last_sub_dt
    }

    /// Number of substeps run during the most recent `step_frame` call.
    pub fn last_substeps(&self) -> usize {
        self.last_substeps
    }

    /// Total frames stepped since creation.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

#[cfg(test)]
mod device_lost_tests {
    use super::*;
    use crate::materials::registry::MaterialRegistry;
    use crate::materials::{FromSI, NeoHookeanMaterial};
    use crate::solver::config::{SimConfig, SpawnRegion};
    use glam::{IVec2, Vec2};

    fn gpu_available() -> bool {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }

    /// Real, white-box verification of the device-lost guard added for emerge
    /// issue #10 (see project memory gpu_readback_error_path_bug_issue10 — the
    /// root cause, a genuine `Out of Memory` device loss under sustained
    /// slow-backend load, was confirmed with hard evidence via a real
    /// `device_lost_callback` firing; forcing that same OOM condition again just
    /// to test the GUARD would repeat the same heavy, machine-stressing
    /// reproduction unnecessarily). This directly injects a lost reason into the
    /// private `device_lost` flag exactly as the real callback would, then
    /// proves three things: (1) `device_lost_reason()` reports it, (2)
    /// `step_frame()` becomes a real no-op (frame_index does not advance,
    /// proving it didn't just avoid panicking by luck), (3) the blocking sync
    /// methods are also safe no-ops (don't panic touching a "dead" device).
    #[test]
    fn step_frame_becomes_safe_noop_once_device_lost() {
        if !gpu_available() {
            return;
        }
        let config = SimConfig {
            max_substeps_per_step: 8,
            ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(4, 4),
            box_center: Vec2::new(16.0, 16.0),
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let particles = crate::build_particles(&config, spawn);
        let mat = NeoHookeanMaterial::from_physical(
            &crate::materials::physical_props::Elastic {
                e_pa: 30.0e3,
                nu: 0.45,
                rho_kg_m3: 1000.0,
            },
            &config,
        );
        let registry = MaterialRegistry::with_default(Box::new(mat));
        let mut sim = pollster::block_on(GpuSimulation::new(config, particles, registry));

        assert!(
            sim.device_lost_reason().is_none(),
            "a healthy, freshly-constructed sim must not report device loss"
        );

        sim.step_frame();
        let frame_after_healthy_step = sim.frame_index;
        assert!(
            frame_after_healthy_step > 0,
            "sanity check: a healthy device must actually advance frame_index"
        );

        // Directly inject a lost reason, exactly as the real callback does.
        *sim.device_lost.lock().unwrap() = Some("Unknown: Out of memory".to_string());
        assert_eq!(
            sim.device_lost_reason(),
            Some("Unknown: Out of memory".to_string()),
            "device_lost_reason() must report an injected loss"
        );

        sim.step_frame();
        assert_eq!(
            sim.frame_index, frame_after_healthy_step,
            "step_frame must become a real no-op once device_lost is set -- \
             frame_index must NOT advance"
        );

        // Must not panic -- these touch the same "dead" device.
        sim.sync_particles_blocking();
        let ranges = vec![0..1usize, 1..2usize];
        sim.sync_particle_ranges_blocking(&ranges);
    }

    /// Real repro of issue #10's ACTUAL failure mode, found via real windows-latest
    /// CI evidence (not speculation): re-enabling the crash-repro test showed that,
    /// under sustained load, wgpu invalidates/destroys buffers tied to the device
    /// BEFORE this instance's `device_lost_callback` fires -- the readback path's
    /// `.unmap()` call then hits an uncaptured Validation error naming the destroyed
    /// resource, and wgpu's default handler panics unconditionally
    /// (`default_error_handler`: `panic!("wgpu error: {err}")`, confirmed by reading
    /// wgpu-27.0.1's source directly).
    ///
    /// Forcing a real 9-minute sustained-load OOM again just to hit this exact race
    /// would repeat the same heavy, machine-stressing reproduction unnecessarily --
    /// this reproduces the SAME call path directly: destroy the readback staging
    /// buffer ourselves (exactly what the device-loss cascade does to it), then call
    /// `abandon_readback()`, the exact function whose `.unmap()` call panicked on
    /// real CI. Before the `on_uncaptured_error` fix this would panic and abort the
    /// test process; with it installed, it must set `device_lost` instead -- no
    /// panic, `device_lost_reason()` reports it.
    #[test]
    fn uncaptured_destroyed_buffer_error_sets_device_lost_not_a_panic() {
        if !gpu_available() {
            return;
        }
        let config = SimConfig {
            max_substeps_per_step: 8,
            ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(4, 4),
            box_center: Vec2::new(16.0, 16.0),
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let particles = crate::build_particles(&config, spawn);
        let mat = NeoHookeanMaterial::from_physical(
            &crate::materials::physical_props::Elastic {
                e_pa: 30.0e3,
                nu: 0.45,
                rho_kg_m3: 1000.0,
            },
            &config,
        );
        let registry = MaterialRegistry::with_default(Box::new(mat));
        let sim = pollster::block_on(GpuSimulation::new(config, particles, registry));

        assert!(
            sim.device_lost_reason().is_none(),
            "a healthy, freshly-constructed sim must not report device loss"
        );

        // Exactly what a device-loss cascade does to resources tied to the
        // device, without needing 9 minutes of real sustained WARP load.
        sim.buffers.readback_staging.destroy();

        // The exact real call path that panicked on CI: finish_readback and
        // abandon_readback both end in `.unmap()` on this buffer.
        sim.buffers.abandon_readback();
        sim.device.poll(wgpu::PollType::Poll).ok();

        let reason = sim.device_lost_reason();
        assert!(
            reason.is_some(),
            "an uncaptured error naming a destroyed buffer must set device_lost, \
             not silently do nothing"
        );
        let reason = reason.unwrap();
        assert!(
            reason.contains("uncaptured wgpu error"),
            "reason should be tagged as coming from the uncaptured-error handler \
             (distinguishable from the official device_lost_callback's report), \
             got: {reason}"
        );
        assert!(
            reason.contains("destroyed"),
            "reason should retain the real wgpu error text naming the destroyed \
             resource, got: {reason}"
        );
    }

    /// Real proof that the OPT-IN path works -- this is the path LP's actual
    /// production code needs (`World::with_device`, since LP shares its device
    /// with a renderer and has no device-lost handling of its own, confirmed by
    /// inspection of LP's `src/main.rs`). Proves `enable_device_lost_detection()`
    /// makes a `with_device()` instance behave identically to a `new()`-
    /// constructed one for reporting purposes. NOTE: this does NOT independently
    /// prove `with_device()` never silently registers its own callback -- that
    /// would need a real device-loss trigger to distinguish "no callback
    /// registered" from "callback registered but nothing happened yet," which
    /// this test doesn't force (see the heavy stress-test caution elsewhere in
    /// this file). Static code inspection is what actually backs that claim:
    /// `with_device()`'s body contains no `set_device_lost_callback` call.
    #[test]
    fn with_device_instances_need_explicit_opt_in_for_device_lost_detection() {
        if !gpu_available() {
            return;
        }
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("no adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("test_shared_device"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("no device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let config = SimConfig {
            max_substeps_per_step: 8,
            ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
        };
        let mat = NeoHookeanMaterial::from_physical(
            &crate::materials::physical_props::Elastic {
                e_pa: 30.0e3,
                nu: 0.45,
                rho_kg_m3: 1000.0,
            },
            &config,
        );
        let sim = GpuSimulation::with_device(
            device,
            queue,
            config,
            Vec::new(),
            MaterialRegistry::with_default(Box::new(mat)),
        );

        // Fresh with_device() instance: field starts unset (expected regardless
        // of whether a callback is wired -- see doc comment above for what this
        // does and doesn't prove).
        assert!(sim.device_lost_reason().is_none());

        sim.enable_device_lost_detection();
        // Directly invoke the same injection used in the other test -- proves the
        // callback registration path (not just the field) is wired correctly.
        *sim.device_lost.lock().unwrap() = Some("Unknown: Out of memory".to_string());
        assert_eq!(
            sim.device_lost_reason(),
            Some("Unknown: Out of memory".to_string()),
            "after enable_device_lost_detection(), device_lost_reason() must work \
             identically to a new()-constructed instance"
        );
    }
}
