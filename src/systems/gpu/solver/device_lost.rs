use super::GpuSimulation;

impl GpuSimulation {
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

    pub(super) fn is_device_lost(&self) -> bool {
        self.device_lost
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }
}
