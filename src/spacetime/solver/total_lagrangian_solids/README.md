# total_lagrangian_solids

**Level:** discretization. **Status:** the precise path, the engine's target.

Runs the step of `spacetime/total_lagrangian/` frame by frame inside `Simulation`.
The state that counts is kept here in `f64` and SI; the particles' `f32` copies
are written each frame for the renderer and the queries.

**Derives from:** `spacetime/total_lagrangian/` (the step), `forces/boundary/` (the
domain walls, through `walls.rs`), `matter/materials/` (each body's record).

| File | What it is |
|---|---|
| `../total_lagrangian_solids.rs` | the frame loop, and what the path refuses, with the reason |
| `frame.rs` | one frame's record: its energy ledger and counts |
| `flight.rs` | a body in free flight carried by its exact modes |
| `handoff.rs` | when flight hands back to the step inside a frame |
| `rest.rs`, `rest/` | a body at rest carried by its constrained modes |
| `walls.rs` | the wall laws as contact planes: an adapter, no law of its own |
| `probes/` | test builds only; see its README |
