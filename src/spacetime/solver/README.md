# solver

**Level:** discretization and host. **Status:** mixed, by file.

`Simulation` lives here, and `Simulation::step` picks the path:
`SimConfig::total_lagrangian` set runs the precise path, otherwise the explicit
solver runs.

| Files | Status | What they are |
|---|---|---|
| `mod.rs`, `lifecycle.rs`, `particles.rs`, `queries.rs`, `handle.rs`, `body_state.rs`, `spatial_hash.rs`, `grid_units.rs`, `config/` | shared host | the world: bodies, queries, units, configuration |
| `total_lagrangian_solids.rs`, `total_lagrangian_solids/` | **precise path** | the frame loop on `Simulation` |
| `step.rs`, `cfl.rs`, `projection.rs`, `density.rs`, `boundary_diagnostics.rs`, `implicit_corotated.rs` | **deprecated** | the explicit MPM solver: maintenance only, see `spacetime/README.md` |
