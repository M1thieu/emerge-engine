//! Spacetime domain: the arena.
//!
//! `grid` — `Grid`/`Cell`, the Eulerian background grid + quadratic B-spline
//! kernel. `solver` — `Simulation`, `SimConfig`, `SpawnRegion`: orchestrates
//! the whole substep loop over that grid. `transfer` — P2G/G2P transfer
//! kernels, the actual particle-grid-particle bridge each substep.
//!
//! Part of the emerge/LP domain taxonomy (matter/forces/energy/information/
//! spacetime/organism/systems) -- see `project_domain_taxonomy` design notes.
//! Re-exported at crate root (`pub use spacetime::grid;` etc in `lib.rs`) so
//! every existing `crate::grid::`/`crate::solver::`/`crate::transfer::` path
//! and every LP equivalent keeps resolving unchanged -- this move only
//! changes where the files physically live, not any public API.

pub mod grid;
pub mod solver;
pub mod transfer;
