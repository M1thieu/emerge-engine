//! Forces domain: what acts on matter.
//!
//! `fields` — the `Field` trait + force-field impls (gravity, Coulomb, EM,
//! confinement, buoyancy, chemotaxis). `boundary` — the `BoundaryCondition`
//! trait + wall/friction/terrain impls. Both apply forces/constraints to
//! particles; Forces' core-gravity and EM-force-application pieces live here
//! too (EM's radiative/energy-transfer half is Energy's, not Forces').
//!
//! Part of the emerge/LP domain taxonomy (matter/forces/energy/information/
//! spacetime/organism/systems) -- see `project_domain_taxonomy` design notes.
//! Re-exported at crate root (`pub use forces::fields;` etc in `lib.rs`) so
//! every existing `crate::fields::`/`crate::boundary::` path and every LP
//! `emerge::fields::`/`emerge::boundary::` path keeps resolving unchanged --
//! this move only changes where the files physically live, not any public API.

pub mod boundary;
pub mod fields;
