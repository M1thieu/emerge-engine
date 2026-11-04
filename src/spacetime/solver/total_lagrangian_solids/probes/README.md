# probes/

Measurements of the total Lagrangian solid path, compiled in test builds only
(`#[cfg(test)]`, most with the `experimental` feature). Nothing here is part of
the engine's behaviour: the engine files of the path stay in the parent folder
(`frame.rs`, `flight.rs`, `handoff.rs`, `rest.rs` with `rest/`, `walls.rs`,
and `total_lagrangian_solids.rs` above them).

A probe is a measured draft. It is written with its criteria before it runs.
Once its question is answered, it is either:
- promoted: the mechanism it measured becomes engine code, with its own tests;
- kept as a test: it guards a property the engine must keep;
- removed: it stays recoverable from the history.

The modules keep their names and paths in the crate (`total_lagrangian_solids::<name>`),
so moving a file in or out of this folder changes only its `#[path]` line.
