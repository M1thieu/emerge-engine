# probes/

Measurements and test instrumentation of the total Lagrangian step, compiled in
test builds only. Nothing here is part of the engine's behaviour. A probe is a
measured draft, written with its criteria before it runs. Once its question is
answered it is promoted to engine code with its own tests, kept as a test, or
removed (it stays recoverable from the history).

`hooks.rs` is the instrumentation the step calls under `#[cfg(test)]`; `tests.rs`
is the step's unit tests.

The modules keep their names in the crate (`total_lagrangian::<name>`), so moving
a file in or out of this folder changes only its `#[path]` line.
