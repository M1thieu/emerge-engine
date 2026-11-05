# grains

**Level:** discretization, a second solver for discrete grains. **Status:** CPU only,
coupled to the explicit solver's grid (`coupling.rs`), not to the precise path.

The dynamics of discrete-element grains: their container and integration
(`population.rs`, `implicit.rs`), grid scatter and gather (`coupling.rs`), and the
signal for where grains are needed (`oracle.rs`).

**Derives from:** `matter/materials/granular/grain_contact_law.rs` (the contact force
law) and `matter/particle/` (`Grain`, the per-grain state).
