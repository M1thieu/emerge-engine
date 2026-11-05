# rod

**Level:** discretization, a second solver for slender bodies. **Status:** CPU only,
coupled to the explicit solver's grid (`coupling.rs`), not to the precise path.

Discrete elastic rods (Bergou et al. 2008) for bodies whose length far exceeds their
width.

**Derives from:** `matter/materials/rod_material.rs` (`EA`, `EI` from measured
properties), `matter/particle/` (`RodPoints`).

`gravitropism.rs`, `growth.rs`, `secondary_growth.rs` and `plasticity.rs` hold laws
that belong outside `spacetime`: they stay here until the folders are brought in line.
