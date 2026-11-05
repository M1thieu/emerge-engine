# total_lagrangian

**Level:** discretization. **Status:** the precise path, the engine's target.

The conserving implicit step for a solid, in `f64` and SI: one step, from the
state at its start to the state at its end, with no knowledge of frames or of
`Simulation`. `spacetime/solver/total_lagrangian_solids/` runs it frame by frame.

**Derives from:**
- `matter/materials/`: each law's energy and stress (`elastic.rs` dispatches to them);
- `forces/boundary/signorini_coulomb.rs`: the contact law (`contact.rs` and
  `weighted_gap.rs` enforce it on this path's unknowns);
- the balance of momentum in its total Lagrangian form, and the energy-momentum
  midpoint of Love and Sulsky (head of `mod.rs`).

| File | What it is |
|---|---|
| `step.rs` | one step, with its contact passes |
| `solve.rs` | Newton on the step's equations |
| `krylov.rs`, `band.rs` | the linear solves and their preconditioner |
| `basis.rs` | this path's own grid: quadratic B-splines on the reference configuration |
| `elastic.rs`, `relaxation.rs` | the solid on the step: discrete-gradient stress, relaxation branches |
| `contact.rs`, `weighted_gap.rs`, `multipliers.rs` | where surfaces touch, and the multipliers that hold them |
| `step_control.rs` | how long each step is, from its estimated error |
| `modes.rs` | the body's vibration modes, for flight and rest |
| `probes/`, `research/` | test builds only; see their READMEs |
