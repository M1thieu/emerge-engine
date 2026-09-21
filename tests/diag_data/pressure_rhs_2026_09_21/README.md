# Pressure projection RHS, variant C (core audit, 2026-09-21)

Variant C, in `src/spacetime/grid/pressure.rs` on this branch: the divergence
right-hand side is computed on fluid cells only (mass > 0), and a massless
neighbour reads the centre cell's own velocity. The current code instead reads
a missing neighbour as velocity zero, on every cell of the padded box.

The 2026-09-17 attempt (`velocity_at_or_extrapolated`, reverted, see the note in
`pressure.rs`) cleaned the fluid cells but left the same fake source on the air
cells next to the body, which the solver keeps as unknowns.

Measured with variant C (quick profile, CPU):

| Scene | Current code | Variant C |
|---|---|---|
| Falling droplet, full gravity | J at the [0.5, 2] clamp at frame 1 | passes, J = 1.0000, v = g t |
| Falling droplet, 0.003 g | clamp at frame 6-7 | J = 1.0000 to frame 19, clamp at frame 37 |
| Wall column, 0.003 g | no NaN, but J at the clamp frames 20-120 | exact at frame 1, panics at frame 2 |

`wall_scene_variant_c_rhs_trace.txt`: `EMERGE_DEBUG_PRESSURE=1` output of the
wall column run. The RHS is zero for 6 substeps, then a 1e-6 roundoff seed grows
about 8 to 10 times per substep up to 6.5e7: the projection amplifies
divergence instead of removing it. Variant C is necessary, not sufficient.
