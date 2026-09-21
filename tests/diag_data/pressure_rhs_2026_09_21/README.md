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
| Wall column, 0.003 g | no NaN, but J at the clamp frames 20-120 | exact until frame 9 (first wall contact), panics at frame 9 |

`wall_scene_variant_c_rhs_trace.txt`: `EMERGE_DEBUG_PRESSURE=1` output of the
wall column run. The RHS is zero for 6 substeps, then a 1e-6 roundoff seed grows
about 8 to 10 times per substep up to 6.5e7: the projection amplifies
divergence instead of removing it. Variant C is necessary, not sufficient.

## Suspects removed one at a time (`toggle_matrix.txt`)

`tests/diag_pressure_projection_amplification.rs` runs the gate (a random
velocity field on a real spawned mass distribution must lose divergence under
one projection, and keep losing it under a second) and the three scenes, under
the `EMERGE_P_*` toggles added to `pressure.rs` on this branch.

- The amplifier in free flight is the correction dividing by the nodal mass
  (`1 / mass`) at partially filled surface nodes, while the pressure was solved
  with the average density and only 10 Gauss-Seidel sweeps. Spawned droplet:
  36 of 100 fluid nodes under 0.3 x the average mass. Using one alpha in both
  (`EMERGE_P_CONST_ALPHA`), or solving with the nodal alpha to convergence
  (`EMERGE_P_SURFACE_IN_SOLVE`), keeps the 0.003 g droplet at J = 1.0000 for
  all 37 frames (variant C alone: clamp at frame 37).
- `EMERGE_P_COMPACT` alone and `EMERGE_P_NO_FILTER` alone do not remove it.
  `EMERGE_P_RELAX1` alone makes it far worse (clamp at frame 16, gate grows
  x1.7 to x6.2 on the second pass): the 0.2 relaxation was masking it.
- All five together: interior divergence goes to exactly 0 in one pass and
  stays there.
- Wall contact is a separate mechanism: every configuration but the lone
  `EMERGE_P_RELAX1`, all five included, keeps |J - 1| < 1 % until the first
  wall contact (frame 9), then reaches the clamp by frame 9 to 14. With the surface solved inside the
  system, the column survives 120 frames without a panic, but J stays clamped.
