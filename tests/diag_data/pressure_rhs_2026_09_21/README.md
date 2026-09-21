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
wall column run (one substep per frame until contact). The RHS is zero for
frames 1 to 5, a roundoff seed of 1e-6 to 8e-6 appears at frames 6 to 8, then
at the first wall contact (frame 9) a real divergence of 1.3 appears and is
amplified about 8 to 10 times per substep, up to 6.5e7 within that frame.
Variant C is necessary, not sufficient.

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

## Wall contact, all five suspects removed (`wall_contact_matrix.txt`)

Hypothesis checked: the solve does not represent the slip wall, which owns the
nodes within `boundary_thickness` (2) of the domain edge (`apply_slip_wall_velocity`),
while the DCT and Gauss-Seidel put their edges on the padded box (index 0).
`wall_contact_where_j_leaves_one` prints, around the first contact, where J
drifts, how wall-band nodes are classified, and the post-step divergence by
distance to the floor.

- Measured, five suspects removed: at the first contact (frame 9) the 16
  wall-band nodes that gain mass are all light, so the air rule pins them to
  p = 0 as open air; the projection corrects them and the boundary condition
  clips them afterwards. The fluid unknowns next to the wall keep an RMS
  divergence of 2.07 after the step (interior 1e-7 before contact).
- S6 (`EMERGE_P_WALL_IN_SOLVE=2`, wall nodes solid in the solve): those fluid
  unknowns drop to 1e-5, but partly filled fluid nodes pressed against the wall
  are still classified as air by the 0.3 rule and carry the compression
  (RMS 5.8 to 7.9). J drifts more (frame 9: 54 particles off by more than 1 %).
- S7 (`EMERGE_P_AIR_FRACTION=0`, only near-empty nodes are air): contact much
  better (frame 9: 2 particles instead of 21) but still clamps at frame 16.
- S6 and S7 together: J drifts near the left wall at frame 5, before contact;
  panic at frame 50.

Reading: the remaining mechanism is how the solve classifies nodes as fluid,
air or wall (nodal mass thresholds and the domain edge instead of geometry).
Each local patch fixes one symptom and exposes another. The standard answer
replaces all three rules at once: liquid level set from the particles, solid
fractions at the wall's real position, ghost-fluid free surface (apic2d,
Batty et al. 2007, Gibou et al. 2002).
