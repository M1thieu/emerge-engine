# transfer

**Level:** discretization. **Status:** deprecated (explicit solver), also carrying the
adjoint chain that `spacetime/diff` uses.

Particle-to-grid and grid-to-particle transfers (MLS-APIC), and the frictional heat
carried back to the particles. The precise path does not transfer: its unknowns are
coefficients on the reference configuration, read at the points directly (it shares
one constant, the smallest applied temperature rise of `friction_heat`).

**Derives from:** `matter/materials/` (each law's Kirchhoff stress), `spacetime/grid/`.
