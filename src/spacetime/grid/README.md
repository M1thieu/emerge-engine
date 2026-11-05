# grid

**Level:** discretization. **Status:** deprecated (the explicit solver's nodal grid),
except `mac/`, which is experimental and has its own README.

The sparse background grid of the explicit MPM solver: cells, the quadratic
B-spline kernel, multi-field contact (`contact.rs`, Bardenhagen et al. 2001),
material-reported friction at nodes, mixtures. The precise path does not use this
grid: it has its own on the reference configuration
(`spacetime/total_lagrangian/basis.rs`).

**Derives from:** `matter/materials/` (stresses, through `transfer/`), `forces/`
(walls and fields applied at nodes).
