# research/

Variants of the total Lagrangian step kept for comparison or still being built,
compiled in test builds only: the eG(2) time discretization (`galerkin2`),
prescribed boundary motion (`dirichlet`), a one-solve error estimate measured
beside the controller's (`estimate`), and the axisymmetric path's slices
(`axisymmetric`). None of them decides anything on the default path. A variant
either becomes engine code, with its own gates, or is removed.

The modules keep their names in the crate (`total_lagrangian::<name>`), so moving
a file in or out of this folder changes only its `#[path]` line.
