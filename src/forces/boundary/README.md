# boundary

**Level:** law. **Status:** two forms, one per path.

What a wall does.
- `signorini_coulomb.rs`: the contact law (push but never pull, Coulomb friction). The
  precise path enforces it on its own unknowns.
- The `BoundaryCondition` models (`slip.rs`, `static_box.rs`, `heightmap.rs`,
  `kinematic_obstacle.rs`, `friction/`): walls by grid-node label, for the explicit
  solver. A wall's `clamp_particle_position` is a declared non-physical safety net.

The precise path takes the domain walls from here (`domain_walls()`) and turns them
into contact planes. `spacetime/grid/mac/solid.rs` defines walls separately, as a
signed distance: unifying the definitions is open.
