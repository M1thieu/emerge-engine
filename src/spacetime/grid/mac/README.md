# mac

**Level:** discretization. **Status:** experimental (feature `experimental`); nothing
in the engine calls it.

Pressure projection on a staggered grid for incompressible liquids, built apart
from `Simulation::step` to be judged on its own first. Sources are listed at the
head of `mod.rs`.

Two things here are not what the folder's name says:
- `solid.rs` defines walls as a signed distance, a second definition of the wall
  beside `forces/boundary/`. Unifying them is open.
- Several of `gates/` (`bars`, `disc_slide`, `manufactured`, `vortex`, `scene`) test
  the precise solid step of `spacetime/total_lagrangian/`, for historical reasons.
