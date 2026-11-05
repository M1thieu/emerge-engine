# materials

**Level:** law. **Status:** used by both paths.

What a substance is: how it pushes back on its own deformation, and its measured
properties. Each law sits in the folder of the state of matter it describes
(`solid/`, `liquid/`, `gas/`, `granular/`, `mixture/`). The files at this level are
shared by all of them.

How each path reads a law:
- the explicit solver through the `MaterialModel` trait, in `f32` grid units;
- the precise path through each law's `f64` energy and stress, dispatched in
  `spacetime/total_lagrangian/elastic.rs`. It has these forms for some laws only
  (corotated, neo-Hookean, J2 on Hencky, Ogden, barotropic liquid); it refuses the
  others, with the reason.

**Derives from:** measurements and sources, cited at each law. Nothing here knows about
grids or steps.
