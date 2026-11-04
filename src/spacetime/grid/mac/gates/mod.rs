//! Stop criteria for the MAC projection and the solids built on it, each
//! written before its code and frozen: a criterion moves only by a declared
//! amendment, with its reason.
//!
//! # What is under test
//!
//! The standard formulation (module doc) on the engine's own particle state
//! and units, apart from `Simulation::step`. How it couples to the nodal
//! MLS-MPM grid is the next step, with its own sources read first.
//!
//! # Fixed setup
//!
//! - Gravity is the only thing that changes between runs of the column at
//!   rest and the droplet in free fall: 0.38, 1 and 2.5 times 9.81 m/s^2.
//!   The dam break and the drop into a pool run at 1.
//! - Cells of 1 cm unless stated. No equation of state, no viscosity, no
//!   surface tension: the projection alone holds the fluid.
//! - Frame 1/60 s. Substeps are chosen so a particle moving at the largest
//!   speed, plus what gravity adds over the substep, travels at most one
//!   cell (Zhu and Bridson 4.2.5). Particles keep the velocity they
//!   gather. Positions advance by the midpoint rule through the projected
//!   velocity (Bridson and Muller-Fischer 3.1) instead of the engine's
//!   `x += v dt`, a declared change of setup with its reason at
//!   `Scene::substep`. The criteria below did not move with it.
//! - Four particles per cell, each placed at random inside its quarter of
//!   the cell; those within one cell of a free surface are moved along its
//!   normal to half a cell from it (Zhu and Bridson 4.2.1). Particle radius
//!   in the level set: the particle spacing; kernel radius: twice that
//!   (their section 5).
//! - J is the fluid's own: its logarithm advances by `dt div v` each
//!   substep through `advance_log_volume_ratio`, with the engine's bounds
//!   [0.5, 2.0], `div v` the divergence of the liquid cells at the particle
//!   rather than the trace of the gathered velocity gradient, a declared
//!   change of setup with its reason at `Scene::substep`.
//!   The fluid's volume is `sum(V0 J)`.
//! - Named settings, each with its reason at its definition: the ghost
//!   fraction floor (0.01, as `apic2d`), the conjugate gradient tolerance
//!   and iteration cap, the MIC(0) parameter, the extrapolation depth, the
//!   travel per substep.
//! - Not allowed: a relaxation factor on the correction, a velocity clamp,
//!   damping, pushing particles back out of walls, or a J at its bounds.
//!
//! # Column at rest
//!
//! Water filling a tank 0.40 m wide from wall to wall, 0.30 m deep, open
//! top, at rest, 5 s.
//!
//! - At every frame and every fluid cell of the central column, pressure
//!   within half a cell of head of `g (h - y)`, `h` the fill height.
//! - At 0.5 cm cells, at 1 g, the largest pressure error, in metres of
//!   head, is not larger than at 1 cm.
//! - No particle ever more than one cell from where it started.
//! - The surface, where the level set crosses zero on the central column,
//!   within half a cell of its height at the start.
//!
//! # Droplet in free fall
//!
//! Disc of radius 5 cm, at rest, falling 0.3 s in a domain 1.44 m tall
//! that it never touches, at 2.5 g included. Exact answer: zero pressure,
//! translation at g, unchanged shape.
//!
//! - Largest pressure magnitude at most a hundredth of one cell of head.
//! - Centre-of-mass acceleration over each frame equal to g within 0.1
//!   percent.
//! - Every particle's speed relative to the centre of mass at most 0.1
//!   percent of `g t`.
//! - Radius of gyration within 0.5 percent of its start.
//! - Every J within 0.001 of 1.
//!
//! # Dam break
//!
//! Column 0.20 m wide and 0.40 m high against the left wall of a closed
//! tank 0.64 m by 0.64 m, 2 s.
//!
//! - No NaN, and no particle outside the tank.
//! - Volume, `sum(V0 J)`, within 2 percent of its start at every frame.
//! - Kinetic plus potential energy never above its start by more than 1
//!   percent: the projection only removes kinetic energy (Bridson and
//!   Muller-Fischer eq. 4.41) and still walls do no work.
//! - Recorded, not judged: the front position over time, compared with an
//!   experiment only once its paper has been read, and the number of
//!   particles per interior fluid cell, a volume measure independent of J.
//!
//! # Drop into a pool
//!
//! Disc of radius 5 cm released at rest with its centre 0.40 m above the
//! floor, over water 0.15 m deep filling the same closed tank, 2 s. Same
//! criteria as the dam break.
//!
//! # Each component alone
//!
//! The staggered transfer, the level set, the solid weights, the ghost
//! fluid assembly, the solver and the extrapolation each have their own
//! tests, with no state shared between them. The solver takes a system and
//! returns a solution.
//!
//! # Measured and reported, not judged
//!
//! Substeps per frame, conjugate gradient iterations per substep and their
//! largest value, solves stopped by the iteration cap, and milliseconds per
//! simulated second (debug build, so only for comparison between scenes).
//!
//! # Counting failures
//!
//! A failure is a full run of the scenes in which any criterion above is
//! missed, once the implementation is believed complete. After the first:
//! one diagnosis, one fix, labelled real fix or declared approximation; a
//! crutch is not allowed. After the second: stop and record it in
//! `KNOWN_LIMITATIONS.md`.
//!
//! # Compressible projection, criteria written before its code
//!
//! The generalized Chorin projection of Stomakhin, Schroeder, Jiang, Chai,
//! Teran and Selle, *Augmented MPM for phase-change and varied materials*,
//! 2014, eqs. 14 to 18: `dp/dt = -K div v` taken implicitly with the
//! pressure update, so the pressure system gains `1 / (c^2 dt)` on its
//! diagonal and `q_n / (c^2 dt)` on its right-hand side (`q = p / rho`,
//! `c^2 = K / rho`, `q_n` from the particles' J through the linear
//! equation of state `q = -c^2 (J - 1)`, their eq. 10 without plasticity).
//! As `c` grows the system becomes the incompressible one above. Same
//! rules as above: no clamp, no damping, no relaxation; positions and J as
//! declared above.
//!
//! - Compressible column at rest: the column at rest's tank, 1 cm cells,
//!   1 g, `c = 10 m/s`, 5 s. Pressure within half a cell of head of
//!   `g (h - y)` with `h` the surface height of that frame, from t = 1 s
//!   (amended: a column released uncompressed rings while it settles, and
//!   no initial state met both this criterion at every frame and the
//!   surface one below); the mean J of the particles in each 5-cell band
//!   within 0.005 of `1 - g (h - y) / c^2` over the last second; the
//!   surface, averaged over the last second, lowered from its start by
//!   `g h^2 / (2 c^2)` within 0.15 cell.
//! - Sound speed: a closed tube 200 cells long and 8 high, full, no
//!   gravity, `c = 10 m/s`, the 10 cells at one end starting at J = 0.99,
//!   substeps of `dx / c`. The pressure peak reaches the cell 150 from that
//!   end at `(150 - 5) dx / c` within 5 percent.
//! - Real water: the dam break and the drop into a pool with `K = 2.2e9
//!   Pa`, the same criteria as there, and substeps per frame at most 1.1
//!   times those of the incompressible runs: the real bulk modulus must not
//!   bring back an acoustic time step.
//! - The incompressible scenes keep passing unchanged.
//!
//! Failures counted as above.
//!
//! # Elastic solid on the staggered grid, criteria first
//!
//! Stomakhin et al. 2014's split (their eqs. 7, 8, 13): the shear part of
//! the fixed corotated energy taken on the isochoric deformation,
//! `mu |J^(-1/d) F - R|^2`, gives a deviatoric Kirchhoff stress applied as
//! forces on the faces, explicitly; the volume part `lambda / 2 (J - 1)^2`
//! is the compressible projection above with `c^2 = lambda / rho`. Plane
//! strain, `lambda` and `mu` from `E` and `nu`. Same rules: no clamp, no
//! damping, no relaxation. Since the free-body change below, the volume
//! part acts through the particles instead, with the shear part.
//!
//! - Confined elastic column at rest: the compressible column's tank
//!   filled with the solid, `nu = 0.3`, `E` such that `(lambda + mu) / rho
//!   = (10 m/s)^2`, the uniaxial-strain stiffness of this split in 2D;
//!   starting unstressed, 5 s. Over the last second, the mean J of each
//!   5-cell band within 0.005 of `1 - g (h - y) rho / (lambda + mu)`; no
//!   particle out of the tank; no NaN. Amended with the material matrix
//!   below: that stiffness was the split's own, with `lambda` as its volume
//!   modulus; real plane-strain elasticity has `lambda + 2 mu`, which the
//!   volume part now gives (`Scene::with_lame`), and the expected J is `1 -
//!   g (h - y) rho / (lambda + 2 mu)`, read from the scene's own
//!   coefficients.
//! - Plane shear wave (amended: a free strip 16 cells high timed by the
//!   peak of its velocity measured no speed. A free surface along the strip
//!   forces `sigma_xy = 0`, which no plane shear wave satisfies, so the
//!   strip bends instead, and the peak at 150 cells was 1 to 6 percent of
//!   the kick and moved from 14 percent early to 50 percent late with the
//!   height). Now a free block 100 cells long and 208 high, no gravity,
//!   `nu = 0.3`, `E` such that `c_s = sqrt(mu / rho) = 5 m/s`, the 10
//!   cells at one end starting with an upward velocity `v0` = 0.05 m/s
//!   over the whole height, substeps of `dx / (2 c_s)`. Motion uniform
//!   along y obeys the 1D wave equation for `v_y` at `c_s`, so by
//!   d'Alembert a front of height `v0 / 2` leaves the kicked slab. Within 4
//!   cells of mid-height, a free edge's disturbance needs `(104 - 4) dx /
//!   c_p` (`c_p / c_s = sqrt(2 (1 - nu) / (1 - 2 nu))`, 53 `dx / c_s`) to
//!   arrive, after the window below. The mean vertical velocity of the
//!   particles within one cell of `x = 50` cells from that end, and within
//!   4 cells of mid-height, first reaches `v0 / 4` (half the front) at
//!   `40 dx / c_s` within 5 percent: the front leaves from the slab's inner
//!   edge, 10 cells from the end, 40 cells before the probe.
//! - Elastic block dropped: a 10 by 10 cell block, `nu = 0.3`, `c_s = 5
//!   m/s`, released at rest with its base 20 cells above the floor of the
//!   dam break's closed tank, 1 g, 2 s. Kinetic, potential and elastic
//!   energy never above the start by more than 1 percent; no particle out;
//!   no NaN; at 2 s the block's radius of gyration about its centre within
//!   5 percent of its start (it springs back).
//! - The incompressible and compressible scenes keep passing unchanged.
//!
//! Failures counted as above.
//!
//! Amended: a particle found beyond a wall after its step is moved to its
//! mirror point (`x -> 2 w - x`), the map the walls' image construction
//! implies (`extrapolate::constrain_to_solids`; Zhu and Bridson 4.2.5 push
//! particles back out of solids). Why: a wall is an invariant line of the
//! mirrored flow, where the normal velocity is zero, and where the liquid
//! leaves the wall it is an unstable one. The image continues the same
//! line, `u = a (x - w)`, behind the wall, so a particle put an ulp beyond
//! it by rounding is carried away as `exp(a t)`. Measured: a dam-break
//! particle gliding on the left wall plane went 4 ulps beyond it with the
//! f32 noise of the gathered normal velocity (1.6e-5 cells/s), then 4.45
//! cells into the wall with `a = 26 1/s`. So that the reflection cannot
//! hide a real crossing, every reflection is counted with its depth before
//! it, and every scene with walls requires the deepest at most 4 ulps of
//! the wall's coordinate: one rounding of the position's sum is half an
//! ulp, and the noise step, `1.6e-5 x 3 ms`, a tenth of one, with a
//! declared factor of 4 over that.
//!
//! Amended again, for precision, speed and every material: that budget
//! left out the pressure solve's own tolerance. The solve stops at a
//! residual divergence `r` (1/s), which leaves a face velocity off by about
//! `r dx`, a step of `dt dx r` towards a wall. Measured with MIC(0) on the
//! resting column at 2.5 g: `r` near 4e-4 1/s at substeps of 16.7 ms gives
//! 6.8e-6 cell, 14 ulps at the floor; the crossing was 13.5. The multigrid
//! cycle overshoots the tolerance and had hidden the term. Each
//! reflection's depth is now held to `4 ulp + 4 dt dx r`, `r` the residual
//! that substep's solve actually reached, the same declared factor of 4
//! over both terms. A crossing at a physical velocity stays orders of
//! magnitude above it.
//!
//! # The material matrix, criteria first
//!
//! Every material runs the same scenes, set up and judged from its
//! measured coefficients alone, `rho`, Lame `lambda` and `mu` (plane
//! strain): no scene, tolerance or expected value is written for one
//! material. A row is a material; adding one adds no code. Rows: water
//! (`K = 2.2e9 Pa`, `rho = 1000 kg/m^3`, `mu = 0`), and the elastic part
//! of the very soft and medium clays of `examples/cpu/vonmises_clay_scene`
//! (FHWA NHI-06-088 and NAVFAC DM 7.01: `E = 300 c`, `c = q_u / 2`, `q_u`
//! 20 and 75 kPa, `nu = 0.45`, `rho = 1700 kg/m^3`).
//!
//! - P wave: a free block, no gravity, its first 10 cells kicked along x
//!   at 0.05 m/s over the whole height. The mean x velocity within one
//!   cell of 50 cells from the end, and 4 cells of mid-height, first
//!   reaches a quarter of the kick at `40 dx / c_p` within 5 percent, `c_p
//!   = sqrt((lambda + 2 mu) / rho)` (d'Alembert, as the plane shear wave).
//!   The block is high enough that its free edges' disturbance, at most
//!   `c_p`, arrives after the window.
//! - S wave, for `mu > 0`: the plane shear wave with the row's
//!   coefficients, the height from the row's own `c_p / c_s`; `c_s =
//!   sqrt(mu / rho)`.
//! - Every scene: no NaN, no particle beyond a wall past its budget, and
//!   its wall time per simulated second against real time, reported.
//!
//! Added, criteria first: the scenes at the substep each material runs at
//! in a game, where its cost against real time is the verdict.
//!
//! - Rest: a column 40 cells wide and 30 deep filling a tank of slip
//!   walls, 1 cm cells, released unstressed under 1 g, 2 s. Over the last
//!   second the mean J of each 5-cell band within `0.05 rho g h / M + a T`
//!   of `1 - rho g (h - y) / M`, `M = lambda + 2 mu` (uniaxial strain),
//!   where `a T` is the drift the pressure solve's absolute tolerance `a`
//!   (1/s) allows J over the run's `T` seconds.
//! - Drop: a 10 by 10 cell block of the material released at rest 20
//!   cells above the floor of a closed tank, 1 g, 1 s. Kinetic,
//!   potential, shear and volume energy never above the start by more than
//!   1 percent; for `mu > 0`, the radius of gyration about the centre
//!   within 5 percent of the start at the end (it springs back); for `mu =
//!   0`, the volume within 2 percent (the dam break's bound).
//! - Both: the wall time per simulated second, reported against real time
//!   (1000 ms); no criterion yet, as the explicit shear bounds the substep
//!   by the shear wave (by the P wave since the free-body change below), so
//!   the stiffer the solid the slower.
//!
//! Failures counted as above.
//!
//! # A free body keeps its momentum, criteria first
//!
//! A body with no external force keeps its linear and angular momentum,
//! whatever its material. Measured before this change: a free block of the
//! very soft clay, vibrating, accelerated its centre to 0.27 m/s in 1 s
//! at 1 cm cells and to 0.45 m/s, spinning at 6.3 rad/s, at 0.5 cm, the
//! free surface's pressure giving it a net force (tanks hid it: their walls
//! take any net force).
//!
//! - Free body: every row, a 10 by 10 cell block at the centre of a 72
//!   cell tank, no gravity, never reaching a wall, started with the pure
//!   strain rate `v = s (y - c_y, x - c_x)` whose largest particle speed is
//!   0.5 m/s, the same for every row, at 1 and 0.5 cm cells, 1 s. (Set
//!   before any run: a strain over a sound crossing gave water 37 m/s.) The
//!   centre's velocity stays within 1 percent of the start's largest
//!   particle speed, and the angular momentum about the centre within 1
//!   percent of `I s` (the start's inertia times the strain rate), at every
//!   frame; no NaN; energy never above the start by more than 1 percent.
//! - Every scene above keeps passing.
//!
//! Failures counted as above.
//!
//! Amended: a row with no shear stiffness (`mu = 0`) cannot hold a strain
//! rate, so its block stretched into the walls within the second, whose
//! forces then changed its momentum (the decomposition below did not
//! telescope to zero for water alone). Such a row starts in pure
//! translation instead: the block 4 cells from two walls, moving at 0.5
//! m/s along the tank's diagonal, which keeps it 14 cells from the far
//! walls after 1 s. The bounds are unchanged, the centre's velocity read
//! against its start, the angular momentum's scale `I s` with `s` the
//! largest start speed over the half diagonal.
//!
//! Measured on the way (probes and runs, every change reverted but the
//! last):
//! - The projection changes the faces' momentum by exactly `sum_f (m_f -
//!   rho V_f) du_f`, `V_f` the volume the kinematic update assumes of face
//!   `f`. On the free very soft clay the terms came from the bulk's sampled
//!   mass, from the surface's kernel-spread mass against the level set's
//!   `theta`, and from the faces past the surface that take the continued
//!   pressure.
//! - The surface's `theta` from the face mass and no continued pressure:
//!   the clays' drift unchanged, the column at rest broke.
//! - The solved pressure applied as each particle's stress: the clay's
//!   centre held, water lost about 10 percent of its volume, the solved
//!   pressure belonging to the MAC operator, not to that one.
//! - Newton's law on every face, `du = dt F / m_f`. A resting liquid
//!   convected, as Ding, Shinar and Schroeder (MAC APIC, JCP 408, 2020,
//!   2.4.3) report of grid masses in the projection, "an initially
//!   stationary pool of water would develop currents in it", and the clays
//!   blew up on faces of vanishing mass.
//! - The power weights of Qu, Li, de Goes and Jiang (The Power
//!   Particle-In-Cell Method, TOG 41(4) 118, 2022), face masses exact to
//!   the transport plan's tolerance. Water clean, the clay still drifted: a
//!   compressed solid's density really varies, `rho0 / J`, which a
//!   density-blind projection cannot conserve.
//!
//! Result: a solid is not projected. Its whole elastic stress, the shear
//! part and the volume part `K J (J - 1) I` with `J = det F` and `K =
//! lambda + mu` the plane-strain bulk modulus, acts through its particles
//! (`transfer::stress_to_faces`, the engine's MPM force). The kernel's
//! gradients sum to zero over the faces, so a particle's stress moves no
//! momentum in or out whatever the stress or the sampling, and a symmetric
//! stress no angular momentum. Liquids keep the projection. A solid's
//! substep follows its P wave, `c_p^2 = (K + mu) / rho`: 3.3 times the
//! substeps on the matrix's clays (`nu = 0.45`, `c_p / c_s = sqrt(11)`). A
//! face on a wall plane takes no normal velocity, the grid's own wall
//! condition, which the projection's variational form held before. The
//! free body then passes, the centre at 0.0000 of the start speed and the
//! angular momentum within 0.0005 of `I s` (0.0004 before solids were
//! seeded without the surface pull, `seed_body`), and every scene above
//! passes with the plane shear wave's substep amended below. Lost against
//! the split: the clays' P wave arrives 4.2 percent late, 1.1 with the
//! split.
//!
//! Measured since, not a criterion: the free body's energy bound catches a
//! rise only, and the explicit path loses most of the free block's energy
//! within the second (95 to 99.95 percent, the conserving step below). The
//! conserving step has its own gate with the bound either way
//! (`gate_conserving_free_vibration`).
//!
//! Amended with this change: the plane shear wave fixed its substep at
//! half a cell per shear wave crossing, which a solid carrying its P wave
//! crosses at 0.94 cell (limit 0.577), and blew up. Its substep now follows
//! the faster wave, as the matrix's plane waves set it; the arrival is 3.3
//! percent late (3.7 with the split).
//!
//! # A solid keeps its energy at large substeps, criteria first
//!
//! Real time for a solid needs substeps far longer than the explicit P
//! wave limit, 273 to 529 per frame on the matrix's clays at 1 cm, and no
//! loss of precision means no numerical damping. Measured before this
//! change: the free body's clay block ends its second with 95 to 99.95
//! percent less energy, all of it taken by the method
//! (`probe_free_body_frames`: 99.7 and 99.95 percent at 1 cm for the very
//! soft and medium clays, 95 at 0.5 cm for the very soft one). Love and
//! Sulsky (An Unconditionally Stable, Energy-Momentum Consistent
//! Implementation of the Material-Point Method, HPC@UNM technical report
//! 2004-002, the report of CMAME 195, 2006) prove both the cause and the
//! remedy for a hyperelastic solid. A lumped mass loses
//! `1/2 dv . (M_lumped - M) dv` each step (their eq. 68), and a transfer
//! that overwrites the particles' velocity loses the velocity it cannot
//! carry (their eq. 49).
//! With the consistent mass `M_AB = sum_p m_p N_A N_B` (eq. 24), a
//! particle update that adds only the grid's change (eq. 32), and an
//! implicit midpoint step on the grid whose stress is the discrete
//! gradient of the stored energy (eqs. 29, 54, 56), energy and both
//! momenta are kept exactly, at any substep.
//!
//! - Free vibration: every solid row, the free body's block with its
//!   strain start, no gravity, 1 s, at 1 and 0.5 cm, at three substeps: the
//!   explicit limit `0.5 dx / c_p`, four times it, and one frame (1/60 s,
//!   or the travel limit if shorter). The total energy, kinetic and
//!   stored, within 1 percent of the start at every frame; the free body's
//!   momentum bounds; no NaN.
//! - Bounce: the matrix's drop for every solid row, 2 s, at one frame's
//!   substep. The total energy, kinetic, gravitational and stored, within 1
//!   percent of the start at every frame; the radius of gyration back
//!   within 5 percent at the end (the drop's bound); nothing out of the
//!   tank.
//! - Rest: the matrix's rest for every solid row at one frame's substep,
//!   within its bounds.
//! - Waves: the matrix's P and S waves keep their bounds at the substeps
//!   the matrix sets. Past one crossing per substep a step cannot carry the
//!   wave, so there only the energy bound above holds.
//! - Every scene above keeps passing.
//! - Cost: the wall time per simulated second of the matrix's drop for
//!   each solid row, against real time and against the explicit run before
//!   this change (16.5 and 32.0 s per simulated second at 1 cm, release).
//!   Reported, not a criterion: reaching real time is the next stage, kept
//!   apart so that a loss of energy and a loss of speed are never confused.
//!
//! Amendments after the bounce first failed (both clays NaN at their
//! impact), each measured before the next run (`gate_conserving_bounce`):
//! - Walls: the tank holds a face on or beyond a wall's plane while the
//!   force the wall exerts on it is compressive and lets it go once it
//!   would pull, the contact condition itself (no penetration, no pull;
//!   Bardenhagen et al. 2001, `total_lagrangian::step`). Held faces keep the
//!   step's energy balance exactly: the drop's measured loss equalled the
//!   mass stabilisation's own cost to five digits.
//! - Mass stabilisation: the extended B-splines (`conserving`'s grid) in
//!   place of the ghost penalty, which took 26 to 46 percent of the drop's
//!   energy through its impacts (measured, `gamma` 0.05 down to 0.002); the
//!   extended basis adds nothing to the mass and took none.
//! - The energy bound is two-sided, as written: `|E / E_0 - 1|` within 1
//!   percent at every frame.
//!
//! Measured since, not an amendment: at one frame per substep the drop
//! keeps its energy to 1.5e-6 and bounces when the block reaches the floor
//! at about 0.66 m/s (1.1 cells per frame at 1 cm), and breaks at its
//! first contact at 2 m/s (3.3 cells per frame) for every stiffness from 3
//! MPa to 10 GPa (`probe_conserving_spread`): a step that carries the block
//! further than about a cell into its contact has no admissible solution
//! on the grid. The bounce at the drop's 2 m/s is therefore open at one
//! frame per substep; the contact refinement (`Scene::contact_substep`), a
//! declared stepping stone, resolves it at the P wave step.
//!
//! Failures counted as above.

use std::time::Instant;

use glam::{IVec2, Mat2, Vec2};

use super::elastic::{deformation_step, shear_energy, shear_kirchhoff};
use super::extrapolate::fold_onto_images;
use super::field::{Field2, MacLayout};
use super::level_set::{SurfaceSettings, liquid_phi};
use super::solid::{
    FaceWeights, box_container, box_container_image, face_weights, sample_centres, sample_corners,
};
use super::transfer::{faces_to_particles, particles_to_faces, stress_to_faces};
use super::{Liquid, ProjectionSettings, Walls, project, travel_limited_dt};
use super::{elastic, extrapolate, field, pcg, pressure};
use crate::diagnostics::{OCCUPANCY_BANDS, scene_map};
use crate::fields::EARTH_GRAVITY_M_S2;
use crate::matter::kinematics::volume_ratio::advance_log_volume_ratio;
use crate::particle::{Particle, Particles};
use crate::solver::LcgRng;

const FRAME: f32 = 1.0 / 60.0;
/// The fluid material's own J bounds (`NewtonianFluidMaterial::update_particle`).
const J_BOUNDS: (f32, f32) = (0.5, 2.0);
const GRAVITIES: [f32; 3] = [0.38, 1.0, 2.5];

/// Particles and the still tank they sit in, stepped by the standard
/// projection alone. Lengths are in cells (`dx = 1`); `cell_m` gives the
/// metres per cell.
struct Scene {
    layout: MacLayout,
    cell_m: f32,
    /// Cells per second squared.
    gravity: Vec2,
    tank: (Vec2, Vec2),
    weights: FaceWeights,
    solid_centres: Field2,
    surface: SurfaceSettings,
    settings: ProjectionSettings,
    x: Vec<Vec2>,
    v: Vec<Vec2>,
    c: Vec<Mat2>,
    mass: Vec<f32>,
    v0: Vec<f32>,
    log_j: Vec<f32>,
    j_at_bounds: usize,
    time: f32,
    probe: Probe,
    /// `c^2` in cells^2/s^2 for the compressible projection;
    /// `None` is the incompressible limit.
    sound_speed2: Option<f32>,
    /// A substep length that replaces the travel limit (the sound-speed
    /// tube).
    fixed_dt: Option<f32>,
    /// `mu / rho` in cells^2/s^2 for an elastic solid; `None`
    /// for a liquid.
    shear: Option<f32>,
    /// A solid moves by the energy-momentum conserving implicit step
    /// (`conserving`) instead of the explicit one.
    conserving: Option<crate::spacetime::total_lagrangian::ConservingSettings>,
    /// One material per particle for the conserving step, in place of the
    /// scene's one (a probe's several bodies).
    solids: Option<Vec<crate::spacetime::total_lagrangian::Elastic>>,
    /// The conserving step's contact refinement, a declared stepping stone
    /// (`contact_substep`): its substep in P wave steps, `0.5 dx / c_p`;
    /// set by a probe.
    refine_contact: Option<f32>,
    /// Each particle's deformation gradient, for the shear stress.
    deformation: Vec<Mat2>,
    /// The conserving step's reference configuration (taken at its first
    /// step) and its grid, kept across steps (`conserving_substep`).
    lagrangian: Option<(
        Vec<glam::DVec2>,
        Option<crate::spacetime::total_lagrangian::LagrangianCache>,
    )>,
    /// The first of the bodies' contact points (`with_contact_outline`):
    /// only they meet the walls. `None`: every particle does.
    contact_from: Option<usize>,
}

/// Where things happen, for diagnosis only: no criterion reads it.
#[derive(Default)]
struct Probe {
    /// J reaching its bounds, by where the particle was: in a liquid cell
    /// of the tank, in an air cell of the tank, inside a wall.
    j_at_bounds_by_place: [usize; 3],
    /// Particles moved back from beyond a wall (`substep`), and the
    /// deepest of them before the move, in ulps of the wall's coordinate.
    reflections: u32,
    deepest_reflection_ulps: f32,
    /// The largest reflection's depth over its budget (criteria, second
    /// amendment), and the budget's solver term of the current substep,
    /// `4 dt dx r`.
    worst_reflection_ratio: f32,
    solve_drift: f32,
    /// Particles leaving the tank: time, position before, velocity,
    /// distance to the nearest wall before.
    crossings: Vec<(f32, Vec2, Vec2, f32)>,
    /// Wall-clock time per phase of the substep, in microseconds: P2G and
    /// gravity, level set, assembly for the frame record, projection
    /// (assembly, solve, update, extrapolation, walls), the two gathers,
    /// the particle update.
    phase_us: [u128; 6],
    /// Cell whose pressure every substep records, with the time (the
    /// sound-speed tube).
    watch: Option<usize>,
    watch_history: Vec<(f32, f32)>,
    /// Particles within one cell of this x and between these two heights:
    /// their mean vertical velocity every substep (the plane shear wave).
    watch_x: Option<(f32, f32, f32)>,
    /// The watch reads the x velocity instead of y (a P wave along x).
    watch_longitudinal: bool,
    watch_x_history: Vec<(f32, f32)>,
    /// Where and when the deepest reflection happened, and the particle's
    /// position before the step that took it there.
    deepest_reflection_at: Option<(f32, Vec2)>,
    /// The conserving step's implicit steps: Newton and GMRES iterations
    /// summed, the most GMRES iterations in one step, the worst final residual
    /// and the steps that did not converge.
    conserving: (u64, u64, u32, f64, u32),
    /// Conserving steps rejected and halved (step-size control).
    rejected: u32,
    /// The stabilised mass's energy loss summed over the conserving steps.
    stabilisation_loss: f64,
    /// The worst relative residual of the conserving steps' projection.
    worst_projection: f64,
    /// The last conserving step's energy ledger and work
    /// (`total_lagrangian::StepReport`).
    ledger: (f64, f64, f64),
    work: (f64, f64),
    /// The most particles the conserving step held on a wall's plane at
    /// once.
    contacts: usize,
    /// The conserving steps' wall time by phase (`total_lagrangian::StepReport`)
    /// and how many steps.
    phases: [f64; 5],
    steps: u32,
    /// The conserving steps' solves summed (the walls' passes).
    passes: u64,
    /// The local error estimate of Zienkiewicz and Xie (1991), as Vetter,
    /// Wittel, Stoop and Herrmann write it for the Newmark family (arXiv
    /// 1111.5128, eq. 64), `|beta - 1/6| dt^2 |a_{n+1} - a_n|_inf / u_ref`,
    /// here with the midpoint's `beta = 1/4` and each step's average
    /// acceleration `(v_{n+1} - v_n) / dt` per particle in place of their
    /// predicted one, an adaptation of this engine's. Measured when
    /// `estimator_reference` (`u_ref`, cells) is positive: the last step's
    /// accelerations and the largest estimate since it was last reset.
    estimator_reference: f32,
    last_acceleration: Vec<Vec2>,
    largest_estimate: f32,
    /// The bodies whose mass conditioning each conserving step measures
    /// before it moves (`total_lagrangian::mass_conditioning`): the union's and
    /// each body's own, the smallest since last reset; empty, none.
    conditioning_bodies: Vec<std::ops::Range<usize>>,
    conditioning: (f64, Vec<f64>),
    /// Below this conditioning a body's basis is dissected
    /// (`total_lagrangian::basis_report`), the worst since last reset kept with
    /// its conditioning; zero, never.
    basis_threshold: f64,
    basis_worst: Option<(f64, String)>,
    /// Since last reset: the largest grid velocity of a projection over the
    /// fastest particle's speed, and Newton's worst start
    /// (`total_lagrangian::StepReport`).
    grid_speed_ratio: f64,
    start_residual: f64,
}

/// What one frame's last substep left, and what the frame cost.
struct Frame {
    pressure: Vec<f32>,
    active: Vec<bool>,
    phi: Field2,
    /// Face velocity the last substep's particles gathered from.
    vel: super::field::MacVelocity,
    substeps: u32,
    iterations: Vec<u32>,
    capped: u32,
    largest_travel: f32,
}

impl Scene {
    fn new(
        (nx, ny): (usize, usize),
        cell_m: f32,
        g_fraction: f32,
        tank: (Vec2, Vec2),
        x: Vec<Vec2>,
    ) -> Self {
        let layout = MacLayout::new(nx, ny, 1.0);
        let solid = box_container(tank.0, tank.1);
        let corners = sample_corners(&layout, solid);
        let spacing = 0.5 * layout.dx;
        let n = x.len();
        Self {
            layout,
            cell_m,
            gravity: Vec2::new(0.0, -g_fraction * EARTH_GRAVITY_M_S2 / cell_m),
            tank,
            weights: face_weights(&layout, &corners),
            solid_centres: sample_centres(&layout, solid),
            surface: SurfaceSettings::from_spacing(spacing),
            settings: ProjectionSettings::default(),
            x,
            v: vec![Vec2::ZERO; n],
            c: vec![Mat2::ZERO; n],
            mass: vec![spacing * spacing; n],
            v0: vec![spacing * spacing; n],
            log_j: vec![0.0; n],
            j_at_bounds: 0,
            time: 0.0,
            probe: Probe::default(),
            sound_speed2: None,
            fixed_dt: None,
            shear: None,
            conserving: None,
            solids: None,
            refine_contact: None,
            deformation: vec![Mat2::IDENTITY; n],
            lagrangian: None,
            contact_from: None,
        }
    }

    fn g(&self) -> f32 {
        self.gravity.length()
    }

    /// A plane-strain elastic solid of Young's modulus over density
    /// `e_over_rho` (m^2/s^2) and Poisson's ratio `nu` (`with_lame`).
    fn with_elastic(self, e_over_rho: f32, nu: f32) -> Self {
        let mu = e_over_rho / (2.0 * (1.0 + nu));
        let lambda = e_over_rho * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        self.with_lame(lambda, mu)
    }

    /// A material by its plane-strain Lame coefficients over its density,
    /// `lambda / rho` and `mu / rho` in m^2/s^2. With `mu > 0` a solid: its
    /// whole stress, shear and volume, through its particles (module doc:
    /// a solid is not projected);
    /// with `mu = 0` a liquid, its volume in the compressible projection.
    /// Adds contact points of no mass and no volume on the bodies'
    /// surface, which alone meet the walls in the conserving step. A
    /// contact holds a body's surface, as the node-to-surface contact of
    /// Laursen and Chawla, IJNME 40 (1997), sec. 3, holds its slave
    /// surface's nodes: a particle inside the body held where it crossed a
    /// wall would be held there while its neighbours move (measured: the
    /// very soft clay's second, corner-first impact blew up with every
    /// particle held).
    fn with_contact_outline(mut self, points: Vec<Vec2>) -> Self {
        self.contact_from = Some(self.x.len());
        for q in points {
            self.x.push(q);
            self.v.push(Vec2::ZERO);
            self.c.push(Mat2::ZERO);
            self.mass.push(0.0);
            self.v0.push(0.0);
            self.log_j.push(0.0);
            self.deformation.push(Mat2::IDENTITY);
        }
        self
    }

    fn with_lame(mut self, lambda_over_rho: f32, mu_over_rho: f32) -> Self {
        let cells2 = 1.0 / (self.cell_m * self.cell_m);
        self.shear = (mu_over_rho > 0.0).then_some(mu_over_rho * cells2);
        // The shear part acts on the isochoric deformation only, so all
        // the volume stiffness is the volume part's (the particles' for a
        // solid, the projection's for a liquid): the
        // plane-strain bulk modulus `kappa = lambda + mu`. With it, `2 mu dev(e) + kappa tr(e)
        // I = 2 mu e + lambda tr(e) I`, Lame's law; with `lambda` alone, as
        // Stomakhin et al.'s split writes the volume energy, the P wave ran
        // at `sqrt((lambda + mu) / rho)`: 6.0 percent late on the clays of
        // the matrix (`nu = 0.45`), against water's 1.0.
        self.sound_speed2 = Some((lambda_over_rho + mu_over_rho) * cells2);
        // Probe switch: a solid moves by the conserving implicit step, at
        // one frame per substep (`frame`) or at the explicit P wave limit
        // (`p`), when `EMERGE_PROBE_CONSERVING` names one.
        if let (Some(mu), Ok(step)) = (self.shear, std::env::var("EMERGE_PROBE_CONSERVING")) {
            let defaults = crate::spacetime::total_lagrangian::ConservingSettings::default();
            self.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
                occupation: std::env::var("EMERGE_PROBE_OCCUPATION")
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .or(defaults.occupation),
                ..defaults
            });
            let c_p = (self.sound_speed2.unwrap_or(0.0) + mu).sqrt();
            let step = Substep::parse(&step).unwrap_or(Substep::Frame);
            self.fixed_dt = Some(step.length(self.layout.dx, c_p));
        }
        self
    }

    /// The compressible projection at `c_m_s`.
    fn with_sound_speed(mut self, c_m_s: f32) -> Self {
        let c = c_m_s / self.cell_m;
        self.sound_speed2 = Some(c * c);
        self
    }

    /// Mass-weighted J at the cell centres, bilinear, and the pressure the
    /// linear equation of state gives it, `q = -c^2 (J - 1)`; zero where no
    /// particle reaches.
    fn cell_pressure_from_j(&self, sound_speed2: f32) -> Vec<f32> {
        let (nx, ny) = (self.layout.nx, self.layout.ny);
        let mut sum_j = vec![0.0f32; nx * ny];
        let mut sum_w = vec![0.0f32; nx * ny];
        for p in 0..self.x.len() {
            let t = self.x[p] / self.layout.dx - Vec2::splat(0.5);
            let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
            let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
            let j_p = self.j(p);
            for (di, dj, w) in [
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let (i, j) = (i0 + di, j0 + dj);
                if i < 0 || j < 0 || i as usize >= nx || j as usize >= ny {
                    continue;
                }
                let c = i as usize + nx * j as usize;
                sum_w[c] += w * self.mass[p];
                sum_j[c] += w * self.mass[p] * j_p;
            }
        }
        (0..nx * ny)
            .map(|c| {
                if sum_w[c] > 0.0 {
                    -sound_speed2 * (sum_j[c] / sum_w[c] - 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The liquid level set with the tank's walls folded in (`apic2d`).
    fn phi(&self) -> Field2 {
        let mut phi = liquid_phi(&self.layout, &self.x, &self.surface);
        for (value, &solid) in phi.data_mut().iter_mut().zip(self.solid_centres.data()) {
            *value = value.min(solid);
        }
        phi
    }

    fn substep(&mut self, dt: f32, frame: &mut Frame) {
        if let (Some(mu), Some(settings)) = (self.shear, self.conserving) {
            self.conserving_substep(mu, settings, dt, frame);
            return;
        }
        let mut clock = Instant::now();
        let mut lap = |probe: &mut Probe, k: usize| {
            probe.phase_us[k] += clock.elapsed().as_micros();
            clock = Instant::now();
        };
        let (mut vel, face_mass) =
            particles_to_faces(&self.layout, &self.x, &self.v, &self.c, &self.mass);
        if let Some(mu) = self.shear {
            // The whole elastic stress on the particles (module doc, a solid is
            // not projected): the shear part on the isochoric deformation and
            // the volume part `K J (J - 1) I`, `J = det F`, with `K` the
            // plane-strain bulk modulus `with_lame` keeps.
            let bulk = self.sound_speed2.unwrap_or(0.0);
            let tau_v0: Vec<Mat2> = (0..self.x.len())
                .map(|p| {
                    let f = self.deformation[p];
                    let j = f.determinant();
                    let volume = bulk * j * (j - 1.0) * Mat2::IDENTITY;
                    self.v0[p] * (shear_kirchhoff(f, mu) + volume)
                })
                .collect();
            let mut force = stress_to_faces(&self.layout, &self.x, &tau_v0);
            let mut lent = super::field::MacVelocity {
                u: face_mass.u.clone(),
                v: face_mass.v.clone(),
            };
            let image = box_container_image(self.tank.0, self.tank.1);
            fold_onto_images(&self.layout, &mut force, &self.weights, &image, true);
            fold_onto_images(&self.layout, &mut lent, &self.weights, &image, false);
            for (field, f, m) in [
                (&mut vel.u, &force.u, &lent.u),
                (&mut vel.v, &force.v, &lent.v),
            ] {
                for ((value, &f), &m) in field.data_mut().iter_mut().zip(f.data()).zip(m.data()) {
                    if m > 0.0 {
                        *value += dt * f / m;
                    }
                }
            }
        }
        for value in vel.u.data_mut() {
            *value += self.gravity.x * dt;
        }
        for value in vel.v.data_mut() {
            *value += self.gravity.y * dt;
        }
        lap(&mut self.probe, 0);
        let phi = self.phi();
        lap(&mut self.probe, 1);
        let system = super::pressure::assemble(
            &self.layout,
            dt,
            &vel,
            &self.weights,
            super::pressure::Surface {
                phi: &phi,
                theta_floor: self.settings.theta_floor,
            },
        );
        lap(&mut self.probe, 2);
        let image = box_container_image(self.tank.0, self.tank.1);
        let solution = if self.shear.is_some() {
            let mut valid = super::field::FaceFlags {
                u: face_mass.u.data().iter().map(|&m| m > 0.0).collect(),
                v: face_mass.v.data().iter().map(|&m| m > 0.0).collect(),
            };
            // A solid is not projected (module doc). The wall
            // condition the projection's variational form holds (`pressure`
            // module doc) is then the grid's own, as standard MPM sets it:
            // a face on a wall plane has no normal velocity and keeps none
            // through the extension. Without it a settling column crossed
            // the floor by 3704 ulps in its first frame.
            let (lo, hi) = self.tank;
            let on_plane = |a: f32, b: f32| (a - b).abs() <= 1e-6 * self.layout.dx;
            for j in 0..self.layout.ny {
                for i in 0..=self.layout.nx {
                    let q = self.layout.u_position(i, j);
                    if on_plane(q.x, lo.x) || on_plane(q.x, hi.x) {
                        vel.u.set(i, j, 0.0);
                        valid.u[vel.u.index(i, j)] = true;
                    }
                }
            }
            for j in 0..=self.layout.ny {
                for i in 0..self.layout.nx {
                    let q = self.layout.v_position(i, j);
                    if on_plane(q.y, lo.y) || on_plane(q.y, hi.y) {
                        vel.v.set(i, j, 0.0);
                        valid.v[vel.v.index(i, j)] = true;
                    }
                }
            }
            super::extrapolate::extrapolate_velocity(
                &mut vel,
                &mut valid,
                self.settings.extrapolation_layers,
            );
            super::extrapolate::constrain_to_solids(&self.layout, &mut vel, &self.weights, &image);
            super::pcg::Solution {
                pressure: vec![0.0; self.layout.nx * self.layout.ny],
                iterations: 0,
                residual: 0.0,
                converged: true,
            }
        } else {
            let q_before = self
                .sound_speed2
                .map(|c2| (c2, self.cell_pressure_from_j(c2)));
            let material: Vec<bool> = self.solid_centres.data().iter().map(|&d| d > 0.0).collect();
            project(
                &self.layout,
                dt,
                &mut vel,
                Walls {
                    weights: &self.weights,
                    image: &image,
                    open: &material,
                },
                Liquid {
                    phi: &phi,
                    face_mass: &face_mass,
                },
                &self.settings,
                q_before
                    .as_ref()
                    .map(|(c2, q)| (*c2, q.as_slice(), material.as_slice())),
            )
        };
        lap(&mut self.probe, 3);
        faces_to_particles(&self.layout, &vel, &self.x, &mut self.v, &mut self.c);
        // Positions advance by the midpoint rule (RK2) through the projected
        // face velocity, as Bridson and Muller-Fischer 3.1 recommend over
        // forward Euler for trajectories. Against a wall the normal velocity
        // falls linearly to zero, `v = -a d`: forward Euler moves `d` to `d (1
        // - a dt)`, which crosses the wall once `a dt > 1`, while the midpoint
        // rule gives `d (1 - a dt + (a dt)^2 / 2)`, positive for every step.
        // After the corner fix, 71 of the 72 remaining crossings were water
        // decelerating against the lid within one cell. The particle keeps
        // the velocity and `C` gathered at its start, as before.
        let midpoint: Vec<Vec2> = (0..self.x.len())
            .map(|p| self.x[p] + 0.5 * dt * self.v[p])
            .collect();
        let mut v_mid = vec![Vec2::ZERO; self.x.len()];
        let mut c_mid = vec![Mat2::ZERO; self.x.len()];
        faces_to_particles(&self.layout, &vel, &midpoint, &mut v_mid, &mut c_mid);
        lap(&mut self.probe, 4);
        if self.shear.is_some() {
            for p in 0..self.x.len() {
                self.deformation[p] = deformation_step(dt * self.c[p]) * self.deformation[p];
            }
        }
        if let Some((x0, y_lo, y_hi)) = self.probe.watch_x {
            let (sum, n) = (0..self.x.len())
                .filter(|&p| {
                    (self.x[p].x - x0).abs() < self.layout.dx && (y_lo..y_hi).contains(&self.x[p].y)
                })
                .fold((0.0f32, 0u32), |(s, n), p| {
                    let v = if self.probe.watch_longitudinal {
                        self.v[p].x
                    } else {
                        self.v[p].y
                    };
                    (s + v, n + 1)
                });
            self.probe
                .watch_x_history
                .push((self.time + dt, sum / n.max(1) as f32));
        }
        let ln_bounds = (J_BOUNDS.0.ln(), J_BOUNDS.1.ln());
        // J advances by the divergence the projection holds: the cells'
        // divergence interpolated bilinearly at the particle, over liquid
        // cells only. `tr C` reads the same in the bulk (the bulk divergence
        // probe: 1.32e-5 against 1.31e-5 1/s), but near the free surface the
        // gather reaches the velocity extrapolated into the air, which no
        // solve made divergence free (Bridson and Muller-Fischer 6.3's
        // constant extension is not either), and J drifted 5 % there while
        // the particles per interior cell, a volume measure that does not
        // read J, held at 4.01. A particle with no liquid cell around it is
        // in free flight and reads zero.
        let (nx, ny) = (self.layout.nx as i32, self.layout.ny as i32);
        let liquid = |i: i32, j: i32| {
            i >= 0
                && j >= 0
                && i < nx
                && j < ny
                && phi.get(i as usize, j as usize) < 0.0
                && self.solid_centres.get(i as usize, j as usize) > 0.0
        };
        let cell_div = |i: i32, j: i32| {
            let (i, j) = (i as usize, j as usize);
            (vel.u.get(i + 1, j) - vel.u.get(i, j) + vel.v.get(i, j + 1) - vel.v.get(i, j))
                / self.layout.dx
        };
        for (p, v_step) in v_mid.iter().enumerate() {
            let dt_div = {
                let t = self.x[p] / self.layout.dx - Vec2::splat(0.5);
                let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
                let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
                let (mut sum, mut weight) = (0.0f32, 0.0f32);
                for (di, dj, w) in [
                    (0, 0, (1.0 - fx) * (1.0 - fy)),
                    (1, 0, fx * (1.0 - fy)),
                    (0, 1, (1.0 - fx) * fy),
                    (1, 1, fx * fy),
                ] {
                    if liquid(i0 + di, j0 + dj) {
                        sum += w * cell_div(i0 + di, j0 + dj);
                        weight += w;
                    }
                }
                if weight > 0.0 { dt * sum / weight } else { 0.0 }
            };
            let advanced = self.log_j[p] + dt_div;
            let step = *v_step * dt;
            let was_inside = self.inside_tank(p);
            if advanced <= ln_bounds.0 || advanced >= ln_bounds.1 {
                self.j_at_bounds += 1;
                let cell = self.x[p].floor().as_ivec2();
                let wet = phi.get_signed(cell.x, cell.y).is_some_and(|f| f < 0.0);
                let place = if !was_inside {
                    2
                } else if wet {
                    0
                } else {
                    1
                };
                self.probe.j_at_bounds_by_place[place] += 1;
            }
            self.log_j[p] =
                advance_log_volume_ratio(self.log_j[p], dt_div, J_BOUNDS.0, J_BOUNDS.1).0;
            frame.largest_travel = frame.largest_travel.max(step.length());
            let before = self.x[p];
            self.x[p] += step;
            if was_inside && !self.inside_tank(p) {
                let (lo, hi) = (before - self.tank.0, self.tank.1 - before);
                let wall = lo.x.min(lo.y).min(hi.x).min(hi.y);
                self.probe
                    .crossings
                    .push((self.time, before, self.v[p], wall));
            }
        }
        self.probe.solve_drift = 4.0 * dt * self.layout.dx * solution.residual;
        for p in 0..self.x.len() {
            self.reflect_off_walls(p);
        }
        lap(&mut self.probe, 5);
        self.time += dt;
        frame.substeps += 1;
        frame.iterations.push(solution.iterations);
        frame.capped += u32::from(!solution.converged);
        if let Some(c) = self.probe.watch {
            self.probe
                .watch_history
                .push((self.time, solution.pressure[c]));
        }
        frame.pressure = solution.pressure;
        frame.active = system.active;
        frame.phi = phi;
        frame.vel = vel;
    }

    /// Fixed substeps (`fixed_dt`) until `t_end`, for scenes shorter than
    /// a frame.
    fn run_until(&mut self, t_end: f32) {
        let dt = self.fixed_dt.expect("run_until needs a fixed substep");
        let mut frame = Frame {
            pressure: Vec::new(),
            active: Vec::new(),
            phi: self.layout.cells(0.0),
            vel: super::field::MacVelocity::zeros(&self.layout),
            substeps: 0,
            iterations: Vec::new(),
            capped: 0,
            largest_travel: 0.0,
        };
        while self.time < t_end {
            self.substep(dt.min(t_end - self.time).max(1e-9), &mut frame);
        }
    }

    fn frame(&mut self) -> Frame {
        let mut frame = Frame {
            pressure: Vec::new(),
            active: Vec::new(),
            phi: self.layout.cells(0.0),
            vel: super::field::MacVelocity::zeros(&self.layout),
            substeps: 0,
            iterations: Vec::new(),
            capped: 0,
            largest_travel: 0.0,
        };
        let mut left = FRAME;
        let contact = self.contact_substep();
        while left > 1e-7 {
            let speed = self.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            // Half a cell per crossing of the fastest wave a solid's
            // particles carry, the P wave `c_p^2 = (K + mu) / rho`, under
            // the explicit limit `2 / sqrt(12) = 0.577` the transfer's
            // stiffest mode allows (`transfer::the_stiffest_mode_stays_
            // bounded_...`).
            let shear_limit = match (self.shear, self.conserving) {
                (Some(mu), None) => {
                    0.5 * self.layout.dx / (mu + self.sound_speed2.unwrap_or(0.0)).sqrt()
                }
                _ => f32::INFINITY,
            };
            let limit = contact.or(self.fixed_dt).unwrap_or_else(|| {
                shear_limit.min(travel_limited_dt(
                    speed,
                    self.g(),
                    self.layout.dx,
                    self.settings.max_cells_per_substep,
                ))
            });
            let dt = limit.min(left);
            self.substep(dt, &mut frame);
            left -= dt;
        }
        frame
    }

    /// The conserving step's contact refinement, a stepping stone toward
    /// resolving impacts only where they happen, declared as such: a frame
    /// during which any particle is, or would come, within reach of a wall
    /// moves at the P wave step `0.5 dx / c_p` instead of one step. An impact
    /// lasts about `2 L / c_p` (Love and Sulsky's contact time, their section
    /// 4.1), far under a frame, and the elastodynamic contact problem keeps its
    /// energy through it (Lebeau and Schatzman 1984, as Dabaghi et al., arXiv
    /// 1601.00778, theorem 2.1, state it), which one frame-long step cannot
    /// reproduce (measured: a body's centre 10 mm off the resolved one after a
    /// 0.66 m/s impact, and 76 percent of the energy lost at a 2 m/s one with
    /// the contact held at the step's end). The reach is the quadratic
    /// B-spline's support, 1.5 cells, plus the half cell a particle stands for
    /// (the region BFEMP, arXiv 2108.03349, notes a particle represents); the
    /// prediction is each particle's own trajectory over the frame with its
    /// velocity and the weight, the linear motion continuous collision
    /// detection assumes over a step. Its cost: a body resting on a wall stays
    /// refined. `None` keeps the scene's own substep.
    fn contact_substep(&self) -> Option<f32> {
        let multiple = self.refine_contact?;
        let (mu, _) = (self.shear?, self.conserving?);
        let c_p = (self.sound_speed2.unwrap_or(0.0) + mu).sqrt();
        let (lo, hi) = self.tank;
        let reach = 2.0 * self.layout.dx;
        let distance = |q: Vec2| (q.x - lo.x).min(hi.x - q.x).min(q.y - lo.y).min(hi.y - q.y);
        let near = self.x.iter().zip(&self.v).any(|(&q, &v)| {
            let ahead = q + v * FRAME + 0.5 * self.gravity * FRAME * FRAME;
            distance(q).min(distance(ahead)) < reach
        });
        near.then_some(multiple * 0.5 * self.layout.dx / c_p)
    }

    /// One implicit, energy-momentum conserving step of the solid: the
    /// total Lagrangian step (`total_lagrangian::step_total_lagrangian_contact`),
    /// its grid on the configuration of its first step, the tank's walls
    /// held by the contact condition in velocity (route C), frictionless.
    fn conserving_substep(
        &mut self,
        mu: f32,
        settings: crate::spacetime::total_lagrangian::ConservingSettings,
        dt: f32,
        frame: &mut Frame,
    ) {
        use glam::{DMat2, DVec2};
        let n = self.x.len();
        let mut x: Vec<DVec2> = self.x.iter().map(|q| q.as_dvec2()).collect();
        let mut v: Vec<DVec2> = self.v.iter().map(|q| q.as_dvec2()).collect();
        let mut f: Vec<DMat2> = self.deformation.iter().map(|m| m.as_dmat2()).collect();
        let mass: Vec<f64> = self.mass.iter().map(|&m| f64::from(m)).collect();
        let v0: Vec<f64> = self.v0.iter().map(|&m| f64::from(m)).collect();
        let (lo, hi) = (self.tank.0.as_dvec2(), self.tank.1.as_dvec2());
        // The probes' grid reports see no wall: walls are contacts now.
        let held = |_: usize, _: usize, _: usize| -> Option<f64> { None };
        let solids = self.solids.clone().unwrap_or_else(|| {
            vec![
                crate::spacetime::total_lagrangian::Elastic::isochoric(
                    f64::from(mu),
                    f64::from(self.sound_speed2.unwrap_or(0.0)),
                );
                n
            ]
        });
        if !self.probe.conditioning_bodies.is_empty() {
            let volume: Vec<f64> = (0..n).map(|p| v0[p] * f[p].determinant()).collect();
            let at = |r: std::ops::Range<usize>| {
                crate::spacetime::total_lagrangian::mass_conditioning(
                    &self.layout,
                    &x[r.clone()],
                    (&mass[r.clone()], &volume[r]),
                    &held,
                    (settings.occupation, settings.epsilon),
                )
            };
            let union = at(0..n);
            let each: Vec<f64> = self
                .probe
                .conditioning_bodies
                .iter()
                .map(|r| at(r.clone()))
                .collect();
            // The worst body below the threshold, dissected.
            let threshold = self.probe.basis_threshold;
            let worst = each
                .iter()
                .enumerate()
                .filter(|(_, c)| **c < threshold)
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, &c)| (i, c));
            if let Some((i, c)) = worst
                && self.probe.basis_worst.as_ref().is_none_or(|(w, _)| c < *w)
            {
                let r = self.probe.conditioning_bodies[i].clone();
                let report = crate::spacetime::total_lagrangian::basis_report(
                    &self.layout,
                    &x[r.clone()],
                    (&mass[r.clone()], &volume[r]),
                    &held,
                    (settings.occupation, settings.epsilon),
                );
                self.probe.basis_worst = Some((c, format!("body {i}: {report}")));
            }
            let (low, lows) = &mut self.probe.conditioning;
            *low = low.min(union);
            if lows.len() != each.len() {
                *lows = vec![f64::INFINITY; each.len()];
            }
            for (l, e) in lows.iter_mut().zip(each) {
                *l = l.min(e);
            }
        }
        let fastest = v.iter().fold(0.0f64, |m, q| m.max(q.length()));
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut v,
            f: &mut f,
            mass: &mass,
            v0: &v0,
        };
        let plane =
            |axis, at, inward| crate::spacetime::total_lagrangian::Plane { axis, at, inward };
        let planes = [
            plane(0, lo.x, 1.0),
            plane(0, hi.x, -1.0),
            plane(1, lo.y, 1.0),
            plane(1, hi.y, -1.0),
        ];
        // The penalty, the Uzawa iteration's parameter: the P wave's
        // stiffness (factor 1, `block_slide_run`'s measured choice).
        let from = self.contact_from.unwrap_or(0);
        let penalty: Vec<f64> = solids
            .iter()
            .enumerate()
            .map(|(p, s)| if p >= from { s.bulk + s.mu } else { 0.0 })
            .collect();
        if self.lagrangian.as_ref().is_none_or(|(r, _)| r.len() != n) {
            self.lagrangian = Some((self.x.iter().map(|q| q.as_dvec2()).collect(), None));
        }
        let (reference, cache) = self.lagrangian.as_mut().expect("set above");
        let report = crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
            &self.layout,
            particles,
            reference,
            &solids,
            (
                f64::from(dt),
                crate::spacetime::total_lagrangian::ContactMode::Velocity,
                self.gravity.as_dvec2(),
            ),
            (
                crate::spacetime::total_lagrangian::ContactSet {
                    planes: &planes,
                    pairs: &[],
                    outlines: &[],
                    corners: &[],
                    friction: 0.0,
                },
                &penalty,
                &settings,
            ),
            cache,
        );
        if std::env::var_os("EMERGE_PROBE_SCENE_TRACE").is_some() {
            println!(
                "  step {}: lowest {:.3} (floor {:.3}), Newton {} residual {:.2e} (start {:.2e}) converged {}, contacts {}, push {:.3e}, passes {}, fastest {:.3e}",
                self.probe.steps,
                self.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min),
                lo.y,
                report.newton,
                report.residual,
                report.start_residual,
                report.converged,
                report.contacts,
                report.push,
                report.passes,
                fastest
            );
        }
        self.probe.contacts = self.probe.contacts.max(report.contacts);
        for (total, part) in self.probe.phases.iter_mut().zip(report.phases) {
            *total += part;
        }
        self.probe.steps += 1;
        self.probe.passes += u64::from(report.passes);
        self.probe.stabilisation_loss += report.stabilisation_loss;
        self.probe.grid_speed_ratio = self
            .probe
            .grid_speed_ratio
            .max(report.grid_speed / fastest.max(f64::MIN_POSITIVE));
        self.probe.start_residual = self.probe.start_residual.max(report.start_residual);
        self.probe.worst_projection = self.probe.worst_projection.max(report.projection);
        self.probe.ledger = report.ledger;
        self.probe.work = report.work;
        let stats = &mut self.probe.conserving;
        stats.0 += u64::from(report.newton);
        stats.1 += u64::from(report.gmres);
        stats.2 = stats.2.max(report.gmres);
        stats.3 = stats.3.max(report.residual);
        stats.4 += u32::from(!report.converged);
        if self.probe.estimator_reference > 0.0 {
            let acceleration: Vec<Vec2> =
                (0..n).map(|p| (v[p].as_vec2() - self.v[p]) / dt).collect();
            if self.probe.last_acceleration.len() == n {
                let jump = acceleration
                    .iter()
                    .zip(&self.probe.last_acceleration)
                    .fold(0.0f32, |m, (a, b)| m.max((*a - *b).abs().max_element()));
                let estimate = (0.25 - 1.0 / 6.0) * dt * dt * jump / self.probe.estimator_reference;
                self.probe.largest_estimate = self.probe.largest_estimate.max(estimate);
            }
            self.probe.last_acceleration = acceleration;
        }
        for p in 0..n {
            let before = self.x[p];
            self.x[p] = x[p].as_vec2();
            self.v[p] = v[p].as_vec2();
            self.c[p] = Mat2::ZERO;
            self.deformation[p] = f[p].as_mat2();
            frame.largest_travel = frame.largest_travel.max((self.x[p] - before).length());
        }
        self.probe.solve_drift = 0.0;
        // Counted, not moved: the contact holds the walls, and a moved
        // particle would leave the grid's reference configuration (`x = X +
        // u(X)`); the depth beyond a wall stays a measured criterion.
        for p in 0..n {
            self.beyond_walls(p, false);
        }
        self.time += dt;
        frame.substeps += 1;
        frame.phi = self.phi();
    }

    /// Moves particle `p`, if beyond a wall, to its mirror point across it,
    /// and counts it (the reflection amendment, module doc).
    fn reflect_off_walls(&mut self, p: usize) {
        self.beyond_walls(p, true);
    }

    /// Counts particle `p` if it is beyond a wall, by how far, and with
    /// `mirror` moves it to its mirror point across it.
    fn beyond_walls(&mut self, p: usize, mirror: bool) {
        let ulp = |w: f32| f32::from_bits(w.abs().to_bits() + 1) - w.abs();
        let (lo, hi) = self.tank;
        let mut q = self.x[p];
        let drift = self.probe.solve_drift;
        let (mut deepest, mut ratio) = (0.0f32, 0.0f32);
        for (coord, low, high) in [(&mut q.x, lo.x, hi.x), (&mut q.y, lo.y, hi.y)] {
            let wall = if *coord < low {
                low
            } else if *coord > high {
                high
            } else {
                continue;
            };
            let depth = (*coord - wall).abs();
            deepest = deepest.max(depth / ulp(wall));
            ratio = ratio.max(depth / (4.0 * ulp(wall) + drift));
            *coord = 2.0 * wall - *coord;
        }
        if deepest > 0.0 {
            if mirror {
                self.x[p] = q;
            }
            self.probe.reflections += 1;
            if deepest > self.probe.deepest_reflection_ulps {
                self.probe.deepest_reflection_at = Some((self.time, self.x[p]));
            }
            self.probe.deepest_reflection_ulps = self.probe.deepest_reflection_ulps.max(deepest);
            self.probe.worst_reflection_ratio = self.probe.worst_reflection_ratio.max(ratio);
        }
    }

    /// The reflection criterion (module doc).
    fn check_reflections(&self, label: &str, report: &mut Report) {
        let (n, deepest) = (self.probe.reflections, self.probe.deepest_reflection_ulps);
        let ratio = self.probe.worst_reflection_ratio;
        println!(
            "{label}: {n} reflections off a wall, deepest {deepest:.2} ulps before it, worst {ratio:.2} of its budget"
        );
        if let Some((t, at)) = self.probe.deepest_reflection_at {
            println!("{label}: deepest reflection at t = {t:.4} s, mirrored to {at:?}");
        }
        report.check(ratio <= 1.0, || {
            format!("{label}: a particle went {ratio:.2} times its budget beyond a wall ({deepest:.1} ulps)")
        });
    }

    /// Which cells have their centre outside every solid.
    fn open_centres(&self) -> Vec<bool> {
        self.solid_centres.data().iter().map(|&d| d > 0.0).collect()
    }

    /// A particle's volume ratio: a solid's is its own `det F`, the `J` its
    /// stress reads; a liquid's integrates the cells' divergence.
    fn j(&self, p: usize) -> f32 {
        if self.shear.is_some() {
            self.deformation[p].determinant()
        } else {
            self.log_j[p].exp()
        }
    }

    fn volume(&self) -> f32 {
        (0..self.x.len()).map(|p| self.v0[p] * self.j(p)).sum()
    }

    fn centre_of_mass(&self) -> (Vec2, Vec2) {
        let total: f32 = self.mass.iter().sum();
        let x: Vec2 = (0..self.x.len()).map(|p| self.x[p] * self.mass[p]).sum();
        let v: Vec2 = (0..self.x.len()).map(|p| self.v[p] * self.mass[p]).sum();
        (x / total, v / total)
    }

    /// Kinetic plus potential energy, the floor of the tank as zero.
    fn energy(&self) -> f32 {
        let floor = self.tank.0.y;
        let elastic = |p: usize| match (self.shear, self.sound_speed2) {
            (Some(mu), Some(lambda)) => {
                let j = self.j(p);
                self.v0[p]
                    * (shear_energy(self.deformation[p], mu) + 0.5 * lambda * (j - 1.0).powi(2))
            }
            _ => 0.0,
        };
        (0..self.x.len())
            .map(|p| {
                let kinetic = 0.5 * self.v[p].length_squared();
                self.mass[p] * (kinetic + self.g() * (self.x[p].y - floor)) + elastic(p)
            })
            .sum()
    }

    /// In the tank or on its walls: out means beyond a wall. A particle gliding
    /// along a wall approaches it geometrically (the normal velocity falls
    /// linearly to zero there, `substep` doc) and, in f32, ends on it: measured
    /// in the drop into a pool once the elastic solid's transfer was in, a
    /// particle reached `x = 4.0000005`, one ulp from the wall at 4, with `v_x
    /// = -1.6e-5` cells/s, sat exactly on the wall for one frame and went back
    /// in, never beyond it. The strict test counted it out.
    fn inside_tank(&self, p: usize) -> bool {
        let q = self.x[p];
        q.x >= self.tank.0.x && q.x <= self.tank.1.x && q.y >= self.tank.0.y && q.y <= self.tank.1.y
    }

    /// Particles per cell over cells whose eight neighbours also hold
    /// particles: a volume measure that does not read J.
    fn interior_particles_per_cell(&self) -> f32 {
        let (nx, ny) = (self.layout.nx as i32, self.layout.ny as i32);
        let mut count = vec![0u32; (nx * ny) as usize];
        for q in &self.x {
            let c = q.floor().as_ivec2();
            if c.x >= 0 && c.y >= 0 && c.x < nx && c.y < ny {
                count[(c.x + nx * c.y) as usize] += 1;
            }
        }
        let (mut sum, mut cells) = (0u32, 0u32);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let full = (-1..=1)
                    .all(|dj| (-1..=1).all(|di| count[(i + di + nx * (j + dj)) as usize] > 0));
                if full {
                    sum += count[(i + nx * j) as usize];
                    cells += 1;
                }
            }
        }
        if cells == 0 {
            0.0
        } else {
            sum as f32 / cells as f32
        }
    }

    /// A text picture of the particles, printed when `EMERGE_GATE_MAPS` is
    /// set.
    fn print_map(&self, label: &str) {
        if crate::diagnostics::research_switch("EMERGE_GATE_MAPS").is_none() {
            return;
        }
        let mut particles = Particles::default();
        for &x in &self.x {
            particles.push(Particle {
                x,
                ..bytemuck::Zeroable::zeroed()
            });
        }
        let size = Vec2::new(self.layout.nx as f32, self.layout.ny as f32);
        let cols = self.layout.nx.min(72);
        let rows = (cols as f32 * size.y / size.x / 2.0).ceil() as usize;
        println!("{label} t={:.3}s", self.time);
        let map = scene_map(
            &particles,
            (Vec2::ZERO, size),
            cols,
            rows.max(1),
            |_| 1.0,
            &OCCUPANCY_BANDS,
        );
        for row in map {
            println!("|{row}|");
        }
    }
}

/// Where the level set crosses zero going up column `i` from the tank
/// floor, by linear interpolation between cell centres.
fn surface_height(phi: &Field2, i: usize, floor_cell: usize) -> Option<f32> {
    (floor_cell + 1..phi.nj()).find_map(|j| {
        let (below, above) = (phi.get(i, j - 1), phi.get(i, j));
        (below < 0.0 && above >= 0.0).then(|| (j as f32 - 0.5) + below / (below - above))
    })
}

/// Four particles per cell over `cells`, each at a random point of its
/// quarter of the cell, kept where `inside` holds; those within one cell
/// of a free surface (`surface` gives the distance and outward normal)
/// moved along the normal to the particle spacing from it (Zhu and
/// Bridson 4.2.1).
fn seed(
    cells: (IVec2, IVec2),
    inside: impl Fn(Vec2) -> bool,
    surface: impl Fn(Vec2) -> (f32, Vec2),
    rng: &mut LcgRng,
) -> Vec<Vec2> {
    let spacing = 0.5;
    let mut x = Vec::new();
    for j in cells.0.y..cells.1.y {
        for i in cells.0.x..cells.1.x {
            for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let quarter = Vec2::new(0.25 + 0.5 * a as f32, 0.25 + 0.5 * b as f32);
                let jitter = Vec2::new(rng.next_f32() - 0.5, rng.next_f32() - 0.5) * spacing;
                let mut q = Vec2::new(i as f32, j as f32) + quarter + jitter;
                if !inside(q) {
                    continue;
                }
                let (distance, normal) = surface(q);
                if distance < 1.0 {
                    q += normal * (distance - spacing);
                }
                x.push(q);
            }
        }
    }
    x
}

/// `seed` for a body: a liquid's particles pulled to the particle spacing
/// from its free surface (Zhu and Bridson's technique, `seed`, which
/// smooths the surface their level set rebuilds), a solid's left where they
/// fall. A solid's particles are the quadrature points of its volume, and
/// the pull lays the two outer rows of a box onto one line (the particles a
/// quarter and three quarters of a spacing from a face both end half a
/// spacing from it), a degenerate quadrature (measured: at one frame per
/// substep Newton fails on it within four frames and a particle inverts).
fn seed_body(
    solid: bool,
    cells: (IVec2, IVec2),
    inside: impl Fn(Vec2) -> bool,
    surface: impl Fn(Vec2) -> (f32, Vec2),
    rng: &mut LcgRng,
) -> Vec<Vec2> {
    if solid {
        seed(cells, inside, |_| (f32::INFINITY, Vec2::Y), rng)
    } else {
        seed(cells, inside, surface, rng)
    }
}

/// Distance to the nearest of a box's free sides and that side's normal.
fn box_free_sides(q: Vec2, top: Option<f32>, right: Option<f32>) -> (f32, Vec2) {
    let mut best = (f32::INFINITY, Vec2::ZERO);
    if let Some(y) = top
        && y - q.y < best.0
    {
        best = (y - q.y, Vec2::Y);
    }
    if let Some(x) = right
        && x - q.x < best.0
    {
        best = (x - q.x, Vec2::X);
    }
    best
}

fn disc_side(q: Vec2, centre: Vec2, radius: f32) -> (f32, Vec2) {
    let d = q - centre;
    let r = d.length();
    (radius - r, if r > 0.0 { d / r } else { Vec2::Y })
}

/// Every missed criterion of a run, printed as it is found and asserted
/// together at the end, so one run shows them all.
struct Report {
    failures: Vec<String>,
}

impl Report {
    fn new() -> Self {
        Self {
            failures: Vec::new(),
        }
    }

    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        if !ok {
            let line = what();
            println!("  FAIL {line}");
            self.failures.push(line);
        }
    }

    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} criteria missed:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

/// Totals over a run, for the cost lines.
#[derive(Default)]
struct Cost {
    frames: u32,
    substeps: u32,
    max_substeps: u32,
    iterations: u64,
    solves: u64,
    max_iterations: u32,
    capped: u32,
    largest_travel: f32,
}

impl Cost {
    fn add(&mut self, frame: &Frame) {
        self.frames += 1;
        self.substeps += frame.substeps;
        self.max_substeps = self.max_substeps.max(frame.substeps);
        self.iterations += frame.iterations.iter().map(|&i| i as u64).sum::<u64>();
        self.solves += frame.iterations.len() as u64;
        let most = frame.iterations.iter().copied().max().unwrap_or(0);
        self.max_iterations = self.max_iterations.max(most);
        self.capped += frame.capped;
        self.largest_travel = self.largest_travel.max(frame.largest_travel);
    }

    fn print(&self, label: &str, wall_s: f32, simulated_s: f32) {
        println!(
            "  cost {label}: {:.1} substeps/frame (max {}), CG {:.1} it/solve (max {}), \
             {} capped, largest travel {:.3} cell, {:.0} ms per simulated s ({})",
            self.substeps as f32 / self.frames.max(1) as f32,
            self.max_substeps,
            self.iterations as f32 / self.solves.max(1) as f32,
            self.max_iterations,
            self.capped,
            self.largest_travel,
            1000.0 * wall_s / simulated_s,
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
        );
    }
}

/// The Gauss-Legendre rule of `n` points (1 to 5) on `[0, 1]`: nodes and
/// weights, from the closed forms of the roots of the Legendre polynomial
/// `P_n` on `[-1, 1]` mapped by `t = (x + 1) / 2` (weights halved). Exact
/// for polynomials of degree up to `2 n - 1`, which the probe checks.
pub(super) fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (x, w): (Vec<f64>, Vec<f64>) = match n {
        1 => (vec![0.0], vec![2.0]),
        2 => {
            let a = (1.0f64 / 3.0).sqrt();
            (vec![-a, a], vec![1.0, 1.0])
        }
        3 => {
            let a = (3.0f64 / 5.0).sqrt();
            (vec![-a, 0.0, a], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        }
        4 => {
            let r = 2.0 / 7.0 * (6.0f64 / 5.0).sqrt();
            let (a, b) = ((3.0 / 7.0 - r).sqrt(), (3.0 / 7.0 + r).sqrt());
            let s = 30.0f64.sqrt();
            let (wa, wb) = ((18.0 + s) / 36.0, (18.0 - s) / 36.0);
            (vec![-b, -a, a, b], vec![wb, wa, wa, wb])
        }
        _ => {
            let r = 2.0 * (10.0f64 / 7.0).sqrt();
            let (a, b) = ((5.0 - r).sqrt() / 3.0, (5.0 + r).sqrt() / 3.0);
            let s = 13.0 * 70.0f64.sqrt();
            let (wa, wb) = ((322.0 + s) / 900.0, (322.0 - s) / 900.0);
            (vec![-b, -a, 0.0, a, b], vec![wb, wa, 128.0 / 225.0, wa, wb])
        }
    };
    (
        x.iter().map(|x| 0.5 * (x + 1.0)).collect(),
        w.iter().map(|w| 0.5 * w).collect(),
    )
}

/// The seeds of the dam break and the drop into a pool.
fn dam_break_seed() -> Vec<Vec2> {
    let mut rng = LcgRng::new(3);
    seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    )
}

fn drop_into_pool_seed() -> Vec<Vec2> {
    let mut rng = LcgRng::new(5);
    let mut x = seed(
        (IVec2::new(4, 4), IVec2::new(68, 19)),
        |_| true,
        |q| box_free_sides(q, Some(19.0), None),
        &mut rng,
    );
    let (centre, radius) = (Vec2::new(36.0, 44.0), 5.0);
    x.extend(seed(
        (IVec2::new(30, 38), IVec2::new(42, 50)),
        |q| (q - centre).length() < radius,
        |q| disc_side(q, centre, radius),
        &mut rng,
    ));
    x
}

/// `E / rho` (m^2/s^2) giving plane-strain shear speed `c_s` at `nu`.
fn e_over_rho_for_shear_speed(c_s: f32, nu: f32) -> f32 {
    2.0 * (1.0 + nu) * c_s * c_s
}

/// Kinetic, potential, shear and volume energy, the floor as zero: the
/// drop's measure for any row.
fn row_energy(scene: &Scene) -> f64 {
    let floor = scene.tank.0.y;
    (0..scene.x.len())
        .map(|p| {
            let shear = scene
                .shear
                .map_or(0.0, |mu| shear_energy(scene.deformation[p], mu));
            let volume = scene
                .sound_speed2
                .map_or(0.0, |kappa| 0.5 * kappa * (scene.j(p) - 1.0).powi(2));
            f64::from(
                scene.mass[p]
                    * (0.5 * scene.v[p].length_squared() + scene.g() * (scene.x[p].y - floor))
                    + scene.v0[p] * (shear + volume),
            )
        })
        .sum()
}

/// A conserving run's fixed substep: the explicit P wave limit `0.5 dx /
/// c_p`, four times it, or one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Substep {
    PWave,
    FourPWaves,
    Frame,
}

impl Substep {
    /// From a probe's setting: `p`, `4p` or `frame`.
    fn parse(name: &str) -> Option<Self> {
        match name {
            "p" => Some(Self::PWave),
            "4p" => Some(Self::FourPWaves),
            "frame" => Some(Self::Frame),
            _ => None,
        }
    }

    /// Its length in seconds on cells of `dx` for a P wave of `c_p`.
    fn length(self, dx: f32, c_p: f32) -> f32 {
        match self {
            Self::PWave => 0.5 * dx / c_p,
            Self::FourPWaves => 2.0 * dx / c_p,
            Self::Frame => FRAME,
        }
    }
}

impl std::fmt::Display for Substep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PWave => "0.5 dx / c_p",
            Self::FourPWaves => "2 dx / c_p",
            Self::Frame => "one frame",
        })
    }
}

/// A square solid block of `side` cells centred on `centre`, seeded
/// (`seed_body`) from `seed_value`: the per-body probes' bodies.
fn clay_block(centre: Vec2, side: f32, seed_value: u32) -> Vec<Vec2> {
    let (lo, hi) = (
        centre - Vec2::splat(0.5 * side),
        centre + Vec2::splat(0.5 * side),
    );
    seed_body(
        true,
        (lo.floor().as_ivec2(), hi.ceil().as_ivec2()),
        |q| q.cmpge(lo).all() && q.cmple(hi).all(),
        |_| (f32::INFINITY, Vec2::Y),
        &mut LcgRng::new(seed_value),
    )
}

/// One body's energy (`row_energy` over its particles `r`), kinetic
/// energy, centre height in m and vertical centre speed in m/s.
fn body_measure(scene: &Scene, r: std::ops::Range<usize>) -> [f64; 4] {
    let floor = scene.tank.0.y;
    let mu = scene.shear.unwrap_or(0.0);
    let kappa = scene.sound_speed2.unwrap_or(0.0);
    let (mut energy, mut kinetic, mut mass) = (0.0f64, 0.0f64, 0.0f64);
    let (mut y, mut vy) = (0.0f64, 0.0f64);
    for p in r {
        let m = f64::from(scene.mass[p]);
        let k = 0.5 * m * f64::from(scene.v[p].length_squared());
        let stored = f64::from(
            scene.v0[p]
                * (shear_energy(scene.deformation[p], mu)
                    + 0.5 * kappa * (scene.j(p) - 1.0).powi(2)),
        );
        energy += k + m * f64::from(scene.g() * (scene.x[p].y - floor)) + stored;
        kinetic += k;
        mass += m;
        y += m * f64::from(scene.x[p].y);
        vy += m * f64::from(scene.v[p].y);
    }
    let cell = f64::from(scene.cell_m);
    [energy, kinetic, y / mass * cell, vy / mass * cell]
}

/// Quadrature points of the ring `Ri <= R <= Ro` about `centre`: every half
/// cell of the grid (both lattices' B-splines are polynomials on it) wholly
/// inside the ring takes a tensor Gauss-Legendre rule (3x3 by default,
/// `EMERGE_PROBE_VORTEX_GAUSS`); a half
/// cell the circles cut is split in four, down to `depth` levels, and the
/// rule's points of the last level are kept where they lie in the ring.
/// Composite Gauss quadrature over an adaptive subdivision of the cut
/// cells; its geometric error is the run's measured area error.
fn ring_quadrature(
    centre: glam::DVec2,
    (inner, outer): (f64, f64),
    side: usize,
    depth: u32,
) -> (Vec<glam::DVec2>, Vec<f64>) {
    use glam::DVec2;
    // `EMERGE_PROBE_VORTEX_GAUSS` points per direction (default 3).
    let order: usize = std::env::var("EMERGE_PROBE_VORTEX_GAUSS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let (nodes, weights) = gauss_legendre(order);
    let (mut points, mut volumes) = (Vec::new(), Vec::new());
    let mut stack: Vec<(DVec2, f64, u32)> = (0..2 * side)
        .flat_map(|j| (0..2 * side).map(move |i| (DVec2::new(i as f64, j as f64) * 0.5, 0.5, 0)))
        .collect();
    while let Some((lo, size, level)) = stack.pop() {
        let nearest = DVec2::new(
            centre.x.clamp(lo.x, lo.x + size),
            centre.y.clamp(lo.y, lo.y + size),
        );
        let near = (nearest - centre).length();
        let far = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .iter()
            .map(|&(a, b)| (lo + size * DVec2::new(a, b) - centre).length())
            .fold(0.0f64, f64::max);
        if far <= inner || near >= outer {
            continue;
        }
        let whole = near >= inner && far <= outer;
        if whole || level >= depth {
            for (b, wb) in nodes.iter().zip(&weights) {
                for (a, wa) in nodes.iter().zip(&weights) {
                    let q = lo + size * DVec2::new(*a, *b);
                    let r = (q - centre).length();
                    if whole || (inner..=outer).contains(&r) {
                        points.push(q);
                        volumes.push(size * size * wa * wb);
                    }
                }
            }
        } else {
            let half = 0.5 * size;
            for (a, b) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                stack.push((lo + half * DVec2::new(a, b), half, level + 1));
            }
        }
    }
    (points, volumes)
}

mod bars;
mod disc_slide;
mod elastic_column;
mod liquid;
mod manufactured;
mod matrix;
mod per_body;
mod spread;
mod vortex;
mod walls;
