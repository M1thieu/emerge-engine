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

mod common;
mod scene;
use common::*;
use scene::*;

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
