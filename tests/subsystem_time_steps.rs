//! Time steps of the subsystems beside the mechanics: grains, rods, heat and
//! scalar fields. Criteria written before the code and frozen: they do not
//! move once results have been seen.
//!
//! # What reading the code found
//!
//! - Heat and scalar fields are applied once per `step()`, the whole frame's
//!   time in one explicit step. The heat bound is folded into the mechanics
//!   substep, where it slows the mechanics and protects nothing. The scalar
//!   field has no bound, and decays by `1 - lambda dt`, negative past
//!   `lambda dt = 1`, while its doc says `exp(-lambda dt)`.
//! - Grain contacts are integrated at the mechanics substep; their critical
//!   step exists but only tests call it.
//! - A stiff explicit rod slows the whole scene; the implicit rod is wired
//!   but chosen by a flag set by hand.
//!
//! # The junction
//!
//! - Every subsystem declares the largest step its own scheme stays stable
//!   at, from its own parameters, through one interface shared by grains,
//!   rods, heat, scalar fields, granular fluidity and the Cosserat field.
//!   Each limit is derived from its scheme, next to the code.
//! - The step a subsystem takes is `material_cfl_coefficient` times its
//!   limit: the definition the materials already use, the fraction of the
//!   stability limit, and no new free number.
//! - Explicit rods, advanced inside the mechanics substep: the substep
//!   takes the minimum, as `timestep_bound` does for the materials.
//! - Grains: their contacts are sub-cycled inside the mechanics substep to
//!   their own step, rather than clamping the substep; with a real
//!   mineral's stiffness a minimum would collapse the whole scene's step.
//!   The cost is measured and reported.
//! - Heat and scalar fields, advanced once per frame on their own clock:
//!   they sub-cycle to their own step and no longer clamp the mechanics.
//! - A rod goes implicit by itself when its explicit step would fall below
//!   the rest of the scene's; no flag set by hand. An override remains, so a
//!   test can compare the two integrations.
//! - A scene with none of these subsystems gives bit-identical results.
//!
//! # Gate 1: the diffusion of a peak matches the analytic
//!
//! Through the real path of `ScalarDiffusionField` (particles to grid,
//! finite difference, grid to particles) with its transfer blend at the
//! default, 1.0, and through `ThermalDiffusion`. A body of particles at rest,
//! no gravity, a Gaussian peak of standard deviation 3 cells.
//!
//! - First, with no physical diffusion, the variance change the transfers
//!   alone make over the same time is measured and reported. The criteria
//!   below count only if it is under 1 percent of `2 D T`, so they cannot
//!   pass or fail because of the transfers.
//! - With a frame at half the explicit limit, and at twenty times it: the
//!   variance along each axis grows as `sigma0^2 + 2 D t` within 5
//!   percent; without decay the total is conserved within 0.1 percent; with
//!   decay at `lambda * frame = 3`, the total follows `exp(-lambda t)`
//!   within 0.5 percent (for heat, the excess over ambient under Newton
//!   cooling).
//!
//! # Gate 2: two stiff grains on soft sand invent no energy
//!
//! A bed of soft sand at real gravity settles alone for 1 s. Two grains are
//! placed on it, one on the other, at rest, and the scene runs 3 s. Their
//! contact stiffness comes from the existing physical preset with the Young's
//! modulus of quartz, read from two sources, never softened to pass.
//!
//! - The energy counted is the whole system, sand and grains, kinetic,
//!   rotational and gravitational, because energy passes between them. The
//!   same settled bed also runs alone over the same 3 s; the rise of the
//!   energy with the grains, above the bed alone's, is at most 1 percent of
//!   the grains' own `m g R`.
//! - Reported: grain contact sub-steps per mechanics substep, and the wall
//!   time with and without the grains.
//!
//! # Gate 3: a rod bends the same explicit and implicit
//!
//! A cantilever under its own weight, clamped at one end, its tip deflection
//! under 5 percent of its length.
//!
//! - Settled tip deflection, explicit and implicit, within 1 percent of each
//!   other, and within 5 percent of the Euler-Bernoulli value
//!   `q L^4 / (8 E I)`, derived in the test.
//! - A stiff rod, whose explicit step is below the scene's, goes implicit by
//!   itself: adding it leaves the scene's substeps per frame unchanged.
//!
//! # Gate 4: nothing else moves
//!
//! The existing suite is unchanged, except tests that assert the old fold of
//! the heat bound into the mechanics substep, rewritten with the reason.
//!
//! # Order
//!
//! Diffusion, then grains, then rods, each measured before its change and
//! after.
