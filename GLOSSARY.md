# emerge — Glossary / Quick Reference

A flat, lookup-first index of emerge's concepts, materials, fields, and APIs —
for pulling up "what does X do" fast when onboarding or working with other
people, without re-reading the narrative docs. `ARCHITECTURE.md`
explains *why*; this file is *what exists and where*. First draft (2026-07-20)
— real content pulled from existing source/doc comments, not invented; expand
as gaps are found rather than treating this as finished.

---

## Particle fields (`src/matter/particle.rs`, 128 bytes, `repr(C)`)

| Field | Type | Meaning |
|---|---|---|
| `x` | `Vec2` | position (grid coords) |
| `v` | `Vec2` | velocity |
| `velocity_gradient` | `Mat2` | APIC affine matrix C, ∂v/∂x |
| `deformation_gradient` | `Mat2` | F |
| `mass`, `initial_volume`, `volume`, `density` | `f32` | standard MPM state |
| `material_id` | `u32` | slot index into `MaterialRegistry` |
| `plastic_volume_ratio` | `f32` | Jₚ = det(Fₚ) |
| `hardening_scale` | `f32` | h = exp(ξ(1−Jₚ)) |
| `friction_hardening` | `f32` | shared plasticity scratch — DP `q` / Von Mises `κ` / Rankine damage / SandMuI µ(I). One particle runs one material, so one field safely serves all of them (see ARCHITECTURE.md §2). |
| `log_volume_strain` | `f32` | DP εᵥ |
| `temperature` | `f32` | used by thermal diffusion + phase rules |
| `user_tag` | `u32` | LP: creature/body ownership, no engine meaning |
| `activation` | `f32` | [0,1] active-matter drive (muscle contraction) |
| `activation_dir` | `Vec2` | muscle fiber direction, material frame |
| `muscle_group_id` | `u32` | tags a subset of particles for independent activation control — same continuum, different control group |
| `contact_group` | `u32` | 0 = ordinary particle; nonzero = opts into multi-field frictional contact (see ARCHITECTURE.md §7). Zero-cost when unused. |
| `sleeping` | `u32` | GPU sleep flag |
| `internal_pressure` | `f32` | pre-stress pressure (already SI-converted to grid units), see `MaterialModel::pressure_scale` |
| `pinned` | `u32` | real Dirichlet anchor — forces v=0, velocity_gradient=0 every substep in G2P |

---

## Core mechanisms (by name)

- **`MaterialModel::activation_scale()`** — scaling coefficient for activation-driven deviatoric stress. Muscle/active-matter hook. Default 0.0 (opt-in per material).
- **`MaterialModel::pressure_scale()`** — scaling coefficient for internal pre-stress. Turgor-pressure-style hook (any internally-pressurized body, not plant-specific). Default 0.0.
- **`MixturePhase`** (`Solid`/`Fluid`) — two-phase mixture coupling role (Tampubolon et al. 2017, Darcy drag between interpenetrating granular/fluid phases). `None` (default) = no coupling.
- **`contact_group`** — real multi-field frictional contact (Bardenhagen 2001 + Nairn, Hammerquist, Smith 2020 normal fit + Baumgarte stabilization). See ARCHITECTURE.md §7 for the full algorithm.
- **`WithLatentHeat<M>` / `WithMixturePhase<M>` / `WithPreStress<M>`** — delegating wrapper structs that bolt one extra behavior onto any `MaterialModel` without rewriting it. All three share a `forward_material_model_common!()` macro for the ~10 always-identical pass-through methods (`src/matter/materials/mod.rs`).
- **`add_phase_rule(Fn(&Particle) -> Option<u32>)`** — automatic material_id transition evaluated every substep (freezing, melting, evaporation).
- **`ScalarDiffusionField`** — generic diffusion field (heat, pheromone, nutrients, morphogen). Reaction-diffusion (Gray-Scott/Turing) ready via its `source` closure.
- **Sleep/wake** — flag-based active/sleeping partition, not memory compaction (Phase 1 only; see ARCHITECTURE.md §6). Particles: `SimConfig::sleep_threshold`, per-particle swap into a sleeping tail. Rods: `SimConfig::rod_sleep_threshold`, whole-rod `Rod::sleeping` flag (points are elastically coupled, can't sleep individually) — skips scatter/gather/force integration *and* `rod_cfl_dt` in the substep bound, the real per-rod cost fix for a field of many rods.
- **Adaptive substeps** — `Simulation::step()` always advances exactly `config.dt`, internally split into as many CFL-safe substeps as needed. Not a tuning knob — real physics.
- **`Lnn`** (`information::control::lnn`) — Liquid Time-constant Network CPG (Hasani et al. 2020), a standalone locomotion controller. Does not participate in the substep loop; writes into `activation`/`activation_dir` between steps.
- **`spacetime::diff`** — separate, hand-derived-adjoint forward+reverse MLS-MPM implementation for gradient-based offline controller training. Not used at real-time/play time.

---

## Materials (`use emerge::prelude::*`)

| Type | Best for | Key preset |
|---|---|---|
| `NeoHookeanMaterial` | soft elastic solids, muscle base | `from_young_modulus(E, nu)` |
| `CorotatedMaterial` | stiffer elastic solids | `from_young_modulus(E, nu)` |
| `ViscoelasticMaterial` | near-incompressible damped solids (Kelvin-Voigt) | `.near_incompressible()` `.moderately_compressible()` |
| `NewtonianFluidMaterial` | low-viscosity fluids | `.low_viscosity(density, stiffness)` |
| `BinghamFluidMaterial` | viscoplastic fluids with yield | `.low_yield()` `.medium_yield()` `.high_yield()` |
| `StomakhinMaterial` | snow (SVD crushing/tension, hardening + cohesion) | `from_young_modulus(E, nu)` `.low_cohesion()` |
| `DruckerPragerMaterial` | cohesionless/frictional granular (sand) | `.cohesionless()` `.low_friction()` `.dilatant()` |
| `MuIRheologyMaterial` | rate-dependent dense granular flow | `.small_grain()` `.dense_packed()` |
| `VonMisesMaterial` | ductile yield with hardening | `from_young_modulus(E, nu, yield_stress)` |
| `RankineMaterial` | brittle fracture with softening | `.stiff_brittle()` `.high_tensile()` |
| `NaccMaterial` | wet soil/clay/tissue under compression | `.soft_clay(E, nu)` `.wet_soil(E, nu)` |
| `GranularFluidMaterial` | granular-fluid mixture (Tait EOS + corotated + SVD) | `.saturated_loam(E, nu)` `.cytoplasmic(E, nu)` |
| `NoCompressionMaterial` | tension-only (cables, membranes, tendons) | `FromSI<Elastic>` |

Full real-citation grounding lives in each material's own source-file doc
comment (`src/matter/materials/*.rs`) — this table is for "which one do I
reach for," not the physics derivation.

---

## Rod (`spacetime::rod`) — for slender (length ≫ width) bodies

| Type / API | Meaning |
|---|---|
| `rod::Rod` | embedded in a `Simulation` — `points` (`RodPoints`), `material`, `wind_velocity`/`wind_drag_coeff`, `push_center`/`push_strength`/`push_radius` (read fresh every substep), `sleeping` |
| `rod::RodPoints` | own SoA — `x`/`v`/`mass`/`pinned`/`rest_edge_length`/`rest_curvature` |
| `rod::RodMaterial` | real `EA`/`EI` — `from_young_modulus_rectangular(E, width, thickness, axial_damping, bending_damping)`, `.critical_damping(l0, mass, ea, ei)` |
| `rod::build_straight_rod(start, end, n_points, linear_density, dx_meters)` | construct a straight `RodPoints` |
| `rod::rod_cfl_dt(&points, &material, safety)` | Gershgorin-bound CFL dt, folded into `choose_substep_dt` automatically |
| `Simulation::add_rod`/`with_rod`/`rods()`/`rods_mut()` | lifecycle, same fluent convention as `with_default_material` |
| `SimConfig::rod_sleep_threshold` | 0.0 = disabled; see Sleep/wake above |

---

## Force fields (`src/forces/fields/`)

| Field | Real basis |
|---|---|
| `NBodyGravity` | Barnes-Hut N-body gravitation |
| `GravityWell`, `RadialConfinement`, `AabbConfinement` | point/region attraction or bounding |
| `Coulomb` | electrostatic |
| `UniformEM` | uniform electromagnetic field |
| `LinearDragField` | Stokes/Rayleigh linear drag (wind, water currents) |
| `BuoyancyField` | Archimedes buoyancy — `Δv = g·(ρ_fluid/ρ_particle − 1)`, per-particle. Real calibration table in its own doc comment (wood/steel/ice vs water). |
| chemotaxis fields | gradient-following force along a `ScalarDiffusionField` |

## Boundary conditions (`src/forces/boundary/`)

`SlipBoundary`, `PredictiveBoundary`, `FrictionBoundary` — grid-level, applied
during grid update. See ARCHITECTURE.md §10 table for where each plugs in.

---

## Key APIs

```rust
// Construct
let mut solver = Simulation::new(config, SpawnRegion::for_sim(&config).with_material(WATER_ID));

// Spawn a body later (e.g. a creature born mid-run)
let creature_id = solver.add_body(SpawnRegion::for_sim(&config));

// Carve a shape out of one continuous lattice (no seams, no detachment risk —
// see feedback_puzzle_piece_architecture / basic_plant.rs's own history for
// why NEVER stack multiple separately-spawned SpawnRegions edge-to-edge)
solver.retain_particles(|p| /* keep predicate */ true);

// Phase transitions
solver.phase_transition(|p| p.temperature > 373.0, STEAM_ID);
solver.add_phase_rule(|p| if p.material_id == WATER && p.temperature < 273.0 { Some(ICE) } else { None });

// Neighbor queries
for idx in solver.particles_near(center, radius) { .. }
let n = solver.count_near(center, radius, FOOD_ID);

// Impulses
solver.apply_impulse(center, radius, force);
solver.apply_radial_impulse(center, radius, strength);

// Queries
solver.material_state(material_id) -> BodyState
solver.region_state(center, radius) -> BodyState
solver.particles() / particles_mut()
solver.diagnostics_snapshot() -> SimSnapshot   // min/max_deformation_j, total_kinetic_energy, max_pinned_particle_speed, ...
```

---

## Extension seams (from ARCHITECTURE.md §10)

| Seam | Trait | Applied |
|---|---|---|
| Constitutive response | `MaterialModel` / `ConstitutiveModel` + `PlasticityModel` | P2G stress |
| External body forces | `Field` | after G2P |
| Grid boundaries | `BoundaryCondition` | grid update |
| Multi-field contact | `Particle::contact_group` (opt-in, not a trait) | grid update |
| Scalar transport | `ScalarDiffusionField` | per substep |
| Phase change | phase rules (`Fn(&Particle) -> Option<u32>`) | per substep |
| Observation | `DiagnosticsRegistry` plugins | per step |

---

## Where things live (top-level docs, not duplicated here)

- `ARCHITECTURE.md` — how the engine works, narrative, start here for design intent.
- `CONTRIBUTING.md` — external-contributor module tree.
- `LP_MPM_SPEC.md` — design spec / LP integration contract.
- `PHYSICS_PROOFS.md` — what "visually correct" means per system, real gaps.

## Known gaps in this glossary (real, not filled in yet)

- Per-material real-citation summary (each material's own file has this; not
  yet pulled into one table here).
- Diagnostics plugin system (`DiagnosticsRegistry`, `RollingPlugin`, etc.) —
  not yet indexed.
- GPU-specific concepts (sparse-grid active-block dispatch, bind-group
  layout groups 0-3) — covered in `src/systems/gpu/pipeline.rs`'s own module
  doc, not duplicated here yet.
