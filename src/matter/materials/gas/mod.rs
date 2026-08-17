//! Gas-state materials — real compressible ideal-gas equation of state,
//! genuinely different from a liquid's Tait EOS: pressure vanishes as
//! density does (no rest-pressure offset), and real shock-capturing
//! matters far more than for a weakly-compressible liquid.
//!
//! PLACEHOLDER (2026-08-17): folder scaffolding for the real
//! solid/liquid/gas/plasma/mixture taxonomy restructuring of
//! `matter/materials/` — not yet wired into `materials/mod.rs`. Confirmed
//! real and tractable via MPM literature (Guilkey et al., *"An
//! Introduction to the Material Point Method using a Case Study from Gas
//! Dynamics"*) — see the design artifact for the full plan:
//! <https://claude.ai/code/artifact/90290560-8992-4d7c-ae4b-11ede12a737f>
//!
//! Real math already shipped and verified (2026-08-17), NOT yet a
//! `MaterialModel`: `energy::thermodynamics::ideal_gas` — `p=ρRT`, real
//! adiabatic sound speed, matches the real ~343 m/s air reference. Real
//! next steps before a `GasMaterial` lands here: shock-capturing tuned for
//! gas (reuse `matter::materials::liquid::fluid::artificial_bulk_viscosity`, von
//! Neumann & Richtmyer 1950, already shock-capturing for liquids),
//! verified against Sod's shock tube (Toro, *Riemann Solvers and
//! Numerical Methods for Fluid Dynamics* — the standard exact-analytical-
//! solution benchmark for a compressible-gas solver).
