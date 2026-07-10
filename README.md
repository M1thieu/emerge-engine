# emerge

[![crates.io](https://img.shields.io/crates/v/emerge-engine.svg)](https://crates.io/crates/emerge-engine)
[![docs.rs](https://docs.rs/emerge-engine/badge.svg)](https://docs.rs/emerge-engine)
[![license](https://img.shields.io/crates/l/emerge-engine.svg)](LICENSE-MIT)

An MLS-MPM continuum solver (Hu et al. 2018). Fluids, sand, snow, elastic and plastic solids — one particle-grid transfer for all of them. No rigid bodies, no separate fluid/cloth/soft-body systems bolted together. Pure Rust on the CPU path; an optional wgpu backend runs the whole pipeline on GPU.

Built for [Life's Progress](https://github.com/erematorg/LP). Not a game engine — no ECS, no game loop, no asset pipeline. It steps particles forward and answers queries about regions of space; everything else is up to the caller.

```toml
[dependencies]
emerge = { package = "emerge-engine", version = "0.1" }
# GPU compute, all plasticity included:
emerge = { package = "emerge-engine", version = "0.1", features = ["gpu"] }
```

## Quick start

```rust
use emerge::prelude::*;

const WATER: u32 = 1;

let config = SimConfig::standard(64, 0.05, Vec2::NEG_Y);

let mut sim = Simulation::empty(config)
    .with_default_material(Box::new(NeoHookeanMaterial::new(400.0, 200.0)))
    .with_material(WATER, Box::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1e4)))
    .with_boundary(Box::new(SlipBoundary::new(2)));

let _ = sim.add_body(SpawnRegion {
    box_size: IVec2::new(12, 12),
    box_center: Vec2::new(24.0, 40.0),
    precompute_initial_volumes: true,
    ..SpawnRegion::for_sim(&config)
});

let _ = sim.add_body(SpawnRegion {
    box_size: IVec2::new(12, 8),
    box_center: Vec2::new(40.0, 36.0),
    material_id: WATER,
    precompute_initial_volumes: true,
    ..SpawnRegion::for_sim(&config)
});

sim.step_n(60);

let state = sim.region_state(Vec2::new(40.0, 36.0), 10.0);
println!("avg speed: {:.3}", state.avg_speed);
```

## Materials

Twelve constitutive models, grouped by what they're for:

| Group | Models |
|---|---|
| **Elastic solids** | `NeoHookeanMaterial` (finite-strain), `CorotatedMaterial` (stiffer, corotated-linear), `ViscoelasticMaterial` (Kelvin-Voigt) |
| **Fluids** | `NewtonianFluidMaterial` (Tait EOS + viscosity), `BinghamFluidMaterial` (adds a yield stress — mud, not water) — both take `surface_tension_coeff` for free |
| **Granular** | `StomakhinMaterial` (snow), `DruckerPragerMaterial` / `MuIRheologyMaterial` (two ways to get sand right), `GranularFluidMaterial` (granular suspensions) |
| **Plastic / failure** | `VonMisesMaterial` (ductile), `RankineMaterial` (brittle, damage softening), `NaccMaterial` (Cam-Clay soil) |

Each cites its source paper in the doc comment — see [Physics references](#physics-references).

## Features

- `gpu` — the whole pipeline (P2G, grid update, G2P, every plasticity model) as WGSL compute
- `render` — instanced particle debug renderer, on top of `gpu`
- `experimental` — acoustics, electromagnetics, information-theoretic measures (real, just not API-stable yet)

## Examples

```sh
cargo run --example headless                        # no feature flags, start here
cargo run --example basic_sand      --features render
cargo run --example basic_fluids    --features render
cargo run --example basic_snow      --features render
cargo run --example basic_jellies   --features render
cargo run --example basic_creature  --features render  # LNN-driven muscle locomotion
cargo run --example basic_showcase  --features render  # three materials at once
cargo run --example basic_sand_gpu  --features render
```

Windowed examples (everything except `headless` and `validate_materials`) need `--features render` — they draw via wgpu/winit directly, no Bevy.

## Physics references

| Module | Paper |
|---|---|
| MLS-APIC transfer | Hu et al. 2018, *A Moving Least Squares Material Point Method* |
| NeoHookean / Corotated | Stomakhin et al. 2012, *Energetically Consistent Invertible Elasticity* |
| Snow | Stomakhin et al. 2013, *A Material Point Method for Snow Simulation* |
| Sand | Klar et al. 2016, *Drucker-Prager Elastoplasticity for Sand Animation* |
| µ(I)-rheology | Dunatunga & Kamrin 2015, *Continuum modelling and simulation of granular flow* |
| Surface tension | Stomakhin et al. 2014, *Augmented MPM for cloth and soft bodies* |
| N-body gravity | Barnes & Hut 1986, *A hierarchical O(N log N) force-calculation algorithm* |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
