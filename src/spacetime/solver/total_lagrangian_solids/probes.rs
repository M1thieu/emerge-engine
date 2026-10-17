//! Probes of the path inside `Simulation`.

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// The frictionless slide of `tests/total_lagrangian_path.rs` with
/// every Newton iteration of frame `EMERGE_PROBE_TL_FRAME` (default 3)
/// printed. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_wired_slide_newton() {
    super::cap::capped("probe_wired_slide_newton", 1800, || {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let props = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 2.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        for v in sim.particles_mut().v.iter_mut() {
            *v = Vec2::new(50.0, 0.0);
        }
        let watched: u64 = std::env::var("EMERGE_PROBE_TL_FRAME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        for frame in 1..=watched {
            if frame == watched {
                crate::spacetime::total_lagrangian::hooks::start(0);
            }
            sim.step();
            if frame == watched {
                for (solve, lines) in crate::spacetime::total_lagrangian::hooks::finish()
                    .iter()
                    .enumerate()
                {
                    for line in lines {
                        println!("  solve {solve}: {line}");
                    }
                }
            }
            println!("frame {frame}: {:?}", sim.total_lagrangian_frame());
        }
    });
}
