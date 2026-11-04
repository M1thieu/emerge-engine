//! The explicit path (`Simulation::step` without `total_lagrangian`, the
//! deprecated MLS-MPM solver) on `impact_local`'s clay drop: the same
//! material, block and frames, for EXPLICIT_BASELINE_PLAN.md. A probe:
//! measurement only, the explicit solver unchanged.

use crate::materials::physical_props::Elastic as ElasticProps;
use crate::materials::solid::corotated::stress::corotated_elastic_energy_density;
use crate::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{DMat2, IVec2, Vec2};

fn clay() -> ElasticProps {
    ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    }
}

/// Per frame of the explicit path: its substeps, the simulated time it
/// dropped, its wall time, the centre's height above the floor and vertical
/// velocity, the lowest particle's height, the kinetic, elastic and gravity
/// energies (J/m, evaluated at `f64` from the `f32` state) and the internal
/// energy (kinetic about the centre of mass plus elastic): the stand-in for
/// `impact_local`'s modal vibration energy, the explicit path having no body
/// modes.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_explicit_clay_drop() {
    super::cap::capped("probe_explicit_clay_drop", 7200, || {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        // `EMERGE_PROBE_EXPLICIT_BUDGET`: the per-frame substep budget
        // (`max_substeps_per_step`, a work cap, not a physics parameter by
        // its doc), raised so the whole frame is simulated; the default
        // otherwise (declared in the result).
        if let Some(budget) = std::env::var("EMERGE_PROBE_EXPLICIT_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            config.max_substeps_per_step = budget;
        }
        let mut sim = Simulation::empty(config);
        let material = CorotatedMaterial::from_physical(&clay(), &config);
        let (lambda, mu) = (f64::from(material.lambda), f64::from(material.mu));
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&clay(), &config),
        );
        let dx = f64::from(config.dx_meters);
        let unit = f64::from(config.grid_mass_unit_kg_per_m()) * dx * dx;
        let g = f64::from(config.gravity.length());
        let floor_m = f64::from(floor) * dx;
        let state = |sim: &Simulation| -> [f64; 7] {
            let p = sim.particles();
            let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
            let (mut y, mut vx, mut vy, mut kinetic, mut elastic, mut gravity) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let mut lowest = f64::INFINITY;
            for i in 0..p.x.len() {
                let m = f64::from(p.mass[i]);
                let (x, v) = (p.x[i].as_dvec2(), p.v[i].as_dvec2());
                y += m * x.y;
                vx += m * v.x;
                vy += m * v.y;
                kinetic += 0.5 * m * v.length_squared();
                gravity += m * g * x.y;
                let f = DMat2::from_cols(
                    p.deformation_gradient[i].x_axis.as_dvec2(),
                    p.deformation_gradient[i].y_axis.as_dvec2(),
                );
                elastic += f64::from(p.initial_volume[i])
                    * corotated_elastic_energy_density(f, lambda, mu);
                lowest = lowest.min(x.y);
            }
            let centre_v2 = (vx * vx + vy * vy) / (mass * mass);
            [
                (y / mass) * dx - floor_m,
                vy / mass * dx,
                lowest * dx - floor_m,
                kinetic * unit,
                elastic * unit,
                gravity * unit,
                (kinetic - 0.5 * mass * centre_v2 + elastic) * unit,
            ]
        };
        let start = state(&sim);
        let total0 = start[3] + start[4] + start[5];
        println!(
            "explicit: {} particles, lambda {lambda:.4e} mu {mu:.4e} (grid units), energy unit {unit:.4e} J/m",
            sim.particles().len()
        );
        for n in 0..45 {
            let clock = std::time::Instant::now();
            sim.step();
            let wall = clock.elapsed().as_secs_f64();
            let s = state(&sim);
            let total = s[3] + s[4] + s[5];
            println!(
                "explicit frame {n}: substeps {} dropped {:.3e} s wall {wall:.4} s; height {:.6e} m, vy {:+.6e} m/s, lowest {:+.4e} m; kinetic {:.6e} elastic {:.6e} gravity {:.6e} internal {:.4e} J/m; energy {:+.4e} of the start",
                sim.last_substeps(),
                sim.last_sim_time_dropped,
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                s[5],
                s[6],
                total / total0 - 1.0
            );
        }
    });
}
