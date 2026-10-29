//! The domain's walls as this path's contact planes (moved here from
//! `total_lagrangian_solids.rs` unchanged).

use super::*;

impl Simulation {
    /// The domain's walls as contact planes and their friction coefficient
    /// (one for every contact, walls and bodies alike: declared).
    pub(super) fn total_lagrangian_walls(&self) -> (Vec<Plane>, f64) {
        let cell = f64::from(self.config.dx_meters);
        let res = self.config.grid_res as f64 * cell;
        let mut planes = Vec::new();
        let mut friction = None::<f32>;
        for wall in self.boundaries.iter().filter_map(|b| b.domain_walls()) {
            let t = wall.thickness as f64 * cell;
            for axis in 0..2 {
                planes.push(Plane {
                    axis,
                    at: t,
                    inward: 1.0,
                });
                planes.push(Plane {
                    axis,
                    at: res - t,
                    inward: -1.0,
                });
            }
            match friction {
                Some(mu) if mu != wall.friction => panic!(
                    "the total Lagrangian solid path takes one friction coefficient for every contact; \
                     the walls give {mu} and {}",
                    wall.friction
                ),
                _ => friction = Some(wall.friction),
            }
        }
        (planes, f64::from(friction.unwrap_or(0.0)))
    }
}
