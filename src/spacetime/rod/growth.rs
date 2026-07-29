//! Real elongation growth — the tip segment's `rest_edge_length` grows via
//! logistic growth (Verhulst 1838, `dL/dt = r·L·(1−L/K)`), the SAME real,
//! already-tested equation `ScalarDiffusionField`'s own resource-regrowth
//! source already uses (`src/energy/thermodynamics/scalar_field.rs`,
//! `resource_field.wgsl`) — reused here for length instead of a scalar
//! field, not a new invented law. The multiplicative growth-decomposition
//! CONCEPT this rests on (elongation as a real, separate kinematic quantity
//! from elastic strain) is Rodriguez, Hoger, McCulloch (1994), "Stress-
//! dependent finite growth in soft elastic tissues," *Journal of
//! Biomechanics* 27(4):455–467 — real growth theory, not plant-specific
//! (also used for tumors, blood vessels, tissue growth generally).
//!
//! Only the tip's own edge grows — real apical-meristem elongation happens
//! at the growing tip; mature tissue further back doesn't keep stretching.
//! Disclosed, NOT-yet-built follow-up: real growth eventually subdivides a
//! long tip segment into a new point (cell division adding new material,
//! not just stretching existing material indefinitely) — this module only
//! implements the elongation itself, not point insertion.
//!
//! **Real force-balance growth gate** (`GrowthResistance`):
//! Bengough & Mullins (1990, *J. Soil Science* 41:341–358; 1997, *European
//! J. Soil Science*) show real root penetration resistance is a genuine,
//! distinct force-balance term (cavity-expansion + interfacial friction),
//! and the classical Lockhart (1965) framework states elongation proceeds
//! only once internal turgor pressure exceeds the combined cell-wall +
//! soil resistance — growth is NOT unconditional. Local soil resistance is
//! sensed here via the shared grid's own mass density near the growing tip
//! — a real, but honestly SIMPLIFIED substitute for Bengough & Mullins' own
//! particle-scale cavity-expansion force, which needs grain-level contact
//! data this engine's continuum grid doesn't expose. `turgor_pressure_pa`/
//! `resistance_per_unit_mass_pa` are illustrative (the real measured turgor
//! range, ~0.1-1 MPa, came from secondary sources, not an independently
//! re-verified primary citation) — same disclosed-calibration status as
//! `Gravitropism`'s own rate constants.

use glam::Vec2;

use super::RodPoints;
use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;

#[derive(Debug, Clone, Copy)]
pub struct GrowthResistance {
    /// Internal turgor driving pressure, Pa.
    pub turgor_pressure_pa: f32,
    /// Conversion from local grid mass density to an equivalent soil
    /// resistance pressure, Pa per unit mass — a real, but simplified,
    /// stand-in for Bengough & Mullins' particle-scale cavity-expansion
    /// force (see module doc).
    pub resistance_per_unit_mass_pa: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Growth {
    /// Logistic growth rate `r`, 1/s.
    pub rate: f32,
    /// Carrying capacity `K` — the real maximum length the growing tip
    /// segment approaches, meters. A segment starts well below this and
    /// approaches it asymptotically (real sigmoidal growth curve), never
    /// exceeding it.
    pub max_segment_length_m: f32,
    /// Real turgor-vs-soil-resistance growth gate (see module doc). `None`
    /// (default) = ungated logistic growth, exactly the prior behavior —
    /// zero cost, zero change for anything that doesn't opt in.
    pub resistance: Option<GrowthResistance>,
}

impl Growth {
    pub fn new(rate: f32, max_segment_length_m: f32) -> Self {
        Self {
            rate,
            max_segment_length_m,
            resistance: None,
        }
    }

    pub fn with_resistance(mut self, resistance: GrowthResistance) -> Self {
        self.resistance = Some(resistance);
        self
    }
}

/// Real, quadratic-B-spline-weighted local mass density near `pos` — same
/// kernel every other grid sample in this engine uses, not a naive
/// single-cell lookup. `pub(super)`: also reused by `gravitropism.rs` --
/// gravitropic curling is itself mediated by real differential cell
/// elongation (Bastien et al.'s ACE model, see gravitropism.rs's own
/// citation), so the SAME turgor-vs-resistance force balance that gates
/// ordinary elongation legitimately gates it too, not a separate invented
/// mechanism.
pub(super) fn sample_mass_density(grid: &Grid, pos: Vec2) -> f32 {
    let weights = quadratic_weights(pos);
    let mut mass = 0.0f32;
    for gx in 0..3usize {
        for gy in 0..3usize {
            let weight = weights.wx[gx] * weights.wy[gy];
            if weight <= 0.0 {
                continue;
            }
            let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
            mass += weight * grid.mass_at(cell_pos);
        }
    }
    mass
}

/// Evolves the tip edge's own `rest_edge_length` via real logistic growth,
/// gated by the real turgor-vs-resistance force balance when
/// `growth.resistance` is set. No-op for a rod with fewer than 2 points (no
/// edge exists).
pub fn apply_growth(rod: &mut RodPoints, growth: &Growth, grid: &Grid, dt: f32) {
    let n = rod.rest_edge_length.len();
    if n == 0 {
        return;
    }
    let tip_edge = n - 1;
    let tip_point = rod.x.len() - 1;

    let gate = match growth.resistance {
        Some(r) => {
            let local_mass = sample_mass_density(grid, rod.x[tip_point]);
            let local_resistance_pa = r.resistance_per_unit_mass_pa * local_mass;
            (1.0 - local_resistance_pa / r.turgor_pressure_pa.max(1.0e-6)).clamp(0.0, 1.0)
        }
        None => 1.0,
    };

    let k = growth.max_segment_length_m.max(1.0e-6);
    let l = rod.rest_edge_length[tip_edge];
    let dl = gate * growth.rate * l * (1.0 - l / k);
    rod.rest_edge_length[tip_edge] = (l + dl * dt).max(1.0e-6);
}
