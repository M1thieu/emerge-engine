//! The domain's walls as this path's contact planes, and the friction
//! coefficient of each pair of surfaces in contact.

use super::*;
use crate::boundary::Surface;

/// The friction coefficients of a frame's contacts, owned: one for every
/// pair, or a table by pair of surfaces (`Friction::Pairs`).
pub(super) enum ContactFriction {
    Uniform(f64),
    Pairs {
        surfaces: usize,
        table: Vec<f64>,
        of_particle: Vec<u32>,
        setae: Vec<Setae>,
        axis_of_particle: Vec<DVec2>,
    },
}

impl ContactFriction {
    /// The view the step reads.
    pub(super) fn view(&self) -> Friction<'_> {
        match self {
            Self::Uniform(mu) => Friction::Uniform(*mu),
            Self::Pairs {
                surfaces,
                table,
                of_particle,
                setae,
                axis_of_particle,
            } => Friction::Pairs {
                surfaces: *surfaces,
                table,
                of_particle,
                setae,
                axis_of_particle,
            },
        }
    }

    /// The one coefficient, when every pair has the same.
    pub(super) fn uniform(&self) -> Option<f64> {
        match self {
            Self::Uniform(mu) => Some(*mu),
            Self::Pairs { .. } => None,
        }
    }
}

impl Simulation {
    /// The domain's walls as contact planes, each carrying its boundary's
    /// surface (`Plane::surface`).
    pub(super) fn total_lagrangian_planes(&self) -> Vec<Plane> {
        let cell = f64::from(self.config.dx_meters);
        let res = self.config.grid_res as f64 * cell;
        let mut planes = Vec::new();
        for (b, wall) in self
            .boundaries
            .iter()
            .enumerate()
            .filter_map(|(b, boundary)| Some((b, boundary.domain_walls()?)))
        {
            let t = wall.thickness as f64 * cell;
            for axis in 0..2 {
                for (at, inward) in [(t, 1.0), (res - t, -1.0)] {
                    planes.push(Plane {
                        axis,
                        at,
                        inward,
                        speed: 0.0,
                        // The boundary's surface, after the materials'
                        // (`total_lagrangian_friction`).
                        surface: (self.materials.len() + b) as u32,
                    });
                }
            }
        }
        planes
    }

    /// The domain's walls as contact planes and their one friction
    /// coefficient, for the callers that take one (the settle, the probes):
    /// refused when the walls give different ones.
    pub(super) fn total_lagrangian_walls(&self) -> (Vec<Plane>, f64) {
        let planes = self.total_lagrangian_planes();
        let mut friction = None::<f32>;
        for wall in self.boundaries.iter().filter_map(|b| b.domain_walls()) {
            match friction {
                Some(mu) if mu != wall.friction => panic!(
                    "the walls give friction {mu} and {}; this caller takes one coefficient \
                     for every contact",
                    wall.friction
                ),
                _ => friction = Some(wall.friction),
            }
        }
        (planes, f64::from(friction.unwrap_or(0.0)))
    }

    /// Each contact's friction coefficient by the pair of surfaces it
    /// joins. A wall against a material: the coefficient the wall declares
    /// (`DomainWalls::friction`), unless a pair set by `set_surface_friction`
    /// replaces it. Two materials: the pair set for them; with none set, the
    /// walls' coefficient when they all give one (the coefficient every
    /// contact took before pairs existed, kept as a declared convention:
    /// kind: convention; why: a scene that names no pair keeps its result;
    /// removal: every scene names its pairs), refused otherwise.
    /// With no pair set and one wall coefficient, one coefficient for all.
    /// A pair set by `set_surface_setae` replaces the table's entry, its
    /// axis carried by each particle's deformation (`F axis`) at the
    /// frame's start (kind: discretization; why: the table is built once
    /// per frame, and only the axis's sense along the contact's tangent
    /// enters the law, which turns only when the axis crosses the contact
    /// normal; removal: the axis read at each step's start).
    pub(super) fn total_lagrangian_friction(
        &self,
        state: &TotalLagrangianState,
    ) -> ContactFriction {
        let walls: Vec<(usize, f64)> = self
            .boundaries
            .iter()
            .enumerate()
            .filter_map(|(b, boundary)| Some((b, f64::from(boundary.domain_walls()?.friction))))
            .collect();
        let common = walls
            .first()
            .map(|&(_, mu)| mu)
            .filter(|&mu| walls.iter().all(|&(_, other)| other == mu));
        if self.surface_friction.is_empty()
            && self.surface_setae.is_empty()
            && let Some(mu) = common.or(walls.is_empty().then_some(0.0))
        {
            return ContactFriction::Uniform(mu);
        }
        let materials = self.materials.len();
        let surfaces = materials + self.boundaries.len();
        let index = |s: Surface| match s {
            Surface::Material(id) => id as usize,
            Surface::Boundary(b) => materials + b,
        };
        let mut table = vec![f64::NAN; surfaces * surfaces];
        let mut set = |a: usize, b: usize, mu: f64| {
            table[a * surfaces + b] = mu;
            table[b * surfaces + a] = mu;
        };
        for &(b, mu) in &walls {
            for m in 0..materials {
                set(m, materials + b, mu);
            }
        }
        if let Some(mu) = common {
            for a in 0..materials {
                for b in 0..materials {
                    set(a, b, mu);
                }
            }
        }
        for &(a, b, mu) in &self.surface_friction {
            set(index(a), index(b), mu);
        }
        let setae: Vec<Setae> = self
            .surface_setae
            .iter()
            .map(|s| {
                assert!(
                    !self.surface_setae.iter().any(|t| {
                        t.other == Surface::Material(s.material)
                            && s.other == Surface::Material(t.material)
                    }),
                    "materials {} and {:?} both carry setae against each other: the pair has no one sense of sliding",
                    s.material,
                    s.other
                );
                let (a, b) = (s.material as usize, index(s.other));
                // The table's entry only marks the pair as named; the setae
                // replace it (`Friction::of`).
                set(a, b, s.forward.max(s.backward));
                Setae {
                    surface: a as u32,
                    other: b as u32,
                    forward: s.forward,
                    backward: s.backward,
                }
            })
            .collect();
        // The solids' particles, then the bodies' contact points, each of
        // its body's material (`TotalLagrangianState::outlines`, in the
        // order of `total_lagrangian_outlines`).
        let mut of_particle = self.particles.material_id.clone();
        of_particle.resize(state.x.len(), 0);
        for (body, points) in self.total_lagrangian_outlines.iter().zip(&state.outlines) {
            let mut own = (0..state.particles)
                .filter(|&q| self.particles.user_tag[q] == body.tag)
                .map(|q| self.particles.material_id[q]);
            let material = own.next().unwrap_or(0);
            assert!(
                own.all(|m| m == material),
                "body {} is made of several materials: its surface has no one friction coefficient per pair yet",
                body.tag
            );
            for &q in points {
                of_particle[q] = material;
            }
        }
        let mut present = of_particle.clone();
        present.sort_unstable();
        present.dedup();
        for &m in &present {
            for &n in &present {
                assert!(
                    !table[m as usize * surfaces + n as usize].is_nan(),
                    "no friction coefficient for materials {m} and {n}: the walls give different \
                     ones, so name the pair (Simulation::set_surface_friction)"
                );
            }
        }
        let axis_of_particle = self.total_lagrangian_setae_axes(state, &of_particle);
        ContactFriction::Pairs {
            surfaces,
            table,
            of_particle,
            setae,
            axis_of_particle,
        }
    }

    /// Each point's setae axis in the current configuration, `F axis` with
    /// `F` its own deformation gradient, for the points of a material that
    /// has setae (`set_surface_setae`); zero elsewhere. A body's contact
    /// point beyond the particles takes the `F` of its body's particle
    /// nearest in the reference configuration (kind: discretization; why:
    /// the sense of the axis is what the law reads; removal: the contact
    /// points' own deformation gradient).
    fn total_lagrangian_setae_axes(
        &self,
        state: &TotalLagrangianState,
        of_particle: &[u32],
    ) -> Vec<DVec2> {
        if self.surface_setae.is_empty() {
            return Vec::new();
        }
        let axis_of = |material: u32| {
            self.surface_setae
                .iter()
                .find(|s| s.material == material)
                .map(|s| s.axis)
        };
        let mut axes = vec![DVec2::ZERO; state.x.len()];
        for q in 0..state.x.len() {
            let Some(axis) = axis_of(of_particle[q]) else {
                continue;
            };
            let p = if q < state.particles {
                q
            } else {
                let body = self
                    .total_lagrangian_outlines
                    .iter()
                    .zip(&state.outlines)
                    .find(|(_, points)| points.contains(&q))
                    .map(|(body, _)| body.tag)
                    .expect("a contact point beyond the particles belongs to a body");
                (0..state.particles)
                    .filter(|&p| self.particles.user_tag[p] == body)
                    .min_by(|&a, &b| {
                        let d =
                            |p: usize| (state.reference[p] - state.reference[q]).length_squared();
                        d(a).total_cmp(&d(b))
                    })
                    .expect("a body has particles")
            };
            axes[q] = state.f[p] * axis;
        }
        axes
    }
}
