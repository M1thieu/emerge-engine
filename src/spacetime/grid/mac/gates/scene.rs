use super::*;

/// Particles and the still tank they sit in, stepped by the standard
/// projection alone. Lengths are in cells (`dx = 1`); `cell_m` gives the
/// metres per cell.
pub(super) struct Scene {
    pub(super) layout: MacLayout,
    pub(super) cell_m: f32,
    /// Cells per second squared.
    pub(super) gravity: Vec2,
    pub(super) tank: (Vec2, Vec2),
    pub(super) weights: FaceWeights,
    pub(super) solid_centres: Field2,
    pub(super) surface: SurfaceSettings,
    pub(super) settings: ProjectionSettings,
    pub(super) x: Vec<Vec2>,
    pub(super) v: Vec<Vec2>,
    pub(super) c: Vec<Mat2>,
    pub(super) mass: Vec<f32>,
    pub(super) v0: Vec<f32>,
    pub(super) log_j: Vec<f32>,
    pub(super) j_at_bounds: usize,
    pub(super) time: f32,
    pub(super) probe: Probe,
    /// `c^2` in cells^2/s^2 for the compressible projection;
    /// `None` is the incompressible limit.
    pub(super) sound_speed2: Option<f32>,
    /// A substep length that replaces the travel limit (the sound-speed
    /// tube).
    pub(super) fixed_dt: Option<f32>,
    /// `mu / rho` in cells^2/s^2 for an elastic solid; `None`
    /// for a liquid.
    pub(super) shear: Option<f32>,
    /// A solid moves by the energy-momentum conserving implicit step
    /// (`conserving`) instead of the explicit one.
    pub(super) conserving: Option<crate::spacetime::total_lagrangian::ConservingSettings>,
    /// One material per particle for the conserving step, in place of the
    /// scene's one (a probe's several bodies).
    pub(super) solids: Option<Vec<crate::spacetime::total_lagrangian::Elastic>>,
    /// The conserving step's contact refinement, a declared stepping stone
    /// (`contact_substep`): its substep in P wave steps, `0.5 dx / c_p`;
    /// set by a probe.
    pub(super) refine_contact: Option<f32>,
    /// Each particle's deformation gradient, for the shear stress.
    pub(super) deformation: Vec<Mat2>,
    /// The conserving step's reference configuration (taken at its first
    /// step) and its grid, kept across steps (`conserving_substep`).
    pub(super) lagrangian: Option<(
        Vec<glam::DVec2>,
        Option<crate::spacetime::total_lagrangian::LagrangianCache>,
    )>,
    /// The first of the bodies' contact points (`with_contact_outline`):
    /// only they meet the walls. `None`: every particle does.
    pub(super) contact_from: Option<usize>,
}

/// What one frame's last substep left, and what the frame cost.
pub(super) struct Frame {
    pub(super) pressure: Vec<f32>,
    pub(super) active: Vec<bool>,
    pub(super) phi: Field2,
    /// Face velocity the last substep's particles gathered from.
    pub(super) vel: super::field::MacVelocity,
    pub(super) substeps: u32,
    pub(super) iterations: Vec<u32>,
    pub(super) capped: u32,
    pub(super) largest_travel: f32,
}

impl Scene {
    pub(super) fn new(
        (nx, ny): (usize, usize),
        cell_m: f32,
        g_fraction: f32,
        tank: (Vec2, Vec2),
        x: Vec<Vec2>,
    ) -> Self {
        let layout = MacLayout::new(nx, ny, 1.0);
        let solid = box_container(tank.0, tank.1);
        let corners = sample_corners(&layout, solid);
        let spacing = 0.5 * layout.dx;
        let n = x.len();
        Self {
            layout,
            cell_m,
            gravity: Vec2::new(0.0, -g_fraction * EARTH_GRAVITY_M_S2 / cell_m),
            tank,
            weights: face_weights(&layout, &corners),
            solid_centres: sample_centres(&layout, solid),
            surface: SurfaceSettings::from_spacing(spacing),
            settings: ProjectionSettings::default(),
            x,
            v: vec![Vec2::ZERO; n],
            c: vec![Mat2::ZERO; n],
            mass: vec![spacing * spacing; n],
            v0: vec![spacing * spacing; n],
            log_j: vec![0.0; n],
            j_at_bounds: 0,
            time: 0.0,
            probe: Probe::default(),
            sound_speed2: None,
            fixed_dt: None,
            shear: None,
            conserving: None,
            solids: None,
            refine_contact: None,
            deformation: vec![Mat2::IDENTITY; n],
            lagrangian: None,
            contact_from: None,
        }
    }

    pub(super) fn g(&self) -> f32 {
        self.gravity.length()
    }

    /// A plane-strain elastic solid of Young's modulus over density
    /// `e_over_rho` (m^2/s^2) and Poisson's ratio `nu` (`with_lame`).
    pub(super) fn with_elastic(self, e_over_rho: f32, nu: f32) -> Self {
        let mu = e_over_rho / (2.0 * (1.0 + nu));
        let lambda = e_over_rho * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        self.with_lame(lambda, mu)
    }

    /// A material by its plane-strain Lame coefficients over its density,
    /// `lambda / rho` and `mu / rho` in m^2/s^2. With `mu > 0` a solid: its
    /// whole stress, shear and volume, through its particles (module doc:
    /// a solid is not projected);
    /// with `mu = 0` a liquid, its volume in the compressible projection.
    /// Adds contact points of no mass and no volume on the bodies'
    /// surface, which alone meet the walls in the conserving step. A
    /// contact holds a body's surface, as the node-to-surface contact of
    /// Laursen and Chawla, IJNME 40 (1997), sec. 3, holds its slave
    /// surface's nodes: a particle inside the body held where it crossed a
    /// wall would be held there while its neighbours move (measured: the
    /// very soft clay's second, corner-first impact blew up with every
    /// particle held).
    pub(super) fn with_contact_outline(mut self, points: Vec<Vec2>) -> Self {
        self.contact_from = Some(self.x.len());
        for q in points {
            self.x.push(q);
            self.v.push(Vec2::ZERO);
            self.c.push(Mat2::ZERO);
            self.mass.push(0.0);
            self.v0.push(0.0);
            self.log_j.push(0.0);
            self.deformation.push(Mat2::IDENTITY);
        }
        self
    }

    pub(super) fn with_lame(mut self, lambda_over_rho: f32, mu_over_rho: f32) -> Self {
        let cells2 = 1.0 / (self.cell_m * self.cell_m);
        self.shear = (mu_over_rho > 0.0).then_some(mu_over_rho * cells2);
        // The shear part acts on the isochoric deformation only, so all
        // the volume stiffness is the volume part's (the particles' for a
        // solid, the projection's for a liquid): the
        // plane-strain bulk modulus `kappa = lambda + mu`. With it, `2 mu dev(e) + kappa tr(e)
        // I = 2 mu e + lambda tr(e) I`, Lame's law; with `lambda` alone, as
        // Stomakhin et al.'s split writes the volume energy, the P wave ran
        // at `sqrt((lambda + mu) / rho)`: 6.0 percent late on the clays of
        // the matrix (`nu = 0.45`), against water's 1.0.
        self.sound_speed2 = Some((lambda_over_rho + mu_over_rho) * cells2);
        // Probe switch: a solid moves by the conserving implicit step, at
        // one frame per substep (`frame`) or at the explicit P wave limit
        // (`p`), when `EMERGE_PROBE_CONSERVING` names one.
        if let (Some(mu), Ok(step)) = (self.shear, std::env::var("EMERGE_PROBE_CONSERVING")) {
            let defaults = crate::spacetime::total_lagrangian::ConservingSettings::default();
            self.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
                occupation: std::env::var("EMERGE_PROBE_OCCUPATION")
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .or(defaults.occupation),
                ..defaults
            });
            let c_p = (self.sound_speed2.unwrap_or(0.0) + mu).sqrt();
            let step = Substep::parse(&step).unwrap_or(Substep::Frame);
            self.fixed_dt = Some(step.length(self.layout.dx, c_p));
        }
        self
    }

    /// The compressible projection at `c_m_s`.
    pub(super) fn with_sound_speed(mut self, c_m_s: f32) -> Self {
        let c = c_m_s / self.cell_m;
        self.sound_speed2 = Some(c * c);
        self
    }

    /// Mass-weighted J at the cell centres, bilinear, and the pressure the
    /// linear equation of state gives it, `q = -c^2 (J - 1)`; zero where no
    /// particle reaches.
    pub(super) fn cell_pressure_from_j(&self, sound_speed2: f32) -> Vec<f32> {
        let (nx, ny) = (self.layout.nx, self.layout.ny);
        let mut sum_j = vec![0.0f32; nx * ny];
        let mut sum_w = vec![0.0f32; nx * ny];
        for p in 0..self.x.len() {
            let t = self.x[p] / self.layout.dx - Vec2::splat(0.5);
            let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
            let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
            let j_p = self.j(p);
            for (di, dj, w) in [
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let (i, j) = (i0 + di, j0 + dj);
                if i < 0 || j < 0 || i as usize >= nx || j as usize >= ny {
                    continue;
                }
                let c = i as usize + nx * j as usize;
                sum_w[c] += w * self.mass[p];
                sum_j[c] += w * self.mass[p] * j_p;
            }
        }
        (0..nx * ny)
            .map(|c| {
                if sum_w[c] > 0.0 {
                    -sound_speed2 * (sum_j[c] / sum_w[c] - 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The liquid level set with the tank's walls folded in (`apic2d`).
    pub(super) fn phi(&self) -> Field2 {
        let mut phi = liquid_phi(&self.layout, &self.x, &self.surface);
        for (value, &solid) in phi.data_mut().iter_mut().zip(self.solid_centres.data()) {
            *value = value.min(solid);
        }
        phi
    }

    pub(super) fn substep(&mut self, dt: f32, frame: &mut Frame) {
        if let (Some(mu), Some(settings)) = (self.shear, self.conserving) {
            self.conserving_substep(mu, settings, dt, frame);
            return;
        }
        let mut clock = Instant::now();
        let mut lap = |probe: &mut Probe, k: usize| {
            probe.phase_us[k] += clock.elapsed().as_micros();
            clock = Instant::now();
        };
        let (mut vel, face_mass) =
            particles_to_faces(&self.layout, &self.x, &self.v, &self.c, &self.mass);
        if let Some(mu) = self.shear {
            // The whole elastic stress on the particles (module doc, a solid is
            // not projected): the shear part on the isochoric deformation and
            // the volume part `K J (J - 1) I`, `J = det F`, with `K` the
            // plane-strain bulk modulus `with_lame` keeps.
            let bulk = self.sound_speed2.unwrap_or(0.0);
            let tau_v0: Vec<Mat2> = (0..self.x.len())
                .map(|p| {
                    let f = self.deformation[p];
                    let j = f.determinant();
                    let volume = bulk * j * (j - 1.0) * Mat2::IDENTITY;
                    self.v0[p] * (shear_kirchhoff(f, mu) + volume)
                })
                .collect();
            let mut force = stress_to_faces(&self.layout, &self.x, &tau_v0);
            let mut lent = super::field::MacVelocity {
                u: face_mass.u.clone(),
                v: face_mass.v.clone(),
            };
            let image = box_container_image(self.tank.0, self.tank.1);
            fold_onto_images(&self.layout, &mut force, &self.weights, &image, true);
            fold_onto_images(&self.layout, &mut lent, &self.weights, &image, false);
            for (field, f, m) in [
                (&mut vel.u, &force.u, &lent.u),
                (&mut vel.v, &force.v, &lent.v),
            ] {
                for ((value, &f), &m) in field.data_mut().iter_mut().zip(f.data()).zip(m.data()) {
                    if m > 0.0 {
                        *value += dt * f / m;
                    }
                }
            }
        }
        for value in vel.u.data_mut() {
            *value += self.gravity.x * dt;
        }
        for value in vel.v.data_mut() {
            *value += self.gravity.y * dt;
        }
        lap(&mut self.probe, 0);
        let phi = self.phi();
        lap(&mut self.probe, 1);
        let system = super::pressure::assemble(
            &self.layout,
            dt,
            &vel,
            &self.weights,
            super::pressure::Surface {
                phi: &phi,
                theta_floor: self.settings.theta_floor,
            },
        );
        lap(&mut self.probe, 2);
        let image = box_container_image(self.tank.0, self.tank.1);
        let solution = if self.shear.is_some() {
            let mut valid = super::field::FaceFlags {
                u: face_mass.u.data().iter().map(|&m| m > 0.0).collect(),
                v: face_mass.v.data().iter().map(|&m| m > 0.0).collect(),
            };
            // A solid is not projected (module doc). The wall
            // condition the projection's variational form holds (`pressure`
            // module doc) is then the grid's own, as standard MPM sets it:
            // a face on a wall plane has no normal velocity and keeps none
            // through the extension. Without it a settling column crossed
            // the floor by 3704 ulps in its first frame.
            let (lo, hi) = self.tank;
            let on_plane = |a: f32, b: f32| (a - b).abs() <= 1e-6 * self.layout.dx;
            for j in 0..self.layout.ny {
                for i in 0..=self.layout.nx {
                    let q = self.layout.u_position(i, j);
                    if on_plane(q.x, lo.x) || on_plane(q.x, hi.x) {
                        vel.u.set(i, j, 0.0);
                        valid.u[vel.u.index(i, j)] = true;
                    }
                }
            }
            for j in 0..=self.layout.ny {
                for i in 0..self.layout.nx {
                    let q = self.layout.v_position(i, j);
                    if on_plane(q.y, lo.y) || on_plane(q.y, hi.y) {
                        vel.v.set(i, j, 0.0);
                        valid.v[vel.v.index(i, j)] = true;
                    }
                }
            }
            super::extrapolate::extrapolate_velocity(
                &mut vel,
                &mut valid,
                self.settings.extrapolation_layers,
            );
            super::extrapolate::constrain_to_solids(&self.layout, &mut vel, &self.weights, &image);
            super::pcg::Solution {
                pressure: vec![0.0; self.layout.nx * self.layout.ny],
                iterations: 0,
                residual: 0.0,
                converged: true,
            }
        } else {
            let q_before = self
                .sound_speed2
                .map(|c2| (c2, self.cell_pressure_from_j(c2)));
            let material: Vec<bool> = self.solid_centres.data().iter().map(|&d| d > 0.0).collect();
            project(
                &self.layout,
                dt,
                &mut vel,
                Walls {
                    weights: &self.weights,
                    image: &image,
                    open: &material,
                },
                Liquid {
                    phi: &phi,
                    face_mass: &face_mass,
                },
                &self.settings,
                q_before
                    .as_ref()
                    .map(|(c2, q)| (*c2, q.as_slice(), material.as_slice())),
            )
        };
        lap(&mut self.probe, 3);
        faces_to_particles(&self.layout, &vel, &self.x, &mut self.v, &mut self.c);
        // Positions advance by the midpoint rule (RK2) through the projected
        // face velocity, as Bridson and Muller-Fischer 3.1 recommend over
        // forward Euler for trajectories. Against a wall the normal velocity
        // falls linearly to zero, `v = -a d`: forward Euler moves `d` to `d (1
        // - a dt)`, which crosses the wall once `a dt > 1`, while the midpoint
        // rule gives `d (1 - a dt + (a dt)^2 / 2)`, positive for every step.
        // After the corner fix, 71 of the 72 remaining crossings were water
        // decelerating against the lid within one cell. The particle keeps
        // the velocity and `C` gathered at its start, as before.
        let midpoint: Vec<Vec2> = (0..self.x.len())
            .map(|p| self.x[p] + 0.5 * dt * self.v[p])
            .collect();
        let mut v_mid = vec![Vec2::ZERO; self.x.len()];
        let mut c_mid = vec![Mat2::ZERO; self.x.len()];
        faces_to_particles(&self.layout, &vel, &midpoint, &mut v_mid, &mut c_mid);
        lap(&mut self.probe, 4);
        if self.shear.is_some() {
            for p in 0..self.x.len() {
                self.deformation[p] = deformation_step(dt * self.c[p]) * self.deformation[p];
            }
        }
        if let Some((x0, y_lo, y_hi)) = self.probe.watch_x {
            let (sum, n) = (0..self.x.len())
                .filter(|&p| {
                    (self.x[p].x - x0).abs() < self.layout.dx && (y_lo..y_hi).contains(&self.x[p].y)
                })
                .fold((0.0f32, 0u32), |(s, n), p| {
                    let v = if self.probe.watch_longitudinal {
                        self.v[p].x
                    } else {
                        self.v[p].y
                    };
                    (s + v, n + 1)
                });
            self.probe
                .watch_x_history
                .push((self.time + dt, sum / n.max(1) as f32));
        }
        let ln_bounds = (J_BOUNDS.0.ln(), J_BOUNDS.1.ln());
        // J advances by the divergence the projection holds: the cells'
        // divergence interpolated bilinearly at the particle, over liquid
        // cells only. `tr C` reads the same in the bulk (the bulk divergence
        // probe: 1.32e-5 against 1.31e-5 1/s), but near the free surface the
        // gather reaches the velocity extrapolated into the air, which no
        // solve made divergence free (Bridson and Muller-Fischer 6.3's
        // constant extension is not either), and J drifted 5 % there while
        // the particles per interior cell, a volume measure that does not
        // read J, held at 4.01. A particle with no liquid cell around it is
        // in free flight and reads zero.
        let (nx, ny) = (self.layout.nx as i32, self.layout.ny as i32);
        let liquid = |i: i32, j: i32| {
            i >= 0
                && j >= 0
                && i < nx
                && j < ny
                && phi.get(i as usize, j as usize) < 0.0
                && self.solid_centres.get(i as usize, j as usize) > 0.0
        };
        let cell_div = |i: i32, j: i32| {
            let (i, j) = (i as usize, j as usize);
            (vel.u.get(i + 1, j) - vel.u.get(i, j) + vel.v.get(i, j + 1) - vel.v.get(i, j))
                / self.layout.dx
        };
        for (p, v_step) in v_mid.iter().enumerate() {
            let dt_div = {
                let t = self.x[p] / self.layout.dx - Vec2::splat(0.5);
                let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
                let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
                let (mut sum, mut weight) = (0.0f32, 0.0f32);
                for (di, dj, w) in [
                    (0, 0, (1.0 - fx) * (1.0 - fy)),
                    (1, 0, fx * (1.0 - fy)),
                    (0, 1, (1.0 - fx) * fy),
                    (1, 1, fx * fy),
                ] {
                    if liquid(i0 + di, j0 + dj) {
                        sum += w * cell_div(i0 + di, j0 + dj);
                        weight += w;
                    }
                }
                if weight > 0.0 { dt * sum / weight } else { 0.0 }
            };
            let advanced = self.log_j[p] + dt_div;
            let step = *v_step * dt;
            let was_inside = self.inside_tank(p);
            if advanced <= ln_bounds.0 || advanced >= ln_bounds.1 {
                self.j_at_bounds += 1;
                let cell = self.x[p].floor().as_ivec2();
                let wet = phi.get_signed(cell.x, cell.y).is_some_and(|f| f < 0.0);
                let place = if !was_inside {
                    2
                } else if wet {
                    0
                } else {
                    1
                };
                self.probe.j_at_bounds_by_place[place] += 1;
            }
            self.log_j[p] =
                advance_log_volume_ratio(self.log_j[p], dt_div, J_BOUNDS.0, J_BOUNDS.1).0;
            frame.largest_travel = frame.largest_travel.max(step.length());
            let before = self.x[p];
            self.x[p] += step;
            if was_inside && !self.inside_tank(p) {
                let (lo, hi) = (before - self.tank.0, self.tank.1 - before);
                let wall = lo.x.min(lo.y).min(hi.x).min(hi.y);
                self.probe
                    .crossings
                    .push((self.time, before, self.v[p], wall));
            }
        }
        self.probe.solve_drift = 4.0 * dt * self.layout.dx * solution.residual;
        for p in 0..self.x.len() {
            self.reflect_off_walls(p);
        }
        lap(&mut self.probe, 5);
        self.time += dt;
        frame.substeps += 1;
        frame.iterations.push(solution.iterations);
        frame.capped += u32::from(!solution.converged);
        if let Some(c) = self.probe.watch {
            self.probe
                .watch_history
                .push((self.time, solution.pressure[c]));
        }
        frame.pressure = solution.pressure;
        frame.active = system.active;
        frame.phi = phi;
        frame.vel = vel;
    }

    /// Fixed substeps (`fixed_dt`) until `t_end`, for scenes shorter than
    /// a frame.
    pub(super) fn run_until(&mut self, t_end: f32) {
        let dt = self.fixed_dt.expect("run_until needs a fixed substep");
        let mut frame = Frame {
            pressure: Vec::new(),
            active: Vec::new(),
            phi: self.layout.cells(0.0),
            vel: super::field::MacVelocity::zeros(&self.layout),
            substeps: 0,
            iterations: Vec::new(),
            capped: 0,
            largest_travel: 0.0,
        };
        while self.time < t_end {
            self.substep(dt.min(t_end - self.time).max(1e-9), &mut frame);
        }
    }

    pub(super) fn frame(&mut self) -> Frame {
        let mut frame = Frame {
            pressure: Vec::new(),
            active: Vec::new(),
            phi: self.layout.cells(0.0),
            vel: super::field::MacVelocity::zeros(&self.layout),
            substeps: 0,
            iterations: Vec::new(),
            capped: 0,
            largest_travel: 0.0,
        };
        let mut left = FRAME;
        let contact = self.contact_substep();
        while left > 1e-7 {
            let speed = self.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            // Half a cell per crossing of the fastest wave a solid's
            // particles carry, the P wave `c_p^2 = (K + mu) / rho`, under
            // the explicit limit `2 / sqrt(12) = 0.577` the transfer's
            // stiffest mode allows (`transfer::the_stiffest_mode_stays_
            // bounded_...`).
            let shear_limit = match (self.shear, self.conserving) {
                (Some(mu), None) => {
                    0.5 * self.layout.dx / (mu + self.sound_speed2.unwrap_or(0.0)).sqrt()
                }
                _ => f32::INFINITY,
            };
            let limit = contact.or(self.fixed_dt).unwrap_or_else(|| {
                shear_limit.min(travel_limited_dt(
                    speed,
                    self.g(),
                    self.layout.dx,
                    self.settings.max_cells_per_substep,
                ))
            });
            let dt = limit.min(left);
            self.substep(dt, &mut frame);
            left -= dt;
        }
        frame
    }

    /// The conserving step's contact refinement, a stepping stone toward
    /// resolving impacts only where they happen, declared as such: a frame
    /// during which any particle is, or would come, within reach of a wall
    /// moves at the P wave step `0.5 dx / c_p` instead of one step. An impact
    /// lasts about `2 L / c_p` (Love and Sulsky's contact time, their section
    /// 4.1), far under a frame, and the elastodynamic contact problem keeps its
    /// energy through it (Lebeau and Schatzman 1984, as Dabaghi et al., arXiv
    /// 1601.00778, theorem 2.1, state it), which one frame-long step cannot
    /// reproduce (measured: a body's centre 10 mm off the resolved one after a
    /// 0.66 m/s impact, and 76 percent of the energy lost at a 2 m/s one with
    /// the contact held at the step's end). The reach is the quadratic
    /// B-spline's support, 1.5 cells, plus the half cell a particle stands for
    /// (the region BFEMP, arXiv 2108.03349, notes a particle represents); the
    /// prediction is each particle's own trajectory over the frame with its
    /// velocity and the weight, the linear motion continuous collision
    /// detection assumes over a step. Its cost: a body resting on a wall stays
    /// refined. `None` keeps the scene's own substep.
    pub(super) fn contact_substep(&self) -> Option<f32> {
        let multiple = self.refine_contact?;
        let (mu, _) = (self.shear?, self.conserving?);
        let c_p = (self.sound_speed2.unwrap_or(0.0) + mu).sqrt();
        let (lo, hi) = self.tank;
        let reach = 2.0 * self.layout.dx;
        let distance = |q: Vec2| (q.x - lo.x).min(hi.x - q.x).min(q.y - lo.y).min(hi.y - q.y);
        let near = self.x.iter().zip(&self.v).any(|(&q, &v)| {
            let ahead = q + v * FRAME + 0.5 * self.gravity * FRAME * FRAME;
            distance(q).min(distance(ahead)) < reach
        });
        near.then_some(multiple * 0.5 * self.layout.dx / c_p)
    }

    /// One implicit, energy-momentum conserving step of the solid: the
    /// total Lagrangian step (`total_lagrangian::step_total_lagrangian_contact`),
    /// its grid on the configuration of its first step, the tank's walls
    /// held by the contact condition in velocity (route C), frictionless.
    pub(super) fn conserving_substep(
        &mut self,
        mu: f32,
        settings: crate::spacetime::total_lagrangian::ConservingSettings,
        dt: f32,
        frame: &mut Frame,
    ) {
        use glam::{DMat2, DVec2};
        let n = self.x.len();
        let mut x: Vec<DVec2> = self.x.iter().map(|q| q.as_dvec2()).collect();
        let mut v: Vec<DVec2> = self.v.iter().map(|q| q.as_dvec2()).collect();
        let mut f: Vec<DMat2> = self.deformation.iter().map(|m| m.as_dmat2()).collect();
        let mass: Vec<f64> = self.mass.iter().map(|&m| f64::from(m)).collect();
        let v0: Vec<f64> = self.v0.iter().map(|&m| f64::from(m)).collect();
        let (lo, hi) = (self.tank.0.as_dvec2(), self.tank.1.as_dvec2());
        // The probes' grid reports see no wall: walls are contacts now.
        let held = |_: usize, _: usize, _: usize| -> Option<f64> { None };
        let solids = self.solids.clone().unwrap_or_else(|| {
            vec![
                crate::spacetime::total_lagrangian::Elastic::isochoric(
                    f64::from(mu),
                    f64::from(self.sound_speed2.unwrap_or(0.0)),
                );
                n
            ]
        });
        if !self.probe.conditioning_bodies.is_empty() {
            let volume: Vec<f64> = (0..n).map(|p| v0[p] * f[p].determinant()).collect();
            let at = |r: std::ops::Range<usize>| {
                crate::spacetime::total_lagrangian::mass_conditioning(
                    &self.layout,
                    &x[r.clone()],
                    (&mass[r.clone()], &volume[r]),
                    &held,
                    (settings.occupation, settings.epsilon),
                )
            };
            let union = at(0..n);
            let each: Vec<f64> = self
                .probe
                .conditioning_bodies
                .iter()
                .map(|r| at(r.clone()))
                .collect();
            // The worst body below the threshold, dissected.
            let threshold = self.probe.basis_threshold;
            let worst = each
                .iter()
                .enumerate()
                .filter(|(_, c)| **c < threshold)
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, &c)| (i, c));
            if let Some((i, c)) = worst
                && self.probe.basis_worst.as_ref().is_none_or(|(w, _)| c < *w)
            {
                let r = self.probe.conditioning_bodies[i].clone();
                let report = crate::spacetime::total_lagrangian::basis_report(
                    &self.layout,
                    &x[r.clone()],
                    (&mass[r.clone()], &volume[r]),
                    &held,
                    (settings.occupation, settings.epsilon),
                );
                self.probe.basis_worst = Some((c, format!("body {i}: {report}")));
            }
            let (low, lows) = &mut self.probe.conditioning;
            *low = low.min(union);
            if lows.len() != each.len() {
                *lows = vec![f64::INFINITY; each.len()];
            }
            for (l, e) in lows.iter_mut().zip(each) {
                *l = l.min(e);
            }
        }
        let fastest = v.iter().fold(0.0f64, |m, q| m.max(q.length()));
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut v,
            f: &mut f,
            mass: &mass,
            v0: &v0,
        };
        let plane = |axis, at, inward| crate::spacetime::total_lagrangian::Plane {
            axis,
            at,
            inward,
            speed: 0.0,
            surface: 0,
        };
        let planes = [
            plane(0, lo.x, 1.0),
            plane(0, hi.x, -1.0),
            plane(1, lo.y, 1.0),
            plane(1, hi.y, -1.0),
        ];
        // The penalty, the Uzawa iteration's parameter: the P wave's
        // stiffness (factor 1, `block_slide_run`'s measured choice).
        let from = self.contact_from.unwrap_or(0);
        let penalty: Vec<f64> = solids
            .iter()
            .enumerate()
            .map(|(p, s)| if p >= from { s.bulk + s.mu } else { 0.0 })
            .collect();
        if self.lagrangian.as_ref().is_none_or(|(r, _)| r.len() != n) {
            self.lagrangian = Some((self.x.iter().map(|q| q.as_dvec2()).collect(), None));
        }
        let (reference, cache) = self.lagrangian.as_mut().expect("set above");
        let report = crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
            &self.layout,
            particles,
            reference,
            &solids,
            (
                f64::from(dt),
                crate::spacetime::total_lagrangian::ContactMode::Velocity,
                self.gravity.as_dvec2(),
            ),
            (
                crate::spacetime::total_lagrangian::ContactSet {
                    planes: &planes,
                    pairs: &[],
                    outlines: &[],
                    corners: &[],
                    friction: crate::spacetime::total_lagrangian::Friction::Uniform(0.0),
                },
                &penalty,
                &settings,
            ),
            cache,
        );
        if std::env::var_os("EMERGE_PROBE_SCENE_TRACE").is_some() {
            println!(
                "  step {}: lowest {:.3} (floor {:.3}), Newton {} residual {:.2e} (start {:.2e}) converged {}, contacts {}, push {:.3e}, passes {}, fastest {:.3e}",
                self.probe.steps,
                self.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min),
                lo.y,
                report.newton,
                report.residual,
                report.start_residual,
                report.converged,
                report.contacts,
                report.push,
                report.passes,
                fastest
            );
        }
        self.probe.contacts = self.probe.contacts.max(report.contacts);
        for (total, part) in self.probe.phases.iter_mut().zip(report.phases) {
            *total += part;
        }
        self.probe.steps += 1;
        self.probe.passes += u64::from(report.passes);
        self.probe.stabilisation_loss += report.stabilisation_loss;
        self.probe.grid_speed_ratio = self
            .probe
            .grid_speed_ratio
            .max(report.grid_speed / fastest.max(f64::MIN_POSITIVE));
        self.probe.start_residual = self.probe.start_residual.max(report.start_residual);
        self.probe.worst_projection = self.probe.worst_projection.max(report.projection);
        self.probe.ledger = report.ledger;
        self.probe.work = report.work;
        let stats = &mut self.probe.conserving;
        stats.0 += u64::from(report.newton);
        stats.1 += u64::from(report.gmres);
        stats.2 = stats.2.max(report.gmres);
        stats.3 = stats.3.max(report.residual);
        stats.4 += u32::from(!report.converged);
        if self.probe.estimator_reference > 0.0 {
            let acceleration: Vec<Vec2> =
                (0..n).map(|p| (v[p].as_vec2() - self.v[p]) / dt).collect();
            if self.probe.last_acceleration.len() == n {
                let jump = acceleration
                    .iter()
                    .zip(&self.probe.last_acceleration)
                    .fold(0.0f32, |m, (a, b)| m.max((*a - *b).abs().max_element()));
                let estimate = (0.25 - 1.0 / 6.0) * dt * dt * jump / self.probe.estimator_reference;
                self.probe.largest_estimate = self.probe.largest_estimate.max(estimate);
            }
            self.probe.last_acceleration = acceleration;
        }
        for p in 0..n {
            let before = self.x[p];
            self.x[p] = x[p].as_vec2();
            self.v[p] = v[p].as_vec2();
            self.c[p] = Mat2::ZERO;
            self.deformation[p] = f[p].as_mat2();
            frame.largest_travel = frame.largest_travel.max((self.x[p] - before).length());
        }
        self.probe.solve_drift = 0.0;
        // Counted, not moved: the contact holds the walls, and a moved
        // particle would leave the grid's reference configuration (`x = X +
        // u(X)`); the depth beyond a wall stays a measured criterion.
        for p in 0..n {
            self.beyond_walls(p, false);
        }
        self.time += dt;
        frame.substeps += 1;
        frame.phi = self.phi();
    }

    /// Moves particle `p`, if beyond a wall, to its mirror point across it,
    /// and counts it (the reflection amendment, module doc).
    pub(super) fn reflect_off_walls(&mut self, p: usize) {
        self.beyond_walls(p, true);
    }

    /// Counts particle `p` if it is beyond a wall, by how far, and with
    /// `mirror` moves it to its mirror point across it.
    pub(super) fn beyond_walls(&mut self, p: usize, mirror: bool) {
        let ulp = |w: f32| f32::from_bits(w.abs().to_bits() + 1) - w.abs();
        let (lo, hi) = self.tank;
        let mut q = self.x[p];
        let drift = self.probe.solve_drift;
        let (mut deepest, mut ratio) = (0.0f32, 0.0f32);
        for (coord, low, high) in [(&mut q.x, lo.x, hi.x), (&mut q.y, lo.y, hi.y)] {
            let wall = if *coord < low {
                low
            } else if *coord > high {
                high
            } else {
                continue;
            };
            let depth = (*coord - wall).abs();
            deepest = deepest.max(depth / ulp(wall));
            ratio = ratio.max(depth / (4.0 * ulp(wall) + drift));
            *coord = 2.0 * wall - *coord;
        }
        if deepest > 0.0 {
            if mirror {
                self.x[p] = q;
            }
            self.probe.reflections += 1;
            if deepest > self.probe.deepest_reflection_ulps {
                self.probe.deepest_reflection_at = Some((self.time, self.x[p]));
            }
            self.probe.deepest_reflection_ulps = self.probe.deepest_reflection_ulps.max(deepest);
            self.probe.worst_reflection_ratio = self.probe.worst_reflection_ratio.max(ratio);
        }
    }

    /// The reflection criterion (module doc).
    pub(super) fn check_reflections(&self, label: &str, report: &mut Report) {
        let (n, deepest) = (self.probe.reflections, self.probe.deepest_reflection_ulps);
        let ratio = self.probe.worst_reflection_ratio;
        println!(
            "{label}: {n} reflections off a wall, deepest {deepest:.2} ulps before it, worst {ratio:.2} of its budget"
        );
        if let Some((t, at)) = self.probe.deepest_reflection_at {
            println!("{label}: deepest reflection at t = {t:.4} s, mirrored to {at:?}");
        }
        report.check(ratio <= 1.0, || {
            format!("{label}: a particle went {ratio:.2} times its budget beyond a wall ({deepest:.1} ulps)")
        });
    }

    /// Which cells have their centre outside every solid.
    pub(super) fn open_centres(&self) -> Vec<bool> {
        self.solid_centres.data().iter().map(|&d| d > 0.0).collect()
    }

    /// A particle's volume ratio: a solid's is its own `det F`, the `J` its
    /// stress reads; a liquid's integrates the cells' divergence.
    pub(super) fn j(&self, p: usize) -> f32 {
        if self.shear.is_some() {
            self.deformation[p].determinant()
        } else {
            self.log_j[p].exp()
        }
    }

    pub(super) fn volume(&self) -> f32 {
        (0..self.x.len()).map(|p| self.v0[p] * self.j(p)).sum()
    }

    pub(super) fn centre_of_mass(&self) -> (Vec2, Vec2) {
        let total: f32 = self.mass.iter().sum();
        let x: Vec2 = (0..self.x.len()).map(|p| self.x[p] * self.mass[p]).sum();
        let v: Vec2 = (0..self.x.len()).map(|p| self.v[p] * self.mass[p]).sum();
        (x / total, v / total)
    }

    /// Kinetic plus potential energy, the floor of the tank as zero.
    pub(super) fn energy(&self) -> f32 {
        let floor = self.tank.0.y;
        let elastic = |p: usize| match (self.shear, self.sound_speed2) {
            (Some(mu), Some(lambda)) => {
                let j = self.j(p);
                self.v0[p]
                    * (shear_energy(self.deformation[p], mu) + 0.5 * lambda * (j - 1.0).powi(2))
            }
            _ => 0.0,
        };
        (0..self.x.len())
            .map(|p| {
                let kinetic = 0.5 * self.v[p].length_squared();
                self.mass[p] * (kinetic + self.g() * (self.x[p].y - floor)) + elastic(p)
            })
            .sum()
    }

    /// In the tank or on its walls: out means beyond a wall. A particle gliding
    /// along a wall approaches it geometrically (the normal velocity falls
    /// linearly to zero there, `substep` doc) and, in f32, ends on it: measured
    /// in the drop into a pool once the elastic solid's transfer was in, a
    /// particle reached `x = 4.0000005`, one ulp from the wall at 4, with `v_x
    /// = -1.6e-5` cells/s, sat exactly on the wall for one frame and went back
    /// in, never beyond it. The strict test counted it out.
    pub(super) fn inside_tank(&self, p: usize) -> bool {
        let q = self.x[p];
        q.x >= self.tank.0.x && q.x <= self.tank.1.x && q.y >= self.tank.0.y && q.y <= self.tank.1.y
    }

    /// Particles per cell over cells whose eight neighbours also hold
    /// particles: a volume measure that does not read J.
    pub(super) fn interior_particles_per_cell(&self) -> f32 {
        let (nx, ny) = (self.layout.nx as i32, self.layout.ny as i32);
        let mut count = vec![0u32; (nx * ny) as usize];
        for q in &self.x {
            let c = q.floor().as_ivec2();
            if c.x >= 0 && c.y >= 0 && c.x < nx && c.y < ny {
                count[(c.x + nx * c.y) as usize] += 1;
            }
        }
        let (mut sum, mut cells) = (0u32, 0u32);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let full = (-1..=1)
                    .all(|dj| (-1..=1).all(|di| count[(i + di + nx * (j + dj)) as usize] > 0));
                if full {
                    sum += count[(i + nx * j) as usize];
                    cells += 1;
                }
            }
        }
        if cells == 0 {
            0.0
        } else {
            sum as f32 / cells as f32
        }
    }

    /// A text picture of the particles, printed when `EMERGE_GATE_MAPS` is
    /// set.
    pub(super) fn print_map(&self, label: &str) {
        if crate::diagnostics::research_switch("EMERGE_GATE_MAPS").is_none() {
            return;
        }
        let mut particles = Particles::default();
        for &x in &self.x {
            particles.push(Particle {
                x,
                ..bytemuck::Zeroable::zeroed()
            });
        }
        let size = Vec2::new(self.layout.nx as f32, self.layout.ny as f32);
        let cols = self.layout.nx.min(72);
        let rows = (cols as f32 * size.y / size.x / 2.0).ceil() as usize;
        println!("{label} t={:.3}s", self.time);
        let map = scene_map(
            &particles,
            (Vec2::ZERO, size),
            cols,
            rows.max(1),
            |_| 1.0,
            &OCCUPANCY_BANDS,
        );
        for row in map {
            println!("|{row}|");
        }
    }
}
