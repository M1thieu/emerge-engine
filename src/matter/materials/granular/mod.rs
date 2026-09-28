//! Granular constitutive models and their supporting physics, grouped by
//! research thread (dry-sand repose angle and rolling resistance) rather than
//! by rheological class alone. `sand` (Drucker-Prager) and `sand_mui` (µ(I)
//! rheology) are the constitutive models; `cosserat` (micropolar grain
//! kinematics) and `grain_contact_law` (DEM contact force law) are the two
//! candidate mechanisms for sand's self-arrest; `scale_contract` is the REV
//! grid-resolution check for grain-diameter scenes.
//!
//! Under a purely rheological taxonomy `sand` would sit with the other
//! elastoplastic solids (`von_mises`/`rankine`/`nacc` in `materials/`). It
//! lives here because it is developed together with
//! `cosserat`/`grain_contact_law`.

pub mod cosserat;
pub mod disc_contact;
pub mod grain_contact_law;
pub mod sand;
pub mod sand_mui;
pub mod scale_contract;
