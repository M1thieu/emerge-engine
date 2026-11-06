//! Measured viscoplastic records: Norton's power law (`j2_hencky::Norton`)
//! with the elastic moduli it was measured with, the strain rates it covers
//! and where it comes from. A record is data; a scene builds its material
//! from one (`ViscoplasticRecord::material`).
//!
//! Two plasticines are never blended: Sapozhnikov and Ignatova's technical
//! plasticine and Chijiiwa, Hatamura and Hasegawa's (Trans. ISIJ 21, 1981)
//! differ by a factor of about 4 at 1/s, and the latter's law has a strain
//! hardening and a temperature term this law does not carry; it would be a
//! record of its own law.

use super::j2_hencky::{Norton, StrainHardening};
use super::von_mises::VonMisesMaterial;
use crate::materials::FromSI;
use crate::materials::physical_props::{DuctileProps, Elastic};

/// A measured Norton law and its elastic moduli.
#[derive(Clone, Copy, Debug)]
pub struct ViscoplasticRecord {
    pub law: Norton,
    /// Young's modulus and its stated uncertainty, Pa; Poisson's ratio.
    pub e_pa: f64,
    pub e_uncertainty_pa: f64,
    pub nu: f64,
    /// The material, its state and the source, page by page.
    pub material: &'static str,
    pub source: &'static str,
}

impl ViscoplasticRecord {
    /// Technical plasticine at 20 C, Sapozhnikov and Ignatova, PNRPU Mech.
    /// Bull. 2014 no. 2, 200-219 (doi 10.15593/perm.mech/2014.2.09): the flow
    /// stress `sigma = 6.37e5 eps_dot^0.266` Pa in tension, compression and
    /// indentation over 4e-4 to 80 1/s (eq. 7, p. 212), read as a Norton law
    /// with `rate_0 = 1/s` (p. 213: "elastic and viscous elements in
    /// series"); `E = 30 +- 5 MPa`, `nu = 0.30` initially (pp. 204-205).
    /// Read from a summary; to be checked on the page
    /// before a presented result rests on it.
    ///
    /// Declared gaps, each with its removal criterion in that report: no
    /// density is given (the scene states its own, `material`); engineering
    /// stress and nominal rate, while the law here is in Kirchhoff stress and
    /// log rate (about 15 % apart at 15 % strain); compression-dominated
    /// states only (tension damages it beyond 3-5 %; counted,
    /// `Stepped::outside`); Poisson's ratio rises to 0.45 at 3 % strain; the
    /// shear law (eq. 8, another exponent) is not this one.
    pub fn technical_plasticine_sapozhnikov_2014() -> Self {
        Self {
            law: Norton {
                k: 6.37e5,
                m: 0.266,
                rate_0: 1.0,
                valid_rates: (4.0e-4, 80.0),
                hardening: None,
            },
            e_pa: 30.0e6,
            e_uncertainty_pa: 5.0e6,
            nu: 0.30,
            material: "technical plasticine, 20 C",
            source: "Sapozhnikov and Ignatova 2014, eq. 7 p. 212, pp. 204-205, 213",
        }
    }

    /// Roma Plastilina No. 1 at 23 C, Hernandez, Buchely, Casas-Rodriguez and
    /// Maranon, J. Def. Model. Simul. 20(3) (2023) 371-381, read in full:
    /// uniaxial compression (ASTM D695, 1:1, talc-lubricated) at 6.56e-4 to
    /// 3.28e-1 1/s, regressed on the curves' two log-log lines, `sigma = E
    /// eps` and `sigma = k eps^m epsdot^n` (eq. 9, Table 3): `E = 1.70 +- 0.28
    /// MPa`, `k = 0.173 +- 0.011 MPa`, `m = 0.119 +- 0.050`, `n = 0.037 +-
    /// 0.001`. No drop test entered these numbers. Buchely, Maranon and
    /// Silberschmidt, IJIE 90 (2016), Table 1, measure `E = 1.73 MPa` on the
    /// same clay. `eps_0 = (k / E)^(1 / (1 - m))` is where the two lines meet
    /// at the reference rate (`StrainHardening`): derived, 0.0747.
    ///
    /// Declared gaps, each with its removal criterion:
    /// - Poisson's ratio 0.49 is ASSUMED, by Hernandez et al. 2015 (IJMS 100,
    ///   p. 162), citing Crandall's 0.434 for another clay; a measurement on
    ///   this clay replaces it (Crandall et al. 1971, not held). The
    ///   uniaxial curve moves under 1 % between the two.
    /// - beyond 0.33 1/s the law is extrapolated, as far as a drop's 400 1/s
    ///   (counted, `Stepped::outside`): a flow law measured at those rates and
    ///   not fitted through a drop.
    /// - beyond the compression curves' strain, and under cycling, it hardens
    ///   without bound where the clay softens (measured against NIST's
    ///   oscillatory shear): a law with structure breakdown.
    /// - the law comes from compression: states in tension are applied outside
    ///   its data (counted, `Stepped::outside`): a tensile test of this clay.
    /// - no viscoelastic part and no stored fraction of the plastic work.
    pub fn roma_plastilina_hernandez_2023() -> Self {
        let (e_pa, k, m, n): (f64, f64, f64, f64) = (1.70e6, 0.173e6, 0.119, 0.037);
        Self {
            law: Norton {
                k,
                m: n,
                rate_0: 1.0,
                valid_rates: (6.56e-4, 3.28e-1),
                hardening: Some(StrainHardening {
                    eps_0: (k / e_pa).powf(1.0 / (1.0 - m)),
                    exponent: m,
                }),
            },
            e_pa,
            e_uncertainty_pa: 0.28e6,
            nu: 0.49,
            material: "Roma Plastilina No. 1, 23 C, conditioned; nu assumed",
            source: "Hernandez et al. 2023, eq. 9 and Table 3 (uniaxial compression); nu from Hernandez et al. 2015, p. 162, assumed",
        }
    }

    /// The ductile material of this record for a scene, with Young's modulus
    /// `e_pa` (the record's or a value inside its uncertainty) and the
    /// density `rho_kg_m3` the scene states with its own source: the record
    /// gives none, and no default is taken.
    pub fn material(
        &self,
        (e_pa, rho_kg_m3): (f64, f64),
        config: &crate::SimConfig,
    ) -> VonMisesMaterial {
        let props = DuctileProps {
            elastic: Elastic {
                e_pa: e_pa as f32,
                nu: self.nu as f32,
                rho_kg_m3: rho_kg_m3 as f32,
            },
            yield_stress_pa: 0.0,
        };
        let mut material = VonMisesMaterial::from_physical(&props, config);
        material.norton = Some(self.law);
        material
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Roma Plastilina record carries what its source's Table 3 prints,
    /// the two lines' meeting strain derived from them, and says where its
    /// numbers and its assumed Poisson's ratio come from.
    #[test]
    fn the_roma_plastilina_record_reads_back_its_table_and_its_source() {
        let record = ViscoplasticRecord::roma_plastilina_hernandez_2023();
        assert_eq!(
            (
                record.e_pa,
                record.e_uncertainty_pa,
                record.law.k,
                record.law.m
            ),
            (1.70e6, 0.28e6, 0.173e6, 0.037)
        );
        assert_eq!(record.law.valid_rates, (6.56e-4, 3.28e-1));
        let hardening = record.law.hardening.expect("the law hardens");
        assert_eq!(hardening.exponent, 0.119);
        assert!(
            (hardening.eps_0 - 0.0747).abs() < 5e-5,
            "{}",
            hardening.eps_0
        );
        // The elastic line meets the plastic line at `eps_0` at 1/s.
        let elastic = record.e_pa * hardening.eps_0;
        let plastic = record.law.k * hardening.eps_0.powf(hardening.exponent);
        assert!(
            (elastic / plastic - 1.0).abs() < 1e-12,
            "{elastic} {plastic}"
        );
        assert!(record.source.contains("Table 3"));
        assert!(record.material.contains("assumed") && record.source.contains("assumed"));
    }
}
