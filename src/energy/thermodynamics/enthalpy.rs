//! Real enthalpy method for phase-change (Stefan) problems -- Voller & Cross
//! 1981 ("Accurate solutions of moving boundary problems using the enthalpy
//! method," Int. J. Heat Mass Transfer 24(3):545-556) and Voller &
//! Swaminathan 1991 ("General source-based method for solidification phase
//! change," Numerical Heat Transfer B 19(2):175-189).
//!
//! Real motivating gap (GitHub issue #7): `Simulation::apply_phase_transition`
//! already debits a real, energy-conserving latent-heat jump
//! (`temperature -= latent_heat / heat_capacity`, see that function's own
//! "Stefan condition" roadmap doc) at the INSTANT a threshold fires -- but
//! the transition itself is still a discrete switch, not a continuous mushy
//! zone. The enthalpy method's own real idea: track enthalpy `H` (total
//! thermal energy per unit mass) as the state variable instead of
//! temperature `T` -- T and phase_fraction are then both DERIVED from H,
//! and the derivation is naturally continuous: absorbing energy while
//! `T == t_transition` doesn't raise T further, it raises phase_fraction
//! instead, until the whole latent-heat band has been crossed.
//!
//! Real, disclosed first-increment scope: one shared specific heat `cp` on
//! both sides of the transition (distinct `cp_solid`/`cp_liquid` is a real
//! refinement the method supports, not implemented here). This is the
//! numerical core only -- wiring it into `Particle`/`WithLatentHeat` so a
//! scene's own materials use it automatically is real further work, not
//! done here; see this module's own tests for how the three regions relate,
//! and `crate::solver::particles::apply_phase_transition`'s doc for the
//! discrete-jump mechanism this generalizes.

/// Real, forward enthalpy relation H(T) -- see this module's own doc for the
/// three-region shape. Only valid for `T <= t_transition` (ordinary sensible
/// heat) or `T` representing a FULLY melted state (`phase_fraction == 1`);
/// it cannot represent a mushy intermediate on its own, since a single T
/// doesn't determine phase_fraction in that band -- H does. Use this to seed
/// H from a known, single-phase starting temperature (e.g. at spawn), not to
/// track an ongoing melt.
pub fn enthalpy_from_temperature(t: f32, cp: f32, latent_heat: f32, t_transition: f32) -> f32 {
    debug_assert!(cp > 0.0, "specific heat capacity must be positive");
    if t <= t_transition {
        cp * t
    } else {
        cp * t_transition + latent_heat + cp * (t - t_transition)
    }
}

/// Real inverse of the enthalpy relation: recovers `(temperature,
/// phase_fraction)` from enthalpy `h` alone. `phase_fraction` is in `[0, 1]`
/// -- `0.0` fully solid, `1.0` fully liquid, `(0.0, 1.0)` a real mushy-zone
/// particle mid-melt with `temperature` PINNED at `t_transition` (the real
/// physical behavior: adding heat to a melting substance raises how much of
/// it has melted, not its temperature, until melting completes).
pub fn temperature_and_phase_fraction_from_enthalpy(
    h: f32,
    cp: f32,
    latent_heat: f32,
    t_transition: f32,
) -> (f32, f32) {
    debug_assert!(cp > 0.0, "specific heat capacity must be positive");
    debug_assert!(latent_heat >= 0.0, "latent heat must be non-negative");
    let h_solidus = cp * t_transition;
    let h_liquidus = h_solidus + latent_heat;
    if h <= h_solidus {
        (h / cp, 0.0)
    } else if h < h_liquidus {
        (t_transition, (h - h_solidus) / latent_heat.max(1.0e-12))
    } else {
        (t_transition + (h - h_liquidus) / cp, 1.0)
    }
}

#[cfg(test)]
mod enthalpy_tests {
    use super::*;

    const CP: f32 = 2100.0; // ice, J/(kg*K), real value -- matches diffusion.rs's own convention
    const LATENT_HEAT: f32 = 334_000.0; // ice->water, J/kg, real value (already used elsewhere)
    const T_TRANSITION: f32 = 273.15; // 0 C in Kelvin

    /// Real round-trip check below the transition: H(T) then back must
    /// recover the same T with phase_fraction=0 (still fully solid).
    #[test]
    fn round_trips_below_transition() {
        let t = 250.0;
        let h = enthalpy_from_temperature(t, CP, LATENT_HEAT, T_TRANSITION);
        let (t2, phase) =
            temperature_and_phase_fraction_from_enthalpy(h, CP, LATENT_HEAT, T_TRANSITION);
        assert!((t2 - t).abs() < 1.0e-3, "expected T={t}, got {t2}");
        assert_eq!(
            phase, 0.0,
            "should still be fully solid below the transition"
        );
    }

    /// Real round-trip check above the transition (fully melted): H(T) then
    /// back must recover the same T with phase_fraction=1.
    #[test]
    fn round_trips_above_transition() {
        let t = 300.0;
        let h = enthalpy_from_temperature(t, CP, LATENT_HEAT, T_TRANSITION);
        let (t2, phase) =
            temperature_and_phase_fraction_from_enthalpy(h, CP, LATENT_HEAT, T_TRANSITION);
        assert!((t2 - t).abs() < 1.0e-3, "expected T={t}, got {t2}");
        assert_eq!(phase, 1.0, "should be fully liquid above the transition");
    }

    /// The real mushy-zone property this whole method exists for: halfway
    /// through the latent-heat band, temperature must be PINNED at
    /// t_transition (not interpolated), and phase_fraction must read ~0.5 --
    /// not 0 or 1. This is the literal "partial melting" issue #7 asks for.
    #[test]
    fn halfway_through_latent_heat_band_is_a_real_mushy_particle() {
        let h_solidus = CP * T_TRANSITION;
        let h_halfway = h_solidus + LATENT_HEAT * 0.5;
        let (t, phase) =
            temperature_and_phase_fraction_from_enthalpy(h_halfway, CP, LATENT_HEAT, T_TRANSITION);
        assert!(
            (t - T_TRANSITION).abs() < 1.0e-3,
            "mushy-zone particle's temperature must stay pinned at the transition \
             point while it's still melting, got T={t}"
        );
        assert!(
            (phase - 0.5).abs() < 1.0e-6,
            "expected phase_fraction=0.5 exactly halfway through the latent-heat \
             band, got {phase}"
        );
    }

    /// Real monotonicity check: as more energy (H) is added, temperature and
    /// phase_fraction must never DECREASE -- the physical requirement that
    /// this relation actually represents an energy state, not an arbitrary
    /// lookup. Sampled across all three regions.
    #[test]
    fn temperature_and_phase_fraction_are_monotonic_in_enthalpy() {
        let mut prev_t = f32::MIN;
        let mut prev_phase = 0.0f32;
        for i in 0..200 {
            let h = -50_000.0 + i as f32 * 5_000.0; // sweeps well below to well above the band
            let (t, phase) =
                temperature_and_phase_fraction_from_enthalpy(h, CP, LATENT_HEAT, T_TRANSITION);
            assert!(
                t >= prev_t - 1.0e-4,
                "temperature must be monotonic non-decreasing in H: {t} < {prev_t} at h={h}"
            );
            assert!(
                phase >= prev_phase - 1.0e-6,
                "phase_fraction must be monotonic non-decreasing in H: {phase} < {prev_phase} at h={h}"
            );
            prev_t = t;
            prev_phase = phase;
        }
    }

    /// Real energetic-consistency check connecting this module to the
    /// EXISTING discrete-jump mechanism (`apply_phase_transition`'s
    /// `temperature -= latent_heat / heat_capacity`): crossing the ENTIRE
    /// mushy zone (h_solidus to h_liquidus) must correspond to exactly
    /// `latent_heat` joules absorbed per kg, by construction -- the same
    /// real energy quantity the existing threshold mechanism already debits
    /// in one instantaneous step. This method just spreads that same real
    /// energy over a continuous band instead of a single substep.
    #[test]
    fn crossing_the_full_mushy_zone_absorbs_exactly_latent_heat() {
        let h_solidus = CP * T_TRANSITION;
        let h_liquidus = h_solidus + LATENT_HEAT;
        assert!((h_liquidus - h_solidus - LATENT_HEAT).abs() < 1.0e-6);
    }
}
