//! Generic two-sided intensity control
//! =====================================
//!
//! Considers a controlled bilateral jump process where the controller picks
//! two non-negative intensities `λ_+, λ_-` for upward and downward jumps of
//! a scalar state `q`.  The instantaneous reward density is
//!
//! ```text
//! r(q, λ_+, λ_-) = λ_+ (δ_+(λ_+) - ΔV(q, +1)) + λ_- (δ_-(λ_-) - ΔV(q, -1))
//! ```
//!
//! where `δ_±(λ) = α_± + κ_± λ` is an affine *generic* per-jump premium and
//! `ΔV(q, ±1) = V(q ± 1) - V(q)` is the value-function differential.  The
//! first-order condition gives the optimal symmetric pair
//!
//! ```text
//! λ_*± = max(0, (α_± - ΔV(q, ±1)) / (2 κ_±))
//! ```
//!
//! This module exposes a single helper that, given the value-function
//! differentials and the affine coefficients, returns the optimal
//! intensities and the resulting reward.  All quantities are kept in
//! purely abstract / generic form.

use crate::core::{OptimizrError, Result};

#[derive(Clone, Debug)]
pub struct TwoSidedConfig {
    /// Affine intercept `α_+` of the upward premium.
    pub alpha_plus: f64,
    /// Affine intercept `α_-` of the downward premium.
    pub alpha_minus: f64,
    /// Affine slope `κ_+` of the upward premium (must be > 0).
    pub kappa_plus: f64,
    /// Affine slope `κ_-` of the downward premium (must be > 0).
    pub kappa_minus: f64,
}

#[derive(Clone, Debug)]
pub struct TwoSidedIntensities {
    pub lambda_plus: f64,
    pub lambda_minus: f64,
    pub reward_density: f64,
}

pub fn optimal_two_sided_intensities(
    cfg: &TwoSidedConfig,
    delta_v_plus: f64,
    delta_v_minus: f64,
) -> Result<TwoSidedIntensities> {
    if !(cfg.kappa_plus > 0.0 && cfg.kappa_minus > 0.0) {
        return Err(OptimizrError::InvalidParameter(
            "kappa_plus and kappa_minus must be > 0".into(),
        ));
    }
    let raw_plus = (cfg.alpha_plus - delta_v_plus) / (2.0 * cfg.kappa_plus);
    let raw_minus = (cfg.alpha_minus - delta_v_minus) / (2.0 * cfg.kappa_minus);
    let lambda_plus = raw_plus.max(0.0);
    let lambda_minus = raw_minus.max(0.0);
    let premium_plus = cfg.alpha_plus + cfg.kappa_plus * lambda_plus;
    let premium_minus = cfg.alpha_minus + cfg.kappa_minus * lambda_minus;
    let reward_density = lambda_plus * (premium_plus - delta_v_plus)
        + lambda_minus * (premium_minus - delta_v_minus);
    Ok(TwoSidedIntensities {
        lambda_plus,
        lambda_minus,
        reward_density,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_zero_gradient_recovers_alpha_over_two_kappa() {
        let cfg = TwoSidedConfig {
            alpha_plus: 1.0,
            alpha_minus: 1.0,
            kappa_plus: 0.5,
            kappa_minus: 0.5,
        };
        let r = optimal_two_sided_intensities(&cfg, 0.0, 0.0).unwrap();
        let expected = 1.0 / (2.0 * 0.5);
        assert!((r.lambda_plus - expected).abs() < 1e-12);
        assert!((r.lambda_minus - expected).abs() < 1e-12);
        // Reward density = λ (α + κ λ) = 1.0 * (1.0 + 0.5 * 1.0) = 1.5; symmetric → 3.0.
        assert!((r.reward_density - 3.0).abs() < 1e-12);
    }

    #[test]
    fn high_value_gradient_kills_intensity() {
        let cfg = TwoSidedConfig {
            alpha_plus: 1.0,
            alpha_minus: 1.0,
            kappa_plus: 0.5,
            kappa_minus: 0.5,
        };
        let r = optimal_two_sided_intensities(&cfg, 5.0, 5.0).unwrap();
        assert_eq!(r.lambda_plus, 0.0);
        assert_eq!(r.lambda_minus, 0.0);
        assert!(r.reward_density.abs() < 1e-12);
    }

    #[test]
    fn rejects_zero_kappa() {
        let cfg = TwoSidedConfig {
            alpha_plus: 1.0,
            alpha_minus: 1.0,
            kappa_plus: 0.0,
            kappa_minus: 0.5,
        };
        assert!(optimal_two_sided_intensities(&cfg, 0.0, 0.0).is_err());
    }
}
