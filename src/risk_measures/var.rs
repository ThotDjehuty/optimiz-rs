//! Value-at-Risk estimators.
//!
//! For a real-valued loss random variable `L` and confidence level
//! `alpha in (0, 1)`,
//!
//! ```text
//!     VaR_alpha(L) = inf { x in R : P(L <= x) >= alpha }.
//! ```
//!
//! Two estimators are provided:
//!
//! * `historical_var`  -- `alpha`-quantile of an empirical loss sample.
//! * `parametric_var`  -- Gaussian closed form `mu + sigma * Phi^{-1}(alpha)`.

use crate::core::{OptimizrError, Result};

use super::check_alpha;

/// Empirical (historical) Value-at-Risk at confidence level `alpha`.
///
/// `losses` is a non-empty sample of realised losses (positive losses
/// = bad outcomes). The function returns the smallest order statistic
/// `L_(k)` with rank `k = ceil(alpha * n)`.
pub fn historical_var(losses: &[f64], alpha: f64) -> Result<f64> {
    check_alpha(alpha)?;
    if losses.is_empty() {
        return Err(OptimizrError::EmptyData);
    }
    let mut sorted: Vec<f64> = losses.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let k = ((alpha * n as f64).ceil() as usize).max(1).min(n);
    Ok(sorted[k - 1])
}

/// Closed-form Gaussian Value-at-Risk
/// `VaR = mu + sigma * Phi^{-1}(alpha)`.
pub fn parametric_var(mu: f64, sigma: f64, alpha: f64) -> Result<f64> {
    check_alpha(alpha)?;
    if !(sigma >= 0.0) {
        return Err(OptimizrError::InvalidParameter(
            "sigma must be non-negative".into(),
        ));
    }
    Ok(mu + sigma * inverse_normal_cdf(alpha))
}

/// Beasley--Springer--Moro inverse standard normal CDF.
pub(crate) fn inverse_normal_cdf(p: f64) -> f64 {
    // Acklam's algorithm (high accuracy).
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_var_matches_quantile() {
        let losses: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let v = historical_var(&losses, 0.95).unwrap();
        assert_eq!(v, 95.0);
    }

    #[test]
    fn parametric_var_gaussian_95() {
        let v = parametric_var(0.0, 1.0, 0.95).unwrap();
        assert!((v - 1.6448536).abs() < 1e-4);
    }

    #[test]
    fn rejects_bad_alpha() {
        assert!(historical_var(&[1.0, 2.0], 0.0).is_err());
        assert!(historical_var(&[1.0, 2.0], 1.0).is_err());
    }
}
