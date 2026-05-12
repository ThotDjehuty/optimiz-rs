//! Robust drift estimator via the Huber M-estimator
//! ==================================================
//!
//! Given observations `(x_k)_{k=0..N}` of an Ornstein–Uhlenbeck-type
//! discrete dynamical system
//!
//! ```text
//! x_{k+1} = x_k + (a + b x_k) Δt + σ ε_k,    ε_k iid centred
//! ```
//!
//! the routine fits `(a, b)` by minimising the Huber loss
//!
//! ```text
//! L(a, b) = Σ ρ_δ( (Δx_k - (a + b x_k) Δt) / s )
//! ```
//!
//! where `s` is a robust scale (median absolute deviation) and `ρ_δ` is the
//! Huber loss (`x²/2` for `|x| ≤ δ`, `δ |x| - δ²/2` otherwise).  We use
//! iteratively reweighted least squares (IRLS).

use crate::core::{OptimizrError, Result};

#[derive(Clone, Debug)]
pub struct RobustDriftConfig {
    pub dt: f64,
    pub huber_delta: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for RobustDriftConfig {
    fn default() -> Self {
        Self { dt: 1.0, huber_delta: 1.345, max_iterations: 200, tolerance: 1e-9 }
    }
}

#[derive(Clone, Debug)]
pub struct RobustDriftResult {
    pub a: f64,
    pub b: f64,
    pub iterations: usize,
}

fn median_abs_dev(r: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = r.iter().copied().collect();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = sorted[sorted.len() / 2];
    let mut absdev: Vec<f64> = sorted.iter().map(|x| (x - med).abs()).collect();
    absdev.sort_by(|x, y| x.partial_cmp(y).unwrap());
    absdev[absdev.len() / 2].max(1e-12) * 1.4826 // consistency factor for normality
}

pub fn estimate_robust_drift(observations: &[f64], cfg: &RobustDriftConfig) -> Result<RobustDriftResult> {
    if observations.len() < 3 {
        return Err(OptimizrError::InvalidInput("need ≥ 3 observations".into()));
    }
    if cfg.dt <= 0.0 {
        return Err(OptimizrError::InvalidParameter("dt > 0".into()));
    }
    let n = observations.len() - 1;
    let dt = cfg.dt;
    // First-stage OLS for initial guess.
    let mut xb = vec![0.0f64; n]; // x_k
    let mut yb = vec![0.0f64; n]; // (x_{k+1} - x_k) / dt
    for k in 0..n {
        xb[k] = observations[k];
        yb[k] = (observations[k + 1] - observations[k]) / dt;
    }
    let mean_x = xb.iter().sum::<f64>() / n as f64;
    let mean_y = yb.iter().sum::<f64>() / n as f64;
    let mut s_xx = 0.0; let mut s_xy = 0.0;
    for k in 0..n {
        s_xx += (xb[k] - mean_x).powi(2);
        s_xy += (xb[k] - mean_x) * (yb[k] - mean_y);
    }
    if s_xx.abs() < 1e-15 {
        return Err(OptimizrError::NumericalError("design matrix is singular".into()));
    }
    let mut b = s_xy / s_xx;
    let mut a = mean_y - b * mean_x;

    let mut iter = 0;
    for it in 0..cfg.max_iterations {
        iter = it + 1;
        // Residuals r_k = y_k - (a + b x_k)
        let r: Vec<f64> = (0..n).map(|k| yb[k] - (a + b * xb[k])).collect();
        let s = median_abs_dev(&r);
        // Huber weights w_k = 1 if |r/s| ≤ δ else δ s / |r|.
        let mut w = vec![1.0f64; n];
        for k in 0..n {
            let z = (r[k] / s).abs();
            if z > cfg.huber_delta {
                w[k] = cfg.huber_delta / z;
            }
        }
        // Weighted least squares.
        let mut sw = 0.0; let mut swx = 0.0; let mut swy = 0.0;
        let mut swxx = 0.0; let mut swxy = 0.0;
        for k in 0..n {
            let wk = w[k];
            sw += wk;
            swx += wk * xb[k];
            swy += wk * yb[k];
            swxx += wk * xb[k] * xb[k];
            swxy += wk * xb[k] * yb[k];
        }
        let det = sw * swxx - swx * swx;
        if det.abs() < 1e-15 {
            return Err(OptimizrError::NumericalError("weighted normal eqs singular".into()));
        }
        let new_a = (swxx * swy - swx * swxy) / det;
        let new_b = (sw * swxy - swx * swy) / det;
        let delta = (new_a - a).abs() + (new_b - b).abs();
        a = new_a; b = new_b;
        if delta < cfg.tolerance { break; }
    }

    Ok(RobustDriftResult { a, b, iterations: iter })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    /// Simulate `x_{k+1} = x_k + (1 - 0.5 x_k) Δt + 0.1 ε` with a few
    /// outliers; the robust fit should recover `(a, b) = (1, -0.5)` to a
    /// few percent even when 5% of innovations are 10× larger.
    #[test]
    fn robust_estimator_resists_outliers() {
        let mut rng = StdRng::seed_from_u64(7);
        let dt = 0.01;
        let true_a = 1.0_f64; let true_b = -0.5_f64;
        let n = 5000;
        let mut x = 0.0;
        let mut obs = vec![x];
        for k in 0..n {
            let noise = if k % 20 == 0 { rng.gen_range(-2.0..2.0) } else { rng.gen_range(-0.1..0.1) };
            x = x + (true_a + true_b * x) * dt + noise * dt.sqrt();
            obs.push(x);
        }
        let res = estimate_robust_drift(
            &obs,
            &RobustDriftConfig { dt, ..Default::default() },
        ).unwrap();
        assert!((res.a - true_a).abs() < 0.2, "a estimate {} vs {}", res.a, true_a);
        assert!((res.b - true_b).abs() < 0.2, "b estimate {} vs {}", res.b, true_b);
    }

    #[test]
    fn rejects_short_input() {
        let res = estimate_robust_drift(&[1.0, 2.0], &RobustDriftConfig::default());
        assert!(res.is_err());
    }
}
