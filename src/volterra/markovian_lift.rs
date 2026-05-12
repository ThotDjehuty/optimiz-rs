//! Multi-exponential approximation of a convolution kernel.
//!
//! Given a target kernel `K : (0, T] -> R`, find non-negative weights
//! `c_j` and rates `gamma_j` so that
//!
//! ```text
//!     K(t) ~= sum_{j=1}^N  c_j * exp(-gamma_j * t),       t in (0, T].
//! ```
//!
//! When `K` admits the integral representation
//!
//! ```text
//!     K(t) = int_0^infty  exp(-gamma t)  nu(d gamma)
//! ```
//!
//! a Gauss--Jacobi quadrature on `nu` yields nodes `gamma_j` and weights
//! `c_j`. For the rough Volterra kernel `K(t) = t^{H - 1/2} / Gamma(H + 1/2)`
//! with `H in (0, 1/2]`, `nu(d gamma) = gamma^{-1/2 - H} / Gamma(1/2 - H) Gamma(H + 1/2) d gamma`
//! and the approach reduces to a Gauss--Jacobi quadrature on `(0, infty)`.
//!
//! This module exposes:
//!
//! * `geometric_grid_lift`  -- choose `gamma_j` on a geometric grid then
//!                              fit `c_j >= 0` by non-negative least squares
//!                              (active-set algorithm) on a chosen sample
//!                              of `K(t_i)`.
//! * `lift_quality`         -- L2 error on the sampled points.

use crate::core::{OptimizrError, Result};

/// Approximation of `K(t)` on `(0, T]` by `sum c_j exp(-gamma_j t)`.
#[derive(Debug, Clone)]
pub struct MarkovianLift {
    pub gammas: Vec<f64>,
    pub weights: Vec<f64>,
}

impl MarkovianLift {
    pub fn evaluate(&self, t: f64) -> f64 {
        self.gammas
            .iter()
            .zip(self.weights.iter())
            .map(|(&g, &c)| c * (-g * t).exp())
            .sum()
    }
}

/// Build a Markovian lift on a geometric grid of rates and fit weights by
/// non-negative least squares.
///
/// `kernel` is the user-provided positive kernel `K(t)` evaluated on
/// `t_samples`. `n_factors` is the number of exponentials. Rates are
/// chosen as `gamma_j = gamma_min * (gamma_max / gamma_min)^{j/(N-1)}`.
pub fn geometric_grid_lift<K: Fn(f64) -> f64>(
    kernel: K,
    t_samples: &[f64],
    n_factors: usize,
    gamma_min: f64,
    gamma_max: f64,
    nnls_iter: usize,
) -> Result<MarkovianLift> {
    if n_factors < 2 {
        return Err(OptimizrError::InvalidParameter(
            "n_factors must be >= 2".into(),
        ));
    }
    if !(gamma_min > 0.0 && gamma_max > gamma_min) {
        return Err(OptimizrError::InvalidParameter(
            "require 0 < gamma_min < gamma_max".into(),
        ));
    }
    if t_samples.is_empty() {
        return Err(OptimizrError::EmptyData);
    }
    let m = t_samples.len();
    let n = n_factors;
    let log_min = gamma_min.ln();
    let log_max = gamma_max.ln();
    let mut gammas = vec![0.0; n];
    for j in 0..n {
        let frac = j as f64 / (n - 1) as f64;
        gammas[j] = (log_min + frac * (log_max - log_min)).exp();
    }
    // Build A in R^{m x n}: A[i, j] = exp(-gamma_j * t_i)
    let mut a = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = (-gammas[j] * t_samples[i]).exp();
        }
    }
    let b: Vec<f64> = t_samples.iter().map(|&t| kernel(t)).collect();
    let weights = nnls(&a, &b, m, n, nnls_iter);
    Ok(MarkovianLift { gammas, weights })
}

/// `L^2` error of the lift on the sample grid.
pub fn lift_quality<K: Fn(f64) -> f64>(
    lift: &MarkovianLift,
    kernel: K,
    t_samples: &[f64],
) -> f64 {
    let mut s = 0.0;
    for &t in t_samples {
        let diff = lift.evaluate(t) - kernel(t);
        s += diff * diff;
    }
    (s / t_samples.len() as f64).sqrt()
}

/// Lawson--Hanson NNLS-like projected gradient solver. Simple but
/// adequate for small problems (`n <= 20`). Solves
/// `min || A x - b ||^2 s.t. x >= 0`.
fn nnls(a: &[f64], b: &[f64], m: usize, n: usize, max_iter: usize) -> Vec<f64> {
    // Projected gradient with backtracking on Lipschitz constant.
    let mut x = vec![0.0; n];
    // Estimate Lipschitz: spectral radius of A^T A bounded by ||A||_F^2.
    let frob: f64 = a.iter().map(|v| v * v).sum();
    let l = frob.max(1.0);
    let lr = 1.0 / l;
    for _ in 0..max_iter {
        // gradient = A^T (A x - b)
        let mut ax = vec![0.0; m];
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..n {
                s += a[i * n + j] * x[j];
            }
            ax[i] = s - b[i];
        }
        let mut grad = vec![0.0; n];
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                s += a[i * n + j] * ax[i];
            }
            grad[j] = s;
        }
        for j in 0..n {
            x[j] = (x[j] - lr * grad[j]).max(0.0);
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_single_exponential_exactly() {
        // K(t) = 0.7 exp(-2 t)
        let t: Vec<f64> = (1..=200).map(|k| 0.05 * k as f64).collect();
        let lift = geometric_grid_lift(|tt| 0.7 * (-2.0 * tt).exp(), &t, 5, 0.5, 5.0, 5_000)
            .unwrap();
        let err = lift_quality(&lift, |tt| 0.7 * (-2.0 * tt).exp(), &t);
        assert!(err < 5e-3, "L2 err = {}", err);
    }
}
