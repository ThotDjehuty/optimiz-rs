//! Generative calibration hooks
//! ==============================
//!
//! Provides a [`GenerativeSampler`] trait wrapping any external
//! parameterised sampler `θ ↦ X^θ_1, ..., X^θ_M` and a Maximum Mean
//! Discrepancy (MMD) loss with a Gaussian kernel
//!
//! ```text
//! MMD²(P, Q) = (1/M²) Σ_{i,j} k(x_i, x_j) + (1/N²) Σ_{i,j} k(y_i, y_j)
//!              - (2/(MN)) Σ_{i,j} k(x_i, y_j)
//! ```
//!
//! and a one-step finite-difference calibration routine that returns a new
//! parameter vector with a centred-difference gradient descent update.

use crate::core::{OptimizrError, Result};

pub trait GenerativeSampler {
    /// Draw `n_samples` from the distribution parameterised by `theta`.
    fn sample(&self, theta: &[f64], n_samples: usize, seed: u64) -> Result<Vec<f64>>;
}

#[derive(Clone, Debug)]
pub struct MmdLoss {
    /// Gaussian kernel bandwidth (σ > 0).
    pub sigma: f64,
}

fn gaussian_kernel(x: f64, y: f64, sigma: f64) -> f64 {
    let d = (x - y) / sigma;
    (-0.5 * d * d).exp()
}

pub fn mmd_distance(x: &[f64], y: &[f64], loss: &MmdLoss) -> Result<f64> {
    if loss.sigma <= 0.0 {
        return Err(OptimizrError::InvalidParameter("sigma > 0".into()));
    }
    if x.is_empty() || y.is_empty() {
        return Err(OptimizrError::EmptyData);
    }
    let m = x.len();
    let n = y.len();
    let mut s_xx = 0.0;
    for i in 0..m { for j in 0..m { s_xx += gaussian_kernel(x[i], x[j], loss.sigma); } }
    let mut s_yy = 0.0;
    for i in 0..n { for j in 0..n { s_yy += gaussian_kernel(y[i], y[j], loss.sigma); } }
    let mut s_xy = 0.0;
    for i in 0..m { for j in 0..n { s_xy += gaussian_kernel(x[i], y[j], loss.sigma); } }
    let mmd2 = s_xx / (m * m) as f64 + s_yy / (n * n) as f64 - 2.0 * s_xy / (m * n) as f64;
    Ok(mmd2.max(0.0).sqrt())
}

pub fn calibration_step<S: GenerativeSampler>(
    sampler: &S,
    target: &[f64],
    theta: &[f64],
    loss: &MmdLoss,
    learning_rate: f64,
    finite_diff_eps: f64,
    n_samples: usize,
    seed: u64,
) -> Result<Vec<f64>> {
    let p = theta.len();
    if p == 0 {
        return Err(OptimizrError::InvalidParameter("theta is empty".into()));
    }
    let mut grad = vec![0.0f64; p];
    for k in 0..p {
        let mut tp = theta.to_vec();
        let mut tm = theta.to_vec();
        tp[k] += finite_diff_eps;
        tm[k] -= finite_diff_eps;
        let xp = sampler.sample(&tp, n_samples, seed)?;
        let xm = sampler.sample(&tm, n_samples, seed)?;
        let lp = mmd_distance(&xp, target, loss)?;
        let lm = mmd_distance(&xm, target, loss)?;
        grad[k] = (lp - lm) / (2.0 * finite_diff_eps);
    }
    let new_theta: Vec<f64> = theta.iter().zip(grad.iter()).map(|(&t, &g)| t - learning_rate * g).collect();
    Ok(new_theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MMD between two identical samples is zero.
    #[test]
    fn mmd_self_distance_is_zero() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let d = mmd_distance(&x, &x, &MmdLoss { sigma: 1.0 }).unwrap();
        assert!(d.abs() < 1e-10);
    }

    /// MMD between samples shifted by 5σ is large (positive).
    #[test]
    fn mmd_increases_with_shift() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|v| v + 5.0).collect();
        let loss = MmdLoss { sigma: 1.0 };
        let d_self = mmd_distance(&x, &x, &loss).unwrap();
        let d_shift = mmd_distance(&x, &y, &loss).unwrap();
        assert!(d_shift > d_self + 0.5);
    }

    /// Calibration step decreases MMD on a trivially identifiable scalar
    /// shift problem `sample(θ) = θ + i/10`.
    struct ShiftSampler;
    impl GenerativeSampler for ShiftSampler {
        fn sample(&self, theta: &[f64], n: usize, _seed: u64) -> Result<Vec<f64>> {
            let t = theta[0];
            Ok((0..n).map(|i| t + i as f64 / 10.0).collect())
        }
    }

    #[test]
    fn calibration_step_descends() {
        let s = ShiftSampler;
        let target: Vec<f64> = (0..30).map(|i| 1.0 + i as f64 / 10.0).collect();
        let loss = MmdLoss { sigma: 1.0 };
        let theta = vec![0.0];
        let l0 = mmd_distance(&s.sample(&theta, 30, 0).unwrap(), &target, &loss).unwrap();
        let new_theta = calibration_step(&s, &target, &theta, &loss, 1.0, 1e-2, 30, 0).unwrap();
        let l1 = mmd_distance(&s.sample(&new_theta, 30, 0).unwrap(), &target, &loss).unwrap();
        assert!(l1 < l0, "loss did not decrease: {l0} → {l1}");
    }
}
