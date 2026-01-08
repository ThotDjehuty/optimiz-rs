//! Optimal Transport Methods for MFG
use ndarray::Array1;
use crate::core::Result;

pub fn wasserstein_distance(m1: &Array1<f64>, m2: &Array1<f64>, dx: f64) -> f64 {
    m1.iter().zip(m2.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() * dx
}

pub fn sinkhorn_divergence(m1: &Array1<f64>, m2: &Array1<f64>, _eps: f64) -> Result<f64> {
    Ok(wasserstein_distance(m1, m2, 1.0 / m1.len() as f64))
}
