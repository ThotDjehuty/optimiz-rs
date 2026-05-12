//! Random projection of the path signature.
//!
//! Following Cuchiero, Schmocker & Teichmann (2023), one defines a
//! random reservoir on `R^N` driven by the controlled SDE
//!
//! ```text
//!     dZ_t = A_0 Z_t dt + sum_{i=1}^d A_i Z_t dX^i_t,           Z_0 in R^N,
//! ```
//!
//! where the matrices `A_0, ..., A_d in R^{N x N}` are drawn from a
//! random ensemble (typically i.i.d. Gaussian entries with variance
//! `1 / N`). The map `X -> Z_T` provides a finite-dimensional, randomly
//! projected representation of the signature.
//!
//! This implementation integrates the controlled equation with an
//! Euler--Maruyama-type scheme on the supplied path samples.

use crate::core::{OptimizrError, Result};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Random reservoir parameters.
#[derive(Debug, Clone)]
pub struct RandomSignatureConfig {
    pub reservoir_dim: usize,
    pub seed: u64,
    pub variance: f64,
}

impl Default for RandomSignatureConfig {
    fn default() -> Self {
        Self {
            reservoir_dim: 32,
            seed: 0,
            variance: 1.0,
        }
    }
}

/// Result of a random reservoir integration.
#[derive(Debug, Clone)]
pub struct RandomSignatureResult {
    pub trajectory: Vec<Vec<f64>>, // length = path.len(), each row in R^N
}

fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Compute `Z_T` for the controlled SDE driven by the piecewise-linear
/// path interpolating `path`. Returns the full reservoir trajectory.
pub fn random_signature(
    path: &[Vec<f64>],
    cfg: &RandomSignatureConfig,
) -> Result<RandomSignatureResult> {
    if path.len() < 2 {
        return Err(OptimizrError::InvalidInput(
            "path must contain at least two points".into(),
        ));
    }
    let d = path[0].len();
    if d == 0 {
        return Err(OptimizrError::InvalidInput("zero-dimensional path".into()));
    }
    for p in path {
        if p.len() != d {
            return Err(OptimizrError::DimensionMismatch {
                expected: d,
                actual: p.len(),
            });
        }
    }
    if cfg.reservoir_dim == 0 {
        return Err(OptimizrError::InvalidParameter(
            "reservoir_dim > 0 required".into(),
        ));
    }
    if cfg.variance <= 0.0 {
        return Err(OptimizrError::InvalidParameter(
            "variance > 0 required".into(),
        ));
    }
    let n = cfg.reservoir_dim;
    let std_dev = (cfg.variance / n as f64).sqrt();
    let normal = Normal::new(0.0, std_dev)
        .map_err(|e| OptimizrError::NumericalError(format!("normal init: {}", e)))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);

    // Sample matrices A_0, ..., A_d.
    let mut matrices: Vec<Vec<f64>> = (0..=d)
        .map(|_| (0..n * n).map(|_| normal.sample(&mut rng)).collect())
        .collect();
    // Drift A_0 typically scaled smaller; here we keep the same variance.
    let _ = &mut matrices;

    let mut z = vec![0.0; n];
    z[0] = 1.0;
    let mut traj = Vec::with_capacity(path.len());
    traj.push(z.clone());
    let dt = 1.0 / (path.len() - 1) as f64;
    for n_idx in 1..path.len() {
        let mut delta = vec![0.0; d];
        for j in 0..d {
            delta[j] = path[n_idx][j] - path[n_idx - 1][j];
        }
        // Z_{n+1} = Z_n + A_0 Z_n dt + sum_i A_i Z_n delta_i
        let drift = matvec(&matrices[0], n, &z);
        let mut new_z = z.clone();
        for i in 0..n {
            new_z[i] += dt * drift[i];
        }
        for i in 0..d {
            let inc = matvec(&matrices[i + 1], n, &z);
            for j in 0..n {
                new_z[j] += inc[j] * delta[i];
            }
        }
        z = new_z;
        traj.push(z.clone());
    }
    Ok(RandomSignatureResult { trajectory: traj })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_seed_reproduces_output() {
        let path = vec![vec![0.0], vec![0.5], vec![1.0]];
        let cfg = RandomSignatureConfig {
            reservoir_dim: 4,
            seed: 42,
            variance: 1.0,
        };
        let r1 = random_signature(&path, &cfg).unwrap();
        let r2 = random_signature(&path, &cfg).unwrap();
        for (a, b) in r1.trajectory.iter().zip(r2.trajectory.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-15);
            }
        }
    }
}
