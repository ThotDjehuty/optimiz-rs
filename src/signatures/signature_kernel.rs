//! Signature kernel via the Salvi--Cass--Lyons PDE.
//!
//! For two paths `X : [0, S] -> R^d` and `Y : [0, T] -> R^d`, the
//! signature kernel
//!
//! ```text
//!     K(s, t) = < S(X)_{0, s} , S(Y)_{0, t} >
//! ```
//!
//! satisfies the linear hyperbolic PDE
//!
//! ```text
//!     d^2 K / (ds dt) = < dX_s, dY_t >  K(s, t),     K(s, 0) = K(0, t) = 1.
//! ```
//!
//! This module integrates the PDE on a uniform grid via the Goursat
//! finite-difference scheme of Salvi et al. (2021):
//!
//! ```text
//!     K_{i+1, j+1} = K_{i+1, j} + K_{i, j+1} - K_{i, j}
//!                    + (Delta x_i . Delta y_j) * 0.5 * (K_{i+1, j} + K_{i, j+1})
//! ```
//!
//! and returns the value `K(S, T)`.

use crate::core::{OptimizrError, Result};

/// Signature kernel evaluation result.
#[derive(Debug, Clone)]
pub struct SignatureKernelResult {
    pub value: f64,
    pub grid: Vec<Vec<f64>>, // K_{i, j}, shape (n_x, n_y)
}

/// Compute the signature kernel between piecewise-linear paths `x` and
/// `y` (both shape `(n, d)` with arbitrary `n`).
pub fn signature_kernel(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<SignatureKernelResult> {
    if x.len() < 2 || y.len() < 2 {
        return Err(OptimizrError::InvalidInput(
            "both paths must have at least two points".into(),
        ));
    }
    let d = x[0].len();
    if d == 0 {
        return Err(OptimizrError::InvalidInput("zero-dimensional path".into()));
    }
    for p in x.iter().chain(y.iter()) {
        if p.len() != d {
            return Err(OptimizrError::DimensionMismatch {
                expected: d,
                actual: p.len(),
            });
        }
    }
    let nx = x.len();
    let ny = y.len();
    // Pre-compute increments
    let dx: Vec<Vec<f64>> = (0..nx - 1)
        .map(|i| (0..d).map(|k| x[i + 1][k] - x[i][k]).collect())
        .collect();
    let dy: Vec<Vec<f64>> = (0..ny - 1)
        .map(|j| (0..d).map(|k| y[j + 1][k] - y[j][k]).collect())
        .collect();

    let mut k_grid = vec![vec![0.0; ny]; nx];
    for i in 0..nx {
        k_grid[i][0] = 1.0;
    }
    for j in 0..ny {
        k_grid[0][j] = 1.0;
    }
    for i in 0..nx - 1 {
        for j in 0..ny - 1 {
            let dot: f64 = (0..d).map(|c| dx[i][c] * dy[j][c]).sum();
            let avg = 0.5 * (k_grid[i + 1][j] + k_grid[i][j + 1]);
            k_grid[i + 1][j + 1] = k_grid[i + 1][j] + k_grid[i][j + 1] - k_grid[i][j] + dot * avg;
        }
    }
    Ok(SignatureKernelResult {
        value: k_grid[nx - 1][ny - 1],
        grid: k_grid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identical linear segments yield K(1, 1) = exp(||Delta||^2) (closed
    /// form for the signature inner product of a single segment with
    /// itself).
    #[test]
    fn identical_linear_paths_give_exponential_kernel() {
        let delta = vec![0.4, -0.1];
        let n = 200;
        let path: Vec<Vec<f64>> = (0..=n)
            .map(|k| {
                let t = k as f64 / n as f64;
                vec![t * delta[0], t * delta[1]]
            })
            .collect();
        let res = signature_kernel(&path, &path).unwrap();
        let exact = (delta.iter().map(|v| v * v).sum::<f64>()).exp();
        let err = (res.value - exact).abs();
        assert!(err < 1e-2, "K = {} expected {} err = {}", res.value, exact, err);
    }
}
