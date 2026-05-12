//! Graph Laplacians for dense weighted similarity matrices.
//!
//! For a non-negative symmetric weight matrix `W in R^{n x n}` and degree
//! diagonal `D = diag(W * 1)`, the three standard Laplacians are
//!
//! ```text
//!     L         = D - W                     (combinatorial)
//!     L_sym     = I - D^{-1/2} W D^{-1/2}   (symmetric normalised)
//!     L_rw      = I - D^{-1} W              (random-walk)
//! ```
//!
//! All operate on dense `ndarray::Array2<f64>`.

use ndarray::{Array2, ArrayView2};

use crate::core::{OptimizrError, Result};

/// Laplacian variant.
#[derive(Debug, Clone, Copy)]
pub enum LaplacianKind {
    Combinatorial,
    SymmetricNormalised,
    RandomWalk,
}

fn check_weight(w: ArrayView2<f64>) -> Result<()> {
    let (n, m) = (w.nrows(), w.ncols());
    if n == 0 || n != m {
        return Err(OptimizrError::InvalidInput(
            "weight matrix must be square and non-empty".into(),
        ));
    }
    for &x in w.iter() {
        if !x.is_finite() || x < 0.0 {
            return Err(OptimizrError::InvalidInput(
                "weight matrix must be non-negative and finite".into(),
            ));
        }
    }
    Ok(())
}

/// Combinatorial Laplacian `L = D - W`.
pub fn combinatorial_laplacian(w: ArrayView2<f64>) -> Result<Array2<f64>> {
    check_weight(w)?;
    let n = w.nrows();
    let mut l = -w.to_owned();
    for i in 0..n {
        let d_i: f64 = w.row(i).sum();
        l[[i, i]] += d_i;
    }
    Ok(l)
}

/// Symmetric normalised Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}`.
pub fn normalised_laplacian(w: ArrayView2<f64>) -> Result<Array2<f64>> {
    check_weight(w)?;
    let n = w.nrows();
    let mut d_inv_sqrt = vec![0.0; n];
    for i in 0..n {
        let d_i: f64 = w.row(i).sum();
        d_inv_sqrt[i] = if d_i > 0.0 { 1.0 / d_i.sqrt() } else { 0.0 };
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let off = -d_inv_sqrt[i] * w[[i, j]] * d_inv_sqrt[j];
            l[[i, j]] = if i == j { 1.0 + off } else { off };
        }
    }
    Ok(l)
}

/// Random-walk Laplacian `L_rw = I - D^{-1} W`.
pub fn random_walk_laplacian(w: ArrayView2<f64>) -> Result<Array2<f64>> {
    check_weight(w)?;
    let n = w.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let d_i: f64 = w.row(i).sum();
        let inv = if d_i > 0.0 { 1.0 / d_i } else { 0.0 };
        for j in 0..n {
            l[[i, j]] = if i == j { 1.0 - inv * w[[i, j]] } else { -inv * w[[i, j]] };
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn laplacian_kernel_includes_constant() {
        let w = array![
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let l = combinatorial_laplacian(w.view()).unwrap();
        let one = ndarray::Array1::<f64>::ones(3);
        let lone = l.dot(&one);
        for v in lone.iter() {
            assert!(v.abs() < 1e-12);
        }
    }
}
