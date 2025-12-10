//! Sparse Optimization Algorithms
//!
//! Generic implementations of sparse optimization and decomposition methods:
//! - Sparse PCA with L1 regularization
//! - Box & Tao decomposition (Robust PCA)
//! - Elastic Net for sparse linear models
//! - ADMM (Alternating Direction Method of Multipliers)
//!
//! Based on:
//! - d'Aspremont (2011): "Identifying Small Mean Reverting Portfolios"
//! - Candès et al. (2011): "Robust Principal Component Analysis?"
//! - Zou & Hastie (2005): "Regularization and Variable Selection via Elastic Net"

use crate::core::{OptimizrError, OptimizrResult};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Norm, SVD};

/// Soft thresholding operator
///
/// S_λ(x) = sign(x) * max(|x| - λ, 0)
#[inline]
pub fn soft_threshold(x: f64, lambda: f64) -> f64 {
    let abs_x = x.abs();
    if abs_x <= lambda {
        0.0
    } else {
        x.signum() * (abs_x - lambda)
    }
}

/// Vectorized soft thresholding
pub fn soft_threshold_array(arr: &Array1<f64>, lambda: f64) -> Array1<f64> {
    arr.mapv(|x| soft_threshold(x, lambda))
}

/// Soft thresholding for matrices
pub fn soft_threshold_matrix(mat: &Array2<f64>, lambda: f64) -> Array2<f64> {
    mat.mapv(|x| soft_threshold(x, lambda))
}

/// SVD soft thresholding for nuclear norm regularization
///
/// Returns U * S_λ(Σ) * V^T where S_λ is soft thresholding on singular values
pub fn svd_soft_threshold(mat: &Array2<f64>, lambda: f64) -> OptimizrResult<Array2<f64>> {
    let (u, s, vt) = mat
        .svd(true, true)
        .map_err(|e| OptimizrError::ComputationError(format!("SVD failed: {}", e)))?;

    let u = u.ok_or_else(|| OptimizrError::ComputationError("SVD U is None".to_string()))?;
    let vt = vt.ok_or_else(|| OptimizrError::ComputationError("SVD Vt is None".to_string()))?;

    // Soft threshold singular values
    let s_thresh = s.mapv(|val| soft_threshold(val, lambda));

    // Reconstruct: U * diag(s_thresh) * V^T
    let s_diag = Array2::from_diag(&s_thresh);
    let us = u.dot(&s_diag);
    Ok(us.dot(&vt))
}

/// Result from Sparse PCA
#[derive(Debug, Clone)]
pub struct SparsePCAResult {
    pub weights: Array2<f64>, // (n_components, n_features) sparse weights
    pub variance_explained: Array1<f64>, // Variance explained per component
    pub sparsity: Array1<f64>, // Sparsity level per component
    pub iterations: Array1<usize>, // Iterations to converge per component
    pub converged: Vec<bool>, // Convergence flag per component
}

/// Sparse Principal Component Analysis with L1 regularization
///
/// Finds sparse principal components that maximize variance with sparsity constraint:
///
/// max_w  w^T Σ w - λ ||w||_1  s.t. ||w||_2 = 1
///
/// # Arguments
/// * `covariance` - Covariance matrix (n_features, n_features)
/// * `n_components` - Number of sparse components to extract
/// * `lambda` - Sparsity parameter (larger = sparser)
/// * `max_iter` - Maximum iterations per component
/// * `tol` - Convergence tolerance
///
/// # Returns
/// `SparsePCAResult` with sparse principal components
pub fn sparse_pca(
    covariance: &Array2<f64>,
    n_components: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizrResult<SparsePCAResult> {
    let n_features = covariance.nrows();

    if covariance.ncols() != n_features {
        return Err(OptimizrError::InvalidInput(
            "Covariance matrix must be square".to_string(),
        ));
    }

    if n_components > n_features {
        return Err(OptimizrError::InvalidInput(format!(
            "n_components ({}) > n_features ({})",
            n_components, n_features
        )));
    }

    let mut weights = Array2::zeros((n_components, n_features));
    let mut variance_explained = Array1::zeros(n_components);
    let mut sparsity = Array1::zeros(n_components);
    let mut iterations_vec = Array1::zeros(n_components);
    let mut converged_vec = vec![false; n_components];

    let mut residual_cov = covariance.clone();

    for comp in 0..n_components {
        // Initialize with leading eigenvector
        let (_, _s, vt) = residual_cov
            .svd(false, true)
            .map_err(|e| OptimizrError::ComputationError(format!("SVD failed: {}", e)))?;

        let vt = vt.ok_or_else(|| OptimizrError::ComputationError("SVD Vt is None".to_string()))?;
        let mut w = vt.row(0).to_owned();

        let mut converged = false;
        let mut iter = 0;

        // Iterative soft-thresholding
        for _ in 0..max_iter {
            let w_old = w.clone();

            // Update: w_new = Σ * w
            let mut w_new = residual_cov.dot(&w);

            // Apply soft thresholding
            w_new = soft_threshold_array(&w_new, lambda);

            // Normalize
            let norm = w_new.norm_l2();
            if norm > 1e-10 {
                w_new /= norm;
            } else {
                // Degenerate case - use previous
                break;
            }

            // Check convergence
            let diff = (&w_new - &w_old).norm_l2();
            if diff < tol {
                converged = true;
                w = w_new;
                break;
            }

            w = w_new;
            iter += 1;
        }

        // Store results for this component
        weights.row_mut(comp).assign(&w);

        // Variance explained
        let var_exp = w.dot(&residual_cov.dot(&w));
        variance_explained[comp] = var_exp.max(0.0);

        // Sparsity (proportion of non-zero elements)
        let non_zero = w.iter().filter(|&&x| x.abs() > 1e-10).count();
        sparsity[comp] = 1.0 - (non_zero as f64 / n_features as f64);

        iterations_vec[comp] = iter as f64;
        converged_vec[comp] = converged;

        // Deflate covariance matrix
        let w_outer = outer_product(&w, &w);
        residual_cov = &residual_cov - &(w_outer * var_exp);
    }

    Ok(SparsePCAResult {
        weights,
        variance_explained,
        sparsity,
        iterations: iterations_vec.mapv(|x| x as usize),
        converged: converged_vec,
    })
}

/// Helper: outer product of two vectors
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }

    result
}

/// Result from Box & Tao Decomposition
#[derive(Debug, Clone)]
pub struct BoxTaoResult {
    pub low_rank: Array2<f64>,      // Low-rank component (common factors)
    pub sparse: Array2<f64>,        // Sparse component (idiosyncratic)
    pub noise: Array2<f64>,         // Noise/residual
    pub rank: usize,                // Rank of low-rank component
    pub sparsity: f64,              // Sparsity of sparse component
    pub iterations: usize,          // Number of ADMM iterations
    pub converged: bool,            // Convergence flag
    pub objective_values: Vec<f64>, // Objective function history
}

/// Box & Tao Decomposition (Robust PCA)
///
/// Decomposes matrix into low-rank + sparse + noise:
///
/// X = L + S + N
///
/// min_{L,S}  ||L||_* + λ ||S||_1  s.t. ||X - L - S||_F ≤ ε
///
/// Solved via ADMM (Alternating Direction Method of Multipliers)
///
/// # Arguments
/// * `matrix` - Input matrix to decompose
/// * `lambda` - Sparsity parameter for sparse component
/// * `mu` - Penalty parameter for ADMM
/// * `max_iter` - Maximum ADMM iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// `BoxTaoResult` with decomposed components
pub fn box_tao_decomposition(
    matrix: &Array2<f64>,
    lambda: f64,
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizrResult<BoxTaoResult> {
    let (m, n) = matrix.dim();

    // Initialize L, S, Y (dual variable)
    let mut low_rank = Array2::<f64>::zeros((m, n));
    let mut sparse = Array2::<f64>::zeros((m, n));
    let mut dual = Array2::<f64>::zeros((m, n));

    let mut objective_values = Vec::with_capacity(max_iter);
    let mut converged = false;
    let mut iter = 0;

    let rho = mu; // ADMM penalty parameter

    for _ in 0..max_iter {
        // Update L: SVD soft-thresholding
        let l_update = matrix - &sparse + &(&dual / rho);
        low_rank = svd_soft_threshold(&l_update, 1.0 / rho)?;

        // Update S: Element-wise soft-thresholding
        let s_update = matrix - &low_rank + &(&dual / rho);
        sparse = soft_threshold_matrix(&s_update, lambda / rho);

        // Update dual variable Y
        let residual = matrix - &low_rank - &sparse;
        let primal_residual = residual.norm_l2();
        dual = &dual + &(residual * rho);

        // Compute objective
        let nuclear_norm = low_rank
            .svd(false, false)
            .map(|(_, s, _)| s.sum())
            .unwrap_or(0.0);
        let l1_norm = sparse.iter().map(|x| x.abs()).sum::<f64>();
        let objective = nuclear_norm + lambda * l1_norm;
        objective_values.push(objective);

        // Check convergence
        let dual_residual = if iter > 0 {
            rho * (&sparse - matrix).norm_l2()
        } else {
            f64::INFINITY
        };

        if primal_residual < tol && dual_residual < tol {
            converged = true;
            break;
        }

        iter += 1;
    }

    let noise = matrix - &low_rank - &sparse;

    // Compute rank of low-rank component
    let rank = low_rank
        .svd(false, false)
        .map(|(_, s, _)| s.iter().filter(|&&x| x > 1e-10).count())
        .unwrap_or(0);

    // Compute sparsity of sparse component
    let total_elements = (m * n) as f64;
    let non_zero = sparse.iter().filter(|&&x| x.abs() > 1e-10).count() as f64;
    let sparsity = 1.0 - (non_zero / total_elements);

    Ok(BoxTaoResult {
        low_rank,
        sparse,
        noise,
        rank,
        sparsity,
        iterations: iter,
        converged,
        objective_values,
    })
}

/// Result from Elastic Net regression
#[derive(Debug, Clone)]
pub struct ElasticNetResult {
    pub weights: Array1<f64>, // Sparse regression coefficients
    pub intercept: f64,       // Intercept term
    pub sparsity: f64,        // Sparsity level
    pub iterations: usize,    // Iterations to converge
    pub converged: bool,      // Convergence flag
    pub objective_value: f64, // Final objective value
}

/// Elastic Net Regression
///
/// Sparse linear regression with L1 + L2 regularization:
///
/// min_w  (1/2n) ||y - Xw||_2^2 + λ₁ ||w||_1 + (λ₂/2) ||w||_2^2
///
/// Combines LASSO (L1) sparsity with Ridge (L2) smoothness
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `y` - Target vector (n_samples,)
/// * `lambda_l1` - L1 regularization (sparsity)
/// * `lambda_l2` - L2 regularization (smoothness)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// `ElasticNetResult` with sparse coefficients
pub fn elastic_net(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lambda_l1: f64,
    lambda_l2: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizrResult<ElasticNetResult> {
    let (n_samples, n_features) = x.dim();

    if y.len() != n_samples {
        return Err(OptimizrError::InvalidInput(format!(
            "y length ({}) != n_samples ({})",
            y.len(),
            n_samples
        )));
    }

    // Center data
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let y_mean = y.mean().unwrap();

    let x_centered = x - &x_mean;
    let y_centered = y - y_mean;

    // Initialize weights
    let mut w = Array1::zeros(n_features);
    let mut converged = false;
    let mut iter = 0;

    // Coordinate descent
    for _ in 0..max_iter {
        let w_old = w.clone();

        for j in 0..n_features {
            // Compute residual excluding feature j
            let mut residual = y_centered.clone();
            for k in 0..n_features {
                if k != j {
                    let x_k = x_centered.column(k);
                    residual = &residual - &(&x_k * w[k]);
                }
            }

            // Update weight j
            let x_j = x_centered.column(j);
            let rho = x_j.dot(&residual) / n_samples as f64;
            let z_j = x_j.dot(&x_j) / n_samples as f64 + lambda_l2;

            w[j] = soft_threshold(rho, lambda_l1) / z_j;
        }

        // Check convergence
        let diff = (&w - &w_old).norm_l2();
        if diff < tol {
            converged = true;
            break;
        }

        iter += 1;
    }

    // Compute intercept
    let intercept = y_mean - x_mean.dot(&w);

    // Compute sparsity
    let non_zero = w.iter().filter(|&&x| x.abs() > 1e-10).count() as f64;
    let sparsity = 1.0 - (non_zero / n_features as f64);

    // Compute objective value
    let predictions = &x_centered.dot(&w) + y_mean;
    let residuals = y - &predictions;
    let mse = residuals.dot(&residuals) / (2.0 * n_samples as f64);
    let l1_penalty = lambda_l1 * w.iter().map(|x| x.abs()).sum::<f64>();
    let l2_penalty = 0.5 * lambda_l2 * w.dot(&w);
    let objective_value = mse + l1_penalty + l2_penalty;

    Ok(ElasticNetResult {
        weights: w,
        intercept,
        sparsity,
        iterations: iter,
        converged,
        objective_value,
    })
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;

#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (covariance, n_components=1, lambda=0.1, max_iter=1000, tol=1e-6))]
pub fn sparse_pca_py(
    py: Python,
    covariance: PyReadonlyArray2<f64>,
    n_components: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyObject> {
    let cov = covariance.as_array().to_owned();

    let result = sparse_pca(&cov, n_components, lambda, max_iter, tol)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "weights",
        PyArray2::from_owned_array_bound(py, result.weights),
    )?;
    dict.set_item(
        "variance_explained",
        PyArray1::from_owned_array_bound(py, result.variance_explained),
    )?;
    dict.set_item(
        "sparsity",
        PyArray1::from_owned_array_bound(py, result.sparsity),
    )?;
    dict.set_item("iterations", result.iterations.to_vec())?;
    dict.set_item("converged", result.converged)?;

    Ok(dict.into())
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (matrix, lambda=0.1, mu=1.0, max_iter=500, tol=1e-5))]
pub fn box_tao_decomposition_py(
    py: Python,
    matrix: PyReadonlyArray2<f64>,
    lambda: f64,
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyObject> {
    let mat = matrix.as_array().to_owned();

    let result = box_tao_decomposition(&mat, lambda, mu, max_iter, tol)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "low_rank",
        PyArray2::from_owned_array_bound(py, result.low_rank),
    )?;
    dict.set_item(
        "sparse",
        PyArray2::from_owned_array_bound(py, result.sparse),
    )?;
    dict.set_item("noise", PyArray2::from_owned_array_bound(py, result.noise))?;
    dict.set_item("rank", result.rank)?;
    dict.set_item("sparsity", result.sparsity)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("objective_values", result.objective_values)?;

    Ok(dict.into())
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (x, y, lambda_l1=0.1, lambda_l2=0.1, max_iter=1000, tol=1e-6))]
pub fn elastic_net_py(
    py: Python,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    lambda_l1: f64,
    lambda_l2: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyObject> {
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();

    let result = elastic_net(&x_arr, &y_arr, lambda_l1, lambda_l2, max_iter, tol)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "weights",
        PyArray1::from_owned_array_bound(py, result.weights),
    )?;
    dict.set_item("intercept", result.intercept)?;
    dict.set_item("sparsity", result.sparsity)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("objective_value", result.objective_value)?;

    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_soft_threshold() {
        assert_relative_eq!(soft_threshold(3.0, 1.0), 2.0);
        assert_relative_eq!(soft_threshold(-3.0, 1.0), -2.0);
        assert_relative_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_relative_eq!(soft_threshold(-0.5, 1.0), 0.0);
    }

    #[test]
    fn test_sparse_pca_simple() {
        // Simple 3x3 covariance matrix
        let cov = Array2::from_shape_vec((3, 3), vec![4.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0])
            .unwrap();

        let result = sparse_pca(&cov, 1, 0.1, 100, 1e-6).unwrap();

        assert_eq!(result.weights.nrows(), 1);
        assert_eq!(result.weights.ncols(), 3);
        assert!(result.variance_explained[0] > 0.0);
        assert!(result.converged[0]);
    }

    #[test]
    fn test_elastic_net_simple() {
        // Simple linear problem: y = 2*x1 + 3*x2
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0],
        )
        .unwrap();

        let y = Array1::from_vec(vec![5.0, 7.0, 12.0, 17.0, 22.0]);

        let result = elastic_net(&x, &y, 0.01, 0.01, 1000, 1e-6).unwrap();

        assert!(result.converged);
        // Coefficients should be approximately [2, 3]
        assert!((result.weights[0] - 2.0).abs() < 0.5);
        assert!((result.weights[1] - 3.0).abs() < 0.5);
    }
}
