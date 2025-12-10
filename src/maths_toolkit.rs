//! Mathematical Toolkit
//!
//! Common mathematical operations used across optimization algorithms:
//! - Numerical differentiation (gradients, Hessians, Jacobians)
//! - Statistical functions (moments, correlations, distributions)
//! - Linear algebra utilities (matrix operations, decompositions)
//! - Numerical integration and interpolation
//! - Special functions and approximations
//!
//! This module provides generic, reusable mathematical operations
//! that are independent of any specific application domain.

use crate::core::{OptimizrError, OptimizrResult};
use ndarray::{Array1, Array2};

// ============================================================================
// Numerical Differentiation
// ============================================================================

/// Compute gradient using central finite differences
///
/// ∇f(x) ≈ [f(x + h·e_i) - f(x - h·e_i)] / (2h)
///
/// # Arguments
/// * `f` - Function to differentiate
/// * `x` - Point at which to compute gradient
/// * `h` - Step size (default: 1e-5)
///
/// # Returns
/// Gradient vector ∇f(x)
///
/// # Example
/// ```rust
/// use ndarray::array;
/// use optimizr::maths_toolkit::gradient;
///
/// // f(x,y) = x² + 2y²
/// let f = |x: &[f64]| x[0].powi(2) + 2.0 * x[1].powi(2);
/// let x = array![1.0, 2.0];
/// let grad = gradient(&f, &x, 1e-5).unwrap();
/// // grad ≈ [2.0, 8.0]
/// ```
pub fn gradient<F>(f: &F, x: &Array1<f64>, h: f64) -> OptimizrResult<Array1<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = Array1::zeros(n);
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..n {
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * h);

        // Reset for next iteration
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    Ok(grad)
}

/// Compute Hessian matrix using finite differences
///
/// H_ij = ∂²f/∂x_i∂x_j ≈ [f(x+h·e_i+h·e_j) - f(x+h·e_i-h·e_j) - f(x-h·e_i+h·e_j) + f(x-h·e_i-h·e_j)] / (4h²)
///
/// # Arguments
/// * `f` - Function to differentiate
/// * `x` - Point at which to compute Hessian
/// * `h` - Step size (default: 1e-4)
///
/// # Returns
/// Hessian matrix H(x)
pub fn hessian<F>(f: &F, x: &Array1<f64>, h: f64) -> OptimizrResult<Array2<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    #[allow(non_snake_case)]  // H is standard mathematical notation for Hessian
    let mut H = Array2::zeros((n, n));
    let mut x_work = x.to_vec();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
                x_work[i] = x[i] + h;
                let f_plus = f(&x_work);

                x_work[i] = x[i] - h;
                let f_minus = f(&x_work);

                x_work[i] = x[i];
                let f_center = f(&x_work);

                H[[i, i]] = (f_plus - 2.0 * f_center + f_minus) / (h * h);
            } else {
                // Off-diagonal: mixed partial derivative
                x_work[i] = x[i] + h;
                x_work[j] = x[j] + h;
                let f_pp = f(&x_work);

                x_work[j] = x[j] - h;
                let f_pm = f(&x_work);

                x_work[i] = x[i] - h;
                let f_mm = f(&x_work);

                x_work[j] = x[j] + h;
                let f_mp = f(&x_work);

                H[[i, j]] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);

                // Reset
                x_work[i] = x[i];
                x_work[j] = x[j];
            }
        }
    }

    Ok(H)
}

/// Compute Jacobian matrix for vector-valued function
///
/// J_ij = ∂f_i/∂x_j
///
/// # Arguments
/// * `f` - Vector-valued function f: ℝⁿ → ℝᵐ
/// * `x` - Point at which to compute Jacobian
/// * `h` - Step size
///
/// # Returns
/// Jacobian matrix J(x) of shape (m, n)
pub fn jacobian<F>(f: &F, x: &Array1<f64>, h: f64) -> OptimizrResult<Array2<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x.len();
    let f_x = f(&x.to_vec());
    let m = f_x.len();

    #[allow(non_snake_case)]  // J is standard mathematical notation for Jacobian
    let mut J = Array2::zeros((m, n));
    let mut x_work = x.to_vec();

    for j in 0..n {
        x_work[j] = x[j] + h;
        let f_plus = f(&x_work);

        x_work[j] = x[j] - h;
        let f_minus = f(&x_work);

        for i in 0..m {
            J[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
        }

        x_work[j] = x[j];
    }

    Ok(J)
}

// ============================================================================
// Statistical Functions
// ============================================================================

/// Compute mean of a data series
#[inline]
pub fn mean(data: &Array1<f64>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sum() / data.len() as f64
}

/// Compute variance with optional Bessel's correction
#[inline]
pub fn variance(data: &Array1<f64>, ddof: usize) -> f64 {
    if data.len() <= ddof {
        return 0.0;
    }
    let m = mean(data);
    let sum_sq: f64 = data.iter().map(|&x| (x - m).powi(2)).sum();
    sum_sq / (data.len() - ddof) as f64
}

/// Compute standard deviation
#[inline]
pub fn std_dev(data: &Array1<f64>, ddof: usize) -> f64 {
    variance(data, ddof).sqrt()
}

/// Compute skewness (3rd standardized moment)
///
/// Skewness = E[(X - μ)³] / σ³
///
/// - Skewness > 0: Right-skewed (long right tail)
/// - Skewness < 0: Left-skewed (long left tail)
/// - Skewness ≈ 0: Symmetric
pub fn skewness(data: &Array1<f64>) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }

    let m = mean(data);
    let std = std_dev(data, 1);

    if std < 1e-10 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_cubed: f64 = data.iter().map(|&x| ((x - m) / std).powi(3)).sum();

    sum_cubed / n
}

/// Compute kurtosis (4th standardized moment)
///
/// Kurtosis = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
///
/// - Kurtosis > 0: Heavy tails (leptokurtic)
/// - Kurtosis < 0: Light tails (platykurtic)
/// - Kurtosis ≈ 0: Normal-like tails (mesokurtic)
pub fn kurtosis(data: &Array1<f64>) -> f64 {
    if data.len() < 4 {
        return 0.0;
    }

    let m = mean(data);
    let std = std_dev(data, 1);

    if std < 1e-10 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_fourth: f64 = data.iter().map(|&x| ((x - m) / std).powi(4)).sum();

    (sum_fourth / n) - 3.0 // Excess kurtosis
}

/// Compute autocorrelation at lag k
///
/// ρ(k) = Cov(X_t, X_{t-k}) / Var(X_t)
///
/// # Arguments
/// * `data` - Time series
/// * `lag` - Lag value k
///
/// # Returns
/// Autocorrelation coefficient ρ(k) ∈ [-1, 1]
pub fn autocorrelation(data: &Array1<f64>, lag: usize) -> f64 {
    if lag >= data.len() {
        return 0.0;
    }

    let n = data.len();
    let m = mean(data);
    let var = variance(data, 0);

    if var < 1e-10 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in lag..n {
        sum += (data[i] - m) * (data[i - lag] - m);
    }

    sum / ((n - lag) as f64 * var)
}

/// Compute full autocorrelation function up to max_lag
///
/// Returns ACF values [ρ(0), ρ(1), ..., ρ(max_lag)]
pub fn acf(data: &Array1<f64>, max_lag: usize) -> Array1<f64> {
    let lags = (0..=max_lag.min(data.len() - 1))
        .map(|k| autocorrelation(data, k))
        .collect();
    Array1::from_vec(lags)
}

/// Compute correlation between two series
///
/// ρ(X,Y) = Cov(X,Y) / (σ_X · σ_Y)
pub fn correlation(x: &Array1<f64>, y: &Array1<f64>) -> OptimizrResult<f64> {
    if x.len() != y.len() {
        return Err(OptimizrError::InvalidInput(
            "Series must have same length".to_string(),
        ));
    }

    if x.len() < 2 {
        return Err(OptimizrError::InvalidInput(
            "Need at least 2 points".to_string(),
        ));
    }

    let mx = mean(x);
    let my = mean(y);
    let sx = std_dev(x, 1);
    let sy = std_dev(y, 1);

    if sx < 1e-10 || sy < 1e-10 {
        return Ok(0.0);
    }

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mx) * (yi - my))
        .sum::<f64>()
        / (x.len() - 1) as f64;

    Ok(cov / (sx * sy))
}

/// Compute correlation matrix for multiple series
///
/// # Arguments
/// * `data` - Matrix where each column is a time series
///
/// # Returns
/// Correlation matrix C where C_ij = ρ(X_i, X_j)
pub fn correlation_matrix(data: &Array2<f64>) -> OptimizrResult<Array2<f64>> {
    let n_series = data.ncols();
    let mut corr_mat = Array2::eye(n_series);

    for i in 0..n_series {
        for j in (i + 1)..n_series {
            let col_i = data.column(i).to_owned();
            let col_j = data.column(j).to_owned();
            let rho = correlation(&col_i, &col_j)?;
            corr_mat[[i, j]] = rho;
            corr_mat[[j, i]] = rho;
        }
    }

    Ok(corr_mat)
}

// ============================================================================
// Linear Algebra Utilities
// ============================================================================

/// Compute matrix norm (Frobenius norm by default)
///
/// ||A||_F = sqrt(Σ_ij a_ij²)
pub fn matrix_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Compute vector L2 norm
///
/// ||x||_2 = sqrt(Σ_i x_i²)
#[inline]
pub fn vector_norm(vec: &Array1<f64>) -> f64 {
    vec.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Compute vector L1 norm
///
/// ||x||_1 = Σ_i |x_i|
#[inline]
pub fn vector_norm_l1(vec: &Array1<f64>) -> f64 {
    vec.iter().map(|&x| x.abs()).sum()
}

/// Compute vector L∞ norm (maximum absolute value)
///
/// ||x||_∞ = max_i |x_i|
#[inline]
pub fn vector_norm_linf(vec: &Array1<f64>) -> f64 {
    vec.iter().map(|&x| x.abs()).fold(0.0, f64::max)
}

/// Normalize vector to unit length
///
/// Returns x / ||x||_2
pub fn normalize(vec: &Array1<f64>) -> OptimizrResult<Array1<f64>> {
    let norm = vector_norm(vec);
    if norm < 1e-10 {
        return Err(OptimizrError::InvalidInput(
            "Cannot normalize zero vector".to_string(),
        ));
    }
    Ok(vec / norm)
}

/// Compute trace of a square matrix
///
/// Tr(A) = Σ_i A_ii
pub fn trace(matrix: &Array2<f64>) -> OptimizrResult<f64> {
    if matrix.nrows() != matrix.ncols() {
        return Err(OptimizrError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    Ok((0..matrix.nrows()).map(|i| matrix[[i, i]]).sum())
}

/// Compute outer product of two vectors
///
/// A = x ⊗ y where A_ij = x_i · y_j
pub fn outer_product(x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((x.len(), y.len()));
    for i in 0..x.len() {
        for j in 0..y.len() {
            result[[i, j]] = x[i] * y[j];
        }
    }
    result
}

// ============================================================================
// Numerical Integration
// ============================================================================

/// Trapezoidal rule for numerical integration
///
/// ∫f(x)dx ≈ h/2 · [f(x_0) + 2f(x_1) + ... + 2f(x_{n-1}) + f(x_n)]
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of intervals
///
/// # Returns
/// Approximate integral value
pub fn trapz<F>(f: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 0 {
        return 0.0;
    }

    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(x);
    }

    sum * h
}

/// Simpson's rule for numerical integration (more accurate than trapezoidal)
///
/// ∫f(x)dx ≈ h/3 · [f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + ... + f(x_n)]
///
/// # Arguments
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of intervals (must be even)
pub fn simpson<F>(f: &F, a: f64, b: f64, n: usize) -> OptimizrResult<f64>
where
    F: Fn(f64) -> f64,
{
    if n % 2 != 0 {
        return Err(OptimizrError::InvalidInput(
            "Simpson's rule requires even number of intervals".to_string(),
        ));
    }

    if n == 0 {
        return Ok(0.0);
    }

    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;
        let coef = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coef * f(x);
    }

    Ok(sum * h / 3.0)
}

// ============================================================================
// Interpolation
// ============================================================================

/// Linear interpolation between two points
///
/// y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
#[inline]
pub fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    if (x1 - x0).abs() < 1e-10 {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Linear interpolation on a grid
///
/// # Arguments
/// * `x_grid` - Sorted grid points
/// * `y_values` - Function values at grid points
/// * `x` - Point to interpolate
///
/// # Returns
/// Interpolated value
pub fn interp1d(x_grid: &Array1<f64>, y_values: &Array1<f64>, x: f64) -> OptimizrResult<f64> {
    if x_grid.len() != y_values.len() {
        return Err(OptimizrError::InvalidInput(
            "Grid and values must have same length".to_string(),
        ));
    }

    if x_grid.len() < 2 {
        return Err(OptimizrError::InvalidInput(
            "Need at least 2 points for interpolation".to_string(),
        ));
    }

    // Find bracketing indices
    if x <= x_grid[0] {
        return Ok(y_values[0]);
    }
    if x >= x_grid[x_grid.len() - 1] {
        return Ok(y_values[y_values.len() - 1]);
    }

    for i in 0..(x_grid.len() - 1) {
        if x >= x_grid[i] && x <= x_grid[i + 1] {
            return Ok(lerp(
                x_grid[i],
                y_values[i],
                x_grid[i + 1],
                y_values[i + 1],
                x,
            ));
        }
    }

    Ok(y_values[y_values.len() - 1])
}

// ============================================================================
// Special Functions
// ============================================================================

/// Logistic sigmoid function
///
/// σ(x) = 1 / (1 + e^(-x))
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus function (smooth approximation to ReLU)
///
/// softplus(x) = ln(1 + e^x)
#[inline]
pub fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

/// ReLU (Rectified Linear Unit)
///
/// relu(x) = max(0, x)
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Soft thresholding operator (for LASSO/L1 regularization)
///
/// S_λ(x) = sign(x) · max(|x| - λ, 0)
#[inline]
pub fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Apply soft thresholding to vector
pub fn soft_threshold_vec(x: &Array1<f64>, lambda: f64) -> Array1<f64> {
    x.mapv(|xi| soft_threshold(xi, lambda))
}

// ============================================================================
// Optimization Helpers
// ============================================================================

/// Check if a point satisfies box constraints
///
/// Returns true if lower[i] ≤ x[i] ≤ upper[i] for all i
pub fn check_bounds(x: &Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) -> bool {
    if x.len() != lower.len() || x.len() != upper.len() {
        return false;
    }

    x.iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .all(|((&xi, &li), &ui)| xi >= li && xi <= ui)
}

/// Project point onto box constraints
///
/// Returns x' where x'[i] = clamp(x[i], lower[i], upper[i])
pub fn project_bounds(x: &Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) -> Array1<f64> {
    x.iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|((&xi, &li), &ui)| xi.max(li).min(ui))
        .collect()
}

/// Compute numerical condition number estimate
///
/// κ(A) ≈ ||A|| · ||A^(-1)||
///
/// High condition number indicates ill-conditioned matrix
pub fn condition_number_estimate(matrix: &Array2<f64>) -> f64 {
    // Simple estimate using Frobenius norm
    // For more accurate estimate, use SVD
    let norm = matrix_norm(matrix);

    // This is a crude estimate - for production use SVD-based method
    if norm < 1e-10 {
        return f64::INFINITY;
    }

    // Placeholder - proper implementation needs matrix inverse
    norm * 1000.0 // Conservative estimate
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gradient() {
        // f(x,y) = x² + 2y²
        let f = |x: &[f64]| x[0].powi(2) + 2.0 * x[1].powi(2);
        let x = array![1.0, 2.0];
        let grad = gradient(&f, &x, 1e-5).unwrap();

        assert!((grad[0] - 2.0).abs() < 1e-3); // ∂f/∂x = 2x = 2
        assert!((grad[1] - 8.0).abs() < 1e-3); // ∂f/∂y = 4y = 8
    }

    #[test]
    fn test_statistics() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((mean(&data) - 3.0).abs() < 1e-10);
        assert!((std_dev(&data, 1) - 1.5811).abs() < 1e-3);
    }

    #[test]
    fn test_autocorrelation() {
        let data = array![1.0, 2.0, 1.5, 2.5, 2.0, 3.0];
        let acf_0 = autocorrelation(&data, 0);

        assert!((acf_0 - 1.0).abs() < 1e-10); // ACF at lag 0 is always 1
    }

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(3.0, 1.0), 2.0);
        assert_eq!(soft_threshold(-3.0, 1.0), -2.0);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
    }

    #[test]
    fn test_integration() {
        // ∫x² dx from 0 to 1 = 1/3
        let f = |x: f64| x * x;
        let result = trapz(&f, 0.0, 1.0, 1000);

        assert!((result - 1.0 / 3.0).abs() < 1e-3);
    }
}
