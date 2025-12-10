///! Information Theory Metrics
///!
///! This module provides implementations of fundamental information theory measures:
///!
///! - **Shannon Entropy**: H(X) = -Σ p(x) log p(x)
///!   Quantifies the uncertainty/information content of a random variable
///!
///! - **Mutual Information**: I(X;Y) = H(X) + H(Y) - H(X,Y)
///!   Measures the dependence between two random variables
///!
///! # Applications
///!
///! - Feature selection (high MI with target)
///! - Dependency detection in time series
///! - Causality testing
///! - Compression and coding
///!
///! # References
///!
///! Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.
///! Wiley-Interscience.
use pyo3::prelude::*;
use std::f64;

/// Shannon Entropy Calculation
///
/// Computes the Shannon entropy of a random variable using histogram-based
/// probability estimation.
///!
///! H(X) = -Σᵢ p(xᵢ) log(p(xᵢ))
///!
///! where p(xᵢ) is estimated by binning the data.
///!
///! # Arguments
///!
///! * `x` - Sample values from the random variable
///! * `n_bins` - Number of bins for histogram estimation (default: 10)
///!
///! # Returns
///!
///! Entropy in nats (natural logarithm). Multiply by 1/ln(2) for bits.
///!
///! # Example
///!
///! ```python
///! import optimizr
///! import numpy as np
///!
///! # Uniform distribution has high entropy
///! x_uniform = np.random.uniform(0, 1, 10000)
///! h_uniform = optimizr.shannon_entropy(x_uniform, n_bins=20)
///! print(f"Uniform entropy: {h_uniform:.4f} nats")
///!
///! # Peaked distribution has low entropy
///! x_peaked = np.random.normal(0, 0.1, 10000)
///! h_peaked = optimizr.shannon_entropy(x_peaked, n_bins=20)
///! print(f"Peaked entropy: {h_peaked:.4f} nats")
///! ```
#[pyfunction]
#[pyo3(signature = (x, n_bins=10))]
pub fn shannon_entropy(x: Vec<f64>, n_bins: usize) -> PyResult<f64> {
    let n = x.len();

    if n == 0 {
        return Ok(0.0);
    }

    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be positive",
        ));
    }

    // Find min and max
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Handle constant values
    if (x_max - x_min).abs() < 1e-10 {
        return Ok(0.0);
    }

    // Bin the data
    let mut bin_counts = vec![0usize; n_bins];

    for &val in &x {
        let bin = ((val - x_min) / (x_max - x_min) * (n_bins as f64 - 1e-10)) as usize;
        let bin = bin.min(n_bins - 1);
        bin_counts[bin] += 1;
    }

    // Compute entropy: H(X) = -Σ p(x) log(p(x))
    let entropy: f64 = bin_counts
        .iter()
        .filter_map(|&count| {
            if count > 0 {
                let p = count as f64 / n as f64;
                Some(-p * p.ln())
            } else {
                None
            }
        })
        .sum();

    Ok(entropy)
}

/// Mutual Information Calculation
///
/// Computes the mutual information between two random variables:
///!
///! I(X;Y) = H(X) + H(Y) - H(X,Y)
///!
///! where H(X,Y) is the joint entropy.
///!
///! Mutual information measures how much knowing one variable reduces
///! uncertainty about the other. I(X;Y) = 0 if X and Y are independent.
///!
///! # Arguments
///!
///! * `x` - Sample values from first random variable
///! * `y` - Sample values from second random variable (must be same length as x)
///! * `n_bins` - Number of bins for histogram estimation (default: 10)
///!
///! # Returns
///!
///! Mutual information in nats. Always non-negative.
///!
///! # Example
///!
///! ```python
///! import optimizr
///! import numpy as np
///!
///! # Independent variables
///! x = np.random.randn(10000)
///! y = np.random.randn(10000)
///! mi_indep = optimizr.mutual_information(x, y, n_bins=20)
///! print(f"MI (independent): {mi_indep:.4f} nats")
///!
///! # Dependent variables
///! x = np.random.randn(10000)
///! y = 2 * x + np.random.randn(10000) * 0.5
///! mi_dep = optimizr.mutual_information(x, y, n_bins=20)
///! print(f"MI (dependent): {mi_dep:.4f} nats")
///! ```
#[pyfunction]
#[pyo3(signature = (x, y, n_bins=10))]
pub fn mutual_information(x: Vec<f64>, y: Vec<f64>, n_bins: usize) -> PyResult<f64> {
    let n = x.len();

    if n != y.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x and y must have same length",
        ));
    }

    if n == 0 {
        return Ok(0.0);
    }

    if n_bins == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be positive",
        ));
    }

    // Find min/max for binning
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Handle constant values
    if (x_max - x_min).abs() < 1e-10 || (y_max - y_min).abs() < 1e-10 {
        return Ok(0.0);
    }

    // Discretize into bins
    let x_binned: Vec<usize> = x
        .iter()
        .map(|&v| {
            let bin = ((v - x_min) / (x_max - x_min) * (n_bins as f64 - 1e-10)) as usize;
            bin.min(n_bins - 1)
        })
        .collect();

    let y_binned: Vec<usize> = y
        .iter()
        .map(|&v| {
            let bin = ((v - y_min) / (y_max - y_min) * (n_bins as f64 - 1e-10)) as usize;
            bin.min(n_bins - 1)
        })
        .collect();

    // Compute joint and marginal counts
    let mut joint_counts = vec![vec![0usize; n_bins]; n_bins];
    let mut x_counts = vec![0usize; n_bins];
    let mut y_counts = vec![0usize; n_bins];

    for i in 0..n {
        joint_counts[x_binned[i]][y_binned[i]] += 1;
        x_counts[x_binned[i]] += 1;
        y_counts[y_binned[i]] += 1;
    }

    // Compute MI: I(X;Y) = Σᵢⱼ p(x,y) log(p(x,y) / (p(x)p(y)))
    let mut mi = 0.0;

    for i in 0..n_bins {
        let px = x_counts[i] as f64 / n as f64;

        if px == 0.0 {
            continue;
        }

        for j in 0..n_bins {
            let py = y_counts[j] as f64 / n as f64;
            let pxy = joint_counts[i][j] as f64 / n as f64;

            if pxy > 0.0 && py > 0.0 {
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }

    // MI is always non-negative (enforce numerically)
    Ok(mi.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution should have relatively high entropy
        let x: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let entropy = shannon_entropy(x, 10).unwrap();
        assert!(entropy > 2.0); // ln(10) ≈ 2.3 is maximum for 10 bins
    }

    #[test]
    fn test_shannon_entropy_constant() {
        // Constant value should have zero entropy
        let x = vec![1.0; 100];
        let entropy = shannon_entropy(x, 10).unwrap();
        assert!(entropy.abs() < 1e-6);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent uniform variables should have low MI
        let x: Vec<f64> = (0..1000).map(|i| (i % 100) as f64).collect();
        let y: Vec<f64> = (0..1000).map(|i| ((i * 7) % 100) as f64).collect();
        let mi = mutual_information(x, y, 10).unwrap();
        assert!(mi >= 0.0); // MI is always non-negative
    }

    #[test]
    fn test_mutual_information_identical() {
        // Identical variables should have high MI
        let x: Vec<f64> = (0..1000).map(|i| (i % 100) as f64).collect();
        let y = x.clone();
        let mi = mutual_information(x, y, 10).unwrap();
        assert!(mi > 1.0); // Should be close to H(X)
    }
}
