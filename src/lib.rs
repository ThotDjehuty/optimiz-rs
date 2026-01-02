//! OptimizR - High-Performance Optimization Algorithms
//! ===================================================
//!
//! This library provides fast, reliable implementations of advanced optimization
//! and statistical inference algorithms, with Python bindings via PyO3.
//!
//! # Architecture
//!
//! The library is designed with modularity, functional programming patterns,
//! and trait-based abstractions:
//!
//! - `core`: Core traits (Optimizer, Sampler, InformationMeasure) and error types
//! - `functional`: Functional programming utilities (composition, memoization, pipes)
//! - Refactored modules with trait-based design and parallel support
//! - Original modules maintained for backward compatibility
//!
//! # Modules
//!
//! - `hmm`: Hidden Markov Model training and inference
//! - `mcmc`: Markov Chain Monte Carlo sampling
//! - `differential_evolution`: Global optimization algorithm
//! - `grid_search`: Exhaustive parameter space search
//! - `information_theory`: Mutual information and entropy calculations
//! - `sparse_optimization`: Sparse PCA, Box-Tao, Elastic Net
//! - `risk_metrics`: Portfolio risk analysis and Hurst exponent

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyModule;

// Core modules with trait-based architecture
pub mod core;
pub mod functional;
pub mod maths_toolkit; // Mathematical utilities
pub mod timeseries_utils; // Time-series integration helpers

// Modular structure (trait-based, generic)
pub mod de;
pub mod hmm;
pub mod mcmc;
pub mod optimal_control;
pub mod risk_metrics;
pub mod sparse_optimization;

// Python bindings for legacy compatibility
#[cfg(feature = "python-bindings")]
mod differential_evolution;
#[cfg(feature = "python-bindings")]
mod grid_search;
#[cfg(feature = "python-bindings")]
mod information_theory;

/// OptimizR Python module
#[cfg(feature = "python-bindings")]
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ===== New Modular API (Recommended) =====

    // HMM functions (modular structure)
    m.add_class::<hmm::HMMParams>()?;
    m.add_function(wrap_pyfunction!(hmm::fit_hmm, m)?)?;
    m.add_function(wrap_pyfunction!(hmm::viterbi_decode, m)?)?;

    // MCMC functions (modular structure)
    m.add_function(wrap_pyfunction!(mcmc::mcmc_sample, m)?)?;
    m.add_function(wrap_pyfunction!(mcmc::adaptive_mcmc_sample, m)?)?;

    // DE functions (modular structure - uses de_refactored for now)
    m.add_class::<de::DEResult>()?;
    m.add_function(wrap_pyfunction!(de::differential_evolution, m)?)?;

    // ===== Additional Algorithms =====

    // Optimization functions
    m.add_function(wrap_pyfunction!(
        differential_evolution::differential_evolution,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(grid_search::grid_search, m)?)?;

    // Information theory functions
    m.add_function(wrap_pyfunction!(information_theory::mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(information_theory::shannon_entropy, m)?)?;

    // ===== New Optimization Algorithms =====

    // Sparse optimization functions
    m.add_function(wrap_pyfunction!(sparse_optimization::sparse_pca_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        sparse_optimization::box_tao_decomposition_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sparse_optimization::elastic_net_py, m)?)?;

    // Risk metrics functions
    m.add_function(wrap_pyfunction!(risk_metrics::hurst_exponent_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk_metrics::compute_risk_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk_metrics::estimate_half_life_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk_metrics::bootstrap_returns_py, m)?)?;

    // Time-series utility functions
    timeseries_utils::python_bindings::register_python_functions(m)?;

    Ok(())
}
