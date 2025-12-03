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

use pyo3::prelude::*;
use pyo3::types::PyModule;

// Core modules with trait-based architecture
pub mod core;
pub mod functional;

// Refactored modules with advanced patterns
pub mod hmm_refactored;
pub mod mcmc_refactored;
pub mod de_refactored;

// Original modules for backward compatibility
mod hmm;
mod mcmc;
mod differential_evolution;
mod grid_search;
mod information_theory;

/// OptimizR Python module
#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ===== Original API (Backward Compatible) =====
    
    // HMM functions
    m.add_class::<hmm::HMMParams>()?;
    m.add_function(wrap_pyfunction!(hmm::fit_hmm, m)?)?;
    m.add_function(wrap_pyfunction!(hmm::viterbi_decode, m)?)?;
    
    // MCMC functions
    m.add_function(wrap_pyfunction!(mcmc::mcmc_sample, m)?)?;
    
    // Optimization functions
    m.add_function(wrap_pyfunction!(differential_evolution::differential_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search::grid_search, m)?)?;
    
    // Information theory functions
    m.add_function(wrap_pyfunction!(information_theory::mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(information_theory::shannon_entropy, m)?)?;
    
    // ===== New Refactored API (Advanced Features) =====
    
    // Refactored HMM with trait-based design
    m.add_class::<hmm_refactored::HMMParams>()?;
    m.add_function(wrap_pyfunction!(hmm_refactored::fit_hmm, m)?)?;
    m.add_function(wrap_pyfunction!(hmm_refactored::viterbi_decode, m)?)?;
    
    // Refactored MCMC with strategy pattern
    m.add_function(wrap_pyfunction!(mcmc_refactored::mcmc_sample, m)?)?;
    m.add_function(wrap_pyfunction!(mcmc_refactored::adaptive_mcmc_sample, m)?)?;
    
    // Refactored DE with parallel support and multiple strategies
    m.add_class::<de_refactored::DEResult>()?;
    m.add_function(wrap_pyfunction!(de_refactored::differential_evolution, m)?)?;
    
    Ok(())
}
