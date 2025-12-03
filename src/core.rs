//! Core traits and types for optimization algorithms
//!
//! This module defines the foundational traits and types used across all
//! optimization and inference algorithms in OptimizR.

use thiserror::Error;

/// Custom error type for OptimizR operations
#[derive(Error, Debug, Clone)]
pub enum OptimizrError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Empty data provided")]
    EmptyData,
    
    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Result type for OptimizR operations
pub type Result<T> = std::result::Result<T, OptimizrError>;

/// Trait for optimization algorithms
pub trait Optimizer {
    type Config;
    type Output;
    
    /// Optimize to find best solution
    fn optimize(&mut self) -> Result<Self::Output>;
    
    /// Get current best solution
    fn best(&self) -> Result<Vec<f64>>;
}

/// Trait for sampling algorithms (MCMC, etc.)
pub trait Sampler {
    type Config;
    type Output;
    
    /// Draw samples from the target distribution
    fn sample(&mut self) -> Result<Self::Output>;
    
    /// Get diagnostics about sampling performance
    fn diagnostics(&self, samples: &Self::Output) -> Result<SamplerDiagnostics>;
}

/// Diagnostics for sampling algorithms
#[derive(Debug, Clone)]
pub struct SamplerDiagnostics {
    pub n_samples: usize,
    pub means: Vec<f64>,
    pub std_devs: Vec<f64>,
    pub autocorrelations: Vec<f64>,
}

/// Trait for configuration builders
pub trait ConfigBuilder {
    type Config;
    
    fn build(self) -> Result<Self::Config>;
}

/// Trait for information measures (entropy, MI, etc.)
pub trait InformationMeasure {
    /// Compute the measure for given data
    fn compute(&self, data: &[f64]) -> Result<f64>;
    
    /// Compute pairwise measure (for MI)
    fn compute_pairwise(&self, _x: &[f64], _y: &[f64]) -> Result<f64> {
        Err(OptimizrError::ComputationError(
            "Pairwise computation not supported".to_string(),
        ))
    }
}

/// Bounds for optimization
#[derive(Debug, Clone)]
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl Bounds {
    pub fn new(bounds: Vec<(f64, f64)>) -> Result<Self> {
        if bounds.is_empty() {
            return Err(OptimizrError::InvalidParameter(
                "Bounds cannot be empty".to_string(),
            ));
        }
        
        for (lower, upper) in &bounds {
            if lower >= upper {
                return Err(OptimizrError::InvalidParameter(format!(
                    "Invalid bounds: lower ({}) >= upper ({})",
                    lower, upper
                )));
            }
        }
        
        let (lower, upper): (Vec<_>, Vec<_>) = bounds.into_iter().unzip();
        Ok(Self { lower, upper })
    }
    
    pub fn dim(&self) -> usize {
        self.lower.len()
    }
    
    pub fn clip(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &val)| val.max(self.lower[i]).min(self.upper[i]))
            .collect()
    }
    
    pub fn is_valid(&self, x: &[f64]) -> bool {
        x.len() == self.dim()
            && x.iter()
                .enumerate()
                .all(|(i, &val)| val >= self.lower[i] && val <= self.upper[i])
    }
    
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Vec<f64> {
        (0..self.dim())
            .map(|i| rng.gen_range(self.lower[i]..self.upper[i]))
            .collect()
    }
}

/// Trait for parallel execution strategies
pub trait ParallelExecutor {
    fn execute_parallel<F, T>(&self, tasks: Vec<F>) -> Vec<T>
    where
        F: Fn() -> T + Send,
        T: Send;
}

/// Standard rayon-based parallel executor
#[cfg(feature = "parallel")]
pub struct RayonExecutor;

#[cfg(feature = "parallel")]
impl ParallelExecutor for RayonExecutor {
    fn execute_parallel<F, T>(&self, tasks: Vec<F>) -> Vec<T>
    where
        F: Fn() -> T + Send,
        T: Send,
    {
        use rayon::prelude::*;
        tasks.into_par_iter().map(|f| f()).collect()
    }
}

/// Sequential executor (fallback)
pub struct SequentialExecutor;

impl ParallelExecutor for SequentialExecutor {
    fn execute_parallel<F, T>(&self, tasks: Vec<F>) -> Vec<T>
    where
        F: Fn() -> T + Send,
        T: Send,
    {
        tasks.into_iter().map(|f| f()).collect()
    }
}
