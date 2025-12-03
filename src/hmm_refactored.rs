//! Refactored Hidden Markov Model with trait-based design
//!
//! This module provides a more modular, functional, and trait-based implementation
//! of HMMs with support for different emission models and parallel computation.

use crate::core::{OptimizrError, Result};
use pyo3::prelude::*;
use std::f64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Trait for emission probability models
pub trait EmissionModel: Send + Sync + Clone {
    /// Compute emission probability for observation given state
    fn probability(&self, observation: f64, state: usize) -> f64;
    
    /// Update parameters from weighted observations
    fn update(&mut self, observations: &[f64], weights: &[f64], state: usize) -> Result<()>;
    
    /// Initialize parameters from observations
    fn initialize(&mut self, observations: &[f64], n_states: usize, state: usize) -> Result<()>;
    
    /// Get number of states
    fn n_states(&self) -> usize;
}

/// Gaussian emission model
#[derive(Clone, Debug)]
pub struct GaussianEmission {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl GaussianEmission {
    pub fn new(n_states: usize) -> Self {
        Self {
            means: vec![0.0; n_states],
            stds: vec![1.0; n_states],
        }
    }
}

impl EmissionModel for GaussianEmission {
    fn probability(&self, observation: f64, state: usize) -> f64 {
        let mean = self.means[state];
        let std = self.stds[state];
        let z = (observation - mean) / std;
        let coef = 1.0 / (std * (2.0 * f64::consts::PI).sqrt());
        (coef * (-0.5 * z * z).exp()).max(1e-10)
    }
    
    fn update(&mut self, observations: &[f64], weights: &[f64], state: usize) -> Result<()> {
        let sum_weights: f64 = weights.iter().sum();
        
        if sum_weights < 1e-10 {
            return Ok(());
        }
        
        // Weighted mean
        let mean = observations
            .iter()
            .zip(weights.iter())
            .map(|(obs, w)| obs * w)
            .sum::<f64>()
            / sum_weights;
        
        // Weighted variance
        let var = observations
            .iter()
            .zip(weights.iter())
            .map(|(obs, w)| w * (obs - mean).powi(2))
            .sum::<f64>()
            / sum_weights;
        
        self.means[state] = mean;
        self.stds[state] = var.sqrt().max(1e-6);
        
        Ok(())
    }
    
    fn initialize(&mut self, observations: &[f64], n_states: usize, state: usize) -> Result<()> {
        let mut sorted = observations.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = observations.len();
        let start_idx = (state * n) / n_states;
        let end_idx = ((state + 1) * n) / n_states;
        let segment = &sorted[start_idx..end_idx];
        
        if !segment.is_empty() {
            self.means[state] = segment.iter().sum::<f64>() / segment.len() as f64;
            let var: f64 = segment
                .iter()
                .map(|x| (x - self.means[state]).powi(2))
                .sum::<f64>()
                / segment.len() as f64;
            self.stds[state] = var.sqrt().max(1e-6);
        }
        
        Ok(())
    }
    
    fn n_states(&self) -> usize {
        self.means.len()
    }
}

/// HMM Configuration Builder
#[derive(Clone)]
pub struct HMMConfig<E: EmissionModel> {
    pub n_states: usize,
    pub n_iterations: usize,
    pub tolerance: f64,
    pub emission_model: E,
    pub use_parallel: bool,
}

impl<E: EmissionModel> HMMConfig<E> {
    pub fn builder(n_states: usize) -> HMMConfigBuilder<E> {
        HMMConfigBuilder::new(n_states)
    }
}

/// Builder pattern for HMM configuration
pub struct HMMConfigBuilder<E: EmissionModel> {
    n_states: usize,
    n_iterations: usize,
    tolerance: f64,
    emission_model: Option<E>,
    use_parallel: bool,
}

impl<E: EmissionModel> HMMConfigBuilder<E> {
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            n_iterations: 100,
            tolerance: 1e-6,
            emission_model: None,
            use_parallel: cfg!(feature = "parallel"),
        }
    }
    
    pub fn iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }
    
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    
    pub fn emission_model(mut self, model: E) -> Self {
        self.emission_model = Some(model);
        self
    }
    
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled && cfg!(feature = "parallel");
        self
    }
    
    pub fn build(self) -> Result<HMMConfig<E>>
    where
        E: EmissionModel + Default,
    {
        if self.n_states < 2 {
            return Err(OptimizrError::InvalidParameter(
                "n_states must be at least 2".to_string(),
            ));
        }
        
        Ok(HMMConfig {
            n_states: self.n_states,
            n_iterations: self.n_iterations,
            tolerance: self.tolerance,
            emission_model: self.emission_model.unwrap_or_default(),
            use_parallel: self.use_parallel,
        })
    }
}

impl Default for GaussianEmission {
    fn default() -> Self {
        Self::new(2)
    }
}

/// Refactored HMM with generic emission model
pub struct HMM<E: EmissionModel> {
    pub config: HMMConfig<E>,
    pub transition_matrix: Vec<Vec<f64>>,
    pub initial_probs: Vec<f64>,
}

impl<E: EmissionModel> HMM<E> {
    pub fn new(config: HMMConfig<E>) -> Self {
        let n_states = config.n_states;
        let uniform = 1.0 / n_states as f64;
        
        Self {
            config,
            transition_matrix: vec![vec![uniform; n_states]; n_states],
            initial_probs: vec![uniform; n_states],
        }
    }
    
    /// Fit HMM using functional pipeline
    pub fn fit(&mut self, observations: &[f64]) -> Result<()> {
        if observations.is_empty() {
            return Err(OptimizrError::EmptyData);
        }
        
        // Initialize emission parameters
        for s in 0..self.config.n_states {
            self.config
                .emission_model
                .initialize(observations, self.config.n_states, s)?;
        }
        
        // EM iterations with functional approach
        let mut prev_ll = f64::NEG_INFINITY;
        
        for _iter in 0..self.config.n_iterations {
            // E-step: Compute posteriors
            let alpha = self.forward(observations)?;
            let beta = self.backward(observations)?;
            let gamma = Self::compute_gamma(&alpha, &beta);
            let xi = self.compute_xi(observations, &alpha, &beta)?;
            
            // M-step: Update parameters
            self.update_parameters(observations, &gamma, &xi)?;
            
            // Check convergence
            let log_likelihood = Self::compute_log_likelihood(&alpha);
            
            if (log_likelihood - prev_ll).abs() < self.config.tolerance {
                break; // Converged
            }
            
            prev_ll = log_likelihood;
        }
        
        Ok(())
    }
    
    /// Forward algorithm with parallel option
    fn forward(&self, observations: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;
        let mut alpha = vec![vec![0.0; n_states]; n_obs];
        
        // Initialize
        for s in 0..n_states {
            alpha[0][s] = self.initial_probs[s]
                * self.config.emission_model.probability(observations[0], s);
        }
        Self::normalize_row(&mut alpha[0]);
        
        // Recursion (sequential for dependencies)
        for t in 1..n_obs {
            for s in 0..n_states {
                let sum: f64 = (0..n_states)
                    .map(|prev_s| alpha[t - 1][prev_s] * self.transition_matrix[prev_s][s])
                    .sum();
                alpha[t][s] = sum * self.config.emission_model.probability(observations[t], s);
            }
            Self::normalize_row(&mut alpha[t]);
        }
        
        Ok(alpha)
    }
    
    /// Backward algorithm
    fn backward(&self, observations: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;
        let mut beta = vec![vec![0.0; n_states]; n_obs];
        
        // Initialize
        beta[n_obs - 1].fill(1.0);
        
        // Recursion
        for t in (0..n_obs - 1).rev() {
            for s in 0..n_states {
                let sum: f64 = (0..n_states)
                    .map(|next_s| {
                        self.transition_matrix[s][next_s]
                            * self.config.emission_model.probability(observations[t + 1], next_s)
                            * beta[t + 1][next_s]
                    })
                    .sum();
                beta[t][s] = sum;
            }
            Self::normalize_row(&mut beta[t]);
        }
        
        Ok(beta)
    }
    
    /// Compute state occupation probabilities (pure function)
    fn compute_gamma(alpha: &[Vec<f64>], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
        alpha
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| {
                let sum: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
                a.iter()
                    .zip(b.iter())
                    .map(|(ai, bi)| {
                        if sum > 1e-10 {
                            ai * bi / sum
                        } else {
                            1.0 / a.len() as f64
                        }
                    })
                    .collect()
            })
            .collect()
    }
    
    /// Compute transition probabilities
    fn compute_xi(
        &self,
        observations: &[f64],
        alpha: &[Vec<f64>],
        beta: &[Vec<f64>],
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;
        
        let xi: Vec<Vec<Vec<f64>>> = (0..n_obs - 1)
            .map(|t| {
                let mut xi_t = vec![vec![0.0; n_states]; n_states];
                let mut sum = 0.0;
                
                for i in 0..n_states {
                    for j in 0..n_states {
                        xi_t[i][j] = alpha[t][i]
                            * self.transition_matrix[i][j]
                            * self.config.emission_model.probability(observations[t + 1], j)
                            * beta[t + 1][j];
                        sum += xi_t[i][j];
                    }
                }
                
                // Normalize
                if sum > 1e-10 {
                    for row in &mut xi_t {
                        for val in row {
                            *val /= sum;
                        }
                    }
                }
                
                xi_t
            })
            .collect();
        
        Ok(xi)
    }
    
    /// Update parameters using functional patterns
    fn update_parameters(
        &mut self,
        observations: &[f64],
        gamma: &[Vec<f64>],
        xi: &[Vec<Vec<f64>>],
    ) -> Result<()> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;
        
        // Update transitions
        for i in 0..n_states {
            let denom: f64 = gamma[..n_obs - 1].iter().map(|g| g[i]).sum();
            
            for j in 0..n_states {
                let numer: f64 = xi.iter().map(|x| x[i][j]).sum();
                self.transition_matrix[i][j] = if denom > 1e-10 {
                    numer / denom
                } else {
                    1.0 / n_states as f64
                };
            }
        }
        
        // Update emissions
        for s in 0..n_states {
            let weights: Vec<f64> = gamma.iter().map(|g| g[s]).collect();
            self.config
                .emission_model
                .update(observations, &weights, s)?;
        }
        
        Ok(())
    }
    
    /// Viterbi decoding with functional style
    pub fn viterbi(&self, observations: &[f64]) -> Result<Vec<usize>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;
        
        if n_obs == 0 {
            return Ok(Vec::new());
        }
        
        let mut delta = vec![vec![f64::NEG_INFINITY; n_states]; n_obs];
        let mut psi = vec![vec![0usize; n_states]; n_obs];
        
        // Initialize
        for s in 0..n_states {
            delta[0][s] = self.initial_probs[s].ln()
                + self.config.emission_model.probability(observations[0], s).ln();
        }
        
        // Recursion
        for t in 1..n_obs {
            for s in 0..n_states {
                let (max_state, max_val) = (0..n_states)
                    .map(|prev_s| {
                        (
                            prev_s,
                            delta[t - 1][prev_s] + self.transition_matrix[prev_s][s].ln(),
                        )
                    })
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                
                psi[t][s] = max_state;
                delta[t][s] = max_val
                    + self.config.emission_model.probability(observations[t], s).ln();
            }
        }
        
        // Backtrack
        let mut path = vec![0usize; n_obs];
        path[n_obs - 1] = (0..n_states)
            .max_by(|&a, &b| delta[n_obs - 1][a].partial_cmp(&delta[n_obs - 1][b]).unwrap())
            .unwrap();
        
        for t in (0..n_obs - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }
        
        Ok(path)
    }
    
    // Helper functions
    fn normalize_row(row: &mut [f64]) {
        let sum: f64 = row.iter().sum();
        if sum > 1e-10 {
            row.iter_mut().for_each(|v| *v /= sum);
        } else {
            let uniform = 1.0 / row.len() as f64;
            row.fill(uniform);
        }
    }
    
    fn compute_log_likelihood(alpha: &[Vec<f64>]) -> f64 {
        alpha.last().unwrap().iter().sum::<f64>().max(1e-10).ln()
    }
}

// Python bindings remain similar but use the new modular structure
#[pyclass]
#[derive(Clone, Debug)]
pub struct HMMParams {
    #[pyo3(get, set)]
    pub n_states: usize,
    #[pyo3(get, set)]
    pub transition_matrix: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub emission_means: Vec<f64>,
    #[pyo3(get, set)]
    pub emission_stds: Vec<f64>,
    #[pyo3(get, set)]
    pub initial_probs: Vec<f64>,
}

#[pymethods]
impl HMMParams {
    #[new]
    pub fn new(n_states: usize) -> Self {
        let uniform_prob = 1.0 / n_states as f64;
        HMMParams {
            n_states,
            transition_matrix: vec![vec![uniform_prob; n_states]; n_states],
            emission_means: vec![0.0; n_states],
            emission_stds: vec![1.0; n_states],
            initial_probs: vec![uniform_prob; n_states],
        }
    }
    
    fn __repr__(&self) -> String {
        format!(
            "HMMParams(n_states={}, transition_shape={}x{})",
            self.n_states, self.n_states, self.n_states
        )
    }
}

#[pyfunction]
#[pyo3(signature = (observations, n_states, n_iterations=100, tolerance=1e-6))]
pub fn fit_hmm(
    observations: Vec<f64>,
    n_states: usize,
    n_iterations: usize,
    tolerance: f64,
) -> PyResult<HMMParams> {
    let emission = GaussianEmission::new(n_states);
    
    let config = HMMConfig {
        n_states,
        n_iterations,
        tolerance,
        emission_model: emission.clone(),
        use_parallel: false,
    };
    
    let mut hmm = HMM::new(config);
    hmm.fit(&observations)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    Ok(HMMParams {
        n_states,
        transition_matrix: hmm.transition_matrix,
        emission_means: hmm.config.emission_model.means,
        emission_stds: hmm.config.emission_model.stds,
        initial_probs: hmm.initial_probs,
    })
}

#[pyfunction]
pub fn viterbi_decode(observations: Vec<f64>, params: HMMParams) -> PyResult<Vec<usize>> {
    let emission = GaussianEmission {
        means: params.emission_means,
        stds: params.emission_stds,
    };
    
    let config = HMMConfig {
        n_states: params.n_states,
        n_iterations: 0,
        tolerance: 0.0,
        emission_model: emission,
        use_parallel: false,
    };
    
    let mut hmm = HMM::new(config);
    hmm.transition_matrix = params.transition_matrix;
    hmm.initial_probs = params.initial_probs;
    
    hmm.viterbi(&observations)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_builder() {
        let config = HMMConfig::<GaussianEmission>::builder(3)
            .iterations(50)
            .tolerance(1e-5)
            .build()
            .unwrap();
        
        assert_eq!(config.n_states, 3);
        assert_eq!(config.n_iterations, 50);
    }

    #[test]
    fn test_hmm_fit() {
        let observations: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = HMMConfig::<GaussianEmission>::builder(2).build().unwrap();
        
        let mut hmm = HMM::new(config);
        assert!(hmm.fit(&observations).is_ok());
    }
}
