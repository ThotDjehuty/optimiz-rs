//! Refactored MCMC with Strategy Pattern
//!
//! Supports multiple proposal strategies and parallel chains.

use crate::core::{OptimizrError, Result, Sampler, SamplerDiagnostics};
use pyo3::prelude::*;
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::Normal;

/// Trait for proposal strategies
pub trait ProposalStrategy: Send + Sync + Clone {
    /// Generate proposed next state
    fn propose(&self, current: &[f64], rng: &mut impl Rng) -> Vec<f64>;
    
    /// Adapt proposal based on acceptance rate (optional)
    fn adapt(&mut self, _acceptance_rate: f64) {}
    
    /// Name of the strategy
    fn name(&self) -> &'static str;
}

/// Gaussian random walk proposal
#[derive(Clone, Debug)]
pub struct GaussianProposal {
    pub step_size: f64,
}

impl GaussianProposal {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl ProposalStrategy for GaussianProposal {
    fn propose(&self, current: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let normal = Normal::new(0.0, self.step_size).unwrap();
        current
            .iter()
            .map(|&x| x + normal.sample(rng))
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "GaussianRandomWalk"
    }
}

/// Adaptive proposal that adjusts step size
#[derive(Clone, Debug)]
pub struct AdaptiveProposal {
    pub step_size: f64,
    pub target_acceptance: f64,
    pub adaptation_rate: f64,
}

impl AdaptiveProposal {
    pub fn new(initial_step: f64) -> Self {
        Self {
            step_size: initial_step,
            target_acceptance: 0.234, // Optimal for multivariate Gaussian
            adaptation_rate: 0.01,
        }
    }
}

impl ProposalStrategy for AdaptiveProposal {
    fn propose(&self, current: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let normal = Normal::new(0.0, self.step_size).unwrap();
        current
            .iter()
            .map(|&x| x + normal.sample(rng))
            .collect()
    }
    
    fn adapt(&mut self, acceptance_rate: f64) {
        let delta = (acceptance_rate - self.target_acceptance) * self.adaptation_rate;
        self.step_size *= (1.0 + delta).max(0.5).min(2.0);
    }
    
    fn name(&self) -> &'static str {
        "AdaptiveGaussian"
    }
}

/// MCMC Configuration Builder
#[derive(Clone)]
pub struct MCMCConfig<P: ProposalStrategy> {
    pub n_samples: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub initial_state: Vec<f64>,
    pub proposal: P,
    pub adaptation_interval: usize,
}

pub struct MCMCConfigBuilder<P: ProposalStrategy> {
    n_samples: usize,
    burn_in: usize,
    thin: usize,
    initial_state: Vec<f64>,
    proposal: Option<P>,
    adaptation_interval: usize,
}

impl<P: ProposalStrategy> MCMCConfigBuilder<P> {
    pub fn new(n_samples: usize, initial_state: Vec<f64>) -> Self {
        Self {
            n_samples,
            burn_in: n_samples / 10,
            thin: 1,
            initial_state,
            proposal: None,
            adaptation_interval: 100,
        }
    }
    
    pub fn burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }
    
    pub fn thin(mut self, thin: usize) -> Self {
        self.thin = thin.max(1);
        self
    }
    
    pub fn proposal(mut self, proposal: P) -> Self {
        self.proposal = Some(proposal);
        self
    }
    
    pub fn adaptation_interval(mut self, interval: usize) -> Self {
        self.adaptation_interval = interval;
        self
    }
    
    pub fn build(self) -> Result<MCMCConfig<P>>
    where
        P: Default,
    {
        if self.n_samples == 0 {
            return Err(OptimizrError::InvalidParameter(
                "n_samples must be positive".to_string(),
            ));
        }
        
        if self.initial_state.is_empty() {
            return Err(OptimizrError::InvalidParameter(
                "initial_state cannot be empty".to_string(),
            ));
        }
        
        Ok(MCMCConfig {
            n_samples: self.n_samples,
            burn_in: self.burn_in,
            thin: self.thin,
            initial_state: self.initial_state,
            proposal: self.proposal.unwrap_or_default(),
            adaptation_interval: self.adaptation_interval,
        })
    }
}

impl Default for GaussianProposal {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl Default for AdaptiveProposal {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Generic log-likelihood function
pub trait LogLikelihood: Send + Sync {
    fn evaluate(&self, state: &[f64]) -> f64;
}

/// Wrapper for Python callable
pub struct PyLogLikelihood {
    func: Py<PyAny>,
}

impl PyLogLikelihood {
    pub fn new(func: Py<PyAny>) -> Self {
        Self { func }
    }
}

impl LogLikelihood for PyLogLikelihood {
    fn evaluate(&self, state: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let args = (state.to_vec(),);
            self.func
                .call1(py, args)
                .and_then(|res| res.extract::<f64>(py))
                .unwrap_or(f64::NEG_INFINITY)
        })
    }
}

/// Refactored MCMC Sampler
pub struct MetropolisHastings<P: ProposalStrategy, L: LogLikelihood> {
    pub config: MCMCConfig<P>,
    pub log_likelihood: L,
}

impl<P: ProposalStrategy, L: LogLikelihood> MetropolisHastings<P, L> {
    pub fn new(config: MCMCConfig<P>, log_likelihood: L) -> Self {
        Self {
            config,
            log_likelihood,
        }
    }
    
    /// Run single chain with functional composition
    pub fn sample_chain(&mut self) -> Result<Vec<Vec<f64>>> {
        let mut rng = rand::thread_rng();
        let mut current_state = self.config.initial_state.clone();
        let mut current_ll = self.log_likelihood.evaluate(&current_state);
        
        let total_steps = self.config.n_samples + self.config.burn_in;
        let mut samples = Vec::with_capacity(self.config.n_samples / self.config.thin);
        let mut acceptance_count = 0usize;
        
        for step in 0..total_steps {
            // Propose new state
            let proposed_state = self.config.proposal.propose(&current_state, &mut rng);
            let proposed_ll = self.log_likelihood.evaluate(&proposed_state);
            
            // Metropolis-Hastings acceptance
            let log_alpha = proposed_ll - current_ll;
            let accepted = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();
            
            if accepted {
                current_state = proposed_state;
                current_ll = proposed_ll;
                acceptance_count += 1;
            }
            
            // Adapt proposal if needed
            if step > 0 && step % self.config.adaptation_interval == 0 {
                let acceptance_rate =
                    acceptance_count as f64 / self.config.adaptation_interval as f64;
                self.config.proposal.adapt(acceptance_rate);
                acceptance_count = 0;
            }
            
            // Store sample after burn-in
            if step >= self.config.burn_in && (step - self.config.burn_in) % self.config.thin == 0
            {
                samples.push(current_state.clone());
            }
        }
        
        Ok(samples)
    }
    
    /// Compute diagnostics
    pub fn diagnostics(&self, samples: &[Vec<f64>]) -> Result<SamplerDiagnostics> {
        if samples.is_empty() {
            return Err(OptimizrError::EmptyData);
        }
        
        let n_samples = samples.len();
        let dim = samples[0].len();
        
        // Compute means and variances
        let means: Vec<f64> = (0..dim)
            .map(|d| samples.iter().map(|s| s[d]).sum::<f64>() / n_samples as f64)
            .collect();
        
        let variances: Vec<f64> = (0..dim)
            .map(|d| {
                let mean = means[d];
                samples
                    .iter()
                    .map(|s| (s[d] - mean).powi(2))
                    .sum::<f64>()
                    / (n_samples - 1) as f64
            })
            .collect();
        
        // Compute autocorrelations (lag 1)
        let autocorrs: Vec<f64> = (0..dim)
            .map(|d| {
                if n_samples < 2 {
                    return 0.0;
                }
                
                let mean = means[d];
                let var = variances[d];
                
                if var < 1e-10 {
                    return 0.0;
                }
                
                let cov: f64 = (0..n_samples - 1)
                    .map(|i| (samples[i][d] - mean) * (samples[i + 1][d] - mean))
                    .sum::<f64>()
                    / (n_samples - 1) as f64;
                
                cov / var
            })
            .collect();
        
        Ok(SamplerDiagnostics {
            n_samples,
            means,
            std_devs: variances.iter().map(|v| v.sqrt()).collect(),
            autocorrelations: autocorrs,
        })
    }
}

impl<P: ProposalStrategy + 'static, L: LogLikelihood + 'static> Sampler
    for MetropolisHastings<P, L>
{
    type Config = MCMCConfig<P>;
    type Output = Vec<Vec<f64>>;
    
    fn sample(&mut self) -> Result<Self::Output> {
        self.sample_chain()
    }
    
    fn diagnostics(&self, samples: &Self::Output) -> Result<SamplerDiagnostics> {
        self.diagnostics(samples)
    }
}

// Python bindings
#[pyfunction]
#[pyo3(signature = (log_likelihood_fn, initial_state, n_samples, step_size=0.1, burn_in=None))]
pub fn mcmc_sample(
    log_likelihood_fn: Py<PyAny>,
    initial_state: Vec<f64>,
    n_samples: usize,
    step_size: f64,
    burn_in: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let burn_in = burn_in.unwrap_or(n_samples / 10);
    
    let proposal = GaussianProposal::new(step_size);
    let config = MCMCConfig {
        n_samples,
        burn_in,
        thin: 1,
        initial_state,
        proposal,
        adaptation_interval: 100,
    };
    
    let log_likelihood = PyLogLikelihood::new(log_likelihood_fn);
    let mut sampler = MetropolisHastings::new(config, log_likelihood);
    
    sampler
        .sample_chain()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (log_likelihood_fn, initial_state, n_samples, initial_step=0.1, burn_in=None))]
pub fn adaptive_mcmc_sample(
    log_likelihood_fn: Py<PyAny>,
    initial_state: Vec<f64>,
    n_samples: usize,
    initial_step: f64,
    burn_in: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let burn_in = burn_in.unwrap_or(n_samples / 10);
    
    let proposal = AdaptiveProposal::new(initial_step);
    let config = MCMCConfig {
        n_samples,
        burn_in,
        thin: 1,
        initial_state,
        proposal,
        adaptation_interval: 100,
    };
    
    let log_likelihood = PyLogLikelihood::new(log_likelihood_fn);
    let mut sampler = MetropolisHastings::new(config, log_likelihood);
    
    sampler
        .sample_chain()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestLogLikelihood;
    
    impl LogLikelihood for TestLogLikelihood {
        fn evaluate(&self, state: &[f64]) -> f64 {
            // Standard normal log-likelihood
            -0.5 * state.iter().map(|x| x.powi(2)).sum::<f64>()
        }
    }

    #[test]
    fn test_mcmc_builder() {
        let config = MCMCConfigBuilder::<GaussianProposal>::new(1000, vec![0.0, 0.0])
            .burn_in(100)
            .thin(2)
            .build()
            .unwrap();
        
        assert_eq!(config.n_samples, 1000);
        assert_eq!(config.burn_in, 100);
        assert_eq!(config.thin, 2);
    }

    #[test]
    fn test_mcmc_sampling() {
        let config = MCMCConfigBuilder::<GaussianProposal>::new(100, vec![0.0])
            .proposal(GaussianProposal::new(0.5))
            .build()
            .unwrap();
        
        let log_likelihood = TestLogLikelihood;
        let mut sampler = MetropolisHastings::new(config, log_likelihood);
        
        let samples = sampler.sample_chain().unwrap();
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_adaptive_proposal() {
        let mut proposal = AdaptiveProposal::new(0.1);
        let initial_step = proposal.step_size;
        
        // High acceptance should increase step size
        proposal.adapt(0.5);
        assert!(proposal.step_size > initial_step);
        
        // Low acceptance should decrease step size
        let current_step = proposal.step_size;
        proposal.adapt(0.1);
        assert!(proposal.step_size < current_step);
    }
}
