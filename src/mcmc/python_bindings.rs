//! Python bindings for MCMC module

use super::config::MCMCConfig;
use super::likelihood::PyLogLikelihood;
use super::proposal::{AdaptiveProposal, GaussianProposal};
use super::sampler::MetropolisHastings;
use pyo3::prelude::*;

/// Basic MCMC sampling with Gaussian proposal
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

/// Adaptive MCMC sampling with automatic step size tuning
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
