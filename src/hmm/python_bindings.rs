//! Python bindings for HMM module
//!
//! Provides PyO3 wrappers for easy use from Python.

use super::config::HMMConfig;
use super::emission::GaussianEmission;
use super::model::HMM;
use pyo3::prelude::*;

/// Python-facing HMM parameters
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

/// Fit HMM using Baum-Welch algorithm
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

/// Decode most likely state sequence using Viterbi algorithm
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
