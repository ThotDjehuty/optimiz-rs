//! Python bindings for `agent_based`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::{simulate_agent_based, AgentBasedConfig};

/// Bounded-confidence consensus: `T(s, ngh, k) = (1 - α) s + α mean(ngh)`.
#[pyfunction]
#[pyo3(signature = (initial, alpha, noise_sigma, n_steps, seed=0))]
fn consensus_dynamics(
    py: Python<'_>,
    initial: Vec<f64>,
    alpha: f64,
    noise_sigma: f64,
    n_steps: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let cfg = AgentBasedConfig {
        n_agents: initial.len(),
        n_steps,
        noise_sigma,
        seed,
    };
    let res = simulate_agent_based(
        &initial,
        |s, ngh, _k| {
            let m: f64 = ngh.iter().sum::<f64>() / ngh.len() as f64;
            (1.0 - alpha) * s + alpha * m
        },
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    let flat: Vec<f64> = res.states.iter().copied().collect();
    dict.set_item("states_flat", flat)?;
    dict.set_item("n_steps", n_steps + 1)?;
    dict.set_item("n_agents", initial.len())?;
    dict.set_item("mean_trajectory", res.mean_trajectory.to_vec())?;
    Ok(dict.into())
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(consensus_dynamics, m)?)?;
    Ok(())
}
