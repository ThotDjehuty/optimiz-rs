//! Python bindings for `mean_field::mckean_vlasov`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::mckean_vlasov::{simulate_mckean_vlasov, McKeanVlasovConfig};

/// Mean-reverting toward the empirical mean: `b(x, μ) = θ (m̄ - x)`.
#[pyfunction]
#[pyo3(signature = (initial, theta, sigma, n_steps, t_horizon, seed=0))]
fn mean_reverting_mckean_vlasov(
    py: Python<'_>,
    initial: Vec<f64>,
    theta: f64,
    sigma: f64,
    n_steps: usize,
    t_horizon: f64,
    seed: u64,
) -> PyResult<PyObject> {
    let cfg = McKeanVlasovConfig {
        n_particles: initial.len(),
        n_steps,
        t_horizon,
        sigma,
        seed,
    };
    let res = simulate_mckean_vlasov(
        &initial,
        |x, mu| {
            let m: f64 = mu.iter().sum::<f64>() / mu.len() as f64;
            theta * (m - x)
        },
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    let flat: Vec<f64> = res.paths.iter().copied().collect();
    dict.set_item("paths_flat", flat)?;
    dict.set_item("n_steps", n_steps + 1)?;
    dict.set_item("n_particles", initial.len())?;
    dict.set_item("time_grid", res.time_grid.to_vec())?;
    Ok(dict.into())
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean_reverting_mckean_vlasov, m)?)?;
    Ok(())
}
