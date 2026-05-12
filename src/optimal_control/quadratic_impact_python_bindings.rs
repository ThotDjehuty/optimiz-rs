//! Python bindings for `optimal_control::quadratic_impact_control`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::quadratic_impact_control::{
    solve_quadratic_impact_control, QuadraticImpactConfig,
};

#[pyfunction]
#[pyo3(signature = (gamma, phi, a_terminal, t_horizon, n_steps))]
fn quadratic_impact_control_py(
    py: Python<'_>,
    gamma: f64,
    phi: f64,
    a_terminal: f64,
    t_horizon: f64,
    n_steps: usize,
) -> PyResult<PyObject> {
    let cfg = QuadraticImpactConfig { gamma, phi, a_terminal, t_horizon, n_steps };
    let res = solve_quadratic_impact_control(&cfg)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("time_grid", res.time_grid.to_vec())?;
    dict.set_item("h", res.h.to_vec())?;
    dict.set_item("feedback_gain", res.feedback_gain.to_vec())?;
    Ok(dict.into())
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quadratic_impact_control_py, m)?)?;
    Ok(())
}
