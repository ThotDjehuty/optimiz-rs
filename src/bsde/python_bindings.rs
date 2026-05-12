//! Python bindings for the BSDE module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::theta_scheme::{solve_linear_bsde, ThetaSchemeConfig};

/// Solve the linear BSDE -dY = (a Y + b Z + c) dt - Z dW with deterministic
/// constant coefficients.  Returns `{y, z, time_grid}` (numpy arrays).
#[pyfunction]
#[pyo3(signature = (a_const, b_const, c_const, terminal, n_steps, t_horizon, theta=0.5))]
fn linear_bsde_constant_coeffs(
    py: Python<'_>,
    a_const: f64,
    b_const: f64,
    c_const: f64,
    terminal: f64,
    n_steps: usize,
    t_horizon: f64,
    theta: f64,
) -> PyResult<PyObject> {
    let cfg = ThetaSchemeConfig { n_steps, t_horizon, theta };
    let res = solve_linear_bsde(|_| a_const, |_| b_const, |_| c_const, terminal, &cfg)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("y", res.y.to_vec())?;
    dict.set_item("z", res.z.to_vec())?;
    dict.set_item("time_grid", res.time_grid.to_vec())?;
    Ok(dict.into())
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linear_bsde_constant_coeffs, m)?)?;
    Ok(())
}
