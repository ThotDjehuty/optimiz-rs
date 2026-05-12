//! Python bindings for `inference`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::robust_drift::{estimate_robust_drift, RobustDriftConfig};

#[pyfunction]
#[pyo3(signature = (observations, dt, huber_delta=1.345, max_iterations=200, tolerance=1e-9))]
fn robust_drift(
    py: Python<'_>,
    observations: Vec<f64>,
    dt: f64,
    huber_delta: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyObject> {
    let cfg = RobustDriftConfig { dt, huber_delta, max_iterations, tolerance };
    let res = estimate_robust_drift(&observations, &cfg)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("a", res.a)?;
    dict.set_item("b", res.b)?;
    dict.set_item("iterations", res.iterations)?;
    Ok(dict.into())
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(robust_drift, m)?)?;
    Ok(())
}
