//! Python bindings for `optimization::generative_calibration_hooks`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::generative_calibration_hooks::{mmd_distance, MmdLoss};

/// Maximum Mean Discrepancy with Gaussian kernel of bandwidth `sigma`.
#[pyfunction]
#[pyo3(signature = (x, y, sigma=1.0))]
fn mmd_gaussian(x: Vec<f64>, y: Vec<f64>, sigma: f64) -> PyResult<f64> {
    let loss = MmdLoss { sigma };
    mmd_distance(&x, &y, &loss).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mmd_gaussian, m)?)?;
    Ok(())
}
