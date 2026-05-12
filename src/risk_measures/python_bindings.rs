//! Python bindings for empirical risk measures.
//!
//! Exposes Value-at-Risk and Conditional Value-at-Risk estimators
//! together with the simplex-constrained CVaR minimiser.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ndarray::Array2;

use super::cvar::{cvar_value, minimize_cvar, CVaRConfig};
use super::var::{historical_var, parametric_var};

/// Empirical Value-at-Risk at confidence level `alpha`.
#[pyfunction]
#[pyo3(signature = (losses, alpha=0.95))]
fn historical_var_py(losses: Vec<f64>, alpha: f64) -> PyResult<f64> {
    historical_var(&losses, alpha).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

/// Closed-form Gaussian Value-at-Risk `mu + sigma * Phi^{-1}(alpha)`.
#[pyfunction]
#[pyo3(signature = (mu, sigma, alpha=0.95))]
fn parametric_var_py(mu: f64, sigma: f64, alpha: f64) -> PyResult<f64> {
    parametric_var(mu, sigma, alpha).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

/// Empirical Conditional Value-at-Risk at confidence level `alpha`.
#[pyfunction]
#[pyo3(signature = (losses, alpha=0.95))]
fn cvar_value_py(losses: Vec<f64>, alpha: f64) -> PyResult<f64> {
    cvar_value(&losses, alpha).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

/// Minimise empirical CVaR of `L(w) = -<r^{(s)}, w>` over the unit
/// simplex. `samples` has shape `(S, d)` (S samples, d decision
/// components).
#[pyfunction]
#[pyo3(signature = (samples, alpha=0.95, n_iter=5000, step_size=0.01, tol=1e-8))]
fn minimize_cvar_py(
    py: Python<'_>,
    samples: Vec<Vec<f64>>,
    alpha: f64,
    n_iter: usize,
    step_size: f64,
    tol: f64,
) -> PyResult<PyObject> {
    let s = samples.len();
    if s == 0 {
        return Err(PyValueError::new_err("samples must be non-empty"));
    }
    let d = samples[0].len();
    if d == 0 {
        return Err(PyValueError::new_err("samples rows must be non-empty"));
    }
    let mut flat = Vec::with_capacity(s * d);
    for row in &samples {
        if row.len() != d {
            return Err(PyValueError::new_err(
                "all sample rows must share the same length",
            ));
        }
        flat.extend_from_slice(row);
    }
    let arr = Array2::from_shape_vec((s, d), flat)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let cfg = CVaRConfig {
        alpha,
        n_iter,
        step_size,
        tol,
    };
    let result = minimize_cvar(arr.view(), &cfg)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("w", result.w.to_vec())?;
    dict.set_item("zeta", result.zeta)?;
    dict.set_item("cvar", result.cvar)?;
    dict.set_item("iterations", result.iterations)?;
    Ok(dict.into())
}

/// Register all risk-measure functions with the Python module.
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(historical_var_py, m)?)?;
    m.add_function(wrap_pyfunction!(parametric_var_py, m)?)?;
    m.add_function(wrap_pyfunction!(cvar_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_cvar_py, m)?)?;
    Ok(())
}
