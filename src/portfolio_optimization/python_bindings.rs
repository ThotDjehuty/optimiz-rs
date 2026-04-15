//! Python bindings for portfolio optimisation.
//!
//! Exposes CARA, mean-variance, minimum-variance, and ERC optimisers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::cara::CARAOptimizer;
use super::mean_variance::{equal_risk_contribution, minimum_variance, MeanVarianceOptimizer};
use super::traits::PortfolioOptimizer;

/// CARA optimal weights: maximize w'μ − (γ/2)w'Σw  s.t. Σw=1, 0≤w≤max.
#[pyfunction]
#[pyo3(signature = (mu, cov, risk_aversion=2.0, max_weight=0.3))]
fn cara_optimal_weights(
    py: Python<'_>,
    mu: Vec<f64>,
    cov: Vec<Vec<f64>>,
    risk_aversion: f64,
    max_weight: f64,
) -> PyResult<PyObject> {
    let opt =
        CARAOptimizer::new(risk_aversion).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let result = opt
        .optimize(&mu, &cov, max_weight)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("weights", result.weights.clone())?;
    dict.set_item("utility", result.utility)?;
    dict.set_item("expected_return", result.expected_return)?;
    dict.set_item("portfolio_variance", result.portfolio_variance)?;
    dict.set_item("sharpe_ratio", result.sharpe_ratio(0.05))?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict.into())
}

/// Mean-variance optimal weights with optional score tilting.
#[pyfunction]
#[pyo3(signature = (mu, cov, risk_aversion=2.0, max_weight=0.3, scores=None))]
fn mean_variance_optimal_weights(
    py: Python<'_>,
    mu: Vec<f64>,
    cov: Vec<Vec<f64>>,
    risk_aversion: f64,
    max_weight: f64,
    scores: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mut opt = MeanVarianceOptimizer::new(risk_aversion);
    if let Some(s) = scores {
        opt = opt.with_scores(s);
    }
    let result = opt
        .optimize(&mu, &cov, max_weight)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("weights", result.weights.clone())?;
    dict.set_item("utility", result.utility)?;
    dict.set_item("expected_return", result.expected_return)?;
    dict.set_item("portfolio_variance", result.portfolio_variance)?;
    dict.set_item("sharpe_ratio", result.sharpe_ratio(0.05))?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict.into())
}

/// Minimum variance portfolio.
#[pyfunction]
#[pyo3(signature = (cov, max_weight=0.3))]
fn min_variance_weights(
    py: Python<'_>,
    cov: Vec<Vec<f64>>,
    max_weight: f64,
) -> PyResult<PyObject> {
    let result =
        minimum_variance(&cov, max_weight).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("weights", result.weights.clone())?;
    dict.set_item("portfolio_variance", result.portfolio_variance)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict.into())
}

/// Equal-risk-contribution (Risk Parity) portfolio.
#[pyfunction]
#[pyo3(signature = (cov, max_weight=0.3))]
fn erc_weights(py: Python<'_>, cov: Vec<Vec<f64>>, max_weight: f64) -> PyResult<PyObject> {
    let result = equal_risk_contribution(&cov, max_weight)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("weights", result.weights.clone())?;
    dict.set_item("portfolio_variance", result.portfolio_variance)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict.into())
}

/// Register all portfolio optimization functions with the Python module.
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cara_optimal_weights, m)?)?;
    m.add_function(wrap_pyfunction!(mean_variance_optimal_weights, m)?)?;
    m.add_function(wrap_pyfunction!(min_variance_weights, m)?)?;
    m.add_function(wrap_pyfunction!(erc_weights, m)?)?;
    Ok(())
}
