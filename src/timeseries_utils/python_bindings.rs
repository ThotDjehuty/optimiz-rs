//! Python bindings for time-series utilities

use pyo3::prelude::*;

use super::{
    create_lagged_features, prepare_for_hmm, return_statistics, rolling_correlation,
    rolling_half_life, rolling_hurst_exponent,
};

#[pyfunction]
#[pyo3(signature = (prices, lag_periods))]
pub fn prepare_for_hmm_py(prices: Vec<f64>, lag_periods: Vec<usize>) -> Vec<Vec<f64>> {
    prepare_for_hmm(&prices, &lag_periods)
}

#[pyfunction]
#[pyo3(signature = (returns, window_size))]
pub fn rolling_hurst_exponent_py(returns: Vec<f64>, window_size: usize) -> Vec<f64> {
    rolling_hurst_exponent(&returns, window_size)
}

#[pyfunction]
#[pyo3(signature = (prices, window_size))]
pub fn rolling_half_life_py(prices: Vec<f64>, window_size: usize) -> Vec<f64> {
    rolling_half_life(&prices, window_size)
}

#[pyfunction]
#[pyo3(signature = (returns,))]
pub fn return_statistics_py(returns: Vec<f64>) -> (f64, f64, f64, f64, f64) {
    return_statistics(&returns)
}

#[pyfunction]
#[pyo3(signature = (series, lags, include_original=true))]
pub fn create_lagged_features_py(
    series: Vec<f64>,
    lags: Vec<usize>,
    include_original: bool,
) -> Vec<Vec<f64>> {
    create_lagged_features(&series, &lags, include_original)
}

#[pyfunction]
#[pyo3(signature = (series1, series2, window_size))]
pub fn rolling_correlation_py(
    series1: Vec<f64>,
    series2: Vec<f64>,
    window_size: usize,
) -> Vec<f64> {
    rolling_correlation(&series1, &series2, window_size)
}

pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prepare_for_hmm_py, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_hurst_exponent_py, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_half_life_py, m)?)?;
    m.add_function(wrap_pyfunction!(return_statistics_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_lagged_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_correlation_py, m)?)?;
    Ok(())
}
