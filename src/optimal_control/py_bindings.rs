//! Python bindings for optimal control module

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use crate::optimal_control::*;

/// Solve HJB equation for optimal switching boundaries
#[pyfunction]
#[pyo3(signature = (kappa, theta, sigma, rho=0.04, transaction_cost=0.001, n_points=200, max_iter=2000, tolerance=1e-6, n_std=4.0))]
pub fn solve_hjb_py(
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    transaction_cost: f64,
    n_points: usize,
    max_iter: usize,
    tolerance: f64,
    n_std: f64,
) -> PyResult<(f64, f64, f64, usize)> {
    let config = HJBConfig {
        kappa,
        theta,
        sigma,
        rho,
        transaction_cost,
        n_points,
        max_iter,
        tolerance,
        n_std,
    };
    
    let solver = HJBSolver::new(config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    let result = solver.solve()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    
    Ok((
        result.lower_boundary,
        result.upper_boundary,
        result.residual,
        result.iterations,
    ))
}

/// Estimate Ornstein-Uhlenbeck parameters
#[pyfunction]
#[pyo3(signature = (spread, dt=1.0/252.0))]
pub fn estimate_ou_params_py(
    spread: PyReadonlyArray1<f64>,
    dt: f64,
) -> PyResult<(f64, f64, f64, f64)> {
    let spread = spread.as_slice()?;
    
    let params = estimate_ou_params(spread, dt)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    Ok((params.kappa, params.theta, params.sigma, params.half_life))
}

/// Engle-Granger cointegration test
#[pyfunction]
#[pyo3(signature = (y, x, significance=0.05))]
pub fn engle_granger_test_py(
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    significance: f64,
) -> PyResult<(f64, f64, f64, bool)> {
    let y = y.as_slice()?;
    let x = x.as_slice()?;
    
    let result = engle_granger_test(y, x, significance)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    Ok((
        result.beta,
        result.adf_statistic,
        result.p_value,
        result.is_cointegrated,
    ))
}

/// Calculate Hurst exponent
#[pyfunction]
#[pyo3(signature = (series, max_lag=20))]
pub fn hurst_exponent_py(
    series: PyReadonlyArray1<f64>,
    max_lag: usize,
) -> PyResult<f64> {
    let series = series.as_slice()?;
    
    hurst_exponent(series, max_lag)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
}

/// Backtest optimal switching strategy
#[pyfunction]
#[pyo3(signature = (spread, lower_bound, upper_bound, transaction_cost=0.001))]
pub fn backtest_optimal_switching_py(
    py: Python,
    spread: PyReadonlyArray1<f64>,
    lower_bound: f64,
    upper_bound: f64,
    transaction_cost: f64,
) -> PyResult<(f64, f64, f64, usize, f64, Py<PyArray1<f64>>)> {
    let spread = spread.as_slice()?;
    
    let result = backtest_optimal_switching(spread, lower_bound, upper_bound, transaction_cost)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    let pnl_array = PyArray1::from_vec_bound(py, result.pnl.clone());
    
    Ok((
        result.total_return,
        result.sharpe_ratio,
        result.max_drawdown,
        result.num_trades,
        result.win_rate,
        pnl_array.unbind(),
    ))
}

/// Test a pair comprehensively (cointegration + OU + HJB + backtest)
#[pyfunction]
#[pyo3(signature = (y, x, significance=0.05, min_hurst=0.45, transaction_cost=0.001))]
pub fn test_pair_py(
    py: Python,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    significance: f64,
    min_hurst: f64,
    transaction_cost: f64,
) -> PyResult<Option<PyObject>> {
    let y = y.as_slice()?;
    let x = x.as_slice()?;
    
    // 1. Cointegration test
    let coint_result = match engle_granger_test(y, x, significance) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    
    if !coint_result.is_cointegrated {
        return Ok(None);
    }
    
    // 2. Hurst exponent
    let h = match hurst_exponent(&coint_result.spread, 20) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };
    
    if h.is_nan() || h > min_hurst {
        return Ok(None);
    }
    
    // 3. OU parameters
    let ou_params = match estimate_ou_params(&coint_result.spread, 1.0 / 252.0) {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    
    if ou_params.kappa <= 0.0 || ou_params.kappa.is_nan() {
        return Ok(None);
    }
    
    // 4. HJB solver
    let hjb_config = HJBConfig {
        kappa: ou_params.kappa,
        theta: ou_params.theta,
        sigma: ou_params.sigma,
        rho: 0.04,
        transaction_cost,
        n_points: 200,
        max_iter: 2000,
        tolerance: 1e-6,
        n_std: 4.0,
    };
    
    let solver = match HJBSolver::new(hjb_config) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    
    let hjb_result = match solver.solve() {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    
    // 5. Backtest
    let backtest_result = match backtest_optimal_switching(
        &coint_result.spread,
        hjb_result.lower_boundary,
        hjb_result.upper_boundary,
        transaction_cost,
    ) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    
    // Build result dictionary
    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("beta", coint_result.beta)?;
    result.set_item("p_value", coint_result.p_value)?;
    result.set_item("hurst", h)?;
    result.set_item("kappa", ou_params.kappa)?;
    result.set_item("theta", ou_params.theta)?;
    result.set_item("sigma", ou_params.sigma)?;
    result.set_item("half_life", ou_params.half_life)?;
    result.set_item("lower_boundary", hjb_result.lower_boundary)?;
    result.set_item("upper_boundary", hjb_result.upper_boundary)?;
    result.set_item("total_return", backtest_result.total_return)?;
    result.set_item("sharpe_ratio", backtest_result.sharpe_ratio)?;
    result.set_item("max_drawdown", backtest_result.max_drawdown)?;
    result.set_item("num_trades", backtest_result.num_trades)?;
    result.set_item("win_rate", backtest_result.win_rate)?;
    
    // Scores
    let coint_score = 1.0 - coint_result.p_value;
    let meanrev_score = (0.5 - h) / 0.2;
    let profit_score = backtest_result.total_return.max(0.0);
    let combined_score = coint_score * meanrev_score * (1.0 + profit_score);
    
    result.set_item("coint_score", coint_score)?;
    result.set_item("meanrev_score", meanrev_score)?;
    result.set_item("profit_score", profit_score)?;
    result.set_item("combined_score", combined_score)?;
    
    Ok(Some(result.into()))
}

pub fn register_py_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_hjb_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_ou_params_py, m)?)?;
    m.add_function(wrap_pyfunction!(engle_granger_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_exponent_py, m)?)?;
    m.add_function(wrap_pyfunction!(backtest_optimal_switching_py, m)?)?;
    m.add_function(wrap_pyfunction!(test_pair_py, m)?)?;
    Ok(())
}
