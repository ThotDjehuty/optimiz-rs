//! Time Series Risk Metrics and Analysis
//!
//! Generic statistical risk metrics for time series evaluation:
//! - Hurst exponent (mean-reversion and persistence detection)
//! - Performance ratios (Sharpe, Sortino, Calmar)
//! - Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)
//! - Maximum Drawdown and recovery metrics
//! - Statistical moments and distribution analysis
//! - Bootstrap and Monte Carlo simulation utilities
//!
//! These metrics apply to any return/increment series (portfolio returns,
//! spread dynamics, trading signals, residuals, etc.)

use crate::core::{OptimizrError, OptimizrResult};
#[allow(unused_imports)]  // Array2 used in bootstrap_returns_py
use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Result from Hurst exponent calculation
#[derive(Debug, Clone)]
pub struct HurstResult {
    pub hurst_exponent: f64,             // H < 0.5 = mean-reverting
    pub confidence_interval: (f64, f64), // 95% CI
    pub standard_error: f64,             // SE of estimate
    pub is_mean_reverting: bool,         // True if CI upper < 0.5
    pub window_sizes: Vec<usize>,        // Window sizes used
    pub rs_values: Vec<f64>,             // R/S values
}

/// Hurst Exponent via Rescaled Range (R/S) Analysis
///
/// Estimates the Hurst exponent H:
/// - H = 0.5: Random walk (Brownian motion)
/// - H < 0.5: Mean-reverting (anti-persistent)
/// - H > 0.5: Trending (persistent)
///
/// # Arguments
/// * `series` - Time series data
/// * `window_sizes` - Window sizes for R/S analysis (e.g., [8, 16, 32, 64])
///
/// # Returns
/// `HurstResult` with Hurst exponent and statistics
pub fn hurst_exponent(series: &Array1<f64>, window_sizes: &[usize]) -> OptimizrResult<HurstResult> {
    let n = series.len();

    if n < 20 {
        return Err(OptimizrError::InvalidInput(
            "Series too short for Hurst analysis (need >= 20)".to_string(),
        ));
    }

    if window_sizes.is_empty() {
        return Err(OptimizrError::InvalidInput(
            "window_sizes cannot be empty".to_string(),
        ));
    }

    let mut rs_values = Vec::with_capacity(window_sizes.len());
    let mut log_lags = Vec::with_capacity(window_sizes.len());

    for &lag in window_sizes {
        if lag >= n {
            continue;
        }

        let n_windows = n / lag;
        let mut rs_window_values = Vec::with_capacity(n_windows);

        for w in 0..n_windows {
            let start = w * lag;
            let end = start + lag;
            let window = series.slice(ndarray::s![start..end]);

            // Compute mean
            let mean = window.mean().unwrap_or(0.0);

            // Cumulative deviations from mean
            let mut y = Vec::with_capacity(lag);
            let mut cumsum = 0.0;
            for &x in window.iter() {
                cumsum += x - mean;
                y.push(cumsum);
            }

            // Range R
            let max_y = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_y = y.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_y - min_y;

            // Standard deviation S
            let variance: f64 =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / lag as f64;
            let std = variance.sqrt();

            if std > 1e-10 {
                rs_window_values.push(range / std);
            }
        }

        if !rs_window_values.is_empty() {
            let mean_rs: f64 = rs_window_values.iter().sum::<f64>() / rs_window_values.len() as f64;
            rs_values.push(mean_rs);
            log_lags.push((lag as f64).ln());
        }
    }

    if rs_values.len() < 2 {
        return Err(OptimizrError::ComputationError(
            "Not enough valid window sizes for regression".to_string(),
        ));
    }

    // Linear regression: log(R/S) = H * log(lag) + c
    let log_rs: Vec<f64> = rs_values.iter().map(|x| x.ln()).collect();
    let (slope, _intercept, se) = linear_regression(&log_lags, &log_rs)?;

    let hurst = slope;

    // 95% confidence interval
    let t_critical = 1.96; // Approximate for large n
    let ci_lower = hurst - t_critical * se;
    let ci_upper = hurst + t_critical * se;

    let is_mean_reverting = ci_upper < 0.5;

    Ok(HurstResult {
        hurst_exponent: hurst,
        confidence_interval: (ci_lower, ci_upper),
        standard_error: se,
        is_mean_reverting,
        window_sizes: window_sizes.to_vec(),
        rs_values,
    })
}

/// Simple linear regression
fn linear_regression(x: &[f64], y: &[f64]) -> OptimizrResult<(f64, f64, f64)> {
    if x.len() != y.len() || x.is_empty() {
        return Err(OptimizrError::InvalidInput(
            "x and y must have same non-zero length".to_string(),
        ));
    }

    let n = x.len() as f64;
    let x_mean: f64 = x.iter().sum::<f64>() / n;
    let y_mean: f64 = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..x.len() {
        let x_dev = x[i] - x_mean;
        let y_dev = y[i] - y_mean;
        numerator += x_dev * y_dev;
        denominator += x_dev * x_dev;
    }

    if denominator.abs() < 1e-10 {
        return Err(OptimizrError::ComputationError(
            "Degenerate regression (zero variance in x)".to_string(),
        ));
    }

    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;

    // Standard error of slope
    let mut residual_sum = 0.0;
    for i in 0..x.len() {
        let predicted = slope * x[i] + intercept;
        residual_sum += (y[i] - predicted).powi(2);
    }

    let mse = residual_sum / (n - 2.0).max(1.0);
    let se = (mse / denominator).sqrt();

    Ok((slope, intercept, se))
}

/// Portfolio risk metrics
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub sharpe_ratio: f64,            // Risk-adjusted return
    pub sortino_ratio: f64,           // Downside risk-adjusted return
    pub calmar_ratio: f64,            // Return / Max Drawdown
    pub max_drawdown: f64,            // Maximum peak-to-trough decline
    pub max_drawdown_duration: usize, // Days in max drawdown
    pub var_95: f64,                  // Value at Risk (95%)
    pub cvar_95: f64,                 // Conditional VaR (Expected Shortfall)
    pub volatility: f64,              // Annualized volatility
    pub downside_deviation: f64,      // Downside risk
    pub skewness: f64,                // Return distribution skewness
    pub kurtosis: f64,                // Return distribution kurtosis
}

/// Compute comprehensive risk metrics for a return series
///
/// # Arguments
/// * `returns` - Daily returns (not log returns)
/// * `risk_free_rate` - Annual risk-free rate (e.g., 0.02 for 2%)
/// * `periods_per_year` - 252 for daily, 52 for weekly, 12 for monthly
///
/// # Returns
/// `RiskMetrics` with all computed metrics
pub fn compute_risk_metrics(
    returns: &Array1<f64>,
    risk_free_rate: f64,
    periods_per_year: f64,
) -> OptimizrResult<RiskMetrics> {
    if returns.is_empty() {
        return Err(OptimizrError::InvalidInput(
            "Returns array is empty".to_string(),
        ));
    }

    let n = returns.len() as f64;
    let daily_rf = risk_free_rate / periods_per_year;

    // Mean return
    let mean_return = returns.mean().unwrap_or(0.0);
    let excess_return = mean_return - daily_rf;

    // Volatility
    let variance = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>()
        / n;
    let volatility = variance.sqrt() * periods_per_year.sqrt();

    // Sharpe ratio
    let sharpe_ratio = if volatility > 1e-10 {
        excess_return * periods_per_year.sqrt() / volatility
    } else {
        0.0
    };

    // Downside deviation (semi-deviation)
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    let downside_deviation = if !downside_returns.is_empty() {
        let dd_variance = downside_returns.iter().map(|&r| r.powi(2)).sum::<f64>()
            / downside_returns.len() as f64;
        dd_variance.sqrt() * periods_per_year.sqrt()
    } else {
        0.0
    };

    // Sortino ratio
    let sortino_ratio = if downside_deviation > 1e-10 {
        excess_return * periods_per_year.sqrt() / downside_deviation
    } else {
        0.0
    };

    // Maximum drawdown
    let cum_returns = returns
        .iter()
        .scan(1.0, |state, &r| {
            *state *= 1.0 + r;
            Some(*state)
        })
        .collect::<Vec<f64>>();

    let mut running_max = cum_returns[0];
    let mut max_dd: f64 = 0.0;
    let mut current_dd_duration = 0;
    let mut max_dd_duration = 0;
    let mut in_drawdown = false;

    for &cum_ret in &cum_returns {
        if cum_ret > running_max {
            running_max = cum_ret;
            if in_drawdown {
                max_dd_duration = max_dd_duration.max(current_dd_duration);
                current_dd_duration = 0;
                in_drawdown = false;
            }
        } else {
            let dd = (running_max - cum_ret) / running_max;
            max_dd = max_dd.max(dd);
            in_drawdown = true;
            current_dd_duration += 1;
        }
    }

    if in_drawdown {
        max_dd_duration = max_dd_duration.max(current_dd_duration);
    }

    // Calmar ratio
    let annual_return = mean_return * periods_per_year;
    let calmar_ratio = if max_dd > 1e-10 {
        annual_return / max_dd
    } else {
        0.0
    };

    // VaR and CVaR (95% confidence)
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let var_95_idx = (n * 0.05) as usize;
    let var_95 = -sorted_returns[var_95_idx.min(sorted_returns.len() - 1)];

    let cvar_95 = if var_95_idx > 0 {
        -sorted_returns[..var_95_idx].iter().sum::<f64>() / var_95_idx as f64
    } else {
        var_95
    };

    // Skewness and kurtosis
    let skewness = if n > 2.0 {
        let m3: f64 = returns
            .iter()
            .map(|&r| (r - mean_return).powi(3))
            .sum::<f64>()
            / n;
        m3 / variance.powf(1.5)
    } else {
        0.0
    };

    let kurtosis = if n > 3.0 {
        let m4: f64 = returns
            .iter()
            .map(|&r| (r - mean_return).powi(4))
            .sum::<f64>()
            / n;
        m4 / variance.powi(2) - 3.0 // Excess kurtosis
    } else {
        0.0
    };

    Ok(RiskMetrics {
        sharpe_ratio,
        sortino_ratio,
        calmar_ratio,
        max_drawdown: max_dd,
        max_drawdown_duration: max_dd_duration,
        var_95,
        cvar_95,
        volatility,
        downside_deviation,
        skewness,
        kurtosis,
    })
}

/// Half-life estimation for mean-reverting series
///
/// Estimates the half-life of mean reversion using AR(1) model:
/// ΔX_t = θ X_{t-1} + ε_t
///
/// Half-life = -ln(2) / ln(|θ|)
///
/// # Arguments
/// * `series` - Time series (typically portfolio value or spread)
///
/// # Returns
/// Half-life in time periods (e.g., days if daily data)
pub fn estimate_half_life(series: &Array1<f64>) -> OptimizrResult<f64> {
    if series.len() < 10 {
        return Err(OptimizrError::InvalidInput(
            "Series too short for half-life estimation".to_string(),
        ));
    }

    let mean = series.mean().unwrap_or(0.0);
    let deviations: Vec<f64> = series.iter().map(|&x| x - mean).collect();

    // AR(1) regression: dev[t] = theta * dev[t-1] + error
    let x: Vec<f64> = deviations[..deviations.len() - 1].to_vec();
    let y: Vec<f64> = deviations[1..].to_vec();

    let (theta, _, _) = linear_regression(&x, &y)?;

    if theta.abs() >= 1.0 || theta.abs() < 1e-10 {
        // Not mean-reverting or degenerate
        return Ok(f64::INFINITY);
    }

    let half_life = -2.0_f64.ln() / theta.abs().ln();

    Ok(half_life)
}

/// Monte Carlo bootstrap simulation
///
/// Performs bootstrap resampling to estimate uncertainty in metrics
///
/// # Arguments
/// * `returns` - Historical returns
/// * `n_simulations` - Number of bootstrap samples
/// * `block_size` - Optional block size for block bootstrap (preserves autocorrelation)
///
/// # Returns
/// Vector of simulated return series
pub fn bootstrap_returns(
    returns: &Array1<f64>,
    n_simulations: usize,
    block_size: Option<usize>,
) -> Vec<Array1<f64>> {
    use rand::Rng;
    use rand::SeedableRng;

    let n = returns.len();
    let block_size = block_size.unwrap_or(1);

    (0..n_simulations)
        .into_par_iter()
        .map(|seed| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
            let mut sim_returns = Vec::with_capacity(n);

            if block_size == 1 {
                // Simple bootstrap
                while sim_returns.len() < n {
                    let idx = rng.gen_range(0..n);
                    sim_returns.push(returns[idx]);
                }
            } else {
                // Block bootstrap
                let n_blocks = (n as f64 / block_size as f64).ceil() as usize;

                for _ in 0..n_blocks {
                    let start_idx = rng.gen_range(0..=(n.saturating_sub(block_size)));
                    let end_idx = (start_idx + block_size).min(n);

                    for idx in start_idx..end_idx {
                        if sim_returns.len() < n {
                            sim_returns.push(returns[idx]);
                        }
                    }
                }
            }

            Array1::from_vec(sim_returns[..n].to_vec())
        })
        .collect()
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "python-bindings")]
use numpy::{PyArray2, PyReadonlyArray1};
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;

#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (series, window_sizes=None))]
pub fn hurst_exponent_py(
    py: Python,
    series: PyReadonlyArray1<f64>,
    window_sizes: Option<Vec<usize>>,
) -> PyResult<PyObject> {
    let series_arr = series.as_array().to_owned();
    let windows = window_sizes.unwrap_or_else(|| vec![8, 16, 32, 64, 128]);

    let result = hurst_exponent(&series_arr, &windows)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("hurst_exponent", result.hurst_exponent)?;
    dict.set_item("confidence_interval", result.confidence_interval)?;
    dict.set_item("standard_error", result.standard_error)?;
    dict.set_item("is_mean_reverting", result.is_mean_reverting)?;
    dict.set_item("window_sizes", result.window_sizes)?;
    dict.set_item("rs_values", result.rs_values)?;

    Ok(dict.into())
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (returns, risk_free_rate=0.02, periods_per_year=252.0))]
pub fn compute_risk_metrics_py(
    py: Python,
    returns: PyReadonlyArray1<f64>,
    risk_free_rate: f64,
    periods_per_year: f64,
) -> PyResult<PyObject> {
    let returns_arr = returns.as_array().to_owned();

    let result = compute_risk_metrics(&returns_arr, risk_free_rate, periods_per_year)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("sharpe_ratio", result.sharpe_ratio)?;
    dict.set_item("sortino_ratio", result.sortino_ratio)?;
    dict.set_item("calmar_ratio", result.calmar_ratio)?;
    dict.set_item("max_drawdown", result.max_drawdown)?;
    dict.set_item("max_drawdown_duration", result.max_drawdown_duration)?;
    dict.set_item("var_95", result.var_95)?;
    dict.set_item("cvar_95", result.cvar_95)?;
    dict.set_item("volatility", result.volatility)?;
    dict.set_item("downside_deviation", result.downside_deviation)?;
    dict.set_item("skewness", result.skewness)?;
    dict.set_item("kurtosis", result.kurtosis)?;

    Ok(dict.into())
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn estimate_half_life_py(series: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let series_arr = series.as_array().to_owned();

    estimate_half_life(&series_arr).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (returns, n_simulations=1000, block_size=None))]
pub fn bootstrap_returns_py(
    py: Python,
    returns: PyReadonlyArray1<f64>,
    n_simulations: usize,
    block_size: Option<usize>,
) -> PyResult<PyObject> {
    let returns_arr = returns.as_array().to_owned();

    let simulations = bootstrap_returns(&returns_arr, n_simulations, block_size);

    // Convert Vec<Array1> to 2D array
    let n = returns_arr.len();
    let mut result = Array2::zeros((n_simulations, n));

    for (i, sim) in simulations.iter().enumerate() {
        for (j, &val) in sim.iter().enumerate() {
            result[[i, j]] = val;
        }
    }

    Ok(PyArray2::from_owned_array_bound(py, result).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hurst_random_walk() {
        // Random walk should have H ≈ 0.5
        let n = 1000;
        let mut series = vec![0.0];
        for i in 1..n {
            series.push(series[i - 1] + if i % 2 == 0 { 1.0 } else { -1.0 });
        }

        let series = Array1::from_vec(series);
        let result = hurst_exponent(&series, &[8, 16, 32, 64]).unwrap();

        // Should be close to 0.5
        assert!((result.hurst_exponent - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_risk_metrics() {
        // Simple return series
        let returns = Array1::from_vec(vec![0.01, -0.005, 0.015, -0.01, 0.02]);
        let metrics = compute_risk_metrics(&returns, 0.02, 252.0).unwrap();

        assert!(metrics.volatility > 0.0);
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.var_95 >= 0.0);
    }

    #[test]
    fn test_half_life() {
        // Mean-reverting series
        let series = Array1::from_vec(vec![1.0, 0.8, 0.6, 0.7, 0.9, 1.0, 1.1, 0.9, 1.0, 0.95]);

        let half_life = estimate_half_life(&series).unwrap();
        assert!(half_life > 0.0 && half_life < 100.0);
    }
}
