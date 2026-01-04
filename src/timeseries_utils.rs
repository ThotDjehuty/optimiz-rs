//! Time-Series Utilities for Finance and Trading
//!
//! Helper functions for common workflows combining time-series preprocessing
//! with optimization and statistical inference. Designed to work seamlessly
//! with Polaroid time-series operations and OptimizR algorithms.
//!
//! # Use Cases
//!
//! - Regime detection with HMM
//! - Rolling risk metrics (Hurst exponent, half-life)
//! - Trading strategy parameter optimization
//! - Feature engineering for financial data
//!
//! # Examples
//!
//! ```rust
//! use optimizr::timeseries_utils::*;
//!
//! // Prepare features for HMM regime detection
//! let prices = vec![100.0, 101.0, 99.5, 102.0];
//! let features = prepare_for_hmm(&prices, &[1, 5]);
//!
//! // Rolling Hurst exponent
//! let returns = vec![0.01, -0.02, 0.015, 0.005];
//! let rolling_hurst = rolling_hurst_exponent(&returns, 20);
//! ```

use crate::risk_metrics::{hurst_exponent, estimate_half_life};
use ndarray::Array1;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

/// Prepare time-series price data for HMM regime detection
///
/// Creates features from price series including:
/// - Returns (percent change)
/// - Lagged returns
/// - Log returns
/// - Volatility proxy (absolute returns)
///
/// # Arguments
///
/// * `prices` - Raw price series (e.g., stock prices, crypto prices)
/// * `lag_periods` - Lags to compute for returns (e.g., [1, 5, 20] for daily, weekly, monthly)
///
/// # Returns
///
/// Matrix where each row is a feature vector suitable for HMM training.
/// First row will have NaN values due to lagging.
///
/// # Example
///
/// ```
/// use optimizr::timeseries_utils::prepare_for_hmm;
///
/// let prices = vec![100.0, 101.0, 99.5, 102.0, 103.5];
/// let features = prepare_for_hmm(&prices, &[1, 2]);
///
/// // Features: [return_t, return_t-1, return_t-2, log_return, abs_return]
/// assert_eq!(features[0].len(), 5);
/// ```
pub fn prepare_for_hmm(prices: &[f64], lag_periods: &[usize]) -> Vec<Vec<f64>> {
    let n = prices.len();
    if n < 2 {
        return vec![];
    }

    // Calculate returns
    let mut returns = Vec::with_capacity(n - 1);
    let mut log_returns = Vec::with_capacity(n - 1);
    let mut abs_returns = Vec::with_capacity(n - 1);

    for i in 1..n {
        let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
        returns.push(ret);
        log_returns.push((prices[i] / prices[i - 1]).ln());
        abs_returns.push(ret.abs());
    }

    // Create feature matrix
    let max_lag = *lag_periods.iter().max().unwrap_or(&0);
    let mut features = Vec::new();

    for t in max_lag..returns.len() {
        let mut feature_vec = Vec::new();

        // Current return
        feature_vec.push(returns[t]);

        // Lagged returns
        for &lag in lag_periods {
            if t >= lag {
                feature_vec.push(returns[t - lag]);
            } else {
                feature_vec.push(f64::NAN);
            }
        }

        // Log return and volatility proxy
        feature_vec.push(log_returns[t]);
        feature_vec.push(abs_returns[t]);

        features.push(feature_vec);
    }

    features
}

/// Compute Hurst exponent in rolling windows
///
/// The Hurst exponent H indicates:
/// - H < 0.5: Mean-reverting series
/// - H = 0.5: Random walk (no memory)
/// - H > 0.5: Trending series (momentum)
///
/// # Arguments
///
/// * `returns` - Return series (percent changes or log returns)
/// * `window_size` - Rolling window size (e.g., 252 for 1 year of daily data)
///
/// # Returns
///
/// Vector of Hurst exponents, one per window. Length = returns.len() - window_size + 1
///
/// # Example
///
/// ```
/// use optimizr::timeseries_utils::rolling_hurst_exponent;
///
/// let returns = vec![0.01; 300]; // Synthetic data
/// let rolling_h = rolling_hurst_exponent(&returns, 252);
///
/// assert_eq!(rolling_h.len(), 300 - 252 + 1);
/// ```
pub fn rolling_hurst_exponent(returns: &[f64], window_size: usize) -> Vec<f64> {
    let n = returns.len();
    if n < window_size {
        return vec![];
    }

    let mut rolling_h = Vec::with_capacity(n - window_size + 1);

    for i in 0..=(n - window_size) {
        let window = &returns[i..i + window_size];
        let window_arr = Array1::from_vec(window.to_vec());
        // Use standard window sizes for R/S analysis
        let window_sizes = vec![8, 16, 32].into_iter().filter(|&w| w <= window_size / 4).collect::<Vec<_>>();
        let h = if window_sizes.is_empty() {
            0.5 // Default to random walk if window too small
        } else {
            hurst_exponent(&window_arr, &window_sizes)
                .map(|result| result.hurst_exponent)
                .unwrap_or(0.5)
        };
        rolling_h.push(h);
    }

    rolling_h
}

/// Compute half-life of mean reversion in rolling windows
///
/// Half-life is the expected time for a price series to revert halfway
/// to its mean. Useful for pairs trading and mean-reversion strategies.
///
/// # Arguments
///
/// * `prices` - Price series (not returns)
/// * `window_size` - Rolling window size
///
/// # Returns
///
/// Vector of half-life estimates in same time units as data
/// (e.g., days if daily prices)
///
/// # Example
///
/// ```
/// use optimizr::timeseries_utils::rolling_half_life;
///
/// let prices = vec![100.0; 300]; // Synthetic data
/// let rolling_hl = rolling_half_life(&prices, 100);
///
/// assert_eq!(rolling_hl.len(), 300 - 100 + 1);
/// ```
pub fn rolling_half_life(prices: &[f64], window_size: usize) -> Vec<f64> {
    let n = prices.len();
    if n < window_size {
        return vec![];
    }

    let mut rolling_hl = Vec::with_capacity(n - window_size + 1);

    for i in 0..=(n - window_size) {
        let window = &prices[i..i + window_size];
        let window_arr = Array1::from_vec(window.to_vec());
        let hl = estimate_half_life(&window_arr).unwrap_or(f64::INFINITY);
        rolling_hl.push(hl);
    }

    rolling_hl
}

/// Calculate basic time-series statistics
///
/// Returns summary statistics useful for risk analysis and diagnostics.
///
/// # Arguments
///
/// * `returns` - Return series
///
/// # Returns
///
/// Tuple of (mean, std_dev, skewness, kurtosis, sharpe_ratio)
///
/// # Example
///
/// ```
/// use optimizr::timeseries_utils::return_statistics;
///
/// let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01];
/// let (mean, std, skew, kurt, sharpe) = return_statistics(&returns);
/// ```
pub fn return_statistics(returns: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = returns.len() as f64;
    if returns.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Mean
    let mean = returns.iter().sum::<f64>() / n;

    // Standard deviation
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Skewness
    let skewness = if std_dev > 1e-10 {
        returns.iter().map(|r| ((r - mean) / std_dev).powi(3)).sum::<f64>() / n
    } else {
        0.0
    };

    // Excess kurtosis
    let kurtosis = if std_dev > 1e-10 {
        returns.iter().map(|r| ((r - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0
    } else {
        0.0
    };

    // Sharpe ratio (assuming 252 trading days, risk-free rate = 0)
    let sharpe_ratio = if std_dev > 1e-10 {
        (mean * 252.0_f64.sqrt()) / std_dev
    } else {
        0.0
    };

    (mean, std_dev, skewness, kurtosis, sharpe_ratio)
}

/// Create lagged features for machine learning
///
/// Useful for creating supervised learning datasets from time series.
///
/// # Arguments
///
/// * `series` - Input time series (prices, returns, etc.)
/// * `lags` - Vector of lag periods (e.g., [1, 2, 5, 10])
/// * `include_original` - Whether to include t=0 (current value)
///
/// # Returns
///
/// Matrix where each row is [t, t-lag1, t-lag2, ...] if include_original=true
/// or [t-lag1, t-lag2, ...] if false
///
/// # Example
///
/// ```
/// use optimizr::timeseries_utils::create_lagged_features;
///
/// let prices = vec![100.0, 101.0, 99.5, 102.0, 103.5, 104.0];
/// let features = create_lagged_features(&prices, &[1, 2], true);
///
/// // Each row: [price_t, price_t-1, price_t-2]
/// ```
pub fn create_lagged_features(series: &[f64], lags: &[usize], include_original: bool) -> Vec<Vec<f64>> {
    let n = series.len();
    let max_lag = *lags.iter().max().unwrap_or(&0);

    if n <= max_lag {
        return vec![];
    }

    let mut features = Vec::new();

    for t in max_lag..n {
        let mut feature_vec = Vec::new();

        if include_original {
            feature_vec.push(series[t]);
        }

        for &lag in lags {
            feature_vec.push(series[t - lag]);
        }

        features.push(feature_vec);
    }

    features
}

/// Compute rolling correlation between two series
///
/// Useful for pairs trading and correlation analysis.
///
/// # Arguments
///
/// * `series1` - First time series
/// * `series2` - Second time series (must be same length as series1)
/// * `window_size` - Rolling window size
///
/// # Returns
///
/// Vector of correlation coefficients
///
/// # Example
///
/// ```
/// use optimizr::timeseries_utils::rolling_correlation;
///
/// let spy = vec![100.0, 101.0, 99.5, 102.0, 103.5];
/// let qqq = vec![200.0, 202.0, 199.0, 204.0, 207.0];
/// let corr = rolling_correlation(&spy, &qqq, 3);
/// ```
pub fn rolling_correlation(series1: &[f64], series2: &[f64], window_size: usize) -> Vec<f64> {
    let n = series1.len();
    if n != series2.len() || n < window_size {
        return vec![];
    }

    let mut rolling_corr = Vec::with_capacity(n - window_size + 1);

    for i in 0..=(n - window_size) {
        let x = &series1[i..i + window_size];
        let y = &series2[i..i + window_size];

        // Compute correlation
        let mean_x = x.iter().sum::<f64>() / window_size as f64;
        let mean_y = y.iter().sum::<f64>() / window_size as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for j in 0..window_size {
            let dx = x[j] - mean_x;
            let dy = y[j] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let corr = if var_x > 1e-10 && var_y > 1e-10 {
            cov / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        };

        rolling_corr.push(corr);
    }

    rolling_corr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_for_hmm() {
        let prices = vec![100.0, 101.0, 99.5, 102.0, 103.5, 104.0];
        let features = prepare_for_hmm(&prices, &[1, 2]);

        assert!(!features.is_empty());
        // Each feature vector: [return, lag1, lag2, log_return, abs_return]
        assert_eq!(features[0].len(), 5);
    }

    #[test]
    fn test_rolling_hurst_exponent() {
        let returns = vec![0.01; 50];
        let rolling_h = rolling_hurst_exponent(&returns, 20);

        assert_eq!(rolling_h.len(), 50 - 20 + 1);
        // Constant series should have H close to 0.5
        for h in rolling_h {
            assert!((h - 0.5).abs() < 0.3); // Allow some numerical variation
        }
    }

    #[test]
    fn test_return_statistics() {
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01];
        let (mean, std, _skew, _kurt, _sharpe) = return_statistics(&returns);

        assert!((mean).abs() < 0.1); // Small mean
        assert!(std > 0.0); // Non-zero volatility
        // Skewness and kurtosis can vary widely
    }

    #[test]
    fn test_create_lagged_features() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let features = create_lagged_features(&series, &[1, 2], true);

        assert_eq!(features.len(), 3); // 5 - 2 = 3 valid rows
        assert_eq!(features[0], vec![3.0, 2.0, 1.0]); // t=2: [val_t, val_t-1, val_t-2]
    }

    #[test]
    fn test_rolling_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation
        let corr = rolling_correlation(&x, &y, 3);

        assert_eq!(corr.len(), 3);
        for c in corr {
            assert!((c - 1.0).abs() < 1e-6); // Perfect positive correlation
        }
    }
}
