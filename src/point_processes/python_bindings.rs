//! Python bindings for point processes module

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};

use super::kernels::{ExcitationKernel, PowerLawKernel, ExponentialKernel};
use super::hawkes::{HawkesProcess, HawkesProcessConfig, BivariateHawkes};
use super::mittag_leffler::{mittag_leffler, f_alpha_lambda};
use super::mixed_fbm::{FractionalBM, MixedFractionalBM};
use super::order_flow::{OrderFlowAnalyzer, UnifiedTheoryParams, MarketImpact};

/// Simulate a univariate Hawkes process
///
/// # Arguments
/// * `baseline` - Baseline intensity ν
/// * `alpha` - Kernel amplitude (for exponential) or K₀ (for power-law)
/// * `beta` - Decay rate (for exponential) or α₀ (for power-law)
/// * `t_max` - Maximum simulation time
/// * `kernel_type` - "exponential" or "power_law"
/// * `seed` - Optional random seed
///
/// # Returns
/// Array of event times
#[pyfunction]
#[pyo3(signature = (baseline, alpha, beta, t_max, kernel_type="exponential", seed=None))]
pub fn simulate_hawkes<'py>(
    py: Python<'py>,
    baseline: f64,
    alpha: f64,
    beta: f64,
    t_max: f64,
    kernel_type: &str,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let events = match kernel_type {
        "exponential" => {
            let kernel = ExponentialKernel::new(alpha, beta);
            let mut process = HawkesProcess::new(baseline, kernel);
            process.simulate(t_max, seed)
        }
        "power_law" => {
            let kernel = PowerLawKernel::new(beta, alpha);  // beta = α₀, alpha = K₀
            let mut process = HawkesProcess::new(baseline, kernel);
            process.simulate(t_max, seed)
        }
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown kernel type: {}. Use 'exponential' or 'power_law'", kernel_type)
        )),
    };

    Ok(PyArray1::from_vec_bound(py, events))
}

/// Simulate a bivariate Hawkes process for buy/sell reaction flow
///
/// # Arguments
/// * `core_buy_times` - Core buy order times (driver process)
/// * `core_sell_times` - Core sell order times (driver process)
/// * `phi1_alpha` - Same-side kernel amplitude
/// * `phi1_beta` - Same-side kernel decay
/// * `phi2_alpha` - Cross-side kernel amplitude
/// * `phi2_beta` - Cross-side kernel decay
/// * `t_max` - Maximum simulation time
/// * `seed` - Optional random seed
///
/// # Returns
/// Tuple of (buy_times, sell_times)
#[pyfunction]
#[pyo3(signature = (core_buy_times, core_sell_times, phi1_alpha, phi1_beta, phi2_alpha, phi2_beta, t_max, seed=None))]
pub fn simulate_bivariate_hawkes<'py>(
    py: Python<'py>,
    core_buy_times: PyReadonlyArray1<f64>,
    core_sell_times: PyReadonlyArray1<f64>,
    phi1_alpha: f64,
    phi1_beta: f64,
    phi2_alpha: f64,
    phi2_beta: f64,
    t_max: f64,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let core_buys = core_buy_times.as_slice()?;
    let core_sells = core_sell_times.as_slice()?;

    let phi_1 = ExponentialKernel::new(phi1_alpha, phi1_beta);
    let phi_2 = ExponentialKernel::new(phi2_alpha, phi2_beta);

    let mut process = BivariateHawkes::new(phi_1, phi_2);
    let (buys, sells) = process.simulate_driven(core_buys, core_sells, t_max, seed);

    Ok((
        PyArray1::from_vec_bound(py, buys),
        PyArray1::from_vec_bound(py, sells),
    ))
}

/// Compute Mittag-Leffler function E_{α,β}(z)
#[pyfunction]
#[pyo3(signature = (alpha, beta, z))]
pub fn mittag_leffler_py(alpha: f64, beta: f64, z: f64) -> PyResult<f64> {
    Ok(mittag_leffler(alpha, beta, z))
}

/// Compute f_{α₀,λ₀}(x) function from unified theory scaling limits
#[pyfunction]
#[pyo3(signature = (alpha0, lambda0, x))]
pub fn f_alpha_lambda_py(alpha0: f64, lambda0: f64, x: f64) -> PyResult<f64> {
    Ok(f_alpha_lambda(alpha0, lambda0, x))
}

/// Simulate fractional Brownian motion
///
/// # Arguments
/// * `hurst` - Hurst parameter H ∈ (0, 1)
/// * `n` - Number of time steps
/// * `dt` - Time step size
/// * `seed` - Optional random seed
#[pyfunction]
#[pyo3(signature = (hurst, n, dt=1.0, seed=None))]
pub fn simulate_fbm<'py>(
    py: Python<'py>,
    hurst: f64,
    n: usize,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut fbm = FractionalBM::new(hurst);
    let values = fbm.simulate_hosking(n, dt, seed);
    Ok(PyArray1::from_vec_bound(py, values))
}

/// Simulate mixed fractional Brownian motion
///
/// # Arguments
/// * `a` - Coefficient for standard BM component
/// * `b` - Coefficient for fBM component
/// * `hurst` - Hurst parameter of fBM component
/// * `n` - Number of time steps
/// * `dt` - Time step size
/// * `seed` - Optional random seed
#[pyfunction]
#[pyo3(signature = (a, b, hurst, n, dt=1.0, seed=None))]
pub fn simulate_mixed_fbm<'py>(
    py: Python<'py>,
    a: f64,
    b: f64,
    hurst: f64,
    n: usize,
    dt: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut mfbm = MixedFractionalBM::new(a, b, hurst);
    let values = mfbm.simulate(n, dt, seed);
    Ok(PyArray1::from_vec_bound(py, values))
}

/// Estimate Hurst exponent using R/S analysis
#[pyfunction]
pub fn estimate_hurst<'py>(
    _py: Python<'py>,
    data: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let slice = data.as_slice()?;
    Ok(FractionalBM::estimate_hurst(slice))
}

/// Compute scale-dependent Hurst exponents (for mfBM identification)
#[pyfunction]
#[pyo3(signature = (data, scales=None))]
pub fn scale_dependent_hurst<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f64>,
    scales: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = data.as_slice()?;
    let scales = scales.unwrap_or_else(|| vec![10, 50, 100, 500, 1000, 2000, 5000]);
    
    let results = MixedFractionalBM::scale_dependent_hurst(slice, &scales);
    
    let dict = PyDict::new_bound(py);
    for (scale, h) in results {
        dict.set_item(scale, h)?;
    }
    Ok(dict)
}

/// Analyze order flow data using unified theory framework
///
/// # Arguments
/// * `flow` - Signed order flow data (positive = buy, negative = sell)
///
/// # Returns
/// Dictionary with metrics:
/// - h0: Estimated H₀ (core flow persistence)
/// - h_signed_fbm: Hurst under pure fBM assumption
/// - h_signed_mfbm: Hurst under mfBM assumption
/// - h_unsigned: Hurst of unsigned volume
/// - h_volatility: Implied volatility Hurst (2H₀ - 3/2)
/// - impact_exponent: Implied market impact exponent (2 - 2H₀)
/// - acf_1: First-order autocorrelation
/// - scale_hurst: Scale-dependent Hurst estimates
#[pyfunction]
pub fn analyze_order_flow<'py>(
    py: Python<'py>,
    flow: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let slice = flow.as_slice()?;
    
    let analyzer = OrderFlowAnalyzer::new();
    let metrics = analyzer.analyze_signed_flow(slice);
    
    let dict = PyDict::new_bound(py);
    dict.set_item("h0", metrics.h0)?;
    dict.set_item("h_signed_fbm", metrics.h_signed_fbm)?;
    dict.set_item("h_signed_mfbm", metrics.h_signed_mfbm)?;
    dict.set_item("h_unsigned", metrics.h_unsigned)?;
    dict.set_item("h_volatility", metrics.h_volatility)?;
    dict.set_item("impact_exponent", metrics.impact_exponent)?;
    dict.set_item("total_signed", metrics.total_signed)?;
    dict.set_item("total_unsigned", metrics.total_unsigned)?;
    dict.set_item("acf_1", metrics.acf_1)?;
    
    // Scale-dependent Hurst as nested dict
    let scale_dict = PyDict::new_bound(py);
    for (scale, h) in metrics.scale_hurst {
        scale_dict.set_item(scale, h)?;
    }
    dict.set_item("scale_hurst", scale_dict)?;
    
    Ok(dict)
}

/// Get unified theory derived quantities from H₀
///
/// # Arguments
/// * `h0` - Hurst index of signed order flow (typically ~0.75)
///
/// # Returns
/// Dictionary with derived parameters:
/// - h0: Input H₀
/// - alpha0: Tail exponent α₀ = H₀/2
/// - h_volume: Volume Hurst H₁ = H₀ - 0.5
/// - h_volatility: Volatility Hurst = 2H₀ - 1.5
/// - impact_exponent: Market impact exponent δ = 2 - 2H₀
/// - is_semimartingale: Whether mfBM is a semimartingale (H₀ > 3/4)
#[pyfunction]
pub fn unified_theory_params<'py>(
    py: Python<'py>,
    h0: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let params = UnifiedTheoryParams::new(h0);
    
    let dict = PyDict::new_bound(py);
    dict.set_item("h0", params.h0)?;
    dict.set_item("alpha0", params.alpha0())?;
    dict.set_item("h_volume", params.volume_hurst())?;
    dict.set_item("h_volatility", params.volatility_hurst())?;
    dict.set_item("impact_exponent", params.impact_exponent())?;
    dict.set_item("is_semimartingale", params.is_semimartingale())?;
    
    Ok(dict)
}

/// Compute market impact for given order size
///
/// Impact(Q) = scale * |Q|^δ * sign(Q)
/// where δ = 2 - 2*H₀
#[pyfunction]
#[pyo3(signature = (q, h0=0.75, scale=1.0))]
pub fn market_impact<'py>(
    _py: Python<'py>,
    q: f64,
    h0: f64,
    scale: f64,
) -> PyResult<f64> {
    let impact = MarketImpact::from_h0(h0, scale);
    Ok(impact.impact(q))
}

/// Register all point process functions to PyO3 module
pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Hawkes processes
    m.add_function(wrap_pyfunction!(simulate_hawkes, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_bivariate_hawkes, m)?)?;
    
    // Special functions
    m.add_function(wrap_pyfunction!(mittag_leffler_py, m)?)?;
    m.add_function(wrap_pyfunction!(f_alpha_lambda_py, m)?)?;
    
    // Fractional Brownian motion
    m.add_function(wrap_pyfunction!(simulate_fbm, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_mixed_fbm, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hurst, m)?)?;
    m.add_function(wrap_pyfunction!(scale_dependent_hurst, m)?)?;
    
    // Order flow analysis
    m.add_function(wrap_pyfunction!(analyze_order_flow, m)?)?;
    m.add_function(wrap_pyfunction!(unified_theory_params, m)?)?;
    m.add_function(wrap_pyfunction!(market_impact, m)?)?;
    
    Ok(())
}
