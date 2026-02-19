//! Python bindings for point processes module
//!
//! Provides Python access to:
//! - Hawkes process simulation (univariate and bivariate)
//! - Fractional Brownian motion simulation
//! - Mixed fractional Brownian motion
//! - Hurst exponent estimation
//! - Mittag-Leffler special functions

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};

use super::kernels::{ExcitationKernel, PowerLawKernel, ExponentialKernel};
use super::hawkes::{HawkesProcess, BivariateHawkes};
use super::mittag_leffler::{mittag_leffler, f_alpha_lambda};
use super::mixed_fbm::{FractionalBM, MixedFractionalBM};

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
    
    Ok(())
}
