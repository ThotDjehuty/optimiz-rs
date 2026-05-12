//! Python bindings for the `volterra` module.
//!
//! Exposes the four core primitives:
//!
//! * `solve_fractional_ode`  -- Caputo fractional ODE Adams scheme.
//! * `geometric_grid_lift`   -- multi-exponential approximation of a kernel.
//! * `solve_volterra`        -- generic second-kind Volterra equation.
//! * `fourier_invert`        -- characteristic function -> density.
//!
//! Python callables are accepted as `&Bound<'_, PyAny>` and called
//! through `.call1(...)?.extract::<...>()?`. Errors of type
//! [`crate::core::OptimizrError`] are mapped to [`PyValueError`].
//!
//! The underlying Rust solvers require an immutable [`Fn`] callback,
//! so the first error raised by the Python callable is captured via a
//! [`RefCell`] and re-raised after the solver returns.

use std::cell::RefCell;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::fourier_inversion::fourier_invert as rs_fourier_invert;
use super::fractional_riccati::solve_fractional_ode as rs_solve_fractional_ode;
use super::markovian_lift::geometric_grid_lift as rs_geometric_grid_lift;
use super::volterra_solver::solve_volterra as rs_solve_volterra;

/// Solve the Caputo fractional ODE `D^alpha h = rhs(t, h)` on `[0, t_horizon]`.
///
/// `rhs` is a Python callable `(t: float, h: float) -> float`.
///
/// Returns a dict `{"t_grid": [...], "h": [...]}`.
#[pyfunction]
#[pyo3(signature = (h0, alpha, t_horizon, n_steps, rhs))]
fn solve_fractional_ode(
    py: Python<'_>,
    h0: f64,
    alpha: f64,
    t_horizon: f64,
    n_steps: usize,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let cb_err: RefCell<Option<PyErr>> = RefCell::new(None);
    let rust_rhs = |t: f64, h: f64| -> f64 {
        if cb_err.borrow().is_some() {
            return 0.0;
        }
        match rhs.call1((t, h)).and_then(|v| v.extract::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                *cb_err.borrow_mut() = Some(e);
                0.0
            }
        }
    };
    let result = rs_solve_fractional_ode(h0, alpha, t_horizon, n_steps, rust_rhs)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    if let Some(e) = cb_err.into_inner() {
        return Err(e);
    }
    let dict = PyDict::new_bound(py);
    dict.set_item("t_grid", result.t_grid)?;
    dict.set_item("h", result.h)?;
    Ok(dict.into())
}

/// Build a Markovian lift `K(t) ~= sum c_j exp(-gamma_j t)` on a
/// geometric grid of rates with non-negative least-squares weights.
///
/// `kernel` is a Python callable `(t: float) -> float`.
///
/// Returns a dict `{"gammas": [...], "weights": [...]}`.
#[pyfunction]
#[pyo3(signature = (kernel, t_samples, n_factors, gamma_min, gamma_max, nnls_iter=5000))]
fn geometric_grid_lift(
    py: Python<'_>,
    kernel: &Bound<'_, PyAny>,
    t_samples: Vec<f64>,
    n_factors: usize,
    gamma_min: f64,
    gamma_max: f64,
    nnls_iter: usize,
) -> PyResult<PyObject> {
    let cb_err: RefCell<Option<PyErr>> = RefCell::new(None);
    let rust_kernel = |t: f64| -> f64 {
        if cb_err.borrow().is_some() {
            return 0.0;
        }
        match kernel.call1((t,)).and_then(|v| v.extract::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                *cb_err.borrow_mut() = Some(e);
                0.0
            }
        }
    };
    let lift = rs_geometric_grid_lift(
        rust_kernel,
        &t_samples,
        n_factors,
        gamma_min,
        gamma_max,
        nnls_iter,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    if let Some(e) = cb_err.into_inner() {
        return Err(e);
    }
    let dict = PyDict::new_bound(py);
    dict.set_item("gammas", lift.gammas)?;
    dict.set_item("weights", lift.weights)?;
    Ok(dict.into())
}

/// Solve the scalar second-kind Volterra equation
/// `y(t) = g(t) + int_0^t K(t - s, y(s)) ds` by trapezoidal product
/// integration on `n_steps + 1` equispaced nodes.
///
/// `g` is a Python callable `(t: float) -> float`.
/// `kernel` is a Python callable `(dt: float, y: float) -> float`.
///
/// Returns a dict `{"t_grid": [...], "y": [...]}`.
#[pyfunction]
#[pyo3(signature = (g, kernel, t_horizon, n_steps, fixed_point_iter=50, fixed_point_tol=1e-12))]
fn solve_volterra(
    py: Python<'_>,
    g: &Bound<'_, PyAny>,
    kernel: &Bound<'_, PyAny>,
    t_horizon: f64,
    n_steps: usize,
    fixed_point_iter: usize,
    fixed_point_tol: f64,
) -> PyResult<PyObject> {
    let cb_err: RefCell<Option<PyErr>> = RefCell::new(None);
    let rust_g = |t: f64| -> f64 {
        if cb_err.borrow().is_some() {
            return 0.0;
        }
        match g.call1((t,)).and_then(|v| v.extract::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                *cb_err.borrow_mut() = Some(e);
                0.0
            }
        }
    };
    let rust_k = |dt: f64, y: f64| -> f64 {
        if cb_err.borrow().is_some() {
            return 0.0;
        }
        match kernel.call1((dt, y)).and_then(|v| v.extract::<f64>()) {
            Ok(v) => v,
            Err(e) => {
                *cb_err.borrow_mut() = Some(e);
                0.0
            }
        }
    };
    let result = rs_solve_volterra(
        rust_g,
        rust_k,
        t_horizon,
        n_steps,
        fixed_point_iter,
        fixed_point_tol,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    if let Some(e) = cb_err.into_inner() {
        return Err(e);
    }
    let dict = PyDict::new_bound(py);
    dict.set_item("t_grid", result.t_grid)?;
    dict.set_item("y", result.y)?;
    Ok(dict.into())
}

/// Recover a probability density on `x_grid` from a characteristic
/// function `phi`.
///
/// `phi` is a Python callable `(u: float) -> (re: float, im: float)`.
///
/// Returns a dict `{"x_grid": [...], "density": [...]}`.
#[pyfunction]
#[pyo3(signature = (phi, x_grid, u_max, n_u))]
fn fourier_invert(
    py: Python<'_>,
    phi: &Bound<'_, PyAny>,
    x_grid: Vec<f64>,
    u_max: f64,
    n_u: usize,
) -> PyResult<PyObject> {
    let cb_err: RefCell<Option<PyErr>> = RefCell::new(None);
    let rust_phi = |u: f64| -> (f64, f64) {
        if cb_err.borrow().is_some() {
            return (0.0, 0.0);
        }
        match phi.call1((u,)).and_then(|v| v.extract::<(f64, f64)>()) {
            Ok(v) => v,
            Err(e) => {
                *cb_err.borrow_mut() = Some(e);
                (0.0, 0.0)
            }
        }
    };
    let result = rs_fourier_invert(rust_phi, &x_grid, u_max, n_u)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    if let Some(e) = cb_err.into_inner() {
        return Err(e);
    }
    let dict = PyDict::new_bound(py);
    dict.set_item("x_grid", result.x_grid)?;
    dict.set_item("density", result.density)?;
    Ok(dict.into())
}

/// Register all Volterra-related functions with the Python module.
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_fractional_ode, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_grid_lift, m)?)?;
    m.add_function(wrap_pyfunction!(solve_volterra, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_invert, m)?)?;
    Ok(())
}
