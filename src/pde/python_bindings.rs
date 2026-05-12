//! Python bindings for the PDE module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::elliptic_fd::{solve_poisson_2d, EllipticFdConfig};
use super::fokker_planck::{solve_fokker_planck_1d, FokkerPlanckConfig};
use super::hjb_multid::{solve_hjb_multid, HjbMultidConfig};

/// Forward Fokker–Planck on `[x_min, x_max] × [0, T]` with a constant drift
/// `mu` and constant diffusion variance `sigma_sq` and a centred Gaussian
/// initial density of standard deviation `init_sigma`.
#[pyfunction]
#[pyo3(signature = (mu, sigma_sq, init_sigma, x_min, x_max, n_x, t_horizon, n_t))]
fn fokker_planck_constant(
    py: Python<'_>,
    mu: f64,
    sigma_sq: f64,
    init_sigma: f64,
    x_min: f64,
    x_max: f64,
    n_x: usize,
    t_horizon: f64,
    n_t: usize,
) -> PyResult<PyObject> {
    let cfg = FokkerPlanckConfig { n_x, x_min, x_max, n_t, t_horizon };
    let res = solve_fokker_planck_1d(
        |_| mu,
        |_| sigma_sq,
        |x| (-(x * x) / (2.0 * init_sigma * init_sigma)).exp(),
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("x_grid", res.x_grid.to_vec())?;
    dict.set_item("time_grid", res.time_grid.to_vec())?;
    dict.set_item("density", res.density.clone())?;
    dict.set_item("n_x", n_x)?;
    dict.set_item("n_t", n_t)?;
    Ok(dict.into())
}

/// 2-D Poisson `-Δu = f` SOR solver.  `rhs_grid` is a flat row-major
/// `n_x × n_y` array of pre-evaluated source values.  Boundary `u = 0`.
#[pyfunction]
#[pyo3(signature = (rhs_grid, n_x, n_y, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, omega=1.7, max_iterations=20000, tolerance=1e-6))]
fn poisson_2d_zero_boundary(
    py: Python<'_>,
    rhs_grid: Vec<f64>,
    n_x: usize,
    n_y: usize,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    omega: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyObject> {
    if rhs_grid.len() != n_x * n_y {
        return Err(PyValueError::new_err("rhs_grid length must equal n_x*n_y"));
    }
    let cfg = EllipticFdConfig {
        n_x, n_y, x_min, x_max, y_min, y_max,
        max_iterations, tolerance, omega,
    };
    let dx = (x_max - x_min) / (n_x - 1) as f64;
    let dy = (y_max - y_min) / (n_y - 1) as f64;
    let res = solve_poisson_2d(
        |x, y| {
            let i = ((x - x_min) / dx).round() as usize;
            let j = ((y - y_min) / dy).round() as usize;
            let i = i.min(n_x - 1);
            let j = j.min(n_y - 1);
            rhs_grid[i * n_y + j]
        },
        |_, _| 0.0,
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("u", res.u.clone())?;
    dict.set_item("n_x", n_x)?;
    dict.set_item("n_y", n_y)?;
    dict.set_item("iterations", res.iterations)?;
    dict.set_item("residual", res.residual)?;
    Ok(dict.into())
}

/// 2-D HJB on `[x_min, x_max]²` with quadratic Hamiltonian `½ |∇v|²`,
/// constant isotropic diffusion `sigma_sq`, terminal condition
/// `g(x, y) = ½ (x² + y²)`.
#[pyfunction]
#[pyo3(signature = (n_per_dim, x_min, x_max, n_t, t_horizon, sigma_sq))]
fn hjb_quadratic_2d(
    py: Python<'_>,
    n_per_dim: usize,
    x_min: f64,
    x_max: f64,
    n_t: usize,
    t_horizon: f64,
    sigma_sq: f64,
) -> PyResult<PyObject> {
    let cfg = HjbMultidConfig { dim: 2, n_per_dim, x_min, x_max, n_t, t_horizon, sigma_sq };
    let res = solve_hjb_multid(
        |_x, grad| 0.5 * grad.iter().map(|g| g * g).sum::<f64>(),
        |x| 0.5 * (x[0] * x[0] + x[1] * x[1]),
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("value", res.value.clone())?;
    dict.set_item("n_per_dim", n_per_dim)?;
    dict.set_item("axis", res.grid_axes[0].to_vec())?;
    Ok(dict.into())
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fokker_planck_constant, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_2d_zero_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(hjb_quadratic_2d, m)?)?;
    Ok(())
}
