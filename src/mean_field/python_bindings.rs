//! Python bindings for Mean Field Games module

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use super::{MFGConfig, forward_backward_fixed_point, Grid};

/// Python-facing configuration for MFG solver
#[cfg_attr(feature = "python-bindings", pyclass)]
#[derive(Clone, Debug)]
pub struct MFGConfigPy {
    pub nx: usize,
    pub nt: usize,
    pub x_min: f64,
    pub x_max: f64,
    #[allow(non_snake_case)]
    pub T: f64,
    pub nu: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub alpha: f64,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl MFGConfigPy {
    #[new]
    #[allow(non_snake_case)]
    #[pyo3(signature = (nx=100, nt=100, x_min=0.0, x_max=1.0, T=1.0, nu=0.01, max_iter=50, tol=1e-5, alpha=0.5))]
    fn new(
        nx: usize,
        nt: usize,
        x_min: f64,
        x_max: f64,
        T: f64,
        nu: f64,
        max_iter: usize,
        tol: f64,
        alpha: f64,
    ) -> Self {
        Self {
            nx,
            nt,
            x_min,
            x_max,
            T,
            nu,
            max_iter,
            tol,
            alpha,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MFGConfig(nx={}, nt={}, domain=[{:.2},{:.2}], T={:.2}, nu={:.4}, max_iter={}, tol={:.2e}, alpha={:.2})",
            self.nx, self.nt, self.x_min, self.x_max,
            self.T, self.nu, self.max_iter, self.tol, self.alpha
        )
    }
}

impl MFGConfigPy {
    /// Convert to internal MFGConfig type
    pub fn to_mfg_config(&self) -> MFGConfig {
        MFGConfig {
            dim: 1,
            nx: self.nx,
            nt: self.nt,
            domain: (self.x_min, self.x_max),
            time_horizon: self.T,
            viscosity: self.nu,
            tolerance: self.tol,
            max_iterations: self.max_iter,
            relaxation: self.alpha,
        }
    }
}

/// Solve 1D Mean Field Game using forward-backward iteration (Rust implementation)
///
/// # Arguments
///
/// * `m0` - Initial distribution (nx,)
/// * `u_terminal` - Terminal cost (nx,)
/// * `config` - MFG configuration
/// * `lambda_congestion` - Congestion penalty coefficient
///
/// # Returns
///
/// Tuple of (u, m, iterations):
/// * `u` - Value function (nx, nt)
/// * `m` - Distribution (nx, nt)
/// * `iterations` - Number of iterations to convergence
#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(name = "solve_mfg_1d_rust")]
fn solve_mfg_1d_rust_py<'py>(
    py: Python<'py>,
    m0: PyReadonlyArray2<f64>,
    u_terminal: PyReadonlyArray2<f64>,
    config: &MFGConfigPy,
    lambda_congestion: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, usize)> {
    // Convert numpy arrays to ndarray - m0 should be (nx, 1)
    let m0_array = m0.as_array();
    let m0_vec = m0_array.column(0).to_owned();
    
    let u_terminal_array = u_terminal.as_array();
    let u_terminal_vec = u_terminal_array.column(0).to_owned();
    
    // Build MFGConfig from Python config
    let mfg_config = config.to_mfg_config();
    let grid = Grid::new(config.nx, config.nt, (config.x_min, config.x_max), config.T);
    
    // Define problem functions
    let hamiltonian = |_x: f64, _m: f64, p: f64| 0.5 * p * p; // Quadratic H(p) = ½p²
    let running_cost = move |_x: f64, m: f64| lambda_congestion * m; // Congestion cost
    let terminal_cost = |x: f64, _m: f64| u_terminal_vec[((x - config.x_min) / grid.dx) as usize]; // From array
    
    // Solve MFG
    let (u, m, iterations) = forward_backward_fixed_point(
        &mfg_config,
        hamiltonian,
        running_cost,
        terminal_cost,
        &m0_vec,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("MFG solver failed: {}", e)))?;

    // Convert back to numpy arrays
    Ok((u.to_pyarray_bound(py), m.to_pyarray_bound(py), iterations))
}

/// Register Python bindings for mean_field module
#[cfg(feature = "python-bindings")]
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<MFGConfigPy>()?;
    m.add_function(wrap_pyfunction!(solve_mfg_1d_rust_py, m)?)?;
    Ok(())
}
