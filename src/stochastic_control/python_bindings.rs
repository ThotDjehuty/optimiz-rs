//! Python bindings for the stochastic_control module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::optimal_switching::{solve_optimal_switching, SwitchingConfig};
use super::pontryagin::{solve_pontryagin_lqr, PontryaginConfig};
use super::two_sided_intensity_control::{
    optimal_two_sided_intensities, TwoSidedConfig,
};

#[pyfunction]
#[pyo3(signature = (stage_reward_table, terminal_payoff, switching_cost, n_modes, n_steps))]
fn optimal_switching_dp(
    py: Python<'_>,
    stage_reward_table: Vec<f64>, // length n_steps * n_modes (row-major: [k, i])
    terminal_payoff: Vec<f64>,    // length n_modes
    switching_cost: Vec<f64>,     // length n_modes * n_modes
    n_modes: usize,
    n_steps: usize,
) -> PyResult<PyObject> {
    if stage_reward_table.len() != n_steps * n_modes {
        return Err(PyValueError::new_err("stage_reward_table size mismatch"));
    }
    if terminal_payoff.len() != n_modes {
        return Err(PyValueError::new_err("terminal_payoff size mismatch"));
    }
    let cfg = SwitchingConfig { n_modes, n_steps };
    let res = solve_optimal_switching(
        |k, i| stage_reward_table[k * n_modes + i],
        |i| terminal_payoff[i],
        &switching_cost,
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("value", res.value.clone())?;
    dict.set_item("policy", res.policy.clone())?;
    dict.set_item("n_modes", n_modes)?;
    dict.set_item("n_steps", n_steps)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (a, b, q, r, s_terminal, x0, t_horizon, n_steps))]
fn pontryagin_lqr(
    py: Python<'_>,
    a: f64, b: f64, q: f64, r: f64, s_terminal: f64,
    x0: f64, t_horizon: f64, n_steps: usize,
) -> PyResult<PyObject> {
    let cfg = PontryaginConfig { a, b, q, r, s_terminal, x0, t_horizon, n_steps };
    let res = solve_pontryagin_lqr(&cfg).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("time_grid", res.time_grid.to_vec())?;
    dict.set_item("state", res.state.to_vec())?;
    dict.set_item("control", res.control.to_vec())?;
    dict.set_item("riccati", res.riccati.to_vec())?;
    dict.set_item("cost", res.cost)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (alpha_plus, alpha_minus, kappa_plus, kappa_minus, delta_v_plus, delta_v_minus))]
fn two_sided_intensities(
    py: Python<'_>,
    alpha_plus: f64, alpha_minus: f64,
    kappa_plus: f64, kappa_minus: f64,
    delta_v_plus: f64, delta_v_minus: f64,
) -> PyResult<PyObject> {
    let cfg = TwoSidedConfig { alpha_plus, alpha_minus, kappa_plus, kappa_minus };
    let res = optimal_two_sided_intensities(&cfg, delta_v_plus, delta_v_minus)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("lambda_plus", res.lambda_plus)?;
    dict.set_item("lambda_minus", res.lambda_minus)?;
    dict.set_item("reward_density", res.reward_density)?;
    Ok(dict.into())
}

pub fn register_python_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimal_switching_dp, m)?)?;
    m.add_function(wrap_pyfunction!(pontryagin_lqr, m)?)?;
    m.add_function(wrap_pyfunction!(two_sided_intensities, m)?)?;
    Ok(())
}
