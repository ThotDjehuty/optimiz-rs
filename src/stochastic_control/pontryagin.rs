//! Pontryagin maximum principle — 1-D LQR shooting solver
//! ========================================================
//!
//! Closed-form Pontryagin solution for the 1-D controlled linear-quadratic
//! regulator
//!
//! ```text
//! dx/dt = a x + b u,    x(0) = x_0
//! J = ∫_0^T (q x² + r u²) dt + s_T x(T)²
//! ```
//!
//! The Hamiltonian `H = p (a x + b u) + q x² + r u²` is minimised at
//! `u* = -b p / (2 r)`, giving the costate ODE `-dp/dt = 2 q x + a p`.
//! We integrate the resulting Riccati equation `dP/dt = -2 a P + b² P² / r - 2 q`
//! backwards from `P(T) = s_T` and recover `u*(t) = -(b/r) P(t) x(t)`.

use crate::core::{OptimizrError, Result};
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct PontryaginConfig {
    pub a: f64,
    pub b: f64,
    pub q: f64,
    pub r: f64,
    pub s_terminal: f64,
    pub x0: f64,
    pub t_horizon: f64,
    pub n_steps: usize,
}

#[derive(Clone, Debug)]
pub struct PontryaginResult {
    pub time_grid: Array1<f64>,
    pub state: Array1<f64>,
    pub control: Array1<f64>,
    pub riccati: Array1<f64>,
    pub cost: f64,
}

pub fn solve_pontryagin_lqr(cfg: &PontryaginConfig) -> Result<PontryaginResult> {
    if cfg.r <= 0.0 {
        return Err(OptimizrError::InvalidParameter("r must be > 0".into()));
    }
    if cfg.n_steps == 0 || cfg.t_horizon <= 0.0 {
        return Err(OptimizrError::InvalidParameter("n_steps>0 and T>0".into()));
    }
    let n = cfg.n_steps;
    let dt = cfg.t_horizon / n as f64;
    let time_grid: Array1<f64> = Array1::from_iter((0..=n).map(|k| k as f64 * dt));

    // Backward Riccati (explicit Euler).
    let mut p = vec![0.0f64; n + 1];
    p[n] = cfg.s_terminal;
    for k in (0..n).rev() {
        let pn = p[k + 1];
        let dp = -2.0 * cfg.a * pn + cfg.b * cfg.b * pn * pn / cfg.r - 2.0 * cfg.q;
        p[k] = pn - dt * dp;
    }

    // Forward state integration with optimal feedback u* = -(b/r) P x.
    let mut x = Array1::<f64>::zeros(n + 1);
    let mut u = Array1::<f64>::zeros(n);
    x[0] = cfg.x0;
    let mut cost = 0.0;
    for k in 0..n {
        let u_k = -(cfg.b / cfg.r) * p[k] * x[k];
        u[k] = u_k;
        let dx = cfg.a * x[k] + cfg.b * u_k;
        x[k + 1] = x[k] + dt * dx;
        cost += dt * (cfg.q * x[k] * x[k] + cfg.r * u_k * u_k);
    }
    cost += cfg.s_terminal * x[n] * x[n];

    Ok(PontryaginResult {
        time_grid,
        state: x,
        control: u,
        riccati: Array1::from(p),
        cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed-form check: with a = 0, b = 1, q = 0, r = 1, s_T > 0, T = 1,
    /// the Riccati ODE becomes dP/dt = P²; solution P(t) = s_T / (1 + s_T (T-t)).
    #[test]
    fn riccati_matches_closed_form() {
        let cfg = PontryaginConfig {
            a: 0.0,
            b: 1.0,
            q: 0.0,
            r: 1.0,
            s_terminal: 1.0,
            x0: 1.0,
            t_horizon: 1.0,
            n_steps: 2000,
        };
        let res = solve_pontryagin_lqr(&cfg).unwrap();
        let analytic_p0 = 1.0 / (1.0 + 1.0 * 1.0); // 0.5
        let err = (res.riccati[0] - analytic_p0).abs();
        assert!(err < 5e-3, "P(0) = {}, analytic = {}", res.riccati[0], analytic_p0);
        // Optimal cost analytic = P(0) * x0²
        let analytic_cost = analytic_p0;
        assert!((res.cost - analytic_cost).abs() < 5e-3);
    }

    #[test]
    fn rejects_zero_control_weight() {
        let cfg = PontryaginConfig {
            a: 0.0, b: 1.0, q: 1.0, r: 0.0, s_terminal: 0.0,
            x0: 1.0, t_horizon: 1.0, n_steps: 10,
        };
        assert!(solve_pontryagin_lqr(&cfg).is_err());
    }
}
