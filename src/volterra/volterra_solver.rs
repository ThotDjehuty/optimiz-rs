//! Generic second-kind Volterra integral equation solver.
//!
//! Solves
//!
//! ```text
//!     y(t) = g(t) + int_0^t K(t - s, y(s)) ds,        t in [0, T],
//! ```
//!
//! by a product-trapezoidal quadrature on a uniform grid:
//!
//! ```text
//!     y_n = g_n + dt * [ K(t_n, y_0) / 2
//!                       + sum_{k=1}^{n-1} K(t_n - t_k, y_k)
//!                       + K(0, y_n) / 2 ].
//! ```
//!
//! The implicit equation in `y_n` is solved by fixed-point iteration
//! (good behaviour for Lipschitz `K(., .)` with small `dt`).

use crate::core::{OptimizrError, Result};

/// Result of a Volterra integration on a uniform grid.
#[derive(Debug, Clone)]
pub struct VolterraResult {
    pub t_grid: Vec<f64>,
    pub y: Vec<f64>,
}

/// Solve a scalar second-kind Volterra equation by trapezoidal product
/// integration on `n_steps + 1` uniformly spaced nodes.
pub fn solve_volterra<G, K>(
    g: G,
    kernel: K,
    t_horizon: f64,
    n_steps: usize,
    fixed_point_iter: usize,
    fixed_point_tol: f64,
) -> Result<VolterraResult>
where
    G: Fn(f64) -> f64,
    K: Fn(f64, f64) -> f64,
{
    if t_horizon <= 0.0 {
        return Err(OptimizrError::InvalidParameter("t_horizon > 0".into()));
    }
    if n_steps == 0 {
        return Err(OptimizrError::InvalidParameter("n_steps > 0".into()));
    }
    let dt = t_horizon / n_steps as f64;
    let mut t = vec![0.0; n_steps + 1];
    let mut y = vec![0.0; n_steps + 1];
    for k in 0..=n_steps {
        t[k] = k as f64 * dt;
    }
    y[0] = g(0.0);

    for n in 1..=n_steps {
        let g_n = g(t[n]);
        let mut explicit_part = 0.5 * kernel(t[n], y[0]);
        for k in 1..n {
            explicit_part += kernel(t[n] - t[k], y[k]);
        }
        explicit_part *= dt;

        // Implicit fixed-point on y_n
        let mut yn = y[n - 1]; // initial guess
        for _ in 0..fixed_point_iter {
            let yn_new = g_n + explicit_part + 0.5 * dt * kernel(0.0, yn);
            if (yn_new - yn).abs() < fixed_point_tol {
                yn = yn_new;
                break;
            }
            yn = yn_new;
        }
        y[n] = yn;
    }
    Ok(VolterraResult { t_grid: t, y })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact solution: y' = 1, y(0) = 0 corresponds to
    /// y(t) = t = 0 + int_0^t 1 ds, with K(s, y) = 1, g(t) = 0.
    #[test]
    fn constant_kernel_recovers_linear_growth() {
        let res = solve_volterra(|_t| 0.0, |_dt, _y| 1.0, 1.0, 1000, 50, 1e-12).unwrap();
        for k in 0..=1000 {
            let analytic = res.t_grid[k];
            assert!(
                (res.y[k] - analytic).abs() < 1e-10,
                "k={} y={} t={}",
                k,
                res.y[k],
                analytic
            );
        }
    }
}
