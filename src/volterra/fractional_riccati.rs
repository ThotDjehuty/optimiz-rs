//! Adams predictor--corrector for Caputo fractional ODEs.
//!
//! Solves
//!
//! ```text
//!     D^alpha h(t) = F(t, h(t)),         h(0) = h_0,    alpha in (0, 1)
//! ```
//!
//! by the Diethelm--Ford--Freed (2002) fractional Adams scheme.
//!
//! Predictor (Adams--Bashforth):
//!
//! ```text
//!     h^P_{n+1} = h_0 + (1 / Gamma(alpha)) * sum_{k=0}^n b_{n+1,k} F(t_k, h_k)
//! ```
//!
//! Corrector (Adams--Moulton):
//!
//! ```text
//!     h_{n+1} = h_0 + (1 / Gamma(alpha+2)) * [ F(t_{n+1}, h^P_{n+1})
//!                + sum_{k=0}^n a_{n+1,k} F(t_k, h_k) ]
//! ```
//!
//! with weights
//!
//! ```text
//!     b_{n+1,k} = h^alpha / alpha * ((n+1-k)^alpha - (n-k)^alpha)
//! ```
//!
//! and the standard `a_{n+1,k}` coefficients of Diethelm 2002 eq. (14).
//!
//! The scheme is `O(h^{1 + alpha})` accurate.
//!
//! # Reference
//!
//! Diethelm, Ford, Freed (2002), *A predictor--corrector approach for the
//! numerical solution of fractional differential equations*, Nonlinear
//! Dynamics 29.

use crate::core::{OptimizrError, Result};

/// Result of a fractional ODE integration.
#[derive(Debug, Clone)]
pub struct FractionalOdeResult {
    pub t_grid: Vec<f64>,
    pub h: Vec<f64>,
}

#[inline]
fn gamma(x: f64) -> f64 {
    statrs::function::gamma::gamma(x)
}

/// Solve `D^alpha h = F(t, h)` on `[0, T]` with `n_steps + 1` equally
/// spaced points and Caputo derivative of order `alpha in (0, 1)`.
pub fn solve_fractional_ode<F>(
    h0: f64,
    alpha: f64,
    t_horizon: f64,
    n_steps: usize,
    rhs: F,
) -> Result<FractionalOdeResult>
where
    F: Fn(f64, f64) -> f64,
{
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(OptimizrError::InvalidParameter(
            "alpha must lie in (0, 1)".into(),
        ));
    }
    if t_horizon <= 0.0 {
        return Err(OptimizrError::InvalidParameter("t_horizon > 0 required".into()));
    }
    if n_steps == 0 {
        return Err(OptimizrError::InvalidParameter("n_steps > 0 required".into()));
    }
    let dt = t_horizon / n_steps as f64;
    let mut t = vec![0.0; n_steps + 1];
    let mut h = vec![0.0; n_steps + 1];
    let mut f_cache = vec![0.0; n_steps + 1];
    h[0] = h0;
    for k in 1..=n_steps {
        t[k] = k as f64 * dt;
    }
    f_cache[0] = rhs(t[0], h[0]);

    let g_alpha = gamma(alpha);
    let g_alpha_plus_2 = gamma(alpha + 2.0);
    let dt_alpha = dt.powf(alpha);

    for n in 0..n_steps {
        // ===== Predictor (fractional Adams--Bashforth) =====
        let mut sum_b = 0.0;
        for k in 0..=n {
            let nk = (n + 1 - k) as f64;
            let nkm = (n - k) as f64;
            let b = nk.powf(alpha) - nkm.powf(alpha);
            sum_b += b * f_cache[k];
        }
        let h_pred = h0 + dt_alpha / (alpha * g_alpha) * sum_b;

        // ===== Corrector (fractional Adams--Moulton) =====
        let f_pred = rhs(t[n + 1], h_pred);
        let mut sum_a = 0.0;
        for k in 0..=n {
            sum_a += a_weight(n, k, alpha) * f_cache[k];
        }
        let h_new = h0 + dt_alpha / g_alpha_plus_2 * (f_pred + sum_a);
        h[n + 1] = h_new;
        f_cache[n + 1] = rhs(t[n + 1], h[n + 1]);
    }

    Ok(FractionalOdeResult { t_grid: t, h })
}

fn a_weight(n: usize, k: usize, alpha: f64) -> f64 {
    // a_{n+1, k} for 0 <= k <= n  (Diethelm 2002 eq. 14):
    //
    //   a_{n+1, 0}     = n^{alpha+1} - (n - alpha)*(n+1)^alpha
    //   a_{n+1, k}     = (n - k + 2)^{alpha+1} + (n - k)^{alpha+1}
    //                    - 2 (n - k + 1)^{alpha+1}            (1 <= k <= n)
    //
    // (Final h^alpha / Gamma(alpha + 2) factor handled in caller.)
    let n_f = n as f64;
    if k == 0 {
        n_f.powf(alpha + 1.0) - (n_f - alpha) * (n_f + 1.0).powf(alpha)
    } else {
        let a = (n_f - k as f64 + 2.0).powf(alpha + 1.0);
        let b = (n_f - k as f64).powf(alpha + 1.0);
        let c = 2.0 * (n_f - k as f64 + 1.0).powf(alpha + 1.0);
        a + b - c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For alpha -> 1, the Caputo equation reduces to the classical ODE
    /// h' = -h, h(0) = 1, with solution h(t) = exp(-t).
    #[test]
    fn limit_alpha_one_matches_exponential_decay() {
        let alpha = 0.999;
        let res = solve_fractional_ode(1.0, alpha, 2.0, 4000, |_t, h| -h).unwrap();
        let final_t = res.t_grid[4000];
        let analytic = (-final_t).exp();
        let err = (res.h[4000] - analytic).abs();
        assert!(err < 5e-2, "err = {}", err);
    }

    #[test]
    fn rejects_invalid_alpha() {
        assert!(solve_fractional_ode(1.0, 0.0, 1.0, 10, |_, _| 0.0).is_err());
        assert!(solve_fractional_ode(1.0, 1.5, 1.0, 10, |_, _| 0.0).is_err());
    }
}
