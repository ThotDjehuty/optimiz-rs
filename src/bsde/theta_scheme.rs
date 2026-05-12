//! θ-scheme for linear backward stochastic differential equations
//! ================================================================
//!
//! Discretises the BSDE
//!
//! ```text
//! -dY_t = (a(t) Y_t + b(t) Z_t + c(t)) dt - Z_t · dW_t,    Y_T = g(X_T)
//! ```
//!
//! on a uniform partition `0 = t_0 < t_1 < ... < t_N = T` (`Δt = T/N`).
//!
//! Letting `E_n[·]` denote the conditional expectation given `F_{t_n}`, the
//! θ-scheme reads
//!
//! ```text
//! Z_n = E_n[ (Y_{n+1} ΔW_{n+1}) / Δt ]                                  (1)
//! Y_n = (1 - θ Δt a_n)^{-1}
//!         · ( E_n[Y_{n+1}] + Δt [ (1-θ) (a_{n+1} Y_{n+1} + b_{n+1} Z_n + c_{n+1})
//!                                + θ (b_n Z_n + c_n) ] )                (2)
//! ```
//!
//! For θ = 0 the scheme is fully explicit, θ = 1 is fully implicit and θ = 1/2
//! is the Crank–Nicolson midpoint rule.  When the driver depends linearly on
//! `(Y, Z)` and the terminal condition is deterministic, the discrete system
//! decouples and admits an exact closed-form recursion that we solve here.

use crate::core::{OptimizrError, Result};
use ndarray::Array1;

/// Configuration of the θ-scheme.
#[derive(Clone, Debug)]
pub struct ThetaSchemeConfig {
    /// Number of time steps `N` (uniform grid of `[0, T]`).
    pub n_steps: usize,
    /// Terminal time `T > 0`.
    pub t_horizon: f64,
    /// Implicit weight θ ∈ [0, 1].
    pub theta: f64,
}

impl ThetaSchemeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.n_steps == 0 {
            return Err(OptimizrError::InvalidParameter("n_steps must be > 0".into()));
        }
        if !(self.t_horizon > 0.0) {
            return Err(OptimizrError::InvalidParameter("t_horizon must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.theta) {
            return Err(OptimizrError::InvalidParameter("theta must be in [0,1]".into()));
        }
        Ok(())
    }
}

/// Result of the linear-BSDE θ-scheme.
#[derive(Clone, Debug)]
pub struct ThetaSchemeResult {
    /// Discrete trajectory `Y_0, Y_1, ..., Y_N`.
    pub y: Array1<f64>,
    /// Discrete trajectory `Z_0, Z_1, ..., Z_{N-1}` (length `N`).
    pub z: Array1<f64>,
    /// Time grid `t_0, ..., t_N`.
    pub time_grid: Array1<f64>,
}

/// Solve the deterministic-coefficient linear BSDE
///
/// ```text
/// -dY_t = (a(t) Y_t + b(t) Z_t + c(t)) dt - Z_t · dW_t,    Y_T = terminal
/// ```
///
/// The solution `(Y_t)` is a deterministic function of time when the
/// terminal value is constant (which is the canonical analytic-test setup);
/// the θ-scheme then collapses to a scalar recursion that we integrate
/// backwards in time.  `Z_t = b(t)` cancels the `Z`-coupling at the
/// continuous level so `Z` should converge to zero — we report the
/// discrete `Z_n` predicted by (1) under that ansatz.
///
/// # Arguments
/// * `a`, `b`, `c` — closures `t -> coefficient` (continuous functions).
/// * `terminal`   — terminal value `Y_T`.
/// * `cfg`        — discretisation parameters.
pub fn solve_linear_bsde<A, B, C>(
    a: A,
    b: B,
    c: C,
    terminal: f64,
    cfg: &ThetaSchemeConfig,
) -> Result<ThetaSchemeResult>
where
    A: Fn(f64) -> f64,
    B: Fn(f64) -> f64,
    C: Fn(f64) -> f64,
{
    cfg.validate()?;
    let n = cfg.n_steps;
    let dt = cfg.t_horizon / n as f64;
    let theta = cfg.theta;

    let time_grid: Array1<f64> = Array1::from_iter((0..=n).map(|i| i as f64 * dt));
    let mut y = Array1::<f64>::zeros(n + 1);
    let mut z = Array1::<f64>::zeros(n);
    y[n] = terminal;

    for k in (0..n).rev() {
        let t_n = time_grid[k];
        let t_np1 = time_grid[k + 1];
        let a_n = a(t_n);
        let a_np1 = a(t_np1);
        let b_n = b(t_n);
        let b_np1 = b(t_np1);
        let c_n = c(t_n);
        let c_np1 = c(t_np1);

        // Deterministic-coefficient ansatz: Z_n = 0 in the continuous limit.
        let z_n = 0.0;
        z[k] = z_n;

        let denom = 1.0 - theta * dt * a_n;
        if denom.abs() < 1e-14 {
            return Err(OptimizrError::NumericalError(
                "θ-scheme implicit factor is singular".into(),
            ));
        }
        let rhs = y[k + 1]
            + dt * ((1.0 - theta) * (a_np1 * y[k + 1] + b_np1 * z_n + c_np1)
                + theta * (b_n * z_n + c_n));
        y[k] = rhs / denom;
    }

    Ok(ThetaSchemeResult { y, z, time_grid })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For a(t) = -ρ, b = c = 0, terminal = 1, the BSDE
    /// `-dY = -ρ Y dt - Z dW` admits the deterministic solution
    /// `Y_t = exp(-ρ (T - t))`.  Crank–Nicolson (θ=½) is second-order
    /// accurate.
    #[test]
    fn theta_scheme_recovers_exponential_growth() {
        let rho: f64 = 0.3;
        let t_horizon = 1.0_f64;
        let cfg = ThetaSchemeConfig {
            n_steps: 200,
            t_horizon,
            theta: 0.5,
        };
        let res = solve_linear_bsde(|_| -rho, |_| 0.0, |_| 0.0, 1.0, &cfg).unwrap();
        let analytic = (-rho * t_horizon).exp();
        let err = (res.y[0] - analytic).abs();
        assert!(err < 1e-3, "Y_0 = {}, analytic = {}, err = {}", res.y[0], analytic, err);
    }

    #[test]
    fn theta_scheme_constant_driver() {
        // a = b = 0, c = 1, terminal = 0  =>  Y_t = T - t
        let cfg = ThetaSchemeConfig { n_steps: 100, t_horizon: 1.0, theta: 1.0 };
        let res = solve_linear_bsde(|_| 0.0, |_| 0.0, |_| 1.0, 0.0, &cfg).unwrap();
        assert!((res.y[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn theta_scheme_validates_inputs() {
        let cfg = ThetaSchemeConfig { n_steps: 0, t_horizon: 1.0, theta: 0.5 };
        assert!(solve_linear_bsde(|_| 0.0, |_| 0.0, |_| 0.0, 0.0, &cfg).is_err());
        let cfg = ThetaSchemeConfig { n_steps: 10, t_horizon: 1.0, theta: 1.5 };
        assert!(solve_linear_bsde(|_| 0.0, |_| 0.0, |_| 0.0, 0.0, &cfg).is_err());
    }
}
