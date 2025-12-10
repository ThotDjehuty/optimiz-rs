//! Markov Regime Switching Models
//! ==============================
//!
//! Generic implementation of regime-switching dynamics following
//! "Markov Regime Switching Jump Diffusion Model and the Control Problem"
//!
//! # Mathematical Framework
//!
//! ## Regime-Switching Process
//!
//! State space: (X_t, α_t) where X_t ∈ ℝⁿ is the continuous state
//! and α_t ∈ {1,...,K} is the discrete regime indicator.
//!
//! Dynamics: dX_t = μ(X_t, α_t)dt + σ(X_t, α_t)dW_t
//! Regime transitions: P(α_{t+dt} = j | α_t = i) = q_{ij}dt + o(dt)
//!
//! ## Value Function
//!
//! V^i(x) = value function when in regime i at state x
//! System of coupled HJB equations:
//!
//! ρV^i(x) = sup_u [μ^i(x,u)·∇V^i(x) + (1/2)tr(σ^i(x,u)^T H^i(x) σ^i(x,u))
//!            + L^i(x,u) + Σ_{j≠i} q_{ij}(V^j(x) - V^i(x))]

use crate::optimal_control::{OptimalControlError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
// use statrs::distribution::{ContinuousCDF, Normal};

/// Regime-specific parameters
pub struct RegimeParameters {
    /// Drift coefficient μ(x)
    pub drift: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Diffusion coefficient σ(x)
    pub diffusion: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Running cost L(x, u)
    pub cost: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
}

/// Regime switching configuration
pub struct RegimeSwitchingConfig {
    /// Number of regimes
    pub n_regimes: usize,
    /// Transition rate matrix Q (q_ij for i ≠ j, q_ii computed)
    pub transition_rates: Array2<f64>,
    /// Discount rate
    pub rho: f64,
    /// Transaction cost
    pub transaction_cost: f64,
    /// State space bounds [x_min, x_max]
    pub state_bounds: (f64, f64),
    /// Number of grid points
    pub n_points: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for RegimeSwitchingConfig {
    fn default() -> Self {
        // Default: 2-regime model (bull/bear)
        let mut q = Array2::<f64>::zeros((2, 2));
        q[[0, 1]] = 0.5; // Bull -> Bear rate
        q[[1, 0]] = 0.3; // Bear -> Bull rate

        Self {
            n_regimes: 2,
            transition_rates: q,
            rho: 0.04,
            transaction_cost: 0.001,
            state_bounds: (-3.0, 3.0),
            n_points: 200,
            max_iter: 2000,
            tolerance: 1e-6,
        }
    }
}

/// Result from regime-switching solver
#[derive(Debug, Clone)]
pub struct RegimeSwitchingResult {
    /// State space grid
    pub x: Array1<f64>,
    /// Value functions for each regime V^i(x)
    pub values: Array2<f64>, // (n_regimes, n_points)
    /// Optimal controls for each regime u^i(x)
    pub controls: Array2<f64>,
    /// Gradients ∇V^i(x)
    pub gradients: Array2<f64>,
    /// Stationary distribution of regimes
    pub stationary_distribution: Array1<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual
    pub residual: f64,
}

/// Regime Switching Solver
pub struct RegimeSwitchingSolver {
    config: RegimeSwitchingConfig,
    regime_params: Vec<RegimeParameters>,
}

impl RegimeSwitchingSolver {
    /// Create new solver with configuration and regime-specific parameters
    pub fn new(
        config: RegimeSwitchingConfig,
        regime_params: Vec<RegimeParameters>,
    ) -> Result<Self> {
        // Validation
        if config.n_regimes < 2 {
            return Err(OptimalControlError::InvalidParameters(
                "Must have at least 2 regimes".to_string(),
            ));
        }

        if regime_params.len() != config.n_regimes {
            return Err(OptimalControlError::InvalidParameters(format!(
                "Need {} regime parameters, got {}",
                config.n_regimes,
                regime_params.len()
            )));
        }

        if config.transition_rates.shape() != [config.n_regimes, config.n_regimes] {
            return Err(OptimalControlError::InvalidParameters(
                "Transition rate matrix dimension mismatch".to_string(),
            ));
        }

        // Check transition rates are non-negative (except diagonal)
        for i in 0..config.n_regimes {
            for j in 0..config.n_regimes {
                if i != j && config.transition_rates[[i, j]] < 0.0 {
                    return Err(OptimalControlError::InvalidParameters(
                        "Transition rates must be non-negative".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            config,
            regime_params,
        })
    }

    /// Solve coupled system of HJB equations
    pub fn solve(&self) -> Result<RegimeSwitchingResult> {
        let cfg = &self.config;

        // Create state space grid
        let (x_min, x_max) = cfg.state_bounds;
        let dx = (x_max - x_min) / (cfg.n_points - 1) as f64;
        let x = Array1::from_iter((0..cfg.n_points).map(|i| x_min + i as f64 * dx));

        // Initialize value functions and controls
        let mut v = Array2::<f64>::zeros((cfg.n_regimes, cfg.n_points));
        let mut v_old = Array2::<f64>::zeros((cfg.n_regimes, cfg.n_points));
        let mut u = Array2::<f64>::zeros((cfg.n_regimes, cfg.n_points));

        // Compute diagonal elements of Q (ensure row sum = 0)
        let mut q = cfg.transition_rates.clone();
        for i in 0..cfg.n_regimes {
            let row_sum: f64 = (0..cfg.n_regimes)
                .filter(|&j| j != i)
                .map(|j| q[[i, j]])
                .sum();
            q[[i, i]] = -row_sum;
        }

        // Iterative solver
        let mut iterations = 0;
        let mut residual = f64::INFINITY;

        for iter in 0..cfg.max_iter {
            v_old.assign(&v);

            // Solve for each regime (can be parallelized)
            for regime in 0..cfg.n_regimes {
                self.solve_regime_hjb(regime, &x, &mut v, &v_old, &mut u, &q, dx)?;
            }

            // Check convergence
            residual =
                (&v - &v_old).mapv(|x| x.abs()).sum() / (cfg.n_regimes * cfg.n_points) as f64;

            iterations = iter + 1;

            if residual < cfg.tolerance {
                break;
            }

            // Relaxation for stability
            let omega = 0.7;
            v = &v * omega + &v_old * (1.0 - omega);
        }

        if residual >= cfg.tolerance {
            return Err(OptimalControlError::ConvergenceError(format!(
                "Failed to converge after {} iterations, residual = {}",
                iterations, residual
            )));
        }

        // Compute gradients
        let gradients = self.compute_gradients(&v, dx);

        // Compute stationary distribution
        let stationary_dist = self.compute_stationary_distribution(&q)?;

        Ok(RegimeSwitchingResult {
            x,
            values: v,
            controls: u,
            gradients,
            stationary_distribution: stationary_dist,
            iterations,
            residual,
        })
    }

    /// Solve HJB equation for a single regime
    fn solve_regime_hjb(
        &self,
        regime: usize,
        x: &Array1<f64>,
        v: &mut Array2<f64>,
        v_old: &Array2<f64>,
        u: &mut Array2<f64>,
        q: &Array2<f64>,
        dx: f64,
    ) -> Result<()> {
        let cfg = &self.config;
        let params = &self.regime_params[regime];

        // Interior points (parallel)
        let updates: Vec<(usize, f64, f64)> = (1..cfg.n_points - 1)
            .into_par_iter()
            .map(|i| {
                let xi = x[i];

                // Current value and neighbors
                let v_center = v_old[[regime, i]];
                let v_forward = v_old[[regime, i + 1]];
                let v_backward = v_old[[regime, i - 1]];

                // Gradients (finite differences)
                let dv_forward = (v_forward - v_center) / dx;
                let dv_backward = (v_center - v_backward) / dx;
                let d2v = (v_forward - 2.0 * v_center + v_backward) / (dx * dx);

                // Regime-specific drift and diffusion
                let _mu_xi = (params.drift)(xi);
                let _sigma_xi = (params.diffusion)(xi);

                // Optimal control via pointwise optimization
                // For portfolio: u* = argmax_u [μ(x,u)·dV/dx + L(x,u)]
                let optimal_control = self.optimize_control(xi, dv_forward, dv_backward, &params);

                // HJB operator with optimal control
                let mu_optimal = (params.drift)(xi); // Could depend on control
                let sigma_optimal = (params.diffusion)(xi);
                let cost = (params.cost)(xi, optimal_control);

                // Upwind scheme for drift
                let drift_term = if mu_optimal >= 0.0 {
                    mu_optimal * dv_backward
                } else {
                    mu_optimal * dv_forward
                };

                // Diffusion term
                let diffusion_term = 0.5 * sigma_optimal * sigma_optimal * d2v;

                // Regime switching term: Σ_{j≠i} q_ij(V^j(x) - V^i(x))
                let switching_term: f64 = (0..cfg.n_regimes)
                    .filter(|&j| j != regime)
                    .map(|j| q[[regime, j]] * (v_old[[j, i]] - v_center))
                    .sum();

                // Update: ρV = drift + diffusion + cost + switching
                let new_value = (drift_term + diffusion_term + cost + switching_term) / cfg.rho;

                (i, new_value, optimal_control)
            })
            .collect();

        // Apply updates
        for (i, new_value, optimal_control) in updates {
            v[[regime, i]] = new_value;
            u[[regime, i]] = optimal_control;
        }

        // Boundary conditions (reflecting or absorbing)
        v[[regime, 0]] = v[[regime, 1]];
        v[[regime, cfg.n_points - 1]] = v[[regime, cfg.n_points - 2]];

        Ok(())
    }

    /// Optimize control at a point (can be overridden for specific problems)
    fn optimize_control(
        &self,
        x: f64,
        dv_forward: f64,
        dv_backward: f64,
        params: &RegimeParameters,
    ) -> f64 {
        // Simple grid search for now (can be replaced with analytical solution)
        let u_grid = Array1::linspace(-1.0, 1.0, 21);

        let mut best_u = 0.0;
        let mut best_value = f64::NEG_INFINITY;

        for &u in u_grid.iter() {
            let mu = (params.drift)(x);
            let dv = if mu >= 0.0 { dv_backward } else { dv_forward };
            let cost = (params.cost)(x, u);

            let objective = mu * dv + cost;

            if objective > best_value {
                best_value = objective;
                best_u = u;
            }
        }

        best_u
    }

    /// Compute gradients for all regimes
    fn compute_gradients(&self, v: &Array2<f64>, dx: f64) -> Array2<f64> {
        let (n_regimes, n_points) = v.dim();
        let mut grad = Array2::<f64>::zeros((n_regimes, n_points));

        for i in 0..n_regimes {
            // Interior points: central differences
            for j in 1..n_points - 1 {
                grad[[i, j]] = (v[[i, j + 1]] - v[[i, j - 1]]) / (2.0 * dx);
            }

            // Boundaries: one-sided differences
            grad[[i, 0]] = (v[[i, 1]] - v[[i, 0]]) / dx;
            grad[[i, n_points - 1]] = (v[[i, n_points - 1]] - v[[i, n_points - 2]]) / dx;
        }

        grad
    }

    /// Compute stationary distribution of regime chain
    fn compute_stationary_distribution(&self, q: &Array2<f64>) -> Result<Array1<f64>> {
        let n = q.nrows();

        // Solve π^T Q = 0 subject to Σπ_i = 1
        // Convert to (Q^T - I)π = 0 with last equation Σπ_i = 1

        use ndarray_linalg::Solve;

        let mut a = q.t().to_owned();
        for i in 0..n {
            a[[i, i]] -= q[[i, i]];
        }

        // Replace last equation with Σπ_i = 1
        for j in 0..n {
            a[[n - 1, j]] = 1.0;
        }

        let mut b = Array1::<f64>::zeros(n);
        b[n - 1] = 1.0;

        match a.solve(&b) {
            Ok(pi) => Ok(pi),
            Err(_) => Err(OptimalControlError::MatrixError(
                "Failed to compute stationary distribution".to_string(),
            )),
        }
    }

    /// Create a two-regime model (bull/bear markets)
    pub fn two_regime_model(
        mu_bull: f64,
        mu_bear: f64,
        sigma_bull: f64,
        sigma_bear: f64,
        q_bull_to_bear: f64,
        q_bear_to_bull: f64,
    ) -> Result<Self> {
        let mut q = Array2::<f64>::zeros((2, 2));
        q[[0, 1]] = q_bull_to_bear;
        q[[1, 0]] = q_bear_to_bull;

        let config = RegimeSwitchingConfig {
            n_regimes: 2,
            transition_rates: q,
            ..Default::default()
        };

        // Bull regime parameters
        let params_bull = RegimeParameters {
            drift: Box::new(move |_x| mu_bull),
            diffusion: Box::new(move |_x| sigma_bull),
            cost: Box::new(|_x, _u| 0.0), // No running cost
        };

        // Bear regime parameters
        let params_bear = RegimeParameters {
            drift: Box::new(move |_x| mu_bear),
            diffusion: Box::new(move |_x| sigma_bear),
            cost: Box::new(|_x, _u| 0.0),
        };

        Self::new(config, vec![params_bull, params_bear])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_regime_solver() {
        let solver = RegimeSwitchingSolver::two_regime_model(
            0.1,   // bull drift
            -0.05, // bear drift
            0.15,  // bull volatility
            0.25,  // bear volatility
            0.5,   // bull->bear rate
            0.3,   // bear->bull rate
        )
        .unwrap();

        let result = solver.solve().unwrap();

        // Check dimensions
        assert_eq!(result.values.nrows(), 2);
        assert_eq!(result.values.ncols(), 200);

        // Check stationary distribution sums to 1
        let sum: f64 = result.stationary_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Value function should be higher in bull regime
        let mid_idx = result.x.len() / 2;
        assert!(result.values[[0, mid_idx]] > result.values[[1, mid_idx]]);
    }
}
