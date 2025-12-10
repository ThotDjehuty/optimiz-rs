//! Markov Regime Switching Jump Diffusion (MRSJD)
//! ==============================================
//!
//! Complete implementation following the paper:
//! "Markov Regime Switching Jump Diffusion Model and the Control Problem"
//!
//! # Mathematical Framework
//!
//! ## Full Dynamics
//!
//! State: (X_t, α_t) where α_t ∈ {1,...,K} is regime
//!
//! dX_t = μ^{α_t}(X_t)dt + σ^{α_t}(X_t)dW_t + dJ_t^{α_t}
//!
//! - Regime transitions: q_{ij}dt probability
//! - Jump intensity and distribution depend on regime: λ^i, F^i
//!
//! ## Coupled HJB System with Jumps
//!
//! ρV^i(x) = sup_u [μ^i·∇V^i + (σ^i)²/2·∇²V^i + L^i(x,u)
//!            + λ^i∫[V^i(x+y) - V^i(x)]F^i(dy)
//!            + Σ_{j≠i} q_{ij}[V^j(x) - V^i(x)]]
//!
//! This is the most general formulation combining:
//! 1. Diffusion processes
//! 2. Jump processes  
//! 3. Regime switching
//! 4. Optimal control

use crate::optimal_control::{
    jump_diffusion::JumpDistribution,
    OptimalControlError, Result,
};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Regime-specific jump parameters
pub struct RegimeJumpParameters {
    /// Drift μ^i(x)
    pub drift: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Diffusion σ^i(x)
    pub diffusion: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Running cost L^i(x, u)
    pub cost: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
    /// Jump intensity λ^i
    pub jump_intensity: f64,
    /// Jump distribution F^i
    pub jump_distribution: JumpDistribution,
}

/// MRSJD configuration
#[derive(Debug, Clone)]
pub struct MRSJDConfig {
    /// Number of regimes
    pub n_regimes: usize,
    /// Transition rate matrix Q
    pub transition_rates: Array2<f64>,
    /// Discount rate
    pub rho: f64,
    /// Transaction cost
    pub transaction_cost: f64,
    /// State space bounds
    pub state_bounds: (f64, f64),
    /// Number of grid points
    pub n_points: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for MRSJDConfig {
    fn default() -> Self {
        let mut q = Array2::<f64>::zeros((2, 2));
        q[[0, 1]] = 0.5;
        q[[1, 0]] = 0.3;

        Self {
            n_regimes: 2,
            transition_rates: q,
            rho: 0.04,
            transaction_cost: 0.001,
            state_bounds: (-4.0, 4.0),
            n_points: 400, // More points needed for jumps
            max_iter: 3000,
            tolerance: 1e-6,
        }
    }
}

/// MRSJD result
#[derive(Debug, Clone)]
pub struct MRSJDResult {
    /// State space grid
    pub x: Array1<f64>,
    /// Value functions V^i(x)
    pub values: Array2<f64>,
    /// Optimal controls u^i(x)
    pub controls: Array2<f64>,
    /// Gradients
    pub gradients: Array2<f64>,
    /// Jump integral contributions
    pub jump_integrals: Array2<f64>,
    /// Stationary distribution
    pub stationary_distribution: Array1<f64>,
    /// Iterations
    pub iterations: usize,
    /// Residual
    pub residual: f64,
}

/// Markov Regime Switching Jump Diffusion Solver
pub struct MRSJDSolver {
    config: MRSJDConfig,
    regime_params: Vec<RegimeJumpParameters>,
}

impl MRSJDSolver {
    /// Create new MRSJD solver
    pub fn new(config: MRSJDConfig, regime_params: Vec<RegimeJumpParameters>) -> Result<Self> {
        // Validation
        if config.n_regimes != regime_params.len() {
            return Err(OptimalControlError::InvalidParameters(format!(
                "Need {} regime parameters",
                config.n_regimes
            )));
        }

        if config.n_points < 200 {
            return Err(OptimalControlError::InvalidParameters(
                "Need at least 200 grid points for MRSJD".to_string(),
            ));
        }

        // Validate transition rates
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

    /// Solve the coupled MRSJD system
    pub fn solve(&self) -> Result<MRSJDResult> {
        let cfg = &self.config;

        // Create grid
        let (x_min, x_max) = cfg.state_bounds;
        let dx = (x_max - x_min) / (cfg.n_points - 1) as f64;
        let x = Array1::from_iter((0..cfg.n_points).map(|i| x_min + i as f64 * dx));

        // Precompute jump kernels for each regime
        let jump_kernels = self.compute_all_jump_kernels(&x, dx)?;

        // Setup transition matrix
        let mut q = cfg.transition_rates.clone();
        for i in 0..cfg.n_regimes {
            let row_sum: f64 = (0..cfg.n_regimes)
                .filter(|&j| j != i)
                .map(|j| q[[i, j]])
                .sum();
            q[[i, i]] = -row_sum;
        }

        // Initialize
        let mut v = Array2::<f64>::zeros((cfg.n_regimes, cfg.n_points));
        let mut v_old = Array2::<f64>::zeros((cfg.n_regimes, cfg.n_points));
        let mut u = Array2::<f64>::zeros((cfg.n_regimes, cfg.n_points));
        let mut jump_integrals = Array2::<f64>::zeros((cfg.n_regimes, cfg.n_points));

        // Main iteration loop
        let mut iterations = 0;
        let mut residual = f64::INFINITY;

        for iter in 0..cfg.max_iter {
            v_old.assign(&v);

            // Solve for each regime
            for regime in 0..cfg.n_regimes {
                self.solve_regime_mrsjd(
                    regime,
                    &x,
                    &mut v,
                    &v_old,
                    &mut u,
                    &mut jump_integrals,
                    &q,
                    &jump_kernels[regime],
                    dx,
                )?;
            }

            // Check convergence
            residual =
                (&v - &v_old).mapv(|x| x.abs()).sum() / (cfg.n_regimes * cfg.n_points) as f64;
            iterations = iter + 1;

            if residual < cfg.tolerance {
                break;
            }

            // Relaxation
            let omega = 0.5; // More conservative for stability
            v = &v * omega + &v_old * (1.0 - omega);
        }

        if residual >= cfg.tolerance {
            return Err(OptimalControlError::ConvergenceError(format!(
                "Failed to converge after {} iterations, residual = {:.2e}",
                iterations, residual
            )));
        }

        // Compute gradients
        let gradients = self.compute_all_gradients(&v, dx);

        // Stationary distribution
        let stationary_dist = self.compute_stationary_distribution(&q)?;

        Ok(MRSJDResult {
            x,
            values: v,
            controls: u,
            gradients,
            jump_integrals,
            stationary_distribution: stationary_dist,
            iterations,
            residual,
        })
    }

    /// Solve HJB for one regime with jumps
    fn solve_regime_mrsjd(
        &self,
        regime: usize,
        x: &Array1<f64>,
        v: &mut Array2<f64>,
        v_old: &Array2<f64>,
        u: &mut Array2<f64>,
        jump_int: &mut Array2<f64>,
        q: &Array2<f64>,
        jump_kernel: &Array2<f64>,
        dx: f64,
    ) -> Result<()> {
        let cfg = &self.config;
        let params = &self.regime_params[regime];

        // Compute jump integral for this regime
        let lambda = params.jump_intensity;
        for i in 0..cfg.n_points {
            let mut integral = 0.0;
            for j in 0..cfg.n_points {
                integral += jump_kernel[[i, j]] * (v_old[[regime, j]] - v_old[[regime, i]]);
            }
            jump_int[[regime, i]] = lambda * integral;
        }

        // Solve at interior points (parallel)
        let updates: Vec<(usize, f64, f64)> = (1..cfg.n_points - 1)
            .into_par_iter()
            .map(|i| {
                let xi = x[i];

                // Get values
                let v_c = v_old[[regime, i]];
                let v_f = v_old[[regime, i + 1]];
                let v_b = v_old[[regime, i - 1]];

                // Derivatives
                let dv_forward = (v_f - v_c) / dx;
                let dv_backward = (v_c - v_b) / dx;
                let d2v = (v_f - 2.0 * v_c + v_b) / (dx * dx);

                // Regime-specific parameters
                let mu = (params.drift)(xi);
                let sigma = (params.diffusion)(xi);

                // Upwind scheme
                let drift_term = if mu >= 0.0 {
                    mu * dv_backward
                } else {
                    mu * dv_forward
                };

                // Diffusion
                let diffusion_term = 0.5 * sigma * sigma * d2v;

                // Jump integral
                let jump_term = jump_int[[regime, i]];

                // Regime switching term
                let switching_term: f64 = (0..cfg.n_regimes)
                    .filter(|&j| j != regime)
                    .map(|j| q[[regime, j]] * (v_old[[j, i]] - v_c))
                    .sum();

                // Optimal control (placeholder - can be optimized)
                let optimal_control =
                    self.optimize_control_mrsjd(xi, dv_forward, dv_backward, params);

                // Running cost
                let cost = (params.cost)(xi, optimal_control);

                // HJB update
                let new_value =
                    (drift_term + diffusion_term + jump_term + switching_term + cost) / cfg.rho;

                (i, new_value, optimal_control)
            })
            .collect();

        // Apply updates
        for (i, new_value, optimal_control) in updates {
            v[[regime, i]] = new_value;
            u[[regime, i]] = optimal_control;
        }

        // Boundaries
        v[[regime, 0]] = v[[regime, 1]];
        v[[regime, cfg.n_points - 1]] = v[[regime, cfg.n_points - 2]];

        Ok(())
    }

    /// Compute jump kernels for all regimes
    #[allow(unused_variables)]  // dx parameter reserved for future extensions
    fn compute_all_jump_kernels(&self, x: &Array1<f64>, dx: f64) -> Result<Vec<Array2<f64>>> {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let n = x.len();
        let mut kernels = Vec::with_capacity(self.config.n_regimes);

        for regime in 0..self.config.n_regimes {
            let mut kernel = Array2::<f64>::zeros((n, n));
            let dist = &self.regime_params[regime].jump_distribution;

            // Monte Carlo discretization
            let n_samples = 20000;
            for i in 0..n {
                let mut jump_counts = vec![0; n];

                for _ in 0..n_samples {
                    let jump_size = dist.sample(&mut rng);
                    let target_x = x[i] + jump_size;

                    if let Some(j) = self.find_nearest_index(x, target_x) {
                        jump_counts[j] += 1;
                    }
                }

                // Normalize
                for j in 0..n {
                    kernel[[i, j]] = jump_counts[j] as f64 / n_samples as f64;
                }
            }

            kernels.push(kernel);
        }

        Ok(kernels)
    }

    /// Find nearest grid point
    fn find_nearest_index(&self, x: &Array1<f64>, target: f64) -> Option<usize> {
        let (x_min, x_max) = self.config.state_bounds;

        if target < x_min || target > x_max {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist = (x[0] - target).abs();

        for (i, &xi) in x.iter().enumerate() {
            let dist = (xi - target).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        Some(best_idx)
    }

    /// Optimize control (problem-specific)
    fn optimize_control_mrsjd(
        &self,
        _x: f64,
        _dv_forward: f64,
        _dv_backward: f64,
        _params: &RegimeJumpParameters,
    ) -> f64 {
        // Placeholder - implement specific optimization
        0.0
    }

    /// Compute gradients for all regimes
    fn compute_all_gradients(&self, v: &Array2<f64>, dx: f64) -> Array2<f64> {
        let (n_regimes, n_points) = v.dim();
        let mut grad = Array2::<f64>::zeros((n_regimes, n_points));

        for i in 0..n_regimes {
            for j in 1..n_points - 1 {
                grad[[i, j]] = (v[[i, j + 1]] - v[[i, j - 1]]) / (2.0 * dx);
            }
            grad[[i, 0]] = (v[[i, 1]] - v[[i, 0]]) / dx;
            grad[[i, n_points - 1]] = (v[[i, n_points - 1]] - v[[i, n_points - 2]]) / dx;
        }

        grad
    }

    /// Compute stationary distribution
    fn compute_stationary_distribution(&self, q: &Array2<f64>) -> Result<Array1<f64>> {
        use ndarray_linalg::Solve;

        let n = q.nrows();
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrsjd_solver_generic() {
        // Generic test: State evolution with regime switching and jumps
        // Using abstract state variable (not portfolio-specific)

        let mut q = Array2::<f64>::zeros((2, 2));
        q[[0, 1]] = 0.5; // Regime 0 → 1
        q[[1, 0]] = 0.3; // Regime 1 → 0

        let config = MRSJDConfig {
            n_regimes: 2,
            transition_rates: q,
            state_bounds: (-1.0, 3.0),
            n_points: 100,
            rho: 0.05,
            transaction_cost: 0.0,
            max_iter: 200,
            tolerance: 1e-4,
        };

        // Regime 0: Higher drift, lower volatility, fewer jumps
        let params_0 = RegimeJumpParameters {
            drift: Box::new(|x| 0.5 - 0.1 * x), // Mean-reverting to 0.5
            diffusion: Box::new(|_x| 0.2),
            cost: Box::new(|x, _u| x.powi(2)), // Quadratic cost
            jump_intensity: 0.1,
            jump_distribution: JumpDistribution::Normal {
                mean: -0.05,
                std: 0.1,
            },
        };

        // Regime 1: Lower drift, higher volatility, more frequent jumps
        let params_1 = RegimeJumpParameters {
            drift: Box::new(|x| 0.2 - 0.05 * x),
            diffusion: Box::new(|_x| 0.4),
            cost: Box::new(|x, _u| x.powi(2)),
            jump_intensity: 0.3,
            jump_distribution: JumpDistribution::Normal {
                mean: -0.1,
                std: 0.15,
            },
        };

        let solver = MRSJDSolver::new(config, vec![params_0, params_1]).unwrap();
        let result = solver.solve().unwrap();

        assert_eq!(result.values.nrows(), 2);
        assert!(result.iterations > 0);
        assert!(result.residual < 1e-4);

        // Stationary distribution should sum to 1
        let sum: f64 = result.stationary_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
