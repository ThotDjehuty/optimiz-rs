//! HJB PDE Solver
//! ==============
//!
//! Generic Hamilton-Jacobi-Bellman equation solver using finite differences.

use crate::optimal_control::{OptimalControlError, Result};
use ndarray::Array1;
use rayon::prelude::*;

/// Configuration for HJB solver
#[derive(Debug, Clone)]
pub struct HJBConfig {
    /// Mean-reversion speed (κ in OU process)
    pub kappa: f64,
    /// Long-term mean (θ in OU process)
    pub theta: f64,
    /// Volatility (σ in OU process)
    pub sigma: f64,
    /// Discount rate
    pub rho: f64,
    /// Transaction cost per trade
    pub transaction_cost: f64,
    /// Number of grid points
    pub n_points: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of standard deviations for domain
    pub n_std: f64,
}

impl Default for HJBConfig {
    fn default() -> Self {
        Self {
            kappa: 0.5,
            theta: 0.0,
            sigma: 0.1,
            rho: 0.04,
            transaction_cost: 0.001,
            n_points: 200,
            max_iter: 2000,
            tolerance: 1e-6,
            n_std: 4.0,
        }
    }
}

/// Result from HJB solver
#[derive(Debug, Clone)]
pub struct HJBResult {
    /// State space grid
    pub x: Array1<f64>,
    /// Value function V(x)
    pub value: Array1<f64>,
    /// First derivative V'(x)
    pub gradient: Array1<f64>,
    /// Second derivative V''(x)
    pub hessian: Array1<f64>,
    /// Lower boundary (buy signal)
    pub lower_boundary: f64,
    /// Upper boundary (sell signal)
    pub upper_boundary: f64,
    /// Number of iterations until convergence
    pub iterations: usize,
    /// Final residual
    pub residual: f64,
}

/// Generic HJB PDE Solver
pub struct HJBSolver {
    config: HJBConfig,
}

impl HJBSolver {
    /// Create new HJB solver with configuration
    pub fn new(config: HJBConfig) -> Result<Self> {
        // Validate parameters
        if config.kappa <= 0.0 {
            return Err(OptimalControlError::InvalidParameters(
                "kappa must be positive".to_string(),
            ));
        }
        if config.sigma <= 0.0 {
            return Err(OptimalControlError::InvalidParameters(
                "sigma must be positive".to_string(),
            ));
        }
        if config.rho <= 0.0 {
            return Err(OptimalControlError::InvalidParameters(
                "rho must be positive".to_string(),
            ));
        }
        if config.n_points < 50 {
            return Err(OptimalControlError::InvalidParameters(
                "n_points must be at least 50".to_string(),
            ));
        }

        Ok(Self { config })
    }

    /// Solve HJB equation using finite differences
    pub fn solve(&self) -> Result<HJBResult> {
        let cfg = &self.config;

        // Compute stationary standard deviation
        let sigma_inf = cfg.sigma / (2.0 * cfg.kappa).sqrt();

        // State space: θ ± n_std * σ_∞
        let x_min = cfg.theta - cfg.n_std * sigma_inf;
        let x_max = cfg.theta + cfg.n_std * sigma_inf;
        let dx = (x_max - x_min) / (cfg.n_points - 1) as f64;

        // Create grid
        let x = Array1::from_iter((0..cfg.n_points).map(|i| x_min + i as f64 * dx));

        // Initialize value function
        let mut v = Array1::<f64>::zeros(cfg.n_points);
        let mut v_old = Array1::<f64>::zeros(cfg.n_points);

        // Coefficients for finite differences
        let drift_coeff = cfg.kappa / (2.0 * dx);
        let diffusion_coeff = 0.5 * cfg.sigma.powi(2) / dx.powi(2);

        // Iterative solver
        let mut iterations = 0;
        let mut residual = f64::INFINITY;

        for iter in 0..cfg.max_iter {
            v_old.assign(&v);

            // Interior points (parallel computation)
            let _v_slice = v.as_slice().unwrap();
            let x_slice = x.as_slice().unwrap();
            let v_old_slice = v_old.as_slice().unwrap();

            let interior_values: Vec<f64> = (1..cfg.n_points - 1)
                .into_par_iter()
                .map(|i| {
                    let xi = x_slice[i];

                    // Drift term: κ(θ - x) * dV/dx
                    let drift = cfg.kappa
                        * (cfg.theta - xi)
                        * (v_old_slice[i + 1] - v_old_slice[i - 1])
                        * drift_coeff
                        / cfg.kappa;

                    // Diffusion term: (σ²/2) * d²V/dx²
                    let diffusion = (v_old_slice[i + 1] - 2.0 * v_old_slice[i]
                        + v_old_slice[i - 1])
                        * diffusion_coeff;

                    // Update: ρV = drift + diffusion
                    (drift + diffusion) / cfg.rho
                })
                .collect();

            // Update interior points
            for (i, &val) in interior_values.iter().enumerate() {
                v[i + 1] = val;
            }

            // Boundary conditions (Neumann: dV/dx = 0 at boundaries)
            v[0] = v[1];
            v[cfg.n_points - 1] = v[cfg.n_points - 2];

            // Check convergence
            residual = (&v - &v_old)
                .mapv(|x| x.abs())
                .iter()
                .fold(0.0f64, |acc, &x| acc.max(x));

            iterations = iter + 1;

            if residual < cfg.tolerance {
                break;
            }
        }

        if residual >= cfg.tolerance {
            return Err(OptimalControlError::ConvergenceError(format!(
                "Failed to converge after {} iterations (residual: {:.2e})",
                iterations, residual
            )));
        }

        // Compute gradient (first derivative)
        let gradient = self.compute_gradient(&v, dx);

        // Compute hessian (second derivative)
        let hessian = self.compute_hessian(&v, dx);

        // Find optimal boundaries
        let (lower_boundary, upper_boundary) = self.find_boundaries(&x, &gradient, cfg.theta);

        Ok(HJBResult {
            x,
            value: v,
            gradient,
            hessian,
            lower_boundary,
            upper_boundary,
            iterations,
            residual,
        })
    }

    /// Compute first derivative using central differences
    fn compute_gradient(&self, v: &Array1<f64>, dx: f64) -> Array1<f64> {
        let n = v.len();
        let mut gradient = Array1::<f64>::zeros(n);

        // Interior points (central difference)
        for i in 1..n - 1 {
            gradient[i] = (v[i + 1] - v[i - 1]) / (2.0 * dx);
        }

        // Boundaries (forward/backward difference)
        gradient[0] = (v[1] - v[0]) / dx;
        gradient[n - 1] = (v[n - 1] - v[n - 2]) / dx;

        gradient
    }

    /// Compute second derivative using finite differences
    fn compute_hessian(&self, v: &Array1<f64>, dx: f64) -> Array1<f64> {
        let n = v.len();
        let mut hessian = Array1::<f64>::zeros(n);

        // Interior points
        for i in 1..n - 1 {
            hessian[i] = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / dx.powi(2);
        }

        // Boundaries (one-sided)
        hessian[0] = hessian[1];
        hessian[n - 1] = hessian[n - 2];

        hessian
    }

    /// Find optimal switching boundaries
    #[allow(unused_variables)]  // theta parameter reserved for future use
    fn find_boundaries(&self, x: &Array1<f64>, gradient: &Array1<f64>, theta: f64) -> (f64, f64) {
        let n = x.len();
        let mid_idx = n / 2;

        // Lower boundary: V' ≈ 1 (below mean)
        let mut lower_idx = 0;
        let mut min_dist = f64::INFINITY;
        for i in 0..mid_idx {
            let dist = (gradient[i] - 1.0).abs();
            if dist < min_dist {
                min_dist = dist;
                lower_idx = i;
            }
        }

        // Upper boundary: V' ≈ -1 (above mean)
        let mut upper_idx = n - 1;
        min_dist = f64::INFINITY;
        for i in mid_idx..n {
            let dist = (gradient[i] + 1.0).abs();
            if dist < min_dist {
                min_dist = dist;
                upper_idx = i;
            }
        }

        (x[lower_idx], x[upper_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hjb_solver_convergence() {
        let config = HJBConfig {
            kappa: 0.5,
            theta: 0.0,
            sigma: 0.1,
            rho: 0.04,
            transaction_cost: 0.001,
            n_points: 100,
            max_iter: 1000,
            tolerance: 1e-5,
            n_std: 3.0,
        };

        let solver = HJBSolver::new(config).unwrap();
        let result = solver.solve().unwrap();

        assert!(result.iterations < 1000);
        assert!(result.residual < 1e-5);
        assert!(result.lower_boundary < result.upper_boundary);
        assert!(result.lower_boundary < 0.0);
        assert!(result.upper_boundary > 0.0);
    }

    #[test]
    fn test_hjb_solver_symmetry() {
        let config = HJBConfig {
            kappa: 1.0,
            theta: 0.0,
            sigma: 0.2,
            ..Default::default()
        };

        let solver = HJBSolver::new(config).unwrap();
        let result = solver.solve().unwrap();

        // For symmetric OU process, boundaries should be symmetric
        assert_relative_eq!(
            result.lower_boundary.abs(),
            result.upper_boundary.abs(),
            epsilon = 0.1
        );
    }
}
