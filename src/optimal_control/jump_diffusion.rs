//! Jump Diffusion Processes
//! ========================
//!
//! Implementation of jump-diffusion models (Lévy processes) for optimal control.
//!
//! # Mathematical Framework
//!
//! ## Jump-Diffusion Dynamics
//!
//! dX_t = μ(X_t)dt + σ(X_t)dW_t + dJ_t
//!
//! where J_t is a compound Poisson process:
//! - Jump times follow Poisson(λ)
//! - Jump sizes Y_i ~ distribution F
//!
//! ## HJB Equation with Jumps
//!
//! ρV(x) = sup_u [μ(x,u)·∇V(x) + (1/2)σ²(x,u)·∇²V(x) + L(x,u)
//!          + λ∫[V(x+y) - V(x)]F(dy)]
//!
//! The integral term captures the expected change from jumps.

use crate::optimal_control::{OptimalControlError, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Distribution;
use rayon::prelude::*;
use statrs::distribution::{Exp as Exponential, Normal};

/// Jump distribution types
#[derive(Debug, Clone)]
pub enum JumpDistribution {
    /// Normal jumps: Y ~ N(μ_j, σ_j²)
    Normal { mean: f64, std: f64 },
    /// Exponential jumps: Y ~ Exp(λ)
    Exponential { rate: f64 },
    /// Two-sided exponential (Laplace)
    Laplace { location: f64, scale: f64 },
    /// Uniform jumps: Y ~ U(a, b)
    Uniform { min: f64, max: f64 },
    /// Custom distribution (discretized)
    Discrete {
        jumps: Vec<f64>,
        probabilities: Vec<f64>,
    },
}

impl JumpDistribution {
    /// Sample from the jump distribution
    pub fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            JumpDistribution::Normal { mean, std } => {
                let normal = Normal::new(*mean, *std).unwrap();
                normal.sample(rng)
            }
            JumpDistribution::Exponential { rate } => {
                let exp = Exponential::new(*rate).unwrap();
                exp.sample(rng)
            }
            JumpDistribution::Laplace { location, scale } => {
                let u: f64 = rng.gen_range(-0.5..0.5);
                location - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
            }
            JumpDistribution::Uniform { min, max } => rng.gen_range(*min..*max),
            JumpDistribution::Discrete {
                jumps,
                probabilities,
            } => {
                let u: f64 = rng.gen();
                let mut cumsum = 0.0;
                for (i, &p) in probabilities.iter().enumerate() {
                    cumsum += p;
                    if u < cumsum {
                        return jumps[i];
                    }
                }
                jumps[jumps.len() - 1]
            }
        }
    }

    /// Expected value of jump
    pub fn mean(&self) -> f64 {
        match self {
            JumpDistribution::Normal { mean, .. } => *mean,
            JumpDistribution::Exponential { rate } => 1.0 / rate,
            JumpDistribution::Laplace { location, .. } => *location,
            JumpDistribution::Uniform { min, max } => (min + max) / 2.0,
            JumpDistribution::Discrete {
                jumps,
                probabilities,
            } => jumps
                .iter()
                .zip(probabilities.iter())
                .map(|(j, p)| j * p)
                .sum(),
        }
    }

    /// Variance of jump
    pub fn variance(&self) -> f64 {
        match self {
            JumpDistribution::Normal { std, .. } => std * std,
            JumpDistribution::Exponential { rate } => 1.0 / (rate * rate),
            JumpDistribution::Laplace { scale, .. } => 2.0 * scale * scale,
            JumpDistribution::Uniform { min, max } => {
                let range = max - min;
                range * range / 12.0
            }
            JumpDistribution::Discrete {
                jumps,
                probabilities,
            } => {
                let mean = self.mean();
                jumps
                    .iter()
                    .zip(probabilities.iter())
                    .map(|(j, p)| (j - mean).powi(2) * p)
                    .sum()
            }
        }
    }
}

/// Jump diffusion configuration
pub struct JumpDiffusionConfig {
    /// Drift coefficient μ
    pub drift: f64,
    /// Diffusion coefficient σ
    pub sigma: f64,
    /// Jump intensity λ (expected number of jumps per unit time)
    pub jump_intensity: f64,
    /// Jump size distribution
    pub jump_distribution: JumpDistribution,
    /// Discount rate
    pub rho: f64,
    /// State space bounds
    pub state_bounds: (f64, f64),
    /// Number of grid points
    pub n_points: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for JumpDiffusionConfig {
    fn default() -> Self {
        Self {
            drift: 0.05,
            sigma: 0.2,
            jump_intensity: 0.1, // 0.1 jumps per year on average
            jump_distribution: JumpDistribution::Normal {
                mean: -0.05,
                std: 0.1,
            },
            rho: 0.04,
            state_bounds: (-3.0, 3.0),
            n_points: 300, // Need more points for jumps
            max_iter: 2000,
            tolerance: 1e-6,
        }
    }
}

/// Result from jump diffusion solver
#[derive(Debug, Clone)]
pub struct JumpDiffusionResult {
    /// State space grid
    pub x: Array1<f64>,
    /// Value function V(x)
    pub value: Array1<f64>,
    /// Optimal control u(x)
    pub control: Array1<f64>,
    /// Gradient ∇V(x)
    pub gradient: Array1<f64>,
    /// Jump integral contribution
    pub jump_integral: Array1<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual
    pub residual: f64,
}

/// Jump Diffusion HJB Solver
pub struct JumpDiffusionSolver {
    config: JumpDiffusionConfig,
}

impl JumpDiffusionSolver {
    /// Create new solver
    pub fn new(config: JumpDiffusionConfig) -> Result<Self> {
        // Validation
        if config.sigma <= 0.0 {
            return Err(OptimalControlError::InvalidParameters(
                "sigma must be positive".to_string(),
            ));
        }

        if config.jump_intensity < 0.0 {
            return Err(OptimalControlError::InvalidParameters(
                "jump_intensity must be non-negative".to_string(),
            ));
        }

        if config.rho <= 0.0 {
            return Err(OptimalControlError::InvalidParameters(
                "rho must be positive".to_string(),
            ));
        }

        if config.n_points < 100 {
            return Err(OptimalControlError::InvalidParameters(
                "n_points must be at least 100 for jump models".to_string(),
            ));
        }

        Ok(Self { config })
    }

    /// Solve HJB equation with jumps
    pub fn solve(&self) -> Result<JumpDiffusionResult> {
        let cfg = &self.config;

        // Create state space grid
        let (x_min, x_max) = cfg.state_bounds;
        let dx = (x_max - x_min) / (cfg.n_points - 1) as f64;
        let x = Array1::from_iter((0..cfg.n_points).map(|i| x_min + i as f64 * dx));

        // Precompute jump integral for each grid point
        let jump_kernel = self.compute_jump_kernel(&x, dx)?;

        // Initialize value function
        let mut v = Array1::<f64>::zeros(cfg.n_points);
        let mut v_old = Array1::<f64>::zeros(cfg.n_points);
        let mut u = Array1::<f64>::zeros(cfg.n_points);

        // Iterative solver
        let mut iterations = 0;
        let mut residual = f64::INFINITY;

        for iter in 0..cfg.max_iter {
            v_old.assign(&v);

            // Compute jump integral: λ∫[V(x+y) - V(x)]F(dy)
            let jump_integral = self.apply_jump_integral(&v_old, &jump_kernel);

            // Solve HJB at each grid point (parallel)
            let updates: Vec<(usize, f64, f64)> = (1..cfg.n_points - 1)
                .into_par_iter()
                .map(|i| {
                    let _xi = x[i];

                    // Finite differences
                    let v_c = v_old[i];
                    let v_f = v_old[i + 1];
                    let v_b = v_old[i - 1];

                    let dv_forward = (v_f - v_c) / dx;
                    let dv_backward = (v_c - v_b) / dx;
                    let d2v = (v_f - 2.0 * v_c + v_b) / (dx * dx);

                    // Upwind scheme for drift
                    let drift_term = if cfg.drift >= 0.0 {
                        cfg.drift * dv_backward
                    } else {
                        cfg.drift * dv_forward
                    };

                    // Diffusion term
                    let diffusion_term = 0.5 * cfg.sigma * cfg.sigma * d2v;

                    // Jump term
                    let jump_term = jump_integral[i];

                    // Optimal control (placeholder - problem-specific)
                    let optimal_control = 0.0; // TODO: optimize based on objective

                    // Running cost (placeholder)
                    let cost = 0.0;

                    // HJB: ρV = drift + diffusion + jump + cost
                    let new_value = (drift_term + diffusion_term + jump_term + cost) / cfg.rho;

                    (i, new_value, optimal_control)
                })
                .collect();

            // Apply updates
            for (i, new_value, optimal_control) in updates {
                v[i] = new_value;
                u[i] = optimal_control;
            }

            // Boundary conditions
            v[0] = v[1];
            v[cfg.n_points - 1] = v[cfg.n_points - 2];

            // Check convergence
            residual = (&v - &v_old).mapv(|x| x.abs()).sum() / cfg.n_points as f64;
            iterations = iter + 1;

            if residual < cfg.tolerance {
                break;
            }

            // Relaxation
            let omega = 0.6;
            v = &v * omega + &v_old * (1.0 - omega);
        }

        if residual >= cfg.tolerance {
            return Err(OptimalControlError::ConvergenceError(format!(
                "Failed to converge after {} iterations",
                iterations
            )));
        }

        // Compute final jump integral for output
        let jump_integral = self.apply_jump_integral(&v, &jump_kernel);

        // Compute gradient
        let gradient = self.compute_gradient(&v, dx);

        Ok(JumpDiffusionResult {
            x,
            value: v,
            control: u,
            gradient,
            jump_integral,
            iterations,
            residual,
        })
    }

    /// Precompute jump kernel matrix
    /// K[i,j] = probability of jumping from x[i] to x[j]
    #[allow(unused_variables)]  // dx parameter reserved for future extensions
    fn compute_jump_kernel(&self, x: &Array1<f64>, dx: f64) -> Result<Array2<f64>> {
        let n = x.len();
        let mut kernel = Array2::<f64>::zeros((n, n));

        // Discretize jump distribution
        match &self.config.jump_distribution {
            JumpDistribution::Discrete {
                jumps,
                probabilities,
            } => {
                // Use provided discrete distribution
                for i in 0..n {
                    for (jump, prob) in jumps.iter().zip(probabilities.iter()) {
                        let target_x = x[i] + jump;
                        // Find nearest grid point
                        if let Some(j) = self.find_nearest_index(x, target_x) {
                            kernel[[i, j]] += prob;
                        }
                    }
                }
            }
            _ => {
                // Discretize continuous distribution via Monte Carlo
                use rand::thread_rng;
                let mut rng = thread_rng();
                let n_samples = 10000;

                for i in 0..n {
                    let mut jump_counts = vec![0; n];

                    for _ in 0..n_samples {
                        let jump_size = self.config.jump_distribution.sample(&mut rng);
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
            }
        }

        Ok(kernel)
    }

    /// Apply jump integral: λ∫[V(x+y) - V(x)]F(dy)
    fn apply_jump_integral(&self, v: &Array1<f64>, kernel: &Array2<f64>) -> Array1<f64> {
        let n = v.len();
        let lambda = self.config.jump_intensity;

        let mut result = Array1::<f64>::zeros(n);

        for i in 0..n {
            let mut integral = 0.0;
            for j in 0..n {
                integral += kernel[[i, j]] * (v[j] - v[i]);
            }
            result[i] = lambda * integral;
        }

        result
    }

    /// Find nearest grid index to target value
    fn find_nearest_index(&self, x: &Array1<f64>, target: f64) -> Option<usize> {
        let (x_min, x_max) = self.config.state_bounds;

        if target < x_min || target > x_max {
            return None;
        }

        // Binary search would be better, but for simplicity:
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

    /// Compute gradient
    fn compute_gradient(&self, v: &Array1<f64>, dx: f64) -> Array1<f64> {
        let n = v.len();
        let mut grad = Array1::<f64>::zeros(n);

        // Interior: central differences
        for i in 1..n - 1 {
            grad[i] = (v[i + 1] - v[i - 1]) / (2.0 * dx);
        }

        // Boundaries: one-sided
        grad[0] = (v[1] - v[0]) / dx;
        grad[n - 1] = (v[n - 1] - v[n - 2]) / dx;

        grad
    }

    /// Simulate a jump-diffusion path
    pub fn simulate_path(&self, x0: f64, t_max: f64, dt: f64) -> Result<(Vec<f64>, Vec<f64>)> {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let n_steps = (t_max / dt) as usize;
        let mut times = Vec::with_capacity(n_steps + 1);
        let mut states = Vec::with_capacity(n_steps + 1);

        let mut x = x0;
        let mut t = 0.0;

        times.push(t);
        states.push(x);

        let sqrt_dt = dt.sqrt();

        for _ in 0..n_steps {
            // Diffusion increment
            let dw: f64 = rng.sample(Normal::new(0.0, sqrt_dt).unwrap());
            let diffusion = self.config.drift * dt + self.config.sigma * dw;

            // Jump component (Poisson process)
            let jump_prob = self.config.jump_intensity * dt;
            let jump = if rng.gen::<f64>() < jump_prob {
                self.config.jump_distribution.sample(&mut rng)
            } else {
                0.0
            };

            x += diffusion + jump;
            t += dt;

            times.push(t);
            states.push(x);
        }

        Ok((times, states))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_distribution_moments() {
        let dist = JumpDistribution::Normal {
            mean: 0.5,
            std: 0.1,
        };
        assert!((dist.mean() - 0.5).abs() < 1e-10);
        assert!((dist.variance() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_jump_diffusion_solver() {
        let config = JumpDiffusionConfig {
            jump_intensity: 0.5,
            jump_distribution: JumpDistribution::Normal {
                mean: -0.1,
                std: 0.05,
            },
            ..Default::default()
        };

        let solver = JumpDiffusionSolver::new(config).unwrap();
        let result = solver.solve().unwrap();

        assert_eq!(result.x.len(), 300);
        assert!(result.iterations > 0);
        assert!(result.residual < 1e-6);
    }

    #[test]
    fn test_path_simulation() {
        let config = JumpDiffusionConfig::default();
        let solver = JumpDiffusionSolver::new(config).unwrap();

        let (times, states) = solver.simulate_path(0.0, 1.0, 0.01).unwrap();

        assert_eq!(times.len(), states.len());
        assert_eq!(times.len(), 101);
    }
}
