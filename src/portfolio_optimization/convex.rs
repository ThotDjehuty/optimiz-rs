//! Convex optimisation via projected gradient descent.
//!
//! Implements constrained optimisation over the simplex with box constraints.
//! Supports generic `ConvexObjective` + `ConvexConstraint` traits.

use super::traits::ConvexObjective;
use crate::core::OptimizrError;

/// Result of convex optimisation.
#[derive(Debug, Clone)]
pub struct ConvexResult {
    pub x: Vec<f64>,
    pub objective_value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub gradient_norm: f64,
}

/// Projected gradient descent solver for convex problems on the simplex.
///
/// Solves: $\min f(w)$ subject to $\sum w_i = 1$, $l \le w_i \le u$.
pub struct ProjectedGradientSolver {
    pub max_iter: usize,
    pub learning_rate: f64,
    pub tolerance: f64,
    pub box_lower: f64,
    pub box_upper: f64,
}

impl Default for ProjectedGradientSolver {
    fn default() -> Self {
        Self {
            max_iter: 2000,
            learning_rate: 0.1,
            tolerance: 1e-8,
            box_lower: 0.0,
            box_upper: 1.0,
        }
    }
}

impl ProjectedGradientSolver {
    pub fn new(max_iter: usize, lr: f64, tol: f64, lower: f64, upper: f64) -> Self {
        Self {
            max_iter,
            learning_rate: lr,
            tolerance: tol,
            box_lower: lower,
            box_upper: upper,
        }
    }

    /// Solve min f(x) subject to Σx_i = 1, lower ≤ x_i ≤ upper.
    pub fn solve(&self, objective: &dyn ConvexObjective) -> Result<ConvexResult, OptimizrError> {
        let n = objective.dim();
        if n == 0 {
            return Err(OptimizrError::EmptyData);
        }

        let mut x = vec![1.0 / n as f64; n];
        let mut best_val = f64::INFINITY;
        let mut best_x = x.clone();

        for iter in 0..self.max_iter {
            let grad = objective.gradient(&x);
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            let current_val = objective.value(&x);

            if current_val < best_val {
                best_val = current_val;
                best_x = x.clone();
            }

            // Gradient step
            let mut x_new: Vec<f64> = x
                .iter()
                .zip(grad.iter())
                .map(|(xi, gi)| xi - self.learning_rate * gi)
                .collect();

            // Project onto box
            for xi in x_new.iter_mut() {
                *xi = xi.max(self.box_lower).min(self.box_upper);
            }

            // Project onto simplex
            let sum: f64 = x_new.iter().sum();
            if sum > 1e-15 {
                for xi in x_new.iter_mut() {
                    *xi /= sum;
                }
            }

            // Re-clip after normalisation
            for xi in x_new.iter_mut() {
                *xi = xi.max(self.box_lower).min(self.box_upper);
            }
            let sum2: f64 = x_new.iter().sum();
            if sum2 > 1e-15 {
                for xi in x_new.iter_mut() {
                    *xi /= sum2;
                }
            }

            let diff: f64 = x
                .iter()
                .zip(x_new.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            x = x_new;

            if diff < self.tolerance {
                return Ok(ConvexResult {
                    x: best_x.clone(),
                    objective_value: best_val,
                    iterations: iter + 1,
                    converged: true,
                    gradient_norm: grad_norm,
                });
            }
        }

        Ok(ConvexResult {
            x: best_x,
            objective_value: best_val,
            iterations: self.max_iter,
            converged: false,
            gradient_norm: 0.0,
        })
    }
}

/// Mean-variance objective: min γ/2 w'Σw - w'μ  (+ optional score tilting).
pub struct MeanVarianceObjective {
    pub mu: Vec<f64>,
    pub cov: Vec<Vec<f64>>,
    pub gamma: f64,
    pub score_weights: Option<Vec<f64>>,
}

impl ConvexObjective for MeanVarianceObjective {
    fn value(&self, w: &[f64]) -> f64 {
        let n = w.len();
        let ret: f64 = (0..n).map(|i| w[i] * self.mu[i]).sum();
        let var: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| w[i] * w[j] * self.cov[i][j]))
            .sum();
        let mut val = 0.5 * self.gamma * var - ret;
        if let Some(ref scores) = self.score_weights {
            let bonus: f64 = (0..n.min(scores.len())).map(|i| w[i] * scores[i]).sum();
            val -= 0.1 * bonus;
        }
        val
    }

    fn gradient(&self, w: &[f64]) -> Vec<f64> {
        let n = w.len();
        let mut grad = vec![0.0; n];
        for i in 0..n {
            grad[i] = -self.mu[i];
            for j in 0..n {
                grad[i] += self.gamma * self.cov[i][j] * w[j];
            }
            if let Some(ref scores) = self.score_weights {
                if i < scores.len() {
                    grad[i] -= 0.1 * scores[i];
                }
            }
        }
        grad
    }

    fn dim(&self) -> usize {
        self.mu.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_diagonal_cov() {
        let obj = MeanVarianceObjective {
            mu: vec![0.05, 0.03],
            cov: vec![vec![0.04, 0.0], vec![0.0, 0.01]],
            gamma: 2.0,
            score_weights: None,
        };
        let solver = ProjectedGradientSolver {
            box_upper: 0.8,
            ..Default::default()
        };
        let res = solver.solve(&obj).unwrap();
        assert!(res.converged);
        let total: f64 = res.x.iter().sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}
