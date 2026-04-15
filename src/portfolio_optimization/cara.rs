//! CARA and CRRA utility function implementations.
//!
//! CARA: $U(x) = -\frac{1}{\gamma} e^{-\gamma x}$
//! CRRA: $U(x) = \frac{x^{1-\gamma}}{1-\gamma}$

use super::traits::{PortfolioOptimizer, PortfolioResult, UtilityFunction};
use crate::core::OptimizrError;

// ── CARA Utility ────────────────────────────────────────────────────────────

/// Constant Absolute Risk Aversion utility: U(x) = -exp(-γx) / γ
pub struct CARAUtility {
    pub gamma: f64,
}

impl CARAUtility {
    pub fn new(gamma: f64) -> Result<Self, OptimizrError> {
        if gamma <= 0.0 {
            return Err(OptimizrError::InvalidParameter(
                "CARA gamma must be > 0".into(),
            ));
        }
        Ok(Self { gamma })
    }
}

impl UtilityFunction for CARAUtility {
    fn utility(&self, x: f64) -> f64 {
        -(-self.gamma * x).exp() / self.gamma
    }
    fn marginal_utility(&self, x: f64) -> f64 {
        (-self.gamma * x).exp()
    }
    fn inverse_marginal(&self, y: f64) -> f64 {
        if y <= 0.0 {
            return f64::INFINITY;
        }
        -y.ln() / self.gamma
    }
    fn risk_aversion(&self, _x: f64) -> f64 {
        self.gamma
    }
    fn name(&self) -> &str {
        "CARA"
    }
}

// ── CRRA Utility ────────────────────────────────────────────────────────────

/// Constant Relative Risk Aversion utility.
/// γ ≠ 1: U(x) = x^{1-γ} / (1-γ)
/// γ = 1: U(x) = ln(x)
pub struct CRRAUtility {
    pub gamma: f64,
}

impl CRRAUtility {
    pub fn new(gamma: f64) -> Result<Self, OptimizrError> {
        if gamma <= 0.0 {
            return Err(OptimizrError::InvalidParameter(
                "CRRA gamma must be > 0".into(),
            ));
        }
        Ok(Self { gamma })
    }
}

impl UtilityFunction for CRRAUtility {
    fn utility(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if (self.gamma - 1.0).abs() < 1e-12 {
            x.ln()
        } else {
            x.powf(1.0 - self.gamma) / (1.0 - self.gamma)
        }
    }
    fn marginal_utility(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        x.powf(-self.gamma)
    }
    fn inverse_marginal(&self, y: f64) -> f64 {
        if y <= 0.0 {
            return f64::INFINITY;
        }
        y.powf(-1.0 / self.gamma)
    }
    fn risk_aversion(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        self.gamma / x
    }
    fn name(&self) -> &str {
        "CRRA"
    }
}

// ── CARA Portfolio Optimizer ────────────────────────────────────────────────

/// CARA portfolio optimizer.
///
/// Maximizes $w^T \mu - \frac{\gamma}{2} w^T \Sigma w$
/// subject to $\sum w_i = 1$, $0 \le w_i \le w_{\max}$.
pub struct CARAOptimizer {
    pub gamma: f64,
}

impl CARAOptimizer {
    pub fn new(gamma: f64) -> Result<Self, OptimizrError> {
        if gamma <= 0.0 {
            return Err(OptimizrError::InvalidParameter(
                "CARA gamma must be > 0".into(),
            ));
        }
        Ok(Self { gamma })
    }
}

impl PortfolioOptimizer for CARAOptimizer {
    fn optimize(
        &self,
        mu: &[f64],
        cov: &[Vec<f64>],
        max_weight: f64,
    ) -> Result<PortfolioResult, OptimizrError> {
        let n = mu.len();
        if n == 0 {
            return Err(OptimizrError::EmptyData);
        }
        if cov.len() != n {
            return Err(OptimizrError::DimensionMismatch {
                expected: n,
                actual: cov.len(),
            });
        }

        let max_iter = 2000;
        let lr = 0.01;
        let tol = 1e-8;
        let mut w = vec![1.0 / n as f64; n];

        for iter_count in 0..max_iter {
            // ∇[-U] = -μ + γΣw
            let mut grad = vec![0.0; n];
            for i in 0..n {
                grad[i] = -mu[i];
                for j in 0..n {
                    grad[i] += self.gamma * cov[i][j] * w[j];
                }
            }

            // Gradient step
            let mut w_new: Vec<f64> = (0..n).map(|i| w[i] - lr * grad[i]).collect();

            // Project onto box [0, max_weight]
            for v in w_new.iter_mut() {
                *v = v.max(0.0).min(max_weight);
            }

            // Project onto simplex (normalise to sum = 1)
            let sum: f64 = w_new.iter().sum();
            if sum > 1e-15 {
                for v in w_new.iter_mut() {
                    *v /= sum;
                }
            }
            // Re-clip after normalisation
            for v in w_new.iter_mut() {
                *v = v.min(max_weight);
            }
            let sum2: f64 = w_new.iter().sum();
            if sum2 > 1e-15 {
                for v in w_new.iter_mut() {
                    *v /= sum2;
                }
            }

            let diff: f64 = w
                .iter()
                .zip(w_new.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            w = w_new;

            if diff < tol {
                let (ret, var) = portfolio_stats(&w, mu, cov);
                return Ok(PortfolioResult {
                    weights: w,
                    utility: ret - 0.5 * self.gamma * var,
                    expected_return: ret,
                    portfolio_variance: var,
                    iterations: iter_count + 1,
                    converged: true,
                });
            }
        }

        let (ret, var) = portfolio_stats(&w, mu, cov);
        Ok(PortfolioResult {
            weights: w,
            utility: ret - 0.5 * self.gamma * var,
            expected_return: ret,
            portfolio_variance: var,
            iterations: max_iter,
            converged: false,
        })
    }
}

/// Compute portfolio expected return and variance.
pub fn portfolio_stats(w: &[f64], mu: &[f64], cov: &[Vec<f64>]) -> (f64, f64) {
    let n = w.len();
    let ret: f64 = (0..n).map(|i| w[i] * mu[i]).sum();
    let var: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| w[i] * w[j] * cov[i][j]))
        .sum();
    (ret, var)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cara_utility_basic() {
        let u = CARAUtility::new(2.0).unwrap();
        assert!((u.utility(0.0) - (-0.5)).abs() < 1e-10);
        assert!((u.marginal_utility(0.0) - 1.0).abs() < 1e-10);
        assert!((u.risk_aversion(42.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_crra_utility_log() {
        let u = CRRAUtility::new(1.0).unwrap();
        let val = u.utility(std::f64::consts::E);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cara_optimizer_equal() {
        // Equal means, no covariance → equal weights
        let mu = vec![0.01, 0.01, 0.01];
        let cov = vec![
            vec![0.04, 0.0, 0.0],
            vec![0.0, 0.04, 0.0],
            vec![0.0, 0.0, 0.04],
        ];
        let opt = CARAOptimizer::new(2.0).unwrap();
        let res = opt.optimize(&mu, &cov, 0.5).unwrap();
        assert!(res.converged);
        for w in &res.weights {
            assert!((w - 1.0 / 3.0).abs() < 0.05);
        }
    }
}
