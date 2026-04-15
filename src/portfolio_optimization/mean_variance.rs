//! Classic mean-variance (Markowitz) portfolio optimisation + ERC.

use super::cara::portfolio_stats;
use super::convex::{MeanVarianceObjective, ProjectedGradientSolver};
use super::traits::{PortfolioOptimizer, PortfolioResult};
use crate::core::OptimizrError;

/// Markowitz mean-variance optimizer with optional score tilting.
pub struct MeanVarianceOptimizer {
    pub risk_aversion: f64,
    pub scores: Option<Vec<f64>>,
}

impl MeanVarianceOptimizer {
    pub fn new(risk_aversion: f64) -> Self {
        Self {
            risk_aversion,
            scores: None,
        }
    }

    pub fn with_scores(mut self, scores: Vec<f64>) -> Self {
        self.scores = Some(scores);
        self
    }
}

impl PortfolioOptimizer for MeanVarianceOptimizer {
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

        let obj = MeanVarianceObjective {
            mu: mu.to_vec(),
            cov: cov.to_vec(),
            gamma: self.risk_aversion,
            score_weights: self.scores.clone(),
        };

        let solver = ProjectedGradientSolver {
            box_upper: max_weight,
            ..Default::default()
        };

        let res = solver.solve(&obj)?;
        let (ret, var) = portfolio_stats(&res.x, mu, cov);

        Ok(PortfolioResult {
            weights: res.x,
            utility: ret - 0.5 * self.risk_aversion * var,
            expected_return: ret,
            portfolio_variance: var,
            iterations: res.iterations,
            converged: res.converged,
        })
    }
}

/// Minimum variance portfolio (γ → ∞, ignores expected returns).
pub fn minimum_variance(
    cov: &[Vec<f64>],
    max_weight: f64,
) -> Result<PortfolioResult, OptimizrError> {
    let n = cov.len();
    if n == 0 {
        return Err(OptimizrError::EmptyData);
    }
    let mu = vec![0.0; n];
    let opt = MeanVarianceOptimizer {
        risk_aversion: 100.0,
        scores: None,
    };
    opt.optimize(&mu, cov, max_weight)
}

/// Equal-Risk-Contribution (ERC / Risk Parity) portfolio.
///
/// Iterates: $w_i \propto 1 / (\Sigma w)_i$.
pub fn equal_risk_contribution(
    cov: &[Vec<f64>],
    max_weight: f64,
) -> Result<PortfolioResult, OptimizrError> {
    let n = cov.len();
    if n == 0 {
        return Err(OptimizrError::EmptyData);
    }

    let mut w = vec![1.0 / n as f64; n];
    let max_iter = 500;

    for _ in 0..max_iter {
        // Marginal risk contribution: (Σw)_i
        let mut mrc = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                mrc[i] += cov[i][j] * w[j];
            }
        }

        // New weights ∝ 1/|mrc_i|
        let mut w_new: Vec<f64> = mrc
            .iter()
            .map(|m| {
                if m.abs() > 1e-15 {
                    1.0 / m.abs()
                } else {
                    1.0
                }
            })
            .collect();

        // Normalise
        let sum: f64 = w_new.iter().sum();
        for v in w_new.iter_mut() {
            *v /= sum;
        }
        // Clip
        for v in w_new.iter_mut() {
            *v = v.min(max_weight);
        }
        let sum2: f64 = w_new.iter().sum();
        for v in w_new.iter_mut() {
            *v /= sum2;
        }

        let diff: f64 = w
            .iter()
            .zip(w_new.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        w = w_new;
        if diff < 1e-10 {
            break;
        }
    }

    let var: f64 = {
        let w_ref = &w;
        (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| w_ref[i] * w_ref[j] * cov[i][j])
            .sum()
    };

    Ok(PortfolioResult {
        weights: w,
        utility: -var,
        expected_return: 0.0,
        portfolio_variance: var,
        iterations: max_iter,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_variance_identity_cov() {
        let mu = vec![0.10, 0.05, 0.08];
        let cov = vec![
            vec![0.04, 0.0, 0.0],
            vec![0.0, 0.04, 0.0],
            vec![0.0, 0.0, 0.04],
        ];
        let opt = MeanVarianceOptimizer::new(2.0);
        let res = opt.optimize(&mu, &cov, 0.5).unwrap();
        // Highest mu (0.10) should get the largest weight
        assert!(res.weights[0] > res.weights[1]);
        assert!(res.converged);
    }

    #[test]
    fn test_erc_diagonal() {
        let cov = vec![
            vec![0.04, 0.0, 0.0],
            vec![0.0, 0.04, 0.0],
            vec![0.0, 0.0, 0.04],
        ];
        let res = equal_risk_contribution(&cov, 0.5).unwrap();
        // Identical variances → equal weights
        for w in &res.weights {
            assert!((w - 1.0 / 3.0).abs() < 0.01);
        }
    }
}
