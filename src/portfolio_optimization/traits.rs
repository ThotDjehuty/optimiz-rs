//! Core traits for portfolio utility and convex optimisation.

use crate::core::OptimizrError;

/// A utility function U(x) mapping wealth → utility.
pub trait UtilityFunction: Send + Sync {
    /// U(x) — the utility of wealth level x.
    fn utility(&self, x: f64) -> f64;

    /// U'(x) — first derivative (marginal utility).
    fn marginal_utility(&self, x: f64) -> f64;

    /// (U')^{-1}(y) — inverse marginal utility (used in duality).
    fn inverse_marginal(&self, y: f64) -> f64;

    /// Risk-aversion coefficient A(x) = -U''(x) / U'(x).
    fn risk_aversion(&self, x: f64) -> f64;

    /// Name identifier for logging / serialisation.
    fn name(&self) -> &str;
}

/// Convex objective f(w) over portfolio weights w ∈ ℝ^n.
///
/// Used by convex solvers (projected gradient, ADMM, etc.).
pub trait ConvexObjective: Send + Sync {
    /// f(w) — objective value.
    fn value(&self, w: &[f64]) -> f64;

    /// ∇f(w) — gradient vector.
    fn gradient(&self, w: &[f64]) -> Vec<f64>;

    /// Dimension of the weight vector.
    fn dim(&self) -> usize;
}

/// Convex constraint g(w) ≤ 0.
pub trait ConvexConstraint: Send + Sync {
    /// g(w) — constraint value (feasible when ≤ 0).
    fn value(&self, w: &[f64]) -> f64;

    /// ∇g(w) — gradient of constraint function.
    fn gradient(&self, w: &[f64]) -> Vec<f64>;
}

/// Result of a portfolio optimisation.
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    pub weights: Vec<f64>,
    pub utility: f64,
    pub expected_return: f64,
    pub portfolio_variance: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl PortfolioResult {
    pub fn sharpe_ratio(&self, risk_free: f64) -> f64 {
        let vol = self.portfolio_variance.sqrt();
        if vol < 1e-15 {
            return 0.0;
        }
        (self.expected_return - risk_free) / vol
    }
}

/// Generic portfolio optimiser trait.
pub trait PortfolioOptimizer: Send + Sync {
    fn optimize(
        &self,
        mu: &[f64],
        cov: &[Vec<f64>],
        max_weight: f64,
    ) -> Result<PortfolioResult, OptimizrError>;
}
