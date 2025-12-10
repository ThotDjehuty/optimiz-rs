//! Proposal strategies for MCMC sampling
//!
//! Defines the ProposalStrategy trait and common implementations.

use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::Normal;

/// Trait for MCMC proposal strategies
pub trait ProposalStrategy: Send + Sync + Clone {
    /// Generate proposed next state from current state
    fn propose(&self, current: &[f64], rng: &mut impl Rng) -> Vec<f64>;

    /// Adapt proposal based on acceptance rate (optional)
    fn adapt(&mut self, _acceptance_rate: f64) {}

    /// Name of the strategy
    fn name(&self) -> &'static str;
}

/// Gaussian random walk proposal
#[derive(Clone, Debug)]
pub struct GaussianProposal {
    pub step_size: f64,
}

impl GaussianProposal {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl ProposalStrategy for GaussianProposal {
    fn propose(&self, current: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let normal = Normal::new(0.0, self.step_size).unwrap();
        current.iter().map(|&x| x + normal.sample(rng)).collect()
    }

    fn name(&self) -> &'static str {
        "GaussianRandomWalk"
    }
}

impl Default for GaussianProposal {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Adaptive proposal that adjusts step size based on acceptance rate
#[derive(Clone, Debug)]
pub struct AdaptiveProposal {
    pub step_size: f64,
    pub target_acceptance: f64,
    pub adaptation_rate: f64,
}

impl AdaptiveProposal {
    pub fn new(initial_step: f64) -> Self {
        Self {
            step_size: initial_step,
            target_acceptance: 0.234, // Optimal for multivariate Gaussian
            adaptation_rate: 0.01,
        }
    }

    pub fn with_target_acceptance(mut self, target: f64) -> Self {
        self.target_acceptance = target;
        self
    }
}

impl ProposalStrategy for AdaptiveProposal {
    fn propose(&self, current: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let normal = Normal::new(0.0, self.step_size).unwrap();
        current.iter().map(|&x| x + normal.sample(rng)).collect()
    }

    fn adapt(&mut self, acceptance_rate: f64) {
        let delta = (acceptance_rate - self.target_acceptance) * self.adaptation_rate;
        self.step_size *= (1.0 + delta).max(0.5).min(2.0);
    }

    fn name(&self) -> &'static str {
        "AdaptiveGaussian"
    }
}

impl Default for AdaptiveProposal {
    fn default() -> Self {
        Self::new(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_gaussian_proposal() {
        let proposal = GaussianProposal::new(0.5);
        let current = vec![0.0, 1.0];
        let mut rng = thread_rng();

        let proposed = proposal.propose(&current, &mut rng);
        assert_eq!(proposed.len(), 2);
    }

    #[test]
    fn test_adaptive_proposal() {
        let mut proposal = AdaptiveProposal::new(0.1);
        let initial_step = proposal.step_size;

        // High acceptance should increase step size
        proposal.adapt(0.5);
        assert!(proposal.step_size > initial_step);

        // Low acceptance should decrease step size
        let current_step = proposal.step_size;
        proposal.adapt(0.1);
        assert!(proposal.step_size < current_step);
    }
}
