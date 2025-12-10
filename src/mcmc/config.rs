//! MCMC configuration and builder pattern
//!
//! Provides flexible configuration for MCMC sampling.

use super::proposal::ProposalStrategy;
use crate::core::{OptimizrError, Result};

/// MCMC Configuration
#[derive(Clone)]
pub struct MCMCConfig<P: ProposalStrategy> {
    pub n_samples: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub initial_state: Vec<f64>,
    pub proposal: P,
    pub adaptation_interval: usize,
}

/// Builder for MCMC configuration
pub struct MCMCConfigBuilder<P: ProposalStrategy> {
    n_samples: usize,
    burn_in: usize,
    thin: usize,
    initial_state: Vec<f64>,
    proposal: Option<P>,
    adaptation_interval: usize,
}

impl<P: ProposalStrategy> MCMCConfigBuilder<P> {
    pub fn new(n_samples: usize, initial_state: Vec<f64>) -> Self {
        Self {
            n_samples,
            burn_in: n_samples / 10,
            thin: 1,
            initial_state,
            proposal: None,
            adaptation_interval: 100,
        }
    }

    pub fn burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    pub fn thin(mut self, thin: usize) -> Self {
        self.thin = thin.max(1);
        self
    }

    pub fn proposal(mut self, proposal: P) -> Self {
        self.proposal = Some(proposal);
        self
    }

    pub fn adaptation_interval(mut self, interval: usize) -> Self {
        self.adaptation_interval = interval;
        self
    }

    pub fn build(self) -> Result<MCMCConfig<P>>
    where
        P: Default,
    {
        if self.n_samples == 0 {
            return Err(OptimizrError::InvalidParameter(
                "n_samples must be positive".to_string(),
            ));
        }

        if self.initial_state.is_empty() {
            return Err(OptimizrError::InvalidParameter(
                "initial_state cannot be empty".to_string(),
            ));
        }

        Ok(MCMCConfig {
            n_samples: self.n_samples,
            burn_in: self.burn_in,
            thin: self.thin,
            initial_state: self.initial_state,
            proposal: self.proposal.unwrap_or_default(),
            adaptation_interval: self.adaptation_interval,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcmc::proposal::GaussianProposal;

    #[test]
    fn test_config_builder() {
        let config = MCMCConfigBuilder::<GaussianProposal>::new(1000, vec![0.0, 0.0])
            .burn_in(100)
            .thin(2)
            .build()
            .unwrap();

        assert_eq!(config.n_samples, 1000);
        assert_eq!(config.burn_in, 100);
        assert_eq!(config.thin, 2);
    }

    #[test]
    fn test_invalid_config() {
        let result = MCMCConfigBuilder::<GaussianProposal>::new(0, vec![0.0]).build();
        assert!(result.is_err());
    }
}
