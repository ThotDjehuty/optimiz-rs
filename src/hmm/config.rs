//! Configuration and builder for HMM training
//!
//! Provides flexible configuration options using the builder pattern.

use super::emission::EmissionModel;
use crate::core::{OptimizrError, Result};

/// HMM Configuration
#[derive(Clone)]
pub struct HMMConfig<E: EmissionModel> {
    pub n_states: usize,
    pub n_iterations: usize,
    pub tolerance: f64,
    pub emission_model: E,
    pub use_parallel: bool,
}

impl<E: EmissionModel> HMMConfig<E> {
    /// Create a new configuration builder
    pub fn builder(n_states: usize) -> HMMConfigBuilder<E> {
        HMMConfigBuilder::new(n_states)
    }
}

/// Builder pattern for HMM configuration
pub struct HMMConfigBuilder<E: EmissionModel> {
    n_states: usize,
    n_iterations: usize,
    tolerance: f64,
    emission_model: Option<E>,
    use_parallel: bool,
}

impl<E: EmissionModel> HMMConfigBuilder<E> {
    /// Create a new configuration builder
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            n_iterations: 100,
            tolerance: 1e-6,
            emission_model: None,
            use_parallel: cfg!(feature = "parallel"),
        }
    }

    /// Set the number of EM iterations
    pub fn iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set custom emission model
    pub fn emission_model(mut self, model: E) -> Self {
        self.emission_model = Some(model);
        self
    }

    /// Enable/disable parallel computation
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled && cfg!(feature = "parallel");
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<HMMConfig<E>>
    where
        E: EmissionModel + Default,
    {
        if self.n_states < 2 {
            return Err(OptimizrError::InvalidParameter(
                "n_states must be at least 2".to_string(),
            ));
        }

        Ok(HMMConfig {
            n_states: self.n_states,
            n_iterations: self.n_iterations,
            tolerance: self.tolerance,
            emission_model: self.emission_model.unwrap_or_default(),
            use_parallel: self.use_parallel,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm::emission::GaussianEmission;

    #[test]
    fn test_config_builder() {
        let config = HMMConfig::<GaussianEmission>::builder(3)
            .iterations(50)
            .tolerance(1e-5)
            .build()
            .unwrap();

        assert_eq!(config.n_states, 3);
        assert_eq!(config.n_iterations, 50);
        assert_eq!(config.tolerance, 1e-5);
    }

    #[test]
    fn test_invalid_states() {
        let result = HMMConfig::<GaussianEmission>::builder(1).build();
        assert!(result.is_err());
    }
}
