//! Markov Chain Monte Carlo (MCMC) module
//!
//! Modular implementation of MCMC sampling algorithms with:
//!
//! - Multiple proposal strategies (Gaussian, Adaptive)
//! - Builder pattern for configuration
//! - Metropolis-Hastings algorithm
//! - Diagnostic tools (acceptance rate, autocorrelation)
//! - Python bindings via PyO3
//!
//! # Example
//!
//! ```rust
//! use optimizr::mcmc::{MCMCConfig, MetropolisHastings, GaussianProposal};
//!
//! // Define your log-likelihood function
//! // Create config and sample
//! ```

mod config;
mod likelihood;
mod proposal;
#[cfg(feature = "python-bindings")]
mod python_bindings;
mod sampler;

// Re-export public API
pub use config::{MCMCConfig, MCMCConfigBuilder};
pub use likelihood::LogLikelihood;
#[cfg(feature = "python-bindings")]
pub use likelihood::PyLogLikelihood;
pub use proposal::{AdaptiveProposal, GaussianProposal, ProposalStrategy};
#[cfg(feature = "python-bindings")]
pub use python_bindings::{adaptive_mcmc_sample, mcmc_sample};
pub use sampler::MetropolisHastings;
