//! Hidden Markov Model module
//!
//! Modular implementation of HMM with trait-based design for flexibility
//! and code reusability. This module provides:
//!
//! - Multiple emission models (Gaussian, discrete, etc.)
//! - Builder pattern for configuration
//! - Baum-Welch (EM) training algorithm
//! - Viterbi decoding
//! - Python bindings via PyO3
//!
//! # Example
//!
//! ```rust
//! use optimizr::hmm::{HMMConfig, HMM, GaussianEmission};
//!
//! let config = HMMConfig::<GaussianEmission>::builder(3)
//!     .iterations(100)
//!     .tolerance(1e-6)
//!     .build()
//!     .unwrap();
//!
//! let mut hmm = HMM::new(config);
//! let observations = vec![/* your data */];
//! hmm.fit(&observations).unwrap();
//! let states = hmm.viterbi(&observations).unwrap();
//! ```

mod config;
mod emission;
mod model;
#[cfg(feature = "python-bindings")]
mod python_bindings;
mod viterbi;

// Re-export public API
pub use config::{HMMConfig, HMMConfigBuilder};
pub use emission::{EmissionModel, GaussianEmission};
pub use model::HMM;
#[cfg(feature = "python-bindings")]
pub use python_bindings::{fit_hmm, viterbi_decode, HMMParams};
