//! Statistical inference primitives (v2.0.0)
//! ==========================================
//!
//! Currently exposes:
//! - [`robust_drift`] — Huber-loss robust drift estimator for a stationary
//!   1-D OU process observed on a uniform grid.

pub mod robust_drift;

pub use robust_drift::{RobustDriftConfig, RobustDriftResult, estimate_robust_drift};
