//! Portfolio Optimization — CARA utility, convex duality, mean-variance.
//!
//! Provides generic trait-based abstractions for:
//! - **CARA** (Constant Absolute Risk Aversion) utility maximisation
//! - **CRRA** (Constant Relative Risk Aversion) utility
//! - **Convex duality** optimization (Legendre-Fenchel transform)
//! - **Mean-variance** portfolio weights (Markowitz)
//!
//! All heavy lifting uses `ndarray` + `rayon` for parallelism.
//!
//! # References
//! - Markowitz (1952) — Portfolio Selection
//! - Merton (1969) — Lifetime Portfolio Selection under Uncertainty
//! - Rockafellar (1970) — Convex Analysis (duality)

pub mod traits;
pub mod cara;
pub mod convex;
pub mod mean_variance;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;
