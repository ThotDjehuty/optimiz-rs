//! Backward Stochastic Differential Equations (BSDE)
//! ===================================================
//!
//! Generic numerical schemes for BSDEs of the form
//!
//! ```text
//! -dY_t = f(t, Y_t, Z_t) dt - Z_t · dW_t,    Y_T = g(X_T)
//! ```
//!
//! where `Y` is an adapted real-valued process and `Z` is its predictable
//! integrand.  Implementations rely solely on CPU-side ndarray primitives.
//!
//! Modules:
//!
//! - [`theta_scheme`] — implicit/explicit time-stepping with parameter θ ∈ [0,1]
//! - [`deep_bsde_bridge`] — abstract trait providing a hook for an external
//!   neural-network calibrator (the bridge itself does **not** ship a deep
//!   learning runtime — it exposes the conditional expectation interface that
//!   higher-level frameworks plug into).

pub mod theta_scheme;
pub mod deep_bsde_bridge;

pub use theta_scheme::{ThetaSchemeConfig, ThetaSchemeResult, solve_linear_bsde};
pub use deep_bsde_bridge::{ConditionalExpectation, DeepBsdeBridge, DeepBsdeStep};
