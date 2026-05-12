//! Stochastic control primitives
//! ===============================
//!
//! - [`optimal_switching`] — backward dynamic-programming Snell envelope for
//!   discrete-time multi-mode optimal switching with mode-dependent running
//!   reward and switching costs.
//! - [`pontryagin`] — forward shooting solver for the Pontryagin maximum
//!   principle on a 1-D controlled SDE with quadratic cost.
//! - [`two_sided_intensity_control`] — generic bilateral intensity control
//!   for a doubly-controlled jump process; computes the optimal symmetric
//!   intensities from the value function gradient.

pub mod optimal_switching;
pub mod pontryagin;
pub mod two_sided_intensity_control;
#[cfg(feature = "python-bindings")]
pub mod python_bindings;

pub use optimal_switching::{SwitchingConfig, SwitchingResult, solve_optimal_switching};
pub use pontryagin::{PontryaginConfig, PontryaginResult, solve_pontryagin_lqr};
pub use two_sided_intensity_control::{
    TwoSidedConfig, TwoSidedIntensities, optimal_two_sided_intensities,
};
