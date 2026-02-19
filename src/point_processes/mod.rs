//! Point Processes Module
//!
//! This module provides general-purpose implementations of point processes,
//! fractional Brownian motion, and related mathematical tools.
//!
//! # Overview
//!
//! - **Hawkes Processes**: Self-exciting point processes with flexible kernels
//! - **Excitation Kernels**: Power-law and exponential kernels for temporal dependence
//! - **Fractional Brownian Motion (fBM)**: Long-memory Gaussian processes
//! - **Mixed Fractional Brownian Motion (mfBM)**: BM + fBM combinations
//! - **Mittag-Leffler Functions**: Special functions for scaling limit analysis
//! - **Hurst Estimation**: R/S analysis and scale-dependent methods
//!
//! # Mathematical Background
//!
//! ## Hawkes Processes
//!
//! A Hawkes process is a self-exciting point process with intensity:
//!
//! $$\lambda(t) = \nu + \sum_{t_i < t} \phi(t - t_i)$$
//!
//! where $\nu$ is the baseline intensity and $\phi$ is the excitation kernel.
//!
//! Supported kernels:
//! - **Exponential**: $\phi(t) = \alpha e^{-\beta t}$ (short memory)
//! - **Power-law**: $\phi(t) = K_0 (1+t)^{-(1+\alpha_0)}$ (long memory)
//!
//! ## Fractional Brownian Motion
//!
//! fBM $B^H_t$ with Hurst parameter $H \in (0,1)$ has covariance:
//!
//! $$Cov(B^H_s, B^H_t) = \frac{1}{2}(|t|^{2H} + |s|^{2H} - |t-s|^{2H})$$
//!
//! - $H < 0.5$: Anti-persistent (mean-reverting)
//! - $H = 0.5$: Standard Brownian motion
//! - $H > 0.5$: Persistent (trending)
//!
//! # Example
//!
//! ```rust,ignore
//! use optimizr::point_processes::{
//!     HawkesProcess, PowerLawKernel, FractionalBM
//! };
//!
//! // Create a Hawkes process with power-law kernel
//! let kernel = PowerLawKernel::new(0.5, 1.0);  // α₀ = 0.5, K₀ = 1.0
//! let hawkes = HawkesProcess::new(0.1, kernel);  // baseline ν = 0.1
//!
//! // Simulate point process
//! let arrivals = hawkes.simulate(1000.0, Some(42));
//!
//! // Simulate fBM with H = 0.75
//! let mut fbm = FractionalBM::new(0.75);
//! let path = fbm.simulate_hosking(1000, 1.0, Some(42));
//!
//! // Estimate Hurst exponent
//! let h_est = FractionalBM::estimate_hurst(&path);
//! ```

mod kernels;
mod hawkes;
mod mittag_leffler;
mod mixed_fbm;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

// Re-export public API
pub use kernels::{ExcitationKernel, PowerLawKernel, ExponentialKernel};
pub use hawkes::{HawkesProcess, HawkesProcessConfig, BivariateHawkes};
pub use mittag_leffler::{mittag_leffler, mittag_leffler_derivative, f_alpha_lambda};
pub use mixed_fbm::{MixedFractionalBM, FractionalBM};
