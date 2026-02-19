//! Point Processes Module for Order Flow Modeling
//!
//! This module implements the mathematical framework from "A Unified Theory of
//! Order Flow, Market Impact, and Volatility" (Muhle-Karbe et al., 2026).
//!
//! # Overview
//!
//! The module provides tools for modeling order flow in financial markets using:
//!
//! - **Hawkes Processes**: Self-exciting point processes for core and reaction flows
//! - **Excitation Kernels**: Power-law and exponential kernels for temporal dependence
//! - **Mixed Fractional Brownian Motion**: Scaling limits of aggregate order flow
//! - **Mittag-Leffler Functions**: Key functions for scaling limit analysis
//! - **Order Flow Analysis**: Signed/unsigned flow, Hurst estimation, market impact
//!
//! # Key Relationships (Unified Theory)
//!
//! All quantities are determined by a single parameter H₀ ≈ 3/4:
//!
//! - Signed order flow: Hurst index H₀
//! - Unsigned volume: Hurst index H₀ - 1/2 (rough, ~0.25)
//! - Volatility: Hurst index 2H₀ - 3/2 (~0 for H₀=3/4)
//! - Market impact: power law exponent 2 - 2H₀ (~0.5, square-root law)
//!
//! # Example
//!
//! ```rust,ignore
//! use optimizr::point_processes::{
//!     HawkesProcess, PowerLawKernel, OrderFlowAnalyzer
//! };
//!
//! // Create a Hawkes process for core order flow
//! let kernel = PowerLawKernel::new(0.5, 1.0);  // α₀ = 0.5, K₀ = 1.0
//! let hawkes = HawkesProcess::new(0.1, kernel);  // ν = 0.1
//!
//! // Simulate order arrivals
//! let arrivals = hawkes.simulate(1000.0, Some(42));
//!
//! // Analyze order flow
//! let analyzer = OrderFlowAnalyzer::new();
//! let h0 = analyzer.estimate_h0(&arrivals);
//! println!("Estimated H₀: {:.4}", h0);
//! ```

mod kernels;
mod hawkes;
mod mittag_leffler;
mod mixed_fbm;
mod order_flow;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

// Re-export public API
pub use kernels::{ExcitationKernel, PowerLawKernel, ExponentialKernel};
pub use hawkes::{HawkesProcess, HawkesProcessConfig, BivariateHawkes};
pub use mittag_leffler::{mittag_leffler, mittag_leffler_derivative, f_alpha_lambda};
pub use mixed_fbm::{MixedFractionalBM, FractionalBM};
pub use order_flow::{
    OrderFlowAnalyzer, OrderFlowMetrics, UnifiedTheoryParams,
    signed_order_flow, unsigned_volume, market_impact_exponent,
    volatility_hurst, volume_hurst
};
