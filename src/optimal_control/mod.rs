//! Optimal Control Module
//! ======================
//!
//! Generic optimal control algorithms for dynamical systems.
//!
//! # Features
//!
//! - Generic HJB PDE solver with finite differences
//! - Viscosity solutions for non-smooth value functions
//! - Upwind schemes for numerical stability
//! - Parallel processing support
//!
//! # Mathematical Foundation
//!
//! ## Hamilton-Jacobi-Bellman Equation
//!
//! For a general stochastic process dX_t = μ(X_t)dt + σ(X_t)dW_t,
//! the HJB equation is:
//!
//! ρV(x) = sup_u [μ(x,u)V'(x) + (σ²(x,u)/2)V''(x) + L(x,u)]
//!
//! where:
//! - V(x) is the value function
//! - ρ is the discount rate
//! - u is the control
//! - L(x,u) is the running cost
//!
//! ## Viscosity Solutions
//!
//! Handle non-smooth value functions via:
//! 1. Finite difference discretization
//! 2. Upwind schemes for stability
//! 3. Iterative convergence to viscosity solution

pub mod hjb_solver;
pub mod jump_diffusion;
pub mod mrsjd;
pub mod regime_switching;
pub mod viscosity;

pub use hjb_solver::{HJBConfig, HJBResult, HJBSolver};
pub use jump_diffusion::{
    JumpDiffusionConfig, JumpDiffusionResult, JumpDiffusionSolver, JumpDistribution,
};
pub use mrsjd::{MRSJDConfig, MRSJDResult, MRSJDSolver, RegimeJumpParameters};
pub use regime_switching::{
    RegimeParameters, RegimeSwitchingConfig, RegimeSwitchingResult, RegimeSwitchingSolver,
};
pub use viscosity::{ViscosityConfig, ViscositySolver};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimalControlError {
    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Convergence failed: {0}")]
    ConvergenceError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Insufficient data: need at least {0} points")]
    InsufficientData(usize),

    #[error("Matrix computation error: {0}")]
    MatrixError(String),
}

pub type Result<T> = std::result::Result<T, OptimalControlError>;
