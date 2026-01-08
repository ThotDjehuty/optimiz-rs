//! Mean Field Games Module
//!
//! This module implements numerical methods for Mean Field Games (MFG) and Mean Field Type Control.
//! Based on: "Numerical Methods for Mean Field Games and Mean Field Type Control"
//!
//! # Overview
//!
//! Mean Field Games (MFG) study strategic decision-making in large populations where each agent
//! optimizes their cost functional while being influenced by the aggregate behavior (mean field)
//! of all agents.
//!
//! ## Mathematical Framework
//!
//! A Mean Field Game consists of two coupled PDEs:
//!
//! 1. **Hamilton-Jacobi-Bellman (HJB) Equation** (backward in time):
//!    ```text
//!    -∂ₜu - νΔu + H(x, ∇u) = f(x, m)    in Ω × (0,T)
//!    u(x,T) = g(x, m(T))                in Ω
//!    ```
//!
//! 2. **Fokker-Planck (FP) Equation** (forward in time):
//!    ```text
//!    ∂ₜm - νΔm - div(m · Hₚ(x, ∇u)) = 0    in Ω × (0,T)
//!    m(x,0) = m₀(x)                         in Ω
//!    ```
//!
//! where:
//! - u(x,t): value function
//! - m(x,t): distribution of agents
//! - H: Hamiltonian (typically H(x,p) = ½|p|²)
//! - ν: viscosity coefficient
//!
//! ## Numerical Methods
//!
//! This module implements:
//! - Finite difference schemes for HJB and FP equations
//! - Fixed-point iteration for MFG system
//! - Primal-dual methods
//! - Newton-type methods
//! - Monotone schemes
//!
//! # References
//!
//! - Achdou, Y., & Capuzzo-Dolcetta, I. (2010). "Mean field games: numerical methods."
//! - Carmona, R., & Delarue, F. (2018). "Probabilistic Theory of Mean Field Games."
//! - Cardaliaguet, P. (2013). "Notes on Mean Field Games."

pub mod types;
pub mod pde_solvers;
pub mod forward_backward;
pub mod nash_equilibrium;
pub mod optimal_transport;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

pub use types::*;
pub use pde_solvers::*;
pub use forward_backward::*;
pub use nash_equilibrium::*;
pub use optimal_transport::*;

use ndarray::{Array1, Array2};
use crate::core::Result;

/// Configuration for Mean Field Games solver
#[derive(Clone, Debug)]
pub struct MFGConfig {
    /// Spatial dimension
    pub dim: usize,
    /// Number of spatial grid points per dimension
    pub nx: usize,
    /// Number of time steps
    pub nt: usize,
    /// Spatial domain bounds [xmin, xmax]
    pub domain: (f64, f64),
    /// Time horizon
    pub time_horizon: f64,
    /// Viscosity coefficient
    pub viscosity: f64,
    /// Convergence tolerance for fixed-point iteration
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Relaxation parameter for updates
    pub relaxation: f64,
}

impl Default for MFGConfig {
    fn default() -> Self {
        Self {
            dim: 1,
            nx: 100,
            nt: 100,
            domain: (0.0, 1.0),
            time_horizon: 1.0,
            viscosity: 0.01,
            tolerance: 1e-6,
            max_iterations: 1000,
            relaxation: 0.5,
        }
    }
}

/// Main Mean Field Games solver
pub struct MFGSolver {
    config: MFGConfig,
}

impl MFGSolver {
    /// Create a new MFG solver with given configuration
    pub fn new(config: MFGConfig) -> Self {
        Self { config }
    }

    /// Solve the MFG system using fixed-point iteration
    ///
    /// # Arguments
    /// - `hamiltonian`: Hamiltonian function H(x, p, m)
    /// - `running_cost`: Running cost f(x, m)
    /// - `terminal_cost`: Terminal cost g(x, m(T))
    /// - `initial_dist`: Initial distribution m₀(x)
    ///
    /// # Returns
    /// Tuple of (value_function, distribution, number_of_iterations)
    pub fn solve<H, F, G>(
        &self,
        hamiltonian: H,
        running_cost: F,
        terminal_cost: G,
        initial_dist: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, usize)>
    where
        H: Fn(f64, f64, f64) -> f64 + Send + Sync,
        F: Fn(f64, f64) -> f64 + Send + Sync,
        G: Fn(f64, f64) -> f64 + Send + Sync,
    {
        // Implemented in forward_backward.rs
        forward_backward_fixed_point(
            &self.config,
            hamiltonian,
            running_cost,
            terminal_cost,
            initial_dist,
        )
    }

    /// Solve using primal-dual method (faster convergence)
    pub fn solve_primal_dual<H, F, G>(
        &self,
        hamiltonian: H,
        running_cost: F,
        terminal_cost: G,
        initial_dist: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, usize)>
    where
        H: Fn(f64, f64, f64) -> f64 + Send + Sync,
        F: Fn(f64, f64) -> f64 + Send + Sync,
        G: Fn(f64, f64) -> f64 + Send + Sync,
    {
        nash_equilibrium::primal_dual_mfg(
            &self.config,
            hamiltonian,
            running_cost,
            terminal_cost,
            initial_dist,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mfg_config_default() {
        let config = MFGConfig::default();
        assert_eq!(config.dim, 1);
        assert_eq!(config.nx, 100);
        assert_eq!(config.nt, 100);
    }

    #[test]
    fn test_mfg_solver_creation() {
        let config = MFGConfig::default();
        let _solver = MFGSolver::new(config);
    }
}
