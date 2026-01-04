//! Type definitions for Mean Field Games

use ndarray::{Array1, Array2};

/// Grid structure for spatial and temporal discretization
#[derive(Clone, Debug)]
pub struct Grid {
    /// Spatial grid points
    pub x: Array1<f64>,
    /// Time grid points
    pub t: Array1<f64>,
    /// Spatial step size
    pub dx: f64,
    /// Time step size
    pub dt: f64,
}

impl Grid {
    /// Create a new grid from configuration
    pub fn new(nx: usize, nt: usize, domain: (f64, f64), time_horizon: f64) -> Self {
        let dx = (domain.1 - domain.0) / (nx as f64 - 1.0);
        let dt = time_horizon / (nt as f64 - 1.0);
        
        let x = Array1::from_iter((0..nx).map(|i| domain.0 + i as f64 * dx));
        let t = Array1::from_iter((0..nt).map(|i| i as f64 * dt));
        
        Self { x, t, dx, dt }
    }
}

/// Solution of a Mean Field Game
#[derive(Clone, Debug)]
pub struct MFGSolution {
    /// Value function u(x,t)
    pub value_function: Array2<f64>,
    /// Distribution m(x,t)
    pub distribution: Array2<f64>,
    /// Grid information
    pub grid: Grid,
    /// Number of iterations to converge
    pub iterations: usize,
    /// Final residual
    pub residual: f64,
}

/// Hamiltonian types commonly used in MFG
pub enum HamiltonianType {
    /// Quadratic: H(p) = ½|p|²
    Quadratic,
    /// Linear: H(p) = p
    Linear,
    /// Power law: H(p) = |p|^α / α
    PowerLaw(f64),
    /// Custom function
    Custom(Box<dyn Fn(f64, f64) -> f64 + Send + Sync>),
}

impl HamiltonianType {
    /// Evaluate the Hamiltonian
    pub fn evaluate(&self, x: f64, p: f64) -> f64 {
        match self {
            Self::Quadratic => 0.5 * p * p,
            Self::Linear => p,
            Self::PowerLaw(alpha) => p.abs().powf(*alpha) / alpha,
            Self::Custom(f) => f(x, p),
        }
    }

    /// Compute H_p (derivative with respect to p)
    pub fn derivative_p(&self, _x: f64, p: f64) -> f64 {
        match self {
            Self::Quadratic => p,
            Self::Linear => 1.0,
            Self::PowerLaw(alpha) => p.abs().powf(alpha - 1.0) * p.signum(),
            Self::Custom(_) => {
                // Finite difference approximation
                let eps = 1e-8;
                (self.evaluate(_x, p + eps) - self.evaluate(_x, p - eps)) / (2.0 * eps)
            }
        }
    }
}

/// Boundary condition types
#[derive(Clone, Debug)]
pub enum BoundaryCondition {
    /// Dirichlet: u = value on boundary
    Dirichlet(f64),
    /// Neumann: ∂u/∂n = value on boundary
    Neumann(f64),
    /// Periodic boundary conditions
    Periodic,
}
