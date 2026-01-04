//! PDE Solvers for Mean Field Games
//!
//! This module implements high-performance numerical solvers for:
//! - Hamilton-Jacobi-Bellman (HJB) equations
//! - Fokker-Planck (FP) equations
//! - Coupled MFG systems
//!
//! Uses finite difference schemes with parallel computation via Rayon.

use ndarray::{Array1, Array2, s};
use rayon::prelude::*;
use crate::core::{OptimizrError, Result};
use super::{Grid, MFGConfig};

/// Solve the HJB equation backward in time
///
/// Solves: -∂ₜu - νΔu + H(x, ∇u) = f(x, m)
///
/// # Arguments
/// - `config`: MFG configuration
/// - `grid`: Spatial-temporal grid
/// - `hamiltonian`: H(x, p, m) where p = ∇u
/// - `running_cost`: f(x, m)
/// - `terminal_condition`: u(x, T) = g(x, m(T))
/// - `distribution`: Current distribution m(x,t)
///
/// # Returns
/// Value function u(x,t) as Array2 (nx × nt)
pub fn solve_hjb<H, F>(
    config: &MFGConfig,
    grid: &Grid,
    hamiltonian: H,
    running_cost: F,
    terminal_condition: &Array1<f64>,
    distribution: &Array2<f64>,
) -> Result<Array2<f64>>
where
    H: Fn(f64, f64, f64) -> f64 + Send + Sync,
    F: Fn(f64, f64) -> f64 + Send + Sync,
{
    let nx = config.nx;
    let nt = config.nt;
    let dx = grid.dx;
    let dt = grid.dt;
    let nu = config.viscosity;
    
    // Initialize value function
    let mut u = Array2::zeros((nx, nt));
    
    // Set terminal condition
    for i in 0..nx {
        u[[i, nt - 1]] = terminal_condition[i];
    }
    
    // Backward time stepping with upwind scheme
    for n in (0..nt - 1).rev() {
        // Parallel computation over spatial grid
        let u_next: Vec<f64> = (1..nx - 1)
            .into_par_iter()
            .map(|i| {
                let x = grid.x[i];
                let m = distribution[[i, n]];
                
                // Central difference for second derivative (Laplacian)
                let u_xx = (u[[i + 1, n + 1]] - 2.0 * u[[i, n + 1]] + u[[i - 1, n + 1]]) / (dx * dx);
                
                // Upwind scheme for first derivative
                let u_plus = (u[[i + 1, n + 1]] - u[[i, n + 1]]) / dx;
                let u_minus = (u[[i, n + 1]] - u[[i - 1, n + 1]]) / dx;
                
                // Choose upwind direction based on Hamiltonian
                let h_plus = hamiltonian(x, u_plus, m);
                let h_minus = hamiltonian(x, u_minus, m);
                let h = if h_plus.abs() < h_minus.abs() { h_plus } else { h_minus };
                
                // Implicit scheme: u^n = u^{n+1} + dt*(νΔu - H + f)
                let f = running_cost(x, m);
                u[[i, n + 1]] - dt * (nu * u_xx - h + f)
            })
            .collect();
        
        // Update interior points
        for (idx, i) in (1..nx - 1).enumerate() {
            u[[i, n]] = u_next[idx];
        }
        
        // Boundary conditions (Neumann: zero derivative)
        u[[0, n]] = u[[1, n]];
        u[[nx - 1, n]] = u[[nx - 2, n]];
    }
    
    Ok(u)
}

/// Solve the Fokker-Planck equation forward in time
///
/// Solves: ∂ₜm - νΔm - div(m · Hₚ(x, ∇u)) = 0
///
/// # Arguments
/// - `config`: MFG configuration
/// - `grid`: Spatial-temporal grid
/// - `hamiltonian_p`: Derivative of Hamiltonian H_p(x, p)
/// - `initial_distribution`: m(x, 0) = m₀(x)
/// - `value_function`: Current value function u(x,t)
///
/// # Returns
/// Distribution m(x,t) as Array2 (nx × nt)
pub fn solve_fokker_planck<Hp>(
    config: &MFGConfig,
    grid: &Grid,
    hamiltonian_p: Hp,
    initial_distribution: &Array1<f64>,
    value_function: &Array2<f64>,
) -> Result<Array2<f64>>
where
    Hp: Fn(f64, f64) -> f64 + Send + Sync,
{
    let nx = config.nx;
    let nt = config.nt;
    let dx = grid.dx;
    let dt = grid.dt;
    let nu = config.viscosity;
    
    // Initialize distribution
    let mut m = Array2::zeros((nx, nt));
    
    // Set initial condition
    for i in 0..nx {
        m[[i, 0]] = initial_distribution[i];
    }
    
    // Normalize initial distribution
    let sum: f64 = m.slice(s![.., 0]).sum();
    for i in 0..nx {
        m[[i, 0]] /= sum * dx;
    }
    
    // Forward time stepping with upwind scheme
    for n in 0..nt - 1 {
        // Parallel computation over spatial grid
        let m_next: Vec<f64> = (1..nx - 1)
            .into_par_iter()
            .map(|i| {
                let x = grid.x[i];
                
                // Gradient of value function at (x, t^n)
                let u_x = (value_function[[i + 1, n]] - value_function[[i - 1, n]]) / (2.0 * dx);
                
                // Velocity field from Hamiltonian
                let v = hamiltonian_p(x, u_x);
                
                // Diffusion term: νΔm
                let m_xx = (m[[i + 1, n]] - 2.0 * m[[i, n]] + m[[i - 1, n]]) / (dx * dx);
                
                // Advection term: -div(m · v) with upwind
                let flux_plus = if v > 0.0 {
                    v * m[[i, n]]
                } else {
                    v * m[[i + 1, n]]
                };
                let flux_minus = if v > 0.0 {
                    v * m[[i - 1, n]]
                } else {
                    v * m[[i, n]]
                };
                let div_flux = (flux_plus - flux_minus) / dx;
                
                // Forward Euler: m^{n+1} = m^n + dt*(νΔm - div(m·v))
                m[[i, n]] + dt * (nu * m_xx - div_flux)
            })
            .collect();
        
        // Update interior points
        for (idx, i) in (1..nx - 1).enumerate() {
            m[[i, n + 1]] = m_next[idx].max(0.0); // Ensure non-negativity
        }
        
        // Boundary conditions (Neumann)
        m[[0, n + 1]] = m[[1, n + 1]];
        m[[nx - 1, n + 1]] = m[[nx - 2, n + 1]];
        
        // Normalize to maintain probability
        let sum: f64 = m.slice(s![.., n + 1]).sum();
        if sum > 1e-10 {
            for i in 0..nx {
                m[[i, n + 1]] /= sum * dx;
            }
        }
    }
    
    Ok(m)
}

/// Compute L² norm of difference between two arrays
pub fn l2_norm_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute relative L² error
pub fn relative_l2_error(computed: &Array2<f64>, reference: &Array2<f64>) -> f64 {
    let diff_norm = l2_norm_diff(computed, reference);
    let ref_norm = reference.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    if ref_norm < 1e-14 {
        diff_norm
    } else {
        diff_norm / ref_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_grid_creation() {
        let config = MFGConfig::default();
        let grid = Grid::new(config.nx, config.nt, config.domain, config.time_horizon);
        assert_eq!(grid.x.len(), config.nx);
        assert_eq!(grid.t.len(), config.nt);
    }

    #[test]
    fn test_hjb_solver_initialization() {
        let config = MFGConfig::default();
        let grid = Grid::new(config.nx, config.nt, config.domain, config.time_horizon);
        let terminal = Array1::zeros(config.nx);
        let distribution = Array2::zeros((config.nx, config.nt));
        
        let result = solve_hjb(
            &config,
            &grid,
            |_x, p, _m| 0.5 * p * p,
            |_x, _m| 0.0,
            &terminal,
            &distribution,
        );
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_l2_norm() {
        let a = Array2::from_elem((10, 10), 1.0);
        let b = Array2::from_elem((10, 10), 2.0);
        let norm = l2_norm_diff(&a, &b);
        assert!((norm - 10.0).abs() < 1e-10);
    }
}
