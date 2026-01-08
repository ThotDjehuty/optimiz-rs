//! Forward-Backward Fixed-Point Iteration for MFG
//!
//! Implements the classical fixed-point algorithm:
//! 1. Solve HJB backward given current m
//! 2. Solve FP forward given current u
//! 3. Update m with relaxation
//! 4. Repeat until convergence

use ndarray::{Array1, Array2};
use crate::core::Result;
use super::{MFGConfig, Grid, pde_solvers};

pub fn forward_backward_fixed_point<H, F, G>(
    config: &MFGConfig,
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
    let grid = Grid::new(config.nx, config.nt, config.domain, config.time_horizon);
    
    // Initialize with uniform distribution
    let mut m_old = Array2::from_elem((config.nx, config.nt), 1.0 / config.nx as f64);
    let mut m_new;
    
    for iter in 0..config.max_iterations {
        // Step 1: Solve HJB backward with current distribution
        let terminal_cond = Array1::from_iter((0..config.nx).map(|i| {
            terminal_cost(grid.x[i], m_old[[i, config.nt - 1]])
        }));
        
        let u = pde_solvers::solve_hjb(config, &grid, &hamiltonian, &running_cost, &terminal_cond, &m_old)?;
        
        // Step 2: Solve FP forward with current value function
        let hp = |_x: f64, p: f64| p; // H_p for quadratic Hamiltonian
        m_new = pde_solvers::solve_fokker_planck(config, &grid, hp, initial_dist, &u)?;
        
        // Step 3: Check convergence
        let error = pde_solvers::relative_l2_error(&m_new, &m_old);
        if error < config.tolerance {
            return Ok((u, m_new, iter + 1));
        }
        
        // Step 4: Relaxation update
        for i in 0..config.nx {
            for n in 0..config.nt {
                m_old[[i, n]] = config.relaxation * m_new[[i, n]] + (1.0 - config.relaxation) * m_old[[i, n]];
            }
        }
    }
    
    // Return best solution even if not converged
    let terminal_cond = Array1::from_iter((0..config.nx).map(|i| {
        terminal_cost(grid.x[i], m_old[[i, config.nt - 1]])
    }));
    let u = pde_solvers::solve_hjb(config, &grid, &hamiltonian, &running_cost, &terminal_cond, &m_old)?;
    Ok((u, m_old, config.max_iterations))
}
