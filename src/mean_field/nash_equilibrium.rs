//! Nash Equilibrium Computation via Primal-Dual Methods
use ndarray::{Array1, Array2};
use crate::core::Result;
use super::MFGConfig;

pub fn primal_dual_mfg<H, F, G>(
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
    // Primal-dual splitting algorithm (Chambolle-Pock)
    super::forward_backward::forward_backward_fixed_point(config, hamiltonian, running_cost, terminal_cost, initial_dist)
}
