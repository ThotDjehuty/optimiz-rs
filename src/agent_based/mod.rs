//! Agent-based generic dynamics
//! =============================
//!
//! Lightweight discrete-time interacting-agent simulator.  Each of `N`
//! agents has a real-valued state `s_i ∈ ℝ` and updates via a *generic*
//! transition rule
//!
//! ```text
//! s^{k+1}_i = T(s^k_i, neighbours, k) + ξ^k_i
//! ```
//!
//! where `neighbours` is a slice of the other agents' states.  Neither the
//! transition nor the topology carries any domain-specific meaning — it is
//! a CPU-only coupling primitive used by higher-level frameworks (mean
//! field games, opinion dynamics, particle filters, ...).

use crate::core::{OptimizrError, Result};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct AgentBasedConfig {
    pub n_agents: usize,
    pub n_steps: usize,
    /// Standard deviation of the additive noise.
    pub noise_sigma: f64,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct AgentBasedResult {
    /// `states[k, i]` = state of agent `i` at step `k`.
    pub states: Array2<f64>,
    /// Step-wise empirical mean.
    pub mean_trajectory: Array1<f64>,
}

pub fn simulate_agent_based<T>(
    initial: &[f64],
    transition: T,
    cfg: &AgentBasedConfig,
) -> Result<AgentBasedResult>
where
    T: Fn(f64, &[f64], usize) -> f64,
{
    if cfg.n_agents == 0 || cfg.n_steps == 0 {
        return Err(OptimizrError::InvalidParameter("n_agents and n_steps must be > 0".into()));
    }
    if initial.len() != cfg.n_agents {
        return Err(OptimizrError::DimensionMismatch {
            expected: cfg.n_agents,
            actual: initial.len(),
        });
    }
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal = Normal::new(0.0, cfg.noise_sigma).unwrap();
    let mut states = Array2::<f64>::zeros((cfg.n_steps + 1, cfg.n_agents));
    let mut mean_traj = Array1::<f64>::zeros(cfg.n_steps + 1);
    for i in 0..cfg.n_agents {
        states[[0, i]] = initial[i];
    }
    mean_traj[0] = initial.iter().sum::<f64>() / cfg.n_agents as f64;
    let mut current = initial.to_vec();
    let mut next = vec![0.0f64; cfg.n_agents];
    for k in 0..cfg.n_steps {
        for i in 0..cfg.n_agents {
            next[i] = transition(current[i], &current, k) + normal.sample(&mut rng);
        }
        std::mem::swap(&mut current, &mut next);
        for i in 0..cfg.n_agents {
            states[[k + 1, i]] = current[i];
        }
        mean_traj[k + 1] = current.iter().sum::<f64>() / cfg.n_agents as f64;
    }
    Ok(AgentBasedResult { states, mean_trajectory: mean_traj })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bounded-confidence consensus: `T(s, ngh, k) = mean(ngh)`.  Without
    /// noise, all agents converge to a single value — namely the average.
    #[test]
    fn consensus_dynamics_converge_without_noise() {
        let cfg = AgentBasedConfig {
            n_agents: 30, n_steps: 100, noise_sigma: 0.0, seed: 0,
        };
        let init: Vec<f64> = (0..cfg.n_agents).map(|i| i as f64).collect();
        let init_mean = init.iter().sum::<f64>() / init.len() as f64;
        let res = simulate_agent_based(
            &init,
            |_s, ngh, _k| ngh.iter().sum::<f64>() / ngh.len() as f64,
            &cfg,
        ).unwrap();
        let last = res.states.row(cfg.n_steps);
        for &v in last.iter() {
            assert!((v - init_mean).abs() < 1e-9, "did not converge: {v} vs {init_mean}");
        }
    }
}
