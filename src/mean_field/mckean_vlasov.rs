//! McKean–Vlasov SDE simulation by interacting particle method
//! ============================================================
//!
//! Simulates the nonlinear McKean–Vlasov SDE
//!
//! ```text
//! dX_t = b(X_t, μ_t) dt + σ dW_t,    μ_t = Law(X_t)
//! ```
//!
//! by the standard propagation-of-chaos Euler scheme on `N` interacting
//! particles where `μ_t` is approximated by the empirical measure
//! `μ^N_t = (1/N) Σ_i δ_{X^i_t}`.  The user supplies a *generic* drift
//! `b(x, μ^N)` taking the current particle position and the slice of all
//! particle positions at the same time.

use crate::core::{OptimizrError, Result};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct McKeanVlasovConfig {
    pub n_particles: usize,
    pub n_steps: usize,
    pub t_horizon: f64,
    pub sigma: f64,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct McKeanVlasovResult {
    /// `paths[k, i]` is `X^i_{t_k}`.
    pub paths: Array2<f64>,
    pub time_grid: Array1<f64>,
}

pub fn simulate_mckean_vlasov<B>(
    initial: &[f64],
    drift: B,
    cfg: &McKeanVlasovConfig,
) -> Result<McKeanVlasovResult>
where
    B: Fn(f64, &[f64]) -> f64,
{
    if cfg.n_particles == 0 || cfg.n_steps == 0 || cfg.t_horizon <= 0.0 {
        return Err(OptimizrError::InvalidParameter("invalid config".into()));
    }
    if initial.len() != cfg.n_particles {
        return Err(OptimizrError::DimensionMismatch {
            expected: cfg.n_particles,
            actual: initial.len(),
        });
    }
    let n = cfg.n_steps;
    let np = cfg.n_particles;
    let dt = cfg.t_horizon / n as f64;
    let sqrt_dt = dt.sqrt();
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut paths = Array2::<f64>::zeros((n + 1, np));
    for i in 0..np {
        paths[[0, i]] = initial[i];
    }
    let mut current = initial.to_vec();
    let mut next = vec![0.0f64; np];
    for k in 0..n {
        // Use the current empirical measure for all particles at this step.
        for i in 0..np {
            let dw = normal.sample(&mut rng) * sqrt_dt;
            next[i] = current[i] + drift(current[i], &current) * dt + cfg.sigma * dw;
        }
        std::mem::swap(&mut current, &mut next);
        for i in 0..np {
            paths[[k + 1, i]] = current[i];
        }
    }
    let time_grid = Array1::from_iter((0..=n).map(|k| k as f64 * dt));
    Ok(McKeanVlasovResult { paths, time_grid })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean-reverting toward the empirical mean: `b(x, μ) = θ (m̄ - x)`.
    /// The empirical mean should be approximately preserved; the variance
    /// shrinks toward the diffusion-only equilibrium `σ² / (2 θ)`.
    #[test]
    fn mean_field_mean_is_preserved() {
        let cfg = McKeanVlasovConfig {
            n_particles: 200,
            n_steps: 1000,
            t_horizon: 1.0,
            sigma: 0.1,
            seed: 42,
        };
        let init: Vec<f64> = (0..cfg.n_particles)
            .map(|i| (i as f64 - cfg.n_particles as f64 / 2.0) / 50.0)
            .collect();
        let init_mean = init.iter().sum::<f64>() / init.len() as f64;
        let theta = 1.0;
        let res = simulate_mckean_vlasov(
            &init,
            |x, mu| {
                let m = mu.iter().sum::<f64>() / mu.len() as f64;
                theta * (m - x)
            },
            &cfg,
        )
        .unwrap();
        let last_row = res.paths.row(cfg.n_steps);
        let final_mean = last_row.iter().sum::<f64>() / cfg.n_particles as f64;
        assert!((final_mean - init_mean).abs() < 0.05,
            "mean drifted: {final_mean} vs {init_mean}");
    }
}
