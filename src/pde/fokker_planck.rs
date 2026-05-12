//! 1-D Fokker–Planck (Kolmogorov forward) equation
//! =================================================
//!
//! Solves
//!
//! ```text
//! ∂_t m(x, t) + ∂_x [μ(x) m(x, t)] - (1/2) ∂_xx [σ²(x) m(x, t)] = 0,    t > 0
//! m(x, 0) = m_0(x),    Dirichlet boundary m = 0 on the box ends.
//! ```
//!
//! Discretisation: forward Euler in time, conservative central differences in
//! space.  The CFL-type stability condition `Δt · (max|μ|/Δx + max σ²/Δx²) ≤ 1`
//! must hold; the routine returns an error when it would be violated.

use crate::core::{OptimizrError, Result};
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct FokkerPlanckConfig {
    pub n_x: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub n_t: usize,
    pub t_horizon: f64,
}

impl FokkerPlanckConfig {
    pub fn validate(&self) -> Result<()> {
        if self.n_x < 5 {
            return Err(OptimizrError::InvalidParameter("n_x must be ≥ 5".into()));
        }
        if !(self.x_max > self.x_min) {
            return Err(OptimizrError::InvalidParameter("x_max > x_min required".into()));
        }
        if self.n_t == 0 {
            return Err(OptimizrError::InvalidParameter("n_t must be > 0".into()));
        }
        if !(self.t_horizon > 0.0) {
            return Err(OptimizrError::InvalidParameter("t_horizon must be > 0".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FokkerPlanckResult {
    pub x_grid: Array1<f64>,
    pub time_grid: Array1<f64>,
    /// Density at each `(t_k, x_i)` flattened in row-major order
    /// `density[k * n_x + i]`.
    pub density: Vec<f64>,
}

pub fn solve_fokker_planck_1d<Mu, Sigma2, M0>(
    drift: Mu,
    diffusion_sq: Sigma2,
    initial_density: M0,
    cfg: &FokkerPlanckConfig,
) -> Result<FokkerPlanckResult>
where
    Mu: Fn(f64) -> f64,
    Sigma2: Fn(f64) -> f64,
    M0: Fn(f64) -> f64,
{
    cfg.validate()?;
    let nx = cfg.n_x;
    let nt = cfg.n_t;
    let dx = (cfg.x_max - cfg.x_min) / (nx - 1) as f64;
    let dt = cfg.t_horizon / nt as f64;
    let x_grid: Array1<f64> = Array1::from_iter((0..nx).map(|i| cfg.x_min + i as f64 * dx));
    let time_grid: Array1<f64> = Array1::from_iter((0..=nt).map(|k| k as f64 * dt));

    // Stability check (very mild upper bound on coefficients sampled on the grid).
    let mut max_mu = 0.0_f64;
    let mut max_sig = 0.0_f64;
    for &x in x_grid.iter() {
        max_mu = max_mu.max(drift(x).abs());
        max_sig = max_sig.max(diffusion_sq(x).abs());
    }
    let cfl = dt * (max_mu / dx + max_sig / (dx * dx));
    if cfl > 1.0 {
        return Err(OptimizrError::NumericalError(format!(
            "CFL condition violated: dt·(|μ|/dx + σ²/dx²) = {cfl:.3} > 1"
        )));
    }

    let mut density = vec![0.0; nx * (nt + 1)];
    for i in 0..nx {
        density[i] = initial_density(x_grid[i]).max(0.0);
    }
    // Renormalise initial density to mass 1 (trapezoidal).
    let mut mass = 0.0;
    for i in 0..nx - 1 {
        mass += 0.5 * dx * (density[i] + density[i + 1]);
    }
    if mass > 0.0 {
        for i in 0..nx {
            density[i] /= mass;
        }
    }

    for k in 0..nt {
        let off = k * nx;
        let new_off = (k + 1) * nx;
        // Boundaries enforced to zero
        density[new_off] = 0.0;
        density[new_off + nx - 1] = 0.0;
        for i in 1..nx - 1 {
            let x_im = x_grid[i - 1];
            let x_ip = x_grid[i + 1];
            let m_im = density[off + i - 1];
            let m_i = density[off + i];
            let m_ip = density[off + i + 1];
            let mu_im = drift(x_im);
            let mu_ip = drift(x_ip);
            let s_im = diffusion_sq(x_im);
            let s_i = diffusion_sq(x_grid[i]);
            let s_ip = diffusion_sq(x_ip);

            let drift_term = (mu_ip * m_ip - mu_im * m_im) / (2.0 * dx);
            let diff_term = (s_ip * m_ip - 2.0 * s_i * m_i + s_im * m_im) / (dx * dx);
            density[new_off + i] = m_i - dt * drift_term + 0.5 * dt * diff_term;
            if density[new_off + i] < 0.0 {
                density[new_off + i] = 0.0; // positivity safeguard
            }
        }
    }

    Ok(FokkerPlanckResult {
        x_grid,
        time_grid,
        density,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Pure diffusion `μ = 0, σ² = 1` with Gaussian initial condition centred
    /// at 0 should remain centred and stay non-negative; total mass should be
    /// approximately conserved before any boundary loss.
    #[test]
    fn pure_diffusion_keeps_mean_at_zero() {
        let cfg = FokkerPlanckConfig {
            n_x: 401,
            x_min: -8.0,
            x_max: 8.0,
            n_t: 8000,
            t_horizon: 0.5,
        };
        let res = solve_fokker_planck_1d(
            |_| 0.0,
            |_| 1.0,
            |x| (-(x * x) / 2.0).exp() / (2.0 * PI).sqrt(),
            &cfg,
        )
        .unwrap();
        let nx = cfg.n_x;
        let off = cfg.n_t * nx;
        let dx = (cfg.x_max - cfg.x_min) / (nx - 1) as f64;
        let mut mean = 0.0;
        let mut mass = 0.0;
        for i in 0..nx {
            let x = cfg.x_min + i as f64 * dx;
            let m = res.density[off + i];
            mean += x * m * dx;
            mass += m * dx;
        }
        assert!(mass > 0.5, "lost too much mass: {mass}");
        assert!(mean.abs() < 0.05, "mean drifted: {mean}");
    }

    #[test]
    fn cfl_violation_is_detected() {
        let cfg = FokkerPlanckConfig {
            n_x: 11,
            x_min: 0.0,
            x_max: 1.0,
            n_t: 1,
            t_horizon: 1.0,
        };
        let res = solve_fokker_planck_1d(|_| 0.0, |_| 1.0, |_| 1.0, &cfg);
        assert!(res.is_err());
    }
}
