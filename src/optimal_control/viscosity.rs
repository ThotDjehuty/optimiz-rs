//! Viscosity Solutions
//! ===================
//!
//! Advanced numerical schemes for viscosity solutions of HJB equations

use crate::optimal_control::{OptimalControlError, Result};
use ndarray::Array1;

/// Configuration for viscosity solver
#[derive(Debug, Clone)]
pub struct ViscosityConfig {
    /// Spatial grid points
    pub n_space: usize,
    /// Time grid points
    pub n_time: usize,
    /// Spatial domain bounds
    pub x_min: f64,
    pub x_max: f64,
    /// Time horizon
    pub t_max: f64,
    /// Viscosity parameter (artificial diffusion)
    pub epsilon: f64,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for ViscosityConfig {
    fn default() -> Self {
        Self {
            n_space: 200,
            n_time: 100,
            x_min: -3.0,
            x_max: 3.0,
            t_max: 1.0,
            epsilon: 0.01,
            tolerance: 1e-6,
        }
    }
}

/// Viscosity solver for HJB equations
pub struct ViscositySolver {
    config: ViscosityConfig,
}

impl ViscositySolver {
    pub fn new(config: ViscosityConfig) -> Self {
        Self { config }
    }

    /// Solve using upwind finite differences
    pub fn solve_upwind(
        &self,
        drift: impl Fn(f64) -> f64,
        diffusion: impl Fn(f64) -> f64,
        terminal_condition: impl Fn(f64) -> f64,
    ) -> Result<Array1<f64>> {
        let cfg = &self.config;

        let dx = (cfg.x_max - cfg.x_min) / (cfg.n_space - 1) as f64;
        let dt = cfg.t_max / cfg.n_time as f64;

        // Spatial grid
        let x: Vec<f64> = (0..cfg.n_space)
            .map(|i| cfg.x_min + i as f64 * dx)
            .collect();

        // Initialize with terminal condition
        let mut v: Vec<f64> = x.iter().map(|&xi| terminal_condition(xi)).collect();
        let mut v_new = v.clone();

        // Backward in time
        for _t in 0..cfg.n_time {
            for i in 1..cfg.n_space - 1 {
                let xi = x[i];
                let mu = drift(xi);
                let sigma = diffusion(xi);

                // Upwind scheme based on drift direction
                let dv_dx = if mu > 0.0 {
                    (v[i] - v[i - 1]) / dx
                } else {
                    (v[i + 1] - v[i]) / dx
                };

                // Second derivative (central difference)
                let d2v_dx2 = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / dx.powi(2);

                // Explicit Euler step
                v_new[i] = v[i] + dt * (-mu * dv_dx + 0.5 * sigma.powi(2) * d2v_dx2);
            }

            // Boundary conditions
            v_new[0] = v_new[1];
            v_new[cfg.n_space - 1] = v_new[cfg.n_space - 2];

            v = v_new.clone();
        }

        Ok(Array1::from_vec(v))
    }

    /// Solve using Lax-Friedrichs scheme (more stable)
    #[allow(unused_variables)]  // diffusion parameter reserved for future extensions
    pub fn solve_lax_friedrichs(
        &self,
        drift: impl Fn(f64) -> f64,
        diffusion: impl Fn(f64) -> f64,
        hamiltonian: impl Fn(f64, f64, f64) -> f64,
    ) -> Result<Array1<f64>> {
        let cfg = &self.config;

        let dx = (cfg.x_max - cfg.x_min) / (cfg.n_space - 1) as f64;
        let dt = cfg.t_max / cfg.n_time as f64;

        // CFL condition check
        let max_drift = (0..cfg.n_space)
            .map(|i| drift(cfg.x_min + i as f64 * dx).abs())
            .fold(0.0f64, |a, b| a.max(b));

        if dt > 0.5 * dx / (max_drift + cfg.epsilon) {
            return Err(OptimalControlError::NumericalError(
                "CFL condition violated".to_string(),
            ));
        }

        // Spatial grid
        let x: Vec<f64> = (0..cfg.n_space)
            .map(|i| cfg.x_min + i as f64 * dx)
            .collect();

        // Initialize
        let mut v = vec![0.0; cfg.n_space];
        let mut v_new = v.clone();

        // Backward in time
        for _t in 0..cfg.n_time {
            for i in 1..cfg.n_space - 1 {
                let xi = x[i];

                // Lax-Friedrichs numerical flux
                let v_avg = 0.5 * (v[i + 1] + v[i - 1]);
                let _flux_diff = 0.5 * (v[i + 1] - v[i - 1]) / dx;  // Reserved for future use

                let dv_dx = (v[i + 1] - v[i - 1]) / (2.0 * dx);
                let d2v_dx2 = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / dx.powi(2);

                let h = hamiltonian(xi, v[i], dv_dx);

                // Lax-Friedrichs step with artificial viscosity
                v_new[i] = v_avg + dt * (-h + cfg.epsilon * d2v_dx2);
            }

            v_new[0] = v_new[1];
            v_new[cfg.n_space - 1] = v_new[cfg.n_space - 2];

            v = v_new.clone();
        }

        Ok(Array1::from_vec(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viscosity_solver_linear() {
        let config = ViscosityConfig {
            n_space: 50,
            n_time: 50,
            x_min: -1.0,
            x_max: 1.0,
            t_max: 0.1,
            epsilon: 0.01,
            tolerance: 1e-6,
        };

        let solver = ViscositySolver::new(config);

        // Simple drift-diffusion: ∂v/∂t = -v + ∂²v/∂x²
        let drift = |_x: f64| 0.0;
        let diffusion = |_x: f64| 1.0;
        let terminal = |x: f64| x.powi(2);

        let result = solver.solve_upwind(drift, diffusion, terminal);
        assert!(result.is_ok());
    }
}
