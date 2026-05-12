//! 2-D elliptic Poisson solver `-Δu = f` via Gauss–Seidel iteration
//! ==================================================================
//!
//! Solves
//!
//! ```text
//! -Δu(x, y) = f(x, y)   on  Ω = [x_min, x_max] × [y_min, y_max]
//! u = g                 on  ∂Ω
//! ```
//!
//! using the standard 5-point finite difference Laplacian and successive
//! over-relaxation (SOR) with optional `omega` parameter.  Convergence is
//! declared when the L∞ residual drops below `tolerance`.

use crate::core::{OptimizrError, Result};

#[derive(Clone, Debug)]
pub struct EllipticFdConfig {
    pub n_x: usize,
    pub n_y: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub omega: f64,
}

impl Default for EllipticFdConfig {
    fn default() -> Self {
        Self {
            n_x: 65,
            n_y: 65,
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
            max_iterations: 20_000,
            tolerance: 1e-6,
            omega: 1.7,
        }
    }
}

impl EllipticFdConfig {
    pub fn validate(&self) -> Result<()> {
        if self.n_x < 3 || self.n_y < 3 {
            return Err(OptimizrError::InvalidParameter("n_x, n_y ≥ 3".into()));
        }
        if !(self.x_max > self.x_min) || !(self.y_max > self.y_min) {
            return Err(OptimizrError::InvalidParameter("box must be non-degenerate".into()));
        }
        if self.tolerance <= 0.0 {
            return Err(OptimizrError::InvalidParameter("tolerance > 0".into()));
        }
        if !(0.0 < self.omega && self.omega < 2.0) {
            return Err(OptimizrError::InvalidParameter("omega must be in (0, 2)".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct EllipticFdResult {
    /// Solution `u(x_i, y_j)` flattened row-major: `u[i * n_y + j]`.
    pub u: Vec<f64>,
    pub iterations: usize,
    pub residual: f64,
}

pub fn solve_poisson_2d<F, G>(
    rhs: F,
    boundary: G,
    cfg: &EllipticFdConfig,
) -> Result<EllipticFdResult>
where
    F: Fn(f64, f64) -> f64,
    G: Fn(f64, f64) -> f64,
{
    cfg.validate()?;
    let nx = cfg.n_x;
    let ny = cfg.n_y;
    let dx = (cfg.x_max - cfg.x_min) / (nx - 1) as f64;
    let dy = (cfg.y_max - cfg.y_min) / (ny - 1) as f64;
    let dx2 = dx * dx;
    let dy2 = dy * dy;
    let denom = 2.0 * (dx2 + dy2);

    let mut u = vec![0.0f64; nx * ny];
    // Set boundary values.
    for i in 0..nx {
        let x = cfg.x_min + i as f64 * dx;
        u[i * ny] = boundary(x, cfg.y_min);
        u[i * ny + ny - 1] = boundary(x, cfg.y_max);
    }
    for j in 0..ny {
        let y = cfg.y_min + j as f64 * dy;
        u[j] = boundary(cfg.x_min, y);
        u[(nx - 1) * ny + j] = boundary(cfg.x_max, y);
    }

    let mut residual = f64::INFINITY;
    let mut iter = 0;
    while iter < cfg.max_iterations {
        let mut max_res: f64 = 0.0;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let x = cfg.x_min + i as f64 * dx;
                let y = cfg.y_min + j as f64 * dy;
                let f_ij = rhs(x, y);
                let u_old = u[i * ny + j];
                let new_val = (dy2 * (u[(i + 1) * ny + j] + u[(i - 1) * ny + j])
                    + dx2 * (u[i * ny + j + 1] + u[i * ny + j - 1])
                    + dx2 * dy2 * f_ij)
                    / denom;
                let updated = (1.0 - cfg.omega) * u_old + cfg.omega * new_val;
                u[i * ny + j] = updated;
                let r = (updated - u_old).abs();
                if r > max_res {
                    max_res = r;
                }
            }
        }
        residual = max_res;
        iter += 1;
        if residual < cfg.tolerance {
            break;
        }
    }
    if residual >= cfg.tolerance {
        return Err(OptimizrError::ConvergenceFailed(iter));
    }
    Ok(EllipticFdResult { u, iterations: iter, residual })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// `-Δu = 2π² sin(πx) sin(πy)` on [0,1]² with zero boundary admits the
    /// exact solution `u(x,y) = sin(πx) sin(πy)`.
    #[test]
    fn poisson_sine_eigenfunction() {
        let cfg = EllipticFdConfig {
            n_x: 41,
            n_y: 41,
            tolerance: 1e-7,
            max_iterations: 50_000,
            ..Default::default()
        };
        let f = |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
        let g = |_x: f64, _y: f64| 0.0;
        let res = solve_poisson_2d(f, g, &cfg).unwrap();
        // Compare at (0.5, 0.5)  →  sin(π/2)² = 1.
        let mid = (cfg.n_x / 2) * cfg.n_y + cfg.n_y / 2;
        let exact = 1.0_f64;
        let err = (res.u[mid] - exact).abs();
        assert!(err < 5e-3, "u(0.5,0.5) = {} vs 1.0 (err={})", res.u[mid], err);
    }
}
