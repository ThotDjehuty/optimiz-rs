//! Multidimensional explicit HJB finite-difference solver
//! =======================================================
//!
//! Solves the parabolic Hamilton–Jacobi–Bellman equation
//!
//! ```text
//! -∂_t v(t, x) + H(x, ∇v) - (σ²/2) Δv = 0,    v(T, x) = g(x)
//! ```
//!
//! on a regular Cartesian box `[x_min, x_max]^d` with Neumann (zero-flux)
//! boundary conditions and explicit Euler time-stepping.  A user-supplied
//! Hamiltonian closure `H(x, ∇v) -> ℝ` evaluates the inf/sup over controls;
//! the solver only requires the resulting scalar.
//!
//! The implementation supports `d = 1, 2, 3` (more dimensions are accepted
//! but memory grows as `n_per_dim^d`).

use crate::core::{OptimizrError, Result};
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct HjbMultidConfig {
    /// Number of spatial dimensions.
    pub dim: usize,
    /// Number of grid points per dimension (uniform).
    pub n_per_dim: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub n_t: usize,
    pub t_horizon: f64,
    /// Constant isotropic diffusion coefficient σ² ≥ 0.
    pub sigma_sq: f64,
}

impl HjbMultidConfig {
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.dim > 3 {
            return Err(OptimizrError::InvalidParameter("dim must be 1, 2 or 3".into()));
        }
        if self.n_per_dim < 3 {
            return Err(OptimizrError::InvalidParameter("n_per_dim must be ≥ 3".into()));
        }
        if !(self.x_max > self.x_min) {
            return Err(OptimizrError::InvalidParameter("x_max > x_min".into()));
        }
        if self.n_t == 0 {
            return Err(OptimizrError::InvalidParameter("n_t > 0".into()));
        }
        if !(self.t_horizon > 0.0) {
            return Err(OptimizrError::InvalidParameter("t_horizon > 0".into()));
        }
        if self.sigma_sq < 0.0 {
            return Err(OptimizrError::InvalidParameter("sigma_sq ≥ 0".into()));
        }
        Ok(())
    }

    pub fn dx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.n_per_dim - 1) as f64
    }

    pub fn dt(&self) -> f64 {
        self.t_horizon / self.n_t as f64
    }

    pub fn total_size(&self) -> usize {
        self.n_per_dim.pow(self.dim as u32)
    }
}

#[derive(Clone, Debug)]
pub struct HjbMultidResult {
    pub value: Vec<f64>,
    pub grid_axes: Vec<Array1<f64>>,
}

/// Convert a flat index to multi-index (lexicographic, last dim fastest).
fn flat_to_multi(idx: usize, n: usize, dim: usize) -> Vec<usize> {
    let mut out = vec![0usize; dim];
    let mut r = idx;
    for d in (0..dim).rev() {
        out[d] = r % n;
        r /= n;
    }
    out
}

fn multi_to_flat(mi: &[usize], n: usize) -> usize {
    let mut idx = 0usize;
    for &m in mi {
        idx = idx * n + m;
    }
    idx
}

pub fn solve_hjb_multid<H, G>(
    hamiltonian: H,
    terminal: G,
    cfg: &HjbMultidConfig,
) -> Result<HjbMultidResult>
where
    H: Fn(&[f64], &[f64]) -> f64, // H(x, grad_v)
    G: Fn(&[f64]) -> f64,
{
    cfg.validate()?;
    let n = cfg.n_per_dim;
    let d = cfg.dim;
    let dx = cfg.dx();
    let dt = cfg.dt();
    let size = cfg.total_size();

    let cfl = dt * (cfg.sigma_sq * d as f64 / (dx * dx));
    if cfl > 0.5 {
        return Err(OptimizrError::NumericalError(format!(
            "explicit HJB CFL violated: dt·d·σ²/dx² = {cfl:.3} > 0.5"
        )));
    }

    let grid_axes: Vec<Array1<f64>> =
        (0..d).map(|_| Array1::from_iter((0..n).map(|i| cfg.x_min + i as f64 * dx))).collect();

    // Cache positions per flat index.
    let mut pos = vec![0.0f64; size * d];
    for idx in 0..size {
        let mi = flat_to_multi(idx, n, d);
        for k in 0..d {
            pos[idx * d + k] = grid_axes[k][mi[k]];
        }
    }

    // Terminal condition.
    let mut v = vec![0.0f64; size];
    for idx in 0..size {
        v[idx] = terminal(&pos[idx * d..(idx + 1) * d]);
    }
    let mut v_new = vec![0.0f64; size];
    let mut grad = vec![0.0f64; d];
    let mut x_local = vec![0.0f64; d];

    for _step in 0..cfg.n_t {
        for idx in 0..size {
            let mi = flat_to_multi(idx, n, d);
            for k in 0..d {
                x_local[k] = pos[idx * d + k];
            }
            // central differences with reflective (Neumann) boundary.
            let mut lap = 0.0;
            for k in 0..d {
                let mut mi_p = mi.clone();
                let mut mi_m = mi.clone();
                if mi[k] + 1 < n { mi_p[k] += 1; } else { mi_p[k] = mi[k]; }
                if mi[k] >= 1 { mi_m[k] -= 1; } else { mi_m[k] = mi[k]; }
                let v_p = v[multi_to_flat(&mi_p, n)];
                let v_m = v[multi_to_flat(&mi_m, n)];
                grad[k] = (v_p - v_m) / (2.0 * dx);
                lap += (v_p - 2.0 * v[idx] + v_m) / (dx * dx);
            }
            let h_val = hamiltonian(&x_local, &grad);
            // Backward time: v^{n} = v^{n+1} + dt * ((σ²/2) Δv - H)
            v_new[idx] = v[idx] + dt * (0.5 * cfg.sigma_sq * lap - h_val);
        }
        std::mem::swap(&mut v, &mut v_new);
    }

    Ok(HjbMultidResult { value: v, grid_axes })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `H = 0`, `σ² = 0` and constant terminal should stay constant.
    #[test]
    fn trivial_problem_preserves_constant() {
        let cfg = HjbMultidConfig {
            dim: 2,
            n_per_dim: 9,
            x_min: -1.0,
            x_max: 1.0,
            n_t: 50,
            t_horizon: 1.0,
            sigma_sq: 0.0,
        };
        let res = solve_hjb_multid(|_, _| 0.0, |_| 3.14, &cfg).unwrap();
        for v in res.value.iter() {
            assert!((v - 3.14).abs() < 1e-12);
        }
    }

    /// Pure heat (`H = 0`, σ² > 0) preserves the integral and average value
    /// of a constant initial condition (Neumann BCs).
    #[test]
    fn pure_heat_preserves_constant() {
        let cfg = HjbMultidConfig {
            dim: 1,
            n_per_dim: 21,
            x_min: -1.0,
            x_max: 1.0,
            n_t: 100,
            t_horizon: 0.1,
            sigma_sq: 0.5,
        };
        let res = solve_hjb_multid(|_, _| 0.0, |_| 1.0, &cfg).unwrap();
        for v in res.value.iter() {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }
}
