//! Recover a probability density from its characteristic function via a
//! direct Fourier inversion on a uniform frequency grid.
//!
//! For a real-valued random variable with characteristic function
//! `phi(u) = E[ exp(i u X) ]`, the density (when it exists) is
//!
//! ```text
//!     f(x) = (1 / (2 pi)) * int_{-infty}^{+infty}  exp(-i u x) phi(u) du.
//! ```
//!
//! On a truncated symmetric grid `u in [-U, U]` with spacing `du`, the
//! discrete approximation is
//!
//! ```text
//!     f_hat(x) ~= (du / (2 pi)) * sum_k  exp(-i u_k x) phi(u_k).
//! ```
//!
//! Symmetry of real densities (`phi(-u) = conj(phi(u))`) reduces the
//! computation to a cosine sum over `u >= 0`.
//!
//! # Notes
//!
//! * Implementation uses a direct `O(N_u * N_x)` evaluation. For very
//!   large grids, swap in an FFT-based Carr--Madan style routine.
//! * `phi` is supplied as a closure returning `(re, im)`.

use crate::core::{OptimizrError, Result};

/// Inversion result.
#[derive(Debug, Clone)]
pub struct DensityResult {
    pub x_grid: Vec<f64>,
    pub density: Vec<f64>,
}

/// Recover the density on `x_grid` from the characteristic function `phi`.
///
/// `u_max` is the truncation bound; `n_u` the number of nodes on
/// `[0, u_max]`. The total grid is `2*n_u - 1` symmetric nodes.
pub fn fourier_invert<P>(
    phi: P,
    x_grid: &[f64],
    u_max: f64,
    n_u: usize,
) -> Result<DensityResult>
where
    P: Fn(f64) -> (f64, f64),
{
    if x_grid.is_empty() {
        return Err(OptimizrError::EmptyData);
    }
    if u_max <= 0.0 {
        return Err(OptimizrError::InvalidParameter("u_max > 0 required".into()));
    }
    if n_u < 2 {
        return Err(OptimizrError::InvalidParameter("n_u >= 2 required".into()));
    }
    let du = u_max / (n_u - 1) as f64;
    // Use the symmetric form: f(x) = (1 / pi) * int_0^inf [ Re(phi(u)) cos(u x)
    //                                                  + Im(phi(u)) sin(u x) ] du
    // (when X is not necessarily symmetric, the imaginary part contributes
    // an antisymmetric kernel).
    let mut density = vec![0.0; x_grid.len()];
    // Trapezoidal weights
    let weight = |k: usize| -> f64 {
        if k == 0 || k == n_u - 1 {
            0.5
        } else {
            1.0
        }
    };
    let cache: Vec<(f64, f64)> = (0..n_u).map(|k| phi(k as f64 * du)).collect();
    for (idx, &x) in x_grid.iter().enumerate() {
        let mut s = 0.0;
        for k in 0..n_u {
            let u = k as f64 * du;
            let (re, im) = cache[k];
            let cos_ux = (u * x).cos();
            let sin_ux = (u * x).sin();
            s += weight(k) * (re * cos_ux + im * sin_ux);
        }
        density[idx] = du * s / std::f64::consts::PI;
    }
    Ok(DensityResult {
        x_grid: x_grid.to_vec(),
        density,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard normal: phi(u) = exp(-u^2 / 2), density = (1 / sqrt(2 pi)) exp(-x^2 / 2).
    #[test]
    fn recovers_standard_normal_density() {
        let phi = |u: f64| ((-0.5 * u * u).exp(), 0.0);
        let x: Vec<f64> = (-50..=50).map(|k| 0.05 * k as f64).collect();
        let res = fourier_invert(phi, &x, 25.0, 2000).unwrap();
        let exact = |xx: f64| (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * xx * xx).exp();
        let mut max_err = 0.0_f64;
        for (xi, fi) in x.iter().zip(res.density.iter()) {
            let e = (fi - exact(*xi)).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(max_err < 1e-3, "max err = {}", max_err);
    }
}
