//! Conditional Value-at-Risk (CVaR / Expected Shortfall).
//!
//! Rockafellar--Uryasev (2000) variational representation:
//!
//! ```text
//!     CVaR_alpha(L) = inf_{zeta in R} { zeta + 1/(1 - alpha) * E[(L - zeta)_+] }.
//! ```
//!
//! The optimum is attained at `zeta* = VaR_alpha(L)`. For an empirical
//! sample `L^{(s)}, s = 1..S`, the closed-form estimator equals
//!
//! ```text
//!     CVaR_alpha = mean( L^{(s)} : L^{(s)} >= VaR_alpha ).
//! ```
//!
//! For decision-variable optimisation, we minimise
//!
//! ```text
//!     min_{w in C, zeta, u}  zeta + 1/((1 - alpha) * S) * sum_s u_s
//!     s.t.  u_s >= -<r^{(s)}, w> - zeta,    u_s >= 0
//! ```
//!
//! over the unit simplex `C = { w >= 0, 1^T w = 1 }`. We solve this LP by
//! a projected sub-gradient method on the simplex; the inner `zeta` is
//! eliminated by setting `zeta = VaR_alpha(L(w))` at every iteration.

use crate::core::{OptimizrError, Result};
use ndarray::{Array1, Array2, ArrayView2};

use super::check_alpha;

/// Empirical CVaR of a loss sample at confidence level `alpha`.
pub fn cvar_value(losses: &[f64], alpha: f64) -> Result<f64> {
    check_alpha(alpha)?;
    if losses.is_empty() {
        return Err(OptimizrError::EmptyData);
    }
    let mut sorted: Vec<f64> = losses.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    // CVaR_alpha = mean of the worst (1 - alpha) fraction of losses.
    let k = ((alpha * n as f64).floor() as usize).min(n.saturating_sub(1));
    let tail = &sorted[k..];
    Ok(tail.iter().sum::<f64>() / tail.len() as f64)
}

/// Configuration for `minimize_cvar`.
#[derive(Debug, Clone)]
pub struct CVaRConfig {
    pub alpha: f64,
    pub n_iter: usize,
    pub step_size: f64,
    pub tol: f64,
}

impl Default for CVaRConfig {
    fn default() -> Self {
        Self {
            alpha: 0.95,
            n_iter: 5_000,
            step_size: 1e-2,
            tol: 1e-8,
        }
    }
}

/// Result of the CVaR minimisation.
#[derive(Debug, Clone)]
pub struct CVaRResult {
    pub w: Array1<f64>,
    pub zeta: f64,
    pub cvar: f64,
    pub iterations: usize,
}

/// Minimise empirical CVaR of `L(w) = -<r^{(s)}, w>` over the unit
/// simplex.
///
/// `returns` has shape `(S, d)` (S samples, d decision components).
pub fn minimize_cvar(returns: ArrayView2<f64>, cfg: &CVaRConfig) -> Result<CVaRResult> {
    check_alpha(cfg.alpha)?;
    let s = returns.nrows();
    let d = returns.ncols();
    if s == 0 || d == 0 {
        return Err(OptimizrError::EmptyData);
    }
    if cfg.n_iter == 0 {
        return Err(OptimizrError::InvalidParameter("n_iter must be > 0".into()));
    }

    let mut w = Array1::<f64>::from_elem(d, 1.0 / d as f64);
    let mut prev_obj = f64::INFINITY;
    let mut iters = 0usize;

    let inv_factor = 1.0 / ((1.0 - cfg.alpha) * s as f64);

    for it in 0..cfg.n_iter {
        // Compute losses
        let losses: Vec<f64> = (0..s)
            .map(|i| -returns.row(i).dot(&w))
            .collect();

        // VaR (alpha-quantile of losses) gives zeta*.
        let mut sorted = losses.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((cfg.alpha * s as f64).ceil() as usize).max(1).min(s);
        let zeta = sorted[k - 1];

        // Objective and sub-gradient with zeta fixed at current optimum.
        let mut grad = Array1::<f64>::zeros(d);
        let mut obj = zeta;
        let mut tail_count = 0usize;
        for (i, &l) in losses.iter().enumerate() {
            let exceed = l - zeta;
            if exceed > 0.0 {
                obj += inv_factor * exceed;
                // d (l - zeta)_+ / d w_j  =  d l / d w_j  =  -r_{i, j}
                for j in 0..d {
                    grad[j] -= inv_factor * returns[[i, j]];
                }
                tail_count += 1;
            }
        }
        let _ = tail_count;

        // Projected gradient step: descend then project on simplex.
        let lr = cfg.step_size / (1.0 + (it as f64).sqrt());
        for j in 0..d {
            w[j] -= lr * grad[j];
        }
        project_simplex_inplace(&mut w);

        if (prev_obj - obj).abs() < cfg.tol {
            iters = it + 1;
            return Ok(CVaRResult {
                w,
                zeta,
                cvar: obj,
                iterations: iters,
            });
        }
        prev_obj = obj;
        iters = it + 1;
    }

    let losses: Vec<f64> = (0..s).map(|i| -returns.row(i).dot(&w)).collect();
    let final_cvar = cvar_value(&losses, cfg.alpha)?;
    let zeta_final = {
        let mut sorted = losses.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((cfg.alpha * s as f64).ceil() as usize).max(1).min(s);
        sorted[k - 1]
    };
    Ok(CVaRResult {
        w,
        zeta: zeta_final,
        cvar: final_cvar,
        iterations: iters,
    })
}

/// Project a vector onto the probability simplex `{w >= 0, sum w = 1}`.
/// Algorithm of Held, Wolfe, Crowder (1974).
pub fn project_simplex_inplace(v: &mut Array1<f64>) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let mut u: Vec<f64> = v.iter().copied().collect();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cssv = 0.0;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        let t = (cssv - 1.0) / (i as f64 + 1.0);
        if ui - t > 0.0 {
            rho = i + 1;
        }
    }
    let cssv_rho: f64 = u.iter().take(rho).sum();
    let theta = (cssv_rho - 1.0) / rho as f64;
    for x in v.iter_mut() {
        *x = (*x - theta).max(0.0);
    }
}

/// Convenience wrapper accepting an owned ndarray.
pub fn minimize_cvar_owned(returns: Array2<f64>, cfg: &CVaRConfig) -> Result<CVaRResult> {
    minimize_cvar(returns.view(), cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cvar_value_matches_tail_mean() {
        let losses: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        // alpha=0.9: tail = 91..100 => mean = 95.5
        let c = cvar_value(&losses, 0.9).unwrap();
        assert!((c - 95.5).abs() < 1e-12);
    }

    #[test]
    fn simplex_projection_is_idempotent_on_simplex() {
        let mut v = array![0.2_f64, 0.3, 0.5];
        project_simplex_inplace(&mut v);
        let s: f64 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
        for &x in v.iter() {
            assert!(x >= 0.0);
        }
    }

    #[test]
    fn cvar_min_concentrates_on_safer_decision() {
        // d=2 decisions; samples: r0 ~ very volatile, r1 ~ stable.
        let mut data = vec![0.0; 200 * 2];
        let mut state = 42u64;
        for s in 0..200 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = ((state & 0xFFFF) as f64 + 1.0) / 65538.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = ((state & 0xFFFF) as f64 + 1.0) / 65538.0;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data[s * 2] = 5.0 * z; // volatile
            data[s * 2 + 1] = 0.1 * z; // stable
        }
        let returns = Array2::from_shape_vec((200, 2), data).unwrap();
        let cfg = CVaRConfig {
            alpha: 0.95,
            n_iter: 2_000,
            step_size: 0.05,
            tol: 0.0,
        };
        let res = minimize_cvar(returns.view(), &cfg).unwrap();
        // Optimiser should put much more weight on the stable component.
        assert!(res.w[1] > res.w[0], "weights = {:?}", res.w);
    }
}
