//! Generic quadratic-impact controlled SDE (Phase 6 of the v2.0.0 plan)
//! =====================================================================
//!
//! Considers a 1-D controlled SDE of the form
//!
//! ```text
//! dq_t = u_t dt + σ dW_t,    q_0 given,    t ∈ [0, T]
//! ```
//!
//! with a quadratic running cost `L(q, u) = γ u² + φ q²` and terminal cost
//! `g(q) = A q²`.  The associated HJB equation is solvable in closed form:
//! `V(t, q) = h(t) q²` with the Riccati ODE
//!
//! ```text
//! h'(t) = h(t)² / γ - φ,    h(T) = A.
//! ```
//!
//! We integrate `h` backwards in time and return the resulting time-varying
//! optimal feedback `u*_t = -(1/γ) h(t) q`.  The exposition is purely
//! generic — no domain-specific vocabulary.

use crate::core::{OptimizrError, Result};
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct QuadraticImpactConfig {
    /// Control penalty `γ > 0`.
    pub gamma: f64,
    /// State penalty `φ ≥ 0`.
    pub phi: f64,
    /// Terminal weight `A ≥ 0`.
    pub a_terminal: f64,
    pub t_horizon: f64,
    pub n_steps: usize,
}

#[derive(Clone, Debug)]
pub struct QuadraticImpactResult {
    pub time_grid: Array1<f64>,
    pub h: Array1<f64>,
    /// Feedback gain `k(t) = h(t) / γ`.
    pub feedback_gain: Array1<f64>,
}

pub fn solve_quadratic_impact_control(
    cfg: &QuadraticImpactConfig,
) -> Result<QuadraticImpactResult> {
    if cfg.gamma <= 0.0 {
        return Err(OptimizrError::InvalidParameter("gamma > 0 required".into()));
    }
    if cfg.phi < 0.0 || cfg.a_terminal < 0.0 {
        return Err(OptimizrError::InvalidParameter("phi, a_terminal ≥ 0".into()));
    }
    if cfg.n_steps == 0 || cfg.t_horizon <= 0.0 {
        return Err(OptimizrError::InvalidParameter("n_steps>0 and T>0".into()));
    }
    let n = cfg.n_steps;
    let dt = cfg.t_horizon / n as f64;
    let time_grid: Array1<f64> = Array1::from_iter((0..=n).map(|k| k as f64 * dt));
    let mut h = vec![0.0f64; n + 1];
    h[n] = cfg.a_terminal;
    for k in (0..n).rev() {
        let hn = h[k + 1];
        let dh = hn * hn / cfg.gamma - cfg.phi;
        h[k] = hn - dt * dh;
        if !h[k].is_finite() {
            return Err(OptimizrError::NumericalError("Riccati blew up".into()));
        }
    }
    let h_arr = Array1::from(h);
    let feedback_gain = h_arr.mapv(|hv| hv / cfg.gamma);
    Ok(QuadraticImpactResult { time_grid, h: h_arr, feedback_gain })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `φ = 0`, `A = 0` makes the ODE trivial: `h ≡ 0`, optimal control is 0.
    #[test]
    fn no_running_or_terminal_penalty_gives_zero_feedback() {
        let cfg = QuadraticImpactConfig {
            gamma: 1.0, phi: 0.0, a_terminal: 0.0,
            t_horizon: 1.0, n_steps: 100,
        };
        let res = solve_quadratic_impact_control(&cfg).unwrap();
        for v in res.h.iter() { assert!(v.abs() < 1e-12); }
        for v in res.feedback_gain.iter() { assert!(v.abs() < 1e-12); }
    }

    /// `γ = φ = A = 1`: closed-form `h(t) = tanh(T - t + atanh(1)) → ∞`. We
    /// avoid the singularity by checking against the analytic ODE on a short
    /// horizon `T = 0.4`.  At t = T, h = 1; the ODE yields h decreasing as
    /// we integrate backward (h² - 1 < 0 for h < 1)? Actually h² - 1 = 0 at
    /// h = 1, so h stays exactly 1.  Verify numerically.
    #[test]
    fn unit_riccati_fixed_point() {
        let cfg = QuadraticImpactConfig {
            gamma: 1.0, phi: 1.0, a_terminal: 1.0,
            t_horizon: 0.5, n_steps: 500,
        };
        let res = solve_quadratic_impact_control(&cfg).unwrap();
        for v in res.h.iter() {
            assert!((v - 1.0).abs() < 1e-9, "h drifted from fixed point: {v}");
        }
    }
}
