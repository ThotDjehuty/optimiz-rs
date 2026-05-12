//! Discrete-time optimal switching (Snell envelope)
//! ==================================================
//!
//! Solves
//!
//! ```text
//! V_k(i) = ψ_k(i) + max_{ j ≠ i } [ V_{k+1}(j) - c(i, j) ],   V_N(i) = G(i)
//! ```
//!
//! by backward induction.  Here `i ∈ {0, ..., M-1}` is the current operating
//! mode, `ψ_k(i)` is the stage reward, `c(i, j) ≥ 0` is the cost of
//! switching from `i` to `j` (zero on the diagonal), and `G` is the
//! terminal payoff.

use crate::core::{OptimizrError, Result};

#[derive(Clone, Debug)]
pub struct SwitchingConfig {
    pub n_modes: usize,
    pub n_steps: usize,
}

#[derive(Clone, Debug)]
pub struct SwitchingResult {
    /// `value[k * n_modes + i] = V_k(i)`.
    pub value: Vec<f64>,
    /// `policy[k * n_modes + i] = optimal next mode at time k from mode i`.
    pub policy: Vec<usize>,
}

pub fn solve_optimal_switching<R, T>(
    stage_reward: R,
    terminal_payoff: T,
    switching_cost: &[f64],
    cfg: &SwitchingConfig,
) -> Result<SwitchingResult>
where
    R: Fn(usize, usize) -> f64,
    T: Fn(usize) -> f64,
{
    if cfg.n_modes == 0 || cfg.n_steps == 0 {
        return Err(OptimizrError::InvalidParameter(
            "n_modes and n_steps must be > 0".into(),
        ));
    }
    if switching_cost.len() != cfg.n_modes * cfg.n_modes {
        return Err(OptimizrError::DimensionMismatch {
            expected: cfg.n_modes * cfg.n_modes,
            actual: switching_cost.len(),
        });
    }
    for i in 0..cfg.n_modes {
        if switching_cost[i * cfg.n_modes + i].abs() > 1e-12 {
            return Err(OptimizrError::InvalidParameter(
                "diagonal of switching cost must be zero".into(),
            ));
        }
        for j in 0..cfg.n_modes {
            if switching_cost[i * cfg.n_modes + j] < 0.0 {
                return Err(OptimizrError::InvalidParameter(
                    "switching costs must be non-negative".into(),
                ));
            }
        }
    }

    let m = cfg.n_modes;
    let n = cfg.n_steps;
    let mut value = vec![0.0f64; (n + 1) * m];
    let mut policy = vec![0usize; (n + 1) * m];
    for i in 0..m {
        value[n * m + i] = terminal_payoff(i);
        policy[n * m + i] = i;
    }
    for k in (0..n).rev() {
        for i in 0..m {
            let mut best_val = f64::NEG_INFINITY;
            let mut best_j = i;
            for j in 0..m {
                let candidate = value[(k + 1) * m + j] - switching_cost[i * m + j];
                if candidate > best_val {
                    best_val = candidate;
                    best_j = j;
                }
            }
            value[k * m + i] = stage_reward(k, i) + best_val;
            policy[k * m + i] = best_j;
        }
    }
    Ok(SwitchingResult { value, policy })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two modes, mode 1 always pays 1, mode 0 pays 0.  Switching is free.
    /// Optimal: stay in mode 1 from the start. V_0(1) = T, V_0(0) = T.
    #[test]
    fn free_switching_picks_paying_mode() {
        let m = 2;
        let n = 5;
        let cost = vec![0.0; m * m]; // free
        let cfg = SwitchingConfig { n_modes: m, n_steps: n };
        let res = solve_optimal_switching(
            |_, i| if i == 1 { 1.0 } else { 0.0 },
            |_| 0.0,
            &cost,
            &cfg,
        )
        .unwrap();
        // V_k(0) = sum_{l=k}^{n-1} max stage reward = 1·(n-k) (switch immediately at no cost,
        // but stage reward at time k is paid in current mode i; here we receive 0 at time k
        // and best continuation from any next mode j chosen at time k).
        // Concretely: at time k, choose next mode 1; pay no cost; V_{k+1}(1) accrues.
        // V_n(·) = 0 → V_{n-1}(0) = 0 + (V_n(1) - 0) = 0
        //          V_{n-1}(1) = 1 + 0 = 1
        // V_{n-2}(0) = 0 + V_{n-1}(1) = 1
        // V_{n-2}(1) = 1 + V_{n-1}(1) = 2
        // ⇒ V_0(0) = n - 1 = 4, V_0(1) = n = 5
        assert!((res.value[0] - 4.0).abs() < 1e-12);
        assert!((res.value[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn high_switch_cost_locks_mode() {
        let m = 2;
        let n = 3;
        let cost = vec![0.0, 1e6, 1e6, 0.0];
        let cfg = SwitchingConfig { n_modes: m, n_steps: n };
        let res = solve_optimal_switching(|_, i| i as f64, |_| 0.0, &cost, &cfg).unwrap();
        // Starting in mode 0: switching to 1 costs 1e6 → never switch.
        // V_0(0) = 0 (stage) + V_1(0) = 0 + 0 + V_2(0) = 0 + V_3(0) = 0.
        assert!((res.value[0]).abs() < 1e-9);
    }
}
