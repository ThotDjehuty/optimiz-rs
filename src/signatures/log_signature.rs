//! Log-signature in the free Lie algebra.
//!
//! Given the truncated signature `S` of a path, the log-signature
//!
//! ```text
//!     log(S) = sum_{n>=1} (-1)^{n+1} / n  * (S - 1)^{otimes n}
//! ```
//!
//! is a Lie series; its coefficients in any Hall basis form a more
//! parsimonious representation of the path.
//!
//! This implementation:
//!
//! * computes the truncated tensor logarithm `log_M(S)` of a
//!   `TruncatedSignature` (Lyons-Victoir 2007 eq. 2.13);
//! * stores the result as flat tensors `[L_1, L_2, ..., L_M]` (the
//!   level-0 component is identically zero).
//!
//! No projection onto a Hall basis is performed -- the truncated tensor
//! logarithm itself already lives in the truncated free Lie algebra,
//! and consumers can project it onto any preferred basis.

use crate::core::{OptimizrError, Result};

use super::path_signature::TruncatedSignature;

/// Truncated tensor logarithm.
#[derive(Debug, Clone)]
pub struct TruncatedLogSignature {
    pub channels: usize,
    pub level: usize,
    /// `tensors[k]` has length `channels^k`; `tensors[0]` is empty by
    /// convention.
    pub tensors: Vec<Vec<f64>>,
}

fn truncated_tensor_product(a: &[Vec<f64>], b: &[Vec<f64>], d: usize, level: usize) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = (0..=level).map(|k| vec![0.0; d.pow(k as u32)]).collect();
    for n in 0..=level {
        for k in 0..=n {
            let nk = n - k;
            let stride = d.pow(nk as u32);
            for alpha in 0..d.pow(k as u32) {
                let av = a[k][alpha];
                if av == 0.0 {
                    continue;
                }
                let base = alpha * stride;
                for beta in 0..stride {
                    out[n][base + beta] += av * b[nk][beta];
                }
            }
        }
    }
    out
}

/// Compute the truncated tensor logarithm of a signature.
pub fn log_signature(sig: &TruncatedSignature) -> Result<TruncatedLogSignature> {
    let d = sig.channels;
    let m = sig.level;
    if m == 0 {
        return Err(OptimizrError::InvalidParameter(
            "log-signature requires level >= 1".into(),
        ));
    }
    // Build (S - 1) by removing the level-0 unit element.
    let mut s_minus_one: Vec<Vec<f64>> = sig.tensors.clone();
    s_minus_one[0][0] = 0.0;
    // Power series: log(1 + x) = sum_{n>=1} (-1)^{n+1}/n x^n
    let mut log_tensors: Vec<Vec<f64>> = (0..=m).map(|k| vec![0.0; d.pow(k as u32)]).collect();
    let mut x_power = s_minus_one.clone();
    for n in 1..=m {
        let coef = if n % 2 == 0 { -1.0 / (n as f64) } else { 1.0 / (n as f64) };
        for k in 0..=m {
            for (i, v) in x_power[k].iter().enumerate() {
                log_tensors[k][i] += coef * v;
            }
        }
        if n < m {
            x_power = truncated_tensor_product(&x_power, &s_minus_one, d, m);
        }
    }
    Ok(TruncatedLogSignature {
        channels: d,
        level: m,
        tensors: log_tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signatures::path_signature::path_signature;

    #[test]
    fn linear_segment_log_signature_first_level_equals_increment() {
        let path = vec![vec![0.0, 0.0], vec![0.3, -0.2]];
        let sig = path_signature(&path, 2).unwrap();
        let log = log_signature(&sig).unwrap();
        // The first-level part of the log-signature equals the path increment.
        assert!((log.tensors[1][0] - 0.3).abs() < 1e-12);
        assert!((log.tensors[1][1] + 0.2).abs() < 1e-12);
    }
}
