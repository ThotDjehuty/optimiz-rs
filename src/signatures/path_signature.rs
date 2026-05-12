//! Truncated tensor path signature (Lyons 1998).
//!
//! For a continuous path `X : [0, T] -> R^d` of bounded variation, the
//! signature is the formal series
//!
//! ```text
//!     S(X)_{0,T} = 1 + sum_{k>=1} sum_{i_1, ..., i_k}  S^{i_1, ..., i_k}_{0,T} e_{i_1} otimes ... otimes e_{i_k}
//! ```
//!
//! with iterated Stieltjes integrals
//!
//! ```text
//!     S^{i_1, ..., i_k}_{0,T} = int_{0 < u_1 < ... < u_k < T} dX^{i_1}_{u_1} ... dX^{i_k}_{u_k}.
//! ```
//!
//! For a piecewise-linear path with increments `Delta_n in R^d`, the
//! signature truncated at level `M` satisfies the recursion
//!
//! ```text
//!     S^{(M)}_{0, t_n} = S^{(M)}_{0, t_{n-1}} otimes_M exp_M(Delta_n)
//! ```
//!
//! where `otimes_M` is the truncated tensor product and `exp_M` the
//! truncated tensor exponential.
//!
//! Storage convention: a signature truncated at level `M` is a vector of
//! flat tensors `S = [s_0, s_1, ..., s_M]` where `s_k in R^{d^k}` is
//! stored row-major with strides `(d^{k-1}, d^{k-2}, ..., 1)`.

use crate::core::{OptimizrError, Result};

/// Signature truncated at level `M`.
#[derive(Debug, Clone)]
pub struct TruncatedSignature {
    pub channels: usize,
    pub level: usize,
    pub tensors: Vec<Vec<f64>>, // tensors[k] has length channels^k
}

impl TruncatedSignature {
    pub fn identity(channels: usize, level: usize) -> Self {
        let mut tensors = Vec::with_capacity(level + 1);
        for k in 0..=level {
            let len = channels.pow(k as u32);
            let mut t = vec![0.0; len];
            if k == 0 {
                t[0] = 1.0;
            }
            tensors.push(t);
        }
        Self {
            channels,
            level,
            tensors,
        }
    }
}

/// Truncated tensor product `(a otimes b)` up to level `M`.
fn truncated_tensor_product(a: &TruncatedSignature, b: &TruncatedSignature) -> TruncatedSignature {
    let d = a.channels;
    let m = a.level;
    let mut out = TruncatedSignature::identity(d, m);
    for n in 0..=m {
        let len_n = d.pow(n as u32);
        for buf in out.tensors[n].iter_mut() {
            *buf = 0.0;
        }
        for k in 0..=n {
            let nk = n - k;
            let a_k = &a.tensors[k];
            let b_nk = &b.tensors[nk];
            // out[n][i_1...i_n] += a[k][i_1...i_k] * b[nk][i_{k+1}...i_n]
            // Linearised: for each pair (alpha in [0, d^k), beta in [0, d^{nk})), out[alpha * d^{nk} + beta] += a[alpha] * b[beta]
            let stride = d.pow(nk as u32);
            for alpha in 0..d.pow(k as u32) {
                let av = a_k[alpha];
                if av == 0.0 {
                    continue;
                }
                let base = alpha * stride;
                for beta in 0..stride {
                    out.tensors[n][base + beta] += av * b_nk[beta];
                }
            }
            let _ = len_n;
        }
    }
    out
}

/// Truncated tensor exponential of an increment `Delta in R^d`.
///
/// `exp_M(Delta) = 1 + Delta + Delta^{otimes 2}/2! + ... + Delta^{otimes M}/M!`
fn truncated_exponential(delta: &[f64], level: usize) -> TruncatedSignature {
    let d = delta.len();
    let mut sig = TruncatedSignature::identity(d, level);
    if level == 0 || d == 0 {
        return sig;
    }
    // Build delta^{otimes k}/k! incrementally.
    let mut current = vec![1.0]; // delta^0
    let mut current_dim = 0usize;
    let mut factorial = 1.0_f64;
    for k in 1..=level {
        let new_len = d.pow(k as u32);
        let mut new = vec![0.0; new_len];
        let stride = d;
        for alpha in 0..current.len() {
            let cv = current[alpha];
            if cv == 0.0 {
                continue;
            }
            let base = alpha * stride;
            for j in 0..d {
                new[base + j] = cv * delta[j];
            }
        }
        current = new;
        current_dim = k;
        factorial *= k as f64;
        let inv = 1.0 / factorial;
        for (i, v) in sig.tensors[k].iter_mut().enumerate() {
            *v = current[i] * inv;
        }
    }
    let _ = current_dim;
    sig
}

/// Compute the truncated signature of a piecewise-linear path given by
/// its sample points `path` of shape `(n_steps + 1, d)` (row-major).
pub fn path_signature(path: &[Vec<f64>], level: usize) -> Result<TruncatedSignature> {
    if path.len() < 2 {
        return Err(OptimizrError::InvalidInput(
            "path must contain at least two points".into(),
        ));
    }
    let d = path[0].len();
    if d == 0 {
        return Err(OptimizrError::InvalidInput("zero-dimensional path".into()));
    }
    for p in path {
        if p.len() != d {
            return Err(OptimizrError::DimensionMismatch {
                expected: d,
                actual: p.len(),
            });
        }
    }
    let mut sig = TruncatedSignature::identity(d, level);
    for n in 1..path.len() {
        let mut delta = vec![0.0; d];
        for j in 0..d {
            delta[j] = path[n][j] - path[n - 1][j];
        }
        let inc_sig = truncated_exponential(&delta, level);
        sig = truncated_tensor_product(&sig, &inc_sig);
    }
    Ok(sig)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For a single linear segment `X(t) = t * Delta`, the signature reads
    /// `S^{i_1, ..., i_k} = (Delta^{i_1} ... Delta^{i_k}) / k!`.
    #[test]
    fn linear_segment_signature_matches_exponential() {
        let delta = vec![0.3, -0.2];
        let path = vec![vec![0.0, 0.0], delta.clone()];
        let sig = path_signature(&path, 3).unwrap();
        // level 1: Delta itself
        for j in 0..2 {
            assert!((sig.tensors[1][j] - delta[j]).abs() < 1e-12);
        }
        // level 2: Delta_i Delta_j / 2
        for i in 0..2 {
            for j in 0..2 {
                let exp = delta[i] * delta[j] / 2.0;
                assert!((sig.tensors[2][i * 2 + j] - exp).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn empty_path_rejected() {
        assert!(path_signature(&Vec::<Vec<f64>>::new(), 2).is_err());
    }
}
