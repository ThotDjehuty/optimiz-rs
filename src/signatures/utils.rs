//! Helper algebraic operations on flat tensors of the truncated tensor
//! algebra `T^M((R^d))`.
//!
//! Provides:
//!
//! * `shuffle_product`         -- shuffle product of two flat words.
//! * `concatenate_signatures`  -- Chen identity-driven concatenation.

use super::path_signature::TruncatedSignature;
use crate::core::Result;

/// Shuffle product of two words `u, v` of length `p, q` over the
/// alphabet `{0, ..., d - 1}`.
///
/// Returned as a `HashMap<Vec<usize>, f64>` keyed by word.
pub fn shuffle_product(
    u: &[usize],
    v: &[usize],
) -> std::collections::HashMap<Vec<usize>, f64> {
    let mut acc = std::collections::HashMap::new();
    fn rec(
        u: &[usize],
        v: &[usize],
        prefix: &mut Vec<usize>,
        acc: &mut std::collections::HashMap<Vec<usize>, f64>,
    ) {
        if u.is_empty() && v.is_empty() {
            *acc.entry(prefix.clone()).or_insert(0.0) += 1.0;
            return;
        }
        if !u.is_empty() {
            prefix.push(u[0]);
            rec(&u[1..], v, prefix, acc);
            prefix.pop();
        }
        if !v.is_empty() {
            prefix.push(v[0]);
            rec(u, &v[1..], prefix, acc);
            prefix.pop();
        }
    }
    let mut prefix = Vec::new();
    rec(u, v, &mut prefix, &mut acc);
    acc
}

/// Concatenate two signatures via Chen's identity: `S(X * Y) = S(X) otimes S(Y)`.
pub fn concatenate_signatures(
    a: &TruncatedSignature,
    b: &TruncatedSignature,
) -> Result<TruncatedSignature> {
    assert_eq!(a.channels, b.channels);
    assert_eq!(a.level, b.level);
    let d = a.channels;
    let m = a.level;
    let mut out = TruncatedSignature::identity(d, m);
    for n in 0..=m {
        for buf in out.tensors[n].iter_mut() {
            *buf = 0.0;
        }
        for k in 0..=n {
            let nk = n - k;
            let stride = d.pow(nk as u32);
            for alpha in 0..d.pow(k as u32) {
                let av = a.tensors[k][alpha];
                if av == 0.0 {
                    continue;
                }
                let base = alpha * stride;
                for beta in 0..stride {
                    out.tensors[n][base + beta] += av * b.tensors[nk][beta];
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_one_letter_words() {
        let res = shuffle_product(&[0], &[1]);
        assert_eq!(res.len(), 2);
        assert!((res[&vec![0, 1]] - 1.0).abs() < 1e-12);
        assert!((res[&vec![1, 0]] - 1.0).abs() < 1e-12);
    }
}
