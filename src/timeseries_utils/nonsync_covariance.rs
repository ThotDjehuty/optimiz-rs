//! Hayashi--Yoshida non-synchronous covariance estimator.
//!
//! Given two real-valued processes observed on heterogeneous time grids
//! `t^{(1)}_0 < ... < t^{(1)}_{n_1}` and `t^{(2)}_0 < ... < t^{(2)}_{n_2}`,
//! the Hayashi--Yoshida estimator is
//!
//! ```text
//!     Sigma_{1,2}^{HY} = sum_k sum_l  Dr1_k * Dr2_l * 1{ I1_k inter I2_l != empty }
//! ```
//!
//! where `I_i^{(k)} = ( t_{i,k-1}, t_{i,k} ]` is the k-th native
//! observation interval of series i and `Dr_i^{(k)}` the corresponding
//! increment.
//!
//! The implementation is `O(n_1 + n_2)` thanks to the monotonicity of the
//! grids: a two-pointer sweep finds the first index `l` of grid 2 whose
//! interval intersects the current interval of grid 1 and stops as soon as
//! the intervals separate.
//!
//! # Reference
//!
//! Hayashi & Yoshida (2005), *On covariance estimation of non-synchronously
//! observed diffusion processes*, Bernoulli 11(2).

use rayon::prelude::*;

use crate::core::{OptimizrError, Result};

/// Compute the Hayashi--Yoshida covariance between two non-synchronous
/// observation series.
///
/// `t1` (resp. `t2`) is a strictly increasing time grid of length
/// `v1.len()` (resp. `v2.len()`). The increments used by the estimator
/// are `dv_i^{(k)} = v_i[k] - v_i[k-1]` over `(t_i[k-1], t_i[k]]`.
pub fn hayashi_yoshida_covariance(
    t1: &[f64],
    v1: &[f64],
    t2: &[f64],
    v2: &[f64],
) -> Result<f64> {
    if t1.len() != v1.len() || t2.len() != v2.len() {
        return Err(OptimizrError::InvalidInput(
            "time grid and value series lengths differ".into(),
        ));
    }
    if t1.len() < 2 || t2.len() < 2 {
        return Err(OptimizrError::InvalidInput(
            "each series needs at least two observations".into(),
        ));
    }
    for w in t1.windows(2) {
        if !(w[0] < w[1]) {
            return Err(OptimizrError::InvalidInput(
                "t1 must be strictly increasing".into(),
            ));
        }
    }
    for w in t2.windows(2) {
        if !(w[0] < w[1]) {
            return Err(OptimizrError::InvalidInput(
                "t2 must be strictly increasing".into(),
            ));
        }
    }

    let n1 = t1.len() - 1;
    let n2 = t2.len() - 1;

    // For each interval of series 1, sweep series 2 and accumulate.
    // Use a sequential search start index that only moves forward, so the
    // cost is O(n1 + n2). Wrapped in Rayon when n1*n2 > 1e5 for parallel
    // accumulation; the sweep is preserved per worker via local pointer.
    let parallel = (n1 as u64) * (n2 as u64) > 100_000;

    let intervals1: Vec<(f64, f64, f64)> = (0..n1)
        .map(|k| (t1[k], t1[k + 1], v1[k + 1] - v1[k]))
        .collect();
    let intervals2: Vec<(f64, f64, f64)> = (0..n2)
        .map(|k| (t2[k], t2[k + 1], v2[k + 1] - v2[k]))
        .collect();

    let acc = if parallel {
        intervals1
            .par_iter()
            .map(|&(a1, b1, dr1)| {
                // Binary search for first interval of series 2 that ends
                // strictly above a1 (left-open right-closed convention).
                let start =
                    intervals2.partition_point(|&(_, b2, _)| b2 <= a1);
                let mut s = 0.0;
                for &(a2, b2, dr2) in intervals2[start..].iter() {
                    if a2 >= b1 {
                        break;
                    }
                    let _ = b2;
                    s += dr1 * dr2;
                }
                s
            })
            .sum::<f64>()
    } else {
        let mut s = 0.0;
        let mut start = 0usize;
        for &(a1, b1, dr1) in intervals1.iter() {
            // advance start until b2 > a1
            while start < n2 && intervals2[start].1 <= a1 {
                start += 1;
            }
            let mut l = start;
            while l < n2 {
                let (a2, _b2, dr2) = intervals2[l];
                if a2 >= b1 {
                    break;
                }
                s += dr1 * dr2;
                l += 1;
            }
        }
        s
    };

    Ok(acc)
}

/// Build the full Hayashi--Yoshida covariance matrix for `d` series.
///
/// `series` is a slice of `(time_grid, values)` pairs. Returns a
/// `d x d` symmetric matrix flattened in row-major order.
pub fn hayashi_yoshida_matrix(
    series: &[(Vec<f64>, Vec<f64>)],
) -> Result<Vec<Vec<f64>>> {
    let d = series.len();
    if d == 0 {
        return Err(OptimizrError::EmptyData);
    }
    let mut out = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in i..d {
            let cij = hayashi_yoshida_covariance(
                &series[i].0,
                &series[i].1,
                &series[j].0,
                &series[j].1,
            )?;
            out[i][j] = cij;
            out[j][i] = cij;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synchronous_grid_matches_classical_covariance() {
        let n = 1000;
        let t: Vec<f64> = (0..=n).map(|k| k as f64).collect();
        let mut x = vec![0.0_f64; n + 1];
        let mut y = vec![0.0_f64; n + 1];
        // X = W,  Y = W + N(0)  perfectly correlated increments
        let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
        for k in 0..n {
            // simple xorshift gauss approximation via Box-Muller
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u1 = ((rng_state & 0xFFFF_FFFF) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let u2 = ((rng_state & 0xFFFF_FFFF) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x[k + 1] = x[k] + z;
            y[k + 1] = x[k + 1]; // perfect correlation
        }
        let cov = hayashi_yoshida_covariance(&t, &x, &t, &y).unwrap();
        let var_x = hayashi_yoshida_covariance(&t, &x, &t, &x).unwrap();
        let var_y = hayashi_yoshida_covariance(&t, &y, &t, &y).unwrap();
        let rho = cov / (var_x.sqrt() * var_y.sqrt());
        assert!((rho - 1.0).abs() < 1e-12, "rho = {}", rho);
    }

    #[test]
    fn nonsync_subsampling_recovers_correlation() {
        // Generate correlated bivariate Brownian increments, then subsample
        // each series independently. HY estimator should recover rho.
        let n = 20_000;
        let dt = 1.0 / n as f64;
        let rho_true = 0.6_f64;
        let t_full: Vec<f64> = (0..=n).map(|k| k as f64 * dt).collect();
        let mut x = vec![0.0_f64; n + 1];
        let mut y = vec![0.0_f64; n + 1];

        let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
        let mut gauss = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = ((state & 0xFFFF_FFFF) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = ((state & 0xFFFF_FFFF) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let sqrt_dt = dt.sqrt();
        for k in 0..n {
            let z1 = gauss();
            let z2 = rho_true * z1 + (1.0 - rho_true * rho_true).sqrt() * gauss();
            x[k + 1] = x[k] + sqrt_dt * z1;
            y[k + 1] = y[k] + sqrt_dt * z2;
        }

        // Subsample with Bernoulli(0.5) keeping endpoints
        let mut keep_x = vec![true; n + 1];
        let mut keep_y = vec![true; n + 1];
        for k in 1..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            keep_x[k] = (state & 1) == 0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            keep_y[k] = (state & 1) == 0;
        }
        let collect = |keep: &[bool], v: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let mut tt = Vec::new();
            let mut vv = Vec::new();
            for k in 0..=n {
                if keep[k] {
                    tt.push(t_full[k]);
                    vv.push(v[k]);
                }
            }
            (tt, vv)
        };
        let (t_x, x_s) = collect(&keep_x, &x);
        let (t_y, y_s) = collect(&keep_y, &y);

        let cov = hayashi_yoshida_covariance(&t_x, &x_s, &t_y, &y_s).unwrap();
        let var_x = hayashi_yoshida_covariance(&t_x, &x_s, &t_x, &x_s).unwrap();
        let var_y = hayashi_yoshida_covariance(&t_y, &y_s, &t_y, &y_s).unwrap();
        let rho = cov / (var_x.sqrt() * var_y.sqrt());
        assert!(
            (rho - rho_true).abs() < 0.05,
            "estimated rho = {}",
            rho
        );
    }
}
