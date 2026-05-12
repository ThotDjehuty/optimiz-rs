//! Discrete Wavelet Transform (DWT) and Maximal-Overlap DWT (MODWT).
//!
//! Pyramid algorithm with orthogonal mirror filter pairs (`h`, `g`).
//! For a level `j`,
//!
//! ```text
//!     W_{j,k} = sum_n h_n * V_{j-1, 2k - n}
//!     V_{j,k} = sum_n g_n * V_{j-1, 2k - n}
//! ```
//!
//! `MODWT` (a.k.a. translation-invariant wavelet transform) skips
//! down-sampling and rescales the filters by `2^{-j/2}`:
//!
//! ```text
//!     W~_{j,t} = sum_n (h_n / sqrt(2^j)) * V~_{j-1, t - 2^{j-1} n}
//! ```
//!
//! # Reference
//!
//! Daubechies (1992), *Ten Lectures on Wavelets*. Percival & Walden
//! (2000), *Wavelet Methods for Time Series Analysis*.

use crate::core::{OptimizrError, Result};

/// Orthogonal scaling-filter families.
#[derive(Debug, Clone, Copy)]
pub enum WaveletFamily {
    Haar,
    /// Daubechies-N: `n` even integer in {2,4,6,8,10}.
    Daubechies(u8),
}

/// Returns the (decomposition) low-pass scaling filter `g_n` for the
/// requested family. The high-pass wavelet filter is obtained by the QMF
/// relation `h_n = (-1)^n g_{L-1-n}`.
pub fn scaling_filter(family: WaveletFamily) -> Result<Vec<f64>> {
    let g = match family {
        WaveletFamily::Haar => vec![std::f64::consts::FRAC_1_SQRT_2; 2],
        WaveletFamily::Daubechies(n) => match n {
            2 => vec![std::f64::consts::FRAC_1_SQRT_2; 2],
            4 => {
                let s3 = 3.0_f64.sqrt();
                let denom = 4.0 * 2.0_f64.sqrt();
                vec![
                    (1.0 + s3) / denom,
                    (3.0 + s3) / denom,
                    (3.0 - s3) / denom,
                    (1.0 - s3) / denom,
                ]
            }
            6 => vec![
                0.332_670_552_950_082_6,
                0.806_891_509_311_092_5,
                0.459_877_502_118_491_7,
                -0.135_011_020_010_254_6,
                -0.085_441_273_882_026_7,
                0.035_226_291_882_100_7,
            ],
            8 => vec![
                0.230_377_813_308_896_5,
                0.714_846_570_552_915_6,
                0.630_880_767_929_858_9,
                -0.027_983_769_416_859_8,
                -0.187_034_811_719_092_3,
                0.030_841_381_835_560_8,
                0.032_883_011_666_885_2,
                -0.010_597_401_785_069_0,
            ],
            10 => vec![
                0.160_102_397_974_193_0,
                0.603_829_269_797_473_5,
                0.724_308_528_438_574_0,
                0.138_428_145_901_320_3,
                -0.242_294_887_066_382_4,
                -0.032_244_869_585_030_3,
                0.077_571_493_840_065_1,
                -0.006_241_490_212_798_3,
                -0.012_580_751_999_082_0,
                0.003_335_725_285_473_8,
            ],
            other => {
                return Err(OptimizrError::InvalidParameter(format!(
                    "Daubechies({}) not supported (use 2,4,6,8,10)",
                    other
                )))
            }
        },
    };
    Ok(g)
}

#[inline]
fn qmf(g: &[f64]) -> Vec<f64> {
    let l = g.len();
    let mut h = vec![0.0; l];
    for n in 0..l {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        h[n] = sign * g[l - 1 - n];
    }
    h
}

/// One-level DWT step with periodic boundary extension.
///
/// Returns `(approximation, detail)` each of length `signal.len() / 2`.
pub fn dwt_step(signal: &[f64], family: WaveletFamily) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = signal.len();
    if n < 2 || n % 2 != 0 {
        return Err(OptimizrError::InvalidInput(
            "signal length must be even and >= 2".into(),
        ));
    }
    let g = scaling_filter(family)?;
    let h = qmf(&g);
    let l = g.len();
    let half = n / 2;
    let mut approx = vec![0.0; half];
    let mut detail = vec![0.0; half];
    for k in 0..half {
        let mut a = 0.0;
        let mut d = 0.0;
        for j in 0..l {
            // periodic boundary
            let idx = ((2 * k + l - 1 - j) % n + n) % n;
            a += g[j] * signal[idx];
            d += h[j] * signal[idx];
        }
        approx[k] = a;
        detail[k] = d;
    }
    Ok((approx, detail))
}

/// Multi-level pyramid DWT. Returns `(approx_J, [d_1, ..., d_J])`.
pub fn dwt(signal: &[f64], family: WaveletFamily, n_levels: usize) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut approx = signal.to_vec();
    let mut details = Vec::with_capacity(n_levels);
    for _ in 0..n_levels {
        let (a, d) = dwt_step(&approx, family)?;
        details.push(d);
        approx = a;
    }
    Ok((approx, details))
}

/// Maximal-overlap DWT step (no down-sampling, dilated filter at level `j`).
pub fn modwt_step(
    signal: &[f64],
    family: WaveletFamily,
    level: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = signal.len();
    if n == 0 {
        return Err(OptimizrError::EmptyData);
    }
    if level == 0 {
        return Err(OptimizrError::InvalidParameter("level must be >= 1".into()));
    }
    let g = scaling_filter(family)?;
    let h = qmf(&g);
    let scale = (2.0_f64).powf(level as f64 / 2.0);
    let g_tilde: Vec<f64> = g.iter().map(|x| x / scale).collect();
    let h_tilde: Vec<f64> = h.iter().map(|x| x / scale).collect();
    let stride = 1usize << (level - 1); // 2^{j-1}
    let l = g.len();

    let mut approx = vec![0.0; n];
    let mut detail = vec![0.0; n];
    for t in 0..n {
        let mut a = 0.0;
        let mut d = 0.0;
        for j in 0..l {
            let idx = (t + n - (j * stride) % n) % n;
            a += g_tilde[j] * signal[idx];
            d += h_tilde[j] * signal[idx];
        }
        approx[t] = a;
        detail[t] = d;
    }
    Ok((approx, detail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_step_constant_signal_zero_detail() {
        let signal = vec![1.0_f64; 8];
        let (a, d) = dwt_step(&signal, WaveletFamily::Haar).unwrap();
        assert_eq!(a.len(), 4);
        assert_eq!(d.len(), 4);
        for v in d.iter() {
            assert!(v.abs() < 1e-12, "detail should be zero, got {}", v);
        }
        let expected_a = std::f64::consts::SQRT_2;
        for v in a.iter() {
            assert!((v - expected_a).abs() < 1e-12);
        }
    }

    #[test]
    fn db4_filters_have_unit_norm() {
        let g = scaling_filter(WaveletFamily::Daubechies(4)).unwrap();
        let s: f64 = g.iter().map(|x| x * x).sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn modwt_runs_on_short_signal() {
        let signal: Vec<f64> = (0..16).map(|k| (k as f64).sin()).collect();
        let (a, d) = modwt_step(&signal, WaveletFamily::Daubechies(4), 1).unwrap();
        assert_eq!(a.len(), 16);
        assert_eq!(d.len(), 16);
    }
}
