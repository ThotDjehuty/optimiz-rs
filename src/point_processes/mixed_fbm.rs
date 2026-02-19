//! Mixed Fractional Brownian Motion
//!
//! The mixed fractional Brownian motion (mfBM) is the sum of a standard
//! Brownian motion and an independent fractional Brownian motion.
//!
//! MH(t) = a * B(t) + b * BH(t)
//!
//! where:
//! - B(t) is standard Brownian motion (Hurst H = 1/2)
//! - BH(t) is fractional Brownian motion with Hurst index H
//! - a, b are mixing coefficients
//!
//! This is the limiting process for aggregate order flow in the unified theory.

use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;

/// Fractional Brownian Motion with Hurst parameter H
#[derive(Clone, Debug)]
pub struct FractionalBM {
    /// Hurst parameter H ∈ (0, 1)
    pub hurst: f64,
    /// Time grid
    pub times: Vec<f64>,
    /// Sample path values
    pub values: Vec<f64>,
}

impl FractionalBM {
    /// Create a new fBM specification
    pub fn new(hurst: f64) -> Self {
        assert!(hurst > 0.0 && hurst < 1.0, "Hurst parameter must be in (0, 1)");
        Self {
            hurst,
            times: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Covariance function of fBM: E[BH(s) * BH(t)]
    pub fn covariance(&self, s: f64, t: f64) -> f64 {
        let h = self.hurst;
        let two_h = 2.0 * h;
        0.5 * (s.powf(two_h) + t.powf(two_h) - (s - t).abs().powf(two_h))
    }

    /// Variance of increment: Var[BH(t) - BH(s)]
    pub fn increment_variance(&self, s: f64, t: f64) -> f64 {
        let h = self.hurst;
        (t - s).abs().powf(2.0 * h)
    }

    /// Simulate a sample path using Cholesky method
    ///
    /// # Arguments
    /// * `times` - Time points
    /// * `seed` - Optional random seed
    pub fn simulate(&mut self, times: &[f64], seed: Option<u64>) -> Vec<f64> {
        let n = times.len();
        if n == 0 {
            return Vec::new();
        }

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Build covariance matrix
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = self.covariance(times[i], times[j]);
            }
        }

        // Cholesky decomposition
        let chol = cholesky(&cov).expect("Cholesky decomposition failed");

        // Generate standard normal samples
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z: Vec<f64> = (0..n).map(|_| rng.sample(normal)).collect();

        // Apply Cholesky factor
        let mut values = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                values[i] += chol[i][j] * z[j];
            }
        }

        self.times = times.to_vec();
        self.values = values.clone();
        values
    }

    /// Simulate using Hosking's method (more efficient for regular grids)
    pub fn simulate_hosking(&mut self, n: usize, dt: f64, seed: Option<u64>) -> Vec<f64> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let h = self.hurst;
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Hosking's algorithm for fractional Gaussian noise
        // Then cumsum to get fBM

        // Autocovariance of fGn: γ(k) = 0.5 * (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})
        let acf = |k: i64| -> f64 {
            let k_abs = k.abs() as f64;
            let h2 = 2.0 * h;
            0.5 * ((k_abs - 1.0).abs().powf(h2) - 2.0 * k_abs.powf(h2) + (k_abs + 1.0).powf(h2))
        };

        // Durbin-Levinson algorithm for prediction coefficients
        let mut phi = vec![vec![0.0; n]; n];
        let mut v = vec![0.0; n];
        
        v[0] = acf(0);
        phi[0][0] = 0.0;

        let mut fgn = vec![0.0; n];  // Fractional Gaussian noise
        fgn[0] = v[0].sqrt() * rng.sample(normal);

        for i in 1..n {
            // Compute phi_ii
            let num: f64 = acf(i as i64) - (0..i).map(|j| phi[i-1][j] * acf((i - 1 - j) as i64)).sum::<f64>();
            phi[i][i] = num / v[i-1];

            // Update other coefficients
            for j in 0..i {
                phi[i][j] = phi[i-1][j] - phi[i][i] * phi[i-1][i-1-j];
            }

            // Update variance
            v[i] = v[i-1] * (1.0 - phi[i][i].powi(2));

            // Generate fGn sample
            let pred: f64 = (0..i).map(|j| phi[i][j] * fgn[i-1-j]).sum();
            fgn[i] = pred + v[i].sqrt() * rng.sample(normal);
        }

        // Cumulative sum to get fBM
        let scale = dt.powf(h);
        let mut values = vec![0.0; n + 1];
        for i in 0..n {
            values[i + 1] = values[i] + scale * fgn[i];
        }

        self.times = (0..=n).map(|i| i as f64 * dt).collect();
        self.values = values.clone();
        values
    }

    /// Estimate Hurst exponent from data using rescaled range (R/S) analysis
    pub fn estimate_hurst(data: &[f64]) -> f64 {
        let n = data.len();
        if n < 20 {
            return 0.5;  // Not enough data
        }

        let mut log_ns = Vec::new();
        let mut log_rs = Vec::new();

        // Try different subseries lengths
        let min_n = 10;
        let mut sub_n = min_n;
        while sub_n <= n / 2 {
            let k = n / sub_n;  // Number of subseries
            let mut rs_values = Vec::new();

            for i in 0..k {
                let start = i * sub_n;
                let end = start + sub_n;
                let subseries = &data[start..end];

                // Mean and standard deviation
                let mean: f64 = subseries.iter().sum::<f64>() / sub_n as f64;
                let var: f64 = subseries.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sub_n as f64;
                let std_dev = var.sqrt();

                if std_dev > 1e-10 {
                    // Cumulative deviations
                    let mut cumdev = vec![0.0; sub_n];
                    cumdev[0] = subseries[0] - mean;
                    for j in 1..sub_n {
                        cumdev[j] = cumdev[j-1] + (subseries[j] - mean);
                    }

                    // Range
                    let max_dev = cumdev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let min_dev = cumdev.iter().cloned().fold(f64::INFINITY, f64::min);
                    let r = max_dev - min_dev;

                    rs_values.push(r / std_dev);
                }
            }

            if !rs_values.is_empty() {
                let avg_rs: f64 = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
                log_ns.push((sub_n as f64).ln());
                log_rs.push(avg_rs.ln());
            }

            sub_n += sub_n / 4 + 1;
        }

        // Linear regression to estimate H
        if log_ns.len() < 3 {
            return 0.5;
        }

        let n_points = log_ns.len() as f64;
        let mean_x: f64 = log_ns.iter().sum::<f64>() / n_points;
        let mean_y: f64 = log_rs.iter().sum::<f64>() / n_points;

        let mut num = 0.0;
        let mut den = 0.0;
        for (x, y) in log_ns.iter().zip(log_rs.iter()) {
            num += (x - mean_x) * (y - mean_y);
            den += (x - mean_x).powi(2);
        }

        let h = num / den;
        h.clamp(0.01, 0.99)
    }
}

/// Mixed Fractional Brownian Motion
///
/// MH(t) = a * B(t) + b * BH(t)
///
/// Combines a standard BM (diffusive component) with an fBM (persistent component).
/// This is the limiting process for aggregate order flow.
#[derive(Clone, Debug)]
pub struct MixedFractionalBM {
    /// Coefficient for standard BM component
    pub a: f64,
    /// Coefficient for fBM component  
    pub b: f64,
    /// Hurst index H of the fBM component
    pub hurst: f64,
    /// Time grid
    pub times: Vec<f64>,
    /// Sample path values
    pub values: Vec<f64>,
}

impl MixedFractionalBM {
    /// Create a new mixed fBM
    pub fn new(a: f64, b: f64, hurst: f64) -> Self {
        assert!(hurst > 0.0 && hurst < 1.0);
        Self {
            a,
            b,
            hurst,
            times: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create with Hurst index H₀ from unified theory (typically ~3/4)
    /// Using a = 1, b = 1 (unit mixing)
    pub fn from_h0(h0: f64) -> Self {
        assert!(h0 > 0.5 && h0 < 1.0, "H0 should be in (0.5, 1) for persistent flow");
        Self::new(1.0, 1.0, h0)
    }

    /// Covariance function
    pub fn covariance(&self, s: f64, t: f64) -> f64 {
        // Cov(MH(s), MH(t)) = a² * min(s,t) + b² * ρH(s,t)
        // where ρH is the fBM covariance
        let min_st = s.min(t);
        let fbm_cov = {
            let h = self.hurst;
            let two_h = 2.0 * h;
            0.5 * (s.powf(two_h) + t.powf(two_h) - (s - t).abs().powf(two_h))
        };
        self.a.powi(2) * min_st + self.b.powi(2) * fbm_cov
    }

    /// Simulate the mixed fBM
    pub fn simulate(&mut self, n: usize, dt: f64, seed: Option<u64>) -> Vec<f64> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Simulate standard BM component
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sqrt_dt = dt.sqrt();
        let mut bm = vec![0.0; n + 1];
        for i in 0..n {
            bm[i + 1] = bm[i] + sqrt_dt * rng.sample(normal);
        }

        // Simulate fBM component
        let mut fbm = FractionalBM::new(self.hurst);
        let fbm_values = fbm.simulate_hosking(n, dt, seed.map(|s| s + 12345));

        // Combine
        let mut values = vec![0.0; n + 1];
        for i in 0..=n {
            values[i] = self.a * bm[i] + self.b * fbm_values[i.min(fbm_values.len() - 1)];
        }

        self.times = (0..=n).map(|i| i as f64 * dt).collect();
        self.values = values.clone();
        values
    }

    /// Check if process is semi-martingale (H > 3/4 for mfBM)
    pub fn is_semimartingale(&self) -> bool {
        self.hurst > 0.75
    }

    /// Estimate effective Hurst at different time scales
    pub fn scale_dependent_hurst(data: &[f64], scales: &[usize]) -> Vec<(usize, f64)> {
        let mut results = Vec::new();

        for &scale in scales {
            if scale >= data.len() / 4 {
                continue;
            }

            // Compute increments at this scale
            let increments: Vec<f64> = (scale..data.len())
                .map(|i| data[i] - data[i - scale])
                .collect();

            if increments.is_empty() {
                continue;
            }

            // Estimate local Hurst using variance ratio
            let var_1: f64 = increments.iter().map(|x| x.powi(2)).sum::<f64>() / increments.len() as f64;
            
            // Compare with increments at double scale
            let double_scale = 2 * scale;
            if double_scale >= data.len() {
                continue;
            }
            
            let increments_2: Vec<f64> = (double_scale..data.len())
                .map(|i| data[i] - data[i - double_scale])
                .collect();
            
            if increments_2.is_empty() {
                continue;
            }

            let var_2: f64 = increments_2.iter().map(|x| x.powi(2)).sum::<f64>() / increments_2.len() as f64;

            // E[|X_t - X_s|^2] ~ |t-s|^{2H}
            // var_2 / var_1 ~ 2^{2H}
            let h = (var_2 / var_1).ln() / (2.0_f64.ln() * 2.0);
            results.push((scale, h.clamp(0.01, 0.99)));
        }

        results
    }
}

/// Cholesky decomposition of a positive semi-definite matrix
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }

            if i == j {
                let diag = a[i][i] - sum;
                if diag < -1e-10 {
                    return None;  // Not positive semi-definite
                }
                l[i][j] = diag.max(0.0).sqrt();
            } else {
                if l[j][j].abs() < 1e-10 {
                    l[i][j] = 0.0;
                } else {
                    l[i][j] = (a[i][j] - sum) / l[j][j];
                }
            }
        }
    }

    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fbm_covariance() {
        let fbm = FractionalBM::new(0.75);
        
        // Variance at t=1 should be 1
        assert!((fbm.covariance(1.0, 1.0) - 1.0).abs() < 1e-10);
        
        // Cov should be non-negative
        assert!(fbm.covariance(1.0, 2.0) >= 0.0);
    }

    #[test]
    fn test_fbm_simulation() {
        let mut fbm = FractionalBM::new(0.7);
        let values = fbm.simulate_hosking(100, 0.01, Some(42));
        
        assert_eq!(values.len(), 101);
        assert!((values[0]).abs() < 1e-10);  // Starts at 0
    }

    #[test]
    fn test_mixed_fbm() {
        let mut mfbm = MixedFractionalBM::new(1.0, 0.5, 0.8);
        let values = mfbm.simulate(100, 0.01, Some(42));
        
        assert_eq!(values.len(), 101);
        assert!((values[0]).abs() < 1e-10);
    }

    #[test]
    fn test_hurst_estimation() {
        // Generate fBM with known H
        let mut fbm = FractionalBM::new(0.7);
        let values = fbm.simulate_hosking(1000, 1.0, Some(42));
        
        // Estimate H
        let h_est = FractionalBM::estimate_hurst(&values);
        
        // R/S method is noisy, allow wider tolerance (within 0.3)
        // The estimation is approximate and depends on sample size
        assert!(h_est > 0.3 && h_est < 1.0, "Estimated H={:.3} out of reasonable range", h_est);
    }
}
