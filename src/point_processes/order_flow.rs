//! Order Flow Analysis Module
//!
//! Implements the unified theory framework for analyzing signed and unsigned
//! order flow, estimating H₀, and deriving market impact and volatility.
//!
//! # Key Relationships (from unified theory)
//!
//! Given H₀ ≈ 3/4 (persistence of core flow):
//! - Signed order flow: Hurst index H₀
//! - Unsigned volume: Hurst index H₁ = H₀ - 1/2 ≈ 0.25 (rough)
//! - Volatility: Hurst index H_vol = 2H₀ - 3/2 ≈ 0 (very rough)
//! - Market impact: power law exponent δ = 2 - 2H₀ ≈ 0.5 (square root)

use crate::point_processes::mixed_fbm::{FractionalBM, MixedFractionalBM};
use std::f64::consts::PI;

/// Parameters from the unified theory
#[derive(Clone, Debug)]
pub struct UnifiedTheoryParams {
    /// H₀: Hurst index of signed order flow / core flow persistence
    /// Typically H₀ ≈ 0.75 empirically
    pub h0: f64,
    
    /// μ₀: Baseline intensity scaling constant
    pub mu0: f64,
    
    /// λ₀: Decay rate parameter
    pub lambda0: f64,
}

impl UnifiedTheoryParams {
    pub fn new(h0: f64) -> Self {
        assert!(h0 > 0.5 && h0 < 1.0, "H0 must be in (0.5, 1)");
        Self {
            h0,
            mu0: 1.0,
            lambda0: 1.0,
        }
    }

    /// Create with typical empirical value H₀ ≈ 0.75
    pub fn empirical() -> Self {
        Self::new(0.75)
    }

    /// Hurst index of unsigned volume: H₁ = H₀ - 1/2
    pub fn volume_hurst(&self) -> f64 {
        self.h0 - 0.5
    }

    /// Hurst index of volatility: H_vol = 2H₀ - 3/2
    pub fn volatility_hurst(&self) -> f64 {
        2.0 * self.h0 - 1.5
    }

    /// Market impact exponent: δ = 2 - 2H₀
    pub fn impact_exponent(&self) -> f64 {
        2.0 - 2.0 * self.h0
    }

    /// Check if mfBM is semimartingale (H₀ > 3/4)
    pub fn is_semimartingale(&self) -> bool {
        self.h0 > 0.75
    }

    /// α₀ (tail exponent): H₀ = 2α₀, so α₀ = H₀/2
    pub fn alpha0(&self) -> f64 {
        self.h0 / 2.0
    }
}

/// Calculate signed order flow Hurst index (= H₀)
pub fn signed_order_flow(h0: f64) -> f64 {
    h0
}

/// Calculate unsigned volume Hurst index: H₁ = H₀ - 1/2
pub fn unsigned_volume(h0: f64) -> f64 {
    h0 - 0.5
}

/// Alias for unsigned_volume
pub fn volume_hurst(h0: f64) -> f64 {
    unsigned_volume(h0)
}

/// Calculate volatility Hurst index: H_vol = 2H₀ - 3/2
pub fn volatility_hurst(h0: f64) -> f64 {
    2.0 * h0 - 1.5
}

/// Calculate market impact exponent: δ = 2 - 2H₀
/// Impact ~ Q^δ where Q is order size
pub fn market_impact_exponent(h0: f64) -> f64 {
    2.0 - 2.0 * h0
}

/// Order flow metrics computed from data
#[derive(Clone, Debug)]
pub struct OrderFlowMetrics {
    /// Estimated H₀ (core flow persistence)
    pub h0: f64,
    
    /// Estimated Hurst of signed flow (under fBM assumption)
    pub h_signed_fbm: f64,
    
    /// Estimated Hurst of signed flow (under mfBM assumption)
    pub h_signed_mfbm: f64,
    
    /// Estimated Hurst of unsigned volume
    pub h_unsigned: f64,
    
    /// Total signed flow
    pub total_signed: f64,
    
    /// Total unsigned volume
    pub total_unsigned: f64,
    
    /// Implied volatility Hurst
    pub h_volatility: f64,
    
    /// Implied impact exponent
    pub impact_exponent: f64,
    
    /// Autocorrelation at lag 1
    pub acf_1: f64,
    
    /// Scale-dependent Hurst estimates
    pub scale_hurst: Vec<(usize, f64)>,
}

/// Analyzer for order flow data
#[derive(Clone, Debug, Default)]
pub struct OrderFlowAnalyzer {
    /// Whether to compute scale-dependent statistics
    pub compute_scales: bool,
    
    /// Scales for multi-scale analysis
    pub scales: Vec<usize>,
}

impl OrderFlowAnalyzer {
    pub fn new() -> Self {
        Self {
            compute_scales: true,
            scales: vec![10, 50, 100, 500, 1000, 2000, 5000],
        }
    }

    /// Analyze signed order flow
    ///
    /// # Arguments
    /// * `flow` - Signed order flow data (positive = buy, negative = sell)
    pub fn analyze_signed_flow(&self, flow: &[f64]) -> OrderFlowMetrics {
        let n = flow.len();
        if n < 100 {
            return self.default_metrics();
        }

        // Cumulative signed flow
        let mut cum_flow = vec![0.0; n + 1];
        for i in 0..n {
            cum_flow[i + 1] = cum_flow[i] + flow[i];
        }

        // Estimate H under fBM assumption (R/S analysis)
        let h_fbm = FractionalBM::estimate_hurst(&cum_flow);

        // Estimate H under mfBM assumption (scale-dependent)
        let scale_hurst = MixedFractionalBM::scale_dependent_hurst(
            &cum_flow,
            &self.scales,
        );
        
        // Average of high-frequency estimates (mfBM martingale component dominates)
        let h_mfbm_hf: f64 = scale_hurst
            .iter()
            .filter(|(s, _)| *s < 100)
            .map(|(_, h)| *h)
            .sum::<f64>() / scale_hurst.iter().filter(|(s, _)| *s < 100).count().max(1) as f64;

        // Average of low-frequency estimates (fBM component dominates)
        let h_mfbm_lf: f64 = scale_hurst
            .iter()
            .filter(|(s, _)| *s >= 500)
            .map(|(_, h)| *h)
            .sum::<f64>() / scale_hurst.iter().filter(|(s, _)| *s >= 500).count().max(1) as f64;

        // H₀ estimate: use low-frequency Hurst (persistent component)
        let h0 = if h_mfbm_lf > 0.5 { h_mfbm_lf } else { h_fbm };

        // Unsigned volume (absolute values)
        let unsigned: Vec<f64> = flow.iter().map(|x| x.abs()).collect();
        let cum_unsigned: Vec<f64> = unsigned.iter()
            .scan(0.0, |acc, &x| { *acc += x; Some(*acc) })
            .collect();
        let h_unsigned = estimate_hurst_variance_ratio(&cum_unsigned);

        // Autocorrelation
        let mean: f64 = flow.iter().sum::<f64>() / n as f64;
        let var: f64 = flow.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let acf_1 = if var > 1e-10 {
            let cov: f64 = flow[..n-1].iter().zip(flow[1..].iter())
                .map(|(x, y)| (x - mean) * (y - mean))
                .sum::<f64>() / (n - 1) as f64;
            cov / var
        } else {
            0.0
        };

        OrderFlowMetrics {
            h0,
            h_signed_fbm: h_fbm,
            h_signed_mfbm: h_mfbm_lf,
            h_unsigned,
            total_signed: cum_flow.last().copied().unwrap_or(0.0),
            total_unsigned: cum_unsigned.last().copied().unwrap_or(0.0),
            h_volatility: volatility_hurst(h0),
            impact_exponent: market_impact_exponent(h0),
            acf_1,
            scale_hurst,
        }
    }

    /// Analyze order arrival times  
    ///
    /// # Arguments
    /// * `buy_times` - Buy order arrival times
    /// * `sell_times` - Sell order arrival times
    /// * `t_max` - Maximum time  
    /// * `bin_size` - Time bin for aggregation
    pub fn analyze_arrivals(
        &self,
        buy_times: &[f64],
        sell_times: &[f64],
        t_max: f64,
        bin_size: f64,
    ) -> OrderFlowMetrics {
        // Bin the arrivals
        let n_bins = (t_max / bin_size).ceil() as usize;
        let mut signed_flow = vec![0.0; n_bins];
        
        for &t in buy_times {
            let bin = ((t / bin_size).floor() as usize).min(n_bins - 1);
            signed_flow[bin] += 1.0;
        }
        for &t in sell_times {
            let bin = ((t / bin_size).floor() as usize).min(n_bins - 1);
            signed_flow[bin] -= 1.0;
        }

        self.analyze_signed_flow(&signed_flow)
    }

    /// Estimate H₀ from signed order flow data
    pub fn estimate_h0(&self, flow: &[f64]) -> f64 {
        self.analyze_signed_flow(flow).h0
    }

    fn default_metrics(&self) -> OrderFlowMetrics {
        OrderFlowMetrics {
            h0: 0.75,
            h_signed_fbm: 0.5,
            h_signed_mfbm: 0.75,
            h_unsigned: 0.25,
            total_signed: 0.0,
            total_unsigned: 0.0,
            h_volatility: 0.0,
            impact_exponent: 0.5,
            acf_1: 0.0,
            scale_hurst: Vec::new(),
        }
    }
}

/// Estimate Hurst exponent using variance ratio method
fn estimate_hurst_variance_ratio(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 50 {
        return 0.5;
    }

    let scales = [5, 10, 20, 40, 80, 160];
    let mut log_scales = Vec::new();
    let mut log_vars = Vec::new();

    for &s in &scales {
        if s >= n / 4 {
            break;
        }

        // Compute increments at scale s
        let increments: Vec<f64> = (s..n).map(|i| data[i] - data[i - s]).collect();
        if increments.is_empty() {
            continue;
        }

        let var: f64 = increments.iter().map(|x| x.powi(2)).sum::<f64>() / increments.len() as f64;
        if var > 1e-15 {
            log_scales.push((s as f64).ln());
            log_vars.push(var.ln());
        }
    }

    if log_scales.len() < 3 {
        return 0.5;
    }

    // Linear regression: log(var) = 2H * log(scale) + const
    let n_pts = log_scales.len() as f64;
    let mean_x: f64 = log_scales.iter().sum::<f64>() / n_pts;
    let mean_y: f64 = log_vars.iter().sum::<f64>() / n_pts;

    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in log_scales.iter().zip(log_vars.iter()) {
        num += (x - mean_x) * (y - mean_y);
        den += (x - mean_x).powi(2);
    }

    let slope = num / den;
    (slope / 2.0).clamp(0.01, 0.99)
}

/// Market impact function: Impact(Q) ~ Q^δ where δ = 2 - 2H₀
#[derive(Clone, Debug)]
pub struct MarketImpact {
    /// Impact exponent δ
    pub delta: f64,
    /// Scaling constant
    pub scale: f64,
}

impl MarketImpact {
    /// Create from unified theory parameter H₀
    pub fn from_h0(h0: f64, scale: f64) -> Self {
        Self {
            delta: 2.0 - 2.0 * h0,
            scale,
        }
    }

    /// Create square-root impact (H₀ = 0.75)
    pub fn square_root(scale: f64) -> Self {
        Self {
            delta: 0.5,
            scale,
        }
    }

    /// Calculate market impact for order size Q
    pub fn impact(&self, q: f64) -> f64 {
        self.scale * q.abs().powf(self.delta) * q.signum()
    }

    /// Inverse: order size needed for target impact
    pub fn order_size(&self, target_impact: f64) -> f64 {
        (target_impact.abs() / self.scale).powf(1.0 / self.delta) * target_impact.signum()
    }
}

/// Temporary price impact with decay
#[derive(Clone, Debug)]
pub struct TransientImpact {
    /// Impact function
    pub impact: MarketImpact,
    /// Decay kernel exponent (controls how impact dissipates)
    pub decay_exponent: f64,
}

impl TransientImpact {
    pub fn from_h0(h0: f64, scale: f64) -> Self {
        Self {
            impact: MarketImpact::from_h0(h0, scale),
            decay_exponent: 2.0 * h0 - 1.0,  // Derived from no-arbitrage
        }
    }

    /// Decay kernel G(t) ~ t^{-(2H₀-1)}
    pub fn decay(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        t.powf(-self.decay_exponent)
    }

    /// Total impact at time t from orders (times, sizes)
    pub fn total_impact(&self, t: f64, orders: &[(f64, f64)]) -> f64 {
        let mut total = 0.0;
        for (order_time, size) in orders {
            if *order_time < t {
                let elapsed = t - order_time;
                total += self.impact.impact(*size) * self.decay(elapsed);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_theory_params() {
        let params = UnifiedTheoryParams::new(0.75);
        
        assert!((params.volume_hurst() - 0.25).abs() < 1e-10);
        assert!((params.volatility_hurst() - 0.0).abs() < 1e-10);
        assert!((params.impact_exponent() - 0.5).abs() < 1e-10);
        assert!(params.is_semimartingale() == false);  // H₀ = 0.75 is boundary
        
        let params_high = UnifiedTheoryParams::new(0.8);
        assert!(params_high.is_semimartingale());
    }

    #[test]
    fn test_market_impact() {
        let impact = MarketImpact::square_root(1.0);
        
        // Impact should be proportional to sqrt(Q)
        let i1 = impact.impact(100.0);
        let i4 = impact.impact(400.0);
        
        // i4 / i1 ≈ 2 (since sqrt(400) / sqrt(100) = 2)
        assert!((i4 / i1 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_order_flow_analysis() {
        // Generate some synthetic order flow
        use rand::prelude::*;
        use rand_distr::Normal;
        
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // Add some persistence
        let mut flow = vec![0.0; 1000];
        flow[0] = rng.sample(normal);
        for i in 1..1000 {
            flow[i] = 0.3 * flow[i-1] + rng.sample(normal);
        }
        
        let analyzer = OrderFlowAnalyzer::new();
        let metrics = analyzer.analyze_signed_flow(&flow);
        
        // Hurst should be between 0 and 1
        assert!(metrics.h0 > 0.0 && metrics.h0 < 1.0);
        assert!(metrics.h_signed_fbm > 0.0 && metrics.h_signed_fbm < 1.0);
    }
}
