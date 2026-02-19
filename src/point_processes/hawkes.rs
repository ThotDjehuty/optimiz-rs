//! Hawkes Process Implementation
//!
//! Self-exciting point processes where past events increase the probability
//! of future events. Used to model order flow clustering in financial markets.

use super::kernels::ExcitationKernel;
use rand::prelude::*;
use rand_distr::{Exp, Uniform};
use std::collections::VecDeque;

/// Configuration for a Hawkes process
#[derive(Clone, Debug)]
pub struct HawkesProcessConfig<K: ExcitationKernel> {
    /// Baseline intensity ν > 0
    pub baseline_intensity: f64,
    /// Excitation kernel
    pub kernel: K,
    /// Maximum history to track (for efficiency)
    pub max_history: usize,
    /// Numerical tolerance
    pub tolerance: f64,
}

impl<K: ExcitationKernel> HawkesProcessConfig<K> {
    pub fn new(baseline_intensity: f64, kernel: K) -> Self {
        assert!(baseline_intensity > 0.0);
        Self {
            baseline_intensity,
            kernel,
            max_history: 10000,
            tolerance: 1e-10,
        }
    }

    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.max_history = max_history;
        self
    }
}

/// Univariate Hawkes process
///
/// Models self-exciting point processes where the intensity is:
/// λ(t) = ν + ∫₀ᵗ⁻ φ(t - s) dN(s)
///
/// where ν is the baseline intensity and φ is the excitation kernel.
#[derive(Clone, Debug)]
pub struct HawkesProcess<K: ExcitationKernel> {
    config: HawkesProcessConfig<K>,
    /// Event times
    events: Vec<f64>,
    /// Current time
    current_time: f64,
}

impl<K: ExcitationKernel> HawkesProcess<K> {
    /// Create a new Hawkes process
    pub fn new(baseline_intensity: f64, kernel: K) -> Self {
        let config = HawkesProcessConfig::new(baseline_intensity, kernel);
        Self {
            config,
            events: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Create from configuration
    pub fn from_config(config: HawkesProcessConfig<K>) -> Self {
        Self {
            config,
            events: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Get the intensity at time t
    pub fn intensity(&self, t: f64) -> f64 {
        let mut lambda = self.config.baseline_intensity;
        for &event_time in &self.events {
            if event_time < t {
                lambda += self.config.kernel.evaluate(t - event_time);
            }
        }
        lambda
    }

    /// Simulate the process up to time T
    pub fn simulate(&mut self, t_max: f64, seed: Option<u64>) -> Vec<f64> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        self.events.clear();
        self.current_time = 0.0;

        // Use Ogata's thinning algorithm
        let lambda_max_factor = 1.5;  // Safety factor for upper bound

        while self.current_time < t_max {
            // Upper bound for intensity
            let lambda_max = self.intensity(self.current_time) * lambda_max_factor + 
                             self.config.baseline_intensity;

            // Generate candidate inter-arrival time
            let exp_dist = Exp::new(lambda_max).unwrap();
            let tau: f64 = rng.sample(exp_dist);
            let candidate_time = self.current_time + tau;

            if candidate_time >= t_max {
                break;
            }

            // Accept/reject
            let u: f64 = rng.gen();
            let lambda_at_candidate = self.intensity(candidate_time);

            if u <= lambda_at_candidate / lambda_max {
                // Accept the event
                self.events.push(candidate_time);
                self.current_time = candidate_time;

                // Prune old events if needed
                if self.events.len() > self.config.max_history {
                    let recent_events: Vec<f64> = self.events
                        .iter()
                        .rev()
                        .take(self.config.max_history)
                        .cloned()
                        .collect();
                    self.events = recent_events.into_iter().rev().collect();
                }
            } else {
                self.current_time = candidate_time;
            }
        }

        self.events.clone()
    }

    /// Estimate kernel parameters from event times using MLE
    pub fn fit(&mut self, events: &[f64]) -> Result<FitResult, &'static str> {
        if events.is_empty() {
            return Err("No events provided");
        }

        self.events = events.to_vec();
        let t_max = *events.last().unwrap();
        let n = events.len();

        // Compute log-likelihood
        let log_likelihood = self.log_likelihood(t_max);

        // Estimate baseline intensity
        let lambda_avg = n as f64 / t_max;
        let branching_ratio = self.config.kernel.l1_norm();
        let estimated_baseline = lambda_avg * (1.0 - branching_ratio);

        Ok(FitResult {
            log_likelihood,
            n_events: n,
            duration: t_max,
            estimated_baseline,
            estimated_branching_ratio: branching_ratio,
        })
    }

    /// Compute log-likelihood of the process
    pub fn log_likelihood(&self, t_max: f64) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }

        // Log-likelihood = Σ log(λ(tᵢ)) - ∫₀ᵀ λ(t) dt
        let mut ll = 0.0;

        // Sum of log intensities at event times
        for &t in &self.events {
            let lambda_t = self.intensity(t);
            if lambda_t > self.config.tolerance {
                ll += lambda_t.ln();
            }
        }

        // Compensator (integrated intensity)
        // ∫₀ᵀ λ(t) dt = νT + Σᵢ ∫₀^{T-tᵢ} φ(s) ds
        let compensator = self.config.baseline_intensity * t_max
            + self.events.iter()
                .map(|&ti| self.config.kernel.integrate(t_max - ti))
                .sum::<f64>();

        ll - compensator
    }

    /// Get all event times
    pub fn events(&self) -> &[f64] {
        &self.events
    }

    /// Get number of events
    pub fn n_events(&self) -> usize {
        self.events.len()
    }
}

/// Result of fitting a Hawkes process
#[derive(Clone, Debug)]
pub struct FitResult {
    pub log_likelihood: f64,
    pub n_events: usize,
    pub duration: f64,
    pub estimated_baseline: f64,
    pub estimated_branching_ratio: f64,
}

/// Bivariate Hawkes process for reaction order flow
///
/// Models the interplay between buy and sell reaction orders.
/// N = (N⁺, N⁻) with intensity:
///
/// λ⁺(t) = μ⁺(t) + ∫ [φ₁(t-s)dN⁺(s) + φ₂(t-s)dN⁻(s)]
/// λ⁻(t) = μ⁻(t) + ∫ [φ₂(t-s)dN⁺(s) + φ₁(t-s)dN⁻(s)]
///
/// where μ(t) is driven by the core order flow.
#[derive(Clone, Debug)]
pub struct BivariateHawkes<K: ExcitationKernel> {
    /// Same-side kernel φ₁ (buy->buy, sell->sell)
    pub phi_1: K,
    /// Cross-side kernel φ₂ (buy->sell, sell->buy)  
    pub phi_2: K,
    /// Buy events N⁺
    pub buy_events: Vec<f64>,
    /// Sell events N⁻
    pub sell_events: Vec<f64>,
    /// Current time
    current_time: f64,
}

impl<K: ExcitationKernel> BivariateHawkes<K> {
    pub fn new(phi_1: K, phi_2: K) -> Self {
        Self {
            phi_1,
            phi_2,
            buy_events: Vec::new(),
            sell_events: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Get buy intensity at time t given external baseline μ⁺(t)
    pub fn buy_intensity(&self, t: f64, baseline: f64) -> f64 {
        let mut lambda = baseline;
        
        // Self-excitation from buy events
        for &ti in &self.buy_events {
            if ti < t {
                lambda += self.phi_1.evaluate(t - ti);
            }
        }
        
        // Cross-excitation from sell events
        for &ti in &self.sell_events {
            if ti < t {
                lambda += self.phi_2.evaluate(t - ti);
            }
        }
        
        lambda
    }

    /// Get sell intensity at time t given external baseline μ⁻(t)
    pub fn sell_intensity(&self, t: f64, baseline: f64) -> f64 {
        let mut lambda = baseline;
        
        // Cross-excitation from buy events
        for &ti in &self.buy_events {
            if ti < t {
                lambda += self.phi_2.evaluate(t - ti);
            }
        }
        
        // Self-excitation from sell events
        for &ti in &self.sell_events {
            if ti < t {
                lambda += self.phi_1.evaluate(t - ti);
            }
        }
        
        lambda
    }

    /// Simulate given core order flow as driver
    ///
    /// # Arguments
    /// * `core_buy_events` - Core buy order arrival times F⁺
    /// * `core_sell_events` - Core sell order arrival times F⁻
    /// * `t_max` - Maximum simulation time
    /// * `seed` - Optional random seed
    pub fn simulate_driven(
        &mut self,
        core_buy_events: &[f64],
        core_sell_events: &[f64],
        t_max: f64,
        seed: Option<u64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        self.buy_events.clear();
        self.sell_events.clear();
        self.current_time = 0.0;

        // Compute baseline intensity from core flow reaction
        let baseline_buy = |t: f64| -> f64 {
            let mut mu = 0.0;
            for &ti in core_buy_events {
                if ti < t { mu += self.phi_1.evaluate(t - ti); }
            }
            for &ti in core_sell_events {
                if ti < t { mu += self.phi_2.evaluate(t - ti); }
            }
            mu.max(0.001)  // Ensure positive
        };

        let baseline_sell = |t: f64| -> f64 {
            let mut mu = 0.0;
            for &ti in core_buy_events {
                if ti < t { mu += self.phi_2.evaluate(t - ti); }
            }
            for &ti in core_sell_events {
                if ti < t { mu += self.phi_1.evaluate(t - ti); }
            }
            mu.max(0.001)
        };

        // Use thinning algorithm for bivariate process
        let safety_factor = 2.0;

        while self.current_time < t_max {
            let lambda_buy = self.buy_intensity(self.current_time, baseline_buy(self.current_time));
            let lambda_sell = self.sell_intensity(self.current_time, baseline_sell(self.current_time));
            let lambda_max = (lambda_buy + lambda_sell) * safety_factor + 0.1;

            let exp_dist = Exp::new(lambda_max).unwrap();
            let tau: f64 = rng.sample(exp_dist);
            let candidate_time = self.current_time + tau;

            if candidate_time >= t_max {
                break;
            }

            // Accept/reject and determine type
            let u: f64 = rng.gen();
            let lambda_buy_cand = self.buy_intensity(candidate_time, baseline_buy(candidate_time));
            let lambda_sell_cand = self.sell_intensity(candidate_time, baseline_sell(candidate_time));
            let total_intensity = lambda_buy_cand + lambda_sell_cand;

            if u <= total_intensity / lambda_max {
                // Accept - determine buy or sell
                let p_buy = lambda_buy_cand / total_intensity;
                if rng.gen::<f64>() < p_buy {
                    self.buy_events.push(candidate_time);
                } else {
                    self.sell_events.push(candidate_time);
                }
            }

            self.current_time = candidate_time;
        }

        (self.buy_events.clone(), self.sell_events.clone())
    }

    /// Spectral radius of kernel matrix (stability condition)
    pub fn spectral_radius(&self) -> f64 {
        // For symmetric 2x2 [[φ₁, φ₂], [φ₂, φ₁]], eigenvalues are φ₁+φ₂ and φ₁-φ₂
        // Spectral radius = max(|φ₁+φ₂|, |φ₁-φ₂|) in L¹ norm
        let l1_phi1 = self.phi_1.l1_norm();
        let l1_phi2 = self.phi_2.l1_norm();
        (l1_phi1 + l1_phi2).max((l1_phi1 - l1_phi2).abs())
    }

    /// Check if the process is stable (spectral radius < 1)
    pub fn is_stable(&self) -> bool {
        self.spectral_radius() < 1.0
    }

    /// Get signed reaction flow N⁺ - N⁻
    pub fn signed_flow(&self) -> Vec<(f64, i32)> {
        let mut events: Vec<(f64, i32)> = Vec::new();
        events.extend(self.buy_events.iter().map(|&t| (t, 1)));
        events.extend(self.sell_events.iter().map(|&t| (t, -1)));
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        events
    }

    /// Get unsigned reaction volume N⁺ + N⁻
    pub fn unsigned_volume(&self) -> usize {
        self.buy_events.len() + self.sell_events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point_processes::kernels::{ExponentialKernel, PowerLawKernel};

    #[test]
    fn test_hawkes_simulation() {
        let kernel = ExponentialKernel::new(0.3, 1.0);
        let mut process = HawkesProcess::new(1.0, kernel);
        
        let events = process.simulate(100.0, Some(42));
        assert!(!events.is_empty());
        
        // Check events are sorted
        for i in 1..events.len() {
            assert!(events[i] > events[i-1]);
        }
    }

    #[test]
    fn test_hawkes_intensity() {
        let kernel = ExponentialKernel::new(0.5, 1.0);
        let mut process = HawkesProcess::new(1.0, kernel);
        
        // Initial intensity should be baseline
        assert!((process.intensity(0.0) - 1.0).abs() < 1e-10);
        
        // Add an event and check intensity increases
        process.events.push(1.0);
        let intensity_after = process.intensity(1.001);
        assert!(intensity_after > 1.0);
    }

    #[test]
    fn test_power_law_hawkes() {
        let kernel = PowerLawKernel::nearly_critical(0.375, 0.05);  // H₀ ≈ 0.75
        let mut process = HawkesProcess::new(0.1, kernel);
        
        let events = process.simulate(100.0, Some(123));
        println!("Power-law Hawkes: {} events in [0, 100]", events.len());
        assert!(!events.is_empty());
    }

    #[test]
    fn test_bivariate_stability() {
        let phi_1 = ExponentialKernel::new(0.3, 1.0);  // L¹ = 0.3
        let phi_2 = ExponentialKernel::new(0.2, 1.0);  // L¹ = 0.2
        
        let process = BivariateHawkes::new(phi_1, phi_2);
        assert!(process.is_stable());  // 0.3 + 0.2 = 0.5 < 1
        
        // Unstable case
        let phi_1_unstable = ExponentialKernel::new(0.8, 1.0);
        let phi_2_unstable = ExponentialKernel::new(0.5, 1.0);
        let process_unstable = BivariateHawkes::new(phi_1_unstable, phi_2_unstable);
        assert!(!process_unstable.is_stable());  // 0.8 + 0.5 = 1.3 > 1
    }
}
