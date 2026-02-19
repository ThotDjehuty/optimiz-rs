//! Excitation Kernels for Hawkes Processes
//!
//! This module provides different kernel functions that describe how past events
//! influence the intensity of future arrivals in a Hawkes process.
//!
//! # Kernels
//!
//! - **ExponentialKernel**: φ(t) = α * exp(-β * t), for short-range dependence
//! - **PowerLawKernel**: φ(t) = K₀ * (1 + t)^(-1-α₀), for long-range dependence

use std::f64::consts::PI;

/// Trait for excitation kernels in Hawkes processes
pub trait ExcitationKernel: Clone + Send + Sync {
    /// Evaluate the kernel at time t (t >= 0)
    fn evaluate(&self, t: f64) -> f64;

    /// Integrate the kernel from 0 to t
    fn integrate(&self, t: f64) -> f64;

    /// L¹ norm of the kernel (total mass)
    fn l1_norm(&self) -> f64;

    /// Check if the process is stable (L¹ norm < 1 for stability)
    fn is_stable(&self) -> bool {
        self.l1_norm() < 1.0
    }

    /// Tail exponent α₀ (for asymptotic analysis)
    fn tail_exponent(&self) -> Option<f64>;
}

/// Exponential kernel: φ(t) = α * exp(-β * t)
///
/// Suitable for processes with short-range temporal dependence.
/// The L¹ norm is α/β.
#[derive(Clone, Debug)]
pub struct ExponentialKernel {
    /// Peak intensity α > 0
    pub alpha: f64,
    /// Decay rate β > 0
    pub beta: f64,
}

impl ExponentialKernel {
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(beta > 0.0, "beta must be positive");
        Self { alpha, beta }
    }

    /// Create a stable kernel with given L¹ norm (< 1)
    pub fn with_branching_ratio(branching_ratio: f64, beta: f64) -> Self {
        assert!(branching_ratio > 0.0 && branching_ratio < 1.0);
        assert!(beta > 0.0);
        Self {
            alpha: branching_ratio * beta,
            beta,
        }
    }
}

impl ExcitationKernel for ExponentialKernel {
    fn evaluate(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        self.alpha * (-self.beta * t).exp()
    }

    fn integrate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        (self.alpha / self.beta) * (1.0 - (-self.beta * t).exp())
    }

    fn l1_norm(&self) -> f64 {
        self.alpha / self.beta
    }

    fn tail_exponent(&self) -> Option<f64> {
        // Exponential kernel decays faster than any power law
        None
    }
}

/// Power-law kernel: φ(t) = K₀ * (1 + t)^(-1-α₀)
///
/// Suitable for processes with long-range temporal dependence (memory).
/// Used in the unified theory paper for core order flow modeling.
///
/// Parameters:
/// - α₀ ∈ (0, 1): tail exponent controlling memory persistence
/// - K₀ > 0: scaling constant
///
/// Smaller α₀ means stronger persistence (longer memory).
#[derive(Clone, Debug)]
pub struct PowerLawKernel {
    /// Tail exponent α₀ ∈ (0, 1)
    pub alpha_0: f64,
    /// Scaling constant K₀ > 0
    pub k_0: f64,
    /// Normalization factor to achieve unit L¹ norm
    norm_factor: f64,
}

impl PowerLawKernel {
    /// Create a power-law kernel with given parameters
    ///
    /// # Arguments
    /// * `alpha_0` - Tail exponent in (0, 1)
    /// * `k_0` - Scaling constant > 0
    pub fn new(alpha_0: f64, k_0: f64) -> Self {
        assert!(alpha_0 > 0.0 && alpha_0 < 1.0, "alpha_0 must be in (0, 1)");
        assert!(k_0 > 0.0, "k_0 must be positive");
        
        // L¹ norm = K₀ / α₀ (integral of (1+t)^{-1-α₀} from 0 to ∞)
        let norm_factor = alpha_0 / k_0;
        
        Self { alpha_0, k_0, norm_factor }
    }

    /// Create a kernel with unit L¹ norm (critical regime)
    pub fn unit_norm(alpha_0: f64) -> Self {
        assert!(alpha_0 > 0.0 && alpha_0 < 1.0);
        Self {
            alpha_0,
            k_0: alpha_0,  // This gives L¹ norm = 1
            norm_factor: 1.0,
        }
    }

    /// Create a nearly-critical kernel (L¹ norm = 1 - ε)
    pub fn nearly_critical(alpha_0: f64, epsilon: f64) -> Self {
        assert!(alpha_0 > 0.0 && alpha_0 < 1.0);
        assert!(epsilon > 0.0 && epsilon < 1.0);
        let k_0 = alpha_0 * (1.0 - epsilon);
        Self::new(alpha_0, k_0)
    }

    /// Get the corresponding Hurst exponent H₀ = 2 * α₀
    pub fn hurst_exponent(&self) -> f64 {
        2.0 * self.alpha_0
    }
}

impl ExcitationKernel for PowerLawKernel {
    fn evaluate(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        self.k_0 * (1.0 + t).powf(-1.0 - self.alpha_0)
    }

    fn integrate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        // ∫₀ᵗ K₀(1+s)^{-1-α₀} ds = (K₀/α₀) * [1 - (1+t)^{-α₀}]
        (self.k_0 / self.alpha_0) * (1.0 - (1.0 + t).powf(-self.alpha_0))
    }

    fn l1_norm(&self) -> f64 {
        self.k_0 / self.alpha_0
    }

    fn tail_exponent(&self) -> Option<f64> {
        Some(self.alpha_0)
    }
}

/// Completely monotone power-law kernel (as in Assumption A of the paper)
///
/// This satisfies the complete monotonicity requirement for the scaling limit
/// theorems. φ(t) = K₀ * t^{-α₀} * E_{1-α₀}(-λ * t^{1-α₀})
/// where E is the Mittag-Leffler function.
#[derive(Clone, Debug)]
pub struct CompletelyMonotoneKernel {
    pub alpha_0: f64,
    pub k_0: f64,
    pub lambda: f64,
}

impl CompletelyMonotoneKernel {
    pub fn new(alpha_0: f64, k_0: f64, lambda: f64) -> Self {
        assert!(alpha_0 > 0.0 && alpha_0 < 1.0);
        assert!(k_0 > 0.0);
        assert!(lambda > 0.0);
        Self { alpha_0, k_0, lambda }
    }
}

impl ExcitationKernel for CompletelyMonotoneKernel {
    fn evaluate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        // Use the Mittag-Leffler approximation for now
        // Full implementation requires mittag_leffler function
        let ml_arg = -self.lambda * t.powf(1.0 - self.alpha_0);
        
        // Approximate E_{1-α₀}(z) ≈ exp(z^{1/(1-α₀)}) / (1-α₀) for large |z|
        // For small arguments, E_α(z) ≈ 1 + z/Γ(1+α)
        let ml_approx = if ml_arg.abs() < 0.1 {
            1.0 + ml_arg / gamma_fn(2.0 - self.alpha_0)
        } else {
            // Asymptotic expansion
            (-ml_arg).powf(-1.0) / gamma_fn(self.alpha_0)
        };
        
        self.k_0 * t.powf(-self.alpha_0) * ml_approx
    }

    fn integrate(&self, t: f64) -> f64 {
        // Numerical integration fallback
        let n = 1000;
        let dt = t / n as f64;
        let mut sum = 0.0;
        for i in 0..n {
            let ti = (i as f64 + 0.5) * dt;
            sum += self.evaluate(ti);
        }
        sum * dt
    }

    fn l1_norm(&self) -> f64 {
        // For completely monotone kernels, need numerical approximation
        // or analytical form based on Mittag-Leffler integral
        1.0  // Placeholder for unit norm case
    }

    fn tail_exponent(&self) -> Option<f64> {
        Some(self.alpha_0)
    }
}

/// Gamma function approximation (Lanczos approximation)
fn gamma_fn(z: f64) -> f64 {
    // Use Lanczos approximation for Γ(z)
    if z < 0.5 {
        PI / ((PI * z).sin() * gamma_fn(1.0 - z))
    } else {
        let z = z - 1.0;
        let g = 7;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        let mut x = c[0];
        for i in 1..=(g + 1) {
            x += c[i] / (z + i as f64);
        }

        let t = z + g as f64 + 0.5;
        (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_kernel() {
        let kernel = ExponentialKernel::new(0.5, 1.0);
        assert!((kernel.l1_norm() - 0.5).abs() < 1e-10);
        assert!(kernel.is_stable());
        
        // Check decay
        let v0 = kernel.evaluate(0.0);
        let v1 = kernel.evaluate(1.0);
        assert!(v1 < v0);
        assert!((v0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_power_law_kernel() {
        let kernel = PowerLawKernel::new(0.375, 0.375);  // H₀ = 0.75
        assert!((kernel.l1_norm() - 1.0).abs() < 1e-10);
        assert!((kernel.hurst_exponent() - 0.75).abs() < 1e-10);
        
        // Check power-law decay
        let v1 = kernel.evaluate(1.0);
        let v10 = kernel.evaluate(10.0);
        let v100 = kernel.evaluate(100.0);
        
        // φ(t) ~ t^{-1-α₀}, so φ(10)/φ(1) ≈ 10^{-1-α₀}
        let expected_ratio = 10.0_f64.powf(-1.375);
        assert!((v10 / v1 - expected_ratio).abs() < 0.1);
    }

    #[test]
    fn test_nearly_critical() {
        let kernel = PowerLawKernel::nearly_critical(0.5, 0.01);
        assert!((kernel.l1_norm() - 0.99).abs() < 1e-10);
        assert!(kernel.is_stable());
    }
}
