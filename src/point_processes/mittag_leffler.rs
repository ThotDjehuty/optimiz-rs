//! Mittag-Leffler Functions
//!
//! The Mittag-Leffler function E_{α,β}(z) is a generalization of the
//! exponential function that plays a crucial role in fractional calculus
//! and the scaling limits of Hawkes processes.
//!
//! # Definition
//!
//! E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)
//!
//! Special cases:
//! - E_{1,1}(z) = exp(z)
//! - E_{2,1}(z²) = cosh(z)

use std::f64::consts::PI;

/// Compute the Mittag-Leffler function E_{α,β}(z)
///
/// Uses series expansion for |z| < R and asymptotic expansion for |z| > R.
///
/// # Arguments
/// * `alpha` - Parameter α > 0
/// * `beta` - Parameter β > 0
/// * `z` - Complex argument (real part only)
///
/// # Returns
/// The value E_{α,β}(z)
pub fn mittag_leffler(alpha: f64, beta: f64, z: f64) -> f64 {
    assert!(alpha > 0.0, "alpha must be positive");
    assert!(beta > 0.0, "beta must be positive");

    if z.abs() < 1e-15 {
        return 1.0 / gamma(beta);
    }

    // Use different methods based on |z|
    if z.abs() < 10.0 {
        mittag_leffler_series(alpha, beta, z, 100)
    } else {
        mittag_leffler_asymptotic(alpha, beta, z)
    }
}

/// Series expansion for Mittag-Leffler function
fn mittag_leffler_series(alpha: f64, beta: f64, z: f64, max_terms: usize) -> f64 {
    let mut sum: f64 = 0.0;
    let mut z_pow: f64 = 1.0;  // z^k

    for k in 0..max_terms {
        let term = z_pow / gamma(alpha * k as f64 + beta);
        
        // Check for convergence
        if term.abs() < 1e-15 * sum.abs() && k > 10 {
            break;
        }
        
        sum += term;
        z_pow *= z;

        // Prevent overflow
        if z_pow.abs() > 1e100 {
            break;
        }
    }

    sum
}

/// Asymptotic expansion for large |z|
fn mittag_leffler_asymptotic(alpha: f64, beta: f64, z: f64) -> f64 {
    if alpha < 1.0 {
        // For 0 < α < 1, different behavior for positive/negative z
        if z > 0.0 {
            // Leading term: (1/α) * z^{(1-β)/α} * exp(z^{1/α})
            let z_pow = z.powf(1.0 / alpha);
            (1.0 / alpha) * z.powf((1.0 - beta) / alpha) * z_pow.exp()
        } else {
            // For z < 0, algebraic decay
            // E_{α,β}(-x) ~ -Σ_{k=1}^∞ (-x)^{-k} / Γ(β - αk)
            let mut sum = 0.0;
            let z_abs = z.abs();
            for k in 1..=10 {
                let term = (-1.0_f64).powi(k as i32 + 1)
                    * z_abs.powi(-(k as i32))
                    / gamma(beta - alpha * k as f64);
                sum += term;
            }
            sum
        }
    } else if alpha == 1.0 {
        // E_{1,β}(z) ~ z^{1-β} * exp(z) for large z > 0
        if z > 0.0 {
            z.powf(1.0 - beta) * z.exp()
        } else {
            // E_{1,β}(-x) ~ 0 for large x
            0.0
        }
    } else {
        // For α > 1, more complex behavior
        // Use series with damping
        mittag_leffler_series(alpha, beta, z, 200)
    }
}

/// Derivative of Mittag-Leffler function: d/dz E_{α,β}(z)
pub fn mittag_leffler_derivative(alpha: f64, beta: f64, z: f64) -> f64 {
    // d/dz E_{α,β}(z) = (1/α) * (E_{α,β-1}(z) - (β-1) * E_{α,β}(z))
    // More stable: use series directly
    
    if z.abs() < 1e-15 {
        return 1.0 / gamma(alpha + beta);
    }

    let mut sum = 0.0;
    let mut z_pow = 1.0;

    for k in 1..100 {
        let term = k as f64 * z_pow / gamma(alpha * k as f64 + beta);
        sum += term;
        z_pow *= z;

        if z_pow.abs() > 1e100 || (term.abs() < 1e-15 * sum.abs() && k > 10) {
            break;
        }
    }

    sum / z  // Factor out one z from derivative
}

/// The function f_{α₀,λ₀}(x) from Theorem 3.1 of the paper
///
/// f_{α₀,λ₀}(x) = λ₀ * x^{α₀-1} * E_{α₀,α₀}(-λ₀ * x^{α₀})
///
/// This function controls the scaling limit of the Hawkes process.
pub fn f_alpha_lambda(alpha_0: f64, lambda_0: f64, x: f64) -> f64 {
    assert!(alpha_0 > 0.0 && alpha_0 < 1.0);
    assert!(lambda_0 > 0.0);
    
    if x <= 0.0 {
        return 0.0;
    }
    
    let x_alpha = x.powf(alpha_0);
    let ml_arg = -lambda_0 * x_alpha;
    
    lambda_0 * x.powf(alpha_0 - 1.0) * mittag_leffler(alpha_0, alpha_0, ml_arg)
}

/// Integral of f_{α₀,λ₀} from 0 to t
///
/// ∫₀ᵗ f_{α₀,λ₀}(s) ds = t^{α₀} * E_{α₀,α₀+1}(-λ₀ * t^{α₀})
pub fn f_alpha_lambda_integral(alpha_0: f64, lambda_0: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    
    let t_alpha = t.powf(alpha_0);
    let ml_arg = -lambda_0 * t_alpha;
    
    t_alpha * mittag_leffler(alpha_0, alpha_0 + 1.0, ml_arg)
}

/// Gamma function using Lanczos approximation
pub fn gamma(z: f64) -> f64 {
    if z < 0.5 {
        // Reflection formula: Γ(z) * Γ(1-z) = π / sin(πz)
        PI / ((PI * z).sin() * gamma(1.0 - z))
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

/// Log-gamma function for numerical stability
pub fn lgamma(z: f64) -> f64 {
    gamma(z).abs().ln()
}

/// Incomplete gamma function γ(s, x) = ∫₀ˣ t^{s-1} e^{-t} dt
/// Used for various probability computations
pub fn incomplete_gamma_lower(s: f64, x: f64) -> f64 {
    if x < 0.0 || s <= 0.0 {
        return 0.0;
    }
    
    if x < s + 1.0 {
        // Series expansion
        let mut sum = 0.0;
        let mut term = 1.0 / s;
        sum += term;
        
        for n in 1..100 {
            term *= x / (s + n as f64);
            sum += term;
            if term.abs() < 1e-12 * sum.abs() {
                break;
            }
        }
        
        sum * x.powf(s) * (-x).exp()
    } else {
        // Continued fraction for large x
        gamma(s) - incomplete_gamma_upper(s, x)
    }
}

/// Upper incomplete gamma Γ(s, x) = ∫ₓ^∞ t^{s-1} e^{-t} dt
pub fn incomplete_gamma_upper(s: f64, x: f64) -> f64 {
    if x < 0.0 {
        return gamma(s);
    }
    
    // Continued fraction representation (Lentz's algorithm)
    let mut f = 1.0 + x - s;
    if f.abs() < 1e-30 {
        f = 1e-30;
    }
    
    let mut c = f;
    let mut d = 0.0;
    
    for i in 1..100 {
        let an = (i as f64) * (s - i as f64);
        let bn = 2.0 * i as f64 + 1.0 + x - s;
        
        d = bn + an * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        
        c = bn + an / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }
    
    x.powf(s) * (-x).exp() / f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma() {
        // Γ(1) = 1
        assert!((gamma(1.0) - 1.0).abs() < 1e-10);
        
        // Γ(2) = 1! = 1
        assert!((gamma(2.0) - 1.0).abs() < 1e-10);
        
        // Γ(3) = 2! = 2
        assert!((gamma(3.0) - 2.0).abs() < 1e-10);
        
        // Γ(1/2) = √π
        assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_mittag_leffler_special_cases() {
        // E_{1,1}(z) = exp(z)
        let z = 1.5;
        let ml = mittag_leffler(1.0, 1.0, z);
        assert!((ml - z.exp()).abs() < 1e-8);
        
        // E_{α,β}(0) = 1/Γ(β)
        let ml_zero = mittag_leffler(0.5, 1.0, 0.0);
        assert!((ml_zero - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f_alpha_lambda() {
        // Test that f is positive for positive arguments
        let val = f_alpha_lambda(0.375, 1.0, 1.0);
        assert!(val > 0.0);
        
        // Test integral property
        let integral = f_alpha_lambda_integral(0.5, 1.0, 2.0);
        assert!(integral > 0.0);
    }
}
