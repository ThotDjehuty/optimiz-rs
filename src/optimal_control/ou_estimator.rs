//! Ornstein-Uhlenbeck Parameter Estimation
//! ========================================
//!
//! Estimate parameters of OU process: dX_t = κ(θ - X_t)dt + σdW_t

use ndarray::Array1;
use statrs::distribution::{Normal, ContinuousCDF};
use crate::optimal_control::{OptimalControlError, Result};

/// OU process parameters
#[derive(Debug, Clone, Copy)]
pub struct OUParams {
    /// Mean-reversion speed
    pub kappa: f64,
    /// Long-term mean
    pub theta: f64,
    /// Volatility
    pub sigma: f64,
    /// Half-life in time units
    pub half_life: f64,
}

/// Estimate OU parameters via discrete-time approximation
///
/// Uses OLS regression: X_{t+1} = μ + φ * X_t + ε
/// Then converts to continuous-time parameters
pub fn estimate_ou_params(spread: &[f64], dt: f64) -> Result<OUParams> {
    if spread.len() < 10 {
        return Err(OptimalControlError::InsufficientData(10));
    }
    
    let n = spread.len() - 1;
    let x = &spread[..n];
    let y = &spread[1..];
    
    // OLS: y = μ + φ*x + ε
    let x_mean: f64 = x.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;
    
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov_xy += dx * dy;
        var_x += dx * dx;
    }
    
    if var_x < 1e-10 {
        return Err(OptimalControlError::NumericalError(
            "Insufficient variance in data".to_string()
        ));
    }
    
    let phi = cov_xy / var_x;
    let mu = y_mean - phi * x_mean;
    
    // Residuals
    let mut residuals = Vec::with_capacity(n);
    for i in 0..n {
        residuals.push(y[i] - (mu + phi * x[i]));
    }
    
    let sigma_epsilon = {
        let mean_residual = residuals.iter().sum::<f64>() / n as f64;
        let variance = residuals.iter()
            .map(|r| (r - mean_residual).powi(2))
            .sum::<f64>() / n as f64;
        variance.sqrt()
    };
    
    // Convert to continuous-time parameters
    if phi <= 0.0 || phi >= 1.0 {
        return Err(OptimalControlError::InvalidParameters(
            format!("phi out of valid range: {}", phi)
        ));
    }
    
    let kappa = -phi.ln() / dt;
    let theta = mu / (1.0 - phi);
    
    let variance_term = -2.0 * phi.ln() / dt / (1.0 - phi.powi(2));
    if variance_term <= 0.0 {
        return Err(OptimalControlError::NumericalError(
            "Invalid variance calculation".to_string()
        ));
    }
    
    let sigma = sigma_epsilon * variance_term.sqrt();
    let half_life = 2.0f64.ln() / kappa;
    
    Ok(OUParams {
        kappa,
        theta,
        sigma,
        half_life,
    })
}

/// Maximum Likelihood Estimation for OU parameters
///
/// More accurate but computationally expensive
pub fn estimate_ou_params_mle(spread: &[f64], dt: f64) -> Result<OUParams> {
    if spread.len() < 20 {
        return Err(OptimalControlError::InsufficientData(20));
    }
    
    // Initial guess from OLS
    let ols_params = estimate_ou_params(spread, dt)?;
    
    // Newton-Raphson optimization (simplified)
    let mut kappa = ols_params.kappa;
    let mut theta = ols_params.theta;
    let mut sigma = ols_params.sigma;
    
    let n = spread.len() - 1;
    let max_iter = 100;
    let tol = 1e-6;
    
    for _iter in 0..max_iter {
        let mut log_likelihood = 0.0;
        let mut d_kappa = 0.0;
        let mut d_theta = 0.0;
        let mut d_sigma = 0.0;
        
        for i in 0..n {
            let x_t = spread[i];
            let x_next = spread[i + 1];
            
            // Expected value
            let exp_neg_kappa_dt = (-kappa * dt).exp();
            let mu_t = theta + (x_t - theta) * exp_neg_kappa_dt;
            
            // Variance
            let var_t = sigma.powi(2) / (2.0 * kappa) * (1.0 - (-2.0 * kappa * dt).exp());
            
            if var_t <= 0.0 {
                continue;
            }
            
            let std_t = var_t.sqrt();
            let z = (x_next - mu_t) / std_t;
            
            // Log-likelihood contribution
            log_likelihood -= 0.5 * z.powi(2) + std_t.ln();
            
            // Gradients (simplified)
            d_kappa += z * (x_t - theta) * dt * exp_neg_kappa_dt / std_t;
            d_theta += z * (1.0 - exp_neg_kappa_dt) / std_t;
            d_sigma += z * sigma * (1.0 - (-2.0 * kappa * dt).exp()) / (2.0 * kappa * var_t);
        }
        
        // Update parameters (gradient ascent with small step)
        let step_size = 0.01;
        let kappa_new = kappa + step_size * d_kappa;
        let theta_new = theta + step_size * d_theta;
        let sigma_new = sigma + step_size * d_sigma;
        
        // Check convergence
        if (kappa_new - kappa).abs() < tol
            && (theta_new - theta).abs() < tol
            && (sigma_new - sigma).abs() < tol
        {
            kappa = kappa_new;
            theta = theta_new;
            sigma = sigma_new;
            break;
        }
        
        // Ensure parameters stay in valid range
        kappa = kappa_new.max(0.01).min(10.0);
        theta = theta_new;
        sigma = sigma_new.max(0.001).min(10.0);
    }
    
    let half_life = 2.0f64.ln() / kappa;
    
    Ok(OUParams {
        kappa,
        theta,
        sigma,
        half_life,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    use rand_distr::Normal;
    
    #[test]
    fn test_ou_estimation_simulated_data() {
        // Simulate OU process
        let true_kappa = 0.5;
        let true_theta = 0.0;
        let true_sigma = 0.2;
        let dt: f64 = 1.0 / 252.0;
        let n = 500;
        
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let mut spread = vec![0.0; n];
        spread[0] = true_theta;
        
        for i in 1..n {
            let dw = rng.sample(normal) * dt.sqrt();
            spread[i] = spread[i - 1] + true_kappa * (true_theta - spread[i - 1]) * dt
                + true_sigma * dw;
        }
        
        // Estimate parameters
        let params = estimate_ou_params(&spread, dt).unwrap();
        
        // Check accuracy (within 30% for stochastic simulation)
        assert!((params.kappa - true_kappa).abs() / true_kappa < 0.3);
        assert!((params.theta - true_theta).abs() < 0.1);
        assert!((params.sigma - true_sigma).abs() / true_sigma < 0.3);
    }
}
