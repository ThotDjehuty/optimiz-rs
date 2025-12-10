//! Emission probability models for Hidden Markov Models
//!
//! This module defines the EmissionModel trait and provides
//! implementations for different probability distributions.

use crate::core::Result;
use std::f64;

/// Trait for emission probability models
pub trait EmissionModel: Send + Sync + Clone {
    /// Compute emission probability for observation given state
    fn probability(&self, observation: f64, state: usize) -> f64;

    /// Update parameters from weighted observations
    fn update(&mut self, observations: &[f64], weights: &[f64], state: usize) -> Result<()>;

    /// Initialize parameters from observations
    fn initialize(&mut self, observations: &[f64], n_states: usize, state: usize) -> Result<()>;

    /// Get number of states
    fn n_states(&self) -> usize;
}

/// Gaussian emission model for continuous observations
#[derive(Clone, Debug)]
pub struct GaussianEmission {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl GaussianEmission {
    /// Create a new Gaussian emission model with given number of states
    pub fn new(n_states: usize) -> Self {
        Self {
            means: vec![0.0; n_states],
            stds: vec![1.0; n_states],
        }
    }
}

impl EmissionModel for GaussianEmission {
    fn probability(&self, observation: f64, state: usize) -> f64 {
        let mean = self.means[state];
        let std = self.stds[state];
        let z = (observation - mean) / std;
        let coef = 1.0 / (std * (2.0 * f64::consts::PI).sqrt());
        (coef * (-0.5 * z * z).exp()).max(1e-10)
    }

    fn update(&mut self, observations: &[f64], weights: &[f64], state: usize) -> Result<()> {
        let sum_weights: f64 = weights.iter().sum();

        if sum_weights < 1e-10 {
            return Ok(());
        }

        // Weighted mean
        let mean = observations
            .iter()
            .zip(weights.iter())
            .map(|(obs, w)| obs * w)
            .sum::<f64>()
            / sum_weights;

        // Weighted variance
        let var = observations
            .iter()
            .zip(weights.iter())
            .map(|(obs, w)| w * (obs - mean).powi(2))
            .sum::<f64>()
            / sum_weights;

        self.means[state] = mean;
        self.stds[state] = var.sqrt().max(1e-6);

        Ok(())
    }

    fn initialize(&mut self, observations: &[f64], n_states: usize, state: usize) -> Result<()> {
        let mut sorted = observations.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = observations.len();
        let start_idx = (state * n) / n_states;
        let end_idx = ((state + 1) * n) / n_states;
        let segment = &sorted[start_idx..end_idx];

        if !segment.is_empty() {
            self.means[state] = segment.iter().sum::<f64>() / segment.len() as f64;
            let var: f64 = segment
                .iter()
                .map(|x| (x - self.means[state]).powi(2))
                .sum::<f64>()
                / segment.len() as f64;
            self.stds[state] = var.sqrt().max(1e-6);
        }

        Ok(())
    }

    fn n_states(&self) -> usize {
        self.means.len()
    }
}

impl Default for GaussianEmission {
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_emission() {
        let emission = GaussianEmission {
            means: vec![0.0, 1.0],
            stds: vec![1.0, 1.0],
        };

        assert!(emission.probability(0.0, 0) > emission.probability(0.0, 1));
        assert!(emission.probability(1.0, 1) > emission.probability(1.0, 0));
    }

    #[test]
    fn test_emission_update() {
        let mut emission = GaussianEmission::new(2);
        let observations = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0, 1.0];

        emission.update(&observations, &weights, 0).unwrap();
        assert!((emission.means[0] - 2.0).abs() < 0.01);
    }
}
