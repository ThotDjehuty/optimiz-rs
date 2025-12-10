//! Core HMM model implementation with Baum-Welch training
//!
//! Implements the Hidden Markov Model with Expectation-Maximization
//! training algorithm (Baum-Welch).

use super::config::HMMConfig;
use super::emission::EmissionModel;
use crate::core::{OptimizrError, Result};
use std::f64;

/// Hidden Markov Model with generic emission model
pub struct HMM<E: EmissionModel> {
    pub config: HMMConfig<E>,
    pub transition_matrix: Vec<Vec<f64>>,
    pub initial_probs: Vec<f64>,
}

impl<E: EmissionModel> HMM<E> {
    /// Create a new HMM with uniform initialization
    pub fn new(config: HMMConfig<E>) -> Self {
        let n_states = config.n_states;
        let uniform = 1.0 / n_states as f64;

        Self {
            config,
            transition_matrix: vec![vec![uniform; n_states]; n_states],
            initial_probs: vec![uniform; n_states],
        }
    }

    /// Fit HMM using Baum-Welch (EM) algorithm
    pub fn fit(&mut self, observations: &[f64]) -> Result<()> {
        if observations.is_empty() {
            return Err(OptimizrError::EmptyData);
        }

        // Initialize emission parameters
        for s in 0..self.config.n_states {
            self.config
                .emission_model
                .initialize(observations, self.config.n_states, s)?;
        }

        // EM iterations
        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..self.config.n_iterations {
            // E-step: Compute posteriors
            let alpha = self.forward(observations)?;
            let beta = self.backward(observations)?;
            let gamma = Self::compute_gamma(&alpha, &beta);
            let xi = self.compute_xi(observations, &alpha, &beta)?;

            // M-step: Update parameters
            self.update_parameters(observations, &gamma, &xi)?;

            // Check convergence
            let log_likelihood = Self::compute_log_likelihood(&alpha);

            if (log_likelihood - prev_ll).abs() < self.config.tolerance {
                break; // Converged
            }

            prev_ll = log_likelihood;
        }

        Ok(())
    }

    /// Forward algorithm (alpha pass)
    fn forward(&self, observations: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;
        let mut alpha = vec![vec![0.0; n_states]; n_obs];

        // Initialize
        for s in 0..n_states {
            alpha[0][s] =
                self.initial_probs[s] * self.config.emission_model.probability(observations[0], s);
        }
        Self::normalize_row(&mut alpha[0]);

        // Recursion
        for t in 1..n_obs {
            for s in 0..n_states {
                let sum: f64 = (0..n_states)
                    .map(|prev_s| alpha[t - 1][prev_s] * self.transition_matrix[prev_s][s])
                    .sum();
                alpha[t][s] = sum * self.config.emission_model.probability(observations[t], s);
            }
            Self::normalize_row(&mut alpha[t]);
        }

        Ok(alpha)
    }

    /// Backward algorithm (beta pass)
    fn backward(&self, observations: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;
        let mut beta = vec![vec![0.0; n_states]; n_obs];

        // Initialize
        beta[n_obs - 1].fill(1.0);

        // Recursion
        for t in (0..n_obs - 1).rev() {
            for s in 0..n_states {
                let sum: f64 = (0..n_states)
                    .map(|next_s| {
                        self.transition_matrix[s][next_s]
                            * self
                                .config
                                .emission_model
                                .probability(observations[t + 1], next_s)
                            * beta[t + 1][next_s]
                    })
                    .sum();
                beta[t][s] = sum;
            }
            Self::normalize_row(&mut beta[t]);
        }

        Ok(beta)
    }

    /// Compute state occupation probabilities (gamma)
    fn compute_gamma(alpha: &[Vec<f64>], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
        alpha
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| {
                let sum: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
                a.iter()
                    .zip(b.iter())
                    .map(|(ai, bi)| {
                        if sum > 1e-10 {
                            ai * bi / sum
                        } else {
                            1.0 / a.len() as f64
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute transition probabilities (xi)
    fn compute_xi(
        &self,
        observations: &[f64],
        alpha: &[Vec<f64>],
        beta: &[Vec<f64>],
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;

        let xi: Vec<Vec<Vec<f64>>> = (0..n_obs - 1)
            .map(|t| {
                let mut xi_t = vec![vec![0.0; n_states]; n_states];
                let mut sum = 0.0;

                for i in 0..n_states {
                    for j in 0..n_states {
                        xi_t[i][j] = alpha[t][i]
                            * self.transition_matrix[i][j]
                            * self
                                .config
                                .emission_model
                                .probability(observations[t + 1], j)
                            * beta[t + 1][j];
                        sum += xi_t[i][j];
                    }
                }

                // Normalize
                if sum > 1e-10 {
                    for row in &mut xi_t {
                        for val in row {
                            *val /= sum;
                        }
                    }
                }

                xi_t
            })
            .collect();

        Ok(xi)
    }

    /// M-step: Update parameters
    fn update_parameters(
        &mut self,
        observations: &[f64],
        gamma: &[Vec<f64>],
        xi: &[Vec<Vec<f64>>],
    ) -> Result<()> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;

        // Update transition matrix
        for i in 0..n_states {
            let denom: f64 = gamma[..n_obs - 1].iter().map(|g| g[i]).sum();

            for j in 0..n_states {
                let numer: f64 = xi.iter().map(|x| x[i][j]).sum();
                self.transition_matrix[i][j] = if denom > 1e-10 {
                    numer / denom
                } else {
                    1.0 / n_states as f64
                };
            }
        }

        // Update emission parameters
        for s in 0..n_states {
            let weights: Vec<f64> = gamma.iter().map(|g| g[s]).collect();
            self.config
                .emission_model
                .update(observations, &weights, s)?;
        }

        Ok(())
    }

    /// Helper: Normalize a probability row
    fn normalize_row(row: &mut [f64]) {
        let sum: f64 = row.iter().sum();
        if sum > 1e-10 {
            row.iter_mut().for_each(|v| *v /= sum);
        } else {
            let uniform = 1.0 / row.len() as f64;
            row.fill(uniform);
        }
    }

    /// Helper: Compute log-likelihood from forward probabilities
    fn compute_log_likelihood(alpha: &[Vec<f64>]) -> f64 {
        alpha.last().unwrap().iter().sum::<f64>().max(1e-10).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm::emission::GaussianEmission;

    #[test]
    fn test_hmm_creation() {
        let config = HMMConfig::<GaussianEmission>::builder(3).build().unwrap();
        let hmm = HMM::new(config);

        assert_eq!(hmm.transition_matrix.len(), 3);
        assert_eq!(hmm.initial_probs.len(), 3);
    }

    #[test]
    fn test_hmm_fit() {
        let observations: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = HMMConfig::<GaussianEmission>::builder(2).build().unwrap();

        let mut hmm = HMM::new(config);
        assert!(hmm.fit(&observations).is_ok());
    }
}
