//! Metropolis-Hastings sampler implementation
//!
//! Core MCMC sampling algorithm with diagnostics.

use super::config::MCMCConfig;
use super::likelihood::LogLikelihood;
use super::proposal::ProposalStrategy;
use crate::core::{OptimizrError, Result, Sampler, SamplerDiagnostics};
use rand::Rng;

/// Metropolis-Hastings MCMC Sampler
pub struct MetropolisHastings<P: ProposalStrategy, L: LogLikelihood> {
    pub config: MCMCConfig<P>,
    pub log_likelihood: L,
}

impl<P: ProposalStrategy, L: LogLikelihood> MetropolisHastings<P, L> {
    pub fn new(config: MCMCConfig<P>, log_likelihood: L) -> Self {
        Self {
            config,
            log_likelihood,
        }
    }

    /// Run MCMC chain
    pub fn sample_chain(&mut self) -> Result<Vec<Vec<f64>>> {
        let mut rng = rand::thread_rng();
        let mut current_state = self.config.initial_state.clone();
        let mut current_ll = self.log_likelihood.evaluate(&current_state);

        let total_steps = self.config.n_samples + self.config.burn_in;
        let mut samples = Vec::with_capacity(self.config.n_samples / self.config.thin);
        let mut acceptance_count = 0usize;

        for step in 0..total_steps {
            // Propose new state
            let proposed_state = self.config.proposal.propose(&current_state, &mut rng);
            let proposed_ll = self.log_likelihood.evaluate(&proposed_state);

            // Metropolis-Hastings acceptance
            let log_alpha = proposed_ll - current_ll;
            let accepted = log_alpha >= 0.0 || rng.gen::<f64>() < log_alpha.exp();

            if accepted {
                current_state = proposed_state;
                current_ll = proposed_ll;
                acceptance_count += 1;
            }

            // Adapt proposal if needed
            if step > 0 && step % self.config.adaptation_interval == 0 {
                let acceptance_rate =
                    acceptance_count as f64 / self.config.adaptation_interval as f64;
                self.config.proposal.adapt(acceptance_rate);
                acceptance_count = 0;
            }

            // Store sample after burn-in
            if step >= self.config.burn_in && (step - self.config.burn_in) % self.config.thin == 0 {
                samples.push(current_state.clone());
            }
        }

        Ok(samples)
    }

    /// Compute diagnostics for chain
    pub fn diagnostics(&self, samples: &[Vec<f64>]) -> Result<SamplerDiagnostics> {
        if samples.is_empty() {
            return Err(OptimizrError::EmptyData);
        }

        let n_samples = samples.len();
        let dim = samples[0].len();

        // Compute means and variances
        let means: Vec<f64> = (0..dim)
            .map(|d| samples.iter().map(|s| s[d]).sum::<f64>() / n_samples as f64)
            .collect();

        let variances: Vec<f64> = (0..dim)
            .map(|d| {
                let mean = means[d];
                samples.iter().map(|s| (s[d] - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64
            })
            .collect();

        // Compute autocorrelations (lag 1)
        let autocorrs: Vec<f64> = (0..dim)
            .map(|d| {
                if n_samples < 2 {
                    return 0.0;
                }

                let mean = means[d];
                let var = variances[d];

                if var < 1e-10 {
                    return 0.0;
                }

                let cov: f64 = (0..n_samples - 1)
                    .map(|i| (samples[i][d] - mean) * (samples[i + 1][d] - mean))
                    .sum::<f64>()
                    / (n_samples - 1) as f64;

                cov / var
            })
            .collect();

        Ok(SamplerDiagnostics {
            n_samples,
            means,
            std_devs: variances.iter().map(|v| v.sqrt()).collect(),
            autocorrelations: autocorrs,
        })
    }
}

impl<P: ProposalStrategy + 'static, L: LogLikelihood + 'static> Sampler
    for MetropolisHastings<P, L>
{
    type Config = MCMCConfig<P>;
    type Output = Vec<Vec<f64>>;

    fn sample(&mut self) -> Result<Self::Output> {
        self.sample_chain()
    }

    fn diagnostics(&self, samples: &Self::Output) -> Result<SamplerDiagnostics> {
        self.diagnostics(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcmc::proposal::GaussianProposal;

    struct TestLogLikelihood;

    impl LogLikelihood for TestLogLikelihood {
        fn evaluate(&self, state: &[f64]) -> f64 {
            -0.5 * state.iter().map(|x| x.powi(2)).sum::<f64>()
        }
    }

    #[test]
    fn test_mcmc_sampling() {
        let config = MCMCConfig {
            n_samples: 100,
            burn_in: 10,
            thin: 1,
            initial_state: vec![0.0],
            proposal: GaussianProposal::new(0.5),
            adaptation_interval: 50,
        };

        let log_likelihood = TestLogLikelihood;
        let mut sampler = MetropolisHastings::new(config, log_likelihood);

        let samples = sampler.sample_chain().unwrap();
        assert_eq!(samples.len(), 100);
    }
}
