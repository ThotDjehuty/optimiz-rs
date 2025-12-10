//! Viterbi decoding algorithm for finding the most likely state sequence
//!
//! Implements the Viterbi algorithm for HMM inference.

use super::emission::EmissionModel;
use super::model::HMM;
use crate::core::Result;

impl<E: EmissionModel> HMM<E> {
    /// Viterbi decoding: Find most likely state sequence
    pub fn viterbi(&self, observations: &[f64]) -> Result<Vec<usize>> {
        let n_obs = observations.len();
        let n_states = self.config.n_states;

        if n_obs == 0 {
            return Ok(Vec::new());
        }

        let mut delta = vec![vec![f64::NEG_INFINITY; n_states]; n_obs];
        let mut psi = vec![vec![0usize; n_states]; n_obs];

        // Initialize
        for s in 0..n_states {
            delta[0][s] = self.initial_probs[s].ln()
                + self
                    .config
                    .emission_model
                    .probability(observations[0], s)
                    .ln();
        }

        // Recursion
        for t in 1..n_obs {
            for s in 0..n_states {
                let (max_state, max_val) = (0..n_states)
                    .map(|prev_s| {
                        (
                            prev_s,
                            delta[t - 1][prev_s] + self.transition_matrix[prev_s][s].ln(),
                        )
                    })
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                psi[t][s] = max_state;
                delta[t][s] = max_val
                    + self
                        .config
                        .emission_model
                        .probability(observations[t], s)
                        .ln();
            }
        }

        // Backtrack
        let mut path = vec![0usize; n_obs];
        path[n_obs - 1] = (0..n_states)
            .max_by(|&a, &b| {
                delta[n_obs - 1][a]
                    .partial_cmp(&delta[n_obs - 1][b])
                    .unwrap()
            })
            .unwrap();

        for t in (0..n_obs - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm::config::HMMConfig;
    use crate::hmm::emission::GaussianEmission;

    #[test]
    fn test_viterbi() {
        let observations = vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2];
        let config = HMMConfig::<GaussianEmission>::builder(2).build().unwrap();

        let mut hmm = HMM::new(config);
        hmm.fit(&observations).unwrap();

        let path = hmm.viterbi(&observations).unwrap();
        assert_eq!(path.len(), observations.len());
    }
}
