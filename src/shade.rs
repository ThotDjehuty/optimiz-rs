///! SHADE - Success-History based Adaptive Differential Evolution
///!
///! Implementation of SHADE algorithm from Tanabe & Fukunaga (2013):
///! "Success-history based parameter adaptation for Differential Evolution"
///! IEEE Congress on Evolutionary Computation (CEC) 2013
///!
///! Key improvements over jDE:
///! - Historical memory of successful (F, CR) parameters
///! - Weighted random selection from success history
///! - Cauchy distribution for F sampling (better exploration)
///! - Normal distribution for CR sampling (better exploitation)
///! - 10-20% better convergence on benchmark functions
///!
///! # Algorithm Overview
///!
///! 1. Initialize circular memory buffer (size H=10-100) with 0.5
///! 2. For each individual in population:
///!    a. Randomly select memory index r
///!    b. Sample F ~ Cauchy(memory_f[r], 0.1) and clamp to [0, 1]
///!    c. Sample CR ~ Normal(memory_cr[r], 0.1) and clamp to [0, 1]
///!    d. Generate trial vector using sampled F and CR
///! 3. After generation, update memory with successful parameters:
///!    - Compute weighted mean of successful F and CR values
///!    - Update memory at current position (circular buffer)
///!
///! # Performance
///!
///! SHADE consistently outperforms jDE on:
///! - CEC2013 benchmark suite
///! - High-dimensional problems (D > 30)
///! - Multimodal functions
///! - Convergence speed (fewer evaluations to target)

use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand_distr::{Cauchy, Normal};

/// SHADE memory for storing successful parameter history
#[derive(Clone, Debug)]
pub struct SHADEMemory {
    /// Historical F (mutation factor) values
    history_f: Vec<f64>,
    /// Historical CR (crossover rate) values
    history_cr: Vec<f64>,
    /// Current position in circular buffer
    index: usize,
    /// Memory size H (typically 10-100)
    size: usize,
}

impl SHADEMemory {
    /// Create new SHADE memory with given size
    ///
    /// # Arguments
    /// * `size` - Memory buffer size H (recommended: 10-100)
    ///
    /// # Returns
    /// New SHADEMemory initialized with 0.5 for all entries
    pub fn new(size: usize) -> Self {
        Self {
            history_f: vec![0.5; size],
            history_cr: vec![0.5; size],
            index: 0,
            size,
        }
    }

    /// Sample F from Cauchy distribution centered on random history entry
    ///
    /// Process:
    /// 1. Randomly select memory index r
    /// 2. Sample F ~ Cauchy(history_f[r], 0.1)
    /// 3. Clamp to [0, 1], regenerate if outside bounds
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Sampled F value in [0, 1]
    pub fn sample_f<R: Rng>(&self, rng: &mut R) -> f64 {
        // Randomly select memory entry
        let r = rng.gen_range(0..self.size);
        let mean_f = self.history_f[r];

        // Sample from Cauchy distribution
        let cauchy = Cauchy::new(mean_f, 0.1).unwrap();

        // Regenerate until valid (in [0, 1])
        loop {
            let f = cauchy.sample(rng);
            if (0.0..=1.0).contains(&f) {
                return f;
            }
            // Cauchy has heavy tails, may generate extreme values
            // Clamp instead of infinite loop
            if f < 0.0 {
                return 0.0;
            }
            if f > 1.0 {
                return 1.0;
            }
        }
    }

    /// Sample CR from Normal distribution centered on random history entry
    ///
    /// Process:
    /// 1. Randomly select memory index r
    /// 2. Sample CR ~ Normal(history_cr[r], 0.1)
    /// 3. Clamp to [0, 1]
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Sampled CR value in [0, 1]
    pub fn sample_cr<R: Rng>(&self, rng: &mut R) -> f64 {
        // Randomly select memory entry
        let r = rng.gen_range(0..self.size);
        let mean_cr = self.history_cr[r];

        // Sample from Normal distribution
        let normal = Normal::new(mean_cr, 0.1).unwrap();
        let cr = normal.sample(rng);

        // Clamp to [0, 1]
        cr.clamp(0.0, 1.0)
    }

    /// Update memory with successful parameters
    ///
    /// Uses weighted Lehmer mean for successful parameters:
    /// mean_wL(S) = sum(w_i * s_i^2) / sum(w_i * s_i)
    ///
    /// where w_i = improvement_i / sum(improvements)
    ///
    /// # Arguments
    /// * `successful_f` - F values that led to improvement
    /// * `successful_cr` - CR values that led to improvement
    /// * `improvements` - Fitness improvements for each success
    pub fn update(
        &mut self,
        successful_f: &[f64],
        successful_cr: &[f64],
        improvements: &[f64],
    ) {
        if successful_f.is_empty() {
            return; // No successful parameters this generation
        }

        // Compute weights (normalized improvements)
        let total_improvement: f64 = improvements.iter().sum();
        if total_improvement <= 0.0 {
            return; // No actual improvement
        }

        let weights: Vec<f64> = improvements
            .iter()
            .map(|imp| imp / total_improvement)
            .collect();

        // Weighted Lehmer mean for F
        let numerator_f: f64 = weights
            .iter()
            .zip(successful_f)
            .map(|(w, f)| w * f * f)
            .sum();
        let denominator_f: f64 = weights
            .iter()
            .zip(successful_f)
            .map(|(w, f)| w * f)
            .sum();

        let mean_f = if denominator_f > 0.0 {
            numerator_f / denominator_f
        } else {
            0.5 // Fallback
        };

        // Arithmetic mean for CR (as per SHADE paper)
        let mean_cr: f64 = weights
            .iter()
            .zip(successful_cr)
            .map(|(w, cr)| w * cr)
            .sum();

        // Update memory at current position (circular buffer)
        self.history_f[self.index] = mean_f.clamp(0.0, 1.0);
        self.history_cr[self.index] = mean_cr.clamp(0.0, 1.0);

        // Advance circular buffer index
        self.index = (self.index + 1) % self.size;
    }

    /// Get current memory state (for debugging/visualization)
    pub fn get_state(&self) -> (Vec<f64>, Vec<f64>, usize) {
        (
            self.history_f.clone(),
            self.history_cr.clone(),
            self.index,
        )
    }

    /// Reset memory to initial state
    pub fn reset(&mut self) {
        self.history_f.fill(0.5);
        self.history_cr.fill(0.5);
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shade_memory_creation() {
        let mem = SHADEMemory::new(10);
        assert_eq!(mem.size, 10);
        assert_eq!(mem.history_f.len(), 10);
        assert_eq!(mem.history_cr.len(), 10);
        assert!(mem.history_f.iter().all(|&x| (x - 0.5).abs() < 1e-10));
    }

    #[test]
    fn test_shade_memory_sampling() {
        let mem = SHADEMemory::new(10);
        let mut rng = StdRng::seed_from_u64(42);

        // Sample F and CR multiple times
        for _ in 0..100 {
            let f = mem.sample_f(&mut rng);
            let cr = mem.sample_cr(&mut rng);

            assert!((0.0..=1.0).contains(&f), "F={} out of bounds", f);
            assert!((0.0..=1.0).contains(&cr), "CR={} out of bounds", cr);
        }
    }

    #[test]
    fn test_shade_memory_update() {
        let mut mem = SHADEMemory::new(5);

        // Simulate successful parameters
        let successful_f = vec![0.7, 0.8, 0.9];
        let successful_cr = vec![0.6, 0.7, 0.8];
        let improvements = vec![0.1, 0.2, 0.3]; // Weighted by improvement

        mem.update(&successful_f, &successful_cr, &improvements);

        // Check that memory was updated
        let (hist_f, hist_cr, idx) = mem.get_state();

        // Index should have advanced
        assert_eq!(idx, 1);

        // First entry should be updated with weighted mean
        // Weighted Lehmer mean of [0.7, 0.8, 0.9] with weights [1/6, 2/6, 3/6]
        let expected_f = (0.7 * 0.7 * (0.1 / 0.6) + 0.8 * 0.8 * (0.2 / 0.6) + 0.9 * 0.9 * (0.3 / 0.6))
            / (0.7 * (0.1 / 0.6) + 0.8 * (0.2 / 0.6) + 0.9 * (0.3 / 0.6));

        assert!(
            (hist_f[0] - expected_f).abs() < 0.01,
            "F memory updated incorrectly: {} vs {}",
            hist_f[0],
            expected_f
        );
    }

    #[test]
    fn test_shade_memory_circular_buffer() {
        let mut mem = SHADEMemory::new(3);

        // Fill buffer
        for i in 0..5 {
            let f_val = vec![0.1 * (i + 1) as f64];
            let cr_val = vec![0.1 * (i + 1) as f64];
            let imp = vec![1.0];

            mem.update(&f_val, &cr_val, &imp);
        }

        // Index should wrap around
        let (_, _, idx) = mem.get_state();
        assert_eq!(idx, 2); // (5 % 3) = 2
    }

    #[test]
    fn test_shade_memory_reset() {
        let mut mem = SHADEMemory::new(5);

        // Update memory
        mem.update(&vec![0.8], &vec![0.7], &vec![1.0]);

        // Reset
        mem.reset();

        let (hist_f, hist_cr, idx) = mem.get_state();
        assert_eq!(idx, 0);
        assert!(hist_f.iter().all(|&x| (x - 0.5).abs() < 1e-10));
        assert!(hist_cr.iter().all(|&x| (x - 0.5).abs() < 1e-10));
    }
}
