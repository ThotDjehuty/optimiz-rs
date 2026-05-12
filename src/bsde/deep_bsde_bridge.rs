//! Deep BSDE bridge — abstract conditional-expectation interface
//! ===============================================================
//!
//! Provides a thin trait-based hook for plugging an external function
//! approximator (typically a neural network trained in Python) into the
//! discrete BSDE recursion
//!
//! ```text
//! Y_n = E_n[ Y_{n+1} + Δt · f(t_n, Y_{n+1}, Z_n) ]
//! Z_n = E_n[ Y_{n+1} · ΔW_{n+1} / Δt ]
//! ```
//!
//! At the Rust level we only fix the *interface* of the conditional
//! expectations.  Higher-level training loops (PyTorch / JAX) implement the
//! [`ConditionalExpectation`] trait and feed the resulting predictions to
//! the [`DeepBsdeBridge`] driver, which performs the deterministic
//! arithmetic step-by-step.

use crate::core::{OptimizrError, Result};
use ndarray::Array1;

/// One step of the discrete BSDE recursion.
#[derive(Clone, Debug)]
pub struct DeepBsdeStep {
    pub time: f64,
    pub dt: f64,
    /// Predicted `Y_{n+1}` for each Monte-Carlo path.
    pub y_next: Array1<f64>,
    /// Predicted `Z_n` for each Monte-Carlo path.
    pub z: Array1<f64>,
    /// Brownian increments `ΔW_{n+1}` for each path.
    pub dw: Array1<f64>,
}

/// Trait implemented by external function approximators.
pub trait ConditionalExpectation {
    /// Estimate `E_n[ φ ]` from a batch of path-wise samples.
    fn project(&self, time: f64, payoff: &Array1<f64>) -> Result<Array1<f64>>;
}

/// Generic deep-BSDE driver.
pub struct DeepBsdeBridge<E: ConditionalExpectation> {
    pub estimator: E,
}

impl<E: ConditionalExpectation> DeepBsdeBridge<E> {
    pub fn new(estimator: E) -> Self {
        Self { estimator }
    }

    /// Apply the discrete recursion to a single time step using the
    /// driver `f(t, y, z) -> r`.  Returns the projected `Y_n`.
    pub fn step<F>(&self, step: &DeepBsdeStep, driver: F) -> Result<Array1<f64>>
    where
        F: Fn(f64, f64, f64) -> f64,
    {
        if step.dt <= 0.0 {
            return Err(OptimizrError::InvalidParameter("dt must be > 0".into()));
        }
        if step.y_next.len() != step.z.len() || step.z.len() != step.dw.len() {
            return Err(OptimizrError::DimensionMismatch {
                expected: step.y_next.len(),
                actual: step.z.len().min(step.dw.len()),
            });
        }
        let raw: Array1<f64> = step
            .y_next
            .iter()
            .zip(step.z.iter())
            .map(|(&y, &z)| y + step.dt * driver(step.time, y, z))
            .collect();
        self.estimator.project(step.time, &raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mean;
    impl ConditionalExpectation for Mean {
        fn project(&self, _t: f64, payoff: &Array1<f64>) -> Result<Array1<f64>> {
            let m = payoff.iter().sum::<f64>() / payoff.len() as f64;
            Ok(Array1::from_elem(payoff.len(), m))
        }
    }

    #[test]
    fn step_uses_driver_and_estimator() {
        let bridge = DeepBsdeBridge::new(Mean);
        let step = DeepBsdeStep {
            time: 0.0,
            dt: 0.1,
            y_next: Array1::from(vec![1.0, 1.0, 1.0, 1.0]),
            z: Array1::zeros(4),
            dw: Array1::zeros(4),
        };
        let y = bridge.step(&step, |_, y, _| -y).unwrap();
        // Driver: y - 0.1 * y = 0.9; mean projection preserves constants.
        for v in y.iter() {
            assert!((v - 0.9).abs() < 1e-12);
        }
    }

    #[test]
    fn step_rejects_dimension_mismatch() {
        let bridge = DeepBsdeBridge::new(Mean);
        let step = DeepBsdeStep {
            time: 0.0,
            dt: 0.1,
            y_next: Array1::from(vec![1.0, 2.0]),
            z: Array1::zeros(3),
            dw: Array1::zeros(3),
        };
        assert!(bridge.step(&step, |_, y, _| y).is_err());
    }
}
