//! Log-likelihood interface for MCMC
//!
//! Defines the LogLikelihood trait for target distributions.

/// Generic log-likelihood function trait
pub trait LogLikelihood: Send + Sync {
    fn evaluate(&self, state: &[f64]) -> f64;
}

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Wrapper for Python callable log-likelihood
#[cfg(feature = "python-bindings")]
pub struct PyLogLikelihood {
    func: Py<PyAny>,
}

#[cfg(feature = "python-bindings")]
impl PyLogLikelihood {
    pub fn new(func: Py<PyAny>) -> Self {
        Self { func }
    }
}

#[cfg(feature = "python-bindings")]
impl LogLikelihood for PyLogLikelihood {
    fn evaluate(&self, state: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let args = (state.to_vec(),);
            self.func
                .call1(py, args)
                .and_then(|res| res.extract::<f64>(py))
                .unwrap_or(f64::NEG_INFINITY)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestLogLikelihood;

    impl LogLikelihood for TestLogLikelihood {
        fn evaluate(&self, state: &[f64]) -> f64 {
            // Standard normal log-likelihood
            -0.5 * state.iter().map(|x| x.powi(2)).sum::<f64>()
        }
    }

    #[test]
    fn test_log_likelihood() {
        let ll = TestLogLikelihood;
        assert!(ll.evaluate(&[0.0]) > ll.evaluate(&[1.0]));
    }
}
