//! Generic empirical risk measures.
//!
//! This module exposes purely sample-based risk functionals over a vector
//! of decision variables `w` and a sample of losses
//! `L^{(s)} = -<r^{(s)}, w>` (the loss is signed; users pass the sample
//! of returns `r^{(s)}` directly). All algorithms work on arbitrary
//! datasets and make no domain-specific assumption.

use crate::core::{OptimizrError, Result};

pub mod var;
pub mod cvar;


pub use cvar::{cvar_value, minimize_cvar, CVaRConfig, CVaRResult};
pub use var::{historical_var, parametric_var};

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

#[inline]
pub(crate) fn check_alpha(alpha: f64) -> Result<()> {
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(OptimizrError::InvalidParameter(format!(
            "alpha must lie in (0, 1), got {}",
            alpha
        )));
    }
    Ok(())
}
