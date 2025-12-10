//! Differential Evolution optimization module
//!
//! Modular DE implementation with multiple strategies,
//! adaptive parameter control, and parallel evaluation.

#[cfg(feature = "python-bindings")]
pub use crate::differential_evolution::{
    differential_evolution, ConvergenceRecord, DEResult, DEStrategy,
};
