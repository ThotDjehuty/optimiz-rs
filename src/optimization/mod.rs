//! Generative calibration hooks (v2.0.0)
//! ======================================
//!
//! Skeleton trait suite letting external generative models (normalising
//! flows, GANs, score-based diffusion, ...) plug into Rust-side calibration
//! loops.  All numerics are CPU-only and fully generic — no domain-specific
//! vocabulary.

pub mod generative_calibration_hooks;
#[cfg(feature = "python-bindings")]
pub mod python_bindings;

pub use generative_calibration_hooks::{
    GenerativeSampler, MmdLoss, mmd_distance, calibration_step,
};
