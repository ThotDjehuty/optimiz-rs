//! Path signatures and related tensor algebra utilities.
//!
//! Submodules:
//!
//! * `path_signature`     -- truncated tensor signature.
//! * `log_signature`      -- log-signature in a Hall basis.
//! * `random_signature`   -- random projection of the signature.
//! * `signature_kernel`   -- Salvi--Cass--Lyons signature kernel PDE.
//! * `utils`              -- shuffle product, concatenation helpers.

pub mod log_signature;
pub mod path_signature;
pub mod random_signature;
pub mod signature_kernel;
pub mod utils;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

