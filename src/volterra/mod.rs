//! Volterra integral / fractional ODE solvers.
//!
//! Submodules:
//!
//! * `fractional_riccati`  -- Adams predictor--corrector solver for
//!                            Caputo fractional ODEs (Diethelm--Ford--Freed 2002).
//! * `markovian_lift`      -- multi-exponential approximation of a
//!                            convolution kernel `K(t) ~= sum c_j exp(-gamma_j t)`.
//! * `volterra_solver`     -- generic second-kind Volterra equations.
//! * `fourier_inversion`   -- characteristic function -> density via FFT.

pub mod fourier_inversion;
pub mod fractional_riccati;
pub mod markovian_lift;
pub mod volterra_solver;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

