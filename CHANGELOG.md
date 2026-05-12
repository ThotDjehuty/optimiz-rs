# Changelog

All notable changes to **optimiz-rs** are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-05-12

### Added — purely additive, no existing API changes

- `optimal_control::matrix_riccati` — RK4 backward solver for the
  matrix Riccati differential equation
  `dA/dt = -2 A M A + Q`, plus terminal-condition variants for the
  associated affine and constant components.
- `timeseries_utils::nonsync_covariance` — Hayashi--Yoshida estimator
  for asynchronous covariance, with parallel matrix variant.
- `timeseries_utils::wavelet` — discrete and maximum-overlap wavelet
  transforms (Haar, Daubechies orders 2--10) with periodic boundaries.
- `risk_measures` — empirical and parametric Value-at-Risk and
  Conditional Value-at-Risk estimators, plus a projected sub-gradient
  solver for convex CVaR minimisation over the unit simplex.
- `graph::laplacian` — combinatorial, symmetric-normalised and
  random-walk graph Laplacians.
- `graph::spectral_clustering` — spectral clustering via Jacobi
  diagonalisation and Lloyd's algorithm with k-means++ initialisation.
- `topology::persistent_homology` — Vietoris--Rips persistent homology
  by the standard `Z/2` matrix-reduction algorithm.
- `topology::bottleneck` — bottleneck distance between persistence
  diagrams via Hopcroft--Karp matching with binary search.
- `volterra::fractional_riccati` — Adams predictor--corrector solver
  for Caputo fractional ODEs (Diethelm--Ford--Freed 2002).
- `volterra::markovian_lift` — multi-exponential approximation of
  convolution kernels by non-negative least squares on a geometric
  grid.
- `volterra::volterra_solver` — generic second-kind Volterra integral
  equation solver via product-trapezoidal quadrature.
- `volterra::fourier_inversion` — direct trapezoidal Fourier inversion
  of a characteristic function on a uniform frequency grid.
- `signatures::path_signature` — truncated tensor signature of a
  piecewise-linear path with truncated tensor exponential.
- `signatures::log_signature` — truncated tensor logarithm of a
  signature.
- `signatures::random_signature` — Cuchiero--Schmocker--Teichmann
  random reservoir projection of the signature.
- `signatures::signature_kernel` — Salvi--Cass--Lyons signature kernel
  via the Goursat finite-difference scheme.
- `signatures::utils` — shuffle product and Chen-identity-driven
  signature concatenation.

### Changed

- Bumped crate version from `1.0.1` to `1.1.0`. All previously stable
  symbols remain untouched and binary-compatible at the Python ABI
  level (`abi3-py38`).

### Notes

This release is **CPU-only and additive**. No Python bindings were added
in `1.1.0`; the new modules are exposed via the Rust API only and will
be wrapped behind the `python-bindings` feature in a follow-up release.
