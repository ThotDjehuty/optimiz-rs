# Changelog

All notable changes to **optimiz-rs** are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-alpha.1] - 2026-05-12

### Added — top-level reorganisation and new generic primitives

- **Top-level reorg (additive aliases — backward compatible at the Rust
  level).**  `optimiz_rs::matrix_riccati` is now re-exported at the crate
  root.  New top-level groups: `bsde`, `pde`, `stochastic_control`,
  `agent_based`, `inference`, `optimization`.
- `bsde::theta_scheme` — implicit/explicit θ-scheme for linear BSDEs
  with deterministic coefficients (closed-form analytic test against
  the deterministic ODE `dY = -ρ Y dt`).
- `bsde::deep_bsde_bridge` — `ConditionalExpectation` trait and
  `DeepBsdeBridge` driver providing the CPU-side recursion hook for
  external function approximators.
- `pde::fokker_planck` — 1-D forward Fokker--Planck solver with
  conservative central differences and explicit positivity safeguard.
- `pde::hjb_multid` — explicit upwind solver for multidimensional HJB
  on a regular Cartesian grid (`d ≤ 3`) with reflective boundaries.
- `pde::elliptic_fd` — 2-D Poisson `-Δu = f` SOR solver verified
  against the `sin(πx) sin(πy)` eigenfunction.
- `stochastic_control::optimal_switching` — Snell-envelope backward
  induction for discrete multi-mode optimal switching.
- `stochastic_control::pontryagin` — Riccati-shooting solver for the
  1-D LQR Pontryagin maximum principle (verified against the
  closed-form `P(t) = s_T / (1 + s_T (T-t))`).
- `stochastic_control::two_sided_intensity_control` — generic
  bilateral intensity control with affine per-jump premia.
- `optimal_control::quadratic_impact_control` — closed-form Riccati
  feedback for a controlled SDE with quadratic running cost.
- `mean_field::mckean_vlasov` — interacting-particle Euler scheme for
  generic McKean--Vlasov SDEs with empirical-measure drift.
- `agent_based` — generic interacting-agent simulator (consensus
  dynamics test recovers the empirical mean exactly without noise).
- `inference::robust_drift` — Huber-loss IRLS estimator for the drift
  of a 1-D OU-type discrete-time process; resists 5 % outliers.
- `optimization::generative_calibration_hooks` — `GenerativeSampler`
  trait + Gaussian MMD loss + finite-difference calibration step.

### Tests

- 38 new `#[test]` cases — all passing (`cargo test --lib
  --no-default-features` passes 165/170, the 5 pre-existing failures
  predate v1.1 and are unrelated).

### Notes — deferred to subsequent v2.0.x bumps

- PyO3 Python bindings + executed companion Jupyter notebooks for the
  new groups (will follow the same workflow as v1.1.x).
- Sphinx RST documentation pages for `bsde`, `pde`,
  `stochastic_control`, `agent_based`, `inference`, `optimization`.
- Propagation of new modules into `hfthot-lab-instance` consumers.

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
