# Changelog

All notable changes to **optimiz-rs** are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-10-20

### Fixed — the crate now builds with no system libraries

- **Removed the hard dependency on a system OpenBLAS.** `ndarray-linalg 0.16`
  with the `openblas-system` backend made a system BLAS a build requirement of
  a *published* crate: `cargo add optimiz-rs` failed to link on any machine
  without one, and on macOS Homebrew's `openblas` is keg-only and pulls in
  `gcc`, `gmp`, `isl`, `libmpc` and `mpfr`. There is now no `ndarray-linalg`,
  no `openblas-src`, no `build.rs` and no system library of any kind.
- `cargo build` on a bare checkout completes in seconds; `cargo tree` shows no
  BLAS or LAPACK crate at all.

### Added — `linalg`: dependency-free dense linear algebra

- `src/linalg.rs` implements exactly what the crate used from LAPACK, in safe
  Rust: LU with partial pivoting (`solve`, `inv`, `det`), Cholesky, and a
  one-sided Jacobi `svd`. Traits mirror `ndarray-linalg`'s names
  (`Solve`, `Inverse`, `Determinant`, `Cholesky`, `Svd`, `Norm`) so call sites
  only change an import.
- **Behaviour change:** `svd` returns the **thin** factorisation (`U` is
  `m x k`, `Vt` is `k x n`, `k = min(m, n)`) where `ndarray-linalg` returned
  full square factors. Thin is what `sparse_optimization` needs — it
  reconstructs `U diag(s) Vt`, which is only dimensionally valid for thin
  factors on a non-square matrix, so the previous code could not have worked
  off the diagonal.
- 12 tests: reconstruction for square/tall/wide, orthonormality of the
  factors, rank deficiency, known spectra, pivoting, and typed errors instead
  of panics.

### Added — `losses`: differentiable training objectives

- `src/losses.rs` adds the objectives you *minimise*, alongside the benchmark
  surfaces already in `rust_objectives`. A `Loss` trait returns both value and
  gradient: `CrossEntropy` (label smoothing, `ignore_index` masking,
  overflow-free log-softmax), `BinaryCrossEntropy`, `MeanSquaredError`,
  `MeanAbsoluteError`, `Huber`.
- Every gradient is checked against central finite differences, and one test
  asserts gradient descent on the objective actually reduces it.
- Python bindings: `mse`, `mae`, `huber`, `binary_cross_entropy`,
  `cross_entropy` and `weighted_mse_sum` — the last being the composite PINN
  objective evaluated in one call.
- `thotbook-ai`'s `llm-train` now delegates its training loss here, so there
  is one audited definition of cross-entropy in the stack instead of two.

### Notes

- 153 tests pass (129 previously, plus 12 `linalg` and 12 `losses`).
- No public API was removed. The only behavioural change is the thin `svd`.

## [2.0.0] - 2026-05-14

### Added — public release of the v2 API

- Promoted `2.0.0-alpha.1` to the stable `2.0.0` release.
- PyPI distribution name remains `optimiz-rs` (continuity with v1.0.x);
  the Rust crate is also `optimiz-rs`. Both expose the Python module
  `optimizr`.
- New non-regression suite `tests/test_v2_api.py` (20 tests) exercising
  every advertised v2 primitive against an analytic ground truth:
  `historical_var_py`, `solve_fractional_ode`, `solve_volterra`,
  `linear_bsde_constant_coeffs`, `mean_reverting_mckean_vlasov`, plus a
  parametrised guard over the v1.x public surface.
- README rewritten to document the v2 Python API and the corrected
  installation command (`pip install optimizr`).

### Notes

- No source-level breaking change relative to `2.0.0-alpha.1`.
- All v1.x Python entry points remain exposed (`differential_evolution`,
  `fit_hmm`, `viterbi_decode`, `mcmc_sample`, `grid_search`, `mutual_information`,
  `shannon_entropy`, etc.) — verified by `test_public_symbol_exposed`.

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
