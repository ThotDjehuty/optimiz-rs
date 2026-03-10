# Documentation Improvements Needed

**Date:** 2026-02-17  
**Version:** v1.0.1

## Summary

The user identified several critical gaps in the Optimiz-rs documentation that need to be addressed:

1. **Optimal Control Page (`docs/source/algorithms/optimal_control.md`)**
   - Needs much more detail on HJB equations
   - Needs explanation of viscosity solutions
   - Needs to introduce what's actually in the code/module
   - Needs more mathematical foundations

2. **HMM API Page (`docs/source/api/hmm.md`)**
   - Currently almost empty (only ~16 lines)
   - Needs explanation of what algorithms are implemented
   - Needs details on how they work  
   - Needs guidance on when/how to use them

3. **General Documentation**
   - Make more concise and detailed throughout
   - Better balance of theory and practice

## Required Enhancements

### 1. Optimal Control Documentation

#### Mathematical Foundations Needed:
- **HJB Equation:** Full derivation and intuition
  - General form for stochastic processes
  - Specialization to Ornstein-Uhlenbeck process
  - Connection to optimal stopping/switching problems
  
- **Viscosity Solutions:** Detailed explanation
  - Why classical solutions don't exist (kinks at boundaries)
  - Definition of viscosity solutions
  - Numerical approximation via upwind schemes
  - Monotonicity and convergence properties

- **Finite Difference Methods:**
  - Grid discretization approach
  - Upwind vs central differences
  - Policy iteration algorithm
  - Convergence criteria

#### Implementation Details Needed:
- **What's Actually in the Module:**
  - HJB solver for OU process (src/optimal_control/hjb_solver.rs)
  - Viscosity solution solver (src/optimal_control/viscosity.rs)
  - Regime switching (src/optimal_control/regime_switching.rs)
  - Jump diffusion (src/optimal_control/jump_diffusion.rs)
  - MRSJD - Multi-Regime Switching Jump Diffusion (src/optimal_control/mrsjd.rs)
  - OU parameter estimation (src/optimal_control/ou_estimator.rs)
  - Kalman filters: Linear, EKF, UKF (src/optimal_control/kalman_filter.rs)
  - Backtesting framework (src/optimal_control/backtest.rs)

#### Usage Guidance Needed:
- When to use each algorithm
- Parameter tuning guidelines
- Diagnostic plots and convergence monitoring
- Integration with other modules (HMM, Mean Field Games)
- Real-world trading examples

### 2. HMM API Documentation

#### Algorithms to Document:
- **Forward Algorithm:** Compute P(O|λ) efficiently
  - Forward variable α_t(i)
  - Recursive computation
  - Numerical stability (scaling)
  
- **Backward Algorithm:** Alternative for completeness
  - Backward variable β_t(i)
  - Use in Baum-Welch

- **Viterbi Algorithm:** Most likely state sequence
  - Dynamic programming approach
  - Backtracking for path recovery

- **Baum-Welch (EM) Algorithm:** Parameter learning
  - E-step: compute γ_t(i) and ξ_t(i,j)
  - M-step: update π, A, B parameters
  - Convergence properties

#### API Methods to Explain:
- **`HMM(n_states)`:** Constructor
  - When to use 2 vs 3+ states
  - Initialization strategy

- **`fit(X, n_iterations, tolerance)`:** Training
  - What data X should look like
  - How many iterations needed
  - Convergence diagnostics
  - Multiple random restarts

- **`predict(X)`:** Viterbi decoding
  - Returns most likely state sequence
  - Use cases: regime detection, trading signals

- **`score(X)`:** Log-likelihood
  - Model comparison
  - Convergence monitoring
  - Anomaly detection

#### Usage Examples Needed:
- **Regime Detection:**
  - Market regimes (bull/bear)
  - Volatility regimes (high/low)
  - Integration with optimal control

- **Parameter Estimation Per Regime:**
  - Combine with OU parameter estimation
  - Regime-specific HJB solving
  
- **Model Selection:**
  - BIC/AIC for choosing number of states
  - Cross-validation approaches

- **Numerical Best Practices:**
  - Data requirements (minimum samples)
  - Handling outliers
  - Initialization sensitivity
  - Convergence diagnostics

### 3. API Reference Page (`docs/source/api/optimal_control.md`)

Currently 117 lines but needs:
- Complete function signatures
- Parameter descriptions with types
- Return value specifications
- Detailed examples for each function
- Error handling documentation

## Implementation Plan

### Phase 1: Mathematical Foundations (High Priority)
1. Expand optimal_control.md with HJB equation derivations
2. Add viscosity solutions section with theory and numerics
3. Add finite difference methods explanation

### Phase 2: Algorithm Details (High Priority)
1. HMM API documentation expansion
2. Detail each algorithm (forward, backward, Viterbi, Baum-Welch)
3. Add mathematical formulas and intuition

### Phase 3: Usage Guidance (Medium Priority)
1. Add "When to Use" sections for each algorithm
2. Parameter tuning guidelines
3. Diagnostic procedures
4. Integration examples

### Phase 4: API Reference (Medium Priority)
1. Complete function signatures
2. Parameter and return types
3. Error documentation
4. Cross-references

### Phase 5: Examples and Tutorials (Low Priority)
1. Jupyter notebooks for common use cases
2. End-to-end workflows
3. Performance benchmarking examples

## Technical Notes

### Current Implementation Status:

**Optimal Control Module (`src/optimal_control/`):**
- ✅ HJB solver (hjb_solver.rs)
- ✅ Viscosity solutions (viscosity.rs)
-  ✅ Regime switching (regime_switching.rs)
- ✅ Jump diffusion (jump_diffusion.rs)
- ✅ MRSJD (mrsjd.rs)
- ✅ OU estimation (ou_estimator.rs)
- ✅ Kalman filters (kalman_filter.rs, kalman_py_bindings.rs)
- ✅ Backtesting (backtest.rs)

**HMM Module (`src/hmm/`):**
- ✅ Gaussian emissions (emission.rs)
- ✅ Forward-Backward algorithm (model.rs)
- ✅ Viterbi decoding (viterbi.rs)
- ✅ Baum-Welch training (model.rs)
- ✅ Python bindings (python_bindings.rs)

### Documentation Files to Update:

1. `docs/source/algorithms/optimal_control.md` (currently 94 lines → target: 500+ lines)
2. `docs/source/api/hmm.md` (currently 16 lines → target: 300+ lines)
3. `docs/source/api/optimal_control.md` (currently 117 lines → target: 400+ lines)  
4. `docs/source/algorithms/hmm.md` (currently 607 lines → verify completeness)

### Backup Files Created:

- `docs/source/algorithms/optimal_control.md.backup`
- `docs/source/api/hmm.md.backup`
- `docs/source/api/optimal_control.md.backup`

## Next Steps

1. **Immediate:** Write comprehensive optimal_control mathematical foundations
2. **Immediate:** Expand HMM API documentation with algorithm details
3. **Soon:** Add usage examples and integration guides
4. **Later:** Create Jupyter notebook tutorials

## References Needed

### Optimal Control:
- Fleming & Soner (2006): Controlled Markov Processes and Viscosity Solutions
- Øksendal (2003): Stochastic Differential Equations
- Pham (2009): Continuous-time Stochastic Control and Optimization
- Barles & Souganidis (1991): Convergence of approximation schemes

### HMM:
- Rabiner (1989): Tutorial on HMMs and selected applications
- Murphy (2012): Machine Learning: A Probabilistic Perspective
- Bishop (2006): Pattern Recognition and Machine Learning

### Kalman Filtering:
- Kalman (1960): A New Approach to Linear Filtering
- Julier & Uhlmann (1997): Unscented Kalman Filter

---

**Status:** Documentation gaps identified. Implementation in progress.
**Priority:** High - These are critical for user onboarding and proper usage.
