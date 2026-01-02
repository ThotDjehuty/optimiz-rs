# OptimizR Enhancement Strategy

**Date**: January 2, 2025  
**Context**: Post-Polaroid Phase 4, exploring integration and improvements  
**Based On**: v0.2.0 codebase review, roadmap analysis, synergy opportunities

## Current State Analysis

### ✅ What's Implemented (v0.2.0)

1. **Core Algorithms**:
   - Differential Evolution (5 strategies: rand1, best1, currenttobest1, rand2, best2)
   - Hidden Markov Models (Baum-Welch, Viterbi)
   - MCMC Sampling (Metropolis-Hastings, adaptive proposals)
   - Grid Search
   - Information Theory (mutual information, Shannon entropy)

2. **Advanced Features (v0.2.0)**:
   - Sparse Optimization (Sparse PCA, Box-Tao, Elastic Net)
   - Optimal Control (HJB solver, regime switching, jump diffusion)
   - Risk Metrics (Hurst exponent, half-life, bootstrap)
   - Mathematical Toolkit (numerical differentiation, linear algebra, statistics)

3. **Architecture**:
   - Trait-based design (Optimizer, Sampler, InformationMeasure)
   - Builder pattern for configuration
   - Functional programming utilities (composition, memoization, pipes)
   - Rayon dependency already present
   - Feature flag infrastructure (`parallel` feature exists)

### ⚠️ What's Missing/Incomplete

1. **Parallelization BLOCKED**:
   - Infrastructure exists (Rayon trait, ParallelExecutor trait in core.rs)
   - DE has `parallel` parameter but **disabled** due to Python GIL
   - Comment: "Python callbacks cannot be safely parallelized due to GIL"
   - Grid search marked as "future: Expected 50-100x speedup"

2. **Advanced DE Variants (Roadmap v0.3.0)**:
   - JADE (jDE with archive)
   - SHADE (Success-History based Adaptive DE)
   - L-SHADE (with linear population reduction)
   - Current: Only basic jDE adaptive control

3. **Multi-Objective Optimization (Roadmap)**:
   - NSGA-DE (Non-dominated Sorting)
   - MODE (Multi-Objective DE)
   - Pareto front computation

4. **GPU Acceleration (Roadmap)**:
   - CUDA kernels
   - OpenCL support
   - 10-100× additional speedup

5. **Additional Algorithms (Roadmap)**:
   - Particle Swarm Optimization (PSO)
   - CMA-ES (Covariance Matrix Adaptation)
   - Simulated Annealing
   - Ant Colony Optimization

## Synergy Opportunities: Polaroid + OptimizR

### 1. Time-Series Feature Engineering for HMM
**Description**: Use Polaroid's time-series operations to create features for regime detection

**Implementation**:
```python
# Polaroid: Fast feature creation
df = client.lag(['price'], periods=1)  # Lagged prices
df = client.pct_change(['price'], periods=1)  # Returns
df = client.diff(['price'], periods=1)  # Price changes

# OptimizR: Regime detection on features
returns = df['price_pct_change'].to_numpy()
hmm = HMM(n_states=3)  # Bull, Bear, Sideways
hmm.fit(returns, n_iterations=100)
states = hmm.predict(returns)
```

**Value**:
- Polaroid provides fast feature engineering (50-200× faster for large datasets)
- OptimizR provides statistical inference (HMM regime detection)
- Combined: Real-time regime switching for trading strategies

### 2. Risk Metrics on Time-Series Data
**Description**: Calculate advanced risk metrics using both systems

**Implementation**:
```python
# Polaroid: Efficient return calculation
df = client.pct_change(['price'], periods=1)
returns = df['price_pct_change'].to_numpy()

# OptimizR: Risk analysis
hurst = compute_hurst_exponent(returns)  # Mean-reversion detection
half_life = estimate_half_life(returns)  # Reversion time
risk_metrics = compute_risk_metrics(returns)  # Comprehensive suite
```

**Value**:
- Fast preprocessing (Polaroid) + sophisticated analysis (OptimizR)
- Useful for pairs trading, mean-reversion strategies
- Real-time risk monitoring

### 3. Optimal Control with Market Data
**Description**: Dynamic portfolio rebalancing with regime-dependent strategies

**Implementation**:
```python
# Polaroid: Multi-asset feature creation
df = client.lag(['spy_price', 'vix'], periods=[1, 5, 20])
df = client.pct_change(['spy_price'], periods=1)

# OptimizR: Solve optimal control problem
# State: [price, volatility regime]
# Control: portfolio weights
value_fn = solve_hjb_regime_switching(...)
```

**Value**:
- Combines fast data processing with optimal control theory
- Regime-dependent strategies (bull vs bear market)
- Practical for HFT and algorithmic trading

### 4. Parameter Optimization for Trading Strategies
**Description**: Use DE to optimize strategy parameters on time-series data

**Implementation**:
```python
# Polaroid: Backtest execution (fast data ops)
def backtest_strategy(params):
    df = client.lag(['price'], periods=int(params[0]))
    # ... strategy logic ...
    return -sharpe_ratio  # Minimize negative Sharpe

# OptimizR: Find optimal parameters
result = differential_evolution(
    objective_fn=backtest_strategy,
    bounds=[(1, 50), (0.01, 0.5)],  # [lag_period, threshold]
    maxiter=500,
    strategy='rand1'
)
```

**Value**:
- Polaroid handles heavy data processing
- OptimizR finds optimal parameters
- 74-88× faster than SciPy DE

## High-Priority Enhancements

### Priority 1: Enable Parallelization for Pure-Rust Objectives

**Problem**: `parallel` parameter exists but disabled due to Python GIL issues

**Solution**: Create Rust-native objective function trait for GIL-free parallelization

**Implementation Strategy**:
1. Add `RustObjectiveFn` trait separate from Python callbacks
2. Implement parallel evaluation for Rust-native functions
3. Keep Python callbacks sequential (GIL limitation)
4. Enable parallel grid search (no Python callbacks needed for grid)

**Code Outline**:
```rust
// In src/core.rs or src/differential_evolution.rs

/// Rust-native objective function (no Python, no GIL)
pub trait RustObjective: Send + Sync {
    fn evaluate(&self, x: &[f64]) -> f64;
}

/// Parallel evaluation for Rust objectives
#[cfg(feature = "parallel")]
fn evaluate_population_parallel<F: RustObjective>(
    objective: &F,
    population: &[Vec<f64>]
) -> Vec<f64> {
    use rayon::prelude::*;
    population.par_iter()
        .map(|individual| objective.evaluate(individual))
        .collect()
}

// Python binding for benchmarking
#[pyfunction]
fn differential_evolution_rust(
    objective_name: &str,  // "sphere", "rosenbrock", "rastrigin"
    bounds: Vec<(f64, f64)>,
    parallel: bool,  // Now actually works!
    ...
) -> PyResult<DEResult>
```

**Benefits**:
- 10-100× speedup for built-in test functions (sphere, Rosenbrock, Rastrigin)
- Useful for benchmarking and testing
- Grid search can be parallelized (no callbacks)
- Foundation for future Rust-only mode

**Effort**: Medium (1-2 hours)

### Priority 2: Implement SHADE (Success-History Adaptive DE)

**Problem**: Current adaptive DE uses basic jDE, SHADE is state-of-the-art

**Solution**: Implement SHADE algorithm from Tanabe & Fukunaga (2013)

**Key Features**:
- Historical memory of successful parameters (F, CR)
- Weighted random selection from memory
- Better than jDE on CEC benchmarks

**Implementation Strategy**:
1. Add `SHADE` variant to `DEStrategy` enum
2. Create success history buffer (circular buffer of size H=10-100)
3. Update memory after each successful mutation
4. Sample (F, CR) from history using Cauchy/Normal distributions

**Code Outline**:
```rust
pub enum DEAdaptive {
    None,
    JDE,      // Current implementation
    SHADE,    // New: Success-history based
    LSHADE,   // Future: With linear population reduction
}

struct SHADEMemory {
    history_f: Vec<f64>,    // Successful F values
    history_cr: Vec<f64>,   // Successful CR values
    index: usize,           // Circular buffer index
    size: usize,            // Memory size H
}

impl SHADEMemory {
    fn sample_f(&self) -> f64 {
        // Cauchy distribution centered on random history entry
    }
    
    fn sample_cr(&self) -> f64 {
        // Normal distribution centered on random history entry
    }
    
    fn update(&mut self, successful_f: f64, successful_cr: f64) {
        // Add to circular buffer
    }
}
```

**Benefits**:
- State-of-the-art adaptive control
- Better than jDE empirically
- Aligns with roadmap (v0.3.0)
- Minimal API changes

**Effort**: Medium-High (2-4 hours with testing)

### Priority 3: Time-Series Integration Helpers

**Problem**: Using Polaroid + OptimizR requires manual glue code

**Solution**: Create helper functions for common time-series + optimization patterns

**Implementation Strategy**:
1. Add `timeseries_utils` module to OptimizR
2. Functions for common workflows
3. Optional Polaroid integration (via feature flag)

**Code Outline**:
```rust
// New module: src/timeseries_utils.rs

/// Prepare time-series data for HMM regime detection
pub fn prepare_for_hmm(
    prices: &[f64],
    lag_periods: &[usize],
) -> Vec<Vec<f64>> {
    // Create features: returns, lagged returns, etc.
}

/// Rolling window risk metrics
pub fn rolling_hurst_exponent(
    returns: &[f64],
    window_size: usize,
) -> Vec<f64> {
    // Compute Hurst exponent in rolling windows
}

/// Backtest parameter optimization
pub fn optimize_strategy_params<F>(
    objective_fn: F,
    param_bounds: Vec<(f64, f64)>,
    n_trials: usize,
) -> DEResult
where F: Fn(&[f64]) -> f64
{
    // Wrapper around DE with sensible defaults
}
```

**Python Bindings**:
```python
from optimizr import timeseries_utils as tsu

# Prepare features
features = tsu.prepare_for_hmm(prices, lag_periods=[1, 5, 20])

# Rolling risk metrics
rolling_hurst = tsu.rolling_hurst_exponent(returns, window_size=252)

# Strategy optimization
def my_strategy(params):
    # ... backtesting logic ...
    return sharpe_ratio

result = tsu.optimize_strategy_params(
    my_strategy,
    param_bounds=[(1, 50), (0.01, 0.5)],
    n_trials=500
)
```

**Benefits**:
- Reduces boilerplate for common use cases
- Makes integration obvious
- Encourages adoption
- Low effort, high value

**Effort**: Low-Medium (1-2 hours)

## Secondary Enhancements (Future Work)

### 4. Multi-Objective Optimization (NSGA-DE)
- **Roadmap**: v0.3.0
- **Use Case**: Portfolio optimization (maximize return, minimize risk)
- **Effort**: High (5-8 hours)

### 5. GPU Acceleration
- **Roadmap**: v0.3.0
- **Use Case**: Massive population sizes (10K-100K individuals)
- **Effort**: Very High (multi-day project)

### 6. Additional Algorithms (PSO, CMA-ES, etc.)
- **Roadmap**: v0.3.0
- **Use Case**: Algorithm portfolio for different problem types
- **Effort**: High per algorithm (3-5 hours each)

## Recommended Implementation Order

1. **Session 1 (Current)**: Time-Series Integration Helpers (1-2 hours)
   - Low effort, immediate value
   - Makes Polaroid + OptimizR integration obvious
   - Creates examples for documentation

2. **Session 2**: Enable Rust-Native Parallelization (1-2 hours)
   - Unblocks major performance gain
   - Grid search parallelization
   - Foundation for future work

3. **Session 3**: Implement SHADE (2-4 hours)
   - State-of-the-art adaptive DE
   - Aligns with roadmap
   - Publishable improvement

4. **Future**: Multi-objective, GPU, additional algorithms
   - Larger projects
   - Requires more research

## Testing Strategy

For each enhancement:
1. **Unit tests**: Algorithm correctness (sphere function, Rosenbrock)
2. **Benchmarks**: Performance comparison (before/after)
3. **Integration tests**: Polaroid + OptimizR workflows
4. **Documentation**: Usage examples, API docs

## Git Commit Strategy (per MANDATORY rules)

Each enhancement gets:
1. Feature branch: `feature/shade-algorithm` or `feature/rust-parallelization`
2. Implementation commits with tests
3. Benchmark results documented
4. Final commit: `feat(de): implement SHADE adaptive DE variant`
5. Push to origin
6. Log to historia/

## Success Metrics

1. **Performance**:
   - Rust parallelization: 10-100× speedup on multi-core
   - SHADE: 10-20% better convergence than jDE on benchmarks
   - Time-series helpers: Zero overhead (pure convenience)

2. **Usability**:
   - Integration examples in documentation
   - Clear API documentation
   - Python usage examples

3. **Completeness**:
   - All tests passing
   - Benchmarks documented
   - Changes committed to git

---

**Next Action**: Implement Priority 3 (Time-Series Integration Helpers) as it's lowest effort with immediate value for demonstrating Polaroid + OptimizR synergy.
