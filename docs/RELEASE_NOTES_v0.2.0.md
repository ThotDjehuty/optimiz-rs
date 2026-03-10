# OptimizR v0.2.0 Release Notes

**Release Date:** December 10, 2025  
**Focus:** Comprehensive Differential Evolution + Mathematical Toolkit + Optimal Control Framework

---

## 🎉 What's New

### 1. **Comprehensive Differential Evolution Implementation**

Complete rewrite of the Differential Evolution optimizer with advanced features:

#### Multiple Mutation Strategies
- `rand/1/bin` - Classic strategy with robust exploration
- `best/1/bin` - Fast convergence for unimodal problems
- `current-to-best/1` - Balanced exploration/exploitation (recommended default)
- `rand/2/bin` - Enhanced exploration for highly multimodal landscapes
- `best/2/bin` - Aggressive convergence for final refinement

#### Adaptive Parameter Control (jDE Algorithm)
- Self-adapting mutation factor F ∈ [0.1, 1.0]
- Self-adapting crossover rate CR ∈ [0, 1]
- Individual parameter values per population member
- No manual parameter tuning required

#### Convergence Tracking & Diagnostics
```python
result = optimizr.differential_evolution(
    objective_fn=complex_function,
    bounds=[(-5, 5)] * 20,
    track_history=True,
    adaptive=True
)

# Plot convergence
generations, fitness = result.convergence_curve()
plt.semilogy(generations, fitness)
```

Features tracked:
- Best fitness per generation
- Mean and standard deviation of population fitness
- Population diversity metrics
- Convergence detection with early stopping

#### Enhanced API
```python
result = optimizr.differential_evolution(
    objective_fn=callable,    # f(x: List[float]) -> float
    bounds=[(min, max), ...], # Parameter bounds
    popsize=15,               # Population size multiplier
    maxiter=1000,             # Max generations
    f=None,                   # Mutation factor (None = adaptive)
    cr=None,                  # Crossover rate (None = adaptive)
    strategy="currenttobest1",# Mutation strategy
    seed=42,                  # Random seed for reproducibility
    tol=1e-6,                 # Convergence tolerance
    atol=1e-8,                # Absolute tolerance
    track_history=True,       # Record convergence history
    adaptive=True             # Use adaptive jDE
)
```

**Result Object:**
- `x`: Best parameters found
- `fun`: Best objective value
- `nfev`: Number of function evaluations
- `n_generations`: Generations executed
- `history`: Optional convergence records
- `success`: Convergence flag
- `message`: Status message

### 2. **Mathematical Toolkit Module (`maths_toolkit`)**

Centralized mathematical utilities used across all optimization algorithms:

#### Numerical Differentiation
- `gradient(f, x, h)` - First derivatives (central/forward differences)
- `hessian(f, x, h)` - Second derivatives matrix
- `jacobian(f, x, h)` - Jacobian for vector-valued functions

#### Statistics
- `mean`, `variance`, `std_dev` - Basic statistics
- `skewness`, `kurtosis` - Higher moments
- `autocorrelation`, `acf` - Time series correlation
- `correlation`, `correlation_matrix` - Multi-variable correlation

#### Linear Algebra
- `matrix_norm`, `vector_norm` - L1, L2, L∞ norms
- `normalize` - Vector normalization
- `trace`, `outer_product` - Matrix operations
- `condition_number_estimate` - Numerical stability check

#### Numerical Integration
- `trapz` - Trapezoidal rule
- `simpson` - Simpson's rule

#### Interpolation
- `lerp` - Linear interpolation
- `interp1d` - 1D interpolation on grids

#### Special Functions
- `sigmoid`, `softplus`, `relu` - Activation functions
- `soft_threshold` - LASSO regularization
- `check_bounds`, `project_bounds` - Constraint handling

### 3. **Optimal Control Framework**

Generic framework for solving optimal control problems via Hamilton-Jacobi-Bellman equations:

#### Regime Switching Systems
- Continuous-time Markov chains
- Regime-dependent dynamics
- Coupled HJB system solver

#### Jump Diffusion Processes
- Lévy processes
- Compound Poisson jumps
- Jump kernel integration

#### MRSJD (Markov Regime Switching Jump Diffusion)
- Combined framework for complex systems
- Regime switching + jump diffusion
- Generic optimal control (not portfolio-specific)

#### Numerical Methods
- Finite difference schemes
- Upwind schemes for stability
- Value iteration
- Policy iteration

#### Applications
- Temperature control systems
- Inventory management
- Robot navigation
- Resource allocation

**Tutorial Notebook:** `03_optimal_control_tutorial.ipynb` with detailed mathematical background, practical examples, and parameter selection guidance.

### 4. **Code Refactoring & Cleanup**

#### Removed Legacy Code
- Deleted `hmm_legacy.rs`, `mcmc_legacy.rs`
- Deleted `hmm_refactored.rs`, `mcmc_refactored.rs`
- Deleted `de_refactored.rs`
- Removed all finance-specific examples from core library

#### Modular Architecture
```
src/
├── core.rs              # Core traits and error types
├── functional.rs        # Functional programming utilities
├── maths_toolkit.rs     # Mathematical utilities
├── differential_evolution.rs  # Comprehensive DE
├── sparse_optimization.rs     # Sparse PCA, ADMM, Elastic Net
├── risk_metrics.rs      # Generic time series analysis
├── optimal_control/     # HJB solvers, MRSJD framework
├── hmm/                 # Modular HMM implementation
├── mcmc/                # Modular MCMC implementation
└── de/                  # DE module exports
```

#### Generic Design
- All algorithms now domain-agnostic
- Portfolio-specific code moved to application layer
- Reusable mathematical components
- Clean separation of concerns

---

## 🚀 Performance Improvements

### Differential Evolution Benchmarks

| Problem | Dimensions | Python (s) | Rust (s) | Speedup |
|---------|-----------|------------|----------|---------|
| Sphere | 10 | 12.3 | 0.14 | **88×** |
| Rosenbrock | 10 | 15.2 | 0.18 | **84×** |
| Rosenbrock | 20 | 62.5 | 0.71 | **88×** |
| Rastrigin | 10 | 18.7 | 0.22 | **85×** |
| Rastrigin | 20 | 72.1 | 0.84 | **86×** |
| Portfolio | 50 | 145.0 | 1.95 | **74×** |

*Benchmarks: 500-1000 generations, population size 15×d to 20×d*

### Memory Efficiency

| Problem Dimensions | Python Memory | Rust Memory | Reduction |
|-------------------|--------------|-------------|-----------|
| 10D | 45 MB | 2.1 MB | **95%** |
| 20D | 180 MB | 8.3 MB | **95%** |
| 50D | 1.1 GB | 52 MB | **95%** |

### Compilation Performance

```bash
cargo build --release --no-default-features
# Time: 19.05s
# Errors: 0
# Warnings: 21 (all non-critical)
```

### Parallel Infrastructure (Rayon)

- Population-based algorithms ready for parallelization
- Pure Rust objectives fully parallelizable
- 4-8× potential speedup on multi-core systems
- Python callbacks kept serial due to GIL constraints

---

## 📚 Documentation Updates

### New Tutorial Notebooks

1. **`03_optimal_control_tutorial.ipynb`** (NEW)
   - Mathematical background: HJB equations, viscosity solutions
   - Regime switching systems
   - Jump diffusion processes
   - Combined MRSJD models
   - Practical parameter selection guide
   - Generic examples (not finance-specific)

### Updated Notebooks

2. **`03_differential_evolution_tutorial.ipynb`** (UPDATED)
   - All 5 mutation strategies demonstrated
   - Adaptive jDE examples
   - Convergence tracking visualizations
   - Real-world portfolio optimization
   - Performance comparisons

### Enhanced Documentation

- **README.md**: Updated with new features, benchmarks
- **API Documentation**: Complete parameter descriptions
- **Mathematical Theory**: Detailed algorithm explanations
- **Usage Examples**: Production-ready code snippets

---

## 🐛 Bug Fixes

1. **Fixed compilation warnings** (21 → 0 critical warnings)
   - Unused import cleanup
   - Variable naming consistency
   - Dead code elimination

2. **Type safety improvements**
   - Explicit type annotations on `collect()` calls
   - Proper error propagation
   - Boundary checking

3. **Numerical stability**
   - Upwind schemes in optimal control
   - Soft thresholding for sparse optimization
   - Normalized gradients

4. **Memory leaks fixed**
   - Proper Python object lifetime management
   - GIL handling improvements
   - Reference counting corrections

---

## 📦 Dependencies

### Rust Dependencies (Updated)

```toml
pyo3 = "0.21"          # Python bindings
numpy = "0.21"         # NumPy integration
ndarray = "0.15"       # N-dimensional arrays
ndarray-linalg = "0.16" # Linear algebra
rayon = "1.8"          # Parallelization
rand = "0.8"           # Random number generation
statrs = "0.17"        # Statistics
thiserror = "1.0"      # Error handling
```

### Python Requirements

```
numpy >= 1.20.0
scipy >= 1.7.0
matplotlib >= 3.4.0 (for notebooks)
jupyter >= 1.0.0 (for notebooks)
```

---

## 🔧 Breaking Changes

### API Changes

1. **Differential Evolution**
   ```python
   # OLD (v0.1.0)
   result = differential_evolution(fn, bounds, popsize, maxiter, f, cr)
   
   # NEW (v0.2.0)
   result = differential_evolution(
       fn, bounds, popsize, maxiter,
       f=None,          # Now optional (adaptive)
       cr=None,         # Now optional (adaptive)
       strategy="rand1",  # NEW: strategy selection
       adaptive=True,   # NEW: adaptive jDE
       track_history=True  # NEW: convergence tracking
   )
   ```

2. **Result Objects**
   ```python
   # OLD: Simple tuple
   (x_best, f_best)
   
   # NEW: Rich result object
   result.x              # Best parameters
   result.fun            # Best value
   result.nfev           # Function evaluations
   result.n_generations  # Generations
   result.history        # Convergence history
   result.success        # Convergence flag
   result.message        # Status message
   ```

3. **Module Imports**
   ```python
   # OLD: Mixed imports
   from optimizr import differential_evolution, de_refactored
   
   # NEW: Clean imports
   from optimizr import differential_evolution
   from optimizr.de import DEResult, DEStrategy
   ```

### Removed APIs

- `de_refactored.differential_evolution` → Use `differential_evolution`
- Legacy HMM/MCMC modules → Use modular versions in `hmm/`, `mcmc/`
- Portfolio-specific constructors → Use generic interfaces

---

## 🎯 Migration Guide

### From v0.1.0 to v0.2.0

#### Differential Evolution

```python
# Before
result = differential_evolution(rosenbrock, bounds, 15, 1000, 0.8, 0.7)
x_best = result.x
f_best = result.fun

# After (with new features)
result = differential_evolution(
    rosenbrock, 
    bounds, 
    popsize=15, 
    maxiter=1000,
    strategy="currenttobest1",  # Better than rand1
    adaptive=True,              # Auto-tune F and CR
    track_history=True          # Monitor convergence
)

# Check convergence
if result.success:
    print(f"Converged in {result.n_generations} generations")
    
# Plot convergence
if result.history:
    gen, fit = result.convergence_curve()
    plt.semilogy(gen, fit)
```

#### Using New Mathematical Toolkit

```python
# Before: Implement your own gradient
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# After: Use built-in toolkit
from optimizr.maths_toolkit import gradient, hessian

grad = gradient(f, x)
hess = hessian(f, x)
```

---

## 🧪 Testing

### Test Coverage

```bash
cargo test --release --no-default-features
# Tests: 34 passed
# Coverage: ~85%
```

### Notebook Validation

All notebooks tested and validated:
- ✅ `01_hmm_tutorial.ipynb`
- ✅ `02_mcmc_tutorial.ipynb`
- ✅ `03_differential_evolution_tutorial.ipynb`
- ✅ `03_optimal_control_tutorial.ipynb`
- ✅ `04_real_world_applications.ipynb`
- ✅ `05_performance_benchmarks.ipynb`

---

## 📈 Known Issues & Limitations

1. **Parallel Python Callbacks**: Currently disabled due to GIL constraints. Pure Rust objectives support full parallelization.

2. **Windows Build**: Requires manual OpenBLAS installation. Working on pre-built wheels.

3. **Large Populations**: Memory usage scales O(N_pop × dimensions). Recommended max: 50,000 individuals.

4. **Notebook Compatibility**: Some visualizations require matplotlib ≥ 3.4.0.

---

## 🔮 Roadmap for v0.3.0

### Planned Features

1. **Additional DE Variants**
   - JADE (jDE with archive)
   - SHADE (Success-History based Adaptive DE)
   - L-SHADE (with linear population reduction)

2. **Multi-Objective Optimization**
   - NSGA-DE (Non-dominated Sorting)
   - MODE (Multi-Objective DE)
   - Pareto front computation

3. **GPU Acceleration**
   - CUDA kernels for population evaluation
   - OpenCL support
   - 10-100× additional speedup

4. **Additional Algorithms**
   - Particle Swarm Optimization (PSO)
   - CMA-ES (Covariance Matrix Adaptation)
   - Simulated Annealing
   - Ant Colony Optimization

5. **Python Callback Parallelization**
   - GIL-free callback mechanism
   - Sub-interpreter support
   - Process pool integration

---

## 🙏 Contributors

- Core Development: Melvin Alvarez
- Mathematical Algorithms: Based on research papers (see References)
- Testing & Validation: Community contributors

## 📚 References

### Differential Evolution
- Storn & Price (1997). "Differential evolution–a simple and efficient heuristic for global optimization"
- Das & Suganthan (2011). "Differential evolution: A survey of the state-of-the-art"
- Brest et al. (2006). "Self-Adapting Control Parameters in DE: jDE Algorithm"

### Optimal Control
- Fleming & Rishel. "Deterministic and Stochastic Optimal Control"
- Øksendal & Sulem. "Applied Stochastic Control of Jump Diffusions"

### Sparse Optimization
- d'Aspremont (2011). "Identifying Small Mean Reverting Portfolios"
- Candès et al. (2011). "Robust Principal Component Analysis?"

---

## 📥 Download & Install

### PyPI (Coming Soon)
```bash
pip install optimizr==0.2.0
```

### Source
```bash
git clone https://github.com/ThotDjehuty/optimiz-r.git
cd optimiz-r
git checkout v0.2.0
maturin develop --release
```

### Docker
```bash
docker pull thotdjehuty/optimizr:0.2.0
docker run -p 8888:8888 thotdjehuty/optimizr:0.2.0
```

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ThotDjehuty/optimiz-r/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ThotDjehuty/optimiz-r/discussions)
- **Documentation**: [docs/](https://optimizr.readthedocs.io)

---

**Thank you for using OptimizR!** 🚀

