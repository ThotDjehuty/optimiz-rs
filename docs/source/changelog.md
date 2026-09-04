# Optimiz-rs Changelog

Product updates and releases across the Optimiz-rs optimization library, Rust crates, Python bindings (optimiz-rs on PyPI), and documentation, in reverse-chronological order.

---

## February 2026

### Optimiz-rs v1.0.0 — First Stable Release
*Released February 16, 2026 — Semantic Versioning Begins*

**📦 Distribution**
- Published to **crates.io**: `cargo add optimiz-rs`
- Published to **PyPI**: `pip install optimiz-rs` (package name: `optimiz-rs`)
- **Stable API commitment**: Semantic versioning from v1.0.0 forward
- **Production ready**: Comprehensive testing and validation

**🚀 Stable Features**
- **Differential Evolution**: 5 strategies (rand/1, best/1, current-to-best/1, rand/2, best/2) + adaptive jDE
- **Hidden Markov Models**: Baum-Welch training, Viterbi decoding, Gaussian emissions
- **MCMC Sampling**: Metropolis-Hastings, adaptive proposals, convergence diagnostics
- **Mean Field Games**: 1D MFG solver, HJB-Fokker-Planck coupling, agent population dynamics
- **Mathematical Toolkit**: Numerical differentiation, statistics, linear algebra, information theory
- **Grid Search**: Exhaustive parameter space exploration

**⚡ Performance Benchmarks**
| Algorithm | Problem | OptimizR (Rust) | Python Baseline | Speedup |
|-----------|---------|-----------------|-----------------|---------|
| Differential Evolution | Rosenbrock 10D | 0.12s | 8.9s (SciPy) | **74×** |
| HMM Training | 1000 obs, 3 states | 0.03s | 2.4s (hmmlearn) | **80×** |
| Mean Field Games | 100×100 grid | 0.4s | 45s (Pure Python) | **112×** |

**🔧 Breaking Changes from v0.3.0**
- **Cargo features**: `python-bindings` moved from default to opt-in
  - Rust-only users: No changes needed
  - Python users: No impact (maturin auto-enables)
  - Explicit Rust library users: Add `features = ["python-bindings"]` to Cargo.toml

**🐛 Bug Fixes**
- Fixed linking errors when using as Rust-only library
- Fixed PyInit__core symbol warning in maturin builds
- Resolved flate2 yanked dependency warning

**📚 Documentation**
- Complete ReadTheDocs site: https://optimiz-r.readthedocs.io
- 7 validated tutorial notebooks (HMM, MCMC, DE, Optimal Control, Real-World, Benchmarks, MFG)

---

## January 2025

### Optimiz-rs v0.3.0 — Mean Field Games & Maturin Build
*Released January 4, 2025 — Major Feature Release*

**✨ New Features**

**Mean Field Games (MFG) Framework**
- Complete 1D MFG solver with HJB backward + Fokker-Planck forward
- Fixed-point iteration for coupled equations
- Upwind finite difference schemes with Neumann boundaries
- Convergence diagnostics and stability guarantees
- **Performance**: 0.4s for 100×100 grid, 50 iterations (112× vs pure Python)
- Tutorial notebook: `mean_field_games_tutorial.ipynb` with 3D visualizations

**Maturin Build System**
- Replaced cargo with maturin for reliable cross-platform builds
- Works on macOS (fixes linker issues)
- Creates proper Python wheels for abi3 (Python ≥ 3.8)
- Editable installs with `maturin develop`
- Better integration with Python packaging ecosystem

**Python Wrapper Architecture**
- Two-layer design: Rust core (PyO3) + Python OOP wrappers
- User-friendly interfaces (scikit-learn style)
- Automatic Rust acceleration with graceful Python fallback
- Example: `HMM` class wraps `_rust_fit_hmm()` and `_rust_viterbi()`

**📚 Documentation & Validation**
- All 7 example notebooks audited and tested ✅
- New MFG tutorial notebook fully working (12/12 cells)
- Fixed `04_real_world_applications.ipynb` (removed invalid `random_state`)
- New documentation: `MFG_TUTORIAL_COMPLETE.md`, `NOTEBOOK_AUDIT_REPORT.md`, `COMPLETE_NOTEBOOK_PROOF.md`

**🐛 Bug Fixes**
- MFGConfig: Removed `ny` parameter for 1D problems (was 2D-only)
- HMM: Removed non-existent `random_state` parameter
- macOS build: Resolved via maturin migration
- Numerical stability: MFG solver handles large gradients without overflow
- Convergence reporting: Fixed misleading "converged" message

---

## December 2025

### Optimiz-rs v0.2.0 — Comprehensive Differential Evolution & Mathematical Toolkit
*Released December 10, 2025*

**🎉 Major Additions**

**Comprehensive Differential Evolution**
- 5 mutation strategies: rand/1, best/1, current-to-best/1, rand/2, best/2
- **Adaptive jDE**: Self-adapting F ∈ [0.1, 1.0] and CR ∈ [0, 1] per individual
- Convergence tracking: best fitness, mean/std, diversity metrics, early stopping
- Rich result object with history, generations, function evaluations
- **Performance**: 74-88× speedup vs pure Python across benchmark suite

**Mathematical Toolkit Module (`maths_toolkit`)**
- Numerical differentiation: gradient, hessian, jacobian
- Statistics: mean, variance, skewness, kurtosis, autocorrelation, correlation matrix
- Linear algebra: norms, normalization, trace, outer product, condition number
- Numerical integration: trapezoidal, Simpson's rule
- Interpolation: linear, 1D grid interpolation
- Special functions: sigmoid, softplus, relu, soft_threshold, bounds checking

**Optimal Control Framework**
- Generic HJB solver for continuous-time optimal control
- Regime switching systems with Markov chains
- Jump diffusion processes (Lévy, compound Poisson)
- MRSJD: Combined Markov regime switching + jump diffusion
- Finite difference schemes: upwind, value iteration, policy iteration
- Generic applications: temperature control, inventory, robot navigation, resource allocation

**🏗️ Architecture Refactoring**
- Removed legacy code: `hmm_legacy.rs`, `mcmc_legacy.rs`, `de_refactored.rs`
- Modular structure: `core`, `functional`, `maths_toolkit`, `differential_evolution`, `sparse_optimization`, `risk_metrics`, `optimal_control/`, `hmm/`, `mcmc/`, `de/`
- All algorithms now domain-agnostic (finance code moved to application layer)

**🚀 Performance**
| Problem | Dimensions | Python | Rust | Speedup |
|---------|-----------|--------|------|---------|
| Sphere | 10 | 12.3s | 0.14s | **88×** |
| Rosenbrock | 10 | 15.2s | 0.18s | **84×** |
| Rosenbrock | 20 | 62.5s | 0.71s | **88×** |
| Rastrigin | 10 | 18.7s | 0.22s | **85×** |
| Rastrigin | 20 | 72.1s | 0.84s | **86×** |
| Portfolio | 50 | 145.0s | 1.95s | **74×** |

**Memory**: 95% reduction vs NumPy/SciPy across all dimensions

**🔧 Breaking Changes**
- DE API: New parameters (strategy, adaptive, track_history)
- Result objects: Rich objects replacing simple tuples
- Module imports: Clean imports from `optimizr` and `optimizr.de`
- Removed: `de_refactored`, legacy HMM/MCMC modules

---

## Earlier Releases

### Optimiz-rs v0.1.x — Initial Development
- Core HMM implementation (Baum-Welch, Viterbi)
- MCMC sampling (Metropolis-Hastings)
- Basic differential evolution (single strategy)
- Python bindings via PyO3
- Initial benchmark infrastructure

---

## Cross-Project Integration

### Time-Series Analysis with Polarway
Optimiz-rs provides **statistical primitives** that complement Polarway's **high-performance DataFrame engine**:

| Polarway (DataFrame) | Optimiz-rs (Algorithms) |
|----------------------|------------------------|
| OHLCV resampling & rolling windows | Hurst exponent, half-life estimation |
| VWAP/TWAP calculations | Regime detection (HMM) |
| Distributed time-series storage | MCMC for Bayesian inference |
| gRPC streaming for real-time data | Differential evolution for strategy optimization |
| Hybrid storage (Parquet + DuckDB) | Mean Field Games for market dynamics |

**Integration Pattern:**
```python
import polarway as pw
from optimizr import HMM, DifferentialEvolution, mutual_information

# 1. Load data via Polarway (streaming, distributed)
client = pw.connect("localhost:50051")
df = client.scan_parquet("s3://bucket/trades/*.parquet")

# 2. Feature engineering with Polarway
features = df.with_columns([
    pw.col("returns").rolling_std(20).alias("vol_20"),
    pw.col("volume").rolling_mean(50).alias("vol_avg_50"),
])

# 3. Regime detection with Optimiz-rs
hmm = HMM(n_states=3)
regimes = hmm.fit_predict(features.select("returns").collect().to_numpy())

# 4. Strategy optimization
de = DifferentialEvolution(bounds=[(-1, 1)] * 10, strategy="currenttobest1", adaptive=True)
result = de.optimize(lambda x: -sharpe_ratio(x, features, regimes))
```

### Distributed Topological Data Analysis
Optimiz-rs v1.1+ (planned) will include **topological primitives** for distributed analysis:

- **Persistent homology** for time-series shape analysis
- **Graph spectral methods** for network topology
- **Signature methods** for path-dependent data
- **Wavelet transforms** for multi-scale analysis

Polarway's distributed computing framework (`polarway-distributed`) will provide the **execution layer** for running these algorithms at scale across multiple nodes.

---

## Migration Guides

| From Version | To Version | Guide |
|--------------|------------|-------|
| v0.3.x | v1.0.0 | [Cargo Feature Flags](#breaking-changes-from-v030) |
| v0.2.x | v0.3.0 | [Add MFG imports](RELEASE_NOTES_v0.3.0.md#api-changes) |
| v0.1.x | v0.2.0 | [DE API Migration](RELEASE_NOTES_v0.2.0.md#migration-guide) |

---

## Links

- **GitHub Releases**: https://github.com/ThotDjehuty/optimiz-r/releases
- **Documentation**: https://optimiz-r.readthedocs.io/
- **PyPI (Python)**: https://pypi.org/project/optimiz-rs/
- **crates.io (Rust)**: https://crates.io/crates/optimiz-rs
- **Issues**: https://github.com/ThotDjehuty/optimiz-r/issues
- **Discussions**: https://github.com/ThotDjehuty/optimiz-r/discussions

---

*Last updated: August 2026*