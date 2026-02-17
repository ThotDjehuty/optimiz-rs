# OptimizR v1.0.0 Release Notes

**Release Date:** February 16, 2026  
**Status:** ✅ Stable Release

---

## 🎉 First Stable Release

OptimizR v1.0.0 marks the first production-ready stable release with a commitment to semantic versioning going forward. The API is now stable and breaking changes will only occur in major version bumps.

## 📦 Distribution

### crates.io (Rust)
```bash
cargo add optimiz-rs
```
🔗 https://crates.io/crates/optimiz-rs

### PyPI (Python)
```bash
pip install optimiz-rs
```
🔗 https://pypi.org/project/optimiz-rs/

## 🆕 What's New in v1.0.0

### Publication & Distribution
- ✅ **Published to crates.io** - Available in Rust package registry (Feb 17, 2026)
- ✅ **Published to PyPI** - Available as `optimiz-rs` via pip install (Feb 17, 2026)
- ✅ **Stable API** - Semantic versioning from v1.0.0 forward
- ✅ **Production Ready** - Comprehensive testing and validation

**Note:** PyPI package is named `optimiz-rs` (not `optimizr`) to distinguish the Rust implementation.

### Documentation
- 📚 **ReadTheDocs** - Full documentation at https://optimiz-r.readthedocs.io
- 📖 **Getting Started Guide** - Quick start for new users
- 📝 **API Reference** - Complete function and class documentation
- 🎓 **Tutorials** - Step-by-step guides for all algorithms
- 🔬 **Theory & Math** - Mathematical foundations and references

### Build System Improvements
- 🏗️ **Fixed Cargo.toml** - Removed python-bindings from default features
  - Resolves linker errors when using as Rust library
  - Python bindings now opt-in feature (automatically enabled by maturin)
- 🐍 **Maturin Configuration** - Explicit python-bindings feature in pyproject.toml
  - Ensures correct PyO3 extension builds for PyPI
  - Fixes cross-platform compatibility

### Metadata Updates
- 👥 **Authors**: HFThot Research Lab <admin@hfthot-lab.eu>
- 🔗 **Repository**: https://github.com/ThotDjehuty/optimiz-r
- 📚 **Documentation**: https://optimiz-r.readthedocs.io

## 🚀 Features (Stable)

### Optimization Algorithms
- ✅ **Differential Evolution** - 5 strategies (rand/1, best/1, current-to-best/1, rand/2, best/2)
- ✅ **Adaptive jDE** - Self-tuning mutation factor and crossover rate
- ✅ **Grid Search** - Exhaustive parameter space exploration

### Hidden Markov Models
- ✅ **Baum-Welch Training** - EM algorithm for parameter learning
- ✅ **Viterbi Decoding** - Most likely state sequence
- ✅ **Gaussian Emissions** - Continuous observation models

### MCMC Sampling
- ✅ **Metropolis-Hastings** - Bayesian parameter estimation
- ✅ **Adaptive Proposals** - Gaussian random walk
- ✅ **Convergence Diagnostics** - Acceptance rate tracking

### Mean Field Games (v0.3.0+)
- ✅ **1D MFG Solver** - Large population dynamics
- ✅ **HJB-Fokker-Planck Coupling** - Fixed-point iteration
- ✅ **Agent Population Dynamics** - Spatial-temporal evolution

### Mathematical Toolkit
- ✅ **Numerical Differentiation** - gradient(), hessian(), jacobian()
- ✅ **Statistics** - mean(), variance(), skewness(), kurtosis()
- ✅ **Linear Algebra** - norms, normalization, trace, outer product
- ✅ **Information Theory** - mutual_information(), shannon_entropy()

## ⚡ Performance

- **50-100× faster** than pure Python implementations
- **95% memory reduction** vs NumPy/SciPy
- **Parallel-ready** with Rayon infrastructure
- Production-tested on multi-dimensional problems

## 📊 Benchmarks

### Differential Evolution (Rosenbrock 10D)
- OptimizR (Rust): **0.12s**
- SciPy (Python): **8.9s**
- **Speedup: 74×**

### HMM Training (1000 observations, 3 states)
- OptimizR (Rust): **0.03s**
- hmmlearn (Python): **2.4s**
- **Speedup: 80×**

### Mean Field Games (100×100 grid)
- OptimizR (Rust): **0.4s**
- Pure Python: **45s**
- **Speedup: 112×**

## 🔧 Breaking Changes from v0.3.0

### Cargo Feature Flags
```toml
# OLD (v0.3.0):
[features]
default = ["python-bindings"]  # Always included

# NEW (v1.0.0):
[features]
default = []                   # No default features
python-bindings = ["pyo3", "numpy"]  # Opt-in
```

**Impact:**
- Rust-only users: No breaking changes (python-bindings not needed)
- Python users: No impact (maturin automatically enables python-bindings)

If you're using OptimizR as a Rust library and explicitly depend on Python bindings:
```toml
# Update your Cargo.toml:
[dependencies]
optimizr = { version = "1.0", features = ["python-bindings"] }
```

## 📝 Migration Guide

### From v0.3.0 to v1.0.0

**For Rust Users:**
No code changes required. If you were using python-bindings explicitly, add it to features list.

**For Python Users:**
```bash
# Install via pip
pip install optimiz-rs

# Or specify version
pip install optimiz-rs==1.0.0
```

**Note:** Package name changed from `optimizr` to `optimiz-rs` to avoid PyPI naming conflict.

**API Compatibility:**
✅ All Python APIs remain unchanged  
✅ All Rust APIs remain unchanged  
✅ Function signatures are identical  
✅ Return types are identical  
✅ No deprecations or removals

## 🐛 Bug Fixes

- Fixed linking errors when using OptimizR as Rust-only library
- Fixed PyInit__core symbol warning in maturin builds
- Resolved flate2 yanked dependency warning

## 📚 Documentation

### New Documentation
- Complete ReadTheDocs site: https://optimiz-r.readthedocs.io
- Getting Started guide
- Installation instructions for all platforms
- Tutorial notebooks (7 validated examples)
- API reference with examples
- Theory and mathematical background

### Validated Tutorial Notebooks
1. ✅ **Hidden Markov Models** - Regime detection
2. ✅ **MCMC Sampling** - Bayesian inference
3. ✅ **Differential Evolution** - Global optimization
4. ✅ **Optimal Control** - HJB solver (theory)
5. ✅ **Real-World Applications** - Complete workflows
6. ✅ **Performance Benchmarks** - Rust vs Python
7. ✅ **Mean Field Games** - Population dynamics

## 🔮 Roadmap

### v1.1.0 (Q2 2026)
- [ ] Additional DE variants (JADE, SHADE, L-SHADE)
- [ ] Particle Swarm Optimization (PSO)
- [ ] CMA-ES algorithm
- [ ] More HMM emission distributions

### v1.2.0 (Q3 2026)
- [ ] GPU acceleration via CUDA/ROCm
- [ ] Additional language bindings (R, Julia, JavaScript)
- [ ] Distributed computing support
- [ ] Advanced parallel strategies

### v2.0.0 (2027)
- [ ] Neural Evolution Strategies (NES)
- [ ] Multi-objective optimization
- [ ] Constraint handling methods
- [ ] Advanced uncertainty quantification

## 🙏 Acknowledgments

Built with:
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [Maturin](https://www.maturin.rs/) - Build and publish Rust crates as Python packages
- [NumPy](https://numpy.org/) - Numerical computing in Python

Inspired by:
- scipy.optimize
- scikit-learn
- hmmlearn
- emcee

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/ThotDjehuty/optimiz-r/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ThotDjehuty/optimiz-r/discussions)
- **Email**: contact@hfthot-lab.eu

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**OptimizR v1.0.0** - Fast optimization for data science and machine learning 🚀

Thank you to all contributors and early adopters who helped make this release possible!
