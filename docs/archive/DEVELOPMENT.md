# OptimizR Development Guide

## Quick Start

### Prerequisites

- Python 3.8+
- Rust 1.70+ (install from https://rustup.rs)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/ThotDjehuty/optimiz-r.git
cd optimiz-r

# Install development dependencies
pip install -e ".[dev]"
pip install maturin

# Build Rust extension
maturin develop --release

# Run tests
pytest tests/ -v

# Run example
python examples/hmm_regime_detection.py
```

## Project Structure

```
optimiz-r/
├── src/                          # Rust source code
│   ├── lib.rs                   # Main library entry point
│   ├── hmm.rs                   # Hidden Markov Model
│   ├── mcmc.rs                  # MCMC sampling
│   ├── differential_evolution.rs # Differential Evolution
│   ├── grid_search.rs           # Grid Search
│   └── information_theory.rs    # MI and Entropy
│
├── python/optimizr/             # Python package
│   ├── __init__.py             # Package exports
│   ├── core.py                 # Core functions with fallbacks
│   └── hmm.py                  # HMM Python wrapper class
│
├── tests/                       # Python tests
│   └── test_optimizr.py        # Test suite
│
├── examples/                    # Example scripts
│   └── hmm_regime_detection.py # HMM example
│
├── docs/                        # Documentation
│   └── ...                     # API docs, theory, guides
│
├── Cargo.toml                  # Rust dependencies
├── pyproject.toml              # Python project config
├── Makefile                    # Common development tasks
└── README.md                   # Main documentation
```

## Building

### Development Build (faster, with debug symbols)
```bash
maturin develop
```

### Release Build (optimized)
```bash
maturin develop --release
```

### Build Wheel
```bash
maturin build --release --out dist/
```

## Testing

### Run all tests
```bash
make test-all
```

### Python tests only
```bash
pytest tests/ -v
```

### Rust tests only
```bash
cargo test
```

### With coverage
```bash
pytest tests/ --cov=optimizr --cov-report=html
```

## Code Quality

### Format code
```bash
make format
```

### Lint code
```bash
make lint
```

### Type checking
```bash
mypy python/optimizr/
```

### Run all checks
```bash
make check
```

## Common Commands

See all available commands:
```bash
make help
```

Common workflows:
```bash
make dev       # Setup dev environment
make build     # Build release version
make test      # Run tests
make lint      # Check code quality
make format    # Format code
make clean     # Remove build artifacts
make ci        # Run all CI checks locally
```

## Algorithm Implementation Guide

### Adding a New Algorithm

1. **Create Rust module** (`src/new_algorithm.rs`):
   ```rust
   ///! Algorithm Description
   
   use pyo3::prelude::*;
   
   #[pyfunction]
   pub fn my_algorithm(params: Vec<f64>) -> PyResult<f64> {
       // Implementation
       Ok(result)
   }
   ```

2. **Register in `src/lib.rs`**:
   ```rust
   mod new_algorithm;
   
   #[pymodule]
   fn _core(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add_function(wrap_pyfunction!(new_algorithm::my_algorithm, m)?)?;
       Ok(())
   }
   ```

3. **Add Python wrapper** (`python/optimizr/core.py`):
   ```python
   def my_algorithm(params: np.ndarray) -> float:
       if RUST_AVAILABLE:
           return _rust_my_algorithm(params.tolist())
       else:
           return _my_algorithm_python(params)
   ```

4. **Export in `__init__.py`**:
   ```python
   from optimizr.core import my_algorithm
   __all__ = [..., "my_algorithm"]
   ```

5. **Add tests** (`tests/test_optimizr.py`):
   ```python
   def test_my_algorithm():
       result = my_algorithm(np.array([1.0, 2.0]))
       assert result > 0
   ```

6. **Add documentation and examples**

## Benchmarking

Run Rust benchmarks:
```bash
cargo bench
```

Create benchmark:
```rust
// benches/benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_my_algo(c: &mut Criterion) {
    c.bench_function("my_algorithm", |b| {
        b.iter(|| {
            // Benchmark code
        });
    });
}

criterion_group!(benches, benchmark_my_algo);
criterion_main!(benches);
```

## Publishing

### Test on TestPyPI
```bash
maturin publish --repository testpypi
```

### Publish to PyPI
```bash
maturin publish
```

## Troubleshooting

### Build fails with "rustc not found"
Install Rust: https://rustup.rs

### Import error: "cannot import name '_core'"
Rebuild the extension: `maturin develop --release`

### Tests fail with "module not found"
Install in editable mode: `pip install -e .`

### Slow builds
Use development build: `maturin develop` (without --release)

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 Guide](https://pyo3.rs/)
- [Maturin Docs](https://www.maturin.rs/)
- [NumPy Docs](https://numpy.org/doc/)

## Getting Help

- GitHub Issues: Report bugs or request features
- GitHub Discussions: Ask questions, share ideas
- Email: your.email@example.com
