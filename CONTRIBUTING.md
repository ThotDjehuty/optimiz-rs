# Contributing to OptimizR

Thank you for your interest in contributing to OptimizR! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThotDjehuty/optimiz-r.git
   cd optimiz-r
   ```

2. **Install Rust** (if not already installed)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   pip install maturin
   ```

5. **Build Rust extension**
   ```bash
   maturin develop
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes in appropriate files:
   - Rust code: `src/*.rs`
   - Python code: `python/optimizr/*.py`
   - Tests: `tests/test_*.py`
   - Documentation: `docs/*.md`, `README.md`

3. Rebuild after Rust changes:
   ```bash
   maturin develop
   ```

### Code Quality

1. **Format code**
   ```bash
   # Python
   black python/
   
   # Rust
   cargo fmt
   ```

2. **Lint code**
   ```bash
   # Python
   ruff check python/
   
   # Rust
   cargo clippy -- -D warnings
   ```

3. **Type checking**
   ```bash
   mypy python/optimizr/
   ```

### Testing

1. **Run Python tests**
   ```bash
   pytest tests/ -v
   ```

2. **Run Rust tests**
   ```bash
   cargo test
   ```

3. **Run with coverage**
   ```bash
   pytest tests/ --cov=optimizr --cov-report=html
   ```

4. **Run benchmarks**
   ```bash
   cargo bench
   ```

## Contribution Guidelines

### Code Style

- **Python**: Follow PEP 8, use type hints, docstrings in NumPy style
- **Rust**: Follow Rust conventions, document public APIs with `///` comments
- **Line length**: 100 characters for both Python and Rust
- **Imports**: Group and sort imports (use `isort` for Python)

### Documentation

- Document all public APIs with examples
- Update README.md if adding major features
- Add docstrings to Python functions
- Add doc comments (`///`) to Rust functions
- Include mathematical background for algorithms

### Commit Messages

Use conventional commit format:
```
type(scope): brief description

Detailed explanation if needed.

Fixes #issue_number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(hmm): add support for multivariate emissions
fix(mcmc): correct acceptance probability calculation
docs(readme): update installation instructions
```

### Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Add tests for new functionality
   - Update documentation
   - Run code formatters and linters
   - Squash minor commits if appropriate

2. **PR Description should include:**
   - What changes were made and why
   - Link to related issues
   - Any breaking changes
   - Testing performed

3. **Review process:**
   - Maintainers will review within 1-2 weeks
   - Address review comments
   - Once approved, maintainer will merge

## Areas for Contribution

### High Priority

- Additional optimization algorithms (PSO, CMA-ES, Simulated Annealing)
- More HMM variants (discrete emissions, semi-Markov, etc.)
- GPU acceleration via CUDA
- Improved documentation and examples
- Performance benchmarks

### Medium Priority

- Additional probability distributions
- Parallel execution support
- More information theory metrics
- Visualization utilities
- R language bindings

### Documentation

- Tutorial notebooks
- API reference improvements
- Algorithm explanations
- Performance comparisons
- Use case examples

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Email maintainers for sensitive matters

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for making OptimizR better! 🚀
