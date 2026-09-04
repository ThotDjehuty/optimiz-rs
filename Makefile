.PHONY: help build install test lint format clean benchmark docs

help:  ## Show this help message
	@echo "OptimizR - Makefile commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build:  ## Build Rust extension in release mode
	@echo "Building Rust extension..."
	maturin develop --release

build-dev:  ## Build Rust extension in debug mode
	@echo "Building Rust extension (debug)..."
	maturin develop

install:  ## Install package in editable mode with dev dependencies
	@echo "Installing optimizr..."
	pip install -e ".[dev]"

test:  ## Run Python tests
	@echo "Running tests..."
	pytest tests/ -v

test-rust:  ## Run Rust tests
	@echo "Running Rust tests..."
	cargo test

test-all:  ## Run all tests (Python + Rust)
	@make test-rust
	@make test

test-cov:  ## Run tests with coverage
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=optimizr --cov-report=html --cov-report=term

lint:  ## Run all linters
	@echo "Linting Python..."
	ruff check python/
	@echo "Linting Rust..."
	cargo clippy -- -D warnings

lint-fix:  ## Fix auto-fixable lint issues
	@echo "Fixing Python lint issues..."
	ruff check --fix python/
	@echo "Fixing Rust lint issues..."
	cargo clippy --fix --allow-dirty

format:  ## Format code
	@echo "Formatting Python..."
	black python/
	@echo "Formatting Rust..."
	cargo fmt

format-check:  ## Check code formatting without modifying
	@echo "Checking Python formatting..."
	black --check python/
	@echo "Checking Rust formatting..."
	cargo fmt --check

typecheck:  ## Run type checking
	@echo "Type checking..."
	mypy python/optimizr/ --ignore-missing-imports

benchmark:  ## Run Rust benchmarks
	@echo "Running benchmarks..."
	cargo bench

clean:  ## Clean build artifacts
	@echo "Cleaning..."
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf python/optimizr.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.dylib" -delete

wheel:  ## Build wheel distribution
	@echo "Building wheel..."
	maturin build --release --out dist/

publish-test:  ## Publish to TestPyPI
	@echo "Publishing to TestPyPI..."
	maturin publish --repository testpypi

publish:  ## Publish to PyPI
	@echo "Publishing to PyPI..."
	maturin publish

docs:  ## Build documentation (HTML)
	@echo "Building documentation..."
	sphinx-build -b html docs/source docs/build/html

example:  ## Run example script
	@echo "Running HMM example..."
	python examples/hmm_regime_detection.py

check:  ## Run all checks (format, lint, typecheck, test)
	@make format-check
	@make lint
	@make typecheck
	@make test-all

dev:  ## Setup development environment
	@echo "Setting up development environment..."
	pip install -e ".[dev]"
	pip install maturin
	@make build-dev
	@echo "✓ Development environment ready!"

ci:  ## Run CI checks locally
	@echo "Running CI checks..."
	@make format-check
	@make lint
	@make typecheck
	@make test-all
	@echo "✓ All CI checks passed!"

# ============================================================================
# Documentation Commands
# ============================================================================

.PHONY: docs-install
docs-install:  ## Install documentation dependencies
	@echo "📚 Installing documentation dependencies..."
	@pip install -q -r docs/requirements.txt
	@pip install -q sphinx-copybutton sphinxcontrib-mermaid furo
	@echo "✅ Documentation dependencies installed!"

.PHONY: docs-serve
docs-serve: docs-install  ## Serve documentation locally with live reload
	@echo "🚀 Starting documentation server at http://localhost:8000"
	@cd docs && sphinx-autobuild source build/html --host 0.0.0.0 --port 8000 --watch ../python --open-browser

.PHONY: docs-build
docs-build: docs-install  ## Build documentation for production
	@echo "🔨 Building documentation..."
	@cd docs && sphinx-build -b html source build/html --keep-going
	@echo "✅ Documentation built in docs/build/html"

.PHONY: docs-build-pdf
docs-build-pdf: docs-install  ## Build PDF documentation
	@echo "🔨 Building PDF documentation..."
	@cd docs && sphinx-build -b latex source build/latex
	@cd docs/build/latex && make
	@echo "✅ PDF built in docs/build/latex/optimiz-rs.pdf"

.PHONY: docs-deploy
docs-deploy: docs-build  ## Deploy documentation to ReadTheDocs (triggered via webhook)
	@echo "📤 Documentation ready for ReadTheDocs deployment"
	@echo "   Push to main branch to trigger automatic build on ReadTheDocs"

.PHONY: docs-validate
docs-validate: docs-install  ## Validate documentation (check for warnings/errors)
	@echo "🔍 Validating documentation..."
	@cd docs && sphinx-build -b html source build/html -W --keep-going -q
	@echo "✅ Documentation validation passed!"

.PHONY: docs-check-links
docs-check-links: docs-build  ## Check for broken links in documentation
	@echo "🔗 Checking for broken links..."
	@cd docs && sphinx-build -b linkcheck source build/html
	@echo "✅ Link check complete (see docs/build/html/output.txt)"

.PHONY: docs-preview
docs-preview: docs-build  ## Open built documentation in browser
	@echo "🌐 Opening documentation preview..."
	@open docs/build/html/index.html 2>/dev/null || xdg-open docs/build/html/index.html 2>/dev/null || echo "Open docs/build/html/index.html manually"

.PHONY: docs-clean
docs-clean:  ## Clean documentation build artifacts
	@echo "🧹 Cleaning documentation build..."
	@rm -rf docs/build
	@echo "✅ Documentation cleaned"
