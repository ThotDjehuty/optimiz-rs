# Getting Started

This guide prepares a fresh environment, builds the Rust extension, and validates the install.

## 1. Install dependencies

```bash
# Python deps for docs and benchmarks
python -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
pip install maturin numpy
```

## 2. Build and install Optimiz-rs locally

```bash
pip install .
# or editable mode for development
maturin develop --release
```

## 3. Quick verification

```bash
python - <<'PY'
import optimizr
from optimizr import differential_evolution, HMM
print("Optimiz-rs version:", optimizr.__version__)

# Simple objective
f = lambda x: sum(v * v for v in x)
pt, val = differential_evolution(f, bounds=[(-2, 2)] * 3, maxiter=50)
print("DE ok →", round(float(val), 4))

model = HMM(n_states=2).fit([0.01, -0.02, 0.0, 0.03])
print("HMM states →", model.predict([0.01, -0.02, 0.0, 0.03]))
PY
```

## 4. Build docs locally

```bash
cd docs
make html  # or: sphinx-build -b html source build/html
open build/html/index.html
```

## 5. Troubleshooting

- If the Rust extension fails to compile, ensure `rustc --version` ≥ 1.70 and `maturin` is installed.
- On Apple Silicon, set `export MACOSX_DEPLOYMENT_TARGET=12.0` before building if you hit ABI errors.
- Delete stale builds with `rm -rf build dist target *.egg-info` then reinstall.
