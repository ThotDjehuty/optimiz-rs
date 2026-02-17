# Installation Guide

## Requirements

- Python 3.8 or higher
- Rust 1.70 or higher (for building from source)
- pip

## Install from PyPI

**Coming soon**: Optimiz-rs will be available on PyPI.

```bash
pip install optimizr
```

## Install from Source

### Step 1: Clone Repository

```bash
git clone https://github.com/ThotDjehuty/optimiz-r.git
cd optimiz-r
```

### Step 2: Install Maturin

[Maturin](https://github.com/PyO3/maturin) is required to build Rust-Python bindings:

```bash
pip install maturin
```

### Step 3: Build and Install

**Development mode** (editable install, useful for development):

```bash
maturin develop --release
```

**Production install** (creates wheel and installs):

```bash
maturin build --release
pip install target/wheels/optimizr-*.whl
```

### Step 4: Verify Installation

```python
import optimizr
print(optimizr.__version__)  # Should print "0.3.0"
```

## Platform-Specific Notes

### macOS

If you encounter build errors on macOS:

1. Ensure Xcode Command Line Tools are installed:
   ```bash
   xcode-select --install
   ```

2. Install Rust via rustup:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

### Windows

1. Install Visual Studio Build Tools (2019 or later)
2. Install Rust via [rustup-init.exe](https://rustup.rs/)
3. Follow standard installation steps

### Linux

Requires GCC or Clang:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# If you see OpenBLAS link errors during wheels/docs build
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install gcc gcc-c++
```

## Troubleshooting

**Issue**: `maturin: command not found`

**Solution**: Ensure pip bin directory is in PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"  # Linux/macOS
```

**Issue**: Rust compiler errors

**Solution**: Update Rust to latest stable:
```bash
rustup update stable
```

**Issue**: ImportError when importing optimizr

**Solution**: Rebuild with correct Python version:
```bash
maturin develop --release -i python3.10  # Replace with your Python version
```

**Issue**: BLAS/LAPACK linkage errors on Linux

**Solution**: Install OpenBLAS headers (see Linux section above) and rebuild with `maturin develop --release`.
