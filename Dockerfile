# OptimizR Development Container
# Multi-stage build for efficient image size

FROM rust:1.75-slim as rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust source and build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Final stage with Python
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    libopenblas-dev \
    gfortran \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for maturin)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir maturin numpy scipy matplotlib pytest jupyter notebook ipykernel pandas seaborn

# Build the wheel and install it (without needing virtualenv)
RUN maturin build --release --out dist && \
    pip install --no-cache-dir dist/*.whl

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
