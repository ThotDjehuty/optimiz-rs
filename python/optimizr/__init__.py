"""
OptimizR - High-Performance Optimization Algorithms
===================================================

Fast, reliable implementations of advanced optimization and statistical
inference algorithms with Rust acceleration and pure Python fallbacks.

.. moduleauthor:: OptimizR Contributors

"""

from optimizr.hmm import HMM
from optimizr.core import (
    mcmc_sample,
    differential_evolution,
    parallel_differential_evolution_rust,
    grid_search,
    mutual_information,
    shannon_entropy,
    sparse_pca_py,
    box_tao_decomposition_py,
    elastic_net_py,
    hurst_exponent_py,
    compute_risk_metrics_py,
    estimate_half_life_py,
    bootstrap_returns_py,
    # Time-series utilities
    prepare_for_hmm_py,
    rolling_hurst_exponent_py,
    rolling_half_life_py,
    return_statistics_py,
    create_lagged_features_py,
    rolling_correlation_py,
    # Benchmark functions
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
    Griewank,
)

# Try to import maths_toolkit and mean_field from Rust backend
try:
    from optimizr import _core
    maths_toolkit = _core
    MFGConfig = _core.MFGConfigPy
    solve_mfg_1d_rust = _core.solve_mfg_1d_rust
except (ImportError, AttributeError):
    maths_toolkit = None
    MFGConfig = None
    solve_mfg_1d_rust = None

# Portfolio Optimization (CARA, Mean-Variance, ERC)
try:
    from optimizr._core import (
        cara_optimal_weights,
        mean_variance_optimal_weights,
        min_variance_weights,
        erc_weights,
    )
except (ImportError, AttributeError):
    cara_optimal_weights = None
    mean_variance_optimal_weights = None
    min_variance_weights = None
    erc_weights = None

__version__ = "0.2.0"
__all__ = [
    "HMM",
    "mcmc_sample",
    "differential_evolution",
    "parallel_differential_evolution_rust",
    "grid_search",
    "mutual_information",
    "shannon_entropy",
    "sparse_pca_py",
    "box_tao_decomposition_py",
    "elastic_net_py",
    "hurst_exponent_py",
    "compute_risk_metrics_py",
    "estimate_half_life_py",
    "bootstrap_returns_py",
    # Time-series utilities
    "prepare_for_hmm_py",
    "rolling_hurst_exponent_py",
    "rolling_half_life_py",
    "return_statistics_py",
    "create_lagged_features_py",
    "rolling_correlation_py",
    # Benchmark functions
    "Sphere",
    "Rosenbrock",
    "Rastrigin",
    "Ackley",
    "Griewank",
    "maths_toolkit",
    # Mean Field Games
    "MFGConfig",
    "solve_mfg_1d_rust",
    # Portfolio Optimization
    "cara_optimal_weights",
    "mean_variance_optimal_weights",
    "min_variance_weights",
    "erc_weights",
]
