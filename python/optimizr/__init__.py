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

# ===== v2.0 primitives (lazy via __getattr__, but eagerly bound when possible) =====
try:
    from optimizr._core import (
        # Volterra / fractional
        solve_fractional_ode,
        solve_volterra,
        geometric_grid_lift,
        fourier_invert,
        mittag_leffler_py,
        # BSDE
        linear_bsde_constant_coeffs,
        # Mean-field / agent-based
        mean_reverting_mckean_vlasov,
        consensus_dynamics,
        # Risk measures
        historical_var_py,
        parametric_var_py,
        cvar_value_py,
        minimize_cvar_py,
        # PDE
        fokker_planck_constant,
        hjb_quadratic_2d,
        poisson_2d_zero_boundary,
        # Stochastic control
        optimal_switching_dp,
        pontryagin_lqr,
        two_sided_intensities,
        quadratic_impact_control_py,
        # Topology
        vietoris_rips_filtration,
        persistent_homology,
        bottleneck_distance,
        # Graph
        combinatorial_laplacian_py,
        normalised_laplacian_py,
        random_walk_laplacian_py,
        spectral_cluster_py,
        # Signatures
        path_signature,
        path_log_signature,
        random_signature,
        signature_kernel,
        shuffle_product,
        concatenate_signatures,
        # Inference / optimisation
        robust_drift,
        estimate_hurst,
        scale_dependent_hurst,
        f_alpha_lambda_py,
        mmd_gaussian,
        # Point processes
        simulate_hawkes,
        simulate_bivariate_hawkes,
        simulate_fbm,
        simulate_mixed_fbm,
        # Kalman / smoothing
        LinearKalmanFilter,
        UnscentedKalmanFilter,
        RTSSmoother,
        FilterResult,
        SmootherResult,
        KalmanState,
    )
except (ImportError, AttributeError):
    pass

__version__ = "2.0.0"
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
    # ===== v2.0 primitives =====
    "solve_fractional_ode",
    "solve_volterra",
    "geometric_grid_lift",
    "fourier_invert",
    "mittag_leffler_py",
    "linear_bsde_constant_coeffs",
    "mean_reverting_mckean_vlasov",
    "consensus_dynamics",
    "historical_var_py",
    "parametric_var_py",
    "cvar_value_py",
    "minimize_cvar_py",
    "fokker_planck_constant",
    "hjb_quadratic_2d",
    "poisson_2d_zero_boundary",
    "optimal_switching_dp",
    "pontryagin_lqr",
    "two_sided_intensities",
    "quadratic_impact_control_py",
    "vietoris_rips_filtration",
    "persistent_homology",
    "bottleneck_distance",
    "combinatorial_laplacian_py",
    "normalised_laplacian_py",
    "random_walk_laplacian_py",
    "spectral_cluster_py",
    "path_signature",
    "path_log_signature",
    "random_signature",
    "signature_kernel",
    "shuffle_product",
    "concatenate_signatures",
    "robust_drift",
    "estimate_hurst",
    "scale_dependent_hurst",
    "f_alpha_lambda_py",
    "mmd_gaussian",
    "simulate_hawkes",
    "simulate_bivariate_hawkes",
    "simulate_fbm",
    "simulate_mixed_fbm",
    "LinearKalmanFilter",
    "UnscentedKalmanFilter",
    "RTSSmoother",
]


def __getattr__(name):
    """Transparent fallback: forward any unresolved top-level attribute access
    to the compiled `_core` extension.  This keeps `from optimizr import X`
    working for every Rust-backed function (v1.x and v2.0 primitives) without
    having to enumerate the full list above."""
    try:
        from optimizr import _core as _ext
    except ImportError as exc:  # pragma: no cover
        raise AttributeError(
            f"module 'optimizr' has no attribute {name!r} "
            f"(_core extension is not built: {exc})"
        ) from exc
    if hasattr(_ext, name):
        return getattr(_ext, name)
    raise AttributeError(f"module 'optimizr' has no attribute {name!r}")
