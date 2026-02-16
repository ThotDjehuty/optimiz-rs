"""
Core optimization functions with Rust acceleration
"""

import warnings
from typing import Callable, List, Tuple, Optional
import numpy as np

# Try to import Rust backend
try:
    from optimizr._core import (
        mcmc_sample as _rust_mcmc_sample,
        differential_evolution as _rust_differential_evolution,
        parallel_differential_evolution_rust,
        grid_search as _rust_grid_search,
        mutual_information as _rust_mutual_information,
        shannon_entropy as _rust_shannon_entropy,
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
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    warnings.warn(
        "Rust backend not available. Using pure Python fallbacks. "
        "Install with 'pip install optimizr' to enable Rust acceleration.",
        RuntimeWarning
    )


def mcmc_sample(
    log_likelihood_fn: Callable[[List[float]], float],  # Updated: closure captures data
    initial_params: np.ndarray,
    param_bounds: List[Tuple[float, float]],
    n_samples: int = 10000,
    burn_in: int = 1000,
    proposal_std: float = 0.1,
    data: Optional[np.ndarray] = None,  # Deprecated: use closure instead
) -> np.ndarray:
    """
    MCMC Metropolis-Hastings sampler.
    
    Generates samples from a target distribution using the Metropolis-Hastings
    algorithm with Gaussian random walk proposals.
    
    Parameters
    ----------
    log_likelihood_fn : callable
        Function that computes log P(data | params). Should accept
        (params: list) and return float. Data should be captured in closure.
    data : np.ndarray, optional (deprecated)
        This parameter is deprecated and ignored. Capture data in the
        log_likelihood_fn closure instead.
    initial_params : np.ndarray
        Starting parameter values
    param_bounds : list of (float, float)
        [(min, max), ...] bounds for each parameter
    n_samples : int, default=10000
        Number of samples to generate (after burn-in)
    burn_in : int, default=1000
        Number of initial samples to discard
    proposal_std : float, default=0.1
        Standard deviation of Gaussian proposals
        
    Returns
    -------
    samples : np.ndarray
        Array of shape (n_samples, n_params) with parameter samples
        
    Examples
    --------
    >>> def log_likelihood(params, data):
    ...     mu, sigma = params
    ...     residuals = (data - mu) / sigma
    ...     return -0.5 * np.sum(residuals**2) - len(data) * np.log(sigma)
    >>> data = np.random.randn(100) + 2.0
    >>> samples = mcmc_sample(
    ...     log_likelihood_fn=log_likelihood,
    ...     data=data,
    ...     initial_params=np.array([0.0, 1.0]),
    ...     param_bounds=[(-10, 10), (0.1, 10)],
    ...     n_samples=10000,
    ...     burn_in=1000
    ... )
    >>> print(f"Posterior mean: {np.mean(samples[:, 0]):.2f}")
    """
    if RUST_AVAILABLE:
        # Convert to lists if numpy arrays
        # Note: data is now captured in log_likelihood_fn closure
        params_list = initial_params.tolist() if hasattr(initial_params, 'tolist') else list(initial_params)
        
        # Rust function uses different parameter names
        samples = _rust_mcmc_sample(
            log_likelihood_fn=log_likelihood_fn,
            initial_state=params_list,  # Rust expects 'initial_state'
            n_samples=n_samples,
            step_size=proposal_std,  # Rust expects 'step_size'
            burn_in=burn_in,
        )
        return np.array(samples)
    else:
        # Pure Python fallback
        return _mcmc_sample_python(
            log_likelihood_fn, data, initial_params, param_bounds,
            n_samples, burn_in, proposal_std
        )


def differential_evolution(
    objective_fn: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    popsize: int = 15,
    maxiter: int = 1000,
    f: Optional[float] = None,
    cr: Optional[float] = None,
    strategy: str = "rand1",
    seed: Optional[int] = None,
    tol: float = 1e-6,
    atol: float = 1e-8,
    track_history: bool = False,
    parallel: bool = False,
    adaptive: bool = False,
    constraint_penalty: float = 1000.0,
) -> Tuple[np.ndarray, float]:
    """
    Differential Evolution global optimizer.
    
    Population-based stochastic optimization effective for non-convex,
    multimodal objective functions.
    
    Parameters
    ----------
    objective_fn : callable
        Function to minimize: f(x) -> float where x is np.ndarray
    bounds : list of (float, float)
        [(min, max), ...] bounds for each parameter
    popsize : int, default=15
        Population size multiplier (total size = popsize × n_params)
    maxiter : int, default=1000
        Maximum number of generations
    f : float, optional
        Mutation factor (typically 0.5-2.0). If None, uses 0.8
    cr : float, optional
        Crossover probability (typically 0.1-0.9). If None, uses 0.7
    strategy : str, default="rand1"
        Mutation strategy: "rand1", "best1", "currenttobest1", "rand2", "best2"
    seed : int, optional
        Random seed for reproducibility
    tol : float, default=1e-6
        Convergence tolerance for function value changes
    atol : float, default=1e-8
        Absolute convergence tolerance
    track_history : bool, default=False
        Whether to track convergence history
    parallel : bool, default=False
        Whether to use parallel evaluation (not supported for Python callbacks)
    adaptive : bool, default=False
        Whether to use adaptive jDE parameter control
    constraint_penalty : float, default=1000.0
        Penalty for constraint violations
        
    Returns
    -------
    x : np.ndarray
        Best parameters found
    fun : float
        Best objective value (minimum)
        
    Examples
    --------
    >>> def rosenbrock(x):
    ...     return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 
    ...                for i in range(len(x)-1))
    >>> result = differential_evolution(
    ...     objective_fn=rosenbrock,
    ...     bounds=[(-5, 5)] * 10,
    ...     strategy="best1",
    ...     adaptive=True,
    ...     maxiter=1000
    ... )
    >>> print(f"Minimum: {result[1]:.6f} at {result[0]}")
    """
    if RUST_AVAILABLE:
        result = _rust_differential_evolution(
            objective_fn=objective_fn,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            f=f,
            cr=cr,
            strategy=strategy,
            seed=seed,
            tol=tol,
            atol=atol,
            track_history=track_history,
            parallel=parallel,
            adaptive=adaptive,
            constraint_penalty=constraint_penalty,
        )
        return np.array(result.x), result.fun
    else:
        # Pure Python fallback (scipy)
        try:
            from scipy.optimize import differential_evolution as scipy_de
            mutation = f if f is not None else 0.8
            recombination = cr if cr is not None else 0.7
            result = scipy_de(
                objective_fn, 
                bounds=bounds, 
                maxiter=maxiter,
                popsize=popsize, 
                mutation=mutation, 
                recombination=recombination,
                seed=seed,
                tol=tol,
                atol=atol
            )
            return result.x, result.fun
        except ImportError:
            raise ImportError(
                "Rust backend not available and scipy not installed. "
                "Install scipy or build OptimizR with Rust support."
            )


def grid_search(
    objective_fn: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    n_points: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Grid search optimizer.
    
    Exhaustively evaluates objective function at all points on a regular grid.
    
    Parameters
    ----------
    objective_fn : callable
        Function to maximize: f(x) -> float where x is np.ndarray
    bounds : list of (float, float)
        [(min, max), ...] bounds for each parameter
    n_points : int, default=10
        Number of grid points per dimension
        
    Returns
    -------
    x : np.ndarray
        Best parameters found
    fun : float
        Best objective value (maximum)
        
    Examples
    --------
    >>> def objective(x):
    ...     return -(x[0]**2 + x[1]**2)  # Peak at (0, 0)
    >>> result = grid_search(
    ...     objective_fn=objective,
    ...     bounds=[(-5, 5), (-5, 5)],
    ...     n_points=50
    ... )
    >>> print(f"Maximum: {result[1]:.6f} at {result[0]}")
    """
    if RUST_AVAILABLE:
        result = _rust_grid_search(
            objective_fn=objective_fn,
            bounds=bounds,
            n_points=n_points,
        )
        return np.array(result.x), result.fun
    else:
        # Pure Python fallback
        return _grid_search_python(objective_fn, bounds, n_points)


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute mutual information between two variables.
    
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Measures how much knowing one variable reduces uncertainty about the other.
    
    Parameters
    ----------
    x : np.ndarray
        Sample values from first variable
    y : np.ndarray
        Sample values from second variable (must be same length as x)
    n_bins : int, default=10
        Number of bins for histogram estimation
        
    Returns
    -------
    mi : float
        Mutual information in nats (multiply by 1/ln(2) for bits)
        
    Examples
    --------
    >>> x = np.random.randn(10000)
    >>> y = 2 * x + np.random.randn(10000) * 0.5
    >>> mi = mutual_information(x, y, n_bins=20)
    >>> print(f"MI: {mi:.4f} nats")
    """
    if RUST_AVAILABLE:
        return _rust_mutual_information(x.tolist(), y.tolist(), n_bins=n_bins)
    else:
        # Pure Python fallback
        return _mutual_information_python(x, y, n_bins)


def shannon_entropy(
    x: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Shannon entropy of a variable.
    
    H(X) = -Σ p(x) log(p(x))
    
    Quantifies the uncertainty/information content of a random variable.
    
    Parameters
    ----------
    x : np.ndarray
        Sample values from the variable
    n_bins : int, default=10
        Number of bins for histogram estimation
        
    Returns
    -------
    entropy : float
        Shannon entropy in nats (multiply by 1/ln(2) for bits)
        
    Examples
    --------
    >>> x_uniform = np.random.uniform(0, 1, 10000)
    >>> h_uniform = shannon_entropy(x_uniform, n_bins=20)
    >>> x_peaked = np.random.normal(0, 0.1, 10000)
    >>> h_peaked = shannon_entropy(x_peaked, n_bins=20)
    >>> print(f"Uniform: {h_uniform:.4f}, Peaked: {h_peaked:.4f}")
    """
    if RUST_AVAILABLE:
        return _rust_shannon_entropy(x.tolist(), n_bins=n_bins)
    else:
        # Pure Python fallback
        return _shannon_entropy_python(x, n_bins)


# Pure Python fallback implementations
def _mcmc_sample_python(log_likelihood_fn, data, initial_params, param_bounds,
                       n_samples, burn_in, proposal_std):
    """Pure Python MCMC implementation"""
    current_params = initial_params.copy()
    samples = []
    current_ll = log_likelihood_fn(current_params.tolist(), data.tolist())
    
    for _ in range(n_samples + burn_in):
        # Propose
        proposed = current_params + np.random.randn(len(current_params)) * proposal_std
        for i, (low, high) in enumerate(param_bounds):
            proposed[i] = np.clip(proposed[i], low, high)
        
        # Accept/reject
        proposed_ll = log_likelihood_fn(proposed.tolist(), data.tolist())
        if np.log(np.random.rand()) < proposed_ll - current_ll:
            current_params = proposed
            current_ll = proposed_ll
        
        if len(samples) >= burn_in:
            samples.append(current_params.copy())
    
    return np.array(samples)


def _grid_search_python(objective_fn, bounds, n_points):
    """Pure Python grid search implementation"""
    n_params = len(bounds)
    grids = [np.linspace(low, high, n_points) for low, high in bounds]
    
    best_params = None
    best_score = float('-inf')
    
    import itertools
    for point in itertools.product(*grids):
        score = objective_fn(np.array(point))
        if score > best_score:
            best_score = score
            best_params = np.array(point)
    
    return best_params, best_score


def _mutual_information_python(x, y, n_bins):
    """Pure Python MI implementation"""
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    px_py = px[:, None] * py[None, :]
    
    # Only compute where both are nonzero
    nonzero = (pxy > 0) & (px_py > 0)
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))
    
    return max(0.0, mi)


def _shannon_entropy_python(x, n_bins):
    """Pure Python entropy implementation"""
    hist, _ = np.histogram(x, bins=n_bins)
    probs = hist[hist > 0] / np.sum(hist)
    return -np.sum(probs * np.log(probs))
