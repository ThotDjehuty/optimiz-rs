# Differential Evolution API

## Overview

The Differential Evolution (DE) module provides a global optimization algorithm for non-convex, multimodal objective functions. It's particularly effective for problems where gradient information is unavailable or unreliable, and for escaping local optima.

## Function: `differential_evolution`

```python
from optimizr import differential_evolution
```

### Signature

```python
differential_evolution(
    objective_fn: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    popsize: int = 15,
    maxiter: int = 1000,
    f: float = 0.8,
    cr: float = 0.7,
) -> Tuple[np.ndarray, float]
```

### Parameters

- **`objective_fn`** (callable): Function to minimize.
  - **Signature**: `objective_fn(x: np.ndarray) -> float`
  - Takes a 1D array of parameters and returns a scalar objective value.
  - Lower values are better.

- **`bounds`** (List[Tuple[float, float]]): List of (min, max) bounds for each parameter dimension.

- **`popsize`** (int, optional): Population size multiplier. Total population size will be `popsize × n_params`. Default is 15.

- **`maxiter`** (int, optional): Maximum number of generations. Default is 1,000.

- **`f`** (float, optional): Mutation factor, typically in range [0.5, 2.0]. Controls the amplification of differential variation. Default is 0.8.

- **`cr`** (float, optional): Crossover probability, typically in range [0.1, 0.9]. Controls the fraction of parameter values copied from the mutant. Default is 0.7.

### Returns

Returns a tuple `(x, fun)`:
- **`x`** (np.ndarray): Best parameters found (minimum).
- **`fun`** (float): Best objective value (minimum).

Alternatively, when using the Rust backend directly, returns a `DEResult` object with attributes:
- `x`: Best parameters
- `fun`: Best objective value
- `nfev`: Number of function evaluations

## Basic Example

```python
import numpy as np
from optimizr import differential_evolution

# Define the Rosenbrock function (global minimum at [1, 1, ..., 1])
def rosenbrock(x):
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x) - 1))

# Optimize
x_opt, f_min = differential_evolution(
    objective_fn=rosenbrock,
    bounds=[(-5, 5)] * 10,
    popsize=15,
    maxiter=1000
)

print(f"Optimal parameters: {x_opt}")
print(f"Minimum value: {f_min:.6f}")
print(f"Expected: {rosenbrock(np.ones(10)):.6f}")
```

## Advanced Examples

### 1. Rastrigin Function (Many Local Minima)

```python
import numpy as np
from optimizr import differential_evolution

def rastrigin(x):
    """Highly multimodal function with many local minima"""
    A = 10
    n = len(x)
    return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

# True global minimum is at origin with f(0, ..., 0) = 0
x_opt, f_min = differential_evolution(
    objective_fn=rastrigin,
    bounds=[(-5.12, 5.12)] * 10,
    popsize=20,
    maxiter=2000,
    f=0.8,
    cr=0.9
)

print(f"Minimum found: {f_min:.6f}")
print(f"Distance from optimum: {np.linalg.norm(x_opt):.6f}")
```

### 2. Constrained Optimization

```python
import numpy as np
from optimizr import differential_evolution

def constrained_objective(x):
    """Minimize x^2 + y^2 subject to x + y >= 1"""
    obj = x[0]**2 + x[1]**2
    
    # Add penalty for constraint violation
    constraint = x[0] + x[1] - 1
    if constraint < 0:
        obj += 1000 * constraint**2  # Penalty term
    
    return obj

x_opt, f_min = differential_evolution(
    objective_fn=constrained_objective,
    bounds=[(-5, 5), (-5, 5)],
    popsize=15,
    maxiter=500
)

print(f"Optimal point: ({x_opt[0]:.3f}, {x_opt[1]:.3f})")
print(f"Constraint: x + y = {x_opt[0] + x_opt[1]:.3f} (should be ≥ 1)")
print(f"Objective: {f_min:.3f}")
```

### 3. Hyperparameter Tuning

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from optimizr import differential_evolution

# Load data
X, y = load_digits(return_X_y=True)

def svm_objective(params):
    """Optimize SVM hyperparameters"""
    C, gamma = params
    
    # Convert to log scale
    C = 10 ** C
    gamma = 10 ** gamma
    
    # Cross-validation score (negative because we minimize)
    model = SVC(C=C, gamma=gamma)
    score = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    
    return -score.mean()  # Negative because we minimize

# Optimize
params_opt, score_min = differential_evolution(
    objective_fn=svm_objective,
    bounds=[(-3, 3), (-5, 1)],  # log10 scale for C and gamma
    popsize=10,
    maxiter=30
)

C_opt = 10 ** params_opt[0]
gamma_opt = 10 ** params_opt[1]

print(f"Best C: {C_opt:.4f}")
print(f"Best gamma: {gamma_opt:.6f}")
print(f"Best CV accuracy: {-score_min:.4f}")
```

### 4. Portfolio Optimization

```python
import numpy as np
from optimizr import differential_evolution

# Sample returns (rows = assets, columns = time periods)
returns = np.random.randn(5, 1000) * 0.01
returns += np.array([0.08, 0.10, 0.12, 0.06, 0.09])[:, np.newaxis] / 252

def portfolio_objective(weights):
    """Maximize Sharpe ratio (minimize negative Sharpe)"""
    # Ensure weights sum to 1
    weights = weights / weights.sum()
    
    # Calculate portfolio return and volatility
    portfolio_return = np.sum(returns.mean(axis=1) * weights) * 252
    portfolio_vol = np.sqrt(
        np.dot(weights, np.dot(np.cov(returns), weights))
    ) * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 2%)
    sharpe = (portfolio_return - 0.02) / portfolio_vol
    
    return -sharpe  # Negative because we minimize

# Optimize
n_assets = 5
weights_opt, sharpe_neg = differential_evolution(
    objective_fn=portfolio_objective,
    bounds=[(0, 1)] * n_assets,  # Long-only portfolio
    popsize=20,
    maxiter=500
)

# Normalize weights
weights_opt = weights_opt / weights_opt.sum()

print("Optimal Portfolio Weights:")
for i, w in enumerate(weights_opt):
    print(f"  Asset {i+1}: {w:.2%}")

print(f"\nSharpe Ratio: {-sharpe_neg:.3f}")
```

### 5. Function Fitting

```python
import numpy as np
import matplotlib.pyplot as plt
from optimizr import differential_evolution

# Generate noisy data
x_data = np.linspace(0, 10, 100)
y_true = 2.5 * np.sin(0.8 * x_data + 1.2) + 1.5
y_data = y_true + np.random.normal(0, 0.3, len(x_data))

def fitting_objective(params):
    """Fit y = A * sin(B * x + C) + D"""
    A, B, C, D = params
    y_pred = A * np.sin(B * x_data + C) + D
    mse = np.mean((y_data - y_pred)**2)
    return mse

# Optimize
params_opt, mse_min = differential_evolution(
    objective_fn=fitting_objective,
    bounds=[(0, 10), (0, 2), (0, 2*np.pi), (-5, 5)],
    popsize=15,
    maxiter=1000
)

A, B, C, D = params_opt
print(f"Fitted parameters: A={A:.2f}, B={B:.2f}, C={C:.2f}, D={D:.2f}")
print(f"MSE: {mse_min:.4f}")

# Plot
y_fitted = A * np.sin(B * x_data + C) + D
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.5, label='Data')
plt.plot(x_data, y_true, 'g--', label='True', linewidth=2)
plt.plot(x_data, y_fitted, 'r-', label='Fitted', linewidth=2)
plt.legend()
plt.title('Differential Evolution Function Fitting')
plt.show()
```

## Parameter Tuning Guide

### Population Size (`popsize`)

- **Small (5-10)**: Fast but may converge prematurely
- **Medium (15-20)**: Good balance for most problems
- **Large (30+)**: Better exploration, slower convergence

Rule of thumb: `popsize ≥ 10` for problems with up to 10 parameters.

### Mutation Factor (`f`)

- **Low (0.4-0.6)**: Conservative, good for fine-tuning
- **Medium (0.7-0.9)**: Standard, works for most problems
- **High (1.0-2.0)**: Aggressive exploration, avoids local minima

### Crossover Probability (`cr`)

- **Low (0.1-0.3)**: Preserves more of original vector
- **Medium (0.5-0.7)**: Balanced mixing
- **High (0.8-1.0)**: Aggressive recombination

### Maximum Iterations (`maxiter`)

- Depends on problem difficulty and dimensions
- Monitor convergence: if still improving at `maxiter`, increase it
- Typical values: 500-5000

## Convergence Analysis

```python
# Track convergence history (requires modification to return history)
import matplotlib.pyplot as plt

history = []

def tracked_objective(x):
    result = objective_fn(x)
    history.append(result)
    return result

x_opt, f_min = differential_evolution(
    objective_fn=tracked_objective,
    bounds=bounds,
    popsize=15,
    maxiter=1000
)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.semilogy(history)
plt.xlabel('Function Evaluation')
plt.ylabel('Objective Value')
plt.title('Convergence History')
plt.grid(True)
plt.show()
```

## Performance Notes

- **Rust Backend**: 50-100x faster than pure Python implementations for compute-intensive objectives.

- **Python Fallback**: Falls back to `scipy.optimize.differential_evolution` if Rust is unavailable.

- **Parallelization**: Population evaluations are independent and can be parallelized (future enhancement).

- **Complexity**: O(`popsize` × `n_params` × `maxiter` × cost_per_eval)

## Common Use Cases

| Application | Typical Settings | Notes |
|-------------|------------------|-------|
| Hyperparameter tuning | popsize=10-15, maxiter=50-200 | Fast evaluations |
| Engineering design | popsize=20-30, maxiter=500-2000 | Complex constraints |
| Function fitting | popsize=15-20, maxiter=500-1000 | Multiple local minima |
| Portfolio optimization | popsize=15-20, maxiter=200-500 | Moderate dimensions |
| Neural network training | popsize=30-50, maxiter=1000+ | High dimensions |

## Tips and Best Practices

1. **Scaling**: Normalize parameters to similar ranges for better performance.

2. **Bounds**: Set reasonable bounds based on domain knowledge.

3. **Stochastic Objectives**: For noisy functions, use larger population and more iterations.

4. **Warm Start**: Use results from previous runs as initial population.

5. **Hybrid Approach**: Use DE for global search, then local optimizer for refinement.

6. **Early Stopping**: Implement custom stopping criteria based on improvement rate.

## Comparison with Other Optimizers

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **Differential Evolution** | No gradients needed, global search, robust | Slow for high dimensions | Non-convex, derivative-free |
| Gradient Descent | Fast, precise | Needs gradients, local only | Smooth, differentiable |
| Genetic Algorithm | Very flexible | Slower convergence | Discrete, combinatorial |
| Simulated Annealing | Simple, global search | Sensitive to temperature schedule | Simple problems |
| Grid Search | Guaranteed coverage | Exponential cost | Few dimensions only |

## See Also

- [Grid Search API](grid_search.md) - For exhaustive parameter search
- [MCMC API](mcmc.md) - For Bayesian parameter estimation
- [Differential Evolution Theory](theory/differential_evolution.md) - Mathematical background
- [Examples](../examples/) - Complete working examples
