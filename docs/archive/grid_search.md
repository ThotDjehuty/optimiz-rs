# Grid Search API

## Overview

The Grid Search module provides exhaustive parameter space exploration by evaluating the objective function at all points on a regular grid. While computationally expensive, it guarantees finding the best solution within the discretized search space.

## Function: `grid_search`

```python
from optimizr import grid_search
```

### Signature

```python
grid_search(
    objective_fn: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    n_points: int = 10,
) -> Tuple[np.ndarray, float]
```

### Parameters

- **`objective_fn`** (callable): Function to **maximize**.
  - **Signature**: `objective_fn(x: np.ndarray) -> float`
  - Takes a 1D array of parameters and returns a scalar objective value.
  - **Higher values are better** (maximization).

- **`bounds`** (List[Tuple[float, float]]): List of (min, max) bounds for each parameter dimension.

- **`n_points`** (int, optional): Number of equally spaced grid points per dimension. Default is 10.

### Returns

Returns a tuple `(x, fun)`:
- **`x`** (np.ndarray): Best parameters found (maximum).
- **`fun`** (float): Best objective value (maximum).

Alternatively, when using the Rust backend directly, returns a `GridSearchResult` object with attributes:
- `x`: Best parameters
- `fun`: Best objective value
- `nfev`: Number of function evaluations (= `n_points^n_params`)

### Complexity

- **Time**: O(`n_points`^`n_params` × cost_per_eval)
- **Space**: O(`n_points`^`n_params`)

Exponential in the number of parameters!

## Basic Example

```python
import numpy as np
from optimizr import grid_search

# Simple quadratic function with maximum at (0, 0)
def objective(x):
    return -(x[0]**2 + x[1]**2)

# Find maximum
x_opt, f_max = grid_search(
    objective_fn=objective,
    bounds=[(-5, 5), (-5, 5)],
    n_points=50
)

print(f"Optimal point: ({x_opt[0]:.3f}, {x_opt[1]:.3f})")
print(f"Maximum value: {f_max:.6f}")
print(f"Total evaluations: {50**2}")
```

## Advanced Examples

### 1. Hyperparameter Tuning

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from optimizr import grid_search

# Load data
X, y = load_iris(return_X_y=True)

def rf_objective(params):
    """Optimize Random Forest hyperparameters"""
    n_estimators, max_depth = params
    
    # Convert to integers
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    
    # Cross-validation accuracy
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    return scores.mean()

# Grid search
params_opt, acc_max = grid_search(
    objective_fn=rf_objective,
    bounds=[(10, 200), (2, 20)],  # n_estimators, max_depth
    n_points=20
)

print(f"Best n_estimators: {int(params_opt[0])}")
print(f"Best max_depth: {int(params_opt[1])}")
print(f"Best CV accuracy: {acc_max:.4f}")
print(f"Total evaluations: {20**2 = 400}")
```

### 2. Feature Engineering

```python
import numpy as np
from optimizr import grid_search
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1]**2 - X[:, 2] + np.random.randn(100)*0.1

def feature_objective(params):
    """Optimize polynomial degree and regularization"""
    degree, alpha_log = params
    degree = int(degree)
    alpha = 10 ** alpha_log
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Ridge regression with CV
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_poly, y, cv=5, 
                            scoring='neg_mean_squared_error')
    
    return scores.mean()  # Negative MSE (higher is better)

params_opt, score_max = grid_search(
    objective_fn=feature_objective,
    bounds=[(1, 4), (-3, 2)],  # degree, log10(alpha)
    n_points=15
)

print(f"Best polynomial degree: {int(params_opt[0])}")
print(f"Best alpha: {10**params_opt[1]:.6f}")
print(f"Best CV score: {score_max:.6f}")
```

### 3. Signal Processing

```python
import numpy as np
from scipy import signal
from optimizr import grid_search

# Generate noisy signal
t = np.linspace(0, 1, 1000)
true_signal = np.sin(2 * np.pi * 5 * t)
noisy_signal = true_signal + np.random.normal(0, 0.5, len(t))

def filter_objective(params):
    """Optimize Butterworth filter parameters"""
    order, cutoff = params
    order = int(order)
    
    # Design and apply filter
    b, a = signal.butter(order, cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, noisy_signal)
    
    # Minimize MSE with true signal (negative for maximization)
    mse = np.mean((filtered - true_signal)**2)
    return -mse

params_opt, neg_mse = grid_search(
    objective_fn=filter_objective,
    bounds=[(2, 8), (0.05, 0.3)],  # order, cutoff frequency
    n_points=20
)

print(f"Best filter order: {int(params_opt[0])}")
print(f"Best cutoff frequency: {params_opt[1]:.3f}")
print(f"MSE: {-neg_mse:.6f}")
```

### 4. Economic Optimization

```python
import numpy as np
from optimizr import grid_search

def profit_function(params):
    """Maximize profit given price and advertising budget"""
    price, advertising = params
    
    # Demand model: q = 1000 - 20*price + 5*sqrt(advertising)
    quantity = 1000 - 20*price + 5*np.sqrt(advertising)
    quantity = max(0, quantity)  # Can't be negative
    
    # Cost model
    fixed_cost = 5000
    variable_cost = 10  # per unit
    total_cost = fixed_cost + variable_cost * quantity + advertising
    
    # Revenue
    revenue = price * quantity
    
    # Profit
    profit = revenue - total_cost
    
    return profit

params_opt, profit_max = grid_search(
    objective_fn=profit_function,
    bounds=[(15, 60), (0, 10000)],  # price, advertising
    n_points=30
)

price_opt, ad_opt = params_opt
quantity_opt = 1000 - 20*price_opt + 5*np.sqrt(ad_opt)

print(f"Optimal price: ${price_opt:.2f}")
print(f"Optimal advertising: ${ad_opt:.2f}")
print(f"Expected quantity: {quantity_opt:.0f} units")
print(f"Maximum profit: ${profit_max:.2f}")
```

### 5. Portfolio Allocation

```python
import numpy as np
from optimizr import grid_search

# Historical returns for 3 assets
returns = np.array([
    [0.10, 0.12, 0.08],  # Expected annual returns
])
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.09, 0.01],
    [0.02, 0.01, 0.03]
])

def portfolio_objective(params):
    """Maximize risk-adjusted return (Sharpe ratio)"""
    # Only optimize 2 weights; third is determined
    w1, w2 = params
    w3 = 1 - w1 - w2
    
    # Invalid if weights are negative
    if w3 < 0 or w1 < 0 or w2 < 0:
        return -1e10
    
    weights = np.array([w1, w2, w3])
    
    # Portfolio return
    port_return = np.sum(returns * weights)
    
    # Portfolio volatility
    port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
    # Sharpe ratio (assuming risk-free rate = 0.02)
    sharpe = (port_return - 0.02) / port_vol
    
    return sharpe

params_opt, sharpe_max = grid_search(
    objective_fn=portfolio_objective,
    bounds=[(0, 1), (0, 1)],  # weights for assets 1 and 2
    n_points=50
)

w1, w2 = params_opt
w3 = 1 - w1 - w2

print(f"Optimal allocation:")
print(f"  Asset 1: {w1:.2%}")
print(f"  Asset 2: {w2:.2%}")
print(f"  Asset 3: {w3:.2%}")
print(f"Sharpe Ratio: {sharpe_max:.3f}")
```

## Visualization

### 1D Grid Search

```python
import numpy as np
import matplotlib.pyplot as plt
from optimizr import grid_search

# 1D function
def func_1d(x):
    return -(x[0] - 2)**2 + 5

# Create fine grid for plotting
x_plot = np.linspace(-5, 8, 1000)
y_plot = [func_1d([x]) for x in x_plot]

# Grid search
x_opt, f_max = grid_search(
    objective_fn=func_1d,
    bounds=[(-5, 8)],
    n_points=15
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', label='Function', linewidth=2)

# Show grid points
grid_points = np.linspace(-5, 8, 15)
grid_values = [func_1d([x]) for x in grid_points]
plt.scatter(grid_points, grid_values, c='red', s=50, 
           label='Grid points', zorder=3)

plt.scatter(x_opt[0], f_max, c='green', s=200, marker='*',
           label=f'Optimum: ({x_opt[0]:.2f}, {f_max:.2f})', zorder=4)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Grid Search Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2D Grid Search Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt
from optimizr import grid_search

# 2D function
def func_2d(x):
    return np.exp(-((x[0]-1)**2 + (x[1]+1)**2))

# Create grid for visualization
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[func_2d([x1, x2]) for x1, x2 in zip(row1, row2)] 
              for row1, row2 in zip(X1, X2)])

# Grid search
x_opt, f_max = grid_search(
    objective_fn=func_2d,
    bounds=[(-3, 3), (-3, 3)],
    n_points=15
)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(X1, X2, Z, levels=20, cmap='viridis')
plt.colorbar(label='Objective Value')

# Show grid points
grid_1d = np.linspace(-3, 3, 15)
for x1 in grid_1d:
    for x2 in grid_1d:
        plt.plot(x1, x2, 'r.', markersize=3)

plt.scatter(x_opt[0], x_opt[1], c='red', s=300, marker='*',
           edgecolors='white', linewidths=2,
           label=f'Optimum: ({x_opt[0]:.2f}, {x_opt[1]:.2f})')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('2D Grid Search')
plt.legend()
plt.axis('equal')
plt.show()
```

## Performance Analysis

### Computational Cost

```python
import time
from optimizr import grid_search

def expensive_function(x):
    """Simulate expensive computation"""
    time.sleep(0.001)  # 1ms per evaluation
    return -(x[0]**2 + x[1]**2)

# Test different grid sizes
for n_points in [5, 10, 20, 30]:
    n_evals = n_points ** 2
    
    start = time.time()
    x_opt, f_max = grid_search(
        objective_fn=expensive_function,
        bounds=[(-5, 5), (-5, 5)],
        n_points=n_points
    )
    elapsed = time.time() - start
    
    print(f"n_points={n_points:2d}: {n_evals:4d} evaluations, "
          f"{elapsed:.2f}s ({elapsed/n_evals*1000:.2f}ms per eval)")
```

### Scaling with Dimensions

```python
# Demonstrate exponential growth
dimensions = [1, 2, 3, 4, 5]
n_points = 10

for n_dim in dimensions:
    n_evals = n_points ** n_dim
    estimated_time = n_evals * 0.001  # Assuming 1ms per eval
    
    print(f"{n_dim}D: {n_evals:,} evaluations "
          f"(~{estimated_time:.1f}s with 1ms/eval)")
```

Output:
```
1D: 10 evaluations (~0.0s with 1ms/eval)
2D: 100 evaluations (~0.1s with 1ms/eval)
3D: 1,000 evaluations (~1.0s with 1ms/eval)
4D: 10,000 evaluations (~10.0s with 1ms/eval)
5D: 100,000 evaluations (~100.0s with 1ms/eval)
```

## Performance Notes

- **Rust Backend**: When available, grid point generation and evaluation is highly optimized.

- **Python Fallback**: Pure Python/NumPy fallback using `itertools.product`.

- **Parallelization**: Grid evaluations are independent and can be parallelized (future enhancement).

- **Memory**: All grid points are evaluated, so memory usage is O(n_points^n_params).

## When to Use Grid Search

### ✅ Good For

- **Small parameter spaces** (≤ 3 dimensions with reasonable resolution)
- **Expensive models** where you want guaranteed coverage
- **Visualization** and understanding the objective landscape
- **Benchmarking** other optimization methods
- **Discrete parameters** that naturally fit on a grid
- **Verifying global optimum** in small problems

### ❌ Not Good For

- **High-dimensional problems** (exponential cost)
- **Continuous optimization** (infinitely many points)
- **Large-scale hyperparameter tuning** (use random search or Bayesian optimization instead)
- **Time-critical applications** (too slow)

## Tips and Best Practices

### 1. Start Coarse, Then Refine

```python
# First pass: coarse grid
x_coarse, f_coarse = grid_search(
    objective_fn=objective,
    bounds=[(-10, 10), (-10, 10)],
    n_points=10
)

# Second pass: fine grid around optimum
margin = 2.0
x_fine, f_fine = grid_search(
    objective_fn=objective,
    bounds=[
        (x_coarse[0] - margin, x_coarse[0] + margin),
        (x_coarse[1] - margin, x_coarse[1] + margin)
    ],
    n_points=20
)

print(f"Refined optimum: {x_fine}")
```

### 2. Use Logarithmic Scales

```python
# For parameters that span orders of magnitude
def objective_log(params):
    # Convert from log scale
    learning_rate = 10 ** params[0]
    regularization = 10 ** params[1]
    
    # Evaluate model...
    score = model_score(learning_rate, regularization)
    return score

x_opt, f_max = grid_search(
    objective_fn=objective_log,
    bounds=[(-5, -1), (-4, 0)],  # log10 scale
    n_points=20
)

lr_opt = 10 ** x_opt[0]
reg_opt = 10 ** x_opt[1]
```

### 3. Intelligent Bounds Selection

```python
# Use domain knowledge to set reasonable bounds
def intelligent_bounds(parameter_type):
    bounds_dict = {
        'learning_rate': (1e-5, 1e-1),
        'n_estimators': (10, 500),
        'max_depth': (2, 20),
        'alpha': (1e-4, 10),
    }
    return bounds_dict.get(parameter_type, (0, 1))
```

## Comparison with Other Methods

| Method | Coverage | Speed | Use Case |
|--------|----------|-------|----------|
| **Grid Search** | Complete | Slow | Small spaces, verification |
| Random Search | Incomplete | Fast | High dimensions |
| Differential Evolution | Adaptive | Medium | Non-convex functions |
| Bayesian Optimization | Intelligent | Medium | Expensive evaluations |
| Gradient Descent | Local | Very fast | Smooth, differentiable |

## See Also

- [Differential Evolution API](differential_evolution.md) - For large-scale optimization
- [MCMC API](mcmc.md) - For Bayesian inference
- [Examples](../examples/) - Complete working examples and tutorials
