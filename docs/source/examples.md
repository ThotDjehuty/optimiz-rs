# Examples

Practical snippets for every Optimiz-rs component.

## Differential Evolution (global optimization)

```python
import numpy as np
from optimizr import differential_evolution

def sphere(x):
    return np.sum(x**2)

best_x, best_fx = differential_evolution(
    objective_fn=sphere,
    bounds=[(-10, 10)] * 5,
    strategy="rand1",
    maxiter=300,
    adaptive=True,
)

print(best_fx)
```

## Grid Search (hyper-parameter sweep)

```python
from optimizr import grid_search

def objective(params):
    lr, momentum = params["lr"], params["momentum"]
    return (lr - 0.05)**2 + (momentum - 0.9)**2

best_params, best_score = grid_search(
    objective_fn=objective,
    param_grid={"lr": [0.01, 0.05, 0.1], "momentum": [0.8, 0.9, 0.95]},
)

print(best_params, best_score)
```

## Hidden Markov Models (regime detection)

```python
import numpy as np
from optimizr import HMM

returns = np.random.randn(800) * 0.02 + 0.005
returns[400:] -= 0.015  # regime shift

model = HMM(n_states=2).fit(returns)
states = model.predict(returns)
print(np.bincount(states))
```

## MCMC (posterior sampling)

```python
import numpy as np
from optimizr import mcmc_sample

def log_likelihood(params, data):
    mu, sigma = params
    residuals = (data - mu) / sigma
    return -0.5 * np.sum(residuals**2) - len(data) * np.log(sigma)

data = np.random.randn(500) + 1.0
samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=np.array([0.0, 1.0]),
    param_bounds=[(-5, 5), (0.1, 5.0)],
)
print(samples.mean(axis=0))
```

## Mean Field Games (1D solver)

```python
from optimizr import MFGConfig, solve_mfg_1d_rust

config = MFGConfig(nx=64, nt=32, x_min=-2.0, x_max=2.0, T=1.0, epsilon=0.1, kappa=1.0)
solution = solve_mfg_1d_rust(config)
print(solution.converged)
```

## Sparse Optimization (Sparse PCA)

```python
import numpy as np
from optimizr import sparse_pca_py

X = np.random.randn(200, 10)
components = sparse_pca_py(X, n_components=3, l1_ratio=0.2)
print(components.shape)
```

## Risk Metrics (time series)

```python
import numpy as np
from optimizr import hurst_exponent_py, estimate_half_life_py

returns = np.random.randn(1000) * 0.01
print("Hurst:", hurst_exponent_py(returns))
print("Half-life:", estimate_half_life_py(returns))
```

## Notebooks

- Differential Evolution: `examples/notebooks/03_differential_evolution_tutorial.ipynb`
- Mean Field Games: `examples/notebooks/mean_field_games_tutorial.ipynb`
- HMM: `examples/notebooks/01_hmm_tutorial.ipynb`
- MCMC: `examples/notebooks/02_mcmc_tutorial.ipynb`
- Optimal Control & Kalman: `examples/notebooks/03_optimal_control_tutorial.ipynb`
- Performance benchmarks: `examples/notebooks/05_performance_benchmarks.ipynb`

## Contribute Examples

1. Fork the repository and add notebooks under `examples/notebooks/`
2. Keep dependencies minimal (NumPy/Matplotlib preferred)
3. Ensure the notebook runs end-to-end before submitting a PR
