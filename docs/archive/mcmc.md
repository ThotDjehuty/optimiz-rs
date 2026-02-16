# MCMC Sampling API

## Overview

The MCMC (Markov Chain Monte Carlo) module implements the Metropolis-Hastings algorithm for sampling from arbitrary probability distributions. This is particularly useful for Bayesian parameter estimation, posterior inference, and uncertainty quantification.

## Function: `mcmc_sample`

```python
from optimizr import mcmc_sample
```

### Signature

```python
mcmc_sample(
    log_likelihood_fn: Callable[[List[float], List[float]], float],
    data: np.ndarray,
    initial_params: np.ndarray,
    param_bounds: List[Tuple[float, float]],
    n_samples: int = 10000,
    burn_in: int = 1000,
    proposal_std: float = 0.1,
) -> np.ndarray
```

### Parameters

- **`log_likelihood_fn`** (callable): Function that computes log P(data | params).
  - **Signature**: `log_likelihood_fn(params: list, data: list) -> float`
  - Should return the natural logarithm of the likelihood.
  - Higher values indicate better fit.

- **`data`** (np.ndarray): Observed data passed to the log-likelihood function.

- **`initial_params`** (np.ndarray): Starting parameter values. Should be a 1D array.

- **`param_bounds`** (List[Tuple[float, float]]): List of (min, max) bounds for each parameter. Must have the same length as `initial_params`.

- **`n_samples`** (int, optional): Number of samples to generate after burn-in. Default is 10,000.

- **`burn_in`** (int, optional): Number of initial samples to discard. Default is 1,000.

- **`proposal_std`** (float, optional): Standard deviation of Gaussian random walk proposals. Default is 0.1.

### Returns

- **`samples`** (np.ndarray): Array of shape `(n_samples, n_params)` containing parameter samples from the posterior distribution.

### Raises

- `ValueError`: If `initial_params` and `param_bounds` have different lengths.

## Basic Example

```python
import numpy as np
from optimizr import mcmc_sample

# Define log-likelihood for Gaussian model
def log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return -np.inf
    residuals = (data - mu) / sigma
    return -0.5 * np.sum(residuals**2) - len(data) * np.log(sigma)

# Generate synthetic data
np.random.seed(42)
true_mu, true_sigma = 2.5, 1.2
data = np.random.normal(true_mu, true_sigma, 100)

# Sample from posterior
samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=np.array([0.0, 1.0]),
    param_bounds=[(-10, 10), (0.1, 10)],
    n_samples=10000,
    burn_in=1000,
    proposal_std=0.1
)

# Analyze results
print(f"True mean: {true_mu:.2f}, Estimated: {np.mean(samples[:, 0]):.2f}")
print(f"True std: {true_sigma:.2f}, Estimated: {np.mean(samples[:, 1]):.2f}")

# Posterior credible intervals
print(f"Mean 95% CI: {np.percentile(samples[:, 0], [2.5, 97.5])}")
print(f"Std 95% CI: {np.percentile(samples[:, 1], [2.5, 97.5])}")
```

## Advanced Examples

### 1. Linear Regression

```python
import numpy as np
from optimizr import mcmc_sample

# Log-likelihood for linear regression
def log_likelihood(params, data):
    x, y = data
    slope, intercept, sigma = params
    
    if sigma <= 0:
        return -np.inf
    
    predictions = slope * x + intercept
    residuals = (y - predictions) / sigma
    
    return -0.5 * np.sum(residuals**2) - len(y) * np.log(sigma)

# Generate data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2.5 * x + 1.0 + np.random.normal(0, 0.5, 100)

# Sample from posterior
samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=[x, y],
    initial_params=np.array([1.0, 0.0, 1.0]),
    param_bounds=[(-10, 10), (-10, 10), (0.01, 10)],
    n_samples=20000,
    burn_in=2000,
    proposal_std=0.05
)

print(f"Slope: {np.mean(samples[:, 0]):.3f} ± {np.std(samples[:, 0]):.3f}")
print(f"Intercept: {np.mean(samples[:, 1]):.3f} ± {np.std(samples[:, 1]):.3f}")
print(f"Sigma: {np.mean(samples[:, 2]):.3f} ± {np.std(samples[:, 2]):.3f}")
```

### 2. Mixture Model

```python
import numpy as np
from optimizr import mcmc_sample
from scipy.stats import norm

def log_likelihood(params, data):
    """Two-component Gaussian mixture"""
    mu1, sigma1, mu2, sigma2, weight1 = params
    
    # Ensure valid parameters
    if sigma1 <= 0 or sigma2 <= 0:
        return -np.inf
    if not (0 <= weight1 <= 1):
        return -np.inf
    
    weight2 = 1 - weight1
    
    # Mixture likelihood
    likelihood = (weight1 * norm.pdf(data, mu1, sigma1) + 
                  weight2 * norm.pdf(data, mu2, sigma2))
    
    return np.sum(np.log(likelihood + 1e-10))

# Generate mixture data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 300),
    np.random.normal(5, 1.5, 200)
])

samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=np.array([0.0, 1.0, 5.0, 1.5, 0.6]),
    param_bounds=[(-5, 5), (0.1, 5), (0, 10), (0.1, 5), (0.1, 0.9)],
    n_samples=15000,
    burn_in=3000,
    proposal_std=0.15
)

print("Component 1:")
print(f"  Mean: {np.mean(samples[:, 0]):.2f}")
print(f"  Std: {np.mean(samples[:, 1]):.2f}")
print("\nComponent 2:")
print(f"  Mean: {np.mean(samples[:, 2]):.2f}")
print(f"  Std: {np.mean(samples[:, 3]):.2f}")
print(f"\nMixing weight: {np.mean(samples[:, 4]):.2f}")
```

### 3. Time Series Model (AR process)

```python
import numpy as np
from optimizr import mcmc_sample

def log_likelihood(params, data):
    """Autoregressive AR(2) model"""
    phi1, phi2, sigma = params
    
    if sigma <= 0:
        return -np.inf
    
    # Check stationarity conditions
    if abs(phi1) + abs(phi2) >= 1:
        return -np.inf
    
    # Compute residuals
    predictions = phi1 * data[1:-1] + phi2 * data[:-2]
    residuals = (data[2:] - predictions) / sigma
    
    return -0.5 * np.sum(residuals**2) - (len(data) - 2) * np.log(sigma)

# Generate AR(2) process
np.random.seed(42)
n = 500
phi1_true, phi2_true = 0.6, -0.2
sigma_true = 0.5

data = np.zeros(n)
for t in range(2, n):
    data[t] = (phi1_true * data[t-1] + 
               phi2_true * data[t-2] + 
               np.random.normal(0, sigma_true))

samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=np.array([0.5, -0.1, 0.5]),
    param_bounds=[(-0.99, 0.99), (-0.99, 0.99), (0.01, 5)],
    n_samples=15000,
    burn_in=2000,
    proposal_std=0.05
)

print(f"φ₁: True={phi1_true:.2f}, Est={np.mean(samples[:, 0]):.2f}")
print(f"φ₂: True={phi2_true:.2f}, Est={np.mean(samples[:, 1]):.2f}")
print(f"σ: True={sigma_true:.2f}, Est={np.mean(samples[:, 2]):.2f}")
```

## Diagnostics and Visualization

### Trace Plots

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
param_names = ['Mean', 'Std Dev', 'Parameter 3']

for i, (ax, name) in enumerate(zip(axes, param_names)):
    ax.plot(samples[:, i], alpha=0.7)
    ax.set_ylabel(name)
    ax.axhline(np.mean(samples[:, i]), color='r', linestyle='--', 
               label='Mean')
    ax.legend()

axes[-1].set_xlabel('Iteration')
plt.tight_layout()
plt.show()
```

### Posterior Distributions

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ax in enumerate(axes):
    ax.hist(samples[:, i], bins=50, density=True, alpha=0.7)
    ax.axvline(np.mean(samples[:, i]), color='r', linestyle='--',
               label=f'Mean: {np.mean(samples[:, i]):.3f}')
    ax.set_xlabel(f'Parameter {i+1}')
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.show()
```

### Autocorrelation

```python
from scipy.stats import acf

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, ax in enumerate(axes):
    autocorr = acf(samples[:, i], nlags=100)
    ax.plot(autocorr)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Parameter {i+1}')

plt.tight_layout()
plt.show()
```

### Acceptance Rate

```python
# Estimate acceptance rate from consecutive samples
def acceptance_rate(samples):
    changes = np.sum(np.diff(samples, axis=0) != 0, axis=1)
    return np.mean(changes > 0)

rate = acceptance_rate(samples)
print(f"Acceptance rate: {rate:.2%}")

# Ideal range: 20-40% for Metropolis-Hastings
if rate < 0.15:
    print("⚠ Acceptance rate too low. Try decreasing proposal_std.")
elif rate > 0.50:
    print("⚠ Acceptance rate too high. Try increasing proposal_std.")
else:
    print("✓ Acceptance rate is in good range.")
```

## Performance Notes

- **Rust Backend**: When available, MCMC sampling is 50-100x faster than pure Python implementations.

- **Python Fallback**: A pure NumPy fallback is automatically used if Rust is unavailable.

- **Proposal Tuning**: The `proposal_std` parameter significantly affects convergence:
  - Too small: Slow exploration, high acceptance rate
  - Too large: Poor acceptance rate, slow convergence
  - Optimal: 20-40% acceptance rate

## Tips and Best Practices

### 1. Choosing Burn-in Period

```python
# Run a short chain to visualize convergence
test_samples = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=initial_params,
    param_bounds=bounds,
    n_samples=5000,
    burn_in=0,  # Keep all samples for inspection
    proposal_std=0.1
)

# Plot to determine burn-in
plt.plot(test_samples[:, 0])
plt.xlabel('Iteration')
plt.ylabel('Parameter')
plt.title('Determine burn-in period')
plt.show()
```

### 2. Multiple Chains

```python
# Run multiple chains with different starting points
n_chains = 4
all_samples = []

for i in range(n_chains):
    # Random initialization
    init = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds]
    )
    
    samples = mcmc_sample(
        log_likelihood_fn=log_likelihood,
        data=data,
        initial_params=init,
        param_bounds=bounds,
        n_samples=5000,
        burn_in=1000
    )
    all_samples.append(samples)

# Check convergence across chains
means = [np.mean(s[:, 0]) for s in all_samples]
print(f"Chain means: {means}")
print(f"Variance: {np.var(means):.6f}")
```

### 3. Adaptive Proposal

```python
# Start with exploration, then refine
samples_phase1 = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=initial_params,
    param_bounds=bounds,
    n_samples=5000,
    burn_in=1000,
    proposal_std=0.2  # Larger for exploration
)

# Use posterior mean as new starting point
new_init = np.mean(samples_phase1, axis=0)

samples_phase2 = mcmc_sample(
    log_likelihood_fn=log_likelihood,
    data=data,
    initial_params=new_init,
    param_bounds=bounds,
    n_samples=10000,
    burn_in=1000,
    proposal_std=0.05  # Smaller for refinement
)
```

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Chains don't converge | Poor initialization | Use multiple chains or better starting values |
| High autocorrelation | Proposal too small | Increase `proposal_std` |
| Low acceptance rate | Proposal too large | Decrease `proposal_std` |
| Bimodal posterior | Multiple modes | Use longer chains or parallel tempering |
| Numerical errors | Overflow in likelihood | Use log-likelihood correctly |

## See Also

- [HMM API](hmm.md) - For regime detection and sequence modeling
- [MCMC Theory](theory/mcmc.md) - Mathematical background
- [Examples](../examples/) - Complete working examples
