# Hidden Markov Model (HMM) API

## Overview

The Hidden Markov Model (HMM) module provides efficient implementations of the Baum-Welch algorithm for parameter estimation and the Viterbi algorithm for state sequence decoding. This is particularly useful for regime detection in time series, speech recognition, biological sequence analysis, and financial market state identification.

## Class: `HMM`

```python
from optimizr import HMM
```

### Constructor

```python
HMM(n_states: int = 2)
```

**Parameters:**
- `n_states` (int): Number of hidden states. Must be at least 2. Default is 2.

**Raises:**
- `ValueError`: If `n_states < 2`

### Attributes

After fitting, the following attributes are populated:

- **`transition_matrix_`** (np.ndarray): State transition probabilities matrix of shape `(n_states, n_states)`. Entry `[i, j]` represents the probability of transitioning from state `i` to state `j`.

- **`emission_means_`** (np.ndarray): Mean parameters of Gaussian emissions for each state. Array of shape `(n_states,)`.

- **`emission_stds_`** (np.ndarray): Standard deviation parameters of Gaussian emissions for each state. Array of shape `(n_states,)`.

### Methods

#### `fit(X, n_iterations=100, tolerance=1e-6)`

Fit HMM parameters using the Baum-Welch (Expectation-Maximization) algorithm.

**Parameters:**
- `X` (np.ndarray): Time series observations as a 1D array.
- `n_iterations` (int, optional): Maximum number of EM iterations. Default is 100.
- `tolerance` (float, optional): Convergence threshold for log-likelihood change. Default is 1e-6.

**Returns:**
- `self` (HMM): The fitted model instance.

**Raises:**
- `ValueError`: If `X` is empty.

**Example:**
```python
import numpy as np
from optimizr import HMM

# Generate sample data with regime changes
returns = np.concatenate([
    np.random.normal(0.01, 0.02, 500),  # Bull market
    np.random.normal(-0.01, 0.03, 500),  # Bear market
])

# Create and fit HMM
hmm = HMM(n_states=2)
hmm.fit(returns, n_iterations=100)

print("Transition Matrix:")
print(hmm.transition_matrix_)
print("\nEmission Means:", hmm.emission_means_)
print("Emission Stds:", hmm.emission_stds_)
```

#### `predict(X)`

Predict the most likely state sequence using the Viterbi algorithm.

**Parameters:**
- `X` (np.ndarray): Time series observations as a 1D array.

**Returns:**
- `states` (np.ndarray): Array of integers representing the most likely state at each time step. Same length as `X`.

**Raises:**
- `ValueError`: If the model has not been fitted yet.

**Example:**
```python
# Decode most likely state sequence
states = hmm.predict(returns)

print(f"Detected states: {np.unique(states)}")
print(f"State distribution: {np.bincount(states)}")

# Visualize regime changes
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(returns, alpha=0.6, label='Returns')
plt.scatter(range(len(returns)), returns, c=states, cmap='viridis', 
            alpha=0.3, s=1, label='States')
plt.legend()
plt.title('HMM Regime Detection')
plt.show()
```

#### `score(X)`

Compute the log-likelihood of observations given the model.

**Parameters:**
- `X` (np.ndarray): Time series observations as a 1D array.

**Returns:**
- `log_likelihood` (float): The log probability of the observations given the model parameters.

**Raises:**
- `ValueError`: If the model has not been fitted yet.

**Example:**
```python
# Calculate model fit quality
ll = hmm.score(returns)
print(f"Log-likelihood: {ll:.2f}")

# Compare different numbers of states
for n in [2, 3, 4]:
    hmm_temp = HMM(n_states=n)
    hmm_temp.fit(returns)
    ll = hmm_temp.score(returns)
    print(f"States: {n}, Log-likelihood: {ll:.2f}")
```

## Complete Example

```python
import numpy as np
from optimizr import HMM

# Simulate financial returns with regime switching
np.random.seed(42)

# Create synthetic data with 3 regimes
n_samples = 1000
regime_1 = np.random.normal(0.02, 0.01, 300)   # High return, low vol
regime_2 = np.random.normal(0.00, 0.02, 400)   # Neutral return, medium vol
regime_3 = np.random.normal(-0.01, 0.03, 300)  # Negative return, high vol

returns = np.concatenate([regime_1, regime_2, regime_3])

# Fit HMM
hmm = HMM(n_states=3)
hmm.fit(returns, n_iterations=100, tolerance=1e-6)

# Decode states
states = hmm.predict(returns)

# Analyze results
print("Transition Matrix:")
print(hmm.transition_matrix_)
print("\nState Statistics:")
for i in range(3):
    mask = states == i
    print(f"State {i}:")
    print(f"  Mean: {hmm.emission_means_[i]:.4f}")
    print(f"  Std:  {hmm.emission_stds_[i]:.4f}")
    print(f"  Count: {np.sum(mask)} ({100*np.sum(mask)/len(states):.1f}%)")

# Calculate model quality
ll = hmm.score(returns)
print(f"\nLog-likelihood: {ll:.2f}")
```

## Performance Notes

- **Rust Backend**: When the Rust backend is available, HMM operations are 50-100x faster than pure Python implementations.

- **Python Fallback**: If the Rust backend is not available, a pure Python implementation using NumPy is automatically used. A warning will be issued.

- **Memory Efficiency**: The implementation uses log-space computations to prevent numerical underflow for long sequences.

- **Numerical Stability**: Forward-backward probabilities are normalized at each time step to maintain numerical stability.

## Algorithm Details

### Baum-Welch (EM) Algorithm

The Baum-Welch algorithm iteratively refines HMM parameters:

1. **E-step**: Compute expected state occupancies using the Forward-Backward algorithm
2. **M-step**: Update transition and emission parameters to maximize expected log-likelihood
3. **Convergence**: Repeat until log-likelihood change is below tolerance

### Viterbi Algorithm

The Viterbi algorithm finds the most likely state sequence:

1. **Initialization**: Set initial state probabilities
2. **Recursion**: For each time step, find the most likely path to each state
3. **Backtracking**: Trace back the optimal path from the final state

## Common Use Cases

### 1. Financial Regime Detection

```python
# Detect bull/bear markets in stock returns
hmm = HMM(n_states=2)
hmm.fit(stock_returns)
market_regimes = hmm.predict(stock_returns)
```

### 2. Volatility Clustering

```python
# Identify high/low volatility periods
abs_returns = np.abs(returns)
hmm = HMM(n_states=2)
hmm.fit(abs_returns)
volatility_regimes = hmm.predict(abs_returns)
```

### 3. Multi-State Analysis

```python
# Analyze complex market dynamics
hmm = HMM(n_states=4)
hmm.fit(returns)
states = hmm.predict(returns)

# States might represent: crash, bear, normal, bull
```

## Tips and Best Practices

1. **Choosing n_states**: Start with 2-3 states. Use cross-validation or information criteria (AIC/BIC) to select the optimal number.

2. **Data Preprocessing**: Standardize or normalize data before fitting, especially when combining multiple time series.

3. **Initialization**: The algorithm initializes parameters based on data quantiles. For better results, you can manually initialize parameters.

4. **Convergence**: If the algorithm doesn't converge, try:
   - Increasing `n_iterations`
   - Adjusting `tolerance`
   - Preprocessing the data to remove outliers

5. **Overfitting**: Too many states can lead to overfitting. Use a validation set to assess generalization.

## See Also

- [MCMC API](mcmc.md) - For Bayesian parameter estimation
- [HMM Theory](theory/hmm.md) - Mathematical background and references
- [Examples](../examples/) - Complete working examples and tutorials
