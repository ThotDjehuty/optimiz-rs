# Information Theory API

## Overview

The Information Theory module provides implementations of fundamental information measures: Shannon Entropy and Mutual Information. These metrics are essential for feature selection, dependency detection, causality testing, and understanding information content in data.

## Functions

```python
from optimizr import shannon_entropy, mutual_information
```

## Function: `shannon_entropy`

Computes the Shannon entropy of a random variable using histogram-based probability estimation.

### Signature

```python
shannon_entropy(
    x: np.ndarray,
    n_bins: int = 10,
) -> float
```

### Parameters

- **`x`** (np.ndarray): Sample values from the random variable (1D array).

- **`n_bins`** (int, optional): Number of bins for histogram-based probability estimation. Default is 10.

### Returns

- **`entropy`** (float): Shannon entropy in nats (natural logarithm). Multiply by 1/ln(2) ≈ 1.4427 to convert to bits.

### Formula

$$H(X) = -\sum_{i} p(x_i) \log p(x_i)$$

where $p(x_i)$ is estimated from the histogram.

### Example

```python
import numpy as np
from optimizr import shannon_entropy

# Uniform distribution has high entropy
x_uniform = np.random.uniform(0, 1, 10000)
h_uniform = shannon_entropy(x_uniform, n_bins=20)
print(f"Uniform entropy: {h_uniform:.4f} nats")
print(f"Uniform entropy: {h_uniform/np.log(2):.4f} bits")

# Peaked distribution has low entropy
x_peaked = np.random.normal(0, 0.1, 10000)
h_peaked = shannon_entropy(x_peaked, n_bins=20)
print(f"Peaked entropy: {h_peaked:.4f} nats")

# Constant has zero entropy
x_constant = np.ones(1000)
h_constant = shannon_entropy(x_constant, n_bins=20)
print(f"Constant entropy: {h_constant:.4f} nats")
```

---

## Function: `mutual_information`

Computes the mutual information between two random variables.

### Signature

```python
mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> float
```

### Parameters

- **`x`** (np.ndarray): Sample values from the first random variable (1D array).

- **`y`** (np.ndarray): Sample values from the second random variable (1D array). Must be the same length as `x`.

- **`n_bins`** (int, optional): Number of bins for histogram estimation. Default is 10.

### Returns

- **`mi`** (float): Mutual information in nats (natural logarithm). Multiply by 1/ln(2) to convert to bits.

### Formula

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

or equivalently:

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

### Example

```python
import numpy as np
from optimizr import mutual_information

# Generate correlated variables
np.random.seed(42)
x = np.random.randn(10000)
y = 2 * x + np.random.randn(10000) * 0.5  # Strongly correlated

mi = mutual_information(x, y, n_bins=20)
print(f"Mutual Information: {mi:.4f} nats")

# Independent variables
x_ind = np.random.randn(10000)
y_ind = np.random.randn(10000)

mi_ind = mutual_information(x_ind, y_ind, n_bins=20)
print(f"MI (independent): {mi_ind:.4f} nats (should be near 0)")

# Perfectly correlated
y_perfect = x.copy()
mi_perfect = mutual_information(x, y_perfect, n_bins=20)
print(f"MI (perfect): {mi_perfect:.4f} nats")
```

---

## Advanced Examples

### 1. Feature Selection

```python
import numpy as np
import pandas as pd
from optimizr import mutual_information

# Generate dataset
np.random.seed(42)
n_samples = 1000

# Features
x1 = np.random.randn(n_samples)
x2 = np.random.randn(n_samples)
x3 = np.random.randn(n_samples)
x4 = np.random.randn(n_samples)
x5 = np.random.randn(n_samples)

# Target: depends on x1 and x3, not others
y = 2*x1 + 3*x3 + np.random.randn(n_samples)*0.5

# Calculate MI with target
features = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}
mi_scores = {}

for name, feature in features.items():
    mi = mutual_information(feature, y, n_bins=15)
    mi_scores[name] = mi

# Rank features
ranked = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

print("Feature Importance (by MI):")
for name, score in ranked:
    print(f"  {name}: {score:.4f}")

# Select top features
threshold = 0.5
selected = [name for name, score in ranked if score > threshold]
print(f"\nSelected features: {selected}")
```

### 2. Time Series Dependency

```python
import numpy as np
from optimizr import mutual_information
import matplotlib.pyplot as plt

# Generate time series
np.random.seed(42)
n = 1000
x = np.random.randn(n)

# Calculate MI at different lags
max_lag = 50
mi_lags = []

for lag in range(1, max_lag + 1):
    x_lagged = x[:-lag]
    x_current = x[lag:]
    mi = mutual_information(x_lagged, x_current, n_bins=15)
    mi_lags.append(mi)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_lag + 1), mi_lags, 'b-', linewidth=2)
plt.xlabel('Lag')
plt.ylabel('Mutual Information (nats)')
plt.title('Time Series Autocorrelation via MI')
plt.grid(True, alpha=0.3)
plt.show()

# For an AR(1) process
ar_coef = 0.8
y = np.zeros(n)
for t in range(1, n):
    y[t] = ar_coef * y[t-1] + np.random.randn()

mi_ar = mutual_information(y[:-1], y[1:], n_bins=20)
print(f"AR(1) process MI(lag=1): {mi_ar:.4f}")
```

### 3. Nonlinear Dependency Detection

```python
import numpy as np
from optimizr import mutual_information

np.random.seed(42)
n = 5000

# Linear relationship
x_lin = np.random.randn(n)
y_lin = 2*x_lin + np.random.randn(n)*0.3

# Nonlinear relationship
x_nonlin = np.random.uniform(-3, 3, n)
y_nonlin = x_nonlin**2 + np.random.randn(n)*0.5

# No relationship
x_indep = np.random.randn(n)
y_indep = np.random.randn(n)

# Calculate MI
mi_lin = mutual_information(x_lin, y_lin, n_bins=20)
mi_nonlin = mutual_information(x_nonlin, y_nonlin, n_bins=20)
mi_indep = mutual_information(x_indep, y_indep, n_bins=20)

# Compare with Pearson correlation
corr_lin = np.corrcoef(x_lin, y_lin)[0, 1]
corr_nonlin = np.corrcoef(x_nonlin, y_nonlin)[0, 1]
corr_indep = np.corrcoef(x_indep, y_indep)[0, 1]

print("Linear Relationship:")
print(f"  MI: {mi_lin:.4f}, Correlation: {corr_lin:.4f}")

print("\nNonlinear Relationship:")
print(f"  MI: {mi_nonlin:.4f}, Correlation: {corr_nonlin:.4f}")
print("  (MI detects dependency, correlation doesn't)")

print("\nIndependent:")
print(f"  MI: {mi_indep:.4f}, Correlation: {corr_indep:.4f}")
```

### 4. Image Processing

```python
import numpy as np
from optimizr import shannon_entropy, mutual_information
from skimage import data, filters
import matplotlib.pyplot as plt

# Load image
image = data.camera()  # Grayscale image
h, w = image.shape

# Calculate entropy
entropy_original = shannon_entropy(image.flatten(), n_bins=50)
print(f"Original image entropy: {entropy_original:.4f} nats")

# Apply Gaussian blur
blurred = filters.gaussian(image, sigma=3)
entropy_blurred = shannon_entropy((blurred * 255).astype(int).flatten(), 
                                  n_bins=50)
print(f"Blurred image entropy: {entropy_blurred:.4f} nats")

# Add noise
noisy = image + np.random.randn(h, w) * 20
entropy_noisy = shannon_entropy(noisy.flatten(), n_bins=50)
print(f"Noisy image entropy: {entropy_noisy:.4f} nats")

# Mutual information between patches
patch1 = image[100:200, 100:200].flatten()
patch2 = image[200:300, 200:300].flatten()[:len(patch1)]

mi_patches = mutual_information(patch1, patch2, n_bins=30)
print(f"MI between patches: {mi_patches:.4f}")
```

### 5. Model Comparison

```python
import numpy as np
from optimizr import mutual_information
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Generate data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
y_true = 2*X[:, 0] + 3*X[:, 1]**2 - X[:, 2]*X[:, 3]
y = y_true + np.random.randn(n)*0.5

# Train models
models = {
    'Linear': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Neural Net': MLPRegressor(hidden_layer_sizes=(50, 50), random_state=42)
}

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Mutual information between predictions and true values
    mi = mutual_information(y_pred, y, n_bins=20)
    
    # Also calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    
    print(f"{name}:")
    print(f"  MI(pred, true): {mi:.4f}")
    print(f"  R²: {r2:.4f}")
```

### 6. Causality Testing

```python
import numpy as np
from optimizr import mutual_information

# Test if X causes Y
np.random.seed(42)
n = 1000

# X causes Y
x = np.random.randn(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.5*x[t-1] + 0.3*y[t-1] + np.random.randn()*0.1

# MI(X_t-1, Y_t) should be high
mi_xy = mutual_information(x[:-1], y[1:], n_bins=20)
print(f"MI(X_t-1, Y_t): {mi_xy:.4f}")

# MI(Y_t-1, X_t) should be low (Y doesn't cause X)
mi_yx = mutual_information(y[:-1], x[1:], n_bins=20)
print(f"MI(Y_t-1, X_t): {mi_yx:.4f}")

if mi_xy > 2 * mi_yx:
    print("Evidence suggests X → Y causality")
else:
    print("No clear causal direction")
```

## Choosing the Number of Bins

The choice of `n_bins` affects the bias-variance tradeoff:

### Too Few Bins
- High bias, low variance
- Underestimates entropy/MI
- Use when: limited data, smooth distributions

### Too Many Bins
- Low bias, high variance
- Overestimates due to noise
- Use when: large datasets, complex distributions

### Rule of Thumb

```python
import numpy as np

def optimal_bins(n_samples):
    """Sturges' rule and alternatives"""
    
    # Sturges' rule (for normal distributions)
    sturges = int(np.ceil(np.log2(n_samples) + 1))
    
    # Square root rule
    sqrt_rule = int(np.ceil(np.sqrt(n_samples)))
    
    # Rice rule
    rice = int(np.ceil(2 * n_samples**(1/3)))
    
    # Scott's rule (data-dependent)
    # Would need the actual data
    
    return {
        'sturges': sturges,
        'sqrt': sqrt_rule,
        'rice': rice
    }

# Example
n = 10000
bins = optimal_bins(n)
print(f"For {n} samples:")
print(f"  Sturges: {bins['sturges']} bins")
print(f"  Sqrt: {bins['sqrt']} bins")
print(f"  Rice: {bins['rice']} bins")
```

## Sensitivity Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from optimizr import mutual_information

# Generate correlated data
np.random.seed(42)
n = 5000
x = np.random.randn(n)
y = 2*x + np.random.randn(n)

# Test different bin counts
bin_range = range(5, 51, 5)
mi_values = []

for n_bins in bin_range:
    mi = mutual_information(x, y, n_bins=n_bins)
    mi_values.append(mi)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(bin_range, mi_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Bins')
plt.ylabel('Mutual Information (nats)')
plt.title('MI Sensitivity to Bin Count')
plt.grid(True, alpha=0.3)
plt.show()

# Recommend stable range
mi_std = np.std(mi_values)
stable_range = [b for b, m in zip(bin_range, mi_values) 
                if abs(m - np.mean(mi_values)) < mi_std]
print(f"Stable bin range: {min(stable_range)}-{max(stable_range)}")
```

## Performance Notes

- **Rust Backend**: 20-50x faster than pure Python/NumPy implementations.

- **Python Fallback**: Uses NumPy's `histogram` and `histogram2d` functions.

- **Memory**: O(n + n_bins²) for MI, O(n + n_bins) for entropy.

- **Time Complexity**: O(n) for histogram construction, O(n_bins²) for MI computation.

## Properties and Interpretations

### Shannon Entropy

- **Range**: [0, ∞)
- **Zero**: Only for deterministic (constant) variables
- **Maximum**: For continuous uniform distribution: log(range)
- **Units**: nats (natural log) or bits (log₂)

### Mutual Information

- **Range**: [0, ∞)
- **Zero**: For independent variables
- **Maximum**: min(H(X), H(Y)) - when one determines the other
- **Symmetric**: I(X;Y) = I(Y;X)
- **Non-negative**: I(X;Y) ≥ 0 always

### Normalized Mutual Information

```python
def normalized_mi(x, y, n_bins=10):
    """Normalize MI to [0, 1] range"""
    from optimizr import mutual_information, shannon_entropy
    
    mi = mutual_information(x, y, n_bins)
    hx = shannon_entropy(x, n_bins)
    hy = shannon_entropy(y, n_bins)
    
    # Normalized by arithmetic mean
    nmi_arithmetic = mi / ((hx + hy) / 2)
    
    # Normalized by geometric mean
    nmi_geometric = mi / np.sqrt(hx * hy)
    
    # Normalized by minimum
    nmi_min = mi / min(hx, hy)
    
    return {
        'arithmetic': nmi_arithmetic,
        'geometric': nmi_geometric,
        'min': nmi_min
    }
```

## Common Pitfalls

1. **Insufficient Data**: Need enough samples for reliable histogram estimation. Rule of thumb: n > 10 × n_bins².

2. **Outliers**: Can dominate histogram bins. Consider robust binning or outlier removal.

3. **Different Scales**: Variables on very different scales may need normalization.

4. **Discrete vs Continuous**: Binning discretizes continuous variables, losing some information.

5. **Interpretation**: MI measures dependency, not causation.

## See Also

- [HMM API](hmm.md) - For regime detection using information-theoretic principles
- [Information Theory Theory](theory/information_theory.md) - Mathematical background
- [Examples](../examples/) - Complete working examples
