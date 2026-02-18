# API Reference: Hidden Markov Model (HMM)

The `HMM` class provides a complete implementation of Hidden Markov Models with Gaussian emissions for regime detection, time series modeling, and state inference.

## Quick Start

```python
from optimizr import HMM
import numpy as np

# Create model with 2 hidden states (e.g., bull/bear market)
model = HMM(n_states=2)

# Train on returns data
returns = np.random.randn(1000, 1)  # Should be 2D: (n_samples, n_features)
model.fit(returns, n_iterations=100, tolerance=1e-6)

# Decode most likely state sequence (Viterbi)
states = model.predict(returns)

# Compute log-likelihood (for model comparison)
logp = model.score(returns)

print(f"Log-likelihood: {logp:.2f}")
print(f"Decoded states: {states[:10]}")
```

## Constructor

### `HMM(n_states: int)`

Creates a new Hidden Markov Model with Gaussian emissions.

**Parameters:**
- `n_states` (int): Number of hidden states/regimes. Common choices:
  - `n_states=2`: Binary regime (e.g., bull/bear, high/low volatility)
  - `n_states=3`: Three-regime model (e.g., bull/sideways/bear)
  - `n_states>3`: Fine-grained regime detection (requires more data)

**Returns:**
- `HMM` object with random initialization

**Initialization:**
- Transition matrix $A$: Uniform with slight self-transition bias
- Initial state distribution $\pi$: Uniform
- Emission parameters (means $\mu_i$, covariances $\Sigma_i$): From K-means clustering

**Example:**
```python
# Binary regime model
hmm_2 = HMM(n_states=2)

# Three-regime model for more nuanced detection
hmm_3 = HMM(n_states=3)
```

**When to use:**
- `n_states=2`: Most common, sufficient for many applications
- Higher `n_states`: When you have strong prior belief in multiple regimes and sufficient data (>1000 samples per state)

## Methods

### `fit(X, n_iterations=100, tolerance=1e-6, n_init=1, random_state=None)`

Trains the HMM on observed data using the Baum-Welch (Expectation-Maximization) algorithm.

**Parameters:**
- `X` (np.ndarray): Training data of shape `(n_samples, n_features)`
  - For univariate time series: reshape to `(n, 1)` with `X.reshape(-1, 1)`
  - For multivariate: pass directly as `(n, d)` where `d` is feature dimension
- `n_iterations` (int, default=100): Maximum number of EM iterations
  - Typical range: 50-200
  - More iterations → better convergence but slower training
- `tolerance` (float, default=1e-6): Convergence threshold
  - Algorithm stops when log-likelihood improvement < `tolerance`
  - Typical range: 1e-8 to 1e-4
  - Smaller values → tighter convergence but more iterations
- `n_init` (int, default=1): Number of random initializations
  - The best model (highest log-likelihood) is kept
  - Recommended: 5-10 for production models (helps avoid local minima)
- `random_state` (int, optional): Random seed for reproducibility

**Returns:**
- `self`: The fitted HMM object (for method chaining)

**Algorithm: Baum-Welch (EM for HMMs)**

The Baum-Welch algorithm iteratively refines model parameters:

1. **E-step**: Compute state occupation probabilities
   - Forward pass: $\alpha_t(i) = P(O_1, \ldots, O_t, S_t = i \mid \lambda)$
   - Backward pass: $\beta_t(i) = P(O_{t+1}, \ldots, O_T \mid S_t = i, \lambda)$
   - State probabilities: $\gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{\sum_j \alpha_t(j)\beta_t(j)}$
   - Transition probabilities: $\xi_t(i,j) = \frac{\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}{\sum_{i,j}\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}$

2. **M-step**: Update model parameters
   - Initial probabilities: $\pi_i = \gamma_1(i)$
   - Transition matrix: $a_{ij} = \frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}$
   - Emission means: $\mu_i = \frac{\sum_{t=1}^T \gamma_t(i) O_t}{\sum_{t=1}^T \gamma_t(i)}$
   - Emission covariances: $\Sigma_i = \frac{\sum_{t=1}^T \gamma_t(i)(O_t - \mu_i)(O_t - \mu_i)^T}{\sum_{t=1}^T \gamma_t(i)}$

3. **Convergence**: Repeat until log-likelihood change < tolerance

**Example:**
```python
import numpy as np
from optimizr import HMM

# Simulate two-regime data
np.random.seed(42)
n = 2000

# Regime 1: low volatility (first 1000 samples)
regime1 = np.random.normal(0.0, 0.5, 1000)
# Regime 2: high volatility (last 1000 samples)
regime2 = np.random.normal(0.0, 2.0, 1000)
data = np.concatenate([regime1, regime2]).reshape(-1, 1)

# Train HMM
hmm = HMM(n_states=2)
hmm.fit(data, n_iterations=200, tolerance=1e-6, n_init=5)

print("Training complete")
```

**Convergence diagnostics:**
```python
# Plot log-likelihood over iterations (requires storing history)
# Check if converged before max_iter
# Verify parameters make sense (e.g., distinct means for each state)
```

**Typical training time:**
- 1000 samples, 2 states, 100 iterations: ~50-100ms
- 10000 samples, 3 states, 200 iterations: ~500ms-1s

### `predict(X)`

Decodes the most likely sequence of hidden states using the Viterbi algorithm.

**Parameters:**
- `X` (np.ndarray): Observation sequence of shape `(n_samples, n_features)`
  - Must match feature dimension used in `fit()`

**Returns:**
- `states` (np.ndarray): Most likely state sequence of shape `(n_samples,)`
  - Values are integers in range `[0, n_states-1]`

**Algorithm: Viterbi**

The Viterbi algorithm finds the globally optimal state sequence:

1. **Initialization**: $\delta_1(i) = \pi_i \cdot b_i(O_1)$
2. **Recursion**: $\delta_t(j) = \max_i[\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(O_t)$
3. **Termination**: $P^* = \max_i[\delta_T(i)]$
4. **Backtracking**: Trace back from $\arg\max_i[\delta_T(i)]$ to recover state sequence

**Complexity:** $O(T \cdot K^2)$ where $T$ is sequence length, $K$ is number of states

**Example:**
```python
# After training (see fit() example)
states = hmm.predict(data)

# Analyze regime distribution
unique, counts = np.unique(states, return_counts=True)
for state, count in zip(unique, counts):
    print(f"State {state}: {count} samples ({count/len(states)*100:.1f}%)")

# Identify regime switches
switches = np.where(np.diff(states) != 0)[0]
print(f"Number of regime switches: {len(switches)}")

# Use for trading: buy in regime 0, sell in regime 1
current_state = states[-1]
if current_state == 0:
    print("Signal: BUY (low volatility regime)")
else:
    print("Signal: SELL (high volatility regime)")
```

**Use cases:**
- **Regime detection**: Identify market states (bull/bear, high/low vol)
- **Trading signals**: Generate buy/sell signals based on regime
- **Risk management**: Adjust position size based on estimated regime
- **Anomaly detection**: Flag unusual regime transitions

### `score(X)`

Computes the log-likelihood of the observation sequence under the fitted model.

**Parameters:**
- `X` (np.ndarray): Observation sequence of shape `(n_samples, n_features)`

**Returns:**
- `logp` (float): Log-likelihood $\log P(O \mid \lambda)$

**Algorithm: Forward Algorithm**

The forward algorithm efficiently computes the likelihood:

1. **Initialization**: $\alpha_1(i) = \pi_i \cdot b_i(O_1)$
2. **Induction**: $\alpha_t(j) = \left[\sum_{i=1}^K \alpha_{t-1}(i) \cdot a_{ij}\right] \cdot b_j(O_t)$
3. **Termination**: $P(O \mid \lambda) = \sum_{i=1}^K \alpha_T(i)$

**Numerical stability:** Uses log-space computation with scaling to avoid underflow.

**Example:**
```python
# Model comparison: which number of states fits best?
logp_scores = {}
for n_states in [2, 3, 4]:
    hmm = HMM(n_states=n_states)
    hmm.fit(data, n_iterations=100)
    logp = hmm.score(data)
    logp_scores[n_states] = logp
    print(f"{n_states} states: log-likelihood = {logp:.2f}")

# Higher log-likelihood is better (but watch for overfitting)
best_k = max(logp_scores, key=logp_scores.get)
print(f"Best model: {best_k} states")

# Use BIC for model selection (penalizes complexity)
def bic(logp, n_params, n_samples):
    return -2 * logp + n_params * np.log(n_samples)

n_samples = len(data)
for n_states in [2, 3, 4]:
    n_params = n_states**2 + 2*n_states  # Approx: A, pi, means, variances
    bic_score = bic(logp_scores[n_states], n_params, n_samples)
    print(f"{n_states} states: BIC = {bic_score:.2f}")
```

**Use cases:**
- **Model selection**: Compare models with different `n_states` using BIC/AIC
- **Convergence monitoring**: Track log-likelihood during training
- **Outlier detection**: Low likelihood → data doesn't match model
- **Model reliability**: Higher likelihood → better fit (but watch overfitting)

## Complete Example: Market Regime Detection

Here's a complete workflow for detecting market regimes in financial data:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optimizr import HMM

# 1. Load financial data (example: S&P 500 returns)
# In practice, load from your data source
np.random.seed(42)
n_samples = 2000

# Simulate returns with regime changes
returns = []
for i in range(n_samples):
    if i < 500:  # Bull market
        returns.append(np.random.normal(0.001, 0.01))
    elif i < 1000:  # Correction
        returns.append(np.random.normal(-0.002, 0.02))
    elif i < 1500:  # Recovery
        returns.append(np.random.normal(0.001, 0.015))
    else:  # Bear market
        returns.append(np.random.normal(-0.001, 0.025))

returns = np.array(returns).reshape(-1, 1)

# 2. Train HMM with multiple initializations
print("Training HMM...")
hmm = HMM(n_states=3)  # 3 regimes: bull, neutral, bear
hmm.fit(returns, n_iterations=200, tolerance=1e-6, n_init=10)

# 3. Decode regimes
states = hmm.predict(returns)

# 4. Analyze regimes
print("\nRegime Statistics:")
for state_id in range(3):
    mask = (states == state_id)
    state_returns = returns[mask]
    mean_ret = np.mean(state_returns)
    std_ret = np.std(state_returns)
    count = np.sum(mask)
    print(f"State {state_id}:")
    print(f"  Count: {count} ({count/len(returns)*100:.1f}%)")
    print(f"  Mean return: {mean_ret:.4f}")
    print(f"  Volatility: {std_ret:.4f}")
    print(f"  Sharpe (annualized): {mean_ret/std_ret * np.sqrt(252):.2f}")

# 5. Identify regime switches
switches = np.where(np.diff(states) != 0)[0] + 1
print(f"\nRegime switches: {len(switches)}")
print(f"Average regime duration: {len(returns)/len(switches):.1f} days")

# 6. Visualize regimes
plt.figure(figsize=(14, 8))

# Plot returns with regime colors
plt.subplot(3, 1, 1)
colors = ['green', 'yellow', 'red']
for state_id in range(3):
    mask = (states == state_id)
    plt.scatter(np.where(mask)[0], returns[mask], 
                c=colors[state_id], alpha=0.5, s=10,
                label=f'State {state_id}')
plt.ylabel('Returns')
plt.title('Returns colored by HMM regime')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot cumulative returns per regime
plt.subplot(3, 1, 2)
cumulative = np.cumsum(returns.flatten())
plt.plot(cumulative, color='black', linewidth=1)
for switch in switches:
    plt.axvline(switch, color='red', alpha=0.3, linestyle='--')
plt.ylabel('Cumulative Returns')
plt.title('Cumulative returns with regime switches')
plt.grid(True, alpha=0.3)

# Plot state sequence
plt.subplot(3, 1, 3)
plt.plot(states, linewidth=0.5)
plt.ylabel('State')
plt.xlabel('Time')
plt.title('Decoded state sequence')
plt.yticks(range(3))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hmm_regime_detection.png', dpi=150)
print("\nPlot saved to hmm_regime_detection.png")

# 7. Generate trading signals
current_state = states[-1]
state_returns = returns[states == current_state]
expected_return = np.mean(state_returns)
expected_vol = np.std(state_returns)

print(f"\nCurrent regime: State {current_state}")
print(f"Expected return: {expected_return:.4f}")
print(f"Expected volatility: {expected_vol:.4f}")

if expected_return > 0.0005:
    signal = "BUY"
    position_size = 1.0
elif expected_return < -0.0005:
    signal = "SELL"
    position_size = 0.0
else:
    signal = "HOLD"
    position_size = 0.5

print(f"Trading signal: {signal}")
print(f"Recommended position size: {position_size*100:.0f}%")
```

## Advanced Usage

### Model Selection with BIC

Choose the optimal number of states using Bayesian Information Criterion:

```python
from optimizr import HMM
import numpy as np

def bic_score(hmm, X):
    """Compute BIC for HMM: BIC = -2*log(L) + k*log(n)"""
    logp = hmm.score(X)
    n_states = hmm.n_states  # Assuming this attribute exists
    n_features = X.shape[1]
    # Parameters: transition matrix + initial prob + means + covariances
    k = n_states**2 + n_states + n_states*n_features + n_states*n_features**2
    n = X.shape[0]
    return -2*logp + k*np.log(n)

# Test different numbers of states
results = []
for n_states in range(2, 6):
    hmm = HMM(n_states=n_states)
    hmm.fit(data, n_iterations=100, n_init=5)
    bic = bic_score(hmm, data)
    logp = hmm.score(data)
    results.append((n_states, logp, bic))
    print(f"{n_states} states: log-likelihood={logp:.2f}, BIC={bic:.2f}")

# Best model has lowest BIC
best_n_states = min(results, key=lambda x: x[2])[0]
print(f"\nBest model: {best_n_states} states")
```

### Integration with Optimal Control

Combine HMM regime detection with regime-specific optimal control:

```python
from optimizr import HMM, estimate_ou_params_py, solve_hjb_py

# 1. Detect regimes with HMM
returns = np.diff(spread)
hmm = HMM(n_states=2)
hmm.fit(returns.reshape(-1, 1), n_iterations=100)
regimes = hmm.predict(returns.reshape(-1, 1))

# 2. Estimate OU parameters per regime
thresholds = {}
for regime_id in range(2):
    mask = (regimes == regime_id)
    spread_regime = spread[1:][mask]  # Align with returns
    
    # Estimate OU parameters
    kappa, theta, sigma, half_life = estimate_ou_params_py(
        spread_regime, dt=1/252
    )
    
    # Solve HJB for regime-specific thresholds
    lower, upper, _, _ = solve_hjb_py(
        kappa=kappa, theta=theta, sigma=sigma,
        rho=0.04, transaction_cost=0.001
    )
    
    thresholds[regime_id] = (lower, upper)
    print(f"Regime {regime_id}: κ={kappa:.2f}, thresholds=({lower:.3f}, {upper:.3f})")

# 3. Apply regime-aware trading
current_regime = regimes[-1]
lower, upper = thresholds[current_regime]
current_spread = spread[-1]

if current_spread < lower:
    action = "BUY"
elif current_spread > upper:
    action = "SELL"
else:
    action = "HOLD"

print(f"\nCurrent regime: {current_regime}")
print(f"Current spread: {current_spread:.3f}")
print(f"Thresholds: ({lower:.3f}, {upper:.3f})")
print(f"Action: {action}")
```

### Multivariate HMM

For multiple features (e.g., returns + volume + volatility):

```python
# Prepare multivariate data
returns = np.random.randn(1000, 1)
volume = np.random.randn(1000, 1)
volatility = np.random.randn(1000, 1)

# Stack features
X = np.hstack([returns, volume, volatility])  # Shape: (1000, 3)

# Train multivariate HMM
hmm = HMM(n_states=3)
hmm.fit(X, n_iterations=150)

# Decode regimes based on all features
states = hmm.predict(X)

# Each state now captures joint patterns in returns, volume, and volatility
```

## Best Practices

### Data Preparation

1. **Scaling**: Standardize features to similar scales
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Stationarity**: Ensure time series is stationary (use returns, not prices)
   ```python
   returns = np.diff(np.log(prices))  # Log returns
   ```

3. **Outlier handling**: Winsorize extreme values
   ```python
   from scipy.stats import mstats
   X_winsorized = mstats.winsorize(X, limits=[0.01, 0.01])
   ```

### Model Training

1. **Multiple initializations**: Use `n_init=5-10` to avoid local minima
2. **Convergence**: Monitor log-likelihood, ensure convergence before `max_iter`
3. **Validation**: Use held-out data to verify generalization

### Parameter Selection

1. **Number of states**: Start with 2-3, increase if necessary
2. **Iterations**: 100-200 typically sufficient
3. **Tolerance**: 1e-6 for production, 1e-4 for quick experimentation

### Practical Tips

1. **Minimum data**: Use at least 500 samples per state (1000+ for 2-state model)
2. **Regime persistence**: Check average regime duration is meaningful (not too short)
3. **Physical interpretation**: Verify decoded regimes make sense (e.g., high-vol state has higher variance)
4. **Robustness**: Test on multiple time periods, verify stability

## Troubleshooting

### Model not converging
- **Symptom**: Log-likelihood oscillating or not improving
- **Fix**: Increase `n_iterations`; try different `n_init`; check data scaling

### All samples assigned to one state
- **Symptom**: `predict()` returns all 0s or all 1s
- **Fix**: Reduce `n_states`; check data has sufficient variation; verify stationarity

### Unrealistic regime switches
- **Symptom**: State changes every few samples
- **Fix**: Add transition probability constraints (requires model extension); increase minimum regime duration

### Poor out-of-sample performance
- **Symptom**: High in-sample log-likelihood but poor predictions on new data
- **Fix**: Reduce `n_states` (overfitting); use cross-validation; add regularization

## Performance Characteristics

### Computational Complexity

- **Training (Baum-Welch)**: $O(I \cdot T \cdot K^2)$
  - $I$: number of iterations (~100-200)
  - $T$: sequence length
  - $K$: number of states
  
- **Prediction (Viterbi)**: $O(T \cdot K^2)$
  
- **Scoring (Forward)**: $O(T \cdot K^2)$

### Memory Requirements

- Model parameters: $O(K^2 + K \cdot d^2)$ where $d$ is feature dimension
- Forward/backward matrices: $O(T \cdot K)$

### Typical Runtimes (on modern CPU)

- Train (1000 samples, 2 states, 100 iter): ~50-100ms
- Train (10000 samples, 3 states, 200 iter): ~500ms-1s
- Predict (1000 samples, 2 states): ~5-10ms
- Score (1000 samples, 2 states): ~5-10ms

## References

### Hidden Markov Models
- **Rabiner, L. R.** (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
- **Murphy, K. P.** (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. (Chapter 17: Markov and hidden Markov models)

### Financial Applications
- **Guidolin, M., & Timmermann, A.** (2008). International asset allocation under regime switching, skew, and kurtosis preferences. *The Review of Financial Studies*, 21(2), 889-935.
- **Nystrup, P., Madsen, H., & Lindström, E.** (2015). Stylised facts of financial time series and hidden Markov models in continuous time. *Quantitative Finance*, 15(9), 1531-1541.
- **Ang, A., & Bekaert, G.** (2002). Regime switches in interest rates. *Journal of Business & Economic Statistics*, 20(2), 163-182.

### Algorithms
- **Forney, G. D.** (1973). The Viterbi algorithm. *Proceedings of the IEEE*, 61(3), 268-278.
- **Baum, L. E., Petrie, T., Soules, G., & Weiss, N.** (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. *The Annals of Mathematical Statistics*, 41(1), 164-171.

## See Also

- [HMM Algorithms](../algorithms/hmm.md) - Detailed mathematical foundations (Forward-Backward, Viterbi, Baum-Welch)
- [Optimal Control](../algorithms/optimal_control.md) - Integrate HMM regimes with optimal control
- [Optimal Control API](optimal_control.md) - API reference for control algorithms
