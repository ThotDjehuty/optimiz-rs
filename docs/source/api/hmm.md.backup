# API: HMM

```python
from optimizr import HMM

model = HMM(n_states=2)
model.fit(X, n_iterations=100, tolerance=1e-6)
states = model.predict(X)
logp = model.score(X)
```

Parameters
- `n_states`: number of hidden regimes
- `fit(X, n_iterations=100, tolerance=1e-6)`: train with Baum-Welch
- `predict(X)`: Viterbi decoding → `np.ndarray`
- `score(X)`: log-likelihood
