# Verified defects: Baum-Welch normalization and the Hurst estimator

Found while benchmarking `optimizr` against NumPy baselines. Every claim below is
reproduced by `tests/repro_hmm_hurst_defects.py`, which uses only synthetic data
(fixed seeds, no network, no input files).

Environment: Python 3.11.2, numpy 2.x, Linux x86_64, `optimizr` built from this tree.

## A. `estimate_hurst` saturates at exactly 0.99

| input | observed H | expected H |
| --- | --- | --- |
| `np.cumsum(randn(2000))` | 0.990000 | ~1.0 |
| `np.arange(2000)` | 0.990000 | 1.0 |
| `randn(2000)` | 0.553551 | ~0.5 |

A deterministic straight line and a random walk are different processes, yet both
return the identical value. The estimator is clamped on integrated input and therefore
carries no information about price paths. It behaves correctly on stationary input.

Suggested fix: widen or remove the upper clamp, and document that `estimate_hurst`
expects a stationary series while `scale_dependent_hurst` accepts a level path.

## B. `fit_hmm` never re-estimates the initial distribution

`initial_probs` is returned exactly uniform in every configuration tested:
`0.500000 0.500000` for `n_states=2` and `0.333333 0.333333 0.333333` for `n_states=3`,
at n = 1000, 5000 and 12151, at 100 and 500 iterations, with tolerance down to 1e-12.

The test series is generated from a two-regime process that provably begins in the
low-volatility state, so a converged Baum-Welch run should put nearly all initial mass
on that state. The M-step appears to skip the update of pi.

## C. Transition-matrix rows lose normalization as observation kurtosis rises

Row sums of `transition_matrix` are exact for near-Gaussian input and degrade
monotonically with tail weight:

| input | kurtosis | row sums |
| --- | --- | --- |
| `t(df=30)` | 3.2 | 1.000000000 / 1.000000000 / 1.000000000 |
| `t(df=6)` | 6.9 | 0.999819465 / 0.999718279 / 0.999715532 |
| `t(df=3)` | 63.1 | 0.998413170 / 0.998929101 / 0.998920347 |
| `t(df=2)` | 141.9 | 0.998299272 / 0.998067027 / 0.998232526 |

A transition matrix must be row-stochastic to within floating-point error. A shortfall
of 2e-03 is far outside that. Gaussian input, and zero-inflated input at 0/20/50/80 per
cent exact zeros, all return 1.000000000, so this is driven by tail weight rather than by
series length or ties. The likely cause is underflow in the forward-backward scaling
rather than a missing normalization call.

## D. Not a defect: VaR quantile convention

On a heavy-tailed `t(3)` loss series, `cvar_value_py` matches the NumPy tail mean to
`8.88e-16`, while `historical_var_py` differs from `np.quantile(..., 0.95)` by `3.33e-03`.
That is consistent with nearest-rank versus linearly interpolated quantiles, not an
error. Recording it here so the convention is documented rather than rediscovered.

## Reproducing

```
python tests/repro_hmm_hurst_defects.py
```

The script prints one block per item above and needs no arguments.
