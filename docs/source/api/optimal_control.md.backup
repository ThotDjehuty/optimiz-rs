# API: Optimal Control / Kalman

High-level bindings exposed by the `optimizr` Python package. All functions require the Rust extension (`optimizr._core`).

**When to use this module**
- Threshold trading / switching problems solved via HJB (with and without frictions)
- State estimation and smoothing (Kalman, EKF, UKF)
- Parameter inference for mean-reverting spreads (OU) feeding into control logic

## Hamilton–Jacobi–Bellman (HJB) solvers

```python
from optimizr import solve_hjb_py, solve_hjb_full_py

# Switching boundaries for a mean-reverting spread (OU process)
lower, upper, residual, iters = solve_hjb_py(
    kappa=3.0, theta=0.0, sigma=0.2, rho=0.04,
    transaction_cost=0.001, n_points=400, max_iter=4000,
    tolerance=1e-7, n_std=5.0,
)

# Full state (grid + derivatives) for research/visualization
(lower, upper, residual, iters, x_grid, value, grad, hess) = solve_hjb_full_py(
    kappa=3.0, theta=0.0, sigma=0.2, rho=0.04,
    transaction_cost=0.001, n_points=400,
)
```

### Model

We assume an Ornstein–Uhlenbeck process $dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t$ with quadratic transaction costs. The HJB on grid $x \in [-n_{std}\,\sigma/\sqrt{\kappa},\; n_{std}\,\sigma/\sqrt{\kappa}]$ solves
$$
\rho V(x) = \min\Big\{ \tfrac12 \sigma^2 V_{xx}(x) + \kappa(\theta - x) V_x(x),\; V(x) + c_{\text{buy}},\; V(x) + c_{\text{sell}} \Big\}.
$$
`solve_hjb_py` returns optimal buy/sell thresholds; `solve_hjb_full_py` also returns $V$, $V_x$, and $V_{xx}$ for diagnostics.

**Diagnostic tips:**
- Plot $V_x$ to verify smoothness near the boundaries; kinks often signal insufficient grid resolution.
- Track `residual` and `iterations` to spot non-convergence; loosen `tolerance` or increase `max_iter` if needed.

## OU parameter estimation

```python
import numpy as np
from optimizr import estimate_ou_params_py

spread = np.random.randn(10_000)
kappa, theta, sigma, half_life = estimate_ou_params_py(spread, dt=1/252)
```

Method-of-moments / MLE estimation for
$$
X_{t+1} = X_t e^{-\kappa \Delta t} + \theta(1-e^{-\kappa \Delta t}) + \eta_t, \quad \eta_t \sim \mathcal{N}\Big(0,\; \tfrac{\sigma^2}{2\kappa}(1-e^{-2\kappa \Delta t})\Big).
$$
Returns $(\kappa, \theta, \sigma, \text{half\_life})$.

**Practical guidance:** Use at least a few thousand samples for stable estimates; heavy-tailed series benefit from pre-whitening or winsorizing before fitting.

## Backtesting optimal switching

```python
from optimizr import backtest_optimal_switching_py

metrics = backtest_optimal_switching_py(
    spread=spread,
    lower_bound=lower,
    upper_bound=upper,
    transaction_cost=0.001,
)
(total_return, sharpe, max_dd, n_trades, win_rate, pnl_path) = metrics
```

Applies HJB thresholds to historical spreads and reports return, Sharpe ratio, drawdown, trade count, win rate, and PnL path.

**What to inspect:**
- `win_rate` alongside `max_drawdown` to balance aggressiveness
- PnL path for regime shifts; combine with HMM states if you need regime-aware controls

## Kalman filtering (linear, EKF, UKF)

```python
import numpy as np
from optimizr import LinearKalmanFilter, KalmanState

F = [[1.0, 1.0], [0.0, 1.0]]  # constant-velocity model
H = [[1.0, 0.0]]               # observe position only
Q = [[1e-4, 0.0], [0.0, 1e-4]]
R = [[1e-2]]

kf = LinearKalmanFilter(
    f_matrix=F,
    h_matrix=H,
    q_matrix=Q,
    r_matrix=R,
    initial_state=[0.0, 0.0],
    initial_covariance=[[1.0, 0.0], [0.0, 1.0]],
)

kf.predict(control=[0.0, 0.0])       # optional control input via B matrix
kf.update(observation=[1.2])
state = kf.get_state()               # KalmanState with getters for mean/cov

# Batch filtering
result = kf.filter(observations=[[1.0], [1.4], [1.9]], controls=None)
states = result.get_states()
log_likelihoods = result.get_log_likelihoods()
```

### Notes
- `LinearKalmanFilter` implements `predict`, `update`, and batch `filter`.
- `KalmanState` exposes `get_state()`, `get_covariance()`, and `get_log_likelihood()`.
- Extended/Unscented Kalman filters share the same interface (see `UnscentedKalmanFilter` in the Rust module) and are exported through the same bindings.
- For smoothing, use the Rauch–Tung–Striebel smoother (`RTSSmoother`) available in the bindings.

**Conceptual picture:** Kalman filtering = prediction (dynamics prior) + correction (measurement residual). EKF linearizes $f, h$; UKF propagates sigma points for better nonlinear fidelity. RTS smoothing runs backward in time to refine all past states.

See [`examples/notebooks/03_optimal_control_tutorial.ipynb`](https://github.com/ThotDjehuty/optimiz-r/blob/main/examples/notebooks/03_optimal_control_tutorial.ipynb) for end-to-end usage combining HJB thresholds, OU estimation, and filtering.
