# Optimal Control

Hamilton–Jacobi–Bellman (HJB) solvers, regime-switching thresholds, OU parameter estimation, and Kalman filtering utilities backed by Rust.

## HJB switching boundaries (OU process)

```python
from optimizr import solve_hjb_py, solve_hjb_full_py

lower, upper, residual, iters = solve_hjb_py(
    kappa=3.0,
    theta=0.0,
    sigma=0.2,
    rho=0.04,
    transaction_cost=0.001,
    n_points=400,
    max_iter=4000,
    tolerance=1e-7,
    n_std=5.0,
)
print(f"bounds=({lower:.3f}, {upper:.3f}), residual={residual:.2e}, iters={iters}")
```

- Model: $dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t$ with quadratic transaction costs.
- Output: optimal buy/sell thresholds; `solve_hjb_full_py` also returns $V, V_x, V_{xx}$ for diagnostics.
- Diagnostics: plot $V_x$ for smoothness near thresholds; monitor `residual` and increase `max_iter` if not converged.

## Backtesting optimal switching

```python
from optimizr import backtest_optimal_switching_py

metrics = backtest_optimal_switching_py(
    spread=spread,
    lower_bound=lower,
    upper_bound=upper,
    transaction_cost=0.001,
)
(
    total_return,
    sharpe,
    max_dd,
    n_trades,
    win_rate,
    pnl_path,
) = metrics
```

Inspect `win_rate` vs `max_dd` to tune aggressiveness; combine with HMM regimes for state-aware controls.

## OU parameter estimation

```python
import numpy as np
from optimizr import estimate_ou_params_py

spread = np.random.randn(10_000)
kappa, theta, sigma, half_life = estimate_ou_params_py(spread, dt=1/252)
```

Method-of-moments / MLE fit returns $(\kappa, \theta, \sigma, \text{half-life})$. Use a few thousand samples for stability; winsorize heavy tails if needed.

## Kalman filtering (linear, EKF, UKF)

```python
import numpy as np
from optimizr import LinearKalmanFilter

F = [[1.0, 1.0], [0.0, 1.0]]
H = [[1.0, 0.0]]
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

kf.predict(control=[0.0, 0.0])
kf.update(observation=[1.2])
state = kf.get_state()
```

- Interfaces: `LinearKalmanFilter`, `UnscentedKalmanFilter`, and `KalmanState` for batch `filter` and smoothing.
- Concept: prediction (dynamics prior) + correction (measurement residual); RTS smoother refines past states.

## Practical notes
- Rust backend (`optimizr._core`) must be present for control utilities; install from source if wheels are unavailable.
- Grids: for HJB, `n_points≈400` is stable; widen `n_std` for volatile spreads.
- Combine with Mean Field Games: see `mean_field_games.md` for population dynamics; use Kalman estimates as control inputs if needed.
