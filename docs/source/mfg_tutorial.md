# Mean Field Games Tutorial (Production)

This page summarizes the full MFG tutorial notebook (`examples/notebooks/mean_field_games_tutorial.ipynb`) and the accompanying audit in `docs/MFG_TUTORIAL_COMPLETE.md`.

## What the notebook demonstrates

- Rust-backed 1D MFG solver (`solve_mfg_1d_rust`) with PyO3 bindings
- Coupled HJB–Fokker-Planck fixed-point iteration with congestion term
- Execution time: ~0.4 s for a 100×100 grid (agents × time)
- Stable mass conservation and no NaNs across iterations
- Visual outputs: convergence plot, 3D density evolution, 3D value surface, time-slice snapshots

## Problem setup

- Spatial grid: $x \in [0, 1]$, 100 points; time grid: 100 steps, $T = 1.0$
- Viscosity $\nu = 0.01$, relaxation $\alpha = 0.5$, congestion penalty $\lambda = 0.5$
- Initial distribution $m_0$: Gaussian centered at $x=0.3$
- Terminal cost $u_T(x) = 0.5(x - 0.7)^2$ (agents target $x=0.7$)

### Core equations

.. math::
   -\partial_t u - \nu\,\partial_{xx} u + H\big(x, \partial_x u, m\big) = 0,\\
   \partial_t m - \nu\,\partial_{xx} m - \operatorname{div}\big(m\, \partial_p H\big) = 0.

We iterate between backward $u$ and forward $m$ with mass renormalization to keep $\int m \, dx = 1$.

## Usage snippet

```python
import numpy as np
from optimizr import MFGConfig, solve_mfg_1d_rust

x = np.linspace(0, 1, 100)
m0 = np.exp(-50 * (x - 0.3) ** 2)
m0 /= np.trapz(m0, x)

u_terminal = 0.5 * (x - 0.7) ** 2
config = MFGConfig(nx=100, nt=100, x_min=0.0, x_max=1.0, T=1.0, nu=0.01, max_iter=50, tol=1e-5, alpha=0.5)

u, m, iters = solve_mfg_1d_rust(m0, u_terminal, config, lambda_congestion=0.5)
print(f"converged in {iters} iterations: u{u.shape}, m{m.shape}")
```

## Key observations

- Agents split and migrate toward the target region; congestion prevents collapse into a single spike.
- Value function decreases smoothly over time, capturing optimal cost-to-go.
- Convergence is monotone in practice; fixed-point loop hits tolerance within ~50 iterations.

## Why the Rust backend matters

- Implicit diffusion step and upwind transport improve stability over the reference Python solver.
- Rayon parallelism speeds up 2D grids; OpenBLAS accelerates dense linear algebra where applicable.
- Safe bindings via PyO3 with abi3 wheels keep installation friction low.

## Reproducing visuals

- Run the notebook end-to-end to generate 3D surfaces and time-slice plots.
- Export figures from the notebook if you need static assets for papers or presentations.

## Next steps (tracked)

- Add 2D MFG example with separable costs.
- Extend congestion models (e.g., polynomial costs) and compare convergence rates.
- Log convergence metrics to CSV for batch sweeps.
