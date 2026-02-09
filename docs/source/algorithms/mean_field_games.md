# Mean Field Games

Mean Field Games (MFG) provide a powerful framework for modeling strategic interactions 
among a large number of rational agents. Rather than tracking every individual, MFG theory 
replaces the population with a *distribution* and derives equilibrium conditions from 
coupled partial differential equations.

This module implements a **1D Mean Field Games solver** with a high-performance Rust 
backend exposed to Python via PyO3.

---

## Mathematical Foundations

### The State of a Representative Agent

Each agent's state $X_t$ evolves according to a controlled stochastic differential equation:

$$
dX_t = b(X_t, \alpha_t, m_t)\,dt + \sigma\,dW_t
$$

where:

- $\alpha_t$ is the agent's control (decision variable)
- $m_t$ is the population distribution at time $t$
- $W_t$ is standard Brownian motion
- $\sigma$ controls the diffusion intensity (related to `nu` in the solver)

The agent seeks to minimize expected cumulative cost:

$$
J(\alpha) = \mathbb{E}\left[\int_0^T L(X_t, \alpha_t, m_t)\,dt + g(X_T)\right]
$$

---

### The MFG System: Two Coupled PDEs

The MFG equilibrium is characterized by **two coupled PDEs**:

#### 1. Hamilton-Jacobi-Bellman (HJB) Equation — Backward in Time

The value function $u(x,t)$ represents the optimal cost-to-go and satisfies:

$$
-\frac{\partial u}{\partial t} - \nu \frac{\partial^2 u}{\partial x^2} + H\left(x, \frac{\partial u}{\partial x}, m\right) = 0
$$

**Terminal condition:** $u(x, T) = g(x)$ (terminal cost)

The Hamiltonian $H$ captures the running cost. For quadratic control costs:

$$
H(x, p, m) = \frac{|p|^2}{2} - f(x, m)
$$

where $f(x, m)$ is the congestion cost (penalizes crowded regions).

#### 2. Fokker-Planck (FP) Equation — Forward in Time

The population density $m(x,t)$ evolves according to:

$$
\frac{\partial m}{\partial t} - \nu \frac{\partial^2 m}{\partial x^2} - \frac{\partial}{\partial x}\left(m \frac{\partial u}{\partial x}\right) = 0
$$

**Initial condition:** $m(x, 0) = m_0(x)$ (initial population distribution)

This equation propagates the density forward given the optimal velocity field 
$v^*(x,t) = -\partial u / \partial x$ from the HJB solution.

---

### The Fixed-Point Loop

The solver uses an iterative scheme to find the coupled equilibrium:

```
Algorithm: MFG Fixed-Point Iteration
─────────────────────────────────────
1. Initialize: m⁽⁰⁾(x,t) = m₀(x) for all t
2. For k = 0, 1, 2, ... until convergence:
   a. Solve HJB backward:  u⁽ᵏ⁺¹⁾ given m⁽ᵏ⁾
   b. Solve FP forward:    m̃⁽ᵏ⁺¹⁾ given u⁽ᵏ⁺¹⁾
   c. Relax: m⁽ᵏ⁺¹⁾ = α·m̃⁽ᵏ⁺¹⁾ + (1-α)·m⁽ᵏ⁾
   d. Check: ||m⁽ᵏ⁺¹⁾ - m⁽ᵏ⁾|| < tol ?
3. Return: (u*, m*, iterations)
```

The relaxation parameter `alpha` (typically 0.3–0.7) stabilizes convergence by 
damping oscillations between iterations.

---

## Numerical Methods

### Discretization

The solver uses a finite-difference scheme on a uniform grid:

| Parameter | Notation | Description |
|-----------|----------|-------------|
| `nx` | $N_x$ | Number of spatial grid points |
| `nt` | $N_t$ | Number of time steps |
| `dx` | $\Delta x = (x_{max} - x_{min}) / (N_x - 1)$ | Spatial step |
| `dt` | $\Delta t = T / N_t$ | Time step |

### Stability: The CFL Condition

For numerical stability, the scheme requires:

$$
\frac{\nu \cdot \Delta t}{(\Delta x)^2} \leq \frac{1}{2}
$$

**Practical rule**: If you see oscillations or blow-up, either:
- Increase `nt` (smaller $\Delta t$)
- Increase `nu` (more diffusion smooths the solution)
- Decrease `nx` (larger $\Delta x$)

### Transport: Upwind Differencing

The advection term $\partial(m \cdot v)/\partial x$ uses **upwind differencing** 
to ensure stability:

- If $v > 0$: use backward difference
- If $v < 0$: use forward difference

This prevents numerical oscillations in steep density gradients.

### Diffusion: Implicit Scheme

The diffusion term $\nu \partial^2 m / \partial x^2$ is solved **implicitly** 
using a tridiagonal system (Thomas algorithm), making the scheme unconditionally 
stable for diffusion.

### Mass Conservation

After each Fokker-Planck step, the density is renormalized:

$$
m^{(k+1)} \leftarrow \frac{m^{(k+1)}}{\int m^{(k+1)} dx}
$$

This ensures $\int m(x,t)\,dx = 1$ is preserved throughout the simulation.

---

## Python API

### Configuration

```python
from optimizr import MFGConfig

config = MFGConfig(
    nx=100,        # spatial grid points
    nt=100,        # time steps
    x_min=0.0,     # left boundary
    x_max=1.0,     # right boundary
    T=1.0,         # terminal time
    nu=0.01,       # diffusion coefficient (viscosity)
    max_iter=50,   # maximum fixed-point iterations
    tol=1e-5,      # convergence tolerance
    alpha=0.5,     # relaxation parameter
)
```

### Solving the MFG System

```python
import numpy as np
from optimizr import MFGConfig, solve_mfg_1d_rust

# Define spatial grid
x = np.linspace(0, 1, 100)

# Initial population: Gaussian centered at x=0.3
m0 = np.exp(-50 * (x - 0.3) ** 2)
m0 /= np.trapz(m0, x)  # normalize to unit mass

# Terminal cost: quadratic penalty away from x=0.7
u_terminal = 0.5 * (x - 0.7) ** 2

# Create configuration
config = MFGConfig(
    nx=100, nt=100,
    x_min=0.0, x_max=1.0, T=1.0,
    nu=0.01, max_iter=50, tol=1e-5, alpha=0.5,
)

# Solve the MFG system
u, m, iterations = solve_mfg_1d_rust(m0, u_terminal, config, lambda_congestion=0.5)

print(f"Converged in {iterations} iterations")
print(f"Value function shape: {u.shape}")
print(f"Density shape: {m.shape}")
```

**Expected output:**

```
Converged in 34 iterations
Value function shape: (100, 101)
Density shape: (100, 101)
```

---

## Visualization

### Density Evolution Heatmap

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Density heatmap
im0 = axes[0].imshow(m.T, origin='lower', aspect='auto',
                     extent=[0, 1, 0, 1], cmap='viridis')
axes[0].set_xlabel('Position x')
axes[0].set_ylabel('Time t')
axes[0].set_title('Population Density m(x,t)')
plt.colorbar(im0, ax=axes[0])

# Value function heatmap
im1 = axes[1].imshow(u.T, origin='lower', aspect='auto',
                     extent=[0, 1, 0, 1], cmap='plasma')
axes[1].set_xlabel('Position x')
axes[1].set_ylabel('Time t')
axes[1].set_title('Value Function u(x,t)')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig('mfg_heatmaps.png', dpi=150)
```

### Time Slices

```python
t_indices = [0, 25, 50, 75, 100]
colors = plt.cm.viridis(np.linspace(0, 1, len(t_indices)))

plt.figure(figsize=(8, 5))
for i, t_idx in enumerate(t_indices):
    t_val = t_idx / 100.0
    plt.plot(x, m[:, t_idx], color=colors[i], label=f't={t_val:.2f}')

plt.xlabel('Position x')
plt.ylabel('Density m(x,t)')
plt.title('Population Density at Different Times')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('mfg_time_slices.png', dpi=150)
```

---

## Performance

Benchmarks on laptop-class CPU (Apple M1):

| Grid Size | Iterations | Time |
|-----------|------------|------|
| 64×40 | 28 | 0.08 s |
| 100×100 | 34 | 0.37 s |
| 200×200 | 41 | 2.1 s |
| 500×500 | 52 | 18.4 s |

Memory usage scales as $O(N_x \times N_t)$ for storing both arrays.

---

## Convergence Diagnostics

### What to Monitor

1. **Density residual**: $\|m^{(k+1)} - m^{(k)}\|_1$ should decrease monotonically
2. **Value residual**: $\|u^{(k+1)} - u^{(k)}\|_\infty$ should decrease
3. **Mass conservation**: $\int m(x,t)\,dx \approx 1.0$ at all times
4. **No oscillations**: Smooth density profiles without wiggles

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Slow convergence | `alpha` too small | Increase to 0.6–0.7 |
| Oscillating residuals | `alpha` too large | Decrease to 0.3–0.4 |
| Numerical blow-up | CFL violation | Increase `nt` or `nu` |
| Density spikes | Weak diffusion | Increase `nu` or `lambda_congestion` |
| Negative densities | Upwind instability | Increase `nu` |

---

## The Congestion Term

The parameter `lambda_congestion` controls crowd aversion:

$$
f(x, m) = \lambda \cdot m(x)^{\gamma}
$$

| `lambda_congestion` | Effect |
|---------------------|--------|
| 0.0 | No interaction; agents ignore each other |
| 0.1–0.5 | Mild spreading; prefer less crowded regions |
| 1.0+ | Strong dispersion; density stays nearly uniform |

Higher values prevent density spikes but may slow convergence.

---

## Practical Tips

### Grid Resolution

- **Prototyping**: `nx=64, nt=40` — fast iteration, rough results
- **Publication**: `nx=100, nt=100` — good balance of speed and quality
- **High-fidelity**: `nx=200, nt=200` — smooth gradients, longer runtime

### Parameter Tuning

1. Start with `nu=0.01, alpha=0.5, lambda_congestion=0.5`
2. If convergence is slow, try `alpha=0.7`
3. If density has spikes, increase `lambda_congestion` to 1.0
4. If numerical issues appear, increase `nu` to 0.02–0.05

### Initial Conditions

Good choices for `m0`:
- **Gaussian**: `np.exp(-50 * (x - x0)**2)` — localized starting distribution
- **Uniform**: `np.ones(nx) / nx` — spread-out initial population
- **Bimodal**: Sum of two Gaussians — models two subpopulations

---

## References

1. Lasry, J.-M. and Lions, P.-L. (2007). "Mean field games." *Japanese Journal of Mathematics*, 2(1):229–260.

2. Cardaliaguet, P. (2013). "Notes on Mean Field Games." Lecture notes, Collège de France.

3. Achdou, Y. and Capuzzo-Dolcetta, I. (2010). "Mean field games: numerical methods." *SIAM Journal on Numerical Analysis*, 48(3):1136–1162.

4. Huang, M., Malhamé, R., and Caines, P. (2006). "Large population stochastic dynamic games: closed-loop McKean-Vlasov systems and the Nash certainty equivalence principle." *Communications in Information and Systems*, 6(3):221–252.

---

## Notebook Tutorial

For a complete walkthrough with validated outputs and visualizations, see the 
Mean Field Games Tutorial notebook at `examples/notebooks/mean_field_games_tutorial.ipynb`.

The notebook demonstrates:

- Setting up initial distributions
- Running the solver with different parameters
- Visualizing density evolution as 3D surfaces and heatmaps
- Interpreting convergence diagnostics
- Comparing congestion levels

Audit documentation is available at `docs/MFG_TUTORIAL_COMPLETE.md`.
