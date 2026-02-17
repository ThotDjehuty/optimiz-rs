# Differential Evolution

**Differential Evolution (DE)** is a population-based metaheuristic optimization algorithm 
introduced by Storn and Price (1997). It is particularly effective for continuous, non-convex, 
multimodal optimization problems where gradient information is unavailable or unreliable.

This module provides a high-performance Rust implementation with Python bindings, supporting 
multiple mutation strategies, adaptive parameter control (jDE), and parallel evaluation.

---

## Mathematical Foundations

### Problem Formulation

DE solves unconstrained (or box-constrained) minimization problems:

$$
\min_{\mathbf{x} \in \mathbb{R}^D} f(\mathbf{x})
$$

subject to box constraints:

$$
x_j \in [l_j, u_j], \quad j = 1, \ldots, D
$$

**DE is well-suited when:**

- $f$ is continuous but non-differentiable
- Multiple local minima exist
- Gradient information is unavailable or expensive
- Problem dimension is moderate ($D < 100$)

---

### Population

DE maintains a population of $N_P$ candidate solutions:

$$
P_g = \{\mathbf{x}_{1,g}, \mathbf{x}_{2,g}, \ldots, \mathbf{x}_{N_P,g}\}
$$

where $g$ is the generation number and $\mathbf{x}_{i,g} \in \mathbb{R}^D$.

**Rule of thumb:** $N_P = 10 \times D$ where $D$ is the problem dimension.

---

### Main Loop

For each generation $g = 0, 1, 2, \ldots$:

1. **Mutation**: Create mutant vectors by combining existing solutions
2. **Crossover**: Mix mutant with target vector to form trial vector
3. **Selection**: Keep better solution (greedy selection)

---

## Mutation Strategies

The mutation operator creates a **mutant vector** $\mathbf{v}_{i,g+1}$ from existing 
population members:

### DE/rand/1 (Classic Strategy)

$$
\mathbf{v}_{i,g+1} = \mathbf{x}_{r_1,g} + F \cdot (\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g})
$$

where:
- $r_1, r_2, r_3 \in \{1, \ldots, N_P\}$ are randomly chosen, distinct, and $\neq i$
- $F \in (0, 2]$ is the **mutation factor** (typically 0.5–1.0)

**Interpretation:** Start from a random population member $\mathbf{x}_{r_1}$, 
move in direction given by the difference $(\mathbf{x}_{r_2} - \mathbf{x}_{r_3})$, 
scaled by $F$.

**Characteristics:** Most explorative, good for diverse populations.

### DE/best/1

$$
\mathbf{v}_{i,g+1} = \mathbf{x}_{\text{best},g} + F \cdot (\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g})
$$

**Advantage:** Faster convergence toward the best-known solution.

**Disadvantage:** More likely to get stuck in local minima.

### DE/current-to-best/1

$$
\mathbf{v}_{i,g+1} = \mathbf{x}_{i,g} + F \cdot (\mathbf{x}_{\text{best},g} - \mathbf{x}_{i,g}) + F \cdot (\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g})
$$

**Interpretation:** Move current solution toward the best while also exploring.

**Characteristics:** Balanced exploration/exploitation.

### DE/rand/2

$$
\mathbf{v}_{i,g+1} = \mathbf{x}_{r_1,g} + F \cdot (\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}) + F \cdot (\mathbf{x}_{r_4,g} - \mathbf{x}_{r_5,g})
$$

**Characteristics:** More disruptive, better for highly multimodal problems.

### DE/best/2

$$
\mathbf{v}_{i,g+1} = \mathbf{x}_{\text{best},g} + F \cdot (\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g}) + F \cdot (\mathbf{x}_{r_3,g} - \mathbf{x}_{r_4,g})
$$

**Characteristics:** Aggressive convergence to the best solution.

---

## Crossover

After mutation, the **trial vector** $\mathbf{u}_{i,g+1}$ is formed by mixing 
components from the mutant and the target vector.

### Binomial Crossover

For each component $j = 1, \ldots, D$:

$$
u_{i,j,g+1} = \begin{cases}
v_{i,j,g+1} & \text{if } \text{rand}(0,1) \leq CR \text{ or } j = j_{\text{rand}} \\
x_{i,j,g} & \text{otherwise}
\end{cases}
$$

where:
- $CR \in [0, 1]$ is the **crossover probability**
- $j_{\text{rand}} \in \{1, \ldots, D\}$ ensures at least one component comes from the mutant

**Effect:** $CR$ controls how much of the mutant vector is used.

| CR Value | Effect |
|----------|--------|
| Low (0.1–0.3) | Less information exchange, slower convergence. Better for separable problems |
| High (0.7–0.9) | More information exchange, faster convergence. Better for non-separable problems |
| 0.0 | Pure mutation (except $j_{\text{rand}}$) |
| 1.0 | Full crossover |

---

## Selection

Greedy selection (for minimization):

$$
\mathbf{x}_{i,g+1} = \begin{cases}
\mathbf{u}_{i,g+1} & \text{if } f(\mathbf{u}_{i,g+1}) \leq f(\mathbf{x}_{i,g}) \\
\mathbf{x}_{i,g} & \text{otherwise}
\end{cases}
$$

**Property:** Population quality never decreases:

$$
f(\mathbf{x}_{\text{best},g+1}) \leq f(\mathbf{x}_{\text{best},g})
$$

---

## Complete Algorithm

```
Algorithm: Differential Evolution
─────────────────────────────────
Input: objective f, bounds [l, u], pop_size N_P, F, CR, max_iter

1. Initialize population:
   For i = 1 to N_P:
       x_{i,0} = l + rand(0,1) · (u - l)    # uniform in bounds
   
2. Evaluate fitness:
   f_i = f(x_{i,0}) for all i

3. While g < max_iter and not converged:
   
   a. For i = 1 to N_P:
      
      i. Mutation:
         Select r_1, r_2, r_3 distinct and ≠ i
         v_{i,g+1} = x_{r_1,g} + F · (x_{r_2,g} - x_{r_3,g})
      
      ii. Crossover:
          j_rand = randint(1, D)
          For j = 1 to D:
              if rand(0,1) ≤ CR or j = j_rand:
                  u_{i,j,g+1} = v_{i,j,g+1}
              else:
                  u_{i,j,g+1} = x_{i,j,g}
      
      iii. Boundary handling:
           Clip u_{i,g+1} to [l, u]
      
      iv. Selection:
          if f(u_{i,g+1}) ≤ f(x_{i,g}):
              x_{i,g+1} = u_{i,g+1}
          else:
              x_{i,g+1} = x_{i,g}
   
   b. g = g + 1

4. Return x_best and f(x_best)
```

---

## Parameter Selection Guidelines

### Population Size ($N_P$)

| Size Category | Range | Use Case |
|---------------|-------|----------|
| Small | < 4D | Faster convergence; risk premature convergence. Simple unimodal problems |
| Medium | 10D (default) | Good balance for most problems |
| Large | > 20D | Better exploration; slower convergence. Highly multimodal problems |

**Minimum:** $N_P \geq 4$ (needed for mutation with three distinct indices).

### Mutation Factor ($F$)

| F Value | Effect |
|---------|--------|
| Low (0.4–0.6) | Fine-tuning, local search. Safer, less disruptive |
| High (0.8–1.2) | Exploration, global search. Escape local minima |

**Typical range:** $F \in [0.4, 1.0]$, default 0.8.

### Crossover Probability ($CR$)

| CR Value | Effect |
|----------|--------|
| Low (0.1–0.3) | Best for separable problems |
| High (0.7–0.9) | Best for non-separable problems |

**Default:** 0.7–0.9 for most problems.

---

## Python API

### Basic Usage

```python
import numpy as np
from optimizr import differential_evolution

def rastrigin(x):
    """Multimodal benchmark function with many local minima."""
    A = 10
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

best_x, best_fx = differential_evolution(
    objective_fn=rastrigin,
    bounds=[(-5.12, 5.12)] * 10,  # 10-dimensional problem
    strategy="best1",
    popsize=20,
    maxiter=500,
    adaptive=True,
)

print(f"Best fitness: {best_fx:.6f}")
print(f"Best solution: {best_x}")
```

**Expected output:**

```
Best fitness: 0.000042
Best solution: [ 0.00012 -0.00023  0.00018 ... ]
```

### Configuration Options

```python
from optimizr import DifferentialEvolution

de = DifferentialEvolution(
    bounds=[(-5, 5)] * 20,
    strategy="rand1",      # mutation strategy
    popsize=200,           # population size
    maxiter=1000,          # maximum generations
    F=0.8,                 # mutation factor
    CR=0.9,                # crossover probability
    tol=1e-8,              # convergence tolerance
    seed=42,               # reproducibility
)

result = de.minimize(sphere_function)
print(f"Converged in {result.nit} iterations")
print(f"Function evaluations: {result.nfev}")
```

---

## Adaptive Control (jDE)

Optimiz-rs implements **jDE** (self-adaptive DE), where the parameters $F$ and $CR$ 
evolve with the population:

$$
F_{i,g+1} = \begin{cases}
F_l + \text{rand}(0,1) \cdot (F_u - F_l) & \text{if } \text{rand}(0,1) < \tau_1 \\
F_{i,g} & \text{otherwise}
\end{cases}
$$

$$
CR_{i,g+1} = \begin{cases}
\text{rand}(0,1) & \text{if } \text{rand}(0,1) < \tau_2 \\
CR_{i,g} & \text{otherwise}
\end{cases}
$$

**Enable jDE:**

```python
de = DifferentialEvolution(
    bounds=[(-5, 5)] * 20,
    adaptive=True,   # enables jDE
    tau_F=0.1,       # probability of F mutation
    tau_CR=0.1,      # probability of CR mutation
)
```

**Advantages:**
- No need to manually tune $F$ and $CR$
- Adapts to problem landscape during optimization
- Generally robust across problem types

---

## Parallel Evaluation (Rust Backend)

For pure-Rust objectives or when Python callbacks are not needed, enable 
data-parallel evaluation via Rayon:

```python
from optimizr import parallel_differential_evolution_rust

result = parallel_differential_evolution_rust(
    objective="rastrigin",  # built-in benchmark
    dim=50,
    bounds=(-5.12, 5.12),
    popsize=500,
    maxiter=2000,
    n_threads=8,
)

print(f"Best fitness: {result.best_fitness:.8f}")
```

**Speedup:** Near-linear up to $N_P$ processors for expensive objectives.

---

## Convergence Analysis

### Theoretical Properties

**Global Convergence Theorem** (Lampinen, 2001):

Under these sufficient conditions:
- Population size $N_P > 3$
- Mutation factor $F > 0$
- At least one component crossed over ($j_{\text{rand}}$)

DE is a **global optimization method**: any point can be reached with positive probability.

### Diversity Measure

$$
D_g = \frac{1}{N_P D} \sum_{i=1}^{N_P} \sum_{j=1}^D |x_{i,j,g} - \bar{x}_{j,g}|
$$

| Diversity | Behavior |
|-----------|----------|
| High | Exploration (global search) |
| Low | Exploitation (local search) |

### Empirical Budget

**Rule of thumb:** Budget $10^4 \times D$ function evaluations for moderately difficult problems.

---

## Performance Comparison

| Algorithm | Gradient | Global | Constraints | Speed | Best For |
|-----------|----------|--------|-------------|-------|----------|
| **DE** | No | Yes | Box | Medium | Non-convex, continuous |
| Gradient Descent | Yes | No | Yes | Fast | Smooth, convex |
| Genetic Algorithm | No | Yes | Yes | Slow | Discrete, combinatorial |
| Particle Swarm | No | Yes | Box | Fast | Continuous, many dimensions |
| CMA-ES | No | Yes | Box | Fast | Continuous, noisy |

---

## Practical Tips

### 1. Start Simple

Use defaults: $N_P = 10D$, $F = 0.8$, $CR = 0.7$, `strategy="rand1"`.

### 2. Scale Variables

Normalize parameters to similar ranges for better performance.

### 3. Warm Start

If you have a good initial guess, seed the population around it.

### 4. Hybrid Approach

Use DE for global search, then a local optimizer for refinement:

```python
# Global search with DE
best_x, _ = differential_evolution(f, bounds, maxiter=200)

# Local refinement with L-BFGS-B
from scipy.optimize import minimize
result = minimize(f, best_x, method='L-BFGS-B', bounds=bounds)
```

### 5. Monitor Convergence

Plot:
- Best fitness vs. generation
- Average population fitness vs. generation
- Population diversity vs. generation

### 6. Restarts

If premature convergence detected, restart with new random population.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Slow convergence | $F$ or $CR$ too low | Increase $F$ to 0.8, $CR$ to 0.9 |
| Premature convergence | Population too small | Increase $N_P$ to 15–20D |
| Oscillating fitness | $F$ too high | Decrease $F$ to 0.5–0.6 |
| Stuck in local minimum | Using `best1` strategy | Switch to `rand1` or `rand2` |

---

## Benchmark Results

Performance on standard test functions (D=30, $N_P=300$, 1000 generations):

| Function | Best Fitness | Iterations | Time (s) |
|----------|-------------|------------|----------|
| Sphere | 1.2e-28 | 412 | 0.8 |
| Rosenbrock | 2.4e-08 | 891 | 1.4 |
| Rastrigin | 4.1e-05 | 1000 | 2.1 |
| Ackley | 8.8e-15 | 623 | 1.2 |
| Griewank | 3.7e-12 | 548 | 1.0 |

---

## Advantages & Limitations

### Advantages

✅ No gradient information needed

✅ Handles non-convex, multimodal functions well

✅ Few parameters to tune

✅ Simple to implement and understand

✅ Robust across problem types

✅ Naturally handles box constraints

✅ Population maintains diversity

### Limitations

❌ Slower than gradient methods (when gradients are available)

❌ Scales poorly to high dimensions ($D > 100$)

❌ No convergence guarantees in finite time

❌ Requires many function evaluations

❌ Performance sensitive to parameter choices

---

## References

1. Storn, R. & Price, K. (1997). "Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces." *Journal of Global Optimization*, 11(4):341–359.

2. Price, K., Storn, R.M. & Lampinen, J.A. (2005). *Differential Evolution: A Practical Approach to Global Optimization*. Springer.

3. Das, S. & Suganthan, P.N. (2011). "Differential evolution: A survey of the state-of-the-art." *IEEE Transactions on Evolutionary Computation*, 15(1):4–31.

4. Brest, J. et al. (2006). "Self-adapting control parameters in differential evolution." *IEEE Trans. Evolutionary Computation*, 10(6):646–657. (jDE)

5. Tanabe, R. & Fukunaga, A. (2013). "Success-history based parameter adaptation for differential evolution." *IEEE CEC*, pp. 71–78. (SHADE)

---

## Related Topics

- [Grid Search](grid_search.md) – Exhaustive search for small parameter spaces
- [MCMC](mcmc.md) – Sampling-based inference for Bayesian optimization
- [Mean Field Games](mean_field_games.md) – Population dynamics optimization
