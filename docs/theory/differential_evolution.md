# Differential Evolution: Mathematical Theory

## Introduction

Differential Evolution (DE) is a population-based metaheuristic optimization algorithm introduced by Storn and Price (1997). It is particularly effective for continuous, non-convex, multimodal optimization problems where gradient information is unavailable or unreliable.

## Problem Formulation

### Objective

Minimize $f: \mathbb{R}^D \rightarrow \mathbb{R}$:

$$\min_{\mathbf{x} \in \mathbb{R}^D} f(\mathbf{x})$$

subject to box constraints:

$$x_j \in [l_j, u_j], \quad j = 1, ..., D$$

### Characteristics

**DE is suitable when**:
- $f$ is continuous but non-differentiable
- Multiple local minima exist
- Gradient information is unavailable or expensive
- Problem dimension is moderate ($D < 100$)

## Algorithm Overview

### Population

Maintain a population of $N_P$ candidate solutions:

$$P_g = \{\mathbf{x}_{1,g}, \mathbf{x}_{2,g}, ..., \mathbf{x}_{N_P,g}\}$$

where $g$ is the generation number and $\mathbf{x}_{i,g} \in \mathbb{R}^D$.

### Main Loop

For each generation $g = 0, 1, 2, ...$:

1. **Mutation**: Create mutant vectors
2. **Crossover**: Create trial vectors
3. **Selection**: Keep better solutions

## Mutation Strategies

### DE/rand/1 (Classic)

For each target vector $\mathbf{x}_{i,g}$, create mutant:

$$\mathbf{v}_{i,g+1} = \mathbf{x}_{r_1,g} + F \cdot (\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g})$$

where:
- $r_1, r_2, r_3 \in \{1, ..., N_P\}$ are randomly chosen, distinct, and $\neq i$
- $F \in (0, 2]$ is the **mutation factor** (typically 0.5-1.0)

**Interpretation**: 
- Start from a random population member $\mathbf{x}_{r_1}$
- Move in direction given by difference $(\mathbf{x}_{r_2} - \mathbf{x}_{r_3})$
- Scale movement by $F$

### DE/best/1

$$\mathbf{v}_{i,g+1} = \mathbf{x}_{\text{best},g} + F \cdot (\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g})$$

**Advantage**: Faster convergence

**Disadvantage**: More likely to get stuck in local minima

### DE/current-to-best/1

$$\mathbf{v}_{i,g+1} = \mathbf{x}_{i,g} + F \cdot (\mathbf{x}_{\text{best},g} - \mathbf{x}_{i,g}) + F \cdot (\mathbf{x}_{r_1,g} - \mathbf{x}_{r_2,g})$$

**Interpretation**: Move current solution toward best while exploring

### DE/rand/2

$$\mathbf{v}_{i,g+1} = \mathbf{x}_{r_1,g} + F \cdot (\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}) + F \cdot (\mathbf{x}_{r_4,g} - \mathbf{x}_{r_5,g})$$

More disruptive, better for highly multimodal problems.

## Crossover

### Binomial Crossover

For each component $j = 1, ..., D$:

$$u_{i,j,g+1} = \begin{cases}
v_{i,j,g+1} & \text{if } \text{rand}(0,1) \leq CR \text{ or } j = j_{\text{rand}} \\
x_{i,j,g} & \text{otherwise}
\end{cases}$$

where:
- $CR \in [0, 1]$ is the **crossover probability**
- $j_{\text{rand}} \in \{1, ..., D\}$ ensures at least one component is from mutant

**Effect**: Controls how much of the mutant vector is used

### Exponential Crossover

Copy consecutive components from mutant with probability $CR$.

Less common, similar performance to binomial.

## Selection

Greedy selection (for minimization):

$$\mathbf{x}_{i,g+1} = \begin{cases}
\mathbf{u}_{i,g+1} & \text{if } f(\mathbf{u}_{i,g+1}) \leq f(\mathbf{x}_{i,g}) \\
\mathbf{x}_{i,g} & \text{otherwise}
\end{cases}$$

**Property**: Population quality never decreases:
$$f(\mathbf{x}_{\text{best},g+1}) \leq f(\mathbf{x}_{\text{best},g})$$

## Complete Algorithm

```
1. Initialize population:
   For i = 1 to N_P:
       x_{i,0} = l + rand(0,1) · (u - l)
   
2. Evaluate fitness:
   f_i = f(x_{i,0}) for all i

3. While stopping criterion not met:
   
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

## Parameter Selection

### Population Size ($N_P$)

**Rule of thumb**: $N_P = 10D$ where $D$ is problem dimension

**Small population** (< 4D):
- Faster convergence
- Risk premature convergence
- Use for: simple unimodal problems

**Large population** (> 20D):
- Better exploration
- Slower convergence
- Use for: highly multimodal problems

**Minimum**: $N_P \geq 4$ (needed for mutation)

### Mutation Factor ($F$)

**Typical range**: $F \in [0.4, 1.0]$

**Low F** (0.4-0.6):
- Fine-tuning, local search
- Use near end of optimization
- Safer, less disruptive

**High F** (0.8-1.2):
- Exploration, global search
- Escape local minima
- More aggressive

**Adaptive F**: Some variants adjust $F$ during optimization

### Crossover Probability ($CR$)

**Typical range**: $CR \in [0.1, 0.9]$

**Low CR** (0.1-0.3):
- Less information exchange
- Slower convergence
- Use for: separable problems

**High CR** (0.7-0.9):
- More information exchange
- Faster convergence
- Use for: non-separable problems

**Special cases**:
- $CR = 0$: Pure mutation (except $j_{\text{rand}}$)
- $CR = 1$: Full crossover

### Stopping Criteria

1. **Maximum generations**: $g_{\max}$
2. **Function evaluations**: $FE_{\max}$
3. **Target fitness**: $f(\mathbf{x}_{\text{best}}) \leq f_{\text{target}}$
4. **Stagnation**: No improvement for $G_{\text{stag}}$ generations
5. **Diversity loss**: Population variance below threshold

## Convergence Analysis

### Theoretical Results

**Theorem** (Zaharie, 2002): Under certain conditions on $F$ and $CR$, DE converges to a stationary point.

**Conditions**:
- Bounded search space
- Continuous objective function
- Appropriate parameter settings

### Convergence Rate

**Empirical observations**:
- Linear convergence in early stages
- Slows down near optimum
- Faster than genetic algorithms for many problems
- Slower than gradient methods (when gradients available)

### No Free Lunch

DE is not universally optimal. Performance depends on:
- Problem landscape
- Parameter settings
- Population size

## Variants and Extensions

### Self-Adaptive DE (jDE)

Parameters $F$ and $CR$ evolve with the population:

$$F_{i,g+1} = \begin{cases}
F_l + \text{rand}(0,1) \cdot (F_u - F_l) & \text{if } \text{rand}(0,1) < \tau_1 \\
F_{i,g} & \text{otherwise}
\end{cases}$$

$$CR_{i,g+1} = \begin{cases}
\text{rand}(0,1) & \text{if } \text{rand}(0,1) < \tau_2 \\
CR_{i,g} & \text{otherwise}
\end{cases}$$

### SHADE (Success-History Adaptive DE)

Uses historical information about successful parameters.

### L-SHADE

SHADE with linear population size reduction.

### CoDE (Composite DE)

Uses multiple mutation strategies simultaneously.

### Opposition-Based DE

Initialize with both random solutions and their opposites.

### Constraint Handling

For constrained optimization:

1. **Penalty method**: Add penalty to objective
2. **Feasibility rules**: Prefer feasible solutions
3. **ε-constrained**: Relax constraints gradually

## Theoretical Properties

### Global Convergence

**Sufficient conditions** (Lampinen, 2001):
- Population size $N_P > 3$
- Mutation factor $F > 0$
- At least one component crossed over ($j_{\text{rand}}$)

Then DE is a **global optimization method**: Can reach any point with positive probability.

### Diversity Maintenance

Mutation creates diversity, selection reduces it. Balance determines exploration vs. exploitation.

**Diversity measure**:
$$D_g = \frac{1}{N_P D} \sum_{i=1}^{N_P} \sum_{j=1}^D |x_{i,j,g} - \bar{x}_{j,g}|$$

High diversity → exploration

Low diversity → exploitation

### Convergence Speed

**Expected number of generations** to reach near-optimum depends on:
- Problem difficulty (number of local minima, basin sizes)
- Population size
- Parameter settings

**Empirical rule**: Budget $10^4 D$ function evaluations for moderately difficult problems.

## Comparison with Other Algorithms

| Algorithm | Gradient | Global | Constraints | Speed | Best For |
|-----------|----------|--------|-------------|-------|----------|
| **DE** | No | Yes | Penalty | Medium | Non-convex, continuous |
| Gradient Descent | Yes | No | Yes | Fast | Smooth, convex |
| Genetic Algorithm | No | Yes | Yes | Slow | Discrete, combinatorial |
| Particle Swarm | No | Yes | Penalty | Fast | Continuous, many dims |
| Simulated Annealing | No | Yes | Penalty | Slow | Small problems |
| CMA-ES | No | Yes | Penalty | Fast | Continuous, noisy |

## Applications

### 1. Engineering Design

**Example**: Antenna design
- Objective: Maximize gain, minimize side lobes
- Constraints: Physical realizability
- High-dimensional, non-convex

### 2. Machine Learning

**Example**: Neural network hyperparameter tuning
- Objective: Validation accuracy
- Parameters: Learning rate, regularization, architecture
- Noisy, expensive evaluations

### 3. Chemical Engineering

**Example**: Reactor optimization
- Objective: Maximize yield, minimize cost
- Constraints: Safety, temperature, pressure
- Nonlinear dynamics

### 4. Portfolio Optimization

**Example**: Asset allocation
- Objective: Maximize Sharpe ratio
- Constraints: Budget, diversification
- Non-convex risk measures

### 5. System Identification

**Example**: Parameter estimation
- Objective: Minimize prediction error
- Parameters: Model coefficients
- Multimodal likelihood surface

## Computational Complexity

### Time Complexity

Per generation: $O(N_P \cdot D \cdot T_f)$

where $T_f$ is cost of evaluating $f$.

Total: $O(G_{\max} \cdot N_P \cdot D \cdot T_f)$

### Space Complexity

$O(N_P \cdot D)$ for population storage.

### Parallelization

**Embarrassingly parallel**: Each trial vector evaluation is independent.

**Speedup**: Near-linear with number of processors (up to $N_P$ processors).

## Practical Tips

### 1. Start Simple

Use default parameters: $N_P = 10D$, $F = 0.8$, $CR = 0.7$

### 2. Scale Variables

Normalize parameters to similar ranges for better performance.

### 3. Warm Start

If you have a good initial guess, seed population around it.

### 4. Hybrid Approach

Use DE for global search, then local optimizer for refinement:

```
1. Run DE for G_global generations
2. Take best solution x_best
3. Run local optimizer starting from x_best
```

### 5. Monitor Convergence

Plot:
- Best fitness vs. generation
- Average fitness vs. generation
- Population diversity vs. generation

### 6. Restarts

If premature convergence detected, restart with new random population.

## Advantages and Limitations

### Advantages

✅ No gradient information needed

✅ Handles non-convex, multimodal functions well

✅ Few parameters to tune

✅ Simple to implement

✅ Robust across problem types

✅ Naturally handles box constraints

✅ Population maintains diversity

### Limitations

❌ Slower than gradient methods (when gradients available)

❌ Scales poorly to high dimensions ($D > 100$)

❌ No convergence guarantees for finite time

❌ Requires many function evaluations

❌ Performance sensitive to parameters

❌ Difficult to handle complex constraints

❌ No theoretical optimal parameter settings

## Key References

1. **Storn, R., & Price, K.** (1997). *Differential evolution - A simple and efficient heuristic for global optimization over continuous spaces*. Journal of Global Optimization, 11(4), 341-359.
   - Original DE paper

2. **Price, K., Storn, R. M., & Lampinen, J. A.** (2005). *Differential Evolution: A Practical Approach to Global Optimization*. Springer.
   - Comprehensive book on DE

3. **Das, S., & Suganthan, P. N.** (2011). *Differential evolution: A survey of the state-of-the-art*. IEEE Transactions on Evolutionary Computation, 15(1), 4-31.
   - Survey of DE variants and applications

4. **Brest, J., Greiner, S., Bošković, B., Mernik, M., & Žumer, V.** (2006). *Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems*. IEEE Transactions on Evolutionary Computation, 10(6), 646-657.
   - jDE algorithm

5. **Tanabe, R., & Fukunaga, A.** (2013). *Success-history based parameter adaptation for differential evolution*. In IEEE Congress on Evolutionary Computation (pp. 71-78).
   - SHADE algorithm

6. **Qin, A. K., Huang, V. L., & Suganthan, P. N.** (2009). *Differential evolution algorithm with strategy adaptation for global numerical optimization*. IEEE Transactions on Evolutionary Computation, 13(2), 398-417.
   - Self-adaptive DE

## Summary

Differential Evolution is a powerful metaheuristic for global optimization:

**Key Features**:
- Population-based search
- Mutation, crossover, selection operators
- Self-organizing behavior

**Best suited for**:
- Non-convex, multimodal problems
- Moderate dimensions (< 100)
- When gradients unavailable
- Robust optimization needed

**Success factors**:
- Appropriate parameter settings
- Sufficient population size
- Adequate function evaluation budget

## See Also

- [Differential Evolution API Documentation](../differential_evolution.md) - Implementation and usage
- [Grid Search Theory](../grid_search.md) - Alternative for small spaces
- [MCMC Theory](mcmc.md) - Sampling-based inference
