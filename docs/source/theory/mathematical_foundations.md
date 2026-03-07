# Mathematical Foundations

This page develops the core mathematics underlying Optimiz-rs's Rust kernels — from first
principles through advanced theory.  Each section opens with a **definition block**, builds
intuition through **examples and diagrams**, and closes with a **notebook micro-check**.
For complete walkthroughs see `examples/notebooks/`.

---

## 1 · Differential Evolution (DE)

### Background

DE is a gradient-free population-based optimizer for $f: \mathbb{R}^d \to \mathbb{R}$,
not required to be smooth or convex.  At generation $g$ we maintain $N$ candidate
solutions $\{\mathbf{x}_{i,g}\} \subset \mathbb{R}^d$.

**Key insight:** The difference vector $\mathbf{x}_{r_2}-\mathbf{x}_{r_3}$ is an
unbiased directional finite-difference of $f$, so DE implicitly estimates curvature
without Jacobians.

### 1.1 Geometric Intuition — Mutation in $\mathbb{R}^2$

```{figure} ../_static/diagrams/fig_de_mutation.svg
:align: center
:alt: DE mutation geometry in R²
```

- $\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3$ are three **distinct** randomly selected parents.
- The mutant $\mathbf{v}_i$ lands on the other side relative to $\mathbf{x}_{r_1}$.
- **Crossover** then mixes $\mathbf{v}_i$ and $\mathbf{x}_i$ dimension-by-dimension with
  probability $CR$, producing trial vector $\mathbf{u}_i$.
- **Selection** keeps $\mathbf{u}_i$ only if it improves over $\mathbf{x}_i$ — pure greedy.

### 1.2 Operators

| Step | Formula | Role |
|------|---------|------|
| Mutation (rand/1) | $\mathbf{v}_{i,g} = \mathbf{x}_{r_1} + F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3})$ | explore |
| Binomial crossover | $u_{i,j} = v_{i,j}$ if $U(0,1)<CR$ or $j=j_\text{rand}$ | mix dimensions |
| Greedy selection | $\mathbf{x}_{i,g+1} = \mathbf{u}_{i,g}$ iff $f(\mathbf{u})\le f(\mathbf{x})$ | exploit |

**Convergence (informal):** Under bounded population diversity and Lipschitz $f$, the
best-so-far value converges a.s. to a stationary point as $N,g\to\infty$ (Price et al. 2005).

### 1.3 Self-Adaptive jDE (Optimiz-rs default)

Parameters $F,CR$ are per-individual and reset stochastically each generation:

$$
F_i^{g+1} = \begin{cases} F_{\min} + r_1 F_{\max} & r_2 < \tau_1,\\ F_i^g & \text{otherwise,}\end{cases}
\qquad
CR_i^{g+1} = \begin{cases} U(0,1) & r_3 < \tau_2,\\ CR_i^g & \text{otherwise.}\end{cases}
$$

$\tau_1=\tau_2=0.1$ by default.  On rugged landscapes this produces bimodal $F$
histograms concentrated near 0.8 — a sign the landscape is highly multimodal.

### 1.4 Example — Minimising the Rastrigin Function

The Rastrigin function $f(\mathbf{x}) = 10d + \sum_i[x_i^2 - 10\cos(2\pi x_i)]$
has $\approx 10^d$ local minima (global minimum $f^*=0$ at $\mathbf{x}^*=\mathbf{0}$).

**Why gradient methods fail:** The gradient $\partial_{x_i}f = 2x_i + 20\pi\sin(2\pi x_i)$
oscillates rapidly — any gradient step hops between basins.

**Why DE succeeds:** The difference vector $F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3})$
spans the characteristic basin width (~1.0), enabling inter-basin jumps.

```{figure} ../_static/diagrams/fig_rastrigin.svg
:align: center
:alt: Rastrigin function 1D — many local minima with one global optimum at zero
```

**Typical jDE convergence** ($d=10$, $N=100$, $\tau_1=\tau_2=0.1$):

```
Gen    Best f    Mean F    Mean CR
----   -------   -------   -------
  1     48.3      0.50      0.50
 50     12.1      0.78      0.31
200      3.4      0.82      0.24   <- F clusters near 0.8 (bimodal)
500      0.0      0.83      0.22   <- converged
```

::::{admonition} Tip — Diagnosing Stagnation
:class: tip

If best-$f$ does not decrease for 100+ generations:

1. **Check $F$ histogram.** Bimodal near 0.8 -> landscape is multimodal (increase $N$).
   Collapsed near 0 -> diversity loss; restart with random perturbation.
2. **Check $CR$ distribution.** Uniform -> dimensions not interacting.
   Collapsed near 0 -> DE treating dimensions independently (separable function).
3. **Increase $N$** to $\approx 10d$ for $d > 20$.
::::

**Notebook check** (`05_performance_benchmarks.ipynb`): Plot $F_i, CR_i$ histograms
every 50 generations; expect values clustering in $[0.5,0.9]$ on hard problems.

---

## 2 · Stochastic Processes

These form the probabilistic backbone of all continuous-time models in Optimiz-rs.
We build the theory from scratch: random walk → Brownian motion → Itō calculus → SDEs → jump-diffusions.

---

### 2.1 Brownian Motion

#### 2.1.0 Intuitive Construction — From Random Walk to BM

**Step 1 — Discrete random walk.**  Flip a fair coin $n$ times per unit time.
Define $\xi_k = +1$ (heads) or $-1$ (tails) i.i.d.  After $n$ steps of size $1/\sqrt{n}$:

$$S^{(n)}_t = \frac{1}{\sqrt{n}}\sum_{k=1}^{\lfloor nt \rfloor} \xi_k.$$

By the **Central Limit Theorem**, as $n\to\infty$: $S^{(n)}_t \xrightarrow{d} W_t \sim \mathcal{N}(0,t)$.

```{figure} ../_static/diagrams/fig_random_walk.svg
:align: center
:alt: Coin-flip random walk converging to Brownian motion as n grows
```

**Step 2 — Scaling limit.**  The normalization $1/\sqrt{n}$ is crucial:
- Without it, variance grows as $n$ (diverges).
- With $n^{-1/2}$: variance = $n \cdot (1/\sqrt{n})^2 \cdot t = t$ — exactly right.

This is why $W_t \sim \mathcal{N}(0,t)$: **variance accumulates linearly in time**.

::::{admonition} Definition — Wiener Process
:class: definition

A stochastic process $W = (W_t)_{t\ge 0}$ on $(\Omega,\mathcal{F},\mathbb{P})$
is a *standard Brownian motion* if:

1. $W_0 = 0$ a.s.
2. Increments are **independent**: $W_t - W_s \perp \mathcal{F}_s$ for $t>s$.
3. $W_t - W_s \sim \mathcal{N}(0, t-s)$ for all $0\le s<t$.
4. Paths $t\mapsto W_t(\omega)$ are **continuous** a.s.
::::

#### 2.1.1 Key Analytical Properties

| Property | Formula | Intuition |
|----------|---------|-----------|
| Mean | $\mathbb{E}[W_t] = 0$ | No drift — symmetric random walk |
| Variance | $\operatorname{Var}(W_t) = t$ | Uncertainty grows with time |
| Covariance | $\operatorname{Cov}(W_s,W_t) = \min(s,t)$ | Shared history up to first time |
| Non-differentiability | $\lim_{h\to 0}(W_{t+h}-W_t)/h$ diverges a.s. | Too "rough" for ordinary calculus |
| Quadratic variation | $[W]_T = T$ | Core source of Itō correction term |
| Self-similarity | $c^{-1/2}W_{ct} \overset{d}{=} W_t$ | Fractal structure, Hurst $H=\tfrac12$ |

**Quadratic variation derivation (step by step):**

Partition $[0,T]$ into $n$ pieces of width $\Delta = T/n$.  Sum of squared increments:

$$\sum_{k=0}^{n-1}(W_{t_{k+1}}-W_{t_k})^2 \overset{?}{=} T \quad \text{as } n\to\infty.$$

**Step 1** — Each increment: $(W_{t_{k+1}}-W_{t_k})^2 \sim \Delta \cdot \chi_1^2$, so
$\mathbb{E}[(W_{t_{k+1}}-W_{t_k})^2] = \Delta$.

**Step 2** — Sum of means: $\sum_{k=0}^{n-1} \Delta = n\Delta = T$.

**Step 3** — Variance of the sum:
$\operatorname{Var}\!\left(\sum (W_{t_{k+1}}-W_{t_k})^2\right) = n \cdot 2\Delta^2 = 2T^2/n \xrightarrow{n\to\infty} 0$.

**Conclusion:** $\sum (W_{t_{k+1}}-W_{t_k})^2 \xrightarrow{L^2} T$. We write $dW_t^2 = dt$.  
This **single identity** is the engine of all Itō calculus.

::::{admonition} Why dW² = dt is remarkable
:class: tip

In ordinary calculus, $dx^2 \approx dx \cdot dx \to 0$ (second-order infinitesimal).  
For Brownian motion, $(dW)^2 = dt$ is **first-order** — it does **not** vanish!

Physically: BM paths oscillate so rapidly ($\sim t^{0.5}$ scale) that their squared 
increments accumulate at rate $1$ — comparable to the drift $dt$.

This is the **only** reason Itō's lemma has an extra term.
::::

**Multiple sample paths** — the fan widens as $\propto\sqrt{t}$:

```{figure} ../_static/diagrams/fig_bm_fan.svg
:align: center
:alt: Brownian motion fan — multiple sample paths widening as sqrt(t)
```

**Example — Geometric BM:**
$S_t = S_0 \exp\!\bigl((\mu-\tfrac12\sigma^2)t + \sigma W_t\bigr)$
is the Black-Scholes price model.  Log-normal marginals; continuous, nowhere-differentiable paths:

```{figure} ../_static/diagrams/fig_gbm.svg
:align: center
:alt: Geometric Brownian motion — log-normal price paths with drift and volatility
```

### 2.2 Itō Calculus

#### 2.2.0 Why You Cannot Use Ordinary Integration

Attempt to define $\int_0^T W_t\,dW_t$ using a Riemann sum: pick $W_{t_k}$ at the
**left endpoint** → get one answer; pick $(W_{t_k}+W_{t_{k+1}})/2$ (midpoint) → get a *different* answer.

This ambiguity occurs because $W$ is not of bounded variation.  **Itō's convention** 
(left endpoint) is the only one that produces a **martingale** — ensuring no look-ahead.

::::{admonition} Definition — Itō Integral
:class: definition

For adapted $f \in \mathcal{L}^2$ (i.e. $\mathbb{E}\!\int_0^T f_t^2\,dt < \infty$):

$$\int_0^T f_t\,dW_t \;:=\; L^2\text{-}\lim_{|\pi|\to 0} \sum_{k} f_{t_k}(W_{t_{k+1}}-W_{t_k}).$$

Key guarantees:
- **Zero mean:** $\mathbb{E}\!\left[\int_0^T f_t\,dW_t\right] = 0$.
- **Itō isometry:** $\mathbb{E}\!\left[\left(\int_0^T f_t\,dW_t\right)^2\right] = \mathbb{E}\!\int_0^T f_t^2\,dt$.
- **Martingale:** $M_t = \int_0^t f_s\,dW_s$ satisfies $\mathbb{E}[M_t\mid\mathcal{F}_s]=M_s$.
::::

**Itō isometry — proof sketch:**  
Let $I_T = \sum_k f_{t_k}\Delta W_k$ (simple process).  Then:

$$\mathbb{E}[I_T^2] = \sum_{j,k}\underbrace{\mathbb{E}[f_{t_j}\Delta W_j \cdot f_{t_k}\Delta W_k]}_{\text{cross terms}}$$

For $j \neq k$ (say $j < k$): $f_{t_j}\Delta W_j$ and $f_{t_k}$ are both $\mathcal{F}_{t_k}$-measurable,
while $\Delta W_k$ is **independent** of $\mathcal{F}_{t_k}$ with mean 0 → cross term $= 0$.

For $j = k$: $\mathbb{E}[f_{t_j}^2 (\Delta W_j)^2] = \mathbb{E}[f_{t_j}^2]\Delta t_j$ (independence of $f_{t_j}$ and $\Delta W_j$).

$$\Rightarrow \mathbb{E}[I_T^2] = \sum_k \mathbb{E}[f_{t_k}^2]\Delta t_k \xrightarrow{|\pi|\to 0} \mathbb{E}\int_0^T f_t^2\,dt. \quad \checkmark$$

#### 2.2.1 Itō's Lemma — Full Derivation

::::{admonition} Theorem — Itō's Lemma
:class: tip

For $dX_t = \mu_t\,dt + \sigma_t\,dW_t$ and $F \in C^{1,2}([0,T]\times\mathbb{R})$:

$$\boxed{dF(t,X_t) = \partial_t F\,dt + \partial_x F\,dX_t + \tfrac{1}{2}\sigma_t^2\,\partial_{xx}F\,dt}$$

Expanded:

$$dF = \underbrace{\left(\partial_t F + \mu_t\,\partial_x F + \tfrac12\sigma_t^2\,\partial_{xx}F\right)}_{\text{drift}}\,dt
+ \underbrace{\sigma_t\,\partial_x F}_{\text{diffusion}}\,dW_t.$$
::::

**Derivation — Taylor expand $F(t+dt, X_{t+dt})$:**

$$dF = \partial_t F\,dt + \partial_x F\,dX + \tfrac12\partial_{xx}F\,(dX)^2 + \underbrace{\partial_{tx}F\,dt\,dX + \ldots}_{\to 0} $$

Compute $(dX)^2$ using the **Itō multiplication table**:

| × | $dt$ | $dW_t$ |
|---|------|--------|
| $dt$ | $0$ | $0$ |
| $dW_t$ | $0$ | $dt$ |

$$\begin{aligned}
(dX_t)^2 &= (\mu_t\,dt + \sigma_t\,dW_t)^2 \\
          &= \mu_t^2\underbrace{(dt)^2}_{0} + 2\mu_t\sigma_t\underbrace{dt\cdot dW_t}_{0} + \sigma_t^2\underbrace{(dW_t)^2}_{dt}\\
          &= \sigma_t^2\,dt.
\end{aligned}$$

Substituting:

$$dF = \partial_t F\,dt + \partial_x F(\mu_t\,dt + \sigma_t\,dW_t) + \tfrac12\partial_{xx}F\cdot\sigma_t^2\,dt$$

$$= \left(\partial_t F + \mu_t\partial_x F + \tfrac12\sigma_t^2\partial_{xx}F\right)dt + \sigma_t\partial_x F\,dW_t. \quad \checkmark$$

**The extra term $\tfrac12\sigma^2\partial_{xx}F\,dt$ is the "Itō correction".**  
In ordinary calculus $(dW)^2=0$, so it vanishes.  In stochastic calculus, BM oscillates
so rapidly that $(dW)^2 = dt$ — a first-order effect.

**Multidimensional version** (for vector $\mathbf{X}\in\mathbb{R}^n$, matrix noise):

$$dF = \partial_t F\,dt + \sum_i \partial_{x_i}F\,dX_i + \tfrac12\sum_{i,j}\partial_{x_ix_j}F\,d[X_i,X_j]_t$$

where $d[X_i, X_j]_t = d\langle X_i, X_j\rangle_t$ is the quadratic co-variation.

#### 2.2.2 Worked Examples of Itō's Lemma

**Example 1 — GBM, derive explicit solution:**

SDE: $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$.

**Goal:** Find $S_t$ in closed form.

**Step 1** — Guess $F(t,x) = \log x$.  Compute partials:
$\partial_t F = 0$, $\partial_x F = 1/x$, $\partial_{xx}F = -1/x^2$.

**Step 2** — Apply Itō's lemma:

$$d(\log S_t) = 0 + \frac{1}{S_t}\,dS_t + \tfrac12\cdot(-\tfrac{1}{S_t^2})\cdot\sigma^2 S_t^2\,dt$$

$$= \frac{\mu S_t\,dt + \sigma S_t\,dW_t}{S_t} - \tfrac12\sigma^2\,dt$$

$$= \left(\mu - \tfrac12\sigma^2\right)\,dt + \sigma\,dW_t.$$

**Step 3** — Integrate (deterministic integral + Itō integral):

$$\log S_T - \log S_0 = \left(\mu-\tfrac12\sigma^2\right)T + \sigma W_T.$$

**Step 4** — Exponentiate:

$$\boxed{S_T = S_0\exp\!\left[\left(\mu - \tfrac12\sigma^2\right)T + \sigma W_T\right].}$$

The **Itō correction** $-\tfrac12\sigma^2 T$ lowers the expected log-return:
$\mathbb{E}[\log S_T] = \log S_0 + (\mu-\tfrac12\sigma^2)T$, 
but $\mathbb{E}[S_T] = S_0 e^{\mu T}$ (Jensen's inequality explains the gap: 
$e^{\mathbb{E}[X]} < \mathbb{E}[e^X]$ for non-degenerate $X$).

```{figure} ../_static/diagrams/fig_ito_correction.svg
:align: center
:alt: Itō correction — expected log-return is always below the naive slope mu
```

**Example 2 — Itō product rule ($d(X_t Y_t)$):**

By Itō's lemma applied to $F(x,y) = xy$:

$$d(X_t Y_t) = Y_t\,dX_t + X_t\,dY_t + d[X,Y]_t$$

where $d[X,Y]_t = \sigma_X\sigma_Y\,dt$.  Compare to ordinary calculus: $d(xy) = y\,dx + x\,dy$ (no cross term because $(dx)^2=0$).

**Example 3 — Integration by parts for stochastic integrals:**

$$\int_0^T W_t\,dW_t = \tfrac12 W_T^2 - \tfrac12 T.$$

Ordinary calculus would give $\int_0^T W_t\,dW_t = \tfrac12 W_T^2$.
The $-\tfrac12 T$ correction comes from the quadratic variation.

**Verification via Itō's lemma:** Set $F(t,x) = x^2/2$:
$dF = x\,dW + \tfrac12\cdot 1 \cdot dt = W_t\,dW_t + \tfrac12\,dt$.
Integrate: $\tfrac12 W_T^2 - 0 = \int_0^T W_t\,dW_t + \tfrac12 T$ → result follows. ✓

#### 2.2.3 Itō vs Stratonovich

| Property | Itō integral | Stratonovich integral ($\circ$) |
|----------|-------------|----------------------|
| Chain rule | Modified ($+\tfrac12\sigma^2\partial_{xx}F$ term) | Standard calculus chain rule |
| Martingale | Yes (if $f$ adapted) | No in general |
| Use in finance | Natural (no look-ahead) | Physics, geometry |
| Conversion | $\int f\circ dW = \int f\,dW + \tfrac12\int \partial_x f\,\sigma\,dt$ | (same identity) |
| SDE solutions | Different numerics needed | Standard ODE methods work |

**Conversion formula** — Itō $\to$ Stratonovich:

$$\int_0^T f(X_t)\circ dW_t = \int_0^T f(X_t)\,dW_t + \tfrac{1}{2}\int_0^T f'(X_t)\sigma_t\,dt.$$

**Rule of thumb:** Use Itō in finance (causality, no-arbitrage); use Stratonovich in 
physics/differential geometry (coordinate-invariant chain rule).

### 2.3 General Itō SDEs

$$dX_t = b(t, X_t)\,dt + \boldsymbol{\sigma}(t, X_t)\,dW_t,\quad X_0 = x_0.$$

#### 2.3.0 Existence, Uniqueness and Picard Iteration

::::{admonition} Theorem — Strong Solution Existence (Picard–Lindelöf for SDEs)
:class: tip

If $b$ and $\sigma$ are **globally Lipschitz** in $x$ (uniformly in $t$):
$\|b(t,x)-b(t,y)\| + \|\sigma(t,x)-\sigma(t,y)\| \le L\|x-y\|$,

and satisfy **linear growth**: $\|b(t,x)\|^2 + \|\sigma(t,x)\|^2 \le C^2(1+\|x\|^2)$,

then there exists a **unique strong solution** with $\mathbb{E}\!\left[\sup_{t\le T}\|X_t\|^2\right] < \infty$.
::::

**Picard iteration — construct the solution step by step:**

Set $X_t^{(0)} = x_0$ (constant).  For $n\ge 0$:

$$X_t^{(n+1)} = x_0 + \int_0^t b(s, X_s^{(n)})\,ds + \int_0^t \sigma(s, X_s^{(n)})\,dW_s.$$

**Intermediate step — bound the error:**

Let $\varepsilon_n(t) = \mathbb{E}\!\left[\sup_{s\le t}|X_s^{(n+1)}-X_s^{(n)}|^2\right]$.

By Doob's $L^2$-inequality and Lipschitz:

$$\varepsilon_{n+1}(t) \le 2(L^2 T + L^2)\int_0^t \varepsilon_n(s)\,ds.$$

By induction: $\varepsilon_n(t) \le C \cdot \frac{(2L^2(T+1)t)^n}{n!} \to 0$. 
Geometric series → $X^{(n)}$ is Cauchy in $L^2$ → converges to the unique solution.

**Intuition:**

```{figure} ../_static/diagrams/fig_picard.svg
:align: center
:alt: Picard iteration — successive approximations converging to the true SDE solution
```

#### 2.3.1 The Fokker-Planck Equation — How Densities Evolve

If $X_t$ has density $p(t,x)$, then $p$ satisfies the **Fokker-Planck (Kolmogorov forward) PDE**:

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}[b(t,x)\,p] + \frac{1}{2}\frac{\partial^2}{\partial x^2}[\sigma^2(t,x)\,p].$$

**Derivation sketch:** For any test function $\phi$:

$$\frac{d}{dt}\mathbb{E}[\phi(X_t)] = \mathbb{E}[\mathcal{L}\phi(X_t)] = \mathbb{E}\!\left[b\,\phi' + \tfrac12\sigma^2\phi''\right]$$

using Itō's lemma on $\phi(X_t)$.  Integration by parts in the $x$-integral transfers 
derivatives from $\phi$ to $p$, giving the Fokker-Planck equation.

**Visual — density flows rightward (positive drift) and spreads (positive diffusion):**

```{figure} ../_static/diagrams/fig_fokker_planck.svg
:align: center
:alt: Fokker-Planck evolution — probability density drifts right and broadens over time
```

**For OU: $b = \kappa(\theta-x)$, $\sigma$ = const** →
stationary solution $p_\infty(x) = \mathcal{N}(\theta, \sigma^2/2\kappa)$.

#### 2.3.2 Common SDE Reference Table

| Process | SDE | Closed-form $X_t$ | Stationary dist. | Use in Optimiz-rs |
|---------|-----|-------------------|-----------------|-------------------|
| Brownian motion | $dX = \sigma\,dW$ | $X_0 + \sigma W_t$ | — | Noise baseline |
| Geometric BM | $dX = \mu X\,dt + \sigma X\,dW$ | $X_0 e^{(\mu-\sigma^2/2)t+\sigma W_t}$ | Log-normal | Price model |
| Ornstein-Uhlenbeck | $dX = \kappa(\theta-X)\,dt + \sigma\,dW$ | (see §2.4) | $\mathcal{N}(\theta, \sigma^2/2\kappa)$ | Spread model |
| CIR | $dX = \kappa(\theta-X)\,dt + \sigma\sqrt{X}\,dW$ | (Bessel process) | Gamma$(2\kappa\theta/\sigma^2, \sigma^2/2\kappa)$ | Volatility, rates |
| SABR | $dF = \sigma F^\beta dW^1$, $d\sigma = \nu\sigma\,dW^2$ | (no closed form) | — | Volatility model |

#### 2.3.3 Numerical Schemes for SDEs

When no closed form exists, discretize with step $\Delta t$:

**Euler-Maruyama** (simplest, strong order 0.5):

$$X_{t+\Delta t} \approx X_t + b(t,X_t)\,\Delta t + \sigma(t,X_t)\,\Delta W_t$$

where $\Delta W_t = \sqrt{\Delta t}\,Z$, $Z\sim\mathcal{N}(0,1)$.

**Milstein** (includes first-order Itō correction, strong order 1.0):

$$X_{t+\Delta t} \approx X_t + b\,\Delta t + \sigma\,\Delta W_t + \tfrac12\sigma\,\sigma_x\bigl[(\Delta W_t)^2 - \Delta t\bigr].$$

The extra term $\tfrac12\sigma\sigma_x[(\Delta W_t)^2 - \Delta t]$ comes from applying Itō's lemma to $\sigma(X_t)dW_t$.

```{figure} ../_static/diagrams/fig_em_milstein.svg
:align: center
:alt: Strong convergence comparison — Euler-Maruyama order 1/2 vs Milstein order 1
```

### 2.4 Ornstein-Uhlenbeck (Mean-Reversion)

Used in Optimiz-rs's `sparse_mean_reversion` and `ou_estimator` modules:

$$dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t.$$

**Intuition — restoring force:** The drift is a spring pulling $X_t$ back to $\theta$:

```{figure} ../_static/diagrams/fig_ou_path.svg
:align: center
:alt: Ornstein-Uhlenbeck path — mean-reverting diffusion with stationary confidence bands
```

#### 2.4.1 Closed-Form Solution — Step by Step

**Step 1 — Integrating factor.**  Rewrite the SDE as:

$$dX_t + \kappa X_t\,dt = \kappa\theta\,dt + \sigma\,dW_t.$$

Multiply both sides by the integrating factor $e^{\kappa t}$ and recognize the left-hand side:

$$d\!\left(e^{\kappa t}X_t\right) = e^{\kappa t}dX_t + \kappa e^{\kappa t}X_t\,dt = e^{\kappa t}\kappa\theta\,dt + e^{\kappa t}\sigma\,dW_t.$$

(Here we used Itō's product rule: $d(e^{\kappa t}X_t) = e^{\kappa t}dX_t + X_t\cdot\kappa e^{\kappa t}dt$ — no quadratic variation cross term since $e^{\kappa t}$ is deterministic.)

**Step 2 — Integrate both sides from $0$ to $t$:**

$$e^{\kappa t}X_t - X_0 = \kappa\theta\int_0^t e^{\kappa s}\,ds + \sigma\int_0^t e^{\kappa s}\,dW_s$$

$$e^{\kappa t}X_t - X_0 = \theta(e^{\kappa t} - 1) + \sigma\int_0^t e^{\kappa s}\,dW_s.$$

**Step 3 — Divide by $e^{\kappa t}$:**

$$\boxed{X_t = \theta + (X_0 - \theta)e^{-\kappa t} + \sigma\int_0^t e^{-\kappa(t-s)}\,dW_s.}$$

**Interpretation of each term:**

| Term | Meaning |
|------|---------|
| $\theta$ | Long-run equilibrium (the "anchor") |
| $(X_0-\theta)e^{-\kappa t}$ | Deterministic decay: initial displacement shrinks at rate $\kappa$ |
| $\sigma\int_0^t e^{-\kappa(t-s)}dW_s$ | Stochastic part: weighted sum of all past noise shocks, with **exponential forgetting** |

The stochastic integral $I_t = \sigma\int_0^t e^{-\kappa(t-s)}dW_s$ is a **Gaussian** random variable 
(linear functional of Brownian motion) with:

$$\mathbb{E}[I_t] = 0, \qquad \operatorname{Var}(I_t) = \sigma^2\int_0^t e^{-2\kappa(t-s)}\,ds = \frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t}).$$

**Step 4 — Marginal distribution:**

$$X_t \sim \mathcal{N}\!\left(\theta + (X_0-\theta)e^{-\kappa t},\;\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t})\right).$$

As $t\to\infty$: $X_t \to \mathcal{N}(\theta, \sigma^2/2\kappa)$ — the stationary distribution.

#### 2.4.2 Transition Density (Conditional on $X_s$)

$$X_t \mid X_s \sim \mathcal{N}\!\left(\theta + (X_s-\theta)e^{-\kappa(t-s)},\;\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa(t-s)})\right), \quad t > s.$$

This is exact (no approximation) because the OU process is **linear**.  Key formulas:

$$\hat\mu(\tau) = \theta + (X_s-\theta)e^{-\kappa\tau}, \qquad \hat\sigma^2(\tau) = \frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\tau}), \quad \tau=t-s.$$

```{figure} ../_static/diagrams/fig_ou_transition.svg
:align: center
:alt: OU transition density — distribution shifts toward theta and broadens with time
```

#### 2.4.3 Half-Life and Mean-Reversion Speed

**Half-life:** $\tau_{1/2} = \ln 2/\kappa$ — time for the initial displacement to halve.

| $\kappa$ (per year) | Half-life | Typical use |
|--------------------|-----------|-------------|
| 0.2 | 3.5 yr | Long-term macro factors |
| 10 | 25 days | Cross-sectional equity spreads |
| 55 | 4.6 days | Short-term pair spreads |
| 252 | 1 trading day | Intraday alpha signals |

**MLE log-likelihood** (discrete observations at spacing $\Delta t$):

$$\ell(\kappa,\theta,\sigma) = -\frac{1}{2}\sum_{i=1}^{n}\left[\log(2\pi\hat\sigma^2) + \frac{(X_{t_i} - \hat\mu_i)^2}{\hat\sigma^2}\right],$$

where $\hat\mu_i = \theta + (X_{t_{i-1}}-\theta)e^{-\kappa\Delta t}$ and $\hat\sigma^2 = \frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta t})$.

**Score equations** (differentiate $\ell$ and set to zero):

$$\frac{\partial\ell}{\partial\theta} = \sum_i \frac{X_{t_i}-\hat\mu_i}{\hat\sigma^2}(1-e^{-\kappa\Delta t}) = 0,$$

$$\frac{\partial\ell}{\partial\kappa} = \sum_i \frac{(X_{t_i}-\hat\mu_i)}{\hat\sigma^2}(X_{t_{i-1}}-\theta)\Delta t\,e^{-\kappa\Delta t} - \sum_i \frac{\partial\log\hat\sigma^2}{\partial\kappa} = 0.$$

These are nonlinear in $\kappa$; Optimiz-rs solves them with DE (`ou_estimator::fit_mle()`).

::::{admonition} Example — Calibrating OU to an Equity-Pair Spread
:class: note

**Data:** Daily log-spread $X_t = \log(P_A / P_B)$ for a co-integrated pair,
$n=250$ observations, $\Delta t=1/252$ years.

**Step 1 — MLE:** Maximize $\ell(\kappa, \theta, \sigma)$ using `ou_estimator::fit_mle()`.

**Step 2 — Intermediate verification:** The OU log-likelihood surface:

```{figure} ../_static/diagrams/fig_ou_loglik.svg
:align: center
:alt: OU log-likelihood surface — kappa broadly identified, theta tightly localised
```

**Typical results:**

| Parameter | Estimate | Interpretation |
|-----------|----------|----------------|
| $\hat\kappa$ | 55/yr | half-life approx 4.6 days |
| $\hat\theta$ | 0.003 | long-run spread approx 0.3% |
| $\hat\sigma$ | 0.12/yr$^{0.5}$ | daily spread vol approx 0.75% |

**Step 3 — Diagnostic:**

Standardized residuals: $r_i = (X_{t_i} - \hat\mu_i)/\hat\sigma$ should be $\mathcal{N}(0,1)$.

```{figure} ../_static/diagrams/fig_ou_residuals.svg
:align: center
:alt: OU residual diagnostics — standardised residuals histogram vs N(0,1)
```

Ljung-Box test: checks for remaining autocorrelation in $r_i$.

**Step 4 — Trading signal:**
Enter when $|X_t - \hat\theta| > 2\hat\sigma_\infty$ where $\hat\sigma_\infty = \hat\sigma/\sqrt{2\hat\kappa}$.
Exit at $X_t = \hat\theta$.  Expected holding time $\approx \hat\tau_{1/2} = \ln 2/\hat\kappa \approx 4.6$ days.

**P&L decomposition:**
- Gross expected profit per trade $\approx 2\hat\sigma_\infty = 2\hat\sigma/\sqrt{2\hat\kappa}$.
- Transaction costs must be $< 2\hat\sigma_\infty$ for profitability.
::::

---

## 3 · Jump Processes

Many financial time series exhibit sudden large moves that Brownian motion cannot capture.

### 3.1 Poisson Process

::::{admonition} Definition — Poisson Process
:class: definition

A counting process $N = (N_t)_{t\ge 0}$ is a *Poisson process with
intensity* $\lambda > 0$ if:

1. $N_0 = 0$.
2. Independent, stationary increments.
3. $\mathbb{P}(N_{t+h}-N_t=1) = \lambda h + o(h)$ and $\mathbb{P}(\Delta N > 1) = o(h)$.
::::

Equivalently, $N_t \sim \text{Poisson}(\lambda t)$ and inter-arrival times are
$\text{Exp}(\lambda)$.  The *compensated* process $\tilde N_t = N_t - \lambda t$
is a martingale.

**Sample path — step function with random jumps ($\lambda=2$ per unit time):**

```{figure} ../_static/diagrams/fig_poisson.svg
:align: center
:alt: Poisson process sample path — step function with random jump times
```

### 3.2 Compound Poisson Jump-Diffusion (Merton 1976)

$$\frac{dS_t}{S_{t^-}} = \mu\,dt + \sigma\,dW_t + d\Bigl(\sum_{k=1}^{N_t}(e^{J_k}-1)\Bigr),$$

with $N_t$ Poisson($\lambda$) and $J_k \sim \mathcal{N}(\mu_J, \sigma_J^2)$.

**Sample path — smooth diffusion interrupted by sudden jumps:**

```{figure} ../_static/diagrams/fig_jump_diffusion.svg
:align: center
:alt: Merton jump-diffusion path — GBM with sudden discontinuous jumps
```

**Merton option price** — Poisson mixture of Black-Scholes prices:

$$C_{\text{Merton}} = \sum_{n=0}^\infty \frac{e^{-\lambda' T}(\lambda' T)^n}{n!}
\cdot C_{\text{BS}}\!\left(S_0, K, T, r_n, \sigma_n^2\right),$$

where $\lambda' = \lambda e^{\mu_J+\frac12\sigma_J^2}$,
$r_n = r - \lambda(e^{\mu_J+\frac12\sigma_J^2}-1) + n(\mu_J+\tfrac12\sigma_J^2)/T$,
and $\sigma_n^2 = \sigma^2 + n\sigma_J^2/T$.

**Intuition:** Condition on exactly $n$ jumps occurring (probability $e^{-\lambda' T}(\lambda' T)^n/n!$).
In that scenario the world is a BS world with adjusted drift $r_n$ and total variance
$\sigma^2 T + n\sigma_J^2$. Average over the Poisson distribution of $n$.

::::{admonition} Example — Fitting Merton to a Crash Event
:class: note

**Observed:** S&P 500, March 2020.  Implied vol surface shows a vol smile —
OTM puts are expensive (fat left tail), which pure BS cannot explain.

**Merton calibration** (4 parameters: $\sigma, \lambda, \mu_J, \sigma_J$):

| Parameter | Estimated value | Interpretation |
|-----------|----------------|----------------|
| $\sigma$ | 0.18/yr | baseline diffusion vol |
| $\lambda$ | 3/yr | approx 3 crash events per year |
| $\mu_J$ | -0.12 | average log-jump = -12% |
| $\sigma_J$ | 0.08 | jump size std = 8% |

**Fitting procedure:**

1. Collect implied vols for strikes $K$ and maturities $T$.
2. Minimise $\sum_{K,T}(C_{\text{Merton}}(K,T;\theta) - C_{\text{market}})^2$
   via `differential_evolution` (DE is ideal — 4 params, non-convex landscape).
3. **Diagnostic:** Plot Merton vs market smile; expect fit within 0.5 vega.

**Result:** Negative $\mu_J$ captures left-tail skew, explaining costly OTM puts.
::::

### 3.3 Levy Processes and the Levy-Khintchine Representation

::::{admonition} Theorem — Levy-Khintchine
:class: tip

Every Levy process (independent stationary increments) has characteristic function

$$\mathbb{E}[e^{i\xi X_t}] = \exp\!\Bigl(t\Bigl[i b\xi - \tfrac{1}{2}\sigma^2\xi^2
+ \int_{\mathbb{R}\setminus\{0\}} \bigl(e^{i\xi z}-1-i\xi z\mathbf{1}_{|z|\le1}\bigr)\nu(dz)\Bigr]\Bigr)$$

where $(b, \sigma^2, \nu)$ is the *Levy triplet* and $\nu$ the *Levy measure*,
satisfying $\int(1\wedge z^2)\nu(dz)<\infty$.
::::

**Levy measure tail shapes:**

```{figure} ../_static/diagrams/fig_levy_tails.svg
:align: center
:alt: Lévy measure tail comparison — power-law vs Gaussian tails on log scale
```

**Levy Process Zoo**

| Process | Levy measure $\nu$ | Use case |
|---------|-------------------|----------|
| Brownian motion | $\nu=0$ | continuous diffusion |
| Compound Poisson | finite measure | rare large jumps |
| Variance Gamma | $\nu(dz)\propto e^{-c|z|}/|z|$ | equity returns |
| CGMY | power-law with exponential cutoff | heavy tails, $Y\in(0,2)$ |
| $\alpha$-stable | $c|z|^{-1-\alpha}$ | infinite-variance regimes |

### 3.4 SDEs with Jumps — Generator and Ito Formula

$$dX_t = b(X_{t^-})\,dt + \sigma(X_{t^-})\,dW_t
+ \int_{\mathbb{R}} c(X_{t^-}, z)\,\tilde N(dt, dz),$$

where $\tilde N(dt,dz) = N(dt,dz) - \nu(dz)\,dt$ is the *compensated jump measure*.

**Ito formula for jump-diffusions:**

$$dF(X_t) = \mathcal{L}F\,dt + \partial_x F\,\sigma\,dW_t
+ \int\bigl[F(X_{t^-}+c)-F(X_{t^-})\bigr]\tilde N(dt,dz),$$

where the *generator* is

$$\mathcal{L}F = b\,\partial_x F + \tfrac12\sigma^2\partial_{xx}F
+ \int\bigl[F(x+c)-F(x)-c\,\partial_x F\bigr]\nu(dz).$$

---

## 4 · Optimal Control (HJB, PMP, Jumps)

**Big picture.** Optimal control asks: *given a stochastic system we can steer with a
control $u_t$, what policy minimises expected cost?*  Three complementary tools answer this:

| Tool | Solves | Scales to | Intuition |
|------|--------|-----------|-----------|
| HJB PDE | Value function $V(t,x)$ | Low dim (PDE grid) | Dynamic programming |
| PMP | Optimal paths $(X_t,p_t)$ | High dim (ODE) | Adjoint sensitivity |
| HJBI | Same as HJB + jumps | Low dim | Non-local integral term |

---

### 4.1 Stochastic HJB

**Setup.** The state $X_t \in \mathbb{R}^d$ evolves as

$$dX_t = b(X_t,u_t)\,dt + \sigma(X_t,u_t)\,dW_t,$$

and we minimise the total expected cost

$$J(t,x;u) = \mathbb{E}\!\left[\int_t^T \ell(X_s,u_s)\,ds + g(X_T)\,\Big|\,X_t=x\right].$$

The **value function** $V(t,x) = \inf_u J(t,x;u)$ satisfies:

$$-\partial_t V = \inf_{u\in\mathcal{U}}\Bigl[\ell(x,u) + \nabla_x V^{\!\top} b(x,u)
+ \tfrac12\operatorname{Tr}\bigl(\sigma\sigma^{\!\top}(x,u)\,\nabla_x^2 V\bigr)\Bigr],
\quad V(T,\cdot)=g.$$

**Intuition — three terms inside the infimum:**
- $\ell(x,u)$ — instantaneous running cost (pay now).
- $\nabla_x V^\top b$ — drift of the value function (first-order Taylor in state change).
- $\tfrac12\operatorname{Tr}(\sigma\sigma^\top\nabla^2 V)$ — curvature correction due to noise
  (stochastic analogue of the second-order Taylor term).

Under smooth $V$, the **feedback law** is
$u^\star(t,x) = \arg\min_u[\ell(x,u)+\nabla_x V^\top b(x,u)].$

---

::::{admonition} Example — Optimal Portfolio Allocation (Merton 1969)
:class: note

Investor wealth $X_t$ follows
$dX_t = (r + u_t(\mu-r))X_t\,dt + u_t\sigma X_t\,dW_t$,
where $u_t\in\mathbb{R}$ is the fraction invested in the risky asset.

Minimise $-\mathbb{E}[\log X_T]$ (maximise expected log-utility).

**Ansatz:** $V(t,x) = \ln x + f(t)$.  Substituting into HJB:

$$f'(t) = -r - \frac{(\mu-r)^2}{2\sigma^2},\qquad f(T)=0.$$

The **optimal Merton rule** is constant:

$$u^\star = \frac{\mu-r}{\sigma^2} \quad (\text{fraction in risky asset}).$$

Invest a fixed fraction proportional to the Sharpe ratio, inversely to variance —
independent of wealth and time.
::::

---

**LQR special case** ($\ell = x^\top Q x + u^\top R u$, $b=Ax+Bu$,
$\sigma$ constant):
$V(t,x)=x^\top P(t)x + v(t)$ with $P$ solving the *matrix Riccati ODE*:

$$-\dot P = A^\top P + PA - PBR^{-1}B^\top P + Q,\quad P(T)=Q_T.$$

The optimal control is **linear feedback**: $u^\star_t = -R^{-1}B^\top P(t)X_t$.

---

::::{admonition} Example — Optimal Inventory (Almgren-Chriss liquidation)
:class: note

Liquidate $X_0$ shares by time $T$.  Inventory $X_t$, trading rate $u_t<0$:

$$dX_t = u_t\,dt, \quad
\ell(x,u) = \underbrace{\alpha x^2}_{\text{risk}} + \underbrace{\beta u^2}_{\text{impact}}.$$

This is a **deterministic LQR** with
$A=0$, $B=1$, $Q=\alpha$, $R=\beta$.
The Riccati solution gives the TWAP-like schedule

$$u^\star(t,x) = -\frac{\alpha}{\beta}\cdot\frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}\cdot X_0,
\quad \kappa=\sqrt{\alpha/\beta}.$$

Large $\kappa$ (high risk aversion or low impact cost) -> aggressive front-loaded selling.
::::

---

### 4.2 Pontryagin Maximum Principle

The PMP avoids the curse of dimensionality — it converts HJB into a
**two-point boundary-value ODE** in $(X_t, p_t)$, feasible when a PDE grid is intractable.

::::{admonition} Theorem (PMP)
:class: tip

Define the **Hamiltonian** $\mathcal{H}(x,u,p) = \ell(x,u)+p^\top b(x,u)$.
If $(X^\star, u^\star)$ is optimal, there exists a **costate** process $p_t$ with:

$$\dot p_t = -\nabla_x \mathcal{H}(X_t^\star, u_t^\star, p_t),\quad p_T = \nabla_x g(X_T^\star),$$

and the optimality condition $u_t^\star = \arg\min_u \mathcal{H}(X_t^\star, u, p_t)$ holds a.e.
::::

**Costate intuition.** $p_t$ is the *shadow price* of state $X_t$:

$$p_t = \nabla_x V(t, X_t^\star) = \frac{\partial (\text{optimal cost-to-go})}{\partial x}.$$

This is exactly the adjoint / backpropagation equation of deep learning — PMP is the
continuous-time version of gradient backpropagation through a dynamical system.

**Algorithm (shooting method):**

```
1. Guess costate p_0
2. Integrate forward:  dX = b(X, u*(X,p)) dt           (state ODE)
3. Integrate backward: dp = -grad_x H(X, u*, p) dt     (costate ODE)
4. Check boundary condition:  p_T = grad g(X_T)
5. If not satisfied -> update p_0 (Newton / gradient) -> go to 2
```

---

::::{admonition} Example — PMP for the Merton Problem
:class: note

With $\ell = 0$, $g(x) = -\ln x$, $b = (r+u(\mu-r))x$,
the Hamiltonian is $\mathcal{H}(x,u,p) = p(r+u(\mu-r))x$.

**Costate ODE:**
$\dot p_t = -\partial_x \mathcal{H} = -p_t(r+u^\star(\mu-r))$,
with terminal $p_T = -1/X_T^\star$.

**Optimality condition** $\partial_u\mathcal{H}=0$ recovers $u^\star = (\mu-r)/\sigma^2$.

The costate path $p_t = -e^{-(T-t)(r+(\mu-r)u^\star)}/X_t^\star$ confirms that the
shadow price scales inversely with wealth — poorer investors value state more.
::::

---

The costate pair $(X_t^\star, p_t)$ moves along Hamiltonian geodesics on
$T^\star\mathbb{R}^d$ — a direct link to symplectic geometry (§10.4).

---

### 4.3 HJB with Jumps (HJBI)

When the state can jump (§3.4), the HJB equation gains a **non-local integral operator**:

$$-\partial_t V = \inf_{u}\Bigl[\ell + \nabla V^\top b + \tfrac12\operatorname{Tr}(\sigma\sigma^\top\nabla^2 V)
+ \underbrace{\int\bigl[V(x+c(x,u,z))-V(x)-\nabla V^\top c(x,u,z)\bigr]\nu(dz)}_{\text{expected value change from jumps}}\Bigr].$$

**Intuition for the integral term.** A jump of size $c$ moves the state from $x$ to
$x+c$, changing the value function by $V(x+c)-V(x)$. The compensator $\nabla V^\top c$
subtracts the linear part already counted in the drift.

The `optimal_control` module discretises the integral on truncated support using Gaussian quadrature.

---

::::{admonition} Example — Optimal Execution with Jump Risk
:class: note

Extend the inventory model with Poisson order-flow shocks:

$$dX_t = u_t\,dt + \Delta J_t,\quad \Delta J_t \sim \text{Compound Poisson}(\lambda, \mathcal{N}(0,\sigma_J^2)).$$

With Gaussian jumps, the HJBI reduces to the same LQR Riccati ODE but with
**effective diffusion** $\sigma_{\text{eff}}^2 = \lambda\sigma_J^2$.

Key insight: order-flow risk acts like additional Brownian volatility, accelerating
the optimal sell schedule.
::::

---

### 4.4 Viscosity Solutions

When $V$ fails to be $C^{1,2}$ — degenerate diffusion, constraints, or non-smooth terminal
conditions — classical solutions may not exist. **Viscosity solutions** (Crandall-Lions 1983)
provide a rigorous weak notion that restores existence and uniqueness.

::::{admonition} Definition — Viscosity Subsolution
:class: definition

A continuous $V$ is a viscosity *subsolution* if for every smooth $\phi$
touching $V$ **from above** at $(t_0,x_0)$:

$$-\partial_t\phi(t_0,x_0) \le \inf_u\Bigl[\ell(x_0,u) + \nabla_x\phi^\top b + \tfrac12\operatorname{Tr}(\sigma\sigma^\top\nabla^2\phi)\Bigr].$$

A *supersolution* reverses the inequality. The unique viscosity **solution** is both.
::::

**Practical interpretation:** Classical: "$V$ satisfies the PDE pointwise."
Viscosity: "$V$ satisfies the PDE in an averaged sense — even at kinks."

Optimiz-rs's backward DP converges to the viscosity solution under CFL:
$\Delta t \le C\,(\Delta x)^2$.

---

::::{admonition} Example — American Option as a Viscosity Problem
:class: note

American put payoff $g(x) = (K-x)^+$ gives the **variational inequality**:

$$\min\Bigl(-\partial_t V - \mathcal{L}_{\text{BS}}V,\; V - (K-x)^+\Bigr) = 0.$$

- **Continuation region** ($V > g$): Black-Scholes PDE holds.
- **Exercise region** ($V = g$): option exercised immediately.

At the free boundary: $\partial_x V$ is continuous (*smooth-pasting*) but $\partial_{xx}V$ is not
— $V$ is $C^1$ but not $C^2$. Viscosity theory handles this kink rigorously.
::::

**Backward DP grid schema:**

```
t=T    [ g(x_1)  g(x_2)  ...  g(x_n) ]   terminal condition
t=T-1  [ V^1     V^2     ...  V^n    ]   one backward step
 .
 .
t=0    [ V_0^1   V_0^2   ...  V_0^n  ]  -> optimal policy u*(x,0)
```

---

## 5 · Mean Field Games (1D Solver)

MFG couples a **backward HJB** (individual value) with a **forward Fokker-Planck** (population density):

$$\begin{aligned}
\text{HJB (backward): } &
-\partial_t u - \nu\partial_{xx}u + H(x,\partial_x u, m) = 0, & u(T,x)&=g(x),\\
\text{Fokker-Planck (forward): } &
\partial_t m - \nu\partial_{xx}m - \partial_x(m\,\partial_p H) = 0, & m(0,x)&=m_0(x).
\end{aligned}$$

**Coupling:** $H$ depends on $m$ (mean-field interaction), creating a fixed-point problem.

**Backward-forward information flow:**

```
t = 0                                t = T
  m_0 (known)                          g(x) (known)
    |                                    |
    |  Fokker-Planck (forward -->)        |
    |  evolves population density m       |
    |                                    |
    v                                    v
  m(t,x)  <----- mutually consistent ---  u(t,x)
                HJB (backward <--)
                optimal value function

  Each agent uses u to choose optimal control.
  Population density m feeds back into u via H(x, du, m).
  Fixed point: m and u are simultaneously consistent (Nash equilibrium).
```

**Fixed-point algorithm:**

```
1. Initialise m^0 = m_0  (e.g. Gaussian)
2. Solve HJB backward  -> u^{k+1}
3. Extract optimal drift: alpha*(x,t) = -d_p H(x, d_x u^{k+1}, m^k)
4. Solve Fokker-Planck forward with alpha* -> m^{k+1}
5. Check ||m^{k+1} - m^k||_1 < eps; if not, k++ -> go to 2
```

```{figure} ../_static/diagrams/fig_kalman_covariance.svg
:align: center
:alt: Kalman filter covariance convergence — P_t converges to steady state
```
Before observation (predict):    After observation (update):

  +------------------+           +--------+
  |                  |           |        |
  |   p(x | y_1:t-1) |  ---->    |p(x|y_t)|
  |    wide ellipse  |           |  tight |
  +------------------+           +--------+

  Kalman gain K interpolates between:
    K -> 0        (huge R, ignore y_t)  =>  x_hat = prior
    K -> H^-1     (R=0, trust y_t)      =>  x_hat = H^-1 y_t

**Covariance convergence:** $P_t \to P_\infty$ (algebraic Riccati solution) exponentially fast
when $(F,H)$ is observable.

### 6.2 Information-Theoretic View

The Kalman filter computes the exact conditional mean
$\hat{\mathbf{x}}_t = \mathbb{E}[\mathbf{x}_t \mid \mathbf{y}_{1:t}]$ in Gaussian models
and minimises $D_{\mathrm{KL}}(p(\mathbf{x}_t|\mathbf{y}_{1:t})\,\|\,\mathcal{N}(\hat{\mathbf{x}}_t, P_t))$
over all Gaussian approximations.

### 6.3 Continuous-Time Limit (Kalman-Bucy)

For $d\mathbf{X}_t = A\mathbf{X}_t\,dt + B\,d\mathbf{W}_t$,
$d\mathbf{Y}_t = C\mathbf{X}_t\,dt + d\mathbf{V}_t$, the error covariance satisfies
the *Riccati ODE*:

$$\dot P = AP + PA^\top + BQB^\top - PC^\top R^{-1}CP,\qquad P(0)=P_0.$$

::::{admonition} Example — Tracking a Noisy AR(1) Signal
:class: note

**Model:** Latent trend $x_t = 0.95 x_{t-1} + w_t$ ($Q=0.01$);
noisy observation $y_t = x_t + v_t$ ($R=1.0$).

Steady-state: $P_\infty \approx 0.17$, so $K_\infty \approx 0.15$.
Kalman weights the new observation at 15%, prior at 85%.

**Implication:** With $R/Q = 100$ (much noisier obs than process), the filter heavily
smooths observations — useful for noisy financial signals like tick prices.
::::

---

## 7 · MCMC (Metropolis-Hastings and Langevin)

### 7.1 Metropolis-Hastings

For target $\pi(x) \propto e^{-U(x)}$ and proposal $q(x'\mid x)$:

$$\alpha(x\to x') = \min\!\Bigl(1, \frac{\pi(x')q(x\mid x')}{\pi(x)q(x'\mid x)}\Bigr).$$

**Detailed balance** $\pi(x)\alpha(x\to x') = \pi(x')\alpha(x'\to x)$
ensures $\pi$ is the unique stationary distribution.

**Optimal scaling:** With Gaussian proposal, step $h^\star \approx 2.38/\sqrt{d}$
(Roberts-Gelman-Gilks 1997) targets ~23-45% acceptance.

**Energy landscape and accept/reject:**

```{figure} ../_static/diagrams/fig_mcmc_energy.svg
:align: center
:alt: MCMC energy landscape — bimodal potential function
```

**Trace plot of a well-mixed chain:**

```{figure} ../_static/diagrams/fig_mcmc_trace.svg
:align: center
:alt: MCMC trace plot — chain samples and marginal distribution
```

### 7.2 Langevin Dynamics (MALA)

Metropolis-Adjusted Langevin proposal:

$$x' = x - \tfrac{h^2}{2}\nabla U(x) + h\,\xi, \quad \xi\sim\mathcal{N}(0,I_d),$$

a discretisation of the *overdamped Langevin SDE*:

$$dX_t = -\nabla U(X_t)\,dt + \sqrt{2}\,dW_t,$$

whose stationary distribution is exactly $\pi \propto e^{-U}$.

MALA converges in $O(d^{1/3})$ steps vs $O(d)$ for RW-MH — key advantage
for high-dimensional posteriors.

**MALA vs RW-MH trajectory comparison:**

```
  RW-MH vs MALA trajectories
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  RW-MH (random walk):          MALA (gradient-guided):

  ◦  ◦  ◦                         ▲ −∇U ← toward mode
   ◦  ◦  ◦  ◦      vs             ╱
      ◦  ◦                       ◦ ╱ ◦  ◦
   ◦     ◦  ◦                     ◦     ◦

  diffusive, O(d) steps            directed, O(d^{1/3}) steps
  each step ∼ isotropic ξ          each step biased by −∇U(x)
```

::::{admonition} Example — Calibrating OU Parameters via MCMC
:class: note

**Goal:** Full Bayesian inference on $(\kappa, \theta, \sigma)$ of an OU process.
**Prior:** $\kappa \sim \text{Gamma}(2,0.1)$, $\theta \sim \mathcal{N}(0,1)$,
$\sigma \sim \text{HalfNormal}(0.5)$.

**MALA chain** ($d=3$, step $h=0.02$):

```
Iteration  kappa   theta   sigma   log-post
---------  -----   -----   -----   --------
  1000      42.1   0.003   0.11     125.3
  2000      55.3   0.003   0.12     128.7   <- burn-in complete
  3000      58.1   0.003   0.12     129.2
 50000      54.8   0.003   0.12     128.9   <- stable posterior
```

**Marginal posterior** (kappa): 95% CI $[44, 67]$, peak at 55/yr —
wider than the MLE point estimate, reflecting genuine parameter uncertainty.
::::

---

## 8 · Hidden Markov Models (HMM)

### 8.1 Model

Latent Markov chain $Z_t \in \{1,\ldots,K\}$ with transition matrix
$A_{ij}=\mathbb{P}(Z_t=j\mid Z_{t-1}=i)$ generates observations
$Y_t \mid Z_t=k \sim B_k(y)$.

**State machine diagram ($K=3$ regimes):**

```{figure} ../_static/diagrams/fig_hmm_regime.svg
:align: center
:width: 90%

HMM $K=3$ state machine with Bull / Neutral / Bear regimes and Gaussian emission
parameters. Self-transitions $A_{11}=A_{22}=0.97$, $A_{33}=0.90$.
```

### 8.2 Baum-Welch (EM)

**E-step (forward-backward):**

$$\alpha_t(k) = B_k(y_t)\sum_j \alpha_{t-1}(j)A_{jk}, \qquad
\beta_t(k) = \sum_j A_{kj}B_j(y_{t+1})\beta_{t+1}(j).$$

$$\gamma_t(k) = \frac{\alpha_t(k)\beta_t(k)}{\sum_j \alpha_t(j)\beta_t(j)}, \qquad
\xi_t(j,k) = \frac{\alpha_t(j)A_{jk}B_k(y_{t+1})\beta_{t+1}(k)}{\mathcal{L}}.$$

**M-step:**

$$\hat A_{jk} = \frac{\sum_t \xi_t(j,k)}{\sum_t\gamma_t(j)}, \qquad
\hat\mu_k = \frac{\sum_t \gamma_t(k)\,y_t}{\sum_t \gamma_t(k)}.$$

**Information-theoretic view:** Baum-Welch is EM on the complete-data log-likelihood; each
iteration monotonically increases $\mathcal{L}(\theta)$ by Jensen's inequality.

**Viterbi trellis diagram ($K=3$, $T=4$):**

```{figure} ../_static/diagrams/fig_viterbi_trellis.svg
:align: center
:width: 82%

Viterbi trellis ($K=3$, $T=4$). Filled nodes mark the MAP (most probable) state
sequence; arrows show transition candidates. Backtracking via $\psi_t(k)$ recovers
$z_1^\star 	o z_4^\star$.
```

**Viterbi (MAP path):** $\delta_t(k) = \max_j \delta_{t-1}(j)A_{jk} \cdot B_k(y_t)$, $O(TK^2)$.

**Quality check:** Log-likelihood must be non-decreasing; confusion matrix of Viterbi labels
vs ground truth validates regime recovery.

::::{admonition} Example — Equity Regime Detection (S&P 500)
:class: note

**Data:** S&P 500 daily log-returns, 2000-2023, $T=5820$ observations.

**Fit $K=3$ HMM** using `hmm::fit_baum_welch()` with 20 random restarts.

**Estimated regime parameters:**

| Regime | Ann. return | Ann. vol | Avg duration |
|--------|------------|---------|-------------|
| Bull   | +18%       | 10%     | 350 days    |
| Neutral | +2%       | 17%     | 80 days     |
| Bear   | -40%       | 38%     | 25 days     |

**Smoothed state probabilities** $\gamma_t(k)$:

```
P(Bull)   1.0|XXXXXXXXXX        XXXXXXXXXX        XXXXX
              |          XXXXXXXX          XXXXXXXX
          0.0 +-------------------------> t (years)
              2000    2003    2008    2020   2023
                  ^                ^    ^
                  dot-com bust  GFC  COVID crash
```

```{figure} ../_static/diagrams/fig_kl_asymmetry.svg
:align: center
:alt: KL divergence asymmetry — D(P||Q) vs D(Q||P) illustration
```

**Connection to model selection:** AIC $= 2k - 2\ln\hat{\mathcal{L}}$ and
BIC $= k\ln n - 2\ln\hat{\mathcal{L}}$ bound $D_{\mathrm{KL}}(p_{\text{true}}\,\|\,p_\theta)$.

### 9.2 Fisher Information

::::{admonition} Definition — Fisher Information Matrix
:class: definition

For parametric model $p(x;\theta)$:

$$\mathcal{I}(\theta)_{ij}
= \mathbb{E}_{x\sim p}\!\left[\partial_{\theta_i}\log p\;\partial_{\theta_j}\log p\right]
= -\mathbb{E}\!\left[\partial^2_{\theta_i\theta_j}\log p\right].$$
::::

**Fisher information as curvature of the log-likelihood:**

```{figure} ../_static/diagrams/fig_fisher_curvature.svg
:align: center
:alt: Fisher information curvature — log-likelihood and information matrix
```

```{figure} ../_static/diagrams/fig_natural_gradient.svg
:align: center
:alt: Natural gradient descent — steepest descent in information geometry
```

```{figure} ../_static/diagrams/fig_curvatures.svg
:align: center
:alt: Curvature comparison — positive, zero, and negative curvature geodesics
```

**Tangent space — linear approximation at $p$:**

```
   M (curved 2D surface):        TpM (flat tangent plane at p):

     .~~~~.                          ___________
    /      \      -->               |    TpM    |
   | p *    |                       |     * p   |
   |        |                       |___________|
    \      /
     .~~~~.
  (not flat globally, but TpM is flat locally — used for calculus on M)
```

**Geodesics** satisfy:

$$\ddot\gamma^k + \sum_{i,j}\Gamma^k_{ij}\,\dot\gamma^i\dot\gamma^j = 0,$$

where $\Gamma^k_{ij} = \tfrac12 g^{kl}(\partial_i g_{jl}+\partial_j g_{il}-\partial_l g_{ij})$
are the *Christoffel symbols* encoding intrinsic curvature.

### 10.2 Information Geometry and Fisher-Rao Metric

The statistical manifold $\mathcal{M} = \{p(\cdot;\theta)\}$ carries the
**Fisher-Rao metric** $g_{ij}(\theta) = \mathcal{I}(\theta)_{ij}$.

**Standard vs natural gradient:**

```{figure} ../_static/diagrams/fig_std_vs_nat_gradient.svg
:align: center
:width: 88%

Standard versus natural gradient: geometric properties. On exponential families
the natural gradient equals the MLE Newton step, achieving convergence in one
iteration.
```

**Natural gradient (Amari 1998):**

$$\theta \leftarrow \theta - \eta\,\mathcal{I}(\theta)^{-1}\nabla_\theta\mathcal{L}.$$

**KL geometry:**
$D_{\mathrm{KL}}(p_\theta\,\|\,p_{\theta+d\theta}) = \tfrac12\,d\theta^\top\mathcal{I}(\theta)\,d\theta + O(\|d\theta\|^3)$,
confirming Fisher-Rao as the intrinsic KL metric.

**Dually flat structure:** Exponential families
$p(x;\theta)=h(x)\exp(\theta^\top T(x)-A(\theta))$
have $K=0$ — explaining exact Newton/natural-gradient convergence.

::::{admonition} Example — Natural Gradient on a Gaussian Model
:class: note

For $p(x;\theta) = \mathcal{N}(\mu, \sigma^2)$, $\theta=(\mu,\sigma^2)$:

$$\mathcal{I}(\theta) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}.$$

**Natural gradient** of $\mathcal{L} = -\log p(x_{\rm obs};\theta)$:

$$\tilde\nabla_\theta\mathcal{L} = \mathcal{I}^{-1}\nabla\mathcal{L} = \begin{pmatrix}\mu-x \\ \sigma^2 - (x-\mu)^2/2\end{pmatrix}.$$

One Newton step on this exponential family finds the MLE exactly because the
Hessian equals $\mathcal{I}$ (dually flat, $K=0$).
::::

### 10.3 Lie Groups and Geometric Control

::::{admonition} Definition — Lie Group
:class: definition

A *Lie group* $G$ is a smooth manifold with a group structure where
multiplication and inversion are smooth.  The *Lie algebra* $\mathfrak{g} = T_e G$
linearises the group at the identity.
::::

**Matrix Lie group hierarchy:**

```{figure} ../_static/diagrams/fig_lie_group_hierarchy.svg
:align: center
:width: 90%

Matrix Lie group hierarchy: subgroup inclusions and their quantitative-finance
applications. $SO(n)$ underpins PCA factor rotation; $\mathrm{Sp}(2n,\mathbb{R})$
governs Hamiltonian mechanics (PMP §10.4); $H(n)$ drives path-signature features.
```

**Left-invariant control system on $G$:**

$$\dot g(t) = g(t)\,\xi(t), \quad g\in G,\; \xi(t)\in\mathfrak{g}.$$

PMP on Lie groups yields the *Lie-Poisson (Euler-Poincare) equations* (Holm-Marsden-Ratiu),
providing structure-preserving optimal trajectories.

### 10.4 Symplectic Geometry and Hamiltonian Structure

The phase space $(T^\star M, \omega)$ carries the symplectic 2-form
$\omega = \sum_i dp_i \wedge dq_i$.  Hamilton's equations preserve $\omega$
(*Liouville's theorem* — phase-space volume conserved).

**Connection to PMP:** The costate pair $(X_t^\star, p_t)$ solves Hamilton's equations,
i.e., the PMP is a symplectic flow on $T^\star\mathbb{R}^d$.

**Symplectic integrators** (Stormer-Verlet, Ruth-Forest) preserve $\omega$ discretely,
keeping the Hamiltonian nearly constant over long horizons — critical for multi-year
allocation back-tests in Optimiz-rs.

### 10.5 Sectional Curvature and Landscape Geometry

The sectional curvature $K(\sigma)$ governs how quickly nearby geodesics diverge:

```
  Sectional curvature and optimiser geometry
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  K > 0 (sphere)  → geodesics converge   → compact optimiser orbits
  K = 0 (flat)    → Euclidean behaviour  → Newton / nat. grad. exact
  K < 0 (hyper.)  → exponential spread   → fast landscape exploration

  Exponential families live on K = 0 manifold (dually flat).
  DE explores K < 0 terrain: difference vectors "fan out" exponentially.
```

For exponential families in natural/mean parameters $K=0$ — explaining exact
Newton convergence without curvature correction.

---

## Quick Reference

| Concept | Key equation / object | Optimiz-rs module |
|---------|----------------------|-------------------|
| Brownian motion | $W_t - W_s \sim \mathcal{N}(0,t-s)$ | `point_processes` |
| Ito SDE | $dX=b\,dt+\sigma\,dW$ | `ou_estimator` |
| Poisson / Compound Poisson | $N_t\sim\text{Poisson}(\lambda t)$ | `point_processes` |
| Levy process | triplet $(b,\sigma^2,\nu)$ | `point_processes` |
| HJB PDE | $-\partial_t V = \inf_u[\ell + \nabla V^\top b + \tfrac12\operatorname{Tr}\sigma\sigma^\top\nabla^2 V]$ | `optimal_control` |
| HJBI (jumps) | $+\int[V(\cdot+c)-V-\nabla V^\top c]\nu\,dz$ | `optimal_control` |
| PMP costate | $\dot p = -\nabla_x\mathcal{H}$, $u^\star=\arg\min_u\mathcal{H}$ | `optimal_control` |
| MFG (HJB + KFP) | fixed-point $u,m$ | `mean_field_games` |
| Kalman filter | $K_t = P^-H^\top(HP^-H^\top+R)^{-1}$ | `optimal_control` |
| MALA | $x'=x-\tfrac{h^2}{2}\nabla U+h\xi$ | `mcmc` |
| HMM | Baum-Welch EM + Viterbi | `hmm` |
| Fisher information | $\mathcal{I}_{ij}=\mathbb{E}[\partial_i\ell\,\partial_j\ell]$ | `hmm`, `sparse` |
| Natural gradient | $\mathcal{I}^{-1}\nabla_\theta\mathcal{L}$ | `differential_evolution` |
| Riemannian / Lie geometry | Christoffel symbols, Lie-Poisson equations | experimental |
| DE (jDE) | mutation + crossover + selection | `differential_evolution` |

---

## References

1. Oksendal, B. *Stochastic Differential Equations*, 6th ed. Springer, 2003.
2. Cont, R. & Tankov, P. *Financial Modelling with Jump Processes*. CRC Press, 2004.
3. Fleming, W.H. & Soner, H.M. *Controlled Markov Processes and Viscosity Solutions*. Springer, 2006.
4. Lasry, J.-M. & Lions, P.-L. "Mean field games." *Jpn. J. Math.* **2** (2007) 229-260.
5. Amari, S. *Information Geometry and Its Applications*. Springer, 2016.
6. do Carmo, M.P. *Riemannian Geometry*. Birkhauser, 1992.
7. Holm, D.D., Marsden, J.E. & Ratiu, T.S. "The Euler-Poincare equations." *Adv. Math.* **137** (1998).
8. Price, K.V., Storn, R.M. & Lampinen, J.A. *Differential Evolution*. Springer, 2005.
9. Roberts, G.O., Gelman, A. & Gilks, W.R. "Weak convergence of Metropolis algorithms." (1997).
10. Merton, R.C. "Option pricing when underlying stock returns are discontinuous." *JFE* **3** (1976).
11. Crandall, M.G. & Lions, P.-L. "Viscosity solutions of Hamilton-Jacobi equations." *Trans. AMS* (1983).
12. Almgren, R. & Chriss, N. "Optimal execution of portfolio transactions." *J. Risk* **3** (2001).
13. Carmona, R. & Delarue, F. *Probabilistic Theory of Mean Field Games*. Springer, 2018.
