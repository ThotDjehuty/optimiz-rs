# Mathematical Foundations

This page collects the core equations driving Optimiz-rs’s Rust kernels, plus short intuition blurbs and micro-checks you can run in a notebook. For visuals and full walkthroughs, see the example notebooks in `examples/notebooks/`.

## Differential Evolution (DE)

We minimize $f: \mathbb{R}^d \to \mathbb{R}$ with a population $\{\mathbf{x}_{i,g}\}_{i=1}^N$.

**Mutation (rand/1):**
$$
\mathbf{v}_{i,g} = \mathbf{x}_{r_1,g} + F \cdot (\mathbf{x}_{r_2,g} - \mathbf{x}_{r_3,g}),\quad r_1 \neq r_2 \neq r_3 \neq i.
$$

**Intuition:** The differential term is a directional finite-difference estimate of the gradient; scaling $F$ sets the step length. Population diversity controls exploration.

**Crossover (binomial):**
$$
u_{i,j,g} = \begin{cases}
v_{i,j,g} & \text{if } \mathrm{Uniform}(0,1) < CR \text{ or } j = j_{\mathrm{rand}},\\
x_{i,j,g} & \text{otherwise.}
\end{cases}
$$

**Selection (greedy):**
$$
\mathbf{x}_{i,g+1} = \begin{cases}
\mathbf{u}_{i,g} & \text{if } f(\mathbf{u}_{i,g}) \le f(\mathbf{x}_{i,g}),\\
\mathbf{x}_{i,g} & \text{otherwise.}
\end{cases}
$$

**Self-adaptive jDE (used by Optimiz-rs):**
$$
F_i^{g+1} = \begin{cases}
F_{\min} + r_1 \cdot F_{\max} & r_2 < \tau_1,\\
F_i^{g} & \text{otherwise,}
\end{cases}
\qquad
CR_i^{g+1} = \begin{cases}
\mathrm{Uniform}(0,1) & r_3 < \tau_2,\\
CR_i^{g} & \text{otherwise.}
\end{cases}
$$
Typical $\tau_1, \tau_2 = 0.1$. This adaptation reduces manual tuning and improves robustness on multimodal landscapes.

**Notebook check:** In `05_performance_benchmarks.ipynb`, plot $F_i$ and $CR_i$ histograms every 50 generations to verify adaptation is active (expect spread around 0.5–0.9 for $CR$ and 0.5–0.9 for $F$ on hard landscapes).

## Optimal Control (HJB)

For dynamics $dX_t = b(X_t, u_t)\,dt + \sigma(X_t,u_t)\,dW_t$ with running cost $\ell$ and terminal cost $g$, the value function satisfies the Hamilton–Jacobi–Bellman PDE:
$$
-\partial_t V(t,x) = \inf_{u\in\mathcal{U}} \Big[ \ell(x,u) + \nabla_x V(t,x)^{\top} b(x,u) + \tfrac12 \operatorname{Tr}\big(\sigma\sigma^{\top}(x,u) \, \nabla_x^2 V(t,x)\big) \Big],\quad V(T,x) = g(x).
$$

Optimiz-rs uses finite differences with backward time-stepping and optional policy iteration. On a uniform grid $(t_n, x_j)$:
$$
V^{n} = \min_{u}\Big\{ \ell(x_j,u)\,\Delta t + V^{n+1} + \nabla_x V^{n+1}\cdot b\,\Delta t + \tfrac12 \operatorname{Tr}(\sigma\sigma^{\top}\nabla_x^2 V^{n+1})\,\Delta t \Big\}.
$$
The control that attains the minimum yields the feedback policy $u^{\star}(x_j, t_n)$ exported by `compute_policy`.

**Interpretation:** HJB is dynamic programming in continuous time; $V$ encodes the optimal cost-to-go. The quadratic example in `03_optimal_control_tutorial.ipynb` shows $V$ becoming steeper where volatility is high or costs penalize deviation.

## Mean Field Games (1D solver)

Optimiz-rs’s MFG module solves the coupled system for value $u$ and density $m$:
$$
\begin{aligned}
-\partial_t u(t,x) - \nu\,\partial_{xx} u(t,x) + H\big(x,\partial_x u(t,x), m(t,x)\big) &= 0,\\
\partial_t m(t,x) - \nu\,\partial_{xx} m(t,x) - \operatorname{div}\big(m(t,x) \, \partial_p H(x,\partial_x u, m)\big) &= 0,\\
u(T,x) &= g(x), \qquad m(0,x) = m_0(x).
\end{aligned}
$$
We use fixed-point iterations on the transport term with implicit diffusion (stable for $\nu > 0$) and normalize $m$ after each step to preserve mass.

**Practical tip:** Monitor $\|m^{k+1}-m^{k}\|_1$ and $\|u^{k+1}-u^{k}\|_\infty$; both appear in the notebook to diagnose non-convergence.

## Kalman Filtering

For linear-Gaussian state space models
$$
\begin{aligned}
\mathbf{x}_{t} &= F\,\mathbf{x}_{t-1} + \mathbf{w}_{t}, && \mathbf{w}_t \sim \mathcal{N}(0, Q),\\
\mathbf{y}_{t} &= H\,\mathbf{x}_{t}   + \mathbf{v}_{t}, && \mathbf{v}_t \sim \mathcal{N}(0, R),
\end{aligned}
$$
prediction and update follow:
$$
\begin{aligned}
	ext{Predict: } & \hat{\mathbf{x}}^-_t = F \hat{\mathbf{x}}_{t-1}, && P^-_t = F P_{t-1} F^{\top} + Q,\\
	ext{Update: } & K_t = P^-_t H^{\top} (H P^-_t H^{\top} + R)^{-1},\\
& \hat{\mathbf{x}}_t = \hat{\mathbf{x}}^-_t + K_t(\mathbf{y}_t - H \hat{\mathbf{x}}^-_t),\\
& P_t = (I - K_t H) P^-_t.
\end{aligned}
$$
These steps back the `init_kalman_filter`, `kalman_predict`, and `kalman_update` helpers.

## MCMC (Metropolis–Hastings)

For target density $\pi(x)$ and proposal $q(x'\mid x)$:
$$
\alpha(x \to x') = \min\Big(1, \frac{\pi(x')\, q(x \mid x')}{\pi(x)\, q(x' \mid x)}\Big).
$$
Optimiz-rs uses symmetric Gaussian proposals (so $q$ cancels) by default, with optional bounds projection and burn-in.

**Heuristic:** Tune proposal std so acceptance is ~0.25–0.35 for moderate dimensions; see `examples/notebooks/02_mcmc.ipynb` for trace plots.

## Hidden Markov Models (HMM)

We maximize the likelihood of observations $\mathbf{y}$ under latent states $\mathbf{z}$ using Baum–Welch (EM):
$$
\mathcal{L}(\theta) = \sum_{t} \log \Big( \sum_{z_t} p(y_t \mid z_t, \theta) p(z_t \mid z_{t-1}, \theta) \Big).
$$
Forward–backward computes posteriors, then M-step re-estimates transition and emission parameters; Viterbi gives the MAP state path.

**Quality check:** Plot log-likelihood per iteration; it should be non-decreasing. The HMM tutorial notebook includes a simple convergence plot and a confusion matrix for decoded states.
