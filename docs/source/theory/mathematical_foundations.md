# Mathematical Foundations

This page develops the core mathematics underlying Optimiz-rs's Rust kernels — from first
principles through advanced theory.  Each section opens with a **definition block**, builds
intuition through **examples**, and closes with a **notebook micro-check**.  For complete
walkthroughs see ``examples/notebooks/``.

---

## 1 · Differential Evolution (DE)

### Background

DE is a gradient-free population-based optimizer for :math:`f: \mathbb{R}^d \to \mathbb{R}`,
not required to be smooth or convex.  At generation :math:`g` we maintain :math:`N` candidate
solutions :math:`\{\mathbf{x}_{i,g}\} \subset \mathbb{R}^d`.

**Key insight:** The difference vector :math:`\mathbf{x}_{r_2}-\mathbf{x}_{r_3}` is an
unbiased directional finite-difference of :math:`f`, so DE implicitly estimates curvature
without Jacobians.

### Operators

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Step
     - Formula
     - Role
   * - Mutation (rand/1)
     - :math:`\mathbf{v}_{i,g} = \mathbf{x}_{r_1} + F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3})`
     - explore
   * - Binomial crossover
     - :math:`u_{i,j} = v_{i,j}` if :math:`U(0,1)<CR` or :math:`j=j_\text{rand}`
     - mix dimensions
   * - Greedy selection
     - :math:`\mathbf{x}_{i,g+1} = \mathbf{u}_{i,g}` iff :math:`f(\mathbf{u})\le f(\mathbf{x})`
     - exploit

**Convergence (informal):** Under bounded population diversity and Lipschitz :math:`f`, the
best-so-far value converges a.s. to a stationary point as :math:`N,g\to\infty` (Price et al. 2005).

### Self-Adaptive jDE (Optimiz-rs default)

Parameters :math:`F,CR` are per-individual and reset stochastically each generation:

.. math::

   F_i^{g+1} = \begin{cases} F_{\min} + r_1 F_{\max} & r_2 < \tau_1,\\ F_i^g & \text{otherwise,}\end{cases}
   \qquad
   CR_i^{g+1} = \begin{cases} U(0,1) & r_3 < \tau_2,\\ CR_i^g & \text{otherwise.}\end{cases}

:math:`\tau_1=\tau_2=0.1` by default.  On rugged landscapes this produces bimodal :math:`F`
histograms concentrated near 0.8 — a sign the landscape is highly multimodal.

**Notebook check** (``05_performance_benchmarks.ipynb``): Plot :math:`F_i, CR_i` histograms
every 50 generations; expect values clustering in :math:`[0.5,0.9]` on hard problems.

---

## 2 · Stochastic Processes

These form the probabilistic backbone of all continuous-time models in Optimiz-rs.

### 2.1 Brownian Motion

.. admonition:: Definition — Wiener Process

   A stochastic process :math:`W = (W_t)_{t\ge 0}` on :math:`(\Omega,\mathcal{F},\mathbb{P})`
   is a *standard Brownian motion* if:

   1. :math:`W_0 = 0` a.s.
   2. Increments are **independent**: :math:`W_t - W_s \perp \mathcal{F}_s` for :math:`t>s`.
   3. :math:`W_t - W_s \sim \mathcal{N}(0, t-s)` for all :math:`0\le s<t`.
   4. Paths :math:`t\mapsto W_t(\omega)` are **continuous** a.s.

**Key properties:**

- :math:`\mathbb{E}[W_t] = 0`, :math:`\operatorname{Var}(W_t) = t`, :math:`\operatorname{Cov}(W_s,W_t) = \min(s,t)`.
- **Quadratic variation:** :math:`[W]_T = T` (paths are non-differentiable but have finite :math:`p`-variation for :math:`p>2`).
- **Self-similarity:** :math:`c^{-1/2}W_{ct} \overset{d}{=} W_t` (Hurst exponent :math:`H=\tfrac12`).

**Example — Geometric BM:**
:math:`S_t = S_0 \exp\!\bigl((\mu-\tfrac12\sigma^2)t + \sigma W_t\bigr)`
is the Black–Scholes price model.  Sample path sketch::

    S_t
    |       .---.
    |  .--./     \----.
    | /                \---------.
    |/
    +-------------------------------> t
      0             T
    (log-normal marginals; continuous, nowhere-differentiable paths)

### 2.2 Itô Calculus

.. admonition:: Definition — Itô Integral

   For adapted :math:`f \in \mathcal{L}^2` (i.e. :math:`\mathbb{E}\!\int_0^T f_t^2\,dt < \infty`):

   .. math::

      \int_0^T f_t\,dW_t \;=\; L^2\text{-}\lim_{|\pi|\to 0} \sum_{k} f_{t_k}(W_{t_{k+1}}-W_{t_k}).

   The Itô integral is a **martingale** with zero mean and **Itô isometry**
   :math:`\mathbb{E}\bigl[(\int_0^T f_t\,dW_t)^2\bigr] = \mathbb{E}\int_0^T f_t^2\,dt`.

.. admonition:: Theorem — Itô's Lemma

   For :math:`dX_t = \mu_t\,dt + \sigma_t\,dW_t` and :math:`F \in C^{1,2}([0,T]\times\mathbb{R})`:

   .. math::

      dF(t,X_t) = \partial_t F\,dt + \partial_x F\,dX_t + \tfrac{1}{2}\partial_{xx}F\,\sigma_t^2\,dt.

   The correction term :math:`\tfrac12\sigma^2\partial_{xx}F` (absent in ordinary calculus) arises
   from the non-zero quadratic variation :math:`d[W]_t = dt`.

**Example:** Let :math:`X_t = \log S_t` with :math:`dS_t = \mu S_t\,dt + \sigma S_t\,dW_t`.
Itô's Lemma gives :math:`dX_t = (\mu - \tfrac12\sigma^2)\,dt + \sigma\,dW_t`. ✔

### 2.3 General Itô SDEs

.. math::

   dX_t = b(t, X_t)\,dt + \boldsymbol{\sigma}(t, X_t)\,dW_t,\quad X_0 = x_0.

**Existence & uniqueness (Picard–Lindelöf for SDEs):** If :math:`b, \boldsymbol{\sigma}` are
globally Lipschitz with linear growth, there exists a unique strong solution with
:math:`\mathbb{E}[\sup_{t\le T}\|X_t\|^2]<\infty`.

.. list-table:: Common SDE Models
   :header-rows: 1
   :widths: 20 40 40

   * - Process
     - SDE
     - Stationary distribution
   * - Brownian motion
     - :math:`dX = \sigma\,dW`
     - —
   * - Geometric BM
     - :math:`dX = \mu X\,dt + \sigma X\,dW`
     - log-normal
   * - Ornstein–Uhlenbeck
     - :math:`dX = \kappa(\theta-X)\,dt + \sigma\,dW`
     - :math:`\mathcal{N}(\theta, \sigma^2/2\kappa)`
   * - CIR
     - :math:`dX = \kappa(\theta-X)\,dt + \sigma\sqrt{X}\,dW`
     - Gamma(:math:`2\kappa\theta/\sigma^2`, :math:`\sigma^2/2\kappa`)

### 2.4 Ornstein-Uhlenbeck (Mean-Reversion)

Used in Optimiz-rs's ``sparse_mean_reversion`` and ``ou_estimator`` modules:

.. math::

   dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t.

**Closed-form solution:**

.. math::

   X_t = \theta + (X_0 - \theta)e^{-\kappa t} + \sigma\int_0^t e^{-\kappa(t-s)}\,dW_s.

**Half-life:** :math:`\tau_{1/2} = \ln 2/\kappa`.  With :math:`\kappa=0.2`/day,
half-life ≈ 3.5 days — typical for equity-pair spreads.

**MLE log-likelihood** (discrete observations at spacing :math:`\Delta t`):

.. math::

   \ell(\kappa,\theta,\sigma) = -\frac{1}{2}\sum_{i=1}^{n}\left[\log(2\pi\hat\sigma_i^2)
   + \frac{(X_{t_i} - \hat\mu_i)^2}{\hat\sigma_i^2}\right],

where :math:`\hat\mu_i = \theta + (X_{t_{i-1}}-\theta)e^{-\kappa\Delta t}` and
:math:`\hat\sigma_i^2 = \frac{\sigma^2}{2\kappa}(1-e^{-2\kappa\Delta t})`.

---

## 3 · Jump Processes

Many financial time series exhibit sudden large moves that Brownian motion cannot capture.

### 3.1 Poisson Process

.. admonition:: Definition — Poisson Process

   A counting process :math:`N = (N_t)_{t\ge 0}` is a *Poisson process with
   intensity* :math:`\lambda > 0` if:

   1. :math:`N_0 = 0`.
   2. Independent, stationary increments.
   3. :math:`\mathbb{P}(N_{t+h}-N_t=1) = \lambda h + o(h)` and :math:`\mathbb{P}(\Delta N > 1) = o(h)`.

Equivalently, :math:`N_t \sim \text{Poisson}(\lambda t)` and inter-arrival times are
:math:`\text{Exp}(\lambda)`. The *compensated* process :math:`\tilde N_t = N_t - \lambda t`
is a martingale.

### 3.2 Compound Poisson Jump-Diffusion (Merton 1976)

.. math::

   \frac{dS_t}{S_{t^-}} = \mu\,dt + \sigma\,dW_t + d\Bigl(\sum_{k=1}^{N_t}(e^{J_k}-1)\Bigr),

with :math:`N_t` Poisson(:math:`\lambda`) and :math:`J_k \sim \mathcal{N}(\mu_J, \sigma_J^2)`.

**Merton option price** — a Poisson mixture of Black–Scholes prices:

.. math::

   C_{\text{Merton}} = \sum_{n=0}^\infty \frac{e^{-\lambda' T}(\lambda' T)^n}{n!}
   \cdot C_{\text{BS}}\!\left(S_0, K, T, r_n, \sigma_n^2\right),

where :math:`\lambda' = \lambda e^{\mu_J+\frac12\sigma_J^2}`,
:math:`r_n = r - \lambda(e^{\mu_J+\frac12\sigma_J^2}-1) + n(\mu_J+\tfrac12\sigma_J^2)/T`,
and :math:`\sigma_n^2 = \sigma^2 + n\sigma_J^2/T`.

### 3.3 Lévy Processes and the Lévy–Khintchine Representation

.. admonition:: Theorem — Lévy–Khintchine

   Every Lévy process (independent stationary increments) has characteristic function

   .. math::

      \mathbb{E}[e^{i\xi X_t}] = \exp\!\Bigl(t\Bigl[i b\xi - \tfrac{1}{2}\sigma^2\xi^2
      + \int_{\mathbb{R}\setminus\{0\}} \bigl(e^{i\xi z}-1-i\xi z\mathbf{1}_{|z|\le1}\bigr)\nu(dz)\Bigr]\Bigr)

   where :math:`(b, \sigma^2, \nu)` is the *Lévy triplet* and :math:`\nu` the *Lévy measure*,
   satisfying :math:`\int(1\wedge z^2)\nu(dz)<\infty`.

.. list-table:: Lévy Process Zoo
   :header-rows: 1
   :widths: 25 35 40

   * - Process
     - Lévy measure :math:`\nu`
     - Use case
   * - Brownian motion
     - :math:`\nu=0`
     - continuous diffusion
   * - Compound Poisson
     - finite measure
     - rare large jumps
   * - Variance Gamma
     - :math:`\nu(dz)\propto e^{-c|z|}/|z|`
     - equity returns
   * - CGMY
     - :math:`e^{-G|z|}/|z|^{1+Y}` (neg), :math:`e^{-Mx}/x^{1+Y}` (pos)
     - heavy tails, :math:`Y\in(0,2)`
   * - :math:`\alpha`-stable
     - :math:`c|z|^{-1-\alpha}`
     - infinite-variance regimes

### 3.4 SDEs with Jumps — Generator and Itô Formula

.. math::

   dX_t = b(X_{t^-})\,dt + \sigma(X_{t^-})\,dW_t
   + \int_{\mathbb{R}} c(X_{t^-}, z)\,\tilde N(dt, dz),

where :math:`\tilde N(dt,dz) = N(dt,dz) - \nu(dz)\,dt` is the *compensated jump measure*.

**Itô formula for jump-diffusions:**

.. math::

   dF(X_t) = \mathcal{L}F\,dt + \partial_x F\,\sigma\,dW_t
   + \int\bigl[F(X_{t^-}+c)-F(X_{t^-})\bigr]\tilde N(dt,dz),

where the *generator* is

.. math::

   \mathcal{L}F = b\,\partial_x F + \tfrac12\sigma^2\partial_{xx}F
   + \int\bigl[F(x+c)-F(x)-c\,\partial_x F\bigr]\nu(dz).

---

## 4 · Optimal Control (HJB, PMP, Jumps)

### 4.1 Stochastic HJB

For :math:`dX_t = b(X_t,u_t)\,dt + \sigma(X_t,u_t)\,dW_t`, minimising
:math:`J = \mathbb{E}[\int_0^T \ell\,dt + g(X_T)]`, the value function
:math:`V(t,x) = \inf_u J` satisfies:

.. math::

   \boxed{
   -\partial_t V = \inf_{u\in\mathcal{U}}\Bigl[\ell(x,u) + \nabla_x V^{\!\top} b(x,u)
   + \tfrac12\operatorname{Tr}\bigl(\sigma\sigma^{\!\top}(x,u)\,\nabla_x^2 V\bigr)\Bigr],
   \quad V(T,\cdot)=g.
   }

Under smooth :math:`V`, the feedback law is
:math:`u^\star(t,x) = \arg\min_u[\ell(x,u)+\nabla_x V^\top b(x,u)]`.

**LQR special case** (:math:`\ell = x^\top Q x + u^\top R u`, :math:`b=Ax+Bu`):
:math:`V(t,x)=x^\top P(t)x + v(t)` with :math:`P` solving the *matrix Riccati ODE*:

.. math::

   -\dot P = A^\top P + PA - PBR^{-1}B^\top P + Q,\quad P(T)=Q_T.

### 4.2 Pontryagin Maximum Principle

The PMP avoids the curse of dimensionality — it converts the HJB PDE into a
two-point boundary-value problem in :math:`(X_t, p_t)`.

.. admonition:: Theorem (PMP)

   Define the Hamiltonian :math:`\mathcal{H}(x,u,p) = \ell(x,u)+p^\top b(x,u)`.
   If :math:`(X^\star, u^\star)` is optimal, there exists a costate process :math:`p_t` with:

   .. math::

      \dot p_t = -\nabla_x \mathcal{H}(X_t^\star, u_t^\star, p_t),\quad p_T = \nabla_x g(X_T^\star),

   and the optimality condition :math:`u_t^\star = \arg\min_u \mathcal{H}(X_t^\star, u, p_t)` holds a.e.

The costate pair :math:`(X_t^\star, p_t)` moves along Hamiltonian geodesics on
:math:`T^\star\mathbb{R}^d` — a direct link to symplectic geometry (§10.4).

### 4.3 HJB with Jumps (HJBI)

Adding the jump term from §3.4, the HJB equation gains a non-local integral operator:

.. math::

   -\partial_t V = \inf_{u}\Bigl[\ell + \nabla V^\top b + \tfrac12\operatorname{Tr}(\sigma\sigma^\top\nabla^2 V)
   + \underbrace{\int\bigl[V(x+c)-V(x)-\nabla V^\top c\bigr]\nu(dz)}_{\text{non-local jump term}}\Bigr].

Optimiz-rs's ``optimal_control`` module discretises the integral on a truncated support
:math:`[-z_{\max}, z_{\max}]` using Gaussian quadrature.

### 4.4 Viscosity Solutions

When :math:`V` fails to be :math:`C^{1,2}` (degenerate :math:`\sigma`, state constraints),
viscosity solutions (Crandall–Lions 1983) restore uniqueness:

.. admonition:: Definition — Viscosity Subsolution

   A continuous :math:`V` is a viscosity *subsolution* if for every smooth :math:`\phi`
   touching :math:`V` from above at :math:`(t_0,x_0)`:
   :math:`-\partial_t\phi(t_0,x_0) \le \inf_u[\ldots]` evaluated at :math:`\phi`.

Optimiz-rs's backward DP converges to the viscosity solution under the CFL condition
:math:`\Delta t \le C\,(\Delta x)^2`.

**Backward DP grid schema**::

    t=T    [ g(x_1)  g(x_2)  ...  g(x_n) ]   terminal condition
    t=T-1  [ V^1     V^2     ...  V^n    ]   one backward step
     .
     .
    t=0    [ V_0^1   V_0^2   ...  V_0^n  ]  -> optimal policy u*(x,0)

---

## 5 · Mean Field Games (1D Solver)

MFG couples a **backward HJB** (individual value) with a **forward Fokker–Planck** (population density):

.. math::

   \begin{aligned}
   \text{HJB (backward): } &
   -\partial_t u - \nu\partial_{xx}u + H(x,\partial_x u, m) = 0, & u(T,x)&=g(x),\\
   \text{Fokker–Planck (forward): } &
   \partial_t m - \nu\partial_{xx}m - \partial_x(m\,\partial_p H) = 0, & m(0,x)&=m_0(x).
   \end{aligned}

**Coupling:** :math:`H` depends on :math:`m` (mean-field interaction), creating a fixed-point problem.

**Fixed-point algorithm**::

    1. Initialise m^0 = m_0  (e.g. Gaussian)
    2. Solve HJB backward  -> u^{k+1}
    3. Extract optimal drift: alpha*(x,t) = -d_p H(x, d_x u^{k+1}, m^k)
    4. Solve Fokker-Planck forward with alpha* -> m^{k+1}
    5. Check ||m^{k+1} - m^k||_1 < eps; if not, k++ -> go to 2

**Convergence:** For monotone coupling (Lasry–Lions 2007), the system has a unique solution
and the fixed-point iteration contracts.

**Practical tip:** Monitor both :math:`\|m^{k+1}-m^k\|_1` and :math:`\|u^{k+1}-u^k\|_\infty`;
divergence of either signals non-monotone coupling or too large a time step.

---

## 6 · Kalman Filtering

### 6.1 Linear-Gaussian State Space

.. math::

   \mathbf{x}_t = F\mathbf{x}_{t-1} + \mathbf{w}_t,\; \mathbf{w}_t\sim\mathcal{N}(0,Q); \qquad
   \mathbf{y}_t = H\mathbf{x}_t + \mathbf{v}_t,\; \mathbf{v}_t\sim\mathcal{N}(0,R).

Two-step recursion:

.. math::

   \text{Predict:}\quad \hat{\mathbf{x}}^-_t = F\hat{\mathbf{x}}_{t-1},\quad P^-_t = FP_{t-1}F^\top+Q.

.. math::

   K_t = P^-_t H^\top(HP^-_t H^\top + R)^{-1};\quad
   \hat{\mathbf{x}}_t = \hat{\mathbf{x}}^-_t + K_t(\mathbf{y}_t - H\hat{\mathbf{x}}^-_t);\quad
   P_t = (I-K_t H)P^-_t.

:math:`K_t` is the *Kalman gain* — it interpolates between full prior trust (:math:`K\to0`)
and full observation trust (:math:`K\to H^{-1}`).

### 6.2 Information-Theoretic View

The Kalman filter computes the exact conditional mean
:math:`\hat{\mathbf{x}}_t = \mathbb{E}[\mathbf{x}_t \mid \mathbf{y}_{1:t}]` in Gaussian models
and minimises :math:`D_{\mathrm{KL}}(p(\mathbf{x}_t|\mathbf{y}_{1:t})\,\|\,\mathcal{N}(\hat{\mathbf{x}}_t, P_t))`
over all Gaussian approximations.

### 6.3 Continuous-Time Limit (Kalman–Bucy)

For :math:`d\mathbf{X}_t = A\mathbf{X}_t\,dt + B\,d\mathbf{W}_t`,
:math:`d\mathbf{Y}_t = C\mathbf{X}_t\,dt + d\mathbf{V}_t`, the error covariance satisfies
the *Riccati ODE*:

.. math::

   \dot P = AP + PA^\top + BQB^\top - PC^\top R^{-1}CP,\qquad P(0)=P_0,

which converges to the algebraic Riccati solution at steady state.

---

## 7 · MCMC (Metropolis–Hastings and Langevin)

### 7.1 Metropolis–Hastings

For target :math:`\pi(x) \propto e^{-U(x)}` and proposal :math:`q(x'\mid x)`:

.. math::

   \alpha(x\to x') = \min\!\Bigl(1, \frac{\pi(x')q(x\mid x')}{\pi(x)q(x'\mid x)}\Bigr).

**Detailed balance** :math:`\pi(x)\alpha(x\to x') = \pi(x')\alpha(x'\to x)`
ensures :math:`\pi` is the unique stationary distribution.

**Optimal scaling:** With Gaussian proposal :math:`q(x'|x)=\mathcal{N}(x,h^2 I_d)`,
step :math:`h^\star \approx 2.38/\sqrt{d}` (Roberts–Gelman–Gilks 1997) targets ~23–45 % acceptance.

### 7.2 Langevin Dynamics (MALA)

Metropolis-Adjusted Langevin proposal:

.. math::

   x' = x - \tfrac{h^2}{2}\nabla U(x) + h\,\xi, \quad \xi\sim\mathcal{N}(0,I_d),

a discretisation of the *overdamped Langevin SDE*:

.. math::

   dX_t = -\nabla U(X_t)\,dt + \sqrt{2}\,dW_t,

whose stationary distribution is exactly :math:`\pi \propto e^{-U}` (Fokker–Planck analysis).

MALA converges in :math:`O(d^{1/3})` steps vs :math:`O(d)` for RW-MH — a key advantage
for high-dimensional posteriors.

**Heuristic:** Tune proposal std so acceptance is ~25–45 %; see ``examples/notebooks/02_mcmc.ipynb`` for trace plots.

---

## 8 · Hidden Markov Models (HMM)

### 8.1 Model

Latent Markov chain :math:`Z_t \in \{1,\ldots,K\}` with transition matrix
:math:`A_{ij}=\mathbb{P}(Z_t=j\mid Z_{t-1}=i)` generates observations
:math:`Y_t \mid Z_t=k \sim B_k(y)`.

### 8.2 Baum–Welch (EM)

**E-step (forward–backward):**

.. math::

   \alpha_t(k) = B_k(y_t)\sum_j \alpha_{t-1}(j)A_{jk}, \qquad
   \beta_t(k) = \sum_j A_{kj}B_j(y_{t+1})\beta_{t+1}(j).

.. math::

   \gamma_t(k) = \frac{\alpha_t(k)\beta_t(k)}{\sum_j \alpha_t(j)\beta_t(j)}, \qquad
   \xi_t(j,k) = \frac{\alpha_t(j)A_{jk}B_k(y_{t+1})\beta_{t+1}(k)}{\mathcal{L}}.

**M-step:**

.. math::

   \hat A_{jk} = \frac{\sum_t \xi_t(j,k)}{\sum_t\gamma_t(j)}, \qquad
   \hat\mu_k = \frac{\sum_t \gamma_t(k)\,y_t}{\sum_t \gamma_t(k)}.

**Information-theoretic view:** Baum–Welch is EM on the complete-data log-likelihood; each
iteration monotonically increases :math:`\mathcal{L}(\theta)` by Jensen's inequality.

**Viterbi (MAP path):** Replace sum-product with max-product:
:math:`\delta_t(k) = \max_j \delta_{t-1}(j)A_{jk} \cdot B_k(y_t)`, runs in :math:`O(TK^2)`.

**Quality check:** Log-likelihood per EM iteration must be non-decreasing; a confusion matrix
of Viterbi labels vs. ground truth validates regime recovery.

---

## 9 · Information Theory

### 9.1 Entropy and KL Divergence

.. admonition:: Definition — KL Divergence

   For densities :math:`p, q`:

   .. math::

      D_{\mathrm{KL}}(p\,\|\,q) = \int p(x)\log\frac{p(x)}{q(x)}\,dx \;\ge\; 0,

   with equality iff :math:`p=q` a.e. (Gibbs' inequality).  Non-symmetric.

**Connection to model selection:** AIC :math:`= 2k - 2\ln\hat{\mathcal{L}}` and
BIC :math:`= k\ln n - 2\ln\hat{\mathcal{L}}` bound :math:`D_{\mathrm{KL}}(p_{\text{true}}\,\|\,p_\theta)`.

### 9.2 Fisher Information

.. admonition:: Definition — Fisher Information Matrix

   For parametric model :math:`p(x;\theta)`:

   .. math::

      \mathcal{I}(\theta)_{ij}
      = \mathbb{E}_{x\sim p}\!\left[\partial_{\theta_i}\log p\;\partial_{\theta_j}\log p\right]
      = -\mathbb{E}\!\left[\partial^2_{\theta_i\theta_j}\log p\right].

**Cramér–Rao bound:** Any unbiased estimator :math:`\hat\theta` satisfies
:math:`\operatorname{Cov}(\hat\theta) \succeq \mathcal{I}(\theta)^{-1}`.
MLE achieves equality asymptotically.

**Example — Gaussian HMM emission** :math:`B_k = \mathcal{N}(\mu_k,\sigma_k^2)`:
:math:`\mathcal{I}(\mu_k)=\sigma_k^{-2}`, :math:`\mathcal{I}(\sigma_k^2)=(2\sigma_k^4)^{-1}`.

### 9.3 Mutual Information and Feature Relevance

.. math::

   I(X;Y) = D_{\mathrm{KL}}\bigl(p(X,Y)\,\|\,p(X)p(Y)\bigr) = H(X) - H(X\mid Y) \ge 0.

**mRMR criterion** (minimum redundancy, maximum relevance) for the sparse module:

.. math::

   \max_{Y_i} \Bigl[I(Y_i;\text{target}) - \frac{1}{|S|}\sum_{Y_j\in S}I(Y_i;Y_j)\Bigr].

### 9.4 Natural Gradient (Preview)

Classical gradient descent ignores the geometry of parameter space.  The *natural gradient*
replaces :math:`\nabla_\theta\mathcal{L}` with :math:`\mathcal{I}(\theta)^{-1}\nabla_\theta\mathcal{L}`,
giving a reparametrisation-invariant update — see §10.2 for the full geometric development.

---

## 10 · Differential Geometry

### 10.1 Riemannian Manifolds

.. admonition:: Definition — Riemannian Manifold

   A *Riemannian manifold* :math:`(M, g)` is a smooth manifold :math:`M` with a
   *metric tensor* :math:`g_p`: a symmetric, positive-definite bilinear form on each
   tangent space :math:`T_p M`.

**Geodesics** (locally shortest paths) satisfy:

.. math::

   \ddot\gamma^k + \sum_{i,j}\Gamma^k_{ij}\,\dot\gamma^i\dot\gamma^j = 0,

where :math:`\Gamma^k_{ij} = \tfrac12 g^{kl}(\partial_i g_{jl}+\partial_j g_{il}-\partial_l g_{ij})`
are the *Christoffel symbols* encoding intrinsic curvature.

### 10.2 Information Geometry and Fisher–Rao Metric

The statistical manifold :math:`\mathcal{M} = \{p(\cdot;\theta)\}` carries the
**Fisher–Rao metric** :math:`g_{ij}(\theta) = \mathcal{I}(\theta)_{ij}`.

**Natural gradient (Amari 1998):** Steepest descent on :math:`(\mathcal{M}, g)`:

.. math::

   \theta \leftarrow \theta - \eta\,\mathcal{I}(\theta)^{-1}\nabla_\theta\mathcal{L}.

This is *invariant to reparametrisation* and achieves quadratic convergence on convex
objectives — equivalent to Fisher scoring.

**KL geometry:**
:math:`D_{\mathrm{KL}}(p_\theta\,\|\,p_{\theta+d\theta}) = \tfrac12\,d\theta^\top\mathcal{I}(\theta)\,d\theta + O(\|d\theta\|^3)`,
confirming Fisher–Rao as the intrinsic KL metric.

**Dually flat structure:** Exponential families
:math:`p(x;\theta)=h(x)\exp(\theta^\top T(x)-A(\theta))`
are :math:`e`-flat in natural parameters and :math:`m`-flat in mean parameters
:math:`\eta=\nabla A(\theta)`, with vanishing sectional curvature :math:`K=0` —
explaining exact Newton/natural-gradient convergence on these models.

### 10.3 Lie Groups and Geometric Control

.. admonition:: Definition — Lie Group

   A *Lie group* :math:`G` is a smooth manifold with a group structure where
   multiplication and inversion are smooth.  The *Lie algebra* :math:`\mathfrak{g} = T_e G`
   linearises the group at the identity.

**Examples:**

- :math:`SO(d)` — rotation group; portfolio factor rotation and orthogonality constraints.
- Heisenberg group — path-signature feature maps (used in ``lab_signature_methods``).

**Left-invariant control system on :math:`G`:**

.. math::

   \dot g(t) = g(t)\,\xi(t), \quad g\in G,\; \xi(t)\in\mathfrak{g}.

PMP on Lie groups yields the *Lie–Poisson (Euler–Poincaré) equations* (Holm–Marsden–Ratiu),
providing structure-preserving optimal trajectories.

### 10.4 Symplectic Geometry and Hamiltonian Structure

The phase space :math:`(T^\star M, \omega)` carries the symplectic 2-form
:math:`\omega = \sum_i dp_i \wedge dq_i`.  Hamilton's equations preserve :math:`\omega`
(*Liouville's theorem* — phase-space volume conserved).

**Connection to PMP:** The costate pair :math:`(X_t^\star, p_t)` solves Hamilton's equations,
i.e., the PMP is a symplectic flow on :math:`T^\star\mathbb{R}^d`.

**Symplectic integrators** (Störmer–Verlet, Ruth–Forest) preserve :math:`\omega` discretely,
keeping the Hamiltonian nearly constant over long horizons — critical for multi-year
allocation back-tests in Optimiz-rs.

### 10.5 Sectional Curvature and Landscape Geometry

The sectional curvature :math:`K(\sigma)` governs how quickly nearby geodesics diverge::

    K > 0 (sphere): geodesics converge   -> compact optimiser trajectories
    K = 0 (flat  ): Euclidean behaviour  -> Newton / natural gradient exact
    K < 0 (hyper.): exponential spread   -> efficient landscape exploration

For exponential families in natural/mean parameters :math:`K=0` — explaining exact
Newton convergence without curvature correction.

---

## Quick Reference

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Concept
     - Key equation / object
     - Optimiz-rs module
   * - Brownian motion
     - :math:`W_t - W_s \sim \mathcal{N}(0,t-s)`
     - ``point_processes``
   * - Itô SDE
     - :math:`dX=b\,dt+\sigma\,dW`
     - ``ou_estimator``
   * - Poisson / Compound Poisson
     - :math:`N_t\sim\text{Poisson}(\lambda t)`
     - ``point_processes``
   * - Lévy process
     - triplet :math:`(b,\sigma^2,\nu)`
     - ``point_processes``
   * - HJB PDE
     - :math:`-\partial_t V = \inf_u[\ell + \nabla V^\top b + \tfrac12\operatorname{Tr}\sigma\sigma^\top\nabla^2 V]`
     - ``optimal_control``
   * - HJBI (jumps)
     - :math:`+\int[V(\cdot+c)-V-\nabla V^\top c]\nu\,dz`
     - ``optimal_control``
   * - PMP costate
     - :math:`\dot p = -\nabla_x\mathcal{H}`, :math:`u^\star=\arg\min_u\mathcal{H}`
     - ``optimal_control``
   * - MFG (HJB + KFP)
     - fixed-point :math:`u,m`
     - ``mean_field_games``
   * - Kalman filter
     - :math:`K_t = P^-H^\top(HP^-H^\top+R)^{-1}`
     - ``optimal_control``
   * - MALA
     - :math:`x'=x-\tfrac{h^2}{2}\nabla U+h\xi`
     - ``mcmc``
   * - HMM
     - Baum–Welch EM + Viterbi
     - ``hmm``
   * - Fisher information
     - :math:`\mathcal{I}_{ij}=\mathbb{E}[\partial_i\ell\,\partial_j\ell]`
     - ``hmm``, ``sparse``
   * - Natural gradient
     - :math:`\mathcal{I}^{-1}\nabla_\theta\mathcal{L}`
     - ``differential_evolution``
   * - Riemannian / Lie geometry
     - Christoffel symbols, Lie–Poisson equations
     - experimental
   * - DE (jDE)
     - mutation + crossover + selection
     - ``differential_evolution``

---

## References

1. Øksendal, B. *Stochastic Differential Equations*, 6th ed. Springer, 2003.
2. Cont, R. & Tankov, P. *Financial Modelling with Jump Processes*. CRC Press, 2004.
3. Fleming, W.H. & Soner, H.M. *Controlled Markov Processes and Viscosity Solutions*. Springer, 2006.
4. Lasry, J.-M. & Lions, P.-L. "Mean field games." *Jpn. J. Math.* **2** (2007) 229–260.
5. Amari, S. *Information Geometry and Its Applications*. Springer, 2016.
6. do Carmo, M.P. *Riemannian Geometry*. Birkhäuser, 1992.
7. Holm, D.D., Marsden, J.E. & Ratiu, T.S. "The Euler–Poincaré equations." *Adv. Math.* **137** (1998).
8. Price, K.V., Storn, R.M. & Lampinen, J.A. *Differential Evolution*. Springer, 2005.
9. Roberts, G.O., Gelman, A. & Gilks, W.R. "Weak convergence of Metropolis algorithms." (1997).
10. Merton, R.C. "Option pricing when underlying stock returns are discontinuous." *JFE* **3** (1976).
11. Crandall, M.G. & Lions, P.-L. "Viscosity solutions of Hamilton–Jacobi equations." *Trans. AMS* (1983).
