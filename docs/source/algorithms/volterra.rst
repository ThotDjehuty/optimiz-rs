Volterra and Fractional Solvers
================================

The module :code:`volterra` collects four CPU-only generic numerical
primitives for Volterra integral equations and related transforms.

Fractional Caputo Adams Solver
------------------------------

For :math:`\alpha \in (0, 1)`, solve

.. math::

   D^\alpha h(t) = F(t, h(t)),
   \qquad h(0) = h_0,

with the Diethelm--Ford--Freed (2002) fractional Adams predictor--
corrector. Predictor:

.. math::

   h^P_{n+1}
   = h_0
   + \frac{\Delta t^\alpha}{\alpha\,\Gamma(\alpha)}
       \sum_{k=0}^{n}
       \big[(n+1-k)^\alpha - (n-k)^\alpha\big]\, F(t_k, h_k).

Corrector:

.. math::

   h_{n+1}
   = h_0
   + \frac{\Delta t^\alpha}{\Gamma(\alpha + 2)}
   \Big[ F(t_{n+1}, h^P_{n+1}) + \sum_{k=0}^{n} a_{n+1, k}\, F(t_k, h_k) \Big],

with

.. math::

   a_{n+1, 0} = n^{\alpha + 1} - (n - \alpha)\,(n+1)^\alpha,

   a_{n+1, k} = (n - k + 2)^{\alpha + 1} + (n - k)^{\alpha + 1}
                - 2\,(n - k + 1)^{\alpha + 1},
   \qquad 1 \le k \le n.

Markovian Lift
--------------

A convolution kernel :math:`K(t)` admitting

.. math::

   K(t) = \int_0^\infty e^{-\gamma t}\, \nu(d\gamma)

is approximated by

.. math::

   K(t) \;\approx\; \sum_{j=1}^N c_j\, e^{-\gamma_j t},
   \qquad c_j \ge 0,

with rates :math:`\gamma_j` on a geometric grid and weights fitted by
non-negative least squares.

Generic Volterra Equation
-------------------------

For

.. math::

   y(t) = g(t) + \int_0^t K(t - s, y(s))\, ds,

the trapezoidal product-integration scheme reads

.. math::

   y_n = g_n + \Delta t\,\Big[
       \tfrac{1}{2} K(t_n, y_0)
       + \sum_{k=1}^{n-1} K(t_n - t_k, y_k)
       + \tfrac{1}{2} K(0, y_n) \Big],

solved implicitly by fixed-point iteration on :math:`y_n`.

Fourier Inversion
-----------------

Recover a density from a characteristic function :math:`\varphi(u)` via

.. math::

   f(x) \;\approx\;
   \frac{\Delta u}{\pi}\,
   \sum_{k=0}^{N_u - 1}
   w_k \big[\,\Re \varphi(u_k)\,\cos(u_k x)
              + \Im \varphi(u_k)\,\sin(u_k x)\,\big],

with trapezoidal weights :math:`w_k`.

API
---

.. code-block:: rust

   pub fn solve_fractional_ode<F: Fn(f64, f64) -> f64>(
       h0: f64, alpha: f64, t_horizon: f64, n_steps: usize, rhs: F,
   ) -> Result<FractionalOdeResult>;

   pub fn geometric_grid_lift<K: Fn(f64) -> f64>(
       kernel: K, t_samples: &[f64],
       n_factors: usize, gamma_min: f64, gamma_max: f64, nnls_iter: usize,
   ) -> Result<MarkovianLift>;

   pub fn solve_volterra<G, K>(
       g: G, kernel: K, t_horizon: f64, n_steps: usize,
       fixed_point_iter: usize, fixed_point_tol: f64,
   ) -> Result<VolterraResult>;

   pub fn fourier_invert<P: Fn(f64) -> (f64, f64)>(
       phi: P, x_grid: &[f64], u_max: f64, n_u: usize,
   ) -> Result<DensityResult>;
