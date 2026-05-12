Matrix Riccati Solver
=====================

The module :code:`optimal_control::matrix_riccati` integrates backward in time
the matrix Riccati differential equation

.. math::

   \frac{dA(t)}{dt} = -2\,A(t)\,M\,A(t) + Q,
   \qquad A(T) = A_T,

together with the affine and constant components

.. math::

   \frac{dB(t)}{dt} = -2\,A(t)\,M\,B(t),
   \qquad B(T) = B_T,

.. math::

   \frac{dC(t)}{dt} = -B(t)^\top\,M\,B(t),
   \qquad C(T) = C_T.

Discretisation
--------------

The grid :math:`\{t_n = T - n\,\Delta t\}_{n=0}^{N}` with
:math:`\Delta t = T / N` is traversed backward and a classical RK4 step is
applied to the joint vector field :math:`(A, B, C)`. Each macro step is
optionally subdivided into :math:`s` sub-steps for stability on stiff
problems.

Validation
----------

In the scalar case :math:`A, M, Q \in \mathbb{R}` with :math:`A(T) = 0`,

.. math::

   A(t) \;=\; -\sqrt{\frac{Q}{2M}}\;\tanh\!\Big(\sqrt{2QM}\,(T - t)\Big),

a closed form used by the unit test :code:`scalar_riccati_matches_analytic`
to certify :math:`L^\infty` convergence below :math:`10^{-5}` on
:math:`[0, T]`.

API
---

.. code-block:: rust

   pub fn solve_matrix_riccati(
       m_matrix: ArrayView2<f64>,
       q: ArrayView2<f64>,
       n: ArrayView2<f64>,
       a_terminal: ArrayView2<f64>,
       b_terminal: ArrayView1<f64>,
       c_terminal: f64,
       t_horizon: f64,
       config: RiccatiConfig,
   ) -> Result<RiccatiResult>;
