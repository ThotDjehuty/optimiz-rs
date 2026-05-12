Inference — Huber-IRLS drift estimator
======================================

Robust drift estimator (`robust_drift`) for $x_{k+1} = x_k + (a + b x_k) Δt + σ ε_k$ via Huber IRLS — resists 5 % heavy-tailed innovations.

.. note:: Companion executed notebook: `16_robust_drift.ipynb <../../examples/notebooks/16_robust_drift.ipynb>`_

16 — Robust drift estimation
============================

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from optimizr import _core as opt
   plt.rcParams['figure.figsize'] = (7, 4)
   plt.rcParams['figure.dpi'] = 110

Synthetic stationary process with 5 % outliers
----------------------------------------------

.. code-block:: python

   rng = np.random.default_rng(7)
   true_a, true_b = 1.0, -0.5
   dt, n = 0.01, 5000
   x = [0.0]
   for k in range(n):
       if k % 20 == 0:
           eps = rng.uniform(-2.0, 2.0)
       else:
           eps = rng.uniform(-0.1, 0.1)
       x.append(x[-1] + (true_a + true_b * x[-1]) * dt + eps * np.sqrt(dt))
   x = np.array(x)
   print('observation length =', len(x))

.. code-block:: python

   fig, ax = plt.subplots()
   ax.plot(x, lw=0.6)
   ax.axhline(true_a / -true_b, color='red', ls='--', label='OU level a/(-b) = 2')
   ax.set_xlabel('k'); ax.set_ylabel('x_k'); ax.legend(); ax.grid(alpha=0.3)
   ax.set_title('Synthetic series with heavy-tailed innovations')
   fig.tight_layout(); plt.show()







.. AUTO-PLOT-BEGIN
.. image:: ../_static/auto/algorithms__robust_drift/block_03_fig_01.png
   :align: center
   :width: 80%

.. AUTO-PLOT-END
.. image:: ../_static/v2/robust_drift/plot_01.png
   :align: center
   :width: 80%

.. code-block:: python

   res = opt.robust_drift(x.tolist(), dt=dt)
   print(f'a (true 1.0)  ->  {res["a"]:.4f}')
   print(f'b (true -0.5) ->  {res["b"]:.4f}')
   print('IRLS iterations =', res['iterations'])

.. code-block:: python

   # Compare against a naïve OLS that is broken by outliers.
   y = (x[1:] - x[:-1]) / dt
   X = np.vstack([np.ones_like(x[:-1]), x[:-1]]).T
   ols_ab, *_ = np.linalg.lstsq(X, y, rcond=None)
   print('OLS a, b =', ols_ab)
   fig, ax = plt.subplots()
   labels = ['true', 'OLS', 'robust']
   vals_a = [true_a, ols_ab[0], res['a']]
   vals_b = [true_b, ols_ab[1], res['b']]
   ax.bar(np.arange(3) - 0.2, vals_a, width=0.4, label='a')
   ax.bar(np.arange(3) + 0.2, vals_b, width=0.4, label='b')
   ax.set_xticks(range(3)); ax.set_xticklabels(labels)
   ax.legend(); ax.grid(alpha=0.3); ax.set_title('Robust vs OLS drift estimate')
   fig.tight_layout(); plt.show()







.. AUTO-PLOT-BEGIN
.. image:: ../_static/auto/algorithms__robust_drift/block_05_fig_01.png
   :align: center
   :width: 80%

.. AUTO-PLOT-END
.. image:: ../_static/v2/robust_drift/plot_02.png
   :align: center
   :width: 80%

**Verified:** Huber IRLS recovers `(a, b)` within `0.2` even with 5 % heavy outliers.

API
---

.. code-block:: rust

   pub fn estimate_robust_drift(observations: &[f64], cfg: &RobustDriftConfig) -> Result<RobustDriftResult>;
   pub struct RobustDriftConfig { pub dt: f64, pub huber_delta: f64, pub max_iterations: usize, pub tolerance: f64 }
   pub struct RobustDriftResult { pub a: f64, pub b: f64, pub iterations: usize }
