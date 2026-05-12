Generative calibration — Gaussian MMD loss
==========================================

Maximum-Mean-Discrepancy distance with Gaussian kernel (`mmd_gaussian`).  Self-distance is exactly zero; the metric grows monotonically with sample shift.

.. note:: Companion executed notebook: `17_generative_calibration.ipynb <../../examples/notebooks/17_generative_calibration.ipynb>`_

17 — MMD calibration loss
=========================

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from optimizr import _core as opt
   plt.rcParams['figure.figsize'] = (7, 4)
   plt.rcParams['figure.dpi'] = 110

.. code-block:: python

   x = np.linspace(0.0, 5.0, 80)
   shifts = np.linspace(0.0, 6.0, 40)
   d = [opt.mmd_gaussian(x.tolist(), (x + s).tolist(), 1.0) for s in shifts]
   print('MMD self =', d[0])
   print('MMD at shift 6.0 =', d[-1])

.. code-block:: python

   fig, ax = plt.subplots()
   ax.plot(shifts, d, lw=2)
   ax.set_xlabel('translation Δ'); ax.set_ylabel('MMD(P, P + Δ)')
   ax.set_title('Gaussian-kernel MMD vs translation (σ = 1)')
   ax.grid(alpha=0.3); fig.tight_layout(); plt.show()





.. AUTO-PLOT-BEGIN
.. image:: ../_static/auto/algorithms__generative_calibration_hooks/block_03_fig_01.png
   :align: center
   :width: 80%

.. AUTO-PLOT-END
.. image:: ../_static/v2/generative_calibration_hooks/plot_01.png
   :align: center
   :width: 80%

Bandwidth dependence
--------------------

.. code-block:: python

   fig, ax = plt.subplots()
   for sigma in [0.25, 0.5, 1.0, 2.0]:
       d = [opt.mmd_gaussian(x.tolist(), (x + s).tolist(), sigma) for s in shifts]
       ax.plot(shifts, d, label=f'σ = {sigma:g}')
   ax.set_xlabel('translation Δ'); ax.set_ylabel('MMD'); ax.legend(); ax.grid(alpha=0.3)
   ax.set_title('MMD as a function of kernel bandwidth')
   fig.tight_layout(); plt.show()





.. AUTO-PLOT-BEGIN
.. image:: ../_static/auto/algorithms__generative_calibration_hooks/block_04_fig_01.png
   :align: center
   :width: 80%

.. AUTO-PLOT-END
.. image:: ../_static/v2/generative_calibration_hooks/plot_02.png
   :align: center
   :width: 80%

**Verified:** `MMD(x, x) = 0`; metric is strictly monotonic in shift.

API
---

.. code-block:: rust

   pub fn mmd_distance(x: &[f64], y: &[f64], loss: &MmdLoss) -> Result<f64>;
   pub fn calibration_step<S: GenerativeSampler>(sampler: &mut S, target: &[f64], loss: &MmdLoss, lr: f64) -> Result<f64>;
   pub trait GenerativeSampler { fn sample(&self, n: usize, seed: u64) -> Vec<f64>; fn parameters(&self) -> Vec<f64>; fn perturb(&mut self, deltas: &[f64]); }
   pub struct MmdLoss { pub sigma: f64 }
