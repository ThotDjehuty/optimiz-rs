Agent-based — bounded-confidence consensus
==========================================

Generic interacting-agent simulator (`consensus_dynamics`) — linear bounded-confidence rule $s_i^{k+1} = (1-α) s_i^k + α \bar s^k + ξ_i$.

.. note:: Companion executed notebook: `15_agent_based.ipynb <../../examples/notebooks/15_agent_based.ipynb>`_

15 — Agent-based dynamics
=========================

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from optimizr import _core as opt
   plt.rcParams['figure.figsize'] = (7, 4)
   plt.rcParams['figure.dpi'] = 110

.. code-block:: python

   init = np.arange(40.0).tolist()
   init_mean = float(np.mean(init))
   res = opt.consensus_dynamics(init, alpha=0.3, noise_sigma=0.1,
                                 n_steps=80, seed=0)
   n_t = res['n_steps']; n_a = res['n_agents']
   S = np.array(res['states_flat']).reshape(n_t, n_a)
   mean_traj = np.array(res['mean_trajectory'])
   print('initial mean =', init_mean)
   print('final mean   =', mean_traj[-1])
   print('final std    =', float(S[-1].std()))

.. code-block:: python

   fig, ax = plt.subplots()
   for i in range(n_a):
       ax.plot(S[:, i], color='tab:blue', alpha=0.3, lw=0.6)
   ax.plot(mean_traj, color='red', lw=2, label='empirical mean')
   ax.axhline(init_mean, color='k', ls=':', label='initial mean')
   ax.set_xlabel('step k'); ax.set_ylabel('s^k_i'); ax.legend(); ax.grid(alpha=0.3)
   ax.set_title('Bounded-confidence consensus, α = 0.3')
   fig.tight_layout(); plt.show()







.. AUTO-PLOT-BEGIN
.. image:: ../_static/auto/algorithms__agent_based/block_03_fig_01.png
   :align: center
   :width: 80%

.. AUTO-PLOT-END
.. image:: ../_static/v2/agent_based/plot_01.png
   :align: center
   :width: 80%

.. code-block:: python

   fig, ax = plt.subplots()
   for alpha in [0.05, 0.1, 0.3, 0.6, 1.0]:
       r = opt.consensus_dynamics(init, alpha=alpha, noise_sigma=0.0, n_steps=60, seed=0)
       S = np.array(r['states_flat']).reshape(r['n_steps'], r['n_agents'])
       spread = S.max(axis=1) - S.min(axis=1)
       ax.semilogy(spread, label=f'α = {alpha:g}')
   ax.set_xlabel('step k'); ax.set_ylabel('max_i s − min_i s')
   ax.set_title('Convergence rate vs averaging weight α'); ax.legend(); ax.grid(alpha=0.3)
   fig.tight_layout(); plt.show()







.. AUTO-PLOT-BEGIN
.. image:: ../_static/auto/algorithms__agent_based/block_04_fig_01.png
   :align: center
   :width: 80%

.. AUTO-PLOT-END
.. image:: ../_static/v2/agent_based/plot_02.png
   :align: center
   :width: 80%

**Verified:** without noise, the empirical mean is exactly preserved and the spread decays geometrically.

API
---

.. code-block:: rust

   pub fn simulate_agent_based<T>(initial: &[f64], transition: T, cfg: &AgentBasedConfig) -> Result<AgentBasedResult>
   where T: Fn(f64, &[f64], usize) -> f64;

   pub struct AgentBasedConfig { pub n_agents: usize, pub n_steps: usize, pub noise_sigma: f64, pub seed: u64 }
   pub struct AgentBasedResult { pub states: Array2<f64>, pub mean_trajectory: Array1<f64> }
