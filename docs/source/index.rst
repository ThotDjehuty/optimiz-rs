.. Optimiz-rs documentation master file

Optimiz-rs Documentation
======================

**High-performance optimization algorithms in Rust with Python bindings**

.. image:: https://img.shields.io/badge/version-1.0.0-blue.svg
   :target: https://github.com/ThotDjehuty/optimiz-r/releases
   :alt: Version
   
.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/ThotDjehuty/optimiz-r/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/pypi-optimiz--rs-blue.svg
   :target: https://pypi.org/project/optimiz-rs/
   :alt: PyPI

.. image:: https://img.shields.io/badge/crates.io-optimiz--rs-orange.svg
   :target: https://crates.io/crates/optimiz-rs
   :alt: crates.io

Optimiz-rs provides blazingly fast, production-ready implementations of advanced optimization and statistical inference algorithms. Built with Rust for maximum performance and exposed to Python through PyO3, it delivers **50-100× speedup** over pure Python implementations.

**🚀 Quick Integration with Polarway** — Combine Optimiz-rs algorithms with Polarway's high-performance DataFrame engine for streaming time-series analytics:

.. raw:: html

   <div class="grid cards" style="margin: 1.5rem 0;">
     <div style="padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; background: #fff; transition: all 0.2s;">
       <h3 style="margin-top: 0;">🔄 Time-Series Regime Detection</h3>
       <p style="color: #64748b; margin: 0;">Polarway streams OHLCV data → Optimiz-rs HMM classifies market regimes</p>
     </div>
     <div style="padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; background: #fff; transition: all 0.2s;">
       <h3 style="margin-top: 0;">📊 Bayesian Inference at Scale</h3>
       <p style="color: #64748b; margin: 0;">Polarway distributed storage → Optimiz-rs MCMC for probabilistic modeling</p>
     </div>
     <div style="padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; background: #fff; transition: all 0.2s;">
       <h3 style="margin-top: 0;">⚡ Strategy Optimization</h3>
       <p style="color: #64748b; margin: 0;">Polarway hybrid storage → Optimiz-rs Differential Evolution (74-88× speedup)</p>
     </div>
     <div style="padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; background: #fff; transition: all 0.2s;">
       <h3 style="margin-top: 0;">🧠 Mean Field Games</h3>
       <p style="color: #64748b; margin: 0;">Polarway time-series → Optimiz-rs MFG for agent population dynamics</p>
     </div>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   getting-started
   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: Algorithms
   
   algorithms/differential_evolution
   algorithms/mean_field_games
   algorithms/hmm
   algorithms/mcmc
   algorithms/sparse_optimization
   algorithms/optimal_control
   algorithms/risk_metrics
   algorithms/grid_search
   algorithms/point_processes

.. toctree::
   :maxdepth: 2
   :caption: v1.1 Numerical Primitives

   algorithms/matrix_riccati
   algorithms/nonsync_covariance
   algorithms/wavelet
   algorithms/risk_measures
   algorithms/graph_spectral
   algorithms/topology
   algorithms/volterra
   algorithms/signatures

.. toctree::
   :maxdepth: 2
   :caption: v2.0 Generic Stochastic Control & PDE

   algorithms/bsde
   algorithms/pde
   algorithms/stochastic_control
   algorithms/quadratic_impact_control
   algorithms/mckean_vlasov
   algorithms/agent_based
   algorithms/robust_drift
   algorithms/generative_calibration_hooks

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/differential_evolution
   api/grid_search
   api/hmm
   api/mcmc
   api/sparse
   api/optimal_control
   api/risk_metrics
   api/point_processes

.. toctree::
   :maxdepth: 1
   :caption: Advanced
   
   theory/mathematical_foundations
   mfg_tutorial
   benchmarks
   contributing
   changelog

Features
--------

✨ **Algorithms Included:**

- **Mean Field Games**: 1D MFG solver, HJB-Fokker-Planck coupling, agent population dynamics
- **Differential Evolution**: 5 strategies (rand/1, best/1, current-to-best/1, rand/2, best/2), adaptive jDE
- **Optimal Control**: HJB solvers, regime switching, jump diffusion, MRSJD framework
- **Hidden Markov Models**: Baum-Welch training, Viterbi decoding, Gaussian emissions
- **MCMC Sampling**: Metropolis-Hastings, adaptive proposals, Bayesian inference
- **Sparse Optimization**: Sparse PCA, Box-Tao decomposition, Elastic Net, ADMM
- **Risk Metrics**: Hurst exponent, half-life estimation, time series analysis
- **Information Theory**: Mutual information, Shannon entropy, feature selection
- **Time-Series Helpers**: Rolling Hurst/half-life, feature prep for HMM, lagged feature builders
- **Parallelization**: Rust-native population evaluation with Rayon for built-in objectives

🚀 **Performance:**

- **50-100× faster** than pure Python implementations
- **95% memory reduction** vs NumPy/SciPy
- **Parallel-ready** with Rayon infrastructure
- Production-tested on multi-dimensional problems

Quick Example
-------------

.. code-block:: python

    import numpy as np
    from optimizr import DifferentialEvolution
    
    # Define objective function
    def sphere(x):
        return np.sum(x**2)
    
    # Optimize
    de = DifferentialEvolution(
        bounds=[(-5, 5)] * 10,
        strategy="best/1/bin",
        population_size=50
    )
    result = de.optimize(sphere, max_iterations=100)
    
    print(f"Best fitness: {result.best_fitness:.6f}")
    print(f"Best solution: {result.best_solution}")

Installation
------------

From PyPI:

.. code-block:: bash

    pip install optimiz-rs

From source:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/ThotDjehuty/optimiz-r.git
    cd optimiz-r
    
    # Build and install
    pip install maturin
    maturin develop --release

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
