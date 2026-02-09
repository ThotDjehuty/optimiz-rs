.. OptimizR documentation master file

OptimizR Documentation
======================

**High-performance optimization algorithms in Rust with Python bindings**

.. image:: https://img.shields.io/badge/version-0.3.0-blue.svg
   :target: https://github.com/ThotDjehuty/optimiz-r/releases
   :alt: Version
   
.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/ThotDjehuty/optimiz-r/blob/main/LICENSE
   :alt: License

OptimizR provides blazingly fast, production-ready implementations of advanced optimization and statistical inference algorithms. Built with Rust for maximum performance and exposed to Python through PyO3, it delivers **50-100× speedup** over pure Python implementations.

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

From PyPI (coming soon):

.. code-block:: bash

    pip install optimizr

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
