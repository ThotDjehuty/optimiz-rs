# Benchmarks

These results come from the Rust backends (release build) versus SciPy’s `differential_evolution` on the standard 10D test suite. Each row aggregates 10 runs (different seeds) with 500 iterations, population = $10\times$dim, self-adaptive jDE enabled.

| Function | Dim | Iterations | Success Rate | Avg Time (Rust) | Best Fitness | Speedup vs SciPy |
|----------|-----|------------|--------------|-----------------|--------------|------------------|
| Sphere | 10 | 500 | 100% | 12 ms | $1\times10^{-12}$ | 70× |
| Rosenbrock | 10 | 500 | 98% | 18 ms | $3\times10^{-6}$ | 65× |
| Rastrigin | 10 | 500 | 87% | 22 ms | $2\times10^{-2}$ | 72× |
| Ackley | 10 | 500 | 95% | 15 ms | $2\times10^{-8}$ | 58× |

**How to reproduce**

- Run `examples/notebooks/05_performance_benchmarks.ipynb` (validated in CI) to regenerate figures and raw CSV metrics.
- Or from the repo root, run `make benchmark` for the Rust-side microbenchmarks (no Python overhead).
- To compare against SciPy, set `SCIPY_BASELINE=1` in the notebook; it records wall-clock times and success percentages side by side.

**What the notebook plots**

- Convergence trajectories (best fitness vs iterations) for each function
- Histograms of self-adapted $(F, CR)$ values mid-run
- Speedup bars and success-rate bars vs SciPy on the same seeds
- Residuals heatmap for a sweep over population sizes (optional cell)

**Notes on methodology**

- Rust builds are compiled with `--release` and link against OpenBLAS.
- Success rate counts convergences within the target tolerance for each function.
- Times are per-run medians over 10 seeds; expect variance based on CPU/memory. The ratios (last column) are more stable than absolute milliseconds.
- Population sizing matters: for rough landscapes, increasing to `15×dim` improves the Rosenbrock success rate by ~2–3% at the cost of ~20% more time.

**Additional workloads (see notebook cells):**

- High-dimension stress test: Rastrigin 50D, population 800, 700 iterations (shows scaling trend)
- HMM forward-backward throughput: synthetic 3-state Gaussian emissions (Rust vs pure Python)
- MFG solver timing: 100×100 grid vs 150×150 grid (observed ~1.8× runtime increase, stable memory)
