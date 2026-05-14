# optimiz-rs v2.0 benchmark

Best wall-clock over a few runs. Single-threaded. Workloads chosen to be intrinsically loopy / sequential -- the regime where the Rust core delivers a real speedup over a NumPy reference.

Workloads that are fully vectorisable in NumPy (e.g. drift updates for an N-particle SDE with no callback) are not included: a tight NumPy loop on contiguous arrays is hard to beat from Rust through a PyO3 callback boundary.

| Workload | Pure Python / NumPy | optimiz-rs (Rust) | Speedup |
|---|---:|---:|---:|
| HMM Baum-Welch (2 states, 5_000 obs, 10 iters) |   970.94 ms |    14.34 ms | ** 67.7×** |
| Differential evolution (Rastrigin d=5, 50 iters x 20 pop) |   417.45 ms |    30.03 ms | ** 13.9×** |
| Path signature (T=300, d=3, depth=3) |    11.07 ms |     0.99 ms | ** 11.2×** |
| MCMC random-walk MH (5000 samples, d=2) |    35.41 ms |    20.24 ms | **  1.7×** |
| Hawkes process (T=100.0, mu=1.0, alpha=0.6, beta=1.2) |     2.75 ms |     0.83 ms | **  3.3×** |
