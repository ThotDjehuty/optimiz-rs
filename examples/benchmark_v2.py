"""Honest v2 benchmark for optimiz-rs.

We compare each Rust primitive against the most natural pure-Python /
NumPy reference for the same task. The point is **not** to claim a
universal speedup, but to give users a realistic picture of where the
Rust core wins: intrinsically loopy / sequential algorithms that do
not vectorise cleanly in NumPy.

Run with:
    python examples/benchmark_v2.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

import optimizr as opt


def _bench(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# 1. HMM Baum-Welch -- intrinsically loopy
# ---------------------------------------------------------------------------

def _bench_hmm():
    rng = np.random.default_rng(42)
    n_obs = 5_000
    obs = np.concatenate([
        rng.normal(-1.0, 0.5, n_obs // 2),
        rng.normal(+1.0, 0.5, n_obs // 2),
    ]).reshape(-1, 1)

    def py_baum_welch():
        n = obs.shape[0]
        mu = np.array([-0.5, 0.5])
        sigma = np.array([1.0, 1.0])
        pi = np.array([0.5, 0.5])
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        for _ in range(10):
            B = np.exp(-(obs - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
            alpha = np.zeros((n, 2))
            alpha[0] = pi * B[0]
            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                alpha[t] /= alpha[t].sum() + 1e-300
            beta = np.zeros((n, 2))
            beta[-1] = 1.0
            for t in range(n - 2, -1, -1):
                beta[t] = A @ (B[t + 1] * beta[t + 1])
                beta[t] /= beta[t].sum() + 1e-300
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            mu = (gamma * obs).sum(axis=0) / gamma.sum(axis=0)
            sigma = np.sqrt(((obs - mu) ** 2 * gamma).sum(axis=0) / gamma.sum(axis=0))

    def rs_hmm():
        opt.fit_hmm(obs.flatten().tolist(), 2, 10, 1e-6)

    np_t = _bench(py_baum_welch, repeat=2)
    rs_t = _bench(rs_hmm, repeat=3)
    return ("HMM Baum-Welch (2 states, 5_000 obs, 10 iters)", np_t, rs_t)


# ---------------------------------------------------------------------------
# 2. Differential evolution -- multi-modal global optimisation
# ---------------------------------------------------------------------------

def _bench_differential_evolution():
    from scipy.optimize import differential_evolution as scipy_de  # type: ignore

    rastrigin = lambda x: 10 * len(x) + sum(xi * xi - 10 * math.cos(2 * math.pi * xi) for xi in x)
    bounds = [(-5.12, 5.12)] * 5

    def py_de():
        scipy_de(rastrigin, bounds, maxiter=50, popsize=20, seed=0, tol=0.0, polish=False)

    def rs_de():
        opt.differential_evolution(rastrigin, bounds, maxiter=50, popsize=20, seed=0)

    np_t = _bench(py_de, repeat=2)
    rs_t = _bench(rs_de, repeat=3)
    return ("Differential evolution (Rastrigin d=5, 50 iters x 20 pop)", np_t, rs_t)


# ---------------------------------------------------------------------------
# 3. Path signature
# ---------------------------------------------------------------------------

def _bench_signature():
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.standard_normal((300, 3)) * 0.05, axis=0)

    def py_signature():
        d = path.shape[1]
        increments = np.diff(path, axis=0)
        s1 = np.zeros(d)
        s2 = np.zeros((d, d))
        s3 = np.zeros((d, d, d))
        for inc in increments:
            s1 = s1 + inc
            s2 = s2 + 0.5 * np.outer(inc, inc)
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        s3[i, j, k] += inc[i] * inc[j] * inc[k] / 6.0

    def rs_signature():
        opt.path_signature(path.tolist(), 3)

    np_t = _bench(py_signature, repeat=2)
    rs_t = _bench(rs_signature, repeat=3)
    return ("Path signature (T=300, d=3, depth=3)", np_t, rs_t)


# ---------------------------------------------------------------------------
# 4. MCMC random-walk Metropolis
# ---------------------------------------------------------------------------

def _bench_mcmc():
    n_samples = 5_000
    rng = np.random.default_rng(1)

    def py_mh():
        def logp(x):
            a, b = 1.0, 100.0
            return -((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2) / 20.0
        x = np.array([0.0, 0.0])
        lp = logp(x)
        out = np.zeros((n_samples, 2))
        for i in range(n_samples):
            cand = x + rng.normal(scale=0.5, size=2)
            lpc = logp(cand)
            if math.log(rng.random() + 1e-300) < lpc - lp:
                x, lp = cand, lpc
            out[i] = x

    def rs_mh():
        # mcmc_sample expects a Python log-density callable; that's a fair
        # comparison since the python reference also calls a python logp.
        def logp(x):
            a, b = 1.0, 100.0
            return -((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2) / 20.0
        try:
            opt.mcmc_sample(logp, [0.0, 0.0], n_samples, 0.5)
        except TypeError:
            # Fallback signature: positional only
            opt.mcmc_sample(logp, [0.0, 0.0], n_samples)

    np_t = _bench(py_mh, repeat=2)
    rs_t = _bench(rs_mh, repeat=3)
    return (f"MCMC random-walk MH ({n_samples} samples, d=2)", np_t, rs_t)


# ---------------------------------------------------------------------------
# 5. Hawkes process simulation -- sequential, O(N^2) reference
# ---------------------------------------------------------------------------

def _bench_hawkes():
    T = 100.0
    mu = 1.0
    alpha = 0.6
    beta = 1.2

    def py_hawkes():
        rng = np.random.default_rng(0)
        events = []
        t = 0.0
        lam_max = mu
        while t < T:
            t += rng.exponential(1.0 / max(lam_max, 1e-9))
            if t >= T:
                break
            lam = mu + alpha * sum(math.exp(-beta * (t - s)) for s in events)
            if rng.random() <= lam / lam_max:
                events.append(t)
            lam_max = lam + alpha
        return events

    def rs_hawkes():
        opt.simulate_hawkes(mu, alpha, beta, T, "exponential", 0)

    np_t = _bench(py_hawkes, repeat=2)
    rs_t = _bench(rs_hawkes, repeat=3)
    return (f"Hawkes process (T={T}, mu={mu}, alpha={alpha}, beta={beta})", np_t, rs_t)


def main():
    print("Running v2 benchmarks (best of N runs, single-threaded)\n")
    rows = []
    for fn in (_bench_hmm, _bench_differential_evolution, _bench_signature,
               _bench_mcmc, _bench_hawkes):
        try:
            rows.append(fn())
        except Exception as exc:  # pragma: no cover
            rows.append((f"{fn.__name__} (FAILED: {exc})", float('nan'), float('nan')))

    md = ["| Workload | Pure Python / NumPy | optimiz-rs (Rust) | Speedup |",
          "|---|---:|---:|---:|"]
    for name, np_t, rs_t in rows:
        if math.isnan(np_t) or math.isnan(rs_t):
            md.append(f"| {name} | n/a | n/a | n/a |")
            continue
        speedup = np_t / rs_t if rs_t > 0 else float("inf")
        md.append(f"| {name} | {np_t * 1e3:8.2f} ms | {rs_t * 1e3:8.2f} ms | **{speedup:5.1f}×** |")

    table = "\n".join(md)
    print(table)

    out = Path(__file__).with_name("benchmark_v2.md")
    out.write_text(
        "# optimiz-rs v2.0 benchmark\n\n"
        "Best wall-clock over a few runs. Single-threaded. "
        "Workloads chosen to be intrinsically loopy / sequential -- the "
        "regime where the Rust core delivers a real speedup over a NumPy "
        "reference.\n\n"
        "Workloads that are fully vectorisable in NumPy (e.g. drift updates "
        "for an N-particle SDE with no callback) are not included: a tight "
        "NumPy loop on contiguous arrays is hard to beat from Rust through "
        "a PyO3 callback boundary.\n\n"
        + table
        + "\n"
    )
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
