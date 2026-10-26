"""Self-contained synthetic reproduction of two confirmed defects in optimizr.

Run:  python tests/repro_hmm_hurst_defects.py
No market data, no network, no files. Only numpy + optimizr.
"""
import numpy as np
import optimizr as opt


def defect_a_hurst_saturates_at_099():
    """estimate_hurst returns exactly 0.990000 for any integrated series."""
    R = np.random.RandomState(42)
    rw = np.cumsum(R.randn(2000)).astype(np.float64)
    line = np.arange(2000, dtype=np.float64)
    noise = R.randn(2000).astype(np.float64)
    print('[A] random walk   H = %.6f  (expected ~1.0)' % opt.estimate_hurst(rw))
    print('[A] straight line H = %.6f  (expected  1.0)' % opt.estimate_hurst(line))
    print('[A] white noise   H = %.6f  (expected ~0.5)' % opt.estimate_hurst(noise))
    print('[A] DEFECT: a deterministic line and a random walk give the same clamped 0.99')


def defect_b_initial_probs_never_updated():
    """fit_hmm always returns a uniform initial distribution."""
    R = np.random.RandomState(7)
    st, out = 0, []
    for _ in range(12151):
        if R.rand() < 0.002:
            st = 1 - st
        out.append(R.randn() * (0.0003 if st == 0 else 0.0012))
    x = np.asarray(out, dtype=np.float64)
    for n in (1000, 5000, 12151):
        for k in (2, 3):
            p = opt.fit_hmm(x[:n].tolist(), k, 500, 1e-12)
            pi = np.asarray(p.initial_probs, dtype=float)
            print('[B] n=%-6d k=%d initial_probs = %s' % (n, k, ' '.join('%.6f' % v for v in pi)))
    print('[B] DEFECT: pi is exactly uniform everywhere, although the series starts in state 0')


def defect_c_transition_rows_lose_normalization_with_kurtosis():
    """Transition matrix row sums fall below 1.0 as observation kurtosis rises."""
    R = np.random.RandomState(13)
    n = 12151
    for df in (30, 6, 3, 2):
        x = (R.standard_t(df, size=n) * 0.0003).astype(np.float64)
        kurt = ((x - x.mean()) ** 4).mean() / x.std() ** 4
        T = np.asarray(opt.fit_hmm(x.tolist(), 3, 500, 1e-12).transition_matrix, dtype=float)
        print('[C] t(df=%-2d) kurtosis=%7.1f row sums = %s'
              % (df, kurt, ' '.join('%.9f' % v for v in T.sum(axis=1))))
    print('[C] DEFECT: rows sum to 1.0 for near-Gaussian input but drift to ~0.998 as tails fatten')


def note_d_var_quantile_convention():
    """Not a defect: historical_var_py uses a different quantile convention to numpy."""
    R = np.random.RandomState(42)
    loss = -(R.standard_t(3, size=2000).astype(np.float64))
    vr = opt.historical_var_py(loss, 0.95)
    vn = float(np.quantile(loss, 0.95))
    cr = opt.cvar_value_py(loss, 0.95)
    cn = float(loss[loss >= vn].mean())
    print('[D] VaR95  optimizr %.10f  numpy %.10f  diff %.2e' % (vr, vn, abs(vr - vn)))
    print('[D] CVaR95 optimizr %.10f  numpy %.10f  diff %.2e' % (cr, cn, abs(cr - cn)))
    print('[D] NOTE: CVaR agrees exactly; the VaR gap is nearest-rank vs interpolated quantile')


if __name__ == '__main__':
    defect_a_hurst_saturates_at_099()
    print()
    defect_b_initial_probs_never_updated()
    print()
    defect_c_transition_rows_lose_normalization_with_kurtosis()
    print()
    note_d_var_quantile_convention()
