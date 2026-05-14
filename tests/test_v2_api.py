"""Non-regression tests for the optimiz-rs v2 public API.

Each test exercises one of the v2 primitives advertised in the README and
the public blog post and checks it against an analytic ground truth.

Run with:    pytest tests/test_v2_api.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optimizr as opt


# ---------------------------------------------------------------------------
# 1. Risk measures -- historical VaR
# ---------------------------------------------------------------------------

def test_historical_var_gaussian():
    """VaR_0.95 of N(0,1) losses is the 0.95-quantile ~= 1.6449."""
    rng = np.random.default_rng(0)
    losses = rng.standard_normal(200_000).tolist()
    v95 = opt.historical_var_py(losses, 0.95)
    assert math.isclose(v95, 1.6449, abs_tol=2e-2), f"got {v95}"


def test_historical_var_monotone_in_alpha():
    rng = np.random.default_rng(1)
    losses = rng.standard_normal(50_000).tolist()
    v90 = opt.historical_var_py(losses, 0.90)
    v95 = opt.historical_var_py(losses, 0.95)
    v99 = opt.historical_var_py(losses, 0.99)
    assert v90 < v95 < v99


# ---------------------------------------------------------------------------
# 2. Volterra -- fractional ODE (Caputo / Adams scheme)
# ---------------------------------------------------------------------------

def test_solve_fractional_ode_constant_rhs_matches_power_law():
    """For Caputo D^alpha h = c with h(0) = h0, the closed form is
    h(t) = h0 + c * t^alpha / Gamma(alpha + 1)."""
    alpha = 0.7
    h0 = 1.0
    c = 2.0
    T = 1.0
    out = opt.solve_fractional_ode(h0, alpha, T, 400, lambda t, h: c)
    assert set(out.keys()) >= {"t_grid", "h"}
    h_T = out["h"][-1]
    expected = h0 + c * T ** alpha / math.gamma(alpha + 1.0)
    assert math.isclose(h_T, expected, rel_tol=2e-2), f"got {h_T}, expected {expected}"


def test_solve_fractional_ode_zero_rhs_is_constant():
    """If the right-hand side is zero, the Caputo ODE preserves h0."""
    out = opt.solve_fractional_ode(3.14, 0.5, 1.0, 200, lambda t, h: 0.0)
    for hi in out["h"]:
        assert math.isclose(hi, 3.14, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 3. Volterra -- second-kind integral equation
# ---------------------------------------------------------------------------

def test_solve_volterra_zero_kernel_returns_g():
    """If K(dt, y) = 0 the Volterra equation collapses to y(t) = g(t)."""
    out = opt.solve_volterra(lambda t: t, lambda dt, y: 0.0, 1.0, 50)
    grid = list(out["t_grid"])
    y = list(out["y"])
    assert len(grid) == len(y) == 51
    for ti, yi in zip(grid, y):
        assert math.isclose(yi, ti, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 4. BSDE -- linear theta scheme with constant coefficients
# ---------------------------------------------------------------------------

def test_linear_bsde_zero_coefficients_returns_terminal():
    """a = b = c = 0 reduces the BSDE to dY = -Z dW with terminal Y_T = K;
    the unique solution is Y_t = K, Z_t = 0."""
    res = opt.linear_bsde_constant_coeffs(
        a_const=0.0, b_const=0.0, c_const=0.0,
        terminal=2.5, n_steps=100, t_horizon=1.0, theta=0.5,
    )
    y = list(res["y"])
    z = list(res["z"])
    assert len(y) == 101 and len(z) == 100
    for yi in y:
        assert math.isclose(yi, 2.5, abs_tol=1e-9)
    for zi in z:
        assert math.isclose(zi, 0.0, abs_tol=1e-9)


def test_linear_bsde_pure_drift_grows_backward():
    """With a > 0, b = c = 0 and Y_T = 1 we get Y_t = exp(a (T - t))."""
    a = 0.3
    T = 1.0
    res = opt.linear_bsde_constant_coeffs(
        a_const=a, b_const=0.0, c_const=0.0,
        terminal=1.0, n_steps=400, t_horizon=T, theta=0.5,
    )
    grid = list(res["time_grid"])
    y = list(res["y"])
    expected = [math.exp(a * (T - t)) for t in grid]
    err = max(abs(yi - ei) for yi, ei in zip(y, expected))
    assert err < 5e-3, f"max error {err}"


# ---------------------------------------------------------------------------
# 5. McKean-Vlasov -- mean-reverting toward the empirical mean
# ---------------------------------------------------------------------------

def test_mean_reverting_mckean_vlasov_shapes_and_invariance():
    """Empirical mean is conserved in expectation by mean-reversion to it."""
    n_part = 200
    n_steps = 500
    initial = np.linspace(-1.0, 1.0, n_part).tolist()
    out = opt.mean_reverting_mckean_vlasov(
        initial=initial, theta=1.0, sigma=0.0,
        n_steps=n_steps, t_horizon=1.0, seed=42,
    )
    assert set(out.keys()) >= {"paths_flat", "n_steps", "n_particles", "time_grid"}
    assert out["n_particles"] == n_part
    assert out["n_steps"] == n_steps + 1
    paths = np.array(out["paths_flat"]).reshape(n_steps + 1, n_part)
    mean0 = float(np.mean(initial))
    # With sigma = 0 and mean-reversion to the empirical mean,
    # the cross-sectional mean must be preserved exactly.
    assert math.isclose(float(np.mean(paths[-1])), mean0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 6. Module surface -- guard against accidental API removal
# ---------------------------------------------------------------------------

V2_PUBLIC_API = (
    # v2.0 newcomers advertised in the blog post and README
    "solve_fractional_ode",
    "solve_volterra",
    "linear_bsde_constant_coeffs",
    "mean_reverting_mckean_vlasov",
    "historical_var_py",
    # v1.x primitives that must remain available (backward compat)
    "differential_evolution",
    "fit_hmm",
    "viterbi_decode",
    "mcmc_sample",
    "grid_search",
    "mutual_information",
    "shannon_entropy",
)


@pytest.mark.parametrize("name", V2_PUBLIC_API)
def test_public_symbol_exposed(name):
    assert hasattr(opt, name), f"optimizr.{name} is missing"
    assert callable(getattr(opt, name)), f"optimizr.{name} is not callable"
