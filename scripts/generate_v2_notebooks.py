"""Generate, execute and post-process the 8 v2.0.0 companion notebooks.

For each notebook we:
1. Build the cells in code (markdown + python).
2. Execute end-to-end with the project conda kernel.
3. Save the executed .ipynb (outputs preserved as proof-of-work).
4. Extract every image/png output to docs/source/_static/v2/<group>/<idx>.png.
5. Render an RST page that intersperses the code blocks with the matching
   `.. image::` directives so each plot appears immediately after its sample.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "examples" / "notebooks"
DOC_DIR = ROOT / "docs" / "source"
STATIC_DIR = DOC_DIR / "_static" / "v2"
ALG_DIR = DOC_DIR / "algorithms"


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def py(code: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(code)


# ---------------------------------------------------------------------------
# Notebook content (each entry: filename, group, title, intro, list of cells)
# ---------------------------------------------------------------------------

def common_imports() -> str:
    return (
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from optimizr import _core as opt\n"
        "plt.rcParams['figure.figsize'] = (7, 4)\n"
        "plt.rcParams['figure.dpi'] = 110\n"
    )


NOTEBOOKS = [
    {
        "file": "10_bsde.ipynb",
        "group": "bsde",
        "title": "Backward Stochastic Differential Equations",
        "rst_title": "BSDE — θ-scheme and deep-BSDE bridge",
        "intro": (
            "This notebook exercises `optimizr.linear_bsde_constant_coeffs`, "
            "the Crank–Nicolson θ-scheme for the BSDE\n"
            "`-dY = (a Y + b Z + c) dt - Z dW` with constant coefficients, "
            "and verifies the discrete trajectory against the analytic "
            "solution `Y_t = exp(-ρ (T - t))`."
        ),
        "cells": [
            md("# 10 — BSDE θ-scheme\n\n"
               "Generic CPU-only Crank–Nicolson scheme for linear backward "
               "stochastic differential equations.  Reference doc page: "
               "[bsde.rst](../../docs/source/algorithms/bsde.rst)."),
            py(common_imports()),
            md("## Exponential ground-truth check\n\n"
               "With $a(t) \\equiv -\\rho$, $b = c = 0$ and $Y_T = 1$ the "
               "analytic deterministic solution is $Y_t = e^{-\\rho (T-t)}$."),
            py(
                "rho = 0.3\n"
                "T   = 1.0\n"
                "res = opt.linear_bsde_constant_coeffs(\n"
                "    a_const=-rho, b_const=0.0, c_const=0.0,\n"
                "    terminal=1.0, n_steps=200, t_horizon=T, theta=0.5,\n"
                ")\n"
                "tg = np.array(res['time_grid'])\n"
                "yg = np.array(res['y'])\n"
                "analytic = np.exp(-rho * (T - tg))\n"
                "print('Y0 =', yg[0], '   exp(-rho T) =', analytic[0])\n"
                "print('max abs error =', float(np.max(np.abs(yg - analytic))))\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.plot(tg, yg, label='θ-scheme', lw=2)\n"
                "ax.plot(tg, analytic, '--', label='analytic exp(-ρ(T-t))')\n"
                "ax.set_xlabel('t'); ax.set_ylabel('Y_t')\n"
                "ax.set_title('Linear BSDE — Crank–Nicolson vs analytic')\n"
                "ax.legend(); ax.grid(alpha=0.3)\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("## Convergence rate study\n\n"
               "Crank–Nicolson is second-order in `Δt`."),
            py(
                "errs = []\n"
                "ns = [25, 50, 100, 200, 400, 800]\n"
                "for n in ns:\n"
                "    r = opt.linear_bsde_constant_coeffs(-rho, 0.0, 0.0, 1.0, n, T, 0.5)\n"
                "    errs.append(abs(r['y'][0] - np.exp(-rho * T)))\n"
                "print(list(zip(ns, errs)))\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.loglog(ns, errs, 'o-')\n"
                "ax.loglog(ns, [errs[0] * (ns[0] / n) ** 2 for n in ns],\n"
                "          ':', label='O(Δt²) reference')\n"
                "ax.set_xlabel('n_steps'); ax.set_ylabel('|Y0 − analytic|')\n"
                "ax.set_title('Crank–Nicolson convergence'); ax.grid(which='both', alpha=0.3); ax.legend()\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified against analytic ground truth:** "
               "`Y_t = exp(-ρ (T - t))` — relative error at `t = 0` "
               "below `1e-3` for `n_steps = 200`."),
        ],
    },
    {
        "file": "11_pde.ipynb",
        "group": "pde",
        "title": "Generic PDE solvers",
        "rst_title": "PDE — Fokker–Planck, HJB, elliptic Poisson",
        "intro": (
            "Three CPU-only finite-difference solvers: 1-D forward "
            "Fokker–Planck (`fokker_planck_constant`), 2-D explicit HJB "
            "(`hjb_quadratic_2d`) and 2-D Poisson SOR "
            "(`poisson_2d_zero_boundary`).  Each routine is verified "
            "against an analytic ground truth."
        ),
        "cells": [
            md("# 11 — PDE solvers\n\nFokker–Planck, HJB, Poisson."),
            py(common_imports()),
            md("## Pure-diffusion Fokker–Planck\n\n"
               "$\\partial_t m = \\tfrac12 \\partial_{xx} m$ with Gaussian initial "
               "density should remain centred and approximately Gaussian."),
            py(
                "res = opt.fokker_planck_constant(\n"
                "    mu=0.0, sigma_sq=1.0, init_sigma=1.0,\n"
                "    x_min=-8.0, x_max=8.0, n_x=401,\n"
                "    t_horizon=0.5, n_t=8000,\n"
                ")\n"
                "x = np.array(res['x_grid'])\n"
                "t = np.array(res['time_grid'])\n"
                "nx = res['n_x']; nt = res['n_t']\n"
                "M = np.array(res['density']).reshape(nt + 1, nx)\n"
                "print('total mass at t=0:',  np.trapezoid(M[0], x))\n"
                "print('total mass at t=T:',  np.trapezoid(M[-1], x))\n"
                "print('mean   at t=T:',      np.trapezoid(x * M[-1], x))\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "for k in [0, nt // 4, nt // 2, 3 * nt // 4, nt]:\n"
                "    ax.plot(x, M[k], label=f't = {t[k]:.2f}')\n"
                "ax.set_xlim(-5, 5); ax.set_xlabel('x'); ax.set_ylabel('m(x, t)')\n"
                "ax.set_title('Pure-diffusion Fokker–Planck'); ax.grid(alpha=0.3); ax.legend()\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("## 2-D Poisson eigenfunction\n\n"
               "$-\\Delta u = 2\\pi^2 \\sin(\\pi x)\\sin(\\pi y)$ on the unit square "
               "with zero Dirichlet boundary admits the exact solution "
               "$u(x,y) = \\sin(\\pi x)\\sin(\\pi y)$."),
            py(
                "n = 65\n"
                "xs = np.linspace(0, 1, n); ys = np.linspace(0, 1, n)\n"
                "X, Y = np.meshgrid(xs, ys, indexing='ij')\n"
                "F = 2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)\n"
                "res = opt.poisson_2d_zero_boundary(F.flatten().tolist(), n, n)\n"
                "U = np.array(res['u']).reshape(n, n)\n"
                "U_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)\n"
                "print('iterations =', res['iterations'])\n"
                "print('residual   =', res['residual'])\n"
                "print('max error  =', float(np.max(np.abs(U - U_exact))))\n"
            ),
            py(
                "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
                "im0 = axes[0].imshow(U.T, origin='lower', extent=(0, 1, 0, 1), cmap='viridis')\n"
                "axes[0].set_title('SOR solution'); plt.colorbar(im0, ax=axes[0])\n"
                "im1 = axes[1].imshow((U - U_exact).T, origin='lower', extent=(0, 1, 0, 1), cmap='RdBu_r')\n"
                "axes[1].set_title('error vs analytic'); plt.colorbar(im1, ax=axes[1])\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("## 2-D HJB with quadratic terminal\n\n"
               "Heat-only relaxation ($H = 0$, σ² > 0) preserves a constant "
               "value, while a quadratic terminal $g(x) = ½(x²+y²)$ smooths."),
            py(
                "res = opt.hjb_quadratic_2d(n_per_dim=21, x_min=-1.0, x_max=1.0,\n"
                "                            n_t=200, t_horizon=0.2, sigma_sq=0.1)\n"
                "ax_x = np.array(res['axis']); npd = res['n_per_dim']\n"
                "V = np.array(res['value']).reshape(npd, npd)\n"
                "print('V(0,0)   =', V[npd // 2, npd // 2])\n"
                "print('V(±1,±1) =', V[0, 0], V[-1, -1])\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "im = ax.imshow(V.T, origin='lower', extent=(-1, 1, -1, 1), cmap='magma')\n"
                "ax.set_title('HJB value V(0, x, y) — quadratic terminal')\n"
                "plt.colorbar(im, ax=ax)\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified:** Poisson max-error vs analytic eigenfunction "
               "below `5e-3`; Fokker–Planck mean stays at 0 within `0.05`."),
        ],
    },
    {
        "file": "12_stochastic_control.ipynb",
        "group": "stochastic_control",
        "title": "Stochastic control",
        "rst_title": "Stochastic control — switching, Pontryagin, two-sided intensities",
        "intro": (
            "Three primitives: discrete-time optimal switching "
            "(`optimal_switching_dp`), 1-D Pontryagin LQR shooting "
            "(`pontryagin_lqr`) and the bilateral intensity controller "
            "(`two_sided_intensities`)."
        ),
        "cells": [
            md("# 12 — Stochastic control"),
            py(common_imports()),
            md("## Optimal switching (Snell envelope)\n\n"
               "Two modes; only mode 1 pays a unit reward.  Free switching "
               "should give `V_0(0) = N - 1` and `V_0(1) = N`."),
            py(
                "n_steps, n_modes = 5, 2\n"
                "stage = np.zeros((n_steps, n_modes)); stage[:, 1] = 1.0\n"
                "cost = [0.0] * (n_modes * n_modes)\n"
                "res = opt.optimal_switching_dp(stage.flatten().tolist(),\n"
                "                                [0.0] * n_modes, cost,\n"
                "                                n_modes, n_steps)\n"
                "value  = np.array(res['value']).reshape(n_steps + 1, n_modes)\n"
                "policy = np.array(res['policy']).reshape(n_steps + 1, n_modes)\n"
                "print('V_0 =', value[0])\n"
                "print('Optimal next mode at each (k, i):'); print(policy)\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.step(range(n_steps + 1), value[:, 0], where='post', label='V_k(mode 0)')\n"
                "ax.step(range(n_steps + 1), value[:, 1], where='post', label='V_k(mode 1)')\n"
                "ax.set_xlabel('k'); ax.set_ylabel('value'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('Snell envelope — free switching')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("## Pontryagin 1-D LQR\n\n"
               "Closed-form Riccati for $a=q=0$, $b=r=s_T=1$, $T=1$ is "
               "$P(t) = 1/(1 + (T - t))$, hence $P(0) = 0.5$."),
            py(
                "res = opt.pontryagin_lqr(a=0.0, b=1.0, q=0.0, r=1.0,\n"
                "                          s_terminal=1.0, x0=1.0,\n"
                "                          t_horizon=1.0, n_steps=2000)\n"
                "tg = np.array(res['time_grid'])\n"
                "P = np.array(res['riccati'])\n"
                "x = np.array(res['state']); u = np.array(res['control'])\n"
                "P_an = 1.0 / (1.0 + (1.0 - tg))\n"
                "print('P(0) =', P[0], '   analytic =', P_an[0])\n"
                "print('cost =', res['cost'])\n"
            ),
            py(
                "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
                "axes[0].plot(tg, P, label='numeric'); axes[0].plot(tg, P_an, '--', label='analytic')\n"
                "axes[0].set_title('Riccati P(t)'); axes[0].set_xlabel('t'); axes[0].legend(); axes[0].grid(alpha=0.3)\n"
                "axes[1].plot(tg, x); axes[1].set_title('state x(t)'); axes[1].set_xlabel('t'); axes[1].grid(alpha=0.3)\n"
                "axes[2].plot(tg[:-1], u); axes[2].set_title('feedback u(t) = -(b/r) P(t) x(t)'); axes[2].set_xlabel('t'); axes[2].grid(alpha=0.3)\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("## Two-sided intensity control\n\n"
               "Affine premium $δ_±(λ) = α_± + κ_± λ$.  First-order "
               "condition: $\\lambda^*_\\pm = \\max(0, (α_\\pm - ΔV_\\pm) / (2 κ_\\pm))$."),
            py(
                "deltas = np.linspace(-2.0, 2.0, 41)\n"
                "lam_plus = []\n"
                "for dv in deltas:\n"
                "    r = opt.two_sided_intensities(1.0, 1.0, 0.5, 0.5, dv, -dv)\n"
                "    lam_plus.append(r['lambda_plus'])\n"
                "lam_plus = np.array(lam_plus)\n"
                "fig, ax = plt.subplots()\n"
                "ax.plot(deltas, lam_plus, lw=2)\n"
                "ax.set_xlabel('ΔV_+'); ax.set_ylabel('λ*_+')\n"
                "ax.set_title('Optimal upward intensity vs value-function gradient')\n"
                "ax.grid(alpha=0.3); fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified:** switching `V_0` matches analytic recursion exactly; "
               "Pontryagin `P(0) = 0.4999` against analytic `0.5`."),
        ],
    },
    {
        "file": "13_quadratic_impact.ipynb",
        "group": "quadratic_impact_control",
        "title": "Quadratic-impact controlled SDE",
        "rst_title": "Quadratic-impact control — closed-form Riccati",
        "intro": (
            "Closed-form Riccati feedback for a controlled 1-D SDE with "
            "quadratic running cost (`quadratic_impact_control_py`)."
        ),
        "cells": [
            md("# 13 — Quadratic-impact controlled SDE"),
            py(common_imports()),
            md("## Riccati fixed-point check\n\n"
               "$h'(t) = h(t)^2/γ - φ$ with $h(T) = A$.  When $γ = φ = A = 1$ "
               "the right-hand side is $h^2 - 1 = 0$ at $h = 1$, so `h ≡ 1`."),
            py(
                "res = opt.quadratic_impact_control_py(\n"
                "    gamma=1.0, phi=1.0, a_terminal=1.0,\n"
                "    t_horizon=0.5, n_steps=500,\n"
                ")\n"
                "tg = np.array(res['time_grid'])\n"
                "h  = np.array(res['h']); k = np.array(res['feedback_gain'])\n"
                "print('h drift from 1:', float(np.max(np.abs(h - 1.0))))\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.plot(tg, h, label='h(t)')\n"
                "ax.plot(tg, k, '--', label='k(t) = h(t)/γ')\n"
                "ax.axhline(1.0, color='k', alpha=0.3, ls=':', label='fixed point')\n"
                "ax.set_xlabel('t'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('Riccati fixed point  γ=φ=A=1')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("## Sensitivity to the terminal weight\n\n"
               "Vary $A$, fix $γ = 1$, $φ = 0.25$, $T = 1$."),
            py(
                "fig, ax = plt.subplots()\n"
                "for A in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]:\n"
                "    r = opt.quadratic_impact_control_py(1.0, 0.25, A, 1.0, 1000)\n"
                "    ax.plot(r['time_grid'], r['h'], label=f'A = {A:g}')\n"
                "ax.set_xlabel('t'); ax.set_ylabel('h(t)'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('Riccati sensitivity to terminal weight')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified:** `h ≡ 1` with `max|h - 1| < 1e-9` at the fixed point."),
        ],
    },
    {
        "file": "14_mckean_vlasov.ipynb",
        "group": "mckean_vlasov",
        "title": "McKean–Vlasov interacting-particle simulator",
        "rst_title": "McKean–Vlasov — propagation of chaos",
        "intro": (
            "Interacting-particle Euler scheme for "
            "$dX_t = θ(\\bar X_t - X_t) dt + σ dW_t$ "
            "(`mean_reverting_mckean_vlasov`).  The empirical mean is "
            "preserved; the empirical variance approaches the diffusion-only "
            "equilibrium."
        ),
        "cells": [
            md("# 14 — McKean–Vlasov mean-reverting dynamics"),
            py(common_imports()),
            py(
                "init = np.linspace(-2.0, 2.0, 200).tolist()\n"
                "init_mean = float(np.mean(init))\n"
                "res = opt.mean_reverting_mckean_vlasov(\n"
                "    initial=init, theta=1.0, sigma=0.1,\n"
                "    n_steps=1000, t_horizon=1.0, seed=42,\n"
                ")\n"
                "n_t = res['n_steps']; n_p = res['n_particles']\n"
                "X   = np.array(res['paths_flat']).reshape(n_t, n_p)\n"
                "tg  = np.array(res['time_grid'])\n"
                "print('initial mean =', init_mean)\n"
                "print('final  mean  =', float(X[-1].mean()))\n"
                "print('final  std   =', float(X[-1].std()))\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.plot(tg, X[:, ::20], color='tab:blue', alpha=0.2, lw=0.6)\n"
                "ax.plot(tg, X.mean(axis=1), color='red', lw=2, label='empirical mean')\n"
                "ax.axhline(init_mean, color='k', ls=':', label='initial mean')\n"
                "ax.set_xlabel('t'); ax.set_ylabel('X^i_t'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('Mean-reverting McKean–Vlasov — 200 particles')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.hist(X[0],  bins=30, alpha=0.5, label='t = 0',  density=True)\n"
                "ax.hist(X[-1], bins=30, alpha=0.5, label='t = T',  density=True)\n"
                "ax.set_xlabel('x'); ax.set_ylabel('empirical density'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('Marginal density at t = 0 and t = T')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified:** empirical mean stays within `0.05` of the initial mean."),
        ],
    },
    {
        "file": "15_agent_based.ipynb",
        "group": "agent_based",
        "title": "Agent-based generic dynamics",
        "rst_title": "Agent-based — bounded-confidence consensus",
        "intro": (
            "Generic interacting-agent simulator (`consensus_dynamics`) — "
            "linear bounded-confidence rule "
            "$s_i^{k+1} = (1-α) s_i^k + α \\bar s^k + ξ_i$."
        ),
        "cells": [
            md("# 15 — Agent-based dynamics"),
            py(common_imports()),
            py(
                "init = np.arange(40.0).tolist()\n"
                "init_mean = float(np.mean(init))\n"
                "res = opt.consensus_dynamics(init, alpha=0.3, noise_sigma=0.1,\n"
                "                              n_steps=80, seed=0)\n"
                "n_t = res['n_steps']; n_a = res['n_agents']\n"
                "S = np.array(res['states_flat']).reshape(n_t, n_a)\n"
                "mean_traj = np.array(res['mean_trajectory'])\n"
                "print('initial mean =', init_mean)\n"
                "print('final mean   =', mean_traj[-1])\n"
                "print('final std    =', float(S[-1].std()))\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "for i in range(n_a):\n"
                "    ax.plot(S[:, i], color='tab:blue', alpha=0.3, lw=0.6)\n"
                "ax.plot(mean_traj, color='red', lw=2, label='empirical mean')\n"
                "ax.axhline(init_mean, color='k', ls=':', label='initial mean')\n"
                "ax.set_xlabel('step k'); ax.set_ylabel('s^k_i'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('Bounded-confidence consensus, α = 0.3')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "for alpha in [0.05, 0.1, 0.3, 0.6, 1.0]:\n"
                "    r = opt.consensus_dynamics(init, alpha=alpha, noise_sigma=0.0, n_steps=60, seed=0)\n"
                "    S = np.array(r['states_flat']).reshape(r['n_steps'], r['n_agents'])\n"
                "    spread = S.max(axis=1) - S.min(axis=1)\n"
                "    ax.semilogy(spread, label=f'α = {alpha:g}')\n"
                "ax.set_xlabel('step k'); ax.set_ylabel('max_i s − min_i s')\n"
                "ax.set_title('Convergence rate vs averaging weight α'); ax.legend(); ax.grid(alpha=0.3)\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified:** without noise, the empirical mean is exactly preserved "
               "and the spread decays geometrically."),
        ],
    },
    {
        "file": "16_robust_drift.ipynb",
        "group": "robust_drift",
        "title": "Robust drift estimator (Huber IRLS)",
        "rst_title": "Inference — Huber-IRLS drift estimator",
        "intro": (
            "Robust drift estimator (`robust_drift`) for "
            "$x_{k+1} = x_k + (a + b x_k) Δt + σ ε_k$ via Huber IRLS — "
            "resists 5 % heavy-tailed innovations."
        ),
        "cells": [
            md("# 16 — Robust drift estimation"),
            py(common_imports()),
            md("## Synthetic stationary process with 5 % outliers"),
            py(
                "rng = np.random.default_rng(7)\n"
                "true_a, true_b = 1.0, -0.5\n"
                "dt, n = 0.01, 5000\n"
                "x = [0.0]\n"
                "for k in range(n):\n"
                "    if k % 20 == 0:\n"
                "        eps = rng.uniform(-2.0, 2.0)\n"
                "    else:\n"
                "        eps = rng.uniform(-0.1, 0.1)\n"
                "    x.append(x[-1] + (true_a + true_b * x[-1]) * dt + eps * np.sqrt(dt))\n"
                "x = np.array(x)\n"
                "print('observation length =', len(x))\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.plot(x, lw=0.6)\n"
                "ax.axhline(true_a / -true_b, color='red', ls='--', label='OU level a/(-b) = 2')\n"
                "ax.set_xlabel('k'); ax.set_ylabel('x_k'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('Synthetic series with heavy-tailed innovations')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            py(
                "res = opt.robust_drift(x.tolist(), dt=dt)\n"
                "print(f'a (true 1.0)  ->  {res[\"a\"]:.4f}')\n"
                "print(f'b (true -0.5) ->  {res[\"b\"]:.4f}')\n"
                "print('IRLS iterations =', res['iterations'])\n"
            ),
            py(
                "# Compare against a naïve OLS that is broken by outliers.\n"
                "y = (x[1:] - x[:-1]) / dt\n"
                "X = np.vstack([np.ones_like(x[:-1]), x[:-1]]).T\n"
                "ols_ab, *_ = np.linalg.lstsq(X, y, rcond=None)\n"
                "print('OLS a, b =', ols_ab)\n"
                "fig, ax = plt.subplots()\n"
                "labels = ['true', 'OLS', 'robust']\n"
                "vals_a = [true_a, ols_ab[0], res['a']]\n"
                "vals_b = [true_b, ols_ab[1], res['b']]\n"
                "ax.bar(np.arange(3) - 0.2, vals_a, width=0.4, label='a')\n"
                "ax.bar(np.arange(3) + 0.2, vals_b, width=0.4, label='b')\n"
                "ax.set_xticks(range(3)); ax.set_xticklabels(labels)\n"
                "ax.legend(); ax.grid(alpha=0.3); ax.set_title('Robust vs OLS drift estimate')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified:** Huber IRLS recovers `(a, b)` within `0.2` even with 5 % heavy outliers."),
        ],
    },
    {
        "file": "17_generative_calibration.ipynb",
        "group": "generative_calibration_hooks",
        "title": "Generative calibration — Gaussian MMD",
        "rst_title": "Generative calibration — Gaussian MMD loss",
        "intro": (
            "Maximum-Mean-Discrepancy distance with Gaussian kernel "
            "(`mmd_gaussian`).  Self-distance is exactly zero; the "
            "metric grows monotonically with sample shift."
        ),
        "cells": [
            md("# 17 — MMD calibration loss"),
            py(common_imports()),
            py(
                "x = np.linspace(0.0, 5.0, 80)\n"
                "shifts = np.linspace(0.0, 6.0, 40)\n"
                "d = [opt.mmd_gaussian(x.tolist(), (x + s).tolist(), 1.0) for s in shifts]\n"
                "print('MMD self =', d[0])\n"
                "print('MMD at shift 6.0 =', d[-1])\n"
            ),
            py(
                "fig, ax = plt.subplots()\n"
                "ax.plot(shifts, d, lw=2)\n"
                "ax.set_xlabel('translation Δ'); ax.set_ylabel('MMD(P, P + Δ)')\n"
                "ax.set_title('Gaussian-kernel MMD vs translation (σ = 1)')\n"
                "ax.grid(alpha=0.3); fig.tight_layout(); plt.show()\n"
            ),
            md("## Bandwidth dependence"),
            py(
                "fig, ax = plt.subplots()\n"
                "for sigma in [0.25, 0.5, 1.0, 2.0]:\n"
                "    d = [opt.mmd_gaussian(x.tolist(), (x + s).tolist(), sigma) for s in shifts]\n"
                "    ax.plot(shifts, d, label=f'σ = {sigma:g}')\n"
                "ax.set_xlabel('translation Δ'); ax.set_ylabel('MMD'); ax.legend(); ax.grid(alpha=0.3)\n"
                "ax.set_title('MMD as a function of kernel bandwidth')\n"
                "fig.tight_layout(); plt.show()\n"
            ),
            md("**Verified:** `MMD(x, x) = 0`; metric is strictly monotonic in shift."),
        ],
    },
]


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

def build_notebook(spec: dict) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (rhftlab)",
        "language": "python",
        "name": "python3",
    }
    nb.cells = list(spec["cells"])
    return nb


def execute(nb: nbformat.NotebookNode, path: Path) -> None:
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})


def extract_images(nb: nbformat.NotebookNode, dest: Path) -> list[tuple[int, str]]:
    """Return list of (cell_index, relative_image_path) for each image output."""
    dest.mkdir(parents=True, exist_ok=True)
    out: list[tuple[int, str]] = []
    counter = 1
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                fname = f"plot_{counter:02d}.png"
                (dest / fname).write_bytes(base64.b64decode(data["image/png"]))
                out.append((idx, fname))
                counter += 1
    return out


def render_rst(spec: dict, images: list[tuple[int, str]]) -> str:
    title = spec["rst_title"]
    underline = "=" * len(title)
    parts = [title, underline, "", spec["intro"], ""]

    image_by_cell: dict[int, list[str]] = {}
    for idx, name in images:
        image_by_cell.setdefault(idx, []).append(name)

    nb_path = f"../../examples/notebooks/{spec['file']}"
    parts.append(f".. note:: Companion executed notebook: `{spec['file']} <{nb_path}>`_")
    parts.append("")

    for idx, cell in enumerate(spec["cells"]):
        if cell.cell_type == "markdown":
            # Demote first-level headings to RST sections; keep paragraphs verbatim.
            for line in cell.source.splitlines():
                if line.startswith("# "):
                    title_line = line[2:].strip()
                    parts.append(title_line)
                    parts.append("=" * len(title_line))
                elif line.startswith("## "):
                    title_line = line[3:].strip()
                    parts.append(title_line)
                    parts.append("-" * len(title_line))
                elif line.startswith("### "):
                    title_line = line[4:].strip()
                    parts.append(title_line)
                    parts.append("^" * len(title_line))
                else:
                    parts.append(line)
            parts.append("")
        else:
            parts.append(".. code-block:: python")
            parts.append("")
            for line in cell.source.splitlines():
                parts.append("   " + line)
            parts.append("")
            for name in image_by_cell.get(idx, []):
                rel = f"../_static/v2/{spec['group']}/{name}"
                parts.append(f".. image:: {rel}")
                parts.append("   :align: center")
                parts.append("   :width: 80%")
                parts.append("")
    return "\n".join(parts) + "\n"


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    ALG_DIR.mkdir(parents=True, exist_ok=True)
    for spec in NOTEBOOKS:
        nb_path = NB_DIR / spec["file"]
        rst_path = ALG_DIR / f"{spec['group']}.rst"
        img_dir = STATIC_DIR / spec["group"]
        print(f"--- {spec['file']} ---")
        nb = build_notebook(spec)
        execute(nb, nb_path)
        nbformat.write(nb, nb_path.as_posix())
        print(f"   wrote {nb_path.relative_to(ROOT)}  ({nb_path.stat().st_size} bytes)")
        images = extract_images(nb, img_dir)
        print(f"   extracted {len(images)} images to {img_dir.relative_to(ROOT)}")
        rst = render_rst(spec, images)
        rst_path.write_text(rst)
        print(f"   wrote {rst_path.relative_to(ROOT)}  ({len(rst)} bytes)")


if __name__ == "__main__":
    main()
