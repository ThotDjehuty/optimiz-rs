#!/usr/bin/env python3
"""
Generate all matplotlib diagrams for mathematical_foundations.md.

Run from the docs/source directory (or workspace root):
    python docs/source/_gen_diagrams.py

Outputs SVG files to docs/source/_static/diagrams/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.stats import norm

# ─── output dir ─────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static", "diagrams")
os.makedirs(OUT, exist_ok=True)

# ─── palette & defaults ─────────────────────────────────────────────────────
C0   = "#2E6BE5"   # blue
C1   = "#E8850A"   # orange
C2   = "#27AE60"   # green
C3   = "#D62728"   # red
GRAY = "#888888"
BAND = "#AACBE8"

matplotlib.rcParams.update({
    "font.size"           : 11,
    "axes.titlesize"      : 12,
    "axes.labelsize"      : 11,
    "xtick.labelsize"     : 9,
    "ytick.labelsize"     : 9,
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "figure.dpi"          : 150,
    "savefig.bbox"        : "tight",
    "savefig.transparent" : False,
    "figure.facecolor"    : "white",
    "axes.facecolor"      : "white",
    "lines.linewidth"     : 1.8,
    "text.usetex"         : False,
})

def save(name):
    plt.savefig(os.path.join(OUT, name + ".svg"))
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# §1  DIFFERENTIAL EVOLUTION
# ════════════════════════════════════════════════════════════════════════════

def fig_de_mutation():
    r1 = np.array([0.5, 0.3])
    r2 = np.array([1.2, 1.4])
    r3 = np.array([1.8, 0.6])
    F  = 0.7
    vi = r1 + F * (r2 - r3)

    fig, ax = plt.subplots(figsize=(6, 4.2))

    # difference vector r3 → r2
    ax.annotate("", r2, r3,
                arrowprops=dict(arrowstyle="-|>", color=C2, lw=2.0, mutation_scale=14))
    mid = (r2 + r3) / 2
    ax.text(mid[0] - 0.05, mid[1] + 0.09,
            r"$F(\mathbf{x}_{r_2}-\mathbf{x}_{r_3})$",
            ha="center", fontsize=10, color=C2)

    # mutation arrow r1 → vi (dashed)
    ax.annotate("", vi, r1,
                arrowprops=dict(arrowstyle="-|>", color=C1, lw=2.0,
                                mutation_scale=14, linestyle="dashed"))
    ax.text((r1[0]+vi[0])/2, (r1[1]+vi[1])/2 - 0.1,
            r"$+F(\cdots)$", ha="center", fontsize=9, color=C1)

    pts = {
        r"$\mathbf{x}_{r_1}$  (base)": (r1, C0),
        r"$\mathbf{x}_{r_2}$":         (r2, C0),
        r"$\mathbf{x}_{r_3}$":         (r3, C0),
        r"$\mathbf{v}_i$  (mutant)":   (vi, C1),
    }
    for lbl, (p, col) in pts.items():
        ax.scatter(*p, s=90, color=col, zorder=6)
        offset = (0.05, 0.07)
        if "mutant" in lbl:
            offset = (0.07, 0.05)
        ax.text(p[0] + offset[0], p[1] + offset[1], lbl, fontsize=10, color=col)

    ax.set_xlim(0.1, 2.5); ax.set_ylim(0.0, 1.85)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title(r"DE Mutation:  $\mathbf{v}_i = \mathbf{x}_{r_1} + F\,(\mathbf{x}_{r_2} - \mathbf{x}_{r_3})$")
    ax.set_aspect("equal", adjustable="box")
    save("fig_de_mutation")


def fig_rastrigin():
    x = np.linspace(-2.5, 2.5, 800)
    y = 10 + x**2 - 10 * np.cos(2 * np.pi * x)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, y, color=C0, lw=2, label=r"$f(x) = 10 + x^2 - 10\cos(2\pi x)$")
    ax.fill_between(x, y, alpha=0.07, color=C0)
    ax.axhline(0, color=GRAY, lw=0.7, ls=":")

    # global minimum
    ax.scatter([0], [0], s=110, color=C1, zorder=6, label=r"global min  $f^*=0$", marker="*")

    # local minima
    lm_x = np.array([-2.0, -1.0, 1.0, 2.0])
    lm_y = 10 + lm_x**2 - 10 * np.cos(2 * np.pi * lm_x)
    ax.scatter(lm_x, lm_y, s=55, color=C3, zorder=5, label="local minima", marker="o")

    ax.annotate(r"$\approx 10^d$ local pits", (1.0, lm_y[2]),
                (1.5, 12), fontsize=9, color=C3,
                arrowprops=dict(arrowstyle="->", color=C3, lw=1.0))

    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$f(x)$")
    ax.set_title(r"Rastrigin function  ($d = 1$)  — many local minima")
    ax.legend(fontsize=9, framealpha=0.6)
    save("fig_rastrigin")


# ════════════════════════════════════════════════════════════════════════════
# §2.1  BROWNIAN MOTION
# ════════════════════════════════════════════════════════════════════════════

def fig_random_walk():
    rng = np.random.default_rng(42)
    n = 300
    t = np.linspace(0, 1, n)
    W = np.cumsum(rng.choice([-1, 1], size=n)) / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, W, color=C0, lw=1.4)
    ax.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$W_t^{(n)}$")
    ax.set_title(r"Coin-flip random walk ($n=300$)  $\longrightarrow$  Brownian motion as $n\to\infty$")
    save("fig_random_walk")


def fig_bm_fan():
    rng = np.random.default_rng(0)
    n, dt = 500, 0.002
    npaths = 10
    ts = np.linspace(0, 1, n)
    paths = np.cumsum(rng.normal(0, np.sqrt(dt), (npaths, n)), axis=1)
    paths[:, 0] = 0

    fig, ax = plt.subplots(figsize=(7, 4.2))
    lo, hi = -2 * np.sqrt(ts), 2 * np.sqrt(ts)
    ax.fill_between(ts, lo, hi, alpha=0.13, color=C0, label=r"$\pm 2\sqrt{t}$ (95% band)")
    ax.plot(ts, hi, color=C0, lw=1.2, ls="--", alpha=0.55)
    ax.plot(ts, lo, color=C0, lw=1.2, ls="--", alpha=0.55)
    colors_cycle = plt.colormaps["tab10"](np.linspace(0, 0.9, npaths))
    for i, p in enumerate(paths):
        ax.plot(ts, p, lw=0.9, alpha=0.75, color=colors_cycle[i])
    ax.axhline(0, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$W_t$")
    ax.set_title(r"Brownian motion — sample paths spread as $\sqrt{t}$  (trumpet fan)")
    ax.legend(fontsize=9, framealpha=0.7)
    save("fig_bm_fan")


def fig_gbm():
    rng = np.random.default_rng(7)
    T, n, dt = 1.0, 500, 0.002
    mu, sigma, S0 = 0.10, 0.30, 1.0
    ts = np.linspace(0, T, n)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ts, S0 * np.exp(mu * ts), color=C1, lw=1.8, ls="--",
            label=r"$\mathbb{E}[S_t] = S_0 e^{\mu t}$")
    ax.plot(ts, S0 * np.exp((mu - 0.5*sigma**2) * ts), color=C2, lw=1.5, ls=":",
            label=r"median $\approx S_0 e^{(\mu-\sigma^2/2)t}$")
    colors_cycle = plt.colormaps["Blues"](np.linspace(0.4, 0.85, 7))
    for i in range(7):
        W = np.cumsum(rng.normal(0, np.sqrt(dt), n))
        S = S0 * np.exp((mu - 0.5*sigma**2) * ts + sigma * W)
        ax.plot(ts, S, lw=0.9, alpha=0.7, color=colors_cycle[i])
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$S_t$")
    ax.set_title(r"Geometric Brownian motion  ($\mu=0.10,\;\sigma=0.30$)")
    ax.legend(fontsize=9, framealpha=0.6)
    save("fig_gbm")


# ════════════════════════════════════════════════════════════════════════════
# §2.2  ITŌ CALCULUS
# ════════════════════════════════════════════════════════════════════════════

def fig_ito_correction():
    t = np.linspace(0, 2.2, 300)
    mu, sigma = 0.12, 0.30

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, mu * t, color=C1, lw=2, ls="--",
            label=r"Naïve slope $\mu t$  (wrong)")
    ax.plot(t, (mu - 0.5*sigma**2) * t, color=C0, lw=2,
            label=r"Itō slope $(\mu - \sigma^2/2)\,t$  (correct)")

    # gap annotation at t = 1.8
    g_x = 1.8
    y_top = mu * g_x
    y_bot = (mu - 0.5*sigma**2) * g_x
    ax.annotate("", (g_x, y_bot), (g_x, y_top),
                arrowprops=dict(arrowstyle="<->", color=C3, lw=1.6))
    ax.text(g_x + 0.07, (y_top + y_bot) / 2,
            r"gap $= \sigma^2 T/2$", fontsize=9, color=C3, va="center")

    ax.axhline(0, color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\mathbb{E}[\log S_t] - \log S_0$")
    ax.set_title(r"Itō correction: $\mathbb{E}[\log S_t]$ always below the naïve slope $\mu t$")
    ax.legend(fontsize=9)
    save("fig_ito_correction")


# ════════════════════════════════════════════════════════════════════════════
# §2.3  FOKKER-PLANCK
# ════════════════════════════════════════════════════════════════════════════

def fig_fokker_planck():
    x = np.linspace(-0.5, 5.5, 600)
    mu_drift, sigma_diff = 0.8, 0.3
    times  = [0.05, 0.5, 1.5]
    colors = [C3, C2, C0]
    labels = [r"$t = 0.05$  (narrow spike)",
              r"$t = 0.50$",
              r"$t = 1.50$  (wide, drifted)"]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for t, col, lbl in zip(times, colors, labels):
        mean = mu_drift * t
        std  = sigma_diff * np.sqrt(t)
        y = norm.pdf(x, mean, std)
        ax.plot(x, y, color=col, lw=2, label=lbl)
        ax.fill_between(x, y, alpha=0.10, color=col)

    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$p(t, x)$")
    ax.set_title(r"Fokker-Planck: density drifts $(\mu=0.8)$ and broadens $(\sigma=0.3)$")
    ax.legend(fontsize=9)
    save("fig_fokker_planck")


# ════════════════════════════════════════════════════════════════════════════
# §2.3  EULER-MARUYAMA vs MILSTEIN
# ════════════════════════════════════════════════════════════════════════════

def fig_em_milstein():
    dts     = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    em_err  = 0.38 * dts**0.5
    mil_err = 0.19 * dts**1.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dts, em_err,  "o-",  color=C0, lw=2, ms=7,
              label=r"Euler-Maruyama  (order $1/2$)")
    ax.loglog(dts, mil_err, "s--", color=C1, lw=2, ms=7,
              label=r"Milstein          (order $1$)")
    ax.set_xlabel(r"Step size $\Delta t$")
    ax.set_ylabel(r"Strong error  $\|X_T - \hat{X}_T\|$")
    ax.set_title("SDE numerical schemes — strong convergence order")
    ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.3)
    save("fig_em_milstein")


# ════════════════════════════════════════════════════════════════════════════
# §2.4  ORNSTEIN-UHLENBECK
# ════════════════════════════════════════════════════════════════════════════

def fig_ou_path():
    rng = np.random.default_rng(3)
    T, n, dt = 5.0, 2000, 0.0025
    kappa, theta, sigma = 3.0, 0.5, 0.4
    X = np.zeros(n); X[0] = 2.0
    for i in range(1, n):
        X[i] = X[i-1] + kappa * (theta - X[i-1]) * dt + sigma * rng.normal(0, np.sqrt(dt))

    ts = np.linspace(0, T, n)
    sig_inf = sigma / np.sqrt(2 * kappa)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ts, X, color=C0, lw=1.0, alpha=0.9, label=r"$X_t$")
    ax.axhline(theta, color=C1, lw=1.8, ls="--",
               label=fr"$\theta = {theta}$  (long-run mean)")
    ax.fill_between(ts,
                    theta - 2 * sig_inf,
                    theta + 2 * sig_inf,
                    alpha=0.10, color=GRAY, label=r"$\theta \pm 2\sigma_\infty$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$X_t$")
    ax.set_title(fr"Ornstein-Uhlenbeck  ($\kappa={kappa},\;\theta={theta},\;\sigma={sigma}$)  — mean-reversion")
    ax.legend(fontsize=9)
    save("fig_ou_path")


def fig_ou_transition():
    x = np.linspace(-0.3, 2.6, 500)
    kappa, theta, sigma, x0 = 3.0, 0.5, 0.4, 2.0
    taus   = [0.1, 0.5, 2.0]
    colors = [C3, C2, C0]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for tau, col in zip(taus, colors):
        mean = theta + (x0 - theta) * np.exp(-kappa * tau)
        var  = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * tau))
        y = norm.pdf(x, mean, np.sqrt(var))
        ax.plot(x, y, color=col, lw=2,
                label=fr"$\tau = {tau:.1f}$   (mean $= {mean:.2f}$)")
        ax.fill_between(x, y, alpha=0.09, color=col)
    ax.axvline(theta, color=C1, lw=1.3, ls="--", label=fr"$\theta = {theta}$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$p(x_\tau \mid x_0)$")
    ax.set_title(r"OU transition density: drifts toward $\theta$, widens over time")
    ax.legend(fontsize=9)
    save("fig_ou_transition")


def fig_ou_loglik():
    kappa_v = np.linspace(10, 120, 80)
    theta_v = np.linspace(-0.005, 0.011, 80)
    K, T    = np.meshgrid(kappa_v, theta_v)
    Z = -(((K - 55) / 22)**2 + ((T - 0.003) / 0.003)**2)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    cf = ax.contourf(theta_v * 1000, kappa_v, Z.T, levels=20, cmap="Blues")
    ax.contour(theta_v * 1000, kappa_v, Z.T, levels=8,
               colors="white", linewidths=0.7, alpha=0.55)
    ax.plot(3, 55, "*", color=C1, ms=16, zorder=5,
            label=r"MLE $\hat\theta, \hat\kappa$")
    plt.colorbar(cf, ax=ax, label="Log-likelihood (normalised)")
    ax.set_xlabel(r"$\theta \times 10^3$"); ax.set_ylabel(r"$\kappa$")
    ax.set_title(r"OU log-likelihood surface  $\ell(\kappa, \theta \mid \hat\sigma)$")
    ax.legend(fontsize=10)
    save("fig_ou_loglik")


def fig_ou_residuals():
    rng = np.random.default_rng(9)
    r = rng.normal(0, 1, 600)
    x = np.linspace(-4, 4, 300)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(r, bins=32, density=True, color=C0, alpha=0.50,
            label="Standardised residuals")
    ax.plot(x, norm.pdf(x), color=C1, lw=2.2,
            label=r"$\mathcal{N}(0,1)$ theory")
    ax.set_xlabel(r"$r_i$"); ax.set_ylabel("Density")
    ax.set_title(r"OU residual diagnostic:  $r_i = (X_{t_i} - \hat\mu_i)/\hat\sigma$")
    ax.legend(fontsize=9)
    save("fig_ou_residuals")


# ════════════════════════════════════════════════════════════════════════════
# §3  JUMP PROCESSES
# ════════════════════════════════════════════════════════════════════════════

def fig_poisson():
    rng = np.random.default_rng(1)
    lam, T = 2, 4.0
    arrivals, t = [], 0.0
    while True:
        t += rng.exponential(1 / lam)
        if t > T: break
        arrivals.append(t)

    ts = np.concatenate([[0.0], arrivals, [T]])
    ns = np.arange(len(ts) - 1)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (t0, t1, n) in enumerate(zip(ts[:-1], ts[1:], ns)):
        ax.hlines(n, t0, t1, color=C0, lw=2.8)
        if i < len(arrivals):
            ax.vlines(t1, n, n + 1, color=C0, lw=2.0, linestyle=":")
            ax.scatter([t1], [n],     s=45, color="white", edgecolors=C0, zorder=5, lw=1.5)
            ax.scatter([t1], [n + 1], s=45, color=C0, zorder=5)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$N_t$")
    ax.set_title(fr"Poisson process  ($\lambda = {lam}$ jumps/unit)  — inter-arrivals $\sim \mathrm{{Exp}}(\lambda)$")
    save("fig_poisson")


def fig_jump_diffusion():
    rng = np.random.default_rng(11)
    T, n, dt = 1.0, 1000, 0.001
    mu, sigma, lam = 0.05, 0.18, 2.5
    ts = np.linspace(0, T, n)
    S  = np.ones(n)
    jump_times = np.sort(rng.uniform(0, T, rng.poisson(lam * T)))

    for i in range(1, n):
        dW   = rng.normal(0, np.sqrt(dt))
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        if np.any((ts[i-1] < jump_times) & (jump_times <= ts[i])):
            S[i] *= np.exp(rng.normal(0.0, 0.09))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ts, S, color=C0, lw=1.3, label=r"$S_t$  (jump-diffusion path)")
    # mark jump locations
    jt_idx = [np.searchsorted(ts, jt) for jt in jump_times if jt < T]
    ax.scatter(ts[jt_idx], S[jt_idx], s=50, color=C3, zorder=5,
               label=r"Poisson jump $\tau_k$", marker="v")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$S_t$")
    ax.set_title(r"Merton jump-diffusion  ($\lambda = 2.5$/yr,  $\sigma_J = 9\%$)")
    ax.legend(fontsize=9)
    save("fig_jump_diffusion")


def fig_levy_tails():
    x = np.linspace(0.05, 5, 600)
    gauss_tail = norm.pdf(x)
    gauss_tail /= gauss_tail[0]
    vg_tail = np.exp(-1.5 * x) / x
    vg_tail /= vg_tail[0]
    alpha_tail = x ** (-1.8)
    alpha_tail /= alpha_tail[0]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(x, gauss_tail, lw=2, color=C0,
                label=r"Gaussian  ($\nu \equiv 0$)")
    ax.semilogy(x, vg_tail, lw=2, color=C2,
                label=r"Variance Gamma  ($\nu \propto e^{-c|z|}/|z|$)")
    ax.semilogy(x, alpha_tail, lw=2, color=C1, ls="--",
                label=r"$\alpha$-stable  ($\nu \propto |z|^{-1-\alpha}$,  heaviest)")
    ax.set_xlabel(r"Jump size $|z|$")
    ax.set_ylabel(r"Lévy density $\nu(dz)/dz$  (log scale)")
    ax.set_title("Lévy measure tails — heavier tail = more frequent/larger jumps")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.25)
    save("fig_levy_tails")


# ════════════════════════════════════════════════════════════════════════════
# §6  KALMAN FILTER
# ════════════════════════════════════════════════════════════════════════════

def fig_kalman_covariance():
    t    = np.linspace(0, 30, 300)
    Pinf = 0.17
    Pt   = Pinf + (1.0 - Pinf) * np.exp(-0.35 * t)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, Pt, color=C0, lw=2, label=r"$P_t$  (error covariance)")
    ax.axhline(Pinf, color=C1, lw=1.6, ls="--",
               label=fr"$P_\infty \approx {Pinf}$  (steady-state)")
    ax.fill_between(t, Pt, Pinf, alpha=0.10, color=C0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$P_t$")
    ax.set_title(r"Kalman filter: error covariance converges exponentially to $P_\infty$")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
    save("fig_kalman_covariance")


# ════════════════════════════════════════════════════════════════════════════
# §7  MCMC
# ════════════════════════════════════════════════════════════════════════════

def fig_mcmc_energy():
    x = np.linspace(-5, 5, 600)
    pi = 0.5 * norm.pdf(x, -1.5, 0.8) + 0.5 * norm.pdf(x, 1.5, 0.9)
    U  = -np.log(pi + 1e-12)
    U -= U.min()

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, U, color=C0, lw=2)
    ax.fill_between(x, U, alpha=0.08, color=C0)
    ax.scatter([-1.5, 1.5], [U[np.abs(x + 1.5).argmin()],
                              U[np.abs(x - 1.5).argmin()]],
               s=90, color=C2, zorder=5, label=r"modes of $\pi$")
    saddle_i = np.abs(x).argmin()
    ax.scatter([x[saddle_i]], [U[saddle_i]], s=90, color=C3,
               zorder=5, marker="^", label="energy barrier")
    ax.annotate(r"accept with $e^{-\Delta U}$",
                (x[saddle_i] + 0.3, U[saddle_i] - 0.4),
                (2.2, 1.2), fontsize=9, color=C3,
                arrowprops=dict(arrowstyle="->", color=C3, lw=1.0))
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$U(x) = -\log\pi(x)$")
    ax.set_title(r"MCMC energy landscape  (bimodal target $\pi$)")
    ax.legend(fontsize=9)
    save("fig_mcmc_energy")


def fig_mcmc_trace():
    rng = np.random.default_rng(42)
    x_cur = -1.5
    chain = [x_cur]
    for _ in range(2999):
        prop    = x_cur + rng.normal(0, 0.8)
        pi_cur  = 0.5 * norm.pdf(x_cur, -1.5, 0.8) + 0.5 * norm.pdf(x_cur, 1.5, 0.9)
        pi_prop = 0.5 * norm.pdf(prop,  -1.5, 0.8) + 0.5 * norm.pdf(prop,  1.5, 0.9)
        x_cur   = prop if rng.random() < pi_prop / pi_cur else x_cur
        chain.append(x_cur)
    chain = np.array(chain)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(chain, lw=0.6, color=C0, alpha=0.8)
    axes[0].axhline(0, color=GRAY, lw=0.7, ls=":")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel(r"$x_t$")
    axes[0].set_title("Trace plot  — chain mixes between both modes")

    x = np.linspace(-5, 5, 400)
    true_pi = 0.5 * norm.pdf(x, -1.5, 0.8) + 0.5 * norm.pdf(x, 1.5, 0.9)
    axes[1].hist(chain, bins=50, density=True, color=C0, alpha=0.50,
                 label="MCMC samples")
    axes[1].plot(x, true_pi, color=C1, lw=2.2, label=r"true $\pi(x)$")
    axes[1].set_xlabel(r"$x$"); axes[1].set_ylabel("Density")
    axes[1].set_title("Marginal distribution")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    save("fig_mcmc_trace")


# ════════════════════════════════════════════════════════════════════════════
# §9  INFORMATION THEORY
# ════════════════════════════════════════════════════════════════════════════

def fig_kl_asymmetry():
    x = np.linspace(-10, 10, 800)
    p = norm.pdf(x, 0, 1)
    q = norm.pdf(x, 0, 4)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(x, p, color=C0, lw=2,       label=r"$p = \mathcal{N}(0,1)$  (narrow)")
    ax.plot(x, q, color=C1, lw=2, ls="--", label=r"$q = \mathcal{N}(0,4)$  (wide)")
    ax.fill_between(x, p, alpha=0.12, color=C0)
    ax.fill_between(x, q, alpha=0.08, color=C1)

    dx = x[1] - x[0]
    eps = 1e-12
    kl_pq = float(np.sum(p * np.log((p + eps) / (q + eps))) * dx)
    kl_qp = float(np.sum(q * np.log((q + eps) / (p + eps)) * dx))
    ax.text(-9.5, 0.085,
            fr"$D_{{KL}}(p\|q) \approx {kl_pq:.2f}$  (small: $q$ covers $p$)",
            fontsize=9, color=C0)
    ax.text(-9.5, 0.066,
            fr"$D_{{KL}}(q\|p) \approx {kl_qp:.2f}$  (large: $p$ misses tails of $q$)",
            fontsize=9, color=C1)
    ax.set_xlabel(r"$x$"); ax.set_ylabel("Density")
    ax.set_title(r"KL divergence asymmetry:  $D_{KL}(p\|q) \neq D_{KL}(q\|p)$")
    ax.legend(fontsize=9)
    save("fig_kl_asymmetry")


def fig_fisher_curvature():
    theta = np.linspace(-3, 3, 400)
    sigma_vals = [0.5, 1.0, 2.0]
    colors     = [C0, C2, C1]
    labels     = [r"$\sigma=0.5$  (high $\mathcal{I}$, sharp peak)",
                  r"$\sigma=1.0$",
                  r"$\sigma=2.0$  (low $\mathcal{I}$, flat peak)"]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for s, col, lbl in zip(sigma_vals, colors, labels):
        logL = -0.5 * (theta / s)**2 - np.log(s)
        logL -= logL.max()
        ax.plot(theta, logL, lw=2, color=col, label=lbl)

    ax.axvline(0, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\log\mathcal{L}(\theta \mid x_\mathrm{obs})$  (centred)")
    ax.set_title(r"Fisher information = log-likelihood curvature at $\theta^*$")
    ax.legend(fontsize=9); ax.set_ylim(-4.2, 0.3)
    save("fig_fisher_curvature")


# ════════════════════════════════════════════════════════════════════════════
# §10  DIFFERENTIAL GEOMETRY
# ════════════════════════════════════════════════════════════════════════════

def fig_curvatures():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # K > 0  — converging geodesics
    ax = axes[0]
    ax.set_aspect("equal"); ax.axis("off")
    theta_arc = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta_arc), np.sin(theta_arc), color=GRAY, lw=1.5, ls="--", alpha=0.35)
    for ang in np.linspace(-0.45, 0.45, 7):
        r = np.linspace(0, 1, 60)
        ax.plot(r * np.sin(ang), r * np.cos(ang), color=C0, lw=1.5, alpha=0.75)
    ax.scatter([0], [0], s=70, color=C1, zorder=5)
    ax.text(0, -0.12, "meet at N pole", ha="center", fontsize=8, color=GRAY)
    ax.set_title(r"$K > 0$  (sphere $S^2$)" + "\ngeodesics converge", fontsize=10)

    # K = 0  — parallel
    ax = axes[1]; ax.axis("off")
    for y in np.linspace(-0.8, 0.8, 7):
        ax.plot([-1, 1], [y, y], color=C0, lw=1.5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.2, 1.2)
    ax.text(0, -1.1, "remain equidistant", ha="center", fontsize=8, color=GRAY)
    ax.set_title(r"$K = 0$  (flat $\mathbb{R}^2$)" + "\nparallel geodesics", fontsize=10)

    # K < 0  — diverging
    ax = axes[2]; ax.axis("off")
    for ang in np.linspace(-0.55, 0.55, 7):
        r = np.linspace(0, 1.2, 60)
        scale = 1 + 0.55 * r
        ax.plot(r * np.sin(ang * scale), r * np.cos(ang * scale), color=C0, lw=1.5, alpha=0.75)
    ax.scatter([0], [0], s=70, color=C1, zorder=5)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.15, 1.5)
    ax.text(0, -0.12, "spread exponentially", ha="center", fontsize=8, color=GRAY)
    ax.set_title(r"$K < 0$  (hyperbolic $H^2$)" + "\ngeodesics diverge", fontsize=10)

    plt.suptitle("Sectional curvature determines geodesic behaviour", y=1.03, fontsize=12)
    plt.tight_layout()
    save("fig_curvatures")


def fig_natural_gradient():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    theta1 = np.linspace(-2, 2, 300)
    theta2 = np.linspace(-2, 2, 300)
    T1, T2 = np.meshgrid(theta1, theta2)

    # Standard: elongated contours → zigzag
    Z_std = 6 * T1**2 + T2**2
    axes[0].contour(T1, T2, Z_std, levels=7, colors=GRAY, alpha=0.45, linewidths=0.9)
    path_std = [(1.6, 1.6), (0.05, 1.1), (0.75, 0.15), (0.03, 0.06), (0, 0)]
    xs, ys = zip(*path_std)
    axes[0].plot(xs, ys, "o-", color=C0, lw=1.8, ms=5)
    axes[0].scatter([0], [0], s=120, color=C1, zorder=5, marker="*")
    axes[0].set_title("Standard gradient $\\nabla_\\theta \\mathcal{L}$\n(zigzag on ill-conditioned $\\mathcal{I}$)",
                      fontsize=10)
    axes[0].set_xlabel(r"$\theta_1$"); axes[0].set_ylabel(r"$\theta_2$")

    # Natural: circular contours → direct path
    Z_nat = T1**2 + T2**2
    axes[1].contour(T1, T2, Z_nat, levels=7, colors=GRAY, alpha=0.45, linewidths=0.9)
    path_nat = [(1.6, 1.6), (0.8, 0.8), (0.3, 0.3), (0, 0)]
    xs2, ys2 = zip(*path_nat)
    axes[1].plot(xs2, ys2, "o-", color=C2, lw=1.8, ms=5)
    axes[1].scatter([0], [0], s=120, color=C1, zorder=5, marker="*")
    axes[1].set_title(r"Natural gradient $\mathcal{I}^{-1}\nabla_\theta\mathcal{L}$" + "\n(direct, reparametrisation-invariant)",
                      fontsize=10)
    axes[1].set_xlabel(r"$\theta_1$"); axes[1].set_ylabel(r"$\theta_2$")

    plt.tight_layout()
    save("fig_natural_gradient")


# ════════════════════════════════════════════════════════════════════════════
# §2.3  PICARD ITERATION
# ════════════════════════════════════════════════════════════════════════════

def fig_picard():
    t = np.linspace(0, 1.5, 300)
    # True solution: dx = x dt  →  x(t) = e^t
    x_true = np.exp(t)
    # Picard iterates starting at x0 = 1
    x0 = np.ones_like(t)      # n=0: constant 1
    x1 = 1 + t                # n=1: linear
    x2 = 1 + t + t**2 / 2    # n=2: quadratic
    x3 = 1 + t + t**2/2 + t**3/6  # n=3

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, x0,    color=GRAY, lw=1.5, ls=":",  label=r"$X^{(0)}$: constant")
    ax.plot(t, x1,    color=C3,   lw=1.5, ls="-.", label=r"$X^{(1)}$: linear")
    ax.plot(t, x2,    color=C2,   lw=1.5, ls="--", label=r"$X^{(2)}$: quadratic")
    ax.plot(t, x3,    color=C1,   lw=1.8,          label=r"$X^{(3)}$")
    ax.plot(t, x_true, color=C0,  lw=2.2,          label=r"$X^{(\infty)} = e^t$  (true)")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$X^{(n)}_t$")
    ax.set_title(r"Picard iteration  ($dX = X\,dt$,  $X_0 = 1$)  — successive approximations")
    ax.legend(fontsize=9); ax.set_ylim(0.8, 5.0)
    save("fig_picard")


# ════════════════════════════════════════════════════════════════════════════
# §8  HMM REGIME STATE MACHINE  (K = 3)
# ════════════════════════════════════════════════════════════════════════════

def fig_hmm_regime():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4); ax.axis("off")

    states = [
        (1.5, 2.6, "State 1\nBull",    C2),
        (4.5, 2.6, "State 2\nNeutral", GRAY),
        (7.5, 2.6, "State 3\nBear",    C3),
    ]
    box_w, box_h = 2.0, 1.1
    for (cx, cy, label, col) in states:
        fancy = FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h,
                               boxstyle="round,pad=0.08", linewidth=1.6,
                               edgecolor=col, facecolor=col + "22",
                               zorder=2)
        ax.add_patch(fancy)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color=col, zorder=3)

    # Forward arrows A₁₂, A₂₃
    for x0, x1, label in [(2.5, 3.5, r"$A_{12}$"), (5.5, 6.5, r"$A_{23}$")]:
        ax.annotate("", xy=(x1, 2.85), xytext=(x0, 2.85),
                    arrowprops=dict(arrowstyle="-|>", color=C0, lw=1.5))
        ax.text((x0+x1)/2, 2.98, label, ha="center", fontsize=9, color=C0)
    # Backward arrows A₂₁, A₃₂
    for x0, x1, label in [(3.5, 2.5, r"$A_{21}$"), (6.5, 5.5, r"$A_{32}$")]:
        ax.annotate("", xy=(x1, 2.35), xytext=(x0, 2.35),
                    arrowprops=dict(arrowstyle="-|>", color=C1, lw=1.5))
        ax.text((x0+x1)/2, 2.22, label, ha="center", fontsize=9, color=C1)

    # Emission table
    col_labels = ["State", r"$\mu$", r"$\sigma$", "Character"]
    rows = [
        ["Bull",    "+0.05", "0.12", "high return, low vol"],
        ["Neutral", " 0.00", "0.18", "flat, medium vol"],
        ["Bear",    "−0.08", "0.35", "crash, high vol"],
    ]
    row_colors = [[C2+"33", C2+"33", C2+"33", C2+"33"],
                  [GRAY+"33", GRAY+"33", GRAY+"33", GRAY+"33"],
                  [C3+"33", C3+"33", C3+"33", C3+"33"]]
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="bottom",
                   cellColours=row_colors, bbox=[0.05, 0.0, 0.90, 0.42])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C0 + "33")
            cell.set_text_props(fontweight="bold")

    ax.set_title(r"HMM Regime State Machine ($K=3$)  —  Emission $B_k(y)=\mathcal{N}(\mu_k,\sigma_k^2)$",
                 fontsize=11, pad=6)
    plt.tight_layout()
    save("fig_hmm_regime")


# ════════════════════════════════════════════════════════════════════════════
# §8.2  VITERBI TRELLIS  (K=3, T=4)
# ════════════════════════════════════════════════════════════════════════════

def fig_viterbi_trellis():
    from matplotlib.patches import Circle, FancyArrowPatch

    K, T = 3, 4
    state_labels = ["1 (Bull)", "2 (Neutral)", "3 (Bear)"]
    map_path = {(1, 1), (1, 2)}  # state index 1 = "2 (Neutral)" at t=2,3 (0-indexed t)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set_xlim(-0.5, T + 0.5); ax.set_ylim(-0.5, K - 0.3); ax.axis("off")

    # x-positions: t=1..4 → 0.5, 1.5, 2.5, 3.5
    xs = [0.6 * (t + 1) for t in range(T)]
    ys = [K - 1 - k for k in range(K)]  # top = state 1

    # Draw crossing / passing arrows (selective to show crossing)
    arrow_kw = dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.0",
                    color=GRAY, lw=1.1, alpha=0.55)
    cross_kw = dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.18",
                    color=GRAY, lw=1.1, alpha=0.45)
    for t in range(T - 1):
        for k in range(K):
            for k2 in range(K):
                rad = 0.0 if k == k2 else (0.18 if k2 > k else -0.18)
                col = C0 if (k == 1 and k2 == 1 and t >= 1) else GRAY
                alpha = 0.9 if col == C0 else 0.3
                ax.annotate("", xy=(xs[t+1], ys[k2]), xytext=(xs[t], ys[k]),
                            arrowprops=dict(arrowstyle="-|>",
                                            connectionstyle=f"arc3,rad={rad}",
                                            color=col, lw=1.2 if col == C0 else 0.8,
                                            alpha=alpha))

    # Draw nodes
    r = 0.14
    for k in range(K):
        for t in range(T):
            is_map = (k == 1 and 1 <= t <= 2)
            fc = C0 if is_map else "white"
            ec = C0 if is_map else GRAY
            circ = Circle((xs[t], ys[k]), r, facecolor=fc, edgecolor=ec, lw=1.8, zorder=4)
            ax.add_patch(circ)

    # Labels on left
    for k in range(K):
        ax.text(-0.1, ys[k], state_labels[k], ha="right", va="center",
                fontsize=9, color=C0 if k == 1 else "black")

    # x-axis ticks
    for t in range(T):
        ax.text(xs[t], -0.35, f"$t={t+1}$", ha="center", va="top", fontsize=9)

    # Legend
    ax.scatter([], [], color=C0, s=80, label="● MAP (Viterbi) path", zorder=5)
    ax.scatter([], [], facecolor="white", edgecolors=GRAY, s=80, label="○ other nodes", zorder=5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(r"Viterbi Trellis  ($K=3$, $T=4$)  —  $\delta_t(k)=\max_j\,\delta_{t-1}(j)\,A_{jk}\,B_k(y_t)$",
                 fontsize=11)
    plt.tight_layout()
    save("fig_viterbi_trellis")


# ════════════════════════════════════════════════════════════════════════════
# §10.2  STANDARD VS NATURAL GRADIENT — PROPERTY COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def fig_std_vs_nat_gradient():
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))

    panels = [
        ("Standard Gradient\n" + r"$\theta_{k+1} = \theta_k - \eta\nabla\mathcal{L}$",
         ["Flat $\\mathbb{R}^d$ geometry",
          "Ignores manifold curvature",
          "Slow on ill-conditioned $\\mathcal{I}$",
          "$O(\\kappa(\\mathcal{I}))$ iterations"],
         C3, C3 + "18"),
        ("Natural Gradient\n" + r"$\theta_{k+1} = \theta_k - \eta\,\mathcal{I}(\theta)^{-1}\nabla\mathcal{L}$",
         ["Riemannian metric $\\mathcal{I}(\\theta)$",
          "Adapts to manifold geometry",
          "Reparametrisation-invariant",
          "$O(1)$ on exp. families (MLE step)"],
         C2, C2 + "18"),
    ]

    for ax, (title, props, border, bg) in zip(axes, panels):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fancy = FancyBboxPatch((0.03, 0.04), 0.94, 0.92,
                               boxstyle="round,pad=0.04", linewidth=2,
                               edgecolor=border, facecolor=bg)
        ax.add_patch(fancy)
        ax.text(0.5, 0.87, title, ha="center", va="top", fontsize=10,
                fontweight="bold", color=border, transform=ax.transAxes,
                multialignment="center")
        y = 0.68
        for prop in props:
            ax.text(0.12, y, "•  " + prop, ha="left", va="top", fontsize=9.5,
                    transform=ax.transAxes, color="#222222")
            y -= 0.17

    fig.suptitle("Standard  vs  Natural Gradient — geometric properties", fontsize=11, y=1.02)
    plt.tight_layout()
    save("fig_std_vs_nat_gradient")


# ════════════════════════════════════════════════════════════════════════════
# §10.3  MATRIX LIE GROUP HIERARCHY
# ════════════════════════════════════════════════════════════════════════════

def fig_lie_group_hierarchy():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4.6); ax.axis("off")

    nodes = {
        "GL":  (4.5, 4.1,  r"$\mathrm{GL}(n,\mathbb{R})$" + "\nall invertible $n\times n$", C0),
        "SL":  (1.8, 2.85, r"$\mathrm{SL}(n,\mathbb{R})$" + "\n" + r"$\det=1$",              C2),
        "On":  (4.5, 2.85, r"$O(n)$" + "\n$R^\top R=I$",                                    C1),
        "Sp":  (7.2, 2.85, r"$\mathrm{Sp}(2n,\mathbb{R})$" + "\npreserves " + r"$\omega$", C2),
        "SO":  (4.5, 1.55, r"$\mathrm{SO}(n)$" + "\n" + r"$\det=+1$ (rotations)",           C2),
        "Hn":  (1.8, 1.55, r"$H(n)$ Heisenberg" + "\nupper triangular",                      C3),
    }
    notes = {
        "SO": "portfolio factor\nrotation, PCA",
        "Sp": "Hamiltonian\nmechanics, PMP",
        "Hn": "path-signature\nfeature maps",
    }
    edges = [("GL","SL"), ("GL","On"), ("GL","Sp"), ("On","SO")]

    bw, bh = 2.2, 0.76
    for key, (cx, cy, label, col) in nodes.items():
        fbp = FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
                             boxstyle="round,pad=0.07", lw=1.6,
                             edgecolor=col, facecolor=col+"22", zorder=2)
        ax.add_patch(fbp)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8.5,
                multialignment="center", color=col, fontweight="bold", zorder=3)
        if key in notes:
            ax.text(cx + bw/2 + 0.15, cy, notes[key], va="center",
                    fontsize=7.5, color="#555555", fontstyle="italic")

    for src, dst in edges:
        sx, sy = nodes[src][0], nodes[src][1]
        dx, dy = nodes[dst][0], nodes[dst][1]
        ax.annotate("", xy=(dx, dy + bh/2 + 0.04), xytext=(sx, sy - bh/2 - 0.04),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))

    ax.set_title("Matrix Lie Group Hierarchy — subgroup inclusions and finance applications",
                 fontsize=11, pad=5)
    plt.tight_layout()
    save("fig_lie_group_hierarchy")


# ════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    funcs = [
        fig_de_mutation, fig_rastrigin,
        fig_random_walk, fig_bm_fan, fig_gbm,
        fig_ito_correction,
        fig_picard,
        fig_fokker_planck, fig_em_milstein,
        fig_ou_path, fig_ou_transition, fig_ou_loglik, fig_ou_residuals,
        fig_poisson, fig_jump_diffusion, fig_levy_tails,
        fig_kalman_covariance,
        fig_mcmc_energy, fig_mcmc_trace,
        fig_kl_asymmetry, fig_fisher_curvature,
        fig_curvatures, fig_natural_gradient,
        # new §8 & §10 diagrams
        fig_hmm_regime, fig_viterbi_trellis,
        fig_std_vs_nat_gradient, fig_lie_group_hierarchy,
    ]
    for fn in funcs:
        print(f"  {fn.__name__} ... ", end="", flush=True)
        fn()
        print("ok")
    print(f"\nDone — {len(funcs)} SVGs saved to {OUT}")
