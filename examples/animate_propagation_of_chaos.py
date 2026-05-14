"""Propagation of chaos for the McKean-Vlasov mean-reverting flow.

Theorem (Sznitman 1991): for the N-particle system

    dX^{i,N}_t = b(X^{i,N}_t, mu^N_t) dt + sigma dW^i_t,
    mu^N_t = (1/N) sum_j delta_{X^{j,N}_t},

with b Lipschitz, the empirical measure mu^N_t converges (in
Wasserstein-2) to the law mu_t of the McKean-Vlasov limit

    dX_t = b(X_t, mu_t) dt + sigma dW_t,    Law(X_t) = mu_t,

at rate O(1/sqrt(N)). Equivalently, any finite k-tuple
(X^{1,N}_t, ..., X^{k,N}_t) becomes asymptotically independent --
``chaos propagates`` from t = 0 to all later times.

This animation visualises the convergence: we run four McKean-Vlasov
simulations in parallel with N in {20, 200, 2000, 20000}, all sharing
the SAME initial bimodal distribution and the SAME drift / noise.
The coloured histograms are the empirical densities mu^N_t; the white
dashed curve is a high-resolution reference (N = 100_000).

As t advances and N increases, the histograms collapse onto the white
reference -- propagation of chaos in action.

Run with:
    python examples/animate_propagation_of_chaos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import optimizr as opt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_VALUES = [20, 100, 500, 4000]
N_REF = 12_000
N_STEPS = 200
T_HORIZON = 3.0
THETA = 0.7
SIGMA = 0.30
SEED = 11
FRAME_STRIDE = 4

X_GRID = np.linspace(-3.5, 3.5, 200)
BINS = np.linspace(-3.5, 3.5, 50)


def make_initial(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = N // 2
    return np.concatenate([
        rng.normal(-2.0, 0.35, half),
        rng.normal(+2.0, 0.35, N - half),
    ])


def simulate(N: int, seed: int) -> np.ndarray:
    """Return paths of shape (n_steps + 1, N)."""
    initial = make_initial(N, seed)
    out = opt.mean_reverting_mckean_vlasov(
        initial=initial.tolist(),
        theta=THETA,
        sigma=SIGMA,
        n_steps=N_STEPS,
        t_horizon=T_HORIZON,
        seed=seed,
    )
    return np.asarray(out["paths_flat"]).reshape(N_STEPS + 1, N)


print("Simulating propagation-of-chaos panels...")
panels = {}
for i, N in enumerate(N_VALUES):
    print(f"  panel N = {N}...", flush=True)
    panels[N] = simulate(N, SEED + i)
print(f"  reference (N = {N_REF})...", flush=True)
ref_paths = simulate(N_REF, SEED + 999)
print("  done.", flush=True)

# Pre-compute smoothed reference density for each frame using a histogram
# convolved with a Gaussian kernel -- O(N) per frame instead of O(N*G).
print("Building reference density curves...")
DENS_BINS = np.linspace(-3.5, 3.5, 161)
DENS_CENTERS = 0.5 * (DENS_BINS[:-1] + DENS_BINS[1:])
bw = 0.12
kernel_x = np.arange(-int(4 * bw / (DENS_BINS[1] - DENS_BINS[0])),
                     int(4 * bw / (DENS_BINS[1] - DENS_BINS[0])) + 1)
kernel = np.exp(-0.5 * (kernel_x * (DENS_BINS[1] - DENS_BINS[0]) / bw) ** 2)
kernel /= kernel.sum() * (DENS_BINS[1] - DENS_BINS[0])
ref_density = np.empty((N_STEPS + 1, DENS_CENTERS.size))
for k in range(N_STEPS + 1):
    h, _ = np.histogram(ref_paths[k], bins=DENS_BINS, density=True)
    ref_density[k] = np.convolve(h, kernel, mode="same") / kernel.sum() * kernel.sum()
# Interpolate onto display grid
ref_density_grid = np.empty((N_STEPS + 1, X_GRID.size))
for k in range(N_STEPS + 1):
    ref_density_grid[k] = np.interp(X_GRID, DENS_CENTERS, ref_density[k])
ref_density = ref_density_grid

times = np.linspace(0.0, T_HORIZON, N_STEPS + 1)

# Wasserstein-2 distance between empirical mu^N and reference, per frame
print("Computing W2(mu^N_t, mu_t) curves...")
def w2_to_ref(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """1-D Wasserstein-2 via sorted samples (Sklar / quantile transport)."""
    a = np.sort(samples_a)
    b = np.sort(samples_b)
    # Resample b to len(a) via interpolation of empirical quantiles.
    qa = np.linspace(0, 1, len(a))
    qb = np.linspace(0, 1, len(b))
    b_resampled = np.interp(qa, qb, b)
    return float(np.sqrt(np.mean((a - b_resampled) ** 2)))


w2_curves = {N: np.array([w2_to_ref(panels[N][k], ref_paths[k]) for k in range(N_STEPS + 1)])
             for N in N_VALUES}

# ---------------------------------------------------------------------------
# Figure -- 2x2 panels + W2 convergence track
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "axes.facecolor": "#0b1020",
    "figure.facecolor": "#0b1020",
    "axes.edgecolor": "#3a4a72",
    "axes.labelcolor": "#dbe7ff",
    "xtick.color": "#9eb1d8",
    "ytick.color": "#9eb1d8",
    "text.color": "#dbe7ff",
    "axes.grid": True,
    "grid.color": "#1e2a47",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})

fig = plt.figure(figsize=(8.5, 6.5), dpi=80)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.25)

panel_axes = {}
hist_artists = {}
panel_colors = ["#39d2ff", "#7be495", "#ffd166", "#ff7847"]

for ax_idx, (N, color) in enumerate(zip(N_VALUES, panel_colors)):
    ax = fig.add_subplot(gs[ax_idx // 2, ax_idx % 2])
    panel_axes[N] = ax
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0, ref_density.max() * 1.15)
    ax.set_title(f"$N = {N}$", color=color, fontsize=10, pad=4)
    ax.set_xticks([-3, -1.5, 0, 1.5, 3])
    if ax_idx >= 2:
        ax.set_xlabel("$x$", fontsize=9)
    # Initial histogram & reference line
    hist, _ = np.histogram(panels[N][0], bins=BINS, density=True)
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    bars = ax.bar(centers, hist, width=BINS[1] - BINS[0],
                  color=color, alpha=0.55, edgecolor="none")
    (ref_line,) = ax.plot(X_GRID, ref_density[0], color="#ffffff",
                          lw=1.3, ls="--", alpha=0.85,
                          label=r"$\mu_t$ (ref. $N=10^5$)")
    if ax_idx == 0:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.2,
                  edgecolor="#3a4a72")
    hist_artists[N] = (bars, ref_line)

# Bottom row: W2 convergence on a log-log scale wrt time
ax_w2 = fig.add_subplot(gs[2, :])
ax_w2.set_xlim(0, T_HORIZON)
ax_w2.set_yscale("log")
ax_w2.set_ylim(max(1e-3, min(w2_curves[N_VALUES[-1]].min(), 1e-2) * 0.5),
               max(w2_curves[N_VALUES[0]].max() * 1.5, 1.0))
ax_w2.set_xlabel("$t$", fontsize=9)
ax_w2.set_ylabel(r"$W_2(\mu_t^N, \mu_t)$", fontsize=9)
ax_w2.set_title(r"Wasserstein-2 distance to the reference law (log scale)"
                "  --  $W_2 \\sim O(1/\\sqrt{N})$",
                color="#e9efff", fontsize=10, pad=6)

w2_lines = {}
for N, color in zip(N_VALUES, panel_colors):
    (line,) = ax_w2.plot([], [], color=color, lw=1.6, label=f"$N = {N}$")
    w2_lines[N] = line
ax_w2.legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.2,
             edgecolor="#3a4a72")

cursor = ax_w2.axvline(0.0, color="#ffd166", lw=1, ls=":", alpha=0.8)

fig.suptitle(
    r"Propagation of chaos (Sznitman 1991):  $\mu_t^N \to \mu_t$  as  $N \to \infty$",
    color="#e9efff", fontsize=12, y=0.98,
)


def update(frame):
    artists = []
    for N in N_VALUES:
        bars, ref_line = hist_artists[N]
        hist, _ = np.histogram(panels[N][frame], bins=BINS, density=True)
        for rect, h in zip(bars, hist):
            rect.set_height(h)
        ref_line.set_ydata(ref_density[frame])
        artists.extend([*bars, ref_line])
    cursor.set_xdata([times[frame], times[frame]])
    for N in N_VALUES:
        w2_lines[N].set_data(times[: frame + 1], w2_curves[N][: frame + 1])
    artists.append(cursor)
    artists.extend(w2_lines.values())
    return artists


frame_indices = list(range(0, N_STEPS + 1, FRAME_STRIDE))
print(f"Rendering {len(frame_indices)} frames...")
anim = animation.FuncAnimation(
    fig, update, frames=frame_indices, interval=40, blit=False,
)

out_path = Path(__file__).with_name("propagation_of_chaos.gif")
try:
    anim.save(out_path, writer=animation.PillowWriter(fps=24))
    print(f"Saved animation: {out_path}")
except Exception as exc:
    print(f"GIF write failed ({exc}); saving PNG of last frame instead.")
    update(N_STEPS)
    fig.savefig(out_path.with_suffix(".png"), dpi=120)
    print(f"Saved PNG: {out_path.with_suffix('.png')}")
