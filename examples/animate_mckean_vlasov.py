"""Cool animation: mean-reverting McKean-Vlasov particle system.

Uses ``optimizr.mean_reverting_mckean_vlasov`` to simulate N
interacting particles whose drift pulls each toward the empirical
mean of the population, perturbed by a Brownian noise. We render

* a time-evolving particle scatter (top panel)
* the rolling empirical density estimated via a Gaussian KDE (bottom panel)

and save the result to ``examples/mckean_vlasov.gif``.

Run with:
    python examples/animate_mckean_vlasov.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

import optimizr as opt

# ---------------------------------------------------------------------------
# Parameters -- two well-separated initial clouds that fuse over time
# ---------------------------------------------------------------------------
N_PART = 800
N_STEPS = 400
T_HORIZON = 4.0
THETA = 0.6      # mean-reversion strength toward empirical mean
SIGMA = 0.25     # Brownian noise amplitude
SEED = 7

rng = np.random.default_rng(SEED)
left = rng.normal(-2.0, 0.35, N_PART // 2)
right = rng.normal(+2.0, 0.35, N_PART // 2)
initial = np.concatenate([left, right])

print(f"Simulating N={N_PART} particles for {N_STEPS} steps...")
out = opt.mean_reverting_mckean_vlasov(
    initial=initial.tolist(),
    theta=THETA,
    sigma=SIGMA,
    n_steps=N_STEPS,
    t_horizon=T_HORIZON,
    seed=SEED,
)
paths = np.asarray(out["paths_flat"]).reshape(N_STEPS + 1, N_PART)
times = np.asarray(out["time_grid"])

# ---------------------------------------------------------------------------
# Density grid via Gaussian KDE (vectorised)
# ---------------------------------------------------------------------------
x_grid = np.linspace(-3.5, 3.5, 240)
bw = 0.18
density = np.empty((N_STEPS + 1, x_grid.size))
norm = 1.0 / (N_PART * bw * np.sqrt(2 * np.pi))
for k in range(N_STEPS + 1):
    diffs = (x_grid[:, None] - paths[k][None, :]) / bw
    density[k] = norm * np.exp(-0.5 * diffs * diffs).sum(axis=1)

# ---------------------------------------------------------------------------
# Figure setup -- dark, cinematic look
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
    "grid.alpha": 0.5,
})

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7, 4.5), gridspec_kw={"height_ratios": [3, 2]}, dpi=80,
)

# Cool blue-orange diverging colormap for particles by initial position
norm_color = (initial - initial.min()) / (initial.max() - initial.min())
cmap = LinearSegmentedColormap.from_list("cool_warm", ["#39d2ff", "#ff7847"])
colors = cmap(norm_color)

scat = ax_top.scatter(
    paths[0], np.random.uniform(0, 1, N_PART),
    c=colors, s=10, alpha=0.85, edgecolors="none",
)
ax_top.set_xlim(-3.5, 3.5)
ax_top.set_ylim(0, 1)
ax_top.set_yticks([])
ax_top.set_title(
    "McKean–Vlasov mean-reverting flow\n"
    f"$dX_t = \\theta(\\bar m_t - X_t)\\,dt + \\sigma\\,dW_t$"
    f"  ($N = {N_PART}$, $\\theta = {THETA}$, $\\sigma = {SIGMA}$)",
    fontsize=11, color="#e9efff", pad=12,
)
mean_line = ax_top.axvline(initial.mean(), color="#ffd166", lw=1.2, ls="--",
                           label="empirical mean $\\bar m_t$")
ax_top.legend(loc="upper right", framealpha=0.2, edgecolor="#3a4a72")

(line,) = ax_bot.plot(x_grid, density[0], color="#39d2ff", lw=2)
fill = ax_bot.fill_between(x_grid, density[0], color="#39d2ff", alpha=0.25)
ax_bot.set_xlim(-3.5, 3.5)
ax_bot.set_ylim(0, density.max() * 1.05)
ax_bot.set_xlabel("$x$")
ax_bot.set_ylabel("empirical density")
time_text = ax_bot.text(
    0.02, 0.92, "", transform=ax_bot.transAxes,
    fontsize=10, color="#ffd166", family="monospace",
)

# Recompute jitter once -- particles keep their assigned y for visual stability
jitter = np.random.default_rng(SEED + 1).uniform(0, 1, N_PART)


def update(frame):
    global fill
    pts = paths[frame]
    scat.set_offsets(np.column_stack([pts, jitter]))
    mean_line.set_xdata([pts.mean(), pts.mean()])
    line.set_ydata(density[frame])
    fill.remove()
    fill = ax_bot.fill_between(x_grid, density[frame], color="#39d2ff", alpha=0.25)
    time_text.set_text(
        f"t = {times[frame]:5.2f}   |   "
        f"mean = {pts.mean():+.3f}   |   std = {pts.std():.3f}"
    )
    return scat, mean_line, line, fill, time_text


FRAME_STRIDE = 4   # render every 4th time step to keep the GIF small
frame_indices = list(range(0, N_STEPS + 1, FRAME_STRIDE))
print(f"Rendering {len(frame_indices)} frames...")
anim = animation.FuncAnimation(
    fig, update, frames=frame_indices, interval=40, blit=False,
)

out_path = Path(__file__).with_name("mckean_vlasov.gif")
try:
    anim.save(out_path, writer=animation.PillowWriter(fps=25))
    print(f"Saved animation: {out_path}")
except Exception as exc:
    print(f"Could not save GIF ({exc}); saving last frame as PNG instead.")
    update(N_STEPS)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    print(f"Saved PNG: {out_path.with_suffix('.png')}")
