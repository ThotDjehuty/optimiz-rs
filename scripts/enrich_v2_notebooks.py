"""Regenerate the eight v2.0 companion notebooks with the mandatory
pedagogical sandwich structure (PRE-markdown -> code -> POST-markdown)
and an additional real-world / concrete example per notebook.

Each notebook is then executed end-to-end with the rhftlab kernel so the
committed .ipynb carries figures and prints as proof of work.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


PREAMBLE = """import numpy as np
import matplotlib.pyplot as plt
from optimizr import _core as opt
plt.rcParams['figure.figsize'] = (8.5, 4.5)
plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
"""


def write_notebook(name: str, title_md: str, cells: list[dict]) -> Path:
    nb = {
        "cells": [md(title_md), code(PREAMBLE)] + cells,
        "metadata": {
            "kernelspec": {
                "display_name": "rhftlab",
                "language": "python",
                "name": "rhftlab",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1))
    return path


# ---------------------------------------------------------------------------
# 10 — BSDE
# ---------------------------------------------------------------------------
NB10_TITLE = """# 10 — BSDE θ-scheme (linear, constant coefficients)

CPU-only Crank–Nicolson scheme for linear backward stochastic
differential equations. Doc page:
[bsde.rst](../../docs/source/algorithms/bsde.rst).

Each computational cell is sandwiched between a *PRE* markdown
(theoretical reminder) and a *POST* markdown (expected result and
graph reading).
"""

NB10_CELLS = [
    md("""## Cellule 1 — Vérification du schéma θ contre la solution analytique

**Théorème (Pardoux-Peng 1990) — BSDE linéaire à coefficients constants.**
Pour $a$, $b$, $c$ déterministes constants et terminal $\\xi$, la
solution déterministe (lorsque $b = c = 0$) de
$$-dY_t = (a Y_t)\\,dt - Z_t\\,dW_t,\\qquad Y_T = \\xi$$
est $Y_t = \\xi e^{a(T-t)}$.

**Équation pivot.**
$$Y_t = \\xi\\,e^{a(T-t)}.$$

**Démonstration (esquisse).** En l'absence de bruit ($b = 0$) et de
forçage ($c = 0$), l'EDS rétrograde devient l'EDO $\\dot Y = -aY$
intégrée en $Y_t = \\xi e^{a(T-t)}$.  $\\square$

**Ce que la cellule vérifie.** Le schéma $\\theta = 0.5$ (Crank–Nicolson)
reproduit la décroissance exponentielle pour $a = -0.3$, $T = 1$.
"""),
    code("""rho, T, n = 0.3, 1.0, 200
res = opt.linear_bsde_constant_coeffs(-rho, 0.0, 0.0, 1.0, n, T, 0.5)
tg = np.array(res['time_grid'])
yg = np.array(res['y'])
analytic = np.exp(-rho * (T - tg))
err = float(np.max(np.abs(yg - analytic)))
print(f\"Y0 numérique  = {yg[0]:.6f}\")
print(f\"Y0 analytique = {analytic[0]:.6f}\")
print(f\"Erreur max sur la grille : {err:.2e}\")

fig, ax = plt.subplots()
ax.plot(tg, yg, lw=2, label='θ-scheme')
ax.plot(tg, analytic, '--', lw=1.5, label=r'$\\xi e^{-\\rho(T-t)}$')
ax.set_xlabel('t'); ax.set_ylabel(r'$Y_t$')
ax.set_title(\"BSDE linéaire — Crank–Nicolson vs analytique\")
ax.legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $Y_0 \\approx 0.7408$, erreur max $< 10^{-3}$.

**Lecture du graphique.** Les deux courbes se superposent visuellement.

**Conclusion.** Le primitive `linear_bsde_constant_coeffs` est calibré
sur le test analytique exponentiel ; il est utilisé comme brique pour
les BSDE non-linéaires (cellule 3).
"""),
    md("""## Cellule 2 — Étude de convergence en $\\Delta t$

**Théorème (Crank–Nicolson, ordre 2).** Pour une EDO linéaire
$\\dot y = -a y$, le schéma $\\theta = 1/2$ vérifie
$|y_n - y(t_n)| = O(\\Delta t^2)$.

**Équation pivot.**
$$\\log\\,\\text{erreur}(n) \\;\\approx\\; -2\\log n + C.$$

**Ce que la cellule vérifie.** On trace $|Y_0^{(n)} - e^{-\\rho T}|$
en log-log et on compare à la pente $-2$.
"""),
    code("""ns = [25, 50, 100, 200, 400, 800]
errs = []
for n in ns:
    r = opt.linear_bsde_constant_coeffs(-rho, 0.0, 0.0, 1.0, n, T, 0.5)
    errs.append(abs(r['y'][0] - np.exp(-rho * T)))
for n, e in zip(ns, errs):
    print(f\"n = {n:4d}  ->  erreur = {e:.3e}\")

fig, ax = plt.subplots()
ax.loglog(ns, errs, 'o-', lw=2, label='erreur empirique')
ax.loglog(ns, [errs[0] * (ns[0] / n) ** 2 for n in ns], ':',
          label=r'pente $-2$ (référence)')
ax.set_xlabel('n_steps'); ax.set_ylabel(r'$|Y_0 - e^{-\\rho T}|$')
ax.set_title(\"Convergence de Crank–Nicolson\")
ax.legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Les points s'alignent sur la pente $-2$.

**Lecture du graphique.** Parallélisme avec la droite pointillée.

**Conclusion.** Précision $10^{-6}$ atteinte avec $n \\approx 800$.
"""),
    md("""## Cellule 3 — Exemple concret : actualisation d'une espérance terminale

**Modèle utilisé.** Pour un brownien $W_t$, la valeur actualisée
$$Y_t = \\mathbb{E}\\!\\left[e^{-\\rho(T-t)}\\,W_T^2\\,\\big|\\,\\mathcal{F}_t\\right]$$
satisfait l'EDP de Feynman–Kac
$\\partial_t u + \\tfrac{1}{2}\\partial_{xx} u - \\rho u = 0$,
$u(T, x) = x^2$.

**Équation pivot (cas déterministe).** $\\mathbb{E}[W_T^2] = T$, donc
$$Y_0 = e^{-\\rho T}\\,T.$$

**Ce que la cellule vérifie.** On compare le primitive `linear_bsde`
(avec terminal $\\xi = T$ — espérance déterministe de $W_T^2$) à une
simulation Monte Carlo de $10^4$ trajectoires.
"""),
    code("""rng = np.random.default_rng(0)
M = 10_000
W_T = rng.standard_normal(M) * np.sqrt(T)
mc_value = np.exp(-rho * T) * float(np.mean(W_T ** 2))

# valeur déterministe via le primitive (xi = E[W_T^2] = T)
res = opt.linear_bsde_constant_coeffs(-rho, 0.0, 0.0, T, n, T, 0.5)
y0_pde = float(res['y'][0])

print(f\"Monte Carlo (M={M}) : Y0 = {mc_value:.6f}\")
print(f\"BSDE primitive      : Y0 = {y0_pde:.6f}\")
print(f\"Écart relatif       : {abs(y0_pde - mc_value)/mc_value:.2%}\")

ts = np.linspace(0, T, 50)
paths = np.cumsum(rng.standard_normal((20, len(ts))) *
                  np.sqrt(T / len(ts)), axis=1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for p in paths:
    axes[0].plot(ts, p, alpha=0.5)
axes[0].set_title(\"20 trajectoires browniennes\")
axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$W_t$')

tg = np.array(res['time_grid'])
yg = np.array(res['y'])
axes[1].plot(tg, yg, lw=2, color='C3', label='BSDE')
axes[1].axhline(mc_value, ls='--', color='C0',
                label=f'Monte Carlo Y0 = {mc_value:.3f}')
axes[1].set_title(\"Valeur actualisée déterministe\")
axes[1].set_xlabel('t'); axes[1].set_ylabel(r'$Y_t$')
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $Y_0 = T \\cdot e^{-\\rho T} \\approx 0.7408$,
estimateur Monte Carlo à moins de 3 %.

**Lecture du graphique.** Faisceau brownien à gauche ; valeur actualisée
et niveau Monte Carlo à droite — ils coïncident.

**Conclusion.** La primitive BSDE évite la simulation Monte Carlo
quand le terminal est une fonctionnelle simple du brownien.
"""),
]


# ---------------------------------------------------------------------------
# 11 — PDE
# ---------------------------------------------------------------------------
NB11_TITLE = """# 11 — PDE solvers (Fokker–Planck, Poisson)

CPU-only finite-difference solvers. Doc page:
[pde.rst](../../docs/source/algorithms/pde.rst).
"""

NB11_CELLS = [
    md("""## Cellule 1 — Diffusion d'une gaussienne (Fokker–Planck 1-D)

**Théorème (Fokker–Planck / Kolmogorov forward).** Pour la diffusion
$dX_t = \\mu\\,dt + \\sigma\\,dW_t$, la densité $p(t, x)$ vérifie
$$\\partial_t p = -\\mu\\,\\partial_x p + \\tfrac{1}{2}\\sigma^2\\,\\partial_{xx} p.$$

**Équation pivot (densité gaussienne).**
$$p(t, x) = \\frac{1}{\\sqrt{2\\pi(\\sigma^2 t + s_0^2)}}
   \\exp\\!\\left(-\\frac{(x-\\mu t)^2}{2(\\sigma^2 t + s_0^2)}\\right).$$

**Ce que la cellule vérifie.** Le primitive
`fokker_planck_constant(mu, sigma_sq, init_sigma, ...)` reproduit la
gaussienne analytique pour $\\mu = 0.1$, $\\sigma^2 = 0.16$,
$s_0 = 0.2$, $T = 1$.
"""),
    code("""mu, sigma_sq, init_sigma = 0.1, 0.16, 0.2
T, n_t = 1.0, 200
n_x = 201
res = opt.fokker_planck_constant(mu, sigma_sq, init_sigma,
                                 -3.0, 3.0, n_x, T, n_t)
xs = np.array(res['x_grid'])
ts = np.array(res['time_grid'])
density = np.array(res['density']).reshape(n_t + 1, n_x)
p_final = density[-1]

# ground truth gaussienne
sig_eff = np.sqrt(sigma_sq * T + init_sigma ** 2)
analytic = np.exp(-(xs - mu * T) ** 2 / (2 * sig_eff ** 2))
analytic /= np.trapezoid(analytic, xs)

err = float(np.max(np.abs(p_final - analytic)))
mass = float(np.trapezoid(p_final, xs))
print(f\"Erreur sup |p_num - p_ana| = {err:.3e}\")
print(f\"Masse finale (≈ 1)         = {mass:.4f}\")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(xs, density[0], ':', label='p(0, x)', alpha=0.6)
axes[0].plot(xs, p_final, lw=2, label=f'p({T}, x) — schéma')
axes[0].plot(xs, analytic, '--', lw=1.5, label='gaussienne analytique')
axes[0].set_xlabel('x'); axes[0].set_ylabel('densité')
axes[0].set_title(\"Coupes initiale / finale\")
axes[0].legend()
im = axes[1].imshow(density.T, aspect='auto', origin='lower',
                    extent=[0, T, xs.min(), xs.max()],
                    cmap='inferno')
axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
axes[1].set_title(\"Évolution spatio-temporelle\")
plt.colorbar(im, ax=axes[1], label='p(t, x)')
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Erreur sup faible et masse $\\approx 1$.

**Lecture du graphique.** Gauche : pic initial fin (pointillé) qui
s'étale en gaussienne décalée vers la droite (drift $\\mu T = 0.1$).
Droite : carte chaleur de $p(t, x)$, qui s'élargit dans le temps.

**Conclusion.** Le solveur Fokker–Planck est validé sur le cas
gaussien.
"""),
    md("""## Cellule 2 — Poisson 2D sur un carré (équation harmonique)

**Théorème (Poisson Dirichlet).** Pour $f$ régulière et conditions
nulles sur $\\partial\\Omega = [0,1]^2$, $-\\Delta u = f$ admet une
unique solution dans $H^1_0$.

**Équation pivot.** Si $f = 2\\pi^2 \\sin(\\pi x)\\sin(\\pi y)$ alors
$$u(x, y) = \\sin(\\pi x)\\sin(\\pi y).$$

**Démonstration.** Calcul direct : $-\\Delta u = 2\\pi^2 \\sin(\\pi x)
\\sin(\\pi y) = f$.  $\\square$

**Ce que la cellule vérifie.** Le solveur SOR `poisson_2d_zero_boundary`
résout l'équation et l'erreur tend vers zéro.
"""),
    code("""nx_grid = 41
xs = ys = np.linspace(0, 1, nx_grid)
X, Y = np.meshgrid(xs, ys, indexing='ij')
f = 2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
res = opt.poisson_2d_zero_boundary(
    f.flatten().tolist(), nx_grid, nx_grid,
    0.0, 1.0, 0.0, 1.0,
)
u = np.array(res['u']).reshape(nx_grid, nx_grid)

u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
err = float(np.max(np.abs(u - u_exact)))
print(f\"Itérations SOR : {res['iterations']}\")
print(f\"Résidu final   : {res['residual']:.3e}\")
print(f\"Erreur sup     : {err:.3e}\")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
im0 = axes[0].imshow(u, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
axes[0].set_title('u numérique'); plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(u - u_exact, origin='lower',
                      extent=[0, 1, 0, 1], cmap='RdBu_r')
axes[1].set_title('erreur signée'); plt.colorbar(im1, ax=axes[1])
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Erreur sup faible (< $10^{-2}$) ; résidu
SOR petit.

**Lecture du graphique.** Cloche centrée à hauteur $1$ ;
carte d'erreur quasi nulle, symétrique autour de zéro.

**Conclusion.** Le solveur elliptique est validé.
"""),
    md("""## Cellule 3 — Exemple concret : diffusion thermique sur une plaque

**Modèle physique.** Plaque carrée $[0,1]^2$ chauffée par une source
gaussienne au centre, conditions de Dirichlet nulles sur le bord.
Équilibre régi par
$$-\\Delta u = q(x, y),\\qquad u\\big|_{\\partial\\Omega} = 0.$$

**Ce que la cellule vérifie.** Profil radial décroissant de température
sous source ponctuelle.
"""),
    code("""nx_grid = 81
xs = ys = np.linspace(0, 1, nx_grid)
X, Y = np.meshgrid(xs, ys, indexing='ij')
q = 50.0 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.05 ** 2))
res = opt.poisson_2d_zero_boundary(
    q.flatten().tolist(), nx_grid, nx_grid,
    0.0, 1.0, 0.0, 1.0,
)
u = np.array(res['u']).reshape(nx_grid, nx_grid)

print(f\"Température max au centre : {u.max():.3f}\")
print(f\"Température max au bord   : {u[0, :].max():.2e}\")
print(f\"Itérations SOR            : {res['iterations']}\")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(u, origin='lower', extent=[0, 1, 0, 1],
               cmap='inferno', interpolation='bilinear')
ax.contour(X, Y, u, levels=8, colors='white', linewidths=0.5, alpha=0.6)
ax.set_title(\"Plaque chauffée par une source ponctuelle (équilibre)\")
ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax, label='température')
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Température décroissante du centre vers le
bord (Dirichlet), profil radial visible.

**Lecture du graphique.** Iso-températures (lignes blanches) circulaires
autour de la source ; température nulle sur les bords.

**Conclusion.** Le même solveur Poisson 2D résout des problèmes
mathématiques académiques *et* des problèmes physiques concrets
(transfert thermique, électrostatique).
"""),
]


# ---------------------------------------------------------------------------
# 12 — Stochastic control (Pontryagin LQR)
# ---------------------------------------------------------------------------
NB12_TITLE = """# 12 — Stochastic control (Pontryagin LQR)

Doc page: [stochastic_control.rst](../../docs/source/algorithms/stochastic_control.rst).
"""

NB12_CELLS = [
    md("""## Cellule 1 — LQR scalaire et équation de Riccati

**Théorème (Kalman, LQR).** Pour le système $\\dot x = ax + bu$ et le
coût
$$J = \\int_0^T (q x^2 + r u^2)\\,dt + s\\,x(T)^2,$$
le contrôle optimal est $u^*(t) = -(b/r)\\,P(t)\\,x(t)$ où $P(t)$
satisfait la Riccati rétrograde
$$-\\dot P = 2aP - (b^2/r) P^2 + q,\\qquad P(T) = s.$$

**Équation pivot (cas $a = b = q = r = 1$).** Le point fixe
stationnaire est $P^* = 1 + \\sqrt{2} \\approx 2.414$.

**Ce que la cellule vérifie.** Le primitive `pontryagin_lqr` retourne
$P(0)$ proche du point fixe pour $T = 5$.
"""),
    code("""res = opt.pontryagin_lqr(
    a=1.0, b=1.0, q=1.0, r=1.0, s_terminal=0.5,
    x0=1.0, t_horizon=5.0, n_steps=400,
)
ts = np.array(res['time_grid'])
P = np.array(res['riccati'])
state = np.array(res['state'])
control = np.array(res['control'])

P_star = 1.0 + np.sqrt(2.0)
print(f\"P(0) numérique  = {P[0]:.4f}\")
print(f\"P* analytique   = {P_star:.4f}\")
print(f\"P(T) (terminal) = {P[-1]:.4f}\")
print(f\"Coût optimal    = {res['cost']:.4f}\")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].plot(ts, P, lw=2)
axes[0].axhline(P_star, ls='--', color='gray',
                label=f'P* = {P_star:.3f}')
axes[0].set_xlabel('t'); axes[0].set_ylabel('P(t)')
axes[0].set_title(\"Riccati\"); axes[0].legend()
axes[1].plot(ts, state, lw=2, color='C2')
axes[1].set_xlabel('t'); axes[1].set_ylabel('x(t)')
axes[1].set_title(\"État optimal\")
ts_u = ts[:len(control)]
axes[2].plot(ts_u, control, lw=2, color='C3')
axes[2].set_xlabel('t'); axes[2].set_ylabel('u(t)')
axes[2].set_title(\"Commande optimale\")
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $P(0) \\approx 1 + \\sqrt{2}$ ; l'état $x$
décroît rapidement vers $0$ ; la commande est proportionnelle à
$-P(t)\\,x(t)/r$.

**Lecture du graphique.** $P(t)$ plate sur $[0, T-\\epsilon]$ puis
descend vers $s = 0.5$ ; trajectoires d'état/commande lisses.

**Conclusion.** Le LQR scalaire est validé contre le point fixe
analytique.
"""),
    md("""## Cellule 2 — Étude paramétrique : sensibilité au coût terminal

**Théorème.** Lorsque $T \\to \\infty$, $P(0)$ tend vers le point fixe
indépendamment de $s$.

**Équation pivot.** $P^* = (a + \\sqrt{a^2 + b^2 q / r}) \\cdot r / b^2$.
Pour $a = b = q = r = 1$, $P^* = 1 + \\sqrt{2}$.

**Ce que la cellule vérifie.** Sweep sur plusieurs valeurs de $s$ et
de $T$ : $P(0)$ converge vers $P^*$.
"""),
    code("""s_values = [0.1, 0.5, 2.0, 5.0]
T_values = [0.5, 1.0, 2.0, 5.0, 10.0]
P_star = 1.0 + np.sqrt(2.0)

fig, ax = plt.subplots()
for s in s_values:
    p0s = []
    for T_ in T_values:
        r = opt.pontryagin_lqr(1.0, 1.0, 1.0, 1.0, s, 1.0, T_, 200)
        p0s.append(r['riccati'][0])
    ax.plot(T_values, p0s, 'o-', lw=2, label=f's = {s}')
    print(f\"s = {s:4.1f} : P(0) à T=10 = {p0s[-1]:.4f}\")
ax.axhline(P_star, ls='--', color='black',
           label=f'P* = {P_star:.3f}')
ax.set_xlabel('horizon T'); ax.set_ylabel('P(0)')
ax.set_title(\"Convergence vers le point fixe Riccati\")
ax.legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Toutes les courbes convergent vers
$P^* \\approx 2.414$ pour $T$ grand.

**Lecture du graphique.** L'effet du terminal $s$ s'estompe quand $T$
augmente.

**Conclusion.** En horizon long, le coût terminal devient négligeable.
"""),
    md("""## Cellule 3 — Exemple concret : stabilisation d'un pendule inversé

**Modèle physique.** Pendule inversé linéarisé autour de la verticale
$\\ddot\\theta = (g/\\ell)\\theta + (1/m\\ell^2) u$.  Pour le mode
sur-amorti $(\\dot\\theta \\equiv 0)$ on retient l'équation scalaire
$\\dot\\theta = a\\theta + bu$ avec $a = g/\\ell$, $b = 1/(m\\ell^2)$.

**Équation pivot (Riccati).** $P$ vérifie $-\\dot P = 2aP - (b^2/r)P^2 + q$.

**Ce que la cellule vérifie.** Le contrôleur LQR ramène l'angle
initial $0.3$ rad vers $0$.
"""),
    code("""g, ell, m = 9.81, 1.0, 1.0
a, b = g / ell, 1.0 / (m * ell ** 2)
res = opt.pontryagin_lqr(a, b, q=10.0, r=1.0, s_terminal=1.0,
                          x0=0.3, t_horizon=5.0, n_steps=500)
ts = np.array(res['time_grid'])
theta = np.array(res['state'])
u = np.array(res['control'])

print(f\"Angle initial : {theta[0]:.3f} rad ({np.degrees(theta[0]):.1f}°)\")
print(f\"Angle final   : {theta[-1]:.3e} rad\")
print(f\"Effort max    : {np.abs(u).max():.3f}\")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ts, theta, lw=2, color='C3', label=r'$\\theta(t)$')
axes[0].axhline(0, ls='--', color='gray', alpha=0.6)
axes[0].set_xlabel('t (s)'); axes[0].set_ylabel(r'$\\theta$ (rad)')
axes[0].set_title(\"Stabilisation du pendule inversé\")
axes[0].legend()
ts_u = ts[:len(u)]
axes[1].plot(ts_u, u, lw=2, color='C2', label='u(t)')
axes[1].set_xlabel('t (s)'); axes[1].set_ylabel('u (couple)')
axes[1].set_title(\"Couple appliqué (LQR)\")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $\\theta \\to 0$ rapidement, effort borné.

**Lecture du graphique.** Décroissance exponentielle de l'angle ;
couple initial fort puis amorti.

**Conclusion.** Le primitive transforme un système naturellement
instable ($a = 9.81 > 0$) en système stable, illustration directe
du contrôle optimal.
"""),
]


# ---------------------------------------------------------------------------
# 13 — Quadratic impact control
# ---------------------------------------------------------------------------
NB13_TITLE = """# 13 — Quadratic-impact control

Doc page: [quadratic_impact_control.rst](../../docs/source/algorithms/quadratic_impact_control.rst).
"""

NB13_CELLS = [
    md("""## Cellule 1 — Riccati générique sur un horizon fini

**Théorème (HJB quadratique 1-D).** Pour $dq_t = u_t\\,dt + \\sigma\\,dW_t$
et coût $L(q, u) = \\gamma u^2 + \\varphi q^2$, $g(q) = A q^2$, la
fonction valeur est $V(t, q) = h(t) q^2$ avec
$$h'(t) = h(t)^2 / \\gamma - \\varphi,\\qquad h(T) = A.$$
Le feedback optimal est $u^*(t) = -(h(t)/\\gamma)\\,q(t)$.

**Équation pivot.** Pour $\\gamma = \\varphi = A = 1$, le point fixe
est $h^* = 1$ (car $h^2 - 1 = 0 \\Rightarrow h = 1$).

**Ce que la cellule vérifie.** Le primitive
`quadratic_impact_control_py(gamma=1, phi=1, A=1, T, n)` retourne
$h(t) \\equiv 1$ et donc un gain de feedback unitaire.
"""),
    code("""res = opt.quadratic_impact_control_py(gamma=1.0, phi=1.0,
                                       a_terminal=1.0,
                                       t_horizon=0.5, n_steps=500)
ts = np.array(res['time_grid'])
h = np.array(res['h'])
k = np.array(res['feedback_gain'])

err = float(np.max(np.abs(h - 1.0)))
print(f\"max |h(t) - 1| = {err:.3e}\")
print(f\"Feedback gain : k(0) = {k[0]:.4f},  k(T) = {k[-1]:.4f}\")

fig, ax = plt.subplots()
ax.plot(ts, h, lw=2, label=r'$h(t)$')
ax.plot(ts, k, '--', lw=2, label=r'$k(t) = h(t)/\\gamma$')
ax.axhline(1.0, ls=':', color='gray', alpha=0.6, label='point fixe = 1')
ax.set_xlabel('t'); ax.set_ylabel('value')
ax.set_title(\"Riccati quadratique scalaire\")
ax.legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $h(t) \\equiv 1$ à précision machine.

**Lecture du graphique.** Lignes plates à 1.

**Conclusion.** Le primitive est validé sur le point fixe analytique.
"""),
    md("""## Cellule 2 — Étude paramétrique : pénalisation $\\varphi$

**Théorème.** Le point fixe $h^* = \\sqrt{\\gamma \\varphi}$ croît
avec $\\varphi$.

**Équation pivot.**
$$h^* = \\sqrt{\\gamma \\varphi},
   \\qquad k^* = h^*/\\gamma = \\sqrt{\\varphi/\\gamma}.$$

**Ce que la cellule vérifie.** Sweep $\\varphi \\in \\{0.1, 1, 4, 10\\}$ :
$h(0)$ s'aligne sur $\\sqrt{\\varphi}$ pour $\\gamma = 1$.
"""),
    code("""phis = [0.1, 1.0, 4.0, 10.0]
fig, ax = plt.subplots()
for phi in phis:
    r = opt.quadratic_impact_control_py(1.0, phi, np.sqrt(phi),
                                         t_horizon=2.0, n_steps=500)
    h_arr = np.array(r['h'])
    h_star = np.sqrt(phi)
    ax.plot(r['time_grid'], h_arr,
            label=fr'$\\varphi = {phi}$, $h^*={h_star:.2f}$')
    print(f\"phi={phi:5.2f} : h(0) = {h_arr[0]:.4f}, h* = {h_star:.4f}\")
ax.set_xlabel('t'); ax.set_ylabel('h(t)')
ax.set_title(r'Sensibilité de $h$ au coefficient $\\varphi$')
ax.legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Plus $\\varphi$ est grand, plus le gain de
feedback est élevé.

**Lecture du graphique.** Quatre lignes plates à des hauteurs
$\\sqrt{\\varphi}$.

**Conclusion.** Le compromis état/contrôle est gouverné par le
ratio $\\varphi/\\gamma$.
"""),
    md("""## Cellule 3 — Exemple concret : régulation autour d'un set-point

**Modèle.** En boucle fermée $\\dot q = -k q$ avec $k = h/\\gamma$,
la dynamique est $q(t) = q_0 e^{-kt}$.  Cas pratique : régulateur
thermique scalaire qui ramène la température vers $0$ (écart à la
consigne).

**Équation pivot.**
$$q(t) = q_0\\,e^{-(h^*/\\gamma)\\,t}.$$

**Ce que la cellule vérifie.** Pour différents $\\varphi$, on intègre
$\\dot q = -k(t) q$ et on observe la vitesse de retour vers $0$.
"""),
    code("""T, n_steps = 3.0, 600
dt = T / n_steps
q0 = 1.0
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for phi in [0.25, 1.0, 4.0]:
    r = opt.quadratic_impact_control_py(1.0, phi, np.sqrt(phi),
                                         t_horizon=T, n_steps=n_steps)
    k = np.array(r['feedback_gain'])
    q = np.zeros(n_steps + 1); q[0] = q0
    for i in range(n_steps):
        q[i + 1] = q[i] - dt * k[i] * q[i]
    ts = np.array(r['time_grid'])
    axes[0].plot(ts, q, lw=2, label=fr'$\\varphi = {phi}$')
    axes[1].plot(ts, k, lw=2, label=fr'$\\varphi = {phi}$')
    print(f\"phi={phi:.2f} : q(T) = {q[-1]:.4f}\")
axes[0].set_xlabel('t'); axes[0].set_ylabel('q(t)')
axes[0].set_title(\"Écart à la consigne\")
axes[0].legend()
axes[1].set_xlabel('t'); axes[1].set_ylabel('k(t) (gain)')
axes[1].set_title(\"Gain de feedback\")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Plus $\\varphi$ est grand, plus $q$ retourne
rapidement à $0$.

**Lecture du graphique.** Décroissance exponentielle visible ; gain
plat (régime stationnaire de la Riccati).

**Conclusion.** Le primitive fournit le gain optimal pour tout
problème linéaire-quadratique scalaire à pénalité d'impact.
"""),
]


# ---------------------------------------------------------------------------
# 14 — McKean-Vlasov
# ---------------------------------------------------------------------------
NB14_TITLE = """# 14 — Mean-reverting McKean–Vlasov

Doc page: [mckean_vlasov.rst](../../docs/source/algorithms/mckean_vlasov.rst).
"""

NB14_CELLS = [
    md("""## Cellule 1 — Conservation de la moyenne

**Théorème (McKean 1966).** Pour la dynamique de champ moyen
$$dX_t^i = \\theta\\,(\\bar X_t - X_t^i)\\,dt + \\sigma\\,dW_t^i,
   \\qquad \\bar X_t = \\frac{1}{N}\\sum_j X_t^j,$$
la moyenne empirique est conservée en espérance,
$\\mathbb{E}[\\bar X_t] = \\bar X_0$.

**Équation pivot.** $\\mathbb{E}[\\bar X_t] = \\bar X_0$ pour tout $t$.

**Démonstration.** Sommer l'EDS sur $i$ : la dérive interne s'annule
(moyenne — individu).  $\\square$

**Ce que la cellule vérifie.** `mean_reverting_mckean_vlasov` préserve
la moyenne et concentre la variance.
"""),
    code("""N, T, n_steps = 500, 1.0, 200
theta, sigma = 1.5, 0.3
x0 = np.linspace(-1.0, 1.0, N).tolist()  # mean = 0

res = opt.mean_reverting_mckean_vlasov(x0, theta, sigma, n_steps, T, 42)
n_t = res['n_steps']
n_part = res['n_particles']
paths = np.array(res['paths_flat']).reshape(n_t, n_part)
ts = np.array(res['time_grid'])

mean = paths.mean(axis=1)
var = paths.var(axis=1)
print(f\"n_steps stocké : {n_t} (snapshots)\")
print(f\"n_particles    : {n_part}\")
print(f\"Moyenne initiale : {mean[0]:.3e}\")
print(f\"Moyenne finale   : {mean[-1]:.3e}\")
print(f\"Variance init    : {var[0]:.3f}\")
print(f\"Variance finale  : {var[-1]:.3f}\")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in range(0, n_part, 25):
    axes[0].plot(ts, paths[:, i], alpha=0.4, lw=0.7)
axes[0].plot(ts, mean, 'k-', lw=2, label='moyenne empirique')
axes[0].set_xlabel('t'); axes[0].set_ylabel(r'$X_t^i$')
axes[0].set_title(\"Trajectoires McKean–Vlasov\")
axes[0].legend()
axes[1].plot(ts, var, lw=2, color='C2')
axes[1].set_xlabel('t'); axes[1].set_ylabel(r'Var($X_t$)')
axes[1].set_title(\"Variance empirique\")
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $|\\bar X_t - 0|$ très petit, variance qui
décroît vers une borne stationnaire.

**Lecture du graphique.** Faisceau qui se contracte autour de la
moyenne ; variance qui diminue.

**Conclusion.** Le solveur reproduit la concentration mean-field.
"""),
    md("""## Cellule 2 — Exemple concret : dynamique d'opinion polarisée

**Modèle.** Modèle de DeGroot continu : opinions initialement
bimodales (deux groupes opposés).  La dynamique de champ moyen
détruit progressivement la polarisation.

**Équation pivot.**
$$dX_t^i = \\theta\\,(\\bar X_t - X_t^i)\\,dt + \\sigma\\,dW_t^i.$$

**Ce que la cellule vérifie.** Une distribution initiale bimodale
fusionne en distribution unimodale centrée sur $\\bar X_0$.
"""),
    code("""rng = np.random.default_rng(7)
N = 600
half = N // 2
x0 = np.concatenate([
    rng.normal(-1.0, 0.2, half),
    rng.normal(+1.0, 0.2, N - half),
]).tolist()

res = opt.mean_reverting_mckean_vlasov(x0, theta=2.0, sigma=0.15,
                                        n_steps=400, t_horizon=2.0, seed=11)
n_t = res['n_steps']
n_part = res['n_particles']
paths = np.array(res['paths_flat']).reshape(n_t, n_part)

mid = n_t // 2
print(f\"Variance t=0   : {paths[0].var():.3f}\")
print(f\"Variance t=mid : {paths[mid].var():.3f}\")
print(f\"Variance t=T   : {paths[-1].var():.3f}\")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, idx, label in zip(axes, [0, mid, -1], ['t=0', 't=T/2', 't=T']):
    ax.hist(paths[idx], bins=40, density=True,
            color='C0', edgecolor='white', alpha=0.85)
    ax.set_title(f\"Distribution {label}\")
    ax.set_xlabel('opinion'); ax.set_ylabel('densité')
    ax.set_xlim(-2, 2)
fig.suptitle(\"Convergence d'une population polarisée vers le consensus\",
             fontsize=12, y=1.02)
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** L'histogramme bimodal initial fusionne en un
pic unique centré à $0$.

**Lecture du graphique.** Trois snapshots montrent la fusion
progressive des deux modes.

**Conclusion.** L'attraction par la moyenne (mean-field) détruit la
polarisation initiale en temps fini — illustration du théorème du
consensus dans les dynamiques de DeGroot continues.
"""),
]


# ---------------------------------------------------------------------------
# 15 — Agent-based consensus
# ---------------------------------------------------------------------------
NB15_TITLE = """# 15 — Agent-based consensus dynamics

Doc page: [agent_based.rst](../../docs/source/algorithms/agent_based.rst).
"""

NB15_CELLS = [
    md("""## Cellule 1 — Convergence vers le consensus

**Théorème (consensus DeGroot, version $\\alpha$-pondérée).** Pour la
dynamique discrète
$$X^i_{k+1} = (1 - \\alpha)\\,X^i_k + \\alpha\\,\\bar X_k + \\sigma\\,\\varepsilon_k^i,$$
la moyenne $\\bar X_k$ est conservée en espérance et la variance
décroît à taux $(1 - \\alpha)^2$ par pas (pour $\\sigma = 0$).

**Équation pivot.**
$$\\mathbb{E}[\\text{Var}(X_k)] = (1 - \\alpha)^{2k}\\,\\text{Var}(X_0).$$

**Démonstration.** Récurrence linéaire : la composante hors moyenne
est multipliée par $(1-\\alpha)$ à chaque pas.  $\\square$

**Ce que la cellule vérifie.** `consensus_dynamics(initial, alpha,
noise_sigma, n_steps, seed)` reproduit la décroissance géométrique.
"""),
    code("""N, alpha, sigma, n_steps = 50, 0.3, 0.0, 60
rng = np.random.default_rng(1)
x0 = rng.uniform(-2, 2, N).tolist()

res = opt.consensus_dynamics(x0, alpha=alpha, noise_sigma=sigma,
                              n_steps=n_steps, seed=0)
states = np.array(res['states_flat']).reshape(n_steps + 1, N)
mean_traj = np.array(res['mean_trajectory'])

variances = states.var(axis=1)
print(f\"Moyenne initiale : {np.mean(x0):.3f}\")
print(f\"Moyenne finale   : {mean_traj[-1]:.3f}\")
print(f\"Var init / final : {variances[0]:.3f} / {variances[-1]:.3e}\")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in range(N):
    axes[0].plot(states[:, i], lw=0.8, alpha=0.5)
axes[0].plot(mean_traj, 'k-', lw=2, label='moyenne')
axes[0].set_xlabel('itération k'); axes[0].set_ylabel(r'$X_k^i$')
axes[0].set_title(f\"Consensus (α = {alpha}, σ = {sigma})\")
axes[0].legend()
axes[1].semilogy(variances, lw=2, label='variance empirique')
axes[1].semilogy(variances[0] * (1 - alpha) ** (2 * np.arange(n_steps + 1)),
                 ':', label=r'$(1-\\alpha)^{2k}$ Var$(X_0)$')
axes[1].set_xlabel('itération k'); axes[1].set_ylabel(r'Var($X_k$)')
axes[1].set_title(\"Décroissance géométrique\")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Variance qui décroît exponentiellement.

**Lecture du graphique.** Faisceau qui se concentre vers la moyenne
(ligne noire) ; pente log-linéaire conforme à la prédiction.

**Conclusion.** Le primitive est validé sur la convergence
géométrique.
"""),
    md("""## Cellule 2 — Effet du bruit

**Théorème.** Lorsque $\\sigma > 0$, la variance ne tombe pas à zéro
mais se stabilise à un niveau d'équilibre proportionnel à $\\sigma^2$.

**Équation pivot.** Variance asymptotique croissante en $\\sigma^2$.

**Ce que la cellule vérifie.** Sweep sur $\\sigma \\in \\{0.05, 0.1,
0.2\\}$ pour $\\alpha = 0.3$.
"""),
    code("""N, alpha, n_steps = 80, 0.3, 200
rng = np.random.default_rng(11)
x0 = rng.uniform(-2, 2, N).tolist()

fig, ax = plt.subplots()
for sigma in [0.05, 0.1, 0.2]:
    r = opt.consensus_dynamics(x0, alpha, sigma, n_steps, seed=42)
    states = np.array(r['states_flat']).reshape(n_steps + 1, N)
    var = states.var(axis=1)
    ax.plot(var, label=fr'$\\sigma = {sigma}$')
    print(f\"σ = {sigma:.2f} : Var(K) numérique = {var[-1]:.4f}\")
ax.set_xlabel('itération k'); ax.set_ylabel(r'Var($X_k$)')
ax.set_title(\"Variance asymptotique vs bruit\")
ax.legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Variance qui plateauise à un niveau
croissant avec $\\sigma^2$.

**Lecture du graphique.** Trois courbes décroissantes qui se
stabilisent à des paliers proportionnels à $\\sigma^2$.

**Conclusion.** Le compromis bruit / convergence est gouverné par
$\\alpha$ et $\\sigma$.
"""),
    md("""## Cellule 3 — Exemple concret : émergence d'une décision collective

**Modèle.** $N = 100$ agents au sein d'un comité décident d'un
nombre (par exemple un score entre $-1$ et $+1$).  Chaque agent
ajuste sa position vers la moyenne avec un peu de bruit.

**Équation pivot.**
$$X^i_{k+1} = (1 - \\alpha) X^i_k + \\alpha \\bar X_k + \\sigma \\varepsilon^i_k.$$

**Ce que la cellule vérifie.** Visualisation : densité des
opinions au cours du temps converge vers une distribution unimodale.
"""),
    code("""N = 100
n_steps = 80
alpha, sigma = 0.2, 0.04
rng = np.random.default_rng(7)
# distribution initiale : trois clusters
x0 = np.concatenate([
    rng.normal(-0.7, 0.15, 35),
    rng.normal(0.0, 0.10, 35),
    rng.normal(+0.7, 0.15, 30),
]).tolist()

r = opt.consensus_dynamics(x0, alpha, sigma, n_steps, seed=0)
states = np.array(r['states_flat']).reshape(n_steps + 1, N)
mean_traj = np.array(r['mean_trajectory'])

print(f\"Moyenne initiale : {np.mean(x0):.3f}\")
print(f\"Moyenne finale   : {mean_traj[-1]:.3f}\")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
im = axes[0].imshow(states.T, aspect='auto', cmap='coolwarm',
                    interpolation='nearest', origin='lower',
                    extent=[0, n_steps, 0, N])
axes[0].set_xlabel('itération k')
axes[0].set_ylabel('agent i')
axes[0].set_title(\"Évolution des opinions individuelles\")
plt.colorbar(im, ax=axes[0], label='X_k^i')
for k, color in zip([0, 20, n_steps], ['C0', 'C2', 'C3']):
    axes[1].hist(states[k], bins=20, alpha=0.55, color=color,
                 density=True, label=f'k={k}', edgecolor='white')
axes[1].set_xlabel('opinion'); axes[1].set_ylabel('densité')
axes[1].set_title(\"Distribution à différents instants\")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Les trois clusters initiaux fusionnent en
distribution unimodale concentrée près de $\\bar X_0$.

**Lecture du graphique.** Gauche : la heatmap montre l'horizon de
fusion.  Droite : trois distributions superposées (initiale tri-modale,
intermédiaire, finale unimodale).

**Conclusion.** Le primitive `consensus_dynamics` modélise toute
dynamique de moyennage par interaction (vote pondéré, opinion
publique, scoring collectif).
"""),
]


# ---------------------------------------------------------------------------
# 16 — Robust drift
# ---------------------------------------------------------------------------
NB16_TITLE = """# 16 — Robust drift estimation (Huber)

Doc page: [robust_drift.rst](../../docs/source/algorithms/robust_drift.rst).
"""

NB16_CELLS = [
    md("""## Cellule 1 — Récupération du drift d'un OU sous bruit gaussien

**Théorème.** Pour le processus d'Ornstein–Uhlenbeck discret
$$x_{k+1} = x_k + (a + b x_k)\\,\\Delta t + \\sigma\\,\\varepsilon_k,$$
l'estimateur de Huber via IRLS converge vers $(a, b)$ pour bruit
symétrique.

**Équation pivot (lien avec OU continu).** $dX_t = (a + b X_t) dt
+ \\sigma dW_t$ ; en posant $a = \\theta\\mu$, $b = -\\theta$, on
récupère le mean-reversion $dX_t = \\theta(\\mu - X_t)dt + \\sigma dW_t$.

**Ce que la cellule vérifie.** $\\theta = 2$, $\\mu = 1$ donc
$a_\\text{vrai} = 2$, $b_\\text{vrai} = -2$.  L'estimateur retrouve
ces valeurs sur $N = 1000$ observations.
"""),
    code("""rng = np.random.default_rng(0)
theta_true, mu_true, sigma = 2.0, 1.0, 0.2
N = 1000
dt = 0.01
x = np.zeros(N)
x[0] = 0.5
for k in range(N - 1):
    x[k + 1] = x[k] + theta_true * (mu_true - x[k]) * dt + \\
               sigma * np.sqrt(dt) * rng.standard_normal()

res = opt.robust_drift(x.tolist(), dt, huber_delta=1.345)
a_hat, b_hat = res['a'], res['b']
a_true, b_true = theta_true * mu_true, -theta_true
print(f\"a vrai / estimé : {a_true:.3f} / {a_hat:.3f}\")
print(f\"b vrai / estimé : {b_true:.3f} / {b_hat:.3f}\")
print(f\"theta_hat = {-b_hat:.3f},  mu_hat = {a_hat / -b_hat:.3f}\")
print(f\"itérations IRLS : {res['iterations']}\")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ts = np.arange(N) * dt
axes[0].plot(ts, x, lw=0.8)
axes[0].axhline(mu_true, ls='--', color='gray', label=fr'$\\mu = {mu_true}$')
axes[0].set_xlabel('t'); axes[0].set_ylabel('x(t)')
axes[0].set_title(\"Trajectoire OU simulée\")
axes[0].legend()
dx = (x[1:] - x[:-1]) / dt
axes[1].scatter(x[:-1], dx, s=4, alpha=0.4, label='observations')
xs_lin = np.linspace(x.min(), x.max(), 50)
axes[1].plot(xs_lin, a_true + b_true * xs_lin, 'k--',
             lw=1.5, label='vrai drift')
axes[1].plot(xs_lin, a_hat + b_hat * xs_lin, 'C3-',
             lw=2, label='Huber estimé')
axes[1].set_xlabel('x'); axes[1].set_ylabel(r'$\\Delta x / \\Delta t$')
axes[1].set_title(\"Régression robuste sur le drift\")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $\\hat a \\approx 2.0$, $\\hat b \\approx -2.0$.

**Lecture du graphique.** Gauche : trajectoire qui oscille autour de
$\\mu = 1$.  Droite : nuage des incréments avec deux droites
superposées (vraie et estimée).

**Conclusion.** Sur bruit gaussien, Huber se comporte comme OLS
(efficacité ~95 %).
"""),
    md("""## Cellule 2 — Robustesse face à un bruit de Cauchy

**Théorème (consistance de Huber sous queues lourdes).** Pour
$\\varepsilon_k \\sim \\text{Cauchy}$, OLS *diverge* (variance
infinie) tandis que Huber reste consistant.

**Équation pivot.**
$$\\hat\\theta_\\text{Huber} \\xrightarrow{a.s.} \\theta_0,
   \\qquad \\hat\\theta_\\text{OLS} \\nrightarrow \\theta_0
   \\text{ (Cauchy)}.$$

**Ce que la cellule vérifie.** On compare Huber vs OLS sur 30 jeux
synthétiques OU avec bruit de Cauchy injecté.
"""),
    code("""rng = np.random.default_rng(123)
theta_true, mu_true = 2.0, 1.0
sigma_cauchy = 0.05
N = 1000; dt = 0.01
n_runs = 30
a_true, b_true = theta_true * mu_true, -theta_true

ols_errs, huber_errs = [], []
for _ in range(n_runs):
    x = np.zeros(N); x[0] = 0.5
    for k in range(N - 1):
        eps = rng.standard_cauchy() * sigma_cauchy
        x[k + 1] = x[k] + theta_true * (mu_true - x[k]) * dt + eps * np.sqrt(dt)
    Xmat = np.column_stack([np.ones(N - 1), x[:-1]])
    yvec = (x[1:] - x[:-1]) / dt
    th_ols, *_ = np.linalg.lstsq(Xmat, yvec, rcond=None)
    r = opt.robust_drift(x.tolist(), dt, huber_delta=1.345)
    th_h = np.array([r['a'], r['b']])
    ols_errs.append(np.linalg.norm(th_ols - np.array([a_true, b_true])))
    huber_errs.append(np.linalg.norm(th_h - np.array([a_true, b_true])))

ols_errs = np.array(ols_errs); huber_errs = np.array(huber_errs)
print(f\"OLS   : médiane = {np.median(ols_errs):.3f}, max = {ols_errs.max():.2f}\")
print(f\"Huber : médiane = {np.median(huber_errs):.3f}, max = {huber_errs.max():.2f}\")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].boxplot([ols_errs, huber_errs], labels=['OLS', 'Huber'])
axes[0].set_ylabel(r'$\\|\\hat\\theta - \\theta_0\\|$')
axes[0].set_yscale('log')
axes[0].set_title(f\"{n_runs} runs OU + bruit Cauchy\")
axes[1].hist(np.clip(ols_errs, 0, 10), bins=20,
             alpha=0.5, label='OLS', color='C3')
axes[1].hist(huber_errs, bins=20, alpha=0.7, label='Huber', color='C0')
axes[1].set_xlabel(r'$\\|\\hat\\theta - \\theta_0\\|$')
axes[1].set_ylabel('fréquence')
axes[1].set_title(\"Distribution des erreurs\")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** Huber médiane bien inférieure à OLS médiane.

**Lecture du graphique.** Gauche : boxplots log, Huber serré.
Droite : histogramme — OLS dispersé, Huber concentré près de $0$.

**Conclusion.** L'estimateur robuste est indispensable dès que les
queues s'éloignent du gaussien (mesures expérimentales bruitées,
signaux de capteurs aberrants).
"""),
]


# ---------------------------------------------------------------------------
# 17 — Generative calibration hooks (MMD)
# ---------------------------------------------------------------------------
NB17_TITLE = """# 17 — Generative calibration hooks (MMD)

Doc page: [generative_calibration_hooks.rst](../../docs/source/algorithms/generative_calibration_hooks.rst).
"""

NB17_CELLS = [
    md("""## Cellule 1 — MMD nulle entre échantillons identiques

**Théorème (Gretton et al. 2012).** Pour le noyau gaussien
$k(x, y) = \\exp(-(x-y)^2 / 2\\sigma^2)$, la MMD au carré empirique
entre $X = (x_i)$ et $Y = (y_j)$ est
$$\\widehat{\\text{MMD}}^2(X, Y) = \\frac{1}{m^2}\\sum_{ij} k(x_i, x_j)
   + \\frac{1}{n^2}\\sum_{ij} k(y_i, y_j)
   - \\frac{2}{mn}\\sum_{ij} k(x_i, y_j).$$

**Équation pivot.** $X = Y \\Rightarrow \\widehat{\\text{MMD}}^2 = 0$.

**Démonstration.** Les trois sommes coïncident terme à terme.  $\\square$

**Ce que la cellule vérifie.** `mmd_gaussian(X, X, sigma)` retourne $0$
à précision machine.
"""),
    code("""rng = np.random.default_rng(0)
X = rng.standard_normal(200).tolist()
m = float(opt.mmd_gaussian(X, X, sigma=1.0))
print(f\"MMD(X, X) = {m:.3e}  (attendu : 0)\")

fig, ax = plt.subplots()
ax.hist(X, bins=30, density=True, color='C0',
        edgecolor='white', alpha=0.85)
ax.set_title(r'Échantillon $X = Y$')
ax.set_xlabel('x'); ax.set_ylabel('densité')
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** $\\sim 0$.

**Lecture du graphique.** Histogramme gaussien standard.

**Conclusion.** L'auto-MMD nulle est validée.
"""),
    md("""## Cellule 2 — MMD entre gaussienne et Laplace

**Théorème.** Deux distributions de moyenne et variance identiques
mais formes différentes ont une MMD strictement positive.

**Équation pivot.**
$$\\text{MMD}^2(\\mathcal{N}(0, 1), \\text{Laplace}(0, 1/\\sqrt{2})) > 0.$$

**Ce que la cellule vérifie.** Sweep en $\\sigma$ : la MMD est
maximale autour de $\\sigma$ commensurable à l'échelle des
distributions.
"""),
    code("""rng = np.random.default_rng(2)
n = 500
X = rng.standard_normal(n).tolist()  # N(0, 1)
Y = rng.laplace(0.0, 1.0 / np.sqrt(2.0), n).tolist()  # même variance

sigmas = np.geomspace(0.1, 10.0, 25)
mmds = [float(opt.mmd_gaussian(X, Y, sigma=s)) for s in sigmas]
print(f\"MMD min : {min(mmds):.3e}, MMD max : {max(mmds):.3e}\")
print(f\"σ optimal : {sigmas[int(np.argmax(mmds))]:.2f}\")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
xs = np.linspace(-5, 5, 400)
axes[0].hist(X, bins=40, density=True, alpha=0.5,
             label=r'$\\mathcal{N}(0,1)$', color='C0')
axes[0].hist(Y, bins=40, density=True, alpha=0.5,
             label='Laplace', color='C3')
axes[0].plot(xs, np.exp(-xs**2/2)/np.sqrt(2*np.pi),
             'C0--', lw=1.5)
axes[0].plot(xs, np.exp(-np.abs(xs)*np.sqrt(2))*np.sqrt(2)/2,
             'C3--', lw=1.5)
axes[0].set_xlabel('x'); axes[0].set_ylabel('densité')
axes[0].set_title(\"Densités comparées\"); axes[0].legend()
axes[1].semilogx(sigmas, mmds, 'o-', lw=2, color='C2')
axes[1].set_xlabel(r'bandwidth $\\sigma$')
axes[1].set_ylabel(r'$\\widehat{\\text{MMD}}^2$')
axes[1].set_title(\"Sensibilité à la bandwidth\")
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** MMD strictement positive, courbe en cloche.

**Lecture du graphique.** Gauche : la Laplace est plus pointue ;
droite : MMD maximale lorsque $\\sigma \\sim 1$.

**Conclusion.** Le choix de bandwidth est crucial ; règle pratique :
$\\sigma \\approx \\text{médiane des distances}$.
"""),
    md("""## Cellule 3 — Exemple concret : MMD entre mélange et gaussienne

**Modèle.** $P = \\mathcal{N}(0, 1)$ vs
$Q_\\mu = \\frac{1}{2}\\mathcal{N}(-\\mu, 1) + \\frac{1}{2}\\mathcal{N}(+\\mu, 1)$.
La MMD doit *croître* avec $\\mu$.

**Équation pivot.**
$$\\text{MMD}^2(P, Q_\\mu) \\xrightarrow[\\mu \\to 0]{} 0,
   \\qquad \\nearrow \\text{ en } \\mu.$$

**Ce que la cellule vérifie.** Pour $\\mu \\in [0, 2.5]$, la courbe
MMD est croissante.
"""),
    code("""rng = np.random.default_rng(11)
n = 400
P = rng.standard_normal(n).tolist()
mus = np.linspace(0.0, 2.5, 11)

mmd_vals = []
for mu in mus:
    half = n // 2
    Q = np.concatenate([
        rng.standard_normal(half) - mu,
        rng.standard_normal(n - half) + mu,
    ]).tolist()
    mmd_vals.append(float(opt.mmd_gaussian(P, Q, sigma=1.0)))

for mu, m in zip(mus, mmd_vals):
    print(f\"μ = {mu:.2f}  ->  MMD² = {m:.3e}\")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(mus, mmd_vals, 'o-', lw=2, color='C2')
axes[0].set_xlabel(r'séparation $\\mu$')
axes[0].set_ylabel(r'$\\widehat{\\text{MMD}}^2$')
axes[0].set_title(\"MMD croissante avec la séparation\")

mu_show = mus[-1]
half = n // 2
Q_show = np.concatenate([
    rng.standard_normal(half) - mu_show,
    rng.standard_normal(n - half) + mu_show,
])
axes[1].hist(P, bins=30, density=True, alpha=0.5,
             label=r'P = $\\mathcal{N}(0,1)$', color='C0')
axes[1].hist(Q_show, bins=30, density=True, alpha=0.5,
             label=fr'Q (μ = {mu_show:.1f})', color='C3')
axes[1].set_xlabel('x'); axes[1].set_ylabel('densité')
axes[1].set_title(\"P vs Q (mélange séparé)\")
axes[1].legend()
fig.tight_layout(); plt.show()
"""),
    md("""**Résultat attendu.** MMD croît monotone en $\\mu$.

**Lecture du graphique.** Gauche : courbe croissante.  Droite : la
gaussienne unique vs le mélange bimodal sont visuellement
distincts à $\\mu = 2.5$.

**Conclusion.** MMD avec noyau gaussien est un détecteur efficace de
différence distributionnelle, applicable à la calibration de modèles
génératifs (GAN, normalizing flows) en boucle d'apprentissage.
"""),
]


# ---------------------------------------------------------------------------
NOTEBOOKS = [
    ("10_bsde.ipynb", NB10_TITLE, NB10_CELLS),
    ("11_pde.ipynb", NB11_TITLE, NB11_CELLS),
    ("12_stochastic_control.ipynb", NB12_TITLE, NB12_CELLS),
    ("13_quadratic_impact.ipynb", NB13_TITLE, NB13_CELLS),
    ("14_mckean_vlasov.ipynb", NB14_TITLE, NB14_CELLS),
    ("15_agent_based.ipynb", NB15_TITLE, NB15_CELLS),
    ("16_robust_drift.ipynb", NB16_TITLE, NB16_CELLS),
    ("17_generative_calibration.ipynb", NB17_TITLE, NB17_CELLS),
]


def main() -> None:
    failures: list[str] = []
    for name, title, cells in NOTEBOOKS:
        path = write_notebook(name, title, cells)
        print(f"  written {path.name}  ({len(cells) + 2} cells)")
        rc = subprocess.run(
            [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.kernel_name=rhftlab",
                "--ExecutePreprocessor.timeout=300",
                str(path),
            ],
            check=False,
        )
        status = "OK" if rc.returncode == 0 else f"FAIL ({rc.returncode})"
        if rc.returncode != 0:
            failures.append(name)
        print(f"  executed {path.name} -> {status}")
    print()
    if failures:
        print(f"FAILURES: {failures}")
    else:
        print("ALL NOTEBOOKS OK")


if __name__ == "__main__":
    main()
