"""Replace the 4 remaining ASCII diagram blocks in mathematical_foundations.md
with {figure} directives pointing to the new SVGs.
"""
import pathlib

MD = pathlib.Path(__file__).parent / "theory" / "mathematical_foundations.md"
text = MD.read_text(encoding="utf-8")

# ── 1.  HMM regime state machine → fig_hmm_regime ──────────────────────────
old1 = '''\
```
  HMM regime state machine  (K = 3)
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

       A₁₂ →                A₂₃ →
  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
  │  State 1   │──────▶│  State 2   │──────▶│  State 3   │
  │   Bull     │◀──────│  Neutral   │◀──────│  Bear     │
  └─────────────┘        └─────────────┘        └─────────────┘
           ← A₂₁              ← A₃₂

  Emission B_k(y) = 𝒩(μ_k, σ_k²):
  ┌────────┬────────┬────────┬──────────────────┐
  │ State  │  μ    │  σ    │ Character             │
  ├────────┼────────┼────────┼──────────────────┤
  │ Bull   │ +0.05 │  0.12 │ high return, low vol  │
  │ Neutral│  0.00 │  0.18 │ flat, medium vol      │
  │ Bear   │ -0.08 │  0.35 │ crash, high vol       │
  └────────┴────────┴────────┴──────────────────┘
  (self-transition: A₁₁=0.97,  A₂₂=0.97,  A₃₃=0.90)
```'''

new1 = '''\
```{figure} ../_static/diagrams/fig_hmm_regime.svg
:align: center
:width: 90%

HMM $K=3$ state machine with Bull / Neutral / Bear regimes and Gaussian emission
parameters. Self-transitions $A_{11}=A_{22}=0.97$, $A_{33}=0.90$.
```'''

# ── 2.  Viterbi trellis → fig_viterbi_trellis ───────────────────────────────
old2 = '''\
```
  Viterbi trellis  (K=3, T=4)
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  State   t=1         t=2         t=3         t=4

    1    ○─────────▶○─────────▶○─────────▶○
           ╲              ╳
    2    ○─────────▶●─────────▶●─────────▶○   ● = MAP path
           ╲       ╲         ╲
    3    ○─────────▶○─────────▶○─────────▶○

  δ_t(k) = max_j [δ_{t−1}(j) · A_jk · B_k(y_t)]
  ψ_t(k) = argmax_j  ← backtrack pointer

  Traceback: z_4★ ← z_3★ ← z_2★ ← z_1★  via ψ
```'''

new2 = r'''\
```{figure} ../_static/diagrams/fig_viterbi_trellis.svg
:align: center
:width: 82%

Viterbi trellis ($K=3$, $T=4$). Filled nodes mark the MAP (most probable) state
sequence; arrows show transition candidates. Backtracking via $\psi_t(k)$ recovers
$z_1^\star \to z_4^\star$.
```'''

# ── 3.  Standard vs natural gradient (text comparison) → fig_std_vs_nat_gradient
old3 = '''\
```
  Standard vs natural gradient
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  Standard:  θ_{k+1} = θ_k − η·∇ℒ       Natural:  θ_{k+1} = θ_k − η·ℐ(θ)^{−1}∇ℒ
  ────────────────────────────────────────────

  ┌────────────────────┐ ┌────────────────────┐
  │ Flat ℝᵈ geometry    │ │ Riemannian metric ℐ(θ) │
  │ Ignores curvature  │ │ Adapts to geometry    │
  │ Slow on ill-cond ℐ │ │ Reparam invariant     │
  │ O(κ(ℐ)) iters      │ │ O(1) on exp families  │
  └────────────────────┘ └────────────────────┘

  On Gaussian / exponential family:  ℐ⁻¹∇ℒ = MLE step → 1 iteration!
```'''

new3 = '''\
```{figure} ../_static/diagrams/fig_std_vs_nat_gradient.svg
:align: center
:width: 88%

Standard versus natural gradient: geometric properties. On exponential families
the natural gradient equals the MLE Newton step, achieving convergence in one
iteration.
```'''

# ── 4.  Matrix Lie group hierarchy → fig_lie_group_hierarchy ─────────────────
old4 = '''\
```
  Matrix Lie group hierarchy
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  GL(n,ℝ)  ─  all invertible n×n real matrices
      │
      ├──▶ SL(n,ℝ)   det = 1
      │
      ├──▶ O(n)      RᵀR = I  (orthogonal)
      │      └─▶ SO(n)  det = +1  (pure rotations)
      │               ↳ portfolio factor rotation, PCA constraints
      │
      └──▶ Sp(2n,ℝ)  preserves symplectic form ω
                     ↳ Hamiltonian mechanics, PMP §4.2 / §10.4

  H(n)  Heisenberg  ─  upper triangular, 1s on diagonal
             ↳ path-signature feature maps
```'''

new4 = r'''\
```{figure} ../_static/diagrams/fig_lie_group_hierarchy.svg
:align: center
:width: 90%

Matrix Lie group hierarchy: subgroup inclusions and their quantitative-finance
applications. $SO(n)$ underpins PCA factor rotation; $\mathrm{Sp}(2n,\mathbb{R})$
governs Hamiltonian mechanics (PMP §10.4); $H(n)$ drives path-signature features.
```'''

replacements = [(old1, new1), (old2, new2), (old3, new3), (old4, new4)]
for i, (old, new) in enumerate(replacements, 1):
    if old in text:
        text = text.replace(old, new, 1)
        print(f"  Block {i}: replaced OK")
    else:
        print(f"  Block {i}: NOT FOUND — check encoding/whitespace")

MD.write_text(text, encoding="utf-8")
print("Done.")
