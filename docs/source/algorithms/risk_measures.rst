Risk Measures: VaR and CVaR
============================

The module :code:`risk_measures` provides Value-at-Risk and Conditional
Value-at-Risk estimators together with a convex CVaR minimisation
solver over the unit simplex.

Definitions
-----------

For a real random variable :math:`L` (a *loss*), the Value-at-Risk at
confidence level :math:`\alpha \in (0, 1)` is the lower :math:`\alpha`-
quantile

.. math::

   \mathrm{VaR}_\alpha(L)
   \;=\;
   \inf\!\big\{ \ell \in \mathbb{R} : \mathbb{P}(L \le \ell) \ge \alpha \big\}.

The Conditional Value-at-Risk (also called Average Value-at-Risk) is

.. math::

   \mathrm{CVaR}_\alpha(L)
   \;=\;
   \frac{1}{1-\alpha}\,
   \int_\alpha^1 \mathrm{VaR}_u(L)\,du.

For a sample :math:`L_1, \dots, L_n` of i.i.d. losses sorted in increasing
order, the empirical CVaR at level :math:`\alpha` is

.. math::

   \widehat{\mathrm{CVaR}}_\alpha
   \;=\;
   \frac{1}{n - k}\, \sum_{i = k+1}^{n} L_{(i)},
   \qquad k = \lfloor \alpha\, n \rfloor.

Convex minimisation
-------------------

Rockafellar--Uryasev (2000) showed that

.. math::

   \mathrm{CVaR}_\alpha(L)
   \;=\;
   \min_{\zeta \in \mathbb{R}}\;
   \zeta + \frac{1}{1 - \alpha}\,\mathbb{E}\!\big[(L - \zeta)_+\big].

Given samples of a vector :math:`r^{(s)} \in \mathbb{R}^d`,
:code:`minimize_cvar` solves

.. math::

   \min_{w \in \Delta_d,\;\zeta \in \mathbb{R}}\;
   \zeta + \frac{1}{(1 - \alpha)\, S}\, \sum_{s=1}^S
       \big(\zeta - \langle r^{(s)}, w\rangle\big)_+,

over the unit simplex :math:`\Delta_d`, by a projected sub-gradient
method using the Held--Wolfe--Crowder simplex projection.

API
---

.. code-block:: rust

   pub fn historical_var(losses: &[f64], alpha: f64) -> Result<f64>;
   pub fn parametric_var(mu: f64, sigma: f64, alpha: f64) -> Result<f64>;
   pub fn cvar_value(losses: &[f64], alpha: f64) -> Result<f64>;
   pub fn minimize_cvar(returns: ArrayView2<f64>, cfg: &CVaRConfig)
       -> Result<CVaRResult>;
