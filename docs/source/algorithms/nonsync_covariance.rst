Asynchronous Covariance (Hayashi--Yoshida)
==========================================

The module :code:`timeseries_utils::nonsync_covariance` implements the
Hayashi--Yoshida estimator of the integrated covariance between two
asynchronously sampled processes :math:`X` and :math:`Y` observed at
distinct, non-overlapping observation grids
:math:`\{t^X_i\}` and :math:`\{t^Y_j\}`.

Estimator
---------

Let :math:`I_i = (t^X_{i-1}, t^X_i]` and :math:`J_j = (t^Y_{j-1}, t^Y_j]`.
The Hayashi--Yoshida estimator is

.. math::

   \widehat{\langle X, Y\rangle}_{[0,T]}
   \;=\;
   \sum_{i, j}\,
   \big(X_{t^X_i} - X_{t^X_{i-1}}\big)\,
   \big(Y_{t^Y_j} - Y_{t^Y_{j-1}}\big)\,
   \mathbf{1}\!\big[I_i \cap J_j \neq \emptyset\big].

It is consistent under non-synchronicity and avoids the *Epps effect*
that plagues naive grid interpolation.

Implementation
--------------

* A two-pointer scan in :math:`O(n_X + n_Y)` collects all overlapping
  pairs.
* For matrices of size :math:`d \times d` with large per-asset sample
  counts, off-diagonal entries are computed in parallel with Rayon.

API
---

.. code-block:: rust

   pub fn hayashi_yoshida_covariance(
       t1: &[f64], v1: &[f64],
       t2: &[f64], v2: &[f64],
   ) -> Result<f64>;

   pub fn hayashi_yoshida_matrix(
       series: &[(Vec<f64>, Vec<f64>)],
   ) -> Result<Vec<Vec<f64>>>;
