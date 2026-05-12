Discrete and Maximum-Overlap Wavelet Transforms
================================================

The module :code:`timeseries_utils::wavelet` provides Haar and Daubechies
wavelet transforms with periodic boundary handling.

Filter banks
------------

For an orthogonal scaling filter :math:`\{h_k\}_{k=0}^{L-1}` the quadrature
mirror filter (QMF) is

.. math::

   g_k = (-1)^k\, h_{L - 1 - k},

so that :math:`\sum_k h_k = \sqrt{2}` and :math:`\sum_k g_k = 0`.

DWT (one level, periodic)
-------------------------

For an input vector :math:`x \in \mathbb{R}^N` with :math:`N` even,

.. math::

   a_n = \sum_{k=0}^{L-1} h_k\, x_{(2n + k)\bmod N},
   \qquad
   d_n = \sum_{k=0}^{L-1} g_k\, x_{(2n + k)\bmod N},
   \qquad n = 0, \dots, N/2 - 1.

Successive levels apply the same filter to the previous approximation
:math:`a^{(j)}`.

MODWT (Maximum Overlap)
-----------------------

The MODWT does not downsample: at level :math:`j`, the filter is dilated
by inserting :math:`2^{j-1} - 1` zeros between successive taps and applied
in a periodic convolution. The result is shift-invariant.

API
---

.. code-block:: rust

   pub enum WaveletFamily { Haar, Daubechies(u8) }
   pub fn scaling_filter(family: WaveletFamily) -> Result<Vec<f64>>;
   pub fn qmf(h: &[f64]) -> Vec<f64>;
   pub fn dwt_step(x: &[f64], h: &[f64], g: &[f64]) -> (Vec<f64>, Vec<f64>);
   pub fn dwt(x: &[f64], family: WaveletFamily, levels: usize) -> Result<Vec<Vec<f64>>>;
   pub fn modwt_step(x: &[f64], h: &[f64], g: &[f64], level: usize) -> (Vec<f64>, Vec<f64>);
