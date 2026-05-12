Path Signatures
===============

The module :code:`signatures` provides truncated tensor signatures
(Lyons 1998), log-signatures, random reservoir projections, and the
Salvi--Cass--Lyons signature kernel.

Truncated Signature
-------------------

For a continuous path :math:`X : [0, T] \to \mathbb{R}^d` of bounded
variation, the *signature* is the formal series

.. math::

   S(X)_{0,T}
   \;=\;
   1 + \sum_{k \ge 1} \sum_{i_1, \dots, i_k}
       S^{i_1, \dots, i_k}_{0, T}\,
       e_{i_1} \otimes \dots \otimes e_{i_k},

with iterated Stieltjes integrals

.. math::

   S^{i_1, \dots, i_k}_{0, T}
   \;=\;
   \int_{0 < u_1 < \dots < u_k < T}
       dX^{i_1}_{u_1}\, \dots\, dX^{i_k}_{u_k}.

For piecewise-linear input with increments :math:`\Delta_n`, the
truncated signature obeys the multiplicative recursion

.. math::

   S^{(M)}_{0, t_n}
   \;=\;
   S^{(M)}_{0, t_{n-1}}\,\otimes_M\,\exp_M(\Delta_n),

where :math:`\exp_M(\Delta) = \sum_{k=0}^M \Delta^{\otimes k} / k!`.

Log-Signature
-------------

The truncated tensor logarithm

.. math::

   \log(S)
   \;=\;
   \sum_{n \ge 1} \frac{(-1)^{n+1}}{n}\,(S - 1)^{\otimes n}

lives in the truncated free Lie algebra and provides a more
parsimonious representation.

Random Signature
----------------

Following Cuchiero--Schmocker--Teichmann (2023), one drives a random
reservoir on :math:`\mathbb{R}^N`,

.. math::

   dZ_t = A_0 Z_t\, dt + \sum_{i=1}^d A_i Z_t\, dX^i_t,

with random matrices :math:`A_i \in \mathbb{R}^{N \times N}` whose
entries are i.i.d. Gaussian with variance :math:`1/N`. The map
:math:`X \mapsto Z_T` is a finite-dimensional random projection of
:math:`S(X)`.

Signature Kernel (Salvi--Cass--Lyons)
-------------------------------------

The signature inner product

.. math::

   K(s, t) \;=\; \langle S(X)_{0, s},\; S(Y)_{0, t}\rangle

solves the linear hyperbolic PDE

.. math::

   \frac{\partial^2 K}{\partial s\,\partial t}
   \;=\;
   \langle \dot X_s, \dot Y_t \rangle\, K(s, t),
   \qquad
   K(s, 0) = K(0, t) = 1.

It is integrated on a uniform grid via the Goursat scheme

.. math::

   K_{i+1, j+1}
   = K_{i+1, j} + K_{i, j+1} - K_{i, j}
     + \langle \Delta x_i, \Delta y_j\rangle\,
       \tfrac{1}{2}(K_{i+1, j} + K_{i, j+1}).

API
---

.. code-block:: rust

   pub struct TruncatedSignature {
       pub channels: usize,
       pub level: usize,
       pub tensors: Vec<Vec<f64>>,
   }
   pub fn path_signature(path: &[Vec<f64>], level: usize) -> Result<TruncatedSignature>;
   pub fn log_signature(sig: &TruncatedSignature) -> Result<TruncatedLogSignature>;

   pub struct RandomSignatureConfig {
       pub reservoir_dim: usize, pub seed: u64, pub variance: f64,
   }
   pub fn random_signature(path: &[Vec<f64>], cfg: &RandomSignatureConfig)
       -> Result<RandomSignatureResult>;

   pub fn signature_kernel(x: &[Vec<f64>], y: &[Vec<f64>])
       -> Result<SignatureKernelResult>;
