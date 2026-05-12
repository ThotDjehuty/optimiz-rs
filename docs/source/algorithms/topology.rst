Topological Data Analysis
=========================

The module :code:`topology` implements Vietoris--Rips persistent
homology and the bottleneck distance between persistence diagrams.

Vietoris--Rips Filtration
-------------------------

For a finite point cloud :math:`X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d`
and scale :math:`\varepsilon \ge 0`, the Vietoris--Rips complex is

.. math::

   \mathrm{VR}_\varepsilon(X)
   \;=\;
   \big\{ \sigma \subseteq X : \mathrm{diam}(\sigma) \le \varepsilon \big\}.

Increasing :math:`\varepsilon` yields a filtration; the persistent
homology of this filtration produces, for each homological degree
:math:`k`, a multiset of birth/death pairs

.. math::

   D_k(X) = \big\{ (b_i, d_i) : 0 \le b_i < d_i \le \infty \big\}.

Persistence Algorithm
---------------------

The boundary matrix :math:`\partial` is built over :math:`\mathbb{Z}/2`
and reduced left-to-right: for each column :math:`j` we cancel its
lowest entry by adding any earlier column with the same low. Pairs
:math:`(\mathrm{low}(j), j)` give birth/death pairs.

Bottleneck Distance
-------------------

For two diagrams :math:`D` and :math:`D'`,

.. math::

   d_B(D, D')
   \;=\;
   \inf_{\eta : D \to D'}\;
   \sup_{x \in D}\, \|x - \eta(x)\|_\infty,

where matchings may pair points with the diagonal
:math:`\Delta = \{(t, t) : t \ge 0\}` at cost :math:`(d - b)/2`.

The implementation binary-searches the threshold :math:`\varepsilon`
and certifies a perfect matching by Hopcroft--Karp on the bipartite
graph of admissible edges.

API
---

.. code-block:: rust

   pub struct PersistencePair { pub dim: usize, pub birth: f64, pub death: f64 }
   pub struct PersistenceDiagram { pub pairs: Vec<PersistencePair> }

   pub fn vietoris_rips_filtration(points: &[Vec<f64>], max_dim: usize, max_eps: f64)
       -> Result<Vec<Simplex>>;
   pub fn persistent_homology(points: &[Vec<f64>], max_dim: usize, max_eps: f64)
       -> Result<PersistenceDiagram>;
   pub fn bottleneck_distance(d1: &[PersistencePair], d2: &[PersistencePair])
       -> Result<f64>;
