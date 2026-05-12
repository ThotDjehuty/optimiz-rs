Graph Laplacians and Spectral Clustering
========================================

The module :code:`graph` provides graph Laplacian operators and a
spectral clustering algorithm built on a Jacobi diagonaliser.

Laplacians
----------

For a weighted undirected graph with adjacency matrix :math:`W \in \mathbb{R}^{n\times n}_{\ge 0}`
and degree matrix :math:`D = \mathrm{diag}(W \mathbf{1})`:

- **Combinatorial**: :math:`L = D - W`.
- **Symmetric normalised**: :math:`L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}`.
- **Random-walk normalised**: :math:`L_{\mathrm{rw}} = I - D^{-1} W`.

Each operator is positive semidefinite and the multiplicity of the
zero eigenvalue equals the number of connected components.

Spectral Clustering
-------------------

Given :math:`W` and a target number of clusters :math:`k`:

1. Build :math:`L_{\mathrm{sym}}` (or another Laplacian).
2. Diagonalise via cyclic Jacobi rotations to obtain the eigenpairs
   :math:`(\lambda_i, u_i)`.
3. Stack the :math:`k` eigenvectors associated with the smallest
   eigenvalues as columns of :math:`U \in \mathbb{R}^{n \times k}`.
4. Normalise rows of :math:`U` and run Lloyd's algorithm with
   k-means++ initialisation on the rows.

The Fiedler eigenvalue :math:`\lambda_2` is reported separately as a
proxy for the spectral gap.

API
---

.. code-block:: rust

   pub enum LaplacianKind { Combinatorial, SymmetricNormalised, RandomWalk }
   pub fn combinatorial_laplacian(w: ArrayView2<f64>) -> Result<Array2<f64>>;
   pub fn normalised_laplacian(w: ArrayView2<f64>) -> Result<Array2<f64>>;
   pub fn random_walk_laplacian(w: ArrayView2<f64>) -> Result<Array2<f64>>;

   pub struct SpectralClusterResult {
       pub labels: Vec<usize>,
       pub eigenvalues: Vec<f64>,
       pub fiedler_value: f64,
   }
   pub fn spectral_cluster(w: ArrayView2<f64>, k: usize, n_kmeans_iter: usize, seed: u64)
       -> Result<SpectralClusterResult>;
