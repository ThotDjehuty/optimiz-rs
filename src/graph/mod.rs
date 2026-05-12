//! Generic graph algorithms.
//!
//! Currently exposes spectral utilities on dense weighted graphs:
//!
//! * `laplacian`           -- combinatorial / normalised / random-walk Laplacians.
//! * `spectral_clustering` -- Ng--Jordan--Weiss algorithm (k-means on
//!                            normalised eigenvectors of L_sym).

pub mod laplacian;
pub mod spectral_clustering;


pub use laplacian::{combinatorial_laplacian, normalised_laplacian, random_walk_laplacian, LaplacianKind};
pub use spectral_clustering::{spectral_cluster, SpectralClusterResult};

#[cfg(feature = "python-bindings")]
pub mod python_bindings;
