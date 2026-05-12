//! Topological data analysis utilities.
//!
//! * `persistent_homology` -- Vietoris--Rips persistence via the standard
//!   matrix-reduction algorithm (Edelsbrunner & Harer, 2010).
//! * `bottleneck`          -- bottleneck distance between two persistence
//!   diagrams using the standard `O(n^{2.5})` matching (Hopcroft--Karp on
//!   thresholded bipartite graph + binary search on epsilon).

pub mod bottleneck;
pub mod persistent_homology;


pub use bottleneck::bottleneck_distance;
pub use persistent_homology::{
    persistent_homology, vietoris_rips_filtration, PersistenceDiagram, PersistencePair,
};
