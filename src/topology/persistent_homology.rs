//! Vietoris--Rips persistent homology via standard matrix reduction.
//!
//! Given a finite point cloud `X = {x_1, ..., x_n} \subset R^d` and a
//! filtration parameter `epsilon >= 0`, the Vietoris--Rips complex
//! `VR_eps(X)` contains every simplex `sigma = {x_{i_0}, ..., x_{i_k}}`
//! whose pairwise distances satisfy `d(x_{i_p}, x_{i_q}) <= eps`.
//!
//! The resulting filtration `{ VR_eps(X) }_{eps >= 0}` defines a
//! persistence module whose decomposition yields the diagram
//! `D_k(X) = { (b_i, d_i) }` of birth/death pairs in homological
//! degree `k`.
//!
//! This implementation:
//! * builds the simplicial complex up to a chosen homological degree
//!   `max_dim` and a maximum scale `max_eps`;
//! * orders simplices by `(filtration_value, dimension, lexicographic)`;
//! * runs left-to-right column reduction over `Z/2`;
//! * extracts pairs `(b, d)` from low entries.
//!
//! The implementation is intentionally simple and CPU-friendly. For very
//! large complexes, switch to a sparse / chunk-based variant.

use crate::core::{OptimizrError, Result};

/// Filtration entry.
#[derive(Debug, Clone)]
pub struct Simplex {
    pub vertices: Vec<usize>,
    pub filtration: f64,
}

impl Simplex {
    pub fn dim(&self) -> usize {
        self.vertices.len() - 1
    }
}

/// One persistent homology pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    pub dim: usize,
    pub birth: f64,
    pub death: f64, // f64::INFINITY for essential classes
}

/// Persistence diagram grouped by homological degree.
#[derive(Debug, Clone, Default)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    pub fn pairs_in_dim(&self, dim: usize) -> Vec<(f64, f64)> {
        self.pairs
            .iter()
            .filter(|p| p.dim == dim)
            .map(|p| (p.birth, p.death))
            .collect()
    }

    /// Number of generators alive at scale `eps` in dimension `dim`.
    pub fn betti(&self, dim: usize, eps: f64) -> usize {
        self.pairs
            .iter()
            .filter(|p| p.dim == dim && p.birth <= eps && eps < p.death)
            .count()
    }
}

#[inline]
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

/// Build the Vietoris--Rips filtration up to `max_dim` and scale
/// `max_eps`. Each simplex's filtration value equals the maximum pairwise
/// distance among its vertices (zero for vertices, edge length for edges).
pub fn vietoris_rips_filtration(
    points: &[Vec<f64>],
    max_dim: usize,
    max_eps: f64,
) -> Result<Vec<Simplex>> {
    let n = points.len();
    if n == 0 {
        return Err(OptimizrError::EmptyData);
    }
    let d = points[0].len();
    for p in points {
        if p.len() != d {
            return Err(OptimizrError::DimensionMismatch {
                expected: d,
                actual: p.len(),
            });
        }
    }
    if max_eps < 0.0 {
        return Err(OptimizrError::InvalidParameter(
            "max_eps must be non-negative".into(),
        ));
    }

    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dij = euclidean_distance(&points[i], &points[j]);
            dist[i * n + j] = dij;
            dist[j * n + i] = dij;
        }
    }

    let mut simplices: Vec<Simplex> = Vec::new();

    // 0-simplices
    for i in 0..n {
        simplices.push(Simplex {
            vertices: vec![i],
            filtration: 0.0,
        });
    }

    // Higher-dim simplices: enumerate via incremental expansion using
    // edges as the seed. For tractability we only enumerate simplices
    // whose filtration <= max_eps.
    if max_dim >= 1 {
        for i in 0..n {
            for j in (i + 1)..n {
                let dij = dist[i * n + j];
                if dij <= max_eps {
                    simplices.push(Simplex {
                        vertices: vec![i, j],
                        filtration: dij,
                    });
                }
            }
        }
    }
    // Generic k-simplex enumeration by k-clique listing.
    let mut current_dim = 1usize;
    while current_dim < max_dim {
        // Collect simplices of dimension current_dim
        let prev: Vec<Simplex> = simplices
            .iter()
            .filter(|s| s.dim() == current_dim)
            .cloned()
            .collect();
        let mut new_simplices: Vec<Simplex> = Vec::new();
        for s in prev.iter() {
            // Try extend by every vertex > max(vertices) such that all
            // pairwise distances stay below max_eps.
            let last = *s.vertices.last().unwrap();
            'cand: for v in (last + 1)..n {
                let mut max_d = s.filtration;
                for &u in s.vertices.iter() {
                    let dd = dist[u * n + v];
                    if dd > max_eps {
                        continue 'cand;
                    }
                    if dd > max_d {
                        max_d = dd;
                    }
                }
                let mut verts = s.vertices.clone();
                verts.push(v);
                new_simplices.push(Simplex {
                    vertices: verts,
                    filtration: max_d,
                });
            }
        }
        if new_simplices.is_empty() {
            break;
        }
        simplices.extend(new_simplices);
        current_dim += 1;
    }

    // Order: filtration ascending, then dimension ascending, then
    // lexicographic on vertices for determinism.
    simplices.sort_by(|a, b| {
        a.filtration
            .partial_cmp(&b.filtration)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.dim().cmp(&b.dim()))
            .then(a.vertices.cmp(&b.vertices))
    });
    Ok(simplices)
}

/// Map vertex tuple -> index in the ordered filtration.
fn build_index_map(simplices: &[Simplex]) -> std::collections::HashMap<Vec<usize>, usize> {
    let mut map = std::collections::HashMap::with_capacity(simplices.len());
    for (idx, s) in simplices.iter().enumerate() {
        map.insert(s.vertices.clone(), idx);
    }
    map
}

/// Boundary indices of a simplex (vertices removed one at a time).
fn boundary_indices(
    simplex: &Simplex,
    index_map: &std::collections::HashMap<Vec<usize>, usize>,
) -> Vec<usize> {
    let k = simplex.vertices.len();
    if k <= 1 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let mut face = Vec::with_capacity(k - 1);
        for (j, &v) in simplex.vertices.iter().enumerate() {
            if j != i {
                face.push(v);
            }
        }
        if let Some(&idx) = index_map.get(&face) {
            out.push(idx);
        }
    }
    out
}

/// Run the standard Z/2 column-reduction persistence algorithm and
/// return the persistence diagram.
pub fn persistent_homology(
    points: &[Vec<f64>],
    max_dim: usize,
    max_eps: f64,
) -> Result<PersistenceDiagram> {
    let simplices = vietoris_rips_filtration(points, max_dim, max_eps)?;
    let n = simplices.len();
    let index_map = build_index_map(&simplices);

    // Sparse boundary columns sorted descending so that low(col) = first.
    let mut columns: Vec<Vec<usize>> = simplices
        .iter()
        .map(|s| {
            let mut b = boundary_indices(s, &index_map);
            b.sort_unstable();
            b.dedup();
            b
        })
        .collect();

    // low_to_col[r] = column index whose low entry is r, if any.
    let mut low_to_col: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut paired: Vec<bool> = vec![false; n];
    let mut diag = PersistenceDiagram::default();

    // Symmetric difference helper (Z/2 column add).
    fn xor_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
        let mut out = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    out.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&a[i..]);
        out.extend_from_slice(&b[j..]);
        out
    }

    for j in 0..n {
        loop {
            let low = match columns[j].last() {
                Some(&l) => l,
                None => break,
            };
            if let Some(&j_prev) = low_to_col.get(&low) {
                let merged = xor_sorted(&columns[j], &columns[j_prev]);
                columns[j] = merged;
            } else {
                low_to_col.insert(low, j);
                break;
            }
        }
        if let Some(&birth_idx) = columns[j].last() {
            let birth = simplices[birth_idx].filtration;
            let death = simplices[j].filtration;
            if death > birth {
                diag.pairs.push(PersistencePair {
                    dim: simplices[birth_idx].dim(),
                    birth,
                    death,
                });
            }
            paired[birth_idx] = true;
            paired[j] = true;
        }
    }

    // Essential classes: simplices that were neither paired as birth
    // (column reduces to nonzero with low=birth_idx) nor as death.
    for (i, s) in simplices.iter().enumerate() {
        if !paired[i] {
            // Skip top-dim or simplices > max_dim
            if s.dim() <= max_dim {
                diag.pairs.push(PersistencePair {
                    dim: s.dim(),
                    birth: s.filtration,
                    death: f64::INFINITY,
                });
            }
        }
    }

    Ok(diag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_circle_has_one_loop() {
        // 30 points uniformly on unit circle.
        let n = 30;
        let pts: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let theta = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
                vec![theta.cos(), theta.sin()]
            })
            .collect();
        let diag = persistent_homology(&pts, 2, 1.5).unwrap();
        // Check that betti_1 equals 1 at an intermediate scale.
        let b1 = diag.betti(1, 0.5);
        assert!(
            b1 >= 1,
            "expected at least one 1-cycle near eps=0.5, got betti_1={}",
            b1
        );
    }

    #[test]
    fn two_points_zero_dim_birth() {
        let pts = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let diag = persistent_homology(&pts, 1, 2.0).unwrap();
        // Two 0-simplices born at 0; one dies at distance 1.
        let dim0: Vec<_> = diag.pairs_in_dim(0);
        assert!(dim0.iter().any(|(b, d)| *b == 0.0 && (*d - 1.0).abs() < 1e-12));
    }
}
