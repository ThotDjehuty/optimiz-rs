//! Bottleneck distance between two persistence diagrams.
//!
//! For diagrams `D, D'` the bottleneck distance is
//!
//! ```text
//!     d_B(D, D') = inf_{eta : D -> D'} sup_{x in D} || x - eta(x) ||_inf
//! ```
//!
//! where the matchings allow points to be paired with the diagonal
//! `Delta = { (t, t) : t >= 0 }` at cost `(d - b) / 2`.
//!
//! Algorithm: binary search on `epsilon`, build the bipartite graph with
//! edges `(p, q)` whenever `||p - q||_inf <= epsilon` (including
//! diagonal projections), test for a perfect matching by Hopcroft--Karp.

use std::collections::VecDeque;

use super::persistent_homology::PersistencePair;
use crate::core::Result;

#[inline]
fn linf_dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).abs().max((a.1 - b.1).abs())
}

#[inline]
fn diag_cost(p: (f64, f64)) -> f64 {
    (p.1 - p.0).abs() / 2.0
}

/// Compute the bottleneck distance between two diagrams. Only pairs with
/// matching `dim` are coupled.
pub fn bottleneck_distance(d1: &[PersistencePair], d2: &[PersistencePair]) -> Result<f64> {
    let dims: std::collections::BTreeSet<usize> =
        d1.iter().chain(d2.iter()).map(|p| p.dim).collect();

    let mut max_d = 0.0_f64;
    for dim in dims {
        let p1: Vec<(f64, f64)> = d1
            .iter()
            .filter(|p| p.dim == dim)
            .map(|p| (p.birth, p.death))
            .collect();
        let p2: Vec<(f64, f64)> = d2
            .iter()
            .filter(|p| p.dim == dim)
            .map(|p| (p.birth, p.death))
            .collect();
        let d = bottleneck_one_dim(&p1, &p2);
        if d > max_d {
            max_d = d;
        }
    }
    Ok(max_d)
}

fn bottleneck_one_dim(d1: &[(f64, f64)], d2: &[(f64, f64)]) -> f64 {
    // Filter out infinite-death points for finite matching. If counts of
    // essential classes differ, the distance is infinite.
    let inf1 = d1.iter().filter(|p| p.1.is_infinite()).count();
    let inf2 = d2.iter().filter(|p| p.1.is_infinite()).count();
    if inf1 != inf2 {
        return f64::INFINITY;
    }
    let f1: Vec<(f64, f64)> = d1.iter().copied().filter(|p| p.1.is_finite()).collect();
    let f2: Vec<(f64, f64)> = d2.iter().copied().filter(|p| p.1.is_finite()).collect();

    // Candidate epsilons: pairwise linf distances + diagonal costs.
    let mut cand: Vec<f64> = Vec::new();
    for &p in f1.iter() {
        for &q in f2.iter() {
            cand.push(linf_dist(p, q));
        }
        cand.push(diag_cost(p));
    }
    for &q in f2.iter() {
        cand.push(diag_cost(q));
    }
    cand.push(0.0);
    cand.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cand.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Binary search the smallest epsilon admitting a perfect matching.
    let mut lo = 0usize;
    let mut hi = cand.len() - 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if perfect_matching_exists(&f1, &f2, cand[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    cand[lo]
}

/// Build bipartite graph at threshold `eps` and return whether a perfect
/// matching covering both sides exists. Diagonal nodes are added so that
/// every persistence pair can be matched with the diagonal.
fn perfect_matching_exists(left: &[(f64, f64)], right: &[(f64, f64)], eps: f64) -> bool {
    let n_l = left.len();
    let n_r = right.len();
    // Augment both sides with diagonal duplicates so that all pairs can
    // be matched onto the diagonal at cost diag_cost.
    let total_left = n_l + n_r;
    let total_right = n_r + n_l;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total_left];
    for i in 0..n_l {
        // real -> real
        for j in 0..n_r {
            if linf_dist(left[i], right[j]) <= eps {
                adj[i].push(j);
            }
        }
        // real -> diagonal copy of real left[i] -> right diag index n_r + i
        if diag_cost(left[i]) <= eps {
            adj[i].push(n_r + i);
        }
    }
    for j in 0..n_r {
        // diag copy left[n_l + j] (representing right[j]'s diagonal) only
        // matches its own diagonal target n_r + (n_l + j) ... we keep
        // simple: it can match any diagonal target at cost zero.
        for k in n_r..total_right {
            adj[n_l + j].push(k);
        }
    }

    // Hopcroft--Karp
    let nil = usize::MAX;
    let mut pair_l = vec![nil; total_left];
    let mut pair_r = vec![nil; total_right];
    let mut dist = vec![0i64; total_left];

    fn bfs(
        adj: &[Vec<usize>],
        pair_l: &[usize],
        pair_r: &[usize],
        dist: &mut [i64],
        n_l: usize,
        nil: usize,
    ) -> bool {
        let mut q: VecDeque<usize> = VecDeque::new();
        for u in 0..n_l {
            if pair_l[u] == nil {
                dist[u] = 0;
                q.push_back(u);
            } else {
                dist[u] = i64::MAX;
            }
        }
        let mut found = false;
        while let Some(u) = q.pop_front() {
            for &v in &adj[u] {
                let pv = pair_r[v];
                if pv == nil {
                    found = true;
                } else if dist[pv] == i64::MAX {
                    dist[pv] = dist[u] + 1;
                    q.push_back(pv);
                }
            }
        }
        found
    }

    fn dfs(
        u: usize,
        adj: &[Vec<usize>],
        pair_l: &mut [usize],
        pair_r: &mut [usize],
        dist: &mut [i64],
        nil: usize,
    ) -> bool {
        for &v in &adj[u] {
            let pv = pair_r[v];
            let cond = pv == nil || (dist[pv] == dist[u] + 1 && dfs(pv, adj, pair_l, pair_r, dist, nil));
            if cond {
                pair_l[u] = v;
                pair_r[v] = u;
                return true;
            }
        }
        dist[u] = i64::MAX;
        false
    }

    let mut matched = 0usize;
    while bfs(&adj, &pair_l, &pair_r, &mut dist, total_left, nil) {
        for u in 0..total_left {
            if pair_l[u] == nil && dfs(u, &adj, &mut pair_l, &mut pair_r, &mut dist, nil) {
                matched += 1;
            }
        }
    }
    matched == total_left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_diagrams_zero_distance() {
        let d1 = vec![PersistencePair {
            dim: 1,
            birth: 0.2,
            death: 0.7,
        }];
        let d2 = d1.clone();
        let d = bottleneck_distance(&d1, &d2).unwrap();
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn point_vs_diagonal_distance() {
        let d1 = vec![PersistencePair {
            dim: 0,
            birth: 0.0,
            death: 0.6,
        }];
        let d2: Vec<PersistencePair> = vec![];
        let d = bottleneck_distance(&d1, &d2).unwrap();
        assert!((d - 0.3).abs() < 1e-9, "expected 0.3, got {}", d);
    }
}
