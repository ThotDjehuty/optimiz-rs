//! Spectral clustering on dense weighted graphs (Ng--Jordan--Weiss).
//!
//! Algorithm:
//!
//! 1. Build `L_sym = I - D^{-1/2} W D^{-1/2}`.
//! 2. Compute the `k` eigenvectors of `L_sym` associated with the `k`
//!    smallest eigenvalues, stack them into `U in R^{n x k}`.
//! 3. Normalise each row of `U` to unit `l_2` norm.
//! 4. Apply Lloyd's `k`-means to the rows of `U`.
//!
//! Eigen-decomposition uses a symmetric Jacobi rotation (no external
//! linear-algebra dependency) which is appropriate for the small
//! to medium dense matrices that spectral clustering targets.

use ndarray::{Array1, Array2, ArrayView2};

use crate::core::{OptimizrError, Result};
use super::laplacian::normalised_laplacian;

/// Result of spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralClusterResult {
    pub labels: Vec<usize>,
    pub eigenvalues: Vec<f64>,
    pub fiedler_value: f64,
}

/// Spectral clustering of `n` items into `k` groups using a non-negative
/// symmetric similarity matrix `w`.
pub fn spectral_cluster(
    w: ArrayView2<f64>,
    k: usize,
    n_kmeans_iter: usize,
    seed: u64,
) -> Result<SpectralClusterResult> {
    if k < 2 {
        return Err(OptimizrError::InvalidParameter("k must be >= 2".into()));
    }
    let n = w.nrows();
    if n < k {
        return Err(OptimizrError::InvalidParameter(
            "k must not exceed n".into(),
        ));
    }
    let l_sym = normalised_laplacian(w)?;
    let (eigvals, eigvecs) = jacobi_symmetric_eig(&l_sym, 200, 1e-10)?;
    // Sort ascending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eigvals[a].partial_cmp(&eigvals[b]).unwrap_or(std::cmp::Ordering::Equal));
    let smallest_k: Vec<usize> = order.iter().take(k).copied().collect();

    // U[i, j] = eigvecs[i, smallest_k[j]]
    let mut u = Array2::<f64>::zeros((n, k));
    for j in 0..k {
        let col = smallest_k[j];
        for i in 0..n {
            u[[i, j]] = eigvecs[[i, col]];
        }
    }
    // Row-normalise
    for i in 0..n {
        let norm: f64 = (0..k).map(|j| u[[i, j]] * u[[i, j]]).sum::<f64>().sqrt();
        if norm > 0.0 {
            for j in 0..k {
                u[[i, j]] /= norm;
            }
        }
    }

    let labels = kmeans_lloyd(&u, k, n_kmeans_iter, seed);
    let sorted_eigvals: Vec<f64> = order.iter().map(|&i| eigvals[i]).collect();
    let fiedler = sorted_eigvals.get(1).copied().unwrap_or(0.0);
    Ok(SpectralClusterResult {
        labels,
        eigenvalues: sorted_eigvals,
        fiedler_value: fiedler,
    })
}

/// Lloyd's k-means with k-means++ seeding.
fn kmeans_lloyd(x: &Array2<f64>, k: usize, n_iter: usize, seed: u64) -> Vec<usize> {
    let n = x.nrows();
    let d = x.ncols();
    let mut state = if seed == 0 { 0xDEAD_BEEFu64 } else { seed };
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut centers = Array2::<f64>::zeros((k, d));
    let first = (next() * n as f64) as usize % n;
    for j in 0..d {
        centers[[0, j]] = x[[first, j]];
    }
    let mut min_d2 = vec![f64::INFINITY; n];
    for c in 1..k {
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..d {
                let diff = x[[i, j]] - centers[[c - 1, j]];
                s += diff * diff;
            }
            if s < min_d2[i] {
                min_d2[i] = s;
            }
        }
        let total: f64 = min_d2.iter().sum();
        let r = next() * total;
        let mut acc = 0.0;
        let mut chosen = 0;
        for i in 0..n {
            acc += min_d2[i];
            if acc >= r {
                chosen = i;
                break;
            }
        }
        for j in 0..d {
            centers[[c, j]] = x[[chosen, j]];
        }
    }

    let mut labels = vec![0usize; n];
    for _ in 0..n_iter {
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_d2 = f64::INFINITY;
            for c in 0..k {
                let mut s = 0.0;
                for j in 0..d {
                    let diff = x[[i, j]] - centers[[c, j]];
                    s += diff * diff;
                }
                if s < best_d2 {
                    best_d2 = s;
                    best = c;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut counts = vec![0usize; k];
        let mut new_centers = Array2::<f64>::zeros((k, d));
        for i in 0..n {
            counts[labels[i]] += 1;
            for j in 0..d {
                new_centers[[labels[i], j]] += x[[i, j]];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    new_centers[[c, j]] /= counts[c] as f64;
                }
            } else {
                for j in 0..d {
                    new_centers[[c, j]] = centers[[c, j]];
                }
            }
        }
        centers = new_centers;
    }
    labels
}

/// Symmetric Jacobi eigen-decomposition. Returns `(eigenvalues, V)`
/// such that `A V = V diag(eigenvalues)` and `V` is orthogonal.
pub fn jacobi_symmetric_eig(
    a: &Array2<f64>,
    max_sweeps: usize,
    tol: f64,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(OptimizrError::InvalidInput("matrix must be square".into()));
    }
    let mut m = a.clone();
    let mut v = Array2::<f64>::eye(n);

    for _ in 0..max_sweeps {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[[p, q]] * m[[p, q]];
            }
        }
        if off.sqrt() < tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[[p, q]];
                if apq.abs() < 1e-16 {
                    continue;
                }
                let app = m[[p, p]];
                let aqq = m[[q, q]];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                m[[p, p]] = app - t * apq;
                m[[q, q]] = aqq + t * apq;
                m[[p, q]] = 0.0;
                m[[q, p]] = 0.0;
                for i in 0..n {
                    if i != p && i != q {
                        let aip = m[[i, p]];
                        let aiq = m[[i, q]];
                        m[[i, p]] = c * aip - s * aiq;
                        m[[p, i]] = m[[i, p]];
                        m[[i, q]] = s * aip + c * aiq;
                        m[[q, i]] = m[[i, q]];
                    }
                }
                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
    }

    let eigvals = Array1::from_iter((0..n).map(|i| m[[i, i]]));
    Ok((eigvals, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_blocks_recovered() {
        // 6 nodes, two cliques {0,1,2} and {3,4,5}, weak link 2-3.
        let w = array![
            [0.0, 1.0, 1.0, 0.05, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.05, 0.0, 0.0],
            [0.05, 0.0, 0.05, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ];
        let res = spectral_cluster(w.view(), 2, 100, 7).unwrap();
        // Members of each block should share their label.
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[1], res.labels[2]);
        assert_eq!(res.labels[3], res.labels[4]);
        assert_eq!(res.labels[4], res.labels[5]);
        assert_ne!(res.labels[0], res.labels[3]);
    }

    #[test]
    fn jacobi_diagonalises_symmetric() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let (eig, v) = jacobi_symmetric_eig(&a, 200, 1e-12).unwrap();
        let mut sorted: Vec<f64> = eig.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Eigenvalues of [[2,1],[1,3]]: (5 +- sqrt(5)) / 2
        let lo = (5.0 - 5.0_f64.sqrt()) / 2.0;
        let hi = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((sorted[0] - lo).abs() < 1e-9);
        assert!((sorted[1] - hi).abs() < 1e-9);
        // Orthogonality
        let vt_v = v.t().dot(&v);
        for i in 0..2 {
            for j in 0..2 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((vt_v[[i, j]] - target).abs() < 1e-9);
            }
        }
    }
}
