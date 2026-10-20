//! Dependency-free dense linear algebra for the handful of factorisations
//! optimiz-rs needs.
//!
//! # Why this exists
//!
//! optimiz-rs used `ndarray-linalg` with the `openblas-system` backend. That
//! made a *system* OpenBLAS a hard requirement of a published crate: on macOS
//! Homebrew's `openblas` is keg-only and drags in `gcc`, `gmp`, `isl`,
//! `libmpc` and `mpfr`, and on a bare Linux image there is no `libopenblas`
//! either. `cargo add optimiz-rs` therefore failed to link for anyone who had
//! not already installed a BLAS, which is the opposite of what an optimisation
//! library competing with scipy and jax can afford.
//!
//! Everything optimiz-rs actually used from LAPACK is in here, in safe Rust
//! with no build script and no system library:
//!
//! | need | implementation |
//! | --- | --- |
//! | `solve(A, b)` | LU with partial pivoting |
//! | `inv(A)`, `det(A)` | the same LU |
//! | `cholesky(A)` | standard Cholesky–Banachiewicz, SPD only |
//! | `svd(A)` | one-sided Jacobi, thin factors |
//! | `norm_l2` | direct |
//!
//! The traits mirror `ndarray-linalg`'s names and signatures, so call sites
//! only change which module they import.
//!
//! # Differences from `ndarray-linalg`
//!
//! * [`Svd::svd`] returns the **thin** factorisation: `U` is `m×k`, `Vt` is
//!   `k×n` with `k = min(m, n)`. `ndarray-linalg` returns the full square
//!   `U` and `Vt`. Thin is what the callers here want — they reconstruct
//!   `U · diag(s) · Vt`, which is only dimensionally valid for thin factors
//!   when the matrix is not square.
//! * Only `f64` is supported, which is all optimiz-rs uses.
//!
//! Enable the `lapack` feature to route these operations through
//! `ndarray-linalg` instead, for large problems where a tuned BLAS wins.

use ndarray::{Array1, Array2};
use std::fmt;

/// Which triangle a factor is stored in. Mirrors `ndarray_linalg::UPLO`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UPLO {
    Upper,
    Lower,
}

/// Failure modes of the factorisations in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinalgError {
    /// The matrix was not square where squareness is required.
    NotSquare { rows: usize, cols: usize },
    /// Shapes did not agree.
    ShapeMismatch { expected: usize, got: usize },
    /// A pivot was numerically zero: the matrix is singular to working
    /// precision.
    Singular { at: usize },
    /// Cholesky hit a non-positive pivot, so the input was not positive
    /// definite.
    NotPositiveDefinite { at: usize },
    /// A Jacobi sweep budget was exhausted without meeting the tolerance.
    NoConvergence { sweeps: usize },
    /// The matrix was empty.
    Empty,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSquare { rows, cols } => {
                write!(f, "matrix must be square, got {rows}x{cols}")
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            Self::Singular { at } => {
                write!(f, "matrix is singular: zero pivot at index {at}")
            }
            Self::NotPositiveDefinite { at } => write!(
                f,
                "matrix is not positive definite: non-positive pivot at index {at}"
            ),
            Self::NoConvergence { sweeps } => {
                write!(f, "no convergence after {sweeps} Jacobi sweeps")
            }
            Self::Empty => write!(f, "matrix is empty"),
        }
    }
}

impl std::error::Error for LinalgError {}

pub type Result<T> = std::result::Result<T, LinalgError>;

// ---------------------------------------------------------------------------
// LU with partial pivoting
// ---------------------------------------------------------------------------

/// `PA = LU`, packed into one matrix with the permutation and its sign.
struct Lu {
    /// Unit-lower `L` below the diagonal, `U` on and above it.
    lu: Array2<f64>,
    perm: Vec<usize>,
    /// `(-1)^{number of row swaps}`, needed for the determinant.
    sign: f64,
}

fn lu_factor(a: &Array2<f64>) -> Result<Lu> {
    let (m, n) = a.dim();
    if m != n {
        return Err(LinalgError::NotSquare { rows: m, cols: n });
    }
    if n == 0 {
        return Err(LinalgError::Empty);
    }

    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;

    for k in 0..n {
        // Partial pivoting: largest magnitude in the remaining column keeps
        // the multipliers bounded by 1 and the factorisation stable.
        let (piv, piv_val) = (k..n)
            .map(|i| (i, lu[[i, k]].abs()))
            .fold((k, 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });

        if piv_val < f64::EPSILON {
            return Err(LinalgError::Singular { at: k });
        }
        if piv != k {
            for j in 0..n {
                let t = lu[[k, j]];
                lu[[k, j]] = lu[[piv, j]];
                lu[[piv, j]] = t;
            }
            perm.swap(k, piv);
            sign = -sign;
        }

        let pivot = lu[[k, k]];
        for i in (k + 1)..n {
            let factor = lu[[i, k]] / pivot;
            lu[[i, k]] = factor;
            if factor != 0.0 {
                for j in (k + 1)..n {
                    lu[[i, j]] -= factor * lu[[k, j]];
                }
            }
        }
    }

    Ok(Lu { lu, perm, sign })
}

impl Lu {
    /// Forward then back substitution for a single right-hand side.
    fn solve_vec(&self, b: &Array1<f64>) -> Array1<f64> {
        let n = self.lu.nrows();
        let mut x = Array1::<f64>::zeros(n);

        // Apply the row permutation while copying.
        for i in 0..n {
            x[i] = b[self.perm[i]];
        }
        // L y = Pb, L unit lower.
        for i in 1..n {
            let mut acc = x[i];
            for j in 0..i {
                acc -= self.lu[[i, j]] * x[j];
            }
            x[i] = acc;
        }
        // U x = y.
        for i in (0..n).rev() {
            let mut acc = x[i];
            for j in (i + 1)..n {
                acc -= self.lu[[i, j]] * x[j];
            }
            x[i] = acc / self.lu[[i, i]];
        }
        x
    }

    fn det(&self) -> f64 {
        (0..self.lu.nrows()).fold(self.sign, |acc, i| acc * self.lu[[i, i]])
    }
}

// ---------------------------------------------------------------------------
// public free functions
// ---------------------------------------------------------------------------

/// Solve `A x = b` for one right-hand side.
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeMismatch {
            expected: a.nrows(),
            got: b.len(),
        });
    }
    Ok(lu_factor(a)?.solve_vec(b))
}

/// Inverse of `A`, by solving against each unit vector.
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let lu = lu_factor(a)?;
    let mut out = Array2::<f64>::zeros((n, n));
    let mut e = Array1::<f64>::zeros(n);
    for j in 0..n {
        e.fill(0.0);
        e[j] = 1.0;
        let col = lu.solve_vec(&e);
        for i in 0..n {
            out[[i, j]] = col[i];
        }
    }
    Ok(out)
}

/// Determinant of `A`, as the signed product of the LU pivots.
pub fn det(a: &Array2<f64>) -> Result<f64> {
    // A singular matrix has determinant zero; that is an answer, not an error.
    match lu_factor(a) {
        Ok(lu) => Ok(lu.det()),
        Err(LinalgError::Singular { .. }) => Ok(0.0),
        Err(e) => Err(e),
    }
}

/// Lower (or upper) Cholesky factor of a symmetric positive-definite `A`.
///
/// Returns `L` with `A = L Lᵀ` for [`UPLO::Lower`], or `Lᵀ` for
/// [`UPLO::Upper`].
pub fn cholesky(a: &Array2<f64>, uplo: UPLO) -> Result<Array2<f64>> {
    let (m, n) = a.dim();
    if m != n {
        return Err(LinalgError::NotSquare { rows: m, cols: n });
    }
    if n == 0 {
        return Err(LinalgError::Empty);
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut acc = a[[i, j]];
            for k in 0..j {
                acc -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if acc <= 0.0 {
                    return Err(LinalgError::NotPositiveDefinite { at: i });
                }
                l[[i, j]] = acc.sqrt();
            } else {
                l[[i, j]] = acc / l[[j, j]];
            }
        }
    }

    Ok(match uplo {
        UPLO::Lower => l,
        UPLO::Upper => l.reversed_axes().to_owned(),
    })
}

const JACOBI_SWEEPS: usize = 60;

/// Thin SVD `A = U · diag(s) · Vt` by one-sided Jacobi.
///
/// `U` is `m×k`, `s` has `k` entries sorted descending, `Vt` is `k×n`, with
/// `k = min(m, n)`. Components are only materialised when asked for.
///
/// One-sided Jacobi is used rather than Golub–Reinsch because it needs no
/// bidiagonalisation, is short enough to audit, and attains high relative
/// accuracy on the small-to-medium dense matrices optimiz-rs factorises.
pub fn svd(
    a: &Array2<f64>,
    want_u: bool,
    want_vt: bool,
) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>)> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err(LinalgError::Empty);
    }

    // One-sided Jacobi orthogonalises columns, so it needs at least as many
    // rows as columns. For a wide matrix factor the transpose and swap the
    // roles: A = (Aᵀ)ᵀ = (U' S V'ᵀ)ᵀ = V' S U'ᵀ.
    if m < n {
        let at = a.t().to_owned();
        let (u_t, s, vt_t) = svd(&at, want_vt, want_u)?;
        let u = vt_t.map(|vt| vt.reversed_axes().to_owned());
        let vt = u_t.map(|u| u.reversed_axes().to_owned());
        return Ok((u, s, vt));
    }

    let k = n; // m >= n here, so min(m, n) == n
    let mut w = a.clone(); // columns get rotated in place
    let mut v = Array2::<f64>::eye(n);

    let mut converged = false;
    for _sweep in 0..JACOBI_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let mut alpha = 0.0; // ⟨w_p, w_p⟩
                let mut beta = 0.0; // ⟨w_q, w_q⟩
                let mut gamma = 0.0; // ⟨w_p, w_q⟩
                for i in 0..m {
                    let wp = w[[i, p]];
                    let wq = w[[i, q]];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                if gamma == 0.0 {
                    continue;
                }
                // Track the largest relative inner product this sweep.
                let scale = (alpha * beta).sqrt();
                if scale > 0.0 {
                    off = off.max(gamma.abs() / scale);
                }
                if scale == 0.0 || gamma.abs() <= f64::EPSILON * scale {
                    continue;
                }

                // Jacobi rotation annihilating ⟨w_p, w_q⟩. The t formula is
                // the numerically stable branch (no cancellation for small
                // zeta, no overflow for large).
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..m {
                    let wp = w[[i, p]];
                    let wq = w[[i, q]];
                    w[[i, p]] = c * wp - s * wq;
                    w[[i, q]] = s * wp + c * wq;
                }
                for i in 0..n {
                    let vp = v[[i, p]];
                    let vq = v[[i, q]];
                    v[[i, p]] = c * vp - s * vq;
                    v[[i, q]] = s * vp + c * vq;
                }
            }
        }
        if off <= 1e-15 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence {
            sweeps: JACOBI_SWEEPS,
        });
    }

    // Singular values are the column norms of the orthogonalised W.
    let mut sigma: Vec<(f64, usize)> = (0..n)
        .map(|j| {
            let mut acc = 0.0;
            for i in 0..m {
                acc += w[[i, j]] * w[[i, j]];
            }
            (acc.sqrt(), j)
        })
        .collect();
    // Descending, matching LAPACK's convention.
    sigma.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let s = Array1::from_iter(sigma.iter().map(|&(val, _)| val));

    let u = want_u.then(|| {
        let mut u = Array2::<f64>::zeros((m, k));
        for (new_j, &(val, old_j)) in sigma.iter().enumerate() {
            if val > 0.0 {
                for i in 0..m {
                    u[[i, new_j]] = w[[i, old_j]] / val;
                }
            }
            // A zero singular value leaves the column at zero: the
            // corresponding direction is not determined by A.
        }
        u
    });

    let vt = want_vt.then(|| {
        let mut vt = Array2::<f64>::zeros((k, n));
        for (new_j, &(_, old_j)) in sigma.iter().enumerate() {
            for i in 0..n {
                vt[[new_j, i]] = v[[i, old_j]];
            }
        }
        vt
    });

    Ok((u, s, vt))
}

// ---------------------------------------------------------------------------
// ndarray-linalg-shaped traits, so call sites only swap an import
// ---------------------------------------------------------------------------

/// Solve `A x = b`.
pub trait Solve {
    fn solve(&self, b: &Array1<f64>) -> Result<Array1<f64>>;
}

impl Solve for Array2<f64> {
    fn solve(&self, b: &Array1<f64>) -> Result<Array1<f64>> {
        solve(self, b)
    }
}

/// Matrix inverse.
pub trait Inverse {
    fn inv(&self) -> Result<Array2<f64>>;
}

impl Inverse for Array2<f64> {
    fn inv(&self) -> Result<Array2<f64>> {
        inv(self)
    }
}

/// Determinant.
pub trait Determinant {
    fn det(&self) -> Result<f64>;
}

impl Determinant for Array2<f64> {
    fn det(&self) -> Result<f64> {
        det(self)
    }
}

/// Cholesky factorisation.
pub trait Cholesky {
    fn cholesky(&self, uplo: UPLO) -> Result<Array2<f64>>;
}

impl Cholesky for Array2<f64> {
    fn cholesky(&self, uplo: UPLO) -> Result<Array2<f64>> {
        cholesky(self, uplo)
    }
}

/// Thin singular value decomposition.
pub trait Svd {
    #[allow(clippy::type_complexity)]
    fn svd(
        &self,
        want_u: bool,
        want_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>)>;
}

impl Svd for Array2<f64> {
    fn svd(
        &self,
        want_u: bool,
        want_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>)> {
        svd(self, want_u, want_vt)
    }
}

/// Euclidean norm for vectors, Frobenius norm for matrices.
pub trait Norm {
    fn norm_l2(&self) -> f64;
}

impl Norm for Array1<f64> {
    fn norm_l2(&self) -> f64 {
        self.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

impl Norm for Array2<f64> {
    fn norm_l2(&self) -> f64 {
        self.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn solve_matches_hand_computed() {
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![3.0, 5.0];
        let x = solve(&a, &b).unwrap();
        // 2x + y = 3, x + 3y = 5  =>  x = 4/5, y = 7/5
        assert!((x[0] - 0.8).abs() < 1e-12, "{x:?}");
        assert!((x[1] - 1.4).abs() < 1e-12, "{x:?}");
    }

    #[test]
    fn solve_needs_pivoting() {
        // A zero leading pivot: only partial pivoting gets this right.
        let a = array![[0.0, 1.0], [1.0, 0.0]];
        let b = array![2.0, 3.0];
        let x = solve(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12, "{x:?}");
        assert!((x[1] - 2.0).abs() < 1e-12, "{x:?}");
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        let a = array![[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]];
        let ai = inv(&a).unwrap();
        assert!(max_abs_diff(&a.dot(&ai), &Array2::eye(3)) < 1e-10);
        assert!(max_abs_diff(&ai.dot(&a), &Array2::eye(3)) < 1e-10);
    }

    #[test]
    fn determinant_known_values() {
        assert!((det(&array![[1.0, 2.0], [3.0, 4.0]]).unwrap() + 2.0).abs() < 1e-12);
        assert!((det(&Array2::<f64>::eye(5)).unwrap() - 1.0).abs() < 1e-12);
        // Singular is an answer, not an error.
        assert_eq!(det(&array![[1.0, 2.0], [2.0, 4.0]]).unwrap(), 0.0);
    }

    #[test]
    fn cholesky_reconstructs_and_rejects_indefinite() {
        let a = array![[4.0, 2.0, 0.6], [2.0, 5.0, 1.2], [0.6, 1.2, 3.0]];
        let l = cholesky(&a, UPLO::Lower).unwrap();
        assert!(max_abs_diff(&l.dot(&l.t()), &a) < 1e-10);
        // Lower factor really is lower triangular.
        assert_eq!(l[[0, 1]], 0.0);
        assert_eq!(l[[0, 2]], 0.0);

        let upper = cholesky(&a, UPLO::Upper).unwrap();
        assert!(max_abs_diff(&upper.t().dot(&upper), &a) < 1e-10);

        let indefinite = array![[1.0, 2.0], [2.0, 1.0]];
        assert!(matches!(
            cholesky(&indefinite, UPLO::Lower),
            Err(LinalgError::NotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn svd_reconstructs_square() {
        let a = array![[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0], [1.0, 1.0, 4.0]];
        let (u, s, vt) = svd(&a, true, true).unwrap();
        let u = u.unwrap();
        let vt = vt.unwrap();
        let recon = u.dot(&Array2::from_diag(&s)).dot(&vt);
        assert!(max_abs_diff(&recon, &a) < 1e-10, "recon={recon:?}");
        // Descending order.
        for i in 1..s.len() {
            assert!(s[i - 1] >= s[i] - 1e-12);
        }
    }

    #[test]
    fn svd_reconstructs_tall_and_wide() {
        let tall = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (u, s, vt) = svd(&tall, true, true).unwrap();
        let recon = u.unwrap().dot(&Array2::from_diag(&s)).dot(&vt.unwrap());
        assert!(max_abs_diff(&recon, &tall) < 1e-10);

        let wide = tall.t().to_owned();
        let (u2, s2, vt2) = svd(&wide, true, true).unwrap();
        let recon2 = u2.unwrap().dot(&Array2::from_diag(&s2)).dot(&vt2.unwrap());
        assert!(max_abs_diff(&recon2, &wide) < 1e-10);
        // Same spectrum either way round.
        for (x, y) in s.iter().zip(s2.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
    }

    #[test]
    fn svd_singular_values_of_a_known_matrix() {
        // diag(3, 1) rotated: singular values must stay {3, 1}.
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let rot = array![[c, -c], [c, c]];
        let a = rot.dot(&array![[3.0, 0.0], [0.0, 1.0]]).dot(&rot.t());
        let (_, s, _) = svd(&a, false, false).unwrap();
        assert!((s[0] - 3.0).abs() < 1e-10, "{s:?}");
        assert!((s[1] - 1.0).abs() < 1e-10, "{s:?}");
    }

    #[test]
    fn svd_handles_rank_deficiency() {
        // Rank 1: second singular value is zero.
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let (u, s, vt) = svd(&a, true, true).unwrap();
        assert!(s[1].abs() < 1e-12, "{s:?}");
        let recon = u.unwrap().dot(&Array2::from_diag(&s)).dot(&vt.unwrap());
        assert!(max_abs_diff(&recon, &a) < 1e-10);
    }

    #[test]
    fn svd_factors_are_orthonormal() {
        let a = array![[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]];
        let (u, _, vt) = svd(&a, true, true).unwrap();
        let u = u.unwrap();
        let vt = vt.unwrap();
        assert!(max_abs_diff(&u.t().dot(&u), &Array2::eye(3)) < 1e-10);
        assert!(max_abs_diff(&vt.dot(&vt.t()), &Array2::eye(3)) < 1e-10);
    }

    #[test]
    fn norms() {
        assert!((array![3.0, 4.0].norm_l2() - 5.0).abs() < 1e-12);
        assert!((array![[3.0, 0.0], [0.0, 4.0]].norm_l2() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn errors_are_typed_not_panics() {
        let rect = Array2::<f64>::zeros((2, 3));
        assert!(matches!(
            inv(&rect),
            Err(LinalgError::NotSquare { rows: 2, cols: 3 })
        ));
        let singular = array![[1.0, 1.0], [1.0, 1.0]];
        assert!(matches!(
            solve(&singular, &array![1.0, 1.0]),
            Err(LinalgError::Singular { .. })
        ));
        assert!(matches!(
            solve(&Array2::<f64>::eye(2), &array![1.0, 2.0, 3.0]),
            Err(LinalgError::ShapeMismatch { expected: 2, got: 3 })
        ));
    }
}
