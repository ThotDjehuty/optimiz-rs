//! Matrix Riccati Backward ODE Solver
//! ==================================
//!
//! Solves the coupled backward matrix Riccati system
//!
//! ```text
//!   dA/dt = -2 A M A + Q,                 A(T) = A_T  (symmetric)
//!   dB/dt = -2 A M B + N^T B,             B(T) = B_T
//!   dC/dt = -2 A M C,                     C(T) = C_T
//! ```
//!
//! using a classical fourth-order Runge--Kutta scheme integrated backward
//! in time with optional sub-stepping for stiff problems.
//!
//! All matrices live in `R^{d x d}`; `Q` is symmetric positive
//! semi-definite, `M` symmetric positive definite. The solver performs no
//! linear-algebra inversion: callers that have a pre-computed `M^{-1}` can
//! pass it directly through `m_matrix` since only the products `A M A`
//! appear in the right-hand side.
//!
//! # Reference
//!
//! Bismut (1976), *Linear-Quadratic Optimal Stochastic Control with Random
//! Coefficients*, SIAM Journal on Control and Optimization 14(3).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::core::{OptimizrError, Result};

/// Configuration for the backward Riccati integrator.
#[derive(Debug, Clone, Copy)]
pub struct RiccatiConfig {
    /// Number of stored time points (including both endpoints).
    pub n_steps: usize,
    /// Number of internal RK4 sub-steps between two stored points.
    pub n_substeps: usize,
}

impl Default for RiccatiConfig {
    fn default() -> Self {
        Self {
            n_steps: 200,
            n_substeps: 1,
        }
    }
}

/// Result of a backward Riccati integration.
///
/// All vectors are ordered forward in time, i.e. `t_grid[0] = 0` and
/// `t_grid[n_steps - 1] = T`.
#[derive(Debug, Clone)]
pub struct RiccatiResult {
    pub t_grid: Vec<f64>,
    pub a: Vec<Array2<f64>>,
    pub b: Vec<Array2<f64>>,
    pub c: Vec<Array1<f64>>,
}

#[inline]
fn rhs(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array1<f64>,
    m: &Array2<f64>,
    q: &Array2<f64>,
    n_t: &Array2<f64>, // N^T precomputed
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let am = a.dot(m);
    let ama = am.dot(a);
    let amb = am.dot(b);
    let amc = am.dot(c);

    let da = -2.0 * &ama + q;
    let db = -2.0 * &amb + n_t.dot(b);
    let dc = -2.0 * amc;
    (da, db, dc)
}

/// Backward integration of the matrix Riccati system.
///
/// # Arguments
///
/// * `m_matrix`     -- `M` (symmetric positive definite) of shape `(d, d)`.
/// * `q`            -- `Q` (symmetric positive semi-definite) of shape `(d, d)`.
/// * `n`            -- `N` of shape `(d, d)`.
/// * `a_terminal`   -- terminal `A(T)` of shape `(d, d)`.
/// * `b_terminal`   -- terminal `B(T)` of shape `(d, d)`.
/// * `c_terminal`   -- terminal `C(T)` of shape `(d,)`.
/// * `t_horizon`    -- final time `T > 0`.
/// * `config`       -- discretisation parameters.
pub fn solve_matrix_riccati(
    m_matrix: ArrayView2<f64>,
    q: ArrayView2<f64>,
    n: ArrayView2<f64>,
    a_terminal: ArrayView2<f64>,
    b_terminal: ArrayView2<f64>,
    c_terminal: ArrayView1<f64>,
    t_horizon: f64,
    config: RiccatiConfig,
) -> Result<RiccatiResult> {
    if t_horizon <= 0.0 {
        return Err(OptimizrError::InvalidParameter(
            "t_horizon must be strictly positive".into(),
        ));
    }
    if config.n_steps < 2 {
        return Err(OptimizrError::InvalidParameter(
            "n_steps must be at least 2".into(),
        ));
    }
    if config.n_substeps == 0 {
        return Err(OptimizrError::InvalidParameter(
            "n_substeps must be at least 1".into(),
        ));
    }

    let d = a_terminal.nrows();
    let check_square = |x: ArrayView2<f64>, name: &str| -> Result<()> {
        if x.nrows() != d || x.ncols() != d {
            return Err(OptimizrError::DimensionMismatch {
                expected: d,
                actual: x.nrows(),
            });
        }
        let _ = name;
        Ok(())
    };
    check_square(m_matrix, "m_matrix")?;
    check_square(q, "q")?;
    check_square(n, "n")?;
    check_square(b_terminal, "b_terminal")?;
    if c_terminal.len() != d {
        return Err(OptimizrError::DimensionMismatch {
            expected: d,
            actual: c_terminal.len(),
        });
    }

    let m = m_matrix.to_owned();
    let q_o = q.to_owned();
    let n_t = n.t().to_owned();

    let n_steps = config.n_steps;
    let n_sub = config.n_substeps;
    let dt = t_horizon / ((n_steps - 1) as f64);
    let h = dt / (n_sub as f64);

    let mut t_grid = Vec::with_capacity(n_steps);
    for k in 0..n_steps {
        t_grid.push((k as f64) * dt);
    }

    let mut a_vec: Vec<Array2<f64>> = vec![Array2::<f64>::zeros((d, d)); n_steps];
    let mut b_vec: Vec<Array2<f64>> = vec![Array2::<f64>::zeros((d, d)); n_steps];
    let mut c_vec: Vec<Array1<f64>> = vec![Array1::<f64>::zeros(d); n_steps];

    a_vec[n_steps - 1] = a_terminal.to_owned();
    b_vec[n_steps - 1] = b_terminal.to_owned();
    c_vec[n_steps - 1] = c_terminal.to_owned();

    // Backward integration: from index k to k-1 (time decreases).
    // The continuous system is integrated with step -h (RK4).
    for k in (1..n_steps).rev() {
        let mut a = a_vec[k].clone();
        let mut b = b_vec[k].clone();
        let mut c = c_vec[k].clone();

        for _ in 0..n_sub {
            let (k1a, k1b, k1c) = rhs(&a, &b, &c, &m, &q_o, &n_t);
            let a2 = &a - 0.5 * h * &k1a;
            let b2 = &b - 0.5 * h * &k1b;
            let c2 = &c - 0.5 * h * &k1c;
            let (k2a, k2b, k2c) = rhs(&a2, &b2, &c2, &m, &q_o, &n_t);
            let a3 = &a - 0.5 * h * &k2a;
            let b3 = &b - 0.5 * h * &k2b;
            let c3 = &c - 0.5 * h * &k2c;
            let (k3a, k3b, k3c) = rhs(&a3, &b3, &c3, &m, &q_o, &n_t);
            let a4 = &a - h * &k3a;
            let b4 = &b - h * &k3b;
            let c4 = &c - h * &k3c;
            let (k4a, k4b, k4c) = rhs(&a4, &b4, &c4, &m, &q_o, &n_t);

            a = &a - (h / 6.0) * (&k1a + 2.0 * &k2a + 2.0 * &k3a + &k4a);
            b = &b - (h / 6.0) * (&k1b + 2.0 * &k2b + 2.0 * &k3b + &k4b);
            c = &c - (h / 6.0) * (&k1c + 2.0 * &k2c + 2.0 * &k3c + &k4c);
        }

        // Symmetrise A to preserve numerical symmetry over long horizons.
        let a_sym = 0.5 * (&a + &a.t());
        a_vec[k - 1] = a_sym;
        b_vec[k - 1] = b;
        c_vec[k - 1] = c;
    }

    // Final integrity check: no NaN/Inf in stored arrays.
    for arr in a_vec.iter() {
        if arr.iter().any(|x| !x.is_finite()) {
            return Err(OptimizrError::NumericalError(
                "non-finite value encountered in A trajectory".into(),
            ));
        }
    }
    let _ = Axis(0);
    Ok(RiccatiResult {
        t_grid,
        a: a_vec,
        b: b_vec,
        c: c_vec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Scalar (d=1) reference solution for A(T) = 0.
    /// The ODE  dA/dt = -2 M A^2 + Q  with terminal A(T)=0 admits
    ///     A(t) = -sqrt(Q / (2 M)) * tanh(sqrt(2 Q M) * (T - t)).
    #[test]
    fn scalar_riccati_matches_analytic() {
        let m = array![[1.5_f64]];
        let q = array![[0.8_f64]];
        let n = array![[0.0_f64]];
        let a_t = array![[0.0_f64]];
        let b_t = array![[0.0_f64]];
        let c_t = array![0.0_f64];

        let t_horizon = 3.0;
        let cfg = RiccatiConfig {
            n_steps: 2001,
            n_substeps: 4,
        };
        let res = solve_matrix_riccati(
            m.view(),
            q.view(),
            n.view(),
            a_t.view(),
            b_t.view(),
            c_t.view(),
            t_horizon,
            cfg,
        )
        .expect("solver");

        let coeff = -(q[[0, 0]] / (2.0 * m[[0, 0]])).sqrt();
        let alpha = (2.0 * q[[0, 0]] * m[[0, 0]]).sqrt();

        let mut max_err = 0.0_f64;
        for (i, &t) in res.t_grid.iter().enumerate() {
            let analytic = coeff * (alpha * (t_horizon - t)).tanh();
            let num = res.a[i][[0, 0]];
            max_err = max_err.max((analytic - num).abs());
        }
        assert!(
            max_err < 1e-5,
            "Riccati scalar L_inf error too large: {}",
            max_err
        );
    }

    #[test]
    fn rejects_invalid_inputs() {
        let m = array![[1.0_f64]];
        let q = array![[1.0_f64]];
        let n = array![[0.0_f64]];
        let a_t = array![[0.0_f64]];
        let b_t = array![[0.0_f64]];
        let c_t = array![0.0_f64];
        assert!(solve_matrix_riccati(
            m.view(),
            q.view(),
            n.view(),
            a_t.view(),
            b_t.view(),
            c_t.view(),
            -1.0,
            RiccatiConfig::default(),
        )
        .is_err());
    }
}
