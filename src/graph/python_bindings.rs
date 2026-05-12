//! Python bindings for the `graph` module.
//!
//! Exposes graph Laplacian operators and the Ng--Jordan--Weiss
//! spectral clustering routine.

use ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::laplacian::{combinatorial_laplacian, normalised_laplacian, random_walk_laplacian};
use super::spectral_clustering::spectral_cluster;

fn vec_of_vec_to_array2(w: &[Vec<f64>]) -> PyResult<Array2<f64>> {
    let n = w.len();
    if n == 0 {
        return Err(PyValueError::new_err(
            "weight matrix must be non-empty",
        ));
    }
    let m = w[0].len();
    if m != n {
        return Err(PyValueError::new_err(
            "weight matrix must be square",
        ));
    }
    let mut a = Array2::<f64>::zeros((n, n));
    for (i, row) in w.iter().enumerate() {
        if row.len() != n {
            return Err(PyValueError::new_err(
                "weight matrix must be square",
            ));
        }
        for (j, &v) in row.iter().enumerate() {
            a[[i, j]] = v;
        }
    }
    Ok(a)
}

fn array2_to_vec_of_vec(a: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = a.nrows();
    let m = a.ncols();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(m);
        for j in 0..m {
            row.push(a[[i, j]]);
        }
        out.push(row);
    }
    out
}

/// Combinatorial Laplacian `L = D - W`.
#[pyfunction]
#[pyo3(signature = (w))]
fn combinatorial_laplacian_py(w: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let arr = vec_of_vec_to_array2(&w)?;
    let l = combinatorial_laplacian(arr.view())
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(array2_to_vec_of_vec(&l))
}

/// Symmetric normalised Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}`.
#[pyfunction]
#[pyo3(signature = (w))]
fn normalised_laplacian_py(w: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let arr = vec_of_vec_to_array2(&w)?;
    let l = normalised_laplacian(arr.view())
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(array2_to_vec_of_vec(&l))
}

/// Random-walk Laplacian `L_rw = I - D^{-1} W`.
#[pyfunction]
#[pyo3(signature = (w))]
fn random_walk_laplacian_py(w: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let arr = vec_of_vec_to_array2(&w)?;
    let l = random_walk_laplacian(arr.view())
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    Ok(array2_to_vec_of_vec(&l))
}

/// Spectral clustering (Ng--Jordan--Weiss) on a non-negative symmetric
/// similarity matrix.
///
/// Returns a dict with keys `labels`, `eigenvalues`, `fiedler_value`.
#[pyfunction]
#[pyo3(signature = (w, k, n_kmeans_iter=100, seed=0))]
fn spectral_cluster_py(
    py: Python<'_>,
    w: Vec<Vec<f64>>,
    k: usize,
    n_kmeans_iter: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let arr = vec_of_vec_to_array2(&w)?;
    let result = spectral_cluster(arr.view(), k, n_kmeans_iter, seed)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("labels", result.labels.clone())?;
    dict.set_item("eigenvalues", result.eigenvalues.clone())?;
    dict.set_item("fiedler_value", result.fiedler_value)?;
    Ok(dict.into())
}

/// Register all graph functions with the Python module.
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(combinatorial_laplacian_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalised_laplacian_py, m)?)?;
    m.add_function(wrap_pyfunction!(random_walk_laplacian_py, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_cluster_py, m)?)?;
    Ok(())
}
