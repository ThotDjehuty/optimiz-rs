//! Python bindings for the `signatures` module.
//!
//! Exposes truncated path signatures, log-signatures, random reservoir
//! projections, the Salvi--Cass--Lyons signature kernel, and the
//! shuffle product / Chen concatenation utilities.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::log_signature::log_signature as rs_log_signature;
use super::path_signature::{path_signature as rs_path_signature, TruncatedSignature};
use super::random_signature::{
    random_signature as rs_random_signature, RandomSignatureConfig,
};
use super::signature_kernel::signature_kernel as rs_signature_kernel;
use super::utils::{concatenate_signatures as rs_concatenate, shuffle_product as rs_shuffle};

fn signature_to_dict(py: Python<'_>, sig: &TruncatedSignature) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("channels", sig.channels)?;
    dict.set_item("level", sig.level)?;
    dict.set_item("tensors", sig.tensors.clone())?;
    Ok(dict.into())
}

fn dict_to_signature(channels: usize, level: usize, tensors: Vec<Vec<f64>>) -> PyResult<TruncatedSignature> {
    if tensors.len() != level + 1 {
        return Err(PyValueError::new_err(format!(
            "tensors length {} does not match level + 1 = {}",
            tensors.len(),
            level + 1
        )));
    }
    for (k, t) in tensors.iter().enumerate() {
        let expected = channels.pow(k as u32);
        if t.len() != expected {
            return Err(PyValueError::new_err(format!(
                "tensors[{}] has length {}, expected channels^k = {}",
                k,
                t.len(),
                expected
            )));
        }
    }
    Ok(TruncatedSignature {
        channels,
        level,
        tensors,
    })
}

/// Truncated tensor signature of a piecewise-linear multivariate path.
#[pyfunction]
#[pyo3(signature = (path, level))]
fn path_signature(py: Python<'_>, path: Vec<Vec<f64>>, level: usize) -> PyResult<PyObject> {
    let sig = rs_path_signature(&path, level)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    signature_to_dict(py, &sig)
}

/// Truncated tensor log-signature of a piecewise-linear multivariate path.
#[pyfunction]
#[pyo3(signature = (path, level))]
fn path_log_signature(py: Python<'_>, path: Vec<Vec<f64>>, level: usize) -> PyResult<PyObject> {
    let sig = rs_path_signature(&path, level)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let log = rs_log_signature(&sig).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("channels", log.channels)?;
    dict.set_item("level", log.level)?;
    dict.set_item("tensors", log.tensors.clone())?;
    Ok(dict.into())
}

/// Random reservoir projection of the path signature.
#[pyfunction]
#[pyo3(signature = (path, reservoir_dim=32, seed=0, variance=1.0))]
fn random_signature(
    py: Python<'_>,
    path: Vec<Vec<f64>>,
    reservoir_dim: usize,
    seed: u64,
    variance: f64,
) -> PyResult<PyObject> {
    let cfg = RandomSignatureConfig {
        reservoir_dim,
        seed,
        variance,
    };
    let res = rs_random_signature(&path, &cfg)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("trajectory", res.trajectory.clone())?;
    Ok(dict.into())
}

/// Salvi--Cass--Lyons signature kernel between two multivariate paths.
#[pyfunction]
#[pyo3(signature = (x, y))]
fn signature_kernel(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
) -> PyResult<PyObject> {
    let res = rs_signature_kernel(&x, &y)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("value", res.value)?;
    dict.set_item("grid", res.grid.clone())?;
    Ok(dict.into())
}

/// Shuffle product of two words `u`, `v` over `{0, ..., d-1}`.
///
/// Returned as a list of `(word, multiplicity)` tuples to remain
/// hashable-key agnostic on the Python side.
#[pyfunction]
#[pyo3(signature = (u, v))]
fn shuffle_product(
    py: Python<'_>,
    u: Vec<usize>,
    v: Vec<usize>,
) -> PyResult<PyObject> {
    let map = rs_shuffle(&u, &v);
    let list = pyo3::types::PyList::empty_bound(py);
    for (word, mult) in map.into_iter() {
        let tup = pyo3::types::PyTuple::new_bound(py, &[word.into_py(py), mult.into_py(py)]);
        list.append(tup)?;
    }
    Ok(list.into())
}

/// Concatenate two truncated signatures via Chen's identity.
///
/// Inputs are passed as `(channels, level, tensors)` triples matching
/// the dict layout returned by :func:`path_signature`.
#[pyfunction]
#[pyo3(signature = (a_channels, a_level, a_tensors, b_channels, b_level, b_tensors))]
fn concatenate_signatures(
    py: Python<'_>,
    a_channels: usize,
    a_level: usize,
    a_tensors: Vec<Vec<f64>>,
    b_channels: usize,
    b_level: usize,
    b_tensors: Vec<Vec<f64>>,
) -> PyResult<PyObject> {
    if a_channels != b_channels || a_level != b_level {
        return Err(PyValueError::new_err(
            "signatures must share channels and level",
        ));
    }
    let a = dict_to_signature(a_channels, a_level, a_tensors)?;
    let b = dict_to_signature(b_channels, b_level, b_tensors)?;
    let out = rs_concatenate(&a, &b).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    signature_to_dict(py, &out)
}

/// Register all signatures functions with the Python module.
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(path_signature, m)?)?;
    m.add_function(wrap_pyfunction!(path_log_signature, m)?)?;
    m.add_function(wrap_pyfunction!(random_signature, m)?)?;
    m.add_function(wrap_pyfunction!(signature_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_product, m)?)?;
    m.add_function(wrap_pyfunction!(concatenate_signatures, m)?)?;
    Ok(())
}
