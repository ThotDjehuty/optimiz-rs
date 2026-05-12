//! Python bindings for topological data analysis.
//!
//! Exposes Vietoris--Rips filtration construction, persistent homology,
//! and bottleneck distance between persistence diagrams.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::bottleneck::bottleneck_distance as rs_bottleneck_distance;
use super::persistent_homology::{
    persistent_homology as rs_persistent_homology,
    vietoris_rips_filtration as rs_vietoris_rips_filtration, PersistencePair,
};

fn diagram_to_pylist(py: Python<'_>, pairs: &[PersistencePair]) -> PyResult<PyObject> {
    let list = pyo3::types::PyList::empty_bound(py);
    for p in pairs {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("dim", p.dim)?;
        d.set_item("birth", p.birth)?;
        d.set_item("death", p.death)?;
        list.append(d)?;
    }
    Ok(list.into())
}

fn pylist_to_diagram(pairs: Vec<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Vec<PersistencePair>> {
    let mut out = Vec::with_capacity(pairs.len());
    for d in pairs {
        let dim: usize = d
            .get_item("dim")?
            .ok_or_else(|| PyValueError::new_err("missing key 'dim'"))?
            .extract()?;
        let birth: f64 = d
            .get_item("birth")?
            .ok_or_else(|| PyValueError::new_err("missing key 'birth'"))?
            .extract()?;
        let death: f64 = d
            .get_item("death")?
            .ok_or_else(|| PyValueError::new_err("missing key 'death'"))?
            .extract()?;
        out.push(PersistencePair { dim, birth, death });
    }
    Ok(out)
}

/// Build the Vietoris--Rips filtration up to ``max_dim`` and scale
/// ``max_eps``. Returns a list of simplices as dicts
/// ``{"vertices": [..], "filtration": float, "dim": int}``.
#[pyfunction]
#[pyo3(signature = (points, max_dim=1, max_eps=1.0))]
fn vietoris_rips_filtration(
    py: Python<'_>,
    points: Vec<Vec<f64>>,
    max_dim: usize,
    max_eps: f64,
) -> PyResult<PyObject> {
    let simplices = rs_vietoris_rips_filtration(&points, max_dim, max_eps)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    let list = pyo3::types::PyList::empty_bound(py);
    for s in simplices {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("vertices", s.vertices.clone())?;
        d.set_item("filtration", s.filtration)?;
        d.set_item("dim", s.dim())?;
        list.append(d)?;
    }
    Ok(list.into())
}

/// Compute the Vietoris--Rips persistence diagram. Returns a list of
/// dicts ``[{"dim": int, "birth": float, "death": float}, ...]``.
#[pyfunction]
#[pyo3(signature = (points, max_dim=1, max_eps=1.0))]
fn persistent_homology(
    py: Python<'_>,
    points: Vec<Vec<f64>>,
    max_dim: usize,
    max_eps: f64,
) -> PyResult<PyObject> {
    let diag = rs_persistent_homology(&points, max_dim, max_eps)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    diagram_to_pylist(py, &diag.pairs)
}

/// Bottleneck distance between two persistence diagrams. Each diagram
/// is a list of dicts ``{"dim": int, "birth": float, "death": float}``.
#[pyfunction]
#[pyo3(signature = (diagram_a, diagram_b))]
fn bottleneck_distance(
    diagram_a: Vec<Bound<'_, pyo3::types::PyDict>>,
    diagram_b: Vec<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<f64> {
    let a = pylist_to_diagram(diagram_a)?;
    let b = pylist_to_diagram(diagram_b)?;
    rs_bottleneck_distance(&a, &b).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

/// Register all topology functions with the Python module.
pub fn register_python_functions(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vietoris_rips_filtration, m)?)?;
    m.add_function(wrap_pyfunction!(persistent_homology, m)?)?;
    m.add_function(wrap_pyfunction!(bottleneck_distance, m)?)?;
    Ok(())
}
