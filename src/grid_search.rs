///! Grid Search Optimization
///!
///! Exhaustive search over a parameter grid to find the global optimum.
///! While computationally expensive, grid search guarantees finding the
///! best solution within the discretized parameter space.
///!
///! # Algorithm
///!
///! For each parameter dimension:
///! 1. Create n_points evenly spaced between bounds
///! 2. Evaluate objective at all grid combinations
///! 3. Return parameters with best score
///!
///! Complexity: O(n_points^n_params × cost_per_eval)
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Bound;

/// Grid Search Result
#[pyclass]
#[derive(Clone)]
pub struct GridSearchResult {
    /// Best parameters found
    #[pyo3(get)]
    pub x: Vec<f64>,

    /// Best objective value
    #[pyo3(get)]
    pub fun: f64,

    /// Number of function evaluations
    #[pyo3(get)]
    pub nfev: usize,
}

#[pymethods]
impl GridSearchResult {
    fn __repr__(&self) -> String {
        format!(
            "GridSearchResult(fun={:.6}, nfev={}, nparams={})",
            self.fun,
            self.nfev,
            self.x.len()
        )
    }
}

/// Grid Search Optimizer
///
/// Exhaustively evaluates objective function at all points on a regular grid.
/// Best for small parameter spaces or when global optimum verification is needed.
///
/// # Arguments
///
/// * `objective_fn` - Python callable to maximize: f(x) -> float
/// * `bounds` - [(min, max), ...] bounds for each parameter
/// * `n_points` - Number of grid points per dimension
///
/// # Returns
///
/// GridSearchResult with best parameters and objective value
///
/// # Example
///
/// ```python
/// import optimizr
///
/// # Maximize simple function
/// def objective(x):
///     return -(x[0]**2 + x[1]**2)  # Peak at (0, 0)
///
/// result = optimizr.grid_search(
///     objective_fn=objective,
///     bounds=[(-5, 5), (-5, 5)],
///     n_points=50
/// )
///
/// print(f"Maximum: {result.fun} at {result.x}")
/// print(f"Evaluated {result.nfev} points")
/// ```
///
/// # Note
///
/// Grid search maximizes the objective (unlike DE which minimizes).
/// To minimize, negate your objective function.
#[pyfunction]
#[pyo3(signature = (objective_fn, bounds, n_points=10))]
pub fn grid_search(
    objective_fn: &Bound<'_, PyAny>,
    bounds: Vec<(f64, f64)>,
    n_points: usize,
) -> PyResult<GridSearchResult> {
    let n_params = bounds.len();

    if n_params == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "bounds cannot be empty",
        ));
    }

    if n_points < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_points must be at least 2",
        ));
    }

    // Generate grid points for each dimension
    let grids: Vec<Vec<f64>> = bounds
        .iter()
        .map(|(low, high)| {
            (0..n_points)
                .map(|i| low + (high - low) * i as f64 / (n_points - 1) as f64)
                .collect()
        })
        .collect();

    let mut best_params = vec![0.0; n_params];
    let mut best_score = f64::NEG_INFINITY;
    let mut nfev = 0;

    // Recursive grid traversal
    evaluate_grid_recursive(
        objective_fn,
        &grids,
        &mut Vec::new(),
        0,
        &mut best_params,
        &mut best_score,
        &mut nfev,
    )?;

    Ok(GridSearchResult {
        x: best_params,
        fun: best_score,
        nfev,
    })
}

/// Recursively evaluate all grid points
fn evaluate_grid_recursive(
    objective_fn: &Bound<'_, PyAny>,
    grids: &[Vec<f64>],
    current: &mut Vec<f64>,
    depth: usize,
    best_params: &mut Vec<f64>,
    best_score: &mut f64,
    nfev: &mut usize,
) -> PyResult<()> {
    if depth == grids.len() {
        // Reached a complete point, evaluate it
        let score = objective_fn.call1((current.clone(),))?.extract::<f64>()?;

        *nfev += 1;

        if score > *best_score {
            *best_score = score;
            *best_params = current.clone();
        }

        return Ok(());
    }

    // Iterate through values for current dimension
    for &value in &grids[depth] {
        current.push(value);
        evaluate_grid_recursive(
            objective_fn,
            grids,
            current,
            depth + 1,
            best_params,
            best_score,
            nfev,
        )?;
        current.pop();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_search_result() {
        let result = GridSearchResult {
            x: vec![1.0, 2.0],
            fun: 10.5,
            nfev: 100,
        };

        assert_eq!(result.x.len(), 2);
        assert_eq!(result.nfev, 100);
    }
}
