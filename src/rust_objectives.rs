///! Rust-native objective functions for GIL-free parallelization

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

pub trait RustObjective: Send + Sync {
    fn evaluate(&self, x: &[f64]) -> f64;
    fn dimension(&self) -> Option<usize> { None }
    fn global_optimum(&self) -> Option<f64> { None }
    fn optimal_solution(&self) -> Option<Vec<f64>> { None }
}

#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
#[derive(Clone)]
pub struct Sphere { pub dim: usize }

impl Sphere {
    pub fn new(dim: usize) -> Self { Sphere { dim } }
    #[cfg(feature = "python-bindings")]
    pub fn __call__(&self, x: Vec<f64>) -> f64 { self.evaluate(&x) }
}

impl RustObjective for Sphere {
    fn evaluate(&self, x: &[f64]) -> f64 { x.iter().map(|xi| xi * xi).sum() }
    fn dimension(&self) -> Option<usize> { Some(self.dim) }
    fn global_optimum(&self) -> Option<f64> { Some(0.0) }
    fn optimal_solution(&self) -> Option<Vec<f64>> { Some(vec![0.0; self.dim]) }
}

#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
#[derive(Clone)]
pub struct Rosenbrock { pub dim: usize }

impl Rosenbrock {
    pub fn new(dim: usize) -> Self { Rosenbrock { dim } }
    #[cfg(feature = "python-bindings")]
    pub fn __call__(&self, x: Vec<f64>) -> f64 { self.evaluate(&x) }
}

impl RustObjective for Rosenbrock {
    fn evaluate(&self, x: &[f64]) -> f64 {
        (0..x.len() - 1).map(|i| {
            let t1 = x[i + 1] - x[i] * x[i];
            let t2 = 1.0 - x[i];
            100.0 * t1 * t1 + t2 * t2
        }).sum()
    }
    fn dimension(&self) -> Option<usize> { Some(self.dim) }
    fn global_optimum(&self) -> Option<f64> { Some(0.0) }
    fn optimal_solution(&self) -> Option<Vec<f64>> { Some(vec![1.0; self.dim]) }
}

#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
#[derive(Clone)]
pub struct Rastrigin { pub dim: usize }

impl Rastrigin {
    pub fn new(dim: usize) -> Self { Rastrigin { dim } }
    #[cfg(feature = "python-bindings")]
    pub fn __call__(&self, x: Vec<f64>) -> f64 { self.evaluate(&x) }
}

impl RustObjective for Rastrigin {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let pi = std::f64::consts::PI;
        10.0 * n + x.iter().map(|xi| xi * xi - 10.0 * (2.0 * pi * xi).cos()).sum::<f64>()
    }
    fn dimension(&self) -> Option<usize> { Some(self.dim) }
    fn global_optimum(&self) -> Option<f64> { Some(0.0) }
    fn optimal_solution(&self) -> Option<Vec<f64>> { Some(vec![0.0; self.dim]) }
}

#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
#[derive(Clone)]
pub struct Ackley { pub dim: usize }

impl Ackley {
    pub fn new(dim: usize) -> Self { Ackley { dim } }
    #[cfg(feature = "python-bindings")]
    pub fn __call__(&self, x: Vec<f64>) -> f64 { self.evaluate(&x) }
}

impl RustObjective for Ackley {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let pi = std::f64::consts::PI;
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let sum_cos = x.iter().map(|xi| (2.0 * pi * xi).cos()).sum::<f64>();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
    }
    fn dimension(&self) -> Option<usize> { Some(self.dim) }
    fn global_optimum(&self) -> Option<f64> { Some(0.0) }
    fn optimal_solution(&self) -> Option<Vec<f64>> { Some(vec![0.0; self.dim]) }
}

#[cfg_attr(feature = "python-bindings", pyo3::pyclass)]
#[derive(Clone)]
pub struct Griewank { pub dim: usize }

impl Griewank {
    pub fn new(dim: usize) -> Self { Griewank { dim } }
    #[cfg(feature = "python-bindings")]
    pub fn __call__(&self, x: Vec<f64>) -> f64 { self.evaluate(&x) }
}

impl RustObjective for Griewank {
    fn evaluate(&self, x: &[f64]) -> f64 {
        let sum_sq = x.iter().map(|xi| xi * xi).sum::<f64>();
        let prod_cos = x.iter().enumerate().map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos()).product::<f64>();
        1.0 + sum_sq / 4000.0 - prod_cos
    }
    fn dimension(&self) -> Option<usize> { Some(self.dim) }
    fn global_optimum(&self) -> Option<f64> { Some(0.0) }
    fn optimal_solution(&self) -> Option<Vec<f64>> { Some(vec![0.0; self.dim]) }
}

#[cfg(feature = "python-bindings")]
pub fn register_benchmark_functions(m: &pyo3::Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<Sphere>()?;
    m.add_class::<Rosenbrock>()?;
    m.add_class::<Rastrigin>()?;
    m.add_class::<Ackley>()?;
    m.add_class::<Griewank>()?;
    Ok(())
}
