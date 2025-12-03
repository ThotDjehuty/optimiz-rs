//! Refactored Differential Evolution with Parallel Support
//!
//! Strategy pattern for mutation operators and parallel fitness evaluation.

use crate::core::{Bounds, OptimizrError, Optimizer, Result};
use pyo3::prelude::*;
use rand::Rng;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Trait for mutation strategies
pub trait MutationStrategy: Send + Sync + Clone {
    fn mutate(
        &self,
        population: &[Vec<f64>],
        target_idx: usize,
        f: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64>;
    
    fn name(&self) -> &'static str;
}

/// DE/rand/1 strategy
#[derive(Clone, Debug)]
pub struct RandOne;

impl MutationStrategy for RandOne {
    fn mutate(
        &self,
        population: &[Vec<f64>],
        target_idx: usize,
        f: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let pop_size = population.len();
        let dim = population[0].len();
        
        // Select three distinct random individuals
        let mut indices = Vec::new();
        while indices.len() < 3 {
            let idx = rng.gen_range(0..pop_size);
            if idx != target_idx && !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        
        let [r1, r2, r3] = [indices[0], indices[1], indices[2]];
        
        // Mutant = r1 + F * (r2 - r3)
        (0..dim)
            .map(|d| population[r1][d] + f * (population[r2][d] - population[r3][d]))
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "DE/rand/1"
    }
}

/// DE/best/1 strategy
#[derive(Clone, Debug)]
pub struct BestOne {
    pub best_idx: usize,
}

impl MutationStrategy for BestOne {
    fn mutate(
        &self,
        population: &[Vec<f64>],
        target_idx: usize,
        f: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let pop_size = population.len();
        let dim = population[0].len();
        
        // Select two distinct random individuals
        let mut indices = Vec::new();
        while indices.len() < 2 {
            let idx = rng.gen_range(0..pop_size);
            if idx != target_idx && idx != self.best_idx && !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        
        let [r1, r2] = [indices[0], indices[1]];
        
        // Mutant = best + F * (r1 - r2)
        (0..dim)
            .map(|d| population[self.best_idx][d] + f * (population[r1][d] - population[r2][d]))
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "DE/best/1"
    }
}

/// DE/rand/2 strategy
#[derive(Clone, Debug)]
pub struct RandTwo;

impl MutationStrategy for RandTwo {
    fn mutate(
        &self,
        population: &[Vec<f64>],
        target_idx: usize,
        f: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let pop_size = population.len();
        let dim = population[0].len();
        
        // Select five distinct random individuals
        let mut indices = Vec::new();
        while indices.len() < 5 {
            let idx = rng.gen_range(0..pop_size);
            if idx != target_idx && !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        
        let [r1, r2, r3, r4, r5] = [indices[0], indices[1], indices[2], indices[3], indices[4]];
        
        // Mutant = r1 + F * (r2 - r3) + F * (r4 - r5)
        (0..dim)
            .map(|d| {
                population[r1][d]
                    + f * (population[r2][d] - population[r3][d])
                    + f * (population[r4][d] - population[r5][d])
            })
            .collect()
    }
    
    fn name(&self) -> &'static str {
        "DE/rand/2"
    }
}

/// Generic objective function
pub trait ObjectiveFunction: Send + Sync {
    fn evaluate(&self, x: &[f64]) -> f64;
}

/// Wrapper for Python callable
pub struct PyObjectiveFunction {
    func: Arc<Py<PyAny>>,
}

impl PyObjectiveFunction {
    pub fn new(func: Py<PyAny>) -> Self {
        Self {
            func: Arc::new(func),
        }
    }
}

impl ObjectiveFunction for PyObjectiveFunction {
    fn evaluate(&self, x: &[f64]) -> f64 {
        Python::with_gil(|py| {
            let args = (x.to_vec(),);
            self.func
                .call1(py, args)
                .and_then(|res| res.extract::<f64>(py))
                .unwrap_or(f64::INFINITY)
        })
    }
}

/// DE Configuration Builder
#[derive(Clone)]
pub struct DEConfig<M: MutationStrategy> {
    pub bounds: Bounds,
    pub pop_size: usize,
    pub max_generations: usize,
    pub mutation_factor: f64,
    pub crossover_rate: f64,
    pub tolerance: f64,
    pub strategy: M,
    pub use_parallel: bool,
}

pub struct DEConfigBuilder<M: MutationStrategy> {
    bounds: Bounds,
    pop_size: Option<usize>,
    max_generations: usize,
    mutation_factor: f64,
    crossover_rate: f64,
    tolerance: f64,
    strategy: Option<M>,
    use_parallel: bool,
}

impl<M: MutationStrategy> DEConfigBuilder<M> {
    pub fn new(bounds: Bounds) -> Self {
        Self {
            bounds,
            pop_size: None,
            max_generations: 1000,
            mutation_factor: 0.8,
            crossover_rate: 0.7,
            tolerance: 1e-6,
            strategy: None,
            use_parallel: cfg!(feature = "parallel"),
        }
    }
    
    pub fn pop_size(mut self, size: usize) -> Self {
        self.pop_size = Some(size);
        self
    }
    
    pub fn max_generations(mut self, gen: usize) -> Self {
        self.max_generations = gen;
        self
    }
    
    pub fn mutation_factor(mut self, f: f64) -> Self {
        self.mutation_factor = f;
        self
    }
    
    pub fn crossover_rate(mut self, cr: f64) -> Self {
        self.crossover_rate = cr;
        self
    }
    
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    
    pub fn strategy(mut self, strategy: M) -> Self {
        self.strategy = Some(strategy);
        self
    }
    
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.use_parallel = enabled && cfg!(feature = "parallel");
        self
    }
    
    pub fn build(self) -> Result<DEConfig<M>>
    where
        M: Default,
    {
        let dim = self.bounds.dim();
        let pop_size = self.pop_size.unwrap_or(10 * dim);
        
        if pop_size < 4 {
            return Err(OptimizrError::InvalidParameter(
                "pop_size must be at least 4".to_string(),
            ));
        }
        
        Ok(DEConfig {
            bounds: self.bounds,
            pop_size,
            max_generations: self.max_generations,
            mutation_factor: self.mutation_factor,
            crossover_rate: self.crossover_rate,
            tolerance: self.tolerance,
            strategy: self.strategy.unwrap_or_default(),
            use_parallel: self.use_parallel,
        })
    }
}

impl Default for RandOne {
    fn default() -> Self {
        RandOne
    }
}

impl Default for RandTwo {
    fn default() -> Self {
        RandTwo
    }
}

/// Refactored Differential Evolution
pub struct DifferentialEvolution<M: MutationStrategy, F: ObjectiveFunction> {
    pub config: DEConfig<M>,
    pub objective: F,
}

impl<M: MutationStrategy, F: ObjectiveFunction> DifferentialEvolution<M, F> {
    pub fn new(config: DEConfig<M>, objective: F) -> Self {
        Self { config, objective }
    }
    
    /// Initialize population
    fn initialize_population(&self, rng: &mut impl Rng) -> Vec<Vec<f64>> {
        (0..self.config.pop_size)
            .map(|_| self.config.bounds.sample(rng))
            .collect()
    }
    
    /// Evaluate fitness in parallel or sequential
    fn evaluate_population(&self, population: &[Vec<f64>]) -> Vec<f64> {
        #[cfg(feature = "parallel")]
        {
            if self.config.use_parallel {
                return population
                    .par_iter()
                    .map(|ind| self.objective.evaluate(ind))
                    .collect();
            }
        }
        
        // Sequential fallback
        population
            .iter()
            .map(|ind| self.objective.evaluate(ind))
            .collect()
    }
    
    /// Perform crossover
    fn crossover(&self, target: &[f64], mutant: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        let dim = target.len();
        let j_rand = rng.gen_range(0..dim);
        
        (0..dim)
            .map(|j| {
                if rng.gen::<f64>() < self.config.crossover_rate || j == j_rand {
                    mutant[j]
                } else {
                    target[j]
                }
            })
            .collect()
    }
    
    /// Run optimization
    pub fn optimize(&mut self) -> Result<(Vec<f64>, f64)> {
        let mut rng = rand::thread_rng();
        
        // Initialize
        let mut population = self.initialize_population(&mut rng);
        let mut fitness = self.evaluate_population(&population);
        
        let mut best_idx = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        let mut best_fitness = fitness[best_idx];
        
        // Evolution loop with functional style
        for _generation in 0..self.config.max_generations {
            let prev_best = best_fitness;
            
            // Generate trial vectors
            let trials: Vec<Vec<f64>> = (0..self.config.pop_size)
                .map(|i| {
                    // Note: BestOne strategy would need special handling here
                    // In practice, use a mutable reference pattern or Arc<Mutex<>>
                    
                    // Mutation
                    let mutant = self.config.strategy.mutate(
                        &population,
                        i,
                        self.config.mutation_factor,
                        &mut rng,
                    );
                    
                    // Crossover
                    let trial = self.crossover(&population[i], &mutant, &mut rng);
                    
                    // Clip to bounds
                    self.config.bounds.clip(&trial)
                })
                .collect();
            
            // Evaluate trials
            let trial_fitness = self.evaluate_population(&trials);
            
            // Selection
            for i in 0..self.config.pop_size {
                if trial_fitness[i] < fitness[i] {
                    population[i] = trials[i].clone();
                    fitness[i] = trial_fitness[i];
                    
                    if trial_fitness[i] < best_fitness {
                        best_idx = i;
                        best_fitness = trial_fitness[i];
                    }
                }
            }
            
            // Check convergence
            if (best_fitness - prev_best).abs() < self.config.tolerance {
                break;
            }
        }
        
        Ok((population[best_idx].clone(), best_fitness))
    }
}

impl<M: MutationStrategy + 'static, F: ObjectiveFunction + 'static> Optimizer
    for DifferentialEvolution<M, F>
{
    type Config = DEConfig<M>;
    type Output = (Vec<f64>, f64);
    
    fn optimize(&mut self) -> Result<Self::Output> {
        self.optimize()
    }
    
    fn best(&self) -> Result<Vec<f64>> {
        // Note: This requires re-optimization. In production, cache the best solution.
        Err(OptimizrError::ComputationError(
            "Call optimize() to get the best solution".to_string(),
        ))
    }
}

// Python bindings
#[pyclass]
#[derive(Clone, Debug)]
pub struct DEResult {
    #[pyo3(get)]
    pub best_solution: Vec<f64>,
    #[pyo3(get)]
    pub best_value: f64,
}

#[pyfunction]
#[pyo3(signature = (objective_fn, bounds, pop_size=None, max_generations=1000, mutation_factor=0.8, crossover_rate=0.7, strategy="rand1"))]
pub fn differential_evolution(
    objective_fn: Py<PyAny>,
    bounds: Vec<(f64, f64)>,
    pop_size: Option<usize>,
    max_generations: usize,
    mutation_factor: f64,
    crossover_rate: f64,
    strategy: &str,
) -> PyResult<DEResult> {
    let bounds = Bounds::new(bounds)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    
    let objective = PyObjectiveFunction::new(objective_fn);
    
    // Select strategy
    match strategy {
        "rand1" | "DE/rand/1" => {
            let mut builder = DEConfigBuilder::new(bounds)
                .max_generations(max_generations)
                .mutation_factor(mutation_factor)
                .crossover_rate(crossover_rate)
                .strategy(RandOne);
            
            if let Some(ps) = pop_size {
                builder = builder.pop_size(ps);
            }
            
            let config = builder
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            let mut optimizer = DifferentialEvolution::new(config, objective);
            let (best_solution, best_value) = optimizer
                .optimize()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            Ok(DEResult {
                best_solution,
                best_value,
            })
        }
        "rand2" | "DE/rand/2" => {
            let mut builder = DEConfigBuilder::new(bounds)
                .max_generations(max_generations)
                .mutation_factor(mutation_factor)
                .crossover_rate(crossover_rate)
                .strategy(RandTwo);
            
            if let Some(ps) = pop_size {
                builder = builder.pop_size(ps);
            }
            
            let config = builder
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            let mut optimizer = DifferentialEvolution::new(config, objective);
            let (best_solution, best_value) = optimizer
                .optimize()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            Ok(DEResult {
                best_solution,
                best_value,
            })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown strategy: {}. Use 'rand1', 'rand2', or 'best1'",
            strategy
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SphereFunction;
    
    impl ObjectiveFunction for SphereFunction {
        fn evaluate(&self, x: &[f64]) -> f64 {
            x.iter().map(|xi| xi.powi(2)).sum()
        }
    }

    #[test]
    fn test_de_builder() {
        let bounds = Bounds::new(vec![(-5.0, 5.0), (-5.0, 5.0)]).unwrap();
        let config = DEConfigBuilder::<RandOne>::new(bounds)
            .pop_size(40)
            .max_generations(100)
            .build()
            .unwrap();
        
        assert_eq!(config.pop_size, 40);
        assert_eq!(config.max_generations, 100);
    }

    #[test]
    fn test_de_optimization() {
        let bounds = Bounds::new(vec![(-5.0, 5.0), (-5.0, 5.0)]).unwrap();
        let config = DEConfigBuilder::<RandOne>::new(bounds)
            .pop_size(20)
            .max_generations(50)
            .build()
            .unwrap();
        
        let objective = SphereFunction;
        let mut optimizer = DifferentialEvolution::new(config, objective);
        
        let (_best, fitness) = optimizer.optimize().unwrap();
        assert!(fitness < 0.1); // Should converge close to 0
    }
}
