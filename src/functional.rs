//! Trait-based functional utilities for OptimizR
//!
//! This module provides functional programming utilities like composition,
//! monadic operations, and higher-order functions.

use crate::core::{OptimizrError, Result};

/// Function composition trait
pub trait Compose<A, B, C>: Sized {
    fn compose<G>(self, g: G) -> impl Fn(A) -> C
    where
        G: Fn(B) -> C,
        Self: Fn(A) -> B;
}

impl<F, A, B, C> Compose<A, B, C> for F
where
    F: Fn(A) -> B,
{
    fn compose<G>(self, g: G) -> impl Fn(A) -> C
    where
        G: Fn(B) -> C,
    {
        move |x| g(self(x))
    }
}

/// Monadic operations for Result
pub trait ResultExt<T> {
    /// Apply a function if Ok, short-circuit on Err
    fn and_then_log<F, U>(self, f: F, msg: &str) -> Result<U>
    where
        F: FnOnce(T) -> Result<U>;
    
    /// Map with context
    fn map_context<F, U>(self, f: F, ctx: &str) -> Result<U>
    where
        F: FnOnce(T) -> U;
}

impl<T> ResultExt<T> for Result<T> {
    fn and_then_log<F, U>(self, f: F, msg: &str) -> Result<U>
    where
        F: FnOnce(T) -> Result<U>,
    {
        match self {
            Ok(val) => f(val),
            Err(e) => {
                eprintln!("Error at {}: {:?}", msg, e);
                Err(e)
            }
        }
    }
    
    fn map_context<F, U>(self, f: F, ctx: &str) -> Result<U>
    where
        F: FnOnce(T) -> U,
    {
        self.map(f).map_err(|e| {
            OptimizrError::ComputationError(format!("{}: {}", ctx, e))
        })
    }
}

/// Retry logic for operations
pub fn retry<F, T>(mut f: F, max_attempts: usize) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let mut last_error = None;
    
    for _ in 0..max_attempts {
        match f() {
            Ok(val) => return Ok(val),
            Err(e) => last_error = Some(e),
        }
    }
    
    Err(last_error.unwrap_or_else(|| {
        OptimizrError::ComputationError("All retry attempts failed".to_string())
    }))
}

/// Memoization for expensive computations
pub struct Memoized<F, T>
where
    F: Fn(&[f64]) -> T,
{
    f: F,
    cache: std::sync::Mutex<std::collections::HashMap<Vec<ordered_float::OrderedFloat<f64>>, T>>,
}

impl<F, T> Memoized<F, T>
where
    F: Fn(&[f64]) -> T,
    T: Clone,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
    
    pub fn call(&self, x: &[f64]) -> T {
        let key: Vec<_> = x.iter().map(|&v| ordered_float::OrderedFloat(v)).collect();
        
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        
        let result = (self.f)(x);
        cache.insert(key, result.clone());
        result
    }
}

/// Lazy evaluation wrapper
pub struct Lazy<T, F>
where
    F: FnOnce() -> T,
{
    f: Option<F>,
    value: Option<T>,
}

impl<T, F> Lazy<T, F>
where
    F: FnOnce() -> T,
{
    pub fn new(f: F) -> Self {
        Self {
            f: Some(f),
            value: None,
        }
    }
    
    pub fn force(&mut self) -> &T {
        if self.value.is_none() {
            let f = self.f.take().unwrap();
            self.value = Some(f());
        }
        self.value.as_ref().unwrap()
    }
}

/// Piping operator - allows chaining operations
pub trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

impl<T> Pipe for T {}

/// Currying utilities
/// Note: Simplified version due to Rust's ownership constraints
/// For full currying, use the partial function instead
pub fn curry2<A, B, R, F>(f: F) -> impl Fn((A, B)) -> R
where
    F: Fn(A, B) -> R + 'static,
    A: 'static,
    B: 'static,
    R: 'static,
{
    move |(a, b)| f(a, b)
}

/// Partial application
pub fn partial<A: Clone + 'static, B, R, F>(f: F, a: A) -> impl Fn(B) -> R
where
    F: Fn(A, B) -> R + 'static,
    B: 'static,
    R: 'static,
{
    move |b| f(a.clone(), b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipe() {
        let result = vec![1, 2, 3]
            .pipe(|v| v.into_iter().map(|x| x * 2).collect::<Vec<_>>())
            .pipe(|v: Vec<_>| v.into_iter().sum::<i32>());
        
        assert_eq!(result, 12);
    }

    #[test]
    fn test_partial() {
        let add = |a: i32, b: i32| a + b;
        let add5 = partial(add, 5);
        
        assert_eq!(add5(3), 8);
    }
}
