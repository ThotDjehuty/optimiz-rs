//! Kalman Filter Module
//! ====================
//!
//! Generic Kalman filter implementations for state estimation in dynamical systems.
//!
//! # Implementations
//!
//! - **Linear Kalman Filter**: For linear state-space models
//! - **Extended Kalman Filter (EKF)**: For nonlinear systems via linearization
//! - **Unscented Kalman Filter (UKF)**: For nonlinear systems with better approximations
//!
//! # Mathematical Foundation
//!
//! ## State-Space Model
//!
//! Consider a discrete-time dynamical system:
//!
//! ```text
//! State equation:        x_{k+1} = F_k x_k + B_k u_k + w_k
//! Observation equation:  z_k = H_k x_k + v_k
//! ```
//!
//! where:
//! - `x_k` ∈ ℝⁿ is the state vector at time k
//! - `z_k` ∈ ℝᵐ is the observation vector
//! - `u_k` ∈ ℝᵖ is the control input
//! - `F_k` is the state transition matrix
//! - `H_k` is the observation matrix
//! - `B_k` is the control input matrix
//! - `w_k ~ N(0, Q_k)` is process noise
//! - `v_k ~ N(0, R_k)` is measurement noise
//!
//! ## Linear Kalman Filter
//!
//! The Kalman filter is the optimal Bayesian estimator for linear Gaussian systems.
//!
//! ### Prediction Step
//!
//! ```text
//! x̂_{k|k-1} = F_k x̂_{k-1|k-1} + B_k u_k
//! P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k
//! ```
//!
//! ### Update Step
//!
//! ```text
//! ỹ_k = z_k - H_k x̂_{k|k-1}              (innovation)
//! S_k = H_k P_{k|k-1} H_k^T + R_k        (innovation covariance)
//! K_k = P_{k|k-1} H_k^T S_k^{-1}         (Kalman gain)
//! x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k          (state update)
//! P_{k|k} = (I - K_k H_k) P_{k|k-1}      (covariance update)
//! ```
//!
//! ## Extended Kalman Filter (EKF)
//!
//! For nonlinear systems:
//!
//! ```text
//! x_{k+1} = f(x_k, u_k) + w_k
//! z_k = h(x_k) + v_k
//! ```
//!
//! The EKF linearizes around the current estimate using Jacobians:
//!
//! ```text
//! F_k = ∂f/∂x |_{x̂_{k-1|k-1}}
//! H_k = ∂h/∂x |_{x̂_{k|k-1}}
//! ```
//!
//! ## Unscented Kalman Filter (UKF)
//!
//! The UKF uses sigma points to capture nonlinear transformations better than EKF:
//!
//! 1. Generate sigma points around current estimate
//! 2. Propagate through nonlinear functions
//! 3. Compute mean and covariance from transformed points
//!
//! UKF has O(n³) complexity but better accuracy than EKF for highly nonlinear systems.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Determinant, Inverse};
use std::f64::consts::PI;
use thiserror::Error;

use super::{OptimalControlError, Result};

#[derive(Error, Debug)]
pub enum KalmanError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Matrix is not positive definite: {0}")]
    NotPositiveDefinite(String),

    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

/// Trait for state transition models
///
/// Enables generic implementation across linear and nonlinear dynamics.
pub trait StateTransitionModel {
    /// Predict next state: x_{k+1} = f(x_k, u_k)
    fn predict(&self, state: &Array1<f64>, control: Option<&Array1<f64>>) -> Array1<f64>;

    /// Get state transition Jacobian (for EKF): F_k = ∂f/∂x
    fn jacobian(&self, state: &Array1<f64>) -> Array2<f64>;

    /// Get process noise covariance Q_k
    fn process_noise(&self) -> &Array2<f64>;
}

/// Trait for observation models
///
/// Enables generic implementation across linear and nonlinear measurements.
pub trait ObservationModel {
    /// Predict observation: z_k = h(x_k)
    fn observe(&self, state: &Array1<f64>) -> Array1<f64>;

    /// Get observation Jacobian (for EKF): H_k = ∂h/∂x
    fn jacobian(&self, state: &Array1<f64>) -> Array2<f64>;

    /// Get measurement noise covariance R_k
    fn measurement_noise(&self) -> &Array2<f64>;
}

/// Linear state transition model: x_{k+1} = F x_k + B u_k + w_k
#[derive(Clone)]
pub struct LinearStateTransition {
    /// State transition matrix F
    pub f_matrix: Array2<f64>,
    /// Control input matrix B
    pub b_matrix: Option<Array2<f64>>,
    /// Process noise covariance Q
    pub q_matrix: Array2<f64>,
}

impl StateTransitionModel for LinearStateTransition {
    fn predict(&self, state: &Array1<f64>, control: Option<&Array1<f64>>) -> Array1<f64> {
        let mut next_state = self.f_matrix.dot(state);

        if let (Some(b), Some(u)) = (&self.b_matrix, control) {
            next_state = next_state + b.dot(u);
        }

        next_state
    }

    fn jacobian(&self, _state: &Array1<f64>) -> Array2<f64> {
        self.f_matrix.clone()
    }

    fn process_noise(&self) -> &Array2<f64> {
        &self.q_matrix
    }
}

/// Linear observation model: z_k = H x_k + v_k
#[derive(Clone)]
pub struct LinearObservation {
    /// Observation matrix H
    pub h_matrix: Array2<f64>,
    /// Measurement noise covariance R
    pub r_matrix: Array2<f64>,
}

impl ObservationModel for LinearObservation {
    fn observe(&self, state: &Array1<f64>) -> Array1<f64> {
        self.h_matrix.dot(state)
    }

    fn jacobian(&self, _state: &Array1<f64>) -> Array2<f64> {
        self.h_matrix.clone()
    }

    fn measurement_noise(&self) -> &Array2<f64> {
        &self.r_matrix
    }
}

/// Kalman Filter State
///
/// Maintains current state estimate and covariance.
#[derive(Clone, Debug)]
pub struct KalmanState {
    /// State estimate x̂_{k|k}
    pub state: Array1<f64>,
    /// Covariance estimate P_{k|k}
    pub covariance: Array2<f64>,
    /// Innovation (residual) ỹ_k
    pub innovation: Option<Array1<f64>>,
    /// Innovation covariance S_k
    pub innovation_covariance: Option<Array2<f64>>,
    /// Log-likelihood of current observation
    pub log_likelihood: Option<f64>,
}

impl KalmanState {
    pub fn new(state: Array1<f64>, covariance: Array2<f64>) -> Self {
        Self {
            state,
            covariance,
            innovation: None,
            innovation_covariance: None,
            log_likelihood: None,
        }
    }

    /// Get state dimension
    pub fn dim(&self) -> usize {
        self.state.len()
    }
}

/// Generic Kalman Filter
///
/// Works with any state transition and observation models implementing the traits.
pub struct KalmanFilter<S: StateTransitionModel, O: ObservationModel> {
    /// State transition model
    state_model: S,
    /// Observation model
    obs_model: O,
    /// Current filter state
    state: KalmanState,
}

impl<S: StateTransitionModel, O: ObservationModel> KalmanFilter<S, O> {
    /// Create new Kalman filter with initial state
    pub fn new(state_model: S, obs_model: O, initial_state: KalmanState) -> Self {
        Self {
            state_model,
            obs_model,
            state: initial_state,
        }
    }

    /// Get current state estimate
    pub fn state(&self) -> &KalmanState {
        &self.state
    }

    /// Prediction step: propagate state and covariance forward
    ///
    /// Computes x̂_{k|k-1} and P_{k|k-1}
    pub fn predict(&mut self, control: Option<&Array1<f64>>) -> Result<()> {
        // Predict state: x̂_{k|k-1} = f(x̂_{k-1|k-1}, u_k)
        let predicted_state = self.state_model.predict(&self.state.state, control);

        // Get Jacobian: F_k = ∂f/∂x
        let f_jacobian = self.state_model.jacobian(&self.state.state);

        // Predict covariance: P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k
        let predicted_cov = f_jacobian.dot(&self.state.covariance).dot(&f_jacobian.t())
            + self.state_model.process_noise();

        self.state.state = predicted_state;
        self.state.covariance = predicted_cov;

        Ok(())
    }

    /// Update step: incorporate measurement
    ///
    /// Computes x̂_{k|k} and P_{k|k} given observation z_k
    pub fn update(&mut self, observation: &Array1<f64>) -> Result<()> {
        // Predicted observation: ẑ_k = h(x̂_{k|k-1})
        let predicted_obs = self.obs_model.observe(&self.state.state);

        // Innovation: ỹ_k = z_k - ẑ_k
        let innovation = observation - &predicted_obs;

        // Get observation Jacobian: H_k = ∂h/∂x
        let h_jacobian = self.obs_model.jacobian(&self.state.state);

        // Innovation covariance: S_k = H_k P_{k|k-1} H_k^T + R_k
        let s_matrix = h_jacobian.dot(&self.state.covariance).dot(&h_jacobian.t())
            + self.obs_model.measurement_noise();

        // Compute Kalman gain: K_k = P_{k|k-1} H_k^T S_k^{-1}
        let s_inv = s_matrix
            .inv()
            .map_err(|e| OptimalControlError::MatrixError(format!("Cannot invert S: {:?}", e)))?;

        let kalman_gain = self.state.covariance.dot(&h_jacobian.t()).dot(&s_inv);

        // Update state: x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k
        self.state.state = &self.state.state + &kalman_gain.dot(&innovation);

        // Update covariance: P_{k|k} = (I - K_k H_k) P_{k|k-1}
        let n = self.state.covariance.nrows();
        let identity = Array2::eye(n);
        let update_matrix = &identity - &kalman_gain.dot(&h_jacobian);
        self.state.covariance = update_matrix.dot(&self.state.covariance);

        // Compute log-likelihood: log p(z_k | z_{1:k-1})
        let log_likelihood = Self::compute_log_likelihood(&innovation, &s_matrix)?;

        // Store diagnostics
        self.state.innovation = Some(innovation);
        self.state.innovation_covariance = Some(s_matrix);
        self.state.log_likelihood = Some(log_likelihood);

        Ok(())
    }

    /// Run filter on sequence of observations
    ///
    /// Returns state estimates and log-likelihood
    pub fn filter(
        &mut self,
        observations: &[Array1<f64>],
        controls: Option<&[Array1<f64>]>,
    ) -> Result<FilterResult> {
        let n_obs = observations.len();

        let mut states = Vec::with_capacity(n_obs);
        let mut covariances = Vec::with_capacity(n_obs);
        let mut log_likelihood = 0.0;

        for (i, obs) in observations.iter().enumerate() {
            // Predict
            let control = controls.and_then(|c| c.get(i));
            self.predict(control)?;

            // Update
            self.update(obs)?;

            // Store results
            states.push(self.state.state.clone());
            covariances.push(self.state.covariance.clone());
            if let Some(ll) = self.state.log_likelihood {
                log_likelihood += ll;
            }
        }

        Ok(FilterResult {
            states,
            covariances,
            log_likelihood,
        })
    }

    /// Compute Gaussian log-likelihood
    fn compute_log_likelihood(innovation: &Array1<f64>, cov: &Array2<f64>) -> Result<f64> {
        let m = innovation.len() as f64;
        let det = cov
            .det()
            .map_err(|e| OptimalControlError::MatrixError(format!("Cannot compute det: {:?}", e)))?;

        if det <= 0.0 {
            return Err(OptimalControlError::NumericalError(
                "Innovation covariance not positive definite".to_string(),
            ));
        }

        let cov_inv = cov
            .inv()
            .map_err(|e| OptimalControlError::MatrixError(format!("Cannot invert cov: {:?}", e)))?;

        let mahalanobis = innovation.dot(&cov_inv.dot(innovation));

        let log_likelihood = -0.5 * (m * (2.0 * PI).ln() + det.ln() + mahalanobis);

        Ok(log_likelihood)
    }
}

/// Results from filtering
#[derive(Clone, Debug)]
pub struct FilterResult {
    /// Filtered state estimates
    pub states: Vec<Array1<f64>>,
    /// Filtered covariances
    pub covariances: Vec<Array2<f64>>,
    /// Total log-likelihood
    pub log_likelihood: f64,
}

/// Rauch-Tung-Striebel (RTS) Smoother
///
/// Performs backward pass to compute smoothed estimates given all observations.
///
/// # Mathematical Foundation
///
/// Given filtered estimates from forward pass, the smoother computes:
///
/// ```text
/// x̂_{k|T} = x̂_{k|k} + C_k (x̂_{k+1|T} - x̂_{k+1|k})
/// P_{k|T} = P_{k|k} + C_k (P_{k+1|T} - P_{k+1|k}) C_k^T
/// ```
///
/// where the smoother gain is:
///
/// ```text
/// C_k = P_{k|k} F_{k+1}^T P_{k+1|k}^{-1}
/// ```
pub struct RTSSmoother<S: StateTransitionModel> {
    state_model: S,
}

impl<S: StateTransitionModel> RTSSmoother<S> {
    pub fn new(state_model: S) -> Self {
        Self { state_model }
    }

    /// Smooth filtered estimates
    pub fn smooth(&self, filter_result: &FilterResult) -> Result<SmootherResult> {
        let n = filter_result.states.len();

        let mut smoothed_states = filter_result.states.clone();
        let mut smoothed_covariances = filter_result.covariances.clone();

        // Backward pass
        for k in (0..n - 1).rev() {
            // Get Jacobian at filtered state
            let f_jacobian = self.state_model.jacobian(&filter_result.states[k]);

            // Predicted covariance: P_{k+1|k} = F P_{k|k} F^T + Q
            let predicted_cov = f_jacobian
                .dot(&filter_result.covariances[k])
                .dot(&f_jacobian.t())
                + self.state_model.process_noise();

            let predicted_cov_inv = predicted_cov.inv().map_err(|e| {
                OptimalControlError::MatrixError(format!("Cannot invert predicted cov: {:?}", e))
            })?;

            // Smoother gain: C_k = P_{k|k} F^T P_{k+1|k}^{-1}
            let smoother_gain = filter_result.covariances[k]
                .dot(&f_jacobian.t())
                .dot(&predicted_cov_inv);

            // Smoothed state: x̂_{k|T} = x̂_{k|k} + C_k (x̂_{k+1|T} - x̂_{k+1|k})
            let predicted_state = self
                .state_model
                .predict(&filter_result.states[k], None);
            let state_diff = &smoothed_states[k + 1] - &predicted_state;
            smoothed_states[k] = &smoothed_states[k] + &smoother_gain.dot(&state_diff);

            // Smoothed covariance: P_{k|T} = P_{k|k} + C_k (P_{k+1|T} - P_{k+1|k}) C_k^T
            let cov_diff = &smoothed_covariances[k + 1] - &predicted_cov;
            smoothed_covariances[k] = &smoothed_covariances[k]
                + &smoother_gain.dot(&cov_diff).dot(&smoother_gain.t());
        }

        Ok(SmootherResult {
            states: smoothed_states,
            covariances: smoothed_covariances,
        })
    }
}

/// Results from smoothing
#[derive(Clone, Debug)]
pub struct SmootherResult {
    /// Smoothed state estimates
    pub states: Vec<Array1<f64>>,
    /// Smoothed covariances
    pub covariances: Vec<Array2<f64>>,
}

/// Unscented Kalman Filter (UKF)
///
/// Uses unscented transform for better handling of nonlinearities.
///
/// # Parameters
///
/// - α (alpha): Spread of sigma points (typically 1e-3)
/// - β (beta): Prior knowledge of distribution (2 for Gaussian)
/// - κ (kappa): Secondary scaling parameter (typically 0)
pub struct UnscentedKalmanFilter<S: StateTransitionModel, O: ObservationModel> {
    state_model: S,
    obs_model: O,
    state: KalmanState,
    alpha: f64,
    beta: f64,
    kappa: f64,
}

impl<S: StateTransitionModel, O: ObservationModel> UnscentedKalmanFilter<S, O> {
    pub fn new(
        state_model: S,
        obs_model: O,
        initial_state: KalmanState,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> Self {
        Self {
            state_model,
            obs_model,
            state: initial_state,
            alpha,
            beta,
            kappa,
        }
    }

    /// Generate sigma points using unscented transform
    fn generate_sigma_points(&self) -> Result<(Vec<Array1<f64>>, Vec<f64>, Vec<f64>)> {
        let n = self.state.dim();
        let lambda = self.alpha.powi(2) * (n as f64 + self.kappa) - n as f64;

        // Compute square root of covariance: P = L L^T
        let sqrt_cov = self.matrix_sqrt(&self.state.covariance)?;

        // Generate 2n+1 sigma points
        let mut sigma_points = Vec::with_capacity(2 * n + 1);
        sigma_points.push(self.state.state.clone());

        let scale = ((n as f64 + lambda).sqrt() * &sqrt_cov).to_owned();

        for i in 0..n {
            sigma_points.push(&self.state.state + &scale.column(i));
            sigma_points.push(&self.state.state - &scale.column(i));
        }

        // Compute weights
        let w_m_0 = lambda / (n as f64 + lambda);
        let w_c_0 = w_m_0 + (1.0 - self.alpha.powi(2) + self.beta);
        let w_i = 1.0 / (2.0 * (n as f64 + lambda));

        let mut weights_mean = vec![w_m_0];
        let mut weights_cov = vec![w_c_0];
        for _ in 0..2 * n {
            weights_mean.push(w_i);
            weights_cov.push(w_i);
        }

        Ok((sigma_points, weights_mean, weights_cov))
    }

    /// Compute matrix square root using Cholesky decomposition
    fn matrix_sqrt(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        use ndarray_linalg::Cholesky;

        matrix.cholesky(ndarray_linalg::UPLO::Lower).map_err(|e| {
            OptimalControlError::MatrixError(format!("Cholesky decomposition failed: {:?}", e))
        })
    }

    pub fn state(&self) -> &KalmanState {
        &self.state
    }

    /// UKF prediction step
    pub fn predict(&mut self, control: Option<&Array1<f64>>) -> Result<()> {
        let (sigma_points, weights_mean, weights_cov) = self.generate_sigma_points()?;

        // Propagate sigma points through state transition
        let predicted_sigmas: Vec<_> = sigma_points
            .iter()
            .map(|sp| self.state_model.predict(sp, control))
            .collect();

        // Compute predicted mean
        let predicted_mean = Self::weighted_mean(&predicted_sigmas, &weights_mean);

        // Compute predicted covariance
        let predicted_cov = Self::weighted_covariance(
            &predicted_sigmas,
            &predicted_mean,
            &weights_cov,
            Some(self.state_model.process_noise()),
        );

        self.state.state = predicted_mean;
        self.state.covariance = predicted_cov;

        Ok(())
    }

    /// UKF update step
    pub fn update(&mut self, observation: &Array1<f64>) -> Result<()> {
        let (sigma_points, weights_mean, weights_cov) = self.generate_sigma_points()?;

        // Propagate sigma points through observation model
        let obs_sigmas: Vec<_> = sigma_points
            .iter()
            .map(|sp| self.obs_model.observe(sp))
            .collect();

        // Compute predicted observation mean
        let obs_mean = Self::weighted_mean(&obs_sigmas, &weights_mean);

        // Compute innovation covariance
        let obs_cov = Self::weighted_covariance(
            &obs_sigmas,
            &obs_mean,
            &weights_cov,
            Some(self.obs_model.measurement_noise()),
        );

        // Compute cross-covariance
        let cross_cov = Self::cross_covariance(&sigma_points, &self.state.state, &obs_sigmas, &obs_mean, &weights_cov);

        // Compute Kalman gain
        let obs_cov_inv = obs_cov.inv().map_err(|e| {
            OptimalControlError::MatrixError(format!("Cannot invert observation cov: {:?}", e))
        })?;

        let kalman_gain = cross_cov.dot(&obs_cov_inv);

        // Innovation
        let innovation = observation - &obs_mean;

        // Update state and covariance
        self.state.state = &self.state.state + &kalman_gain.dot(&innovation);
        self.state.covariance = &self.state.covariance - &kalman_gain.dot(&obs_cov).dot(&kalman_gain.t());

        // Store diagnostics
        let log_likelihood = KalmanFilter::<S, O>::compute_log_likelihood(&innovation, &obs_cov)?;
        self.state.innovation = Some(innovation);
        self.state.innovation_covariance = Some(obs_cov);
        self.state.log_likelihood = Some(log_likelihood);

        Ok(())
    }

    fn weighted_mean(samples: &[Array1<f64>], weights: &[f64]) -> Array1<f64> {
        let mut mean = Array1::zeros(samples[0].len());
        for (sample, &weight) in samples.iter().zip(weights.iter()) {
            mean = mean + &(sample * weight);
        }
        mean
    }

    fn weighted_covariance(
        samples: &[Array1<f64>],
        mean: &Array1<f64>,
        weights: &[f64],
        additive: Option<&Array2<f64>>,
    ) -> Array2<f64> {
        let n = samples[0].len();
        let mut cov = Array2::zeros((n, n));

        for (sample, &weight) in samples.iter().zip(weights.iter()) {
            let diff = sample - mean;
            let outer = Self::outer_product(&diff, &diff);
            cov = cov + &(outer * weight);
        }

        if let Some(add) = additive {
            cov = cov + add;
        }

        cov
    }

    fn cross_covariance(
        x_samples: &[Array1<f64>],
        x_mean: &Array1<f64>,
        y_samples: &[Array1<f64>],
        y_mean: &Array1<f64>,
        weights: &[f64],
    ) -> Array2<f64> {
        let n_x = x_samples[0].len();
        let n_y = y_samples[0].len();
        let mut cross_cov = Array2::zeros((n_x, n_y));

        for ((x, y), &weight) in x_samples.iter().zip(y_samples.iter()).zip(weights.iter()) {
            let x_diff = x - x_mean;
            let y_diff = y - y_mean;
            let outer = Self::outer_product(&x_diff, &y_diff);
            cross_cov = cross_cov + &(outer * weight);
        }

        cross_cov
    }

    fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((a.len(), b.len()));
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[[i, j]] = a[i] * b[j];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_kalman_filter() {
        // Simple 1D constant velocity model
        let dt = 0.1;
        let f = Array2::from_shape_vec((2, 2), vec![1.0, dt, 0.0, 1.0]).unwrap();
        let h = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let q = Array2::eye(2) * 0.01;
        let r = Array2::eye(1) * 0.1;

        let state_model = LinearStateTransition {
            f_matrix: f,
            b_matrix: None,
            q_matrix: q,
        };

        let obs_model = LinearObservation {
            h_matrix: h,
            r_matrix: r,
        };

        let initial_state = KalmanState::new(Array1::from_vec(vec![0.0, 0.0]), Array2::eye(2));

        let mut kf = KalmanFilter::new(state_model, obs_model, initial_state);

        // Generate synthetic observations
        let observations = vec![
            Array1::from_vec(vec![0.1]),
            Array1::from_vec(vec![0.2]),
            Array1::from_vec(vec![0.3]),
        ];

        kf.predict(None).unwrap();
        kf.update(&observations[0]).unwrap();

        // State should move toward observation
        assert!(kf.state().state[0] > 0.0);
    }
}
