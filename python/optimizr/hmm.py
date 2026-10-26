"""
Hidden Markov Model implementation
"""

import warnings
from typing import Optional
import numpy as np

# Try to import Rust backend
try:
    from optimizr._core import fit_hmm as _rust_fit_hmm, viterbi_decode as _rust_viterbi

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class HMM:
    """
    Hidden Markov Model for regime detection and sequence analysis.

    Uses the Baum-Welch algorithm (EM) for parameter estimation and
    Viterbi algorithm for finding the most likely state sequence.

    Parameters
    ----------
    n_states : int
        Number of hidden states

    Attributes
    ----------
    transition_matrix_ : np.ndarray or None
        Learned transition probabilities (n_states × n_states)
    emission_means_ : np.ndarray or None
        Mean of Gaussian emission for each state
    emission_stds_ : np.ndarray or None
        Std dev of Gaussian emission for each state

    Examples
    --------
    >>> import numpy as np
    >>> from optimizr import HMM
    >>>
    >>> # Generate data with regime changes
    >>> returns = np.concatenate([
    ...     np.random.normal(0.01, 0.02, 500),  # Bull market
    ...     np.random.normal(-0.01, 0.03, 500),  # Bear market
    ... ])
    >>>
    >>> # Fit HMM
    >>> hmm = HMM(n_states=2)
    >>> hmm.fit(returns, n_iterations=100)
    >>>
    >>> # Decode states
    >>> states = hmm.predict(returns)
    >>> print(f"Detected states: {np.unique(states)}")
    """

    def __init__(self, n_states: int = 2):
        if n_states < 2:
            raise ValueError("n_states must be at least 2")

        self.n_states = n_states
        self.transition_matrix_: np.ndarray = np.zeros((n_states, n_states))
        self.emission_means_: np.ndarray = np.zeros(n_states)
        self.emission_stds_: np.ndarray = np.ones(n_states)
        self._params = None

    def fit(self, X: np.ndarray, n_iterations: int = 100, tolerance: float = 1e-6) -> "HMM":
        """
        Fit HMM parameters using Baum-Welch algorithm.

        Parameters
        ----------
        X : np.ndarray
            Time series observations (1D array)
        n_iterations : int, default=100
            Maximum number of EM iterations
        tolerance : float, default=1e-6
            Convergence threshold for log-likelihood change

        Returns
        -------
        self : HMM
            Fitted model
        """
        X = np.asarray(X).flatten()

        if len(X) == 0:
            raise ValueError("X cannot be empty")

        if RUST_AVAILABLE:
            # Use Rust implementation
            self._params = _rust_fit_hmm(
                observations=X.tolist(),
                n_states=self.n_states,
                n_iterations=n_iterations,
                tolerance=tolerance,
            )

            self.transition_matrix_ = np.array(self._params.transition_matrix)
            self.emission_means_ = np.array(self._params.emission_means)
            self.emission_stds_ = np.array(self._params.emission_stds)
        else:
            # Pure Python fallback
            warnings.warn(
                "Rust backend not available. Using slower Python implementation.", RuntimeWarning
            )
            self._fit_python(X, n_iterations, tolerance)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Parameters
        ----------
        X : np.ndarray
            Time series observations (1D array)

        Returns
        -------
        states : np.ndarray
            Most likely state at each time step
        """
        if self.transition_matrix_ is None:
            raise ValueError("Model must be fitted before prediction")

        X = np.asarray(X).flatten()

        if RUST_AVAILABLE and self._params is not None:
            states = _rust_viterbi(X.tolist(), self._params)
            return np.array(states)
        else:
            return self._viterbi_python(X)

    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.

        Parameters
        ----------
        X : np.ndarray
            Time series observations

        Returns
        -------
        log_likelihood : float
            Log P(X | model)
        """
        if self.transition_matrix_ is None:
            raise ValueError("Model must be fitted before scoring")

        X = np.asarray(X).flatten()
        alpha = self._forward_python(X)
        return np.log(np.sum(alpha[-1]))

    def _fit_python(self, X: np.ndarray, n_iterations: int, tolerance: float):
        """Pure Python implementation of Baum-Welch"""
        n_obs = len(X)

        # Initialize parameters
        self.transition_matrix_ = np.ones((self.n_states, self.n_states)) / self.n_states

        # Initialize emissions based on quantiles
        quantiles = np.linspace(0, 1, self.n_states + 1)
        self.emission_means_ = np.zeros(self.n_states)
        self.emission_stds_ = np.ones(self.n_states)

        for i in range(self.n_states):
            mask = (X >= np.quantile(X, quantiles[i])) & (X < np.quantile(X, quantiles[i + 1]))
            if np.any(mask):
                self.emission_means_[i] = np.mean(X[mask])
                self.emission_stds_[i] = max(np.std(X[mask]), 1e-6)

        prev_ll = float("-inf")

        # EM iterations
        for _ in range(n_iterations):
            # E-step
            alpha = self._forward_python(X)
            beta = self._backward_python(X)
            gamma = self._compute_gamma_python(alpha, beta)
            xi = self._compute_xi_python(X, alpha, beta)

            # M-step
            self._update_parameters_python(X, gamma, xi)

            # Check convergence
            ll = np.log(np.sum(alpha[-1]))
            if abs(ll - prev_ll) < tolerance:
                break
            prev_ll = ll

    def _forward_python(self, X: np.ndarray) -> np.ndarray:
        """Forward algorithm"""
        n_obs = len(X)
        alpha = np.zeros((n_obs, self.n_states))

        # Initialize
        for s in range(self.n_states):
            alpha[0, s] = (1.0 / self.n_states) * self._emission_prob(X[0], s)

        alpha[0] /= np.sum(alpha[0])

        # Recursion
        for t in range(1, n_obs):
            for s in range(self.n_states):
                alpha[t, s] = np.sum(
                    alpha[t - 1] * self.transition_matrix_[:, s]
                ) * self._emission_prob(X[t], s)
            alpha[t] /= max(np.sum(alpha[t]), 1e-10)

        return alpha

    def _backward_python(self, X: np.ndarray) -> np.ndarray:
        """Backward algorithm"""
        n_obs = len(X)
        beta = np.zeros((n_obs, self.n_states))
        beta[-1] = 1.0

        for t in range(n_obs - 2, -1, -1):
            for s in range(self.n_states):
                beta[t, s] = np.sum(
                    self.transition_matrix_[s]
                    * np.array([self._emission_prob(X[t + 1], s2) for s2 in range(self.n_states)])
                    * beta[t + 1]
                )
            beta[t] /= max(np.sum(beta[t]), 1e-10)

        return beta

    def _compute_gamma_python(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute state occupation probabilities"""
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma

    def _compute_xi_python(self, X: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute transition probabilities"""
        n_obs = len(X)
        xi = np.zeros((n_obs - 1, self.n_states, self.n_states))

        for t in range(n_obs - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i]
                        * self.transition_matrix_[i, j]
                        * self._emission_prob(X[t + 1], j)
                        * beta[t + 1, j]
                    )
            xi[t] /= max(np.sum(xi[t]), 1e-10)

        return xi

    def _update_parameters_python(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: update parameters"""
        # Update transitions
        for i in range(self.n_states):
            denom = np.sum(gamma[:-1, i])
            for j in range(self.n_states):
                numer = np.sum(xi[:, i, j])
                self.transition_matrix_[i, j] = numer / max(denom, 1e-10)

        # Update emissions
        for s in range(self.n_states):
            weights = gamma[:, s]
            sum_weights = np.sum(weights)

            if sum_weights > 1e-10:
                self.emission_means_[s] = np.sum(weights * X) / sum_weights
                self.emission_stds_[s] = max(
                    np.sqrt(np.sum(weights * (X - self.emission_means_[s]) ** 2) / sum_weights),
                    1e-6,
                )

    def _emission_prob(self, obs: float, state: int) -> float:
        """Gaussian emission probability"""
        mean = self.emission_means_[state]
        std = self.emission_stds_[state]
        z = (obs - mean) / std
        return max(np.exp(-0.5 * z**2) / (std * np.sqrt(2 * np.pi)), 1e-10)

    def _viterbi_python(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding"""
        n_obs = len(X)
        delta = np.full((n_obs, self.n_states), float("-inf"))
        psi = np.zeros((n_obs, self.n_states), dtype=int)

        # Initialize
        for s in range(self.n_states):
            delta[0, s] = np.log(1.0 / self.n_states) + np.log(self._emission_prob(X[0], s))

        # Recursion
        for t in range(1, n_obs):
            for s in range(self.n_states):
                trans_probs = delta[t - 1] + np.log(self.transition_matrix_[:, s] + 1e-10)
                psi[t, s] = np.argmax(trans_probs)
                delta[t, s] = trans_probs[psi[t, s]] + np.log(self._emission_prob(X[t], s))

        # Backtrack
        path = np.zeros(n_obs, dtype=int)
        path[-1] = np.argmax(delta[-1])

        for t in range(n_obs - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path
