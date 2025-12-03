# Markov Chain Monte Carlo: Mathematical Theory

## Introduction

Markov Chain Monte Carlo (MCMC) methods are a class of algorithms for sampling from probability distributions based on constructing a Markov chain that has the desired distribution as its equilibrium distribution. MCMC is fundamental to Bayesian inference, computational physics, and many areas of computational statistics.

## The Monte Carlo Method

### Goal

Sample from a target distribution $\pi(\theta)$ where:
- Direct sampling is difficult or impossible
- We can evaluate $\pi(\theta)$ up to a normalization constant

### Why Monte Carlo?

Given samples $\theta^{(1)}, ..., \theta^{(N)} \sim \pi(\theta)$, we can approximate:

**Expectations**:
$$\mathbb{E}_\pi[f(\theta)] \approx \frac{1}{N}\sum_{i=1}^N f(\theta^{(i)})$$

**Probabilities**:
$$P(\theta \in A) \approx \frac{1}{N}\sum_{i=1}^N \mathbb{1}[\theta^{(i)} \in A]$$

**Quantiles**, **distributions**, and other properties of $\pi(\theta)$.

## Markov Chains

### Definition

A sequence $\theta^{(0)}, \theta^{(1)}, \theta^{(2)}, ...$ is a Markov chain if:

$$P(\theta^{(t+1)} \mid \theta^{(0)}, ..., \theta^{(t)}) = P(\theta^{(t+1)} \mid \theta^{(t)})$$

The next state depends only on the current state.

### Transition Kernel

$$K(\theta' \mid \theta) = P(\theta^{(t+1)} = \theta' \mid \theta^{(t)} = \theta)$$

### Stationary Distribution

A distribution $\pi(\theta)$ is **stationary** if:

$$\pi(\theta') = \int K(\theta' \mid \theta) \pi(\theta) d\theta$$

If we start with $\theta^{(0)} \sim \pi$, then $\theta^{(t)} \sim \pi$ for all $t$.

### Ergodicity

A Markov chain is **ergodic** if:
1. **Irreducible**: Can reach any state from any state
2. **Aperiodic**: No cyclic behavior

For ergodic chains with stationary distribution $\pi$:
$$\lim_{t \to \infty} P(\theta^{(t)} \in A) = \pi(A)$$

regardless of initial state $\theta^{(0)}$.

### Detailed Balance

A sufficient (but not necessary) condition for $\pi$ to be stationary:

$$\pi(\theta) K(\theta' \mid \theta) = \pi(\theta') K(\theta \mid \theta')$$

**Reversibility**: The probability of going from $\theta$ to $\theta'$ equals the probability of the reverse transition.

## Metropolis-Hastings Algorithm

### Overview

The Metropolis-Hastings (MH) algorithm constructs a Markov chain whose stationary distribution is the target $\pi(\theta)$.

### Algorithm

**Input**: Target distribution $\pi(\theta)$, proposal distribution $q(\theta' \mid \theta)$

1. Initialize $\theta^{(0)}$

2. For $t = 0, 1, 2, ...$:
   
   a. **Propose**: Draw $\theta^* \sim q(\theta^* \mid \theta^{(t)})$
   
   b. **Compute acceptance probability**:
   $$\alpha = \min\left(1, \frac{\pi(\theta^*) q(\theta^{(t)} \mid \theta^*)}{\pi(\theta^{(t)}) q(\theta^* \mid \theta^{(t)})}\right)$$
   
   c. **Accept/Reject**:
   $$\theta^{(t+1)} = \begin{cases}
   \theta^* & \text{with probability } \alpha \\
   \theta^{(t)} & \text{with probability } 1-\alpha
   \end{cases}$$

### Why It Works

**Theorem**: The MH algorithm produces a Markov chain with stationary distribution $\pi(\theta)$.

**Proof sketch**: Show detailed balance holds.

For accepted moves:
$$\pi(\theta) q(\theta' \mid \theta) \alpha(\theta' \mid \theta) = \pi(\theta') q(\theta \mid \theta') \alpha(\theta \mid \theta')$$

For rejected moves, transitions to same state also balance.

### Special Cases

#### Metropolis Algorithm

When proposal is **symmetric**: $q(\theta' \mid \theta) = q(\theta \mid \theta')$

Acceptance probability simplifies:
$$\alpha = \min\left(1, \frac{\pi(\theta^*)}{\pi(\theta^{(t)})}\right)$$

#### Random Walk Metropolis

Use Gaussian proposal:
$$q(\theta' \mid \theta) = \mathcal{N}(\theta' \mid \theta, \sigma^2 I)$$

Symmetric, so use Metropolis acceptance.

#### Independence Sampler

Proposal doesn't depend on current state:
$$q(\theta' \mid \theta) = g(\theta')$$

Good if $g$ approximates $\pi$ well.

## Gibbs Sampling

### Motivation

For multivariate distributions, updating all dimensions at once can be inefficient.

### Algorithm

For $\theta = (\theta_1, ..., \theta_d)$:

1. Initialize $\theta^{(0)} = (\theta_1^{(0)}, ..., \theta_d^{(0)})$

2. For $t = 0, 1, 2, ...$:
   
   Sample each component from its conditional distribution:
   
   $$\theta_1^{(t+1)} \sim \pi(\theta_1 \mid \theta_2^{(t)}, ..., \theta_d^{(t)})$$
   $$\theta_2^{(t+1)} \sim \pi(\theta_2 \mid \theta_1^{(t+1)}, \theta_3^{(t)}, ..., \theta_d^{(t)})$$
   $$\vdots$$
   $$\theta_d^{(t+1)} \sim \pi(\theta_d \mid \theta_1^{(t+1)}, ..., \theta_{d-1}^{(t+1)})$$

### Properties

- Special case of Metropolis-Hastings with acceptance probability = 1
- Requires knowing conditional distributions
- Can be slow if variables are highly correlated

## Bayesian Inference with MCMC

### Bayes' Theorem

$$p(\theta \mid D) = \frac{p(D \mid \theta) p(\theta)}{p(D)}$$

where:
- $p(\theta \mid D)$ is the **posterior** (what we want)
- $p(D \mid \theta)$ is the **likelihood**
- $p(\theta)$ is the **prior**
- $p(D) = \int p(D \mid \theta) p(\theta) d\theta$ is the **evidence** (normalizing constant)

### MCMC for Posterior Sampling

The evidence $p(D)$ is often intractable, but we can evaluate:
$$\pi(\theta) \propto p(D \mid \theta) p(\theta)$$

MCMC only needs $\pi$ up to a constant, so we can sample from the posterior!

### Metropolis-Hastings for Bayesian Inference

Target: $\pi(\theta) = p(D \mid \theta) p(\theta)$ (unnormalized posterior)

Acceptance probability:
$$\alpha = \min\left(1, \frac{p(D \mid \theta^*) p(\theta^*)}{p(D \mid \theta^{(t)}) p(\theta^{(t)})} \cdot \frac{q(\theta^{(t)} \mid \theta^*)}{q(\theta^* \mid \theta^{(t)})}\right)$$

For symmetric proposals:
$$\alpha = \min\left(1, \frac{p(D \mid \theta^*) p(\theta^*)}{p(D \mid \theta^{(t)}) p(\theta^{(t)})}\right)$$

### Example: Normal Mean and Variance

**Model**: $y_i \sim \mathcal{N}(\mu, \sigma^2)$, $i = 1, ..., n$

**Prior**: $p(\mu, \sigma^2) = p(\mu) p(\sigma^2)$
- $p(\mu) = \mathcal{N}(0, 100)$
- $p(\sigma^2) = \text{InvGamma}(0.01, 0.01)$

**Likelihood**:
$$p(D \mid \mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mu)^2}{2\sigma^2}\right)$$

**Log-posterior** (up to constant):
$$\log \pi(\mu, \sigma^2) = \log p(D \mid \mu, \sigma^2) + \log p(\mu) + \log p(\sigma^2)$$

Sample using MH with Gaussian random walk proposals.

## Convergence Diagnostics

### Burn-in Period

Discard initial samples before the chain has converged to the stationary distribution.

**How to choose?**
- Plot trace plots and look for stabilization
- Typically 1000-10000 iterations
- Conservative: discard first 50% of samples

### Effective Sample Size (ESS)

Due to autocorrelation, MCMC samples are not independent.

$$\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}$$

where $\rho_k$ is the autocorrelation at lag $k$.

**Interpretation**: ESS ≈ number of independent samples

### Autocorrelation

$$\rho_k = \frac{\text{Cov}(\theta^{(t)}, \theta^{(t+k)})}{\text{Var}(\theta^{(t)})}$$

**Goal**: Low autocorrelation (faster mixing)

**Solutions**:
- Tune proposal distribution
- Thinning (keep every $k$-th sample)
- Advanced methods (HMC, parallel tempering)

### Gelman-Rubin Diagnostic ($\hat{R}$)

Run multiple chains with different starting points.

$$\hat{R} = \sqrt{\frac{\text{Var}^+}{\text{Within-chain variance}}}$$

**Interpretation**:
- $\hat{R} \approx 1$: Chains have converged
- $\hat{R} > 1.1$: Chains have not mixed

### Geweke Diagnostic

Compare means of first 10% and last 50% of chain.

$$Z = \frac{\bar{\theta}_A - \bar{\theta}_B}{\sqrt{\text{SE}_A^2 + \text{SE}_B^2}}$$

Under null hypothesis of convergence, $Z \sim \mathcal{N}(0, 1)$.

## Proposal Tuning

### Acceptance Rate

**Optimal acceptance rate** (for random walk Metropolis in high dimensions):
- 1D: 44%
- ∞-D: 23.4%
- Practical: 20-40%

**Too high** (> 50%): Proposals too small, slow exploration

**Too low** (< 10%): Proposals too large, many rejections

### Adaptive Metropolis

Automatically tune proposal covariance during burn-in:

$$\Sigma^{(t+1)} = \text{Cov}(\theta^{(1)}, ..., \theta^{(t)})$$

Proposal:
$$q(\theta' \mid \theta) = \mathcal{N}(\theta', \theta, 2.38^2 \Sigma^{(t)} / d)$$

where $d$ is dimension.

### Optimal Scaling

Roberts and Rosenthal (2001): For Gaussian targets in $d$ dimensions, optimal variance:

$$\sigma^2 = \frac{2.38^2}{d} \Sigma$$

where $\Sigma$ is posterior covariance.

## Advanced MCMC Methods

### Hamiltonian Monte Carlo (HMC)

Uses gradient information to propose distant states with high acceptance.

**Advantages**:
- Efficient for high-dimensional problems
- Low autocorrelation

**Disadvantages**:
- Requires gradient computation
- More complex to implement

### Parallel Tempering

Run multiple chains at different "temperatures":

$$\pi_\beta(\theta) \propto \pi(\theta)^\beta$$

Exchange states between chains to improve mixing.

### Reversible Jump MCMC

For problems where dimension changes (model selection).

### Sequential Monte Carlo (SMC)

Particle filters for sequential data.

## Practical Considerations

### Initialization

**Strategies**:
1. **Random**: From prior or broad distribution
2. **MAP estimate**: From optimization
3. **Overdispersed**: Multiple chains, widely separated

### Thinning

Keep every $k$-th sample to reduce autocorrelation and storage.

**Debate**: Some argue thinning wastes information. Better to run longer and keep all samples (if storage permits).

### Reparameterization

Transform parameters to reduce correlation:

**Example**: Instead of $(\mu, \sigma^2)$, use $(\mu, \log\sigma)$.

Better geometry → better sampling.

### Multimodal Distributions

**Challenge**: Single chain may get stuck in one mode.

**Solutions**:
- Multiple independent chains
- Parallel tempering
- Simulated annealing

## Theoretical Guarantees

### Central Limit Theorem

For ergodic chains:

$$\sqrt{N}(\bar{\theta} - \mathbb{E}[\theta]) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

where $\sigma^2$ depends on autocorrelation.

**Implication**: Monte Carlo estimates are asymptotically normal.

### Law of Large Numbers

$$\bar{\theta} = \frac{1}{N}\sum_{i=1}^N \theta^{(i)} \xrightarrow{a.s.} \mathbb{E}_\pi[\theta]$$

**Implication**: Estimates converge to true values.

### Convergence Rate

Geometric ergodicity: $\|P^t(\theta, \cdot) - \pi\| \leq C \rho^t$

for some $C > 0$ and $\rho < 1$.

Faster convergence → fewer samples needed.

## MCMC vs. Alternatives

| Method | Pros | Cons |
|--------|------|------|
| **MCMC** | General, exact (asymptotically) | Slow convergence, diagnostics needed |
| **Variational Inference** | Fast, scalable | Approximate, may be biased |
| **Importance Sampling** | Simple, independent samples | Requires good proposal |
| **Rejection Sampling** | Independent samples | Inefficient in high dimensions |
| **Grid/Quadrature** | Deterministic | Exponential in dimension |

## Applications

### 1. Bayesian Regression

Posterior inference for regression coefficients and variance.

### 2. Hierarchical Models

Multi-level models with group-specific and population parameters.

### 3. Mixture Models

Cluster analysis with unknown number of components.

### 4. Time Series

State space models, GARCH, stochastic volatility.

### 5. Spatial Statistics

Gaussian processes, kriging, disease mapping.

### 6. Computational Biology

Phylogenetic inference, population genetics.

## Software Implementations

### Stan

- Hamiltonian Monte Carlo (NUTS)
- Automatic differentiation
- Interfaces: R, Python, Julia, etc.

### PyMC

- Python library
- Variety of samplers
- Integrates with NumPy, Theano

### JAGS

- Just Another Gibbs Sampler
- BUGS-like syntax
- Interfaces: R (rjags), Python

### TensorFlow Probability / PyTorch

- Probabilistic programming on GPUs
- Integration with deep learning

## Key References

1. **Metropolis, N., et al.** (1953). *Equation of state calculations by fast computing machines*. The Journal of Chemical Physics, 21(6), 1087-1092.
   - Original Metropolis algorithm

2. **Hastings, W. K.** (1970). *Monte Carlo sampling methods using Markov chains and their applications*. Biometrika, 57(1), 97-109.
   - Generalization to Metropolis-Hastings

3. **Geman, S., & Geman, D.** (1984). *Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 6, 721-741.
   - Gibbs sampling

4. **Gelfand, A. E., & Smith, A. F. M.** (1990). *Sampling-based approaches to calculating marginal densities*. Journal of the American Statistical Association, 85(410), 398-409.
   - Popularized MCMC for Bayesian inference

5. **Brooks, S., Gelman, A., Jones, G., & Meng, X. L.** (Eds.). (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.
   - Comprehensive reference

6. **Robert, C. P., & Casella, G.** (2004). *Monte Carlo Statistical Methods*. Springer.
   - Mathematical treatment

## Summary

MCMC provides a powerful framework for:
- Sampling from complex, high-dimensional distributions
- Bayesian inference when posteriors are intractable
- Computing expectations and quantiles

**Key components**:
1. **Markov chain**: Generates dependent samples
2. **Stationary distribution**: Chain converges to target
3. **Metropolis-Hastings**: General acceptance/rejection scheme
4. **Diagnostics**: Ensure convergence and adequate mixing

## See Also

- [MCMC API Documentation](../mcmc.md) - Implementation details and usage
- [HMM Theory](hmm.md) - Alternative for sequential latent variable models
- [Differential Evolution Theory](differential_evolution.md) - Optimization methods
