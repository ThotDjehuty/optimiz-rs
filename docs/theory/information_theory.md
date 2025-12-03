# Information Theory: Mathematical Foundations

## Introduction

Information theory, founded by Claude Shannon in 1948, provides a mathematical framework for quantifying information, uncertainty, and communication. It has applications in data compression, communication, cryptography, machine learning, and statistical inference.

## Shannon Entropy

### Definition

For a discrete random variable $X$ with probability mass function $p(x)$:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

**Convention**: $0 \log 0 = 0$ (limit as $p \to 0$)

### Units

- **Nats**: Natural logarithm (base $e$)
- **Bits**: Logarithm base 2
- **Dits**: Logarithm base 10

**Conversion**: $H_{\text{bits}} = H_{\text{nats}} / \ln(2) \approx 1.4427 \cdot H_{\text{nats}}$

### Interpretation

**Shannon entropy measures**:
1. **Uncertainty** about $X$ before observation
2. **Information content** of a sample from $X$
3. **Average code length** (optimal compression)
4. **Unpredictability** of $X$

### Properties

**Non-negativity**: 
$$H(X) \geq 0$$

Equality iff $X$ is deterministic (probability 1 on one outcome).

**Maximum entropy**:
$$H(X) \leq \log |\mathcal{X}|$$

Achieved by uniform distribution: $p(x) = 1/|\mathcal{X}|$ for all $x$.

**Concavity**:
$H$ is a concave function of the distribution $p$.

### Examples

**Binary variable** ($p$ = probability of success):
$$H(X) = -p\log p - (1-p)\log(1-p)$$

Maximum at $p = 0.5$: $H_{\text{bits}} = 1$ bit.

**Fair die**:
$$H(X) = -\sum_{i=1}^6 \frac{1}{6}\log\frac{1}{6} = \log 6 \approx 1.79 \text{ bits}$$

**Biased die** (probability 0.5 for face 1, 0.1 for others):
$$H(X) = -0.5\log(0.5) - 5 \times 0.1\log(0.1) \approx 1.36 \text{ bits}$$

Less entropy than fair die (more predictable).

## Continuous Entropy (Differential Entropy)

### Definition

For continuous random variable $X$ with density $f(x)$:

$$h(X) = -\int f(x) \log f(x) dx$$

### Differences from Discrete Case

- Can be **negative**
- Not invariant under coordinate transformations
- Measures relative information (to uniform over support)

### Gaussian Distribution

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$$

**Maximal entropy** among all distributions with variance $\sigma^2$.

### Multivariate Gaussian

For $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$h(\mathbf{X}) = \frac{1}{2}\log\det(2\pi e \boldsymbol{\Sigma})$$

## Joint and Conditional Entropy

### Joint Entropy

For pair $(X, Y)$:

$$H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

**Chain rule**:
$$H(X, Y) = H(X) + H(Y|X)$$

### Conditional Entropy

$$H(Y|X) = -\sum_{x,y} p(x,y) \log p(y|x)$$

**Interpretation**: Average uncertainty in $Y$ given $X$.

**Property**:
$$H(Y|X) \leq H(Y)$$

Conditioning reduces entropy (information never increases uncertainty).

Equality iff $X$ and $Y$ are independent.

## Mutual Information

### Definition

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

Alternatively:

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Or:

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

### Interpretation

**Mutual information measures**:
1. **Reduction** in uncertainty about $X$ given $Y$
2. **Shared information** between $X$ and $Y$
3. **Dependence** between $X$ and $Y$
4. **Distance** from independence

### Properties

**Non-negativity**:
$$I(X; Y) \geq 0$$

Equality iff $X$ and $Y$ are independent.

**Symmetry**:
$$I(X; Y) = I(Y; X)$$

**Bounded**:
$$I(X; Y) \leq \min(H(X), H(Y))$$

Equality when one variable completely determines the other.

**Data processing inequality**:

If $X \to Y \to Z$ form a Markov chain:
$$I(X; Z) \leq I(X; Y)$$

Processing can't increase information.

### Relationship to Correlation

For bivariate Gaussian $(X, Y)$ with correlation $\rho$:

$$I(X; Y) = -\frac{1}{2}\log(1 - \rho^2)$$

**Mutual information** detects both linear and nonlinear dependencies, while **Pearson correlation** only detects linear.

## Kullback-Leibler Divergence

### Definition

For distributions $p$ and $q$ over $\mathcal{X}$:

$$D_{KL}(p \| q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}$$

**Continuous case**:
$$D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

### Interpretation

- **Relative entropy**: Information gain when updating from $q$ to $p$
- **Divergence**: How much $p$ differs from $q$
- **Inefficiency**: Extra bits needed when using code for $q$ to encode $p$

### Properties

**Non-negativity** (Gibbs' inequality):
$$D_{KL}(p \| q) \geq 0$$

Equality iff $p = q$ (almost everywhere).

**Asymmetry**:
$$D_{KL}(p \| q) \neq D_{KL}(q \| p)$$

Not a true distance metric (doesn't satisfy triangle inequality).

**Connection to MI**:
$$I(X; Y) = D_{KL}(p(x,y) \| p(x)p(y))$$

MI is the KL divergence from joint to product of marginals.

## Cross Entropy

### Definition

$$H(p, q) = -\sum_x p(x) \log q(x)$$

**Relationship to KL divergence**:
$$H(p, q) = H(p) + D_{KL}(p \| q)$$

### Machine Learning Application

**Loss function** in classification:

For true distribution $p$ (one-hot) and predicted $q$ (softmax):
$$\text{Loss} = H(p, q)$$

Minimizing cross-entropy ≡ minimizing KL divergence ≡ maximizing likelihood.

## Estimation from Data

### Histogram Method

Given samples $x_1, ..., x_n$ from continuous distribution:

1. **Discretize**: Create histogram with $m$ bins
2. **Estimate probabilities**: $\hat{p}_i = n_i / n$ where $n_i$ is count in bin $i$
3. **Compute entropy**: $\hat{H}(X) = -\sum_{i=1}^m \hat{p}_i \log \hat{p}_i$

### Bin Selection

**Too few bins**: Underestimates entropy (over-smoothing)

**Too many bins**: Overestimates entropy (noise)

**Rules of thumb**:
- Sturges: $m = \lceil \log_2 n + 1 \rceil$
- Scott: $m = \lceil (x_{\max} - x_{\min}) / (3.5 \sigma n^{-1/3}) \rceil$
- Square root: $m = \lceil \sqrt{n} \rceil$

### Bias Correction

Histogram estimator is **biased** (tends to overestimate).

**Miller-Madow correction**:
$$\hat{H}_{\text{corrected}} = \hat{H} - \frac{m - 1}{2n}$$

where $m$ is number of non-empty bins.

### Mutual Information Estimation

For samples $(x_i, y_i)$, $i = 1, ..., n$:

1. Create 2D histogram (or separate 1D histograms)
2. Estimate joint and marginal probabilities
3. Compute:
$$\hat{I}(X; Y) = \sum_{i,j} \hat{p}_{ij} \log \frac{\hat{p}_{ij}}{\hat{p}_i \hat{p}_j}$$

**Alternative estimators**:
- k-nearest neighbors (Kraskov et al., 2004)
- Kernel density estimation
- Copula-based methods

## Information-Theoretic Principles

### Maximum Entropy Principle

**Given**: Constraints on moments or expectations

**Find**: Distribution with maximum entropy satisfying constraints

**Result**: Least informative distribution consistent with knowledge

**Example**: Max entropy with mean $\mu$ and variance $\sigma^2$ → Gaussian $\mathcal{N}(\mu, \sigma^2)$

### Minimum Description Length (MDL)

**Model selection**: Choose model that minimizes:
$$\text{Description Length} = \text{Data encoding cost} + \text{Model encoding cost}$$

Related to Bayesian Information Criterion (BIC).

### Information Bottleneck

**Goal**: Compress $X$ to $T$ while preserving information about $Y$

**Objective**:
$$\min_{p(t|x)} I(X; T) - \beta I(T; Y)$$

Trade-off between compression and relevance.

## Applications

### 1. Feature Selection

**Goal**: Select features most informative about target

**Method**: Rank features by $I(X_i; Y)$

**Advantages over correlation**:
- Detects nonlinear relationships
- Handles categorical variables naturally

**Example**:
```
Features: X₁, X₂, X₃, X₄
Target: Y

I(X₁; Y) = 0.8
I(X₂; Y) = 0.3
I(X₃; Y) = 1.2  ← most informative
I(X₄; Y) = 0.1

Select X₃, then X₁
```

### 2. Dependency Detection

**Test independence**: $X \perp Y$ iff $I(X; Y) = 0$

**Hypothesis test**:
- Null: $I(X; Y) = 0$ (independent)
- Alternative: $I(X; Y) > 0$ (dependent)

**Test statistic**: $2n \cdot I(X; Y) / \ln(2)$ approximately $\chi^2$ distributed.

### 3. Clustering

**Information-theoretic clustering** minimizes within-cluster entropy.

**Objective**:
$$\min \sum_{k=1}^K \pi_k H(X | C=k)$$

where $\pi_k$ is cluster proportion.

### 4. Transfer Entropy

**Causality detection** in time series:

$$TE_{X \to Y} = I(Y_{t+1}; X_t | Y_t)$$

Measures information flow from $X$ to $Y$.

### 5. Data Compression

**Shannon's source coding theorem**:

Expected code length $\geq H(X)$ (entropy is fundamental limit).

**Huffman coding**, **arithmetic coding** approach this limit.

### 6. Neural Network Analysis

**Information plane**: Track $I(X; T)$ and $I(T; Y)$ during training

where $T$ is hidden layer representation.

**Observations**:
- Initial phase: Increase both (fitting)
- Later phase: Decrease $I(X; T)$, maintain $I(T; Y)$ (compression)

## Multivariate Extensions

### Joint Mutual Information

$$I(X_1, X_2; Y) = H(Y) - H(Y | X_1, X_2)$$

### Conditional Mutual Information

$$I(X; Y | Z) = H(X|Z) - H(X|Y,Z)$$

**Interpretation**: Information shared by $X$ and $Y$ not contained in $Z$

### Total Correlation

$$C(X_1, ..., X_n) = \sum_{i=1}^n H(X_i) - H(X_1, ..., X_n)$$

Measures total dependence among variables.

### Interaction Information

For three variables:
$$I(X; Y; Z) = I(X; Y|Z) - I(X; Y)$$

Can be positive (synergy) or negative (redundancy).

## Relationship to Other Concepts

### Information and Probability

$$I(E) = -\log p(E)$$

**Self-information** of event $E$.

Rare events carry more information.

### Fisher Information

For parameter estimation:

$$\mathcal{I}(\theta) = \mathbb{E}\left[\left(\frac{\partial \log p(X|\theta)}{\partial \theta}\right)^2\right]$$

Measures precision of estimating $\theta$.

**Cramér-Rao bound**: Variance of any unbiased estimator $\geq 1/\mathcal{I}(\theta)$

### Entropy and Thermodynamics

**Boltzmann entropy**: $S = k_B \ln W$

**Connection**: Statistical mechanics entropy ≈ Shannon entropy of microstates.

### Entropy Rate

For stochastic process $\{X_t\}$:

$$h = \lim_{n \to \infty} \frac{1}{n} H(X_1, ..., X_n)$$

**For Markov chains**: $h = -\sum_{i,j} \pi_i p_{ij} \log p_{ij}$

## Theoretical Results

### Source Coding Theorem

Expected code length $L \geq H(X)$

Equality achieved by Shannon coding.

### Channel Capacity

Maximum rate of reliable communication:

$$C = \max_{p(x)} I(X; Y)$$

where $Y$ is channel output given input $X$.

### Data Processing Inequality

If $X \to Y \to Z$ (Markov chain):

$$I(X; Y) \geq I(X; Z)$$

Processing cannot increase mutual information.

### Fano's Inequality

For estimating $X$ from $Y$ with error probability $P_e$:

$$H(X|Y) \leq H(P_e) + P_e \log(|\mathcal{X}| - 1)$$

Lower bound on conditional entropy given error rate.

## Computational Considerations

### Complexity

**Entropy estimation**: $O(n + m)$ where $n$ = samples, $m$ = bins

**Mutual information**: $O(n + m^2)$ for 2D histogram

**High dimensions**: Curse of dimensionality (need $m^d$ bins for $d$ dimensions)

### Numerical Stability

**Issue**: $\log 0$ is undefined

**Solutions**:
- Add small constant: $p + \epsilon$
- Use convention: $0 \log 0 = 0$
- Laplace smoothing: $(n_i + \alpha) / (n + \alpha m)$

## Key References

1. **Shannon, C. E.** (1948). *A mathematical theory of communication*. Bell System Technical Journal, 27(3), 379-423.
   - Foundational paper

2. **Cover, T. M., & Thomas, J. A.** (2006). *Elements of Information Theory* (2nd ed.). Wiley.
   - Comprehensive textbook

3. **MacKay, D. J.** (2003). *Information Theory, Inference and Learning Algorithms*. Cambridge University Press.
   - Applications to machine learning

4. **Kraskov, A., Stögbauer, H., & Grassberger, P.** (2004). *Estimating mutual information*. Physical Review E, 69(6), 066138.
   - k-NN based MI estimation

5. **Paninski, L.** (2003). *Estimation of entropy and mutual information*. Neural Computation, 15(6), 1191-1253.
   - Bias correction methods

## Summary

Information theory provides fundamental limits and tools for:

**Core concepts**:
- **Entropy**: Uncertainty/information content
- **Mutual Information**: Shared information/dependence
- **KL Divergence**: Difference between distributions

**Key properties**:
- Entropy is maximized by uniform distribution
- Conditioning reduces entropy
- Mutual information detects any dependency

**Applications**:
- Feature selection and dimensionality reduction
- Model selection and compression
- Causality and dependency detection
- Machine learning (cross-entropy loss)

## See Also

- [Information Theory API Documentation](../information_theory.md) - Implementation and usage
- [HMM Theory](hmm.md) - Applications to sequential models
- [MCMC Theory](mcmc.md) - Sampling and inference methods
