//! Differentiable loss / objective primitives for training.
//!
//! `rust_objectives` holds *benchmark* surfaces (Sphere, Rosenbrock, …) used to
//! exercise optimisers. This module holds the objectives you actually minimise
//! when fitting a model: each one returns both its value and its gradient with
//! respect to the predictions, so any optimiser in this crate — `lbfgs`,
//! SHADE, the gradient methods — can drive a training loop without pulling in
//! a deep-learning framework.
//!
//! Every loss here is **non-negative** and every gradient is checked against
//! central finite differences in the tests. That is not a formality: the
//! motivating bug was a hand-rolled `cross_entropy_loss` elsewhere in the
//! stack that returned `+mean(log q)` instead of `-mean(log q)`, so gradient
//! descent on it trained the model *away* from its targets, and its unit test
//! asserted the wrong sign.
//!
//! # Example
//!
//! ```
//! use optimizr::losses::{CrossEntropy, Loss};
//!
//! // Two positions, three classes, uniform logits.
//! let logits = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
//! let targets = vec![0usize, 2];
//! let ce = CrossEntropy::new(3);
//! let (value, grad) = ce.value_and_grad(&logits, &targets).unwrap();
//!
//! // -ln(1/3) for a uniform distribution over three classes.
//! assert!((value - (3.0f64).ln()).abs() < 1e-12);
//! // The gradient sums to zero within each position: it is softmax - onehot.
//! assert!(grad[0..3].iter().sum::<f64>().abs() < 1e-12);
//! ```

use std::fmt;

/// Failure modes shared by the losses in this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LossError {
    /// `predictions.len()` is not `targets.len() * n_classes`.
    ShapeMismatch {
        predictions: usize,
        expected: usize,
    },
    /// A class index was outside `0..n_classes`.
    TargetOutOfRange { target: usize, n_classes: usize },
    /// Nothing to average over: every position was masked, or the input was
    /// empty.
    NoUnmaskedPositions,
    /// A parameter was outside its valid range.
    InvalidParameter(&'static str),
}

impl fmt::Display for LossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { predictions, expected } => write!(
                f,
                "predictions has {predictions} entries, expected {expected}"
            ),
            Self::TargetOutOfRange { target, n_classes } => write!(
                f,
                "target {target} is out of range for {n_classes} classes"
            ),
            Self::NoUnmaskedPositions => {
                write!(f, "no unmasked positions to average over")
            }
            Self::InvalidParameter(p) => write!(f, "invalid parameter: {p}"),
        }
    }
}

impl std::error::Error for LossError {}

pub type Result<T> = std::result::Result<T, LossError>;

/// A scalar objective of a prediction vector, with its gradient.
///
/// `T` is the target representation: class indices for classification,
/// real values for regression.
pub trait Loss<T: ?Sized> {
    /// The loss value. Always finite and, for every loss here, non-negative.
    fn value(&self, predictions: &[f64], targets: &T) -> Result<f64>;

    /// `∂loss/∂predictions`, the same length as `predictions`.
    fn grad(&self, predictions: &[f64], targets: &T) -> Result<Vec<f64>>;

    /// Both at once. Override when the two share most of their work — the
    /// default calls each separately.
    fn value_and_grad(&self, predictions: &[f64], targets: &T) -> Result<(f64, Vec<f64>)> {
        Ok((self.value(predictions, targets)?, self.grad(predictions, targets)?))
    }

    /// A short name for logs and reports.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// classification
// ---------------------------------------------------------------------------

/// Multi-class cross-entropy over logits, with optional label smoothing and
/// masking.
///
/// For `N` unmasked positions, `V` classes, logits `z` and targets `y`:
///
/// ```text
/// q      = softmax(z)
/// L      = -(1/N) Σ_n Σ_i p̃_{n,i} log q_{n,i}
/// p̃_{n,i} = (1-ε)·1[i = y_n] + ε/V
/// ∂L/∂z_{n,i} = (q_{n,i} - p̃_{n,i}) / N
/// ```
///
/// With `ε = 0` this is the usual `-(1/N) Σ log q_{n,y_n}`. The log-softmax is
/// computed by subtracting the row maximum, so large logits do not overflow.
///
/// Masking: positions whose target equals [`CrossEntropy::ignore_index`] are
/// excluded from both the sum and `N`, and receive a zero gradient. That is
/// how padding and prompt tokens are kept out of a language-model loss.
#[derive(Debug, Clone)]
pub struct CrossEntropy {
    n_classes: usize,
    label_smoothing: f64,
    ignore_index: Option<usize>,
}

impl CrossEntropy {
    /// Plain cross-entropy over `n_classes`, no smoothing, nothing masked.
    pub fn new(n_classes: usize) -> Self {
        Self { n_classes, label_smoothing: 0.0, ignore_index: None }
    }

    /// Set label smoothing `ε ∈ [0, 1)`.
    pub fn with_label_smoothing(mut self, epsilon: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&epsilon) {
            return Err(LossError::InvalidParameter("label_smoothing must be in [0, 1)"));
        }
        self.label_smoothing = epsilon;
        Ok(self)
    }

    /// Exclude positions whose target is `index` from the loss and its
    /// gradient.
    pub fn with_ignore_index(mut self, index: usize) -> Self {
        self.ignore_index = Some(index);
        self
    }

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
    pub fn label_smoothing(&self) -> f64 {
        self.label_smoothing
    }
    pub fn ignore_index(&self) -> Option<usize> {
        self.ignore_index
    }

    fn check(&self, predictions: &[f64], targets: &[usize]) -> Result<()> {
        let expected = targets.len() * self.n_classes;
        if predictions.len() != expected {
            return Err(LossError::ShapeMismatch {
                predictions: predictions.len(),
                expected,
            });
        }
        for &t in targets {
            if Some(t) == self.ignore_index {
                continue;
            }
            if t >= self.n_classes {
                return Err(LossError::TargetOutOfRange {
                    target: t,
                    n_classes: self.n_classes,
                });
            }
        }
        Ok(())
    }

    /// Numerically stable log-softmax of one row, returned in place.
    fn log_softmax_row(row: &[f64], out: &mut [f64]) {
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for (o, &z) in out.iter_mut().zip(row) {
            let e = (z - max).exp();
            *o = e;
            sum += e;
        }
        let log_sum = sum.ln() + max;
        for (o, &z) in out.iter_mut().zip(row) {
            *o = z - log_sum;
        }
    }
}

impl Loss<[usize]> for CrossEntropy {
    fn value(&self, predictions: &[f64], targets: &[usize]) -> Result<f64> {
        Ok(self.value_and_grad_inner(predictions, targets, false)?.0)
    }

    fn grad(&self, predictions: &[f64], targets: &[usize]) -> Result<Vec<f64>> {
        Ok(self.value_and_grad_inner(predictions, targets, true)?.1)
    }

    fn value_and_grad(
        &self,
        predictions: &[f64],
        targets: &[usize],
    ) -> Result<(f64, Vec<f64>)> {
        self.value_and_grad_inner(predictions, targets, true)
    }

    fn name(&self) -> &'static str {
        "cross_entropy"
    }
}

impl CrossEntropy {
    fn value_and_grad_inner(
        &self,
        predictions: &[f64],
        targets: &[usize],
        want_grad: bool,
    ) -> Result<(f64, Vec<f64>)> {
        self.check(predictions, targets)?;
        let v = self.n_classes;
        let eps = self.label_smoothing;

        let n_unmasked = targets
            .iter()
            .filter(|&&t| Some(t) != self.ignore_index)
            .count();
        if n_unmasked == 0 {
            return Err(LossError::NoUnmaskedPositions);
        }
        let scale = 1.0 / n_unmasked as f64;

        let mut grad = if want_grad {
            vec![0.0; predictions.len()]
        } else {
            Vec::new()
        };
        let mut logq = vec![0.0; v];
        let mut total = 0.0;

        for (n, &target) in targets.iter().enumerate() {
            let row = &predictions[n * v..(n + 1) * v];
            if Some(target) == self.ignore_index {
                continue; // zero contribution, zero gradient
            }
            Self::log_softmax_row(row, &mut logq);

            // -Σ_i p̃_i log q_i, with p̃ the (possibly smoothed) target.
            if eps == 0.0 {
                total -= logq[target];
            } else {
                let uniform = eps / v as f64;
                let mut acc = 0.0;
                for (i, &lq) in logq.iter().enumerate() {
                    let p = if i == target { 1.0 - eps + uniform } else { uniform };
                    acc += p * lq;
                }
                total -= acc;
            }

            if want_grad {
                let g = &mut grad[n * v..(n + 1) * v];
                let uniform = eps / v as f64;
                for (i, (gi, &lq)) in g.iter_mut().zip(logq.iter()).enumerate() {
                    let q = lq.exp();
                    let p = if i == target { 1.0 - eps + uniform } else { uniform };
                    *gi = (q - p) * scale;
                }
            }
        }

        Ok((total * scale, grad))
    }
}

/// Binary cross-entropy over logits, averaged over all positions.
///
/// ```text
/// σ(z) = 1/(1+e^{-z})
/// L    = -(1/N) Σ_n [ y_n log σ(z_n) + (1-y_n) log(1-σ(z_n)) ]
/// ∂L/∂z_n = (σ(z_n) - y_n) / N
/// ```
///
/// Evaluated as `max(z,0) - z·y + log(1 + e^{-|z|})`, which is the
/// overflow-free form.
#[derive(Debug, Clone, Default)]
pub struct BinaryCrossEntropy;

impl Loss<[f64]> for BinaryCrossEntropy {
    fn value(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(LossError::ShapeMismatch {
                predictions: predictions.len(),
                expected: targets.len(),
            });
        }
        if predictions.is_empty() {
            return Err(LossError::NoUnmaskedPositions);
        }
        let n = predictions.len() as f64;
        let total: f64 = predictions
            .iter()
            .zip(targets)
            .map(|(&z, &y)| z.max(0.0) - z * y + (-z.abs()).exp().ln_1p())
            .sum();
        Ok(total / n)
    }

    fn grad(&self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>> {
        if predictions.len() != targets.len() {
            return Err(LossError::ShapeMismatch {
                predictions: predictions.len(),
                expected: targets.len(),
            });
        }
        if predictions.is_empty() {
            return Err(LossError::NoUnmaskedPositions);
        }
        let n = predictions.len() as f64;
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&z, &y)| (1.0 / (1.0 + (-z).exp()) - y) / n)
            .collect())
    }

    fn name(&self) -> &'static str {
        "binary_cross_entropy"
    }
}

// ---------------------------------------------------------------------------
// regression
// ---------------------------------------------------------------------------

macro_rules! check_same_len {
    ($p:expr, $t:expr) => {{
        if $p.len() != $t.len() {
            return Err(LossError::ShapeMismatch {
                predictions: $p.len(),
                expected: $t.len(),
            });
        }
        if $p.is_empty() {
            return Err(LossError::NoUnmaskedPositions);
        }
        $p.len() as f64
    }};
}

/// `L = (1/N) Σ (ŷ - y)²`, with `∂L/∂ŷ = 2(ŷ - y)/N`.
#[derive(Debug, Clone, Default)]
pub struct MeanSquaredError;

impl Loss<[f64]> for MeanSquaredError {
    fn value(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        let n = check_same_len!(predictions, targets);
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&p, &t)| (p - t) * (p - t))
            .sum::<f64>()
            / n)
    }

    fn grad(&self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>> {
        let n = check_same_len!(predictions, targets);
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&p, &t)| 2.0 * (p - t) / n)
            .collect())
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}

/// `L = (1/N) Σ |ŷ - y|`. The gradient at zero is taken as zero.
#[derive(Debug, Clone, Default)]
pub struct MeanAbsoluteError;

impl Loss<[f64]> for MeanAbsoluteError {
    fn value(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        let n = check_same_len!(predictions, targets);
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&p, &t)| (p - t).abs())
            .sum::<f64>()
            / n)
    }

    fn grad(&self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>> {
        let n = check_same_len!(predictions, targets);
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&p, &t)| {
                let d = p - t;
                if d > 0.0 {
                    1.0 / n
                } else if d < 0.0 {
                    -1.0 / n
                } else {
                    0.0 // subgradient choice at the kink
                }
            })
            .collect())
    }

    fn name(&self) -> &'static str {
        "mae"
    }
}

/// Huber loss: squared near zero, linear in the tails, so large residuals do
/// not dominate the fit.
///
/// ```text
/// L_n = ½ r²                 |r| ≤ δ
///       δ(|r| - ½δ)          otherwise          r = ŷ - y
/// ```
#[derive(Debug, Clone)]
pub struct Huber {
    delta: f64,
}

impl Huber {
    pub fn new(delta: f64) -> Result<Self> {
        if !(delta > 0.0) || !delta.is_finite() {
            return Err(LossError::InvalidParameter("delta must be finite and > 0"));
        }
        Ok(Self { delta })
    }
    pub fn delta(&self) -> f64 {
        self.delta
    }
}

impl Default for Huber {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Loss<[f64]> for Huber {
    fn value(&self, predictions: &[f64], targets: &[f64]) -> Result<f64> {
        let n = check_same_len!(predictions, targets);
        let d = self.delta;
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&p, &t)| {
                let r = (p - t).abs();
                if r <= d {
                    0.5 * r * r
                } else {
                    d * (r - 0.5 * d)
                }
            })
            .sum::<f64>()
            / n)
    }

    fn grad(&self, predictions: &[f64], targets: &[f64]) -> Result<Vec<f64>> {
        let n = check_same_len!(predictions, targets);
        let d = self.delta;
        Ok(predictions
            .iter()
            .zip(targets)
            .map(|(&p, &t)| {
                let r = p - t;
                let g = if r.abs() <= d { r } else { d * r.signum() };
                g / n
            })
            .collect())
    }

    fn name(&self) -> &'static str {
        "huber"
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Central differences on an arbitrary loss, for gradient checking.
    fn fd_grad<T: ?Sized, L: Loss<T>>(loss: &L, p: &[f64], t: &T) -> Vec<f64> {
        let h = 1e-6;
        (0..p.len())
            .map(|i| {
                let mut up = p.to_vec();
                let mut dn = p.to_vec();
                up[i] += h;
                dn[i] -= h;
                (loss.value(&up, t).unwrap() - loss.value(&dn, t).unwrap()) / (2.0 * h)
            })
            .collect()
    }

    fn assert_grad_matches_fd<T: ?Sized, L: Loss<T>>(loss: &L, p: &[f64], t: &T, tol: f64) {
        let analytic = loss.grad(p, t).unwrap();
        let numeric = fd_grad(loss, p, t);
        for (i, (a, n)) in analytic.iter().zip(numeric.iter()).enumerate() {
            assert!(
                (a - n).abs() < tol,
                "{}: component {i} analytic {a} vs finite-difference {n}",
                loss.name()
            );
        }
    }

    #[test]
    fn cross_entropy_uniform_logits_is_ln_v() {
        let ce = CrossEntropy::new(10);
        let logits = vec![0.0; 40]; // 4 positions x 10 classes
        let targets = [1usize, 2, 3, 4];
        let v = ce.value(&logits, &targets).unwrap();
        assert!((v - (10.0f64).ln()).abs() < 1e-12, "got {v}");
        assert!(v > 0.0, "cross-entropy must be non-negative");
    }

    #[test]
    fn cross_entropy_is_never_negative_and_penalises_being_wrong() {
        let ce = CrossEntropy::new(4);
        let right = [8.0, 0.0, 0.0, 0.0];
        let wrong = [0.0, 0.0, 0.0, 8.0];
        let target = [0usize];

        let l_right = ce.value(&right, &target).unwrap();
        let l_wrong = ce.value(&wrong, &target).unwrap();
        assert!(l_right >= 0.0 && l_wrong >= 0.0);
        assert!(
            l_wrong > l_right,
            "a confidently wrong prediction must cost more: {l_wrong} vs {l_right}"
        );
    }

    #[test]
    fn cross_entropy_gradient_is_softmax_minus_onehot() {
        let ce = CrossEntropy::new(3);
        let logits = [0.3, -1.1, 2.0, 0.5, 0.25, -0.75];
        let targets = [2usize, 0];
        let g = ce.grad(&logits, &targets).unwrap();
        // Each row of softmax - onehot sums to zero.
        assert!(g[0..3].iter().sum::<f64>().abs() < 1e-12);
        assert!(g[3..6].iter().sum::<f64>().abs() < 1e-12);
        assert_grad_matches_fd(&ce, &logits, targets.as_slice(), 1e-7);
    }

    #[test]
    fn cross_entropy_label_smoothing_changes_the_loss_and_the_gradient() {
        let plain = CrossEntropy::new(5);
        let smooth = CrossEntropy::new(5).with_label_smoothing(0.1).unwrap();
        let logits = [2.0, 0.5, -1.0, 0.25, 0.0];
        let targets = [0usize];

        let lp = plain.value(&logits, &targets).unwrap();
        let ls = smooth.value(&logits, &targets).unwrap();
        assert!(
            (lp - ls).abs() > 1e-6,
            "smoothing must change the value: {lp} vs {ls}"
        );
        assert!(ls > 0.0);
        assert_grad_matches_fd(&smooth, &logits, targets.as_slice(), 1e-7);

        assert!(matches!(
            CrossEntropy::new(5).with_label_smoothing(1.5),
            Err(LossError::InvalidParameter(_))
        ));
    }

    #[test]
    fn cross_entropy_masking_excludes_positions_from_value_and_gradient() {
        const PAD: usize = 0;
        let ce = CrossEntropy::new(4).with_ignore_index(PAD);
        // Three positions; the middle one is padding.
        let logits = [1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 2.0, 0.0];
        let targets = [2usize, PAD, 2];

        let masked = ce.value(&logits, &targets).unwrap();

        // The same loss computed over only the two real positions.
        let ce_plain = CrossEntropy::new(4);
        let kept: Vec<f64> = logits[0..4].iter().chain(&logits[8..12]).copied().collect();
        let direct = ce_plain.value(&kept, &[2usize, 2]).unwrap();
        assert!(
            (masked - direct).abs() < 1e-12,
            "masked {masked} should equal the unpadded loss {direct}"
        );

        // Padding gets exactly zero gradient.
        let g = ce.grad(&logits, &targets).unwrap();
        assert!(g[4..8].iter().all(|x| *x == 0.0), "pad gradient must be zero");
        assert!(g[0..4].iter().any(|x| *x != 0.0));

        // Everything masked is an error, not a divide-by-zero.
        assert!(matches!(
            ce.value(&logits, &[PAD, PAD, PAD]),
            Err(LossError::NoUnmaskedPositions)
        ));
    }

    #[test]
    fn cross_entropy_is_stable_at_extreme_logits() {
        let ce = CrossEntropy::new(3);
        for scale in [100.0, 1e3, 1e6] {
            let logits = [scale, -scale, 0.0];
            let v = ce.value(&logits, &[0usize]).unwrap();
            assert!(v.is_finite(), "overflow at scale {scale}: {v}");
            assert!(v >= 0.0);
            let g = ce.grad(&logits, &[0usize]).unwrap();
            assert!(g.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn cross_entropy_rejects_bad_shapes_and_targets() {
        let ce = CrossEntropy::new(3);
        assert!(matches!(
            ce.value(&[0.0; 5], &[0usize]),
            Err(LossError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            ce.value(&[0.0; 3], &[7usize]),
            Err(LossError::TargetOutOfRange { target: 7, n_classes: 3 })
        ));
    }

    #[test]
    fn regression_losses_match_hand_values_and_finite_differences() {
        let p = [1.0, 2.0, 3.0];
        let t = [1.5, 1.0, 3.0];

        let mse = MeanSquaredError;
        // (0.25 + 1 + 0) / 3
        assert!((mse.value(&p, &t).unwrap() - 1.25 / 3.0).abs() < 1e-12);
        assert_grad_matches_fd(&mse, &p, t.as_slice(), 1e-6);

        let mae = MeanAbsoluteError;
        // (0.5 + 1 + 0) / 3
        assert!((mae.value(&p, &t).unwrap() - 1.5 / 3.0).abs() < 1e-12);

        let huber = Huber::new(0.75).unwrap();
        assert!(huber.value(&p, &t).unwrap() > 0.0);
        assert_grad_matches_fd(&huber, &p, t.as_slice(), 1e-6);

        assert!(matches!(
            Huber::new(0.0),
            Err(LossError::InvalidParameter(_))
        ));
    }

    #[test]
    fn huber_is_quadratic_inside_delta_and_linear_outside() {
        let h = Huber::new(1.0).unwrap();
        // |r| = 0.5 <= delta: 0.5 * r^2
        assert!((h.value(&[0.5], &[0.0]).unwrap() - 0.125).abs() < 1e-12);
        // |r| = 3 > delta: delta * (|r| - delta/2) = 2.5
        assert!((h.value(&[3.0], &[0.0]).unwrap() - 2.5).abs() < 1e-12);
        // Linear tail means a bounded gradient, which is the point.
        let g = h.grad(&[100.0], &[0.0]).unwrap();
        assert!((g[0] - 1.0).abs() < 1e-12, "tail gradient saturates at delta");
    }

    #[test]
    fn binary_cross_entropy_matches_finite_differences_and_is_stable() {
        let bce = BinaryCrossEntropy;
        let p = [0.4, -1.2, 2.5, 0.0];
        let t = [1.0, 0.0, 1.0, 0.0];
        assert!(bce.value(&p, &t).unwrap() > 0.0);
        assert_grad_matches_fd(&bce, &p, t.as_slice(), 1e-6);

        // z = 0, y = 0.5 -> ln 2
        assert!((bce.value(&[0.0], &[0.5]).unwrap() - (2.0f64).ln()).abs() < 1e-12);
        for z in [1e3, -1e3] {
            assert!(bce.value(&[z], &[1.0]).unwrap().is_finite());
        }
    }

    #[test]
    fn value_and_grad_agrees_with_the_separate_calls() {
        let ce = CrossEntropy::new(4).with_label_smoothing(0.05).unwrap();
        let logits = [0.1, 0.9, -0.4, 1.3, 0.0, -2.0, 0.5, 0.2];
        let targets = [1usize, 3];
        let (v, g) = ce.value_and_grad(&logits, &targets).unwrap();
        assert!((v - ce.value(&logits, &targets).unwrap()).abs() < 1e-15);
        for (a, b) in g.iter().zip(ce.grad(&logits, &targets).unwrap()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn gradient_descent_on_cross_entropy_actually_learns() {
        // The end-to-end property the sign bug violated: stepping against the
        // gradient must raise the probability of the correct class.
        let ce = CrossEntropy::new(3);
        let targets = [0usize];
        let mut logits = vec![0.0, 0.0, 0.0];
        let first = ce.value(&logits, &targets).unwrap();

        for _ in 0..200 {
            let g = ce.grad(&logits, &targets).unwrap();
            for (z, gi) in logits.iter_mut().zip(&g) {
                *z -= 0.5 * gi;
            }
        }
        let last = ce.value(&logits, &targets).unwrap();
        assert!(last < first, "loss must fall: {first} -> {last}");
        assert!(
            logits[0] > logits[1] && logits[0] > logits[2],
            "the target logit must end up largest: {logits:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
// Exposed so `import optimizr` gives Python the *same* objective the Rust
// trainers minimise, rather than a numpy re-implementation that can drift.
#[cfg(feature = "python-bindings")]
mod py {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    fn to_py(e: LossError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }

    /// Mean squared error and its gradient: `(value, grad)`.
    #[pyfunction]
    pub fn mse(predictions: Vec<f64>, targets: Vec<f64>) -> PyResult<(f64, Vec<f64>)> {
        MeanSquaredError
            .value_and_grad(&predictions, &targets)
            .map_err(to_py)
    }

    /// Mean absolute error and its subgradient.
    #[pyfunction]
    pub fn mae(predictions: Vec<f64>, targets: Vec<f64>) -> PyResult<(f64, Vec<f64>)> {
        MeanAbsoluteError
            .value_and_grad(&predictions, &targets)
            .map_err(to_py)
    }

    /// Huber loss with transition point `delta`.
    #[pyfunction]
    #[pyo3(signature = (predictions, targets, delta=1.0))]
    pub fn huber(
        predictions: Vec<f64>,
        targets: Vec<f64>,
        delta: f64,
    ) -> PyResult<(f64, Vec<f64>)> {
        Huber::new(delta)
            .map_err(to_py)?
            .value_and_grad(&predictions, &targets)
            .map_err(to_py)
    }

    /// Binary cross-entropy over logits.
    #[pyfunction]
    pub fn binary_cross_entropy(
        logits: Vec<f64>,
        targets: Vec<f64>,
    ) -> PyResult<(f64, Vec<f64>)> {
        BinaryCrossEntropy
            .value_and_grad(&logits, &targets)
            .map_err(to_py)
    }

    /// Multi-class cross-entropy over flattened `[n_positions * n_classes]`
    /// logits, with optional label smoothing and an ignored class index.
    #[pyfunction]
    #[pyo3(signature = (logits, targets, n_classes, label_smoothing=0.0, ignore_index=None))]
    pub fn cross_entropy(
        logits: Vec<f64>,
        targets: Vec<usize>,
        n_classes: usize,
        label_smoothing: f64,
        ignore_index: Option<usize>,
    ) -> PyResult<(f64, Vec<f64>)> {
        let mut ce = CrossEntropy::new(n_classes);
        if label_smoothing > 0.0 {
            ce = ce.with_label_smoothing(label_smoothing).map_err(to_py)?;
        }
        if let Some(ix) = ignore_index {
            ce = ce.with_ignore_index(ix);
        }
        ce.value_and_grad(&logits, &targets).map_err(to_py)
    }

    /// Sum of weighted MSE terms, the standard composite PINN objective:
    /// `Σ_k w_k · MSE(residual_k, 0)`. Returns the total and the gradient of
    /// each block, concatenated in the order given.
    ///
    /// This is one call instead of one per term, which matters when a PINN
    /// evaluates four or five residual blocks per optimiser step.
    #[pyfunction]
    pub fn weighted_mse_sum(
        blocks: Vec<Vec<f64>>,
        weights: Vec<f64>,
    ) -> PyResult<(f64, Vec<Vec<f64>>)> {
        if blocks.len() != weights.len() {
            return Err(PyValueError::new_err(format!(
                "{} blocks but {} weights",
                blocks.len(),
                weights.len()
            )));
        }
        let mut total = 0.0;
        let mut grads = Vec::with_capacity(blocks.len());
        for (block, &w) in blocks.iter().zip(&weights) {
            let zeros = vec![0.0; block.len()];
            let (v, g) = MeanSquaredError
                .value_and_grad(block, &zeros)
                .map_err(to_py)?;
            total += w * v;
            grads.push(g.into_iter().map(|x| w * x).collect());
        }
        Ok((total, grads))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(mse, m)?)?;
        m.add_function(wrap_pyfunction!(mae, m)?)?;
        m.add_function(wrap_pyfunction!(huber, m)?)?;
        m.add_function(wrap_pyfunction!(binary_cross_entropy, m)?)?;
        m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
        m.add_function(wrap_pyfunction!(weighted_mse_sum, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python-bindings")]
pub use py::register as register_losses;
