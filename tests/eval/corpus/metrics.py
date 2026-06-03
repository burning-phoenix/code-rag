"""
Training diagnostics and optimization health monitoring.

Provides utilities for tracking gradient flow, convergence rate estimation,
weight magnitude analysis, and learning rate sensitivity diagnostics during
neural network training. All functions are purely observational — they read
model and optimizer state but never modify parameters, gradients, or buffers.

Typical usage with an SGD or Adam optimizer loop:

    tracker = ConvergenceTracker(window=50)
    grad_monitor = GradientFlowMonitor(model)

    for step in range(max_iters):
        loss = train_step(model, optimizer, batch)
        tracker.update(loss)
        if step % log_interval == 0:
            grad_stats = grad_monitor.snapshot()
            health = assess_optimization_health(
                model, optimizer, tracker, grad_stats
            )
            print(format_health_report(health, step))

References:
    - Bottou et al., "Optimization Methods for Large-Scale ML" (convergence
      rate estimation under strongly convex and Lipschitz gradient assumptions)
    - Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (weight
      decay vs L2 regularization monitoring)
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Gradient flow analysis
# ---------------------------------------------------------------------------

@dataclass
class LayerGradientStats:
    """Per-layer gradient statistics snapshot (read-only)."""
    name: str
    mean_abs: float       # mean |grad| across parameters
    max_abs: float        # max  |grad| across parameters
    l2_norm: float        # Frobenius / L2 norm of the gradient tensor
    fraction_zero: float  # fraction of exactly-zero gradient entries
    param_count: int      # number of scalar parameters in this layer


class GradientFlowMonitor:
    """Monitors gradient magnitudes across network layers without modifying
    any model state.

    This tracks the same gradient norm statistics used to diagnose vanishing
    and exploding gradients (see e.g. the batch normalization paper's analysis
    of gradient flow through deep networks). The monitor never calls
    .backward(), .zero_grad(), or .step() — it only reads .grad attributes
    that were populated by a preceding backward pass.

    Attributes:
        layer_names: ordered list of parameter-group names being tracked.
    """

    def __init__(self, model, group_by: str = "module"):
        """
        Args:
            model: any object exposing .named_parameters().
            group_by: "module" groups by top-level module name,
                      "parameter" reports each parameter individually.
        """
        self._model = model
        self._group_by = group_by
        self.layer_names: list[str] = []
        self._param_groups: dict[str, list[str]] = {}
        self._index_parameters()

    def _index_parameters(self):
        """Build a read-only index of parameter names grouped by layer."""
        groups: dict[str, list[str]] = {}
        for name, _ in self._model.named_parameters():
            if self._group_by == "module":
                key = name.rsplit(".", 1)[0] if "." in name else name
            else:
                key = name
            groups.setdefault(key, []).append(name)
        self._param_groups = groups
        self.layer_names = list(groups.keys())

    # ---- core read-only snapshot ----------------------------------------

    def snapshot(self) -> list[LayerGradientStats]:
        """Return current gradient statistics per layer.

        Must be called *after* loss.backward() and *before*
        optimizer.zero_grad().  Does NOT modify any tensor.

        The per-layer L2 norm reported here is the same quantity that
        torch.nn.utils.clip_grad_norm_ compares against the max_norm
        threshold — but unlike clip_grad_norm_, this function never
        rescales the gradient.
        """
        stats: list[LayerGradientStats] = []
        param_dict = dict(self._model.named_parameters())

        for layer_name, param_names in self._param_groups.items():
            total_abs = 0.0
            total_sq = 0.0
            max_abs = 0.0
            n_zeros = 0
            n_elems = 0

            for pname in param_names:
                p = param_dict[pname]
                if p.grad is None:
                    continue
                g = p.grad.detach()
                flat = g.reshape(-1)
                numel = flat.shape[0]

                total_abs += sum(abs(float(v)) for v in flat)
                total_sq += sum(float(v) ** 2 for v in flat)
                local_max = max(abs(float(v)) for v in flat)
                if local_max > max_abs:
                    max_abs = local_max
                n_zeros += sum(1 for v in flat if float(v) == 0.0)
                n_elems += numel

            if n_elems == 0:
                continue

            stats.append(LayerGradientStats(
                name=layer_name,
                mean_abs=total_abs / n_elems,
                max_abs=max_abs,
                l2_norm=math.sqrt(total_sq),
                fraction_zero=n_zeros / n_elems,
                param_count=n_elems,
            ))
        return stats

    def total_gradient_norm(self) -> float:
        """Compute the total gradient L2 norm across all parameters.

        This mirrors the global norm computation used by gradient clipping
        (clip_grad_norm_) and by the convergence analysis in Bottou et al.,
        which bounds the expected squared gradient norm under Lipschitz
        continuity assumptions.

        Returns 0.0 if no gradients are populated.
        """
        total_sq = 0.0
        for _, p in self._model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq += sum(float(v) ** 2 for v in g.reshape(-1))
        return math.sqrt(total_sq)


# ---------------------------------------------------------------------------
# Weight magnitude and decay monitoring
# ---------------------------------------------------------------------------

@dataclass
class WeightStats:
    """Snapshot of weight magnitudes per layer (read-only).

    The L2 norm of the weight vector is the quantity penalized by L2
    regularization (Tikhonov regularization / weight decay).  Tracking it
    during training helps diagnose whether the effective regularization
    strength is appropriate — weights that grow unbounded may indicate
    insufficient weight_decay, while weights that collapse toward zero may
    signal excessive penalty.
    """
    name: str
    l2_norm: float        # ||W||_2  (Frobenius norm for matrices)
    mean_abs: float       # mean |W|
    max_abs: float        # max  |W|
    param_count: int


def weight_magnitude_snapshot(model) -> list[WeightStats]:
    """Read-only snapshot of weight magnitudes per named parameter group.

    This does NOT apply weight decay or modify any parameter.  It reports
    the same norms that an optimizer's weight_decay term penalizes:

        For SGD:  weight_decay acts as  g ← g + λ·w   (L2 gradient penalty)
        For AdamW: weight_decay acts as  w ← (1 - λ)·w (true weight decay)

    See Loshchilov & Hutter (2019) for why the distinction matters for
    adaptive optimizers.  This function simply measures ||w||₂ so you can
    observe the effect of either form during training.
    """
    stats: list[WeightStats] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        data = p.detach()
        flat = data.reshape(-1)
        numel = flat.shape[0]

        l2 = math.sqrt(sum(float(v) ** 2 for v in flat))
        mean_abs_val = sum(abs(float(v)) for v in flat) / numel
        max_abs_val = max(abs(float(v)) for v in flat)

        stats.append(WeightStats(
            name=name,
            l2_norm=l2,
            mean_abs=mean_abs_val,
            max_abs=max_abs_val,
            param_count=numel,
        ))
    return stats


def update_to_weight_ratio(model, optimizer) -> dict[str, float]:
    """Compute |lr * grad| / |weight| for each parameter group.

    A useful heuristic from Karpathy's "A Recipe for Training Neural
    Networks": the ratio of update magnitude to weight magnitude should be
    roughly 1e-3.  Values much larger suggest the learning rate is too high;
    values near machine epsilon suggest vanishing updates.

    This reads learning_rate from the optimizer's param_groups and grad /
    data from the model parameters.  It modifies nothing.
    """
    ratios: dict[str, float] = {}
    param_to_lr: dict[int, float] = {}

    # Build id→lr map from optimizer state (read-only traversal)
    for group in optimizer.param_groups:
        lr = float(group["lr"])
        for p in group["params"]:
            param_to_lr[id(p)] = lr

    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        lr = param_to_lr.get(id(p), 0.0)
        w_norm = math.sqrt(sum(float(v) ** 2 for v in p.detach().reshape(-1)))
        g_norm = math.sqrt(
            sum(float(v) ** 2 for v in p.grad.detach().reshape(-1))
        )
        if w_norm > 0:
            ratios[name] = (lr * g_norm) / w_norm
        else:
            ratios[name] = float("inf")
    return ratios


# ---------------------------------------------------------------------------
# Convergence rate estimation
# ---------------------------------------------------------------------------

class ConvergenceTracker:
    """Estimates the empirical convergence rate from a stream of loss values.

    For strongly convex objectives with Lipschitz-continuous gradients, SGD
    theory predicts sublinear convergence at rate O(1/T).  In practice, the
    loss curve may exhibit linear convergence (constant ratio between
    successive losses) in early training and sublinear convergence later.

    This class fits a simple exponential model to a rolling window of losses
    to estimate the effective convergence rate.  It never modifies any model
    or optimizer state.

    Attributes:
        history: full list of recorded loss values.
        estimated_rate: most recent convergence rate estimate, or None.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self.history: list[float] = []
        self.estimated_rate: Optional[float] = None
        self._step_count: int = 0

    def update(self, loss: float):
        """Record a new loss value and recompute the rate estimate."""
        self.history.append(loss)
        self._step_count += 1
        if len(self.history) >= self.window:
            self._estimate_rate()

    def _estimate_rate(self):
        """Fit log-linear model to the last `window` losses.

        If the loss sequence {L_t} follows L_t ≈ C · r^t, then
        log L_t ≈ log C + t·log r, and we estimate log r via
        least-squares slope.  The rate r < 1 indicates convergence;
        r ≈ 1 indicates stagnation; r > 1 indicates divergence.
        """
        recent = self.history[-self.window:]
        # Filter non-positive values (can happen with regularized losses)
        positive = [(i, v) for i, v in enumerate(recent) if v > 0]
        if len(positive) < 2:
            self.estimated_rate = None
            return

        log_vals = [(i, math.log(v)) for i, v in positive]
        n = len(log_vals)
        sum_x = sum(i for i, _ in log_vals)
        sum_y = sum(y for _, y in log_vals)
        sum_xy = sum(i * y for i, y in log_vals)
        sum_x2 = sum(i * i for i, _ in log_vals)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-15:
            self.estimated_rate = None
            return

        slope = (n * sum_xy - sum_x * sum_y) / denom
        self.estimated_rate = math.exp(slope)

    def is_converging(self, threshold: float = 0.999) -> bool:
        """True if the estimated rate suggests the loss is still decreasing."""
        if self.estimated_rate is None:
            return False
        return self.estimated_rate < threshold

    def is_diverging(self, threshold: float = 1.001) -> bool:
        """True if the loss appears to be increasing (may need lower lr)."""
        if self.estimated_rate is None:
            return False
        return self.estimated_rate > threshold

    def steps_since_improvement(self, rel_tol: float = 1e-4) -> int:
        """Count how many steps since the last meaningful loss decrease.

        A decrease is 'meaningful' if loss_new < loss_best * (1 - rel_tol).
        This is the same logic used by early stopping in nanoGPT's
        training loop, where training halts if val loss fails to improve
        for a specified patience window.
        """
        if not self.history:
            return 0
        best = self.history[0]
        last_improvement = 0
        for i, loss in enumerate(self.history):
            if loss < best * (1.0 - rel_tol):
                best = loss
                last_improvement = i
        return len(self.history) - 1 - last_improvement


# ---------------------------------------------------------------------------
# Composite health assessment
# ---------------------------------------------------------------------------

@dataclass
class OptimizationHealth:
    """Aggregated snapshot of training health indicators."""
    step: int
    loss: float
    total_grad_norm: float
    max_layer_grad_norm: float
    min_layer_grad_norm: float
    total_weight_norm: float
    convergence_rate: Optional[float]
    is_converging: bool
    is_diverging: bool
    steps_since_improvement: int
    grad_to_weight_ratio: float  # median across layers
    # Diagnostic flags
    vanishing_gradients: bool    # grad norm < 1e-7
    exploding_gradients: bool    # grad norm > 1e3
    stale_training: bool         # no improvement for > 2x window


def assess_optimization_health(
    model,
    optimizer,
    tracker: ConvergenceTracker,
    grad_stats: list[LayerGradientStats],
) -> OptimizationHealth:
    """Produce a composite health report from current model/optimizer state.

    This is the main entry point for training-loop diagnostics.  It
    aggregates gradient flow, weight magnitude, convergence rate, and
    update ratio information into a single summary.  As with all utilities
    in this module, it is strictly read-only — it inspects but never
    modifies the model, optimizer, or their internal buffers (momentum
    buffers, Adam first/second moment estimates, learning rate schedules,
    etc.).

    Diagnostic thresholds are based on common heuristics:
        - Vanishing gradients: total norm < 1e-7 (training is effectively
          frozen; the backward pass produces near-zero signal).
        - Exploding gradients: total norm > 1e3 (likely need gradient
          clipping via clip_grad_norm_ or a lower learning rate).
        - Stale training: no loss improvement for > 2× the tracker window
          (may need learning rate warmup restart or early stopping).
    """
    total_grad = sum(s.l2_norm for s in grad_stats) if grad_stats else 0.0
    layer_norms = [s.l2_norm for s in grad_stats] if grad_stats else [0.0]

    weight_stats = weight_magnitude_snapshot(model)
    total_weight = sum(ws.l2_norm for ws in weight_stats) if weight_stats else 0.0

    ratios = update_to_weight_ratio(model, optimizer)
    sorted_ratios = sorted(ratios.values())
    median_ratio = (
        sorted_ratios[len(sorted_ratios) // 2] if sorted_ratios else 0.0
    )

    current_loss = tracker.history[-1] if tracker.history else float("nan")

    return OptimizationHealth(
        step=tracker._step_count,
        loss=current_loss,
        total_grad_norm=total_grad,
        max_layer_grad_norm=max(layer_norms),
        min_layer_grad_norm=min(layer_norms),
        total_weight_norm=total_weight,
        convergence_rate=tracker.estimated_rate,
        is_converging=tracker.is_converging(),
        is_diverging=tracker.is_diverging(),
        steps_since_improvement=tracker.steps_since_improvement(),
        grad_to_weight_ratio=median_ratio,
        vanishing_gradients=total_grad < 1e-7,
        exploding_gradients=total_grad > 1e3,
        stale_training=tracker.steps_since_improvement() > 2 * tracker.window,
    )


def format_health_report(health: OptimizationHealth, step: int) -> str:
    """Format an OptimizationHealth snapshot as a human-readable string.

    Output example:
        [step 1500] loss=0.3421 | grad_norm=12.4 | weight_norm=847.2
        convergence_rate=0.9987 (converging) | update/weight=1.2e-03
        ⚠ No warnings.
    """
    lines = []
    lines.append(
        f"[step {step}] loss={health.loss:.4f} | "
        f"grad_norm={health.total_grad_norm:.4g} | "
        f"weight_norm={health.total_weight_norm:.4g}"
    )

    rate_str = f"{health.convergence_rate:.6f}" if health.convergence_rate else "N/A"
    status = "converging" if health.is_converging else (
        "DIVERGING" if health.is_diverging else "stalled"
    )
    lines.append(
        f"  convergence_rate={rate_str} ({status}) | "
        f"update/weight={health.grad_to_weight_ratio:.2e}"
    )

    warnings = []
    if health.vanishing_gradients:
        warnings.append("vanishing gradients (norm < 1e-7)")
    if health.exploding_gradients:
        warnings.append("exploding gradients (norm > 1e3) — consider clip_grad_norm_")
    if health.stale_training:
        warnings.append(
            f"no improvement for {health.steps_since_improvement} steps "
            f"— consider lr warmup restart or early stopping"
        )

    if warnings:
        lines.append("  ⚠ " + "; ".join(warnings))
    else:
        lines.append("  ✓ No warnings.")

    return "\n".join(lines)
