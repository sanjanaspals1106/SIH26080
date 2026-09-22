"""Probabilistic and quantile verification metrics (PRD §16.2, Appendix A).

Formulas:
- Brier score: mean((p - y)^2)
- Brier Skill Score (BSS): 1 - BS / BS_reference
- Pinball loss (level alpha): mean( max(alpha*(y - q), (alpha - 1)*(y - q)) )
- Coverage: share of observations between q_low and q_high (target 0.80 for q10-q90)
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from verification.metrics import MetricResult


def compute_brier_score(
    probabilities: Sequence[float] | np.ndarray,
    observed_binary: Sequence[Union[int, float, bool]] | np.ndarray,
) -> MetricResult:
    """Computes Brier Score: mean((p - y)^2).

    Args:
        probabilities: Forecast probabilities in range [0, 1].
        observed_binary: Observed outcomes where 1 = event occurred, 0 = did not occur.

    Returns:
        MetricResult containing Brier score or undefined reason.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(observed_binary, dtype=float).ravel()

    if p.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: probabilities shape {p.shape}, observed shape {y.shape}."
        )

    # Filter out NaNs
    valid_mask = ~(np.isnan(p) | np.isnan(y))
    p_clean = p[valid_mask]
    y_clean = y[valid_mask]

    n = len(p_clean)
    if n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")

    # Check bounds
    if np.any(p_clean < 0.0) or np.any(p_clean > 1.0):
        raise ValueError("Probabilities must lie strictly within [0.0, 1.0].")

    # Check binary targets
    if not np.all(np.isin(y_clean, [0.0, 1.0])):
        raise ValueError("Observed binary targets must contain only 0 and 1.")

    bs = np.mean((p_clean - y_clean) ** 2)
    return MetricResult(value=float(bs), undefined_reason=None)


def compute_brier_skill_score(
    brier_score: Union[float, MetricResult, None],
    brier_score_ref: Union[float, MetricResult, None],
) -> MetricResult:
    """Computes Brier Skill Score: 1 - (BS / BS_reference).

    Undefined if BS_reference is 0 or if either input is undefined.
    """
    bs_val = brier_score.value if isinstance(brier_score, MetricResult) else brier_score
    ref_val = brier_score_ref.value if isinstance(brier_score_ref, MetricResult) else brier_score_ref

    if bs_val is None or ref_val is None:
        return MetricResult(value=None, undefined_reason="MISSING_BRIER_SCORE")

    if ref_val == 0.0:
        return MetricResult(value=None, undefined_reason="ZERO_REFERENCE_BRIER_SCORE")

    bss = 1.0 - (bs_val / ref_val)
    return MetricResult(value=float(bss), undefined_reason=None)


def compute_pinball_loss(
    quantiles: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
    alpha: float,
) -> MetricResult:
    """Computes Pinball Loss (quantile loss) for quantile level alpha:

    mean( max(alpha * (y - q), (alpha - 1) * (y - q)) )

    Args:
        quantiles: Predicted quantile values.
        observed: Observed values.
        alpha: Quantile target level in (0, 1), e.g. 0.1, 0.5, 0.9.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Quantile alpha must be in open interval (0, 1), got {alpha}.")

    q = np.asarray(quantiles, dtype=float).ravel()
    y = np.asarray(observed, dtype=float).ravel()

    if q.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: quantiles shape {q.shape}, observed shape {y.shape}."
        )

    valid_mask = ~(np.isnan(q) | np.isnan(y))
    q_clean = q[valid_mask]
    y_clean = y[valid_mask]

    n = len(q_clean)
    if n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")

    diff = y_clean - q_clean
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return MetricResult(value=float(np.mean(loss)), undefined_reason=None)


def compute_coverage(
    q_low: Sequence[float] | np.ndarray,
    q_high: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
) -> MetricResult:
    """Computes empirical coverage: fraction of observations in [q_low, q_high].

    For model-estimated range q10-q90, target is 0.80 (PRD §13.7, Appendix A).
    """
    ql = np.asarray(q_low, dtype=float).ravel()
    qh = np.asarray(q_high, dtype=float).ravel()
    y = np.asarray(observed, dtype=float).ravel()

    if not (ql.shape == qh.shape == y.shape):
        raise ValueError(
            f"Shape mismatch: q_low {ql.shape}, q_high {qh.shape}, observed {y.shape}."
        )

    valid_mask = ~(np.isnan(ql) | np.isnan(qh) | np.isnan(y))
    ql_clean = ql[valid_mask]
    qh_clean = qh[valid_mask]
    y_clean = y[valid_mask]

    n = len(y_clean)
    if n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")

    inside = (y_clean >= ql_clean) & (y_clean <= qh_clean)
    cov = np.mean(inside)
    return MetricResult(value=float(cov), undefined_reason=None)
