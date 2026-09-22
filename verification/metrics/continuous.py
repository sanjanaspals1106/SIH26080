"""Continuous verification metrics (PRD §16.2, Appendix A).

Formulas:
- Bias = mean(F - O)
- MAE  = mean(|F - O|)
- RMSE = sqrt(mean((F - O)^2))
"""

from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from verification.metrics import MetricResult


def _prepare_continuous_arrays(
    forecast: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Validates inputs, converts to 1D float arrays, and filters out NaNs."""
    f = np.asarray(forecast, dtype=float).ravel()
    o = np.asarray(observed, dtype=float).ravel()

    if f.shape != o.shape:
        raise ValueError(
            f"Shape mismatch: forecast has shape {f.shape}, observed has shape {o.shape}."
        )

    # Exclude missing observations (NaNs in either forecast or observed)
    valid_mask = ~(np.isnan(f) | np.isnan(o))
    f_clean = f[valid_mask]
    o_clean = o[valid_mask]
    return f_clean, o_clean, int(len(f_clean))


def compute_bias(
    forecast: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
) -> MetricResult:
    """Computes mean forecast error: mean(F - O)."""
    f, o, n = _prepare_continuous_arrays(forecast, observed)
    if n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    diff = f - o
    return MetricResult(value=float(np.mean(diff)), undefined_reason=None)


def compute_mae(
    forecast: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
) -> MetricResult:
    """Computes Mean Absolute Error: mean(|F - O|)."""
    f, o, n = _prepare_continuous_arrays(forecast, observed)
    if n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    diff = np.abs(f - o)
    return MetricResult(value=float(np.mean(diff)), undefined_reason=None)


def compute_rmse(
    forecast: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
) -> MetricResult:
    """Computes Root Mean Squared Error: sqrt(mean((F - O)^2))."""
    f, o, n = _prepare_continuous_arrays(forecast, observed)
    if n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    diff_sq = (f - o) ** 2
    return MetricResult(value=float(np.sqrt(np.mean(diff_sq))), undefined_reason=None)


def compute_continuous_metrics(
    forecast: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
) -> Dict[str, MetricResult]:
    """Computes Bias, MAE, and RMSE together."""
    f, o, n = _prepare_continuous_arrays(forecast, observed)
    if n == 0:
        return {
            "bias": MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES"),
            "mae": MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES"),
            "rmse": MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES"),
        }
    diff = f - o
    return {
        "bias": MetricResult(value=float(np.mean(diff)), undefined_reason=None),
        "mae": MetricResult(value=float(np.mean(np.abs(diff))), undefined_reason=None),
        "rmse": MetricResult(value=float(np.sqrt(np.mean(diff ** 2))), undefined_reason=None),
    }
