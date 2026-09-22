"""Categorical / contingency verification metrics (PRD §16.2, §16.7, Appendix A).

Formulas:
- a = hits (F >= t and O >= t)
- b = false alarms (F >= t and O < t)
- c = misses (F < t and O >= t)
- d = correct negatives (F < t and O < t)
- n = a + b + c + d

Metrics:
- POD            = a / (a + c)
- FAR (ratio)    = b / (a + b)
- CSI            = a / (a + b + c)
- Frequency bias = (a + b) / (a + c)
- a_ref          = (a + b)(a + c) / n
- ETS            = (a - a_ref) / (a + b + c - a_ref)

Edge cases (PRD §16.7):
- If (a + c) == 0 (no observed events) -> POD, Frequency bias, ETS are undefined.
- If (a + b) == 0 (no forecast events) -> FAR, ETS are undefined.
- If (a + b + c) == 0 (no events)      -> CSI is undefined.
- Never return 0 for undefined metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from verification.metrics import MetricResult


@dataclass(frozen=True)
class ContingencyCounts:
    """2x2 contingency table counts."""
    a: int  # hits
    b: int  # false alarms
    c: int  # misses
    d: int  # correct negatives
    n: int  # total valid samples (a + b + c + d)

    def to_dict(self) -> Dict[str, int]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "n": self.n}


def compute_contingency_counts(
    forecast: Sequence[float] | np.ndarray,
    observed: Sequence[float] | np.ndarray,
    threshold: float,
) -> ContingencyCounts:
    """Calculates 2x2 contingency table counts for a given rainfall threshold.

    Excludes pairs where either forecast or observed is NaN.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    o = np.asarray(observed, dtype=float).ravel()

    if f.shape != o.shape:
        raise ValueError(
            f"Shape mismatch: forecast has shape {f.shape}, observed has shape {o.shape}."
        )

    # Filter out missing observations
    valid_mask = ~(np.isnan(f) | np.isnan(o))
    f_valid = f[valid_mask]
    o_valid = o[valid_mask]

    f_ge = f_valid >= threshold
    o_ge = o_valid >= threshold

    a = int(np.count_nonzero(f_ge & o_ge))
    b = int(np.count_nonzero(f_ge & (~o_ge)))
    c = int(np.count_nonzero((~f_ge) & o_ge))
    d = int(np.count_nonzero((~f_ge) & (~o_ge)))
    n = a + b + c + d

    return ContingencyCounts(a=a, b=b, c=c, d=d, n=n)


def compute_pod(a: int, c: int) -> MetricResult:
    """Probability of Detection (Hit Rate): a / (a + c).

    Undefined if there are no observed events (a + c == 0).
    """
    denom = a + c
    if denom == 0:
        return MetricResult(value=None, undefined_reason="NO_OBSERVED_EVENTS")
    return MetricResult(value=float(a / denom), undefined_reason=None)


def compute_far(a: int, b: int) -> MetricResult:
    """False Alarm Ratio: b / (a + b).

    Undefined if there are no forecast events (a + b == 0).
    """
    denom = a + b
    if denom == 0:
        return MetricResult(value=None, undefined_reason="NO_FORECAST_EVENTS")
    return MetricResult(value=float(b / denom), undefined_reason=None)


def compute_csi(a: int, b: int, c: int) -> MetricResult:
    """Critical Success Index (Threat Score): a / (a + b + c).

    Undefined if there are no events at all (a + b + c == 0).
    """
    denom = a + b + c
    if denom == 0:
        return MetricResult(value=None, undefined_reason="NO_EVENTS")
    return MetricResult(value=float(a / denom), undefined_reason=None)


def compute_frequency_bias(a: int, b: int, c: int) -> MetricResult:
    """Frequency Bias: (a + b) / (a + c).

    Undefined if there are no observed events (a + c == 0).
    """
    denom = a + c
    if denom == 0:
        return MetricResult(value=None, undefined_reason="NO_OBSERVED_EVENTS")
    return MetricResult(value=float((a + b) / denom), undefined_reason=None)


def compute_ets(a: int, b: int, c: int, d: int) -> MetricResult:
    """Equitable Threat Score (Gilbert Skill Score): (a - a_ref) / (a + b + c - a_ref).

    where a_ref = (a + b)(a + c) / n.
    Undefined if n == 0, if a + c == 0, if a + b == 0, or if denominator == 0.
    """
    n = a + b + c + d
    if n == 0:
        return MetricResult(value=None, undefined_reason="NO_SAMPLES")
    if (a + c) == 0:
        return MetricResult(value=None, undefined_reason="NO_OBSERVED_EVENTS")
    if (a + b) == 0:
        return MetricResult(value=None, undefined_reason="NO_FORECAST_EVENTS")

    a_ref = ((a + b) * (a + c)) / n
    denom = (a + b + c) - a_ref
    if denom == 0:
        return MetricResult(value=None, undefined_reason="ZERO_DENOMINATOR")

    return MetricResult(value=float((a - a_ref) / denom), undefined_reason=None)


def compute_contingency_metrics(counts: ContingencyCounts) -> Dict[str, MetricResult]:
    """Computes all categorical metrics for a given ContingencyCounts instance."""
    a, b, c, d = counts.a, counts.b, counts.c, counts.d
    return {
        "pod": compute_pod(a, c),
        "far": compute_far(a, b),
        "csi": compute_csi(a, b, c),
        "frequency_bias": compute_frequency_bias(a, b, c),
        "ets": compute_ets(a, b, c, d),
    }
