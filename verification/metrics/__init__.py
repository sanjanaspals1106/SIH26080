"""Verification metrics module (PRD §16, Appendix A)."""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


@dataclass(frozen=True)
class MetricResult:
    """Represents a metric computation result.

    Attributes:
        value: The computed metric value, or None if undefined.
        undefined_reason: Machine-readable reason if value is None, else None.
    """
    value: Optional[float]
    undefined_reason: Optional[str] = None

    def __iter__(self) -> Iterator[Any]:
        """Allows unpacking as (value, undefined_reason)."""
        yield self.value
        yield self.undefined_reason

    def is_defined(self) -> bool:
        """Returns True if the metric value is defined (not None)."""
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to dictionary."""
        return {
            "value": self.value,
            "undefined_reason": self.undefined_reason,
        }


# Re-export metrics from submodules
from verification.metrics.continuous import (
    compute_bias,
    compute_continuous_metrics,
    compute_mae,
    compute_rmse,
)
from verification.metrics.contingency import (
    ContingencyCounts,
    compute_contingency_counts,
    compute_contingency_metrics,
    compute_csi,
    compute_ets,
    compute_far,
    compute_frequency_bias,
    compute_pod,
)
from verification.metrics.probabilistic import (
    compute_brier_score,
    compute_brier_skill_score,
    compute_coverage,
    compute_pinball_loss,
)
from verification.metrics.spatial import (
    FSSComponents,
    aggregate_fss,
    compute_fss,
    compute_fss_components,
    compute_fss_from_components,
    compute_observed_event_fraction,
    compute_useful_skill,
)

__all__ = [
    "MetricResult",
    "compute_bias",
    "compute_mae",
    "compute_rmse",
    "compute_continuous_metrics",
    "ContingencyCounts",
    "compute_contingency_counts",
    "compute_contingency_metrics",
    "compute_pod",
    "compute_far",
    "compute_csi",
    "compute_frequency_bias",
    "compute_ets",
    "compute_brier_score",
    "compute_brier_skill_score",
    "compute_pinball_loss",
    "compute_coverage",
    "FSSComponents",
    "compute_fss_components",
    "compute_fss_from_components",
    "compute_fss",
    "aggregate_fss",
    "compute_observed_event_fraction",
    "compute_useful_skill",
]
