"""Regime-benefit decision rule evaluation per PRD §10.8 (Hypothesis H1).

Decision Rule:
B3 is better than B2 on a metric only if the 95% interval of the paired difference
(B3 - B2) is entirely on B3's side.

Metric directions:
- P1 RMSE: LOWER is better (B3 favored when CI is strictly negative, ci_high < 0)
- P2 ETS:  HIGHER is better (B3 favored when CI is strictly positive, ci_low > 0)
- P3 FSS:  HIGHER is better (B3 favored when CI is strictly positive, ci_low > 0)
- P4 BSS:  HIGHER is better (B3 favored when CI is strictly positive, ci_low > 0)

Per-metric classifications:
- 'B3_BETTER'
- 'B2_BETTER'
- 'NO_SIGNIFICANT_DIFFERENCE' (CI touches or crosses 0)

Overall claim 'regime information helps' (demonstrated_benefit=True) requires:
1. P1 significantly favors B3
2. At least one of P2, P3, P4 significantly favors B3
3. None of P1, P2, P3, P4 significantly favors B2
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from verification.stats.bootstrap import BootstrapResult


STATUS_B3_BETTER = "B3_BETTER"
STATUS_B2_BETTER = "B2_BETTER"
STATUS_NO_DIFF = "NO_SIGNIFICANT_DIFFERENCE"
STATUS_UNAVAILABLE = "UNAVAILABLE"

# Metric direction mapping: True = higher is better, False = lower is better
_METRIC_HIGHER_IS_BETTER: Dict[str, bool] = {
    "P1": False,  # RMSE
    "P2": True,   # ETS
    "P3": True,   # FSS
    "P4": True,   # BSS
}


@dataclass(frozen=True)
class DecisionResult:
    """Structured result of PRD §10.8 decision rule evaluation."""
    demonstrated_benefit: bool
    per_metric_status: Dict[str, str]
    supporting_metrics: List[str]
    worse_metrics: List[str]
    failed_conditions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "demonstrated_benefit": self.demonstrated_benefit,
            "per_metric_status": self.per_metric_status,
            "supporting_metrics": self.supporting_metrics,
            "worse_metrics": self.worse_metrics,
            "failed_conditions": self.failed_conditions,
        }


def _canonicalize_metric_key(key: str) -> str:
    """Normalizes diverse metric key formats to P1, P2, P3, P4."""
    norm = key.strip().upper()
    if norm in ("P1", "RMSE", "P1_RMSE"):
        return "P1"
    if norm in ("P2", "ETS", "P2_ETS"):
        return "P2"
    if norm in ("P3", "FSS", "P3_FSS"):
        return "P3"
    if norm in ("P4", "BSS", "P4_BSS", "BRIER_SKILL_SCORE"):
        return "P4"
    return norm


def classify_paired_metric(
    metric_id: str,
    ci_low: float,
    ci_high: float,
) -> str:
    """Classifies a single metric paired difference CI (B3 - B2).

    Rules:
    - CI touching or crossing 0 => NO_SIGNIFICANT_DIFFERENCE.
    - If lower is better (P1/RMSE):
        ci_high < 0.0 => B3_BETTER (B3 has lower RMSE)
        ci_low > 0.0  => B2_BETTER (B2 has lower RMSE)
    - If higher is better (P2/P3/P4):
        ci_low > 0.0  => B3_BETTER (B3 has higher skill)
        ci_high < 0.0 => B2_BETTER (B2 has higher skill)
    """
    canon = _canonicalize_metric_key(metric_id)
    higher_is_better = _METRIC_HIGHER_IS_BETTER.get(canon, True)

    # Check if CI touches or crosses 0
    if ci_low <= 0.0 <= ci_high:
        return STATUS_NO_DIFF

    if higher_is_better:
        if ci_low > 0.0:
            return STATUS_B3_BETTER
        elif ci_high < 0.0:
            return STATUS_B2_BETTER
    else:  # lower is better (e.g. RMSE)
        if ci_high < 0.0:
            return STATUS_B3_BETTER
        elif ci_low > 0.0:
            return STATUS_B2_BETTER

    return STATUS_NO_DIFF


def evaluate_regime_benefit_decision(
    paired_intervals: Dict[str, Union[Tuple[float, float], BootstrapResult]],
) -> DecisionResult:
    """Evaluates hypothesis H1 (regime information helps) per PRD §10.8.

    Args:
        paired_intervals: Mapping of metric key ('P1', 'P2', 'P3', 'P4') to either
                          a (ci_low, ci_high) tuple or a BootstrapResult.
                          Assumes paired difference is (B3 - B2).

    Returns:
        DecisionResult containing demonstrated_benefit boolean and structured statuses.
    """
    per_metric_status: Dict[str, str] = {}

    # Standardize input keys
    standardized_inputs: Dict[str, Tuple[float, float]] = {}
    for k, v in paired_intervals.items():
        c_key = _canonicalize_metric_key(k)
        if isinstance(v, BootstrapResult):
            standardized_inputs[c_key] = (v.ci_low, v.ci_high)
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            standardized_inputs[c_key] = (float(v[0]), float(v[1]))
        else:
            raise TypeError(
                f"Expected tuple of (ci_low, ci_high) or BootstrapResult for metric {k}, got {type(v)}."
            )

    # Classify each primary metric P1-P4
    for mid in ["P1", "P2", "P3", "P4"]:
        if mid in standardized_inputs:
            ci_low, ci_high = standardized_inputs[mid]
            per_metric_status[mid] = classify_paired_metric(mid, ci_low, ci_high)
        else:
            per_metric_status[mid] = STATUS_UNAVAILABLE

    supporting_metrics: List[str] = [
        m for m, status in per_metric_status.items() if status == STATUS_B3_BETTER
    ]
    worse_metrics: List[str] = [
        m for m, status in per_metric_status.items() if status == STATUS_B2_BETTER
    ]

    failed_conditions: List[str] = []

    # Condition 1: P1 (RMSE) must significantly favor B3
    if per_metric_status.get("P1") != STATUS_B3_BETTER:
        failed_conditions.append("P1_RMSE_DOES_NOT_FAVOR_B3")

    # Condition 2: At least one of P2 (ETS), P3 (FSS), P4 (BSS) must significantly favor B3
    cat_prob_favors = any(
        per_metric_status.get(m) == STATUS_B3_BETTER for m in ["P2", "P3", "P4"]
    )
    if not cat_prob_favors:
        failed_conditions.append("NO_CATEGORICAL_OR_PROBABILITY_METRIC_FAVORS_B3")

    # Condition 3: No primary metric may be significantly worse (favor B2)
    if len(worse_metrics) > 0:
        failed_conditions.append("PRIMARY_METRIC_SIGNIFICANTLY_FAVORS_B2")

    demonstrated_benefit = len(failed_conditions) == 0

    return DecisionResult(
        demonstrated_benefit=demonstrated_benefit,
        per_metric_status=per_metric_status,
        supporting_metrics=supporting_metrics,
        worse_metrics=worse_metrics,
        failed_conditions=failed_conditions,
    )
