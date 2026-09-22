"""Aggregation layer for verification components (PRD §16.4, §16.5, §16.10).

Rules:
1. Components are summed across dates/runs; NEVER average daily metrics.
2. Recompute final metrics from aggregated sums.
3. Preserve undefined metrics as MetricResult(None, reason). Never return 0.
4. n_samples and n_events must be accurately aggregated and preserved.
5. Incompatible forecast_type, evaluation_set, threshold_mm, neighbourhood_cells,
   lead_day, or group identity are rejected unless caller explicitly groups by them.
"""

from collections import defaultdict
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from verification.components import VerificationComponent
from verification.metrics import MetricResult
from verification.metrics.contingency import (
    compute_contingency_metrics,
    compute_csi,
    compute_ets,
    compute_far,
    compute_frequency_bias,
    compute_pod,
)
from verification.metrics.spatial import compute_fss_from_components


def aggregate_components(
    components: Sequence[VerificationComponent],
    allow_incompatible: bool = False,
) -> VerificationComponent:
    """Aggregates a sequence of VerificationComponent records into a single summed component.
    
    Args:
        components: Sequence of VerificationComponent instances.
        allow_incompatible: If False, validates that all components share compatible metadata.
        
    Returns:
        A single VerificationComponent with all counts and sums accumulated.
    """
    if not components:
        return VerificationComponent()

    if len(components) == 1:
        c = components[0]
        return VerificationComponent(
            imd_date=c.imd_date,
            lead_day=c.lead_day,
            forecast_type=c.forecast_type,
            evaluation_set=c.evaluation_set,
            threshold_mm=c.threshold_mm,
            neighbourhood_cells=c.neighbourhood_cells,
            group_type=c.group_type,
            group_value=c.group_value,
            bin_lower=c.bin_lower,
            bin_upper=c.bin_upper,
            n_samples=c.n_samples,
            n_events=c.n_events,
            n=c.n,
            sum_error=c.sum_error,
            sum_abs_error=c.sum_abs_error,
            sum_squared_error=c.sum_squared_error,
            a=c.a,
            b=c.b,
            c=c.c,
            d=c.d,
            fss_numerator_sum=c.fss_numerator_sum,
            fss_forecast_fraction_sq_sum=c.fss_forecast_fraction_sq_sum,
            fss_observed_fraction_sq_sum=c.fss_observed_fraction_sq_sum,
            brier_n=c.brier_n,
            sum_brier_terms=c.sum_brier_terms,
            sum_reference_brier_terms=c.sum_reference_brier_terms,
            sum_predicted_probability=c.sum_predicted_probability,
            n_observed_events=c.n_observed_events,
            pinball_q10_sum=c.pinball_q10_sum,
            pinball_q50_sum=c.pinball_q50_sum,
            pinball_q90_sum=c.pinball_q90_sum,
            range_n=c.range_n,
            coverage_count=c.coverage_count,
        )

    first = components[0]

    # Validate compatibility across all components unless explicitly overridden
    if not allow_incompatible:
        for other in components[1:]:
            first.check_compatibility(other)

    total_n_samples = sum(c.n_samples for c in components)
    any_events = any(c.n_events is not None for c in components)
    total_n_events = sum(c.n_events or 0 for c in components) if any_events else None

    return VerificationComponent(
        imd_date="aggregated",
        lead_day=first.lead_day,
        forecast_type=first.forecast_type,
        evaluation_set=first.evaluation_set,
        threshold_mm=first.threshold_mm,
        neighbourhood_cells=first.neighbourhood_cells,
        group_type=first.group_type,
        group_value=first.group_value,
        bin_lower=first.bin_lower,
        bin_upper=first.bin_upper,
        n_samples=total_n_samples,
        n_events=total_n_events,
        n=sum(c.n for c in components),
        sum_error=sum(c.sum_error for c in components),
        sum_abs_error=sum(c.sum_abs_error for c in components),
        sum_squared_error=sum(c.sum_squared_error for c in components),
        a=sum(c.a for c in components),
        b=sum(c.b for c in components),
        c=sum(c.c for c in components),
        d=sum(c.d for c in components),
        fss_numerator_sum=sum(c.fss_numerator_sum for c in components),
        fss_forecast_fraction_sq_sum=sum(c.fss_forecast_fraction_sq_sum for c in components),
        fss_observed_fraction_sq_sum=sum(c.fss_observed_fraction_sq_sum for c in components),
        brier_n=sum(c.brier_n for c in components),
        sum_brier_terms=sum(c.sum_brier_terms for c in components),
        sum_reference_brier_terms=sum(c.sum_reference_brier_terms for c in components),
        sum_predicted_probability=sum(c.sum_predicted_probability for c in components),
        n_observed_events=sum(c.n_observed_events for c in components),
        pinball_q10_sum=sum(c.pinball_q10_sum for c in components),
        pinball_q50_sum=sum(c.pinball_q50_sum for c in components),
        pinball_q90_sum=sum(c.pinball_q90_sum for c in components),
        range_n=sum(c.range_n for c in components),
        coverage_count=sum(c.coverage_count for c in components),
    )


def group_and_aggregate(
    components: Sequence[VerificationComponent],
    group_keys: Optional[Sequence[str]] = None,
) -> Dict[Tuple[Any, ...], VerificationComponent]:
    """Groups components by specified metadata keys and aggregates each group."""
    if group_keys is None:
        group_keys = [
            "lead_day",
            "forecast_type",
            "evaluation_set",
            "threshold_mm",
            "neighbourhood_cells",
            "group_type",
            "group_value",
            "bin_lower",
            "bin_upper",
        ]

    buckets: Dict[Tuple[Any, ...], List[VerificationComponent]] = defaultdict(list)
    for comp in components:
        key = tuple(getattr(comp, k) for k in group_keys)
        buckets[key].append(comp)

    aggregated: Dict[Tuple[Any, ...], VerificationComponent] = {}
    for key, items in buckets.items():
        aggregated[key] = aggregate_components(items, allow_incompatible=False)

    return aggregated


# ---------------------------------------------------------------------------
# Metric Recomputation Helpers (from aggregated components)
# ---------------------------------------------------------------------------

def compute_rmse_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes RMSE from sum_squared_error and n: sqrt(sum_squared_error / n)."""
    if comp.n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    mse = comp.sum_squared_error / comp.n
    return MetricResult(value=float(math.sqrt(mse)), undefined_reason=None)


def compute_bias_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes Bias from sum_error and n: sum_error / n."""
    if comp.n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    bias = comp.sum_error / comp.n
    return MetricResult(value=float(bias), undefined_reason=None)


def compute_mae_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes MAE from sum_abs_error and n: sum_abs_error / n."""
    if comp.n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    mae = comp.sum_abs_error / comp.n
    return MetricResult(value=float(mae), undefined_reason=None)


def compute_pod_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes Probability of Detection: a / (a + c)."""
    return compute_pod(comp.a, comp.c)


def compute_far_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes False Alarm Ratio: b / (a + b)."""
    return compute_far(comp.a, comp.b)


def compute_csi_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes Critical Success Index: a / (a + b + c)."""
    return compute_csi(comp.a, comp.b, comp.c)


def compute_frequency_bias_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes Frequency Bias: (a + b) / (a + c)."""
    return compute_frequency_bias(comp.a, comp.b, comp.c)


def compute_ets_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes Equitable Threat Score: (a - a_ref) / (a + b + c - a_ref)."""
    return compute_ets(comp.a, comp.b, comp.c, comp.d)


def compute_contingency_metrics_from_components(comp: VerificationComponent) -> Dict[str, MetricResult]:
    """Recomputes all contingency metrics (POD, FAR, CSI, Frequency Bias, ETS)."""
    return compute_contingency_metrics(comp.to_contingency_counts())


def compute_fss_from_components_record(comp: VerificationComponent) -> MetricResult:
    """Recomputes Fractions Skill Score from summed FSS components (PRD §16.4, §16.8)."""
    denom = comp.fss_forecast_fraction_sq_sum + comp.fss_observed_fraction_sq_sum
    if denom == 0.0:
        return MetricResult(value=None, undefined_reason="ZERO_DENOMINATOR")
    return compute_fss_from_components(comp.to_fss_components())


def compute_brier_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes Brier Score from sum_brier_terms and brier_n."""
    if comp.brier_n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    bs = comp.sum_brier_terms / comp.brier_n
    return MetricResult(value=float(bs), undefined_reason=None)


def compute_brier_skill_score_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes Brier Skill Score: 1 - BS / BS_reference (PRD §13.5, §16.2).
    
    Rules:
    - Zero reference Brier score => undefined reason 'ZERO_REFERENCE_BRIER_SCORE' (not numeric 0).
    - Missing samples => undefined reason 'NO_VALID_SAMPLES'.
    """
    if comp.brier_n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    if comp.sum_reference_brier_terms <= 0.0:
        return MetricResult(value=None, undefined_reason="ZERO_REFERENCE_BRIER_SCORE")

    bs = comp.sum_brier_terms / comp.brier_n
    bs_ref = comp.sum_reference_brier_terms / comp.brier_n
    bss = 1.0 - (bs / bs_ref)
    return MetricResult(value=float(bss), undefined_reason=None)


compute_bss_from_components = compute_brier_skill_score_from_components


def compute_pinball_from_components(comp: VerificationComponent) -> Dict[str, MetricResult]:
    """Recomputes Pinball losses for q10, q50, q90 from summed terms and range_n."""
    if comp.range_n == 0:
        return {
            "q10": MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES"),
            "q50": MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES"),
            "q90": MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES"),
        }
    return {
        "q10": MetricResult(value=float(comp.pinball_q10_sum / comp.range_n), undefined_reason=None),
        "q50": MetricResult(value=float(comp.pinball_q50_sum / comp.range_n), undefined_reason=None),
        "q90": MetricResult(value=float(comp.pinball_q90_sum / comp.range_n), undefined_reason=None),
    }


def compute_coverage_from_components(comp: VerificationComponent) -> MetricResult:
    """Recomputes empirical coverage from coverage_count and range_n."""
    if comp.range_n == 0:
        return MetricResult(value=None, undefined_reason="NO_VALID_SAMPLES")
    cov = comp.coverage_count / comp.range_n
    return MetricResult(value=float(cov), undefined_reason=None)


def evaluate_all_metrics_from_components(comp: VerificationComponent) -> Dict[str, Any]:
    """Recomputes all applicable verification metrics from an aggregated VerificationComponent."""
    contingency = compute_contingency_metrics_from_components(comp)
    pinball = compute_pinball_from_components(comp)

    return {
        "rmse": compute_rmse_from_components(comp),
        "bias": compute_bias_from_components(comp),
        "mae": compute_mae_from_components(comp),
        "pod": contingency["pod"],
        "far": contingency["far"],
        "csi": contingency["csi"],
        "frequency_bias": contingency["frequency_bias"],
        "ets": contingency["ets"],
        "fss": compute_fss_from_components_record(comp),
        "brier_score": compute_brier_from_components(comp),
        "brier_skill_score": compute_brier_skill_score_from_components(comp),
        "pinball_q10": pinball["q10"],
        "pinball_q50": pinball["q50"],
        "pinball_q90": pinball["q90"],
        "coverage": compute_coverage_from_components(comp),
        "n_samples": comp.n_samples,
        "n_events": comp.n_events,
    }
