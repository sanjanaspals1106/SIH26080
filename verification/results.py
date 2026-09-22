"""Verification result packaging and PRD §10.8 primary-metric threshold selection.

Combines aggregated verification components, metric recomputations, and bootstrap
outputs into the structured result matching PRD §16.11.

Rules:
1. Primary threshold selection:
   - Default primary threshold for P2/P3/P4 = 64.5 mm.
   - If HOLDOUT has fewer than 30 observed events at 64.5 mm:
     use 15.6 mm for P2/P3/P4.
   - 64.5 mm results may still exist as exploratory.
   - Pure protocol rule, independent of model skill.
   - Boundary: 29 events => 15.6 mm; 30 events => 64.5 mm.
2. Verification result object matching PRD §16.11:
   - Preserves forecast_type, comparison_to, evaluation_set, lead_day,
     threshold_mm, neighbourhood_cells, group: {type, value}.
   - Metrics: rmse, pod, far, csi, ets, fss, frequency_bias, brier_score,
     brier_skill_score, coverage (where applicable).
   - Each metric: value, ci_low, ci_high, undefined_reason.
   - Paired difference: value, ci_low, ci_high when comparison exists.
   - n_samples, n_events preserved accurately.
   - Bootstrap metadata: block_days, resamples.
   - Flags: primary_metric, exploratory.
3. Integrity rules:
   - Never fabricate CIs when bootstrap is absent.
   - Undefined metric values remain None.
   - CI must remain None when metric is undefined.
   - DEVELOPMENT and HOLDOUT results stay strictly distinct.
   - No claim / decision-rule wording in results.
   - No database or API dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import yaml

from verification.aggregation import evaluate_all_metrics_from_components
from verification.components import VerificationComponent
from verification.metrics import MetricResult
from verification.stats.bootstrap import BootstrapResult


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_protocol_config(
    protocol_config_path: Optional[Union[str, Path]] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Loads protocol configuration from config/protocol.yaml."""
    if protocol_config_path is not None:
        p = Path(protocol_config_path)
    elif config_dir is not None:
        p = Path(config_dir) / "protocol.yaml"
    else:
        p = _find_repo_root() / "config" / "protocol.yaml"

    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def load_thresholds_config(
    thresholds_config_path: Optional[Union[str, Path]] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Loads threshold configuration from config/thresholds.yaml."""
    if thresholds_config_path is not None:
        p = Path(thresholds_config_path)
    elif config_dir is not None:
        p = Path(config_dir) / "thresholds.yaml"
    else:
        p = _find_repo_root() / "config" / "thresholds.yaml"

    if p.is_file():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


@dataclass(frozen=True)
class PrimaryThresholdDecision:
    """Decision metadata for primary metric threshold selection per PRD §10.8."""
    selected_threshold_mm: float
    default_threshold_mm: float
    fallback_used: bool
    reason: str
    holdout_event_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_threshold_mm": self.selected_threshold_mm,
            "default_threshold_mm": self.default_threshold_mm,
            "fallback_used": self.fallback_used,
            "reason": self.reason,
            "holdout_event_count": self.holdout_event_count,
        }


def select_primary_threshold(
    holdout_event_count: int,
    config_dir: Optional[Union[str, Path]] = None,
    protocol_config_path: Optional[Union[str, Path]] = None,
    thresholds_config_path: Optional[Union[str, Path]] = None,
    min_events_primary: Optional[int] = None,
    primary_threshold_mm: Optional[float] = None,
    fallback_threshold_mm: Optional[float] = None,
) -> PrimaryThresholdDecision:
    """Selects primary verification threshold for P2/P3/P4 per PRD §10.8.

    Rule:
    - Default primary threshold for P2/P3/P4 = 64.5 mm.
    - If HOLDOUT has fewer than 30 observed events at 64.5 mm:
      use 15.6 mm for P2/P3/P4.
    - 64.5 mm results may still exist as exploratory.
    - This is an automatic pre-written protocol rule, NOT tuning.
    - Boundary: 29 events => 15.6 mm; 30 events => 64.5 mm.

    Args:
        holdout_event_count: Observed event count in the holdout dataset at default threshold.
        config_dir: Optional path to config directory.
        protocol_config_path: Optional explicit path to config/protocol.yaml.
        thresholds_config_path: Optional explicit path to config/thresholds.yaml.
        min_events_primary: Optional override for minimum events cutoff (default 30).
        primary_threshold_mm: Optional override for primary threshold (default 64.5).
        fallback_threshold_mm: Optional override for fallback threshold (default 15.6).

    Returns:
        PrimaryThresholdDecision containing selected threshold, fallback flag, and reason.
    """
    if holdout_event_count < 0:
        raise ValueError(f"holdout_event_count must be non-negative, got {holdout_event_count}")

    proto_cfg = load_protocol_config(protocol_config_path=protocol_config_path, config_dir=config_dir)
    thresh_cfg = load_thresholds_config(thresholds_config_path=thresholds_config_path, config_dir=config_dir)

    ev_cfg = proto_cfg.get("event_counting", {})
    rain_thresh = thresh_cfg.get("rain_thresholds_mm", {})

    cutoff = (
        min_events_primary
        if min_events_primary is not None
        else ev_cfg.get("min_events_primary", 30)
    )
    default_thresh = (
        primary_threshold_mm
        if primary_threshold_mm is not None
        else ev_cfg.get("primary_threshold_mm", rain_thresh.get("heavy", 64.5))
    )
    fallback_thresh = (
        fallback_threshold_mm
        if fallback_threshold_mm is not None
        else ev_cfg.get("fallback_threshold_mm", rain_thresh.get("moderate", 15.6))
    )

    cutoff = int(cutoff)
    default_thresh = float(default_thresh)
    fallback_thresh = float(fallback_thresh)

    if holdout_event_count < cutoff:
        return PrimaryThresholdDecision(
            selected_threshold_mm=fallback_thresh,
            default_threshold_mm=default_thresh,
            fallback_used=True,
            reason=f"INSUFFICIENT_HOLDOUT_EVENTS: {holdout_event_count} < {cutoff}",
            holdout_event_count=holdout_event_count,
        )
    else:
        return PrimaryThresholdDecision(
            selected_threshold_mm=default_thresh,
            default_threshold_mm=default_thresh,
            fallback_used=False,
            reason="SUFFICIENT_HOLDOUT_EVENTS",
            holdout_event_count=holdout_event_count,
        )


@dataclass
class MetricRecord:
    """Represents a single metric evaluation record matching PRD §16.11."""
    value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    undefined_reason: Optional[str] = None

    def __post_init__(self) -> None:
        # Rule: CI must remain None when metric is undefined
        if self.value is None:
            self.ci_low = None
            self.ci_high = None

    def __getitem__(self, key: str) -> Any:
        if key in ("value", "ci_low", "ci_high", "undefined_reason"):
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }
        if self.undefined_reason is not None:
            d["undefined_reason"] = self.undefined_reason
        return d


@dataclass
class VerificationResult:
    """Structured verification result matching PRD §16.11."""
    forecast_type: str
    comparison_to: Optional[str] = None
    evaluation_set: str = "development"
    lead_day: Optional[int] = 1
    threshold_mm: Optional[float] = None
    neighbourhood_cells: Optional[int] = None
    group: Dict[str, str] = field(default_factory=lambda: {"type": "all", "value": "all"})
    metrics: Dict[str, MetricRecord] = field(default_factory=dict)
    paired_difference: Dict[str, MetricRecord] = field(default_factory=dict)
    n_samples: Optional[int] = None
    n_events: Optional[int] = None
    bootstrap: Optional[Dict[str, Any]] = None
    model_version_id: Optional[str] = None
    primary_metric: bool = False
    exploratory: bool = False

    def __getitem__(self, key: str) -> Any:
        if key == "metrics":
            return {k: v.to_dict() for k, v in self.metrics.items()}
        if key == "paired_difference":
            return {k: v.to_dict() for k, v in self.paired_difference.items()}
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_type": self.forecast_type,
            "comparison_to": self.comparison_to,
            "evaluation_set": self.evaluation_set,
            "lead_day": self.lead_day,
            "threshold_mm": self.threshold_mm,
            "neighbourhood_cells": self.neighbourhood_cells,
            "group": dict(self.group),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "paired_difference": {k: v.to_dict() for k, v in self.paired_difference.items()},
            "n_samples": self.n_samples,
            "n_events": self.n_events,
            "bootstrap": dict(self.bootstrap) if self.bootstrap is not None else None,
            "model_version_id": self.model_version_id,
            "primary_metric": self.primary_metric,
            "exploratory": self.exploratory,
        }


CORE_METRIC_NAMES = (
    "rmse",
    "pod",
    "far",
    "csi",
    "ets",
    "fss",
    "frequency_bias",
    "brier_score",
    "brier_skill_score",
)


def create_example_result(
    forecast_type: str = "regime_aware_ml",
    comparison_to: Optional[str] = "raw_nwp",
    evaluation_set: str = "holdout",
    lead_day: Optional[int] = 1,
    threshold_mm: Optional[float] = 64.5,
    neighbourhood_cells: Optional[int] = 5,
    group_type: str = "all",
    group_value: str = "all",
    block_days: int = 7,
    resamples: int = 2000,
    model_version_id: Optional[str] = None,
    primary_metric: bool = True,
    exploratory: bool = False,
) -> VerificationResult:
    """Creates a placeholder VerificationResult matching PRD §16.11 JSON contract."""
    metrics = {
        name: MetricRecord(value=None, ci_low=None, ci_high=None)
        for name in CORE_METRIC_NAMES
    }
    paired_diff = {
        "ets": MetricRecord(value=None, ci_low=None, ci_high=None)
    }

    return VerificationResult(
        forecast_type=forecast_type,
        comparison_to=comparison_to,
        evaluation_set=evaluation_set,
        lead_day=lead_day,
        threshold_mm=threshold_mm,
        neighbourhood_cells=neighbourhood_cells,
        group={"type": group_type, "value": group_value},
        metrics=metrics,
        paired_difference=paired_diff,
        n_samples=None,
        n_events=None,
        bootstrap={"block_days": block_days, "resamples": resamples},
        model_version_id=model_version_id,
        primary_metric=primary_metric,
        exploratory=exploratory,
    )


def package_verification_result(
    component: VerificationComponent,
    comparison_to: Optional[str] = None,
    paired_bootstrap: Optional[Union[Dict[str, BootstrapResult], BootstrapResult]] = None,
    metric_cis: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    bootstrap_metadata: Optional[Dict[str, Any]] = None,
    model_version_id: Optional[str] = None,
    primary_metric: Optional[bool] = None,
    exploratory: Optional[bool] = None,
    primary_threshold_mm: Optional[float] = None,
) -> VerificationResult:
    """Packages aggregated component data and bootstrap results into PRD §16.11 VerificationResult.

    Args:
        component: Aggregated VerificationComponent containing additive sums/counts.
        comparison_to: Reference forecast type (e.g. 'raw_nwp' or 'global_ml').
        paired_bootstrap: Mapping of metric_name -> BootstrapResult, or single BootstrapResult.
        metric_cis: Mapping of metric_name -> (ci_low, ci_high) for model metric values.
        bootstrap_metadata: Optional dictionary with 'block_days' and 'resamples'.
        model_version_id: Optional model artifact identifier.
        primary_metric: Explicit override for primary_metric flag.
        exploratory: Explicit override for exploratory flag.
        primary_threshold_mm: Current primary protocol threshold (e.g. 64.5 or 15.6).

    Returns:
        VerificationResult structured per PRD §16.11.
    """
    raw_metrics = evaluate_all_metrics_from_components(component)

    metrics_records: Dict[str, MetricRecord] = {}

    for name in CORE_METRIC_NAMES:
        m_res: Optional[MetricResult] = raw_metrics.get(name)
        val = m_res.value if m_res is not None else None
        reason = m_res.undefined_reason if m_res is not None else None

        ci_low = None
        ci_high = None
        if val is not None and metric_cis and name in metric_cis:
            ci_pair = metric_cis[name]
            if ci_pair is not None:
                ci_low = ci_pair[0]
                ci_high = ci_pair[1]

        metrics_records[name] = MetricRecord(
            value=val,
            ci_low=ci_low,
            ci_high=ci_high,
            undefined_reason=reason,
        )

    # Coverage metric where applicable (range evaluations)
    if component.range_n > 0 or (metric_cis and "coverage" in metric_cis):
        cov_res = raw_metrics.get("coverage")
        cov_val = cov_res.value if cov_res is not None else None
        cov_reason = cov_res.undefined_reason if cov_res is not None else None

        cov_low = None
        cov_high = None
        if cov_val is not None and metric_cis and "coverage" in metric_cis:
            ci_pair = metric_cis["coverage"]
            if ci_pair is not None:
                cov_low = ci_pair[0]
                cov_high = ci_pair[1]

        metrics_records["coverage"] = MetricRecord(
            value=cov_val,
            ci_low=cov_low,
            ci_high=cov_high,
            undefined_reason=cov_reason,
        )

    # Process paired differences and bootstrap metadata
    paired_diff_records: Dict[str, MetricRecord] = {}
    bs_meta = dict(bootstrap_metadata) if bootstrap_metadata else None

    if paired_bootstrap is not None:
        if isinstance(paired_bootstrap, BootstrapResult):
            m_key = "ets" if component.threshold_mm is not None else "rmse"
            paired_diff_records[m_key] = MetricRecord(
                value=paired_bootstrap.paired_difference,
                ci_low=paired_bootstrap.ci_low,
                ci_high=paired_bootstrap.ci_high,
            )
            if bs_meta is None:
                bs_meta = {
                    "block_days": paired_bootstrap.block_days,
                    "resamples": paired_bootstrap.n_resamples,
                }
        elif isinstance(paired_bootstrap, dict):
            for k, bs in paired_bootstrap.items():
                canon_k = k.lower()
                paired_diff_records[canon_k] = MetricRecord(
                    value=bs.paired_difference,
                    ci_low=bs.ci_low,
                    ci_high=bs.ci_high,
                )
                if bs_meta is None:
                    bs_meta = {
                        "block_days": bs.block_days,
                        "resamples": bs.n_resamples,
                    }

    # Determine primary_metric and exploratory flags per protocol
    if primary_metric is not None:
        is_primary = bool(primary_metric)
    else:
        if component.threshold_mm is None:
            # Continuous metric (P1 RMSE)
            is_primary = True
        else:
            ref_thresh = primary_threshold_mm if primary_threshold_mm is not None else 64.5
            is_primary = abs(float(component.threshold_mm) - float(ref_thresh)) < 1e-4

    if exploratory is not None:
        is_exploratory = bool(exploratory)
    else:
        is_exploratory = not is_primary

    return VerificationResult(
        forecast_type=component.forecast_type,
        comparison_to=comparison_to,
        evaluation_set=component.evaluation_set,
        lead_day=component.lead_day,
        threshold_mm=component.threshold_mm,
        neighbourhood_cells=component.neighbourhood_cells,
        group={"type": component.group_type, "value": component.group_value},
        metrics=metrics_records,
        paired_difference=paired_diff_records,
        n_samples=component.n_samples,
        n_events=component.n_events,
        bootstrap=bs_meta,
        model_version_id=model_version_id,
        primary_metric=is_primary,
        exploratory=is_exploratory,
    )
