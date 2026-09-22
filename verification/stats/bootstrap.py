"""Paired block bootstrap for forecast verification (PRD §16.6, Appendix A).

Rules (PRD §16.6):
- Dates are the resampling unit. Never bootstrap individual cells or days independently.
- Resample blocks of L consecutive dates with replacement until original date count is reached.
- Truncate any excess beyond the original date count.
- Use EXACTLY the same sampled date blocks for both systems in a comparison.
- Recompute the metric from stored additive components each resample.
- Direction is NOT interpreted here; returns numeric paired difference (metric_a - metric_b) and CI.
- Default L = 7 days (also 3 and 14 days), 2000 resamples, deterministic seed.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml

from verification.metrics.contingency import ContingencyCounts, compute_ets
from verification.metrics.spatial import FSSComponents, compute_fss_from_components


@dataclass(frozen=True)
class RMSEComponents:
    """Additive components for Root Mean Squared Error (RMSE)."""
    n_samples: int
    sum_squared_errors: float


@dataclass(frozen=True)
class BrierComponents:
    """Additive components for Brier Score."""
    n_samples: int
    sum_brier_terms: float


@dataclass(frozen=True)
class BootstrapResult:
    """Result of paired block bootstrap evaluation."""
    metric_a: float
    metric_b: float
    paired_difference: float
    ci_low: float
    ci_high: float
    block_days: int
    n_resamples: int
    n_valid_resamples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "paired_difference": self.paired_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "block_days": self.block_days,
            "n_resamples": self.n_resamples,
            "n_valid_resamples": self.n_valid_resamples,
        }


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_bootstrap_config(
    verification_config_path: Optional[Union[str, Path]] = None,
    protocol_config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Loads default bootstrap parameters from verification.yaml and protocol.yaml."""
    repo_root = _find_repo_root()
    v_path = (
        Path(verification_config_path)
        if verification_config_path
        else repo_root / "config" / "verification.yaml"
    )
    p_path = (
        Path(protocol_config_path)
        if protocol_config_path
        else repo_root / "config" / "protocol.yaml"
    )

    block_days = 7
    n_resamples = 2000
    ci_level = 0.95
    seed = 42

    if v_path.is_file():
        with open(v_path, "r", encoding="utf-8") as f:
            v_cfg = yaml.safe_load(f)
        b_sec = v_cfg.get("bootstrap", {})
        block_days = int(b_sec.get("primary_block_days", 7))
        n_resamples = int(b_sec.get("resamples", 2000))
        ci_level = float(b_sec.get("ci_level", 0.95))

    if p_path.is_file():
        with open(p_path, "r", encoding="utf-8") as f:
            p_cfg = yaml.safe_load(f)
        s_val = p_cfg.get("random_seed")
        if s_val is not None:
            seed = int(s_val)

    return {
        "block_days": block_days,
        "n_resamples": n_resamples,
        "ci_level": ci_level,
        "seed": seed,
    }


def _recompute_rmse(components_list: Sequence[RMSEComponents]) -> Optional[float]:
    """Recomputes RMSE: sqrt(sum_squared_errors / n).

    Must be the square root of mean squared error (RMSE != MSE).
    """
    n_total = sum(c.n_samples for c in components_list)
    if n_total == 0:
        return None
    sse_total = sum(c.sum_squared_errors for c in components_list)
    return float(np.sqrt(sse_total / n_total))



def _recompute_ets(components_list: Sequence[ContingencyCounts]) -> Optional[float]:
    """Recomputes ETS from summed contingency counts."""
    a = sum(c.a for c in components_list)
    b = sum(c.b for c in components_list)
    c = sum(c.c for c in components_list)
    d = sum(c.d for c in components_list)
    res = compute_ets(a, b, c, d)
    return res.value


def _recompute_fss(components_list: Sequence[FSSComponents]) -> Optional[float]:
    """Recomputes FSS from summed FSS components."""
    if not components_list:
        return None
    total = components_list[0]
    for c in components_list[1:]:
        total = total + c
    res = compute_fss_from_components(total)
    return res.value


def _recompute_brier(components_list: Sequence[BrierComponents]) -> Optional[float]:
    """Recomputes Brier score from summed squared probability errors."""
    n_total = sum(c.n_samples for c in components_list)
    if n_total == 0:
        return None
    sb_total = sum(c.sum_brier_terms for c in components_list)
    return float(sb_total / n_total)


def _get_metric_evaluator(
    metric: Union[str, Callable[[Sequence[Any]], Optional[float]]],
) -> Callable[[Sequence[Any]], Optional[float]]:
    """Resolves metric name or callable to component evaluation function."""
    if callable(metric):
        return metric

    metric_name = str(metric).lower().strip()
    if metric_name == "rmse":
        return _recompute_rmse
    elif metric_name == "ets":
        return _recompute_ets
    elif metric_name == "fss":
        return _recompute_fss
    elif metric_name in ("brier", "brier_score"):
        return _recompute_brier
    else:
        raise ValueError(
            f"Unsupported metric: {metric!r}. Choose from 'rmse', 'ets', 'fss', 'brier', or pass a callable."
        )


def _parse_date(d: Union[date, datetime, str]) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d.strip()[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        return d.date()
    raise TypeError(f"Cannot convert {d!r} to date.")


def paired_block_bootstrap(
    dates: Sequence[Union[date, datetime, str]],
    components_a: Sequence[Any],
    components_b: Sequence[Any],
    metric: Union[str, Callable[[Sequence[Any]], Optional[float]]],
    block_days: Optional[int] = None,
    n_resamples: Optional[int] = None,
    ci_level: Optional[float] = None,
    random_seed: Optional[int] = None,
    min_valid_ratio: float = 0.5,
) -> BootstrapResult:
    """Executes paired block bootstrap on two forecasting systems (PRD §16.6).

    Args:
        dates: Sequence of observation dates.
        components_a: Additive components for System A on each date.
        components_b: Additive components for System B on each date.
        metric: Metric to evaluate ('rmse', 'ets', 'fss', 'brier' or callable).
        block_days: Block length in days (default 7, supports 3, 14).
        n_resamples: Number of bootstrap resamples (default 2000).
        ci_level: Confidence interval coverage (default 0.95).
        random_seed: Deterministic random seed from config/protocol.yaml.
        min_valid_ratio: Minimum fraction of valid non-undefined resamples required.

    Returns:
        BootstrapResult with point estimates, paired difference, and confidence interval.
    """
    cfg = load_bootstrap_config()
    L = int(block_days if block_days is not None else cfg["block_days"])
    B = int(n_resamples if n_resamples is not None else cfg["n_resamples"])
    level = float(ci_level if ci_level is not None else cfg["ci_level"])
    seed = int(random_seed if random_seed is not None else cfg["seed"])

    if L < 1:
        raise ValueError(f"Block length must be at least 1 day, got {L}.")
    if B < 1:
        raise ValueError(f"Number of resamples must be at least 1, got {B}.")
    if not (0.0 < level < 1.0):
        raise ValueError(f"Confidence level must be in (0, 1), got {level}.")

    n_dates = len(dates)
    if not (len(components_a) == len(components_b) == n_dates):
        raise ValueError(
            f"Length mismatch: dates ({n_dates}), components_a ({len(components_a)}), "
            f"components_b ({len(components_b)})."
        )

    if n_dates == 0:
        raise ValueError("Cannot run bootstrap on empty date sequence.")
    if n_dates < L:
        raise ValueError(
            f"Sample has {n_dates} dates, which is less than block length L={L}. Cannot form blocks."
        )

    # Sort chronologically
    parsed_dates = [_parse_date(d) for d in dates]
    sort_idx = sorted(range(n_dates), key=lambda i: parsed_dates[i])

    sorted_comps_a = [components_a[i] for i in sort_idx]
    sorted_comps_b = [components_b[i] for i in sort_idx]

    eval_func = _get_metric_evaluator(metric)

    # Point estimates on the original full sample
    val_a = eval_func(sorted_comps_a)
    val_b = eval_func(sorted_comps_b)

    if val_a is None or val_b is None:
        raise ValueError(
            f"Metric evaluation on full sample failed or was undefined (metric_a={val_a}, metric_b={val_b})."
        )

    paired_diff_point = float(val_a - val_b)

    # Moving block bootstrap setup
    # Possible block start indices: 0 .. n_dates - L
    k_starts = n_dates - L + 1
    rng = np.random.default_rng(seed)

    paired_diffs: List[float] = []

    for _ in range(B):
        sample_indices: List[int] = []
        while len(sample_indices) < n_dates:
            start = int(rng.integers(0, k_starts))
            sample_indices.extend(range(start, start + L))

        # Truncate excess so sample size matches original date count
        sample_indices = sample_indices[:n_dates]

        # Use EXACTLY the same sampled blocks for both systems
        resamp_a = [sorted_comps_a[i] for i in sample_indices]
        resamp_b = [sorted_comps_b[i] for i in sample_indices]

        ma_res = eval_func(resamp_a)
        mb_res = eval_func(resamp_b)

        # Skip undefined resamples per PRD §16.6 / §16.7
        if ma_res is not None and mb_res is not None:
            paired_diffs.append(float(ma_res - mb_res))

    n_valid = len(paired_diffs)
    # Implementation safeguard: ensure sufficient valid resamples for reliable empirical quantiles.
    # Note: this safeguard is an engineering safeguard against degenerate data, NOT a PRD rule.
    min_required = int(np.ceil(B * min_valid_ratio))

    if n_valid < min_required:
        raise RuntimeError(
            f"Too few valid bootstrap resamples ({n_valid} / {B}). "
            f"At least {min_required} required ({min_valid_ratio*100:.0f}%). "
            "Undefined metrics in resamples prevented reliable confidence interval calculation."
        )

    # Percentiles for two-sided (1 - alpha) CI
    alpha = 1.0 - level
    pct_low = 100.0 * (alpha / 2.0)
    pct_high = 100.0 * (1.0 - alpha / 2.0)

    diff_arr = np.asarray(paired_diffs, dtype=float)
    ci_low = float(np.percentile(diff_arr, pct_low))
    ci_high = float(np.percentile(diff_arr, pct_high))

    return BootstrapResult(
        metric_a=float(val_a),
        metric_b=float(val_b),
        paired_difference=paired_diff_point,
        ci_low=ci_low,
        ci_high=ci_high,
        block_days=L,
        n_resamples=B,
        n_valid_resamples=n_valid,
    )
