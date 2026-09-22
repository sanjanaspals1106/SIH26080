"""Statistical comparison tests for forecast evaluation (PRD §16.6).

Implements:
1. Season consistency (§16.6 item 4):
   Reports how many seasons the corrected forecast was better.
2. Wilcoxon signed-rank test (§16.6 item 4):
   Paired Wilcoxon test on per-season differences for >= 6 seasons.
3. Diebold-Mariano test with HAC variance (§16.6 item 5):
   Equal-accuracy test on daily loss values with autocorrelation-robust variance.

Rules:
- Never use ordinary t-tests or independent sample assumptions on daily/cell data.
- Paired comparisons only on identical dates and seasons.
- Missing values excluded without corrupting alignment.
- Metric direction must be explicitly supplied ("lower" or "higher"); never inferred from names.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import stats
import yaml

MIN_WILCOXON_SEASONS: int = 6
MIN_DM_DATES: int = 10


@dataclass(frozen=True)
class SeasonConsistencyResult:
    """Result of season consistency evaluation (PRD §16.6 item 4)."""
    n_seasons: int
    n_seasons_a_better: int
    n_seasons_b_better: int
    n_ties: int
    direction: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_seasons": self.n_seasons,
            "n_seasons_a_better": self.n_seasons_a_better,
            "n_seasons_b_better": self.n_seasons_b_better,
            "n_ties": self.n_ties,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class WilcoxonResult:
    """Result of Wilcoxon signed-rank test on per-season differences (PRD §16.6 item 4)."""
    statistic: Optional[float]
    p_value: Optional[float]
    n_seasons: int
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_seasons": self.n_seasons,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Result of Diebold-Mariano test on daily loss differentials (PRD §16.6 item 5)."""
    dm_statistic: Optional[float]
    p_value: Optional[float]
    mean_loss_difference: Optional[float]
    n_dates: int
    lag: int
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dm_statistic": self.dm_statistic,
            "p_value": self.p_value,
            "mean_loss_difference": self.mean_loss_difference,
            "n_dates": self.n_dates,
            "lag": self.lag,
            "reason": self.reason,
        }


def _align_paired_series(
    series_a: Union[Dict[Any, float], Sequence[float], np.ndarray],
    series_b: Union[Dict[Any, float], Sequence[float], np.ndarray],
    keys: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Aligns two series by key or position, dropping pairs with missing values (None or NaN)."""
    is_dict_a = isinstance(series_a, dict)
    is_dict_b = isinstance(series_b, dict)

    vals_a: List[float] = []
    vals_b: List[float] = []
    matched_keys: List[Any] = []

    if is_dict_a and is_dict_b:
        common_keys = sorted(set(series_a.keys()) & set(series_b.keys()))
        for k in common_keys:
            va = series_a[k]
            vb = series_b[k]
            if va is not None and vb is not None:
                if not (np.isnan(float(va)) or np.isnan(float(vb))):
                    vals_a.append(float(va))
                    vals_b.append(float(vb))
                    matched_keys.append(k)
    elif keys is not None:
        if len(keys) != len(series_a) or len(keys) != len(series_b):
            raise ValueError("Length of keys must match length of series_a and series_b")
        for k, va, vb in zip(keys, series_a, series_b):
            if va is not None and vb is not None:
                if not (np.isnan(float(va)) or np.isnan(float(vb))):
                    vals_a.append(float(va))
                    vals_b.append(float(vb))
                    matched_keys.append(k)
    else:
        seq_a = list(series_a)
        seq_b = list(series_b)
        n = min(len(seq_a), len(seq_b))
        for idx in range(n):
            va = seq_a[idx]
            vb = seq_b[idx]
            if va is not None and vb is not None:
                if not (np.isnan(float(va)) or np.isnan(float(vb))):
                    vals_a.append(float(va))
                    vals_b.append(float(vb))
                    matched_keys.append(idx)

    return np.array(vals_a, dtype=float), np.array(vals_b, dtype=float), matched_keys


# -----------------------------------------------------------------------------
# 1. Season Consistency (PRD §16.6 item 4)
# -----------------------------------------------------------------------------

def compute_season_consistency(
    metric_a: Union[Dict[Any, float], Sequence[float], np.ndarray],
    metric_b: Union[Dict[Any, float], Sequence[float], np.ndarray],
    direction: str,
    seasons: Optional[Sequence[Any]] = None,
    eps: float = 1e-9,
) -> SeasonConsistencyResult:
    """Evaluates how many seasons system A vs system B was better (PRD §16.6 item 4).
    
    Args:
        metric_a: Per-season metric values for system A (e.g. B3 or AI-corrected).
        metric_b: Per-season metric values for system B (e.g. B2 or raw NWP).
        direction: "lower" (for RMSE) or "higher" (for ETS, FSS, BSS). Must be explicitly supplied.
        seasons: Optional sequence of season labels when metric inputs are sequences.
        eps: Tolerance for counting ties.
        
    Returns:
        SeasonConsistencyResult with n_seasons, n_seasons_a_better, n_seasons_b_better, n_ties.
    """
    dir_clean = str(direction).strip().lower()
    if dir_clean not in ("lower", "higher"):
        raise ValueError(
            f"Invalid metric direction: '{direction}'. Must be explicitly 'lower' or 'higher'."
        )

    arr_a, arr_b, _ = _align_paired_series(metric_a, metric_b, keys=seasons)
    n_seasons = len(arr_a)

    if n_seasons == 0:
        return SeasonConsistencyResult(
            n_seasons=0,
            n_seasons_a_better=0,
            n_seasons_b_better=0,
            n_ties=0,
            direction=dir_clean,
        )

    diff = arr_a - arr_b  # diff < 0 means A < B; diff > 0 means A > B

    if dir_clean == "lower":
        # A is better when A < B (diff < -eps)
        a_better = int(np.count_nonzero(diff < -eps))
        b_better = int(np.count_nonzero(diff > eps))
        ties = int(np.count_nonzero(np.abs(diff) <= eps))
    else:  # "higher"
        # A is better when A > B (diff > eps)
        a_better = int(np.count_nonzero(diff > eps))
        b_better = int(np.count_nonzero(diff < -eps))
        ties = int(np.count_nonzero(np.abs(diff) <= eps))

    return SeasonConsistencyResult(
        n_seasons=n_seasons,
        n_seasons_a_better=a_better,
        n_seasons_b_better=b_better,
        n_ties=ties,
        direction=dir_clean,
    )


# -----------------------------------------------------------------------------
# 2. Wilcoxon Signed-Rank Test (PRD §16.6 item 4)
# -----------------------------------------------------------------------------

def compute_season_wilcoxon(
    metric_a: Union[Dict[Any, float], Sequence[float], np.ndarray],
    metric_b: Union[Dict[Any, float], Sequence[float], np.ndarray],
    seasons: Optional[Sequence[Any]] = None,
    min_seasons: int = MIN_WILCOXON_SEASONS,
) -> WilcoxonResult:
    """Performs Wilcoxon signed-rank test on per-season metric differences (PRD §16.6 item 4).
    
    Rules:
    - Only runs if at least 6 valid paired seasons exist.
    - Operates on paired per-season metric differences.
    - Drops pairs with missing values.
    - If fewer than 6 valid seasons: returns statistic=None, p_value=None, reason='INSUFFICIENT_SEASONS'.
    - Never converts failure/undefined cases to p=1 or 0.
    """
    arr_a, arr_b, _ = _align_paired_series(metric_a, metric_b, keys=seasons)
    n_seasons = len(arr_a)

    if n_seasons < min_seasons:
        return WilcoxonResult(
            statistic=None,
            p_value=None,
            n_seasons=n_seasons,
            reason="INSUFFICIENT_SEASONS",
        )

    diffs = arr_a - arr_b

    # If all differences are zero, ranks are undefined
    if np.all(diffs == 0.0):
        return WilcoxonResult(
            statistic=None,
            p_value=None,
            n_seasons=n_seasons,
            reason="ALL_ZERO_DIFFERENCES",
        )

    try:
        res = stats.wilcoxon(diffs)
        stat = float(res.statistic)
        pval = float(res.pvalue)
        return WilcoxonResult(
            statistic=stat,
            p_value=pval,
            n_seasons=n_seasons,
            reason=None,
        )
    except Exception as e:
        return WilcoxonResult(
            statistic=None,
            p_value=None,
            n_seasons=n_seasons,
            reason=f"ERROR: {e}",
        )


# -----------------------------------------------------------------------------
# 3. Diebold-Mariano Test for RMSE Comparison (PRD §16.6 item 5)
# -----------------------------------------------------------------------------

def get_dm_lag(
    n_dates: int,
    lag: Optional[int] = None,
    lead_day: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> int:
    """Selects deterministic lag for HAC covariance estimation in the Diebold-Mariano test.
    
    Selection hierarchy:
    1. Explicit lag if provided by caller.
    2. Config parameter diebold_mariano.lag from config/verification.yaml if present.
    3. For multi-step forecasts: lag = lead_day - 1 (since h-step forecasts have at most MA(h-1) errors).
    4. Deterministic sample-size based rule: floor(4 * (T / 100)^(2/9)) clamped to [1, T-1].
    """
    if lag is not None:
        return max(1, int(lag))

    # Check config
    cfg_file = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parent.parent.parent / "config" / "verification.yaml"
    )
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg_lag = cfg.get("diebold_mariano", {}).get("lag")
            if cfg_lag is not None:
                return max(1, int(cfg_lag))
        except Exception:
            pass

    # Lead-day rule
    if lead_day is not None and int(lead_day) > 1:
        return max(1, int(lead_day) - 1)

    # Deterministic rule of thumb for daily data
    t_effective = max(n_dates, 1)
    rule_lag = int(np.floor(4.0 * (t_effective / 100.0) ** (2.0 / 9.0)))
    return max(1, min(rule_lag, max(1, t_effective - 1)))


def compute_diebold_mariano(
    loss_a: Union[Dict[Any, float], Sequence[float], np.ndarray],
    loss_b: Union[Dict[Any, float], Sequence[float], np.ndarray],
    dates: Optional[Sequence[Any]] = None,
    lag: Optional[int] = None,
    lead_day: Optional[int] = None,
    min_dates: int = MIN_DM_DATES,
    config_path: Optional[Union[str, Path]] = None,
) -> DieboldMarianoResult:
    """Diebold-Mariano test for equal predictive accuracy with HAC variance (PRD §16.6 item 5).
    
    Uses daily loss values (e.g. squared errors for RMSE).
    Loss differential: d_t = loss_A(t) - loss_B(t).
    Tested under H0: E[d_t] = 0.
    
    Args:
        loss_a: Daily loss series for system A.
        loss_b: Daily loss series for system B.
        dates: Optional sequence of dates for alignment when losses are sequences.
        lag: Truncation lag for HAC variance. If None, selected deterministically.
        lead_day: Forecast lead day. Used to select default lag = lead_day - 1.
        min_dates: Minimum number of paired dates required (default 10).
        config_path: Path to config/verification.yaml.
        
    Returns:
        DieboldMarianoResult with dm_statistic, p_value, mean_loss_difference, n_dates, lag, reason.
    """
    arr_a, arr_b, _ = _align_paired_series(loss_a, loss_b, keys=dates)
    n_dates = len(arr_a)

    effective_lag = get_dm_lag(
        n_dates=n_dates,
        lag=lag,
        lead_day=lead_day,
        config_path=config_path,
    )

    if n_dates < min_dates:
        return DieboldMarianoResult(
            dm_statistic=None,
            p_value=None,
            mean_loss_difference=None,
            n_dates=n_dates,
            lag=effective_lag,
            reason="INSUFFICIENT_DATES",
        )

    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))

    # Check for identical series (loss_A == loss_B)
    if np.all(np.abs(diff) < 1e-12):
        return DieboldMarianoResult(
            dm_statistic=0.0,
            p_value=1.0,
            mean_loss_difference=0.0,
            n_dates=n_dates,
            lag=effective_lag,
            reason="IDENTICAL_SERIES",
        )

    # Compute HAC long-run variance with Bartlett kernel (Newey-West)
    z = diff - mean_diff
    t_float = float(n_dates)

    gamma_0 = float(np.sum(z ** 2) / t_float)
    v_long_run = gamma_0

    # Sum autocovariances with Bartlett damping weights: w_k = 1 - k / (lag + 1)
    for k in range(1, effective_lag + 1):
        if k < n_dates:
            gamma_k = float(np.sum(z[k:] * z[:-k]) / t_float)
            w_k = 1.0 - (float(k) / float(effective_lag + 1))
            v_long_run += 2.0 * w_k * gamma_k

    var_mean = v_long_run / t_float

    if var_mean <= 0.0:
        stat = float("inf") if mean_diff > 0 else float("-inf")
        return DieboldMarianoResult(
            dm_statistic=stat,
            p_value=0.0,
            mean_loss_difference=mean_diff,
            n_dates=n_dates,
            lag=effective_lag,
            reason="ZERO_VARIANCE",
        )

    se = np.sqrt(var_mean)
    dm_stat = float(mean_diff / se)

    # Two-sided p-value against asymptotic standard normal
    p_val = float(2.0 * stats.norm.sf(abs(dm_stat)))

    return DieboldMarianoResult(
        dm_statistic=dm_stat,
        p_value=p_val,
        mean_loss_difference=mean_diff,
        n_dates=n_dates,
        lag=effective_lag,
        reason=None,
    )
