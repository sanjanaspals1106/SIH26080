"""Validation and sanity check for monsoon phase labels.

PRD Section 11.2 & Check CK7 & Appendix B:
July-August statistics check on 1981-2010:
- Average active days per season: 7 +/- 3 (4 to 10 days)
- Average break days per season: 7 +/- 3 (4 to 10 days)
- Share of seasons without break: 26% +/- 10% (16% to 36%)
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def check_july_august_statistics(
    labels_series: pd.Series,
    active_target: float = 7.0,
    active_tol: float = 3.0,
    break_target: float = 7.0,
    break_tol: float = 3.0,
    no_break_share_target: float = 0.26,
    no_break_share_tol: float = 0.10,
) -> Dict[str, Any]:
    """Run July-August statistics check on 1981-2010 labels series.

    Args:
        labels_series: pd.Series containing labels ('active', 'break', 'normal')
                       with DatetimeIndex or string dates.
        active_target: Target active days per season (default 7.0).
        active_tol: Tolerance for active days (default 3.0).
        break_target: Target break days per season (default 7.0).
        break_tol: Tolerance for break days (default 3.0).
        no_break_share_target: Target share of seasons without breaks (default 0.26).
        no_break_share_tol: Tolerance for no-break share (default 0.10).

    Returns:
        Dictionary with summary statistics and boolean 'passed'.
    """
    if not isinstance(labels_series.index, pd.DatetimeIndex):
        s = labels_series.copy()
        s.index = pd.to_datetime(s.index)
    else:
        s = labels_series

    # Filter to July and August only (month 7 and 8)
    ja_mask = s.index.month.isin([7, 8])
    ja_labels = s.loc[ja_mask]

    years = np.unique(ja_labels.index.year)
    if len(years) == 0:
        return {
            "passed": False,
            "error": "No July-August dates found in series.",
        }

    active_counts = []
    break_counts = []
    no_break_count = 0

    for yr in years:
        yr_sub = ja_labels.loc[ja_labels.index.year == yr]
        n_act = int((yr_sub == "active").sum())
        n_brk = int((yr_sub == "break").sum())
        active_counts.append(n_act)
        break_counts.append(n_brk)
        if n_brk == 0:
            no_break_count += 1

    mean_active = float(np.mean(active_counts))
    mean_break = float(np.mean(break_counts))
    no_break_share = float(no_break_count / len(years))

    active_pass = (active_target - active_tol) <= mean_active <= (active_target + active_tol)
    break_pass = (break_target - break_tol) <= mean_break <= (break_target + break_tol)
    no_break_pass = (no_break_share_target - no_break_share_tol) <= no_break_share <= (no_break_share_target + no_break_share_tol)

    all_passed = bool(active_pass and break_pass and no_break_pass)

    return {
        "passed": all_passed,
        "n_seasons": len(years),
        "mean_active_days": mean_active,
        "active_days_pass": bool(active_pass),
        "mean_break_days": mean_break,
        "break_days_pass": bool(break_pass),
        "no_break_share": no_break_share,
        "no_break_share_pass": bool(no_break_pass),
    }


def compare_with_published_list(
    computed_labels: pd.Series,
    published_labels: pd.Series,
) -> Dict[str, Any]:
    """Compare computed active/break labels against a published reference list (CK7).

    Args:
        computed_labels: pd.Series with computed labels ('active', 'break', 'normal').
        published_labels: pd.Series with reference labels.

    Returns:
        Dictionary with matching statistics.
    """
    common_idx = computed_labels.index.intersection(published_labels.index)
    if len(common_idx) == 0:
        return {"matched": 0, "total": 0, "match_rate": 0.0}

    c = computed_labels.loc[common_idx]
    p = published_labels.loc[common_idx]
    matches = int((c == p).sum())
    total = len(common_idx)

    return {
        "matched": matches,
        "total": total,
        "match_rate": float(matches / total) if total > 0 else 0.0,
    }
