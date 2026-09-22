"""Active and break spell detection based on Rajeevan et al. (2010).

PRD Section 11.2:
A break (active) spell is a period when the standardised rainfall of the
core monsoon zone is below -1 (above +1) for at least 3 days in a row.
All days in the qualifying spell are labelled active / break.
Normal = everything else.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def detect_spells(
    z_series: pd.Series,
    active_z_min: float = 1.0,
    break_z_max: float = -1.0,
    min_run_days: int = 3,
) -> pd.Series:
    """Label each date as 'active', 'break', or 'normal'.

    Args:
        z_series: pd.Series of standardized anomalies indexed chronologically by date.
        active_z_min: Threshold for active spell (default: +1.0).
        break_z_max: Threshold for break spell (default: -1.0).
        min_run_days: Minimum consecutive days required to constitute a spell (default: 3).

    Returns:
        pd.Series of strings ('active', 'break', 'normal') indexed by the same dates.
    """
    n = len(z_series)
    labels = np.array(["normal"] * n, dtype=object)
    values = z_series.to_numpy(dtype=float)

    # 1. Identify active spells (>= min_run_days with z > active_z_min)
    in_active = False
    start_idx = 0
    for i in range(n):
        val = values[i]
        is_above = not np.isnan(val) and val > active_z_min
        if is_above:
            if not in_active:
                in_active = True
                start_idx = i
        else:
            if in_active:
                if (i - start_idx) >= min_run_days:
                    labels[start_idx:i] = "active"
                in_active = False
    # Check end of series
    if in_active and (n - start_idx) >= min_run_days:
        labels[start_idx:n] = "active"

    # 2. Identify break spells (>= min_run_days with z < break_z_max)
    in_break = False
    start_idx = 0
    for i in range(n):
        val = values[i]
        is_below = not np.isnan(val) and val < break_z_max
        if is_below:
            if not in_break:
                in_break = True
                start_idx = i
        else:
            if in_break:
                if (i - start_idx) >= min_run_days:
                    labels[start_idx:i] = "break"
                in_break = False
    # Check end of series
    if in_break and (n - start_idx) >= min_run_days:
        labels[start_idx:n] = "break"

    return pd.Series(labels, index=z_series.index, name="phase_label")
