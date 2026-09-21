"""Climatology and standardized anomaly computation.

PRD Section 11.2 & Appendix B:
Base period: IMD 1981-2010.
Computes day-of-year mean and standard deviation using a 31-day window
centred on each day of the year to smooth them.
Standardised anomaly: z = (rain - mean) / std.
"""

from typing import Union
import numpy as np
import pandas as pd


def compute_doy_climatology(
    daily_series: pd.Series,
    window_days: int = 31,
) -> pd.DataFrame:
    """Compute day-of-year mean and std from a daily series with a rolling window.

    The window is centered on each day of the year (doy). To avoid boundary artifacts,
    the calendar wraps around circularly across year boundaries (e.g. late Dec to early Jan).

    Args:
        daily_series: pd.Series of core zone daily rainfall indexed by pd.DatetimeIndex
                      or strings convertible to DatetimeIndex.
        window_days: Window size in days (default 31 days per PRD).

    Returns:
        pd.DataFrame indexed by doy (1 to 366) with columns ['mean', 'std'].
    """
    if not isinstance(daily_series.index, pd.DatetimeIndex):
        s = daily_series.copy()
        s.index = pd.to_datetime(s.index)
    else:
        s = daily_series

    # Group by doy first or collect all values for doy +/- half_win
    half_win = window_days // 2

    # Group values by day of year (1..366)
    doy_to_values = {d: [] for d in range(1, 367)}
    for date, val in s.items():
        if pd.notna(val):
            doy_to_values[date.dayofyear].append(float(val))

    # For each doy, aggregate all observations in [doy - half_win, doy + half_win] modulo 366
    records = []
    for d in range(1, 367):
        window_vals = []
        for offset in range(-half_win, half_win + 1):
            target_doy = ((d + offset - 1) % 366) + 1
            window_vals.extend(doy_to_values[target_doy])

        if len(window_vals) > 0:
            arr = np.array(window_vals, dtype=float)
            m = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
            if sd <= 1e-6:
                sd = 1.0  # Prevent zero division
        else:
            m = 0.0
            sd = 1.0

        records.append({"doy": d, "mean": m, "std": sd})

    df_clim = pd.DataFrame(records).set_index("doy")
    return df_clim


def compute_standardized_anomalies(
    daily_series: pd.Series,
    climatology_df: pd.DataFrame,
) -> pd.Series:
    """Compute standardized anomalies z = (rain - mean) / std.

    Args:
        daily_series: pd.Series of core zone daily rainfall indexed by date.
        climatology_df: DataFrame indexed by doy with ['mean', 'std'].

    Returns:
        pd.Series of z-scores matching daily_series index.
    """
    if not isinstance(daily_series.index, pd.DatetimeIndex):
        s = daily_series.copy()
        s.index = pd.to_datetime(s.index)
    else:
        s = daily_series

    z_vals = []
    for date, val in s.items():
        doy = date.dayofyear
        if doy in climatology_df.index and pd.notna(val):
            m = climatology_df.loc[doy, "mean"]
            sd = climatology_df.loc[doy, "std"]
            z = (val - m) / sd
        else:
            z = np.nan
        z_vals.append(z)

    return pd.Series(z_vals, index=daily_series.index, name="z_score")
