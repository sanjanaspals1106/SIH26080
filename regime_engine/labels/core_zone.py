"""Core monsoon zone daily rainfall aggregator.

PRD Section 11.2 & Appendix B:
Core monsoon zone box: 18.0-28.0 N, 68.0-88.0 E.
Computes daily mean IMD rainfall over valid cells inside this box.
"""

from typing import Optional, Sequence
import numpy as np
import pandas as pd


def compute_core_zone_daily_mean(
    rainfall_df: pd.DataFrame,
    valid_cells: Optional[Sequence[int]] = None,
    lat_min: float = 18.0,
    lat_max: float = 28.0,
    lon_min: float = 68.0,
    lon_max: float = 88.0,
) -> pd.Series:
    """Compute daily average rainfall over valid cells in the core monsoon zone box.

    Args:
        rainfall_df: DataFrame containing at least ['imd_date', 'cell_id', 'latitude', 'longitude', 'rain_mm'].
        valid_cells: Optional set/sequence of valid cell_ids. If provided, filters to these cells.
        lat_min: Minimum latitude of core zone (default: 18.0).
        lat_max: Maximum latitude of core zone (default: 28.0).
        lon_min: Minimum longitude of core zone (default: 68.0).
        lon_max: Maximum longitude of core zone (default: 88.0).

    Returns:
        pd.Series indexed by imd_date with mean rainfall in mm.
    """
    df = rainfall_df

    # Filter to core zone spatial box
    mask = (
        (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
        & (df["longitude"] >= lon_min)
        & (df["longitude"] <= lon_max)
    )

    if valid_cells is not None:
        valid_set = set(valid_cells)
        mask &= df["cell_id"].isin(valid_set)

    # Filter out missing values (-999 or NaN)
    mask &= df["rain_mm"].notna() & (df["rain_mm"] >= 0)

    filtered = df.loc[mask]
    if filtered.empty:
        return pd.Series(dtype=float)

    daily_means = filtered.groupby("imd_date")["rain_mm"].mean()
    return daily_means
