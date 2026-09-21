"""Layer A monsoon phase labeling module.

PRD Section 11.2 & Appendix B.
Implements the published Rajeevan et al. (2010) active/break spell detection
over the core monsoon zone (18-28°N, 68-88°E) with 1981-2010 climatology.
"""

from typing import Optional, Sequence
import pandas as pd

from regime_engine.labels.core_zone import compute_core_zone_daily_mean
from regime_engine.labels.climatology import (
    compute_doy_climatology,
    compute_standardized_anomalies,
)
from regime_engine.labels.spell_detector import detect_spells
from regime_engine.labels.label_checker import (
    check_july_august_statistics,
    compare_with_published_list,
)


def generate_phase_labels(
    rainfall_df: pd.DataFrame,
    climatology_df: pd.DataFrame,
    valid_cells: Optional[Sequence[int]] = None,
    active_z_min: float = 1.0,
    break_z_max: float = -1.0,
    min_run_days: int = 3,
) -> pd.DataFrame:
    """Generate daily phase labels ('active', 'break', 'normal') and z-scores from rainfall.

    Args:
        rainfall_df: DataFrame with ['imd_date', 'cell_id', 'latitude', 'longitude', 'rain_mm'].
        climatology_df: 1981-2010 DOY climatology table with ['mean', 'std'].
        valid_cells: Optional list/set of valid cell IDs.
        active_z_min: Active spell z-score threshold (+1.0).
        break_z_max: Break spell z-score threshold (-1.0).
        min_run_days: Consecutive days required (3).

    Returns:
        pd.DataFrame indexed by imd_date with columns ['core_rain_mm', 'z_score', 'phase_label'].
    """
    daily_core = compute_core_zone_daily_mean(rainfall_df, valid_cells=valid_cells)
    z_series = compute_standardized_anomalies(daily_core, climatology_df)
    labels = detect_spells(
        z_series,
        active_z_min=active_z_min,
        break_z_max=break_z_max,
        min_run_days=min_run_days,
    )

    out_df = pd.DataFrame(
        {
            "core_rain_mm": daily_core,
            "z_score": z_series,
            "phase_label": labels,
        },
        index=daily_core.index,
    )
    return out_df


__all__ = [
    "compute_core_zone_daily_mean",
    "compute_doy_climatology",
    "compute_standardized_anomalies",
    "detect_spells",
    "check_july_august_statistics",
    "compare_with_published_list",
    "generate_phase_labels",
]
