"""Temporal alignment, method C1 (PRD 8.1 to 8.3). Owner: M1.

Input: the normalised TIGGE Datasets of Stage 1 (`read_tigge`), dims `(init_time, lead_hours, lat, lon)`.
Output: fields per **lead day** with dims `(init_time, lead_day, lat, lon)`, still on the TIGGE grid.

* IMD's rain day ends at 08:30 IST = 03:00 UTC (T6, derived in code below).
* Lead day k covers hours `24(k-1)+3` to `24k+3` after the start: 3->27, 27->51, 51->75.
* `tp` is cumulative from step 0. Cumulative rain at any hour is a straight-line interpolation
  between the two nearest 6-hourly steps (cumulative at hour 0 is 0). Window rain = cum(end) - cum(start).
  Values below 0 are set to 0 and counted (T2).
* Atmosphere for lead k = mean of the atmosphere steps inside the window (steps 6..24, 30..48, 54..72).
* Units are not changed: `rain_mm` in mm (kg m-2), atmosphere as stored in TIGGE.

Method C0 is not implemented (PRD D6: only when 6-hourly `tp` is unavailable).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.ingestion.config import AlignmentConfig
from data_pipeline.ingestion.errors import IngestionError, ValidationError

log = logging.getLogger(__name__)

IST_OFFSET = timedelta(hours=5, minutes=30)


def imd_day_end_utc_hours(cfg: AlignmentConfig) -> int:
    """Hour (UTC) at which the IMD rain day ends, converted from IST in code (test T6)."""
    ist = datetime.strptime(cfg.imd_day_end_ist, "%H:%M")
    utc = ist - IST_OFFSET
    if utc.minute != 0:
        raise ValidationError(f"IMD day end {cfg.imd_day_end_ist} IST is not on a UTC hour boundary.")
    if utc.strftime("%H:%M") != cfg.imd_day_end_utc:
        raise ValidationError(
            f"config says the IMD day ends at {cfg.imd_day_end_utc} UTC but {cfg.imd_day_end_ist} IST is "
            f"{utc:%H:%M} UTC (PRD 8.1). Fix truth.imd_day_end_* in config/alignment.yaml."
        )
    return utc.hour


def require_c1(cfg: AlignmentConfig) -> None:
    if cfg.method != "C1":
        raise IngestionError(
            f"time_alignment.method is '{cfg.method}' but only C1 is implemented (PRD D6). C0 is for the "
            "case where 6-hourly tp is not available (check CK2)."
        )


def window_hours(k: int, cfg: AlignmentConfig) -> tuple[int, int]:
    """C1 window of lead day k in hours after the start: (24(k-1)+3, 24k+3)."""
    off = imd_day_end_utc_hours(cfg)
    return 24 * (k - 1) + off, 24 * k + off


def atmosphere_steps(k: int, steps: list[int], cfg: AlignmentConfig) -> list[int]:
    """Atmosphere steps that lie inside the window of lead day k (PRD 8.3)."""
    lo, hi = window_hours(k, cfg)
    return [s for s in steps if lo <= s <= hi]


def imd_date(init_time: pd.Timestamp, k: int, imd_stamp: str) -> pd.Timestamp:
    """IMD date for lead k: I + k for `end_date`, I + k - 1 for `start_date` (PRD 8.3)."""
    if imd_stamp not in ("end_date", "start_date"):
        raise IngestionError(f"imd_stamp must be 'end_date' or 'start_date', got '{imd_stamp}'.")
    return pd.Timestamp(init_time).normalize() + pd.Timedelta(days=k if imd_stamp == "end_date" else k - 1)


def window_bounds_utc(
    init_time: pd.Timestamp, k: int, cfg: AlignmentConfig
) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo, hi = window_hours(k, cfg)
    t = pd.Timestamp(init_time)
    return t + pd.Timedelta(hours=lo), t + pd.Timedelta(hours=hi)


# ---- rain ------------------------------------------------------------------------------------


def check_tp_cumulative(tp: xr.DataArray, cfg: AlignmentConfig) -> float:
    """T1 (CK4): share of series (start x cell) that never decrease along the steps. Raises if too low."""
    diffs = tp.transpose("init_time", "lat", "lon", "lead_hours").diff("lead_hours")
    ok = (diffs >= -cfg.tp_decrease_tolerance_mm).all("lead_hours")
    share = float(ok.mean())
    if share < cfg.tp_non_decreasing_min_share:
        raise ValidationError(
            f"T1 failed: tp is non-decreasing in only {share:.4%} of cells (need "
            f"{cfg.tp_non_decreasing_min_share:.1%}). tp may not be cumulative or not in mm (CK4). Stop and "
            "find the reason before going on."
        )
    return share


def cumulative_at(tp: xr.DataArray, hour: float) -> xr.DataArray:
    """Cumulative rain `hour` hours after the start: straight-line interpolation between 6-hourly steps.

    Cumulative at step 0 is 0 (PRD 8.3), whatever the file holds there.
    """
    steps = np.asarray(tp["lead_hours"].values, dtype=float)
    if hour <= 0:
        return xr.zeros_like(tp.isel(lead_hours=0, drop=True))
    lo = steps[steps <= hour]
    hi = steps[steps >= hour]
    if lo.size == 0 or hi.size == 0:
        raise ValidationError(f"tp steps {steps.tolist()} do not cover hour {hour:g}.")
    s0, s1 = lo.max(), hi.min()

    def at(s: float) -> xr.DataArray:
        return (
            xr.zeros_like(tp.isel(lead_hours=0, drop=True))
            if s == 0
            else tp.sel(lead_hours=int(s), drop=True)
        )

    if s0 == s1:
        return at(s0)
    frac = (hour - s0) / (s1 - s0)
    return at(s0) * (1 - frac) + at(s1) * frac


def c1_rain_windows(tp: xr.DataArray, cfg: AlignmentConfig, check: bool = True) -> xr.Dataset:
    """Window rain for every start time and lead day, on the TIGGE grid.

    Returns a Dataset with `rain_mm` and `clipped` (bool: the raw window was below 0 and was set to 0),
    dims `(init_time, lead_day, lat, lon)`. `attrs` hold the T2 counts. `check=True` raises if the clipped
    share is above `clip_max_share` (T2).
    """
    require_c1(cfg)
    tp = tp.transpose("init_time", "lead_hours", "lat", "lon")
    if check:
        check_tp_cumulative(tp, cfg)
    rain, clipped = [], []
    for k in cfg.lead_days:
        lo, hi = window_hours(k, cfg)
        raw = cumulative_at(tp, hi) - cumulative_at(tp, lo)
        clipped.append(raw < 0)
        rain.append(raw.clip(min=0))
    lead = pd.Index(cfg.lead_days, name="lead_day")
    out = xr.Dataset(
        {
            "rain_mm": xr.concat(rain, dim=lead)
            .transpose("init_time", "lead_day", "lat", "lon")
            .astype("float32"),
            "clipped": xr.concat(clipped, dim=lead).transpose("init_time", "lead_day", "lat", "lon"),
        }
    )
    n_clipped, n_total = int(out["clipped"].sum()), int(out["clipped"].size)
    out.attrs.update(n_clipped=n_clipped, n_values=n_total, method="C1")
    if n_clipped:
        log.warning(
            "T2: %d of %d window values were below 0 and set to 0 (%.4f%%)",
            n_clipped,
            n_total,
            100 * n_clipped / n_total,
        )
    if check and n_total and n_clipped / n_total > cfg.clip_max_share:
        raise ValidationError(
            f"T2 failed: {n_clipped} of {n_total} window values ({n_clipped / n_total:.3%}) had to be set from "
            f"below 0 to 0; the limit is {cfg.clip_max_share:.1%}. Check that tp is cumulative (CK4)."
        )
    return out


# ---- atmosphere ------------------------------------------------------------------------------


def c1_atmosphere_windows(atm: xr.Dataset, cfg: AlignmentConfig) -> xr.Dataset:
    """Mean of the atmosphere steps inside each lead-day window, dims `(init_time, lead_day, lat, lon)`."""
    require_c1(cfg)
    steps = sorted(int(s) for s in atm["lead_hours"].values)
    parts = []
    for k in cfg.lead_days:
        wanted = atmosphere_steps(k, steps, cfg)
        if not wanted:
            raise ValidationError(
                f"No atmosphere steps inside the window of lead day {k}; found steps {steps}."
            )
        parts.append(atm.sel(lead_hours=wanted).mean("lead_hours", keep_attrs=True))
    lead = pd.Index(cfg.lead_days, name="lead_day")
    out = xr.concat(parts, dim=lead)
    for name in out.data_vars:
        out[name] = out[name].transpose("init_time", "lead_day", "lat", "lon").astype("float32")
    return out
