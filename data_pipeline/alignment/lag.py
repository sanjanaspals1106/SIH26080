"""T4 lag test: decides `imd_stamp` (PRD 8.2). Owner: M1.

For each shift s in {-1, 0, +1} days, pair the **lead-1** window rain of start date I (already on the IMD
grid) with the IMD field dated I + 1 + s. Correlate the two fields cell by cell (over the valid cells),
then average over all start dates. The decision rule is exactly the PRD table:

| result                                        | action                                   |
|-----------------------------------------------|------------------------------------------|
| best average and shift 0 differ by < margin   | keep `end_date`                          |
| shift 0 is highest                            | keep `end_date`                          |
| shift -1 is highest                           | `imd_stamp = start_date`                 |
| shift +1 is highest                           | stop: raise, the team must find the cause|

Run it on the newest chosen season, all valid cells. Write the three averages in `docs/data-status.md`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.alignment.cells import load_valid_cells
from data_pipeline.alignment.golden import read_golden
from data_pipeline.ingestion.config import AlignmentConfig, IngestionConfig, load_config
from data_pipeline.ingestion.errors import ValidationError
from data_pipeline.ingestion.imd import read_imd

log = logging.getLogger(__name__)

MIN_CELLS = 10  # a correlation needs at least this many valid pairs


@dataclass(frozen=True)
class LagTestResult:
    averages: dict[int, float]  # shift (days) -> average correlation
    best_shift: int
    imd_stamp: str  # "end_date" or "start_date"
    n_dates: dict[int, int]  # shift -> number of start dates that had a usable correlation

    def summary(self) -> str:
        avg = ", ".join(f"shift {s:+d}: {v:.4f}" for s, v in sorted(self.averages.items()))
        return f"T4 lag test: {avg}; best shift {self.best_shift:+d} -> imd_stamp = {self.imd_stamp}"


def decide_imd_stamp(averages: dict[int, float], keep_margin: float) -> tuple[int, str]:
    """PRD 8.2 decision from the three average correlations. Returns (best shift, imd_stamp)."""
    if not {-1, 0, 1} <= set(averages):
        raise ValidationError(f"The lag test needs shifts -1, 0 and +1; got {sorted(averages)}.")
    best = max(averages, key=lambda s: (averages[s], -abs(s)))  # ties go to the smaller shift
    if best == 0 or averages[best] - averages[0] < keep_margin:
        return best, "end_date"
    if best == -1:
        return best, "start_date"
    raise ValidationError(
        f"T4 lag test: shift +1 has the highest correlation ({averages[1]:.4f} vs {averages[0]:.4f} at shift 0). "
        "Something else is wrong with the time alignment. Stop and find the cause before going on (PRD 8.2)."
    )


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_CELLS:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def lag_test(
    rain_lead1: xr.DataArray, imd_rain: xr.DataArray, valid: xr.DataArray | np.ndarray, cfg: AlignmentConfig
) -> LagTestResult:
    """Run the T4 lag test.

    `rain_lead1`: lead-1 window rain on the IMD grid, dims `(init_time, lat, lon)`.
    `imd_rain`: IMD rain `(time, lat, lon)` on the same grid. `valid`: boolean `(lat, lon)` mask.
    """
    mask = np.asarray(valid, dtype=bool)
    fcst = rain_lead1.transpose("init_time", "lat", "lon")
    obs_times = pd.DatetimeIndex(imd_rain["time"].values)
    averages: dict[int, float] = {}
    counts: dict[int, int] = {}
    for shift in cfg.lag_shifts_days:
        corrs = []
        for init in pd.DatetimeIndex(fcst["init_time"].values):
            pos = obs_times.get_indexer([init.normalize() + pd.Timedelta(days=1 + shift)])[0]
            if pos < 0:
                continue
            corrs.append(
                _correlation(fcst.sel(init_time=init).values[mask], imd_rain.isel(time=pos).values[mask])
            )
        good = [c for c in corrs if np.isfinite(c)]
        if not good:
            raise ValidationError(
                f"T4 lag test: no usable start date for shift {shift:+d} (missing IMD dates or constant fields)."
            )
        averages[shift], counts[shift] = float(np.mean(good)), len(good)
    best, stamp = decide_imd_stamp(averages, cfg.lag_keep_margin)
    result = LagTestResult(averages, best, stamp, counts)
    log.info(result.summary())
    return result


def lag_test_season(
    year: int, config: IngestionConfig | None = None, imd: xr.Dataset | None = None
) -> LagTestResult:
    """T4 on one season: lead-1 rain from the Golden Dataset against the raw IMD fields (all valid cells)."""
    config = config or load_config()
    grid = load_valid_cells(config)
    rows = read_golden(config, seasons=[year])
    rows = rows[rows["lead_day"] == 1]
    inits = pd.DatetimeIndex(sorted(rows["initialization_time"].dt.tz_localize(None).unique()))
    n_lat, n_lon = config.imd.n_lat, config.imd.n_lon
    cube = np.full((len(inits), n_lat * n_lon), np.nan, dtype="float32")
    cube[inits.get_indexer(rows["initialization_time"].dt.tz_localize(None)), rows["cell_id"].to_numpy()] = (
        rows["rain_mm"].to_numpy()
    )
    lat, lon = grid["latitude"].to_numpy()[::n_lon][:n_lat], grid["longitude"].to_numpy()[:n_lon]
    rain = xr.DataArray(
        cube.reshape(len(inits), n_lat, n_lon),
        dims=("init_time", "lat", "lon"),
        coords={"init_time": inits, "lat": lat, "lon": lon},
    )
    imd = imd if imd is not None else read_imd([year], config)
    valid = grid["is_valid"].to_numpy().reshape(n_lat, n_lon)
    return lag_test(rain, imd["rain"], valid, config.alignment)
