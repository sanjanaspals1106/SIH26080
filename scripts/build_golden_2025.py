#!/usr/bin/env python
"""Development-only Golden Dataset for 2025 from the files already on disk (no downloads).

Uses the repo's own golden_rows/write_golden. Differences from scripts/build_golden.py (raw layout):
  * TIGGE GRIBs are read from data/tigge/2025/{tp_2025MM,single_MM,pressure_MM}.grib
  * IMD is read from the NetCDF data/imd/RF25_ind2025_rfp25.nc
  * valid-cell mask comes from JJAS 2025 only (the PRD 1981-2010 base period is NOT available) and is
    labelled as such in data/golden/grid_cells.parquet metadata.
  python scripts/build_golden_2025.py
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.alignment.cells import (  # noqa: E402
    imd_grid,
    save_valid_cells,
    valid_fraction_from_counts,
)
from data_pipeline.alignment.golden import align_static, golden_rows, write_golden  # noqa: E402
from data_pipeline.ingestion.config import load_config  # noqa: E402
from data_pipeline.ingestion.imd import imd_axes, validate_imd  # noqa: E402
from data_pipeline.ingestion.tigge import read_tigge, validate_tigge  # noqa: E402

YEAR = 2025
TIGGE = Path("data/tigge") / str(YEAR)
IMD_NC = Path("data/imd/RF25_ind2025_rfp25.nc")


def load_imd(config) -> xr.Dataset:
    raw = xr.open_dataset(IMD_NC)["RAINFALL"].rename(TIME="time", LATITUDE="lat", LONGITUDE="lon")
    raw = raw.where(raw != config.imd.missing_value)  # -999 -> NaN (already NaN in this file)
    lat, lon = imd_axes(config.imd)
    assert np.allclose(raw["lat"].values, lat) and np.allclose(raw["lon"].values, lon)
    ds = xr.Dataset({"rain": raw.astype("float32").load()})
    validate_imd(ds, config)
    return ds


def build_mask(imd: xr.Dataset, config) -> pd.DataFrame:
    cfg = config.alignment
    jjas = imd["rain"].sel(time=imd["time"].dt.month.isin(cfg.season_months))
    n_ok = jjas.notnull().sum("time").values.ravel()
    frac, ok = valid_fraction_from_counts(n_ok, jjas.sizes["time"], cfg.min_non_missing_fraction)
    grid = imd_grid(config)
    grid["valid_fraction"] = frac.astype("float32")
    grid["is_valid"] = ok
    grid.attrs.update(base_years=[YEAR, YEAR], n_years=1, jjas_days=jjas.sizes["time"],
                      note="DEVELOPMENT mask from JJAS 2025 only, not the PRD 1981-2010 base period")
    return grid


def load_static() -> xr.Dataset:
    parts = cfgrib.open_datasets(str(TIGGE / "single_06.grib"), backend_kwargs={"indexpath": ""},
                                 decode_timedelta=True)
    ds = next(d for d in parts if "orog" in d.data_vars)
    out = ds[["orog", "lsm"]].isel(time=0, step=0, drop=True).reset_coords(drop=True)
    return out.rename(latitude="lat", longitude="lon").sortby(["lat", "lon"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    # T1 tolerance: this GRIB's tp has packing noise (max observed decrease 0.015 mm, 0.03% of series
    # above 0.01 mm), so the 0.001 mm default fails on float noise. The same noise makes 1.08% of dry-cell
    # windows slightly negative (<= 0.015 mm), which are clipped to 0 (T2 limit 0.1% -> 2%). Dev build
    # only; config/alignment.yaml is unchanged.
    config = dataclasses.replace(
        config,
        alignment=dataclasses.replace(
            config.alignment, tp_decrease_tolerance_mm=0.02, clip_max_share=0.02
        ),
    )
    imd = load_imd(config)
    grid = build_mask(imd, config)
    print("mask:", save_valid_cells(grid, config), "valid cells:", int(grid["is_valid"].sum()))
    static_grid = align_static(load_static(), config)
    total = 0
    for month in config.alignment.season_months:
        tp = read_tigge([TIGGE / f"tp_{YEAR}{month:02d}.grib"], "rain", config)["tp"]
        # msl has an extra step 0 that the pressure-level file does not; atmosphere steps are 6..72 (PRD 7.2)
        sfc = read_tigge([TIGGE / f"single_{month:02d}.grib"], "atmosphere", config, validate=False)
        sfc = sfc.sel(lead_hours=sfc["lead_hours"] >= 6)
        pl = read_tigge([TIGGE / f"pressure_{month:02d}.grib"], "atmosphere", config, validate=False)
        atm = xr.merge([sfc, pl], join="exact", compat="override")
        validate_tigge(atm, "atmosphere", config)
        df = golden_rows(tp, atm, static_grid, imd["rain"], grid, config)
        write_golden(df, config)
        total += len(df)
        print(f"month {month}: {len(df)} rows, obs_missing={int(df['obs_missing'].sum())}")
    print("TOTAL golden rows:", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
