"""The M1 Golden Dataset (PRD 9.3, 9.7). Owner: M1. Schema: `docs/golden-dataset.md`.

One row per `(run_id, lead_day, cell_id)`, valid IMD cells only. Rain windows (C1) and atmosphere are
aligned to the IMD grid; observed IMD rain and the two static fields sit beside them. No features, no
regime labels, no model output.

Files: `<DATA_DIR>/golden/season_<YYYY>/golden_<YYYYMM>.parquet`, one per (season, start month), written
with a fixed schema and in row order `(run_id, lead_day, cell_id)`. Writing a month again replaces the
month's file, so re-running never creates duplicates.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import xarray as xr

from data_pipeline.alignment.cells import load_valid_cells
from data_pipeline.alignment.spatial import (
    build_grid_map,
    regrid_bilinear,
    regrid_dataset_bilinear,
    regrid_rain,
)
from data_pipeline.alignment.temporal import (
    c1_atmosphere_windows,
    c1_rain_windows,
    imd_date,
    window_bounds_utc,
)
from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import (
    IngestionError,
    MissingInputError,
    ValidationError,
)
from data_pipeline.ingestion.imd import imd_axes, read_imd
from data_pipeline.ingestion.tigge import read_month, read_tigge, run_id, tigge_raw_paths

log = logging.getLogger(__name__)

SOURCE = "ECMWF-TIGGE-control"
GRID_NAME = "IMD_0.25"
ATMOSPHERE_VARS = ["msl", "u850", "v850", "q850", "u200", "v200"]
STATIC_VARS = ["orog", "lsm"]
_TS = pa.timestamp("us", tz="UTC")

GOLDEN_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("source", pa.string()),
        ("initialization_time", _TS),
        ("season", pa.int16()),
        ("lead_day", pa.int8()),
        ("imd_date", pa.date32()),
        ("window_start_utc", _TS),
        ("window_end_utc", _TS),
        ("alignment_method", pa.string()),
        ("alignment_offset_hours", pa.int8()),
        ("imd_stamp", pa.string()),
        ("unit", pa.string()),
        ("grid", pa.string()),
        ("cell_id", pa.int32()),
        ("latitude", pa.float32()),
        ("longitude", pa.float32()),
        ("rain_mm", pa.float32()),
        ("obs_mm", pa.float32()),
        *[(v, pa.float32()) for v in ATMOSPHERE_VARS],
        *[(v, pa.float32()) for v in STATIC_VARS],
        ("rain_clipped", pa.bool_()),
        ("obs_missing", pa.bool_()),
        ("rain_regrid", pa.string()),
    ]
)
GOLDEN_COLUMNS = GOLDEN_SCHEMA.names
KEY = ["run_id", "lead_day", "cell_id"]


# ---- static fields ---------------------------------------------------------------------------


def align_static(static: xr.Dataset, config: IngestionConfig) -> xr.Dataset:
    """`orog` and `lsm` on the IMD grid (bilinear). Do this once per run and reuse the result."""
    missing = [v for v in STATIC_VARS if v not in static.data_vars]
    if missing:
        raise ValidationError(
            f"Static dataset is missing {missing}; read it with read_tigge(paths, 'static')."
        )
    lat, lon = imd_axes(config.imd)
    gm = build_grid_map(static["lat"].values, static["lon"].values, lat, lon)
    return regrid_dataset_bilinear(static[STATIC_VARS], gm)


# ---- full-grid atmosphere (regime engine input) ------------------------------------------------


def atmosphere_fullgrid_windows(atm: xr.Dataset, config: IngestionConfig | None = None) -> xr.Dataset:
    """C1 atmosphere windows on the whole IMD grid, dims `(init_time, lead_day, lat, lon)`, float32.

    The same time (`c1_atmosphere_windows`) and space (bilinear) steps as `golden_rows`, but nothing is cut to
    the valid cells: the regime engine needs the ocean too (phase boxes reach 5N and 60E, the low-pressure
    detector covers 5-30N, 60-100E).
    """
    config = config or load_config()
    atm_w = c1_atmosphere_windows(atm[ATMOSPHERE_VARS], config.alignment)
    lat, lon = imd_axes(config.imd)
    gm = build_grid_map(atm["lat"].values, atm["lon"].values, lat, lon)
    return xr.Dataset({v: regrid_bilinear(atm_w[v], gm) for v in ATMOSPHERE_VARS})


# ---- rows ------------------------------------------------------------------------------------


def golden_rows(
    tp: xr.DataArray,
    atm: xr.Dataset,
    static_grid: xr.Dataset,
    imd_rain: xr.DataArray,
    grid: pd.DataFrame,
    config: IngestionConfig | None = None,
    imd_stamp: str | None = None,
) -> pd.DataFrame:
    """Golden rows for every start time present in both `tp` and `atm` (Stage 1 normalised objects).

    `static_grid`: output of `align_static`. `imd_rain`: IMD `rain` (time, lat, lon). `grid`: the table of
    `load_valid_cells` (its `is_valid` column selects the cells).
    """
    config = config or load_config()
    cfg = config.alignment
    stamp = imd_stamp or cfg.imd_stamp
    n_lat, n_lon = config.imd.n_lat, config.imd.n_lon

    inits = tp["init_time"].to_index().intersection(atm["init_time"].to_index()).sort_values()
    if len(inits) == 0:
        raise ValidationError("rain and atmosphere have no start time in common.")
    if len(inits) != tp.sizes["init_time"] or len(inits) != atm.sizes["init_time"]:
        log.warning(
            "rain has %d and atmosphere %d start times; using the %d they share",
            tp.sizes["init_time"],
            atm.sizes["init_time"],
            len(inits),
        )
    tp, atm = tp.sel(init_time=inits), atm.sel(init_time=inits)

    # 1. time (native grid), 2. space (IMD grid)
    rain_w = c1_rain_windows(tp, cfg)
    atm_w = c1_atmosphere_windows(atm[ATMOSPHERE_VARS], cfg)
    lat, lon = imd_axes(config.imd)
    gm_rain = build_grid_map(tp["lat"].values, tp["lon"].values, lat, lon)
    gm_atm = build_grid_map(atm["lat"].values, atm["lon"].values, lat, lon)
    rain = regrid_rain(rain_w["rain_mm"], gm_rain)
    clipped = regrid_rain(rain_w["clipped"].astype("float32"), gm_rain) > 0

    valid = grid["is_valid"].to_numpy()
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValidationError("No valid cells in the grid table; nothing to write.")
    n_init, leads = len(inits), list(cfg.lead_days)
    n_lead, n_valid = len(leads), idx.size

    def pick(da: xr.DataArray) -> np.ndarray:  # (init, lead, lat, lon) -> (init*lead*valid,)
        return (
            da.transpose("init_time", "lead_day", "lat", "lon")
            .values.reshape(n_init * n_lead, n_lat * n_lon)[:, idx]
            .ravel()
        )

    # observed IMD rain for each (init, lead)
    dates = [imd_date(t, k, stamp) for t in inits for k in leads]
    obs_times = pd.DatetimeIndex(imd_rain["time"].values)
    pos = obs_times.get_indexer(dates)
    obs_all = np.full((len(dates), n_lat * n_lon), np.nan, dtype="float32")
    have = pos >= 0
    obs_all[have] = imd_rain.transpose("time", "lat", "lon").values[pos[have]].reshape(int(have.sum()), -1)
    obs = obs_all[:, idx].ravel()

    bounds = [window_bounds_utc(t, k, cfg) for t in inits for k in leads]

    def per_row(values) -> np.ndarray:  # one value per (init, lead) -> one per row
        return np.repeat(np.asarray(values), n_valid)

    cols: dict[str, object] = {
        "run_id": per_row([run_id(t) for t in inits for _ in leads]),
        "source": SOURCE,
        "initialization_time": pd.DatetimeIndex(
            per_row([t.to_datetime64() for t in inits for _ in leads])
        ).tz_localize("UTC"),
        "season": per_row([t.year for t in inits for _ in leads]),
        "lead_day": np.tile(np.repeat(np.asarray(leads, dtype="int8"), n_valid), n_init),
        "imd_date": pd.DatetimeIndex(per_row([d.to_datetime64() for d in dates])),
        "window_start_utc": pd.DatetimeIndex(per_row([b[0].to_datetime64() for b in bounds])).tz_localize(
            "UTC"
        ),
        "window_end_utc": pd.DatetimeIndex(per_row([b[1].to_datetime64() for b in bounds])).tz_localize(
            "UTC"
        ),
        "alignment_method": cfg.method,
        "alignment_offset_hours": np.int8(cfg.alignment_offset_hours[cfg.method]),
        "imd_stamp": stamp,
        "unit": "mm",
        "grid": GRID_NAME,
        "cell_id": np.tile(grid["cell_id"].to_numpy()[idx], n_init * n_lead),
        "latitude": np.tile(grid["latitude"].to_numpy()[idx], n_init * n_lead),
        "longitude": np.tile(grid["longitude"].to_numpy()[idx], n_init * n_lead),
        "rain_mm": pick(rain),
        "obs_mm": obs,
    }
    for v in ATMOSPHERE_VARS:
        cols[v] = pick(regrid_bilinear(atm_w[v], gm_atm))
    for v in STATIC_VARS:
        cols[v] = np.tile(static_grid[v].transpose("lat", "lon").values.ravel()[idx], n_init * n_lead)
    cols["rain_clipped"] = pick(clipped)
    cols["obs_missing"] = np.isnan(obs)
    cols["rain_regrid"] = gm_rain.rain_method

    numeric = {
        f.name: f.type.to_pandas_dtype()
        for f in GOLDEN_SCHEMA
        if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)
    }
    df = pd.DataFrame(cols)[GOLDEN_COLUMNS].astype(numeric)
    validate_golden(df, cfg.max_rain_24h_mm)
    return df


def validate_golden(df: pd.DataFrame, max_rain_mm: float = 1000.0) -> None:
    """Schema, key and value checks (PRD 9.2: no empty forecast fields, 0 <= rain < 1000 mm)."""
    problems: list[str] = []
    if list(df.columns) != GOLDEN_COLUMNS:
        problems.append(
            f"columns differ from the golden schema: missing {sorted(set(GOLDEN_COLUMNS) - set(df.columns))}, unexpected {sorted(set(df.columns) - set(GOLDEN_COLUMNS))}"
        )
    else:
        if df.duplicated(KEY).any():
            problems.append(f"{int(df.duplicated(KEY).sum())} duplicate (run_id, lead_day, cell_id) rows")
        for v in ["rain_mm", *ATMOSPHERE_VARS, *STATIC_VARS, "latitude", "longitude"]:
            n_bad = int((~np.isfinite(df[v].to_numpy())).sum())
            if n_bad:
                problems.append(f"{v} has {n_bad} empty values")
        rain = df["rain_mm"].to_numpy()
        if ((rain < 0) | (rain >= max_rain_mm)).any():
            problems.append(f"rain_mm outside [0, {max_rain_mm:g}) mm")
    if problems:
        raise ValidationError("Golden dataset check failed: " + "; ".join(problems) + ".")


# ---- files -----------------------------------------------------------------------------------


def _month_path(config: IngestionConfig, season: int, yyyymm: str) -> Path:
    return config.alignment.golden_dir / f"season_{season}" / f"golden_{yyyymm}.parquet"


def write_golden(df: pd.DataFrame, config: IngestionConfig | None = None) -> list[Path]:
    """Write `df` as one Parquet file per (season, start month), replacing any file of the same month."""
    config = config or load_config()
    validate_golden(df, config.alignment.max_rain_24h_mm)
    df = df.sort_values(KEY, kind="stable")
    paths = []
    months = df["initialization_time"].dt.strftime("%Y%m")
    for yyyymm, part in df.groupby(months, sort=True):
        path = _month_path(config, int(part["season"].iloc[0]), str(yyyymm))
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(part, schema=GOLDEN_SCHEMA, preserve_index=False)
        tmp = path.with_name(path.name + ".part")
        pq.write_table(table.replace_schema_metadata(None), tmp, compression="zstd")
        tmp.replace(path)
        paths.append(path)
        log.info("wrote %s (%d rows)", path, len(part))
    return paths


def read_golden(config: IngestionConfig | None = None, seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Read the Golden Dataset (all seasons, or the given ones) as a DataFrame in the golden schema."""
    config = config or load_config()
    root = config.alignment.golden_dir
    dirs = [root / f"season_{s}" for s in seasons] if seasons is not None else sorted(root.glob("season_*"))
    files = sorted(f for d in dirs for f in d.glob("golden_*.parquet"))
    if not files:
        raise MissingInputError(
            f"No Golden Dataset files under {root}. Build them with scripts/build_golden.py."
        )
    table = pads.dataset([str(f) for f in files], schema=GOLDEN_SCHEMA, format="parquet").to_table()
    return table.to_pandas(date_as_object=False)


# ---- one season from the raw files -----------------------------------------------------------


def build_golden_season(
    year: int,
    config: IngestionConfig | None = None,
    months: Iterable[int] | None = None,
    grid: pd.DataFrame | None = None,
    static: xr.Dataset | None = None,
    imd: xr.Dataset | None = None,
    tigge_dir: Path | str | None = None,
    require_base_period: bool = False,
) -> list[Path]:
    """Stage 1 raw files of one season -> Golden Dataset files, one month at a time (bounded memory).

    Needs the valid-cell mask (`grid`, default: the saved one), the static fields, the IMD year and the
    TIGGE month files under `<DATA_DIR>/raw/` (or, with `tigge_dir`, in the hand-downloaded layout of
    `read_month`). Anything missing raises with a message saying what to fetch.

    `require_base_period=True` (used by `scripts/build_golden.py season`) refuses a valid-cell mask that was not
    computed from the PRD base period, so that every season is built on the same mask.
    """
    config = config or load_config()
    cfg = config.alignment
    grid = grid if grid is not None else load_valid_cells(config)
    base = grid.attrs.get("base_years")
    if require_base_period and str(base) != str([cfg.base_start_year, cfg.base_end_year]):
        raise ValidationError(
            f"The valid-cell mask was computed from base years {base}, not the PRD base period "
            f"{cfg.base_start_year}-{cfg.base_end_year}. Run `python scripts/build_golden.py mask` first "
            "(needs IMD for those years), or pass --allow-dev-mask for a development build."
        )
    if base is not None and str(base) != str([cfg.base_start_year, cfg.base_end_year]):
        log.warning(
            "valid-cell mask was computed from base years %s, not the PRD base period %d-%d",
            base,
            cfg.base_start_year,
            cfg.base_end_year,
        )

    if static is None:
        if not cfg.static_date:
            raise MissingInputError(
                "Set golden.static_date in config/alignment.yaml (date of the orog/lsm download) or pass `static`."
            )
        d = date.fromisoformat(str(cfg.static_date))
        static = read_tigge(tigge_raw_paths("static", f"{d:%Y%m%d}", config), "static", config)
    static_grid = align_static(static, config)  # once for the whole season
    imd = imd if imd is not None else read_imd([year], config)

    written: list[Path] = []
    for month in months if months is not None else cfg.season_months:
        tag = f"{year}{month:02d}"
        tp, atm = read_month(year, month, config, tigge_dir)
        df = golden_rows(tp, atm, static_grid, imd["rain"], grid, config)
        written += write_golden(df, config)
    if not written:
        raise IngestionError(f"No Golden Dataset files were written for {year}.")
    return written


# ---- one mask for every season -----------------------------------------------------------------


def check_mask_consistency(
    config: IngestionConfig | None = None, seasons: Iterable[int] | None = None, strict_base_period: bool = True
) -> dict:
    """Do all built seasons use exactly the cells of the saved valid-cell mask, and is that mask the PRD one?

    Reads only the `cell_id` column of every Golden file (cheap). Raises `ValidationError` listing every problem;
    returns a small report if all is well. `strict_base_period=False` skips the base-period condition (a
    development mask), the cell comparison always runs.
    """
    config = config or load_config()
    cfg = config.alignment
    grid = load_valid_cells(config)
    mask_ids = set(grid.loc[grid["is_valid"], "cell_id"].tolist())
    problems: list[str] = []
    base = grid.attrs.get("base_years")
    want = str([cfg.base_start_year, cfg.base_end_year])
    if strict_base_period and str(base) != want:
        problems.append(f"the mask was computed from base years {base}, not {want}")
    root = cfg.golden_dir
    dirs = [root / f"season_{s}" for s in seasons] if seasons is not None else sorted(root.glob("season_*"))
    per_season: dict[str, int] = {}
    for d in dirs:
        files = sorted(d.glob("golden_*.parquet"))
        if not files:
            problems.append(f"{d.name}: no Golden files")
            continue
        for f in files:
            ids = set(pq.read_table(f, columns=["cell_id"]).column("cell_id").unique().to_pylist())
            if ids != mask_ids:
                problems.append(
                    f"{d.name}/{f.name}: {len(ids)} cells, mask has {len(mask_ids)} "
                    f"({len(ids - mask_ids)} extra, {len(mask_ids - ids)} missing)"
                )
        per_season[d.name] = len(files)
    if problems:
        raise ValidationError("Valid-cell mask is not used consistently: " + "; ".join(problems) + ".")
    return {"n_valid_cells": len(mask_ids), "base_years": base, "seasons": per_season}
