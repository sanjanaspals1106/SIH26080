"""Stage 3 driver: Golden Dataset -> features, district forecasts, district history (PRD 9.4, 9.6, 14). Owner: M1.

One start month at a time (bounded memory). The static geography, the climatology and the district weights are
**cached** and passed in, never recomputed per month or per run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from data_pipeline.districts.products import (
    FORECAST_SCHEMA,
    HISTORY_SCHEMA,
    NULLABLE_INTS,
    district_forecasts,
    district_history,
)
from data_pipeline.districts.products import KEY as DISTRICT_KEY
from data_pipeline.districts.weights import get_district_weights
from data_pipeline.features.climatology import assert_no_holdout, get_climatology
from data_pipeline.features.io import read_month_tables, write_month_tables
from data_pipeline.features.library import FEATURE_SCHEMA, KEY_COLUMNS, build_features
from data_pipeline.features.static import get_static_geography
from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import MissingInputError

log = logging.getLogger(__name__)

TABLES = {"features": "cell_features", "forecasts": "district_forecasts", "history": "district_history"}


def _run_month(run_ids: pd.Series) -> pd.Series:
    """'YYYYMM' of the forecast start, from `tigge_ecmwf_cf_YYYYMMDDHH`."""
    return run_ids.str[-10:-4]


def golden_month_files(config: IngestionConfig, year: int) -> list[Path]:
    files = sorted((config.alignment.golden_dir / f"season_{year}").glob("golden_*.parquet"))
    if not files:
        raise MissingInputError(
            f"No Golden Dataset for season {year} under {config.alignment.golden_dir}. "
            f"Run `python scripts/build_golden.py season {year}` first."
        )
    return files


def build_stage3_season(
    year: int,
    climatology_seasons: Iterable[int],
    config: IngestionConfig | None = None,
    holdout_seasons: Iterable[int] = (),
    static: xr.Dataset | None = None,
    districts: gpd.GeoDataFrame | None = None,
    grid: pd.DataFrame | None = None,
) -> dict[str, list[Path]]:
    """Features, district forecasts and district history for every Golden month of one season.

    `climatology_seasons`: the **training** seasons (required, no default). `holdout_seasons`: if given, the
    climatology is checked against them (PRD L4). `districts` fails clearly (`MissingInputError`) if no district
    file is configured and none is passed.
    """
    config = config or load_config()
    files = golden_month_files(config, year)  # fail early, before any expensive step
    weights, summary = get_district_weights(
        districts, grid, config
    )  # fails clearly if no district file is configured
    static_geo = get_static_geography(config, static=static)
    clim = get_climatology(climatology_seasons, config)
    assert_no_holdout(clim, holdout_seasons)

    written: dict[str, list[Path]] = {k: [] for k in TABLES}
    for file in files:
        golden = pq.read_table(file).to_pandas(date_as_object=False)
        feats = build_features(golden, static_geo, clim, config)
        written["features"] += write_month_tables(
            feats, TABLES["features"], FEATURE_SCHEMA, KEY_COLUMNS, _run_month(feats["run_id"]), config
        )
        fc = district_forecasts(golden, weights, summary, config)
        month = _run_month(fc["run_id"])
        written["forecasts"] += write_month_tables(
            fc, TABLES["forecasts"], FORECAST_SCHEMA, DISTRICT_KEY, month, config
        )
        hist = district_history(fc)
        written["history"] += write_month_tables(
            hist, TABLES["history"], HISTORY_SCHEMA, DISTRICT_KEY, _run_month(hist["run_id"]), config
        )
    return written


def read_features(
    config: IngestionConfig | None = None, seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    return read_month_tables(TABLES["features"], FEATURE_SCHEMA, config, seasons)


def read_district_forecasts(
    config: IngestionConfig | None = None, seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    df = read_month_tables(TABLES["forecasts"], FORECAST_SCHEMA, config, seasons)
    return df.astype(dict.fromkeys(NULLABLE_INTS, "Int32"))  # cell ids: integers with NULL as pd.NA


def read_district_history(
    config: IngestionConfig | None = None, seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    return read_month_tables(TABLES["history"], HISTORY_SCHEMA, config, seasons)
