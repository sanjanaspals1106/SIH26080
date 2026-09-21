"""Cell layers. Cell-level data stays in Parquet (PRD D16), so the grid endpoint reads the Golden Dataset file of
the run's start month with `pyarrow`, selecting only the run, the lead and the needed columns. Nothing is computed:
no regridding, no features, no models."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from backend.app.errors import ApiError

# PRD 18.4 variables. `raw` and `observed` are M1 products (Golden Dataset); the rest need later pipeline stages.
AVAILABLE = {"raw": "rain_mm", "observed": "obs_mm"}
VARIABLES = [
    "raw", "corrected", "difference", "observed", "improvement", "q10", "q50", "q90",
    "p_ge_15_6", "p_ge_64_5", "p_ge_115_6", "lps_influence", "orographic_influence", "coastal_influence",
]  # fmt: skip


def golden_file(golden_dir: Path, run_id: str) -> Path:
    month = run_id[-10:-4]  # tigge_ecmwf_cf_YYYYMMDDHH -> YYYYMM
    return golden_dir / f"season_{month[:4]}" / f"golden_{month}.parquet"


def read_layer(golden_dir: Path, run_id: str, lead_day: int, variable: str, limit: int, offset: int) -> dict:
    if variable not in AVAILABLE:
        raise ApiError(
            503,
            "VARIABLE_NOT_AVAILABLE",
            f"The '{variable}' layer is not available yet: it needs a later pipeline stage (corrected forecast, probabilities or regime).",
        )
    path = golden_file(golden_dir, run_id)
    if not path.is_file():
        raise ApiError(503, "GRID_DATA_UNAVAILABLE", "Cell-level data for this run is not available.")
    column = AVAILABLE[variable]
    table = pq.read_table(
        path,
        columns=["cell_id", "latitude", "longitude", "imd_date", column],
        filters=[("run_id", "=", run_id), ("lead_day", "=", lead_day)],
    )
    if table.num_rows == 0:
        raise ApiError(404, "FORECAST_NOT_FOUND", "No cell data was found for this run and lead day.")
    page = table.sort_by("cell_id").slice(offset, limit).to_pydict()
    cells = [
        {
            "cell_id": c,
            "latitude": la,
            "longitude": lo,
            "value": None if v is None or v != v else round(float(v), 3),
        }
        for c, la, lo, v in zip(
            page["cell_id"], page["latitude"], page["longitude"], page[column], strict=True
        )
    ]
    return {"total": table.num_rows, "imd_date": table["imd_date"][0].as_py(), "cells": cells}
