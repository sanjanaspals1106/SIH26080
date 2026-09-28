"""Cell layers. Cell-level data stays in Parquet (PRD D16), so the grid endpoint reads the Golden Dataset file of
the run's start month with `pyarrow`, selecting only the run, the lead and the needed columns. Nothing is computed:
no regridding, no features, no models."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from backend.app.errors import ApiError

# PRD 18.4 variables. `raw` and `observed` are M1 products (Golden Dataset). `corrected`, `q10/50/90` and the two
# probability thresholds available at this stage come from the trained M3 models (`data/serving/corrected/`,
# `scripts/build_district_corrected.py`); `difference` is derived from `raw` and `corrected` at read time.
# `p_ge_15_6`, `improvement` and the three regime-influence layers still need a later pipeline stage.
AVAILABLE = {"raw": "rain_mm", "observed": "obs_mm"}
CORRECTED = {"corrected": "corrected_mean_mm", "q10": "q10_mm", "q50": "q50_mm", "q90": "q90_mm",
            "p_ge_64_5": "p_ge_64_5", "p_ge_115_6": "p_ge_115_6"}  # fmt: skip
VARIABLES = [
    "raw", "corrected", "difference", "observed", "improvement", "q10", "q50", "q90",
    "p_ge_15_6", "p_ge_64_5", "p_ge_115_6", "lps_influence", "orographic_influence", "coastal_influence",
]  # fmt: skip


def golden_file(golden_dir: Path, run_id: str) -> Path:
    month = run_id[-10:-4]  # tigge_ecmwf_cf_YYYYMMDDHH -> YYYYMM
    return golden_dir / f"season_{month[:4]}" / f"golden_{month}.parquet"


def corrected_file(corrected_dir: Path, run_id: str) -> Path:
    month = run_id[-10:-4]
    return corrected_dir / f"season_{month[:4]}" / f"corrected_{month}.parquet"


def _read(path: Path, columns: list[str], run_id: str, lead_day: int):
    if not path.is_file():
        return None
    table = pq.read_table(path, columns=columns, filters=[("run_id", "=", run_id), ("lead_day", "=", lead_day)])
    if table.num_rows == 0:
        return None
    if "cell_id" in table.column_names and table.field("cell_id").type != pa.int32():
        table = table.set_column(table.column_names.index("cell_id"), "cell_id", table["cell_id"].cast(pa.int32()))
    return table


def read_layer(
    golden_dir: Path, corrected_dir: Path, run_id: str, lead_day: int, variable: str, limit: int, offset: int
) -> dict:
    geo = _read(golden_file(golden_dir, run_id), ["cell_id", "latitude", "longitude", "imd_date"], run_id, lead_day)
    if geo is None:
        raise ApiError(503, "GRID_DATA_UNAVAILABLE", "Cell-level data for this run is not available.")

    if variable in AVAILABLE:
        values, value_col = golden_file(golden_dir, run_id), AVAILABLE[variable]
        vtab = _read(values, ["cell_id", value_col], run_id, lead_day)
    elif variable in CORRECTED:
        vtab = _read(corrected_file(corrected_dir, run_id), ["cell_id", CORRECTED[variable]], run_id, lead_day)
        value_col = CORRECTED[variable]
    elif variable == "difference":
        raw = _read(golden_file(golden_dir, run_id), ["cell_id", "rain_mm"], run_id, lead_day)
        cor = _read(corrected_file(corrected_dir, run_id), ["cell_id", "corrected_mean_mm"], run_id, lead_day)
        if raw is None or cor is None:
            vtab = None
        else:
            j = raw.rename_columns(["cell_id", "raw_mm"]).join(
                cor.rename_columns(["cell_id", "corrected_mean_mm"]), "cell_id"
            )
            j = j.append_column("difference", pc.subtract(j["corrected_mean_mm"], j["raw_mm"]))
            vtab, value_col = j.select(["cell_id", "difference"]), "difference"
    else:
        raise ApiError(
            503,
            "VARIABLE_NOT_AVAILABLE",
            f"The '{variable}' layer is not available yet: it needs a later pipeline stage (probabilities or regime).",
        )
    if vtab is None:
        raise ApiError(
            503, "VARIABLE_NOT_AVAILABLE", f"The '{variable}' layer has not been generated for this run yet."
        )

    table = geo.join(vtab, "cell_id").sort_by("cell_id")
    if table.num_rows == 0:
        raise ApiError(404, "FORECAST_NOT_FOUND", "No cell data was found for this run and lead day.")
    page = table.slice(offset, limit).to_pydict()
    cells = [
        {
            "cell_id": c,
            "latitude": la,
            "longitude": lo,
            "value": None if v is None or v != v else round(float(v), 3),
        }
        for c, la, lo, v in zip(
            page["cell_id"], page["latitude"], page["longitude"], page[value_col], strict=True
        )
    ]
    return {"total": table.num_rows, "imd_date": table["imd_date"][0].as_py(), "cells": cells}
