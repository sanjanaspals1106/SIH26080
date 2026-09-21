"""Offline loader: Stage 2/3 Parquet outputs -> the M1 database tables (PRD 18.2, 20.2). Owner: M1.

This is a batch job run from `scripts/load_m1.py`, never from the API. It reads what the pipeline already
produced (it does not recompute weights, features or climatology) and writes with **upsert** semantics, so
loading the same output twice gives the same database and no duplicate rows.

What is written where:

| Table | Source | On reload |
|---|---|---|
| `grid_cells` | valid-cell mask (Stage 2) + cached static geography (Stage 3) | upsert; `region_code` untouched |
| `districts` | Stage 1 district loader (simplified GeoJSON outline) + Stage 3 district summary | upsert |
| `cell_district_weights` | Stage 3 cached weights | replaced as a whole (M1 owns the table) |
| `nwp_runs` | run metadata of the Golden Dataset | upsert; `evaluation_set`/`status` untouched |
| `district_forecasts` | Stage 3 `district_forecasts` | upsert of **M1 columns only** |
| `district_history` | Stage 3 `district_history` | upsert of **M1 columns only** |

Columns owned by later stages (`corrected_mean_mm`, `wettest_cell_*`, the probabilities, `attention_level`,
`priority_rank`, `product_type`, `fallback_*`, `model_version_id`, `phase`, `lps_near`, `prediction_source`) are never
written here: they stay NULL, and a later stage that fills them is not overwritten when M1 output is reloaded.
Stage 3 columns with no place in the PRD tables (`observed_max_cell_mm` of forecasts, `observed_heavy_area_fraction`,
`raw_wettest_cell_*`, `n_main_cells`) stay in Parquet.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import mapping
from sqlalchemy import Engine, delete, inspect
from sqlalchemy.dialects import postgresql, sqlite

from backend.app.db import tables as t
from data_pipeline.alignment.cells import load_valid_cells
from data_pipeline.districts import load_districts
from data_pipeline.features.pipeline import (
    read_district_forecasts,
    read_district_history,
)
from data_pipeline.features.static import static_path
from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import IngestionError, MissingInputError

log = logging.getLogger(__name__)

CHUNK = 5000


class LoaderError(IngestionError):
    """The database is not ready for loading (for example the schema has not been applied)."""


@dataclass
class LoadReport:
    rows: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return ", ".join(f"{k}: {v}" for k, v in self.rows.items())


# ---- helpers ---------------------------------------------------------------------------------------


def ensure_schema(engine: Engine, create_if_sqlite: bool = True) -> None:
    """Check that the M1 tables exist. On PostgreSQL the schema comes from `database/schema.sql` (see
    database/README.md); on SQLite (tests, demos) the tables are created from the table definitions."""
    missing = [tb.name for tb in t.M1_TABLES if not inspect(engine).has_table(tb.name)]
    if not missing:
        return
    if engine.dialect.name == "sqlite" and create_if_sqlite:
        t.metadata.create_all(engine)
        return
    raise LoaderError(
        f"Database tables missing: {missing}. Create the schema first: "
        "`psql -U sih -h localhost -d sih_rain -f database/schema.sql` (see database/README.md)."
    )


def _plain(value: Any) -> Any:
    """Python value for a DB driver: NaN / NaT / pd.NA -> None, numpy and pandas scalars -> Python scalars."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value.item() if hasattr(value, "item") else value


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    cols = df[columns].astype(object)
    return [
        {c: _plain(v) for c, v in zip(columns, row, strict=True)}
        for row in cols.itertuples(index=False, name=None)
    ]


def upsert(engine: Engine, table, rows: list[dict[str, Any]], keys: list[str], update: list[str]) -> int:
    """INSERT ... ON CONFLICT (keys) DO UPDATE SET update-columns. Only `update` columns are changed on a
    conflict, so anything else already stored in the row (values written by a later stage) survives."""
    if not rows:
        return 0
    insert = (postgresql if engine.dialect.name == "postgresql" else sqlite).insert
    stmt = insert(table)
    stmt = (
        stmt.on_conflict_do_update(index_elements=keys, set_={c: stmt.excluded[c] for c in update})
        if update
        else stmt.on_conflict_do_nothing(index_elements=keys)
    )
    with engine.begin() as conn:
        for i in range(0, len(rows), CHUNK):
            conn.execute(stmt, rows[i : i + CHUNK])
    return len(rows)


def _round_coords(node: Any, digits: int = 4) -> Any:
    if isinstance(node, (list, tuple)):
        return [_round_coords(n, digits) for n in node]
    return round(node, digits) if isinstance(node, float) else node


def outline_feature(geometry, source_id: str | None, tolerance_deg: float) -> dict[str, Any]:
    """GeoJSON Feature for the map: geometry simplified (topology kept) and rounded to 4 decimals (~10 m)."""
    geom = geometry.simplify(tolerance_deg, preserve_topology=True) if tolerance_deg > 0 else geometry
    g = mapping(geom)
    props = {"source_id": source_id} if source_id is not None else {}
    return {
        "type": "Feature",
        "geometry": {"type": g["type"], "coordinates": _round_coords(g["coordinates"])},
        "properties": props,
    }


def _seasons_on_disk(config: IngestionConfig, table: str) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in (config.features.dir / table).glob("season_*"))


# ---- tables ----------------------------------------------------------------------------------------


def load_grid_cells(engine: Engine, config: IngestionConfig) -> int:
    grid = load_valid_cells(config)  # cell_id, latitude, longitude, is_valid, valid_fraction
    path = static_path(config)
    if path.is_file():
        grid = grid.merge(pq.read_table(path).to_pandas(), on="cell_id", how="left")
    else:
        log.warning("no cached static geography at %s: elevation, slope and coast distance stay NULL", path)
        for c in ("elevation_m", "slope", "dist_coast_km"):
            grid[c] = None
    cols = ["cell_id", "latitude", "longitude", "is_valid", "elevation_m", "slope", "dist_coast_km"]
    return upsert(engine, t.grid_cells, _records(grid, cols), ["cell_id"], cols[1:])


def load_district_rows(
    engine: Engine,
    config: IngestionConfig,
    districts: gpd.GeoDataFrame | None = None,
    summary: pd.DataFrame | None = None,
) -> int:
    """Districts with their simplified outline. `districts` defaults to the configured district file (Stage 1
    loader, which fails clearly if none is configured). Nothing is substituted."""
    districts = districts if districts is not None else load_districts(config=config)
    if summary is None:
        path = config.features.dir / "districts" / "district_summary.parquet"
        if not path.is_file():
            raise MissingInputError(
                f"No district summary at {path}. Run `python scripts/build_features.py weights` first."
            )
        summary = pq.read_table(path).to_pandas()
    stats = summary.set_index("district_id")
    tol = config.districts.outline_simplify_deg
    rows = []
    for rec in districts.itertuples(index=False):
        s = stats.loc[rec.district_id] if rec.district_id in stats.index else None
        source_id = getattr(rec, "source_id", None)
        rows.append(
            {
                "district_id": rec.district_id,
                "name": rec.name,
                "state": rec.state,
                "geojson": outline_feature(rec.geometry, None if pd.isna(source_id) else str(source_id), tol),
                "centroid_lat": _plain(s["centroid_lat"]) if s is not None else None,
                "centroid_lon": _plain(s["centroid_lon"]) if s is not None else None,
                "n_effective_cells": _plain(s["n_effective_cells"]) if s is not None else None,
                "is_small": _plain(s["is_small"]) if s is not None else None,
                "source_year": config.districts.census_basis,
            }
        )
    return upsert(
        engine, t.districts, rows, ["district_id"], [c for c in rows[0] if c != "district_id"] if rows else []
    )


def load_weights(engine: Engine, config: IngestionConfig) -> int:
    path = config.features.dir / "districts" / "cell_district_weights.parquet"
    if not path.is_file():
        raise MissingInputError(
            f"No district weights at {path}. Run `python scripts/build_features.py weights` first."
        )
    weights = pq.read_table(path).to_pandas()
    rows = _records(weights, ["cell_id", "district_id", "area_weight", "is_main"])
    with engine.begin() as conn:  # M1 owns this table: replace it as a whole, atomically
        conn.execute(delete(t.cell_district_weights))
        for i in range(0, len(rows), CHUNK):
            conn.execute(t.cell_district_weights.insert(), rows[i : i + CHUNK])
    return len(rows)


def load_runs(engine: Engine, config: IngestionConfig, seasons: Iterable[int] | None = None) -> int:
    """NWP run metadata from the Golden Dataset files (columns only, not the rows)."""
    root = config.alignment.golden_dir
    dirs = [root / f"season_{s}" for s in seasons] if seasons is not None else sorted(root.glob("season_*"))
    files = sorted(f for d in dirs for f in d.glob("golden_*.parquet"))
    if not files:
        raise MissingInputError(
            f"No Golden Dataset files under {root}; run `python scripts/build_golden.py season <year>`."
        )
    cols = [
        "run_id",
        "source",
        "initialization_time",
        "season",
        "alignment_method",
        "alignment_offset_hours",
        "imd_stamp",
    ]
    runs = (
        pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files])
        .drop_duplicates("run_id")
        .sort_values("run_id")
    )
    rows = _records(runs, cols)
    now = datetime.now(UTC)
    for r in rows:
        r["mode"], r["created_at"] = "replay", now
    return upsert(engine, t.nwp_runs, rows, ["run_id"], [c for c in cols if c != "run_id"])


def load_district_forecasts(
    engine: Engine, config: IngestionConfig, seasons: Iterable[int] | None = None
) -> int:
    n = 0
    for season in list(seasons) if seasons is not None else _seasons_on_disk(config, "district_forecasts"):
        df = read_district_forecasts(config, seasons=[season])
        cols = [
            "run_id",
            "lead_day",
            "district_id",
            "imd_date",
            "raw_mean_mm",
            "observed_mean_mm",
            "is_small",
        ]
        n += upsert(
            engine, t.district_forecasts, _records(df, cols), ["run_id", "lead_day", "district_id"], cols[3:]
        )
    return n


def load_district_history(
    engine: Engine, config: IngestionConfig, seasons: Iterable[int] | None = None
) -> int:
    n = 0
    for season in list(seasons) if seasons is not None else _seasons_on_disk(config, "district_history"):
        df = read_district_history(config, seasons=[season])
        cols = [
            "run_id",
            "lead_day",
            "district_id",
            "imd_date",
            "season",
            "observed_mean_mm",
            "observed_max_cell_mm",
            "raw_mean_mm",
        ]
        n += upsert(
            engine, t.district_history, _records(df, cols), ["run_id", "lead_day", "district_id"], cols[3:]
        )
    return n


def load_all(
    engine: Engine,
    config: IngestionConfig | None = None,
    seasons: Iterable[int] | None = None,
    districts: gpd.GeoDataFrame | None = None,
) -> LoadReport:
    """Load every M1 product, in foreign-key order. Safe to run again."""
    config = config or load_config()
    seasons = list(seasons) if seasons is not None else None
    ensure_schema(engine)
    report = LoadReport()
    report.rows["grid_cells"] = load_grid_cells(engine, config)
    report.rows["districts"] = load_district_rows(engine, config, districts)
    report.rows["cell_district_weights"] = load_weights(engine, config)
    report.rows["nwp_runs"] = load_runs(engine, config, seasons)
    report.rows["district_forecasts"] = load_district_forecasts(engine, config, seasons)
    report.rows["district_history"] = load_district_history(engine, config, seasons)
    log.info("loaded M1 products: %s", report)
    return report
