"""Small synthetic inputs in the real file formats. For tests and `scripts/ingest_example.py` only.

Nothing here is used by the pipeline itself. It lets Stage 1 run end to end without credentials,
without network access and without the full historical data.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from data_pipeline.ingestion.config import IngestionConfig, TiggeConfig


def write_synthetic_imd_year(
    config: IngestionConfig,
    year: int,
    missing_fraction: float = 0.3,
    seed: int = 0,
    missing_cells_fraction: float = 0.0,
) -> Path:
    """Write a yearwise IMD `.grd` file (float32, days x 129 x 135) with -999 in some cells.

    `missing_fraction`: random share of values that are -999. `missing_cells_fraction`: share of whole
    cells (like ocean) that are -999 on every day.
    """
    cfg = config.imd
    rng = np.random.default_rng(seed)
    days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    data = rng.gamma(0.5, 8.0, size=(days, cfg.n_lat, cfg.n_lon)).astype("float32")
    data[rng.random(data.shape) < missing_fraction] = cfg.missing_value
    if missing_cells_fraction:
        data[:, rng.random((cfg.n_lat, cfg.n_lon)) < missing_cells_fraction] = cfg.missing_value
    path = cfg.year_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.astype(cfg.dtype).tofile(path)
    return path


def _grib_message(
    cfg: TiggeConfig,
    short_name: str,
    level: int | None,
    day: date,
    step: int,
    accum: bool,
    values: np.ndarray,
    n_lat: int,
    n_lon: int,
    spacing: float,
) -> Any:
    import eccodes as ec

    h = ec.codes_grib_new_from_samples("regular_ll_pl_grib2" if level else "regular_ll_sfc_grib2")
    for key, value in {
        "Ni": n_lon,
        "Nj": n_lat,
        "latitudeOfFirstGridPointInDegrees": cfg.area.north,
        "latitudeOfLastGridPointInDegrees": cfg.area.south,
        "longitudeOfFirstGridPointInDegrees": cfg.area.west,
        "longitudeOfLastGridPointInDegrees": cfg.area.east,
        "iDirectionIncrementInDegrees": spacing,
        "jDirectionIncrementInDegrees": spacing,
        "jScansPositively": 0,
        "dataDate": int(f"{day:%Y%m%d}"),
        "dataTime": cfg.init_hour_utc * 100,
    }.items():
        ec.codes_set(h, key, value)
    if accum:  # statistically processed field (accumulation from step 0), as TIGGE `tp`
        ec.codes_set(h, "productDefinitionTemplateNumber", 8)
        ec.codes_set(h, "typeOfStatisticalProcessing", 1)
    ec.codes_set(h, "shortName", short_name)
    if level:
        ec.codes_set(h, "level", level)
    ec.codes_set(h, "stepUnits", 1)
    ec.codes_set(h, "endStep", step)
    ec.codes_set_values(h, values.ravel())
    return h


def write_synthetic_grib(
    config: IngestionConfig,
    path: Path,
    unit: dict[str, Any],
    days: list[date],
    steps: list[int],
    spacing: float | None = None,
    seed: int = 0,
) -> Path:
    """Write a real GRIB2 file with the variables of one request unit (see tigge._request_units)."""
    import eccodes as ec

    cfg = config.tigge
    spacing = spacing or cfg.grid_spacing_deg
    rng = np.random.default_rng(seed + (unit["level"] or 0))  # differs by level
    n_lat = int(round((cfg.area.north - cfg.area.south) / spacing)) + 1
    n_lon = int(round((cfg.area.east - cfg.area.west) / spacing)) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for day in days:
            for param in unit["params"]:
                base = rng.random((n_lat, n_lon))
                for step in steps:
                    accum = param == "tp"
                    values = base * step if accum else base + step  # tp grows with the step
                    h = _grib_message(
                        cfg, param, unit["level"], day, step, accum, values, n_lat, n_lon, spacing
                    )
                    ec.codes_write(h, f)
                    ec.codes_release(h)
    return path


class SyntheticClient:
    """Stands in for `cdsapi.Client`: `retrieve(dataset, request, target)` writes synthetic GRIB.

    `max_dates` keeps files small by writing only the first dates of a monthly request.
    `calls` records every request so tests can check what would have been sent.
    """

    def __init__(self, config: IngestionConfig, max_dates: int = 2, spacing: float | None = None):
        self.config = config
        self.max_dates = max_dates
        self.spacing = spacing
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def retrieve(self, dataset: str, request: dict[str, Any], target: str) -> str:
        self.calls.append((dataset, request))
        k = self.config.tigge.request_keys
        levels = request.get(k["level"])
        unit = {
            "levtype": request[k["levtype"]],
            "level": levels[0] if levels else None,
            "params": request[k["variable"]],
        }
        days = [date.fromisoformat(d) for d in request[k["date"]]][: self.max_dates]
        steps = [int(s) for s in request[k["step"]]]
        write_synthetic_grib(self.config, Path(target), unit, days, steps, self.spacing)
        return target


def write_synthetic_districts(path: Path, invalid: bool = False, with_crs: bool = True) -> Path:
    """Write 4 square 'districts' over India (columns DISTRICT, ST_NM, censuscode).

    The format follows the suffix (.geojson or .shp). `with_crs=False` is only meaningful for .shp.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, box

    geoms = [box(75, 10, 76, 11), box(76, 10, 77, 11), box(80, 20, 81, 21), box(72, 22, 73, 23)]
    if invalid:  # a self-intersecting "bow tie" that make_valid can repair
        geoms[3] = Polygon([(72, 22), (73, 23), (73, 22), (72, 23)])
    gdf = gpd.GeoDataFrame(
        {
            "DISTRICT": ["Alpha", "Beta", "Gamma", "Delta"],
            "ST_NM": ["State A", "State A", "State B", "State C"],
            "censuscode": [101, 102, 201, 301],
        },
        geometry=geoms,
        crs="EPSG:4326" if with_crs else None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path)
    return path
