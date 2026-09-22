"""Static geography on the IMD grid (PRD 8.6). Owner: M1.

From the TIGGE `orog` and `lsm` fields already regridded to the IMD grid (Stage 2 `align_static`), for **every**
cell of the 129 x 135 grid (sea cells too, they are needed to find the coast):

| column | definition |
|---|---|
| `elevation_m` | `orog`, metres |
| `slope` | size of the height gradient, m per m, central differences on the metric grid (`grid.py`) |
| `aspect_sin`, `aspect_cos` | direction of the **downhill** slope as a bearing clockwise from north: `sin = -dz/dx / slope`, `cos = -dz/dy / slope`. Flat cell (slope 0): both 0 |
| `dist_coast_km` | distance from a land cell (`lsm >= 0.5`) to the nearest sea cell (`lsm < 0.5`), in km. Sea cells: 0. NaN if the grid has no sea cell |

Distance uses the PRD conversion (27.75 km per cell north-south, 27.75 * cos(latitude of the land cell) km per
cell east-west). It is an exact nearest-cell search under that metric (one KD-tree per latitude row) instead of a
distance transform with one fixed east-west size, because the east-west size changes with latitude.

The table is computed once and cached at `<DATA_DIR>/features/static/static_geography.parquet`; the file carries
a hash of the input fields and is recomputed only when they change.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from scipy.spatial import cKDTree

from data_pipeline.alignment.cells import imd_grid
from data_pipeline.alignment.golden import align_static
from data_pipeline.features.grid import cell_size_m, masked_derivative
from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import MissingInputError, ValidationError
from data_pipeline.ingestion.imd import imd_axes

log = logging.getLogger(__name__)

STATIC_COLUMNS = ["cell_id", "elevation_m", "slope", "aspect_sin", "aspect_cos", "dist_coast_km"]
LAND_MIN = 0.5  # lsm >= 0.5 is land, < 0.5 is sea (PRD 8.6)


def distance_to_coast_km(lsm: np.ndarray, lat: np.ndarray, config: IngestionConfig) -> np.ndarray:
    """Distance (km) from each land cell to the nearest sea cell; 0 for sea cells. Shape (n_lat, n_lon)."""
    sea = lsm < LAND_MIN
    out = np.zeros(lsm.shape, dtype="float64")
    if not sea.any():
        out[:] = np.nan
        return out
    dy_m, _ = cell_size_m(lat, config)
    dy_km = dy_m / 1000.0
    sea_i, sea_j = np.nonzero(sea)
    for i in range(lsm.shape[0]):
        land_j = np.flatnonzero(~sea[i])
        if land_j.size == 0:
            continue
        east_km = dy_km * np.cos(np.radians(lat[i]))  # km per cell east-west at this row's latitude
        tree = cKDTree(np.column_stack([sea_i * dy_km, sea_j * east_km]))
        out[i, land_j], _ = tree.query(np.column_stack([np.full(land_j.size, i * dy_km), land_j * east_km]))
    return out


def compute_static_geography(static_grid: xr.Dataset, config: IngestionConfig | None = None) -> pd.DataFrame:
    """Static geography table for all IMD cells from `orog` and `lsm` on the IMD grid (see module docstring)."""
    config = config or load_config()
    lat, lon = imd_axes(config.imd)
    for v in ("orog", "lsm"):
        if v not in static_grid.data_vars:
            raise ValidationError(f"Static grid is missing '{v}'; pass the output of align_static().")
    orog = static_grid["orog"].transpose("lat", "lon").values.astype("float64")
    lsm = static_grid["lsm"].transpose("lat", "lon").values.astype("float64")
    if orog.shape != (lat.size, lon.size) or lsm.shape != orog.shape:
        raise ValidationError(
            f"Static fields are {orog.shape}, expected the IMD grid {(lat.size, lon.size)}."
        )
    if np.isnan(orog).any() or np.isnan(lsm).any():
        raise ValidationError(
            "Static fields have NaN on the IMD grid (is the TIGGE static domain smaller than IMD's?)."
        )

    dy, dx = cell_size_m(lat, config)
    every = np.ones(orog.shape, dtype=bool)
    gy = masked_derivative(orog, every, dy, -2)  # dz/dy, m per m (northward)
    gx = masked_derivative(orog, every, dx, -1)  # dz/dx (eastward)
    slope = np.hypot(gy, gx)
    flat = slope == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        asin, acos = np.where(flat, 0.0, -gx / slope), np.where(flat, 0.0, -gy / slope)

    cells = imd_grid(config)
    table = pd.DataFrame(
        {
            "cell_id": cells["cell_id"].to_numpy(),
            "elevation_m": orog.ravel().astype("float32"),
            "slope": slope.ravel().astype("float32"),
            "aspect_sin": asin.ravel().astype("float32"),
            "aspect_cos": acos.ravel().astype("float32"),
            "dist_coast_km": distance_to_coast_km(lsm, lat, config).ravel().astype("float32"),
        }
    )
    return table[STATIC_COLUMNS]


def _fingerprint(static_grid: xr.Dataset) -> str:
    h = hashlib.sha256()
    for v in ("orog", "lsm"):
        h.update(
            np.ascontiguousarray(static_grid[v].transpose("lat", "lon").values.astype("float32")).tobytes()
        )
    return h.hexdigest()[:16]


def static_path(config: IngestionConfig):
    return config.features.dir / "static" / "static_geography.parquet"


def get_static_geography(
    config: IngestionConfig | None = None, static: xr.Dataset | None = None, recompute: bool = False
) -> pd.DataFrame:
    """Cached static geography. `static` is the Stage 1 static Dataset (`read_tigge(paths, 'static')`).

    If a cached file exists and `static` is None it is used as is; if `static` is given, the cache is reused
    only when it was made from identical fields. Without a cache and without `static` it reads the raw static
    file named by `golden.static_date` in config/alignment.yaml.
    """
    config = config or load_config()
    path = static_path(config)
    grid_static = align_static(static, config) if static is not None else None
    fp = _fingerprint(grid_static) if grid_static is not None else None
    if path.is_file() and not recompute:
        table = pq.read_table(path)
        stored = (table.schema.metadata or {}).get(b"input_hash", b"").decode()
        if fp is None or stored == fp:
            return table.to_pandas()
        log.info("static geography cache is stale (inputs changed); recomputing")
    if grid_static is None:
        from datetime import date

        from data_pipeline.ingestion.tigge import read_tigge, tigge_raw_paths

        if not config.alignment.static_date:
            raise MissingInputError(
                "No static geography yet: set golden.static_date in config/alignment.yaml (date of the orog/lsm "
                "download) or pass `static`."
            )
        d = date.fromisoformat(str(config.alignment.static_date))
        raw = read_tigge(tigge_raw_paths("static", f"{d:%Y%m%d}", config), "static", config)
        grid_static = align_static(raw, config)
        fp = _fingerprint(grid_static)
    result = compute_static_geography(grid_static, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(result, preserve_index=False).replace_schema_metadata({"input_hash": fp})
    tmp = path.with_name(path.name + ".part")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return result
