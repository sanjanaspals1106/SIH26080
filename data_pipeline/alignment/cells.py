"""The IMD project grid, its fixed `cell_id`, and the valid-cell mask (PRD D5, 8.4, 8.5). Owner: M1.

* `cell_id = i_lat * n_lon + i_lon`, counting from the south-west corner (lat 6.5, lon 66.5) with lat and
  lon ascending. It is defined on the **full** 129 x 135 grid, so it never changes when the mask changes.
* A cell is **valid** if IMD has a non-missing value on at least 95% of JJAS days of the base period
  (1981-2010). The mask is computed once from the IMD files (`compute_valid_cells`), saved as
  `<DATA_DIR>/golden/grid_cells.parquet` and reloaded afterwards (`get_valid_cells`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import MissingInputError, ValidationError
from data_pipeline.ingestion.imd import imd_axes, read_imd_year

log = logging.getLogger(__name__)

GRID_FILE = "grid_cells.parquet"
GRID_COLUMNS = ["cell_id", "latitude", "longitude", "is_valid", "valid_fraction"]


def imd_grid(config: IngestionConfig) -> pd.DataFrame:
    """All 129 x 135 cells: `cell_id`, `latitude`, `longitude` (float32; exact for 0.25 degree steps)."""
    lat, lon = imd_axes(config.imd)
    ilat, ilon = np.meshgrid(np.arange(lat.size), np.arange(lon.size), indexing="ij")
    return pd.DataFrame(
        {
            "cell_id": (ilat * lon.size + ilon).ravel().astype("int32"),
            "latitude": lat[ilat.ravel()].astype("float32"),
            "longitude": lon[ilon.ravel()].astype("float32"),
        }
    )


def valid_fraction_from_counts(
    n_ok: np.ndarray, n_days: int, min_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """(fraction non-missing, is_valid) per cell. `is_valid` means fraction >= min_fraction (95% counts)."""
    if n_days <= 0:
        raise ValidationError("No JJAS days found in the base period; cannot compute valid cells.")
    return n_ok / n_days, n_ok >= min_fraction * n_days - 1e-9


def compute_valid_cells(
    config: IngestionConfig | None = None, years: Iterable[int] | None = None
) -> pd.DataFrame:
    """Valid-cell table from IMD files, one year at a time (never all years in memory).

    `years` defaults to the PRD base period 1981-2010. Raises `MissingInputError` if a year is not on
    disk. Nothing is invented: with the data missing there is no mask.
    """
    config = config or load_config()
    cfg = config.alignment
    year_list = (
        sorted(years) if years is not None else list(range(cfg.base_start_year, cfg.base_end_year + 1))
    )
    grid = imd_grid(config)
    n_lat, n_lon = config.imd.n_lat, config.imd.n_lon
    n_ok = np.zeros((n_lat, n_lon), dtype="int64")
    n_days = 0
    for year in year_list:
        rain = read_imd_year(config.imd.year_path(year), year, config)["rain"]
        jjas = rain.sel(time=rain["time"].dt.month.isin(cfg.season_months))
        n_ok += jjas.notnull().sum("time").values
        n_days += jjas.sizes["time"]
        log.info("valid cells: year %d done (%d JJAS days)", year, jjas.sizes["time"])
    frac, ok = valid_fraction_from_counts(n_ok.ravel(), n_days, cfg.min_non_missing_fraction)
    grid["valid_fraction"] = frac.astype("float32")
    grid["is_valid"] = ok
    grid.attrs.update(base_years=[year_list[0], year_list[-1]], n_years=len(year_list), jjas_days=n_days)
    log.info("%d of %d cells are valid (%d JJAS days per cell)", int(ok.sum()), ok.size, n_days)
    return grid


def save_valid_cells(grid: pd.DataFrame, config: IngestionConfig | None = None) -> Path:
    """Write the grid/mask table. The base period and day count go into the file's metadata."""
    config = config or load_config()
    path = config.alignment.golden_dir / GRID_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(grid[GRID_COLUMNS], preserve_index=False)
    meta = {k.encode(): str(v).encode() for k, v in grid.attrs.items()}
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def load_valid_cells(config: IngestionConfig | None = None) -> pd.DataFrame:
    """Read the saved grid/mask table. Raises with instructions if it has not been computed."""
    config = config or load_config()
    path = config.alignment.golden_dir / GRID_FILE
    if not path.is_file():
        raise MissingInputError(
            f"Valid-cell mask not found: {path}. Compute it once from the IMD base period "
            f"({config.alignment.base_start_year}-{config.alignment.base_end_year}) with "
            "`python scripts/build_golden.py mask`, after downloading those IMD years."
        )
    table = pq.read_table(path)
    grid = table.to_pandas()
    grid.attrs.update(
        {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items() if k != b"pandas"}
    )
    return grid


def get_valid_cells(
    config: IngestionConfig | None = None, years: Iterable[int] | None = None, recompute: bool = False
) -> pd.DataFrame:
    """Load the saved mask; compute and save it first if it does not exist yet (or `recompute`)."""
    config = config or load_config()
    if not recompute and (config.alignment.golden_dir / GRID_FILE).is_file():
        return load_valid_cells(config)
    grid = compute_valid_cells(config, years)
    save_valid_cells(grid, config)
    return grid
