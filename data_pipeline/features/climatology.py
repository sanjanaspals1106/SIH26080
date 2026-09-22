"""Cell climatology features `clim_mean` and `clim_p95` (PRD 9.4, 9.5, 10.4). Owner: M1.

For every IMD cell: the mean and the 95th percentile (linear interpolation, `numpy.percentile`) of **observed
IMD rain on JJAS days of the given seasons**, ignoring missing days. NaN if a cell has no observed day.

**No leakage.** The function only ever reads the seasons it is given, and the caller must name them: there
is no default. Give it the *training* seasons only (PRD 9.5, L4). `assert_no_holdout` checks a climatology
against the holdout seasons before it is used. The seasons used are stored with the table.

Memory: one JJAS year (about 8.5 MB) is read at a time and the percentile is taken in blocks of latitude rows.
Cached per season set at `<DATA_DIR>/features/climatology/climatology_<hash>.parquet`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.alignment.cells import imd_grid
from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import ValidationError
from data_pipeline.ingestion.imd import read_imd_year

log = logging.getLogger(__name__)

CLIM_COLUMNS = ["cell_id", "clim_mean", "clim_p95"]
_BLOCK_ROWS = 16  # latitude rows per percentile block


def compute_climatology(seasons: Iterable[int], config: IngestionConfig | None = None) -> pd.DataFrame:
    """Climatology table (`cell_id`, `clim_mean`, `clim_p95`) from the JJAS IMD rain of `seasons`."""
    config = config or load_config()
    years = sorted(set(int(s) for s in seasons))
    if not years:
        raise ValidationError("Climatology needs at least one (training) season.")
    n_lat, n_lon = config.imd.n_lat, config.imd.n_lon
    jjas = []
    for year in years:  # raises MissingInputError if a year is not on disk: nothing is invented
        rain = read_imd_year(config.imd.year_path(year), year, config)["rain"]
        jjas.append(
            rain.sel(time=rain["time"].dt.month.isin(config.alignment.season_months)).values.astype("float32")
        )
        log.info("climatology: read %d JJAS days of %d", jjas[-1].shape[0], year)
    mean = np.full((n_lat, n_lon), np.nan)
    p95 = np.full((n_lat, n_lon), np.nan)
    for r in range(0, n_lat, _BLOCK_ROWS):
        block = np.concatenate([j[:, r : r + _BLOCK_ROWS, :] for j in jjas], axis=0)
        has_data = np.isfinite(block).any(axis=0)
        with warnings.catch_warnings():  # "All-NaN slice" for cells with no data; handled by has_data
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m, q = np.nanmean(block, axis=0), np.nanpercentile(block, 95, axis=0)
        mean[r : r + _BLOCK_ROWS] = np.where(has_data, m, np.nan)
        p95[r : r + _BLOCK_ROWS] = np.where(has_data, q, np.nan)
    table = pd.DataFrame(
        {
            "cell_id": imd_grid(config)["cell_id"].to_numpy(),
            "clim_mean": mean.ravel().astype("float32"),
            "clim_p95": p95.ravel().astype("float32"),
        }
    )
    table.attrs["seasons"] = years
    return table


def climatology_path(seasons: Iterable[int], config: IngestionConfig):
    years = sorted(set(int(s) for s in seasons))
    key = hashlib.sha256(json.dumps(years).encode()).hexdigest()[:10]
    return config.features.dir / "climatology" / f"climatology_{key}.parquet"


def get_climatology(
    seasons: Iterable[int], config: IngestionConfig | None = None, recompute: bool = False
) -> pd.DataFrame:
    """Climatology for the given seasons: read from the cache, or computed and saved first."""
    config = config or load_config()
    years = sorted(set(int(s) for s in seasons))
    path = climatology_path(years, config)
    if path.is_file() and not recompute:
        table = pq.read_table(path)
        out = table.to_pandas()
        out.attrs["seasons"] = json.loads((table.schema.metadata or {})[b"seasons"])
        return out
    out = compute_climatology(years, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False).replace_schema_metadata(
        {"seasons": json.dumps(years)}
    )
    tmp = path.with_name(path.name + ".part")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return out


def assert_no_holdout(climatology: pd.DataFrame, holdout_seasons: Iterable[int]) -> None:
    """Raise if the climatology was made from any holdout season (PRD L4)."""
    used = set(climatology.attrs.get("seasons", []))
    if not used:
        raise ValidationError(
            "This climatology does not record which seasons it was made from; cannot check L4."
        )
    bad = sorted(used & set(int(s) for s in holdout_seasons))
    if bad:
        raise ValidationError(
            f"Climatology was computed with holdout season(s) {bad}: leakage (PRD L4). Rebuild it from training "
            "seasons only."
        )
