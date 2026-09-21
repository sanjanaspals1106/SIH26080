"""IMD gridded daily rainfall ingestion adapter (PRD 7.1, 7.3, 8.4, 9.1). Owner: M1.

* **Download** (`download_imd`): one yearwise file per year through `imdlib`, saved as
  `<DATA_DIR>/raw/imd/rain/<year>.grd`. A year that is already on disk is not downloaded again.
* **Read** (`read_imd`): the binary files -> one normalised `xarray.Dataset`.

Normalised representation (same naming as `tigge.py`, so Stage 2 can regrid TIGGE onto it):

* variable `rain` (mm/day, float32), dims `(time, lat, lon)`, 129 x 135, `lat` from 6.5 to 38.5 and
  `lon` from 66.5 to 100.0, both ascending.
* `time` holds the **IMD date label as stored** (midnight, no time zone). What that label means
  (`imd_stamp`, PRD 8.2) is decided in Stage 2.
* the missing value -999 is already NaN.
"""

from __future__ import annotations

import calendar
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.ingestion.config import ImdConfig, IngestionConfig, load_config
from data_pipeline.ingestion.errors import (
    DownloadError,
    IngestionError,
    MissingInputError,
    ValidationError,
)

log = logging.getLogger(__name__)


def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def imd_axes(cfg: ImdConfig) -> tuple[np.ndarray, np.ndarray]:
    """The expected lat and lon axes (129 and 135 points, ascending)."""
    lat = cfg.lat_origin + cfg.grid_spacing_deg * np.arange(cfg.n_lat)
    lon = cfg.lon_origin + cfg.grid_spacing_deg * np.arange(cfg.n_lon)
    return lat, lon


# --------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------


def _imdlib_fetch(year: int, cfg: ImdConfig) -> None:
    try:
        import imdlib
    except ImportError as exc:
        raise IngestionError(
            "The 'imdlib' package is not installed. Run `conda env create -f environment.yml`."
        ) from exc
    # imdlib creates only the last folder level, so make sure the parents exist.
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    # yearwise -> <file_dir>/rain/<year>.grd. imdlib also reads the file back and prints progress.
    imdlib.get_data(cfg.var_type, year, year, fn_format="yearwise", file_dir=str(cfg.raw_dir))


def download_imd(
    years: Iterable[int],
    config: IngestionConfig | None = None,
    fetch: Callable[[int, ImdConfig], None] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download IMD rainfall for the given years, skipping years already on disk."""
    cfg = (config or load_config()).imd
    fetch = fetch or _imdlib_fetch
    paths = []
    for year in sorted(set(years)):
        path = cfg.year_path(year)
        if path.is_file() and path.stat().st_size > 0 and not force:
            log.info("IMD %d already present, not downloading again: %s", year, path)
        else:
            log.info("Downloading IMD rainfall for %d", year)
            try:
                fetch(year, cfg)
            except IngestionError:
                raise
            except Exception as exc:
                raise DownloadError(
                    f"IMD download for {year} failed: {type(exc).__name__}: {exc}. Check the "
                    "network and the IMD page in docs/data-sources.md, or place the file at "
                    f"{path} by hand."
                ) from exc
            if not path.is_file() or path.stat().st_size == 0:
                # imdlib prints HTTP errors instead of raising them.
                raise DownloadError(
                    f"IMD download for {year} produced no file at {path}. imdlib does not raise "
                    "on HTTP errors; look at its printed message above."
                )
        paths.append(path)
    return paths


# --------------------------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------------------------


def read_imd_year(path: Path | str, year: int, config: IngestionConfig | None = None) -> xr.Dataset:
    """Read one yearwise IMD binary file (float32, days x 129 x 135, C order) and return `rain`."""
    cfg = (config or load_config()).imd
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(
            f"IMD file not found: {path}. Download it with download_imd([{year}]) or check DATA_DIR."
        )
    n_days, n_cells = _days_in_year(year), cfg.n_lat * cfg.n_lon
    dtype = np.dtype(cfg.dtype)
    expected_bytes = n_days * n_cells * dtype.itemsize
    actual = path.stat().st_size
    if actual != expected_bytes:
        raise ValidationError(
            f"{path.name}: size is {actual} bytes but {year} should have {n_days} days x "
            f"{cfg.n_lat} x {cfg.n_lon} x {dtype.itemsize} = {expected_bytes} bytes. The file is "
            "truncated, is not the 0.25 degree rainfall product, or is not a year "
            f"file for {year}. Delete it and download again."
        )
    data = np.fromfile(path, dtype=dtype).reshape(n_days, cfg.n_lat, cfg.n_lon).astype("float32")
    data[data == cfg.missing_value] = np.nan  # PRD 8.4: -999 becomes NaN

    lat, lon = imd_axes(cfg)
    time = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    ds = xr.Dataset(
        {"rain": (("time", "lat", "lon"), data, {"units": "mm/day", "long_name": "IMD daily rainfall"})},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.attrs.update(source="IMD gridded daily rainfall", grid="IMD_0.25", files=[path.name])
    return ds


def read_imd(
    years: Iterable[int],
    config: IngestionConfig | None = None,
    validate: bool = True,
) -> xr.Dataset:
    """Read the raw yearwise files of `years` into one normalised Dataset (see module docstring)."""
    config = config or load_config()
    year_list = sorted(set(years))
    if not year_list:
        raise IngestionError("read_imd() needs at least one year.")
    parts = [read_imd_year(config.imd.year_path(y), y, config) for y in year_list]
    ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    ds.attrs["files"] = [f for p in parts for f in p.attrs["files"]]
    if validate:
        validate_imd(ds, config)
    return ds


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def validate_imd(ds: xr.Dataset, config: IngestionConfig | None = None) -> None:
    """Basic checks on a normalised IMD Dataset. Raises one ValidationError listing every problem."""
    cfg = (config or load_config()).imd
    problems: list[str] = []

    if "rain" not in ds.data_vars:
        raise ValidationError("IMD validation failed: variable 'rain' is missing.")
    absent = [c for c in ("time", "lat", "lon") if c not in ds.coords]
    if absent:
        raise ValidationError(f"IMD validation failed: missing coordinates {absent}.")

    rain = ds["rain"]
    if rain.dims != ("time", "lat", "lon"):
        problems.append(f"rain dims are {rain.dims}, expected ('time', 'lat', 'lon')")
    if ds.sizes["lat"] != cfg.n_lat or ds.sizes["lon"] != cfg.n_lon:
        problems.append(
            f"grid is {ds.sizes['lat']} x {ds.sizes['lon']}, expected {cfg.n_lat} x {cfg.n_lon} (PRD 7.1)"
        )
    else:
        exp_lat, exp_lon = imd_axes(cfg)
        if not (np.allclose(ds["lat"].values, exp_lat) and np.allclose(ds["lon"].values, exp_lon)):
            problems.append(
                f"lat/lon axes differ from {cfg.lat_origin:g}-{cfg.lat_end:g}N, "
                f"{cfg.lon_origin:g}-{cfg.lon_end:g}E at {cfg.grid_spacing_deg:g} deg"
            )

    time = pd.DatetimeIndex(ds["time"].values)
    if time.size == 0:
        problems.append("no dates")
    else:
        if time.isna().any():
            problems.append("some dates are missing (NaT)")
        if time.has_duplicates:
            problems.append("duplicate dates")
        if not time.is_monotonic_increasing:
            problems.append("dates are not in increasing order")

    values = rain.values
    if values.size == 0 or not np.isfinite(values).any():
        problems.append("rain is empty (no finite values)")
    else:
        if (values == cfg.missing_value).any():
            problems.append(f"{cfg.missing_value:g} was not converted to NaN")
        if (values < 0).any():
            problems.append("negative rain values remain after -999 conversion (unknown missing-value code?)")

    if problems:
        raise ValidationError("IMD validation failed: " + "; ".join(problems) + ".")
