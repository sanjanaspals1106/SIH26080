"""IMD gridded daily rainfall ingestion adapter (PRD 7.1, 7.3, 8.4, 9.1). Owner: M1.

* **Download** (`download_imd`): one yearwise file per year through `imdlib`, saved as
  `<DATA_DIR>/raw/imd/rain/<year>.grd`. A year that is already on disk is not downloaded again.
* **Read** (`read_imd`): the binary files -> one normalised `xarray.Dataset`. A year that is on disk as an
  IMD NetCDF file instead of a `.grd` file (`resolve_imd_path`) is read too, into the same representation.

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
import re
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
# Locating a year on disk (.grd or NetCDF)
# --------------------------------------------------------------------------------------------

NETCDF_SUFFIXES = {".nc", ".nc4", ".cdf"}
_RAIN_VARIABLES = ("rain", "rainfall", "rf", "precip", "precipitation")
_DIM_ALIASES = {"time": "time", "lat": "lat", "latitude": "lat", "lon": "lon", "longitude": "lon"}


def _imd_search_dirs(cfg: ImdConfig) -> list[Path]:
    dirs = [cfg.raw_dir / "rain", cfg.raw_dir]
    if len(cfg.raw_dir.parents) >= 2:
        dirs.append(cfg.raw_dir.parents[1] / "imd")  # <DATA_DIR>/imd, where hand-placed IMD files often sit
    return dirs


def resolve_imd_path(year: int, cfg: ImdConfig) -> Path | None:
    """The file that holds `year`: the yearwise `.grd` of the config first, otherwise a NetCDF file.

    NetCDF files are looked for in `raw/imd/rain`, `raw/imd` and `<DATA_DIR>/imd`, as `<year>.nc` or any
    `*.nc` whose name contains the year as a whole number (for example `RF25_ind2025_rfp25.nc`). Two
    different NetCDF files for one year are an error (which one is the truth is not for the code to guess).
    """
    grd = cfg.year_path(year)
    if grd.is_file() and grd.stat().st_size > 0:
        return grd
    year_re = re.compile(rf"(?<!\d){year}(?!\d)")
    found: dict[Path, None] = {}
    for d in _imd_search_dirs(cfg):
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in NETCDF_SUFFIXES and f.is_file() and year_re.search(f.stem):
                    found[f.resolve()] = None
    if len(found) > 1:
        raise ValidationError(
            f"More than one IMD NetCDF file matches {year}: {[str(p) for p in found]}. Keep one per year."
        )
    return next(iter(found), None)


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
        existing = resolve_imd_path(year, cfg)
        if existing is not None and not force:
            log.info("IMD %d already present, not downloading again: %s", year, existing)
            path = existing
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


def _read_imd_netcdf(path: Path, year: int, cfg: ImdConfig) -> xr.Dataset:
    """One year of IMD rain from a NetCDF file, in the same normalised form as the `.grd` reader.

    Accepts any capitalisation of `time/lat/lon` (IMD's own files use `TIME/LATITUDE/LONGITUDE/RAINFALL`),
    a descending latitude axis, extra years in the file (only `year` is kept) and either NaN or -999 as
    missing. The grid must equal the IMD grid of the config; nothing is regridded or guessed.
    """
    try:
        raw = xr.open_dataset(path)
    except Exception as exc:
        raise IngestionError(f"Could not open {path.name} as NetCDF ({type(exc).__name__}: {exc}).") from exc
    with raw:
        names = {v.lower(): v for v in raw.data_vars}
        var = next((names[n] for n in _RAIN_VARIABLES if n in names), None)
        if var is None and len(raw.data_vars) == 1:
            var = next(iter(raw.data_vars))
        if var is None:
            raise ValidationError(f"{path.name}: no rain variable found; variables are {list(raw.data_vars)}.")
        da = raw[var]
        rename = {d: _DIM_ALIASES[str(d).lower()] for d in da.dims if str(d).lower() in _DIM_ALIASES}
        da = da.rename(rename)
        if set(da.dims) != {"time", "lat", "lon"}:
            raise ValidationError(f"{path.name}: rain dims are {da.dims}, expected time, lat and lon.")
        da = da.transpose("time", "lat", "lon").sortby(["lat", "lon"])
        da = da.isel(time=np.flatnonzero((da["time"].dt.year == year).values))
        data = da.values.astype("float32")
        time = pd.DatetimeIndex(da["time"].values).normalize()
        lat, lon = da["lat"].values, da["lon"].values
    exp_lat, exp_lon = imd_axes(cfg)
    if (
        lat.size != exp_lat.size
        or lon.size != exp_lon.size
        or not (np.allclose(lat, exp_lat, atol=1e-3) and np.allclose(lon, exp_lon, atol=1e-3))
    ):
        raise ValidationError(
            f"{path.name}: grid is {lat.size} x {lon.size} ({lat.min():g}-{lat.max():g}N, {lon.min():g}-"
            f"{lon.max():g}E), expected {cfg.n_lat} x {cfg.n_lon} ({cfg.lat_origin:g}-{cfg.lat_end:g}N, "
            f"{cfg.lon_origin:g}-{cfg.lon_end:g}E) at {cfg.grid_spacing_deg:g} degrees."
        )
    if time.size != _days_in_year(year):
        raise ValidationError(
            f"{path.name}: has {time.size} days of {year} but the year has {_days_in_year(year)}. The file is "
            "truncated or is not a full-year file."
        )
    if time[0] != pd.Timestamp(year, 1, 1) or not (np.diff(time.values) == np.timedelta64(1, "D")).all():
        raise ValidationError(f"{path.name}: dates of {year} are not one per day from 1 January.")
    time = pd.date_range(f"{year}-01-01", periods=time.size, freq="D")  # same time axis as the .grd reader
    data[data == cfg.missing_value] = np.nan  # PRD 8.4: -999 becomes NaN (a file may use NaN already)
    ds = xr.Dataset(
        {"rain": (("time", "lat", "lon"), data, {"units": "mm/day", "long_name": "IMD daily rainfall"})},
        coords={"time": time, "lat": exp_lat, "lon": exp_lon},
    )
    ds.attrs.update(source="IMD gridded daily rainfall", grid="IMD_0.25", files=[path.name])
    return ds


def read_imd_year(path: Path | str, year: int, config: IngestionConfig | None = None) -> xr.Dataset:
    """Read one year of IMD rain (`rain`). `path` is normally the yearwise `.grd` file (float32, days x 129 x
    135, C order). If it is not there, the same year is looked for as an IMD NetCDF file
    (`resolve_imd_path`); a NetCDF `path` is read as NetCDF."""
    cfg = (config or load_config()).imd
    path = Path(path)
    if not path.is_file():
        alt = resolve_imd_path(year, cfg)
        if alt is not None:
            path = alt
    if not path.is_file():
        raise MissingInputError(
            f"IMD file not found: {path}. Download it with download_imd([{year}]) or check DATA_DIR."
        )
    if path.suffix.lower() in NETCDF_SUFFIXES:
        return _read_imd_netcdf(path, year, cfg)
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
