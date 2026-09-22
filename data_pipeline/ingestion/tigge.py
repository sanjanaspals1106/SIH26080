"""ECMWF TIGGE ingestion adapter (PRD 7.2, 7.3, 9.1, 9.2). Owner: M1.

Two halves:

* **Download** (`download_tigge_month`, `download_tigge_static`): one ECDS request per month and
  per (group, level type, level), saved under `<DATA_DIR>/raw/tigge/<year>/` with a deterministic
  name. A file that already exists is never downloaded again.
* **Read** (`read_tigge`): GRIB file(s) -> one normalised `xarray.Dataset` per group.

Normalised representation (what Stage 2 imports):

* dims `(init_time, lead_hours, lat, lon)` for `rain` and `atmosphere` variables; `lat` and `lon`
  are ascending, `init_time` is UTC, `lead_hours` is an int (hours after the start time). A
  `valid_time(init_time, lead_hours)` coordinate is added.
* dims `(lat, lon)` for the `static` variables (`orog`, `lsm`).
* variable names are the internal names of `config/alignment.yaml` (`tp`, `msl`, `u850`, ...).
* values are exactly as stored in the file. `tp` is still **cumulative** from step 0 (PRD 8.1);
  differencing is Stage 2.

Credentials: cdsapi reads `CDSAPI_URL` + `CDSAPI_KEY` from the environment or `~/.cdsapirc`.
Nothing is stored in this repository.
"""

from __future__ import annotations

import calendar
import logging
import os
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.ingestion.config import IngestionConfig, TiggeConfig, load_config
from data_pipeline.ingestion.errors import (
    DownloadError,
    IngestionError,
    MissingCredentialsError,
    MissingInputError,
    ValidationError,
)

log = logging.getLogger(__name__)

GROUPS = ("rain", "atmosphere", "static")
_GRIB_SUFFIXES = {".grib", ".grib2", ".grb", ".grb2"}


def run_id(init_time: Any) -> str:
    """Run ID of PRD 9.1: `tigge_ecmwf_cf_YYYYMMDDHH`."""
    return f"tigge_ecmwf_cf_{pd.Timestamp(init_time):%Y%m%d%H}"


# --------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------


def _request_units(cfg: TiggeConfig, group: str) -> list[dict[str, Any]]:
    """Split a group into ECDS requests: one per (level type, pressure level)."""
    if group not in GROUPS:
        raise IngestionError(f"Unknown TIGGE group '{group}'. Use one of {GROUPS}.")
    units: dict[tuple[str, int | None], list[str]] = {}
    for name in cfg.variables_in_group(group):
        spec = cfg.variables[name]
        params = units.setdefault((spec.levtype, spec.level_hpa), [])
        if spec.param not in params:
            params.append(spec.param)
    return [
        {
            "unit": f"{levtype}{level if level is not None else ''}",
            "levtype": levtype,
            "level": level,
            "params": params,
        }
        for (levtype, level), params in units.items()
    ]


def raw_path(cfg: TiggeConfig, group: str, tag: str, unit: str) -> Path:
    """Deterministic raw file name. `tag` is `YYYYMM` (rain, atmosphere) or `YYYYMMDD` (static)."""
    return cfg.raw_dir / tag[:4] / f"tigge_ecmwf_{cfg.forecast_type}_{tag}_{group}_{unit}.grib"


def tigge_raw_paths(group: str, tag: str, config: IngestionConfig | None = None) -> list[Path]:
    """Where the raw files of one group/month (or static date) are expected to be."""
    cfg = (config or load_config()).tigge
    return [raw_path(cfg, group, tag, u["unit"]) for u in _request_units(cfg, group)]


def build_request(
    cfg: TiggeConfig, group: str, unit: dict[str, Any], dates: Sequence[date]
) -> dict[str, Any]:
    """The ECDS request for one unit of one group. Key names come from `config/alignment.yaml`.

    TODO(M1, CK2): compare with "Show API request code" on the ECDS TIGGE form. Only the config
    (`ingestion.tigge.request_keys`) should need to change.
    """
    k = cfg.request_keys
    request: dict[str, Any] = {
        k["origin"]: cfg.origin,
        k["type"]: cfg.forecast_type,
        k["time"]: cfg.time,
        k["date"]: [d.isoformat() for d in dates],
        k["step"]: cfg.steps_hours[group],
        k["variable"]: unit["params"],
        k["levtype"]: unit["levtype"],
        k["area"]: cfg.area.as_list(),
        k["grid"]: [cfg.grid_spacing_deg, cfg.grid_spacing_deg],
        k["format"]: cfg.data_format,
    }
    if unit["level"] is not None:
        request[k["level"]] = [unit["level"]]
    return request


def _check_credentials() -> None:
    have_env = bool(os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"))
    rc = Path(os.environ.get("CDSAPI_RC", "~/.cdsapirc")).expanduser()
    if not have_env and not rc.is_file():
        raise MissingCredentialsError(
            "No ECDS credentials found. Register on the ECMWF Data Store, then either copy the "
            "exact API lines from your ECDS profile page into ~/.cdsapirc, or set the environment "
            "variables CDSAPI_URL and CDSAPI_KEY. Never put the key in the repository (PRD 22.2)."
        )


def _default_client(cfg: TiggeConfig) -> Any:
    _check_credentials()
    try:
        import cdsapi
    except ImportError as exc:
        raise IngestionError(
            "The 'cdsapi' package is not installed. Run `conda env create -f environment.yml`."
        ) from exc
    return cdsapi.Client(timeout=cfg.request_timeout_s)


def _fetch(client: Any, cfg: TiggeConfig, request: dict[str, Any], target: Path, force: bool) -> Path:
    """Download one request to `target`, skipping it if the file already exists."""
    if target.is_file() and target.stat().st_size > 0 and not force:
        log.info("TIGGE file already present, not downloading again: %s", target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    log.info("Requesting TIGGE %s -> %s", cfg.dataset, target.name)
    try:
        client.retrieve(cfg.dataset, request, str(partial))
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"ECDS request for {target.name} failed: {type(exc).__name__}: {exc}. "
            "If the message mentions an unknown key or value, compare the request with "
            "'Show API request code' on the ECDS form and fix ingestion.tigge in "
            "config/alignment.yaml (CK2). If it mentions the size, request fewer dates (CK12). "
            "If it mentions the licence, accept the TIGGE terms on the ECDS dataset page."
        ) from exc
    if not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"ECDS returned no data for {target.name}.")
    os.replace(partial, target)
    return target


def download_tigge_month(
    year: int,
    month: int,
    groups: Iterable[str] = ("rain", "atmosphere"),
    config: IngestionConfig | None = None,
    client: Any = None,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Download one month of 00 UTC control forecasts (PRD 7.3 step 3). Returns paths per group."""
    cfg = (config or load_config()).tigge
    if month not in cfg.season_months:
        raise IngestionError(
            f"Month {month} is outside the JJAS season {cfg.season_months} (PRD 7.2 downloads "
            "1 June to 30 September only)."
        )
    dates = [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    tag = f"{year}{month:02d}"
    out: dict[str, list[Path]] = {}
    for group in groups:
        if group == "static":
            raise IngestionError("Use download_tigge_static() for orog and lsm (one date only).")
        client = client or _default_client(cfg)
        out[group] = [
            _fetch(
                client, cfg, build_request(cfg, group, u, dates), raw_path(cfg, group, tag, u["unit"]), force
            )
            for u in _request_units(cfg, group)
        ]
    return out


def download_tigge_static(
    on_date: date,
    config: IngestionConfig | None = None,
    client: Any = None,
    force: bool = False,
) -> list[Path]:
    """Download `orog` and `lsm` for one date, step 0 (PRD 7.2, D17)."""
    cfg = (config or load_config()).tigge
    client = client or _default_client(cfg)
    tag = f"{on_date:%Y%m%d}"
    return [
        _fetch(
            client,
            cfg,
            build_request(cfg, "static", u, [on_date]),
            raw_path(cfg, "static", tag, u["unit"]),
            force,
        )
        for u in _request_units(cfg, "static")
    ]


# --------------------------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------------------------


def _open_grib(path: Path) -> list[xr.Dataset]:
    if not path.is_file():
        raise MissingInputError(
            f"TIGGE file not found: {path}. Download it with download_tigge_month()/"
            "download_tigge_static(), or check DATA_DIR."
        )
    if path.stat().st_size == 0:
        raise MissingInputError(f"TIGGE file is empty: {path}. Delete it and download again.")
    if path.suffix.lower() not in _GRIB_SUFFIXES:
        raise IngestionError(f"Expected a GRIB file (.grib/.grib2), got: {path.name}")
    try:
        import cfgrib

        # indexpath="" so that no .idx file is written next to the raw data.
        return cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}, decode_timedelta=True)
    except Exception as exc:
        raise IngestionError(
            f"Could not decode {path.name} as GRIB ({type(exc).__name__}: {exc}). The file may be "
            "truncated or an HTML error page; delete it and download again. Also check that "
            "eccodes and cfgrib are installed (conda env create -f environment.yml)."
        ) from exc


def _standardise(da: xr.DataArray, name: str) -> xr.DataArray:
    """cfgrib DataArray -> dims (init_time, lead_hours, lat, lon), ascending lat/lon."""
    da = da.rename({"latitude": "lat", "longitude": "lon"})
    if "time" not in da.dims:
        da = da.expand_dims("time")
    if "step" not in da.dims:
        da = da.expand_dims("step")
    hours = (da["step"] / np.timedelta64(1, "h")).round().astype("int64").values
    da = da.assign_coords(step=hours).rename({"step": "lead_hours", "time": "init_time"})
    da = da.reset_coords(drop=True).sortby(["lat", "lon"])
    da = da.transpose("init_time", "lead_hours", "lat", "lon")
    da = da.assign_coords(
        valid_time=(
            ("init_time", "lead_hours"),
            da["init_time"].values[:, None] + pd.to_timedelta(da["lead_hours"].values, "h").values[None, :],
        )
    )
    da.name = name
    return da


def _extract_variables(datasets: list[xr.Dataset], cfg: TiggeConfig) -> dict[str, xr.DataArray]:
    """Pick the configured variables out of cfgrib's datasets and rename them."""
    wanted: dict[tuple[str, int | None], str] = {
        (s.cfgrib_name, s.level_hpa): n for n, s in cfg.variables.items()
    }
    found: dict[str, xr.DataArray] = {}
    for ds in datasets:
        for var in ds.data_vars:
            da = ds[var]
            if "isobaricInhPa" in da.coords:
                levels = np.atleast_1d(da["isobaricInhPa"].values)
                slices = [
                    (int(lv), da.sel(isobaricInhPa=lv) if da["isobaricInhPa"].ndim else da) for lv in levels
                ]
            else:
                slices = [(None, da)]
            for level, sub in slices:
                name = wanted.get((str(var), level))
                if name is None:
                    log.debug("Ignoring %s level %s (not in the download list)", var, level)
                    continue
                found[name] = _standardise(sub, name)
    return found


def _collapse_static(da: xr.DataArray) -> xr.DataArray:
    if list(da["lead_hours"].values) != [0]:
        raise ValidationError(
            f"{da.name}: static fields must be at step 0 only, found steps {list(da['lead_hours'].values)}."
        )
    if da.sizes["init_time"] > 1:
        log.warning(
            "%s: %d dates in a static file; using the first (T8 checks they are equal)",
            da.name,
            da.sizes["init_time"],
        )
    out = da.isel(init_time=0, lead_hours=0, drop=True).reset_coords(drop=True)
    out.name = da.name
    return out


def _join_paths(arrays: dict[str, list[xr.DataArray]]) -> dict[str, xr.DataArray]:
    """Concatenate the same variable from several files (several months) along init_time."""
    joined: dict[str, xr.DataArray] = {}
    for name, parts in arrays.items():
        if len(parts) == 1:
            joined[name] = parts[0]
            continue
        try:
            da = xr.concat(parts, dim="init_time", join="exact").sortby("init_time")
        except ValueError as exc:
            raise ValidationError(f"{name}: files do not share the same grid or steps ({exc}).") from exc
        if da.indexes["init_time"].has_duplicates:
            raise ValidationError(f"{name}: the same start time appears in more than one file.")
        joined[name] = da
    return joined


def read_tigge(
    paths: Sequence[Path | str] | Path | str,
    group: str,
    config: IngestionConfig | None = None,
    validate: bool = True,
) -> xr.Dataset:
    """Read TIGGE GRIB file(s) of one group into the normalised Dataset (see module docstring).

    `paths` may hold several files: different request units of one month (for example the
    surface and the 850 hPa files) and/or several months. Only the variables of `group` are kept.
    """
    cfg = (config or load_config()).tigge
    if group not in GROUPS:
        raise IngestionError(f"Unknown TIGGE group '{group}'. Use one of {GROUPS}.")
    path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    if not path_list:
        raise MissingInputError("No TIGGE files given to read_tigge().")

    expected = set(cfg.variables_in_group(group))
    parts: dict[str, list[xr.DataArray]] = {}
    for path in path_list:
        for name, da in _extract_variables(_open_grib(path), cfg).items():
            if name in expected:
                parts.setdefault(name, []).append(da)

    arrays = _join_paths(parts)
    if group == "static":
        arrays = {n: _collapse_static(a) for n, a in arrays.items()}
    if not arrays:
        raise ValidationError(
            f"None of the {group} variables {sorted(expected)} were found in {[p.name for p in path_list]}."
        )
    try:
        ds = xr.Dataset(arrays)
    except ValueError as exc:
        raise ValidationError(f"Variables of group '{group}' do not share one grid ({exc}).") from exc
    ds.attrs.update(source="ECMWF-TIGGE-control", group=group, files=[p.name for p in path_list])
    if validate:
        validate_tigge(ds, group, config)
    return ds


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def validate_tigge(ds: xr.Dataset, group: str, config: IngestionConfig | None = None) -> None:
    """Basic checks on a normalised TIGGE Dataset. Raises one ValidationError listing every problem."""
    cfg = (config or load_config()).tigge
    problems: list[str] = []

    missing = [v for v in cfg.variables_in_group(group) if v not in ds.data_vars]
    if missing:
        problems.append(f"missing variables {missing}")

    needed = ["lat", "lon"] + ([] if group == "static" else ["init_time", "lead_hours"])
    absent = [c for c in needed if c not in ds.coords]
    if absent:
        problems.append(f"missing coordinates {absent}")

    for v in ds.data_vars:
        if ds[v].size == 0 or not np.isfinite(ds[v].values).any():
            problems.append(f"{v} is empty (no finite values)")

    if "lat" in ds.coords and "lon" in ds.coords:
        a, tol = cfg.area, cfg.domain_tolerance_deg
        lat, lon = ds["lat"].values, ds["lon"].values
        if lat.size == 0 or lon.size == 0:
            problems.append("lat/lon are empty")
        else:
            for label, got, want in (
                ("south", lat.min(), a.south),
                ("north", lat.max(), a.north),
                ("west", lon.min(), a.west),
                ("east", lon.max(), a.east),
            ):
                if abs(got - want) > tol:
                    problems.append(
                        f"domain {label} edge is {got:g}, expected about {want:g} (PRD 7.2 area N{a.north:g} W{a.west:g} S{a.south:g} E{a.east:g})"
                    )
            for label, axis in (("lat", lat), ("lon", lon)):
                if axis.size > 1 and abs(np.median(np.diff(axis)) - cfg.grid_spacing_deg) > 1e-3:
                    log.warning(
                        "TIGGE %s spacing is %g deg, not %g (CK2: the form may not offer 0.25; regridding is Stage 2)",
                        label,
                        np.median(np.diff(axis)),
                        cfg.grid_spacing_deg,
                    )

    if group != "static" and {"init_time", "lead_hours"} <= set(ds.coords):
        init = pd.DatetimeIndex(ds["init_time"].values)
        if init.size == 0:
            problems.append("no start times")
        elif init.isna().any():
            problems.append("some start times are missing (NaT)")
        elif (init.hour != cfg.init_hour_utc).any() or (init.minute != 0).any():
            problems.append(
                f"start times are not all {cfg.init_hour_utc:02d}:00 UTC (found hours {sorted(set(init.hour))})"
            )
        steps = {int(s) for s in ds["lead_hours"].values}
        want = set(cfg.steps_hours[group])
        if steps != want:
            problems.append(
                f"steps differ from PRD 7.2: missing {sorted(want - steps)}, unexpected {sorted(steps - want)}"
            )

    if problems:
        raise ValidationError(
            "TIGGE validation failed for group '" + group + "': " + "; ".join(problems) + "."
        )
