"""Ingestion configuration, read from `config/alignment.yaml` and `config/districts.yaml`.

All Stage 1 constants live in those two YAML files (PRD 7.2, 7.1, D7). This module only turns them
into typed objects. Secrets are not here: the ECDS key comes from `CDSAPI_URL` + `CDSAPI_KEY` or
`~/.cdsapirc` (see `tigge.py`).

`DATA_DIR` is taken from the environment, then from `.env` in the repository root, then from the
default in the YAML. Raw files go under `<DATA_DIR>/raw/...`, separate from processed data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from data_pipeline.ingestion.errors import IngestionError

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass(frozen=True)
class VariableSpec:
    """One row of the PRD 7.2 download list."""

    name: str  # internal name, for example "u850"
    group: str  # "rain", "atmosphere" or "static"
    levtype: str  # "sfc" or "pl"
    param: str  # TIGGE parameter, for example "u"
    cfgrib_name: str  # short name in the GRIB file
    level_hpa: int | None = None


@dataclass(frozen=True)
class Area:
    north: float
    west: float
    south: float
    east: float

    def as_list(self) -> list[float]:
        """ECDS/MARS order: north, west, south, east."""
        return [self.north, self.west, self.south, self.east]


@dataclass(frozen=True)
class TiggeConfig:
    dataset: str
    origin: str
    forecast_type: str
    init_hour_utc: int
    time: str
    data_format: str
    area: Area
    grid_spacing_deg: float
    domain_tolerance_deg: float
    request_timeout_s: int
    request_keys: dict[str, str]
    variables: dict[str, VariableSpec]
    steps_hours: dict[str, list[int]]  # group -> steps ("static" is [0])
    season_months: list[int]
    raw_dir: Path

    def variables_in_group(self, group: str) -> list[str]:
        return [n for n, v in self.variables.items() if v.group == group]


@dataclass(frozen=True)
class ImdConfig:
    var_type: str
    grid_spacing_deg: float
    n_lat: int
    n_lon: int
    lat_origin: float
    lon_origin: float
    lat_end: float
    lon_end: float
    missing_value: float
    dtype: str
    file_pattern: str
    download_timeout_s: int
    raw_dir: Path

    def year_path(self, year: int) -> Path:
        return self.raw_dir / self.file_pattern.format(year=year)


@dataclass(frozen=True)
class DistrictConfig:
    source_name: str
    census_basis: int
    file: str | None
    name_column: str
    state_column: str | None
    source_id_column: str | None
    assume_crs: str | None
    n_districts: int | None
    outline_simplify_deg: float
    raw_dir: Path


@dataclass(frozen=True)
class AlignmentConfig:
    """Stage 2 settings (PRD 8, 9.3). Read from the `time_alignment`, `valid_cells`, `checks` and `golden` sections."""

    method: str  # "C1" (C0 is not implemented)
    lead_days: list[int]
    imd_stamp: str  # "end_date" or "start_date" (PRD 8.2)
    alignment_offset_hours: dict[str, int]
    imd_day_end_ist: str  # "08:30"
    imd_day_end_utc: str  # "03:00"
    clip_max_share: float  # T2
    lag_shifts_days: list[int]  # T4
    lag_keep_margin: float  # T4
    tp_non_decreasing_min_share: float  # T1
    tp_decrease_tolerance_mm: float
    max_rain_24h_mm: float
    season_months: list[int]
    base_start_year: int
    base_end_year: int
    min_non_missing_fraction: float
    golden_dir: Path
    static_date: str | None


@dataclass(frozen=True)
class FeaturesConfig:
    """Stage 3 settings (PRD 8.4, 9.4, 14.2, 14.4). Read from alignment.yaml, districts.yaml, thresholds.yaml."""

    dir: Path
    doy_period_days: float
    km_per_quarter_degree_lat: float  # PRD 8.6 / Appendix B: 27.75 km per 0.25 degree of latitude
    thresholds_mm: list[float]  # 15.6, 64.5, 115.6 (D8)
    projection_epsg: int  # 6933 (PRD 14.2)
    w_min: float  # 0.05: a cell is "main" if its weight is at least this
    renormalise_after_dropping_invalid_cells: bool
    small_n_effective_max_exclusive: float  # is_small = n_effective_cells < this


@dataclass(frozen=True)
class IngestionConfig:
    data_dir: Path
    tigge: TiggeConfig
    imd: ImdConfig
    districts: DistrictConfig
    alignment: AlignmentConfig
    features: FeaturesConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IngestionError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _dotenv_value(name: str, root: Path) -> str | None:
    """Environment value, or a simple KEY=VALUE line of `.env` in the repository root."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key.strip() == name and value.strip():
                    return value.strip().strip("\"'")
    return None


def env_value(name: str) -> str | None:
    """Environment variable, or the same name in the repository's `.env` (used by the backend for DATABASE_URL)."""
    return _dotenv_value(name, REPO_ROOT)


def _resolve_data_dir(section: dict[str, Any], root: Path, override: Path | str | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    raw = _dotenv_value(section["data_dir_env"], root) or section["default_data_dir"]
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_config(config_dir: Path | str | None = None, data_dir: Path | str | None = None) -> IngestionConfig:
    """Load the ingestion configuration.

    `config_dir` defaults to the repository's `config/`. `data_dir` overrides DATA_DIR (used by
    tests and the synthetic example so that nothing is written into the real data folder).
    """
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    root = cdir.resolve().parent
    align = _read_yaml(cdir / "alignment.yaml")
    dist = _read_yaml(cdir / "districts.yaml")
    thresholds = _read_yaml(cdir / "thresholds.yaml")

    try:
        ing = align["ingestion"]
        fc, truth = align["forecast"], align["truth"]
        base = _resolve_data_dir(ing, root, data_dir)
        raw = {k: base / v for k, v in ing["raw_dirs"].items()}

        variables = {
            name: VariableSpec(
                name=name,
                group=spec["group"],
                levtype=spec["levtype"],
                param=spec["param"],
                cfgrib_name=spec["cfgrib_name"],
                level_hpa=spec.get("level_hpa"),
            )
            for name, spec in fc["variables"].items()
        }
        tg = ing["tigge"]
        tigge = TiggeConfig(
            dataset=tg["dataset"],
            origin=tg["origin"],
            forecast_type=fc["type"],
            init_hour_utc=int(fc["init_hour_utc"]),
            time=tg["time"],
            data_format=tg["data_format"],
            area=Area(**{k: float(v) for k, v in fc["download_area"].items()}),
            grid_spacing_deg=float(fc["grid_spacing_deg"]),
            domain_tolerance_deg=float(tg["domain_tolerance_deg"]),
            request_timeout_s=int(tg["request_timeout_s"]),
            request_keys=dict(tg["request_keys"]),
            variables=variables,
            steps_hours={
                "rain": list(fc["tp_steps_hours"]),
                "atmosphere": list(fc["atmosphere_steps_hours"]),
                "static": [0],
            },
            season_months=list(align["valid_cells"]["season_months"]),
            raw_dir=raw["tigge"],
        )

        im = ing["imd"]
        imd = ImdConfig(
            var_type=im["var_type"],
            grid_spacing_deg=float(truth["grid_spacing_deg"]),
            n_lat=int(truth["grid_shape"]["lat"]),
            n_lon=int(truth["grid_shape"]["lon"]),
            lat_origin=float(truth["grid_origin"]["lat"]),
            lon_origin=float(truth["grid_origin"]["lon"]),
            lat_end=float(truth["grid_end"]["lat"]),
            lon_end=float(truth["grid_end"]["lon"]),
            missing_value=float(truth["missing_value"]),
            dtype=im["dtype"],
            file_pattern=im["file_pattern"],
            download_timeout_s=int(im["download_timeout_s"]),
            raw_dir=raw["imd"],
        )

        ta, vc, ck = align["time_alignment"], align["valid_cells"], align["checks"]
        alignment = AlignmentConfig(
            method=str(ta["method"]),
            lead_days=[int(k) for k in fc["lead_days"]],
            imd_stamp=str(ta["imd_stamp"]),
            alignment_offset_hours={str(k): int(v) for k, v in ta["alignment_offset_hours"].items()},
            imd_day_end_ist=str(truth["imd_day_end_ist"]),
            imd_day_end_utc=str(truth["imd_day_end_utc"]),
            clip_max_share=float(ta["window_negative_clip_max_share"]),
            lag_shifts_days=[int(s) for s in ta["lag_test_shifts_days"]],
            lag_keep_margin=float(ta["lag_test_keep_end_date_margin"]),
            tp_non_decreasing_min_share=float(ck["tp_non_decreasing_min_share"]),
            tp_decrease_tolerance_mm=float(ck["tp_decrease_tolerance_mm"]),
            max_rain_24h_mm=float(ck["max_rain_24h_mm"]),
            season_months=[int(m) for m in vc["season_months"]],
            base_start_year=int(vc["base_period"]["start_year"]),
            base_end_year=int(vc["base_period"]["end_year"]),
            min_non_missing_fraction=float(vc["min_non_missing_fraction"]),
            golden_dir=base / align["golden"]["dir"],
            static_date=align["golden"].get("static_date"),
        )

        features = FeaturesConfig(
            dir=base / align["features"]["dir"],
            doy_period_days=float(align["features"]["doy_period_days"]),
            km_per_quarter_degree_lat=float(align["space"]["km_per_quarter_degree_lat"]),
            thresholds_mm=[float(x) for x in thresholds["rain_thresholds_mm"]],
            projection_epsg=int(dist["weights"]["projection_epsg"]),
            w_min=float(dist["weights"]["w_min"]),
            renormalise_after_dropping_invalid_cells=bool(
                dist["weights"]["renormalise_after_dropping_invalid_cells"]
            ),
            small_n_effective_max_exclusive=float(dist["small_district"]["n_effective_cells_max_exclusive"]),
        )

        src, cols = dist["source"], dist["source"].get("columns", {})
        districts = DistrictConfig(
            source_name=src["name"],
            census_basis=int(src["census_basis"]),
            file=src.get("file"),
            name_column=cols["name"],
            state_column=cols.get("state"),
            source_id_column=cols.get("source_id"),
            assume_crs=src.get("assume_crs"),
            n_districts=src.get("n_districts"),
            outline_simplify_deg=float(dist["outline"]["simplify_tolerance_deg"]),
            raw_dir=raw["districts"],
        )
    except KeyError as exc:
        raise IngestionError(
            f"Config is missing the key {exc}. Check config/alignment.yaml and config/districts.yaml "
            "against the versions in the repository."
        ) from exc

    return IngestionConfig(
        data_dir=base, tigge=tigge, imd=imd, districts=districts, alignment=alignment, features=features
    )
