"""Backend settings. Everything comes from the environment / `.env` and the repository's `config/*.yaml`.

`DATABASE_URL` and `DATA_DIR` are the only environment variables the API needs (PRD 22.6). Nothing is hard-coded
and no secret is stored in the repository. Reading settings is cheap: no database connection, no data files.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from data_pipeline.ingestion.config import IngestionConfig, env_value, load_config


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    data_dir: Path
    golden_dir: Path  # cell-level Parquet stays outside the database (PRD D16)
    lead_days: list[int]
    thresholds_mm: list[float]
    grid: dict[str, float | int]
    district_source: str
    district_census_basis: int


def settings_from_config(cfg: IngestionConfig, database_url: str | None) -> Settings:
    im = cfg.imd
    return Settings(
        database_url=database_url,
        data_dir=cfg.data_dir,
        golden_dir=cfg.alignment.golden_dir,
        lead_days=list(cfg.alignment.lead_days),
        thresholds_mm=list(cfg.features.thresholds_mm),
        grid={
            "n_lat": im.n_lat,
            "n_lon": im.n_lon,
            "spacing_deg": im.grid_spacing_deg,
            "lat_min": im.lat_origin,
            "lat_max": im.lat_end,
            "lon_min": im.lon_origin,
            "lon_max": im.lon_end,
        },
        district_source=cfg.districts.source_name,
        district_census_basis=cfg.districts.census_basis,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return settings_from_config(load_config(), env_value("DATABASE_URL"))
