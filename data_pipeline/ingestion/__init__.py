"""Stage 1: data ingestion (PRD 7, 9.1, 9.2). Owner: M1.

TIGGE and IMD adapters that produce one normalised xarray format (dims named `lat`, `lon`, ...),
plus the shared configuration. District boundaries are loaded by `data_pipeline.districts`.
"""

from data_pipeline.ingestion.config import IngestionConfig, load_config
from data_pipeline.ingestion.errors import (
    DownloadError,
    IngestionError,
    MissingCredentialsError,
    MissingInputError,
    ValidationError,
)
from data_pipeline.ingestion.imd import download_imd, read_imd, validate_imd
from data_pipeline.ingestion.tigge import (
    download_tigge_month,
    download_tigge_static,
    legacy_month_paths,
    read_atmosphere_month,
    read_month,
    read_static_from_grib,
    read_tigge,
    run_id,
    tigge_raw_paths,
    validate_tigge,
)

__all__ = [
    "DownloadError",
    "IngestionConfig",
    "IngestionError",
    "MissingCredentialsError",
    "MissingInputError",
    "ValidationError",
    "download_imd",
    "download_tigge_month",
    "download_tigge_static",
    "legacy_month_paths",
    "load_config",
    "read_imd",
    "read_atmosphere_month",
    "read_month",
    "read_static_from_grib",
    "read_tigge",
    "run_id",
    "tigge_raw_paths",
    "validate_imd",
    "validate_tigge",
]
