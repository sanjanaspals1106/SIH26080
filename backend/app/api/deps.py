"""Shared query-parameter types and dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Path as PathParam
from fastapi import Query

from backend.app.config import get_settings

RUN_ID_PATTERN = r"^tigge_ecmwf_cf_\d{10}$"  # PRD 9.1
DISTRICT_ID_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"

RunIdQuery = Annotated[
    str | None, Query(pattern=RUN_ID_PATTERN, description="Run ID, e.g. tigge_ecmwf_cf_2024071500")
]
RunIdRequired = Annotated[
    str, Query(pattern=RUN_ID_PATTERN, description="Run ID, e.g. tigge_ecmwf_cf_2024071500")
]
LeadQuery = Annotated[int | None, Query(ge=1, le=3, description="Lead day (1, 2 or 3)")]
LeadRequired = Annotated[int, Query(ge=1, le=3, description="Lead day (1, 2 or 3)")]
DistrictQuery = Annotated[str | None, Query(pattern=DISTRICT_ID_PATTERN)]
RunIdPath = Annotated[str, PathParam(pattern=RUN_ID_PATTERN)]
DistrictPath = Annotated[str, PathParam(pattern=DISTRICT_ID_PATTERN)]
Offset = Annotated[int, Query(ge=0, description="Rows to skip")]


def get_golden_dir() -> Path:
    """Where the cell-level Parquet files live (tests replace this dependency)."""
    return get_settings().golden_dir
