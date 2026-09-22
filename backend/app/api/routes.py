"""The M1 endpoints of PRD 18.4, under `/api/v1`: health, runs, district forecasts, the cell grid, map metadata
and district outlines. All read-only."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import Connection, text
from sqlalchemy.exc import SQLAlchemyError

from backend.app.api.deps import (
    DistrictPath,
    DistrictQuery,
    LeadQuery,
    LeadRequired,
    Offset,
    RunIdPath,
    RunIdQuery,
    RunIdRequired,
    get_golden_dir,
)
from backend.app.config import Settings, get_settings
from backend.app.db.session import engine_for, get_connection
from backend.app.errors import ApiError
from backend.app.schemas.models import (
    DistrictFeatureCollection,
    DistrictForecastDetail,
    DistrictForecastList,
    GridLayer,
    Health,
    MapMetadata,
    Run,
    RunList,
)
from backend.app.services import forecasts as forecast_service
from backend.app.services import grid as grid_service
from backend.app.services import map_data
from backend.app.services import runs as run_service

log = logging.getLogger(__name__)
VERSION = "0.1.0"
router = APIRouter()
Conn = Annotated[Connection, Depends(get_connection)]

# ---- health ------------------------------------------------------------------------------------------


@router.get("/health", response_model=Health, tags=["health"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> Health:
    """Service status and whether the database answers. Always 200 while the process is up."""
    if not settings.database_url:
        db = "not_configured"
    else:
        try:
            with engine_for(settings.database_url).connect() as conn:
                conn.execute(text("SELECT 1"))
            db = "ok"
        except (SQLAlchemyError, ApiError) as exc:
            log.warning("health check: database unavailable: %s", exc)
            db = "unavailable"
    return Health(status="ok" if db == "ok" else "degraded", version=VERSION, database=db)


# ---- runs --------------------------------------------------------------------------------------------


@router.get("/runs", response_model=RunList, tags=["runs"])
def list_runs(
    conn: Conn,
    season: Annotated[int | None, Query(ge=1900, le=2200)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Offset = 0,
) -> RunList:
    """Available NWP runs (replay dates), oldest first."""
    total, rows = run_service.list_runs(conn, season, limit, offset)
    return RunList(total=total, limit=limit, offset=offset, runs=rows)


@router.get("/runs/{run_id}", response_model=Run, tags=["runs"])
def get_run(run_id: RunIdPath, conn: Conn) -> Run:
    return Run(**run_service.get_run(conn, run_id))


# ---- district forecasts ------------------------------------------------------------------------------


@router.get("/forecasts/districts", response_model=DistrictForecastList, tags=["forecasts"])
def list_district_forecasts(
    conn: Conn,
    run_id: RunIdQuery = None,
    lead_day: LeadQuery = None,
    district_id: DistrictQuery = None,
    state: Annotated[str | None, Query(max_length=100)] = None,
    imd_date: date | None = None,
    season: Annotated[int | None, Query(ge=1900, le=2200)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Offset = 0,
) -> DistrictForecastList:
    """District table, filtered and paginated. Fields owned by later stages are null until they exist."""
    total, rows, evaluation_set = forecast_service.list_district_forecasts(
        conn,
        run_id=run_id,
        lead_day=lead_day,
        district_id=district_id,
        state=state,
        imd_date=imd_date,
        season=season,
        limit=limit,
        offset=offset,
    )
    return DistrictForecastList(
        total=total, limit=limit, offset=offset, run_id=run_id, evaluation_set=evaluation_set, districts=rows
    )


@router.get("/forecasts/districts/{district_id}", response_model=DistrictForecastDetail, tags=["forecasts"])
def get_district_forecasts(
    district_id: DistrictPath,
    conn: Conn,
    run_id: RunIdQuery = None,
    lead_day: LeadQuery = None,
    imd_date: date | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Offset = 0,
) -> DistrictForecastDetail:
    """One district: its details and its forecasts (for a run and lead if given)."""
    info, total, rows = forecast_service.district_detail(
        conn, district_id, run_id=run_id, lead_day=lead_day, imd_date=imd_date, limit=limit, offset=offset
    )
    return DistrictForecastDetail(total=total, limit=limit, offset=offset, district=info, forecasts=rows)


# ---- grid --------------------------------------------------------------------------------------------


@router.get("/forecasts/grid", response_model=GridLayer, tags=["forecasts"])
def grid_layer(
    conn: Conn,
    golden_dir: Annotated[Path, Depends(get_golden_dir)],
    run_id: RunIdRequired,
    lead_day: LeadRequired,
    variable: Annotated[str, Query(pattern="^(" + "|".join(grid_service.VARIABLES) + ")$")] = "raw",
    limit: Annotated[int, Query(ge=1, le=20000)] = 20000,
    offset: Offset = 0,
) -> GridLayer:
    """One cell layer for a run and lead, read from Parquet. Only `raw` and `observed` exist at this stage."""
    run = run_service.get_run_row(conn, run_id)
    layer = grid_service.read_layer(golden_dir, run_id, lead_day, variable, limit, offset)
    return GridLayer(
        total=layer["total"], limit=limit, offset=offset, run_id=run_id, lead_day=lead_day, imd_date=layer["imd_date"],
        evaluation_set=run.evaluation_set, variable=variable, unit="mm", cells=layer["cells"],
    )  # fmt: skip


# ---- map ---------------------------------------------------------------------------------------------


@router.get("/map/metadata", response_model=MapMetadata, tags=["map"])
def map_metadata(conn: Conn, settings: Annotated[Settings, Depends(get_settings)]) -> MapMetadata:
    return MapMetadata(**map_data.metadata(conn, settings))


@router.get("/map/districts", response_model=DistrictFeatureCollection, tags=["map"])
def map_districts(
    conn: Conn, response: Response, settings: Annotated[Settings, Depends(get_settings)]
) -> DistrictFeatureCollection:
    """The stored (simplified) district outlines as GeoJSON, served once (PRD 19.3)."""
    response.headers["Cache-Control"] = "public, max-age=3600"
    return DistrictFeatureCollection(
        district_file=settings.district_source,
        census_basis_year=settings.district_census_basis,
        features=map_data.district_outlines(conn),
    )
