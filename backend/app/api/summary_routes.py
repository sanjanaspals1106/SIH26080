"""F1 improvement summary and F5 priority table (PRD 15, 18.4): `GET /forecasts/improvement-summary` and
`GET /districts/priority`. Own router, included once in `main.py`. Read-only."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Connection

from backend.app.api.deps import LeadRequired, RunIdRequired, get_golden_dir
from backend.app.config import Settings, get_settings
from backend.app.db.session import get_connection
from backend.app.errors import ApiError
from backend.app.schemas.product_models import ImprovementSummaryOut, PriorityTableOut
from backend.app.services import forecasts as forecast_service
from backend.app.services import improvement as improvement_service
from backend.app.services import runs as run_service

router = APIRouter()
Conn = Annotated[Connection, Depends(get_connection)]


@router.get("/forecasts/improvement-summary", response_model=ImprovementSummaryOut, tags=["forecasts"])
def improvement_summary(
    conn: Conn,
    settings: Annotated[Settings, Depends(get_settings)],
    golden_dir: Annotated[Path, Depends(get_golden_dir)],
    run_id: RunIdRequired,
    lead_day: LeadRequired,
) -> ImprovementSummaryOut:
    """Cells and districts where the corrected forecast is closer to the IMD observation than the raw NWP (F1).
    Counts only rows that have all three values; one day is a description, not evidence."""
    run = run_service.get_run_row(conn, run_id)
    data = improvement_service.improvement_summary(conn, settings.data_dir, golden_dir, run_id, lead_day)
    return ImprovementSummaryOut(run_id=run_id, lead_day=lead_day, evaluation_set=run.evaluation_set, **data)


@router.get("/districts/priority", response_model=PriorityTableOut, tags=["districts"])
def district_priority(
    conn: Conn,
    run_id: RunIdRequired,
    lead_day: LeadRequired,
    state: Annotated[str | None, Query(max_length=100)] = None,
) -> PriorityTableOut:
    """The F5 table: every district of a run and lead, in priority order (level, then P(>=64.5 mm), expected heavy
    area, wettest-cell mean, name), with the attention level computed by `data_pipeline/districts/priority.py`."""
    run = run_service.get_run_row(conn, run_id)
    total, rows = forecast_service.query_forecasts(
        conn, run_id=run_id, lead_day=lead_day, district_id=None, state=state, imd_date=None, season=None,
        limit=1000, offset=0,
    )  # fmt: skip
    if total == 0:
        raise ApiError(404, "FORECAST_NOT_FOUND", "No district forecasts were found for this run and lead.")
    rows.sort(key=lambda r: (r["priority_rank"] is None, r["priority_rank"] or 0, r["district_id"]))
    return PriorityTableOut(
        run_id=run_id, lead_day=lead_day, imd_date=rows[0]["imd_date"], evaluation_set=run.evaluation_set,
        total=total, districts=rows,
    )  # fmt: skip
