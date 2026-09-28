"""`/analogs` (PRD 15 F3): the most similar past monsoon situations and what actually happened in a district.
Own router, included once in `main.py`. Reads `data/regime/` and `data/features/` only."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Connection, select

from backend.app.api.deps import DISTRICT_ID_PATTERN, LeadRequired, RunIdRequired
from backend.app.config import Settings, get_settings
from backend.app.db.session import get_connection
from backend.app.db.tables import nwp_runs
from backend.app.schemas.analog_models import AnalogsResponse
from backend.app.services import analogs as analog_service
from backend.app.services import runs as run_service

router = APIRouter()
Conn = Annotated[Connection, Depends(get_connection)]

METHOD = "Similar monsoon situations: standardized distance on phase probabilities (active, break) and low-pressure-system presence and strength"


@router.get("/analogs", response_model=AnalogsResponse, tags=["regime"])
def historical_analogs(
    conn: Conn,
    settings: Annotated[Settings, Depends(get_settings)],
    run_id: RunIdRequired,
    lead_day: LeadRequired,
    district_id: Annotated[str, Query(pattern=DISTRICT_ID_PATTERN)],
) -> AnalogsResponse:
    """Up to five past days from other development seasons, at least 5 days apart, whose monsoon phase and
    low-pressure-system state most resemble this forecast's, with the district's observed rainfall on each."""
    run = run_service.get_run_row(conn, run_id)
    holdout = {
        int(s) for (s,) in conn.execute(select(nwp_runs.c.season).where(nwp_runs.c.evaluation_set == "holdout").distinct())
    }
    data = analog_service.get_analogs(
        settings.data_dir / "regime", settings.data_dir / "features", run_id, lead_day, district_id, run.season, holdout
    )
    n = data["n_analogs"]
    note = (
        f"{n} cases only. Other development seasons; the query's own season is excluded."
        if n
        else "No comparable past situations were found."
    )
    return AnalogsResponse(run_id=run_id, evaluation_set=run.evaluation_set, method=METHOD, note=note, **data)
