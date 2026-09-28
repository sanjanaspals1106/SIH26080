"""F6 hotspots (PRD 15 F6, 18.4): `GET /hotspots`. Own router, included once in `main.py`. Reads the served
corrected cell store; never trains or writes anything."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import Connection

from backend.app.api.deps import LeadRequired, RunIdRequired, get_golden_dir
from backend.app.config import Settings, get_settings
from backend.app.db.session import get_connection
from backend.app.schemas.product_models import HotspotList
from backend.app.services import hotspots as hotspot_service
from backend.app.services import runs as run_service

router = APIRouter()
Conn = Annotated[Connection, Depends(get_connection)]


@router.get("/hotspots", response_model=HotspotList, tags=["hotspots"])
def hotspots(
    conn: Conn,
    settings: Annotated[Settings, Depends(get_settings)],
    golden_dir: Annotated[Path, Depends(get_golden_dir)],
    run_id: RunIdRequired,
    lead_day: LeadRequired,
) -> HotspotList:
    """8-connected groups of cells with P(>=64.5 mm) >= 0.50 and a corrected mean >= 15.6 mm. The response carries
    the threshold in use (PRD 15 F6: "the threshold in use is visible")."""
    run = run_service.get_run_row(conn, run_id)
    data = hotspot_service.get_hotspots(conn, settings.data_dir, golden_dir, run_id, lead_day)
    return HotspotList(run_id=run_id, lead_day=lead_day, evaluation_set=run.evaluation_set, **data)
