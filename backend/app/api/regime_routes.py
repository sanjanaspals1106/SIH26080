"""The regime endpoints, additive to `api/routes.py` (PRD 15 F1-F4): `/regime` and `/regime/transitions`. Own
router, included once in `main.py`. Reads `data/regime/` only; never recomputes, retrains or writes anything."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.api.deps import DistrictQuery, LeadRequired, RunIdRequired, get_golden_dir
from backend.app.config import Settings, get_settings
from backend.app.db.session import get_connection
from backend.app.schemas.regime_models import RegimeResponse, RegimeTransitionsResponse
from backend.app.services import regime as regime_service
from backend.app.services import runs as run_service
from sqlalchemy import Connection

log = logging.getLogger(__name__)
router = APIRouter()
Conn = Annotated[Connection, Depends(get_connection)]


@router.get("/regime", response_model=RegimeResponse, tags=["regime"])
def regime_for_run(
    conn: Conn,
    settings: Annotated[Settings, Depends(get_settings)],
    golden_dir: Annotated[Path, Depends(get_golden_dir)],
    run_id: RunIdRequired,
    lead_day: LeadRequired,
) -> RegimeResponse:
    """Monsoon phase, low-pressure-system and domain-mean rainfall for one run and lead (PRD 11.6). `indicators`
    is always empty: the A1-A6 raw values were never saved and are out of scope here (see `services/regime.py`).
    """
    run = run_service.get_run_row(conn, run_id)
    data = regime_service.get_regime(
        settings.data_dir / "regime", golden_dir, settings.data_dir / "serving" / "corrected", run_id, lead_day
    )
    return RegimeResponse(run_id=run_id, evaluation_set=run.evaluation_set, **data)


@router.get("/regime/transitions", response_model=RegimeTransitionsResponse, tags=["regime"])
def regime_transitions(
    conn: Conn,
    settings: Annotated[Settings, Depends(get_settings)],
    run_id: RunIdRequired,
    district_id: DistrictQuery = None,
) -> RegimeTransitionsResponse:
    """The season's daily phase/LPS sequence and detected transitions (PRD 15 F4), built from the model's own
    lead-1 forecast chain. `district_id` is accepted but not filtered on -- see `scope: "domain"` in the response.
    `observed_imd` carries the same detector run over the IMD ground-truth labels for the same season, as a
    separate, clearly-labelled comparison; it is never used to fill in the forecast series above.
    """
    run = run_service.get_run_row(conn, run_id)
    data = regime_service.get_transitions(settings.data_dir / "regime", run_id, run.season, district_id)
    return RegimeTransitionsResponse(run_id=run_id, evaluation_set=run.evaluation_set, **data)
