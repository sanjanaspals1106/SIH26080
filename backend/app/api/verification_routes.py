"""Verification report, audit trail and model info endpoints (PRD 15 F2, 16, 17-20), additive to `api/routes.py`.
Own router, included once in `main.py`. Read-only: serves stored metrics, stored forecasts and model metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import Connection, select

from backend.app.api.deps import get_golden_dir
from backend.app.config import Settings, get_settings
from backend.app.db.session import get_connection
from backend.app.db.tables import nwp_runs
from backend.app.schemas.verification_models import AuditTrailResponse, ModelInfoResponse, VerificationReport
from backend.app.services import audit as audit_service
from backend.app.services import model_info as model_info_service
from backend.app.services import verification as verification_service

router = APIRouter()
Conn = Annotated[Connection, Depends(get_connection)]


@router.get("/verification", response_model=VerificationReport, tags=["verification"])
def verification(settings: Annotated[Settings, Depends(get_settings)]) -> VerificationReport:
    """Skill of raw NWP, quantile mapping, global ML and regime-aware ML on the development seasons (out-of-fold)
    and on the locked 2025 holdout, plus probability skill and q10-q90 coverage (development only)."""
    return VerificationReport(**verification_service.get_verification(settings.data_dir / "verification"))


@router.get("/forecasts/{forecast_id}/audit", response_model=AuditTrailResponse, tags=["forecasts"])
def forecast_audit(
    forecast_id: str,
    conn: Conn,
    settings: Annotated[Settings, Depends(get_settings)],
    golden_dir: Annotated[Path, Depends(get_golden_dir)],
) -> AuditTrailResponse:
    """What the model did for one district forecast, step by step. `forecast_id` is `<run_id>_L<lead>_<district_id>`."""
    return AuditTrailResponse(
        **audit_service.get_audit(conn, forecast_id, data_dir=settings.data_dir, golden_dir=golden_dir)
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(conn: Conn, settings: Annotated[Settings, Depends(get_settings)]) -> ModelInfoResponse:
    """Models, data sources, evaluation protocol and stated limitations."""
    alignment = conn.execute(select(nwp_runs.c.alignment_method).limit(1)).scalar()
    return ModelInfoResponse(
        **model_info_service.get_model_info(
            settings.data_dir,
            alignment_method=alignment or "C1",
            lead_days=settings.lead_days,
            district_source=settings.district_source,
            census_basis=settings.district_census_basis,
        )
    )
