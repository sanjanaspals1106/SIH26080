"""Response models for the regime endpoints (`/regime`, `/regime/transitions`). Additive: does not change
`schemas/models.py`. Field names match `frontend/src/types/regime.ts` exactly (checked by hand, not generated).

`indicators` is always `[]`: the raw A1-A6 atmospheric values were never persisted (only the derived regime
features were), and recomputing them means reopening a per-season NetCDF file per request. Out of scope here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# `phase` (below) is a plain dict, not a BaseModel: its keys are active/normal/break/confidence/confidence_band/
# dominant_phase, and `break` is a reserved word in Python -- a dict sidesteps the alias plumbing entirely.


class NearestLps(BaseModel):
    present: bool
    latitude: float | None = None
    longitude: float | None = None
    distance_km: float | None = None
    bearing_deg: float | None = None
    influence: float | None = None
    settings: str | None = None


class RegimeResponse(BaseModel):
    run_id: str
    mode: Literal["replay"] = "replay"
    evaluation_set: str | None
    lead_day: int
    imd_date: str
    phase: dict
    nearest_lps: NearestLps
    indicators: list[dict] = []
    domain_mean_raw_mm: float | None
    domain_mean_corrected_mm: float | None
    regime_available: bool
    ood_flag: bool
    note: str | None = None  # honest caveats: e.g. "regime_available is false for this run/lead"


class TransitionSeriesPoint(BaseModel):
    imd_date: str
    p_active: float
    p_normal: float
    p_break: float
    lps_present: bool


class TransitionEvent(BaseModel):
    event_id: int
    event_type: str
    from_state: str | None
    to_state: str | None
    event_date: str
    confirmed: bool
    confidence: float | None
    district_id: str | None
    domain_mean_corrected_change_mm: float | None
    n_dates_before: int
    n_dates_after: int


class TransitionSet(BaseModel):
    series: list[TransitionSeriesPoint]
    events: list[TransitionEvent]


class RegimeTransitionsResponse(BaseModel):
    run_id: str
    mode: Literal["replay"] = "replay"
    evaluation_set: str | None
    series: list[TransitionSeriesPoint]  # the model-forecast chain (primary; PRD F4)
    events: list[TransitionEvent]
    observed_imd: TransitionSet  # IMD ground-truth transitions, same season window, clearly separate
    scope: str  # "domain" (all endpoints here are domain-level; district_id is accepted but not filtered on)
    district_id: str | None = None
