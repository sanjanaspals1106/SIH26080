"""Response models for `/analogs` (PRD 15 F3, reduced vector). Field names match
`frontend/src/types/regime.ts::AnalogsResponse`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AnalogQuery(BaseModel):
    imd_date: str
    p_active: float | None
    p_break: float | None
    lps_present: bool
    lps_strength: float | None


class AnalogItem(BaseModel):
    rank: int
    analog_run_id: str
    imd_date: str
    season: int
    distance: float
    distance_percentile: float
    p_active: float | None
    p_break: float | None
    lps_present: bool
    lps_strength: float | None
    observed_mean_mm: float | None
    observed_wettest_cell_mm: float | None
    raw_mean_mm: float | None
    error_observed_minus_raw_mm: float | None


class AnalogsResponse(BaseModel):
    run_id: str
    mode: Literal["replay"] = "replay"
    evaluation_set: str | None
    lead_day: int
    district_id: str
    method: str
    query: AnalogQuery
    analogs: list[AnalogItem]
    median_error_observed_minus_raw_mm: float | None
    n_analogs: int
    library_size: int
    note: str
