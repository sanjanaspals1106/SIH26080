"""Response models for the F1 improvement summary, the F5 priority table and the F6 hotspots (PRD 15, 18.5).
Additive to `models.py`; every number is computed from stored data at request time, none is invented."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel

from backend.app.schemas.models import DistrictForecast

PRIORITY_NOTE = "Model-based attention level. Not an official warning."  # PRD 18.5
IMPROVEMENT_NOTE = "One day only, not evidence."  # PRD 15 F1


class LatLon(BaseModel):
    lat: float
    lon: float


class HotspotOut(BaseModel):
    hotspot_id: int
    n_cells: int
    max_probability: float
    max_corrected_mean_mm: float
    centroid: LatLon
    district_ids: list[str]
    outline: dict[str, Any]  # GeoJSON Polygon / MultiPolygon, [lon, lat]


class HotspotList(BaseModel):
    mode: Literal["replay"] = "replay"
    run_id: str
    lead_day: int
    imd_date: date | None
    evaluation_set: str | None
    threshold_mm: float  # the rain threshold whose probability is used: 64.5 (15.6 if the 64.5 model is missing)
    probability_min: float  # a hotspot cell needs P(>= threshold) at least this ...
    corrected_mean_min_mm: float  # ... and a corrected mean of at least this (PRD 15 F6)
    hotspots: list[HotspotOut]


class ImprovementSummaryOut(BaseModel):
    mode: Literal["replay"] = "replay"
    run_id: str
    lead_day: int
    imd_date: date | None
    evaluation_set: str | None
    total_valid_cells: int  # cells with a raw, a corrected and an observed value
    cells_corrected_closer: int
    cells_raw_closer: int
    cells_no_change: int
    total_districts: int  # districts with a raw, a corrected and an observed mean
    districts_corrected_closer: int
    districts_raw_closer: int
    note: str = IMPROVEMENT_NOTE


class PriorityTableOut(BaseModel):
    mode: Literal["replay"] = "replay"
    run_id: str
    lead_day: int
    imd_date: date | None
    evaluation_set: str | None
    total: int
    note: str = PRIORITY_NOTE
    districts: list[DistrictForecast]
