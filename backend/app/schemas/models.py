"""Response models. Fields that a later pipeline stage fills (corrected forecast, probabilities, attention level,
phase, ...) are present and `null` until that stage has written them: the API never invents them."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel


class Page(BaseModel):
    total: int
    limit: int
    offset: int


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    mode: Literal["replay"] = "replay"
    version: str
    database: Literal["ok", "unavailable", "not_configured"]


# ---- runs ---------------------------------------------------------------------------------------


class Run(BaseModel):
    run_id: str
    source: str | None
    initialization_time: datetime
    season: int | None
    evaluation_set: str | None  # development | holdout, assigned by the protocol (M4); null at M1
    mode: Literal["replay"] = "replay"
    alignment_method: str | None
    alignment_offset_hours: int | None
    imd_stamp: str | None
    status: str | None
    lead_days: list[int]
    first_imd_date: date | None
    last_imd_date: date | None
    n_districts: int | None = None  # only in the detail answer


class RunList(Page):
    mode: Literal["replay"] = "replay"
    runs: list[Run]


# ---- district forecasts -------------------------------------------------------------------------


class Flags(BaseModel):
    fallback_used: bool | None
    fallback_reason: str | None


class DistrictForecast(BaseModel):
    run_id: str
    lead_day: int
    imd_date: date | None
    district_id: str
    district_name: str
    state: str
    is_small: bool | None
    product_type: str | None
    raw_mean_mm: float | None
    corrected_mean_mm: float | None
    observed_mean_mm: float | None
    wettest_cell_id: int | None
    wettest_cell_mean_mm: float | None
    wettest_cell_q10_mm: float | None
    wettest_cell_q50_mm: float | None
    wettest_cell_q90_mm: float | None
    heavy_prob_max_cell: float | None
    very_heavy_prob_max_cell: float | None
    heavy_area_fraction_expected: float | None
    very_heavy_area_fraction_expected: float | None
    attention_level: str | None
    priority_rank: int | None
    flags: Flags


class DistrictForecastList(Page):
    mode: Literal["replay"] = "replay"
    run_id: str | None = None
    evaluation_set: str | None = None
    model_version: dict[str, str | None] | None = None
    districts: list[DistrictForecast]


class DistrictInfo(BaseModel):
    district_id: str
    name: str
    state: str
    is_small: bool | None
    n_effective_cells: float | None
    centroid_lat: float | None
    centroid_lon: float | None
    source_year: int | None


class DistrictForecastDetail(Page):
    mode: Literal["replay"] = "replay"
    district: DistrictInfo
    forecasts: list[DistrictForecast]


# ---- grid ---------------------------------------------------------------------------------------


class GridCell(BaseModel):
    cell_id: int
    latitude: float
    longitude: float
    value: float | None


class GridLayer(Page):
    mode: Literal["replay"] = "replay"
    run_id: str
    lead_day: int
    imd_date: date | None
    evaluation_set: str | None
    variable: str
    unit: str
    cells: list[GridCell]


# ---- map ----------------------------------------------------------------------------------------


class MapMetadata(BaseModel):
    mode: Literal["replay"] = "replay"
    grid: dict[str, Any]
    layers: list[dict[str, Any]]
    lead_days: list[int]
    thresholds_mm: list[float]
    runs: dict[str, Any]
    imd_dates: dict[str, date | None]
    districts: dict[str, Any]


class DistrictFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    district_file: str
    census_basis_year: int
    features: list[dict[str, Any]]
