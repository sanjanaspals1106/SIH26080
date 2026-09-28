"""Response models for `/verification`, `/forecasts/{forecast_id}/audit` and `/model-info`. Additive: does not
change `schemas/models.py`. Field names match `frontend/src/types/{verification,forecast,modelInfo}.ts` (checked by
hand, not generated).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ThresholdScores(BaseModel):
    pod: float | None = None
    far: float | None = None
    csi: float | None = None
    ets: float | None = None
    frequency_bias: float | None = None
    fss5: float | None = None
    n_obs_events: int | None = None


class ModelScores(BaseModel):
    key: str
    label: str
    description: str
    forecast_type: str
    scalars: dict[str, float | None]
    thresholds: dict[str, ThresholdScores]


class Highlight(BaseModel):
    kind: Literal["improved", "declined", "note"]
    text: str


class HoldoutLock(BaseModel):
    timestamp: str | None = None
    git_commit: str | None = None
    run_count: int | None = None
    forced_rerun: bool = False


class Evaluation(BaseModel):
    evaluation_set: Literal["development", "holdout"]
    label: str
    seasons: list[int]
    models: list[ModelScores]
    corrected_key: str
    thresholds_mm: list[float]
    highlights: list[Highlight]
    lock: HoldoutLock | None = None


class ProbabilityRow(BaseModel):
    threshold_mm: float
    brier_uncalibrated: float
    brier_calibrated: float
    brier_skill_score: float
    climatology_rate: float
    n_events: int | None = None


class ProbabilityBlock(BaseModel):
    evaluation_set: Literal["development"]
    rows: list[ProbabilityRow]


class CoverageRow(BaseModel):
    lead_day: int
    coverage: float
    target: float
    tolerance: float
    n_samples: int
    within_tolerance: bool


class CoverageBlock(BaseModel):
    evaluation_set: Literal["development"]
    interval: str
    rows: list[CoverageRow]


class VerificationReport(BaseModel):
    mode: Literal["replay"] = "replay"
    development: Evaluation
    holdout: Evaluation | None
    probability: ProbabilityBlock
    range_coverage: CoverageBlock
    pooling: str


# ---- audit trail ---------------------------------------------------------------------------------------------


class AuditRaw(BaseModel):
    district_mean_mm: float | None
    wettest_cell_mm: float | None


class AuditPhase(BaseModel):
    active: float
    normal: float
    # `break` is a Python keyword, so it is set through the alias.
    break_: float = Field(alias="break")
    confidence_band: str

    model_config = ConfigDict(populate_by_name=True)


class AuditLps(BaseModel):
    present: bool
    settings: str | None = None
    distance_km: float | None = None
    bearing_deg: float | None = None
    influence: float | None = None


class AuditRegime(BaseModel):
    phase: AuditPhase
    nearest_lps: AuditLps
    orographic_influence: float | None
    coastal_influence: float | None
    regime_available: bool
    ood_flag: bool
    note: str | None = None


class AuditHistory(BaseModel):
    phase: str
    lps_near: bool
    n_dates: int
    median_diff_mm: float | None
    q25_diff_mm: float | None
    q75_diff_mm: float | None
    note: str | None = None


class AuditRange(BaseModel):
    q10: float | None
    q50: float | None
    q90: float | None


class AuditConfidence(BaseModel):
    heavy_prob_max_cell: float | None
    very_heavy_prob_max_cell: float | None
    range_wettest_cell_mm: AuditRange
    measured_coverage_q10_q90: float | None


class AuditRecord(BaseModel):
    model_version: str
    feature_set_version: str
    alignment_method: str
    fallback_used: bool
    product_type: str
    fallback_reason: str | None = None


class AuditSteps(BaseModel):
    raw: AuditRaw
    regime: AuditRegime | None
    history: AuditHistory | None
    correction: AuditRaw
    corrected: AuditRaw
    confidence: AuditConfidence
    record: AuditRecord


class AuditTrailResponse(BaseModel):
    mode: Literal["replay"] = "replay"
    forecast_id: str
    run_id: str
    evaluation_set: str | None
    lead_day: int
    imd_date: str | None
    district_id: str
    district_name: str
    steps: AuditSteps
    summary: str


# ---- model info ----------------------------------------------------------------------------------------------


class ModelInfoModel(BaseModel):
    role: str
    version: str
    algorithm: str
    feature_set_version: str
    n_features: int
    git_commit: str | None = None
    training_seasons: list[int] | None = None


class ModelInfoData(BaseModel):
    nwp: str
    truth: str
    alignment_method: str
    lead_days: list[int]
    season_scope: str
    district_file: str


class ModelInfoProtocol(BaseModel):
    n_seasons: int | None
    development_seasons: list[int]
    holdout_seasons: list[int]
    holdout_locked: bool
    status: str


class ModelInfoResponse(BaseModel):
    mode: Literal["replay"] = "replay"
    models: list[ModelInfoModel]
    data: ModelInfoData
    protocol: ModelInfoProtocol
    limitations: list[str]
    fallback_share_recent: float | None = None
