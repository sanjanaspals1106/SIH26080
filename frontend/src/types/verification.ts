import type { ApiResponseMeta, EvaluationSet } from './common'

export type ForecastType = 'raw_nwp' | 'quantile_mapping' | 'global_ml' | 'regime_aware_ml'
export type ComparisonType = 'raw_nwp' | 'quantile_mapping' | 'global_ml'

export interface MetricEstimate {
  value: number | null
  ci_low: number | null
  ci_high: number | null
}

export interface VerificationMetrics {
  rmse: MetricEstimate
  mae?: MetricEstimate
  bias?: MetricEstimate
  pod: MetricEstimate
  far: MetricEstimate
  csi: MetricEstimate
  ets: MetricEstimate
  fss: MetricEstimate
  frequency_bias: MetricEstimate
  brier_score?: MetricEstimate
}

export interface VerificationGroup {
  type: string
  value: string
}

export interface VerificationResponse extends ApiResponseMeta {
  forecast_type: ForecastType
  comparison_to: ComparisonType
  evaluation_set: EvaluationSet
  lead_day: number
  threshold_mm: number
  neighbourhood_cells: number
  group: VerificationGroup
  metrics: VerificationMetrics
  paired_difference?: {
    [metricName: string]: MetricEstimate
  }
  n_samples: number | null
  n_events: number | null
  undefined_reason?: string | null
  bootstrap?: {
    block_days: number
    resamples: number
  }
  model_version_id?: number | null
}

export interface ReliabilityBin {
  bin_center: number
  observed_frequency: number | null
  sample_count: number
}

export interface ReliabilityResponse extends ApiResponseMeta {
  forecast_type: ForecastType
  lead_day: number
  threshold_mm: number
  bins: ReliabilityBin[]
  brier_score: number | null
  brier_skill_score: number | null
}

// ---- Verification report (GET /verification) -------------------------------------------------------------
// Point estimates read from the stored metrics files, pooled over lead days 1-3 and all valid IMD cells.

export interface ThresholdScores {
  pod: number | null
  far: number | null
  csi: number | null
  ets: number | null
  frequency_bias: number | null
  fss5: number | null
  n_obs_events: number | null
}

export interface ModelScores {
  key: string
  label: string
  description: string
  forecast_type: ForecastType
  scalars: { rmse: number | null; mae: number | null; bias: number | null }
  thresholds: Record<string, ThresholdScores>
}

export interface Highlight {
  kind: 'improved' | 'declined' | 'note'
  text: string
}

export interface HoldoutLock {
  timestamp: string | null
  git_commit: string | null
  run_count: number | null
  forced_rerun: boolean
}

export interface VerificationEvaluation {
  evaluation_set: EvaluationSet
  label: string
  seasons: number[]
  models: ModelScores[]
  corrected_key: string
  thresholds_mm: number[]
  highlights: Highlight[]
  lock: HoldoutLock | null
}

export interface ProbabilityRow {
  threshold_mm: number
  brier_uncalibrated: number
  brier_calibrated: number
  brier_skill_score: number
  climatology_rate: number
  n_events: number | null
}

export interface CoverageRow {
  lead_day: number
  coverage: number
  target: number
  tolerance: number
  n_samples: number
  within_tolerance: boolean
}

export interface VerificationReport {
  mode: 'replay'
  development: VerificationEvaluation
  holdout: VerificationEvaluation | null
  probability: { evaluation_set: 'development'; rows: ProbabilityRow[] }
  range_coverage: { evaluation_set: 'development'; interval: string; rows: CoverageRow[] }
  pooling: string
}
