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
