import type { ApiResponseMeta, AlignmentMethod } from './common'

export type GridVariable =
  | 'raw'
  | 'corrected'
  | 'difference'
  | 'observed'
  | 'improvement'
  | 'q10'
  | 'q50'
  | 'q90'
  | 'p_ge_15_6'
  | 'p_ge_64_5'
  | 'p_ge_115_6'
  | 'lps_influence'
  | 'orographic_influence'
  | 'coastal_influence'

export interface GridCellData {
  cell_id: number
  lat: number
  lon: number
  value: number | null
}

export interface GridResponse extends ApiResponseMeta {
  lead_day: number
  variable: GridVariable
  threshold_mm?: number
  cells: GridCellData[]
}

export interface ImprovementSummaryResponse extends ApiResponseMeta {
  lead_day: number
  imd_date: string
  total_valid_cells: number
  cells_corrected_closer: number
  cells_raw_closer: number
  cells_no_change: number
  total_districts: number
  districts_corrected_closer: number
  districts_raw_closer: number
  note: string // 'One day only, not evidence.'
}

export interface AuditTrailSteps {
  raw: {
    district_mean_mm: number
    wettest_cell_mm: number
  }
  regime: {
    phase: {
      active: number
      normal: number
      break: number
      confidence_band: 'low' | 'medium' | 'high'
    }
    nearest_lps: {
      distance_km: number
      bearing_deg: number
      influence: number
      settings: string
    }
    orographic_influence: number
    coastal_influence: number
  }
  history: {
    phase: string
    lps_near: boolean
    n_dates: number
    median_diff_mm: number
    q25_diff_mm: number
    q75_diff_mm: number
    note?: string
  }
  correction: {
    district_mean_mm: number
    wettest_cell_mm: number
  }
  corrected: {
    district_mean_mm: number
    wettest_cell_mm: number
  }
  confidence: {
    heavy_prob_max_cell: number | null
    very_heavy_prob_max_cell: number | null
    range_wettest_cell_mm: {
      q10: number | null
      q50: number | null
      q90: number | null
    }
    measured_coverage_q10_q90: number | null
  }
  record: {
    model_version: string
    feature_set_version: string
    alignment_method: AlignmentMethod
    fallback_used: boolean
  }
}

export interface AuditTrailResponse extends ApiResponseMeta {
  forecast_id: string
  steps: AuditTrailSteps
  summary: string
}

export interface Hotspot {
  hotspot_id: number
  n_cells: number
  max_probability: number
  max_corrected_mean_mm: number
  centroid: {
    lat: number
    lon: number
  }
  district_ids: string[]
  outline: {
    type: string
    coordinates: unknown
  }
}

export interface HotspotsResponse extends ApiResponseMeta {
  lead_day: number
  threshold_mm: number
  hotspots: Hotspot[]
}
