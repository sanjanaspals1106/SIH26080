import type { ApiResponseMeta } from './common'

export type MonsoonPhase = 'active' | 'normal' | 'break'
export type ConfidenceBand = 'low' | 'medium' | 'high'

export interface IndicatorA1A6 {
  name: string
  code: string
  value: number
  unit: string
  training_percentile: number
  description: string
}

export interface LpsDetection {
  lps_id: number
  latitude: number
  longitude: number
  zeta_max: number
  mslp_min_hpa: number
  strength_percentile: number
}

export interface RegimeResponse extends ApiResponseMeta {
  lead_day: number
  imd_date: string
  phase: {
    active: number
    normal: number
    break: number
    confidence: number
    confidence_band: ConfidenceBand
    dominant_phase: MonsoonPhase
  }
  nearest_lps: {
    present: boolean
    latitude?: number
    longitude?: number
    distance_km?: number
    bearing_deg?: number
    influence?: number
    settings?: string
  }
  indicators: IndicatorA1A6[]
  domain_mean_raw_mm: number
  domain_mean_corrected_mm: number
  regime_available: boolean
  ood_flag: boolean
}

export interface TransitionSeriesPoint {
  imd_date: string
  p_active: number
  p_normal: number
  p_break: number
  lps_present: boolean
}

export interface TransitionEvent {
  event_id: number
  event_type: string // e.g. 'PHASE:normal->active', 'LPS_FORMS', 'LPS_ENDS', 'LPS_NEAR_DISTRICT'
  from_state?: string
  to_state?: string
  event_date: string
  confirmed: boolean
  confidence: number | null
  district_id: string | null
  domain_mean_corrected_change_mm: number
  n_dates_before: number
  n_dates_after: number
}

export interface RegimeTransitionsResponse extends ApiResponseMeta {
  series: TransitionSeriesPoint[]
  events: TransitionEvent[]
}

export interface AnalogItem {
  rank: number
  imd_date: string
  season: number
  distance: number
  distance_percentile: number
  observed_mean_mm: number
  observed_wettest_cell_mm: number
  raw_mean_mm: number
  corrected_mean_mm: number
  error_observed_minus_raw_mm: number
}

export interface AnalogsResponse extends ApiResponseMeta {
  lead_day: number
  district_id: string
  analogs: AnalogItem[]
  median_error_observed_minus_raw_mm: number
  n_analogs: number
  note: string // '5 cases only'
}
