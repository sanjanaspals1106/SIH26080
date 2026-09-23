import type { ApiResponseMeta } from './common'

export type AttentionLevel = 'HIGH' | 'WATCH' | 'NORMAL' | 'UNAVAILABLE'
export type ProductType = 'raw_nwp' | 'quantile_mapping' | 'global_ml' | 'regime_aware_ml'

export interface QualityFlags {
  fallback_used: boolean
  fallback_reason: string | null
  ood_flag: boolean
  extrapolation_flag: boolean
}

export interface DistrictForecastSummary {
  district_id: string
  district_name: string
  state: string
  is_small: boolean
  product_type: ProductType
  raw_mean_mm: number
  corrected_mean_mm: number
  observed_mean_mm: number | null
  wettest_cell_mean_mm: number
  wettest_cell_q10_mm: number | null
  wettest_cell_q50_mm: number | null
  wettest_cell_q90_mm: number | null
  heavy_prob_max_cell: number | null
  very_heavy_prob_max_cell: number | null
  heavy_area_fraction_expected: number | null
  very_heavy_area_fraction_expected: number | null
  attention_level: AttentionLevel
  priority_rank: number
  flags: QualityFlags
}

export interface DistrictForecastsResponse extends ApiResponseMeta {
  lead_day: number
  imd_date: string
  model_version?: {
    correction: string
    heavy_rain: string
    phase: string
  }
  districts: DistrictForecastSummary[]
}

export interface PriorityTableResponse extends DistrictForecastsResponse {
  note: string // 'Model-based attention level. Not an official warning.'
}

export interface GeoJsonFeatureCollection {
  type: 'FeatureCollection'
  features: Array<{
    type: 'Feature'
    id?: string | number
    properties: {
      district_id: string
      name: string
      state: string
      source_year?: number
      [key: string]: unknown
    }
    geometry: {
      type: string
      coordinates: unknown
    }
  }>
}
