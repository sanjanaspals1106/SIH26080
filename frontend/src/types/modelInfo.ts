import type { ApiResponseMeta } from './common'

export interface ModelRecord {
  role: string
  version: string
  algorithm: string
  feature_set_version: string
  n_features: number
  git_commit?: string
  training_seasons?: number[]
}

export interface DataSourceInfo {
  nwp: string
  truth: string
  alignment_method: string
  lead_days: number[]
  season_scope: string
  district_file: string
}

export interface ProtocolStatus {
  n_seasons: number | null
  development_seasons: number[]
  holdout_seasons: number[]
  holdout_locked: boolean
  status: string // e.g. 'DEVELOPMENT ONLY' or 'LOCKED EVALUATION'
}

export interface ModelInfoResponse extends ApiResponseMeta {
  models: ModelRecord[]
  data: DataSourceInfo
  protocol: ProtocolStatus
  limitations: string[]
  fallback_share_recent: number | null
}

export interface MapMetadataResponse {
  grid_bounds: {
    north: number
    south: number
    west: number
    east: number
    step_deg: number
    n_lat: number
    n_lon: number
    total_valid_cells: number
  }
  available_leads: number[]
  thresholds_mm: number[]
  district_source: string
  district_count_2011: number
  layers: string[]
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error'
  version: string
  database?: string
  cache?: string
}
