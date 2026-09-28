import type { ApiResponseMeta, EvaluationSet, AlignmentMethod } from './common'

export interface NwpRun {
  run_id: string
  source: string
  initialization_time: string
  season: number
  evaluation_set: EvaluationSet | null
  mode: 'replay'
  alignment_method: AlignmentMethod
  alignment_offset_hours: number
  imd_stamp: 'end_date' | 'start_date'
  status: string | null
  created_at?: string
  lead_days?: number[]
  first_imd_date?: string | null
  last_imd_date?: string | null
}

export interface RunsResponse extends ApiResponseMeta {
  total?: number
  limit?: number
  offset?: number
  runs: NwpRun[]
}

export interface RunDetailResponse extends ApiResponseMeta {
  run: NwpRun
}
