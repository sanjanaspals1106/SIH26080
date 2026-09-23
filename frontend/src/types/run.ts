import type { ApiResponseMeta, EvaluationSet, AlignmentMethod } from './common'

export interface NwpRun {
  run_id: string
  source: string
  initialization_time: string
  season: number
  evaluation_set: EvaluationSet
  mode: 'replay'
  alignment_method: AlignmentMethod
  alignment_offset_hours: number
  imd_stamp: 'end_date' | 'start_date'
  status: string
  created_at: string
}

export interface RunsResponse extends ApiResponseMeta {
  runs: NwpRun[]
}

export interface RunDetailResponse extends ApiResponseMeta {
  run: NwpRun
}
