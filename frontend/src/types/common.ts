export type EvaluationSet = 'development' | 'holdout'
export type RunMode = 'replay'
export type AlignmentMethod = 'C0' | 'C1'

export interface ApiResponseMeta {
  _mock?: boolean
  run_id?: string
  mode?: RunMode
  evaluation_set?: EvaluationSet
}

export interface ApiError {
  error: {
    code: string
    message: string
  }
}
