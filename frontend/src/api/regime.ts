import { apiClient, fetchWithMockFallback } from './client'
import {
  MOCK_REGIME,
  MOCK_TRANSITIONS,
  MOCK_ANALOGS,
} from './mock/mockData'
import type {
  RegimeResponse,
  RegimeTransitionsResponse,
  AnalogsResponse,
} from '../types'

export async function getRegime(params?: {
  run_id?: string
  lead_day?: number
}): Promise<RegimeResponse> {
  return fetchWithMockFallback(
    () => apiClient.get<RegimeResponse>('/regime', { params }),
    MOCK_REGIME
  )
}

export async function getRegimeTransitions(params?: {
  run_id?: string
  district_id?: string
}): Promise<RegimeTransitionsResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<RegimeTransitionsResponse>('/regime/transitions', {
        params,
      }),
    MOCK_TRANSITIONS
  )
}

export async function getHistoricalAnalogs(params?: {
  run_id?: string
  lead_day?: number
  district_id?: string
}): Promise<AnalogsResponse> {
  return fetchWithMockFallback(
    () => apiClient.get<AnalogsResponse>('/analogs', { params }),
    MOCK_ANALOGS
  )
}
