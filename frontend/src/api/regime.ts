import { apiClient } from './client'
import type {
  RegimeResponse,
  RegimeTransitionsResponse,
  AnalogsResponse,
} from '../types'

export async function getRegime(params?: {
  run_id?: string
  lead_day?: number
}): Promise<RegimeResponse> {
  const response = await apiClient.get<RegimeResponse>('/regime', { params })
  return response.data
}

export async function getRegimeTransitions(params?: {
  run_id?: string
  district_id?: string
}): Promise<RegimeTransitionsResponse> {
  const response = await apiClient.get<RegimeTransitionsResponse>('/regime/transitions', { params })
  return response.data
}

// Real data only: a district or run with no analogs (404) or a failed call gives null, never mock data.
export async function getHistoricalAnalogs(params?: {
  run_id?: string
  lead_day?: number
  district_id?: string
}): Promise<AnalogsResponse | null> {
  try {
    const response = await apiClient.get<AnalogsResponse>('/analogs', { params })
    return response.data
  } catch {
    return null
  }
}

/** Same endpoint as `getRegime` but never substitutes mock data: callers hide the element on failure. */
export async function getRegimeStrict(params: {
  run_id: string
  lead_day: number
}): Promise<RegimeResponse> {
  const res = await apiClient.get<RegimeResponse>('/regime', { params })
  return res.data
}
