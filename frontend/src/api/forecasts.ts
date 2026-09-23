import { apiClient, fetchWithMockFallback } from './client'
import {
  MOCK_DISTRICTS_FORECAST,
  MOCK_GRID_CELLS,
  MOCK_IMPROVEMENT_SUMMARY,
  MOCK_AUDIT_TRAIL,
  MOCK_HOTSPOTS,
  getMockGridCells,
} from './mock/mockData'
import type {
  DistrictForecastsResponse,
  GridResponse,
  GridVariable,
  ImprovementSummaryResponse,
  AuditTrailResponse,
  HotspotsResponse,
} from '../types'

export interface DistrictForecastParams {
  run_id?: string
  lead_day?: number
  state?: string
  attention_level?: string
}

export interface GridForecastParams {
  run_id?: string
  lead_day?: number
  variable: GridVariable
}

export async function getDistrictForecasts(
  params?: DistrictForecastParams
): Promise<DistrictForecastsResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<DistrictForecastsResponse>('/forecasts/districts', {
        params,
      }),
    MOCK_DISTRICTS_FORECAST
  )
}

export async function getGridForecast(
  params: GridForecastParams
): Promise<GridResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<GridResponse>('/forecasts/grid', {
        params,
      }),
    {
      ...MOCK_GRID_CELLS,
      variable: params.variable,
      lead_day: params.lead_day || 1,
      cells: getMockGridCells(params.variable, params.lead_day || 1),
    }
  )
}

export async function getImprovementSummary(params?: {
  run_id?: string
  lead_day?: number
}): Promise<ImprovementSummaryResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<ImprovementSummaryResponse>(
        '/forecasts/improvement-summary',
        { params }
      ),
    MOCK_IMPROVEMENT_SUMMARY
  )
}

export async function getAuditTrail(
  forecastId: string
): Promise<AuditTrailResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<AuditTrailResponse>(`/forecasts/${forecastId}/audit`),
    {
      ...MOCK_AUDIT_TRAIL,
      forecast_id: forecastId,
    }
  )
}

export async function getHotspots(params?: {
  run_id?: string
  lead_day?: number
  threshold_mm?: number
}): Promise<HotspotsResponse> {
  return fetchWithMockFallback(
    () => apiClient.get<HotspotsResponse>('/hotspots', { params }),
    MOCK_HOTSPOTS
  )
}
