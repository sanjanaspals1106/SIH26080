import { apiClient, fetchWithMockFallback } from './client'
import {
  MOCK_PRIORITY_TABLE,
  MOCK_DISTRICTS_FORECAST,
  MOCK_DISTRICTS_GEOJSON,
} from './mock/mockData'
import type {
  PriorityTableResponse,
  DistrictForecastSummary,
  GeoJsonFeatureCollection,
} from '../types'

export interface PriorityDistrictsParams {
  run_id?: string
  lead_day?: number
  state?: string
}

export async function getPriorityDistricts(
  params?: PriorityDistrictsParams
): Promise<PriorityTableResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<PriorityTableResponse>('/districts/priority', {
        params,
      }),
    MOCK_PRIORITY_TABLE
  )
}

export async function getDistrictDetail(
  districtId: string,
  params?: { run_id?: string; lead_day?: number }
): Promise<DistrictForecastSummary | null> {
  try {
    const response = await apiClient.get<DistrictForecastSummary>(
      `/forecasts/districts/${districtId}`,
      { params }
    )
    return response.data
  } catch {
    const found = MOCK_DISTRICTS_FORECAST.districts.find(
      (d) => d.district_id === districtId
    )
    return found || MOCK_DISTRICTS_FORECAST.districts[0]
  }
}

export async function getDistrictGeoJSON(): Promise<GeoJsonFeatureCollection> {
  return fetchWithMockFallback(
    () => apiClient.get<GeoJsonFeatureCollection>('/map/districts'),
    MOCK_DISTRICTS_GEOJSON
  )
}
