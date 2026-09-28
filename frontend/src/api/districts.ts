import { apiClient } from './client'
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
  const res = await apiClient.get<PriorityTableResponse>('/districts/priority', { params })
  return res.data
}

export async function getDistrictDetail(
  districtId: string,
  params?: { run_id?: string; lead_day?: number }
): Promise<DistrictForecastSummary | null> {
  // Never substitute another district: a missing district is reported as null.
  try {
    const response = await apiClient.get<{ forecasts: DistrictForecastSummary[] }>(
      `/forecasts/districts/${districtId}`,
      { params }
    )
    return response.data.forecasts[0] ?? null
  } catch {
    return null
  }
}

export async function getDistrictGeoJSON(): Promise<GeoJsonFeatureCollection> {
  const res = await apiClient.get<GeoJsonFeatureCollection>('/map/districts')
  return res.data
}
