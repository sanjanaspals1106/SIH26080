import { apiClient } from './client'
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
  // All districts in one page (the API default is 100, which silently hid 537 of the 637 districts).
  const res = await apiClient.get<DistrictForecastsResponse>('/forecasts/districts', {
    params: { limit: 1000, ...params },
  })
  return res.data
}

export async function getGridForecast(
  params: GridForecastParams
): Promise<GridResponse> {
  const res = await apiClient.get<GridResponse>('/forecasts/grid', { params })
  // The API returns cell_id/latitude/longitude/value; GridCellData (and every map component) expects
  // cell_id/lat/lon/value.
  return {
    ...res.data,
    cells: res.data.cells.map((c: any) => ({
      cell_id: c.cell_id,
      lat: c.lat ?? c.latitude,
      lon: c.lon ?? c.longitude,
      value: c.value,
    })),
  }
}

export async function getImprovementSummary(params?: {
  run_id?: string
  lead_day?: number
}): Promise<ImprovementSummaryResponse> {
  try {
    const res = await apiClient.get<ImprovementSummaryResponse>('/forecasts/improvement-summary', { params })
    return res.data
  } catch (err) {
    // A secondary banner: if it cannot be computed, show nothing (ImprovementSummary hides itself when
    // total_valid_cells is 0) rather than break the dashboard or show a made-up number.
    console.error('improvement summary unavailable', err)
    return {
      lead_day: params?.lead_day ?? 1,
      imd_date: '',
      total_valid_cells: 0,
      cells_corrected_closer: 0,
      cells_raw_closer: 0,
      cells_no_change: 0,
      total_districts: 0,
      districts_corrected_closer: 0,
      districts_raw_closer: 0,
      note: 'One day only, not evidence.',
    }
  }
}

export async function getAuditTrail(
  forecastId: string
): Promise<AuditTrailResponse> {
  const res = await apiClient.get<AuditTrailResponse>(`/forecasts/${forecastId}/audit`)
  return res.data
}

export async function getHotspots(params?: {
  run_id?: string
  lead_day?: number
  threshold_mm?: number
}): Promise<HotspotsResponse> {
  try {
    const res = await apiClient.get<HotspotsResponse>('/hotspots', { params })
    return res.data
  } catch (err) {
    // The hotspot layer is an overlay: on failure draw none (the map and table keep working).
    console.error('hotspots unavailable', err)
    return { lead_day: params?.lead_day ?? 1, threshold_mm: 64.5, hotspots: [] }
  }
}
