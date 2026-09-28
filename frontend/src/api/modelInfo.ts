import { apiClient } from './client'
import type {
  ModelInfoResponse,
  MapMetadataResponse,
  MapGridSummary,
  HealthResponse,
} from '../types'

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const response = await apiClient.get<ModelInfoResponse>('/model-info')
  return response.data
}

export async function getMapGridSummary(): Promise<MapGridSummary> {
  const response = await apiClient.get<{ grid: MapGridSummary }>('/map/metadata')
  return response.data.grid
}

export async function getMapMetadata(): Promise<MapMetadataResponse> {
  const response = await apiClient.get<MapMetadataResponse>('/map/metadata')
  return response.data
}

export async function getHealthStatus(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health')
  return response.data
}
