import { apiClient, fetchWithMockFallback } from './client'
import {
  MOCK_MODEL_INFO,
  MOCK_MAP_METADATA,
  MOCK_HEALTH,
} from './mock/mockData'
import type {
  ModelInfoResponse,
  MapMetadataResponse,
  HealthResponse,
} from '../types'

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return fetchWithMockFallback(
    () => apiClient.get<ModelInfoResponse>('/model-info'),
    MOCK_MODEL_INFO
  )
}

export async function getMapMetadata(): Promise<MapMetadataResponse> {
  return fetchWithMockFallback(
    () => apiClient.get<MapMetadataResponse>('/map/metadata'),
    MOCK_MAP_METADATA
  )
}

export async function getHealthStatus(): Promise<HealthResponse> {
  return fetchWithMockFallback(
    () => apiClient.get<HealthResponse>('/health'),
    MOCK_HEALTH
  )
}
