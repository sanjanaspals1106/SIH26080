import { apiClient, fetchWithMockFallback } from './client'
import { MOCK_VERIFICATION, MOCK_RELIABILITY } from './mock/mockData'
import type { VerificationResponse, ReliabilityResponse, ForecastType } from '../types'

export interface VerificationParams {
  forecast_type: string
  comparison_to?: string
  evaluation_set?: string
  lead_day?: number
  threshold_mm?: number
  neighbourhood_cells?: number
  group_type?: string
  group_value?: string
}

export interface ReliabilityParams {
  forecast_type: string
  lead_day?: number
  threshold_mm?: number
}

export async function getVerification(
  params: VerificationParams
): Promise<VerificationResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<VerificationResponse>('/verification', {
        params,
      }),
    {
      ...MOCK_VERIFICATION,
      forecast_type: (params.forecast_type as ForecastType) || 'regime_aware_ml',
      lead_day: params.lead_day || 1,
      threshold_mm: params.threshold_mm || 64.5,
    }
  )
}

export async function getReliability(
  params: ReliabilityParams
): Promise<ReliabilityResponse> {
  return fetchWithMockFallback(
    () =>
      apiClient.get<ReliabilityResponse>('/verification/reliability', {
        params,
      }),
    {
      ...MOCK_RELIABILITY,
      forecast_type: (params.forecast_type as ForecastType) || 'regime_aware_ml',
      lead_day: params.lead_day || 1,
      threshold_mm: params.threshold_mm || 64.5,
    }
  )
}
