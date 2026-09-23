import { apiClient, fetchWithMockFallback } from './client'
import { MOCK_RUNS_RESPONSE } from './mock/mockData'
import type { RunsResponse, RunDetailResponse } from '../types'

/**
 * Fetch available replay forecast runs (PRD 18.4)
 */
export async function getRuns(): Promise<RunsResponse> {
  return fetchWithMockFallback(
    () => apiClient.get<RunsResponse>('/runs'),
    MOCK_RUNS_RESPONSE
  )
}

/**
 * Fetch a specific replay forecast run by ID (PRD 18.4)
 */
export async function getRunById(runId: string): Promise<RunDetailResponse> {
  const defaultRun =
    MOCK_RUNS_RESPONSE.runs.find((r) => r.run_id === runId) ||
    MOCK_RUNS_RESPONSE.runs[0]

  const mockDetail: RunDetailResponse = {
    _mock: true,
    mode: 'replay',
    evaluation_set: defaultRun.evaluation_set,
    run: defaultRun,
  }

  return fetchWithMockFallback(
    () => apiClient.get<RunDetailResponse>(`/runs/${runId}`),
    mockDetail
  )
}
