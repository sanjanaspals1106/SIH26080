import { apiClient } from './client'
import type { NwpRun, RunsResponse, RunDetailResponse } from '../types'

const RUNS_PAGE = 500 // the API caps /runs at 500 rows per page

/**
 * Fetch one page of replay forecast runs (PRD 18.4). No mock fallback: a failure is a real error.
 */
export async function getRuns(params?: { season?: number; limit?: number; offset?: number }): Promise<RunsResponse> {
  const res = await apiClient.get<RunsResponse>('/runs', { params: { limit: RUNS_PAGE, ...params } })
  return res.data
}

/**
 * Fetch every replay run (all seasons), paging through the API.
 */
export async function getAllRuns(): Promise<NwpRun[]> {
  const first = await getRuns({ offset: 0 })
  const total = first.total ?? first.runs.length
  const pages: Promise<RunsResponse>[] = []
  for (let offset = first.runs.length; offset < total; offset += RUNS_PAGE) {
    pages.push(getRuns({ offset }))
  }
  const rest = await Promise.all(pages)
  return [first, ...rest].flatMap((p) => p.runs)
}

/**
 * Fetch a specific replay forecast run by ID (PRD 18.4)
 */
export async function getRunById(runId: string): Promise<RunDetailResponse> {
  const response = await apiClient.get<RunDetailResponse>(`/runs/${runId}`)
  return response.data
}
