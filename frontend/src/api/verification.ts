import { apiClient } from './client'
import type { VerificationReport } from '../types'

export async function getVerificationReport(): Promise<VerificationReport> {
  const response = await apiClient.get<VerificationReport>('/verification')
  return response.data
}
