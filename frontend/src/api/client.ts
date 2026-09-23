import axios, { type AxiosInstance } from 'axios'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
})

// Safe fetcher helper with optional mock fallback
export async function fetchWithMockFallback<T>(
  requestFn: () => Promise<{ data: T }>,
  mockFallbackData: T
): Promise<T> {
  try {
    const response = await requestFn()
    return response.data
  } catch {
    // Return mock data with _mock flag on failure or offline
    return mockFallbackData
  }
}

export default apiClient
