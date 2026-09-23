import type { GridVariable } from '../types'

export interface ColorBin {
  label: string
  color: string
  textColor?: string
}

export const RAIN_BINS: ColorBin[] = [
  { label: '< 1 mm', color: '#e2e8f0', textColor: '#0f172a' },
  { label: '1 – 15.6 mm', color: '#7dd3fc', textColor: '#0f172a' },
  { label: '15.6 – 64.5 mm', color: '#0284c7', textColor: '#ffffff' },
  { label: '64.5 – 115.6 mm', color: '#1e3a8a', textColor: '#ffffff' },
  { label: '> 115.6 mm', color: '#6b21a8', textColor: '#ffffff' },
]

export const DIFFERENCE_BINS: ColorBin[] = [
  { label: '< -10 mm (Lower)', color: '#2563eb', textColor: '#ffffff' },
  { label: '-10 to -2 mm', color: '#93c5fd', textColor: '#0f172a' },
  { label: '-2 to +2 mm', color: '#e2e8f0', textColor: '#0f172a' },
  { label: '+2 to +10 mm', color: '#fca5a5', textColor: '#0f172a' },
  { label: '> +10 mm (Higher)', color: '#dc2626', textColor: '#ffffff' },
]

export const IMPROVEMENT_BINS: ColorBin[] = [
  { label: 'Corrected closer', color: '#10b981', textColor: '#ffffff' },
  { label: 'No significant change', color: '#94a3b8', textColor: '#ffffff' },
  { label: 'Raw closer', color: '#f43f5e', textColor: '#ffffff' },
]

export function getGridCellColor(
  variable: GridVariable,
  val: number | null
): { color: string; opacity: number } {
  if (val === null || isNaN(val)) {
    return { color: '#64748b', opacity: 0.1 }
  }

  if (variable === 'raw' || variable === 'corrected' || variable === 'observed') {
    if (val < 1.0) return { color: '#e2e8f0', opacity: 0.25 }
    if (val < 15.6) return { color: '#7dd3fc', opacity: 0.65 }
    if (val < 64.5) return { color: '#0284c7', opacity: 0.75 }
    if (val < 115.6) return { color: '#1e3a8a', opacity: 0.85 }
    return { color: '#6b21a8', opacity: 0.9 }
  }

  if (variable === 'difference') {
    if (val <= -10) return { color: '#2563eb', opacity: 0.85 }
    if (val < -2) return { color: '#93c5fd', opacity: 0.7 }
    if (val <= 2) return { color: '#e2e8f0', opacity: 0.25 }
    if (val < 10) return { color: '#fca5a5', opacity: 0.7 }
    return { color: '#dc2626', opacity: 0.85 }
  }

  if (variable === 'improvement') {
    if (val > 1.0) return { color: '#10b981', opacity: 0.8 }
    if (val < -1.0) return { color: '#f43f5e', opacity: 0.8 }
    return { color: '#94a3b8', opacity: 0.3 }
  }

  // Generic fallback
  return { color: '#00b4d8', opacity: 0.5 }
}
