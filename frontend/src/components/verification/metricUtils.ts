import type { ModelScores } from '../../types'

export type Better = 'lower' | 'higher' | 'one' | 'zero'

export interface MetricDef {
  key: string
  label: string
  unit?: string
  digits: number
  better: Better
  hint: string
  value: (model: ModelScores, threshold: string) => number | null
}

export const MODEL_COLORS: Record<string, string> = {
  raw_nwp: '#94a3b8',
  quantile_mapping: '#38bdf8',
  global_ml: '#6366f1',
  regime_aware_ml: '#00b4d8',
}

export const THRESHOLD_LABELS: Record<string, string> = {
  '15.6': 'Moderate',
  '64.5': 'Heavy rain',
  '115.6': 'Very heavy',
}

export const METRICS: MetricDef[] = [
  { key: 'rmse', label: 'RMSE', unit: 'mm', digits: 2, better: 'lower', hint: 'Root-mean-square error over all cells and days. Lower is better.', value: (m) => m.scalars.rmse },
  { key: 'mae', label: 'MAE', unit: 'mm', digits: 2, better: 'lower', hint: 'Mean absolute error. Lower is better.', value: (m) => m.scalars.mae },
  { key: 'bias', label: 'Bias', unit: 'mm', digits: 2, better: 'zero', hint: 'Mean forecast minus observed. Closer to zero is better.', value: (m) => m.scalars.bias },
  { key: 'pod', label: 'POD', digits: 3, better: 'higher', hint: 'Probability of detection at the selected threshold. Higher is better.', value: (m, t) => m.thresholds[t]?.pod ?? null },
  { key: 'far', label: 'FAR', digits: 3, better: 'lower', hint: 'False alarm ratio at the selected threshold. Lower is better.', value: (m, t) => m.thresholds[t]?.far ?? null },
  { key: 'csi', label: 'CSI', digits: 3, better: 'higher', hint: 'Critical success index at the selected threshold. Higher is better.', value: (m, t) => m.thresholds[t]?.csi ?? null },
  { key: 'ets', label: 'ETS', digits: 3, better: 'higher', hint: 'Equitable threat score at the selected threshold. Higher is better.', value: (m, t) => m.thresholds[t]?.ets ?? null },
  { key: 'fss5', label: 'FSS 5×5', digits: 3, better: 'higher', hint: 'Fractions skill score on a 5×5-cell (about 140 km) neighbourhood. Higher is better.', value: (m, t) => m.thresholds[t]?.fss5 ?? null },
  { key: 'frequency_bias', label: 'Freq. bias', digits: 2, better: 'one', hint: 'Forecast events divided by observed events. 1.00 is unbiased; below 1 means fewer events forecast than observed.', value: (m, t) => m.thresholds[t]?.frequency_bias ?? null },
]

/** Distance from the ideal value: smaller is better for every metric kind. */
export function badness(better: Better, value: number): number {
  if (better === 'lower') return value
  if (better === 'higher') return -value
  if (better === 'one') return Math.abs(value - 1)
  return Math.abs(value)
}

export function fmt(value: number | null | undefined, digits: number): string {
  return value == null ? '—' : value.toFixed(digits)
}
