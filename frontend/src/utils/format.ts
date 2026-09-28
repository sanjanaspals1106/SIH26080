/** Display rounding for rainfall values (mm). Keeps the API values untouched; only what is shown is rounded. */
export function fmtMm(v: number | null | undefined, dp = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const f = 10 ** dp
  return (Math.round(v * f) / f).toFixed(dp)
}
