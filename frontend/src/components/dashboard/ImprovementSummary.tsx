import { Link } from 'react-router-dom'
import type { ImprovementSummaryResponse } from '../../types'

interface ImprovementSummaryProps {
  summary: ImprovementSummaryResponse | null
  loading?: boolean
}

export default function ImprovementSummary({
  summary,
  loading = false,
}: ImprovementSummaryProps) {
  if (loading) {
    return (
      <div
        style={{
          padding: '0.45rem 0.85rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
        }}
      >
        Calculating F1 improvement metrics…
      </div>
    )
  }

  if (!summary) return null

  const cellPct = Math.round((summary.cells_corrected_closer / summary.total_valid_cells) * 100)
  const districtPct = Math.round((summary.districts_corrected_closer / summary.total_districts) * 100)

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.75rem',
        padding: '0.5rem 0.85rem',
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      <div style={{ fontSize: '0.8rem', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
        <span
          style={{
            padding: '0.15rem 0.45rem',
            backgroundColor: 'var(--accent-glow)',
            color: 'var(--accent-cyan)',
            borderRadius: 'var(--radius-xs)',
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            fontWeight: 800,
            letterSpacing: '0.04em',
          }}
        >
          F1 METRIC
        </span>
        <span>
          AI-Corrected is closer to IMD observation in{' '}
          <strong style={{ color: 'var(--accent-cyan)' }}>
            {summary.cells_corrected_closer.toLocaleString()}
          </strong>{' '}
          of {summary.total_valid_cells.toLocaleString()} cells ({cellPct}%) and{' '}
          <strong style={{ color: 'var(--accent-cyan)' }}>
            {summary.districts_corrected_closer}
          </strong>{' '}
          of {summary.total_districts} districts ({districtPct}%).
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.72rem' }}>
        <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
          One day only, not evidence.
        </span>
        <Link
          to="/verification"
          style={{
            color: 'var(--accent-cyan)',
            fontWeight: 600,
            textDecoration: 'underline',
          }}
        >
          Verification Report →
        </Link>
      </div>
    </div>
  )
}
