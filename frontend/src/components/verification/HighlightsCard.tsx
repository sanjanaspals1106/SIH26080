import type { Highlight } from '../../types'

const DOT: Record<Highlight['kind'], string> = {
  improved: 'var(--status-normal-text)',
  declined: 'var(--status-high-text)',
  note: 'var(--accent-sky)',
}

const TAG: Record<Highlight['kind'], string> = {
  improved: 'Improved',
  declined: 'Lower than raw',
  note: 'Note',
}

export default function HighlightsCard({ highlights }: { highlights: Highlight[] }) {
  if (highlights.length === 0) return null
  return (
    <div
      style={{
        padding: '0.85rem 1rem',
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.4rem',
      }}
    >
      <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-primary)' }}>
        Corrected forecast vs raw NWP
      </div>
      {highlights.map((h) => (
        <div key={h.text} style={{ display: 'flex', gap: '0.6rem', alignItems: 'baseline', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          <span
            style={{
              flex: '0 0 auto',
              minWidth: '92px',
              fontSize: '0.66rem',
              fontWeight: 800,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              color: DOT[h.kind],
            }}
          >
            {TAG[h.kind]}
          </span>
          <span>{h.text}</span>
        </div>
      ))}
    </div>
  )
}
