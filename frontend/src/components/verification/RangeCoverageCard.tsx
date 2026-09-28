import type { CoverageRow } from '../../types'

export default function RangeCoverageCard({ rows, interval }: { rows: CoverageRow[]; interval: string }) {
  if (rows.length === 0) return null
  const target = rows[0].target
  const label = interval.replace('-', ' to ')

  return (
    <div
      style={{
        padding: '1rem',
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
      }}
    >
      <div>
        <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-primary)' }}>Rainfall Range Coverage</div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Share of observations falling inside the {label} range · target {Math.round(target * 100)}% · development seasons, out-of-fold
        </div>
      </div>

      {rows.map((r) => (
        <div key={r.lead_day} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>
              Lead +{r.lead_day * 24}h <span style={{ color: 'var(--text-muted)' }}>({r.n_samples.toLocaleString()} cell-days)</span>
            </span>
            <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{(r.coverage * 100).toFixed(1)}%</strong>
          </div>
          <div style={{ position: 'relative', height: '10px', backgroundColor: 'var(--bg-input)', borderRadius: '5px' }}>
            <div
              style={{
                width: `${Math.min(r.coverage, 1) * 100}%`,
                height: '100%',
                borderRadius: '5px',
                backgroundColor: r.within_tolerance ? 'var(--accent-cyan)' : 'var(--status-watch-border)',
              }}
            />
            <div
              title={`Target ${Math.round(target * 100)}%`}
              style={{ position: 'absolute', left: `${target * 100}%`, top: '-3px', height: '16px', width: '2px', backgroundColor: 'var(--text-primary)' }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
