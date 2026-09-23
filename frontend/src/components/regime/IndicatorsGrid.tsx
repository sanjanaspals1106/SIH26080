import type { IndicatorA1A6 } from '../../types'

interface IndicatorsGridProps {
  indicators: IndicatorA1A6[]
}

export default function IndicatorsGrid({ indicators }: IndicatorsGridProps) {
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
        <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Atmospheric & Climatological Indicators (A1 – A6)
        </div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Core large-scale physical drivers compared against 1981–2010 training distribution percentiles
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '0.65rem',
        }}
      >
        {indicators.map((ind) => {
          const pct = ind.training_percentile
          const pctColor =
            pct >= 75
              ? '#38bdf8'
              : pct <= 25
              ? '#fb923c'
              : '#34d399'

          return (
            <div
              key={ind.code}
              style={{
                padding: '0.65rem 0.75rem',
                backgroundColor: 'var(--bg-input)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.3rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span
                  style={{
                    fontSize: '0.72rem',
                    fontWeight: 800,
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--accent-cyan)',
                  }}
                >
                  {ind.code}
                </span>
                <span
                  style={{
                    fontSize: '0.7rem',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    color: pctColor,
                  }}
                >
                  {pct}th %ile
                </span>
              </div>

              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {ind.name}
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: '1.05rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {ind.value} <span style={{ fontSize: '0.7rem', fontWeight: 400, color: 'var(--text-muted)' }}>{ind.unit}</span>
                </span>
              </div>

              {/* Percentile visual bar */}
              <div style={{ height: '4px', width: '100%', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${pct}%`, backgroundColor: pctColor, borderRadius: '2px' }} />
              </div>

              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', lineHeight: 1.3 }}>
                {ind.description}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
