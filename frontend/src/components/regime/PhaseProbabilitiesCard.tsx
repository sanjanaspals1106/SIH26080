import type { RegimeResponse } from '../../types'

interface PhaseProbabilitiesCardProps {
  regime: RegimeResponse | null
  loading?: boolean
}

export default function PhaseProbabilitiesCard({
  regime,
  loading = false,
}: PhaseProbabilitiesCardProps) {
  if (loading || !regime) {
    return (
      <div
        style={{
          padding: '1.5rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          textAlign: 'center',
          color: 'var(--text-muted)',
        }}
      >
        Loading Monsoon Phase Probabilities…
      </div>
    )
  }

  const { phase } = regime
  const activePct = Math.round(phase.active * 100)
  const normalPct = Math.round(phase.normal * 100)
  const breakPct = Math.round(phase.break * 100)

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Monsoon Phase Classification (Multinomial Logistic Regression)
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            All-India domain classification based on Rajeevan et al. (2010) active/break indices
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>Dominant State:</span>
          <span
            style={{
              padding: '0.2rem 0.55rem',
              backgroundColor: 'var(--accent-glow)',
              color: 'var(--accent-cyan)',
              border: '1px solid var(--accent-cyan)',
              borderRadius: 'var(--radius-xs)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.76rem',
              fontWeight: 800,
              textTransform: 'uppercase',
            }}
          >
            {phase.dominant_phase} (Confidence: {phase.confidence_band})
          </span>
        </div>
      </div>

      {/* Probabilities Distribution Bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.25rem' }}>
        {/* Active Phase */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '3px' }}>
            <span style={{ fontWeight: 600, color: '#38bdf8' }}>ACTIVE MONSOON SPELL</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{activePct}%</span>
          </div>
          <div style={{ height: '8px', width: '100%', backgroundColor: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${activePct}%`, backgroundColor: '#38bdf8', borderRadius: '4px', transition: 'width 0.3s ease' }} />
          </div>
        </div>

        {/* Normal Phase */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '3px' }}>
            <span style={{ fontWeight: 600, color: '#34d399' }}>NORMAL MONSOON STATE</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{normalPct}%</span>
          </div>
          <div style={{ height: '8px', width: '100%', backgroundColor: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${normalPct}%`, backgroundColor: '#34d399', borderRadius: '4px', transition: 'width 0.3s ease' }} />
          </div>
        </div>

        {/* Break Phase */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '3px' }}>
            <span style={{ fontWeight: 600, color: '#fb923c' }}>BREAK MONSOON SPELL (Rain suppressed over central India)</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{breakPct}%</span>
          </div>
          <div style={{ height: '8px', width: '100%', backgroundColor: 'var(--bg-input)', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${breakPct}%`, backgroundColor: '#fb923c', borderRadius: '4px', transition: 'width 0.3s ease' }} />
          </div>
        </div>
      </div>
    </div>
  )
}
