import type { RegimeResponse } from '../../types'

interface LpsDetectionsCardProps {
  regime: RegimeResponse | null
}

export default function LpsDetectionsCard({ regime }: LpsDetectionsCardProps) {
  const lps = regime?.nearest_lps

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
            🌀 Low-Pressure System (LPS) Detector
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Rule-based detection using 850 hPa relative vorticity & sea-level pressure minimums
          </div>
        </div>

        <span
          style={{
            padding: '0.2rem 0.5rem',
            backgroundColor: lps?.present ? 'rgba(239, 68, 68, 0.15)' : 'var(--bg-input)',
            color: lps?.present ? '#f87171' : 'var(--text-secondary)',
            border: `1px solid ${lps?.present ? 'rgba(239, 68, 68, 0.4)' : 'var(--border-subtle)'}`,
            borderRadius: 'var(--radius-xs)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.74rem',
            fontWeight: 700,
          }}
        >
          {lps?.present ? 'LPS ACTIVE' : 'NO LPS DETECTED'}
        </span>
      </div>

      {lps?.present ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: '0.5rem',
          }}
        >
          <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Position (Lat / Lon)</div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {lps.latitude?.toFixed(2)}°N, {lps.longitude?.toFixed(2)}°E
            </div>
          </div>

          <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Distance to Domain Center</div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
              {lps.distance_km} km
            </div>
          </div>

          <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Bearing Angle</div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
              {lps.bearing_deg}° (South-West)
            </div>
          </div>

          <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Influence Index</div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)' }}>
              {lps.influence} ({lps.settings})
            </div>
          </div>
        </div>
      ) : (
        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          No organized synoptic cyclonic low-pressure circulation detected across the Indian monsoon trough domain.
        </div>
      )}
    </div>
  )
}
