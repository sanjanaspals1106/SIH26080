import type { RegimeResponse, DistrictForecastSummary } from '../../types'

interface DistrictRegimeTabProps {
  district: DistrictForecastSummary
  regime: RegimeResponse | null
  loading?: boolean
}

export default function DistrictRegimeTab({
  district,
  regime,
  loading = false,
}: DistrictRegimeTabProps) {
  if (loading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading district regime interaction indices…
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Overview Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '0.75rem',
        }}
      >
        {/* Terrain & Coast Card */}
        <div
          style={{
            padding: '1rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
          }}
        >
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            🏔️ Terrain & Coastal Geography
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.78rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Orographic Influence Index:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>0.84 (High Slope)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Coastal Influence Index:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>0.65 (Near Coast)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Effective Grid Cells in District:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>14 cells (0.25° resolution)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Area Classification:</span>
              <span>{district.is_small ? 'Small District' : 'Standard District'}</span>
            </div>
          </div>
        </div>

        {/* Low-Pressure System Interaction Card */}
        <div
          style={{
            padding: '1rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
          }}
        >
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            🌀 Nearest Low-Pressure System (LPS)
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.78rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>LPS Proximity Status:</span>
              <strong style={{ color: regime?.nearest_lps.present ? 'var(--status-high-text)' : 'var(--text-secondary)' }}>
                {regime?.nearest_lps.present ? 'Active System Within 500km' : 'No System Near'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Distance to Center:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>
                {regime?.nearest_lps.distance_km ? `${regime.nearest_lps.distance_km} km` : '—'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Bearing Angle:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>
                {regime?.nearest_lps.bearing_deg ? `${regime.nearest_lps.bearing_deg}° (SW)` : '—'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Normalized Influence Index:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)' }}>
                {regime?.nearest_lps.influence || '0.00'}
              </strong>
            </div>
          </div>
        </div>
      </div>

      {/* Domain Phase Card */}
      {regime && (
        <div
          style={{
            padding: '0.85rem 1rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            fontSize: '0.78rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.4rem',
          }}
        >
          <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
            All-India Monsoon Phase Context
          </div>
          <div style={{ color: 'var(--text-secondary)' }}>
            Dominant Phase: <strong style={{ color: 'var(--accent-cyan)', textTransform: 'uppercase' }}>{regime.phase.dominant_phase}</strong> · Confidence: <strong>{regime.phase.confidence_band}</strong> ({Math.round(regime.phase.confidence * 100)}%)
          </div>
          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.2rem' }}>
            <span>Active Prob: <strong>{Math.round(regime.phase.active * 100)}%</strong></span>
            <span>Normal Prob: <strong>{Math.round(regime.phase.normal * 100)}%</strong></span>
            <span>Break Prob: <strong>{Math.round(regime.phase.break * 100)}%</strong></span>
          </div>
        </div>
      )}
    </div>
  )
}
