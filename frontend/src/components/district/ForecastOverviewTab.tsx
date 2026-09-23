import type { DistrictForecastSummary } from '../../types'

interface ForecastOverviewTabProps {
  district: DistrictForecastSummary
  leadDay: number
  onSelectLead: (lead: number) => void
}

export default function ForecastOverviewTab({
  district,
  leadDay,
  onSelectLead,
}: ForecastOverviewTabProps) {
  const diff = district.corrected_mean_mm - district.raw_mean_mm
  const diffSign = diff > 0 ? `+${diff.toFixed(1)}` : diff.toFixed(1)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Lead Horizon Selector */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.6rem 0.85rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.76rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Forecast Horizon:
          </span>
          <div className="segmented-group">
            {[1, 2, 3].map((lead) => (
              <button
                key={lead}
                type="button"
                className={`segmented-btn ${leadDay === lead ? 'active' : ''}`}
                onClick={() => onSelectLead(lead)}
              >
                Lead {lead} (+{lead * 24}h)
              </button>
            ))}
          </div>
        </div>

        <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          Product: {district.product_type} {district.flags.fallback_used ? `(Fallback: ${district.flags.fallback_reason})` : ''}
        </div>
      </div>

      {/* 4-Stat Primary Metric Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '0.75rem',
        }}
      >
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
            Raw NWP Mean (ECMWF)
          </div>
          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {district.raw_mean_mm} <span style={{ fontSize: '0.8rem', fontWeight: 500 }}>mm/24h</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Baseline physics control model
          </div>
        </div>

        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--accent-cyan)',
            borderRadius: 'var(--radius-md)',
            boxShadow: '0 0 12px var(--accent-glow)',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)', textTransform: 'uppercase', marginBottom: '4px', fontWeight: 700 }}>
            AI-Corrected Mean (B3)
          </div>
          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
            {district.corrected_mean_mm} <span style={{ fontSize: '0.8rem', fontWeight: 500 }}>mm/24h</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            Regime-aware post-processed prediction
          </div>
        </div>

        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
            Correction Delta
          </div>
          <div
            style={{
              fontSize: '1.35rem',
              fontWeight: 800,
              fontFamily: 'var(--font-mono)',
              color: diff >= 0 ? 'var(--status-high-text)' : 'var(--accent-blue)',
            }}
          >
            {diffSign} <span style={{ fontSize: '0.8rem', fontWeight: 500 }}>mm</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            {diff >= 0 ? 'Model increased rainfall expectation' : 'Model reduced raw overforecast'}
          </div>
        </div>

        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
            Observed Rainfall (IMD Truth)
          </div>
          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {district.observed_mean_mm !== null ? `${district.observed_mean_mm} mm` : 'Pending / Replay'}
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
            24h accumulation ending 08:30 IST
          </div>
        </div>
      </div>

      {/* Range and Heavy Rain Probabilities Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '0.75rem',
        }}
      >
        {/* Model-Estimated Range */}
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
          <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)' }}>
            Wettest Cell & Model-Estimated Range (q10–q90)
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Wettest Cell Expected:</span>
            <span style={{ fontSize: '1.1rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
              {district.wettest_cell_mean_mm} mm
            </span>
          </div>

          <div
            style={{
              padding: '0.6rem 0.75rem',
              backgroundColor: 'var(--bg-input)',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '4px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Uncertainty Range [q10 — q90]:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)' }}>
                {district.wettest_cell_q10_mm !== null
                  ? `[${district.wettest_cell_q10_mm} mm — ${district.wettest_cell_q90_mm} mm]`
                  : 'Not available yet'}
              </strong>
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              PRD Rule H3: Model-estimated quantile spread, not a guaranteed physical interval.
            </div>
          </div>
        </div>

        {/* Heavy Rain Probability & Spatial Area Fraction */}
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
          <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)' }}>
            Heavy & Very Heavy Rain Probabilities
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>P(Rain ≥ 64.5 mm) Heavy:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                {district.heavy_prob_max_cell !== null ? `${Math.round(district.heavy_prob_max_cell * 100)}%` : '—'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>P(Rain ≥ 115.6 mm) Very Heavy:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--status-high-text)' }}>
                {district.very_heavy_prob_max_cell !== null ? `${Math.round(district.very_heavy_prob_max_cell * 100)}%` : '—'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Expected District Area Exposed (≥64.5mm):</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>
                {district.heavy_area_fraction_expected !== null ? `${Math.round(district.heavy_area_fraction_expected * 100)}%` : '—'}
              </strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
