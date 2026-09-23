import { Link } from 'react-router-dom'
import type { DistrictForecastSummary } from '../../types'

interface DistrictContextPanelProps {
  district: DistrictForecastSummary | null
  onClose?: () => void
}

export default function DistrictContextPanel({
  district,
  onClose,
}: DistrictContextPanelProps) {
  if (!district) {
    return (
      <div
        style={{
          width: '310px',
          minWidth: '310px',
          padding: '1.25rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.82rem',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ fontSize: '1.8rem', marginBottom: '0.4rem', opacity: 0.6 }}>📍</div>
        <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
          District Forecast Detail
        </div>
        <div style={{ lineHeight: 1.4 }}>
          Click any district boundary on the map to inspect localized post-processed predictions.
        </div>
      </div>
    )
  }

  const diff = district.corrected_mean_mm - district.raw_mean_mm
  const diffSign = diff > 0 ? `+${diff.toFixed(1)}` : diff.toFixed(1)

  const attentionClass =
    district.attention_level === 'HIGH'
      ? 'var(--status-high-bg)'
      : district.attention_level === 'WATCH'
      ? 'var(--status-watch-bg)'
      : district.attention_level === 'NORMAL'
      ? 'var(--status-normal-bg)'
      : 'var(--status-unavailable-bg)'

  const attentionColor =
    district.attention_level === 'HIGH'
      ? 'var(--status-high-text)'
      : district.attention_level === 'WATCH'
      ? 'var(--status-watch-text)'
      : district.attention_level === 'NORMAL'
      ? 'var(--status-normal-text)'
      : 'var(--status-unavailable-text)'

  const attentionBorder =
    district.attention_level === 'HIGH'
      ? 'var(--status-high-border)'
      : district.attention_level === 'WATCH'
      ? 'var(--status-watch-border)'
      : district.attention_level === 'NORMAL'
      ? 'var(--status-normal-border)'
      : 'var(--status-unavailable-border)'

  return (
    <div
      style={{
        width: '310px',
        minWidth: '310px',
        padding: '1rem 1.15rem',
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.85rem',
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
            {district.district_name}
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
            {district.state} {district.is_small ? '· Small area' : ''}
          </div>
        </div>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              fontSize: '0.9rem',
              padding: '2px',
            }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Attention Level Banner */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
          padding: '0.4rem 0.65rem',
          backgroundColor: attentionClass,
          border: `1px solid ${attentionBorder}`,
          borderRadius: 'var(--radius-sm)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            ATTENTION LEVEL
          </span>
          <span
            style={{
              fontSize: '0.78rem',
              fontWeight: 800,
              fontFamily: 'var(--font-mono)',
              color: attentionColor,
            }}
          >
            {district.attention_level}
          </span>
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          Model-based attention level. Not an official warning.
        </div>
      </div>

      {/* 2x2 Metric Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '0.45rem',
        }}
      >
        <div style={{ padding: '0.45rem 0.55rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', textTransform: 'uppercase' }}>Raw NWP</div>
          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {district.raw_mean_mm} <span style={{ fontSize: '0.68rem', fontWeight: 400 }}>mm</span>
          </div>
        </div>

        <div style={{ padding: '0.45rem 0.55rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', textTransform: 'uppercase' }}>AI-Corrected</div>
          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
            {district.corrected_mean_mm} <span style={{ fontSize: '0.68rem', fontWeight: 400 }}>mm</span>
          </div>
        </div>

        <div style={{ padding: '0.45rem 0.55rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', textTransform: 'uppercase' }}>Difference</div>
          <div
            style={{
              fontSize: '0.95rem',
              fontWeight: 700,
              fontFamily: 'var(--font-mono)',
              color: diff >= 0 ? 'var(--status-high-text)' : 'var(--accent-blue)',
            }}
          >
            {diffSign} <span style={{ fontSize: '0.68rem', fontWeight: 400 }}>mm</span>
          </div>
        </div>

        <div style={{ padding: '0.45rem 0.55rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', textTransform: 'uppercase' }}>Truth (IMD)</div>
          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {district.observed_mean_mm !== null ? `${district.observed_mean_mm} mm` : '—'}
          </div>
        </div>
      </div>

      {/* Uncertainty & Probabilities */}
      <div
        style={{
          padding: '0.55rem 0.65rem',
          backgroundColor: 'var(--bg-card-subtle)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-xs)',
          fontSize: '0.74rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.3rem',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: 'var(--text-muted)' }}>Wettest Cell Mean:</span>
          <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {district.wettest_cell_mean_mm} mm
          </strong>
        </div>

        {district.wettest_cell_q10_mm !== null && (
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>Range (q10–q90):</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)', fontWeight: 600 }}>
              [{district.wettest_cell_q10_mm} – {district.wettest_cell_q90_mm}] mm
            </span>
          </div>
        )}

        {district.heavy_prob_max_cell !== null && (
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: 'var(--text-muted)' }}>P(≥64.5 mm) Heavy:</span>
            <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
              {Math.round(district.heavy_prob_max_cell * 100)}%
            </strong>
          </div>
        )}
      </div>

      {/* Link to District Detail */}
      <Link
        to={`/districts/${district.district_id}`}
        style={{
          display: 'block',
          textAlign: 'center',
          padding: '0.45rem',
          backgroundColor: 'var(--accent-glow)',
          color: 'var(--accent-cyan)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-xs)',
          textDecoration: 'none',
          fontSize: '0.76rem',
          fontWeight: 600,
          transition: 'all 0.15s ease',
        }}
      >
        District Audit Trail & Analogs →
      </Link>
    </div>
  )
}
