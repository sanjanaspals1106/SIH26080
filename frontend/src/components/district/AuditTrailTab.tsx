import type { ReactNode } from 'react'
import type { AuditTrailResponse } from '../../types'
import { fmtMm } from '../../utils/format'

interface AuditTrailTabProps {
  auditTrail: AuditTrailResponse | null
  loading?: boolean
}

const mono = { fontFamily: 'var(--font-mono)' } as const

function signed(value: number, digits = 1): string {
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}`
}

function compass(deg: number): string {
  const names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
  return names[Math.round(deg / 45) % 8]
}

function Strong({ children, color }: { children: ReactNode; color?: string }) {
  return <strong style={{ color: color ?? 'var(--text-primary)', ...mono }}>{children}</strong>
}

function Step({
  n,
  title,
  children,
  highlight = false,
  subtle = false,
}: {
  n: number
  title: string
  children: ReactNode
  highlight?: boolean
  subtle?: boolean
}) {
  return (
    <div
      style={{
        padding: '0.85rem',
        backgroundColor: subtle ? 'var(--bg-card-subtle)' : 'var(--bg-card)',
        border: `1px solid ${highlight ? 'var(--accent-cyan)' : 'var(--border-subtle)'}`,
        borderRadius: 'var(--radius-md)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
        <span
          style={{
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            backgroundColor: highlight ? 'var(--accent-cyan)' : 'var(--bg-input)',
            color: highlight ? 'var(--text-inverse)' : 'var(--text-primary)',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '0.72rem',
            fontWeight: 800,
          }}
        >
          {n}
        </span>
        <strong style={{ fontSize: '0.84rem', color: highlight ? 'var(--accent-cyan)' : 'var(--text-primary)' }}>
          {title}
        </strong>
      </div>
      <div
        style={{
          fontSize: '0.78rem',
          color: 'var(--text-secondary)',
          paddingLeft: '1.75rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.2rem',
        }}
      >
        {children}
      </div>
    </div>
  )
}

export default function AuditTrailTab({ auditTrail, loading = false }: AuditTrailTabProps) {
  if (loading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading AI correction audit trail…
      </div>
    )
  }

  if (!auditTrail || !auditTrail.steps) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No audit trail is stored for this district at the selected run and lead day.
      </div>
    )
  }

  const { steps, summary, forecast_id, evaluation_set } = auditTrail
  const { raw, regime, history, correction, corrected, confidence, record } = steps
  const range = confidence.range_wettest_cell_mm
  const lps = regime?.nearest_lps
  let n = 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Header Summary Box */}
      <div
        style={{
          padding: '0.85rem 1rem',
          backgroundColor: 'var(--accent-glow)',
          border: '1px solid var(--accent-cyan)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.35rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.74rem', fontWeight: 800, ...mono, color: 'var(--accent-cyan)', textTransform: 'uppercase' }}>
            F2 · AI Correction Decision Sequence
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', ...mono }}>
            Forecast ID: {forecast_id}
            {evaluation_set ? ` (${evaluation_set.toUpperCase()})` : ''}
          </span>
        </div>
        <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>{summary}</div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          The audit trail states what the model did. It does not claim to prove why the weather happened.
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        <Step n={++n} title="Raw NWP Input">
          <div>
            Raw district mean: <Strong>{fmtMm(raw.district_mean_mm)} mm</Strong>
            {raw.wettest_cell_mm != null && (
              <>
                {' '}· Raw value at the wettest cell: <Strong>{fmtMm(raw.wettest_cell_mm)} mm</Strong>
              </>
            )}
          </div>
        </Step>

        {regime && (
          <Step n={++n} title="Detected Meteorological Regime">
            <div>
              Phase probabilities: Active <strong>{Math.round(regime.phase.active * 100)}%</strong>, Normal{' '}
              <strong>{Math.round(regime.phase.normal * 100)}%</strong>, Break{' '}
              <strong>{Math.round(regime.phase.break * 100)}%</strong> (confidence:{' '}
              <strong style={{ color: 'var(--accent-cyan)' }}>{regime.phase.confidence_band}</strong>)
            </div>
            {lps?.present && lps.distance_km != null ? (
              <div>
                Nearest low-pressure system: <Strong>{lps.distance_km} km</Strong>
                {lps.bearing_deg != null && (
                  <>
                    {' '}toward <Strong>{lps.bearing_deg}° ({compass(lps.bearing_deg)})</Strong>
                  </>
                )}
                {lps.influence != null && (
                  <>
                    , influence index <Strong>{lps.influence}</Strong>
                  </>
                )}
              </div>
            ) : (
              <div>No low-pressure system detected near this district.</div>
            )}
            {(regime.orographic_influence != null || regime.coastal_influence != null) && (
              <div>
                Terrain effects:{' '}
                {regime.orographic_influence != null && (
                  <>
                    orographic index <Strong>{regime.orographic_influence}</Strong>
                  </>
                )}
                {regime.orographic_influence != null && regime.coastal_influence != null && ', '}
                {regime.coastal_influence != null && (
                  <>
                    coastal index <Strong>{regime.coastal_influence}</Strong>
                  </>
                )}
              </div>
            )}
            {regime.note && <div style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>{regime.note}</div>}
          </Step>
        )}

        {history && history.median_diff_mm != null && (
          <Step n={++n} title="Past Raw-Forecast Error Under a Similar Regime">
            <div>
              Under a <strong>{history.phase}</strong> phase with{' '}
              {history.lps_near ? 'a low-pressure system present' : 'no low-pressure system'}, observed minus raw district
              mean had a median of{' '}
              <Strong color="var(--accent-sky)">{signed(history.median_diff_mm, 2)} mm</Strong>
              {history.q25_diff_mm != null && history.q75_diff_mm != null && (
                <> (IQR {fmtMm(history.q25_diff_mm)} to {fmtMm(history.q75_diff_mm)} mm)</>
              )}{' '}
              across <strong>{history.n_dates} dates</strong>
              {history.note ? ` · ${history.note}` : ''}.
            </div>
          </Step>
        )}

        <Step n={++n} title="Model Correction Applied">
          <div>
            District mean change:{' '}
            <Strong color="var(--accent-cyan)">
              {correction.district_mean_mm != null ? `${signed(correction.district_mean_mm, 2)} mm` : '—'}
            </Strong>
            {correction.wettest_cell_mm != null && (
              <>
                {' '}· Wettest cell change:{' '}
                <Strong color="var(--accent-cyan)">{signed(correction.wettest_cell_mm, 2)} mm</Strong>
              </>
            )}
          </div>
        </Step>

        <Step n={++n} title="AI-Corrected Forecast Output" highlight>
          <div>
            Corrected district mean: <Strong>{fmtMm(corrected.district_mean_mm)} mm</Strong>
            {corrected.wettest_cell_mm != null && (
              <>
                {' '}· Corrected wettest cell: <Strong>{fmtMm(corrected.wettest_cell_mm)} mm</Strong>
              </>
            )}
          </div>
        </Step>

        {(range.q10 != null || confidence.heavy_prob_max_cell != null) && (
          <Step n={++n} title="Uncertainty and Coverage">
            {range.q10 != null && range.q90 != null && (
              <div>
                Wettest-cell range (q10 to q90): <Strong color="var(--accent-sky)">{range.q10} to {range.q90} mm</Strong>
                {range.q50 != null && <> (median {range.q50} mm)</>}
                {confidence.measured_coverage_q10_q90 != null && (
                  <>
                    {' '}· Measured coverage of this interval in development:{' '}
                    <strong>{Math.round(confidence.measured_coverage_q10_q90 * 100)}%</strong>
                  </>
                )}
              </div>
            )}
            {confidence.heavy_prob_max_cell != null && (
              <div>
                Probability of ≥64.5 mm at the wettest cell:{' '}
                <strong>{Math.round(confidence.heavy_prob_max_cell * 100)}%</strong>
                {confidence.very_heavy_prob_max_cell != null && (
                  <>
                    {' '}· ≥115.6 mm: <strong>{Math.round(confidence.very_heavy_prob_max_cell * 100)}%</strong>
                  </>
                )}
              </div>
            )}
          </Step>
        )}

        <Step n={++n} title="Pipeline Execution Record" subtle>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', ...mono }}>
            Model: {record.model_version} | Features: {record.feature_set_version} | Alignment: {record.alignment_method}
            {' '}| Product: {record.product_type} | Fallback: {record.fallback_used ? record.fallback_reason ?? 'used' : 'none'}
          </div>
        </Step>
      </div>
    </div>
  )
}
