import type { AuditTrailResponse } from '../../types'

interface AuditTrailTabProps {
  auditTrail: AuditTrailResponse | null
  loading?: boolean
}

export default function AuditTrailTab({
  auditTrail,
  loading = false,
}: AuditTrailTabProps) {
  if (loading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Loading AI Correction Audit Trail…
      </div>
    )
  }

  if (!auditTrail || !auditTrail.steps) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Audit trail data is not available yet for this district forecast.
      </div>
    )
  }

  const { steps, summary, forecast_id, evaluation_set } = auditTrail

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
          <span style={{ fontSize: '0.74rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', textTransform: 'uppercase' }}>
            F2 · AI Correction Decision Sequence
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Forecast ID: {forecast_id} ({evaluation_set?.toUpperCase()})
          </span>
        </div>
        <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          {summary}
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          PRD Rule H5: The audit trail states what the model did. It does not claim to prove why the weather happened.
        </div>
      </div>

      {/* 7-Step Sequence Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        {/* Step 1: Raw Forecast */}
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--bg-input)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800 }}>1</span>
            <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>1. Raw NWP Input</strong>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', paddingLeft: '1.75rem' }}>
            Raw district mean: <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{steps.raw.district_mean_mm} mm</strong> · Raw wettest cell: <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{steps.raw.wettest_cell_mm} mm</strong>
          </div>
        </div>

        {/* Step 2: Detected Regime */}
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--bg-input)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800 }}>2</span>
            <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>2. Detected Meteorological Regime</strong>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', paddingLeft: '1.75rem', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
            <div>
              Phase Probabilities: Active <strong>{Math.round(steps.regime.phase.active * 100)}%</strong>, Normal <strong>{Math.round(steps.regime.phase.normal * 100)}%</strong>, Break <strong>{Math.round(steps.regime.phase.break * 100)}%</strong> (Confidence: <strong style={{ color: 'var(--accent-cyan)' }}>{steps.regime.phase.confidence_band}</strong>)
            </div>
            <div>
              Nearest Low-Pressure System: Distance <strong>{steps.regime.nearest_lps.distance_km} km</strong>, Bearing <strong>{steps.regime.nearest_lps.bearing_deg}°</strong>, Influence index <strong>{steps.regime.nearest_lps.influence}</strong>
            </div>
            <div>
              Terrain Effects: Orographic index <strong>{steps.regime.orographic_influence}</strong>, Coastal index <strong>{steps.regime.coastal_influence}</strong>
            </div>
          </div>
        </div>

        {/* Step 3: Past Bias Table History */}
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--bg-input)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800 }}>3</span>
            <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>3. Past Empirical Bias Under Similar Regime</strong>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', paddingLeft: '1.75rem' }}>
            Under {steps.history.phase} phase and {steps.history.lps_near ? 'LPS within 500km' : 'No near LPS'}: Historical Median Error = <strong style={{ color: 'var(--accent-sky)', fontFamily: 'var(--font-mono)' }}>{steps.history.median_diff_mm > 0 ? `+${steps.history.median_diff_mm}` : steps.history.median_diff_mm} mm</strong> [IQR: {steps.history.q25_diff_mm} to {steps.history.q75_diff_mm} mm] across <strong>{steps.history.n_dates} dates</strong> ({steps.history.note || 'training archive'}).
          </div>
        </div>

        {/* Step 4: Correction Applied */}
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--bg-input)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800 }}>4</span>
            <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>4. Model Correction Applied</strong>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', paddingLeft: '1.75rem' }}>
            District Mean Correction: <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>+{steps.correction.district_mean_mm} mm</strong> · Wettest Cell Correction: <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>+{steps.correction.wettest_cell_mm} mm</strong>
          </div>
        </div>

        {/* Step 5: Corrected Output */}
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--accent-cyan)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--accent-cyan)', color: 'var(--text-inverse)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800 }}>5</span>
            <strong style={{ fontSize: '0.84rem', color: 'var(--accent-cyan)' }}>5. AI-Corrected Forecast Output</strong>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', paddingLeft: '1.75rem' }}>
            Final Corrected District Mean: <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>{steps.corrected.district_mean_mm} mm</strong> · Corrected Wettest Cell: <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>{steps.corrected.wettest_cell_mm} mm</strong>
          </div>
        </div>

        {/* Step 6: Confidence & Uncertainty */}
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--bg-input)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800 }}>6</span>
            <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>6. Uncertainty and Coverage</strong>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', paddingLeft: '1.75rem' }}>
            Wettest Cell Range: [{steps.confidence.range_wettest_cell_mm.q10} – {steps.confidence.range_wettest_cell_mm.q90} mm] (Measured Empirical Coverage: <strong>{Math.round((steps.confidence.measured_coverage_q10_q90 || 0.78) * 100)}%</strong>) · Heavy Rain Prob: <strong>{Math.round((steps.confidence.heavy_prob_max_cell || 0) * 100)}%</strong>
          </div>
        </div>

        {/* Step 7: Execution Record Metadata */}
        <div
          style={{
            padding: '0.85rem',
            backgroundColor: 'var(--bg-card-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ width: '20px', height: '20px', borderRadius: '50%', backgroundColor: 'var(--bg-input)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.72rem', fontWeight: 800 }}>7</span>
            <strong style={{ fontSize: '0.84rem', color: 'var(--text-primary)' }}>7. Pipeline Execution Record</strong>
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', paddingLeft: '1.75rem' }}>
            Model: {steps.record.model_version} | Feature Set: {steps.record.feature_set_version} | Alignment: {steps.record.alignment_method} | Fallback: {steps.record.fallback_used ? 'True' : 'False'}
          </div>
        </div>
      </div>
    </div>
  )
}
