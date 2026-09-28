import type { VerificationEvaluation } from '../../types'
import { METRICS, MODEL_COLORS, badness, fmt } from './metricUtils'

interface ModelScorecardProps {
  evaluation: VerificationEvaluation
  threshold: string
}

const GOOD = 'var(--status-normal-text)'
const BAD = 'var(--status-high-text)'

export default function ModelScorecard({ evaluation, threshold }: ModelScorecardProps) {
  const raw = evaluation.models.find((m) => m.key === 'raw_nwp')
  const events = evaluation.models[0]?.thresholds[threshold]?.n_obs_events ?? null

  return (
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Model Scorecard · threshold {threshold} mm
        </div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Bold = best in column · small figure = change from raw NWP (green better, red worse)
          {events != null && <> · {events.toLocaleString()} observed events at this threshold</>}
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
          <thead>
            <tr style={{ textAlign: 'right', color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)' }}>
              <th style={{ textAlign: 'left', padding: '0.4rem 0.5rem', fontWeight: 600 }}>Model</th>
              {METRICS.map((c) => (
                <th key={c.key} title={c.hint} style={{ padding: '0.4rem 0.5rem', fontWeight: 600, cursor: 'help', whiteSpace: 'nowrap' }}>
                  {c.label}
                  {c.unit ? <span style={{ fontWeight: 400 }}> ({c.unit})</span> : null}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {evaluation.models.map((model) => {
              const isCorrected = model.key === evaluation.corrected_key
              return (
                <tr
                  key={model.key}
                  style={{
                    borderBottom: '1px solid var(--border-subtle)',
                    backgroundColor: isCorrected ? 'var(--accent-glow)' : 'transparent',
                  }}
                >
                  <td style={{ padding: '0.5rem', minWidth: '190px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <span style={{ width: 9, height: 9, borderRadius: 2, backgroundColor: MODEL_COLORS[model.forecast_type] }} />
                      <strong style={{ color: 'var(--text-primary)' }}>{model.label}</strong>
                    </div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', paddingLeft: '1.05rem' }}>{model.description}</div>
                  </td>
                  {METRICS.map((c) => {
                    const v = c.value(model, threshold)
                    const rawValue = raw ? c.value(raw, threshold) : null
                    const column = evaluation.models.map((m) => c.value(m, threshold)).filter((x): x is number => x != null)
                    const best = column.length ? Math.min(...column.map((x) => badness(c.better, x))) : null
                    const isBest = v != null && best != null && badness(c.better, v) === best
                    let delta: number | null = null
                    let deltaColor = 'var(--text-muted)'
                    if (v != null && rawValue != null && model.key !== 'raw_nwp') {
                      delta = v - rawValue
                      const improvement = badness(c.better, rawValue) - badness(c.better, v)
                      deltaColor = improvement > 0 ? GOOD : improvement < 0 ? BAD : 'var(--text-muted)'
                    }
                    return (
                      <td key={c.key} style={{ padding: '0.5rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        <div style={{ fontWeight: isBest ? 800 : 500, color: 'var(--text-primary)' }}>{fmt(v, c.digits)}</div>
                        {delta != null && (
                          <div style={{ fontSize: '0.66rem', color: deltaColor }}>
                            {delta > 0 ? '+' : ''}
                            {delta.toFixed(c.digits)}
                          </div>
                        )}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
