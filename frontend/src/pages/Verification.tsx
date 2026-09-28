import { useEffect, useState } from 'react'
import ModelScorecard from '../components/verification/ModelScorecard'
import MetricsComparisonChart from '../components/verification/MetricsComparisonChart'
import FssChart from '../components/verification/FssChart'
import ProbabilitySkillCard from '../components/verification/ProbabilitySkillCard'
import RangeCoverageCard from '../components/verification/RangeCoverageCard'
import HighlightsCard from '../components/verification/HighlightsCard'
import { THRESHOLD_LABELS } from '../components/verification/metricUtils'
import LoadingState from '../components/common/LoadingState'
import ErrorState from '../components/common/ErrorState'
import { getVerificationReport } from '../api'
import type { EvaluationSet, VerificationReport } from '../types'

export default function Verification() {
  const [report, setReport] = useState<VerificationReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [evalSet, setEvalSet] = useState<EvaluationSet>('development')
  const [threshold, setThreshold] = useState<string>('64.5')

  useEffect(() => {
    getVerificationReport()
      .then(setReport)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Failed to load the verification report'))
  }, [])

  if (error) return <ErrorState message={error} onRetry={() => window.location.reload()} />
  if (!report) return <LoadingState label="Loading verification report…" />

  const evaluation = evalSet === 'holdout' && report.holdout ? report.holdout : report.development
  const thresholds = evaluation.thresholds_mm.map(String)
  const activeThreshold = thresholds.includes(threshold) ? threshold : thresholds[0]
  const lock = evaluation.lock
  const showProbability = evaluation.evaluation_set === 'development'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', height: '100%' }}>
      {/* Page Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          padding: '0.6rem 0.85rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
            Verification & Skill Report
          </h1>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
            Raw NWP, quantile mapping, global ML and regime-aware ML compared on identical dates and valid IMD cells
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Evaluation:</span>
            <div className="segmented-group">
              <button
                type="button"
                className={`segmented-btn ${evaluation.evaluation_set === 'development' ? 'active' : ''}`}
                onClick={() => setEvalSet('development')}
              >
                Development (out-of-fold)
              </button>
              {report.holdout && (
                <button
                  type="button"
                  className={`segmented-btn ${evaluation.evaluation_set === 'holdout' ? 'active' : ''}`}
                  onClick={() => setEvalSet('holdout')}
                >
                  {report.holdout.seasons.join(', ')} Holdout
                </button>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Threshold:</span>
            <select
              value={activeThreshold}
              onChange={(e) => setThreshold(e.target.value)}
              style={{
                padding: '0.3rem 0.55rem',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-xs)',
                fontSize: '0.76rem',
                fontFamily: 'var(--font-mono)',
                outline: 'none',
              }}
            >
              {thresholds.map((t) => (
                <option key={t} value={t}>
                  {t} mm{THRESHOLD_LABELS[t] ? ` (${THRESHOLD_LABELS[t]})` : ''}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* What is being evaluated */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.5rem',
          padding: '0.5rem 0.85rem',
          backgroundColor: 'var(--bg-card-subtle)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          fontSize: '0.74rem',
          color: 'var(--text-secondary)',
        }}
      >
        <span>
          <strong style={{ color: 'var(--text-primary)' }}>{evaluation.label}</strong> · seasons {evaluation.seasons.join(', ')}
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          {lock
            ? `Evaluated once${lock.timestamp ? ` on ${lock.timestamp.slice(0, 10)}` : ''}`
            : 'Predictions for each season made without training on that season'}
        </span>
      </div>

      <HighlightsCard highlights={evaluation.highlights} />

      <ModelScorecard evaluation={evaluation} threshold={activeThreshold} />

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '0.85rem' }}>
        <MetricsComparisonChart evaluation={evaluation} />
        <FssChart evaluation={evaluation} threshold={activeThreshold} />
      </div>

      {showProbability && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '0.85rem' }}>
          <ProbabilitySkillCard rows={report.probability.rows} />
          <RangeCoverageCard rows={report.range_coverage.rows} interval={report.range_coverage.interval} />
        </div>
      )}

      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', padding: '0 0.25rem 0.5rem' }}>{report.pooling}</div>
    </div>
  )
}
