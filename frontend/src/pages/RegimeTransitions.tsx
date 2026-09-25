import { useEffect, useState } from 'react'
import PhaseProbabilitiesCard from '../components/regime/PhaseProbabilitiesCard'
import LpsDetectionsCard from '../components/regime/LpsDetectionsCard'
import IndicatorsGrid from '../components/regime/IndicatorsGrid'
import TransitionTimeline from '../components/regime/TransitionTimeline'
import LoadingState from '../components/common/LoadingState'
import ErrorState from '../components/common/ErrorState'
import MockDataBanner from '../components/common/MockDataBanner'
import { getRuns, getRegime, getRegimeTransitions } from '../api'
import type { NwpRun, RegimeResponse, RegimeTransitionsResponse } from '../types'

export default function RegimeTransitions() {
  const [runs, setRuns] = useState<NwpRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState<string>('')
  const [selectedLead, setSelectedLead] = useState<number>(1)

  const [regime, setRegime] = useState<RegimeResponse | null>(null)
  const [transitions, setTransitions] = useState<RegimeTransitionsResponse | null>(null)

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Load runs on mount
  useEffect(() => {
    async function init() {
      try {
        setLoading(true)
        const runsRes = await getRuns()
        setRuns(runsRes.runs)
        if (runsRes.runs.length > 0) {
          setSelectedRunId(runsRes.runs[0].run_id)
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to load regime runs')
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  // Load regime data for selected run and lead
  useEffect(() => {
    if (!selectedRunId) return

    async function loadData() {
      try {
        const [regimeRes, transRes] = await Promise.all([
          getRegime({ run_id: selectedRunId, lead_day: selectedLead }),
          getRegimeTransitions({ run_id: selectedRunId }),
        ])

        setRegime(regimeRes)
        setTransitions(transRes)
      } catch (err: unknown) {
        console.error('Failed to load regime data:', err)
      }
    }

    loadData()
  }, [selectedRunId, selectedLead])

  if (loading) {
    return <LoadingState label="Loading Regime & Transitions Workspace…" />
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => window.location.reload()} />
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', height: '100%' }}>
      <MockDataBanner show={Boolean(regime?._mock || transitions?._mock)} />

      {/* Page Header & Controls */}
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
            Monsoon Weather Regime & Transition Engine
          </h1>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
            Physical synoptic clustering · Active/Break spell dynamics · Low-pressure tracking
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {/* Run Select */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Run:
            </span>
            <select
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
              style={{
                padding: '0.3rem 0.6rem',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-xs)',
                fontSize: '0.78rem',
                fontFamily: 'var(--font-mono)',
                outline: 'none',
              }}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.initialization_time.split('T')[0]} ({r.evaluation_set.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          {/* Lead Horizon */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Horizon:
            </span>
            <div className="segmented-group">
              {[1, 2, 3].map((lead) => (
                <button
                  key={lead}
                  type="button"
                  className={`segmented-btn ${selectedLead === lead ? 'active' : ''}`}
                  onClick={() => setSelectedLead(lead)}
                >
                  Lead {lead} (+{lead * 24}h)
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Grid of Regime Components */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
        {/* Phase Probabilities Card */}
        <PhaseProbabilitiesCard regime={regime} />

        {/* LPS Detections Card */}
        <LpsDetectionsCard regime={regime} />

        {/* Indicators A1 - A6 */}
        {regime?.indicators && <IndicatorsGrid indicators={regime.indicators} />}

        {/* F4 Regime Transition Timeline */}
        {transitions && (
          <TransitionTimeline
            series={transitions.series}
            events={transitions.events}
          />
        )}
      </div>
    </div>
  )
}
