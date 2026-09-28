import type { NwpRun } from '../../types'
import EvaluationBadge from '../common/EvaluationBadge'

const labelStyle = {
  fontSize: '0.75rem',
  fontWeight: 600,
  color: 'var(--text-muted)',
  textTransform: 'uppercase' as const,
  letterSpacing: '0.04em',
}

const fieldStyle = {
  padding: '0.35rem 0.65rem',
  backgroundColor: 'var(--bg-input)',
  color: 'var(--text-primary)',
  border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--radius-sm)',
  fontSize: '0.82rem',
  fontFamily: 'var(--font-mono)',
  fontWeight: 600,
  cursor: 'pointer',
  outline: 'none',
}

const dateOf = (r: NwpRun) => r.initialization_time.split('T')[0]

interface ForecastControlsProps {
  runs: NwpRun[]
  selectedRunId: string
  onSelectRunId: (runId: string) => void
  selectedLead: number
  onSelectLead: (lead: number) => void
  isSplitView: boolean
  onToggleSplitView: (split: boolean) => void
  viewMode?: 'globe' | 'analytical'
  onSelectViewMode?: (mode: 'globe' | 'analytical') => void
  showHotspots?: boolean
  onToggleHotspots?: (show: boolean) => void
}

export default function ForecastControls({
  runs,
  selectedRunId,
  onSelectRunId,
  selectedLead,
  onSelectLead,
  isSplitView,
  onToggleSplitView,
  viewMode = 'globe',
  onSelectViewMode,
  showHotspots = false,
  onToggleHotspots,
}: ForecastControlsProps) {
  const currentRun = runs.find((r) => r.run_id === selectedRunId) || runs[0]
  const seasons = Array.from(new Set(runs.map((r) => r.season))).sort((a, b) => a - b)
  const seasonRuns = runs
    .filter((r) => r.season === currentRun?.season)
    .sort((a, b) => a.initialization_time.localeCompare(b.initialization_time))
  const runIndex = seasonRuns.findIndex((r) => r.run_id === currentRun?.run_id)
  const currentDate = currentRun ? dateOf(currentRun) : ''

  const step = (delta: number) => {
    const next = seasonRuns[runIndex + delta]
    if (next) onSelectRunId(next.run_id)
  }

  // A typed date only applies when a run was issued that day (runs are daily, June to September).
  const selectDate = (date: string) => {
    const match = seasonRuns.find((r) => dateOf(r) === date)
    if (match) onSelectRunId(match.run_id)
  }

  // Keep the same calendar day when changing season, falling back to the season's first run.
  const selectSeason = (season: number) => {
    const monthDay = currentDate.slice(5)
    const candidates = runs.filter((r) => r.season === season)
    const same = candidates.find((r) => dateOf(r).slice(5) === monthDay)
    const first = [...candidates].sort((a, b) => a.initialization_time.localeCompare(b.initialization_time))[0]
    const target = same ?? first
    if (target) onSelectRunId(target.run_id)
  }

  // IMD day each lead verifies against: the run's first IMD date is the lead-1 day.
  const validDate = (() => {
    if (!currentRun?.first_imd_date) return null
    const d = new Date(`${currentRun.first_imd_date}T00:00:00Z`)
    d.setUTCDate(d.getUTCDate() + selectedLead - 1)
    return d.toISOString().slice(0, 10)
  })()

  return (
    <div className="control-toolbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap' }}>
        {/* Replay season + forecast date selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap' }}>
          <span style={labelStyle}>Season:</span>
          <select
            id="season-select"
            value={currentRun?.season ?? ''}
            onChange={(e) => selectSeason(Number(e.target.value))}
            style={fieldStyle}
          >
            {seasons.map((season) => (
              <option key={season} value={season}>
                {season}
              </option>
            ))}
          </select>

          <span style={{ ...labelStyle, marginLeft: '0.35rem' }}>Forecast date:</span>
          <button type="button" className="segmented-btn" onClick={() => step(-1)} disabled={runIndex <= 0} aria-label="Previous day">
            ‹
          </button>
          <input
            id="run-date"
            type="date"
            value={currentDate}
            min={seasonRuns[0] ? dateOf(seasonRuns[0]) : undefined}
            max={seasonRuns.length ? dateOf(seasonRuns[seasonRuns.length - 1]) : undefined}
            onChange={(e) => selectDate(e.target.value)}
            style={{ ...fieldStyle, colorScheme: 'dark light' }}
          />
          <button
            type="button"
            className="segmented-btn"
            onClick={() => step(1)}
            disabled={runIndex < 0 || runIndex >= seasonRuns.length - 1}
            aria-label="Next day"
          >
            ›
          </button>
          {currentRun?.evaluation_set && <EvaluationBadge evaluationSet={currentRun.evaluation_set} />}
        </div>

        {/* Lead Day Segmented Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Forecast Horizon:
          </span>
          <div className="segmented-group">
            {[1, 2, 3].map((lead) => (
              <button
                key={lead}
                type="button"
                className={`segmented-btn ${selectedLead === lead ? 'active' : ''}`}
                onClick={() => onSelectLead(lead)}
              >
                Lead {lead} (+{lead * 24}h)
              </button>
            ))}
          </div>
          {validDate && (
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              valid for <strong style={{ color: 'var(--text-primary)' }}>{validDate}</strong>
            </span>
          )}
        </div>

        {/* F6 Hotspots Toggle Button */}
        {onToggleHotspots && (
          <button
            type="button"
            onClick={() => onToggleHotspots(!showHotspots)}
            style={{
              padding: '0.3rem 0.65rem',
              backgroundColor: showHotspots ? 'rgba(239, 68, 68, 0.2)' : 'var(--bg-input)',
              color: showHotspots ? '#f87171' : 'var(--text-secondary)',
              border: `1px solid ${showHotspots ? '#ef4444' : 'var(--border-subtle)'}`,
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.76rem',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              fontFamily: 'var(--font-mono)',
            }}
          >
            <span>🔥</span>
            <span>Hotspots (F6)</span>
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: showHotspots ? '#ef4444' : 'var(--text-muted)',
              }}
            />
          </button>
        )}
      </div>

      {/* Right: View Mode Toggle & Telemetry */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
        {currentRun && (
          <div
            style={{
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              padding: '0.2rem 0.5rem',
              backgroundColor: 'var(--bg-card-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-xs)',
            }}
          >
            {currentRun.source} · {currentRun.alignment_method} Align
          </div>
        )}

        <div className="segmented-group">
          <button
            type="button"
            className={`segmented-btn ${!isSplitView && viewMode === 'globe' ? 'active' : ''}`}
            onClick={() => {
              onToggleSplitView(false)
              if (onSelectViewMode) onSelectViewMode('globe')
            }}
          >
            3D Globe
          </button>
          <button
            type="button"
            className={`segmented-btn ${!isSplitView && viewMode === 'analytical' ? 'active' : ''}`}
            onClick={() => {
              onToggleSplitView(false)
              if (onSelectViewMode) onSelectViewMode('analytical')
            }}
          >
            2D Analytical
          </button>
          <button
            type="button"
            className={`segmented-btn ${isSplitView ? 'active' : ''}`}
            onClick={() => onToggleSplitView(true)}
          >
            Split View (F1)
          </button>
        </div>
      </div>
    </div>
  )
}
