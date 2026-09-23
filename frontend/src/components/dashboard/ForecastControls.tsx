import type { NwpRun } from '../../types'

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

  return (
    <div className="control-toolbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap' }}>
        {/* Replay Run Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Forecast Run:
          </span>
          <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
            <select
              id="run-select"
              value={selectedRunId}
              onChange={(e) => onSelectRunId(e.target.value)}
              style={{
                padding: '0.35rem 1.8rem 0.35rem 0.65rem',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.82rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                cursor: 'pointer',
                appearance: 'none',
                outline: 'none',
              }}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.initialization_time.split('T')[0]} · {r.source.split('-')[0]} 00 UTC ({r.evaluation_set.toUpperCase()})
                </option>
              ))}
            </select>
            <span
              style={{
                position: 'absolute',
                right: '8px',
                pointerEvents: 'none',
                fontSize: '0.65rem',
                color: 'var(--text-muted)',
              }}
            >
              ▼
            </span>
          </div>
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
