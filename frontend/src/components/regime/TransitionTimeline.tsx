import type { TransitionEvent, TransitionSeriesPoint } from '../../types'

interface TransitionTimelineProps {
  series: TransitionSeriesPoint[]
  events: TransitionEvent[]
}

export default function TransitionTimeline({
  series,
  events,
}: TransitionTimelineProps) {
  return (
    <div
      style={{
        padding: '1rem',
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.85rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            F4 · Regime Transition Detection & Event Timeline
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            3-day smoothed probability states tracking Phase Shifts (Active / Normal / Break) and Synoptic LPS Events
          </div>
        </div>
        <span
          style={{
            padding: '0.15rem 0.45rem',
            backgroundColor: 'var(--accent-glow)',
            color: 'var(--accent-cyan)',
            borderRadius: 'var(--radius-xs)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            fontWeight: 800,
          }}
        >
          {events.length} TRANSITIONS DETECTED
        </span>
      </div>

      {/* Series Point Sparkline Table */}
      <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--bg-input)', color: 'var(--text-muted)' }}>
              <th style={{ padding: '0.4rem 0.6rem' }}>IMD Date</th>
              <th style={{ padding: '0.4rem 0.6rem', textAlign: 'right' }}>P(Active)</th>
              <th style={{ padding: '0.4rem 0.6rem', textAlign: 'right' }}>P(Normal)</th>
              <th style={{ padding: '0.4rem 0.6rem', textAlign: 'right' }}>P(Break)</th>
              <th style={{ padding: '0.4rem 0.6rem', textAlign: 'center' }}>LPS Present</th>
            </tr>
          </thead>
          <tbody>
            {series.map((pt) => (
              <tr key={pt.imd_date} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <td style={{ padding: '0.35rem 0.6rem', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                  {pt.imd_date}
                </td>
                <td style={{ padding: '0.35rem 0.6rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                  {Math.round(pt.p_active * 100)}%
                </td>
                <td style={{ padding: '0.35rem 0.6rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: '#34d399' }}>
                  {Math.round(pt.p_normal * 100)}%
                </td>
                <td style={{ padding: '0.35rem 0.6rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: '#fb923c' }}>
                  {Math.round(pt.p_break * 100)}%
                </td>
                <td style={{ padding: '0.35rem 0.6rem', textAlign: 'center' }}>
                  {pt.lps_present ? <span style={{ color: '#ef4444', fontWeight: 700 }}>YES (🌀)</span> : <span style={{ color: 'var(--text-muted)' }}>No</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Events List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
          Detected Transition Events:
        </div>

        {events.map((ev) => (
          <div
            key={ev.event_id}
            style={{
              padding: '0.65rem 0.85rem',
              backgroundColor: 'var(--bg-input)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-xs)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                  {ev.event_type}
                </strong>
                <span
                  style={{
                    fontSize: '0.68rem',
                    padding: '0.1rem 0.35rem',
                    backgroundColor: ev.confirmed ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                    color: ev.confirmed ? '#34d399' : '#fbbf24',
                    borderRadius: '2px',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  {ev.confirmed ? 'CONFIRMED' : 'PENDING'}
                </span>
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                Effective Date: {ev.event_date} {ev.confidence ? `· Confidence: ${Math.round(ev.confidence * 100)}%` : ''}
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Domain Rain Impact:</div>
              <div style={{ fontSize: '0.88rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: ev.domain_mean_corrected_change_mm >= 0 ? '#34d399' : '#fb923c' }}>
                {ev.domain_mean_corrected_change_mm > 0 ? `+${ev.domain_mean_corrected_change_mm}` : ev.domain_mean_corrected_change_mm} mm/day
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
