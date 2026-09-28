import type { AnalogsResponse } from '../../types'

interface HistoricalAnalogsTabProps {
  analogsData: AnalogsResponse | null
  loading?: boolean
}

const fmtMm = (v: number | null | undefined): string => (v == null ? '—' : `${v} mm`)
const fmtPct = (v: number | null | undefined): string => (v == null ? '—' : `${Math.round(v * 100)}%`)
const fmtSigned = (v: number | null | undefined): string =>
  v == null ? '—' : `${v > 0 ? '+' : ''}${v} mm`

const th = { padding: '0.5rem 0.65rem' }
const thRight = { padding: '0.5rem 0.65rem', textAlign: 'right' as const }
const td = { padding: '0.45rem 0.65rem' }
const tdMono = { ...td, textAlign: 'right' as const, fontFamily: 'var(--font-mono)' }

export default function HistoricalAnalogsTab({
  analogsData,
  loading = false,
}: HistoricalAnalogsTabProps) {
  if (loading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Searching past monsoon situations…
      </div>
    )
  }

  if (!analogsData || !analogsData.analogs || analogsData.analogs.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No comparable past monsoon situations are on record for this district and forecast.
      </div>
    )
  }

  const showWettest = analogsData.analogs.some((a) => a.observed_wettest_cell_mm != null)
  const q = analogsData.query
  const median = analogsData.median_error_observed_minus_raw_mm

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Header Info */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.6rem',
          padding: '0.75rem 1rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
        }}
      >
        <div>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            F3 · {analogsData.n_analogs} Similar Past Monsoon Situations
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', maxWidth: '640px' }}>
            Similar monsoon situations (phase + low-pressure-system similarity): ranked by standardized
            distance on active/break phase probability and low-pressure-system presence and strength.
            Drawn from other development seasons, at least 5 days apart
            {analogsData.library_size ? ` (${analogsData.library_size} candidate days)` : ''}.
          </div>
        </div>

        <div
          style={{
            padding: '0.35rem 0.65rem',
            backgroundColor: 'var(--bg-input)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-xs)',
            fontSize: '0.75rem',
            fontFamily: 'var(--font-mono)',
          }}
        >
          Median Error (Obs − Raw): <strong style={{ color: 'var(--accent-cyan)' }}>{fmtSigned(median)}</strong>{' '}
          <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>
            ({analogsData.n_analogs} cases only)
          </span>
        </div>
      </div>

      {/* Today's situation */}
      {q && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '1.25rem',
            padding: '0.6rem 1rem',
            backgroundColor: 'var(--bg-card-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            fontSize: '0.76rem',
            color: 'var(--text-secondary)',
          }}
        >
          <strong style={{ color: 'var(--text-primary)' }}>This forecast · {q.imd_date}</strong>
          <span>
            Active phase: <strong style={{ fontFamily: 'var(--font-mono)' }}>{fmtPct(q.p_active)}</strong>
          </span>
          <span>
            Break phase: <strong style={{ fontFamily: 'var(--font-mono)' }}>{fmtPct(q.p_break)}</strong>
          </span>
          <span>
            Low-pressure system:{' '}
            <strong style={{ fontFamily: 'var(--font-mono)' }}>
              {q.lps_present ? `present (strength ${q.lps_strength ?? '—'})` : 'none detected'}
            </strong>
          </span>
        </div>
      )}

      {/* Analogs Table */}
      <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem', textAlign: 'left' }}>
          <thead>
            <tr
              style={{
                backgroundColor: 'var(--bg-input)',
                borderBottom: '1px solid var(--border-subtle)',
                color: 'var(--text-muted)',
              }}
            >
              <th style={th}>Rank</th>
              <th style={th}>Analog Date</th>
              <th style={th}>Season</th>
              <th style={thRight}>Distance</th>
              <th style={thRight}>Distance %ile</th>
              <th style={thRight}>Active</th>
              <th style={thRight}>Break</th>
              <th style={thRight}>Low-pressure system</th>
              <th style={thRight}>Observed Mean</th>
              {showWettest && <th style={thRight}>Obs. Wettest Cell</th>}
              <th style={thRight}>Raw NWP Mean</th>
              <th style={thRight}>Error (Obs − Raw)</th>
            </tr>
          </thead>
          <tbody>
            {analogsData.analogs.map((a) => (
              <tr key={a.rank} style={{ borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'var(--bg-card)' }}>
                <td style={{ ...td, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                  #{a.rank}
                </td>
                <td style={{ ...td, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{a.imd_date}</td>
                <td style={{ ...td, color: 'var(--text-secondary)' }}>{a.season}</td>
                <td style={tdMono}>{a.distance.toFixed(2)}</td>
                <td style={{ ...tdMono, color: 'var(--accent-sky)' }}>{a.distance_percentile}%</td>
                <td style={tdMono}>{fmtPct(a.p_active)}</td>
                <td style={tdMono}>{fmtPct(a.p_break)}</td>
                <td style={tdMono}>{a.lps_present ? `present (${a.lps_strength ?? '—'})` : 'none'}</td>
                <td style={{ ...tdMono, fontWeight: 700, color: 'var(--text-primary)' }}>
                  {fmtMm(a.observed_mean_mm)}
                </td>
                {showWettest && <td style={tdMono}>{fmtMm(a.observed_wettest_cell_mm)}</td>}
                <td style={{ ...tdMono, color: 'var(--text-secondary)' }}>{fmtMm(a.raw_mean_mm)}</td>
                <td
                  style={{
                    ...tdMono,
                    fontWeight: 700,
                    color:
                      a.error_observed_minus_raw_mm != null && a.error_observed_minus_raw_mm >= 0
                        ? 'var(--status-high-text)'
                        : 'var(--accent-blue)',
                  }}
                >
                  {fmtSigned(a.error_observed_minus_raw_mm)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        Observed values are IMD district rainfall on the analog day; raw NWP is the ECMWF forecast for that day.
        Distance percentiles rank similarity against all candidate days. No similarity percentage is inferred.
      </div>
    </div>
  )
}
