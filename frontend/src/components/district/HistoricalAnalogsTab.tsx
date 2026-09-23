import type { AnalogsResponse } from '../../types'

interface HistoricalAnalogsTabProps {
  analogsData: AnalogsResponse | null
  loading?: boolean
}

export default function HistoricalAnalogsTab({
  analogsData,
  loading = false,
}: HistoricalAnalogsTabProps) {
  if (loading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        Searching 10D feature-space for historical analogs…
      </div>
    )
  }

  if (!analogsData || !analogsData.analogs || analogsData.analogs.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No historical analogs found for this district and regime state.
      </div>
    )
  }

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
            F3 · Top 5 Historical Regime Analogs
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Ranked by Euclidean distance across 10 standardized meteorological regime features (A1–A6, Phase, LPS). Excludes current season.
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
          Median Error (Obs − Raw):{' '}
          <strong style={{ color: 'var(--accent-cyan)' }}>
            {analogsData.median_error_observed_minus_raw_mm > 0
              ? `+${analogsData.median_error_observed_minus_raw_mm}`
              : analogsData.median_error_observed_minus_raw_mm}{' '}
            mm
          </strong>{' '}
          <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>(5 cases only)</span>
        </div>
      </div>

      {/* Analogs Table */}
      <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.78rem',
            textAlign: 'left',
          }}
        >
          <thead>
            <tr
              style={{
                backgroundColor: 'var(--bg-input)',
                borderBottom: '1px solid var(--border-subtle)',
                color: 'var(--text-muted)',
              }}
            >
              <th style={{ padding: '0.5rem 0.65rem' }}>Rank</th>
              <th style={{ padding: '0.5rem 0.65rem' }}>Analog Date</th>
              <th style={{ padding: '0.5rem 0.65rem' }}>Season</th>
              <th style={{ padding: '0.5rem 0.65rem', textAlign: 'right' }}>Distance</th>
              <th style={{ padding: '0.5rem 0.65rem', textAlign: 'right' }}>Distance %ile</th>
              <th style={{ padding: '0.5rem 0.65rem', textAlign: 'right' }}>Observed Mean</th>
              <th style={{ padding: '0.5rem 0.65rem', textAlign: 'right' }}>Obs. Wettest Cell</th>
              <th style={{ padding: '0.5rem 0.65rem', textAlign: 'right' }}>Raw NWP Mean</th>
              <th style={{ padding: '0.5rem 0.65rem', textAlign: 'right' }}>AI-Corrected Mean</th>
              <th style={{ padding: '0.5rem 0.65rem', textAlign: 'right' }}>Error (Obs − Raw)</th>
            </tr>
          </thead>
          <tbody>
            {analogsData.analogs.map((a) => (
              <tr
                key={a.rank}
                style={{
                  borderBottom: '1px solid var(--border-subtle)',
                  backgroundColor: 'var(--bg-card)',
                }}
              >
                <td style={{ padding: '0.45rem 0.65rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                  #{a.rank}
                </td>
                <td style={{ padding: '0.45rem 0.65rem', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                  {a.imd_date}
                </td>
                <td style={{ padding: '0.45rem 0.65rem', color: 'var(--text-secondary)' }}>{a.season}</td>
                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                  {a.distance.toFixed(2)}
                </td>
                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)' }}>
                  {a.distance_percentile}%
                </td>
                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {a.observed_mean_mm} mm
                </td>
                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                  {a.observed_wettest_cell_mm} mm
                </td>
                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  {a.raw_mean_mm} mm
                </td>
                <td style={{ padding: '0.45rem 0.65rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontWeight: 600 }}>
                  {a.corrected_mean_mm} mm
                </td>
                <td
                  style={{
                    padding: '0.45rem 0.65rem',
                    textAlign: 'right',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    color: a.error_observed_minus_raw_mm >= 0 ? 'var(--status-high-text)' : 'var(--accent-blue)',
                  }}
                >
                  {a.error_observed_minus_raw_mm > 0 ? `+${a.error_observed_minus_raw_mm}` : a.error_observed_minus_raw_mm} mm
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        PRD Rule: Distance percentiles measure feature space similarity against development seasons. No similarity percentage is inferred.
      </div>
    </div>
  )
}
