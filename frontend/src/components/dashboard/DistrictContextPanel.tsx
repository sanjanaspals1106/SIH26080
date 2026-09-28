import type { CSSProperties, ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { DistrictForecastSummary } from '../../types'
import { cellSizeKm, neighbourhoodKm, type PinnedPlace } from '../../utils/geo'

/** Values of the picked 0.25° cell for the current run and lead (any of them may be missing). */
export interface CellValues {
  raw?: number | null
  corrected?: number | null
  observed?: number | null
  q10?: number | null
  q50?: number | null
  q90?: number | null
  pHeavy?: number | null
  pVeryHeavy?: number | null
}

interface DistrictContextPanelProps {
  district: DistrictForecastSummary | null
  pin?: PinnedPlace | null
  cellValues?: CellValues | null
  runId?: string
  leadDay?: number
  onClose?: () => void
}

const CARD: CSSProperties = {
  width: '310px',
  minWidth: '310px',
  backgroundColor: 'var(--bg-card)',
  border: '1px solid var(--border-subtle)',
  borderRadius: 'var(--radius-md)',
  boxShadow: 'var(--shadow-sm)',
}

const BOX: CSSProperties = {
  padding: '0.45rem 0.55rem',
  backgroundColor: 'var(--bg-input)',
  borderRadius: 'var(--radius-xs)',
  border: '1px solid var(--border-subtle)',
}

const num = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? null : v.toFixed(digits)

function Metric({ label, value, unit = 'mm', color }: { label: string; value: string | null; unit?: string; color?: string }) {
  return (
    <div style={BOX}>
      <div style={{ color: 'var(--text-muted)', fontSize: '0.68rem', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: '0.95rem', fontWeight: 700, color: color ?? 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
        {value ?? '—'} {value !== null && <span style={{ fontSize: '0.68rem', fontWeight: 400 }}>{unit}</span>}
      </div>
    </div>
  )
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      {children}
    </div>
  )
}

export default function DistrictContextPanel({
  district,
  pin = null,
  cellValues = null,
  runId,
  leadDay,
  onClose,
}: DistrictContextPanelProps) {
  if (!district && !pin) {
    return (
      <div
        style={{
          ...CARD,
          padding: '1.25rem',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.82rem',
        }}
      >
        <div style={{ fontSize: '1.8rem', marginBottom: '0.4rem', opacity: 0.6 }}>📍</div>
        <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem', fontSize: '0.9rem' }}>
          Location Forecast Detail
        </div>
        <div style={{ lineHeight: 1.4 }}>
          Click anywhere on the map, pick a district on the globe, or use the search box to load that place&apos;s
          post-processed prediction.
        </div>
      </div>
    )
  }

  const title = district?.district_name ?? pin?.districtName ?? 'Selected location'
  const subtitle = district?.state ?? pin?.districtState ?? null
  const cell = pin?.cell ?? null
  const size = cell ? cellSizeKm(cell.lat) : null
  const nCells = district?.n_effective_cells ?? null

  const diff =
    district && district.corrected_mean_mm != null && district.raw_mean_mm != null
      ? district.corrected_mean_mm - district.raw_mean_mm
      : null
  const diffSign = diff === null ? null : diff > 0 ? `+${diff.toFixed(1)}` : diff.toFixed(1)

  const cellDiff =
    cellValues && cellValues.raw != null && cellValues.corrected != null ? cellValues.corrected - cellValues.raw : null

  const level = district?.attention_level ?? null
  const levelColors: Record<string, [string, string, string]> = {
    HIGH: ['var(--status-high-bg)', 'var(--status-high-text)', 'var(--status-high-border)'],
    WATCH: ['var(--status-watch-bg)', 'var(--status-watch-text)', 'var(--status-watch-border)'],
    NORMAL: ['var(--status-normal-bg)', 'var(--status-normal-text)', 'var(--status-normal-border)'],
    UNAVAILABLE: ['var(--status-unavailable-bg)', 'var(--status-unavailable-text)', 'var(--status-unavailable-border)'],
  }
  const [levelBg, levelText, levelBorder] = level ? levelColors[level] ?? levelColors.UNAVAILABLE : ['', '', '']

  const detailQuery = new URLSearchParams()
  if (runId) detailQuery.set('run', runId)
  if (leadDay) detailQuery.set('lead', String(leadDay))
  const detailLink = district
    ? `/districts/${district.district_id}${detailQuery.toString() ? `?${detailQuery.toString()}` : ''}`
    : null

  return (
    <div style={{ ...CARD, padding: '1rem 1.15rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
            {title}
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
            {[subtitle, district?.is_small ? 'Small area' : null].filter(Boolean).join(' · ')}
          </div>
        </div>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              fontSize: '0.9rem',
              padding: '2px',
            }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Footprint: what area the numbers below apply to */}
      <div
        style={{
          padding: '0.5rem 0.65rem',
          backgroundColor: 'var(--bg-card-subtle)',
          border: `1px solid ${pin && !cell ? '#f87171' : 'var(--border-subtle)'}`,
          borderRadius: 'var(--radius-xs)',
          fontSize: '0.72rem',
          lineHeight: 1.45,
          color: 'var(--text-secondary)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.25rem',
        }}
      >
        <div style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
          PREDICTION FOOTPRINT
        </div>
        {pin && cell && size ? (
          <>
            <div>
              Applies to a <strong>0.25° cell</strong> (~{Math.round(size.area).toLocaleString()} km²,{' '}
              {size.ns.toFixed(0)} × {size.ew.toFixed(0)} km, area average, not a point).
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              Cell centre {cell.lat.toFixed(3)}°N, {cell.lon.toFixed(3)}°E · pin {pin.lat.toFixed(3)}°N,{' '}
              {pin.lon.toFixed(3)}°E
            </div>
            <div>
              Heavy-rain <em>location</em> is verified only to a 5×5-cell window (~{Math.round(neighbourhoodKm())} km),
              the dashed ring on the map.
            </div>
          </>
        ) : pin && !cell ? (
          <div style={{ color: '#fca5a5', fontWeight: 600 }}>
            No prediction here (outside IMD land grid).
            {district ? ' District values below are still area-weighted over its valid cells.' : ''}
          </div>
        ) : (
          <div>
            District value = area-weighted mean over{' '}
            {nCells !== null ? <strong>{nCells.toFixed(1)} effective</strong> : 'its'} 0.25° cells (each ~28 km). Click the
            map to see a single cell.
          </div>
        )}
        {district?.is_small && (
          <div style={{ color: 'var(--status-watch-text)', fontWeight: 600 }}>
            Small district (fewer than 4 effective cells): the value is dominated by one or two grid cells, so read it
            as regional rather than district-specific.
          </div>
        )}
      </div>

      {/* Cell-level values for the picked place */}
      {pin && cell && cellValues && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
          <div style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
            THIS CELL
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.45rem' }}>
            <Metric label="Raw NWP" value={num(cellValues.raw)} />
            <Metric label="AI-Corrected" value={num(cellValues.corrected)} color="var(--accent-cyan)" />
            {cellDiff !== null && (
              <Metric
                label="Difference"
                value={cellDiff > 0 ? `+${cellDiff.toFixed(1)}` : cellDiff.toFixed(1)}
                color={cellDiff >= 0 ? 'var(--status-high-text)' : 'var(--accent-blue)'}
              />
            )}
            <Metric label="Truth (IMD)" value={num(cellValues.observed)} />
          </div>
          {(cellValues.q10 != null || cellValues.pHeavy != null) && (
            <div
              style={{
                ...BOX,
                fontSize: '0.74rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.3rem',
              }}
            >
              {cellValues.q10 != null && cellValues.q90 != null && (
                <Row label="Range (q10–q90):">
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)', fontWeight: 600 }}>
                    [{num(cellValues.q10)} – {num(cellValues.q90)}] mm
                  </span>
                </Row>
              )}
              {cellValues.q50 != null && (
                <Row label="Median (q50):">
                  <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                    {num(cellValues.q50)} mm
                  </strong>
                </Row>
              )}
              {cellValues.pHeavy != null && (
                <Row label="P(≥64.5 mm) Heavy:">
                  <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                    {Math.round(cellValues.pHeavy * 100)}%
                  </strong>
                </Row>
              )}
              {cellValues.pVeryHeavy != null && (
                <Row label="P(≥115.6 mm) Very heavy:">
                  <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                    {Math.round(cellValues.pVeryHeavy * 100)}%
                  </strong>
                </Row>
              )}
            </div>
          )}
        </div>
      )}

      {/* District-level values */}
      {district && (
        <>
          {level && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '2px',
                padding: '0.4rem 0.65rem',
                backgroundColor: levelBg,
                border: `1px solid ${levelBorder}`,
                borderRadius: 'var(--radius-sm)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-secondary)' }}>ATTENTION LEVEL</span>
                <span style={{ fontSize: '0.78rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: levelText }}>
                  {level}
                </span>
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                Model-based attention level. Not an official warning.
              </div>
            </div>
          )}

          <div style={{ fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
            DISTRICT MEAN
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.45rem', marginTop: '-0.5rem' }}>
            <Metric label="Raw NWP" value={num(district.raw_mean_mm)} />
            <Metric label="AI-Corrected" value={num(district.corrected_mean_mm)} color="var(--accent-cyan)" />
            <Metric
              label="Difference"
              value={diffSign}
              color={diff !== null && diff >= 0 ? 'var(--status-high-text)' : 'var(--accent-blue)'}
            />
            <Metric label="Truth (IMD)" value={num(district.observed_mean_mm)} />
          </div>

          {(district.wettest_cell_mean_mm != null ||
            district.wettest_cell_q10_mm != null ||
            district.heavy_prob_max_cell != null) && (
            <div
              style={{
                padding: '0.55rem 0.65rem',
                backgroundColor: 'var(--bg-card-subtle)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-xs)',
                fontSize: '0.74rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.3rem',
              }}
            >
              {district.wettest_cell_mean_mm != null && (
                <Row label="Wettest Cell Mean:">
                  <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                    {num(district.wettest_cell_mean_mm)} mm
                  </strong>
                </Row>
              )}
              {district.wettest_cell_q10_mm != null && district.wettest_cell_q90_mm != null && (
                <Row label="Range (q10–q90):">
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)', fontWeight: 600 }}>
                    [{num(district.wettest_cell_q10_mm)} – {num(district.wettest_cell_q90_mm)}] mm
                  </span>
                </Row>
              )}
              {district.heavy_prob_max_cell != null && (
                <Row label="P(≥64.5 mm) Heavy:">
                  <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                    {Math.round(district.heavy_prob_max_cell * 100)}%
                  </strong>
                </Row>
              )}
            </div>
          )}
        </>
      )}

      {/* Link to District Detail */}
      {detailLink && (
        <Link
          to={detailLink}
          style={{
            display: 'block',
            textAlign: 'center',
            padding: '0.45rem',
            backgroundColor: 'var(--accent-glow)',
            color: 'var(--accent-cyan)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-xs)',
            textDecoration: 'none',
            fontSize: '0.76rem',
            fontWeight: 600,
            transition: 'all 0.15s ease',
          }}
        >
          District Audit Trail & Analogs →
        </Link>
      )}
    </div>
  )
}
