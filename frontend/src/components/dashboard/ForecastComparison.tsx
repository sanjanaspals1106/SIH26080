import ForecastMap from './ForecastMap'
import type {
  GridCellData,
  GeoJsonFeatureCollection,
  DistrictForecastSummary,
} from '../../types'

interface ForecastComparisonProps {
  rawCells: GridCellData[]
  correctedCells: GridCellData[]
  districtsGeoJson?: GeoJsonFeatureCollection | null
  districtsList?: DistrictForecastSummary[]
  selectedDistrictId?: string | null
  onSelectDistrict?: (district: DistrictForecastSummary) => void
}

export default function ForecastComparison({
  rawCells,
  correctedCells,
  districtsGeoJson,
  districtsList = [],
  selectedDistrictId,
  onSelectDistrict,
}: ForecastComparisonProps) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '0.85rem',
        width: '100%',
        height: '100%',
        minHeight: '460px',
      }}
    >
      {/* Left: Raw NWP Forecast */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div
          style={{
            padding: '0.45rem 0.8rem',
            backgroundColor: 'var(--bg-input)',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontWeight: 700, fontSize: '0.78rem', color: 'var(--text-primary)', letterSpacing: '0.02em' }}>
            RAW NWP FORECAST (ECMWF Control)
          </div>
          <span className="badge badge-replay">RAW B0</span>
        </div>
        <div style={{ flex: 1, position: 'relative', minHeight: '400px' }}>
          <ForecastMap
            variable="raw"
            cells={rawCells}
            districtsGeoJson={districtsGeoJson}
            districtsList={districtsList}
            selectedDistrictId={selectedDistrictId}
            onSelectDistrict={onSelectDistrict}
            titleBadge="RAW ECMWF"
          />
        </div>
      </div>

      {/* Right: AI-Corrected Forecast */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div
          style={{
            padding: '0.45rem 0.8rem',
            backgroundColor: 'var(--bg-input)',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ fontWeight: 700, fontSize: '0.78rem', color: 'var(--accent-cyan)', letterSpacing: '0.02em' }}>
            AI-CORRECTED FORECAST (Regime-Aware ML B3)
          </div>
          <span className="badge badge-development">CORRECTED B3</span>
        </div>
        <div style={{ flex: 1, position: 'relative', minHeight: '400px' }}>
          <ForecastMap
            variable="corrected"
            cells={correctedCells}
            districtsGeoJson={districtsGeoJson}
            districtsList={districtsList}
            selectedDistrictId={selectedDistrictId}
            onSelectDistrict={onSelectDistrict}
            titleBadge="AI-CORRECTED"
          />
        </div>
      </div>
    </div>
  )
}
