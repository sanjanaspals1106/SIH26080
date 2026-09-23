import type { CSSProperties } from 'react'
import type { GeoJsonObject } from 'geojson'
import { Rectangle, Tooltip, GeoJSON } from 'react-leaflet'
import BaseMap from '../../maps/BaseMap'
import type {
  GridCellData,
  GridVariable,
  GeoJsonFeatureCollection,
  DistrictForecastSummary,
  Hotspot,
} from '../../types'
import { getGridCellColor } from '../../utils/colorScales'

interface ForecastMapProps {
  variable: GridVariable
  cells: GridCellData[]
  districtsGeoJson?: GeoJsonFeatureCollection | null
  districtsList?: DistrictForecastSummary[]
  selectedDistrictId?: string | null
  onSelectDistrict?: (district: DistrictForecastSummary) => void
  hotspots?: Hotspot[]
  showHotspots?: boolean
  style?: CSSProperties
  titleBadge?: string
}

export default function ForecastMap({
  variable,
  cells,
  districtsGeoJson,
  districtsList = [],
  selectedDistrictId,
  onSelectDistrict,
  hotspots = [],
  showHotspots = false,
  style,
  titleBadge,
}: ForecastMapProps) {
  const stepHalf = 0.125 // half of 0.25° grid step

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '440px', ...style }}>
      {titleBadge && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            zIndex: 1000,
            padding: '0.25rem 0.6rem',
            backgroundColor: 'rgba(8, 15, 32, 0.85)',
            backdropFilter: 'blur(4px)',
            border: '1px solid var(--border-strong)',
            borderRadius: 'var(--radius-xs)',
            color: 'var(--accent-cyan)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            fontWeight: 700,
            letterSpacing: '0.04em',
            pointerEvents: 'none',
          }}
        >
          {titleBadge}
        </div>
      )}

      {showHotspots && hotspots.length > 0 && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            zIndex: 1000,
            padding: '0.25rem 0.6rem',
            backgroundColor: 'rgba(239, 68, 68, 0.9)',
            border: '1px solid #f87171',
            borderRadius: 'var(--radius-xs)',
            color: '#ffffff',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.72rem',
            fontWeight: 800,
            letterSpacing: '0.04em',
            boxShadow: '0 2px 8px rgba(239, 68, 68, 0.5)',
          }}
        >
          ⚠️ F6 HOTSPOTS: {hotspots.length} DETECTED (≥64.5mm)
        </div>
      )}

      <BaseMap style={{ height: '100%', width: '100%' }}>
        {/* Render District Boundaries (GeoJSON layer - PRD 19.3) */}
        {districtsGeoJson && (
          <GeoJSON
            data={districtsGeoJson as unknown as GeoJsonObject}
            style={(feature) => {
              const districtId = feature?.properties?.district_id
              const isSelected = selectedDistrictId === districtId
              return {
                color: isSelected ? '#00b4d8' : '#475569',
                weight: isSelected ? 2.5 : 1,
                fillColor: '#0a1324',
                fillOpacity: isSelected ? 0.35 : 0.08,
                dashArray: isSelected ? undefined : '2, 3',
              }
            }}
            eventHandlers={{
              click: (e) => {
                const districtId = e.sourceTarget?.feature?.properties?.district_id
                if (districtId && onSelectDistrict) {
                  const match = districtsList.find((d) => d.district_id === districtId)
                  if (match) onSelectDistrict(match)
                }
              },
            }}
          />
        )}

        {/* Render Grid Cells (0.25° Canvas Rectangles) */}
        {cells.map((cell) => {
          const { color, opacity } = getGridCellColor(variable, cell.value)
          const bounds: [[number, number], [number, number]] = [
            [cell.lat - stepHalf, cell.lon - stepHalf],
            [cell.lat + stepHalf, cell.lon + stepHalf],
          ]

          return (
            <Rectangle
              key={cell.cell_id}
              bounds={bounds}
              pathOptions={{
                fillColor: color,
                fillOpacity: opacity,
                stroke: false,
              }}
            >
              <Tooltip sticky>
                <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                  <div>
                    <strong>Cell #{cell.cell_id}</strong> ({cell.lat.toFixed(2)}°N, {cell.lon.toFixed(2)}°E)
                  </div>
                  <div>
                    {variable.toUpperCase()}: <strong>{cell.value !== null ? `${cell.value} mm` : 'NaN'}</strong>
                  </div>
                </div>
              </Tooltip>
            </Rectangle>
          )
        })}

        {/* F6 Spatial Hotspots Layer Overlay */}
        {showHotspots &&
          hotspots.map((h) => (
            <GeoJSON
              key={`hotspot-${h.hotspot_id}`}
              data={h.outline as unknown as GeoJsonObject}
              style={{
                color: '#ef4444',
                weight: 2.5,
                fillColor: '#dc2626',
                fillOpacity: 0.45,
                dashArray: '4, 4',
              }}
            >
              <Tooltip sticky>
                <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#0f172a' }}>
                  <div style={{ fontWeight: 800, color: '#dc2626' }}>
                    🔥 SPATIAL HOTSPOT #{h.hotspot_id} (F6)
                  </div>
                  <div>Cells: <strong>{h.n_cells}</strong> (8-connected)</div>
                  <div>Max Prob: <strong>{Math.round(h.max_probability * 100)}% (≥64.5mm)</strong></div>
                  <div>Max Mean Rain: <strong>{h.max_corrected_mean_mm} mm</strong></div>
                  <div>Districts: {h.district_ids.join(', ')}</div>
                </div>
              </Tooltip>
            </GeoJSON>
          ))}
      </BaseMap>
    </div>
  )
}
