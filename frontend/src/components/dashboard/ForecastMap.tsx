import { useCallback, useEffect, useMemo, type CSSProperties } from 'react'
import type { GeoJsonObject } from 'geojson'
import L from 'leaflet'
import { Rectangle, Tooltip, GeoJSON, CircleMarker, Marker, useMap, useMapEvents } from 'react-leaflet'
import BaseMap from '../../maps/BaseMap'
import type {
  GridCellData,
  GridVariable,
  GeoJsonFeatureCollection,
  Hotspot,
} from '../../types'
import { getGridCellColor } from '../../utils/colorScales'
import { cellBounds, neighbourhoodKm, type PinnedPlace } from '../../utils/geo'
import { fmtMm } from '../../utils/format'

export interface MapFlyTarget {
  lat: number
  lon: number
  zoom?: number
  /** Changes on every request so the same target can be flown to twice. */
  key: number
}

/** Reports every click on the map (cells, district fills, empty sea alike) as a plain lat/lon. */
function MapClickCapture({ onPick }: { onPick?: (lat: number, lon: number) => void }) {
  useMapEvents({
    click: (e) => onPick?.(e.latlng.lat, e.latlng.lng),
  })
  return null
}

function MapFlyController({ target }: { target?: MapFlyTarget | null }) {
  const map = useMap()
  useEffect(() => {
    if (target) map.flyTo([target.lat, target.lon], target.zoom ?? 8, { duration: 0.8 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.key])
  return null
}

const NEIGHBOURHOOD_LABEL = L.divIcon({
  className: '',
  iconSize: [0, 0],
  html: `<div style="display:inline-block;transform:translate(-50%,-135%);white-space:nowrap;padding:2px 7px;border-radius:4px;background:rgba(8,15,32,0.88);border:1px solid #f59e0b;color:#fbbf24;font:700 11px/1.3 var(--font-mono, monospace);pointer-events:none">≈${Math.round(
    neighbourhoodKm()
  )} km (5×5 cells) · heavy-rain location reliability</div>`,
})

interface ForecastMapProps {
  variable: GridVariable
  cells: GridCellData[]
  districtsGeoJson?: GeoJsonFeatureCollection | null
  selectedDistrictId?: string | null
  hotspots?: Hotspot[]
  showHotspots?: boolean
  style?: CSSProperties
  titleBadge?: string
  /** Called with the clicked lat/lon; the parent resolves district and cell. */
  onPick?: (lat: number, lon: number) => void
  pin?: PinnedPlace | null
  flyTo?: MapFlyTarget | null
}

export default function ForecastMap({
  variable,
  cells,
  districtsGeoJson,
  selectedDistrictId,
  hotspots = [],
  showHotspots = false,
  style,
  titleBadge,
  onPick,
  pin = null,
  flyTo = null,
}: ForecastMapProps) {
  const stepHalf = 0.125 // half of 0.25° grid step

  const districtStyle = useCallback(
    (feature?: { properties?: { district_id?: string } }) => {
      const isSelected = selectedDistrictId === feature?.properties?.district_id
      return {
        color: isSelected ? '#00b4d8' : '#475569',
        weight: isSelected ? 2.5 : 1,
        fillColor: '#0a1324',
        fillOpacity: isSelected ? 0.35 : 0.08,
        dashArray: isSelected ? undefined : '2, 3',
      }
    },
    [selectedDistrictId]
  )

  const districtNames = useMemo(
    () => new Map((districtsGeoJson?.features ?? []).map((f) => [f.properties.district_id, f.properties.name])),
    [districtsGeoJson]
  )
  const listDistricts = (ids: string[]) => {
    const names = ids.map((id) => districtNames.get(id) ?? id)
    return names.length > 6 ? `${names.slice(0, 6).join(', ')} +${names.length - 6} more` : names.join(', ')
  }

  // Rebuilt only when the data or layer changes, not on every pin / selection change.
  const cellLayer = useMemo(
    () =>
      cells.map((cell) => {
        const { color, opacity } = getGridCellColor(variable, cell.value)
        const bounds: [[number, number], [number, number]] = [
          [cell.lat - stepHalf, cell.lon - stepHalf],
          [cell.lat + stepHalf, cell.lon + stepHalf],
        ]
        return (
          <Rectangle
            key={cell.cell_id}
            bounds={bounds}
            pathOptions={{ fillColor: color, fillOpacity: opacity, stroke: false }}
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
      }),
    [cells, variable]
  )

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
          ⚠️ F6 HOTSPOTS: {hotspots.length} · cells with P(≥64.5 mm) ≥ 50% and corrected mean ≥ 15.6 mm
        </div>
      )}

      {pin && !pin.cell && (
        <div
          style={{
            position: 'absolute',
            bottom: '12px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 1000,
            padding: '0.35rem 0.8rem',
            backgroundColor: 'rgba(8, 15, 32, 0.92)',
            border: '1px solid #f87171',
            borderRadius: 'var(--radius-xs)',
            color: '#fca5a5',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.74rem',
            fontWeight: 700,
            pointerEvents: 'none',
          }}
        >
          No prediction here (outside IMD land grid)
        </div>
      )}

      <BaseMap style={{ height: '100%', width: '100%' }}>
        {/* Render District Boundaries (GeoJSON layer - PRD 19.3) */}
        {districtsGeoJson && (
          <GeoJSON
            data={districtsGeoJson as unknown as GeoJsonObject}
            style={districtStyle}
          />
        )}

        {/* Render Grid Cells (0.25° Canvas Rectangles) */}
        {cellLayer}

        {/* Click handling and fly-to (map level, so a click anywhere resolves a place) */}
        <MapClickCapture onPick={onPick} />
        <MapFlyController target={flyTo} />

        {/* Footprint of the picked place: its 0.25° cell and the 5×5-cell verification neighbourhood */}
        {pin && (
          <>
            {pin.cell && (
              <>
                <Rectangle
                  bounds={cellBounds(pin.cell.lat, pin.cell.lon, 5)}
                  interactive={false}
                  pathOptions={{ color: '#f59e0b', weight: 1.5, dashArray: '7 6', fill: false }}
                />
                <Marker
                  position={[pin.cell.lat + 0.625, pin.cell.lon]}
                  icon={NEIGHBOURHOOD_LABEL}
                  interactive={false}
                />
                <Rectangle
                  bounds={cellBounds(pin.cell.lat, pin.cell.lon, 1)}
                  interactive={false}
                  pathOptions={{ color: '#00e5ff', weight: 2.5, fillColor: '#00e5ff', fillOpacity: 0.12 }}
                />
              </>
            )}
            <CircleMarker
              center={[pin.lat, pin.lon]}
              radius={5}
              interactive={false}
              pathOptions={{ color: '#ffffff', weight: 2, fillColor: pin.cell ? '#00e5ff' : '#f87171', fillOpacity: 1 }}
            />
          </>
        )}

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
                  <div>Max Mean Rain: <strong>{fmtMm(h.max_corrected_mean_mm)} mm</strong></div>
                  <div>Districts ({h.district_ids.length}): {listDistricts(h.district_ids)}</div>
                </div>
              </Tooltip>
            </GeoJSON>
          ))}
      </BaseMap>
    </div>
  )
}
