import type { CSSProperties, ReactNode } from 'react'
import { MapContainer, TileLayer } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

// Default view: all of India (PRD D1).
const INDIA_CENTER: [number, number] = [22.5, 80.0]
const INDIA_ZOOM = 5

// 1x1 transparent GIF. Used for tiles that fail to load, so no broken-image icon is ever shown.
const BLANK_TILE =
  'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'

interface BaseMapProps {
  /** Optional OpenStreetMap tiles underneath everything else (PRD 19.3). Default: on. */
  useOsmTiles?: boolean
  style?: CSSProperties
  /** Data layers (cells, hotspots, ...). None of them may depend on the tile layer. */
  children?: ReactNode
}

/**
 * Base map (PRD 19.3).
 *
 * - The district outlines will be the base map and must always work without internet.
 * - The OpenStreetMap tile layer is an optional extra underneath. If tiles fail to load the map
 *   keeps working and nothing is shown to the user (tileerror is handled silently).
 * - No other component may depend on the tile layer.
 */
export default function BaseMap({ useOsmTiles = true, style, children }: BaseMapProps) {
  return (
    <MapContainer
      center={INDIA_CENTER}
      zoom={INDIA_ZOOM}
      preferCanvas
      style={{ height: '100%', width: '100%', ...style }}
    >
      {useOsmTiles && (
        <TileLayer
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
          errorTileUrl={BLANK_TILE}
          eventHandlers={{
            // Deliberately silent: a failed tile is not an error for the user (PRD 19.3).
            tileerror: () => {},
          }}
        />
      )}

      {/* ------------------------------------------------------------------------------------
          PLACEHOLDER (M5): district GeoJSON layer. NOT IMPLEMENTED.
          PRD 19.3: the district outlines are the main base map. One GeoJSON file from
          GET /map/districts, simplified to under 10 MB, served once. Add it here, above the
          tile layer, so the map still shows district outlines when tiles are missing.
          ------------------------------------------------------------------------------------ */}

      {children}
    </MapContainer>
  )
}
