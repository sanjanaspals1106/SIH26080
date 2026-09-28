import type { GeoJsonFeatureCollection, GridCellData } from '../types'

/** IMD / model grid step in degrees (PRD D1). */
export const CELL_DEG = 0.25
/** The verification neighbourhood: 5x5 cells (PRD FSS window, about 140 km). */
export const NEIGHBOURHOOD_CELLS = 5

const KM_PER_DEG = 111.195 // mean Earth radius 6371 km

/** A place picked on the map: the click point, the 0.25 degree cell it falls in (null = outside the valid land grid). */
export interface PinnedPlace {
  lat: number
  lon: number
  cell: GridCellData | null
  districtId: string | null
  districtName: string | null
  districtState: string | null
}

type Ring = number[][]

interface IndexedDistrict {
  id: string
  bbox: [number, number, number, number] // minLon, minLat, maxLon, maxLat
  polygons: Ring[][]
}

function inRing(x: number, y: number, ring: Ring): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0]
    const yi = ring[i][1]
    const xj = ring[j][0]
    const yj = ring[j][1]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

function inPolygon(x: number, y: number, rings: Ring[]): boolean {
  if (rings.length === 0 || !inRing(x, y, rings[0])) return false
  for (let h = 1; h < rings.length; h++) if (inRing(x, y, rings[h])) return false
  return true
}

/** Precompute polygon lists and bounding boxes once per GeoJSON so a click is a cheap scan. */
export function buildDistrictIndex(fc: GeoJsonFeatureCollection | null): IndexedDistrict[] {
  if (!fc) return []
  const out: IndexedDistrict[] = []
  for (const f of fc.features) {
    const g = f.geometry as { type: string; coordinates: any } | null
    if (!g || !g.coordinates) continue
    const polygons: Ring[][] = g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : []
    if (polygons.length === 0) continue
    let minX = Infinity
    let minY = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    for (const poly of polygons) {
      for (const [x, y] of poly[0]) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
    out.push({ id: f.properties.district_id, bbox: [minX, minY, maxX, maxY], polygons })
  }
  return out
}

/** The district whose outline contains the point, or null. */
export function findDistrictAt(index: IndexedDistrict[], lat: number, lon: number): string | null {
  for (const d of index) {
    if (lon < d.bbox[0] || lon > d.bbox[2] || lat < d.bbox[1] || lat > d.bbox[3]) continue
    for (const poly of d.polygons) if (inPolygon(lon, lat, poly)) return d.id
  }
  return null
}

const cellKey = (iLat: number, iLon: number) => `${iLat}:${iLon}`

/** Grid cells are centred on multiples of 0.25 degrees, so a cell is found by rounding, not by searching. */
export function buildCellIndex(cells: GridCellData[]): Map<string, GridCellData> {
  const m = new Map<string, GridCellData>()
  for (const c of cells) m.set(cellKey(Math.round(c.lat / CELL_DEG), Math.round(c.lon / CELL_DEG)), c)
  return m
}

/** The valid grid cell containing the point, or null when the point is outside the valid land grid. */
export function cellAt(index: Map<string, GridCellData>, lat: number, lon: number): GridCellData | null {
  return index.get(cellKey(Math.round(lat / CELL_DEG), Math.round(lon / CELL_DEG))) ?? null
}

/** Side lengths (km) and area (km2) of one 0.25 degree cell at a latitude. */
export function cellSizeKm(lat: number): { ns: number; ew: number; area: number } {
  const ns = CELL_DEG * KM_PER_DEG
  const ew = CELL_DEG * KM_PER_DEG * Math.cos((lat * Math.PI) / 180)
  return { ns, ew, area: ns * ew }
}

/** Side length (km) of the 5x5-cell verification neighbourhood, north-south. */
export function neighbourhoodKm(): number {
  return NEIGHBOURHOOD_CELLS * CELL_DEG * KM_PER_DEG
}

export function cellBounds(lat: number, lon: number, cells = 1): [[number, number], [number, number]] {
  const h = (cells * CELL_DEG) / 2
  return [
    [lat - h, lon - h],
    [lat + h, lon + h],
  ]
}
