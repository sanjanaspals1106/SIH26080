import * as THREE from 'three'
import type { DistrictForecastSummary, GridVariable, Hotspot } from '../../types'
import earthBlueMarbleUrl from '../../assets/earth-blue-marble.jpg'
import earthCloudsUrl from '../../assets/earth-clouds.png'
import earthSpecularUrl from '../../assets/earth-specular.jpg'

// Centroid coordinates for meteorological districts (PRD F5 & F6 reference)
export const DISTRICT_COORDINATES: Record<string, { lat: number; lon: number; state: string; name: string }> = {
  D001: { lat: 16.99, lon: 73.30, state: 'Maharashtra', name: 'Ratnagiri' },
  D002: { lat: 16.03, lon: 73.82, state: 'Maharashtra', name: 'Sindhudurg' },
  D003: { lat: 11.68, lon: 76.13, state: 'Kerala', name: 'Wayanad' },
  D004: { lat: 13.92, lon: 75.56, state: 'Karnataka', name: 'Shimoga' },
  D005: { lat: 21.60, lon: 86.80, state: 'Odisha', name: 'Balasore' },
  D006: { lat: 23.83, lon: 78.73, state: 'Madhya Pradesh', name: 'Sagar' },
  D007: { lat: 26.15, lon: 91.77, state: 'Assam', name: 'Kamrup Metro' },
  D008: { lat: 21.14, lon: 79.08, state: 'Maharashtra', name: 'Nagpur' },
  D009: { lat: 28.70, lon: 77.10, state: 'Delhi', name: 'New Delhi' },
  D010: { lat: 13.08, lon: 80.27, state: 'Tamil Nadu', name: 'Chennai' },
  D011: { lat: 12.97, lon: 77.59, state: 'Karnataka', name: 'Bengaluru' },
  D012: { lat: 17.38, lon: 78.48, state: 'Telangana', name: 'Hyderabad' },
  D013: { lat: 22.57, lon: 88.36, state: 'West Bengal', name: 'Kolkata' },
  D014: { lat: 18.98, lon: 72.83, state: 'Maharashtra', name: 'Mumbai' },
  D015: { lat: 23.02, lon: 72.57, state: 'Gujarat', name: 'Ahmedabad' },
}

/**
 * Loads the realistic high-resolution NASA Blue Marble satellite photograph texture
 */
export function loadRealisticEarthTexture(onLoad?: () => void): THREE.Texture {
  const loader = new THREE.TextureLoader()
  const texture = loader.load(earthBlueMarbleUrl, () => {
    if (onLoad) onLoad()
  })
  texture.colorSpace = THREE.SRGBColorSpace
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  texture.generateMipmaps = true
  texture.minFilter = THREE.LinearMipmapLinearFilter
  texture.magFilter = THREE.LinearFilter
  texture.anisotropy = 8
  return texture
}

/**
 * Loads real satellite cloud layer texture with transparency
 */
export function loadRealisticCloudTexture(): THREE.Texture {
  const loader = new THREE.TextureLoader()
  const texture = loader.load(earthCloudsUrl)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  texture.generateMipmaps = true
  texture.minFilter = THREE.LinearMipmapLinearFilter
  texture.magFilter = THREE.LinearFilter
  return texture
}

/**
 * Loads real Earth specular reflection map (ocean reflection vs matte terrain)
 */
export function loadRealisticSpecularTexture(): THREE.Texture {
  const loader = new THREE.TextureLoader()
  const texture = loader.load(earthSpecularUrl)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  return texture
}

/**
 * Creates dynamic high-resolution scientific rainfall overlay texture matching IMD palette:
 * <1mm (trace), 1-15.6mm (light), 15.6-64.5mm (moderate), 64.5-115.6mm (heavy), >115.6mm (extreme)
 */
export function createRainfallOverlayTexture(
  districts: DistrictForecastSummary[] = [],
  hotspots: Hotspot[] = [],
  selectedLayer: GridVariable = 'corrected'
): THREE.CanvasTexture {
  const width = 4096
  const height = 2048
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!

  ctx.clearRect(0, 0, width, height)

  const toX = (lon: number) => ((lon + 180) / 360) * width
  const toY = (lat: number) => ((90 - lat) / 180) * height

  // 1. Broad synoptic monsoon rain belt (Arabian Sea moisture plume -> Western Ghats -> Central Trough -> Bay of Bengal)
  const plumeGrad = ctx.createRadialGradient(toX(74), toY(16), 20, toX(74), toY(16), 340)
  plumeGrad.addColorStop(0, 'rgba(2, 132, 199, 0.42)') // Moderate
  plumeGrad.addColorStop(0.6, 'rgba(96, 165, 250, 0.2)') // Light
  plumeGrad.addColorStop(1, 'rgba(96, 165, 250, 0)')
  ctx.fillStyle = plumeGrad
  ctx.beginPath()
  ctx.ellipse(toX(75), toY(17), 300, 160, -0.4, 0, Math.PI * 2)
  ctx.fill()

  // 2. Bay of Bengal Depression Rainfall Cell
  const bobGrad = ctx.createRadialGradient(toX(87), toY(20), 20, toX(87), toY(20), 260)
  bobGrad.addColorStop(0, 'rgba(234, 88, 12, 0.52)') // Heavy (64.5 - 115.6 mm)
  bobGrad.addColorStop(0.45, 'rgba(2, 132, 199, 0.38)') // Moderate
  bobGrad.addColorStop(1, 'rgba(96, 165, 250, 0)')
  ctx.fillStyle = bobGrad
  ctx.beginPath()
  ctx.arc(toX(87), toY(20), 260, 0, Math.PI * 2)
  ctx.fill()

  // 3. District-specific analytical rainfall overlays
  districts.forEach((d) => {
    const coords = DISTRICT_COORDINATES[d.district_id]
    if (!coords) return

    const rain =
      selectedLayer === 'raw'
        ? d.raw_mean_mm
        : selectedLayer === 'difference'
        ? Math.abs(d.corrected_mean_mm - d.raw_mean_mm)
        : d.corrected_mean_mm

    const x = toX(coords.lon)
    const y = toY(coords.lat)
    const radius = 55

    const grad = ctx.createRadialGradient(x, y, 0, x, y, radius)
    if (rain >= 115.6) {
      // Extreme / Very Heavy: Red
      grad.addColorStop(0, 'rgba(239, 68, 68, 0.85)')
      grad.addColorStop(0.5, 'rgba(239, 68, 68, 0.45)')
      grad.addColorStop(1, 'rgba(239, 68, 68, 0)')
    } else if (rain >= 64.5) {
      // Heavy: Amber/Orange
      grad.addColorStop(0, 'rgba(245, 158, 11, 0.8)')
      grad.addColorStop(0.5, 'rgba(245, 158, 11, 0.38)')
      grad.addColorStop(1, 'rgba(245, 158, 11, 0)')
    } else if (rain >= 15.6) {
      // Moderate: Blue
      grad.addColorStop(0, 'rgba(2, 132, 199, 0.68)')
      grad.addColorStop(0.6, 'rgba(2, 132, 199, 0.28)')
      grad.addColorStop(1, 'rgba(2, 132, 199, 0)')
    } else if (rain >= 1.0) {
      // Light: Sky Blue
      grad.addColorStop(0, 'rgba(96, 165, 250, 0.45)')
      grad.addColorStop(1, 'rgba(96, 165, 250, 0)')
    } else {
      return
    }

    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(x, y, radius, 0, Math.PI * 2)
    ctx.fill()
  })

  // 4. Hotspot Warning Clusters
  hotspots.forEach((h) => {
    const x = toX(h.centroid.lon)
    const y = toY(h.centroid.lat)
    const grad = ctx.createRadialGradient(x, y, 0, x, y, 80)
    grad.addColorStop(0, 'rgba(239, 68, 68, 0.88)')
    grad.addColorStop(0.45, 'rgba(245, 158, 11, 0.5)')
    grad.addColorStop(1, 'rgba(239, 68, 68, 0)')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(x, y, 80, 0, Math.PI * 2)
    ctx.fill()
  })

  // 5. Geographic Subtle Labels for Subcontinent
  ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'
  ctx.font = 'bold 24px "Manrope", sans-serif'
  ctx.textAlign = 'center'
  ctx.fillText('INDIA', toX(78.5), toY(22.5))

  ctx.font = 'italic 16px "Manrope", sans-serif'
  ctx.fillStyle = 'rgba(255, 255, 255, 0.85)'
  ctx.fillText('ARABIAN SEA', toX(66.5), toY(16.5))
  ctx.fillText('BAY OF BENGAL', toX(89.5), toY(15.5))
  ctx.fillText('INDIAN OCEAN', toX(78.5), toY(2.5))

  const texture = new THREE.CanvasTexture(canvas)
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.ClampToEdgeWrapping
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

/**
 * Converts GPS coordinates (lat, lon in degrees) to 3D Cartesian Vector3 on a sphere of radius R
 */
export function latLonToVector3(lat: number, lon: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180)
  const theta = (lon + 180) * (Math.PI / 180)

  const x = -(radius * Math.sin(phi) * Math.cos(theta))
  const z = radius * Math.sin(phi) * Math.sin(theta)
  const y = radius * Math.cos(phi)

  return new THREE.Vector3(x, y, z)
}



