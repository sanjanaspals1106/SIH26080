import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import {
  loadRealisticEarthTexture,
  loadRealisticCloudTexture,
  loadRealisticSpecularTexture,
  createRainfallOverlayTexture,
  latLonToVector3,
  DISTRICT_COORDINATES,
} from './EarthTextures'
import type { DistrictForecastSummary, GridVariable, Hotspot } from '../../types'

interface GlobeHeroProps {
  activeRegime?: string
  leadDay?: number
  hotspots?: Hotspot[]
  districts?: DistrictForecastSummary[]
  selectedDistrict?: DistrictForecastSummary | null
  onSelectDistrict?: (district: DistrictForecastSummary | null) => void
  selectedLayer?: GridVariable
  onSelectLayer?: (layer: GridVariable) => void
  isReplay?: boolean
  showHotspots?: boolean
  onExploreDetailedMap?: () => void
}

export default function GlobeHero({
  activeRegime = 'Active Monsoon (Phase 2)',
  leadDay = 1,
  hotspots = [],
  districts = [],
  selectedDistrict = null,
  onSelectDistrict,
  selectedLayer = 'corrected',
  onSelectLayer,
  isReplay = true,
  showHotspots = true,
  onExploreDetailedMap,
}: GlobeHeroProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [isRotating, setIsRotating] = useState<boolean>(true)
  const [hasWebGL, setHasWebGL] = useState<boolean>(true)
  const [isCloseZoom, setIsCloseZoom] = useState<boolean>(false)
  const [hoveredInfo, setHoveredInfo] = useState<{
    name: string
    rain: number
    type: string
    x: number
    y: number
  } | null>(null)

  // Listen for data-theme changes on document element
  useEffect(() => {
    const updateTheme = () => {
      const current = (document.documentElement.getAttribute('data-theme') as 'dark' | 'light') || 'dark'
      setTheme(current)
    }

    updateTheme()

    const observer = new MutationObserver(() => {
      updateTheme()
    })

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })

    return () => observer.disconnect()
  }, [])

  // References for animation and control
  const earthGroupRef = useRef<THREE.Group | null>(null)
  const cloudMeshRef = useRef<THREE.Mesh | null>(null)
  const rainfallMeshRef = useRef<THREE.Mesh | null>(null)
  const pulseRingsRef = useRef<THREE.Mesh[]>([])
  const isDraggingRef = useRef<boolean>(false)
  const dragTimeoutRef = useRef<number | null>(null)
  const previousMousePosition = useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const targetRotationRef = useRef<{ x: number; y: number } | null>(null)
  const interactiveMeshesRef = useRef<THREE.Mesh[]>([])

  // Large hero globe size
  const GLOBE_RADIUS = 82

  // Center initial view prominently on India: Lon ~78.5°E, Lat ~19.5°N
  const resetToIndia = () => {
    targetRotationRef.current = {
      y: -Math.PI / 2 - (78.5 * Math.PI) / 180,
      x: (19.5 * Math.PI) / 180,
    }
    if (cameraRef.current) {
      cameraRef.current.position.set(0, 0, 185)
      setIsCloseZoom(false)
    }
  }

  // Smooth Zoom controls
  const handleZoomIn = () => {
    if (!cameraRef.current) return
    const newZ = Math.max(110, cameraRef.current.position.z - 25)
    cameraRef.current.position.z = newZ
    setIsCloseZoom(newZ <= 145)
  }

  const handleZoomOut = () => {
    if (!cameraRef.current) return
    const newZ = Math.min(310, cameraRef.current.position.z + 25)
    cameraRef.current.position.z = newZ
    setIsCloseZoom(newZ <= 145)
  }

  // Focus smoothly on selected district when changed
  useEffect(() => {
    if (!selectedDistrict || !earthGroupRef.current) return
    const coords = DISTRICT_COORDINATES[selectedDistrict.district_id]
    if (coords) {
      targetRotationRef.current = {
        y: -Math.PI / 2 - (coords.lon * Math.PI) / 180,
        x: (coords.lat * Math.PI) / 180,
      }
    }
  }, [selectedDistrict])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Verify WebGL support
    try {
      const canvas = document.createElement('canvas')
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
      if (!gl) {
        setHasWebGL(false)
        return
      }
    } catch {
      setHasWebGL(false)
      return
    }

    let animationFrameId: number
    const width = container.clientWidth || 800
    const height = container.clientHeight || 560

    // 1. Scene & Camera (Focused closely on the Indian subcontinent)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, width / height, 1, 1000)
    camera.position.set(0, 0, 185)
    cameraRef.current = camera

    // 2. WebGL Renderer with ACES Tone Mapping
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
      })
      renderer.setSize(width, height)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.toneMapping = THREE.ACESFilmicToneMapping
      renderer.toneMappingExposure = theme === 'light' ? 1.05 : 1.12
      container.appendChild(renderer.domElement)
    } catch (e) {
      console.warn('WebGL initialization failed:', e)
      setHasWebGL(false)
      return
    }

    // 3. Theme-specific Directional Lighting & Ambient Glow
    const ambientLight = new THREE.AmbientLight(
      theme === 'light' ? 0xffffff : 0xdce7f5,
      theme === 'light' ? 1.45 : 1.15
    )
    scene.add(ambientLight)

    // Directional Sun Light positioned from upper-right front
    const sunLight = new THREE.DirectionalLight(
      theme === 'light' ? 0xfffaed : 0xfffdf5,
      theme === 'light' ? 2.65 : 2.4
    )
    sunLight.position.set(180, 120, 220)
    scene.add(sunLight)

    // Soft celestial rim light
    const rimLight = new THREE.DirectionalLight(
      theme === 'light' ? 0x93c5fd : 0x60a5fa,
      theme === 'light' ? 0.95 : 0.85
    )
    rimLight.position.set(-180, -80, -120)
    scene.add(rimLight)

    // 4. Subtle Starfield (Hidden in Light Mode, visible in Dark Mode)
    const starsGeo = new THREE.BufferGeometry()
    const starCount = 650
    const starPositions = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount * 3; i += 3) {
      starPositions[i] = (Math.random() - 0.5) * 850
      starPositions[i + 1] = (Math.random() - 0.5) * 850
      starPositions[i + 2] = -220 - Math.random() * 400
    }
    starsGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
    const starsMat = new THREE.PointsMaterial({
      color: 0x94a3b8,
      size: 1.2,
      transparent: true,
      opacity: theme === 'light' ? 0 : 0.65,
    })
    const starField = new THREE.Points(starsGeo, starsMat)
    scene.add(starField)

    // 5. Earth Parent Group
    const earthGroup = new THREE.Group()
    scene.add(earthGroup)
    earthGroupRef.current = earthGroup

    // Center initial rotation directly on India
    earthGroup.rotation.y = -Math.PI / 2 - (78.5 * Math.PI) / 180
    earthGroup.rotation.x = (19.5 * Math.PI) / 180

    // 6. Base Photorealistic NASA Blue Marble Earth Sphere
    const earthGeo = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64)
    const earthTexture = loadRealisticEarthTexture()
    const specularTexture = loadRealisticSpecularTexture()
    const earthMat = new THREE.MeshStandardMaterial({
      map: earthTexture,
      roughnessMap: specularTexture,
      roughness: theme === 'light' ? 0.65 : 0.6,
      metalness: 0.08,
    })
    const earthMesh = new THREE.Mesh(earthGeo, earthMat)
    earthGroup.add(earthMesh)

    // 7. Scientific Rainfall Translucent Overlay Sphere
    const rainOverlayGeo = new THREE.SphereGeometry(GLOBE_RADIUS + 0.35, 64, 64)
    const rainfallTexture = createRainfallOverlayTexture(districts, hotspots, selectedLayer)
    const rainOverlayMat = new THREE.MeshBasicMaterial({
      map: rainfallTexture,
      transparent: true,
      opacity: theme === 'light' ? 0.95 : 0.92,
      blending: THREE.NormalBlending,
      depthWrite: false,
    })
    const rainOverlayMesh = new THREE.Mesh(rainOverlayGeo, rainOverlayMat)
    earthGroup.add(rainOverlayMesh)
    rainfallMeshRef.current = rainOverlayMesh

    // 8. Translucent Satellite Cloud Layer
    const cloudGeo = new THREE.SphereGeometry(GLOBE_RADIUS + 0.8, 64, 64)
    const cloudTexture = loadRealisticCloudTexture()
    const cloudMat = new THREE.MeshStandardMaterial({
      map: cloudTexture,
      transparent: true,
      opacity: theme === 'light' ? 0.45 : 0.38,
      blending: THREE.NormalBlending,
      depthWrite: false,
    })
    const cloudMesh = new THREE.Mesh(cloudGeo, cloudMat)
    earthGroup.add(cloudMesh)
    cloudMeshRef.current = cloudMesh

    // 9. Soft Rayleigh Atmosphere Airglow Halo
    const atmosphereGeo = new THREE.SphereGeometry(GLOBE_RADIUS * 1.025, 48, 48)
    const atmosphereMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.62 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.8);
          gl_FragColor = vec4(0.22, 0.62, 1.0, 1.0) * intensity * ${theme === 'light' ? '0.75' : '0.9'};
        }
      `,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    })
    const atmosphereMesh = new THREE.Mesh(atmosphereGeo, atmosphereMat)
    scene.add(atmosphereMesh)

    // 10. Restrained 3D Rainfall Columns & Telemetry Beacons
    const rainPillarsGroup = new THREE.Group()
    earthGroup.add(rainPillarsGroup)
    const ringsArray: THREE.Mesh[] = []
    const interactiveMeshes: THREE.Mesh[] = []

    const itemsToRender: Array<{
      name: string
      lat: number
      lon: number
      rain: number
      isHotspot: boolean
      districtObj?: DistrictForecastSummary
    }> = []

    // Add district meteorological points
    if (districts.length > 0) {
      districts.forEach((d) => {
        const coords = DISTRICT_COORDINATES[d.district_id]
        if (coords) {
          const rain =
            selectedLayer === 'raw'
              ? d.raw_mean_mm
              : selectedLayer === 'difference'
              ? Math.abs(d.corrected_mean_mm - d.raw_mean_mm)
              : d.corrected_mean_mm

          itemsToRender.push({
            name: `${d.district_name} (${d.state})`,
            lat: coords.lat,
            lon: coords.lon,
            rain,
            isHotspot: d.attention_level === 'HIGH',
            districtObj: d,
          })
        }
      })
    }

    if (showHotspots && hotspots.length > 0) {
      hotspots.forEach((h) => {
        itemsToRender.push({
          name: `Hotspot #${h.hotspot_id}`,
          lat: h.centroid.lat,
          lon: h.centroid.lon,
          rain: h.max_corrected_mean_mm,
          isHotspot: true,
        })
      })
    }

    itemsToRender.forEach((st) => {
      const pos = latLonToVector3(st.lat, st.lon, GLOBE_RADIUS)
      const normal = pos.clone().normalize()

      // Height proportional to rainfall
      const height = Math.max(4, Math.min(24, (st.rain / 180) * 22))
      const isExtreme = st.rain >= 115.6
      const isHeavy = st.rain >= 64.5
      const isModerate = st.rain >= 15.6

      let color = 0x60a5fa // Light
      if (selectedLayer === 'difference') {
        color = isHeavy ? 0xf59e0b : 0x10b981
      } else {
        color = isExtreme ? 0xef4444 : isHeavy ? 0xf59e0b : isModerate ? 0x0284c7 : 0x60a5fa
      }

      const isSelected = selectedDistrict && st.districtObj?.district_id === selectedDistrict.district_id
      if (isSelected) {
        color = 0x38bdf8
      }

      // Slender Column Geometry
      const colGeo = new THREE.CylinderGeometry(isSelected ? 1.1 : 0.6, isSelected ? 1.4 : 0.9, height, 10)
      colGeo.translate(0, height / 2, 0)
      const colMat = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: isSelected ? 1.0 : 0.7,
        transparent: true,
        opacity: 0.85,
        roughness: 0.3,
      })
      const colMesh = new THREE.Mesh(colGeo, colMat)
      colMesh.position.copy(pos)
      colMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal)
      colMesh.userData = { item: st }
      rainPillarsGroup.add(colMesh)
      interactiveMeshes.push(colMesh)

      // Top Beacon Sphere
      const tipGeo = new THREE.SphereGeometry(isSelected ? 1.8 : 1.1, 14, 14)
      const tipMat = new THREE.MeshBasicMaterial({ color: isSelected ? 0xffffff : color })
      const tipMesh = new THREE.Mesh(tipGeo, tipMat)
      tipMesh.position.copy(pos.clone().add(normal.clone().multiplyScalar(height)))
      tipMesh.userData = { item: st }
      rainPillarsGroup.add(tipMesh)
      interactiveMeshes.push(tipMesh)

      // Ground shockwave ring for High/Extreme warnings
      if (st.isHotspot || isSelected) {
        const ringGeo = new THREE.RingGeometry(1.0, 2.6, 20)
        const ringMat = new THREE.MeshBasicMaterial({
          color: isSelected ? 0x38bdf8 : isExtreme ? 0xef4444 : 0xf59e0b,
          transparent: true,
          opacity: 0.65,
          side: THREE.DoubleSide,
        })
        const ringMesh = new THREE.Mesh(ringGeo, ringMat)
        ringMesh.position.copy(pos.clone().add(normal.clone().multiplyScalar(0.4)))
        ringMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal)
        rainPillarsGroup.add(ringMesh)
        ringsArray.push(ringMesh)
      }
    })

    pulseRingsRef.current = ringsArray
    interactiveMeshesRef.current = interactiveMeshes

    // 11. Mouse & Touch Drag Interactions (3D Orbit) + Raycasting
    let isMouseDown = false
    let dragDist = 0
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()

    const onMouseDown = (e: MouseEvent) => {
      isMouseDown = true
      dragDist = 0
      isDraggingRef.current = true
      if (dragTimeoutRef.current) {
        window.clearTimeout(dragTimeoutRef.current)
      }
      previousMousePosition.current = { x: e.clientX, y: e.clientY }
    }

    const onMouseMove = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

      if (isMouseDown && earthGroupRef.current) {
        const deltaX = e.clientX - previousMousePosition.current.x
        const deltaY = e.clientY - previousMousePosition.current.y
        dragDist += Math.abs(deltaX) + Math.abs(deltaY)

        earthGroupRef.current.rotation.y += deltaX * 0.005
        earthGroupRef.current.rotation.x = Math.max(
          -Math.PI / 3,
          Math.min(Math.PI / 3, earthGroupRef.current.rotation.x + deltaY * 0.005)
        )
        previousMousePosition.current = { x: e.clientX, y: e.clientY }
      } else if (!isMouseDown) {
        // Hover Raycast check
        raycaster.setFromCamera(mouse, camera)
        const intersects = raycaster.intersectObjects(interactiveMeshesRef.current)
        if (intersects.length > 0) {
          const hit = intersects[0].object.userData?.item
          if (hit) {
            setHoveredInfo({
              name: hit.name,
              rain: hit.rain,
              type: hit.isHotspot ? 'High Priority' : 'Normal',
              x: e.clientX - rect.left,
              y: e.clientY - rect.top,
            })
            renderer.domElement.style.cursor = 'pointer'
            return
          }
        }
        setHoveredInfo(null)
        renderer.domElement.style.cursor = 'grab'
      }
    }

    const onMouseUp = (e: MouseEvent) => {
      isMouseDown = false
      // Resume orbit after 2 seconds of inactivity
      dragTimeoutRef.current = window.setTimeout(() => {
        isDraggingRef.current = false
      }, 2000)

      if (dragDist < 6) {
        const rect = renderer.domElement.getBoundingClientRect()
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1

        raycaster.setFromCamera(mouse, camera)
        const intersects = raycaster.intersectObjects(interactiveMeshesRef.current)
        if (intersects.length > 0) {
          const hit = intersects[0].object.userData?.item
          if (hit?.districtObj && onSelectDistrict) {
            onSelectDistrict(hit.districtObj)
          }
        }
      }
    }

    // Zoom on Mouse Wheel with dynamic detail trigger
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      if (!cameraRef.current) return
      const newZ = Math.max(110, Math.min(310, cameraRef.current.position.z + e.deltaY * 0.18))
      cameraRef.current.position.z = newZ
      setIsCloseZoom(newZ <= 145)
    }

    const domEl = renderer.domElement
    domEl.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    domEl.addEventListener('wheel', onWheel, { passive: false })

    // 12. Responsive Resize Observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newW, height: newH } = entry.contentRect
        if (newW > 50 && newH > 50) {
          camera.aspect = newW / newH
          camera.updateProjectionMatrix()
          renderer.setSize(newW, newH)
        }
      }
    })
    resizeObserver.observe(container)

    // 13. Main 60 FPS Render Loop
    let pulseAngle = 0

    const animate = () => {
      // Smooth interpolation to target rotation if specified
      if (targetRotationRef.current && earthGroupRef.current) {
        const diffY = targetRotationRef.current.y - earthGroupRef.current.rotation.y
        const diffX = targetRotationRef.current.x - earthGroupRef.current.rotation.x
        earthGroupRef.current.rotation.y += diffY * 0.08
        earthGroupRef.current.rotation.x += diffX * 0.08
        if (Math.abs(diffY) < 0.001 && Math.abs(diffX) < 0.001) {
          targetRotationRef.current = null
        }
      } else if (isRotating && !isDraggingRef.current && earthGroupRef.current) {
        // Slow subtle auto-orbit
        earthGroupRef.current.rotation.y += 0.0012
      }

      // Subtle atmospheric cloud drift
      if (cloudMeshRef.current) {
        cloudMeshRef.current.rotation.y += 0.0004
      }

      // Pulse ground beacon shockwave rings
      pulseAngle += 0.04
      pulseRingsRef.current.forEach((ring, idx) => {
        const scale = 1 + Math.sin(pulseAngle + idx) * 0.3
        ring.scale.set(scale, scale, 1)
        const mat = ring.material as THREE.MeshBasicMaterial
        mat.opacity = Math.max(0.15, 0.65 - scale * 0.25)
      })

      renderer.render(scene, camera)
      animationFrameId = requestAnimationFrame(animate)
    }

    animate()

    // 14. Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId)
      resizeObserver.disconnect()
      if (dragTimeoutRef.current) window.clearTimeout(dragTimeoutRef.current)
      domEl.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      domEl.removeEventListener('wheel', onWheel)

      if (container.contains(domEl)) {
        container.removeChild(domEl)
      }

      earthGeo.dispose()
      earthMat.dispose()
      rainOverlayGeo.dispose()
      rainOverlayMat.dispose()
      cloudGeo.dispose()
      cloudMat.dispose()
      atmosphereGeo.dispose()
      atmosphereMat.dispose()
      starsGeo.dispose()
      starsMat.dispose()
      renderer.dispose()
    }
  }, [theme, isRotating, hotspots, districts, selectedLayer, selectedDistrict, showHotspots, onSelectDistrict])

  const isLight = theme === 'light'

  return (
    <div
      className="globe-hero-container"
      style={{
        background: isLight
          ? 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)'
          : 'linear-gradient(135deg, rgba(6, 11, 23, 0.98) 0%, rgba(3, 7, 18, 1) 100%)',
        backdropFilter: 'blur(16px)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg)',
        padding: '1.25rem',
        boxShadow: isLight ? '0 4px 20px rgba(0, 0, 0, 0.06)' : 'var(--shadow-lg)',
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        minHeight: '560px',
      }}
    >
      {/* Soft atmospheric ambient backlight */}
      <div
        style={{
          position: 'absolute',
          top: '-15%',
          right: '20%',
          width: '500px',
          height: '500px',
          background: isLight
            ? 'radial-gradient(circle, rgba(14, 165, 233, 0.16) 0%, transparent 70%)'
            : 'radial-gradient(circle, rgba(2, 132, 199, 0.12) 0%, transparent 70%)',
          filter: 'blur(70px)',
          pointerEvents: 'none',
        }}
      />

      {/* Top Header & Telemetry Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          zIndex: 2,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
          <div
            style={{
              padding: '0.25rem 0.75rem',
              borderRadius: '9999px',
              backgroundColor: isLight ? 'rgba(2, 132, 199, 0.1)' : 'rgba(2, 132, 199, 0.15)',
              border: `1px solid ${isLight ? 'rgba(2, 132, 199, 0.35)' : 'rgba(2, 132, 199, 0.4)'}`,
              color: 'var(--accent-cyan)',
              fontSize: '0.76rem',
              fontWeight: 800,
              fontFamily: 'var(--font-mono)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              letterSpacing: '0.04em',
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: 'var(--accent-cyan)',
                boxShadow: isLight ? '0 0 8px rgba(2, 132, 199, 0.6)' : '0 0 10px #38bdf8',
                animation: 'pulse 2s infinite',
              }}
            />
            <span>3D SYNOPTIC MONSOON EARTH</span>
          </div>

          <div
            style={{
              padding: '0.25rem 0.65rem',
              borderRadius: '9999px',
              backgroundColor: 'var(--bg-card-subtle)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-secondary)',
              fontSize: '0.74rem',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
            }}
          >
            Regime: <strong style={{ color: 'var(--accent-cyan)' }}>{activeRegime}</strong>
          </div>

          <div
            style={{
              padding: '0.25rem 0.65rem',
              borderRadius: '9999px',
              backgroundColor: 'var(--bg-card-subtle)',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-secondary)',
              fontSize: '0.74rem',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
            }}
          >
            Horizon: <strong style={{ color: 'var(--text-primary)' }}>Day +{leadDay} (+{leadDay * 24}h)</strong>
          </div>

          {isReplay && (
            <div
              style={{
                padding: '0.25rem 0.65rem',
                borderRadius: '9999px',
                backgroundColor: isLight ? 'rgba(234, 88, 12, 0.12)' : 'rgba(234, 88, 12, 0.18)',
                border: '1px solid rgba(249, 115, 22, 0.4)',
                color: isLight ? '#c2410c' : '#fb923c',
                fontSize: '0.72rem',
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
              }}
            >
              MOCK DATA REPLAY
            </div>
          )}
        </div>

        {/* Layer Selector Tabs */}
        {onSelectLayer && (
          <div
            style={{
              display: 'flex',
              backgroundColor: 'var(--bg-card-subtle)',
              padding: '3px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-subtle)',
              gap: '3px',
            }}
          >
            {[
              { id: 'corrected', label: 'AI-Corrected' },
              { id: 'raw', label: 'Raw NWP' },
              { id: 'difference', label: 'Correction Delta' },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => onSelectLayer(tab.id as GridVariable)}
                style={{
                  padding: '4px 10px',
                  borderRadius: '4px',
                  border: 'none',
                  fontSize: '0.74rem',
                  fontWeight: selectedLayer === tab.id ? 700 : 500,
                  cursor: 'pointer',
                  backgroundColor: selectedLayer === tab.id ? 'var(--accent-glow-strong)' : 'transparent',
                  color: selectedLayer === tab.id ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  transition: 'all 0.15s ease',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Main 3D Earth Globe Canvas Viewport */}
      <div
        style={{
          position: 'relative',
          height: '560px',
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 'var(--radius-lg)',
          overflow: 'hidden',
          backgroundColor: isLight ? '#eaf2fa' : '#01050e',
          backgroundImage: isLight
            ? 'linear-gradient(180deg, #dbeafe 0%, #f0f7ff 100%)'
            : undefined,
          border: isLight ? '1px solid #cbd5e1' : '1px solid rgba(2, 132, 199, 0.25)',
          boxShadow: isLight
            ? 'inset 0 0 40px rgba(2, 132, 199, 0.08)'
            : 'inset 0 0 60px rgba(0, 0, 0, 0.7)',
        }}
      >
        {hasWebGL ? (
          <>
            {/* Three.js Canvas Container */}
            <div
              ref={containerRef}
              style={{
                width: '100%',
                height: '100%',
                cursor: 'grab',
                display: 'block',
              }}
              title="Click & Drag to rotate Earth. Scroll to zoom. Click rainfall pillars to select district."
            />

            {/* District Hover Tooltip */}
            {hoveredInfo && (
              <div
                style={{
                  position: 'absolute',
                  left: hoveredInfo.x + 15,
                  top: hoveredInfo.y - 35,
                  backgroundColor: isLight ? 'rgba(255, 255, 255, 0.96)' : 'rgba(6, 11, 23, 0.95)',
                  backdropFilter: 'blur(10px)',
                  border: isLight ? '1px solid #94a3b8' : '1px solid var(--border-focus)',
                  borderRadius: 'var(--radius-xs)',
                  padding: '5px 9px',
                  fontSize: '0.74rem',
                  color: 'var(--text-primary)',
                  fontFamily: 'var(--font-mono)',
                  boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
                  pointerEvents: 'none',
                  zIndex: 20,
                  whiteSpace: 'nowrap',
                }}
              >
                <div style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>{hoveredInfo.name}</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                  Precipitation: <strong style={{ color: 'var(--text-primary)' }}>{hoveredInfo.rain.toFixed(1)} mm</strong>
                </div>
              </div>
            )}

            {/* Bottom-Left Scientific Rainfall Category Scale Legend */}
            <div
              style={{
                position: 'absolute',
                bottom: '14px',
                left: '14px',
                backgroundColor: isLight ? 'rgba(255, 255, 255, 0.94)' : 'rgba(6, 11, 23, 0.88)',
                backdropFilter: 'blur(10px)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px 12px',
                display: 'flex',
                flexDirection: 'column',
                gap: '5px',
                zIndex: 10,
                boxShadow: isLight ? '0 4px 14px rgba(0,0,0,0.08)' : '0 4px 16px rgba(0,0,0,0.5)',
              }}
            >
              <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Precipitation Scale (mm / 24h)
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem' }}>
                  <span style={{ width: '9px', height: '9px', borderRadius: '2px', backgroundColor: '#ef4444' }} />
                  <span style={{ color: 'var(--text-primary)' }}>&gt;115.6mm (Ext)</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem' }}>
                  <span style={{ width: '9px', height: '9px', borderRadius: '2px', backgroundColor: '#f59e0b' }} />
                  <span style={{ color: 'var(--text-primary)' }}>64.5–115.6mm</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem' }}>
                  <span style={{ width: '9px', height: '9px', borderRadius: '2px', backgroundColor: '#0284c7' }} />
                  <span style={{ color: 'var(--text-primary)' }}>15.6–64.5mm</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.72rem' }}>
                  <span style={{ width: '9px', height: '9px', borderRadius: '2px', backgroundColor: '#60a5fa' }} />
                  <span style={{ color: 'var(--text-primary)' }}>1–15.6mm</span>
                </div>
              </div>
            </div>

            {/* Close-Zoom Regional Transition Banner */}
            {isCloseZoom && onExploreDetailedMap && (
              <div
                style={{
                  position: 'absolute',
                  top: '16px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  backgroundColor: isLight ? 'rgba(255, 255, 255, 0.96)' : 'rgba(6, 11, 23, 0.94)',
                  backdropFilter: 'blur(14px)',
                  border: isLight ? '1px solid #0284c7' : '1px solid rgba(56, 189, 248, 0.5)',
                  borderRadius: '9999px',
                  padding: '6px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  boxShadow: isLight
                    ? '0 6px 20px rgba(0, 0, 0, 0.12), 0 0 12px rgba(2, 132, 199, 0.15)'
                    : '0 8px 24px rgba(0, 0, 0, 0.7), 0 0 16px rgba(56, 189, 248, 0.25)',
                  zIndex: 25,
                  animation: 'fadeIn 0.2s ease-in-out',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '0.85rem' }}>🔍</span>
                  <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                    Regional India Focus Active
                  </span>
                </div>
                <button
                  type="button"
                  onClick={onExploreDetailedMap}
                  style={{
                    backgroundColor: isLight ? 'rgba(2, 132, 199, 0.12)' : 'rgba(2, 132, 199, 0.35)',
                    border: '1px solid var(--accent-cyan)',
                    borderRadius: '9999px',
                    color: 'var(--accent-cyan)',
                    padding: '3px 10px',
                    fontSize: '0.74rem',
                    fontWeight: 700,
                    fontFamily: 'var(--font-mono)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = isLight ? 'var(--accent-cyan)' : 'rgba(2, 132, 199, 0.6)'
                    e.currentTarget.style.color = '#ffffff'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = isLight ? 'rgba(2, 132, 199, 0.12)' : 'rgba(2, 132, 199, 0.35)'
                    e.currentTarget.style.color = 'var(--accent-cyan)'
                  }}
                >
                  <span>Explore Detailed 2D Map ↗</span>
                </button>
              </div>
            )}

            {/* Bottom-Right Compact HUD Control Pill Bar */}
            <div
              style={{
                position: 'absolute',
                bottom: '14px',
                right: '14px',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                backgroundColor: isLight ? 'rgba(255, 255, 255, 0.95)' : 'rgba(6, 11, 23, 0.92)',
                backdropFilter: 'blur(12px)',
                border: isLight ? '1px solid #cbd5e1' : '1px solid rgba(2, 132, 199, 0.4)',
                borderRadius: '9999px',
                padding: '5px 14px',
                fontSize: '0.74rem',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                boxShadow: isLight ? '0 4px 16px rgba(0,0,0,0.1)' : '0 4px 16px rgba(0,0,0,0.6)',
                zIndex: 10,
              }}
            >
              {onExploreDetailedMap && (
                <>
                  <button
                    type="button"
                    onClick={onExploreDetailedMap}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--accent-cyan)',
                      cursor: 'pointer',
                      fontWeight: 700,
                      fontSize: '0.74rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      padding: '2px 4px',
                    }}
                    title="Switch to high-resolution Leaflet analytical map"
                  >
                    <span>🗺️ Detailed 2D Map</span>
                  </button>
                  <span style={{ color: 'var(--border-strong)' }}>|</span>
                </>
              )}

              <button
                type="button"
                onClick={resetToIndia}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '0.74rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '2px 4px',
                }}
                title="Re-center viewpoint on India"
              >
                <span>🎯 Center India</span>
              </button>

              <span style={{ color: 'var(--border-strong)' }}>|</span>

              <button
                type="button"
                onClick={handleZoomIn}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  padding: '0 5px',
                }}
                title="Zoom In"
              >
                +
              </button>

              <button
                type="button"
                onClick={handleZoomOut}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  padding: '0 5px',
                }}
                title="Zoom Out"
              >
                −
              </button>

              <span style={{ color: 'var(--border-strong)' }}>|</span>

              <button
                type="button"
                onClick={() => setIsRotating(!isRotating)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: isRotating ? 'var(--accent-cyan)' : 'var(--text-muted)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '0.74rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '2px 4px',
                }}
              >
                <span>{isRotating ? '⏸ Orbit' : '▶ Orbit'}</span>
              </button>
            </div>
          </>
        ) : (
          /* Graceful WebGL Fallback */
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              width: '100%',
              padding: '2rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '3rem', marginBottom: '0.5rem' }}>🌏</div>
            <h3 style={{ margin: '0 0 0.5rem 0', color: 'var(--text-primary)' }}>
              Monsoon Satellite Telemetry Engine
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', maxWidth: '380px', margin: '0 0 1rem 0' }}>
              WebGL acceleration is unavailable in your browser.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

