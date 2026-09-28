import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import GlobeHero from '../components/dashboard/GlobeHero'
import ForecastControls from '../components/dashboard/ForecastControls'
import ForecastComparison from '../components/dashboard/ForecastComparison'
import ForecastMap from '../components/dashboard/ForecastMap'
import ImprovementSummary from '../components/dashboard/ImprovementSummary'
import DistrictContextPanel, { type CellValues } from '../components/dashboard/DistrictContextPanel'
import DistrictSearch from '../components/dashboard/DistrictSearch'
import type { MapFlyTarget } from '../components/dashboard/ForecastMap'
import PriorityTable from '../components/dashboard/PriorityTable'
import LoadingState from '../components/common/LoadingState'
import ErrorState from '../components/common/ErrorState'
import MockDataBanner from '../components/common/MockDataBanner'
import {
  getAllRuns,
  getRegimeStrict,
  getDistrictForecasts,
  getGridForecast,
  getImprovementSummary,
  getDistrictGeoJSON,
  getHotspots,
} from '../api'
import type {
  NwpRun,
  GridVariable,
  DistrictForecastSummary,
  GridCellData,
  ImprovementSummaryResponse,
  GeoJsonFeatureCollection,
  Hotspot,
} from '../types'
import {
  buildCellIndex,
  buildDistrictIndex,
  cellAt,
  findDistrictAt,
  type PinnedPlace,
} from '../utils/geo'

import ErrorBoundary from '../components/common/ErrorBoundary'

const CELL_VARIABLES: GridVariable[] = ['raw', 'corrected', 'observed', 'q10', 'q50', 'q90', 'p_ge_64_5', 'p_ge_115_6']
type CellLayers = Partial<Record<GridVariable, Map<number, number | null> | null>>

const runDate = (r: NwpRun) => r.initialization_time.split('T')[0]

/** Start on a mid-monsoon day of the first season rather than the 1 June spin-up. */
function defaultRun(runs: NwpRun[]): NwpRun | undefined {
  const sorted = [...runs].sort((a, b) => a.initialization_time.localeCompare(b.initialization_time))
  if (sorted.length === 0) return undefined
  const season = sorted[0].season
  return sorted.find((r) => r.season === season && runDate(r).slice(5) === '07-15') ?? sorted[0]
}

const PHASE_LABEL: Record<string, string> = { active: 'Active monsoon', normal: 'Normal', break: 'Break monsoon' }

export default function Dashboard() {
  const [runs, setRuns] = useState<NwpRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState<string>('')
  const [selectedLead, setSelectedLead] = useState<number>(1)
  const [selectedLayer, setSelectedLayer] = useState<GridVariable>('corrected')
  const [isSplitView, setIsSplitView] = useState<boolean>(false)
  const [viewMode, setViewMode] = useState<'globe' | 'analytical'>('globe')
  const [showHotspots, setShowHotspots] = useState<boolean>(true)

  // Data states
  const [districts, setDistricts] = useState<DistrictForecastSummary[]>([])
  const [selectedDistrict, setSelectedDistrict] = useState<DistrictForecastSummary | null>(null)
  const [gridCells, setGridCells] = useState<GridCellData[]>([])
  const [rawGridCells, setRawGridCells] = useState<GridCellData[]>([])
  const [districtsGeoJson, setDistrictsGeoJson] = useState<GeoJsonFeatureCollection | null>(null)
  const [improvementSummary, setImprovementSummary] = useState<ImprovementSummaryResponse | null>(null)
  const [hotspots, setHotspots] = useState<Hotspot[]>([])
  const [hasMockData, setHasMockData] = useState<boolean>(false)
  const [pin, setPin] = useState<PinnedPlace | null>(null)
  const [cellValues, setCellValues] = useState<CellValues | null>(null)
  const [flyTarget, setFlyTarget] = useState<MapFlyTarget | null>(null)
  const [regimeLabel, setRegimeLabel] = useState<string | null>(null)
  const [dataError, setDataError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState<number>(0)
  const flyKey = useRef(0)
  const cellLayerCache = useRef<Map<string, CellLayers>>(new Map())

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Load runs and GeoJSON on initial mount
  useEffect(() => {
    async function init() {
      try {
        setLoading(true)
        const [allRuns, geoJsonRes] = await Promise.all([
          getAllRuns(),
          getDistrictGeoJSON(),
        ])
        setRuns(allRuns)
        const start = defaultRun(allRuns)
        if (start) setSelectedRunId(start.run_id)
        setDistrictsGeoJson(geoJsonRes)
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to initialize dashboard')
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  // Load forecasts whenever selectedRunId, selectedLead, or selectedLayer changes
  useEffect(() => {
    if (!selectedRunId) return
    let cancelled = false

    async function loadForecastData() {
      try {
        const [districtsRes, gridRes, rawGridRes, impRes, hotspotsRes] = await Promise.all([
          getDistrictForecasts({ run_id: selectedRunId, lead_day: selectedLead }),
          getGridForecast({ run_id: selectedRunId, lead_day: selectedLead, variable: selectedLayer }),
          getGridForecast({ run_id: selectedRunId, lead_day: selectedLead, variable: 'raw' }),
          getImprovementSummary({ run_id: selectedRunId, lead_day: selectedLead }),
          getHotspots({ run_id: selectedRunId, lead_day: selectedLead }),
        ])
        if (cancelled) return

        setDataError(null)
        setDistricts(districtsRes.districts)
        setGridCells(gridRes.cells)
        setRawGridCells(rawGridRes.cells)
        setImprovementSummary(impRes)
        setHotspots(hotspotsRes.hotspots)
        setHasMockData(Boolean(impRes._mock || hotspotsRes._mock))

        // Keep the selected district in step with the new run / lead (matched by id, never defaulted to another)
        setSelectedDistrict((prev) =>
          prev ? districtsRes.districts.find((d) => d.district_id === prev.district_id) ?? null : null
        )
      } catch (err: unknown) {
        if (cancelled) return
        console.error('Error fetching forecast data:', err)
        setDataError(
          'Forecast data for this run and lead day could not be loaded. Choose another date or retry.'
        )
      }
    }

    loadForecastData()
    return () => {
      cancelled = true
    }
  }, [selectedRunId, selectedLead, selectedLayer, reloadKey])

  // The real monsoon phase for the selected run and lead (badge is hidden when unavailable)
  useEffect(() => {
    if (!selectedRunId) return
    let cancelled = false
    getRegimeStrict({ run_id: selectedRunId, lead_day: selectedLead })
      .then((r) => {
        if (cancelled) return
        if (!r.regime_available || !r.phase) {
          setRegimeLabel(null)
          return
        }
        const p = r.phase[r.phase.dominant_phase]
        setRegimeLabel(`${PHASE_LABEL[r.phase.dominant_phase] ?? r.phase.dominant_phase} · ${Math.round(p * 100)}%`)
      })
      .catch(() => {
        if (!cancelled) setRegimeLabel(null)
      })
    return () => {
      cancelled = true
    }
  }, [selectedRunId, selectedLead])

  // Lookup structures for resolving a clicked lat/lon to a district and a 0.25° cell
  const districtIndex = useMemo(() => buildDistrictIndex(districtsGeoJson), [districtsGeoJson])
  const cellIndex = useMemo(() => buildCellIndex(rawGridCells.length ? rawGridCells : gridCells), [rawGridCells, gridCells])
  const districtProps = useMemo(() => {
    const m = new Map<string, { name: string; state: string }>()
    districtsGeoJson?.features.forEach((f) =>
      m.set(f.properties.district_id, { name: f.properties.name, state: f.properties.state })
    )
    return m
  }, [districtsGeoJson])

  const selectDistrict = useCallback((d: DistrictForecastSummary) => {
    setSelectedDistrict(d)
    setPin(null)
    if (d.centroid_lat != null && d.centroid_lon != null) {
      setFlyTarget({ lat: d.centroid_lat, lon: d.centroid_lon, zoom: 8, key: ++flyKey.current })
    }
  }, [])

  const handlePick = useCallback(
    (lat: number, lon: number) => {
      const cell = cellAt(cellIndex, lat, lon)
      const districtId = findDistrictAt(districtIndex, lat, lon)
      const info = districtId ? districtProps.get(districtId) : undefined
      setPin({
        lat,
        lon,
        cell,
        districtId,
        districtName: info?.name ?? null,
        districtState: info?.state ?? null,
      })
      setSelectedDistrict(districtId ? districts.find((d) => d.district_id === districtId) ?? null : null)
    },
    [cellIndex, districtIndex, districtProps, districts]
  )

  // Cell-level values (raw, corrected, truth, range, heavy-rain probability) for the picked cell
  const pinnedCellId = pin?.cell?.cell_id ?? null
  useEffect(() => {
    if (pinnedCellId === null || !selectedRunId) {
      setCellValues(null)
      return
    }
    let cancelled = false
    const key = `${selectedRunId}|${selectedLead}`
    const apply = (layers: CellLayers) => {
      if (cancelled) return
      const at = (v: GridVariable) => layers[v]?.get(pinnedCellId) ?? null
      setCellValues({
        raw: at('raw'),
        corrected: at('corrected'),
        observed: at('observed'),
        q10: at('q10'),
        q50: at('q50'),
        q90: at('q90'),
        pHeavy: at('p_ge_64_5'),
        pVeryHeavy: at('p_ge_115_6'),
      })
    }
    const cached = cellLayerCache.current.get(key)
    if (cached) {
      apply(cached)
      return () => {
        cancelled = true
      }
    }
    Promise.all(
      CELL_VARIABLES.map((variable) =>
        getGridForecast({ run_id: selectedRunId, lead_day: selectedLead, variable })
          .then((r) => [variable, new Map(r.cells.map((c) => [c.cell_id, c.value] as const))] as const)
          .catch(() => [variable, null] as const)
      )
    ).then((entries) => {
      const layers: CellLayers = Object.fromEntries(entries)
      if (cellLayerCache.current.size > 6) cellLayerCache.current.clear()
      cellLayerCache.current.set(key, layers)
      apply(layers)
    })
    return () => {
      cancelled = true
    }
  }, [pinnedCellId, selectedRunId, selectedLead])

  if (loading) {
    return <LoadingState label="Loading Bharat VarshAI forecast dashboard…" />
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => window.location.reload()} />
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
      <MockDataBanner show={hasMockData} />

      {/* Primary Dashboard Hero Header (Canva Reference Alignment) */}
      <div
        className="dashboard-hero-header"
        style={{
          padding: '0.25rem 0.25rem 0 0.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          gap: '1rem',
          flexWrap: 'wrap',
        }}
      >
       <div style={{ minWidth: 0, flex: '1 1 480px' }}>
        <div
          style={{
            fontSize: '0.72rem',
            fontWeight: 800,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'var(--accent-cyan)',
            marginBottom: '0.35rem',
            fontFamily: 'var(--font-mono)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <span
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: 'var(--accent-cyan)',
              display: 'inline-block',
            }}
          />
          <span>DATA-DRIVEN DECISIONS FOR CLIMATE RESILIENCE</span>
        </div>

        <h1
          style={{
            fontSize: '1.95rem',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            color: 'var(--text-primary)',
            lineHeight: 1.2,
            margin: '0 0 0.4rem 0',
            fontFamily: 'var(--font-sans)',
          }}
        >
          Rainfall Insights for a <span style={{ color: 'var(--accent-cyan)' }}>Safer India</span>
        </h1>

        <p
          style={{
            fontSize: '0.9rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.55,
            maxWidth: '820px',
            margin: 0,
            fontWeight: 500,
          }}
        >
          Replay-based rainfall forecasting and regime analysis to support district-level disaster management, planning and preparedness.
        </p>
       </div>
        <DistrictSearch districts={districts} onSelect={selectDistrict} />
      </div>

      {/* Tier 1: Forecast Controls & Replay Timeline */}
      <ForecastControls
        runs={runs}
        selectedRunId={selectedRunId}
        onSelectRunId={setSelectedRunId}
        selectedLead={selectedLead}
        onSelectLead={setSelectedLead}
        isSplitView={isSplitView}
        onToggleSplitView={setIsSplitView}
        viewMode={viewMode}
        onSelectViewMode={setViewMode}
        showHotspots={showHotspots}
        onToggleHotspots={setShowHotspots}
      />

      {/* Tier 2: Workspace — Split View (F1), 3D Earth Globe, OR Detailed 2D Analytical Map */}
      {dataError ? (
        <ErrorState
          title="Forecast data unavailable"
          message={dataError}
          onRetry={() => setReloadKey((k) => k + 1)}
        />
      ) : isSplitView ? (
        /* F1 Side-by-Side Comparison Workspace */
        <ErrorBoundary name="F1 Split View Comparison">
          <div style={{ minHeight: '520px' }}>
            <ForecastComparison
              rawCells={rawGridCells}
              correctedCells={gridCells}
              districtsGeoJson={districtsGeoJson}
              selectedDistrictId={selectedDistrict?.district_id}
              onPick={handlePick}
              pin={pin}
              flyTo={flyTarget}
            />
          </div>
        </ErrorBoundary>
      ) : (
        /* Single View Mode (3D Globe vs Detailed 2D Analytical Map) */
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'stretch' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            {viewMode === 'globe' ? (
              <ErrorBoundary name="3D Synoptic Monsoon Earth">
                <GlobeHero
                  activeRegime={regimeLabel}
                  leadDay={selectedLead}
                  hotspots={hotspots}
                  districts={districts}
                  selectedDistrict={selectedDistrict}
                  onSelectDistrict={(d) => (d ? selectDistrict(d) : setSelectedDistrict(null))}
                  selectedLayer={selectedLayer}
                  onSelectLayer={setSelectedLayer}
                  isReplay={true}
                  showHotspots={showHotspots}
                  onExploreDetailedMap={() => setViewMode('analytical')}
                />
              </ErrorBoundary>
            ) : (
              <ErrorBoundary name="2D Detailed Analytical Map">
                <div
                  className="analytical-map-card"
                  style={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-lg)',
                    padding: '1.25rem',
                    boxShadow: 'var(--shadow-md)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '1rem',
                    minHeight: '560px',
                    position: 'relative',
                  }}
                >
                  {/* Analytical Map Header Toolbar */}
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
                          backgroundColor: 'rgba(2, 132, 199, 0.12)',
                          border: '1px solid rgba(2, 132, 199, 0.35)',
                          color: 'var(--accent-cyan)',
                          fontSize: '0.76rem',
                          fontWeight: 800,
                          fontFamily: 'var(--font-mono)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.45rem',
                        }}
                      >
                        <span>🗺️</span>
                        <span>2D DETAILED ANALYTICAL MAP (0.25° GIS)</span>
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
                        Horizon: <strong style={{ color: 'var(--text-primary)' }}>Day +{selectedLead} (+{selectedLead * 24}h)</strong>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                      {/* Layer Selector Tabs */}
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
                            onClick={() => setSelectedLayer(tab.id as GridVariable)}
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

                      {/* Quick Return to 3D Globe Button */}
                      <button
                        type="button"
                        onClick={() => setViewMode('globe')}
                        style={{
                          padding: '4px 10px',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid rgba(56, 189, 248, 0.5)',
                          backgroundColor: 'rgba(2, 132, 199, 0.2)',
                          color: '#38bdf8',
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
                          e.currentTarget.style.backgroundColor = 'rgba(2, 132, 199, 0.4)'
                          e.currentTarget.style.color = '#ffffff'
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgba(2, 132, 199, 0.2)'
                          e.currentTarget.style.color = '#38bdf8'
                        }}
                      >
                        <span>🌍 Switch to 3D Globe</span>
                      </button>
                    </div>
                  </div>

                  {/* Leaflet Detailed Analytical Map Container */}
                  <div
                    style={{
                      height: '520px',
                      width: '100%',
                      borderRadius: 'var(--radius-md)',
                      overflow: 'hidden',
                      border: '1px solid var(--border-subtle)',
                      boxShadow: 'inset 0 0 20px rgba(0,0,0,0.5)',
                    }}
                  >
                    <ForecastMap
                      variable={selectedLayer}
                      cells={gridCells}
                      districtsGeoJson={districtsGeoJson}
                      selectedDistrictId={selectedDistrict?.district_id}
                      onPick={handlePick}
                      pin={pin}
                      flyTo={flyTarget}
                      hotspots={hotspots}
                      showHotspots={showHotspots}
                      style={{ height: '100%', width: '100%' }}
                    />
                  </div>
                </div>
              </ErrorBoundary>
            )}
          </div>

          {/* Selected District Context Panel */}
          <DistrictContextPanel
            district={selectedDistrict}
            pin={pin}
            cellValues={cellValues}
            runId={selectedRunId}
            leadDay={selectedLead}
            onClose={() => {
              setSelectedDistrict(null)
              setPin(null)
            }}
          />
        </div>
      )}

      {/* Tier 3: F1 AI Improvement Summary Banner */}
      <ImprovementSummary summary={improvementSummary} />

      {/* Tier 4: F5 District Priority Table */}
      <PriorityTable
        districts={districts}
        selectedDistrictId={selectedDistrict?.district_id}
        onSelectDistrict={(district) => {
          selectDistrict(district)
          window.scrollTo({ top: 0, behavior: 'smooth' })
        }}
      />
    </div>
  )
}
