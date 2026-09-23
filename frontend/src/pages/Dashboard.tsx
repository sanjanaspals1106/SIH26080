import { useEffect, useState } from 'react'
import GlobeHero from '../components/dashboard/GlobeHero'
import ForecastControls from '../components/dashboard/ForecastControls'
import ForecastComparison from '../components/dashboard/ForecastComparison'
import ForecastMap from '../components/dashboard/ForecastMap'
import ImprovementSummary from '../components/dashboard/ImprovementSummary'
import DistrictContextPanel from '../components/dashboard/DistrictContextPanel'
import PriorityTable from '../components/dashboard/PriorityTable'
import LoadingState from '../components/common/LoadingState'
import ErrorState from '../components/common/ErrorState'
import {
  getRuns,
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

import ErrorBoundary from '../components/common/ErrorBoundary'

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

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Load runs and GeoJSON on initial mount
  useEffect(() => {
    async function init() {
      try {
        setLoading(true)
        const [runsRes, geoJsonRes] = await Promise.all([
          getRuns(),
          getDistrictGeoJSON(),
        ])
        setRuns(runsRes.runs)
        if (runsRes.runs.length > 0) {
          setSelectedRunId(runsRes.runs[0].run_id)
        }
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

    async function loadForecastData() {
      try {
        const [districtsRes, gridRes, rawGridRes, impRes, hotspotsRes] = await Promise.all([
          getDistrictForecasts({ run_id: selectedRunId, lead_day: selectedLead }),
          getGridForecast({ run_id: selectedRunId, lead_day: selectedLead, variable: selectedLayer }),
          getGridForecast({ run_id: selectedRunId, lead_day: selectedLead, variable: 'raw' }),
          getImprovementSummary({ run_id: selectedRunId, lead_day: selectedLead }),
          getHotspots({ run_id: selectedRunId, lead_day: selectedLead }),
        ])

        setDistricts(districtsRes.districts)
        setGridCells(gridRes.cells)
        setRawGridCells(rawGridRes.cells)
        setImprovementSummary(impRes)
        setHotspots(hotspotsRes.hotspots)

        // Keep selected district updated with new lead/run values using functional update
        setSelectedDistrict((prev) => {
          if (prev) {
            const updated = districtsRes.districts.find(
              (d) => d.district_id === prev.district_id
            )
            return updated || districtsRes.districts[0] || null
          }
          return districtsRes.districts[0] || null
        })
      } catch (err: unknown) {
        console.error('Error fetching forecast data:', err)
      }
    }

    loadForecastData()
  }, [selectedRunId, selectedLead, selectedLayer])

  if (loading) {
    return <LoadingState label="Loading Bharat VarshAI forecast dashboard…" />
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => window.location.reload()} />
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
      {/* Primary Dashboard Hero Header (Canva Reference Alignment) */}
      <div className="dashboard-hero-header" style={{ padding: '0.25rem 0.25rem 0 0.25rem' }}>
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
      {isSplitView ? (
        /* F1 Side-by-Side Comparison Workspace */
        <ErrorBoundary name="F1 Split View Comparison">
          <div style={{ minHeight: '520px' }}>
            <ForecastComparison
              rawCells={rawGridCells}
              correctedCells={gridCells}
              districtsGeoJson={districtsGeoJson}
              districtsList={districts}
              selectedDistrictId={selectedDistrict?.district_id}
              onSelectDistrict={setSelectedDistrict}
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
                  activeRegime="Active Monsoon (Phase 2)"
                  leadDay={selectedLead}
                  hotspots={hotspots}
                  districts={districts}
                  selectedDistrict={selectedDistrict}
                  onSelectDistrict={setSelectedDistrict}
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
                      districtsList={districts}
                      selectedDistrictId={selectedDistrict?.district_id}
                      onSelectDistrict={setSelectedDistrict}
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
            onClose={() => setSelectedDistrict(null)}
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
          setSelectedDistrict(district)
          window.scrollTo({ top: 0, behavior: 'smooth' })
        }}
      />
    </div>
  )
}
