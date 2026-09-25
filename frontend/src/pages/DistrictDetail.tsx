import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ForecastOverviewTab from '../components/district/ForecastOverviewTab'
import AuditTrailTab from '../components/district/AuditTrailTab'
import HistoricalAnalogsTab from '../components/district/HistoricalAnalogsTab'
import DistrictRegimeTab from '../components/district/DistrictRegimeTab'
import LoadingState from '../components/common/LoadingState'
import ErrorState from '../components/common/ErrorState'
import MockDataBanner from '../components/common/MockDataBanner'
import {
  getRuns,
  getDistrictForecasts,
  getAuditTrail,
  getHistoricalAnalogs,
  getRegime,
} from '../api'
import type {
  NwpRun,
  DistrictForecastSummary,
  AuditTrailResponse,
  AnalogsResponse,
  RegimeResponse,
} from '../types'

type DistrictTab = 'forecast' | 'audit' | 'analogs' | 'regime'

export default function DistrictDetail() {
  const { districtId } = useParams<{ districtId?: string }>()
  const navigate = useNavigate()

  const [runs, setRuns] = useState<NwpRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState<string>('')
  const [selectedLead, setSelectedLead] = useState<number>(1)
  const [activeTab, setActiveTab] = useState<DistrictTab>('forecast')

  const [districtsList, setDistrictsList] = useState<DistrictForecastSummary[]>([])
  const [selectedDistrict, setSelectedDistrict] = useState<DistrictForecastSummary | null>(null)

  const [auditTrail, setAuditTrail] = useState<AuditTrailResponse | null>(null)
  const [analogs, setAnalogs] = useState<AnalogsResponse | null>(null)
  const [regime, setRegime] = useState<RegimeResponse | null>(null)

  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)

  // Load runs on initial mount
  useEffect(() => {
    async function init() {
      try {
        setLoading(true)
        const runsRes = await getRuns()
        setRuns(runsRes.runs)
        if (runsRes.runs.length > 0) {
          setSelectedRunId(runsRes.runs[0].run_id)
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : 'Failed to load runs')
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  // Load districts list and detail data
  useEffect(() => {
    if (!selectedRunId) return

    async function loadData() {
      try {
        const [districtsRes, regimeRes] = await Promise.all([
          getDistrictForecasts({ run_id: selectedRunId, lead_day: selectedLead }),
          getRegime({ run_id: selectedRunId, lead_day: selectedLead }),
        ])

        setDistrictsList(districtsRes.districts)
        setRegime(regimeRes)

        // Find active district from URL param or default to first
        let current = districtsRes.districts.find((d) => d.district_id === districtId)
        if (!current && districtsRes.districts.length > 0) {
          current = districtsRes.districts[0]
          navigate(`/districts/${current.district_id}`, { replace: true })
        }
        setSelectedDistrict(current || null)

        // Load audit trail and analogs for current district
        if (current) {
          const forecastId = `${selectedRunId}_L${selectedLead}_${current.district_id}`
          const [auditRes, analogsRes] = await Promise.all([
            getAuditTrail(forecastId),
            getHistoricalAnalogs({
              run_id: selectedRunId,
              lead_day: selectedLead,
              district_id: current.district_id,
            }),
          ])
          setAuditTrail(auditRes)
          setAnalogs(analogsRes)
        }
      } catch (err: unknown) {
        console.error('Failed to load district detail data:', err)
      }
    }

    loadData()
  }, [selectedRunId, selectedLead, districtId, navigate])

  const handleSelectDistrict = (dId: string) => {
    navigate(`/districts/${dId}`)
  }

  if (loading) {
    return <LoadingState label="Loading District Detail Workspace…" />
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => window.location.reload()} />
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', height: '100%' }}>
      <MockDataBanner
        show={Boolean(regime?._mock || auditTrail?._mock || analogs?._mock)}
      />

      {/* Header & District Quick Switcher */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          padding: '0.6rem 0.85rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '1.1rem' }}>📍</span>
          <div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Selected District Analysis:
            </div>
            <select
              value={selectedDistrict?.district_id || ''}
              onChange={(e) => handleSelectDistrict(e.target.value)}
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                color: 'var(--text-primary)',
                fontSize: '1.05rem',
                fontWeight: 800,
                cursor: 'pointer',
                outline: 'none',
                padding: '0',
              }}
            >
              {districtsList.map((d) => (
                <option key={d.district_id} value={d.district_id} style={{ backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}>
                  {d.district_name}, {d.state} (#{d.priority_rank} · {d.attention_level})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Run Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Run:
          </span>
          <select
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            style={{
              padding: '0.3rem 0.6rem',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-xs)',
              fontSize: '0.78rem',
              fontFamily: 'var(--font-mono)',
              outline: 'none',
            }}
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.initialization_time.split('T')[0]} ({r.evaluation_set.toUpperCase()})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div
        style={{
          display: 'flex',
          gap: '0.4rem',
          borderBottom: '1px solid var(--border-subtle)',
          paddingBottom: '2px',
        }}
      >
        <button
          type="button"
          className={`segmented-btn ${activeTab === 'forecast' ? 'active' : ''}`}
          onClick={() => setActiveTab('forecast')}
          style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}
        >
          📊 Forecast Overview
        </button>
        <button
          type="button"
          className={`segmented-btn ${activeTab === 'audit' ? 'active' : ''}`}
          onClick={() => setActiveTab('audit')}
          style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}
        >
          🔍 AI Correction Audit Trail (F2)
        </button>
        <button
          type="button"
          className={`segmented-btn ${activeTab === 'analogs' ? 'active' : ''}`}
          onClick={() => setActiveTab('analogs')}
          style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}
        >
          ⏳ Historical Analogs (F3)
        </button>
        <button
          type="button"
          className={`segmented-btn ${activeTab === 'regime' ? 'active' : ''}`}
          onClick={() => setActiveTab('regime')}
          style={{ padding: '0.45rem 0.85rem', fontSize: '0.82rem' }}
        >
          🌀 District Regime Context
        </button>
      </div>

      {/* Tab Content */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {selectedDistrict && activeTab === 'forecast' && (
          <ForecastOverviewTab
            district={selectedDistrict}
            leadDay={selectedLead}
            onSelectLead={setSelectedLead}
          />
        )}

        {activeTab === 'audit' && (
          <AuditTrailTab auditTrail={auditTrail} />
        )}

        {activeTab === 'analogs' && (
          <HistoricalAnalogsTab analogsData={analogs} />
        )}

        {selectedDistrict && activeTab === 'regime' && (
          <DistrictRegimeTab
            district={selectedDistrict}
            regime={regime}
          />
        )}
      </div>
    </div>
  )
}
