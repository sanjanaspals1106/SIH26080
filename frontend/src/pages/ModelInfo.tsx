import { useEffect, useState } from 'react'
import LoadingState from '../components/common/LoadingState'
import MockDataBanner from '../components/common/MockDataBanner'
import { getModelInfo, getMapMetadata } from '../api'
import type { ModelInfoResponse, MapMetadataResponse } from '../types'

export default function ModelInfo() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [mapMeta, setMapMeta] = useState<MapMetadataResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)

  useEffect(() => {
    async function load() {
      try {
        setLoading(true)
        const [mInfo, mMeta] = await Promise.all([
          getModelInfo(),
          getMapMetadata(),
        ])
        setModelInfo(mInfo)
        setMapMeta(mMeta)
      } catch (err) {
        console.error('Failed to load model info:', err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading || !modelInfo) {
    return <LoadingState label="Loading Model Information & Protocol Status…" />
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', height: '100%' }}>
      <MockDataBanner show={Boolean(modelInfo._mock)} />

      {/* Page Header */}
      <div
        style={{
          padding: '0.6rem 0.85rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <h1 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
          Model Architecture, Data Sources & Evaluation Protocol
        </h1>
        <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
          SIH26080 technical configuration, training splits, fallback hierarchy, and known limitations (PRD §17–§20)
        </div>
      </div>

      {/* 2-Column Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: '0.85rem',
        }}
      >
        {/* Model Ladder & Versions */}
        <div
          style={{
            padding: '1rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.65rem',
          }}
        >
          <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            🤖 Model Hierarchy & Feature Sets
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
            {modelInfo.models.map((m) => (
              <div
                key={m.role}
                style={{
                  padding: '0.55rem 0.75rem',
                  backgroundColor: 'var(--bg-input)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-xs)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontWeight: 700, fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>
                    {m.role}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                    {m.algorithm} · {m.n_features} features ({m.feature_set_version})
                  </div>
                </div>
                <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  {m.version}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Experiment Protocol & Holdout Lock */}
        <div
          style={{
            padding: '1rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.65rem',
          }}
        >
          <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            🔒 Evaluation Protocol & Season Splits
          </div>

          <div style={{ fontSize: '0.78rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Protocol Status:</span>
              <span
                style={{
                  padding: '0.1rem 0.4rem',
                  backgroundColor: 'rgba(168, 85, 247, 0.15)',
                  color: '#d8b4fe',
                  border: '1px solid rgba(168, 85, 247, 0.4)',
                  borderRadius: 'var(--radius-xs)',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 700,
                  fontSize: '0.72rem',
                }}
              >
                {modelInfo.protocol.status}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Total JJAS Seasons:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>{modelInfo.protocol.n_seasons} seasons</strong>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Development Seasons (Leave-One-Season-Out):</span>
              <div style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8', marginTop: '2px' }}>
                {modelInfo.protocol.development_seasons.join(', ')}
              </div>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Locked Holdout Seasons:</span>
              <div style={{ fontFamily: 'var(--font-mono)', color: '#d8b4fe', marginTop: '2px' }}>
                {modelInfo.protocol.holdout_seasons.join(', ')} (Run Once)
              </div>
            </div>
          </div>
        </div>

        {/* Data Sources & Alignment */}
        <div
          style={{
            padding: '1rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.65rem',
          }}
        >
          <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            🛰️ Ingested Datasets & Alignment
          </div>

          <div style={{ fontSize: '0.78rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>NWP Control Forecast:</span>
              <strong>{modelInfo.data.nwp}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Observation Truth:</span>
              <strong>{modelInfo.data.truth}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Time Alignment Method:</span>
              <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                {modelInfo.data.alignment_method} (6-hourly linear interpolation)
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>District Boundaries:</span>
              <strong>{modelInfo.data.district_file}</strong>
            </div>
            {mapMeta && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Valid IMD Cells:</span>
                <strong style={{ fontFamily: 'var(--font-mono)' }}>{mapMeta.grid_bounds.total_valid_cells} cells</strong>
              </div>
            )}
          </div>
        </div>

        {/* Fallbacks & Quality Ladder */}
        <div
          style={{
            padding: '1rem',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.65rem',
          }}
        >
          <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            🛡️ Fallback Ladder & Quality Flags (PRD §17)
          </div>

          <div style={{ fontSize: '0.74rem', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <div style={{ padding: '0.3rem 0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
              1. <strong>Normal Operation</strong> → Serves B3 Regime-Aware ML
            </div>
            <div style={{ padding: '0.3rem 0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
              2. <strong>Regime Features Missing</strong> → Serves B2 Global ML (`REGIME_UNAVAILABLE`)
            </div>
            <div style={{ padding: '0.3rem 0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
              3. <strong>Model Files Missing / OOD</strong> → Serves B1 Quantile Mapping (`ML_UNAVAILABLE` / `OOD_INPUT`)
            </div>
            <div style={{ padding: '0.3rem 0.5rem', backgroundColor: 'var(--bg-input)', borderRadius: 'var(--radius-xs)' }}>
              4. <strong>Extrapolation / Check Fail</strong> → Serves Raw ECMWF (`EXTRAPOLATION` / `VALIDATION_FAILED`)
            </div>
          </div>
        </div>
      </div>

      {/* Known Limitations & Scope Disclaimers */}
      <div
        style={{
          padding: '1rem',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
        }}
      >
        <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          ⚠️ Known Limitations & Operational Scope
        </div>

        <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          {modelInfo.limitations.map((lim) => (
            <li key={lim}>{lim}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}
