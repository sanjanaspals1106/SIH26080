import type { GridVariable } from '../../types'
import {
  RAIN_BINS,
  DIFFERENCE_BINS,
  IMPROVEMENT_BINS,
} from '../../utils/colorScales'

interface LayerToggleProps {
  selectedLayer: GridVariable
  onSelectLayer: (layer: GridVariable) => void
}

const LAYERS: Array<{ id: GridVariable; label: string; desc: string }> = [
  { id: 'raw', label: 'Raw NWP', desc: 'Raw ECMWF control forecast rain (B0)' },
  { id: 'corrected', label: 'AI-Corrected', desc: 'Regime-aware post-processed prediction (B3)' },
  { id: 'difference', label: 'Difference', desc: 'AI-Corrected minus Raw NWP (mm)' },
  { id: 'observed', label: 'Observed', desc: 'IMD 0.25° gridded observation (Truth)' },
  { id: 'improvement', label: 'Improvement', desc: '|Raw − Obs| − |Corr − Obs|' },
]

export default function LayerToggle({
  selectedLayer,
  onSelectLayer,
}: LayerToggleProps) {
  const currentBins =
    selectedLayer === 'difference'
      ? DIFFERENCE_BINS
      : selectedLayer === 'improvement'
      ? IMPROVEMENT_BINS
      : RAIN_BINS

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '0.75rem',
        padding: '0.55rem 0.85rem',
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      {/* Layer Pills */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Layer:
        </span>
        <div className="segmented-group">
          {LAYERS.map((layer) => (
            <button
              key={layer.id}
              type="button"
              className={`segmented-btn ${selectedLayer === layer.id ? 'active' : ''}`}
              onClick={() => onSelectLayer(layer.id)}
              title={layer.desc}
            >
              {layer.label}
            </button>
          ))}
        </div>
      </div>

      {/* Dynamic Color Scale Legend */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          flexWrap: 'wrap',
          padding: '0.25rem 0.5rem',
          backgroundColor: 'var(--bg-input)',
          borderRadius: 'var(--radius-xs)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <span style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.7rem' }}>
          {selectedLayer === 'difference'
            ? 'Difference (mm):'
            : selectedLayer === 'improvement'
            ? 'Improvement:'
            : 'Scale (mm/24h):'}
        </span>
        <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
          {currentBins.map((bin) => (
            <div
              key={bin.label}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.1rem 0.35rem',
                backgroundColor: 'var(--bg-card)',
                borderRadius: 'var(--radius-xs)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <span
                style={{
                  display: 'inline-block',
                  width: '10px',
                  height: '10px',
                  backgroundColor: bin.color,
                  borderRadius: '2px',
                }}
              />
              <span style={{ color: 'var(--text-primary)', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                {bin.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
