import { useState } from 'react'
import ModelScorecard from '../components/verification/ModelScorecard'
import MetricsComparisonChart from '../components/verification/MetricsComparisonChart'
import FssChart from '../components/verification/FssChart'
import ReliabilityCurveChart from '../components/verification/ReliabilityCurveChart'
import MockDataBanner from '../components/common/MockDataBanner'

export default function Verification() {
  const [selectedLead, setSelectedLead] = useState<number>(1)
  const [selectedThreshold, setSelectedThreshold] = useState<number>(64.5)
  const [selectedEvalSet, setSelectedEvalSet] = useState<'development' | 'holdout'>('holdout')
  const [selectedRegion, setSelectedRegion] = useState<string>('all')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', height: '100%' }}>
      <MockDataBanner message="MOCK DATA — this page is not yet wired to the verification engine; every number below is a placeholder illustration, not a measured skill score (PRD rule H2)." />

      {/* Page Header */}
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
        <div>
          <h1 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
            Verification & Statistical Significance Report
          </h1>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
            Fair comparative benchmark on identical dates, valid IMD cells, and paired block bootstraps (PRD §16)
          </div>
        </div>

        {/* Global Filters Toolbar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {/* Evaluation Set Switcher */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Eval Set:
            </span>
            <div className="segmented-group">
              <button
                type="button"
                className={`segmented-btn ${selectedEvalSet === 'development' ? 'active' : ''}`}
                onClick={() => setSelectedEvalSet('development')}
              >
                Development (OOF)
              </button>
              <button
                type="button"
                className={`segmented-btn ${selectedEvalSet === 'holdout' ? 'active' : ''}`}
                onClick={() => setSelectedEvalSet('holdout')}
              >
                Holdout (Locked)
              </button>
            </div>
          </div>

          {/* Lead Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Lead:
            </span>
            <div className="segmented-group">
              {[1, 2, 3].map((lead) => (
                <button
                  key={lead}
                  type="button"
                  className={`segmented-btn ${selectedLead === lead ? 'active' : ''}`}
                  onClick={() => setSelectedLead(lead)}
                >
                  +{lead * 24}h
                </button>
              ))}
            </div>
          </div>

          {/* Threshold Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Threshold:
            </span>
            <select
              value={selectedThreshold}
              onChange={(e) => setSelectedThreshold(Number(e.target.value))}
              style={{
                padding: '0.3rem 0.55rem',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-xs)',
                fontSize: '0.76rem',
                fontFamily: 'var(--font-mono)',
                outline: 'none',
              }}
            >
              <option value={15.6}>15.6 mm (Moderate)</option>
              <option value={64.5}>64.5 mm (Heavy Rain)</option>
              <option value={115.6}>115.6 mm (Very Heavy)</option>
            </select>
          </div>

          {/* Region Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Region:
            </span>
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              style={{
                padding: '0.3rem 0.55rem',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-xs)',
                fontSize: '0.76rem',
                outline: 'none',
              }}
            >
              <option value="all">All India Grid</option>
              <option value="WEST_COAST">West Coast (Demo Region)</option>
              <option value="CENTRAL">Central Monsoon Zone</option>
              <option value="NE">North-East Hills</option>
              <option value="SOUTH">South Peninsula</option>
            </select>
          </div>
        </div>
      </div>

      {/* Model Comparative Scorecard */}
      <ModelScorecard
        thresholdMm={selectedThreshold}
        leadDay={selectedLead}
      />

      {/* Charts Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
          gap: '0.85rem',
        }}
      >
        <MetricsComparisonChart />
        <FssChart />
      </div>

      {/* Probability Calibration Diagram */}
      <ReliabilityCurveChart />
    </div>
  )
}
