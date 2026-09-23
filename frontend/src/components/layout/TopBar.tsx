import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import type { EvaluationSet } from '../../types'

interface TopBarProps {
  evaluationSet?: EvaluationSet
  isMockData?: boolean
}

const ROUTE_LABELS: Record<string, string> = {
  '/': 'Forecast Dashboard',
  '/districts': 'District Detail Analysis',
  '/regime': 'Weather Regime & Transitions',
  '/verification': 'Statistical Verification & Benchmarks',
  '/model-info': 'Model Hierarchy & Data Protocol',
}

export default function TopBar({
  evaluationSet = 'development',
  isMockData = true,
}: TopBarProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const location = useLocation()

  useEffect(() => {
    const savedTheme = localStorage.getItem('sih26080_theme') as 'dark' | 'light' | null
    if (savedTheme) {
      setTheme(savedTheme)
      document.documentElement.setAttribute('data-theme', savedTheme)
    } else {
      document.documentElement.setAttribute('data-theme', 'dark')
    }
  }, [])

  const toggleTheme = () => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark'
    setTheme(nextTheme)
    document.documentElement.setAttribute('data-theme', nextTheme)
    localStorage.setItem('sih26080_theme', nextTheme)
  }

  // Determine current page label
  const basePath = '/' + (location.pathname.split('/')[1] || '')
  const pageLabel = ROUTE_LABELS[basePath] || 'Meteorological Intelligence'

  return (
    <header className="app-topbar">
      {/* Left: Current Page Breadcrumb */}
      <div className="topbar-left">
        <div className="topbar-breadcrumb">
          <span className="breadcrumb-root">Bharat VarshAI</span>
          <span className="breadcrumb-sep">/</span>
          <span className="breadcrumb-page">{pageLabel}</span>
        </div>
      </div>

      {/* Right: Operational Status & Theme Toggle */}
      <div className="topbar-right">
        <div className="system-status-indicator" title="System operational state & replay evaluation protocol">
          <span className="status-beacon" />
          <span className="status-text">
            Operational · Replay Engine ({evaluationSet.toUpperCase()})
            {isMockData && <span style={{ opacity: 0.75, marginLeft: '4px' }}>· Mock Data Preview</span>}
          </span>
        </div>

        <button
          type="button"
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Theme`}
          aria-label="Toggle theme mode"
        >
          <span>{theme === 'dark' ? '☀️' : '🌙'}</span>
          <span className="theme-toggle-label">{theme === 'dark' ? 'Light' : 'Dark'}</span>
        </button>
      </div>
    </header>
  )
}

