import { useState, useEffect, type ReactNode } from 'react'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import type { EvaluationSet } from '../../types'

interface AppShellProps {
  children: ReactNode
  evaluationSet?: EvaluationSet
  isMockData?: boolean
}

export default function AppShell({
  children,
  evaluationSet = 'development',
  isMockData = true,
}: AppShellProps) {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    return localStorage.getItem('sih26080_sidebar_collapsed') === 'true'
  })

  useEffect(() => {
    localStorage.setItem('sih26080_sidebar_collapsed', String(collapsed))
  }, [collapsed])

  return (
    <div className={`app-container ${collapsed ? 'sidebar-is-collapsed' : 'sidebar-is-expanded'}`}>
      <Sidebar collapsed={collapsed} onToggleCollapse={() => setCollapsed(!collapsed)} />
      <div className="app-main">
        <TopBar evaluationSet={evaluationSet} isMockData={isMockData} />
        <div className="app-content-view">{children}</div>
      </div>
    </div>
  )
}

