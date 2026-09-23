import { NavLink } from 'react-router-dom'
import BrandLogo from '../common/BrandLogo'

interface SidebarProps {
  collapsed: boolean
  onToggleCollapse: () => void
}

interface NavItem {
  to: string
  label: string
  icon: string
  badge?: string
  end?: boolean
}

interface NavSection {
  title: string
  items: NavItem[]
}

const NAV_SECTIONS: NavSection[] = [
  {
    title: 'OVERVIEW',
    items: [
      {
        to: '/',
        label: 'Forecast Dashboard',
        icon: '🗺️',
        end: true,
      },
    ],
  },
  {
    title: 'ANALYSIS',
    items: [
      {
        to: '/districts',
        label: 'District Detail',
        icon: '📍',
      },
      {
        to: '/regime',
        label: 'Regime & Transitions',
        icon: '🌀',
      },
    ],
  },
  {
    title: 'EVALUATION',
    items: [
      {
        to: '/verification',
        label: 'Verification & Scores',
        icon: '📊',
      },
    ],
  },
  {
    title: 'REFERENCE',
    items: [
      {
        to: '/model-info',
        label: 'Model Information',
        icon: 'ℹ️',
      },
    ],
  },
]

export default function Sidebar({ collapsed, onToggleCollapse }: SidebarProps) {
  return (
    <aside className={`app-sidebar ${collapsed ? 'collapsed' : 'expanded'}`}>
      {/* Brand Header */}
      <div className="app-sidebar-header">
        <BrandLogo size={36} collapsed={collapsed} />
        <button
          type="button"
          className="sidebar-collapse-toggle"
          onClick={onToggleCollapse}
          title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          aria-label={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {collapsed ? (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
          ) : (
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="15 18 9 12 15 6" />
            </svg>
          )}
        </button>
      </div>

      {/* Navigation Groups */}
      <nav className="app-nav">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title} className="nav-section-group">
            {!collapsed && (
              <div className="nav-section-label">
                {section.title}
              </div>
            )}
            {section.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'active' : ''} ${collapsed ? 'nav-link-collapsed' : ''}`
                }
                title={collapsed ? item.label : undefined}
              >
                <span className="nav-icon">{item.icon}</span>
                {!collapsed && (
                  <span className="nav-text">{item.label}</span>
                )}
                {!collapsed && item.badge && (
                  <span className="nav-item-badge">{item.badge}</span>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      {/* Clean Sidebar Footer */}
      <div className="app-sidebar-footer">
        {!collapsed ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 600 }}>
              SIH26080 Platform
            </span>
            <span
              style={{
                fontSize: '0.72rem',
                color: '#10b981',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
              }}
            >
              <span
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  backgroundColor: '#10b981',
                  boxShadow: '0 0 6px #10b981',
                }}
              />
              Active
            </span>
          </div>
        ) : (
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              width: '100%',
            }}
            title="SIH26080 Active"
          >
            <span
              style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                backgroundColor: '#10b981',
                boxShadow: '0 0 6px #10b981',
              }}
            />
          </div>
        )}
      </div>
    </aside>
  )
}


