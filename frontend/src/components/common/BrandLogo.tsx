import { useEffect, useState } from 'react'
import logoDark from '../../assets/logo-dark.png'
import logoLight from '../../assets/logo-light.png'

interface BrandLogoProps {
  size?: number
  collapsed?: boolean
  className?: string
}

export default function BrandLogo({ size = 36, collapsed = false, className = '' }: BrandLogoProps) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

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

  return (
    <div className={`brand-logo-container ${className}`} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
      <div
        className="brand-logo-icon"
        style={{
          width: `${size}px`,
          height: `${size}px`,
          minWidth: `${size}px`,
          minHeight: `${size}px`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          flexShrink: 0,
        }}
      >
        <img
          src={theme === 'dark' ? logoDark : logoLight}
          alt="Bharat VarshAI Logo"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            filter:
              theme === 'dark'
                ? 'drop-shadow(0 2px 8px rgba(0, 180, 216, 0.35))'
                : 'drop-shadow(0 2px 5px rgba(2, 132, 199, 0.2))',
            transition: 'opacity 0.2s ease',
          }}
        />
      </div>

      {!collapsed && (
        <div className="brand-logo-text" style={{ overflow: 'hidden', whiteSpace: 'nowrap' }}>
          <div
            style={{
              fontSize: '1.02rem',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
              lineHeight: 1.15,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
            }}
          >
            <span>Bharat VarshAI</span>
          </div>
          <div
            style={{
              fontSize: '0.68rem',
              color: 'var(--text-muted)',
              fontWeight: 600,
              letterSpacing: '0.04em',
              marginTop: '1px',
            }}
          >
            SIH26080
          </div>
        </div>
      )}
    </div>
  )
}


