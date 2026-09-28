import { useEffect, useMemo, useRef, useState } from 'react'
import type { DistrictForecastSummary } from '../../types'

interface DistrictSearchProps {
  districts: DistrictForecastSummary[]
  onSelect: (district: DistrictForecastSummary) => void
}

const MAX_RESULTS = 8

/**
 * Type-ahead over district and state names (client-side over every district that has a forecast).
 * Selecting a result loads that district and lets the map / globe fly to it.
 */
export default function DistrictSearch({ districts, onSelect }: DistrictSearchProps) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const boxRef = useRef<HTMLDivElement | null>(null)

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (q.length === 0) return []
    const starts: DistrictForecastSummary[] = []
    const contains: DistrictForecastSummary[] = []
    const byState: DistrictForecastSummary[] = []
    for (const d of districts) {
      const name = d.district_name.toLowerCase()
      if (name.startsWith(q)) starts.push(d)
      else if (name.includes(q)) contains.push(d)
      else if (d.state.toLowerCase().includes(q)) byState.push(d)
    }
    return [...starts, ...contains, ...byState].slice(0, MAX_RESULTS)
  }, [query, districts])

  useEffect(() => {
    setActive(0)
  }, [query])

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [])

  const choose = (d: DistrictForecastSummary) => {
    onSelect(d)
    setQuery('')
    setOpen(false)
  }

  return (
    <div ref={boxRef} style={{ position: 'relative', width: '300px', maxWidth: '100%' }}>
      <div style={{ position: 'relative' }}>
        <span
          aria-hidden
          style={{
            position: 'absolute',
            left: '10px',
            top: '50%',
            transform: 'translateY(-50%)',
            fontSize: '0.8rem',
            opacity: 0.7,
            pointerEvents: 'none',
          }}
        >
          🔍
        </span>
        <input
          type="text"
          value={query}
          placeholder="Search district or state…"
          aria-label="Search district or state"
          onChange={(e) => {
            setQuery(e.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              setActive((a) => Math.min(results.length - 1, a + 1))
            } else if (e.key === 'ArrowUp') {
              e.preventDefault()
              setActive((a) => Math.max(0, a - 1))
            } else if (e.key === 'Enter' && results[active]) {
              e.preventDefault()
              choose(results[active])
            } else if (e.key === 'Escape') {
              setOpen(false)
            }
          }}
          style={{
            width: '100%',
            padding: '0.5rem 0.75rem 0.5rem 2rem',
            backgroundColor: 'var(--bg-input)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.84rem',
            outline: 'none',
            boxSizing: 'border-box',
          }}
        />
      </div>

      {open && query.trim().length > 0 && (
        <div
          role="listbox"
          style={{
            position: 'absolute',
            top: 'calc(100% + 4px)',
            left: 0,
            right: 0,
            zIndex: 2000,
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-strong)',
            borderRadius: 'var(--radius-sm)',
            boxShadow: 'var(--shadow-md)',
            overflow: 'hidden',
          }}
        >
          {results.length === 0 ? (
            <div style={{ padding: '0.6rem 0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              No district or state matches “{query.trim()}”.
            </div>
          ) : (
            results.map((d, i) => (
              <button
                key={d.district_id}
                type="button"
                role="option"
                aria-selected={i === active}
                onMouseEnter={() => setActive(i)}
                onClick={() => choose(d)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'baseline',
                  gap: '0.5rem',
                  width: '100%',
                  padding: '0.45rem 0.75rem',
                  border: 'none',
                  textAlign: 'left',
                  cursor: 'pointer',
                  backgroundColor: i === active ? 'var(--accent-glow)' : 'transparent',
                  color: 'var(--text-primary)',
                  fontSize: '0.82rem',
                }}
              >
                <span style={{ fontWeight: 600 }}>{d.district_name}</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{d.state}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}
