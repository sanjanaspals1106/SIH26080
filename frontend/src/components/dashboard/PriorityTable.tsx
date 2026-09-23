import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { DistrictForecastSummary, AttentionLevel } from '../../types'

interface PriorityTableProps {
  districts: DistrictForecastSummary[]
  selectedDistrictId?: string | null
  onSelectDistrict?: (district: DistrictForecastSummary) => void
}

type SortField =
  | 'priority_rank'
  | 'district_name'
  | 'state'
  | 'attention_level'
  | 'corrected_mean_mm'
  | 'raw_mean_mm'
  | 'heavy_prob_max_cell'
  | 'wettest_cell_mean_mm'

const ATTENTION_ORDER: Record<AttentionLevel, number> = {
  HIGH: 1,
  WATCH: 2,
  NORMAL: 3,
  UNAVAILABLE: 4,
}

export default function PriorityTable({
  districts,
  selectedDistrictId,
  onSelectDistrict,
}: PriorityTableProps) {
  const [search, setSearch] = useState('')
  const [stateFilter, setStateFilter] = useState('ALL')
  const [levelFilter, setLevelFilter] = useState('ALL')
  const [sortField, setSortField] = useState<SortField>('priority_rank')
  const [sortAsc, setSortAsc] = useState(true)

  // Unique states
  const states = Array.from(new Set(districts.map((d) => d.state))).sort()

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(true)
    }
  }

  // Filter and Sort
  const filtered = districts.filter((d) => {
    if (stateFilter !== 'ALL' && d.state !== stateFilter) return false
    if (levelFilter !== 'ALL' && d.attention_level !== levelFilter) return false
    if (search.trim()) {
      const q = search.toLowerCase()
      return (
        d.district_name.toLowerCase().includes(q) ||
        d.state.toLowerCase().includes(q)
      )
    }
    return true
  })

  filtered.sort((a, b) => {
    let diff = 0
    if (sortField === 'attention_level') {
      diff = ATTENTION_ORDER[a.attention_level] - ATTENTION_ORDER[b.attention_level]
    } else if (sortField === 'district_name' || sortField === 'state') {
      diff = a[sortField].localeCompare(b[sortField])
    } else {
      const valA = a[sortField] ?? -999
      const valB = b[sortField] ?? -999
      diff = (valA as number) - (valB as number)
    }
    return sortAsc ? diff : -diff
  })

  return (
    <div
      style={{
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        padding: '0.85rem 1rem',
        boxShadow: 'var(--shadow-sm)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
      }}
    >
      {/* Table Header & Controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.6rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span
            style={{
              fontSize: '0.75rem',
              fontWeight: 800,
              fontFamily: 'var(--font-mono)',
              color: 'var(--accent-cyan)',
              padding: '0.15rem 0.45rem',
              backgroundColor: 'var(--accent-glow)',
              borderRadius: 'var(--radius-xs)',
            }}
          >
            F5 PRIORITY
          </span>
          <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            District Priority & Attention Table
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            ({filtered.length} of {districts.length} districts)
          </span>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexWrap: 'wrap' }}>
          <input
            type="text"
            placeholder="Search district…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              padding: '0.25rem 0.55rem',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-xs)',
              fontSize: '0.78rem',
              outline: 'none',
              width: '130px',
            }}
          />

          <select
            value={stateFilter}
            onChange={(e) => setStateFilter(e.target.value)}
            style={{
              padding: '0.25rem 0.5rem',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-xs)',
              fontSize: '0.78rem',
              outline: 'none',
            }}
          >
            <option value="ALL">All States</option>
            {states.map((st) => (
              <option key={st} value={st}>
                {st}
              </option>
            ))}
          </select>

          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value)}
            style={{
              padding: '0.25rem 0.5rem',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-xs)',
              fontSize: '0.78rem',
              outline: 'none',
            }}
          >
            <option value="ALL">All Attention Levels</option>
            <option value="HIGH">HIGH (P≥50% or P_vh≥20%)</option>
            <option value="WATCH">WATCH (P≥20% or Rain≥15.6mm)</option>
            <option value="NORMAL">NORMAL</option>
            <option value="UNAVAILABLE">UNAVAILABLE (Fallback)</option>
          </select>
        </div>
      </div>

      {/* Table Element */}
      <div style={{ overflowX: 'auto', maxHeight: '360px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-xs)' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.78rem',
            textAlign: 'left',
          }}
        >
          <thead>
            <tr
              style={{
                backgroundColor: 'var(--bg-input)',
                borderBottom: '1px solid var(--border-subtle)',
                color: 'var(--text-muted)',
                position: 'sticky',
                top: 0,
                zIndex: 5,
              }}
            >
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer' }} onClick={() => handleSort('priority_rank')}>
                Rank {sortField === 'priority_rank' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer' }} onClick={() => handleSort('district_name')}>
                District {sortField === 'district_name' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer' }} onClick={() => handleSort('state')}>
                State {sortField === 'state' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer' }} onClick={() => handleSort('attention_level')}>
                Level {sortField === 'attention_level' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer', textAlign: 'right' }} onClick={() => handleSort('corrected_mean_mm')}>
                AI-Corrected {sortField === 'corrected_mean_mm' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer', textAlign: 'right' }} onClick={() => handleSort('raw_mean_mm')}>
                Raw NWP {sortField === 'raw_mean_mm' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer', textAlign: 'right' }} onClick={() => handleSort('wettest_cell_mean_mm')}>
                Wettest Cell {sortField === 'wettest_cell_mean_mm' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', cursor: 'pointer', textAlign: 'right' }} onClick={() => handleSort('heavy_prob_max_cell')}>
                P(≥64.5mm) {sortField === 'heavy_prob_max_cell' ? (sortAsc ? '▲' : '▼') : ''}
              </th>
              <th style={{ padding: '0.45rem 0.6rem', textAlign: 'center' }}>Detail</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={9} style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No districts matching the filter criteria.
                </td>
              </tr>
            ) : (
              filtered.map((d) => {
                const isSelected = selectedDistrictId === d.district_id
                const levelColor =
                  d.attention_level === 'HIGH'
                    ? 'var(--status-high-text)'
                    : d.attention_level === 'WATCH'
                    ? 'var(--status-watch-text)'
                    : d.attention_level === 'NORMAL'
                    ? 'var(--status-normal-text)'
                    : 'var(--status-unavailable-text)'

                const levelBg =
                  d.attention_level === 'HIGH'
                    ? 'var(--status-high-bg)'
                    : d.attention_level === 'WATCH'
                    ? 'var(--status-watch-bg)'
                    : d.attention_level === 'NORMAL'
                    ? 'var(--status-normal-bg)'
                    : 'var(--status-unavailable-bg)'

                return (
                  <tr
                    key={d.district_id}
                    onClick={() => onSelectDistrict && onSelectDistrict(d)}
                    style={{
                      backgroundColor: isSelected
                        ? 'var(--accent-glow)'
                        : 'transparent',
                      borderBottom: '1px solid var(--border-subtle)',
                      cursor: 'pointer',
                      transition: 'background-color 0.1s ease',
                    }}
                    className="table-row-hover"
                  >
                    <td style={{ padding: '0.4rem 0.6rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--text-muted)' }}>
                      #{d.priority_rank}
                    </td>
                    <td style={{ padding: '0.4rem 0.6rem', fontWeight: 600, color: isSelected ? 'var(--accent-cyan)' : 'var(--text-primary)' }}>
                      {d.district_name} {d.is_small && <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>[Small]</span>}
                    </td>
                    <td style={{ padding: '0.4rem 0.6rem', color: 'var(--text-secondary)' }}>{d.state}</td>
                    <td style={{ padding: '0.4rem 0.6rem' }}>
                      <span
                        style={{
                          padding: '0.15rem 0.45rem',
                          backgroundColor: levelBg,
                          color: levelColor,
                          borderRadius: 'var(--radius-xs)',
                          fontFamily: 'var(--font-mono)',
                          fontSize: '0.7rem',
                          fontWeight: 700,
                        }}
                      >
                        {d.attention_level}
                      </span>
                    </td>
                    <td style={{ padding: '0.4rem 0.6rem', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                      {d.corrected_mean_mm} mm
                    </td>
                    <td style={{ padding: '0.4rem 0.6rem', textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                      {d.raw_mean_mm} mm
                    </td>
                    <td style={{ padding: '0.4rem 0.6rem', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                      {d.wettest_cell_mean_mm} mm
                    </td>
                    <td style={{ padding: '0.4rem 0.6rem', textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                      {d.heavy_prob_max_cell !== null ? `${Math.round(d.heavy_prob_max_cell * 100)}%` : '—'}
                    </td>
                    <td style={{ padding: '0.4rem 0.6rem', textAlign: 'center' }}>
                      <Link
                        to={`/districts/${d.district_id}`}
                        style={{
                          color: 'var(--accent-cyan)',
                          textDecoration: 'none',
                          fontSize: '0.72rem',
                          fontWeight: 600,
                        }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        Audit →
                      </Link>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>

      {/* PRD Rule H4 Note */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.7rem',
          color: 'var(--text-muted)',
          fontStyle: 'italic',
        }}
      >
        <span>Model-based attention level. Not an official warning. (Census 2011 boundaries reference)</span>
        <span>Ordering: HIGH &gt; WATCH &gt; NORMAL &gt; UNAVAILABLE</span>
      </div>
    </div>
  )
}
