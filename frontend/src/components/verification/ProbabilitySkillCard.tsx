import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell, ReferenceLine } from 'recharts'
import type { ProbabilityRow } from '../../types'
import { THRESHOLD_LABELS } from './metricUtils'

export default function ProbabilitySkillCard({ rows }: { rows: ProbabilityRow[] }) {
  if (rows.length === 0) return null
  const data = rows.map((r) => ({
    name: `≥${r.threshold_mm} mm`,
    bss: Number(r.brier_skill_score.toFixed(3)),
  }))

  return (
    <div
      style={{
        padding: '1rem',
        backgroundColor: 'var(--bg-card)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.6rem',
      }}
    >
      <div>
        <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Heavy-Rain Probability Skill
        </div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Brier skill score against climatology (above 0 beats always forecasting the climatological rate) · development seasons, out-of-fold
        </div>
      </div>

      <div style={{ height: '170px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
            <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} />
            <YAxis stroke="var(--text-muted)" fontSize={11} domain={[0, 'auto']} />
            <ReferenceLine y={0} stroke="var(--border-strong)" />
            <Tooltip
              contentStyle={{ backgroundColor: 'var(--bg-topbar)', borderColor: 'var(--border-strong)', borderRadius: '6px', fontSize: '11px' }}
              formatter={(v: number) => [v.toFixed(3), 'Brier skill score']}
            />
            <Bar dataKey="bss" radius={[3, 3, 0, 0]}>
              {data.map((d) => (
                <Cell key={d.name} fill={d.bss >= 0 ? '#00b4d8' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.76rem' }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', textAlign: 'right', borderBottom: '1px solid var(--border-subtle)' }}>
            <th style={{ textAlign: 'left', padding: '0.3rem 0.4rem', fontWeight: 600 }}>Threshold</th>
            <th style={{ padding: '0.3rem 0.4rem', fontWeight: 600 }}>Brier (calibrated)</th>
            <th style={{ padding: '0.3rem 0.4rem', fontWeight: 600 }}>Climatology rate</th>
            <th style={{ padding: '0.3rem 0.4rem', fontWeight: 600 }}>Skill score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.threshold_mm} style={{ borderBottom: '1px solid var(--border-subtle)', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
              <td style={{ textAlign: 'left', padding: '0.35rem 0.4rem', fontFamily: 'var(--font-sans)', color: 'var(--text-primary)' }}>
                ≥{r.threshold_mm} mm <span style={{ color: 'var(--text-muted)' }}>{THRESHOLD_LABELS[String(r.threshold_mm)] ?? ''}</span>
              </td>
              <td style={{ padding: '0.35rem 0.4rem' }}>{r.brier_calibrated.toFixed(4)}</td>
              <td style={{ padding: '0.35rem 0.4rem' }}>{(r.climatology_rate * 100).toFixed(2)}%</td>
              <td style={{ padding: '0.35rem 0.4rem', fontWeight: 800, color: 'var(--text-primary)' }}>{r.brier_skill_score.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
