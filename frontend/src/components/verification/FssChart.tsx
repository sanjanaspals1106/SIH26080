import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts'
import type { VerificationEvaluation } from '../../types'
import { MODEL_COLORS } from './metricUtils'

interface Props {
  evaluation: VerificationEvaluation
  threshold: string
}

const SKILLS: { name: string; key: 'pod' | 'ets' | 'csi' | 'fss5' }[] = [
  { name: 'POD', key: 'pod' },
  { name: 'ETS', key: 'ets' },
  { name: 'CSI', key: 'csi' },
  { name: 'FSS 5×5', key: 'fss5' },
]

export default function FssChart({ evaluation, threshold }: Props) {
  const rows = SKILLS.filter((s) => evaluation.models.some((m) => m.thresholds[threshold]?.[s.key] != null)).map((s) => {
    const row: Record<string, string | number | null> = { name: s.name }
    for (const m of evaluation.models) {
      const v = m.thresholds[threshold]?.[s.key]
      row[m.key] = v == null ? null : Number(v.toFixed(3))
    }
    return row
  })

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
        height: '320px',
      }}
    >
      <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-primary)' }}>
        Event skill at {threshold} mm (higher is better)
      </div>
      <div style={{ flex: 1, width: '100%', minHeight: '220px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
            <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} />
            <YAxis stroke="var(--text-muted)" fontSize={11} domain={[0, 'auto']} />
            <Tooltip
              contentStyle={{ backgroundColor: 'var(--bg-topbar)', borderColor: 'var(--border-strong)', borderRadius: '6px', fontSize: '11px' }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
            {evaluation.models.map((m) => (
              <Bar key={m.key} dataKey={m.key} name={m.label} fill={MODEL_COLORS[m.forecast_type]} radius={[2, 2, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
