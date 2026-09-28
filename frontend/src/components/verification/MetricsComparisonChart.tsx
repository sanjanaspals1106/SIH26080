import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from 'recharts'
import type { VerificationEvaluation } from '../../types'
import { MODEL_COLORS } from './metricUtils'

interface Props {
  evaluation: VerificationEvaluation
}

export default function MetricsComparisonChart({ evaluation }: Props) {
  const rows = [
    { name: 'RMSE (mm)', pick: (m: (typeof evaluation.models)[number]) => m.scalars.rmse },
    { name: 'MAE (mm)', pick: (m: (typeof evaluation.models)[number]) => m.scalars.mae },
  ].map((r) => {
    const row: Record<string, string | number | null> = { name: r.name }
    for (const m of evaluation.models) row[m.key] = r.pick(m) == null ? null : Number(r.pick(m)!.toFixed(2))
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
        Error by model, all rainfall (lower is better)
      </div>
      <div style={{ flex: 1, width: '100%', minHeight: '220px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
            <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} />
            <YAxis stroke="var(--text-muted)" fontSize={11} />
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
