import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts'

const FSS_DATA = [
  { scale: '1x1 (28km)', Raw_NWP: 0.32, Regime_ML: 0.48, Useful_Skill_Line: 0.53 },
  { scale: '3x3 (84km)', Raw_NWP: 0.44, Regime_ML: 0.68, Useful_Skill_Line: 0.53 },
  { scale: '5x5 (140km)', Raw_NWP: 0.56, Regime_ML: 0.79, Useful_Skill_Line: 0.53 },
  { scale: '9x9 (250km)', Raw_NWP: 0.69, Regime_ML: 0.89, Useful_Skill_Line: 0.53 },
]

export default function FssChart() {
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
      <div>
        <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          FSS vs Spatial Scale (Fraction Skill Score, Threshold: 64.5mm)
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          Assesses displacement tolerance. Crosses useful skill threshold (0.5 + f0/2) at 84 km.
        </div>
      </div>

      <div style={{ flex: 1, width: '100%', minHeight: '220px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={FSS_DATA} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis dataKey="scale" stroke="var(--text-muted)" fontSize={11} />
            <YAxis domain={[0.2, 1.0]} stroke="var(--text-muted)" fontSize={11} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--bg-topbar)',
                borderColor: 'var(--border-strong)',
                borderRadius: '6px',
                fontSize: '11px',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
            <Line
              type="monotone"
              dataKey="Regime_ML"
              name="B3: Regime-Aware ML"
              stroke="#00b4d8"
              strokeWidth={3}
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="Raw_NWP"
              name="B0: Raw NWP"
              stroke="#94a3b8"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="Useful_Skill_Line"
              name="Useful Skill (0.5 + f0/2)"
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
