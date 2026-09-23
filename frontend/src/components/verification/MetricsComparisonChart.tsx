import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts'

const CHART_DATA = [
  { name: 'RMSE (mm)', B0_Raw: 14.8, B1_QM: 13.6, B2_GlobalML: 11.9, B3_RegimeML: 10.4 },
  { name: 'ETS Skill (x10)', B0_Raw: 2.1, B1_QM: 2.5, B2_GlobalML: 3.1, B3_RegimeML: 3.8 },
  { name: 'CSI Threat (x10)', B0_Raw: 3.2, B1_QM: 3.6, B2_GlobalML: 4.2, B3_RegimeML: 4.8 },
  { name: 'FSS Spatial (x10)', B0_Raw: 4.4, B1_QM: 5.1, B2_GlobalML: 5.9, B3_RegimeML: 6.8 },
]

export default function MetricsComparisonChart() {
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
        Key Skill Metrics Comparison (Lower is better for RMSE, Higher for Skill)
      </div>

      <div style={{ flex: 1, width: '100%', minHeight: '220px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={CHART_DATA} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
            <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} />
            <YAxis stroke="var(--text-muted)" fontSize={11} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--bg-topbar)',
                borderColor: 'var(--border-strong)',
                borderRadius: '6px',
                fontSize: '11px',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
            <Bar dataKey="B0_Raw" name="B0: Raw NWP" fill="#94a3b8" radius={[2, 2, 0, 0]} />
            <Bar dataKey="B1_QM" name="B1: Quantile Mapping" fill="#38bdf8" radius={[2, 2, 0, 0]} />
            <Bar dataKey="B2_GlobalML" name="B2: Global ML" fill="#6366f1" radius={[2, 2, 0, 0]} />
            <Bar dataKey="B3_RegimeML" name="B3: Regime-Aware ML" fill="#00b4d8" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
