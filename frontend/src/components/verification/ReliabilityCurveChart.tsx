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

const RELIABILITY_DATA = [
  { forecastProb: 0.1, observedFreq: 0.09, perfectLine: 0.1, count: 4820 },
  { forecastProb: 0.3, observedFreq: 0.28, perfectLine: 0.3, count: 2150 },
  { forecastProb: 0.5, observedFreq: 0.52, perfectLine: 0.5, count: 980 },
  { forecastProb: 0.7, observedFreq: 0.68, perfectLine: 0.7, count: 410 },
  { forecastProb: 0.9, observedFreq: 0.88, perfectLine: 0.9, count: 180 },
]

export default function ReliabilityCurveChart() {
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
          Probability Reliability Diagram (Heavy Rain ≥64.5mm)
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          Assesses calibration. Curves close to 1:1 diagonal indicate well-calibrated probabilities.
        </div>
      </div>

      <div style={{ flex: 1, width: '100%', minHeight: '220px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={RELIABILITY_DATA} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
            <XAxis dataKey="forecastProb" stroke="var(--text-muted)" fontSize={11} label={{ value: 'Forecast Probability', position: 'insideBottomRight', offset: -5, fontSize: 10 }} />
            <YAxis domain={[0, 1.0]} stroke="var(--text-muted)" fontSize={11} />
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
              dataKey="observedFreq"
              name="Observed Relative Frequency"
              stroke="#00b4d8"
              strokeWidth={3}
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="perfectLine"
              name="1:1 Perfect Reliability"
              stroke="#94a3b8"
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
