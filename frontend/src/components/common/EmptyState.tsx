interface EmptyStateProps {
  title?: string
  message?: string
  isNotAvailableYet?: boolean
  className?: string
}

/**
 * EmptyState (PRD Rule H8):
 * If a required feature is not finished or data is missing, shows "Not available yet".
 */
export default function EmptyState({
  title = 'No Data Available',
  message = 'No records found for the selected parameters.',
  isNotAvailableYet = false,
  className = '',
}: EmptyStateProps) {
  return (
    <div className={`empty-state-box ${className}`}>
      <div style={{ fontSize: '1.75rem', opacity: 0.6 }}>📊</div>
      <div className="state-title">
        {isNotAvailableYet ? 'Not available yet' : title}
      </div>
      <div className="state-message">{message}</div>
    </div>
  )
}
