interface LoadingStateProps {
  label?: string
  className?: string
}

/**
 * LoadingState component for async data requests
 */
export default function LoadingState({
  label = 'Loading forecast data…',
  className = '',
}: LoadingStateProps) {
  return (
    <div className={`loading-state-box ${className}`}>
      <div className="spinner" />
      <div className="state-message" style={{ marginTop: '0.75rem' }}>
        {label}
      </div>
    </div>
  )
}
