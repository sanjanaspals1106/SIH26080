interface ErrorStateProps {
  title?: string
  message?: string
  onRetry?: () => void
  className?: string
}

/**
 * ErrorState component for API errors and missing datasets
 */
export default function ErrorState({
  title = 'Failed to load data',
  message = 'An unexpected error occurred while communicating with the server.',
  onRetry,
  className = '',
}: ErrorStateProps) {
  return (
    <div className={`error-state-box ${className}`}>
      <div style={{ fontSize: '1.75rem', color: 'var(--status-high-border)' }}>⚠️</div>
      <div className="state-title">{title}</div>
      <div className="state-message">{message}</div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="theme-toggle-btn"
          style={{ margin: '1rem auto 0', cursor: 'pointer' }}
        >
          ↻ Retry Request
        </button>
      )}
    </div>
  )
}
