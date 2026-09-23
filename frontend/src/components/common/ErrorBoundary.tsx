import { Component, type ErrorInfo, type ReactNode } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode)
  name?: string
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
    }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`[ErrorBoundary - ${this.props.name || 'Global'}] caught error:`, error, errorInfo)
    this.props.onError?.(error, errorInfo)
  }

  resetError = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        if (typeof this.props.fallback === 'function') {
          return this.props.fallback(this.state.error || new Error('Unknown error'), this.resetError)
        }
        return this.props.fallback
      }

      return (
        <div
          style={{
            padding: '2rem',
            margin: '1.5rem 0',
            backgroundColor: 'var(--bg-card, #0d172e)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            borderRadius: 'var(--radius-md, 8px)',
            color: 'var(--text-primary, #ffffff)',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>⚠️</div>
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#f87171' }}>
            {this.props.name ? `${this.props.name} failed to load` : 'Something went wrong'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary, #94a3b8)', marginBottom: '1rem', fontFamily: 'var(--font-mono, monospace)' }}>
            {this.state.error?.message || 'A runtime component error occurred.'}
          </p>
          <button
            type="button"
            onClick={this.resetError}
            style={{
              padding: '0.45rem 1rem',
              backgroundColor: 'var(--accent-cyan, #00b4d8)',
              color: '#ffffff',
              border: 'none',
              borderRadius: 'var(--radius-xs, 4px)',
              fontWeight: 600,
              cursor: 'pointer',
              fontSize: '0.8rem',
            }}
          >
            Retry Component
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
