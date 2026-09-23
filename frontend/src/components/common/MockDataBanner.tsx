interface MockDataBannerProps {
  show?: boolean
  className?: string
  message?: string
}

/**
 * Mock Data Banner (PRD Rule H7):
 * Mock or made-up data must show a MOCK DATA banner and must never be used for any skill claim.
 */
export default function MockDataBanner({
  show = true,
  className = '',
  message = 'MOCK DATA — Offline preview mock fixtures are active. Not verified real model output.',
}: MockDataBannerProps) {
  if (!show) return null

  return (
    <div className={`banner-mock ${className}`} role="alert">
      <span>⚠️</span>
      <span>{message}</span>
    </div>
  )
}
