interface ReplayBadgeProps {
  className?: string
}

/**
 * Replay Badge (PRD Rule H6): Every screen shows a badge: REPLAY.
 */
export default function ReplayBadge({ className = '' }: ReplayBadgeProps) {
  return (
    <span className={`badge badge-replay ${className}`} title="System running in Replay mode (stored past forecasts)">
      REPLAY
    </span>
  )
}
