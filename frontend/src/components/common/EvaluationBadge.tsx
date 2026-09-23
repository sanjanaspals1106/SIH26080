import type { EvaluationSet } from '../../types'

interface EvaluationBadgeProps {
  evaluationSet?: EvaluationSet
  className?: string
}

/**
 * Evaluation Set Badge (PRD Rule H6):
 * Shows DEVELOPMENT (out-of-fold predictions) or HOLDOUT (final models on locked holdout seasons).
 */
export default function EvaluationBadge({
  evaluationSet = 'development',
  className = '',
}: EvaluationBadgeProps) {
  const isHoldout = evaluationSet === 'holdout'
  const badgeClass = isHoldout ? 'badge-holdout' : 'badge-development'
  const titleText = isHoldout
    ? 'Holdout evaluation season (locked run)'
    : 'Development season (leave-one-season-out out-of-fold predictions)'

  return (
    <span
      className={`badge ${badgeClass} ${className}`}
      title={titleText}
    >
      {evaluationSet.toUpperCase()}
    </span>
  )
}
