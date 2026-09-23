import { Link } from 'react-router-dom'

interface PRDDisclaimerProps {
  type: 'warning-level' | 'one-day-improvement'
  className?: string
}

/**
 * PRD Honesty Disclaimers:
 * - Rule H4: "Model-based attention level. Not an official warning."
 * - Rule H9: "One day only, not evidence." with link to Verification page.
 */
export default function PRDDisclaimer({ type, className = '' }: PRDDisclaimerProps) {
  if (type === 'warning-level') {
    return (
      <div className={`prd-disclaimer ${className}`}>
        <span>ℹ️ Model-based attention level. Not an official warning.</span>
      </div>
    )
  }

  if (type === 'one-day-improvement') {
    return (
      <div className={`prd-disclaimer ${className}`}>
        <span>One day only, not evidence.</span>
        <Link to="/verification">View Verification Report →</Link>
      </div>
    )
  }

  return null
}
