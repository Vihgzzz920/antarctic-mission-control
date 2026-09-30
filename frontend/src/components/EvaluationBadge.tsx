import type { EvaluationMode } from '../api/types'

interface Props {
  mode: EvaluationMode
  isOperationalAssessment: boolean
  icebergExposure?: string
}

/**
 * Which kind of answer the operator is looking at.
 *
 * A long-horizon scientific evaluation and the operational demo are priced
 * differently -- the evaluation omits the iceberg term beyond its configured
 * 12 h horizon and runs on synthetic geometry -- so the two must never look
 * the same. This states the difference in words, not by colour alone, and it
 * stays a single line rather than a banner that takes over the screen.
 */
export default function EvaluationBadge({
  mode,
  isOperationalAssessment,
  icebergExposure,
}: Props) {
  const scientific = mode === 'long_horizon_scientific_evaluation'
  return (
    <span
      className={`eval-badge${scientific ? ' is-scientific' : ''}`}
      data-testid="evaluation-badge"
      data-mode={mode}
      title={
        scientific
          ? 'Forecast-driven long-horizon evaluation. Iceberg exposure is omitted beyond its configured 12 h horizon and the geometry is synthetic.'
          : 'The shipped operational demonstration: persistence environment, iceberg exposure priced within its configured horizon.'
      }
    >
      <b>{scientific ? 'Scientific evaluation' : 'Operational demo'}</b>
      {scientific && (
        <i data-testid="evaluation-badge-note">
          {' '}
          — not an operational assessment
        </i>
      )}
      {scientific && icebergExposure === 'omitted' && (
        <i data-testid="evaluation-badge-iceberg">
          {' '}
          · iceberg exposure omitted beyond 12 h
        </i>
      )}
      {!scientific && isOperationalAssessment && (
        <i> · iceberg exposure priced within its configured horizon</i>
      )}
    </span>
  )
}
