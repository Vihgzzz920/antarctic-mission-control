import type { RoutingReadiness } from '../api/provenance'

interface Props {
  readiness: RoutingReadiness
  /** the gate's own primary action: route, or enable the override */
  onAct: () => void
  /** opens the provenance detail already in the top bar */
  onDetails: () => void
  busy?: boolean
  disabled?: boolean
  /** 'panel' carries its own primary button; 'inline' is copy only, for a
   *  dock that already owns the action */
  variant?: 'inline' | 'panel'
  /** a short state word above the headline, in the panel variant */
  eyebrow?: string
  /** the backend's own message, when it refused the pairing itself */
  note?: string | null
  testId?: string
}

/**
 * An operational condition, stated and actionable.
 *
 * It is not an error: a historical chart is a deliberate demonstration input
 * whose date is disclosed. So it reads as a condition to acknowledge, and the
 * only way past it is the operator pressing the button.
 */
export default function RoutingGate({
  readiness,
  onAct,
  onDetails,
  busy = false,
  disabled = false,
  variant = 'inline',
  eyebrow,
  note,
  testId = 'routing-gate',
}: Props) {
  return (
    <div
      className={`gate gate-${variant} is-${readiness.state}`}
      data-testid={testId}
      data-state={readiness.state}
    >
      <div className="gate-copy">
        {eyebrow && <span className="gate-eyebrow">{eyebrow}</span>}
        <strong data-testid="gate-headline">{readiness.headline}</strong>
        {readiness.detail && <p>{readiness.detail}</p>}
        {note && (
          <p className="gate-note" data-testid="gate-note">
            {note}
          </p>
        )}
      </div>
      <div className="gate-actions">
        <button
          type="button"
          className="text-action"
          onClick={onDetails}
          data-testid="gate-details"
        >
          Provenance details
        </button>
        {variant === 'panel' && (
          <button
            type="button"
            className="primary-action"
            onClick={onAct}
            disabled={busy || disabled}
            data-testid="gate-action"
            data-action={readiness.action}
          >
            {busy ? 'Planning…' : readiness.actionLabel}
          </button>
        )}
      </div>
    </div>
  )
}
