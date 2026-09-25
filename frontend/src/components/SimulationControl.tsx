import { IcebergIcon } from './icons'

interface Props {
  armed: boolean
  active: boolean
  busy: boolean
  disabled: boolean
  onArm: () => void
  onClear: () => void
}

/**
 * The map's one action control. Arming it turns the next map click into a
 * placement; it does not run anything by itself.
 */
export default function SimulationControl({
  armed,
  active,
  busy,
  disabled,
  onArm,
  onClear,
}: Props) {
  return (
    <div className="sim-control" data-testid="simulation-control">
      <button
        type="button"
        className={`sim-button${armed ? ' is-armed' : ''}`}
        onClick={onArm}
        disabled={disabled || busy || active}
        aria-pressed={armed}
      >
        <IcebergIcon size={14} />
        {busy ? 'Replanning…' : armed ? 'Click the map to place' : 'Simulate iceberg'}
      </button>
      {(active || armed) && (
        <button type="button" className="sim-clear" onClick={onClear}>
          Clear simulation
        </button>
      )}
      {armed && !active && (
        <p className="sim-hint">
          The simulated iceberg is placed at the moment the current route is
          predicted to reach it. Nothing is written to any dataset.
        </p>
      )}
    </div>
  )
}
