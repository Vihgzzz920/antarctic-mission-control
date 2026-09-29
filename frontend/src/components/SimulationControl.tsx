import { IcebergIcon } from './icons'

interface Props {
  armed: boolean
  active: boolean
  busy: boolean
  disabled: boolean
  /** the deterministic action: a cell of the route the search returned */
  onInject: () => void
  /** the secondary one: arm the map for a free click */
  onArm: () => void
  onClear: () => void
}

/**
 * The simulated event's controls.
 *
 * The primary action places the berg on the route itself, so what happens next
 * is decided by the planner rather than by the operator's aim. Free placement
 * stays available, secondary, for putting one somewhere specific.
 */
export default function SimulationControl({
  armed,
  active,
  busy,
  disabled,
  onInject,
  onArm,
  onClear,
}: Props) {
  return (
    <div className="sim-control" data-testid="simulation-control">
      {!active && (
        <button
          type="button"
          className="primary-action sim-inject"
          onClick={onInject}
          disabled={disabled || busy || armed}
          data-testid="inject-on-route"
        >
          <IcebergIcon size={14} />
          {busy ? 'Replanning…' : 'Inject iceberg on this route'}
        </button>
      )}
      {!active && (
        <button
          type="button"
          className={`text-action${armed ? ' is-armed' : ''}`}
          onClick={onArm}
          disabled={disabled || busy}
          aria-pressed={armed}
          data-testid="place-manually"
        >
          {armed ? 'Click the map to place' : 'Place manually'}
        </button>
      )}
      {(active || armed) && (
        <button
          type="button"
          className="sim-clear"
          onClick={onClear}
          data-testid="clear-simulation"
        >
          Clear simulation
        </button>
      )}
    </div>
  )
}
