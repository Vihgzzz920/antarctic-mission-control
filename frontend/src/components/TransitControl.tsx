import type { RouteCell } from '../api/types'
import { timestamp } from './format'
import type { VesselPosition } from '../map/forecast'

interface Props {
  playing: boolean
  onToggle: () => void
  position: VesselPosition | null
  cell: RouteCell | null
}

/**
 * Playback of the route the search already returned: the vessel walks the
 * returned cells at their own arrival times. Nothing is recalculated.
 */
export default function TransitControl({
  playing,
  onToggle,
  position,
  cell,
}: Props) {
  return (
    <span className="transit" data-testid="transit-control">
      <button
        type="button"
        className={`transit-button${playing ? ' is-on' : ''}`}
        onClick={onToggle}
      >
        {playing ? 'Pause transit' : 'Play vessel transit'}
      </button>
      {position && cell && (
        <span className="transit-readout" data-testid="transit-readout">
          <span>
            <b>{timestamp(cell.arrival_datetime).slice(11, 16)}</b>
            <i>arrival</i>
          </span>
          <span>
            <b>{position.bucket}</b>
            <i>bucket</i>
          </span>
          <span>
            <b>
              {cell.iceberg_exposure === null
                ? '—'
                : cell.iceberg_exposure.toFixed(3)}
            </b>
            <i>exposure</i>
          </span>
        </span>
      )}
    </span>
  )
}
